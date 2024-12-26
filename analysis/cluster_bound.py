"""
cluster_bound.py - Create two plots showing the dynamical state of clusters

This takes the following parameters:
- Path to save the plot comparing crossing time to age
- Path to save the plot showing the mass dependence of the bound fraction
- The public catalog
"""

import sys
from pathlib import Path
from astropy import table
import numpy as np
from matplotlib import colors
from matplotlib import cm
import betterplotlib as bpl

# set random seed for reproducibility
np.random.seed(123)

bpl.set_style()

mrr_dir = Path(__file__).resolve().parent / "mass_radius_relation"
sys.path.append(str(mrr_dir))
from mass_radius_utils_plotting import age_colors

# ======================================================================================
#
# Load the catalogs that were passed in
#
# ======================================================================================
time_comp_plot_name = Path(sys.argv[1]).resolve()
mass_dependence_plot_name = Path(sys.argv[2]).resolve()
catalog = table.Table.read(sys.argv[3], format="ascii.ecsv")

# restrict to clusters with good masses and radii
mask = np.logical_and(catalog["reliable_radius"], catalog["reliable_mass"])
catalog = catalog[mask]

# then determine which clusters are bound
catalog["bound"] = catalog["SEDfix_age"] > catalog["crossing_time_yr"]
print("bound fraction all", np.sum(catalog["bound"]) / len(catalog))
massive_mask = catalog["SEDfix_mass"] > 5000
print(
    "bound fraction M > 5000",
    np.sum(catalog["bound"][massive_mask]) / len(catalog[massive_mask]),
)
old_mask = catalog["SEDfix_age"] >= 1e7
print(
    "bound fraction age > 1e7",
    np.sum(catalog["bound"][old_mask]) / len(catalog[old_mask]),
)

# ======================================================================================
#
# make the simple plot
#
# ======================================================================================
figsize = [8, 5.5]
fig, ax = bpl.subplots(figsize=figsize)
# make the colormap for masses
# cmap = cm.get_cmap("gist_earth_r")
# cmap = cmocean.cm.thermal_r
# cmap = cmocean.tools.crop_by_percent(cmap, 20, "min")
# make a custom colormap madee manually by taking colors from
# https://sashamaps.net/docs/resources/20-colors/ and fading them
cmap_colors = ["#f58231", "#FFAC71", "#8BA4FD", "#4363d8"]
cmap = colors.ListedColormap(colors=cmap_colors, name="")
norm = colors.LogNorm(vmin=1e3, vmax=1e5)
mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
mass_colors = mappable.to_rgba(catalog["SEDfix_mass"])
# perturb the ages slightly for plotting purposes. Copy them to avoid messing up
# later analysis
plot_ages = catalog["SEDfix_age"].copy()
plot_ages *= np.random.normal(1, 0.15, len(plot_ages))

# then plot and set some limits
ax.scatter(plot_ages, catalog["crossing_time_yr"], s=7, alpha=1, c=mass_colors)
ax.add_labels("Age [yr]", "Crossing Time [yr]")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_limits(1e5, 2e10, 1e5, 3e8)
cbar = fig.colorbar(mappable, ax=ax, extend="both")
cbar.set_label("Mass [$M_\odot$]")

# add the line for boundedness and label it. Have to use the limits to determine the
# proper rotation of the labels
ax.plot([1, 1e12], [1, 1e12], lw=2, c=bpl.almost_black, zorder=0)
frac = 1.25
center = 4e5
shared = {"ha": "center", "va": "center", "rotation": 51, "fontsize": 18}
ax.add_text(x=center * frac, y=center / frac, text="Bound", **shared)
ax.add_text(x=center / frac, y=center * frac, text="Unbound", **shared)

fig.savefig(time_comp_plot_name)

# ======================================================================================
#
# plot bound fraction vs mass
#
# ======================================================================================
mask_all = catalog["SEDfix_age"] > 0
mask_young = catalog["SEDfix_age"] < 1e7
mask_med = np.logical_and(catalog["SEDfix_age"] >= 1e7, catalog["SEDfix_age"] < 1e8)
mask_old = np.logical_and(catalog["SEDfix_age"] >= 1e8, catalog["SEDfix_age"] < 1e9)


def bound_fraction(mask):
    this_subset = catalog[mask]
    mass_bins = np.logspace(2, 6, 13)
    # then figure out which clusters are in the mass bins
    bound_fractions = []
    bound_fraction_errs = []
    mass_centers = []
    for idx_low in range(len(mass_bins) - 1):
        m_lo = mass_bins[idx_low]
        m_hi = mass_bins[idx_low + 1]

        mask_lo = this_subset["SEDfix_mass"] > m_lo
        mask_hi = this_subset["SEDfix_mass"] <= m_hi

        this_mass_subset = this_subset[np.logical_and(mask_lo, mask_hi)]

        if len(this_mass_subset) < 10:
            continue

        this_bound_fraction = np.sum(this_mass_subset["bound"]) / len(this_mass_subset)
        # then calculate error according to Gaussian approximation
        # https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
        p = this_bound_fraction
        this_bound_fraction_err = np.sqrt((p * (1 - p)) / len(this_mass_subset))

        bound_fractions.append(this_bound_fraction)
        bound_fraction_errs.append(this_bound_fraction_err)

        mass_centers.append(10 ** np.mean([np.log10(m_lo), np.log10(m_hi)]))

    return (
        np.array(mass_centers),
        np.array(bound_fractions),
        np.array(bound_fraction_errs),
    )


fig, ax = bpl.subplots(figsize=figsize)

for mask, name, color, zorder in zip(
    [mask_young, mask_med, mask_old],
    ["Age: 1-10 Myr", "Age: 10-100 Myr", "Age: 100 Myr - 1 Gyr"],
    age_colors,
    [5, 6, 7],
):
    plot_mass, plot_frac, plot_frac_err = bound_fraction(mask)
    ax.plot(plot_mass, plot_frac, lw=5, c=color, label=name, zorder=zorder)
    line_above = plot_frac + plot_frac_err
    line_below = plot_frac - plot_frac_err
    ax.fill_between(
        plot_mass,
        y1=line_below,
        y2=line_above,
        color=color,
        alpha=0.5,
        zorder=0,
        lw=0,
    )

    # then plot and set some limits
    ax.add_labels("Mass [$M_\odot$]", "Fraction of Bound Clusters")
ax.set_xscale("log")
ax.set_limits(1e2, 1e6, 0, 1.05)
ax.legend(fontsize=14)
ax.axhline(1.0, ls=":", lw=1, zorder=0)

fig.savefig(mass_dependence_plot_name)
