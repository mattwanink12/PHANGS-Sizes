"""
radius_distribution.py - Create a histogram showing the distribution of effective radii

This takes the following parameters:
- Path to save the plot
- Path to the public catalog
"""

import sys
from pathlib import Path
from astropy import table
from astropy import units as u
import numpy as np
import betterplotlib as bpl

bpl.set_style()

# load the files and catalogs
plot_name = Path(sys.argv[1])
galaxy_catalogs = dict()
catalog = table.Table.read(sys.argv[2], format="ascii.ecsv")

# restrict to the clusters with reliable radii
catalog = catalog[catalog["reliable_radius"]]

# add the pixel scale to be used below
catalog["pixel_scale"] = catalog["r_eff_arcsec"] / catalog["r_eff_pixels"]
for item in catalog["pixel_scale"]:
    assert np.isclose(item, 39.62e-3, atol=0, rtol=1e-2)

# calculate the mean error in log space, will be used for the KDE smothing
r_eff = catalog["r_eff_pc"]
log_r_eff = np.log10(r_eff)
log_err_lo = log_r_eff - np.log10(r_eff - catalog["r_eff_pc_e-"])
log_err_hi = np.log10(r_eff + catalog["r_eff_pc_e+"]) - log_r_eff

catalog["r_eff_log"] = log_r_eff
mean_log_err = 0.5 * (log_err_lo + log_err_hi)
catalog["r_eff_log_smooth"] = 1.75 * mean_log_err  # don't do average, it is too small

# I will have two panels. One will show galaxies with similar radius distributions, the
# other will show ones that deviate. Each panel will have an "other galaxies" category,
# but this "other galaxies" category must be different between the two panels, as the
# sample is different. I have the galaxies manually listed here, but this includes all
# galaxies with more than 180 clusters with well-measured radii. They are also listed
# in order of decreasing cluster number, making them easier to plot
galaxies_1 = ["ngc5194", "ngc628", "ngc1313", "ngc4449"]  # , "ngc3344"]
galaxies_2 = ["ngc1566", "ngc7793"]

individual_cats = dict()
other_cats_1 = []
other_cats_2 = []
all_catalogs = []
for galaxy in np.unique(catalog["galaxy"]):
    this_catalog = catalog[catalog["galaxy"] == galaxy]
    # if it's one of the ones to save, save it
    if galaxy in galaxies_1 or galaxy in galaxies_2:
        individual_cats[galaxy] = this_catalog
    # separately determine the "other" category
    if galaxy not in galaxies_1 and galaxy not in galaxies_2:
        other_cats_1.append(this_catalog)
    if galaxy not in galaxies_2:
        other_cats_2.append(this_catalog)

other_cat_1 = table.vstack(other_cats_1, join_type="inner")
other_cat_2 = table.vstack(other_cats_2, join_type="inner")


def gaussian(x, mean, variance):
    """
    Normalized Gaussian Function at a given value.

    Is normalized to integrate to 1.

    :param x: value to calculate the Gaussian at
    :param mean: mean value of the Gaussian
    :param variance: Variance of the Gaussian distribution
    :return: log of the likelihood at x
    """
    exp_term = np.exp(-((x - mean) ** 2) / (2 * variance))
    normalization = 1.0 / np.sqrt(2 * np.pi * variance)
    return exp_term * normalization


def kde(r_eff_grid, log_r_eff, log_r_eff_err):
    ys = np.zeros(r_eff_grid.size)
    log_r_eff_grid = np.log10(r_eff_grid)

    for lr, lre in zip(log_r_eff, log_r_eff_err):
        ys += gaussian(log_r_eff_grid, lr, lre ** 2)

    # # normalize the y value
    ys = np.array(ys)
    ys = 70 * ys / np.sum(ys)
    return ys


# set the colors to be used on the plots
colors = {
    "ngc5194": bpl.color_cycle[0],
    "ngc628": bpl.color_cycle[1],
    "ngc1313": bpl.color_cycle[3],
    "ngc4449": bpl.color_cycle[4],
    "other_1": bpl.color_cycle[5],
    "ngc1566": bpl.color_cycle[7],
    "ngc7793": bpl.color_cycle[6],
    "other_2": bpl.almost_black,
}

fig, axs = bpl.subplots(ncols=2, figsize=[14, 7])
radii_plot = np.logspace(-1, 1.5, 300)
for idx, galaxy in enumerate(galaxies_1 + galaxies_2):
    if galaxy in galaxies_1:
        ax = axs[0]
    else:
        ax = axs[1]

    cat = individual_cats[galaxy]

    # I want to add an extra space in the legend for NGC628
    label = f"NGC {galaxy[3:]}, "
    if len(galaxy) == 6:
        # chose spaces fine tuned to align in the legend:
        # https://www.overleaf.com/learn/latex/Spacing_in_math_mode
        label += "$\  \ $"
    label += f"N={len(cat)}"

    ax.plot(
        radii_plot,
        kde(
            radii_plot,
            cat["r_eff_log"],
            cat["r_eff_log_smooth"],
        ),
        c=colors[galaxy],
        lw=3,
        label=label,
        zorder=10 - idx,
    )
# plot all other galaxies on both
axs[0].plot(
    radii_plot,
    kde(
        radii_plot,
        other_cat_1["r_eff_log"],
        other_cat_1["r_eff_log_smooth"],
    ),
    lw=3,
    c=colors["other_1"],
    zorder=15,
    label=f"Other Galaxies, N={len(other_cat_1)}",
)
axs[1].plot(
    radii_plot,
    kde(
        radii_plot,
        other_cat_2["r_eff_log"],
        other_cat_2["r_eff_log_smooth"],
    ),
    lw=3,
    c=colors["other_2"],
    zorder=15,
    label=f"All Other Galaxies, N={len(other_cat_2)}",
)


for ax in axs:
    ax.set_xscale("log")
    ax.set_limits(0.1, 20, 0, 1.3)
    ax.set_xticks([0.1, 1, 10])
    ax.set_xticklabels(["0.1", "1", "10"])
    ax.add_labels("$R_{eff}$ [pc]", "Normalized KDE Density")
    ax.legend(frameon=False, fontsize=14, loc=2)

# then add all the pixel sizes
for galaxy in galaxies_1 + galaxies_2:
    if galaxy in galaxies_1:
        ax = axs[0]
    else:
        ax = axs[1]

    pixel_size_arcsec = individual_cats[galaxy]["pixel_scale"][0]
    distance_mpc = individual_cats[galaxy]["galaxy_distance_mpc"][0]

    pixel_size_radian = (pixel_size_arcsec * u.arcsec).to("radian").value
    pixel_size_pc = pixel_size_radian * distance_mpc * 1e6
    ax.plot(
        [pixel_size_pc, pixel_size_pc],
        [0, 0.07],
        lw=3,
        c=colors[galaxy],
        zorder=0,
    )

fig.savefig(plot_name)
