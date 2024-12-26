"""
artificial_comparison.py
Compare the results of the artificial cluster test to the true results

This takes the following parameters:
- Path to save the plot
- Path to the output catalog with fitted radii
- Path to the PSF
- oversampling factor used in the PSF.
"""

import sys
from pathlib import Path
import numpy as np
from astropy import table
from astropy.io import fits
import cmocean
from matplotlib import ticker, colors, cm, gridspec
from matplotlib import pyplot as plt
import betterplotlib as bpl

bpl.set_style()

# ======================================================================================
#
# Load the catalogs that were passed in
#
# ======================================================================================
plot_name = Path(sys.argv[1]).resolve()
catalog_name = Path(sys.argv[2]).resolve()
catalog = table.Table.read(catalog_name, format="ascii.ecsv")
# load the PSF
psf_path = Path(sys.argv[3]).resolve()
psf = fits.open(psf_path)["PRIMARY"].data
# the convolution requires the psf to be normalized, and without any negative values
psf = np.maximum(psf, 0)
psf /= np.sum(psf)

oversampling_factor = int(sys.argv[4])

# ======================================================================================
#
# Then calculate the error for the parameters of interest
#
# ======================================================================================
reff = catalog["r_eff_pixels"]
reff_true = catalog["reff_pixels_true"]
# get the ratio, and its errorbars
reff_ratio = reff / reff_true
catalog["r_eff_ratio_e+"] = catalog["r_eff_pixels_e+"] / reff_true
catalog["r_eff_ratio_e-"] = catalog["r_eff_pixels_e-"] / reff_true

# ======================================================================================
#
# Then calculate the error for the parameters of interest
#
# ======================================================================================
def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def measure_psf_reff(psf, oversampling_factor):
    # the center is the central pixel of the image
    x_cen = int((psf.shape[1] - 1.0) / 2.0)
    y_cen = int((psf.shape[0] - 1.0) / 2.0)
    total = np.sum(psf)
    half_light = total / 2.0
    # then go through all the pixel values to determine the distance from the center.
    # Then we can go through them in order to determine the half mass radius
    radii = []
    values = []
    for x in range(psf.shape[1]):
        for y in range(psf.shape[1]):
            # need to include the oversampling factor in the distance
            radii.append(distance(x, y, x_cen, y_cen) / oversampling_factor)
            values.append(psf[y][x])

    idxs_sort = np.argsort(radii)
    sorted_radii = np.array(radii)[idxs_sort]
    sorted_values = np.array(values)[idxs_sort]

    cumulative_light = 0
    for idx in range(len(sorted_radii)):
        cumulative_light += sorted_values[idx]
        if cumulative_light >= half_light:
            return sorted_radii[idx]


psf_r_eff = measure_psf_reff(psf, oversampling_factor)

# ======================================================================================
#
# Then plot this up
#
# ======================================================================================
# Function to use to set the ticks
@ticker.FuncFormatter
def nice_log_formatter(x, pos):
    exp = np.log10(x)
    # this only works for labels that are factors of 10. Other values will produce
    # misleading results, so check this assumption.
    assert np.isclose(exp, int(exp))

    # for values between 0.01 and 100, just use that value.
    # Otherwise use the log.
    if abs(exp) < 2:
        return f"{x:g}"
    else:
        return "$10^{" + f"{exp:.0f}" + "}$"


# Set up the colormap to use any given value in the catalog.
cbar_quantity = "mag_F555W"
# set up the parameters used for this quantity
cbar_name = {
    "peak_pixel_value_true": "Log Peak Pixel Value",
    "power_law_slope_true": "Power Law Slope $\eta$",
    "mag_F555W": "F555W Magnitude",
}[cbar_quantity]

take_cbar_log = {
    "peak_pixel_value_true": True,
    "power_law_slope_true": False,
    "mag_F555W": False,
}[cbar_quantity]

cmap = {
    "peak_pixel_value_true": cmocean.cm.haline_r,
    "power_law_slope_true": cmocean.cm.thermal_r,
    "mag_F555W": cmocean.cm.haline,
}[cbar_quantity]

cut_percent = {
    "peak_pixel_value_true": 20,
    "power_law_slope_true": 15,
    "mag_F555W": 20,
}[cbar_quantity]

cmap = cmocean.tools.crop_by_percent(cmap, cut_percent, "both")

# then make the colorbar
cbar_values = sorted(np.unique(catalog[cbar_quantity]))
if take_cbar_log:
    cbar_values = np.log10(cbar_values)
# Assume values are equally spaced.
diff = cbar_values[1] - cbar_values[0]
boundaries = np.arange(cbar_values[0] - 0.5 * diff, cbar_values[-1] + 0.51 * diff, diff)
norm = colors.BoundaryNorm(boundaries, ncolors=256)
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

# figure size is optimized to make the axes line up properly while also using
# equal_scale on the main comparison axis
fig = plt.figure(figsize=[8, 8.35])
gs = gridspec.GridSpec(
    nrows=2,
    ncols=2,
    wspace=0.1,
    hspace=0,
    height_ratios=[3, 1],
    width_ratios=[20, 1],
    top=0.97,
    right=0.87,
    left=0.12,
    bottom=0.1,
)
# have two axes: one for the comparison, and one for the ratio
ax_c = fig.add_subplot(gs[0, 0], projection="bpl")
ax_r = fig.add_subplot(gs[1, 0], projection="bpl")

mew = 3
good_size = 5
bad_size = 9
for good_fit, symbol, size in zip([True, False], ["o", "x"], [good_size, bad_size]):
    for v in cbar_values:
        color = mappable.to_rgba(v)
        # get the clusters that have this value and fit quality
        cat_values = catalog[cbar_quantity]
        if take_cbar_log:
            cat_values = np.log10(cat_values)
        v_mask = cat_values == v
        fit_mask = catalog["reliable_radius"] == good_fit
        mask = np.logical_and(v_mask, fit_mask)

        ax_c.errorbar(
            reff_true[mask],
            reff[mask],
            yerr=[
                catalog["r_eff_pixels_e-"][mask],
                catalog["r_eff_pixels_e+"][mask],
            ],
            fmt=symbol,
            alpha=1,
            markersize=size,
            markeredgewidth=mew,
            markeredgecolor=color,
            color=color,
        )
        # only plot the good fits in the ratio plot
        if good_fit:
            ax_r.errorbar(
                reff_true[mask],
                reff_ratio[mask],
                yerr=[
                    catalog["r_eff_ratio_e-"][mask],
                    catalog["r_eff_ratio_e+"][mask],
                ],
                fmt=symbol,
                alpha=1,
                markersize=size,
                markeredgewidth=mew,
                markeredgecolor=color,
                color=color,
            )
# one to one line and horizontal line for ratio of 1
ax_c.plot([1e-5, 100], [1e-5, 100], ls=":", c=bpl.almost_black, zorder=0)
ax_r.axhline(1, ls=":", lw=3)

# line showing the PSF.
ax_r.plot([psf_r_eff, psf_r_eff], [2, 100], ls="--", lw=3, c=bpl.almost_black)
ax_c.plot([psf_r_eff, psf_r_eff], [0.019, 0.04], ls="--", lw=3, c=bpl.almost_black)
ax_c.add_text(psf_r_eff, 0.04, "PSF Size", va="bottom", ha="center", fontsize=17)

# fake symbols for legend
ax_c.errorbar(
    [0],
    [0],
    marker="o",
    markersize=good_size,
    markeredgewidth=mew,
    c=bpl.almost_black,
    label="Success",
)
ax_c.errorbar(
    [0],
    [0],
    marker="x",
    markersize=bad_size,
    markeredgewidth=mew,
    c=bpl.almost_black,
    label="Failure",
)
# plot formatting. Some things common to both axes
for ax in [ax_c, ax_r]:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(axis="x", direction="in", which="both")
    ax.tick_params(axis="y", direction="in", which="both")
    ax.xaxis.set_major_formatter(nice_log_formatter)
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")

# then formatting for each axis separately
x_limits = 0.03, 5
ax_c.set_limits(*x_limits, *x_limits)
ax_c.equal_scale()
ax_c.add_labels("", "Measured $R_{eff}$ [pixels]")
ax_c.yaxis.set_major_formatter(nice_log_formatter)
ax_c.set_xticklabels([])
ax_c.legend(loc=2)

ax_r.set_limits(*x_limits, 1 / 3, 3)
ax_r.add_labels("True $R_{eff}$ [pixels]", "$R_{eff}$ Ratio")
ax_r.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0])
ax_r.set_yticklabels(["", "", "0.5", "", "", "", "", "1", "2", ""])

# the colorbar gets its own axis
cax = fig.add_subplot(gs[:, 1], projection="bpl")
cbar = fig.colorbar(mappable, cax=cax)
cbar.set_label(cbar_name)
cbar.set_ticks(cbar_values)

fig.savefig(plot_name)

# ======================================================================================
#
# Have another plot to compare each of the parameters
#
# ======================================================================================
# I'll have several things that need to be tracked for each parameter
params_to_compare = {
    "scale_radius_pixels": "Scale Radius [pixels]",
    "axis_ratio": "Axis Ratio",
    "position_angle": "Position Angle",
    "power_law_slope": "$\eta$ (Power Law Slope)",
}
param_limits = {
    "scale_radius_pixels": (0.05, 20),
    "axis_ratio": (-0.05, 1.05),
    "position_angle": (0, np.pi),
    "power_law_slope": (0, 3),
}
param_scale = {
    "scale_radius_pixels": "log",
    "axis_ratio": "linear",
    "position_angle": "linear",
    "power_law_slope": "linear",
}

# then plot
fig, axs = bpl.subplots(ncols=2, nrows=2, figsize=[12, 12])
axs = axs.flatten()
plot_colors = [
    mappable.to_rgba(np.log10(v)) if take_cbar_log else mappable.to_rgba(v)
    for v in catalog[cbar_quantity]
]

for p, ax in zip(params_to_compare, axs):
    ax.scatter(
        catalog[p + "_true"],
        catalog[p],
        alpha=1,
        c=plot_colors,
    )

    ax.plot([0, 1e10], [0, 1e10], ls=":", c=bpl.almost_black, zorder=0)
    name = params_to_compare[p]
    ax.add_labels(f"True {name}", f"Measured {name}")
    ax.set_xscale(param_scale[p])
    ax.set_yscale(param_scale[p])
    ax.set_limits(*param_limits[p], *param_limits[p])
    ax.equal_scale()
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label(cbar_name)
    cbar.set_ticks(cbar_values)

fig.savefig(plot_name.parent / "artificial_tests_params.pdf")
