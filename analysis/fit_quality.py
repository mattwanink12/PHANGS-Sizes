"""
fit_quality.py - Make a plot investigating the fit quality

This takes the following parameters
- Path where this plot will be saved
- The name of the run
- All the completed catalogs (not the public catalog, since we have some quantities
  that we want to investigate here that won't be in that catalog)
"""
import sys
from pathlib import Path

import numpy as np
from astropy import table
from matplotlib import pyplot as plt
from matplotlib import gridspec, colors, cm
import cmocean
import colorcet
import betterplotlib as bpl

bpl.set_style()

# ======================================================================================
#
# Load the catalogs that were passed in
#
# ======================================================================================
plot_name = Path(sys.argv[1]).resolve()
# run name is the second argument, parse it a bit
run_name = sys.argv[2]
run_name = run_name.replace("_", " ").title()

catalogs = [table.Table.read(item, format="ascii.ecsv") for item in sys.argv[3:]]
# then stack them together in one master catalog
big_catalog = table.vstack(catalogs, join_type="inner")
# multiply the MAD by 100 to get it to percent
# big_catalog["profile_mad"] *= 100

success_mask = big_catalog["reliable_radius"].data
n_good = np.sum(success_mask)

# ======================================================================================
#
# Plot setup
#
# ======================================================================================
# I'll have several things that need to be tracked for each parameter
params = {
    "log_luminosity": "Log Luminosity [e$^-$]",
    "dx_from_snap_center": "X Difference from Center",
    "dy_from_snap_center": "Y Difference from Center",
    "scale_radius_pixels": "Scale Radius [pixels]",
    "axis_ratio": "Axis Ratio",
    "position_angle": "Position Angle",
    "power_law_slope": "$\eta$ (Power Law Slope)",
    "local_background": "Local Background [e$^-$]",
}
plot_params = ["scale_radius_pixels", "axis_ratio", "power_law_slope"]
param_limits = {
    "log_luminosity": (1, 8),
    "dx_from_snap_center": (-2.5, 2.5),
    "dy_from_snap_center": (-2.5, 2.5),
    "scale_radius_pixels": (0.05, 20),
    "axis_ratio": (-0.05, 1.05),
    "position_angle": (0, np.pi),
    "power_law_slope": (0, 3),
    "local_background": (-500, 1000),
}
# put things on these limits
for param in param_limits:
    big_catalog[param][big_catalog[param] < param_limits[param][0]] = (
        1.01 * param_limits[param][0]
    )
    big_catalog[param][big_catalog[param] > param_limits[param][1]] = (
        0.99 * param_limits[param][1]
    )
param_scale = {
    "log_luminosity": "linear",
    "dx_from_snap_center": "linear",
    "dy_from_snap_center": "linear",
    "scale_radius_pixels": "log",
    "axis_ratio": "linear",
    "position_angle": "linear",
    "power_law_slope": "linear",
    "local_background": "linear",
}
param_bins = {
    "log_luminosity": np.arange(1, 8, 0.5),
    "dx_from_snap_center": np.arange(-2.5, 2.5, 0.125),
    "dy_from_snap_center": np.arange(-2.5, 2.5, 0.125),
    "scale_radius_pixels": np.logspace(-2, 2, 41),
    "axis_ratio": np.arange(-0.1, 1.1, 0.05),
    "position_angle": np.arange(0, 3.5, 0.1),
    "power_law_slope": np.arange(0, 5, 0.1),
    "local_background": np.arange(-300, 1500, 100),
}

indicators = {
    "fit_rms": "Fit RMS",
    "profile_diff_reff": "Deviation of\nCumulative Profile",
    "estimated_local_background_diff_sigma": "Estimated Local\nBackground Error",
}
ind_limits = {
    "fit_rms": (0.1, 100),
    "profile_diff_reff": (1e-3, 1),
    "estimated_local_background_diff_sigma": (-3, 3),
}
ind_scale = {
    "fit_rms": "log",
    "profile_diff_reff": "log",
    "estimated_local_background_diff_sigma": "linear",
}


def make_cumulative_histogram(values):
    """
    Create the line to be plotted for a cumulative histogram

    :param values: data
    :return: List of xs and ys to be plotted for the cumulative histogram
    """
    sorted_values = np.sort(values)
    ys = np.arange(1, 1 + len(sorted_values), 1)
    assert len(ys) == len(sorted_values)
    return sorted_values, ys


def get_percentiles(xs, ys, percentile, bins, bin_scale):
    bin_centers = []
    ys_percentiles = []

    # throw out nans
    mask = ~np.isnan(ys)
    xs = xs[mask]
    ys = ys[mask]
    for idx in range(len(bins) - 1):
        lower = bins[idx]
        upper = bins[idx + 1]

        # then find all clusters in this mass range
        mask_above = xs > lower
        mask_below = xs <= upper
        mask_good = np.logical_and(mask_above, mask_below)

        good_ys = ys[mask_good]
        if len(good_ys) > 0:
            ys_percentiles.append(np.percentile(good_ys, percentile))
            # the bin centers will be the mean in log space
            if bin_scale == "log":
                bin_center = 10 ** np.mean([np.log10(lower), np.log10(upper)])
            else:
                bin_center = np.mean([lower, upper])
            bin_centers.append(bin_center)

    return bin_centers, ys_percentiles


success_color = bpl.color_cycle[0]
failure_color = bpl.color_cycle[3]

# ======================================================================================
#
# Simple plot of cumulative histograms
#
# ======================================================================================
# This will have several columns for different parameters, with the rows being the
# different ways of assessing each parameter
fig, axs = bpl.subplots(
    ncols=3,
    figsize=[16, 6],
    tight_layout=False,
    gridspec_kw={"top": 0.9, "bottom": 0.2, "left": 0.08, "right": 0.98},
)
for ax, param in zip(axs, plot_params):
    # Then the cumulative histogram
    ax.plot(
        *make_cumulative_histogram(big_catalog[param][success_mask]),
        color=success_color,
        lw=2,
        label=f"Success (N={n_good:,})",
    )
    ax.plot(
        *make_cumulative_histogram(big_catalog[param][~success_mask]),
        color=failure_color,
        lw=2,
        label=f"Failure (N={len(big_catalog) - n_good:,})",
    )

    if param == "axis_ratio":
        ax.legend(loc=2)
    ax.add_labels(x_label=params[param])
    if param == "scale_radius_pixels":
        ax.add_labels(y_label="Cumulative Number of Clusters")
        ax.axvline(0.1, ls=":")
        ax.axvline(15, ls=":")
    ax.set_limits(*param_limits[param], 0, 5000)
    ax.set_xscale(param_scale[param])

    # set ticks on top and bottom
    ax.tick_params(
        axis="both",
        top=True,
        bottom=True,
        left=True,
        right=True,
        which="both",
        direction="out",
    )
fig.suptitle(run_name)
fig.savefig(plot_name.parent / "cumulative.png")

# ======================================================================================
#
# Then make the plot
#
# ======================================================================================
# This will have several columns for different parameters, with the rows being the
# different ways of assessing each parameter
fig = plt.figure(figsize=[6 * len(plot_params), 4 * (2 + len(indicators))])
gs = gridspec.GridSpec(
    nrows=len(indicators) + 2,
    ncols=len(plot_params),
    wspace=0.2,
    hspace=0.13,
    left=0.1,
    right=0.98,
    bottom=0.06,
    top=0.96,
)

# Then go through and make the columns
for idx_p, param in enumerate(plot_params):
    # add the histogram
    ax = fig.add_subplot(gs[0, idx_p], projection="bpl")
    ax.hist(
        big_catalog[param][success_mask],
        bins=param_bins[param],
        histtype="step",
        color=success_color,
        lw=2,
        label=f"Success (N={n_good:,})",
    )
    ax.hist(
        big_catalog[param][~success_mask],
        bins=param_bins[param],
        histtype="step",
        color=failure_color,
        lw=2,
        label=f"Failure (N={len(big_catalog) - n_good:,})",
    )

    if idx_p == 1:
        ax.set_title(f"{run_name}\n{params[param]}")
    else:
        ax.set_title(params[param])
    if idx_p == 0:
        ax.add_labels(y_label="Number of Clusters")
    ax.set_limits(*param_limits[param])
    ax.set_xscale(param_scale[param])
    if param == "axis_ratio":
        ax.legend(loc=2)
    # set ticks on top and bottom
    ax.tick_params(
        axis="both",
        top=True,
        bottom=True,
        left=True,
        right=True,
        which="both",
        direction="out",
    )

    # Then the cumulative histogram
    ax = fig.add_subplot(gs[1, idx_p], projection="bpl")
    ax.plot(
        *make_cumulative_histogram(big_catalog[param][success_mask]),
        color=success_color,
        lw=2,
    )
    ax.plot(
        *make_cumulative_histogram(big_catalog[param][~success_mask]),
        color=failure_color,
        lw=2,
    )

    if idx_p == 0:
        ax.add_labels(y_label="Cumulative Number of Clusters")
    ax.set_limits(*param_limits[param], 0)
    ax.set_xscale(param_scale[param])
    # set ticks on top and bottom
    ax.tick_params(
        axis="both",
        top=True,
        bottom=True,
        left=True,
        right=True,
        which="both",
        direction="out",
    )

    # then the indicator plots
    for idx_q, indicator in enumerate(indicators, start=2):
        ax = fig.add_subplot(gs[idx_q, idx_p], projection="bpl")

        ax.scatter(
            big_catalog[param][success_mask],
            big_catalog[indicator][success_mask],
            c=success_color,
            alpha=1,
            s=1,
            zorder=2,
        )
        ax.scatter(
            big_catalog[param][~success_mask],
            big_catalog[indicator][~success_mask],
            c=failure_color,
            alpha=1,
            s=1,
            zorder=2,
        )

        # Draw the percentile lines
        for percentile in [5, 25, 50, 75, 95]:
            xs, ys = get_percentiles(
                big_catalog[param][success_mask],
                big_catalog[indicator][success_mask],
                percentile,
                param_bins[param],
                param_scale[param],
            )
            ax.plot(
                xs,
                ys,
                c=bpl.almost_black,
                lw=1.5 * (1 - (abs(percentile - 50) / 50)) + 0.5,
                zorder=1,
            )
            ax.text(
                x=xs[1],
                y=ys[1],
                ha="center",
                va="bottom",
                s=percentile,
                fontsize=16,
            )

        ax.set_limits(*param_limits[param], *ind_limits[indicator])
        # remove the X label and ticks for all but the last plot
        if idx_q == 4:
            # ax.remove_labels("x")
            ax.add_labels(x_label=params[param])
        if idx_p == 0:
            ax.add_labels(y_label=indicators[indicator])
        # set ticks on top and bottom
        ax.tick_params(
            axis="both",
            top=True,
            bottom=True,
            left=True,
            right=True,
            which="both",
            direction="out",
        )
        ax.set_xscale(param_scale[param])
        ax.set_yscale(ind_scale[indicator])

        if "background" in indicator:
            ax.axhline(0, ls=":", c=bpl.almost_black)


fig.savefig(plot_name)

# ======================================================================================
#
# Then make a plot showing how these parameters relate to one another
#
# ======================================================================================
fig, axs = bpl.subplots(ncols=3, figsize=[20, 6])
axs = axs.flatten()

x_param = "power_law_slope"
y_param = "scale_radius_pixels"
for ax, color_ind in zip(axs, indicators):

    if color_ind == "profile_mad":
        cmap = cmocean.cm.deep
        norm = colors.LogNorm(vmin=0.005, vmax=0.1)
    elif color_ind == "fit_rms":
        cmap = cmocean.cm.rain
        norm = colors.LogNorm(vmin=0.5, vmax=10)
    else:
        cmap = bpl.cm.viridis
        norm = colors.Normalize(vmin=-1, vmax=1)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    point_colors = np.array([mappable.to_rgba(c) for c in big_catalog[color_ind]])
    # # turn failures red
    # for idx in range(len(point_colors)):
    #     if not success_mask[idx]:
    #         point_colors[idx] = "red"

    ax.scatter(
        big_catalog[x_param][success_mask],
        big_catalog[y_param][success_mask],
        c=point_colors[success_mask],
        s=2,
        alpha=1,
    )
    a_grid = np.logspace(-1, 1, 12)
    eta_grid = np.arange(1.1, 3.0, 0.2)
    a_values = []
    eta_values = []
    for a in a_grid:
        for eta in eta_grid:
            a_values.append(a)
            eta_values.append(eta)
    n_grid = len(a_values)

    ax.scatter(
        eta_values, a_values, s=30, marker="o", alpha=1, c=bpl.color_cycle[3], zorder=0
    )

    ax.add_labels(params[x_param], params[y_param])
    ax.set_limits(*param_limits[x_param], *param_limits[y_param])
    ax.set_xscale(param_scale[x_param])
    ax.set_yscale(param_scale[y_param])
    cbar = fig.colorbar(mappable, ax=ax, pad=0)
    cbar.set_label(indicators[color_ind])
    ax.axhline(0.1, ls=":")
fig.savefig(plot_name.parent / "param_correlation.png")

# ======================================================================================
#
# Make a corner plot showing the correlation between all parameters
#
# ======================================================================================
fig, axs = bpl.subplots(
    nrows=len(params),
    ncols=len(params),
    figsize=[5 * len(params), 5 * len(params)],
    sharex=False,
    sharey=False,
    gridspec_kw={"hspace": 0, "wspace": 0},
)
for idx_row, param_row in enumerate(params):
    for idx_col, param_col in enumerate(params):
        # get the axis at this location
        ax = axs[idx_row][idx_col]
        # If the labels are the same, plot the histogram
        if param_col == param_row:
            ax.hist(
                big_catalog[param_col][success_mask],
                bins=param_bins[param_col],
                histtype="step",
                lw=2,
                color=success_color,
            )
            ax.hist(
                big_catalog[param_col][~success_mask],
                bins=param_bins[param_col],
                histtype="step",
                lw=2,
                color=failure_color,
            )
            ax.set_limits(*param_limits[param_col])
            ax.set_xscale(param_scale[param_col])

        # Then plot the scatterplots
        elif idx_col < idx_row:
            ax.scatter(
                big_catalog[param_col][success_mask],
                big_catalog[param_row][success_mask],
                s=1,
                c=success_color,
                alpha=1,
            )
            ax.scatter(
                big_catalog[param_col][~success_mask],
                big_catalog[param_row][~success_mask],
                s=1,
                c=failure_color,
                alpha=1,
            )
            ax.set_limits(*param_limits[param_col], *param_limits[param_row])
            ax.set_xscale(param_scale[param_col])
            ax.set_yscale(param_scale[param_row])
        else:
            # make a 2d histogram
            hist, x_edges, y_edges = np.histogram2d(
                big_catalog[param_col],
                big_catalog[param_row],
                bins=[param_bins[param_col], param_bins[param_row]],
            )
            hist = hist.transpose()
            ax.pcolormesh(x_edges, y_edges, hist, cmap="Greys")
            ax.set_limits(*param_limits[param_col], *param_limits[param_row])
            ax.set_xscale(param_scale[param_col])
            ax.set_yscale(param_scale[param_row])

        # move all ticks to the inside
        ax.tick_params(axis="x", top=True, bottom=True, direction="in")
        ax.tick_params(axis="y", left=True, right=True, direction="in")
        # Then handle where the labels go. If we're in the left column, show the label
        if idx_col == 0:
            ax.add_labels(y_label=params[param_row])
        # put the labels on the right if we're on the last one
        elif idx_col == len(params) - 1:
            ax.add_labels(y_label=params[param_row])
            ax.tick_params(axis="y", labelleft=False, labelright=True)
            ax.yaxis.set_label_position("right")
        else:
            ax.yaxis.set_ticklabels([])

        # add the X label if we're in the bottom row
        if idx_row == len(params) - 1:
            ax.add_labels(x_label=params[param_col])
        elif idx_row == 0:  # put the label on top for the top row
            ax.add_labels(x_label=params[param_col])
            ax.tick_params(axis="x", labelbottom=False, labeltop=True)
            ax.xaxis.set_label_position("top")
        # if we're not on the bottom row, don't show the x labels
        else:
            ax.xaxis.set_ticklabels([])


fig.savefig(plot_name.parent / "corner.png", dpi=100)

# for row in big_catalog:
#     if (
#         row["scale_radius_pixels"] < 1e-6
#         # and row["power_law_slope"] > 1
#         # and row["profile_mad"] < 1
#     ):
#         print(row["galaxy"], row["ID"])
