import numpy as np
from matplotlib import colors, ticker
import betterplotlib as bpl

bpl.set_style()

# set up colors to use for age split
color_young = bpl.color_cycle[0]
color_med = "#729788"
color_old = "#9F4144"
age_colors = [color_young, color_med, color_old]

# ======================================================================================
#
# Plotting functionality - percentiles
#
# ======================================================================================
def get_r_percentiles(radii, masses, percentile, d_log_M):
    bins = np.logspace(0, 7, int(5 / d_log_M) + 1)

    bin_centers = []
    radii_percentiles = []
    for idx in range(len(bins) - 1):
        lower = bins[idx]
        upper = bins[idx + 1]

        # then find all clusters in this mass range
        mask_above = masses > lower
        mask_below = masses < upper
        mask_good = np.logical_and(mask_above, mask_below)

        good_radii = radii[mask_good]
        if len(good_radii) > 20:
            radii_percentiles.append(np.percentile(good_radii, percentile))
            # the bin centers will be the mean in log space
            bin_centers.append(10 ** np.mean([np.log10(lower), np.log10(upper)]))

    return bin_centers, radii_percentiles


def get_r_percentiles_moving(radii, masses, percentile, n, dn):
    # go through the masses in sorted order
    idxs_sort = np.argsort(masses)
    # then go through chunks of them at a time to get the medians
    masses_median = []
    radii_percentiles = []
    for left_idx in range(0, len(radii) - dn, dn):
        right_idx = left_idx + n
        # fix the last bin
        if right_idx > len(idxs_sort):
            right_idx = len(idxs_sort)
            left_idx = right_idx - n

        idxs = idxs_sort[left_idx:right_idx]
        this_masses = masses[idxs]
        this_radii = radii[idxs]

        masses_median.append(np.median(this_masses))
        radii_percentiles.append(np.percentile(this_radii, percentile))
    return masses_median, radii_percentiles


def get_r_percentiles_hybrid(radii, masses, percentile, d_log_M):
    log_m_min = 1.1
    log_m_max = 6
    n_points = 1 + int((log_m_max - log_m_min) / d_log_M)
    bins = np.logspace(log_m_min, log_m_max, n_points)

    bin_centers = []
    radii_percentiles = []
    # I'll manually manipulate the index when going through the loop to get enough
    # clusters in each bin
    idx_lo = 0
    idx_hi = 1
    while idx_hi < len(bins):
        lower = bins[idx_lo]
        upper = bins[idx_hi]

        # then find all clusters in this mass range
        mask_above = masses > lower
        mask_below = masses < upper
        mask_good = np.logical_and(mask_above, mask_below)

        good_radii = radii[mask_good]
        # if there are very few points, move both lo and hi up
        if len(good_radii) < 5:
            idx_lo = idx_hi
            idx_hi = idx_lo + 1
        # if there aren't too many, just move the top limit to get more
        if len(good_radii) < 40:
            idx_hi += 1
        # if there is a decent number, plot them
        else:
            radii_percentiles.append(np.percentile(good_radii, percentile))
            # the bin centers will be the mean in log space
            bin_centers.append(10 ** np.mean([np.log10(lower), np.log10(upper)]))
            # then move the indices along
            idx_lo = idx_hi
            idx_hi = idx_lo + 1

    return bin_centers, radii_percentiles


def get_r_percentiles_unique_values(radii, ages, percentile):
    # get the unique ages
    unique_ages = np.unique(ages)
    # cut off values above 1e9
    unique_ages = unique_ages[unique_ages <= 1e9]
    radii_percentiles = []
    for age in unique_ages:
        mask = ages == age
        radii_percentiles.append(np.percentile(radii[mask], percentile))
    return unique_ages, radii_percentiles


def add_percentile_lines(
    ax,
    mass,
    r_eff,
    style="hybrid",
    color=bpl.almost_black,
    lw_50=3,
    percentiles=None,
    label_percents=True,
    label_legend=None,
    label_percent_fontsize=16,
):
    # set the percentiles to choose
    if percentiles is None:
        percentiles = [5, 25, 50, 75, 95]
    # plot the median and the IQR
    for percentile in percentiles:
        if style == "moving":
            mass_bins, radii_percentile = get_r_percentiles_moving(
                r_eff, mass, percentile, 200, 200
            )
        elif style == "hybrid":
            mass_bins, radii_percentile = get_r_percentiles_hybrid(
                r_eff, mass, percentile, 0.1
            )
        elif style == "unique":
            mass_bins, radii_percentile = get_r_percentiles_unique_values(
                r_eff, mass, percentile
            )
        elif style == "fixed_width":
            mass_bins, radii_percentile = get_r_percentiles(
                r_eff, mass, percentile, 0.1
            )
        else:
            raise ValueError("Style not recognized")
        ax.plot(
            mass_bins,
            radii_percentile,
            c=color,
            lw=lw_50 * (1 - (abs(percentile - 50) / 50)) + 0.5,
            zorder=9,
            label=label_legend,
        )
        if label_percents:
            ax.text(
                x=mass_bins[0],
                y=radii_percentile[0],
                ha="right",
                va="center",
                s=percentile,
                fontsize=label_percent_fontsize,
                zorder=100,
            )


# ======================================================================================
#
# Plotting functionality - psfs
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


def add_psfs_to_plot(ax, x_max=1e6):
    # then add all the PSF widths. Here we load the PSF and directly measure it's R_eff,
    # so we can have a fair comparison to the clusters
    raise NotImplementedError
    # for cat_loc in sys.argv[5:]:
    #     size_home_dir = Path(cat_loc).parent
    #     home_dir = size_home_dir.parent
    #
    #     psf_name = (
    #         f"psf_"
    #         f"{psf_source}_stars_"
    #         f"{psf_width}_pixels_"
    #         f"{oversampling_factor}x_oversampled.fits"
    #     )
    #
    #     psf = fits.open(size_home_dir / psf_name)["PRIMARY"].data
    #     psf_size_pixels = measure_psf_reff(psf)
    #     psf_size_arcsec = utils.pixels_to_arcsec(psf_size_pixels, home_dir)
    #     psf_size_pc = utils.arcsec_to_pc_with_errors(
    #         home_dir, psf_size_arcsec, 0, 0, False
    #     )[0]
    #     ax.plot(
    #         [0.7 * x_max, x_max],
    #         [psf_size_pc, psf_size_pc],
    #         lw=1,
    #         c=bpl.almost_black,
    #         zorder=3,
    #     )


# ======================================================================================
#
# Plotting functionality - fit lines
#
# ======================================================================================
def plot_best_fit_line(
    ax,
    best_fit_params,
    fit_mass_lower_limit=1,
    fit_mass_upper_limit=1e6,
    color=bpl.color_cycle[1],
    fill=True,
    label=None,
    label_intrinsic_scatter=False,
    ls="-",
):
    # first convert the pivot point value into the intercept
    pivot_point_x = 4
    # so params[1] is really y(pivot_point_x) = m (pivot_point_x) + intercept
    intercept = best_fit_params[1] - best_fit_params[0] * pivot_point_x

    # Make the string that will be used for the label
    if label is None:
        label = "$R_{eff} = "
        label += f"{10**best_fit_params[1]:.2f}"
        label += "\left( \\frac{M}{10^4 M_\odot} \\right)^{"
        label += f"{best_fit_params[0]:.2f}"
        label += "}$"
    elif label == "":
        label = None

    plot_log_masses = np.arange(
        np.log10(fit_mass_lower_limit), np.log10(fit_mass_upper_limit), 0.01
    )
    plot_log_radii = best_fit_params[0] * plot_log_masses + intercept
    ax.plot(
        10 ** plot_log_masses,
        10 ** plot_log_radii,
        c=color,
        lw=4,
        ls=ls,
        zorder=8,
        label=label,
    )
    if fill:
        if label_intrinsic_scatter:
            label_sigma = "$\sigma_{int}$" + f" = {best_fit_params[2]:.2f} dex"
        else:
            label_sigma = None
        ax.fill_between(
            x=10 ** plot_log_masses,
            y1=10 ** (plot_log_radii - best_fit_params[2]),
            y2=10 ** (plot_log_radii + best_fit_params[2]),
            color=color,
            alpha=0.5,
            zorder=0,
            label=label_sigma,
        )

    # Filled in bootstrap interval is currently turned off because the itnerval is smaller
    # than the width of the line
    # # Then add the shaded region of regions allowed by bootstrapping. We'll calculate
    # # the fit line for all the iterations, then at each x value calculate the 68
    # # percent range to shade between.
    # lines = [[] for _ in range(len(plot_log_masses))]
    # for i in range(len(param_history[0])):
    #     this_line = param_history[0][i] * plot_log_masses + param_history[1][i]
    #     for j in range(len(this_line)):
    #         lines[j].append(this_line[j])
    # # Then we can calculate the percentile at each location. The y is in log here,
    # # so scale it back up to regular values
    # upper_limits = [10 ** np.percentile(ys, 84.15) for ys in lines]
    # lower_limits = [10 ** np.percentile(ys, 15.85) for ys in lines]
    #
    # ax.fill_between(
    #     x=10 ** plot_log_masses,
    #     y1=lower_limits,
    #     y2=upper_limits,
    #     zorder=0,
    #     alpha=0.5,
    # )


# ======================================================================================
#
# Plotting functionality - datasets
#
# ======================================================================================
def plot_mass_size_dataset_scatter(
    ax,
    mass,
    mass_err_lo,
    mass_err_hi,
    r_eff,
    r_eff_err_lo,
    r_eff_err_hi,
    color,
    label=None,
    size=3,
):
    ax.scatter(
        mass,
        r_eff,
        alpha=1.0,
        s=size,
        c=color,
        zorder=4 + 1 / np.log10(len(mass)),
        label=label,
    )
    # Have errorbars separately so they can easily be turned off
    ax.errorbar(
        x=mass,
        y=r_eff,
        alpha=1.0,
        markersize=0,
        yerr=[r_eff_err_lo, r_eff_err_hi],
        xerr=[mass_err_lo, mass_err_hi],
        lw=0.1,
        zorder=3 + 1 / np.log10(len(mass)),
        c=color,
    )


def create_color_cmap(hex_color, min_saturation=0.1, max_value=0.8):
    """
    Create a colormap that fades from one color to nearly white.

    This is done by converting the color to HSV, then decreasing the saturation while
    increasing the value (which makes it closer to white)

    :param hex_color: Original starting color, must be in hex format
    :param min_saturation: The saturation of the point farthest from the original color
    :param max_value: The value of the point farthest from the original color
    :return: A matplotilb colormap. Calling it with 0 returns the color specififed
             by `min_saturation` and `max_value` while keeping the same hue, while
             1 will return the original color.
    """
    # convert to HSV (rgb required as an intermediate)
    base_color_rgb = colors.hex2color(hex_color)
    h, s, v = colors.rgb_to_hsv(base_color_rgb)
    N = 256  # number of points in final colormap
    # check that this color is within the range specified
    assert s > min_saturation
    assert v < max_value
    # reduce the saturation and up the brightness. Start from the outer values, as these
    # will correspond to 0, while the original color will be 1
    saturations = np.linspace(min_saturation, s, N)
    values = np.linspace(max_value, v, N)
    out_xs = np.linspace(0, 1, N)

    # set up the weird format required by LinearSegmentedColormap
    cmap_dict = {"red": [], "blue": [], "green": []}
    for idx in range(N):
        r, g, b = colors.hsv_to_rgb((h, saturations[idx], values[idx]))
        out_x = out_xs[idx]
        # LinearSegmentedColormap requires a weird format. I don't think the difference
        # in the last two values matters, it seems to work fine without it.
        cmap_dict["red"].append((out_x, r, r))
        cmap_dict["green"].append((out_x, g, g))
        cmap_dict["blue"].append((out_x, b, b))
    return colors.LinearSegmentedColormap(hex_color, cmap_dict, N=256)


def plot_mass_size_dataset_contour(
    ax,
    mass,
    r_eff,
    color,
    zorder=5,
    cmap_min_saturation=0.1,
    cmap_max_value=0.8,
    levels=None,
    alpha=0.6,
):
    # if the user did not specify levels, use 50 and 90 as the default
    if levels is None:
        levels = [0.5, 0.9]

    cmap = create_color_cmap(color, cmap_min_saturation, cmap_max_value)
    # use median errors as the smoothing. First get mean min and max of all clusters,
    # then take the median of that
    # k = 1.25
    # x_smoothing = k * np.median(np.mean([log_r_eff_err_lo, log_r_eff_err_hi], axis=0))
    # y_smoothing = k * np.median(np.mean([log_mass_err_lo, log_mass_err_hi], axis=0))
    x_smoothing = 0.08
    y_smoothing = 0.08

    common = {
        "percent_levels": levels,
        "smoothing": [x_smoothing, y_smoothing],  # dex
        "bin_size": 0.01,  # dex
        "log": True,
        "cmap": cmap,
    }
    ax.density_contourf(mass, r_eff, alpha=alpha, zorder=zorder, **common)
    ax.density_contour(mass, r_eff, zorder=zorder + 1, **common)


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
        return f"$10^{exp:.0f}$"


def format_mass_size_plot(ax, xmin=10**1.5, xmax=1e6, legend_fontsize=18):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_limits(xmin, xmax, 0.1, 40)
    ax.add_labels("Cluster Mass [M$_\odot$]", "Cluster Effective Radius [pc]")
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_major_formatter(nice_log_formatter)
    ax.yaxis.set_major_formatter(nice_log_formatter)
    # determine where to put the legend
    legend = ax.legend(loc=2, frameon=False, fontsize=legend_fontsize)
    # make the points in the legend larger
    for handle in legend.legendHandles:
        # if it's not a handle for a scatter plot, setting the sizes will fail
        try:
            handle.set_sizes([50])
        except:
            continue
