"""
psf_compare.py - Compare the radial profiles of the stars used to make the PSF

This script takes the following command line arguments:
- The path to the 'size' directory for a given cluster. This script will find all the
  files it needs within that folder.
- Oversampling factor for the psf.
- Size used in the cluster snapshots. Used to determine which PSFs to load.
"""

import sys
from pathlib import Path

import betterplotlib as bpl
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import gridspec
import numpy as np
from astropy.nddata import NDData
from astropy import table
from astropy.io import fits
from astropy.table import Table

# need to add the correct path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent / "pipeline"))
import utils

bpl.set_style()

plot_path = Path(sys.argv[1]).resolve()
size_home_dir = plot_path.parent
home_dir = size_home_dir.parent
oversampling_factor = int(sys.argv[2])
psf_width = int(sys.argv[3])

# ======================================================================================
#
# Load the data - image and star catalog
#
# ======================================================================================
band_select = sys.argv[4] # edit this here to get new data
bands = utils.get_drc_image(home_dir)
image_data = bands[band_select][0]
#image_data, _, _ = utils.get_drc_image(home_dir)
# the extract_stars thing below requires the input as a NDData object
nddata = NDData(data=image_data)

bg = Table.read(size_home_dir / "Background.txt", format="ascii.no_header")

# load the input star list.
def get_star_list_and_psf(source):
    name_base = f"_{source}_stars_{psf_width}_pixels_{oversampling_factor}x_oversampled"

    table_name = "psf_star_centers" + name_base + ".txt"
    psf_name = "psf" + name_base + ".fits"

    star_table = table.Table.read(str(size_home_dir / table_name), format="ascii.ecsv")
    psf = fits.open(size_home_dir / psf_name)["PRIMARY"].data

    return star_table, psf


star_table_me, psf_me = get_star_list_and_psf("my")
#star_table_legus, psf_legus = get_star_list_and_psf("legus")

# ======================================================================================
#
# Plot the psfs
#
# ======================================================================================
cmap = bpl.cm.viridis
cmap.set_bad(cmap(0))  # for negative values in log plot


def visualize_psf(fig, ax, psf_data, scale="log"):
    """
    Make a 2D visualization of the PSF. It can be either log or linear scaled.
    """
    vmax = 0.035
    if scale == "log":
        norm = colors.LogNorm(vmin=1e-6, vmax=vmax)
    elif scale == "linear":
        norm = colors.Normalize(vmin=0, vmax=vmax)

    im_data = ax.imshow(psf_data, norm=norm, cmap=cmap, origin="lower")
    fig.colorbar(im_data, ax=ax, pad=0)

    ax.remove_labels("both")
    # ax.remove_spines(["all"])


#for psf, star_source in zip([psf_legus, psf_me], ["legus", "my"]):
for psf, star_source in zip([psf_me], ["my"]):
    fig, axs = bpl.subplots(
        ncols=2,
        figsize=[12, 5],
        tight_layout=False,
        gridspec_kw={"top": 0.9, "left": 0.05, "right": 0.95, "wspace": 0.1},
    )

    visualize_psf(fig, axs[0], psf, "linear")
    visualize_psf(fig, axs[1], psf, "log")

    # reformat the name
    if star_source == "my":
        plot_title = f"{str(home_dir.name).upper()} - Me"
    else:
        plot_title = f"{str(home_dir.name).upper()} - LEGUS"
    fig.suptitle(plot_title, fontsize=24)

    figname = (
        f"psf_{star_source}_"
        + f"{psf_width}_pixels_"
        + f"{oversampling_factor}x_oversampled.png"
    )
    fig.savefig(size_home_dir / "plots" / figname, bbox_inches="tight")
# ======================================================================================
#
# make the radial profiles and plot
#
# ======================================================================================
# Base this on the radial profile functions I used in the fitting debugging plots
def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def bin(bin_size, xs, ys):
    # then sort them by xs first
    idxs_sort = np.argsort(xs)
    xs = np.array(xs)[idxs_sort]
    ys = np.array(ys)[idxs_sort]

    # then go through and put them in bins
    binned_xs = []
    binned_ys = []
    this_bin_ys = []
    max_bin = bin_size  # start at zero
    for idx in range(len(xs)):
        x = xs[idx]
        y = ys[idx]

        # see if we need a new max bin
        if x > max_bin:
            # store our saved data
            if len(this_bin_ys) > 0:
                binned_ys.append(np.mean(this_bin_ys))
                binned_xs.append(max_bin - 0.5 * bin_size)
            # reset the bin
            this_bin_ys = []
            max_bin = np.ceil(x / bin_size) * bin_size

        assert x <= max_bin
        this_bin_ys.append(y)

    return np.array(binned_xs), np.array(binned_ys)


def radial_profile_psf(ax, psf, color, label):
    # the center is the central pixel of the image
    x_cen = int((psf.shape[1] - 1.0) / 2.0)
    y_cen = int((psf.shape[0] - 1.0) / 2.0)
    # then go through all the pixel values to determine the distance from the center
    radii = []
    values = []
    for x in range(psf.shape[1]):
        for y in range(psf.shape[1]):
            # need to include the oversampling factor in the distance
            radii.append(distance(x, y, x_cen, y_cen) / oversampling_factor)
            values.append(psf[y][x])

    assert np.isclose(np.sum(values), 1.0)
    # then bin then
    radii, values = bin(0.1, radii, values)

    ax.plot(radii, values, c=color, lw=3, label=label)


fig, ax = bpl.subplots()
#c_me = "#9EA4CA"
c_me = colors.CSS4_COLORS
c_legus = "#CD8486"
# first go through all the stars
#for star_table, color in zip([star_table_legus, star_table_me], [c_legus, c_me]):
norm_const_list = []
backgrounds = []
for star_table, color in zip([star_table_me], [c_me]):
    i = 0
    for star in star_table:
        # use the centers given in the tables
        x_cen = star["x_center"]
        y_cen = star["y_center"]
        # then go through all the pixel values to determine the distance from the center
        half_width = psf_width / 2.0
        radii = []
        values = []
        background_values = []
        for x in range(int(x_cen - half_width), int(x_cen + half_width)):
            for y in range(int(y_cen - half_width), int(y_cen + half_width)):
                dist = distance(x, y, x_cen, y_cen)
                radii.append(dist)
                values.append(image_data[y][x])
                if dist > 8.0:
                    background_values.append(image_data[y][x])

        values = np.array(values)
        #print(values)
        # background subtract
        values -= np.median(background_values)
        #print(values)
        # normalize the profile
        norm_const = np.sum(values)
        norm_const_list.append(norm_const)
        values /= np.sum(values)
        # then make the normalization match the PSF normalization, which is off by a
        # factor of oversampling_factor**2
        values /= oversampling_factor ** 2
        # then bin then
        radii, values = bin(0.2, radii, values)
        
        #print(np.sum(values))
        local_back = np.median(background_values)#bg["col1"][i]
        backgrounds.append(local_back)
        local_back /= norm_const #np.sum(background_values)
        local_back /= oversampling_factor ** 2
        local_back = abs(local_back)
        print(local_back)

        ax.plot(radii, values, c=list(color)[i], lw=0.5)
        ax.axhline(y=local_back, color=list(color)[i], lw=0.5)
        i += 1


# Then do the same for the psfs
c_me = bpl.color_cycle[0]
c_legus = bpl.color_cycle[3]

norm_const_list = np.array(norm_const_list)
backgrounds = np.array(backgrounds)
avg_background = np.median(abs(backgrounds)) / np.median(abs(norm_const_list))
avg_background /= oversampling_factor ** 2
print("Avg Background: ", avg_background)
ax.axhline(y=avg_background, color=c_me, lw=2)

#for psf, color, label in zip([psf_legus, psf_me], [c_legus, c_me], ["LEGUS", "Me"]):
for psf, color, label in zip([psf_me], [c_me], ["Me"]):
    radial_profile_psf(ax, psf, color, label)

ax.add_labels("Radius (pixels)", "Normalized Pixel Value", home_dir.name.upper())
ax.set_limits(0, 12, 1e-6, 0.04)
ax.axhline(0, ls=":")
ax.legend()
ax.set_yscale("log")

figname = (
    f"psf_comparison_"
    + f"{psf_width}_pixels_"
    + f"{oversampling_factor}x_oversampled.png"
)
fig.savefig(size_home_dir / "plots" / figname, bbox_inches="tight")

# ======================================================================================
#
# Then have the paper plot - one panel is the PSF, the other is the radial profile
#
# ======================================================================================
# fig = plt.figure(figsize=[6, 10.5])
# gs = gridspec.GridSpec(
#     ncols=20, nrows=20, left=0, right=1, bottom=0.07, top=1, hspace=0, wspace=0
# )
# ax0 = fig.add_subplot(gs[:10, :], projection="bpl")
# ax1 = fig.add_subplot(gs[11:, 4:19], projection="bpl")

fig = plt.figure(figsize=[9, 4])
gs = gridspec.GridSpec(
    ncols=40, nrows=20, left=0.01, right=0.97, bottom=0.03, top=0.99, hspace=0, wspace=0
)
ax0 = fig.add_subplot(gs[:, :22], projection="bpl")
ax1 = fig.add_subplot(gs[:17, 27:], projection="bpl")

visualize_psf(fig, ax0, psf_me, "log")
radial_profile_psf(ax1, psf_me, c_me, "")

ax1.add_labels("Radius (pixels)", "Normalized Pixel Value")
ax1.set_limits(0, 10, 1e-6, 0.035)
ax1.set_yscale("log")
ax1.xaxis.set_ticks([0, 2, 4, 6, 8, 10])

fig.savefig(plot_path)
