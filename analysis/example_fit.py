"""
example_fit.py - Plot an example showing a fitted cluster

This takes the following parameters:
- Path to save the plot
- PSF oversampling factor
- The width of the PSF snapshot
- cluster snapshot size
- Path to the public catalog
"""
import sys
from pathlib import Path

import numpy as np
from astropy import table, nddata
from astropy import units as u
from astropy.io import fits
from scipy import signal
from matplotlib import pyplot as plt
from matplotlib import gridspec, colors
import cmocean
import betterplotlib as bpl

bpl.set_style()

# need to add the correct path to import utils
legus_home_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(legus_home_dir / "pipeline"))
import utils
import fit_utils

# get the location to save this plot
plot_name = Path(sys.argv[1]).resolve()
oversampling_factor = int(sys.argv[2])
psf_size = int(sys.argv[3])
snapshot_size = int(sys.argv[4])
snapshot_size_oversampled = snapshot_size * oversampling_factor
catalog = table.Table.read(sys.argv[5], format="ascii.ecsv")

# add the pixel scale for use later
catalog["pixel_scale"] = catalog["r_eff_arcsec"] / catalog["r_eff_pixels"]
for item in catalog["pixel_scale"]:
    assert np.isclose(item, 39.62e-3, atol=0, rtol=1e-2)

# ======================================================================================
#
# Load the data we need
#
# ======================================================================================
# Use NGC1566 ID 2364
galaxy = "ngc1566"
field = "ngc1566"
cluster_id = 2364

data_dir = legus_home_dir / "data" / galaxy
image_data, _, _ = utils.get_drc_image(data_dir)

error_data = fits.open(data_dir / "size" / "sigma_electrons.fits")["PRIMARY"].data
mask = fits.open(data_dir / "size" / "mask_image.fits")["PRIMARY"].data

psf_name = f"psf_my_stars_{psf_size}_pixels_{oversampling_factor}x_oversampled.fits"
psf = fits.open(data_dir / "size" / psf_name)["PRIMARY"].data
psf_cen = int((psf.shape[1] - 1.0) / 2.0)

# then find the correct row
for row in catalog:
    if row["ID"] == cluster_id and row["galaxy"] == galaxy and row["field"] == field:
        break

# Then get the snapshot of this cluster. Have to subtract one to account for the
# difference in zero vs one indexing between the catalog and Python
x_cen = int(np.ceil(row["x"])) - 1
y_cen = int(np.ceil(row["y"])) - 1

# Get the snapshot, based on the size desired
x_min = x_cen - 15
x_max = x_cen + 15
y_min = y_cen - 15
y_max = y_cen + 15

data_snapshot = image_data[y_min:y_max, x_min:x_max]
error_snapshot = error_data[y_min:y_max, x_min:x_max]
mask_snapshot = mask[y_min:y_max, x_min:x_max]
# Use the same mask region as was used in the actual fitting procedure
mask_snapshot = fit_utils.handle_mask(mask_snapshot, row["ID"])

# then have new centers for the fit within this snapshot. See the code in fit.py to
# correct for the oversampling factor. Also have to correct for indexing here too.
x_cen_snap = row["x"] - x_min - 1
y_cen_snap = row["y"] - y_min - 1
x_cen_snap_oversampled = (x_cen_snap + 0.25) * 2
y_cen_snap_oversampled = (y_cen_snap + 0.25) * 2

# ======================================================================================
#
# Creating the EFF profile
#
# ======================================================================================
# The fit utils uses log luminosity rather than mu_0, which is what is in the output
# file. Convert back to that.
def mu_to_logl(mu_0, eta, a):
    logl_term_a = mu_0 * (np.pi * a ** 2) / (eta - 1)
    logl_term_b = 1 - (1 + (15 / a) ** 2) ** (1 - eta)
    return np.log10(logl_term_a * logl_term_b)


models = fit_utils.create_model_image(
    mu_to_logl(row["mu_0"], row["power_law_slope"], row["scale_radius_pixels"]),
    x_cen_snap_oversampled,
    y_cen_snap_oversampled,
    row["scale_radius_pixels"],
    row["axis_ratio"],
    row["position_angle"],
    row["power_law_slope"],
    row["local_background"],
    psf,
    snapshot_size_oversampled,
    oversampling_factor,
)
model_image, model_psf_image, model_psf_bin_image = models

sigma_snapshot = (data_snapshot - model_psf_bin_image) / error_snapshot
sigma_snapshot *= mask_snapshot
# ======================================================================================
#
# Convenience functions for the plot
#
# ======================================================================================
# These don't really need to be functions, but it makes things cleaner in the plot below


def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def radial_profile(snapshot, oversampling_factor, x_c, y_c, y_at_zero=None):
    radii, ys = [], []
    for x in range(snapshot.shape[1]):
        for y in range(snapshot.shape[0]):
            radii.append(distance(x, y, x_c, y_c) / oversampling_factor)
            ys.append(snapshot[y, x])
    idx_sort = np.argsort(radii)

    radii = np.array(radii)[idx_sort]
    ys = np.array(ys)[idx_sort]
    if radii[0] != 0.0 and y_at_zero is not None:
        radii = np.concatenate([[0], radii])
        ys = np.concatenate([[y_at_zero], ys])

    return radii, ys


def binned_radial_profile(
    snapshot, oversampling_factor, x_c, y_c, bin_size, y_at_0=None
):
    radii, ys = radial_profile(snapshot, oversampling_factor, x_c, y_c, y_at_0)
    # then bin this data to make the binned plot
    binned_radii = [0]
    binned_ys = [ys[0]]
    radii = radii[1:]
    ys = ys[1:]

    for r_min in np.arange(0, int(np.ceil(max(radii))), bin_size):
        r_max = r_min + bin_size
        idx_above = np.where(r_min < radii)
        idx_below = np.where(r_max > radii)
        idx_good = np.intersect1d(idx_above, idx_below)

        if len(idx_good) > 0:
            binned_radii.append(r_min + 0.5 * bin_size)
            binned_ys.append(np.mean(ys[idx_good]))
    return binned_radii, binned_ys


# ======================================================================================
#
# Make the plot
#
# ======================================================================================
# parameters that can be adjusted based on individual clusters
data_cmap_vmin = 50
data_cmap_vmax = 1e3
sigma_cmap_vmax = 3
r_eff_y_max = 630  # how far the R_eff line extends in the y direction to hit the line
r_eff_y_label = 110  # where the R_eff label goes on the y axis.
radial_plot_x_max = 7  # how far to extend the radial plot
radial_plot_y_min = 40
radial_plot_y_max = 2e3
model_ymax = 1500  # extrapolation to r=0
model_psf_ymax = 430  # extrapolation to r=0

# vmax = max(np.max(model_image), np.max(model_psf_image), np.max(data_snapshot))
data_norm = colors.LogNorm(vmin=data_cmap_vmin, vmax=data_cmap_vmax)
sigma_norm = colors.Normalize(vmin=-sigma_cmap_vmax, vmax=sigma_cmap_vmax)
data_cmap = bpl.cm.davos
data_cmap.set_bad(data_cmap(0))
sigma_cmap = cmocean.cm.tarn  # "bwr_r" also works

# This will have the data, model, and residual above the plot
fig = plt.figure(figsize=[15, 7])
gs = gridspec.GridSpec(
    nrows=2,
    ncols=3,
    width_ratios=[2, 1, 1],
    wspace=0.1,
    hspace=0.2,
    left=0.07,
    right=0.97,
    bottom=0.1,
    top=0.9,
)
ax_r = fig.add_subplot(gs[0, 1], projection="bpl")  # raw model
ax_f = fig.add_subplot(gs[1, 1], projection="bpl")  # full model (f for fit)
ax_d = fig.add_subplot(gs[1, 2], projection="bpl")  # data
ax_s = fig.add_subplot(gs[0, 2], projection="bpl")  # sigma difference
ax_big = fig.add_subplot(gs[:, 0], projection="bpl")  # radial profile

r_im = ax_r.imshow(model_image, origin="lower", cmap=data_cmap, norm=data_norm)
f_im = ax_f.imshow(model_psf_bin_image, origin="lower", cmap=data_cmap, norm=data_norm)
d_im = ax_d.imshow(data_snapshot, origin="lower", cmap=data_cmap, norm=data_norm)
s_im = ax_s.imshow(sigma_snapshot, origin="lower", cmap=sigma_cmap, norm=sigma_norm)

r_cbar = fig.colorbar(r_im, ax=ax_r, pad=0)
f_cbar = fig.colorbar(f_im, ax=ax_f, pad=0)
d_cbar = fig.colorbar(d_im, ax=ax_d, pad=0)
s_cbar = fig.colorbar(s_im, ax=ax_s, pad=0)

r_cbar.set_label("          Pixel Value [e$^-$]", fontsize=16, labelpad=0)
f_cbar.set_label("          Pixel Value [e$^-$]", fontsize=16, labelpad=0)
d_cbar.set_label("          Pixel Value [e$^-$]", fontsize=16, labelpad=0)

title_fontsize = 16
ax_r.set_title("Raw Cluster Model", fontsize=title_fontsize)
ax_f.set_title("Model Convolved\nWith PSF", fontsize=title_fontsize)
ax_d.set_title("Data", fontsize=title_fontsize)
ax_s.set_title("(Data - Model)/Uncertainty", fontsize=title_fontsize)

for ax in [ax_r, ax_f, ax_d, ax_s]:
    ax.remove_labels("both")
    ax.remove_spines(["all"])

# Then the radial profiles
ax_big.plot(
    *radial_profile(
        model_image, 2, x_cen_snap_oversampled, y_cen_snap_oversampled, model_ymax
    ),
    label="Raw Cluster Model",
    c=bpl.color_cycle[3],
    lw=4,
)
ax_big.plot(
    *binned_radial_profile(
        model_psf_bin_image, 1, x_cen_snap, y_cen_snap, 0.25, model_psf_ymax
    ),
    label="Model Convolved with PSF",
    c=bpl.color_cycle[0],
    lw=4,
)
ax_big.scatter(
    *radial_profile(data_snapshot, 1, x_cen_snap, y_cen_snap),
    label="Data",
    c=bpl.color_cycle[2],
)

# Normalize the PSF to match the PSF smoothed model
psf *= model_psf_ymax / np.max(psf)
ax_big.plot(
    *binned_radial_profile(psf, 2, psf_cen, psf_cen, 0.1),
    label="PSF",
    c=bpl.color_cycle[1],
    lw=4,
)


ax_big.axhline(row["local_background"], ls=":", label="Local Background")
# ax_big.add_text(
#     x=5.5,
#     y=row["local_background"] * 1.1,
#     text="Local Background",
#     ha="left",
#     va="bottom",
#     fontsize=18,
# )

r_eff = row["r_eff_pixels"]
ax_big.plot([r_eff, r_eff], [0, r_eff_y_max], c=bpl.almost_black, ls="--")
ax_big.add_text(
    x=r_eff - 0.05,
    y=r_eff_y_label,
    text="$R_{eff}$",
    ha="right",
    va="bottom",
    rotation=90,
    fontsize=25,
)

# y_at_rmax = profile_at_radius(15)
# ax_big.plot([15, 15], [0, y_at_rmax], c=bpl.almost_black, ls="--")
# ax_big.add_text(
#     x=15 - 0.05,
#     y=y_at_rmax * 0.8,
#     text="$R_{max}$",
#     ha="right",
#     va="top",
#     rotation=90,
#     fontsize=25,
# )

ax_big.legend()
ax_big.set_yscale("log")
ax_big.add_labels("Radius [pixels]", "Pixel Value [e$^-$]")
ax_big.set_limits(0, radial_plot_x_max, radial_plot_y_min, radial_plot_y_max)

# then add a second scale on top translating into parsecs
plot_limit_arcsec = radial_plot_x_max * row["pixel_scale"] * u.arcsec
plot_limit_pc = plot_limit_arcsec.to(u.radian).value * row["galaxy_distance_mpc"] * 1e6
ax_big.twin_axis_simple("x", lower_lim=0, upper_lim=plot_limit_pc, label="Radius [pc]")

fig.savefig(plot_name)
