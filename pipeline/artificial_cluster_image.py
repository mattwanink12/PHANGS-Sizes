"""
artificial_cluster_image.py
Make an image full of artificial clusters to test the pipeline with
"""

import sys
from pathlib import Path
from astropy import table
from astropy.io import fits
import numpy as np

import utils, fit_utils

# set random seed for reproducibility
np.random.seed(123)

# ======================================================================================
#
# Load the parameters that were passed in
#
# ======================================================================================
image_name = Path(sys.argv[1]).resolve()
true_catalog_name = Path(sys.argv[2]).resolve()
true_catalog = table.Table.read(true_catalog_name, format="ascii.ecsv")

oversampling_factor = int(sys.argv[3])
snapshot_size = 2 * int(sys.argv[4])  # extend further for creation of larger clusters
snapshot_size_oversampled = oversampling_factor * snapshot_size

# load the PSF
psf_path = Path(sys.argv[5]).resolve()
psf = fits.open(psf_path)["PRIMARY"].data
# the convolution requires the psf to be normalized, and without any negative values
psf = np.maximum(psf, 0)
psf /= np.sum(psf)

# then load the image from the suggested galaxy
galaxy = sys.argv[6]
galaxy_dir = image_name.parent.parent / galaxy
# I need the header, so I need to use my clunkier function to get the data. It does not
# scale by the exposure time to get the data in electrons, so I need to do that.

band_select = sys.argv[7] # edit this here to get new data

base_image = utils._get_image(galaxy_dir)[band_select][0]
image = base_image.data * base_image.header["EXPTIME"]

# ======================================================================================
#
# Set up the function to measure magnitude
#
# ======================================================================================
# Note that this is calibrated in the testing notebook. The apertures were chosen
# following Adamo et al.17, and the zeropoint is chosen so that I can reproduce the
# real cluster magnitudes
def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def measure_magnitude(snapshot, x_cen, y_cen):
    aperture = 4
    zeropoint = 32.2

    flux_source = 0
    flux_sky = 0
    n_pixels_source = 0
    n_pixels_sky = 0
    for x in range(snapshot.shape[1]):
        for y in range(snapshot.shape[0]):
            dist = distance(x, y, x_cen, y_cen)
            # if it's a cluster pixel, count it
            if dist <= aperture:
                flux_source += snapshot[y][x]
                n_pixels_source += 1

            # then count the sky annulus
            if 6.5 <= dist < 7.5:
                flux_sky += snapshot[y][x]
                n_pixels_sky += 1

    # then subtract the background
    flux_source -= flux_sky * (n_pixels_source / n_pixels_sky)

    return -2.5 * np.log10(flux_source) + zeropoint


# ======================================================================================
#
# Then go through and add the artificial clusters to this image
#
# ======================================================================================
for row in true_catalog:
    # find the appropriate region of the image. To do this I have
    # to take out the region from the image (to make it match the size of this cluster),
    # then add the artificial cluster to that part of the image. Note that getting the
    # region still allows us to modify the image in place.

    # We use ceiling to get the integer pixel values as python indexing does not
    # include the final value.
    x_cen_snap = int(np.ceil(row["x"]))
    y_cen_snap = int(np.ceil(row["y"]))
    # Get the snapshot, based on the size desired.
    # Since we took the ceil of the center, go more in the negative direction (i.e.
    # use ceil to get the minimum values). This only matters if the snapshot size is
    # odd
    x_min = x_cen_snap - int(np.ceil(snapshot_size / 2.0))
    x_max = x_cen_snap + int(np.floor(snapshot_size / 2.0))
    y_min = y_cen_snap - int(np.ceil(snapshot_size / 2.0))
    y_max = y_cen_snap + int(np.floor(snapshot_size / 2.0))

    image_region = image[y_min:y_max, x_min:x_max]

    # get the coordinate of the cluster within this snapshot
    x_cen_cluster = row["x"] - x_min
    y_cen_cluster = row["y"] - y_min

    # Then make the cluster snapshot. We use an arbitrary luminosity, then scale it to
    # match the peak value requested
    cluster_snapshot = fit_utils.create_model_image(
        6,
        fit_utils.image_to_oversampled(x_cen_cluster, oversampling_factor),
        fit_utils.image_to_oversampled(y_cen_cluster, oversampling_factor),
        row["scale_radius_pixels_true"],
        row["axis_ratio_true"],
        row["position_angle_true"],
        row["power_law_slope_true"],
        0,  # background (as it's already in the image)
        psf,
        snapshot_size_oversampled,
        oversampling_factor,
    )[-1]
    # I'll need to scale the cluster to get the appropriate magnitude. Here is the math:
    # m - m_desired = -2.5 log10(f / f_desired)
    #               = -2.5 log10(f / (f * scale_factor))
    #               = -2.5 log10(1 / scale_factor)
    #               = 2.5 log10(scale_factor)
    # scale_factor = 10**((m - m_desired) / 2.5)
    first_mag = measure_magnitude(cluster_snapshot, x_cen_cluster, y_cen_cluster)
    scale_factor = 10 ** ((first_mag - row["mag_F555W"]) / 2.5)
    cluster_snapshot *= scale_factor
    # double check this math.
    assert np.isclose(
        measure_magnitude(cluster_snapshot, x_cen_cluster, y_cen_cluster),
        row["mag_F555W"],
    )

    # remove any pixels below 0. This will only happen due to floating point errors,
    # I believe
    min_value = np.min(cluster_snapshot)
    if min_value < 0:
        print("NEGATIVE VALUE", min_value, np.sum(cluster_snapshot < 0), "pixels")
        cluster_snapshot = np.maximum(cluster_snapshot, 0)

    # add Poisson noise to the clusters as well.
    # The snapshot we have so far is the expected snapshot without any noise. This is
    # equivalent to stating that each pixel is the mean of the Poisson distribution at
    # that pixel. So to get an image with noise, in each pixel we sample from a Poisson
    # distribution with the mean of the expected value in that pixel. Note that this
    # produces the new snapshot itself! We do not need to add the noise. Also, the
    # Poisson function returns integers, we need to convert to floats
    cluster_snapshot = np.random.poisson(cluster_snapshot)
    cluster_snapshot = cluster_snapshot.astype(float)

    # then we can add it to the image
    image_region += cluster_snapshot

# ======================================================================================
#
# write the image
#
# ======================================================================================
new_hdu = fits.PrimaryHDU(image)
# grab the header from the old image
new_hdu.header = base_image.header
# then reset the exposure time to be 1, as I've already put things in electrons.
new_hdu.header["EXPTIME"] = 1
new_hdu.writeto(image_name, overwrite=True)
