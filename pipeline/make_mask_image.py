"""
make_mask_image.py - Create the mask for a given image.

We mask out bright sources near clusters, but do not consider the whole image to save
time. Here the mask will contain these possible values
-2 - never mask
-1 - always mask
any other value - this pixel should be masked for all clusters other than the one with
                  this ID.
These can be postprocessed to turn this into an actual mask. We do this rather
complicated method so that we can mask other clusters if they're close to a cluster, but
then keep the cluster when we need to fit it later.

This script takes the following parameters:
- Path where the final mask image will be saved
- Path to the sigma image
- Path to the cluster catalog
- Size of the snapshots to be used for fitting
"""
import sys
from pathlib import Path

from astropy import table
from astropy.io import fits
import numpy as np
import photutils
from tqdm import tqdm

import utils

# ======================================================================================
#
# Get the parameters the user passed in, load images and catalogs
#
# ======================================================================================
mask_image_path = Path(sys.argv[1]).absolute()
cluster_catalog_path = Path(sys.argv[2]).absolute()
sigma_image_path = Path(sys.argv[3]).absolute()

snapshot_size = 70  # just to be safe, have a large radius where we mask
cluster_mask_radius = 6
min_closeness = 3  # how far stars have to stay away from the center
star_radius_fwhm_multiplier = 2  # We mask pixels that are within this*FWHM of a star

galaxy_name = mask_image_path.parent.parent.name
band_select = sys.argv[4]
bands = utils.get_drc_image(mask_image_path.parent.parent)
image_data = bands[band_select][0]

sigma_data = fits.open(sigma_image_path)["PRIMARY"].data
clusters_table = table.Table.read(cluster_catalog_path, format="ascii.ecsv")

# ======================================================================================
#
# Helper functions
#
# ======================================================================================
def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def find_stars(data_snapshot, uncertainty_snapshot):
    """
    Create mask image

    This wil use IRAFStarFinder from photutils to find stars in the snapshot. Anything
    more than 5 pixels from the center will be masked, with a radius of FWHM. This
    makes the full width masked twice the FWHM.

    :param data_snapshot: Snapshot to be used to identify sources
    :param uncertainty_snapshot: Snapshow showing the uncertainty.
    :return: masked image, where values with 1 are good, zero is bad.
    """
    threshold = 5 * np.min(uncertainty_snapshot)
    star_finder = photutils.detection.IRAFStarFinder(
        threshold=threshold + np.min(data_snapshot),
        fwhm=2.0,  # slightly larger than real PSF, to get extended sources
        #exclude_border=True,
        sharplo=0.8,
        sharphi=5,
        roundlo=0.0,
        roundhi=0.5,
        minsep_fwhm=1.0,
    )
    peaks_table = star_finder.find_stars(data_snapshot)

    # this will be None if nothing was found
    names = [
        "x",
        "y",
        "mask_radius",
        "near_cluster",
        "is_cluster",
    ]
    if peaks_table is None:
        return table.Table([[]] * len(names), names=names)

    # throw away some stars
    to_remove = []
    for idx in range(len(peaks_table)):
        row = peaks_table[idx]
        if (
            # throw away things with large FWHM - are probably clusters!
            row["fwhm"] * star_radius_fwhm_multiplier > cluster_mask_radius
            or row["peak"] < row["sky"]
            # peak is sky-subtracted. This ^ removes ones that aren't very far above
            # a high sky background. This cut stops substructure in clusters from
            # being masked.
            or row["peak"] < threshold
        ):
            to_remove.append(idx)
    peaks_table.remove_rows(to_remove)

    xs = peaks_table["xcentroid"].data
    ys = peaks_table["ycentroid"].data
    mask_radius = peaks_table["fwhm"].data * star_radius_fwhm_multiplier
    blank = np.ones(xs.shape) * -1

    return table.Table([xs, ys, mask_radius, blank, blank], names=names)


# ======================================================================================
#
# Creating the mask around the clusters
#
# ======================================================================================
# As I modify things, I will replace each pixel value with one of the following
# - not set yet
# - is wanted to be unmasked by multiple clusters, so it must be unmasked
# - an isolated star is here
# any nonnegative value: The ID of a cluster than wants this pixel unmasked
# these flag values are chosen to be things that will not be in the output
unset_flag = -10
multiple_cluster_unmasked_flag = -20
isolated_star_mask_flag = -30


def pixel_is_unset(x, y):
    return mask_data[y][x] == unset_flag


def pixel_is_single_cluster(x, y):
    return mask_data[y][x] >= 0


def pixel_is_this_cluster(x, y, cluster_id):
    return mask_data[y][x] == cluster_id


def pixel_is_multiple_clusters(x, y):
    return mask_data[y][x] == multiple_cluster_unmasked_flag


def pixel_is_isolated_star(x, y):
    return mask_data[y][x] == isolated_star_mask_flag


def set_pixel_multiple_clusters(x, y):
    mask_data[y][x] = multiple_cluster_unmasked_flag


def set_pixel_isolated_star(x, y):
    mask_data[y][x] = isolated_star_mask_flag


def set_pixel_single_cluster(x, y, cluster_id):
    mask_data[y][x] = cluster_id


def handle_cluster_pixel(x, y, cluster_id):
    """
    Handle a pixel that wants to be unmasked by one cluster, but masked for all others.

    - If this pixel already wants to be masked by another cluster, we'll mark that it
      wants to be unmasked by multiple clusters.
    - If the pixel is currently told to be always masked, we leave it that way. This is
      because stars are checked against all clusters when seeing if they're isolated.
      Overwriting them would stop any stars between `min_closeness` and
      `cluster_mask_radius` from being masked properly
    - If the pixel is unset, we'll update it to be masked for all clusters except this
      one.
    - If the pixel already wants to be masked by multiple clusters, we will not modify
      that, as we would be adding another cluster there.

    :param x: X pixel location
    :param y: Y pixel location
    :param cluster_id: ID of the cluster that wants to have this pixel unmasked.
    :return: None, but sets the pixel value appropriateoy
    """
    if pixel_is_multiple_clusters(x, y) or pixel_is_isolated_star(x, y):
        return  # just for clarity for myself
    # mark pixels that are not set
    elif pixel_is_unset(x, y):
        set_pixel_single_cluster(x, y, cluster_id)
    # if multiple clusters want this pixel unmasked, mark that multiple
    # clusters want it
    elif pixel_is_single_cluster(x, y) and not pixel_is_this_cluster(x, y, cluster_id):
        set_pixel_multiple_clusters(x, y)
    elif pixel_is_single_cluster(x, y) and pixel_is_this_cluster(x, y, cluster_id):
        # already set, do nothing
        return
    else:
        raise RuntimeError("Should not happen")


mask_data = np.ones(image_data.shape, dtype=int) * unset_flag
for cluster in tqdm(clusters_table):
    # create the snapshot. We use ceiling to get the integer pixel values as python
    # indexing does not include the final value. So when we calcualte the offset, it
    # naturally gets biased low. Moving the center up fixes that in the easiest way.
    x_cen = int(np.ceil(cluster["x"]))
    y_cen = int(np.ceil(cluster["y"]))

    x_min = x_cen - int(np.ceil(snapshot_size / 2.0))
    x_max = x_cen + int(np.floor(snapshot_size / 2.0))
    y_min = y_cen - int(np.ceil(snapshot_size / 2.0))
    y_max = y_cen + int(np.floor(snapshot_size / 2.0))

    data_snapshot = image_data[y_min:y_max, x_min:x_max].copy()
    error_snapshot = sigma_data[y_min:y_max, x_min:x_max].copy()
    # find the stars
    these_stars = find_stars(data_snapshot, error_snapshot)
    # change the x and y coordinates from the snapshot coords to image coords
    these_stars["x"] += x_min
    these_stars["y"] += y_min

    # figure out which stars are near clusters
    for star in these_stars:
        for c in clusters_table:
            dist_to_cluster = distance(star["x"], star["y"], c["x"], c["y"])
            # if it's close to the cluster, we won't even bother marking this source
            if dist_to_cluster < 2:
                star["is_cluster"] = c["ID_PHANGS_CLUSTER"]
            # otherwise, mark stars that will have any overlap with the fit region
            max_dist = star["mask_radius"] + min_closeness
            if dist_to_cluster < max_dist:
                star["near_cluster"] = c["ID_PHANGS_CLUSTER"]

    # then mask the pixels found here, as well as the cluster itself. We do this here
    # so that we don't have to iterate over the whole image
    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            # mark the pixels that need to be unmasked for this cluster because they're
            # close to it.
            if distance(x, y, cluster["x"], cluster["y"]) < cluster_mask_radius:
                handle_cluster_pixel(x, y, cluster["ID_PHANGS_CLUSTER"])

            # then go through each star
            for star in these_stars:
                if star["is_cluster"] >= 0:
                    continue
                # if this pixel is close to the star, we'll need to mask
                if distance(x, y, star["x"], star["y"]) < star["mask_radius"]:
                    # if it's close to a cluster, mark that value
                    if star["near_cluster"] >= 0:
                        # This pixel wants to be masked by a star that is near a cluster
                        # So this pixel needs to be marked with the value of that
                        # cluster, so it will be unmasked for that cluster only.
                        # This is equivalent to what happens when a cluster wants a
                        # pixel masked
                        handle_cluster_pixel(x, y, star["near_cluster"])
                    # we now have an isolated star. Pixels that are identified here
                    # are far from a cluster. They should always be masked, no matter
                    # what is going on elsewhere
                    else:
                        # check some things
                        if pixel_is_multiple_clusters(x, y):
                            #raise RuntimeError(f"{galaxy_name} {x} {y}") # This line was causing issues with f275w files, investigate further!
                            print(f"WARNING: {galaxy_name} {x} {y}")
                        set_pixel_isolated_star(x, y)

# ======================================================================================
#
# Then postprocess this
#
# ======================================================================================
# in the output -2 is always good, -1 is always bad, and any other value means the
# cluster wants to be there.
# if the value has not been set for a given pixel yet, it's good
mask_data[mask_data == unset_flag] = -2
# if it's -100, multiple clusters want it, so we'll make those good
mask_data[mask_data == multiple_cluster_unmasked_flag] = -2
# isolated stars are masked
mask_data[mask_data == isolated_star_mask_flag] = -1
# the only other thing is the clusters, which need to be left alone

# ======================================================================================
#
# Then write this output image
#
# ======================================================================================
new_hdu = fits.PrimaryHDU(mask_data)
new_hdu.writeto(mask_image_path, overwrite=True)
