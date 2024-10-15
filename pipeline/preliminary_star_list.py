"""
Makes a list of stars that will be inspected by the user for creation of the PSF.

This script takes the following command-line arguments:
1 - Path to the star list that will be created
2 - Path to the pre-existing cluster list. We use this to check which stars are near
    existing clusters
"""
# https://photutils.readthedocs.io/en/stable/epsf.html
from pathlib import Path
import sys

from astropy.io import fits
from astropy import table
from astropy import stats
from astropy.table import Table
from astropy.io.fits import getdata
import photutils
import numpy as np

import utils

# ======================================================================================
#
# Get the parameters the user passed in
#
# ======================================================================================
# start by getting the output catalog name, which we can use to get the home directory
final_catalog = Path(sys.argv[1]).absolute()
home_dir = final_catalog.parent.parent
# We'll need to get the cluster catalog too
cluster_catalog_path = Path(sys.argv[2]).absolute()
width = int(sys.argv[3])

def find_dolphot(home_dir):
    """
    Find the name of the dolphot file name. We need a function for this because we
    don't know whether it exists and it could have many name styles for the filename.

    :param home_dir: Directory to search for Dolphot
    :type home_dir: Path
    :return: Path object pointing to the Dolphot file if it exists and a boolean stating whether it exists.
    :rtype: tuple
    """
    galaxy_name = home_dir.name
    
    
    for item in home_dir.iterdir():
        if not item.is_file():
            continue
            
        filename = item.name
        # See if it starts and ends with what the Dolphot psfs file should be. We prefer .psfs files as these are specially selected by the PHANGS team for this purpose
        if filename.startswith(f"{galaxy_name}") and filename.endswith(
            f"_dolphot.psfs"
        ):
            dolphot = item
            
            return dolphot, True, "psfs"
            
    
    for item in home_dir.iterdir():
        if not item.is_file():
            continue
            
        filename = item.name
        # see if it starts and ends with what the Dolphot psfs file should be.
        if filename.startswith(f"{galaxy_name}") and filename.endswith(
            f"_dolphot.fits"
        ):
            dolphot = item
            
            return dolphot, True, "fits"
            
    return "", False

dolphot_name, dolphot_exists, dolphot_type = find_dolphot(home_dir)

# If the Dolphot file exists, write to preliminary stars file the data from the specified columns
if (dolphot_exists):
    
    if dolphot_type == "psfs":
        peaks_table = Table.read(dolphot_name, format = "ascii.no_header")

        peaks_table.rename_column('col3', 'Dolphot_x')
        peaks_table.rename_column('col4', 'Dolphot_y')
        
        
    elif dolphot_type == "fits":
        peaks_table = Table.read(dolphot_name)
        
    # Handle strange coordinates, the same for both file types
    peaks_table['Dolphot_x'] -= 0.5
    peaks_table['Dolphot_y'] -= 0.5
            


    # ======================================================================================
    #
    # Identifying troublesome stars
    #
    # ======================================================================================
    # I want to identify stars that may be problematic, because they're near another star or
    # because they're a cluster.
    one_sided_width = int((width - 1) / 2.0)
    # get duplicates within this box. We initially say that everything has nothing near it,
    # then will modify that as needed. We also track if something is close enough to a
    # cluster to actually be one.
    peaks_table["near_star"] = False
    peaks_table["near_cluster"] = False
    peaks_table["is_cluster"] = False
    # We'll write this up as a function, as we'll use this to check both the stars and
    # clusters, so don't want to have to reuse the same code
    def test_star_near(star_x, star_y, all_x, all_y, min_separation):
        """
        Returns whether a given star is near other objects

        :param star_x: X pixel coordinate of this star
        :param star_y: Y pixel coordinate of this star
        :param all_x: numpy array of x coordinates of objects to check against
        :param all_y: numpy array of y coordinates of objects to check against
        :param min_separation: The minimum separation allowed for two objects to be
                               considered isolated from each other, in pixels. This check
                               is done within a box of 2*min_separation + 1 pixels, not a
                               circle with radius min_separation.
        :return: True if no objects are close to the star
        """
        seps_x = np.abs(all_x - star_x)
        seps_y = np.abs(all_y - star_y)
        # see which other objects are near this one
        near_x = seps_x < min_separation
        near_y = seps_y < min_separation
        # see where both x and y are close
        near = np.logical_and(near_x, near_y)
        # for the star to be near something, any of these can be true
        return np.any(near)


    # read in the clusters table
    clusters_table = table.Table.read(cluster_catalog_path, format="ascii.ecsv")

    # when iterating through the rows, we do need to throw out the star itself when checking
    # it against other stars. So this changes the loop a bit. We also get the data
    # beforehand to reduce accesses
    stars_x = peaks_table["Dolphot_x"].data  # to get as numpy array
    stars_y = peaks_table["Dolphot_y"].data
    clusters_x = clusters_table["PHANGS_X"]
    clusters_y = clusters_table["PHANGS_Y"]
    for idx in range(len(peaks_table)):
        star_x = stars_x[idx]
        star_y = stars_y[idx]

        other_x = np.delete(stars_x, idx)  # returns fresh array, stars_x not modified
        other_y = np.delete(stars_y, idx)

        peaks_table["near_star"][idx] = test_star_near(
            star_x, star_y, other_x, other_y, one_sided_width
        )
        peaks_table["near_cluster"][idx] = test_star_near(
            star_x, star_y, clusters_x, clusters_y, one_sided_width
        )
        peaks_table["is_cluster"][idx] = test_star_near(
            star_x, star_y, clusters_x, clusters_y, 5
        )


    # then write the output catalog
    peaks_table.write(final_catalog, format="ascii.ecsv")
    
else:
    # Otherwise write a single value to the preliminary stars that informs us later in the pipeline that it doesn't exist
    peaks_table = Table()
    peaks_table["Data-Warning:"] = ["No Dolphot file"]
    peaks_table.write(final_catalog, format="ascii.ecsv")
