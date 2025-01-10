"""
psf_percent.py - Compare the average background level to the maximum PSF value for each galaxy

This takes the following parameters:
- Band Selection
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
sys.path.append(str("pipeline"))
import utils

bpl.set_style()

data_dir = Path("data_PHANGS")

gal_names = []
avg_backs = []
max_psf = []
ratios = []

def get_star_list_and_psf(source):
    name_base = f"_{source}_stars_{psf_width}_pixels_{oversampling_factor}x_oversampled"

    table_name = "psf_star_centers" + name_base + ".txt"
    psf_name = "psf" + name_base + ".fits"

    star_table = table.Table.read(str(size_home_dir / table_name), format="ascii.ecsv")
    psf = fits.open(size_home_dir / psf_name)["PRIMARY"].data

    return star_table, psf
    
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
    
def radial_profile_psf(psf, color, label):
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
    
    return max(values)

for galaxy in data_dir.iterdir():
    if galaxy.name == ".DS_Store":
        continue
    galaxy_name = galaxy.name
    gal_names.append(galaxy_name)
    
    home_dir = data_dir / f"{galaxy_name}"
    
    size_home_dir = home_dir / "size"
    
    oversampling_factor = 2
    psf_width = 15

    # ======================================================================================
    #
    # Load the data - image and star catalog
    #
    # ======================================================================================
    band_select = sys.argv[1]
    bands = utils.get_drc_image(home_dir)
    image_data = bands[band_select][0]
    
    # the extract_stars thing below requires the input as a NDData object
    nddata = NDData(data=image_data)


    star_table_me, psf_me = get_star_list_and_psf("my")
    c_me = colors.CSS4_COLORS

    norm_const_list = []
    backgrounds = []
    for star_table, color in zip([star_table_me], [c_me]):
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
            # background subtract
            values -= np.median(background_values)
            # normalize the profile
            norm_const = np.sum(values)
            norm_const_list.append(norm_const)
            values /= np.sum(values)
            # then make the normalization match the PSF normalization, which is off by a
            # factor of oversampling_factor**2
            values /= oversampling_factor ** 2
            
            local_back = np.median(background_values)
            backgrounds.append(local_back)
            local_back /= norm_const
            local_back /= oversampling_factor ** 2
            local_back = abs(local_back)

            
    norm_const_list = np.array(norm_const_list)
    backgrounds = np.array(backgrounds)
    avg_background = np.median(abs(backgrounds)) / np.median(abs(norm_const_list))
    avg_background /= oversampling_factor ** 2
    # print(f"Avg Background for {galaxy_name}: ", avg_background)
    avg_backs.append(avg_background)
    
    for psf, color, label in zip([psf_me], [c_me], ["Me"]):
        max_val = radial_profile_psf(psf, color, label)
        # print(f"Max PSF value for {galaxy_name}: ", max_val)
        max_psf.append(max_val)
        
    # print(f"Ratio for {galaxy_name}: ", avg_background/max_val)
    ratios.append(avg_background/max_val)
    

data = np.array([gal_names, avg_backs, max_psf, ratios])
data = data.T

plot_dir = data_dir.parent
np.savetxt(str(plot_dir / "Background_Info.txt"), data, fmt="%s", delimiter = ' ', header="Galaxy_Name Background Max_PSF Ratio")
    
    
