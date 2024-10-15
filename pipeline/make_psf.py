"""
make_psf.py - Uses the previously-created list of stars to generate a new psf.

Takes the following command line arguments:
- Name to save the PSF as.
- Oversampling factor
- Pixel size for the PSF snapshot
- The source of the coordinate lists. Must either be "my" or "legus"
"""
# https://photutils.readthedocs.io/en/stable/epsf.html
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy import table
from astropy import stats
from astropy.nddata import NDData
import photutils
import betterplotlib as bpl
from matplotlib import colors
from matplotlib import pyplot as plt
from astropy.table import Table

from astropy.table import hstack

#from photutils import detect_sources, detect_threshold, aperture_photometry
from photutils.segmentation import detect_sources, detect_threshold
from photutils.aperture import aperture_photometry, CircularAnnulus, CircularAperture, EllipticalAnnulus, EllipticalAperture
from photutils.background import Background2D, MedianBackground

import utils

bpl.set_style()

# ======================================================================================
#
# Get the parameters the user passed in
#
# ======================================================================================
# start by getting the output name, which we can use to get the home directory
psf_name = Path(sys.argv[1]).absolute()
size_home_dir = psf_name.parent
home_dir = size_home_dir.parent
oversampling_factor = int(sys.argv[2])
psf_width = int(sys.argv[3])
star_source = sys.argv[4]

# check that the source is correct
if not star_source in ["my", "legus"]:
    raise ValueError("Bad final parameter to make_psf.py. Must be 'my' or 'legus'")

# ======================================================================================
#
# Load the data - image and star catalog
#
# ======================================================================================
# Get the image for the band selection chosen
band_select = sys.argv[5]
bands = utils.get_drc_image(home_dir)
image_data = bands[band_select][0]
# the extract_stars thing below requires the input as a NDData object
nddata = NDData(data=image_data)

# get the noise_level, which will be used later
_, _, noise = stats.sigma_clipped_stats(image_data, sigma=2.0)

# load the input star list. This depends on what source we have for these stars
if star_source == "my":
    star_table = table.Table.read(size_home_dir / "psf_stars.txt", format="ascii.ecsv")

else:
    # For the LEGUS list I'll do this myself because of the formatting. We do have
    # to be careful with one galaxy which has a long filename
    galaxy_name = home_dir.name
    if galaxy_name == "ngc5194-ngc5195-mosaic":
        name = "isolated_stars__f435w_f555w_f814w_ngc5194-ngc5195-mosaic.coo"
    else:
        name = f"isolated_stars_{galaxy_name}.coo"
    preliminary_catalog_path = home_dir / name

    star_table = table.Table(names=("x", "y"))
    with open(preliminary_catalog_path, "r") as in_file:
        for row in in_file:
            row = row.strip()
            if (not row.startswith("#")) and row != "":
                star_table.add_row((row.split()[0], row.split()[1]))
                
                
# Remove strange placeholder effects
size = len(star_table)
i = 0
while i < size:
    if (star_table["x"][i] < 0 and star_table["y"][i] < 0):
        print("Removed row:", star_table["x"][i], star_table["y"][i])
        star_table.remove_row(i)
        size -= 1
    i += 1
print(star_table) # Just checking that the star table looks right, feel free to edit this


# ======================================================================================
#
# make the cutouts and selections
#
# ======================================================================================
star_cutouts = photutils.psf.extract_stars(nddata, star_table, size=psf_width)

def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
# Aperture photometry function for CI estimation
def do_ap_photometry(data, source_centroid, r_ap, bg_rin, bg_rout):
    """
    Args:
        data:            The image data array
        source_centroid: Location of source (pix coordinates)
        r_ap:            Size (in pix) of source aperture
        bg_rin:          Inner radius (in pix) of background annulus
        bg_rout:         Outer radius (in pix) of background annulus
        
    Returns:
        Flux, background subtracted
    """
    
    # Create the source and background apertures
    aperture = CircularAperture(source_centroid, r=r_ap)
    bg_aperture = CircularAnnulus(source_centroid, r_in = bg_rin, r_out = bg_rout)

    # Do the aperture photometry
    
    rawflux_table = aperture_photometry(data, aperture)
    bkgflux_table = aperture_photometry(data, bg_aperture)

    phot_table = hstack([rawflux_table, bkgflux_table],
                         table_names=['raw', 'bkg'])
                         
    # Get the mean number of background counts per unit area
    bkg_mean = phot_table['aperture_sum_bkg'] / bg_aperture.area

    # Calculate number of background counts in source aperture
    bkg_sum = bkg_mean*aperture.area

    # Calculate the true number of source photon counts in
    # source aperture
    final_flux = (phot_table["aperture_sum_raw"]-bkg_sum)
    
    print ("Background sum = ", bkg_sum.data, "\n")
    print ("Object Counts = ", final_flux.data, "\n")
    
    return final_flux.data
    
    
# Plot photon counts as a function of radius given the position
# the source. Useful for determining aperture parameters
def plot_radial_counts(data, source_centroid, Name):

    radius = 0.25
    radii = []
    photon_counts = []

    # Most sources are much smaller than 25 pixels in radius
    # But can definitely increase this value if needed!
    # This also gives a good sense of background counts
    while radius < 25:
        # Create the aperture

        aperture = CircularAnnulus(source_centroid, r_in = radius, r_out = radius+0.25)

        # Do the photometry
        phot_table = aperture_photometry(data, aperture, method='subpixel')

        # Determine number of photon counts per unit area
        photon_counts.append(phot_table['aperture_sum']/aperture.area)

        radii.append(radius)
        radius += 0.25


    end = np.ones(len(photon_counts))*photon_counts[-1]
    plt.grid(True)

    l = np.array(photon_counts)
    flat_list = [item for sublist in l for item in sublist]

    
    plt.plot(radii, photon_counts)
    plt.plot(radii, end)
    plt.title("ADU counts vs. Radial Distance " + "{:.3f}".format(Name))
        
    plt.ylabel("Log ADU Counts per Unit Area")

    plt.xlabel("Radial Distance (pixels)")

    plt.xlim(0, 15)
    plt.yscale("log")
    plt.savefig(size_home_dir / Path("Extended_Source_" + "{:.3f}".format(Name) + ".png"))
    plt.close()


# Background estimate and calculate concentration index. Here I use the pixels farther than 8 pixels from the center for the backgrounds.
# This value was determined by looking at the profiles. Concentration Index decribed as magnitude difference between 1 pixel and 3 pixel apertures.
# CI and background estimation useful in creating quality cuts for our PSF stars.
backgrounds = []
con_inds = []
ext_srcs = []
for star in star_cutouts:
    x_cen = star.cutout_center[0]  # yes, this indexing is correct, look at the docs or
    y_cen = star.cutout_center[1]  # at the bottom of this and psf_compare.py to see use
    border_pixels = [
        star.data[y][x]
        for x in range(star.data.shape[1])
        for y in range(star.data.shape[0])
        if distance(x_cen, y_cen, x, y) > 8
    ]
    
    one_pix = do_ap_photometry(star.data, (x_cen, y_cen), 1, 8, 10)
    three_pix = do_ap_photometry(star.data, (x_cen, y_cen), 3, 8, 10)
    
    con_ind = -2.5 * np.log10(one_pix/three_pix)
    con_inds.append(con_ind)
    
    """
    if con_ind > 1.4:
        ext_srcs.append({"CI": con_ind, "Data": star.data, "Centroid": (x_cen, y_cen)})
    """
    
    # quick estimate of how many pixels we expect to have here. 0.9 is fudge factor
    assert len(border_pixels) > 0.9 * (psf_width ** 2 - np.pi * 8 ** 2)
    star._data = star.data - np.median(border_pixels)
    backgrounds.append(np.median(border_pixels))

# Remove stars with CI > 1.4 from our PSF stars
popped = []
for i, ind in enumerate(con_inds):
    if ind > 1.4:
        popped.append(i)
        
star_table.remove_rows(popped)
backgrounds = [i for j, i in enumerate(backgrounds) if j not in popped]
con_inds = [i for j, i in enumerate(con_inds) if j not in popped]

# Select about 10 stars (or however many are available) based on the lowest background values for our final selection of PSF stars
num_stars = np.minimum(10, len(star_table))
idx = abs(np.array(backgrounds)).argsort()[:num_stars]
backgrounds = np.array(backgrounds)[idx]
np.savetxt(size_home_dir / 'Background.txt', backgrounds)

con_inds = np.array(con_inds)[idx]
np.savetxt(size_home_dir / 'Concentration_Index.txt', con_inds)

# This section is for plotting the radial profiles of stars whose CI > 1.4 if you are cocnerend with keeping them. This way you can see
# if the profile appears to be a stellar source or otherwise.
'''
if len(ext_srcs) > 0:
    for ind_pot in ext_srcs:
        for ind_act in con_inds:
            if ind_pot["CI"] == ind_act:
                #print(ind_pot["CI"][0], type(ind_pot["CI"][0]))
                plot_radial_counts(ind_pot["Data"], ind_pot["Centroid"], ind_pot["CI"][0])
'''

plt.hist(con_inds)
plt.title("Concentration Index Histogram", fontsize=16)
plt.xlabel("CI Values", fontsize=13)
plt.savefig(size_home_dir / "CI_histogram.png")
plt.close()

# ======================================================================================
#
# Create actual PSF here and perform background subtraction
#
# ======================================================================================
print("\nNow at actual selection\n")
star_table = star_table[:][idx]

star_cutouts = photutils.psf.extract_stars(nddata, star_table, size=psf_width)
print(star_cutouts.n_stars) # Just a check that the number of stars is correct

# Actual background subtraction
for i, star in enumerate(star_cutouts):

    star._data = star.data - backgrounds[i]

# ======================================================================================
#
# then combine to make the PSF
#
# ======================================================================================
psf_builder = photutils.EPSFBuilder(
    oversampling=oversampling_factor,
    maxiters=20,
    smoothing_kernel="quadratic",  # more stable than quartic
    progress_bar=True,
)
psf, fitted_stars = psf_builder(star_cutouts)

psf_data = psf.data
# the convolution requires the psf to be normalized, and without any negative values
psf_data = np.maximum(psf_data, 0)
psf_data /= np.sum(psf_data)

# ======================================================================================
#
# Plot the cutouts
#
# ======================================================================================
#cmap = bpl.cm.lapaz
#cmap.set_bad(cmap(0))  # for negative values in log plot

ncols = 5
nrows = int(np.ceil(len(star_cutouts) / 5))

top_inches = 1.5
inches_per_row = 3
inches_per_col = 3.4
fig, axs = bpl.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=[inches_per_col * ncols, top_inches + inches_per_row * nrows],
    tight_layout=False,
    gridspec_kw={
        "top": (inches_per_row * nrows) / (top_inches + inches_per_row * nrows),
        "wspace": 0.18,
        "hspace": 0.35,
        "left": 0.01,
        "right": 0.98,
        "bottom": 0.01,
    },
)
axs = axs.flatten()
for ax in axs:
    ax.set_axis_off()

for ax, cutout, row, fitted_star in zip(axs, star_cutouts, star_table, fitted_stars):
    vmax = np.max(cutout)
    vmin = -5 * noise
    linthresh = max(0.01 * vmax, 5 * noise)
    norm = colors.SymLogNorm(vmin=vmin, vmax=vmax * 2, linthresh=linthresh, base=10)
    im = ax.imshow(cutout, norm=norm, origin="lower")
    # add a marker at the location identified as the center
    ax.scatter(
        [fitted_star.cutout_center[0]],
        fitted_star.cutout_center[1],
        c=bpl.almost_black,
        marker="x",
    )
    ax.remove_labels("both")
    #ax.remove_spines(["all"])
    ax.set_title(f"x={row['x']:.0f}\ny={row['y']:.0f}")
    fig.colorbar(im, ax=ax)

# reformat the name
if star_source == "my":
    plot_title = f"{str(home_dir.name).upper()} - Me"
else:
    plot_title = f"{str(home_dir.name).upper()} - LEGUS"

fig.suptitle(plot_title, fontsize=40)

figname = (
    f"psf_stars_{star_source}_stars_"
    + f"{psf_width}_pixels_"
    + f"{oversampling_factor}x_oversampled.png"
)

fig.savefig(size_home_dir / "plots" / figname, dpi=100)

# ======================================================================================
#
# then save it as a fits file
#
# ======================================================================================
new_hdu = fits.PrimaryHDU(psf_data)
new_hdu.writeto(psf_name, overwrite=True)

# also save the fitted stars
x_cens = [star.cutout_center[0] + star.origin[0] for star in fitted_stars.all_stars]
y_cens = [star.cutout_center[1] + star.origin[1] for star in fitted_stars.all_stars]
fitted_star_table = table.Table([x_cens, y_cens], names=("x_center", "y_center"))
savename = (
    f"psf_star_centers_{star_source}_stars_"
    + f"{psf_width}_pixels_"
    + f"{oversampling_factor}x_oversampled.txt"
)
fitted_star_table.write(size_home_dir / savename, format="ascii.ecsv")
