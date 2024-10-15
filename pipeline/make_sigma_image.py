"""
Make the sigma image, representing the uncertainty at each point in the image.

This script takes the following command-line arguments:
1 - Path to the sigma image that will be created
"""
from pathlib import Path
import sys

from astropy.io import fits
from astropy import stats
import betterplotlib as bpl
import numpy as np

import utils

bpl.set_style()

# ======================================================================================
#
# Get the parameters the user passed in, load the image
#
# ======================================================================================
# start by getting the output catalog name, which we can use to get the home directory
final_sigma_image = Path(sys.argv[1]).absolute()
size_dir = final_sigma_image.parent
home_dir = size_dir.parent

band_select = sys.argv[2] # edit this here to get new data
bands = utils.get_drc_image(home_dir)
image_data = bands[band_select][0]
#image_data, _, _ = utils.get_drc_image(home_dir)

# ======================================================================================
#
# Get the sky noise level and make debug plot
#
# ======================================================================================
# first throw out all data that's exactly zero
idxs_nonzero = np.nonzero(image_data)
nonzero_data = image_data[idxs_nonzero]

# we do the simplest thing and do sigma clipping
mean, median, sigma_sky = stats.sigma_clipped_stats(nonzero_data, sigma=2.0)

# make a debug plot to assess how well this worked
fig, ax = bpl.subplots()
bins = np.arange(mean - 10 * sigma_sky, mean + 10.1 * sigma_sky, 0.1 * sigma_sky)
ax.hist(nonzero_data, bins=bins, color=bpl.color_cycle[1])
ax.set_limits(min(bins), max(bins))
ax.axvline(mean, ls="--", label="Mean")
ax.axvline(mean + sigma_sky, ls=":", label="Mean $\pm \sigma_{sky}$")
ax.axvline(mean - sigma_sky, ls=":")
ax.add_labels("Pixel Values [electrons]", "Number of Pixels")
ax.legend()
fig.savefig(size_dir / "plots" / "sky_noise_debug.png")

# ======================================================================================
#
# Then assign noise values for all pixels in the image
#
# ======================================================================================
# start with an array that is infinite (to represent the pixels with no data)
sigma_image = np.ones(image_data.shape) * np.inf

# We add the noise in quadrature. This is the sky value and the poisson noise
poisson_sigma_squared = np.maximum(0, image_data)
final_sigma_squared = poisson_sigma_squared + sigma_sky ** 2

# then assign this noise to all pixels that had actual data
sigma_image[idxs_nonzero] = np.sqrt(final_sigma_squared)[idxs_nonzero]

# ======================================================================================
#
# Then write this output image
#
# ======================================================================================
new_hdu = fits.PrimaryHDU(sigma_image)
new_hdu.writeto(final_sigma_image, overwrite=True)
