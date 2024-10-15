"""
fit_utils.py - Collection of functions to be used in the fitting and elsewhere
"""
import numpy as np
from astropy import nddata
from scipy import signal

# ======================================================================================
#
# Create the functions that will be used in the fitting procedure
#
# ======================================================================================
def eff_profile_2d(x, y, log_luminosity, x_c, y_c, a, q, theta, eta):
    """
    2-dimensional EFF profile, in pixel units

    :param x: X pixel values
    :param y: Y pixel values
    :param log_luminosity: Log of the luminosity of the cluster, in the same units as
                           the image
    :param x_c: Center X pixel coordinate
    :param y_c: Center Y pixel coordinate
    :param a: Scale radius in the major axis direction
    :param q: Axis ratio. The small axis will have scale length qa
    :param theta: Position angle
    :param eta: Power law slope
    :return: Values of the function at the x,y coordinates passed in
    """
    x_prime = (x - x_c) * np.cos(theta) + (y - y_c) * np.sin(theta)
    y_prime = -(x - x_c) * np.sin(theta) + (y - y_c) * np.cos(theta)

    # need to guard against zeros for a and q
    if a == 0:
        a = 1e-15
    if q == 0:
        q = 1e-15

    x_term = (x_prime / a) ** 2
    y_term = (y_prime / (q * a)) ** 2
    # use all the other parameters to turn luminosity into mu (assume 15 pixel
    # maximum radius).
    mu_0_term_a = 10 ** (log_luminosity) * (eta - 1) / (np.pi * a ** 2)
    mu_0_term_b = 1 - (1 + (15 / a) ** 2) ** (1 - eta)
    # sometimes parameters can conspire to make the bottom term essentially or
    # identically zero. Guard against this. We do need to be careful about the sign,
    # since this can be negative.
    if mu_0_term_b == 0:
        mu_0_term_b = 1e-15
    mu_0_term_b = np.sign(mu_0_term_b) * max(abs(mu_0_term_b), 1e-15)

    # then combine the two terms
    mu_0 = mu_0_term_a / mu_0_term_b

    return mu_0 * (1 + x_term + y_term) ** (-eta)


def convolve_with_psf(in_array, psf):
    """ Convolve an array with the PSF """
    # Scipy FFT based convolution was tested to be the fastest. It does have edge
    # affects, but those are minimized by using our padding. We do have to modify the
    # psf to have
    return signal.fftconvolve(in_array, psf, mode="same")


def bin_data_2d(data, oversampling_factor):
    """ Bin a 2d array into square bins determined by the oversampling factor """
    # Astropy has a convenient function to do this
    bin_factors = [oversampling_factor, oversampling_factor]
    return nddata.block_reduce(data, bin_factors, np.mean)


def create_model_image(
    log_luminosity,
    x_c,
    y_c,
    a,
    q,
    theta,
    eta,
    background,
    psf,
    snapshot_size_oversampled,
    oversampling_factor,
):
    """ Create a model image using the EFF parameters. """
    # a is passed in in regular coordinates, shift it to the oversampled ones
    a *= oversampling_factor

    # first generate the x and y pixel coordinates of the model image. We will have
    # an array that's the same size as the cluster snapshot in oversampled pixels,
    # plus padding to account for zero-padded boundaries in the FFT convolution
    padding = 5 * oversampling_factor  # 5 regular pixels on each side
    box_length = snapshot_size_oversampled + 2 * padding

    # correct the center to be at the center of this new array
    x_c_internal = x_c + padding
    y_c_internal = y_c + padding

    x_values = np.zeros([box_length, box_length])
    y_values = np.zeros([box_length, box_length])

    for x in range(box_length):
        x_values[:, x] = x
    for y in range(box_length):
        y_values[y, :] = y

    model_image = eff_profile_2d(
        x_values, y_values, log_luminosity, x_c_internal, y_c_internal, a, q, theta, eta
    )
    # convolve without the background first, to do an ever better job avoiding edge
    # effects, as the model should be zero near the boundaries anyway, matching the zero
    # padding scipy does.
    model_psf_image = convolve_with_psf(model_image, psf)
    model_image += background
    model_psf_image += background

    # crop out the padding before binning the data
    model_image = model_image[padding:-padding, padding:-padding]
    model_psf_image = model_psf_image[padding:-padding, padding:-padding]
    model_psf_bin_image = bin_data_2d(model_psf_image, oversampling_factor)

    # return all of these, since we'll want to use them when plotting
    return model_image, model_psf_image, model_psf_bin_image


def handle_mask(mask_snapshot, cluster_id):
    """
    Mask/unmask the cluster appropriately

    :param mask_snapshot: The original unmodified mask image snapshot
    :return: A modified mask image snapshot where the previously identified cluster is
             now unmasked, but any nearby clusters are masked
    """
    for x in range(mask_snapshot.shape[1]):
        for y in range(mask_snapshot.shape[0]):
            # -2 means it's always good
            if mask_snapshot[y, x] == -2:
                mask_snapshot[y, x] = 1
            # -1 means always bad
            elif mask_snapshot[y, x] == -1:
                mask_snapshot[y, x] = 0
            # otherwise, the value will be the cluster ID for which this pixel needs to
            # be unmasked. Figure out whether that is our cluster
            elif mask_snapshot[y, x] == cluster_id:
                mask_snapshot[y, x] = 1
            else:  # mask this region
                mask_snapshot[y, x] = 0
    return mask_snapshot


def radial_weighting(data_image, x_cen, y_cen, style="none"):
    """
    Produce a map of the radial pixel weighting to be used

    :param sigma_image: data snapshot, only used for its size
    :param x_cen: X coordinate of the cluster center, in units of the data
    :param y_cen: Y coordinate of the cluster center, in units of the data
    :param style: how to create the radial weights. This has a couple options:
                  none - do no modification to the sigma image
                  annulus - weight by 1/r, so that each annulus has equal weight
    :return: new weights image
    """
    weights = np.ones(data_image.shape)
    if style == "none":
        pass  # leave all weights as one
    elif style == "annulus":
        x_values = np.zeros([weights.shape[0], weights.shape[1]])
        y_values = np.zeros([weights.shape[0], weights.shape[1]])

        for x in range(weights.shape[1]):
            x_values[:, x] = x
        for y in range(weights.shape[0]):
            y_values[y, :] = y

        dists = distance(x_values, y_values, x_cen, y_cen)
        # make everything inside 3 pixels be the same to avoid centering artifacts
        dists = np.maximum(3.0, dists)
        weights = 1 / dists
    else:
        raise ValueError("Bad style parameter to radial_weighting")
    # then normalize the weights to have a maximum value at 1
    weights /= np.max(weights)
    return weights


def oversampled_to_image(x, oversampling_factor):
    """
    Turn oversampled pixel coordinates into regular pixel coordinates.

    There are two affects here: first the pixel size is different, and the centers
    (where the pixel is defined to be) are not aligned.

    :param x: Location in oversampled pixel coordinates
    :return: Location in regular pixel coordinates
    """
    # first have to correct for the pixel size
    x /= oversampling_factor
    # then the oversampled pixel centers are offset from the regular pixels,
    # so we need to correct for that too
    if oversampling_factor == 2:
        return x - 0.25
    else:
        raise ValueError("Think about this more")


def image_to_oversampled(x, oversampling_factor):
    """
    inverse of oversampled_to_image()

    :param x: Location in regular pixel coordinates
    :param oversampling_factor: PSF oversampling factor used for the snapshot
    :return: Location in the oversampled coordinates
    """
    if oversampling_factor == 2:
        return (x + 0.25) * 2
    else:
        raise ValueError("Think about this more")


def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
