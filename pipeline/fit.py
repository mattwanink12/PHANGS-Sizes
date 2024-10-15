"""
fit.py - Do the fitting of the clusters in the image

This takes 7 or 8 arguments:
- Path where the resulting cluster catalog will be saved
- Path to the fits image containing the PSF
- Oversampling factor used when creating the PSF
- Path to the sigma image containing uncertainties for each pixel
- Path to the mask image containing whether or not to use each pixel
- Path to the cleaned cluster catalog
- Size of the fitting region to be used
- Optional argument that must be "ryon_like" if present. If it is present, masking will
  not be done, and the power law slope will be restricted to be greater than 1.
"""
from pathlib import Path
import sys
from collections import defaultdict

from astropy import table, stats
from astropy.io import fits
import numpy as np
from scipy import optimize
from tqdm import tqdm

import utils
import fit_utils

# ======================================================================================
#
# Get the parameters the user passed in, load images and catalogs
#
# ======================================================================================
final_catalog = Path(sys.argv[1]).absolute()
psf_path = Path(sys.argv[2]).absolute()
oversampling_factor = int(sys.argv[3])
sigma_image_path = Path(sys.argv[4]).absolute()
mask_image_path = Path(sys.argv[5]).absolute()
cluster_catalog_path = Path(sys.argv[6]).absolute()
snapshot_size = int(sys.argv[7])
band_select = sys.argv[8]
if len(sys.argv) > 9:
    if len(sys.argv) != 10 or sys.argv[9] != "ryon_like":
        raise ValueError("Bad list of parameters to fit.py")
    else:
        ryon_like = True
else:
    ryon_like = False

galaxy_name = final_catalog.parent.parent.name
#band_select = "f555w" # edit this here to get new data
bands = utils.get_drc_image(final_catalog.parent.parent)
image_data = bands[band_select][0]
psf = fits.open(psf_path)["PRIMARY"].data

# the convolution requires the psf to be normalized, and without any negative values
psf = np.maximum(psf, 0)
psf /= np.sum(psf)

sigma_data = fits.open(sigma_image_path)["PRIMARY"].data
mask_data = fits.open(mask_image_path)["PRIMARY"].data
clusters_table = table.Table.read(cluster_catalog_path, format="ascii.ecsv")
# if we're Ryon-like, do no masking, so the mask will just be -2 (the value for no
# masked region). Also handle the radial weighting
if ryon_like:
    mask_data = np.ones(mask_data.shape) * -2
    radial_weighting = "none"
else:
    radial_weighting = "annulus"

snapshot_size_oversampled = snapshot_size * oversampling_factor

# set up the background scale factor. We do this as the range of possible values for the
# background is quite large, so it's hard for the fit to cover this whole range. In
# addition, in the gradient calculation, very small changes in the background have
# basically no change in the fit quality. To alleviate these issues, we simple scale
# the background value used in the fit. We divide it by some large number. This simple
# fix solves both issues. Using the log would also work, but is harder since we have
# negative numbers.
bg_scale_factor = 1e3

# ======================================================================================
#
# Set up the table. We need some dummy columns that we'll fill later
#
# ======================================================================================
def dummy_list_col(n_rows):
    return np.array([[-99.9] * k for k in range(n_rows)], dtype="object")


# Make the grid of a and eta to have the multiple starting points. We'll use this later
a_grid = np.logspace(-1, 1, 12)
eta_grid = np.arange(1.1, 3.0, 0.2)
a_values = []
eta_values = []
for a in a_grid:
    for eta in eta_grid:
        a_values.append(a)
        eta_values.append(eta)
n_grid = len(a_values)

# Add the dummy columns for the attributes
n_rows = len(clusters_table)
new_cols = [
    "x_fitted",
    "y_fitted",
    "x_pix_snapshot_oversampled",
    "y_pix_snapshot_oversampled",
    "log_luminosity",
    "scale_radius_pixels",
    "axis_ratio",
    "position_angle",
    "power_law_slope",
    "local_background",
]

for col in new_cols:
    clusters_table[col] = dummy_list_col(n_rows)
    clusters_table[col + "_x0_variations"] = -99.9 * np.ones((n_rows, n_grid))
    clusters_table[col + "_best"] = -99.9

clusters_table["log_likelihood_x0_variations"] = -99.9 * np.ones((n_rows, n_grid))
clusters_table["num_boostrapping_iterations"] = -99

# ======================================================================================
#
# Functions to be used in the fitting
#
# ======================================================================================
def calculate_chi_squared(params, cluster_snapshot, error_snapshot, mask):
    """
    Calculate the chi-squared value for a given set of parameters.

    :param params: Tuple of parameters of the EFF profile
    :param cluster_snapshot: Cluster snapshot
    :param error_snapshot: Error snapshot
    :param mask: 2D array used as the mask, that contains 1 where there are pixels to
                 use, and zero where the pixels are not to be used.
    :return:
    """
    _, _, model_snapshot = fit_utils.create_model_image(
        *params, psf, snapshot_size_oversampled, oversampling_factor
    )
    assert model_snapshot.shape == cluster_snapshot.shape
    assert model_snapshot.shape == error_snapshot.shape

    diffs = cluster_snapshot - model_snapshot
    sigma_snapshot = diffs / error_snapshot
    # then use the mask and the weights
    sigma_snapshot *= mask
    # do the radial weighting. Need to get the data coordinates of the center.
    # Note that the radial weighting is not used if Ryon-like is set, see initial setup
    sigma_snapshot *= fit_utils.radial_weighting(
        cluster_snapshot,
        fit_utils.oversampled_to_image(params[1], oversampling_factor),
        fit_utils.oversampled_to_image(params[2], oversampling_factor),
        style=radial_weighting,
    )
    # Ryon used squares of differences, we use absolute value
    if ryon_like:
        return np.sum(sigma_snapshot ** 2)
    else:
        return np.sum(np.abs(sigma_snapshot))


def estimate_background(data, mask, x_c, y_c, min_radius):
    """
    Estimate the true background value.

    This will be defined to be the median of all unmasked pixels beyond min_radius

    :param data: Values at each pixel
    :param x_c: X coordinate of the center, in coordinates of the sigmas snapshot
    :param y_c: Y coordinate of the center, in coordinates of the sigmas snapshot
    :param min_radius: Minimum radius to include in the calculation
    :return: median(pixel_value) and std(pixel_value) where r > min_radius,
    """
    good_bg = []
    for x in range(data.shape[1]):
        for y in range(data.shape[0]):
            if fit_utils.distance(x, y, x_c, y_c) > min_radius and mask[y, x] > 0:
                good_bg.append(data[y, x])

    if len(good_bg) > 0:
        mean, median, std = stats.sigma_clipped_stats(
            good_bg,
            sigma_lower=None,
            sigma_upper=3.0,
            maxiters=None,
        )
        return mean, std
    else:
        return np.min(data), np.inf


def log_of_normal(x, mean, sigma):
    """
    Log of the normal distribution PDF. This is normalized to be 0 at the mean.

    :param x: X values to determine the value of the PDF at
    :param mean: Mean of the Gaussian
    :param sigma: Standard deviation of the Gaussian
    :return: natural log of the normal PDF at this location
    """
    return -0.5 * ((x - mean) / sigma) ** 2


def logistic(x, minimum, maximum, x_0, scale):
    """
    Generalized logistic function that goes from some min to some max

    :param x: Value at which to evaluate the logistic function
    :param minimum: Asymptotic value for low values of x
    :param maximum: Asymptotic value for high values of x
    :param x_0: Central value at which the transition happens
    :param scale: Scale factor for the width of the transition
    :return: Value of the logistic function at this value.
    """
    height = maximum - minimum
    return minimum + height / (1 + np.exp((x_0 - x) / scale))


def log_priors(
    log_luminosity,
    x_c,
    y_c,
    a,
    q,
    theta,
    eta,
    background,
    estimated_bg,
    estimated_bg_sigma,
):
    """
    Calculate the prior probability for a given model.

    :param estimated_bg: the estimated background value, to be used as the mean of the
                         Gaussian prior on the background.
    :param estimated_bg_sigma: the scatter in the estimated background. The sigma of the
                               Gaussian prior on the background will be 0.1 times this.


    The parameters passed in are the ones for the EFF profile. The parameters are
    treated independently:
    - For the center we use a Gaussian centered on the center of the image with a
      width of 3 image pixels
    - for the scale radius and power law slope we use a Gamma distribution
      with k=1.5, theta=3
    - for the axis ratio we use a simple trapezoid shape, where it's linearly increasing
      up to 0.3, then flat above that.
    All these independent values are multiplied together then returned.

    :return: Total prior probability for the given model.
    """
    if ryon_like:
        return 0
    log_prior = 0
    # prior are multiplicative, or additive in log space
    # the width of the prior on the background depends on the value of the power law
    # slope. Below 1 it will be strict (0.1 sigma), as this is when we have issues with
    # estimating the background, while for higher values of eta the background prior
    # will be less strict. We want a smooth transition of this width, as any sharp
    # transition will give artifacts in the resulting distributions.
    width = logistic(eta, 0.1, 1.0, 1.0, 0.1) * estimated_bg_sigma
    log_prior += log_of_normal(background, estimated_bg, width)
    return log_prior


def negative_log_likelihood(
    params, cluster_snapshot, error_snapshot, mask, estimated_bg, estimated_bg_sigma
):
    """
    Calculate the negative log likelihood for a model

    We do the negative likelihood becuase scipy likes minimize rather than maximize,
    so minimizing the negative likelihood is maximizing the likelihood

    :param params: Tuple of parameters of the EFF profile
    :param cluster_snapshot: Cluster snapshot
    :param error_snapshot: Error snapshot
    :param mask: 2D array used as the mask, that contains 1 where there are pixels to
                 use, and zero where the pixels are not to be used.
    :param estimated_bg: the estimated background value, to be used as the mean of the
                         Gaussian prior on the background.
    :param estimated_bg_sigma: the scatter in the estimated background. The sigma of the
                               Gaussian prior on the background will be 0.1 times this.
    :return:
    """
    # postprocess them from the beginning to make things simpler. This allows the
    # fitting machinery to do what it wants, but under the hood we only use
    # reasonable parameter value. This also handles the background scaling.
    params = postprocess_params(*params)

    chi_sq = calculate_chi_squared(params, cluster_snapshot, error_snapshot, mask)
    log_data_likelihood = -chi_sq / 2.0
    # Need to postprocess the parameters before calculating the prior, as the prior
    # is on the physically reasonable values, we need to make sure that's correct.
    log_prior = log_priors(*params, estimated_bg, estimated_bg_sigma)
    log_likelihood = log_data_likelihood + log_prior
    # With bad parameters, it is possible to get a nan or infinity value. This is
    # particularly possible with very negative eta values. If we get something bad,
    # make the likelihoood negative infinity. This is fine when using the Powell
    # method as it does not rely on the gradient, so using infinity for this shouldn't
    # affect the fitting dramatically. [Note that I'm not sure what's causing these bad
    # parameter values. I am setting bounds, but it appears that there are function
    # calls being made with values outside these bounds, for reasons I do not
    # understand.]
    try:
        assert not np.isnan(log_prior)
        assert not np.isnan(log_data_likelihood)
        assert not np.isinf(log_prior)
        assert not np.isinf(log_data_likelihood)
        assert not np.isneginf(log_prior)
        assert not np.isneginf(log_data_likelihood)
    except AssertionError:
        log_likelihood = -np.inf  # zero likelihood
    # return the negative of this so we can minimize this value
    return -log_likelihood


def create_boostrap_mask(original_mask, x_c, y_c):
    """
    Create a temporary mask used during a given bootstrapping iteration.

    We will have two separate regions. Within 9 pixels from the center, we will sample
    on all pixels individually. Outside this central region, we create 3x3 pixel
    boxes. We do bootstrapping on both of these regions independently, then combine
    the selection.

    :param original_mask: Original mask, where other sources can be masked out.
    :param x_c: X center of the cluster
    :param y_c: Y center of the cluster
    :return: Mask that contains the number of times each pixel was selected.
    """
    box_size = 5
    # correct for oversampled pixels
    x_c /= oversampling_factor
    y_c /= oversampling_factor

    # first go through and assign all pixels to either a box or the center. We have a
    # dictionary of lists for the boxes, that will have keys of box location and values
    # of all the pixels in that box.
    outside_boxes = defaultdict(list)
    center_pixels = []
    for x in range(original_mask.shape[1]):
        for y in range(original_mask.shape[0]):
            if original_mask[y, x] == 1:  # only keep pixels not already masked out
                if fit_utils.distance(x, y, x_c, y_c) <= 9:
                    center_pixels.append((x, y))
                else:
                    idx_box_x = x // box_size
                    idx_box_y = y // box_size

                    outside_boxes[(idx_box_x, idx_box_y)].append((x, y))

    # freeze the keys so we can have an order to sample from
    outside_boxes_keys = list(outside_boxes.keys())

    # then we can subsample each of those
    idxs_boxes = np.random.randint(0, len(outside_boxes_keys), len(outside_boxes_keys))
    idxs_center = np.random.randint(0, len(center_pixels), len(center_pixels))

    # then put this into the mask
    temp_mask = np.zeros(original_mask.shape)
    for idx in idxs_boxes:
        key = outside_boxes_keys[idx]
        for x, y in outside_boxes[key]:
            temp_mask[y, x] += 1
    for idx in idxs_center:
        x, y = center_pixels[idx]
        temp_mask[y, x] += 1

    return temp_mask


def postprocess_params(log_luminosity, x_c, y_c, a, q, theta, eta, background):
    """
    Postprocess the parameters, namely the axis ratio and position angle.

    This is needed since we let the fit have axis ratios larger than 1, and position
    angles of any value. Axis ratios larger than 1 indicate that we need to flip the
    major and minor axes. This requires rotating the position angle 90 degrees, and
    shifting the value assigned to the major axis to correct for the improper axis
    ratio.
    """
    # handle background
    background *= bg_scale_factor

    # q and a can be negative, fix that before any further processing
    a = abs(a)
    q = abs(q)
    if q > 1.0:
        q_final = 1.0 / q
        a_final = a * q
        theta_final = (theta - (np.pi / 2.0)) % np.pi
        return log_luminosity, x_c, y_c, a_final, q_final, theta_final, eta, background
    else:
        return log_luminosity, x_c, y_c, a, q, theta % np.pi, eta, background


def fit_model(data_snapshot, uncertainty_snapshot, mask, x_guess, y_guess):
    """
    Fits an EFF model to the data passed in, using bootstrapping.

    :param data_snapshot: 2D array holding the pixel values (in units of electrons)
    :param uncertainty_snapshot: 2D array holding the uncertainty in the pixel values,
                                 in units of electrons.
    :param mask: 2D array holding the mask, where 1 is a good pixel, zero is bad.
    :param id_num: The ID of this cluster
    :return: A two-item tuple containing: the fit parameters to all pixels, and the
             history of all parameters took throughout the bootstrapping

    """
    data_center = snapshot_size / 2.0
    # estimate the background to use as a prior
    estimated_bg, bg_scatter = estimate_background(
        data_snapshot, mask, data_center, data_center, 6
    )
    # log luminosity - subtract off the estimated background then apply the mask
    # to estimate the light belonging to the cluster. We then have to correct for
    # the oversampling factor, since the model space has different pixel sizes. We
    # take the mean to combine pixels (this works nicely for the background), but
    # for the luminosity we need to scale.
    estimated_l = np.sum((data_snapshot - estimated_bg) * mask_snapshot)
    estimated_l *= oversampling_factor ** 2
    # check for negative values of luminosity. this does happen for one cluster with an
    # artifact in the background, where a chunk has lower values. This messes up the
    # estimated luminosity calculation, nothing else. We need to correct it to be some
    # nonzero number so we can take a log. 1 is quite low, but it's fine.
    estimated_l = max(estimated_l, 1)

    # Create the initial guesses for the parameters
    start_params = (
        [np.log10(estimated_l)] * n_grid,
        [x_guess] * n_grid,  # X center in the oversampled snapshot
        [y_guess] * n_grid,  # Y center in the oversampled snapshot
        a_values,  # scale radius, in regular pixels.
        [0.8] * n_grid,  # axis ratio
        [0] * n_grid,  # position angle
        eta_values,  # power law slope
        [estimated_bg / bg_scale_factor] * n_grid,  # background
    )

    # some of the bounds are needed .We allow axis ratios greater than 1 to make
    # the fitting routine have more flexibility. For example, if the position angle is
    # correct but it's aligned with what should be the minor axis instead of the major,
    # the axis ratio can go above one to fix that issue. We then have to process things
    # afterwards to flip it back, but that's not an issue. We do similar things with
    # allowing the axis ratio and scale radius to be negative. They get squared in the
    # EFF profile anyway, so their sign doesn't matter. They are singular at a=0 and
    # q=0, but hopefully floating point values will stop that from being an issue
    # Have the center pixel bounds be from the center, so it's easy to detect when
    # we reached a bound, so we can throw it out.
    center = snapshot_size_oversampled / 2.0
    center_half_width = 2 * oversampling_factor
    bounds = [
        (None, 100),  # log of luminosity. Cap needed to stop overflow errors
        (center - center_half_width, center + center_half_width),  # X center
        (center - center_half_width, center + center_half_width),  # Y center
        (None, None),  # scale radius in regular pixels.
        (None, None),  # axis ratio
        (None, None),  # position angle
        (0, None),  # power law slope
        (None, None),  # background
    ]

    # set some of the convergence criteria parameters for the Powell fitting routine.
    # I don't require too strict convergence, since having multiple starting points
    # ensures that we're close. I've found it doesn't converge any better with more
    # strict criteria anyway.
    xtol = 1e-3
    ftol = 1e-3
    maxfev = np.inf
    maxiter = np.inf

    # first get the results when all good pixels are used, to be used as a starting
    # point when bootstrapping is done to save time. This will be done for each
    # starting point, and we'll pick the best one to use
    param_x0_variations_raw = []
    param_x0_variations_postprocessed = []
    log_likelihood_x0_variations = []
    for idx in range(len(start_params[0])):
        initial_result_struct = optimize.minimize(
            negative_log_likelihood,
            args=(data_snapshot, uncertainty_snapshot, mask, estimated_bg, bg_scatter),
            x0=[start_params[j][idx] for j in range(len(start_params))],
            bounds=bounds,
            method="Powell",
            options={
                "xtol": xtol,
                "ftol": ftol,
                "maxfev": maxfev,
                "maxiter": maxiter,
            },
        )

        # Note that we do not postprocess the params until after we've calculated the
        # log likelihood, as the likelihood function does the postprocessing, and we
        # don't want to do that twice, as it will mess up the background and
        # theta values
        initial_result = initial_result_struct.x
        this_log_likelihood = -1 * negative_log_likelihood(
            initial_result,
            data_snapshot,
            uncertainty_snapshot,
            mask,
            estimated_bg,
            bg_scatter,
        )
        param_x0_variations_raw.append(initial_result)
        param_x0_variations_postprocessed.append(postprocess_params(*initial_result))
        log_likelihood_x0_variations.append(this_log_likelihood)

    # then examine the likelihood of each, and pick the one with the highest likelihood
    best_idx = np.argmax(log_likelihood_x0_variations)
    initial_result_best_postprocessed = param_x0_variations_postprocessed[best_idx]
    initial_result_best_raw = param_x0_variations_raw[best_idx]

    # Then we do bootstrapping
    n_variables = len(initial_result_best_raw)
    param_history = [[] for _ in range(n_variables)]
    param_std_last = [np.inf for _ in range(n_variables)]

    converge_criteria = 0.1  # fractional change in std required for convergence
    converged = [False for _ in range(n_variables)]
    check_spacing = 20  # how many iterations between checking the std
    iteration = 0
    while not all(converged):
        iteration += 1

        # make a new mask
        temp_mask = create_boostrap_mask(
            mask, initial_result_best_raw[1], initial_result_best_raw[2]
        )

        # fit to this selection of pixels
        this_result_struct = optimize.minimize(
            negative_log_likelihood,
            args=(
                data_snapshot,
                uncertainty_snapshot,
                temp_mask,
                estimated_bg,
                bg_scatter,
            ),
            # use the best fit results as the initial guess, to get the uncertainty
            # around that value. This should also reduce the time needed to converge
            x0=initial_result_best_raw,
            bounds=bounds,
            method="Powell",
            options={
                "xtol": xtol,
                "ftol": ftol,
                "maxfev": maxfev,
                "maxiter": maxiter,
            },
        )
        # store the results after processing them
        this_result = postprocess_params(*this_result_struct.x)
        for param_idx in range(n_variables):
            param_history[param_idx].append(this_result[param_idx])

        # then check if we're converged
        if iteration % check_spacing == 0:
            for param_idx in range(n_variables):
                # calculate the new standard deviation
                this_std = np.std(param_history[param_idx])
                if this_std == 0:
                    converged[param_idx] = True
                else:  # actually calculate the change
                    last_std = param_std_last[param_idx]
                    diff = abs((this_std - last_std) / this_std)
                    converged[param_idx] = diff < converge_criteria

                # then set the new last value
                param_std_last[param_idx] = this_std

    # then we're done!
    return (
        initial_result_best_postprocessed,
        np.array(param_history),
        param_x0_variations_postprocessed,
        log_likelihood_x0_variations,
    )


# ======================================================================================
#
# Then go through the catalog
#
# ======================================================================================
for row in tqdm(clusters_table):
    # create the snapshot. We use ceiling to get the integer pixel values as
    # python indexing does not include the final value.
    x_cen = int(np.ceil(row["x"]))
    y_cen = int(np.ceil(row["y"]))

    # Get the snapshot, based on the size desired.
    # Since we took the ceil of the center, go more in the negative direction (i.e.
    # use ceil to get the minimum values). This only matters if the snapshot size is
    # odd
    x_min = x_cen - int(np.ceil(snapshot_size / 2.0))
    x_max = x_cen + int(np.floor(snapshot_size / 2.0))
    y_min = y_cen - int(np.ceil(snapshot_size / 2.0))
    y_max = y_cen + int(np.floor(snapshot_size / 2.0))

    data_snapshot = image_data[y_min:y_max, x_min:x_max].copy()
    error_snapshot = sigma_data[y_min:y_max, x_min:x_max].copy()
    mask_snapshot = mask_data[y_min:y_max, x_min:x_max].copy()

    mask_snapshot = fit_utils.handle_mask(mask_snapshot, row["ID_PHANGS_CLUSTER"])

    # Use the LEGUS center to start the fit at this location
    x_legus = fit_utils.image_to_oversampled(row["x"] - x_min, oversampling_factor)
    y_legus = fit_utils.image_to_oversampled(row["y"] - y_min, oversampling_factor)

    # then do this fitting!
    all_results = fit_model(
        data_snapshot, error_snapshot, mask_snapshot, x_legus, y_legus
    )
    results, history, x0_variations, x0_likelihoods = all_results

    # Then add these values to the table
    row["num_boostrapping_iterations"] = len(history[0])

    row["log_luminosity_best"] = results[0]
    row["x_pix_snapshot_oversampled_best"] = results[1]
    row["y_pix_snapshot_oversampled_best"] = results[2]
    row["x_fitted_best"] = x_min + fit_utils.oversampled_to_image(
        results[1], oversampling_factor
    )
    row["y_fitted_best"] = y_min + fit_utils.oversampled_to_image(
        results[2], oversampling_factor
    )
    row["scale_radius_pixels_best"] = results[3]
    row["axis_ratio_best"] = results[4]
    row["position_angle_best"] = results[5]
    row["power_law_slope_best"] = results[6]
    row["local_background_best"] = results[7]

    # Store the bootstrapping history
    row["log_luminosity"] = history[0]
    row["x_pix_snapshot_oversampled"] = history[1]
    row["y_pix_snapshot_oversampled"] = history[2]
    row["x_fitted"] = [
        x_min + fit_utils.oversampled_to_image(v, oversampling_factor)
        for v in history[1]
    ]
    row["y_fitted"] = [
        y_min + fit_utils.oversampled_to_image(v, oversampling_factor)
        for v in history[2]
    ]
    row["scale_radius_pixels"] = history[3]
    row["axis_ratio"] = history[4]
    row["position_angle"] = history[5]
    row["power_law_slope"] = history[6]
    row["local_background"] = history[7]

    # Store the fitting done with different starting points
    row["log_luminosity_x0_variations"] = [p[0] for p in x0_variations]
    row["x_pix_snapshot_oversampled_x0_variations"] = [p[1] for p in x0_variations]
    row["y_pix_snapshot_oversampled_x0_variations"] = [p[2] for p in x0_variations]
    row["x_fitted_x0_variations"] = [
        x_min + fit_utils.oversampled_to_image(p[1], oversampling_factor)
        for p in x0_variations
    ]
    row["y_fitted_x0_variations"] = [
        y_min + fit_utils.oversampled_to_image(p[2], oversampling_factor)
        for p in x0_variations
    ]
    row["scale_radius_pixels_x0_variations"] = [p[3] for p in x0_variations]
    row["axis_ratio_x0_variations"] = [p[4] for p in x0_variations]
    row["position_angle_x0_variations"] = [p[5] for p in x0_variations]
    row["power_law_slope_x0_variations"] = [p[6] for p in x0_variations]
    row["local_background_x0_variations"] = [p[7] for p in x0_variations]

    # and the likelihood of the different starting points
    row["log_likelihood_x0_variations"] = x0_likelihoods


# ======================================================================================
#
# Then write this output catalog
#
# ======================================================================================
# Before saving we need to put all the columns to the same length, which we can do
# by padding with nans, which are easy to remove
def pad(array, total_length):
    final_array = np.zeros(total_length) * np.nan
    final_array[: len(array)] = array
    return final_array


max_length_hist = max([len(row["log_luminosity"]) for row in clusters_table])
for col in new_cols:
    clusters_table[col] = [pad(row[col], max_length_hist) for row in clusters_table]

clusters_table.write(str(final_catalog), format="hdf5", path="table", overwrite=True)
