"""
artificial_cluster_catalog.py
Creates a catalog of artificial clusters with known parameters that will be used to
create an image of artificial clusters
"""

import sys
from pathlib import Path
from astropy import table
from scipy import spatial, optimize
import numpy as np

# set random seed for reproducibility
np.random.seed(41)

# ======================================================================================
#
# Load the catalogs that were passed in
#
# ======================================================================================
catalog_name = Path(sys.argv[1]).resolve()
field = sys.argv[2]
# find the cluster catalog for the field of interest
cat_legus = table.Table.read(sys.argv[3], format="ascii.ecsv")

# create the empty catalog we'll fill later
catalog = table.Table([], names=[])

# ======================================================================================
#
# copy some functions to get the true effective radius
#
# ======================================================================================
# I can't import these easily, unfortunately, which is why I copy them here
def logistic(eta):
    """
    This is the fit to the slopes as a function of eta

    These slopes are used in the ellipticity correction.
    :param eta: Eta (power law slope)
    :return: The slope to go in ellipticity_correction
    """
    ymax = 0.57902801
    scale = 0.2664717
    eta_0 = 0.92404378
    offset = 0.07298404
    return ymax / (1 + np.exp((eta_0 - eta) / scale)) - offset


def ellipticy_correction(q, eta):
    """
    Correction for ellipticity. This gives R_eff(q) / R_eff(q=1)

    This is a generalized form of the simplified form used in Ryon's analysis. It's
    simply a line of arbitrary slope passing through (q=1, correction=1) as circular
    clusters need no correction. This lets us write the correction in point slope form
    as:
    y - 1 = m (q - 1)
    y = 1 + m (q - 1)

    Note that when m = 0.5, this simplifies to y = (1 + q) * 0.5, as used in Ryon.
    The slope here (m) is determined by fitting it as a function of eta.
    """
    return 1 + logistic(eta) * (q - 1)


def eff_profile_r_eff_with_rmax(eta, a, q, rmax):
    """
    Calculate the effective radius of an EFF profile, assuming a maximum radius.

    :param eta: Power law slope of the EFF profile
    :param a: Scale radius of the EFF profile, in any units.
    :param q: Axis ratio of the profile
    :param rmax: Maximum radius for the profile, in the same units as a.
    :return: Effective radius, in the same units as a and rmax
    """
    # This is such an ugly formula, put it in a few steps
    term_1 = 1 + (1 + (rmax / a) ** 2) ** (1 - eta)
    term_2 = (0.5 * (term_1)) ** (1 / (1 - eta)) - 1
    return ellipticy_correction(q, eta) * a * np.sqrt(term_2)


def get_scale_factor(r_eff, eta, q):
    """
    Calculate the scale factor needed to produce a cluster with a given effective
    radius and other parameters

    :param r_eff: effective radius desired
    :param eta: power law slope
    :param q: axis ratio
    :return: scale factor (a) needed to produce the given r_eff
    """

    def to_minimize(log_a):
        this_r_eff = eff_profile_r_eff_with_rmax(eta, 10 ** log_a, q, 15)
        return (np.log10(r_eff) - np.log10(this_r_eff)) ** 2

    result = optimize.minimize(to_minimize, (0), bounds=[[-6, 1]])
    assert result.success
    return 10 ** result.x[0]


# ======================================================================================
#
# first create the parameters for clusters. I'll add x-y later
#
# ======================================================================================
# Here is how I'll pick the parameters for my fake clusters:
# r_eff - will have a range of values
# magnitude - will be in the typical range of clusters in this image
# power_law_slope - will be iterated over in a grid with peak value
# scale_radius_pixels - will change to match r_eff given other parameters
# axis_ratio - will be a fixed value typical of clusters
# position_angle - will be randomly chosen for each cluster
# I'll create a grid of peak value vs eta. Then as I iterate through this grid, I'll
# slowly increment r_eff. No two fake clusters will have the same r_eff, and I can
# iterate through eta and peak value at a range of r_eff
# set up the axis ratio, which is needed to calculate r_eff
axis_ratio = 0.8

# set up the grid
n_eta = 6
n_mag = 5
n_eta_p_repititions = 5
n_r_eff = n_eta * n_mag * n_eta_p_repititions

eta_values = np.linspace(1.25, 2.5, n_eta)
mag_values = np.linspace(20, 24, n_mag)
r_eff_values = np.logspace(-1.5, 0.5, n_r_eff)

# make the ordered grids.
a_final, eta_final, mag_final = [], [], []
r_eff_idx = 0
for _ in range(n_eta_p_repititions):
    for eta in eta_values:
        for m in mag_values:
            # figure out what the needed scale factor is to make the cluster have the
            # desired r_eff. This has to be in this loop because we change the
            # effective radius for each iteration, so a needs to change too.
            a = get_scale_factor(r_eff_values[r_eff_idx], eta, axis_ratio)

            a_final.append(a)
            eta_final.append(eta)
            mag_final.append(m)

            # go to the next r_eff
            r_eff_idx += 1

# double check the lengths of these arrays
assert len(eta_final) == len(a_final) == len(mag_final) == n_r_eff

# then add this all to the table, including IDs
catalog["ID_PHANGS_CLUSTER"] = range(1, len(a_final) + 1)
catalog["mag_F555W"] = mag_final
catalog["scale_radius_pixels_true"] = a_final
catalog["axis_ratio_true"] = axis_ratio
catalog["position_angle_true"] = np.random.uniform(0, np.pi, len(a_final))
catalog["power_law_slope_true"] = eta_final
catalog["reff_pixels_true"] = r_eff_values

# ======================================================================================
#
# Create the x-y positions of the fake clusters
#
# ======================================================================================
# to select x-y values, I have a few rules. First, clusters must be in the region of
# the image with actual data (i.e. not the edges or chip gaps).
# They must also not be near other clusters, which I define as being outside of
# 30 pixels from them.
x_real = cat_legus["x"]
y_real = cat_legus["y"]

# I'll manually define the regions that are allowed. Then I can make use scipy to test
# whether proposed points are in one of these regions
left_chip_points = [
    (25, 2340),
    (1780, 7000),
    (4000, 5950),
    (2200, 1250),
    (25, 2340),
]
right_chip_points = [
    (2300, 1200),
    (4140, 5900),
    (6450, 4830),
    (4550, 110),
    (2300, 1200),
]

# Use these to create a region such that we can test whether proposed clusters lie
# within it. The idea is to use a convex hull, but this stack overflow does it a bit
# differently in scipy: https://stackoverflow.com/a/16898636
class Hull(object):
    def __init__(self, points):
        self.hull = spatial.Delaunay(np.array(points))

    def test_within(self, x, y):
        return self.hull.find_simplex((x, y)) >= 0


hull_clusters = Hull([(xi, yi) for xi, yi in zip(x_real, y_real)])
hull_left = Hull(left_chip_points)
hull_right = Hull(right_chip_points)

# also get the range so I can restrict where I sample from
max_x_real = np.max(x_real)
max_y_real = np.max(y_real)
min_diff = 30

x_fake, y_fake = np.array([]), np.array([])
for _ in range(len(catalog)):
    # set a counter to track when we have a good set of xy
    good_xy = False
    tracker = 0
    while not good_xy:
        # generate a set of xy. I do center clusters on the oversampled pixel
        # coordinates. Since I have an oversampling factor of 2, the subsampled pixels
        # will range from 0->0.5 and 0.5->1, putting the center of the first at 0.25.
        # I do this because for very small clusters, if they are not centered on
        # pixels, the generation will not get the appropriate peak value. This is only
        # so I can properly generate realistic clusters.
        x = np.random.randint(0, max_x_real) + 0.25
        y = np.random.randint(0, max_y_real) + 0.25
        within_clusters = hull_clusters.test_within(x, y)
        within_chips = hull_left.test_within(x, y) or hull_right.test_within(x, y)
        within_all = within_chips and within_clusters

        # if it's not in range, don't test whether it's close to other clusters.
        if not within_all:
            continue

        # then test it against other clusters. The proposed location must be far from
        # every other cluster in either x or y
        far_x_real = np.abs(x_real - x) > min_diff
        far_y_real = np.abs(y_real - y) > min_diff
        far_real = np.all(np.logical_or(far_x_real, far_y_real))
        # and clusters that have been made so far
        far_x_fake = np.abs(x_fake - x) > min_diff
        far_y_fake = np.abs(y_fake - y) > min_diff
        far_fake = np.all(np.logical_or(far_x_fake, far_y_fake))

        far_all = np.logical_and(far_real, far_fake)

        good_xy = np.logical_and(far_all, within_all)

        # make sure we never have an infinite loop
        tracker += 1
        if tracker > 100:
            raise RuntimeError("It appears we can't place any more clusters.")

    x_fake = np.append(x_fake, x)
    y_fake = np.append(y_fake, y)


catalog["x"] = x_fake
catalog["y"] = y_fake

# ======================================================================================
#
# Then add a few other needed parameters before saving the catalog
#
# ======================================================================================
# My pipeline uses these quantities for later analysis, even if I won't ever look at
# the results of this analysis for these artificial clusters.
catalog["SEDfix_mass"] = 1e4
catalog["SEDfix_mass_limlo"] = 1e4
catalog["SEDfix_age"] = 1e7
catalog["PHANGS_REDUCEDCHISQ_MINCHISQ"] = 1

catalog.write(catalog_name, format="ascii.ecsv")
