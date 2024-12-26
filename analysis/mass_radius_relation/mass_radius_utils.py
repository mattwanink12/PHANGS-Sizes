from pathlib import Path
import sys

import numpy as np
from astropy import table

# need to add the correct path to import utils
code_home_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(code_home_dir / "pipeline"))
import utils

# ======================================================================================
#
# functions for writing to an output file with fit info
#
# ======================================================================================
def write_fit_results(fit_out_file, name, best_fit_params, fit_history, masses):
    """
    Write the results of one fit to a file

    :param fit_out_file: Opened file object to write these results to
    :param name: Name of the fitted sample
    :param best_fit_params: The 3 best fit parameters: slope, intercept, scatter
    :param fit_history: The history of these 3 parameters, used to find errors
    :param masses: list of cluster masses, will be used to find percentiles
    :return: None, but the info is written to the file
    """
    print_str = f"\t\t{name} & {len(masses)}"
    # the second parameter is the log of clusters at 10^4. Put it back to linear space
    best_fit_params[1] = 10 ** best_fit_params[1]
    fit_history[1] = [10 ** f for f in fit_history[1]]
    for idx in range(len(best_fit_params)):
        # I did check that the error distributions are roughly symmetric, so the
        # standard deviation is a decent estimate of the error. With separate upper and
        # lower limits, they were often the same or very close to it.
        std = np.std(fit_history[idx])
        print_str += f" & {best_fit_params[idx]:.3f} $\pm$ {std:.3f}"

    p_lo_log_m, p_hi_log_m = np.log10(np.percentile(masses, [1, 99]))
    print_str += f" & {p_lo_log_m:.2f} -- {p_hi_log_m:.2f} "
    print_str += "\\\\ \n"
    fit_out_file.write(print_str)


# ======================================================================================
#
# Data handling
#
# ======================================================================================
def make_big_table(table_loc):
    """
    Read all the catalogs passed in, stack them together, and throw out bad clusters

    :param table_loc: string holding the paths to the catalog
    :return: One astropy table with all the good clusters from this sample
    """
    catalogs = table.Table.read(table_loc, format="ascii.ecsv")

    # filter out the clusters that can't be used in fitting the mass-radius relation
    mask = np.logical_and(catalogs["reliable_radius"], catalogs["reliable_mass"])
    return catalogs[mask]


# get some commonly used items from the table and transform them to log properly
def get_my_masses(catalog, mask):
    """
    Get the masses from my catalog, along with their errors

    :param catalog: Catalog to retrieve the masses from
    :param mask: Mask to apply to the data, to restrict to certain clusters
    :return: Tuple with three elements: mass, lower mass error, upper mass error
    """
    mass = catalog["SEDfix_mass"][mask]
    # mass errors are reported as min and max values
    mass_err_lo = mass - catalog["SEDfix_mass_limlo"][mask]
    mass_err_hi = catalog["SEDfix_mass_limhi"][mask] - mass

    return mass, mass_err_lo, mass_err_hi


def get_my_radii(catalog, mask):
    """
    Get the radii from my catalog, along with their errors

    :param catalog: Catalog to retrieve the radii from
    :param mask: Mask to apply to the data, to restrict to certain clusters
    :return: Tuple with three elements: radius, lower radius error, upper radius error
    """
    r_eff = catalog["r_eff_pc"][mask]
    r_eff_err_lo = catalog["r_eff_pc_e-"][mask]
    r_eff_err_hi = catalog["r_eff_pc_e+"][mask]

    return r_eff, r_eff_err_lo, r_eff_err_hi


def get_my_ages(catalog, mask):
    """
    Get the ages from my catalog, along with their errors

    :param catalog: Catalog to retrieve the ages from
    :param mask: Mask to apply to the data, to restrict to certain clusters
    :return: Tuple with three elements: age, lower age error, upper age error
    """
    age = catalog["SEDfix_age"][mask]
    # age errors are reported as min and max values
    age_err_lo = age - catalog["SEDfix_age_limlo"][mask]
    age_err_hi = catalog["SEDfix_age_limhi"][mask] - age

    return age, age_err_lo, age_err_hi


def transform_to_log(mean, err_lo, err_hi):
    """
    Take a value and its error and transform this into the value and its error in log

    :param mean: Original value
    :param err_lo: Lower error bar
    :param err_hi: Upper error bar
    :return: log(mean), lower error in log, upper error in  log
    """
    log_mean = np.log10(mean)
    log_err_lo = log_mean - np.log10(mean - err_lo)
    log_err_hi = np.log10(mean + err_hi) - log_mean

    return log_mean, log_err_lo, log_err_hi
