"""
mass_radius_legus_external.py
- Fit the mass-size relation for all LEGUS clusters plus clusters from M31
"""
import sys
from pathlib import Path

import numpy as np
import betterplotlib as bpl

import mass_radius_utils as mru
import mass_radius_utils_mle_fitting as mru_mle
import mass_radius_utils_external_data as mru_d

bpl.set_style()

# load the parameters the user passed in
output_name = Path(sys.argv[1])
fit_out_file = open(output_name, "w")
big_catalog = mru.make_big_table(sys.argv[2])

# Filter out clusters older than 1 Gyr
mask = big_catalog["age_yr"] < 1e9
mass, mass_err_lo, mass_err_hi = mru.get_my_masses(big_catalog, mask)
r_eff, r_eff_err_lo, r_eff_err_hi = mru.get_my_radii(big_catalog, mask)

# Then I need to combine all the datasets together. To do this, I iterate through the
# functions that are used to get the data, then concatenate everything together. This
# also lets me plot the datasets one by one. But to do this I need a dummy function to
# handle the legus data
def dummy_legus():
    return mass, mass_err_lo, mass_err_hi, r_eff, r_eff_err_lo, r_eff_err_hi


funcs = [
    dummy_legus,
    mru_d.get_lmc_smc_ocs_mackey_gilmore,
    mru_d.get_m31_open_clusters,
    mru_d.get_m82_sscs_cuevas_otahola,
    mru_d.get_m83_clusters,
]

mass_total = np.concatenate([func()[0] for func in funcs])
mass_err_lo_total = np.concatenate([func()[1] for func in funcs])
mass_err_hi_total = np.concatenate([func()[2] for func in funcs])
r_eff_total = np.concatenate([func()[3] for func in funcs])
r_eff_err_lo_total = np.concatenate([func()[4] for func in funcs])
r_eff_err_hi_total = np.concatenate([func()[5] for func in funcs])

# Then we can do the fit
fit, fit_history = mru_mle.fit_mass_size_relation(
    mass_total,
    mass_err_lo_total,
    mass_err_hi_total,
    r_eff_total,
    r_eff_err_lo_total,
    r_eff_err_hi_total,
    fit_mass_upper_limit=1e5,
)
mru.write_fit_results(
    fit_out_file, "LEGUS + External Galaxies", fit, fit_history, mass_total
)

# finalize output file
fit_out_file.close()
