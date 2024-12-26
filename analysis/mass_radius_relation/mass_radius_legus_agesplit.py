"""
mass_radius_legus_agesplit.py
- Fit the mass-size relation for all LEGUS clusters, splitting by decade in age
"""
import sys
from pathlib import Path

import numpy as np
import betterplotlib as bpl

import mass_radius_utils as mru
import mass_radius_utils_mle_fitting as mru_mle
import mass_radius_utils_plotting as mru_p

bpl.set_style()

# load the parameters the user passed in
plot_name = Path(sys.argv[1])
output_name = Path(sys.argv[2])
fit_out_file = open(output_name, "w")
big_catalog = mru.make_big_table(sys.argv[3])

# Filter out clusters older than 1 Gyr
mask = big_catalog["SEDfix_age"] < 1e9
mass, mass_err_lo, mass_err_hi = mru.get_my_masses(big_catalog, mask)
r_eff, r_eff_err_lo, r_eff_err_hi = mru.get_my_radii(big_catalog, mask)
age, _, _ = mru.get_my_ages(big_catalog, mask)

# Then do several splits by age
mask_young = age < 1e7
mask_med = np.logical_and(age >= 1e7, age < 1e8)
mask_old = np.logical_and(age >= 1e8, age < 1e9)

fig, ax = bpl.subplots(figsize=[8, 5.5])
for age_mask, name, color, zorder in zip(
    [mask_young, mask_med, mask_old],
    ["Age: 1--10 Myr", "Age: 10--100 Myr", "Age: 100 Myr -- 1 Gyr"],
    mru_p.age_colors,
    [1, 3, 2],
):
    fit, fit_history = mru_mle.fit_mass_size_relation(
        mass[age_mask],
        mass_err_lo[age_mask],
        mass_err_hi[age_mask],
        r_eff[age_mask],
        r_eff_err_lo[age_mask],
        r_eff_err_hi[age_mask],
        fit_mass_upper_limit=1e5,
    )

    mru_p.plot_mass_size_dataset_contour(
        ax,
        mass[age_mask],
        r_eff[age_mask],
        color,
        zorder=zorder,
        alpha=0.25,
    )
    mru_p.add_percentile_lines(
        ax,
        mass[age_mask],
        r_eff[age_mask],
        color=color,
        percentiles=[50],
        label_percents=False,
        label_legend=f"{name.replace('--', '-')}, N={np.sum(age_mask)}",
    )
    mru_p.plot_best_fit_line(
        ax,
        fit,
        1,
        1e5,
        color,
        fill=False,
        label="",
        ls=":",
    )
    mru.write_fit_results(fit_out_file, name, fit, fit_history, mass[age_mask])
mru_p.format_mass_size_plot(ax, legend_fontsize=13)
fig.savefig(plot_name)

# finalize output file
fit_out_file.close()
