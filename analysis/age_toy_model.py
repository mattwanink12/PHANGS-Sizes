"""
age_toy_model.py - Create a toy model for the age evolution of the effective radius

This takes the following parameters:
- Path to save the plot
- Path to the file containing the best fit mass-radius relation split by age
- The public catalog
"""

import sys
from pathlib import Path
import numpy as np
from scipy import optimize
from astropy import constants as c
from astropy import units as u
from matplotlib import colors as mpl_colors

import betterplotlib as bpl

bpl.set_style()

# import some utils from the mass radius relation
mrr_dir = Path(__file__).resolve().parent / "mass_radius_relation"
sys.path.append(str(mrr_dir))
import mass_radius_utils as mru
import mass_radius_utils_plotting as mru_p

# Get the input arguments
plot_name = Path(sys.argv[1])
fit_table_loc = Path(sys.argv[2])

# ======================================================================================
#
# handle catalogs
#
# ======================================================================================
catalog = mru.make_big_table(sys.argv[3])
# Filter out clusters older than 1 Gyr
mask = catalog["age_yr"] < 1e9
mass_obs = mru.get_my_masses(catalog, mask)[0] * u.Msun
r_eff_obs = mru.get_my_radii(catalog, mask)[0] * u.pc
age_obs = mru.get_my_ages(catalog, mask)[0] * u.yr

# Then do several splits by age
mask_young = age_obs < 1e7 * u.yr
mask_med = np.logical_and(age_obs >= 1e7 * u.yr, age_obs < 1e8 * u.yr)
mask_old = np.logical_and(age_obs >= 1e8 * u.yr, age_obs < 1e9 * u.yr)
mask_medold = np.logical_or(mask_med, mask_old)

# ======================================================================================
#
# load fit parameters for the age bins
#
# ======================================================================================
def is_fit_line(line):
    return "$\pm$" in line


def get_fit_from_line(line):
    quantities = line.split("&")
    # format nicer
    quantities = [q.strip() for q in quantities]
    name, N, beta, r_4, scatter, percentiles = quantities
    # get rid of the error, only include the quantity
    beta = float(beta.split()[0])
    r_4 = float(r_4.split()[0])
    scatter = float(scatter.split()[0])
    return name, beta, r_4


# then use these to find what we need
fits = dict()
with open(fit_table_loc, "r") as in_file:
    for line in in_file:
        if is_fit_line(line):
            name, beta, r_4 = get_fit_from_line(line)
            if "Age: " in name:
                if "1--10" in name:
                    name = "age1"
                elif "10--100" in name:
                    name = "age2"
                elif "100 Myr" in name:
                    name = "age3"
                else:
                    raise ValueError
                fits[name] = (beta, r_4)

# ======================================================================================
#
# Functions defining the relation as well as some simple evolution
#
# ======================================================================================
def mass_size_relation(mass, beta, r_4):
    return r_4 * u.pc * (mass / (10 ** 4 * u.Msun)) ** beta


def stellar_mass_adiabatic(m_old, m_new, r_old):
    # Portegies Zwart et al. 2010 section 4.3.1 Eq 33
    # Krumholz review section 3.5.2
    r_new = r_old * m_old / m_new
    return r_new


def stellar_mass_rapid(m_old, m_new, r_old):
    # Portegies Zwart et al. 2010 section 4.2.1 Eq 31
    # Krumholz review section 3.5.2
    eta = m_new / m_old
    r_new = r_old * eta / (2 * eta - 1)
    return r_new


def tidal_radius(m_old, m_new, r_old):
    # Krumholz review Equation 9 section 1.3.2
    consts = r_old ** 3 / m_old
    r_new = (consts * m_new) ** (1 / 3)
    return r_new


# ======================================================================================
#
# Functions defining the evolution according to Gieles etal 2010
#
# ======================================================================================
# https://ui.adsabs.harvard.edu/abs/2010MNRAS.408L..16G/abstract
# Equation 6 is the key
# this includes stellar evolution and two-body relaxation. No tidal evolution


def gieles_etal_10_evolution(initial_radius, initial_mass, time):
    # all quantities must have units
    # Equation 6 defines the main thing, but also grabs a few things from elsewhere
    m0 = initial_mass  # shorthand
    t = time  # shorthand
    r0 = initial_radius

    delta = 0.07  # equation 4
    # equation 4 plus text after equation 7 for t_star. I don't have early ages so the
    # minimum doesn't matter to me here.
    t_star = 2e6 * u.year
    chi_t = 3 * (t / t_star) ** (-0.3)  # equation 7

    # equation 1 gives t_rh0
    m_bar = 0.5 * u.Msun
    N = m0 / m_bar
    t_rh0 = 0.138 * np.sqrt(N * r0 ** 3 / (c.G * m_bar * np.log(0.4 * N) ** 2))

    # then fill out equation 6
    term_1 = (t / t_star) ** (2 * delta)
    term_2 = ((chi_t * t) / (t_rh0)) ** (4 / 3)
    r_final = r0 * np.sqrt(term_1 + term_2)

    # final mass given by equation 4
    m_final = m0 * (t / t_star) ** (-delta)
    return m_final.to(u.Msun), r_final.to(u.pc)


# # ======================================================================================
# # duplicate plot from paper to validate this prescription
# # ======================================================================================
# test_mass_initial = np.logspace(3, 10, 100) * u.Msun
# test_r_initial = 10 ** (-3.560 + 0.615 * np.log10(test_mass_initial.to("Msun").value))
# test_r_initial *= u.pc
#
# fig, ax = bpl.subplots(figsize=[7, 7])
# ax.plot(test_mass_initial, test_r_initial, label="Initial")
# # then  go through the different ages
# for age in [10 * u.Myr, 100 * u.Myr, 1 * u.Gyr, 10 * u.Gyr]:
#     this_m, this_r = gieles_etal_10_evolution(test_r_initial, test_mass_initial, age)
#     ax.plot(this_m, this_r, label=age)
#
# ax.add_labels("Mass", "Radius")
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_limits(1e3, 3e8, 0.1, 300)
# ax.legend()
# fig.savefig("test_g10.png")

# ======================================================================================
#
# Functions defining the evolution according to Gieles etal 2016
#
# ======================================================================================
# https://ui.adsabs.harvard.edu/abs/2016MNRAS.463L.103G/abstract
# Equation 12 is essentially the answer here.
# this model is two body relaxation plus tidal shocks. No stellar evolution.
# note their notation uses i for initial rather than 0, and I keep that in my code
g16_f = 3
# zeta is chosen to be 0.5 by G16 right at the end of section 3.2
zeta = 0.5


def gieles_t_dis_equilibrium(mass):
    # t_dis is defined in equation 17. I'll assume the default value for gamma_GMC
    # NOTE THAT THIS IS ONLY ON THE EQUILIBRIUM RELATION
    return 940 * u.Myr * (mass / (10 ** 4 * u.Msun)) ** (2 / 3)


def gieles_t_sh(rho):
    # assumes default value for gamma_GMC
    gamma_gmc = 12.8 * u.Gyr
    return gamma_gmc * (rho / (100 * u.Msun / u.pc ** 3))


def gieles_t_rh(M, rho):
    # kappa is needed for t_rh, G16 use the value for equal mass systems
    kappa = 142 * u.Myr
    return (
        kappa * (M / (1e4 * u.Msun)) * (rho / (1e2 * u.Msun * u.pc ** (-3))) ** (-1 / 2)
    )


# This commented equation does not work, as it uses timescales only valid on the
# equilibrium relation, which is not what I have to start
# def gieles_mass_loss(initial_mass, time):
#     # all quantities should have units
#     # the last paragraph before the conclusion shows how this works.
#     # M_dot / M = 1 / t_dis
#     # M_dot = M / t_dis
#     # I'll numerically integrate this
#     dt = 3 * u.Myr
#     t_now = 0 * u.Myr
#     M_now = initial_mass.copy()
#     assert int(time / dt) == time / dt
#
#     while t_now < time:
#         # calculate the instantaneous t_di
#         t_dis = gieles_t_dis(M_now)
#         # then calculate the mass loss
#         M_dot = M_now / t_dis
#         # don't let the mass go negative
#         M_now = np.maximum(0.1 * u.Msun, M_now - M_dot * dt)
#
#         t_now += dt
#     return M_now
def gieles_etal_16_evolution(initial_radius, initial_mass, end_time):
    # Use Equation 2 to get mass
    # dM = M f dE / E
    # where Equation 4 is used for shocks:
    # dE / E = -dt / tau_sh
    # I'll numerically integrate this
    # then equation 12 to get radius at a given mass

    # use shorthands for the initial values
    r_i = initial_radius
    M_i = initial_mass
    t_end = end_time
    rho_i = calculate_density(M_i, r_i)
    rho_now = rho_i

    dt = 1 * u.Myr
    t_now = 0 * u.Myr
    M_now = initial_mass.copy()

    t_history = [t_now.copy()]
    M_history = [M_now.copy()]
    rho_history = [rho_now.copy()]
    while t_now < t_end:
        # calculate the shock timescale. Mass evolution only comes from shocks,
        # not two-body relaxation
        tau_sh = gieles_t_sh(rho_now)
        # we then use this to determine the mass loss
        dE_E = -dt / tau_sh
        dM = M_now * g16_f * dE_E
        # don't let the mass go negative
        M_now = np.maximum(0.1 * u.Msun, M_now + dM)
        rho_now = gieles_etal_16_density(M_i, M_now, rho_i)

        t_now += dt

        # store variables
        t_history.append(t_now.copy())
        M_history.append(M_now.copy())
        rho_history.append(rho_now.copy())

    # turn history into nice astropy arrays
    t_history = u.Quantity(t_history)
    M_history = u.Quantity(M_history)
    rho_history = u.Quantity(rho_history)

    r_history = density_to_half_mass_radius(rho_history, M_history)
    return t_history, M_history, r_history, rho_history


def gieles_etal_16_density(initial_mass, current_mass, initial_density):
    # quantities must have units

    # shorthands
    M_i = initial_mass
    M = current_mass
    rho_i = initial_density

    # use the A value used in Figure 3, see text right before section 4.2
    A = 0.02 * u.pc ** (-9 / 2) * u.Msun ** (1 / 2)

    # then equation 12 can simply be calculated
    numerator = A * M
    denom_term_1 = A * M_i / (rho_i ** (3 / 2))
    denom_term_2 = (M / M_i) ** (17 / 2 - 9 / (2 * g16_f))
    denominator = 1 + (denom_term_1 - 1) * denom_term_2
    rho = (numerator / denominator) ** (2 / 3)
    return rho


def calculate_density(mass, half_mass_radius):
    # when calculating the density, take (half_mass) / (4/3 pi half_mass_radius^3)
    return 3 * mass / (8 * np.pi * half_mass_radius ** 3)


def density_to_half_mass_radius(density, mass):
    # then turn this back into half mass radius (remember to use half the mass
    return ((3 * mass) / (8 * np.pi * density)) ** (1 / 3)


# # ======================================================================================
# # duplicate plot from paper to validate this prescription
# # ======================================================================================
# # test the gieles etal 16 prescription
# test_rho = 30 * u.Msun / u.pc ** 3
# test_mass_initial = np.logspace(2.5, 5.0, 10) * u.Msun
# test_radius_initial = density_to_half_mass_radius(test_rho, test_mass_initial)
#
# test_run = gieles_etal_16_evolution(test_radius_initial, test_mass_initial, 300 * u.Myr)
# test_t_history, test_M_history, test_r_history, test_rho_history = test_run
# test_idx_30 = np.where(test_t_history == 30 * u.Myr)
# test_idx_300 = np.where(test_t_history == 300 * u.Myr)
#
# test_rho_30 = test_rho_history[test_idx_30]
# test_rho_300 = test_rho_history[test_idx_300]
# test_m_30 = test_M_history[test_idx_30]
# test_m_300 = test_M_history[test_idx_300]
#
# fig, ax = bpl.subplots(figsize=[7, 7])
# ax.scatter(
#     test_mass_initial, [test_rho.to(u.Msun / u.pc ** 3).value] * 10, label="Initial"
# )
# ax.scatter(test_m_30, test_rho_30, label="30 Myr")
# ax.scatter(test_m_300, test_rho_300, label="300 Myr")
# for idx in range(10):
#     test_ms = test_M_history[:, idx]
#     test_rhos = test_rho_history[:, idx]
#     ax.plot(test_ms, test_rhos, c=bpl.almost_black, lw=1, zorder=0)
#
# test_A = 0.02 * u.pc ** (-9 / 2) * u.Msun ** (1 / 2)
# test_eq_m = np.logspace(-1, 6, 1000) * u.Msun
# test_rho_eq = (test_A * test_eq_m) ** (2 / 3)
# ax.plot(
#     test_eq_m.to(u.Msun).value,
#     test_rho_eq.to(u.Msun / u.pc ** 3).value,
#     ls=":",
#     c=bpl.almost_black,
#     lw=1,
#     zorder=0,
# )
#
# ax.add_labels("Mass [$M_\odot$]", "Density [$M_\odot / pc^3$]")
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_limits(1e2, 1e6, 0.1, 1e4)
# ax.legend()
# fig.savefig("test_g16.png")

# ======================================================================================
#
# Functions defining the evolution according to Gieles etal 2016 modified by me
# to not include any mass loss
#
# ======================================================================================
# I derived these equations in my notebook. The basic idea is to take equation 1 of G16,
# assume no mass loss, plug in equation 10 to split the total energy, use equations
# 4 and 7 to get those energies, then turn the density derivative into a radius
# derivative, since we already assumed to mass loss.
# The final equation is:
# dr = r (1 / t_sh + zeta / t_rh) dt
def gieles_etal_16_evolution_no_mass_loss(initial_radius, mass, end_time):
    # use shorthands for the initial values
    r_i = initial_radius
    t_end = end_time
    M = mass

    dt = 1 * u.Myr
    t_now = 0 * u.Myr
    r_now = r_i.copy()

    t_history = [t_now.copy()]
    M_history = [M.copy()]
    r_history = [r_now.copy()]
    while t_now < t_end:
        # calculate the timescales needed
        rho_now = calculate_density(M, r_now)
        tau_sh = gieles_t_sh(rho_now)
        tau_rh = gieles_t_rh(M, rho_now)

        # then calculate and apply the changed value
        dr = r_now * (1 / tau_sh + zeta / tau_rh) * dt
        # don't let the radius go to infinity
        r_now = np.minimum(100 * u.pc, r_now + dr)
        t_now += dt

        # store in history
        t_history.append(t_now.copy())
        r_history.append(r_now.copy())
        M_history.append(M.copy())

    # turn history into nice astropy arrays
    t_history = u.Quantity(t_history)
    r_history = u.Quantity(r_history)
    M_history = u.Quantity(M_history)

    return t_history, M_history, r_history


# ======================================================================================
#
# Functions defining the evolution according to Gieles etal 2016 modified by me
# to include mass loss by both tides and two body relation, but with radius assumed
# to be proportional to tidal radius
#
# ======================================================================================
# I derived these equations in my notebook. The basic idea is to take equation 2 of G16,
# have an analogous one for two body relaxation (to give it mass loss, as it would if
# clusters are tidally limited), then plus in the timescales to get mass loss as a
# function of time. I then can assume the effective radius is proportional to the
# tidal radius. This gives:
# dM / M = -dt (f_sh / tau_sh + f_rlx * zeta / tau_rh)
def gieles_etal_16_evolution_tidal_prop(initial_radius, initial_mass, end_time, f_rlx):
    # use shorthands for the initial values
    r_i = initial_radius
    M_i = initial_mass
    t_end = end_time

    # assume some values for mass loss
    f_sh = 3

    dt = 1 * u.Myr
    t_now = 0 * u.Myr
    r_now = r_i.copy()
    M_now = M_i.copy()

    t_history = [t_now.copy()]
    M_history = [M_now.copy()]
    r_history = [r_now.copy()]

    while t_now < t_end:
        # calculate the timescales needed
        rho_now = calculate_density(M_now, r_now)
        tau_sh = gieles_t_sh(rho_now)
        tau_rh = gieles_t_rh(M_now, rho_now)
        # then calculate and apply the changed value
        dM = -dt * M_now * (f_sh / tau_sh + zeta * f_rlx / tau_rh)
        # don't let the mass go negative
        M_now = np.maximum(0.1 * u.Msun, M_now + dM)

        # assume radius is proportional to tidal radius
        r_now = r_i * (M_now / M_i) ** (1 / 3)
        t_now += dt

        # store in history
        t_history.append(t_now.copy())
        r_history.append(r_now.copy())
        M_history.append(M_now.copy())

    # turn history into nice astropy arrays
    t_history = u.Quantity(t_history)
    r_history = u.Quantity(r_history)
    M_history = u.Quantity(M_history)

    return t_history, M_history, r_history


# ======================================================================================
#
# Functions defining the evolution according to Gieles etal 2016 modified by me
# to include mass loss by both tides and two body relation, with no assumption
# that the radius is proportional to tidal radius
#
# ======================================================================================
# I derived these equations in my notebook. The basic idea is to start with equation 1,
# assume a f_rlx like the last model, but then work everthing through equation 1.
# this gives:
# dr = (r / d) dt [(1 - 8f_sh / 3) / t_sh + zeta(1 - 8 f_rlx / 3) / (tau_rh)]
def gieles_etal_16_evolution_rlx_loss(initial_radius, initial_mass, end_time, f_rlx):
    # use shorthands for the initial values
    r_i = initial_radius
    M_i = initial_mass
    t_end = end_time

    # assume some values for mass loss
    f_sh = 3

    dt = 0.1 * u.Myr
    t_now = 0 * u.Myr
    r_now = r_i.copy()
    M_now = M_i.copy()

    t_history = [t_now.copy()]
    M_history = [M_now.copy()]
    r_history = [r_now.copy()]

    while t_now < t_end:
        # calculate the timescales needed
        rho_now = calculate_density(M_now, r_now)
        tau_sh = gieles_t_sh(rho_now)
        # zeta is chosen to be 0.5 by G16 right at the end of section 3.2
        zeta = 0.5
        # kappa is needed for t_rh, G16 use the value for equal mass systems
        kappa = 142 * u.Myr
        tau_rh = (
            kappa
            * (M_now / (1e4 * u.Msun))
            * (rho_now / (1e2 * u.Msun * u.pc ** (-3))) ** (-1 / 2)
        )
        # then calculate and apply the changed value
        # mass loss is the same as the previous prescription
        dM = -dt * M_now * (f_sh / tau_sh + zeta * f_rlx / tau_rh)
        # but radius is more complicated
        dr_term_1 = (1 - 2 * f_sh) / tau_sh
        dr_term_2 = (1 - 2 * f_rlx) * zeta / tau_rh
        dr = dt * (r_now / 3) * (dr_term_1 + dr_term_2)

        # apply the changes, but don't let them go to crazy values
        M_now = np.maximum(0.1 * u.Msun, M_now + dM)
        r_now = np.maximum(0.01 * u.pc, r_now + dr)

        t_now += dt

        # store in history
        t_history.append(t_now.copy())
        r_history.append(r_now.copy())
        M_history.append(M_now.copy())

    # turn history into nice astropy arrays
    t_history = u.Quantity(t_history)
    r_history = u.Quantity(r_history)
    M_history = u.Quantity(M_history)

    return t_history, M_history, r_history


# ======================================================================================
#
# Run clusters through this evolution - for both mean relation and full clusters
#
# ======================================================================================
mass_toy = np.logspace(2.5, 5, 1000) * u.Msun
reff_t0 = mass_size_relation(mass_toy, *fits["age1"])  # 0.12, 2.3)
reff_bin1_toy = mass_size_relation(mass_toy, *fits["age1"])
reff_bin2_toy = mass_size_relation(mass_toy, *fits["age2"])
reff_bin3_toy = mass_size_relation(mass_toy, *fits["age3"])

# ======================================================================================
# 2010 model
# ======================================================================================
t_history_g10_toy = np.arange(1, 300.01, 1) * u.Myr
history_g10_toy = [
    gieles_etal_10_evolution(reff_t0, mass_toy, t) for t in t_history_g10_toy
]
M_history_g10_toy = u.Quantity([h[0] for h in history_g10_toy])
r_history_g10_toy = u.Quantity([h[1] for h in history_g10_toy])

# ======================================================================================
# 2016 model
# ======================================================================================
(
    t_history_g16_toy,
    M_history_g16_toy,
    r_history_g16_toy,
    rho_history_g16_toy,
) = gieles_etal_16_evolution(reff_t0, mass_toy, 300 * u.Myr)

# then do the same for the full clusters
t_history_obs, M_history_obs, r_history_obs, rho_history_obs = gieles_etal_16_evolution(
    r_eff_obs[mask_young], mass_obs[mask_young], 300 * u.Myr
)

# ======================================================================================
# modified G16 with no mass loss
# ======================================================================================
(
    t_history_g16m_toy,
    M_history_g16m_toy,
    r_history_g16m_toy,
) = gieles_etal_16_evolution_no_mass_loss(reff_t0, mass_toy, 300 * u.Myr)
(
    t_history_g16m_obs,
    M_history_g16m_obs,
    r_history_g16m_obs,
) = gieles_etal_16_evolution_no_mass_loss(
    r_eff_obs[mask_young], mass_obs[mask_young], 300 * u.Myr
)

# ======================================================================================
# modified G16 such that r is proportional to tidal radius
# ======================================================================================
f_rlx = 0.2
(
    t_history_g16t_toy,
    M_history_g16t_toy,
    r_history_g16t_toy,
) = gieles_etal_16_evolution_tidal_prop(reff_t0, mass_toy, 300 * u.Myr, f_rlx)
(
    t_history_g16t_obs,
    M_history_g16t_obs,
    r_history_g16t_obs,
) = gieles_etal_16_evolution_tidal_prop(
    r_eff_obs[mask_young], mass_obs[mask_young], 300 * u.Myr, f_rlx
)

# ======================================================================================
# modified G16 such that relaxation causes mass loss, no tidal proportionality
# ======================================================================================
(
    t_history_g16r_toy,
    M_history_g16r_toy,
    r_history_g16r_toy,
) = gieles_etal_16_evolution_rlx_loss(reff_t0, mass_toy, 300 * u.Myr, f_rlx)
(
    t_history_g16r_obs,
    M_history_g16r_obs,
    r_history_g16r_obs,
) = gieles_etal_16_evolution_rlx_loss(
    r_eff_obs[mask_young], mass_obs[mask_young], 300 * u.Myr, f_rlx
)

# ======================================================================================
#
# Simple fitting routine to get the parameters of resulting relations
#
# ======================================================================================
def negative_log_likelihood(params, xs, ys):
    # first convert the pivot point value into the intercept
    pivot_point_x = 4
    # so params[1] is really y(pivot_point_x) = m (pivot_point_x) + intercept
    intercept = params[1] - params[0] * pivot_point_x

    # calculate the difference
    data_diffs = ys - (params[0] * xs + intercept)

    # calculate the sum of data likelihoods. The total likelihood is the product of
    # individual cluster likelihoods, so when we take the log it turns into a sum of
    # individual log likelihoods.
    return np.sum(data_diffs ** 2)


def fit_mass_size_relation(mass, r_eff):
    log_mass = np.log10(mass.to("Msun").value)
    log_r_eff = np.log10(r_eff.to("pc").value)

    mask = log_mass > 2
    log_mass = log_mass[mask]
    log_r_eff = log_r_eff[mask]

    # Then try the fitting
    best_fit_result = optimize.minimize(
        negative_log_likelihood,
        args=(
            log_mass,
            log_r_eff,
        ),
        bounds=([-1, 1], [None, None]),
        x0=np.array([0.2, np.log10(2)]),
    )
    assert best_fit_result.success
    beta = best_fit_result.x[0]
    r_4 = 10 ** best_fit_result.x[1]
    return beta, r_4


# # ======================================================================================
# #
# # Make the plot
# #
# # ======================================================================================
# def format_params(base_label, beta, r_4):
#     return f"{base_label} - $\\beta={beta:.3f}, r_4={r_4:.3f}$"
#
#
# fig, axs = bpl.subplots(ncols=2, figsize=[20, 7])
# # plot the contours and the mean relation evolution for each model.
# # Start with observed young data set
# mru_p.plot_mass_size_dataset_contour(
#     axs[0],
#     mass_obs[mask_young].to("Msun").value,
#     r_eff_obs[mask_young].to("pc").value,
#     bpl.fade_color(bpl.color_cycle[0]),
#     zorder=0,
# )
# for ax in axs:
#     ax.plot(
#         mass_toy,
#         reff_bin1_toy,
#         c=bpl.color_cycle[0],
#         lw=5,
#         label="Age: 1-10 Myr Observed",
#     )
# # then observed old data set
# # mru_p.plot_mass_size_dataset_contour(
# #     axs[1],
# #     mass_obs[mask_old].to("Msun").value,
# #     r_eff_obs[mask_old].to("pc").value,
# #     bpl.fade_color(bpl.color_cycle[3]),
# #     zorder=0,
# # )
# axs[1].plot(
#     mass_toy,
#     reff_bin3_toy,
#     c=bpl.color_cycle[3],
#     lw=5,
#     label="Age: 100 Myr - 1 Gyr Observed",
# )
#
# # then the Gieles+2010 model
# # mru_p.plot_mass_size_dataset_contour(
# #     axs[1],
# #     m_g10_300myr_obs.to("Msun").value,
# #     r_g10_300myr_obs.to("pc").value,
# #     bpl.fade_color(bpl.color_cycle[5]),
# #     zorder=0,
# # )
# # axs[1].plot(
# #     m_g10_300myr_toy,
# #     r_g10_300myr_toy,
# #     c=bpl.color_cycle[5],
# #     lw=5,
# #     label=format_params(
# #         "G10 - 300 Myr",
# #         *fit_mass_size_relation(m_g10_300myr_toy, r_g10_300myr_toy),
# #     ),
# # )
# # Then the Gieles+2016 model
# # mru_p.plot_mass_size_dataset_contour(
# #     axs[1],
# #     m_g16_300myr_obs.to("Msun").value,
# #     r_g16_300myr_obs.to("pc").value,
# #     bpl.fade_color(bpl.color_cycle[4]),
# #     zorder=0,
# # )
# # axs[1].plot(
# #     m_g16_300myr_toy,
# #     r_g16_300myr_toy,
# #     c=bpl.color_cycle[4],
# #     lw=5,
# #     label=format_params(
# #         "G16 - 300 Myr",
# #         *fit_mass_size_relation(m_g16_300myr_toy, r_g16_300myr_toy),
# #     ),
# # )
# # Then the Gieles+2016 modified model with no mass loss
# # mru_p.plot_mass_size_dataset_contour(
# #     axs[1],
# #     mass_obs[mask_young].to("Msun").value,
# #     r_g16m_300_obs.to("pc").value,
# #     bpl.fade_color(bpl.color_cycle[6]),
# #     zorder=0,
# # )
# axs[1].plot(
#     mass_toy,
#     r_g16m_300_toy,
#     c=bpl.color_cycle[6],
#     lw=5,
#     label="G16 no mass loss - 300 Myr",
# )
# # Then the Gieles+2016 modified model that's not proportional to tidal radius
# # mru_p.plot_mass_size_dataset_contour(
# #     axs[1],
# #     m_g16r_300_obs.to("Msun").value,
# #     r_g16r_300_obs.to("pc").value,
# #     bpl.fade_color(bpl.color_cycle[1]),
# #     zorder=0,
# # )
# axs[1].plot(
#     m_g16r_300_toy,
#     r_g16r_300_toy,
#     c=bpl.color_cycle[4],
#     lw=5,
#     label="G16 - $f_{rlx}$=" + str(f_rlx) + " - 300 Myr",
# )
# # Then the Gieles+2016 modified model that's proportional to tidal radius
# # mru_p.plot_mass_size_dataset_contour(
# #     axs[1],
# #     m_g16t_300_obs.to("Msun").value,
# #     r_g16t_300_obs.to("pc").value,
# #     bpl.fade_color(bpl.color_cycle[7]),
# #     zorder=0,
# # )
# axs[1].plot(
#     m_g16t_300_toy,
#     r_g16t_300_toy,
#     c=bpl.color_cycle[5],
#     lw=5,
#     label="G16 $r_{eff} \propto r_{tid}$ - $f_{rlx}$=" + str(f_rlx) + " - 300 Myr",
# )
#
# # plot the determined initial values
# axs[0].plot(
#     mass_toy,
#     r_initial,
#     c=bpl.color_cycle[1],
#     lw=5,
#     label=format_params("Initial Relation", initial_beta, initial_r_4),
# )
# axs[0].plot(
#     m_initial_to_10,
#     r_initial_to_10,
#     c=bpl.color_cycle[2],
#     lw=5,
#     label=f"Initial Relation Evolved by G10 to {initial_age.to('Myr'):.1f}",
# )
#
# for ax in axs:
#     mru_p.format_mass_size_plot(ax)
# axs[0].legend(loc=2, fontsize=16)
# axs[1].legend(loc=4, fontsize=14)
# fig.savefig(plot_name)

# ======================================================================================
#
# Make a version of this plot showing toy arrows
#
# ======================================================================================
def fade_color(color, f_s=0.666, f_v=0.75):
    rgb = mpl_colors.to_rgb(color)
    hsv = mpl_colors.rgb_to_hsv(rgb)
    h = hsv[0]
    s = hsv[1] * (1 - f_s)  # remove saturation
    # make things lighter - 3/4 of the way to full brightness. In combination
    # with the reduction in saturation, it basically fades things whiter
    v = hsv[2] + (1.0 - hsv[2]) * f_v

    return mpl_colors.hsv_to_rgb([h, s, v])


# set the values used for these fade parameters
f_fill = 0.75
f_line = 0.6

fig, ax = bpl.subplots(figsize=[8, 5.5])
# plot the mean relation evolution for each model.
# Start with contours for all the data sets
# colors are manually selected to look okay here
for mask, color, name in zip(
    [mask_young, mask_med, mask_old],
    mru_p.age_colors,
    ["1-10 Myr Observed", "10-100 Myr Observed", "100 Myr - 1 Gyr Observed"],
):
    # make my own version of the contour function in my mass radius utils
    common = {
        "percent_levels": [0.75],
        "smoothing": 0.08,  # dex
        "bin_size": 0.01,  # dex
        "log": True,
    }
    ax.density_contourf(
        mass_obs[mask].to("Msun").value,
        r_eff_obs[mask].to("pc").value,
        alpha=0.6,
        zorder=0,
        colors=[fade_color(color, f_fill, f_fill)],
        **common
    )
    ax.density_contour(
        mass_obs[mask].to("Msun").value,
        r_eff_obs[mask].to("pc").value,
        zorder=1,
        colors=[fade_color(color, f_line, f_line)],
        **common
    )
    # mru_p.add_percentile_lines(
    #     ax,
    #     mass_obs[mask].to("Msun").value,
    #     r_eff_obs[mask].to("pc").value,
    #     color=color,
    #     percentiles=[50],
    #     label_percents=False,
    #     label_legend=name,
    #     lw_50=2,
    # )
    # ax.plot(
    #     [1, 1],
    #     [1, 1],
    #     c=color,
    #     zorder=200,
    #     label=name,
    # )

# # plot the initial mass-radius relation
# ax.plot(
#     mass_toy,
#     reff_t0,
#     lw=3,
#     c=bpl.color_cycle[2],
#     zorder=500,
#     label="Example t=0 Relation",
# )


def plot_fit_restricted_range(mass_plot, r_eff_plot, mass_obs, color, zorder, label):
    # get the 1-99 percentiles
    m_min, m_max = np.percentile(mass_obs, [1, 99])
    good_idx = np.logical_and(mass_plot > m_min, mass_plot < m_max)
    x_values = mass_plot[good_idx]
    y_values = r_eff_plot[good_idx]
    ax.plot(x_values, y_values, c=color, lw=3, zorder=zorder, label=label)


# then lines for the observed relations
plot_fit_restricted_range(
    mass_toy,
    reff_bin1_toy,
    mass_obs[mask_young],
    mru_p.color_young,
    1000,
    "1-10 Myr Observed",
)
plot_fit_restricted_range(
    mass_toy,
    reff_bin2_toy,
    mass_obs[mask_med],
    mru_p.color_med,
    100,
    "10-100 Myr Observed",
)
plot_fit_restricted_range(
    mass_toy,
    reff_bin3_toy,
    mass_obs[mask_old],
    mru_p.color_old,
    100,
    "100 Myr - 1 Gyr Observed",
)

# Then the models. Each has their own color
c_g16 = "#c9733a"
c_g16m = "#6E0004"
c_g16t = "#95B125"  # "#95B125"  # 79993D # 81B521
# Just plot dummy lines  to include in legend, the actual tracks will done later.
ax.plot([1, 1], [1, 1], c=c_g16, zorder=200, label="GR16")
ax.plot([1, 1], [1, 1], c=c_g16m, zorder=200, label="No Mass Loss")
ax.plot([1, 1], [1, 1], c=c_g16t, zorder=200, label="$R_{eff} \propto r_J$")

plot_limits = 1e2, 3e5, 0.2, 35
for t_history, m_history, r_history, color, fs in zip(
    [t_history_g16_toy, t_history_g16m_toy, t_history_g16t_toy],
    [M_history_g16_toy, M_history_g16m_toy, M_history_g16t_toy],
    [r_history_g16_toy, r_history_g16m_toy, r_history_g16t_toy],
    [c_g16, c_g16m, c_g16t],
    [
        np.concatenate([[0.15, 0.275], np.arange(0.35, 0.96, 0.05)]),
        np.arange(0.45, 0.96, 0.05),
        np.concatenate([[0.3, 0.4], np.arange(0.5, 0.96, 0.05)]),
    ],
):
    # ax.plot(m_model, r_model, lw=2, c=color)
    idxs = [int(f * len(mass_toy)) - 1 for f in fs]
    for idx in idxs:
        m_plot = m_history[:, idx]
        r_plot = r_history[:, idx]

        for t_max, lw in zip([300] * u.Myr, [2.5]):
            t_idxs = t_history <= t_max
            ax.plot(m_plot[t_idxs], r_plot[t_idxs], color=color, lw=lw, zorder=200)

        # The arrows are way more complicated than I expected. I want them to start from
        # the end of the line with the appropriate angle
        # get the log of the values. Doing it separately this way is cleaner
        log_r1 = np.log10(r_plot[-1].to("pc").value)
        log_r2 = np.log10(r_plot[-2].to("pc").value)
        log_m1 = np.log10(m_plot[-1].to("Msun").value)
        log_m2 = np.log10(m_plot[-2].to("Msun").value)

        d_logm = log_m1 - log_m2
        d_logr = log_r1 - log_r2
        theta = np.arctan2(d_logr, d_logm)

        arrow_length = 0.03  # dex, chosen by experimentation
        arrow_d_logm = arrow_length * np.cos(theta)
        arrow_d_logr = arrow_length * np.sin(theta)

        arrow_end_m = 10 ** (log_m1 + arrow_d_logm)
        arrow_end_r = 10 ** (log_r1 + arrow_d_logr)

        # only plot arrows that will be in the plot
        if (
            plot_limits[0] < arrow_end_m < plot_limits[1]
            and plot_limits[2] < arrow_end_r < plot_limits[3]
        ):
            ax.annotate(
                "",
                xy=(arrow_end_m, arrow_end_r),
                xytext=(m_plot[-1].to("Msun").value, r_plot[-1].to("pc").value),
                arrowprops={
                    "edgecolor": "none",
                    "facecolor": color,
                    "width": 1e-10,
                    "headwidth": 9,
                    "headlength": 9,
                },
                zorder=300,
            )

# put arrows in the legend. This is a hack, I just place them in the location to
# make them appear as part of the legend
arrowprops = {
    "edgecolor": "none",
    "width": 1,
    "headwidth": 9,
    "headlength": 9,
}  # 10.5 - 10.7,
for y, color in zip([10.6, 7.9, 5.9], [c_g16, c_g16m, c_g16t]):
    arrowprops["facecolor"] = color
    x0 = 210
    ax.annotate("", xy=(x0, y), xytext=(x0 - 1, y), arrowprops=arrowprops, zorder=300)

mru_p.format_mass_size_plot(ax)
ax.legend(loc=2, fontsize=12, frameon=False)
ax.set_limits(*plot_limits)
fig.savefig(plot_name)
