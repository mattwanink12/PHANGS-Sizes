"""
density.py - Create a plot showing the density of clusters.

This takes the following parameters:
- Path to save the plot
- Path to save the table containing the lognormal fits
- Path to the public catalog
"""

import sys
from pathlib import Path
from astropy import table
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt
from matplotlib import ticker, gridspec
import betterplotlib as bpl

# set random seed for reproducibility
np.random.seed(123)

bpl.set_style()

# import a colormap function from the mass-size relation plots
code_home_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(code_home_dir / "analysis" / "mass_radius_relation"))
from mass_radius_utils_plotting import create_color_cmap, age_colors
import mass_radius_utils_mle_fitting as mru_mle
import mass_radius_utils as mru

# ======================================================================================
#
# Load the catalogs that were passed in
#
# ======================================================================================
plot_name = Path(sys.argv[1]).resolve()
fits_output_name = Path(sys.argv[2]).resolve()
catalog = table.Table.read(sys.argv[3], format="ascii.ecsv")

# restrict to clusters with good masses and radii
good_mask = np.logical_and(catalog["reliable_radius"], catalog["reliable_mass"])
catalog = catalog[good_mask]

# ======================================================================================
#
# Get the quantities we'll need for the plot
#
# ======================================================================================
density_3d = catalog["density"]
density_3d_log_err = catalog["density_log_err"]
density_2d = catalog["surface_density"]
density_2d_log_err = catalog["surface_density_log_err"]

# turn these errors into linear space for plotting
density_3d_err_lo = density_3d - 10 ** (np.log10(density_3d) - density_3d_log_err)
density_3d_err_hi = 10 ** (np.log10(density_3d) + density_3d_log_err) - density_3d

density_2d_err_lo = density_2d - 10 ** (np.log10(density_2d) - density_2d_log_err)
density_2d_err_hi = 10 ** (np.log10(density_2d) + density_2d_log_err) - density_2d

# then mass
mass = catalog["SEDfix_mass"]
m_err_lo = catalog["SEDfix_mass"] - catalog["SEDfix_mass_limlo"] #catalog["SEDfix_mass"] - catalog["mass_msun_min"]
m_err_hi = catalog["SEDfix_mass_limhi"] - catalog["SEDfix_mass"] #catalog["mass_msun_max"] - catalog["SEDfix_mass"]

# also set up the masks for age
age = catalog["SEDfix_age"]
mask_young = age < 1e7
mask_med = np.logical_and(age >= 1e7, age < 1e8)
mask_old = np.logical_and(age >= 1e8, age < 1e9)
mask_all = age < np.inf


# ======================================================================================
#
# Convenience functions
#
# ======================================================================================
def gaussian(x, mean, variance):
    """
    Normalized Gaussian Function at a given value.

    Is normalized to integrate to 1.

    :param x: value to calculate the Gaussian at
    :param mean: mean value of the Gaussian
    :param variance: Variance of the Gaussian distribution
    :return: log of the likelihood at x
    """
    exp_term = np.exp(-((x - mean) ** 2) / (2 * variance))
    normalization = 1.0 / np.sqrt(2 * np.pi * variance)
    return exp_term * normalization


def fit_gaussian(x_data, y_data):
    def to_minimize(params):
        norm, mean, variance = params
        predicted = norm * gaussian(x_data, mean, variance)
        return np.sum((predicted - y_data) ** 2)

    fit = optimize.minimize(
        to_minimize,
        x0=(1, 1, 1),
        bounds=((None, None), (None, None), (0.001, None)),  # variance is positive
    )
    assert fit.success
    return fit.x


def kde(x_grid, log_x, log_x_err):
    ys = np.zeros(x_grid.size)
    log_x_grid = np.log10(x_grid)

    for lx, lxe in zip(log_x, log_x_err):
        ys += gaussian(log_x_grid, lx, lxe ** 2)

    # # normalize the y value
    ys = np.array(ys)
    ys = 150 * ys / np.sum(ys)  # arbitrary scaling to look nice
    return ys


def contour(ax, mass, r_eff, color, zorder):
    cmap = create_color_cmap(color, 0.1, 0.8)
    common = {
        "percent_levels": [0.5, 0.90],
        "smoothing": [0.15, 0.15],  # dex
        "bin_size": 0.02,  # dex
        "log": True,
        "cmap": cmap,
    }
    ax.density_contourf(mass, r_eff, alpha=0.25, zorder=zorder, **common)
    ax.density_contour(mass, r_eff, zorder=zorder + 1, **common)


# Function to use to set the ticks
@ticker.FuncFormatter
def nice_log_formatter(x, pos):
    exp = np.log10(x)
    # this only works for labels that are factors of 10. Other values will produce
    # misleading results, so check this assumption.
    assert np.isclose(exp, int(exp))

    # for values between 0.01 and 100, just use that value.
    # Otherwise use the log.
    if abs(exp) < 2:
        return f"{x:g}"
    else:
        return f"$10^{exp:.0f}$"


# ======================================================================================
#
# Fit the mass-density relation
#
# ======================================================================================
def fit_mass_density_relation_orthogonal(
    mass, mass_err_lo, mass_err_hi, density, density_log_err
):
    # This is basically copied from the mass-radius relation fitting, but with some
    # changes to make it more suitable here.
    log_mass, log_mass_err_lo, log_mass_err_hi = mru.transform_to_log(
        mass, mass_err_lo, mass_err_hi
    )
    log_density = np.log10(density)

    # and symmetrize the mass errors
    log_mass_err = np.mean([log_mass_err_lo, log_mass_err_hi], axis=0)

    # set some of the convergence criteria parameters for the Powell fitting routine.
    xtol = 1e-10
    ftol = 1e-10
    maxfev = np.inf
    maxiter = np.inf
    # Then try the fitting
    best_fit_result = optimize.minimize(
        mru_mle.negative_log_likelihood,
        args=(
            log_mass,
            log_mass_err,
            log_density,
            density_log_err,
        ),
        bounds=([-1, 10], [None, None], [0, 2]),
        x0=np.array([0.4, 2, 1]),
        method="Powell",
        options={
            "xtol": xtol,
            "ftol": ftol,
            "maxfev": maxfev,
            "maxiter": maxiter,
        },
    )
    assert best_fit_result.success
    return best_fit_result.x


def fit_mass_density_relation_vertical(
    mass, mass_err_lo, mass_err_hi, density, density_log_err
):
    # note that mass errors are not used, but I put them here for consistency with
    # the orthogonal fit function call.
    log_mass = np.log10(mass)
    log_density = np.log10(density)

    def to_minimize(params):
        slope, norm, scatter = params
        variance = density_log_err ** 2 + scatter ** 2
        expected_log_density = norm + slope * (log_mass - 4)

        # calculate the negative log likelihood for each cluster (splitting in two
        # terms), then return the sum over all clusters. Note that we return the
        # negative log likelihood since we are minimizing this, which is equivalent
        # to maximizing the likelihood
        neg_log_likelihood = 0.5 * np.log(2 * np.pi * variance)
        neg_log_likelihood += (expected_log_density - log_density) ** 2 / variance
        return np.sum(neg_log_likelihood)

    fit = optimize.minimize(to_minimize, x0=[0.4, 2, 1], method="Powell")
    assert fit.success
    return fit.x


def bootstrap_fit(fit_func, mass, mass_err_lo, mass_err_hi, density, density_log_err):
    """
    Wrapper for a given fit function to do bootstrapping
    """
    # do the best fit using all data
    best_fit_params = fit_func(mass, mass_err_lo, mass_err_hi, density, density_log_err)

    # then do bootstrapping. Code copied from fitting mass-radius relation
    n_variables = len(best_fit_params)
    param_history = [[] for _ in range(n_variables)]
    param_std_last = [np.inf for _ in range(n_variables)]

    converge_criteria = 0.01  # fractional change in std required for convergence
    converged = [False for _ in range(n_variables)]
    check_spacing = 20  # how many iterations between checking the std
    iteration = 0
    while not all(converged):
        iteration += 1

        # create a new sample of x and y coordinates
        sample_idxs = np.random.randint(0, len(mass), len(mass))

        # fit to this set of data
        this_result = fit_func(
            mass[sample_idxs],
            mass_err_lo[sample_idxs],
            mass_err_hi[sample_idxs],
            density[sample_idxs],
            density_log_err[sample_idxs],
        )
        # store the results
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

    return np.concatenate([best_fit_params, param_std_last])


fit_2d_o = bootstrap_fit(
    fit_mass_density_relation_orthogonal,
    mass,
    m_err_lo,
    m_err_hi,
    density_2d,
    density_2d_log_err,
)
fit_3d_o = bootstrap_fit(
    fit_mass_density_relation_orthogonal,
    mass,
    m_err_lo,
    m_err_hi,
    density_3d,
    density_3d_log_err,
)
fit_2d_v = bootstrap_fit(
    fit_mass_density_relation_vertical,
    mass,
    m_err_lo,
    m_err_hi,
    density_2d,
    density_2d_log_err,
)
fit_3d_v = bootstrap_fit(
    fit_mass_density_relation_vertical,
    mass,
    m_err_lo,
    m_err_hi,
    density_3d,
    density_3d_log_err,
)


def print_parameters(name, params):
    print(
        f"{name+':':<14} "
        f"slope={params[0]:.2f}+-{params[3]:.3f}  "
        f"norm={params[1]:.2f}+-{params[4]:.3f}  "
        f"sigma={params[2]:.2f}+-{params[5]:.3f}  "
    )


# print the parameters
print_parameters("3D Orthogonal", fit_3d_o)
print_parameters("3D Vertical", fit_3d_v)
print_parameters("2D Orthogonal", fit_2d_o)
print_parameters("2D Vertical", fit_2d_v)

# ======================================================================================
#
# Start the table to output the fit parameters to
#
# ======================================================================================
out_file = open(fits_output_name, "w")
# write the header
out_file.write("\t\\begin{tabular}{lcccc}\n")
out_file.write("\t\t\\toprule\n")
out_file.write(
    "\t\tAge & "
    "$\log \mu_{\\rho}$ & "
    "$\sigma_{\\rho}$ & "
    "$\log \mu_{\Sigma}$ & "
    "$\sigma_{\Sigma}$ \\\\ \n"
    "\t\t& ($\Msunpc^{-3}$) & "
    "(dex) & "
    "($\Msunpc^{-2}$) & "
    "(dex) \\\\ \n"
)
out_file.write("\t\t\midrule\n")


def write_fit_line(name, mean_2d, std_2d, mean_3d, std_3d):
    out_file.write(
        f"\t\t{name.replace('-', '--').replace('Age: ', '')} & "
        f"{mean_3d:.2f} & "
        f"{std_3d:.2f} & "
        f"{mean_2d:.2f} & "
        f"{std_2d:.2f} \\\\ \n"
    )


# ======================================================================================
#
# Make the plot
#
# ======================================================================================
fig = plt.figure(figsize=[14, 7.5])
gs = gridspec.GridSpec(
    nrows=2,
    ncols=2,
    wspace=0.3,
    hspace=0.1,
    height_ratios=[1.3, 2],
    top=0.94,
    right=0.98,
    left=0.08,
    bottom=0.1,
)
# have two axes: one for the comparison, and one for the ratio
ax_3_k = fig.add_subplot(gs[0, 0], projection="bpl")
ax_3_m = fig.add_subplot(gs[1, 0], projection="bpl")
ax_2_k = fig.add_subplot(gs[0, 1], projection="bpl")
ax_2_m = fig.add_subplot(gs[1, 1], projection="bpl")
ax_legend = ax_2_m

density_grid = np.logspace(-2, 6, 1000)

for mask, name, color, zorder in zip(
    [mask_all, mask_young, mask_med, mask_old],
    ["All", "Age: 1-10 Myr", "Age: 10-100 Myr", "Age: 100 Myr - 1 Gyr"],
    [None] + age_colors,
    [None, 3, 2, 1],
):
    # create the KDE histogram for the top panel
    kde_2d = kde(density_grid, np.log10(density_2d[mask]), density_2d_log_err[mask])
    kde_3d = kde(density_grid, np.log10(density_3d[mask]), density_3d_log_err[mask])

    # fit this with a Gaussian
    norm_2d, mean_2d, variance_2d = fit_gaussian(np.log10(density_grid), kde_2d)
    norm_3d, mean_3d, variance_3d = fit_gaussian(np.log10(density_grid), kde_3d)

    write_fit_line(name, mean_2d, np.sqrt(variance_2d), mean_3d, np.sqrt(variance_3d))

    # plotting doesn't happen for all subsets. Set None as the color to skip plotting
    if color is None:
        continue

    # plot the KDE histograms
    ax_3_k.plot(density_grid, kde_3d, c=color)
    ax_2_k.plot(density_grid, kde_2d, c=color)

    # # then plot the fit to those histograms
    # plot_fit_2d = norm_2d * gaussian(np.log10(density_grid), mean_2d, variance_2d)
    # plot_fit_3d = norm_3d * gaussian(np.log10(density_grid), mean_3d, variance_3d)
    # ax_2_k.plot(density_grid, plot_fit_2d, ls=":", c=color)
    # ax_3_k.plot(density_grid, plot_fit_3d, ls=":", c=color)

    # plot the contours in the lower panels. Mass is on the y axis
    contour(ax_3_m, density_3d[mask], mass[mask], color, zorder)
    contour(ax_2_m, density_2d[mask], mass[mask], color, zorder)

    # plot dummy lines for use in legend
    ax_legend.plot([0, 0], [0, 0], c=color, label=name)

# # plot the expected mass-density relations. Math derived by using my best fit relation,
# # then substituting in to get it in terms of density.
# R_4 = 2.548
# beta = 0.242
# M_0 = 1e4
# test_masses = np.logspace(2, 6, 100)
# mrr_density_3d = (
#     3 * M_0 ** (3 * beta) * test_masses ** (1 - (3 * beta)) / (8 * np.pi * R_4 ** 3)
# )
# mrr_density_2d = (
#     M_0 ** (2 * beta) * test_masses ** (1 - (2 * beta)) / (2 * np.pi * R_4 ** 2)
# )
# ax_2_m.plot(test_masses, mrr_density_2d, ls="-", c=bpl.almost_black, label="Expected")
# ax_3_m.plot(test_masses, mrr_density_3d, ls="-", c=bpl.almost_black)
#
# # plot the fits to the mass-density relation
# plot_fit_2d_o = (10 ** fit_2d_o[1]) * (test_masses / 1e4) ** (fit_2d_o[0])
# plot_fit_3d_o = (10 ** fit_3d_o[1]) * (test_masses / 1e4) ** (fit_3d_o[0])
# plot_fit_2d_v = (10 ** fit_2d_v[1]) * (test_masses / 1e4) ** (fit_2d_v[0])
# plot_fit_3d_v = (10 ** fit_3d_v[1]) * (test_masses / 1e4) ** (fit_3d_v[0])
# ax_2_m.plot(test_masses, plot_fit_2d_o, ls=":", c=bpl.almost_black, label="Orthogonal")
# ax_3_m.plot(test_masses, plot_fit_3d_o, ls=":", c=bpl.almost_black)
# ax_2_m.plot(test_masses, plot_fit_2d_v, ls="--", c=bpl.almost_black, label="Vertical")
# ax_3_m.plot(test_masses, plot_fit_3d_v, ls="--", c=bpl.almost_black)

# format axes
ax_legend.legend(loc=2, fontsize=14, frameon=False)
for ax in [ax_2_k, ax_2_m, ax_3_k, ax_3_m]:
    ax.tick_params(axis="both", which="major", length=8)  # , direction="in", pad=7)
    ax.tick_params(axis="both", which="minor", length=4)  # , direction="in", pad=7)
    ax.xaxis.set_ticks_position("both")
for ax in [ax_2_k, ax_3_k]:
    ax.set_xscale("log")
    ax.yaxis.set_ticks([0, 0.5, 1.0])
    ax.yaxis.set_ticklabels(["0", "0.5", "1"])
    ax.xaxis.set_major_formatter(nice_log_formatter)
    ax.set_limits(0.1, 1e5, 0)
    ax.tick_params(axis="x", which="both", labelbottom=False, labeltop=True)
for ax in [ax_2_m, ax_3_m]:
    ax.yaxis.set_ticks_position("both")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(nice_log_formatter)
    ax.yaxis.set_major_formatter(nice_log_formatter)
    ax.set_limits(0.1, 1e5, 1e2, 1e6)

# add labels to the axes
label_mass = "Mass [$M_\odot$]"
label_kde_2d = "dN/dlog($\\Sigma_h$)"
label_kde_3d = "dN/dlog($\\rho_h$)"
label_3d = "$\\rho_h$ [$M_\odot$/pc$^3$]"
label_2d = "$\\Sigma_h$ [$M_\odot$/pc$^2$]"
ax_3_k.add_labels("", label_kde_3d)
ax_3_m.add_labels(label_3d, label_mass)
ax_2_k.add_labels("", label_kde_2d)
ax_2_m.add_labels(label_2d, label_mass)

fig.savefig(plot_name)

# then finalize the output file
out_file.write("\t\t\\bottomrule\n")
out_file.write("\t\end{tabular}\n")
out_file.close()
