"""
stacked_distribution.py - Find an analytic distribution that fits the radius
distribution

This takes the following parameters:
- Path to save the plot
- Path to the public catalog
"""

import sys
from pathlib import Path
from astropy import table
import numpy as np
from scipy import optimize, stats, integrate
import betterplotlib as bpl

bpl.set_style()

plot_name = Path(sys.argv[1])

# ======================================================================================
#
# Read the catalogs
#
# ======================================================================================
# Here we stack everything except for a few outliers which we exclude
catalog = table.Table.read(sys.argv[2], format="ascii.ecsv")
mask_not_1566 = catalog["galaxy"] != "ngc1566"
mask_not_7793 = catalog["galaxy"] != "ngc7793"
catalog = catalog[np.logical_and(mask_not_1566, mask_not_7793)]

# restrict to the clusters with reliable radii
catalog = catalog[catalog["reliable_radius"]]

# calculate the mean error in log space, will be used for the KDE smothing
r_eff = catalog["r_eff_pc"]
log_r_eff = np.log10(r_eff)
log_err_lo = log_r_eff - np.log10(r_eff - catalog["r_eff_pc_e-"])
log_err_hi = np.log10(r_eff + catalog["r_eff_pc_e+"]) - log_r_eff

catalog["r_eff_log"] = log_r_eff
catalog["r_eff_log_err"] = 0.5 * (log_err_lo + log_err_hi)


# ======================================================================================
#
# Create the universal pdf
#
# ======================================================================================
def kde(r_eff_grid, log_r_eff, log_r_eff_err):
    ys = np.zeros(r_eff_grid.size)
    log_r_eff_grid = np.log10(r_eff_grid)

    for lr, lre in zip(log_r_eff, log_r_eff_err):
        ys += stats.norm.pdf(log_r_eff_grid, lr, lre)

    # # normalize the y value to integrate to 1.
    ys = np.array(ys)
    area = integrate.trapz(y=ys, x=r_eff_grid)
    ys = ys / area
    return ys


pdf_radii = np.logspace(-1, 1.5, 300)
pdf_stacked = kde(
    pdf_radii,
    catalog["r_eff_log"],
    catalog["r_eff_log_err"],
)
# ======================================================================================
#
# class to fit parameters of a distribution
#
# ======================================================================================
class FitOption(object):
    def __init__(self, name, functional_form, x0):
        self.name = name
        self.x0 = x0
        self.functional_form = functional_form

        # then fit this functional form to get the best parameters
        def to_mimimize(params):
            pdf_model = self.functional_form(pdf_radii, *params)
            return np.sum((pdf_stacked - pdf_model) ** 2)

        fit_return = optimize.minimize(to_mimimize, x0=self.x0)
        assert fit_return.success
        self.params = fit_return.x

        # then calculate the RMS using this
        model = self.functional_form(pdf_radii, *self.params)
        self.rms = np.sqrt(np.mean((model - pdf_stacked) ** 2))

    def plot_fit(self, ax, xs):
        ax.plot(
            xs,
            self.functional_form(xs, *self.params),
            lw=4,
            label=f"{self.name}, RMS={self.rms:.4f}",
        )


# I don't understand scipy stats lognormal form, so I'll define my own. I have a
# separate normalization because the normalization is done in log space for this pdf,
# while normally it's done in linear space
def lognorm(x, mean, log_variance, norm):
    return norm * stats.norm.pdf(np.log10(x), np.log10(mean), log_variance)


# double check Weibull so I know I have the equation correct:
def weibull(x, x0, k, l):
    # https://en.wikipedia.org/wiki/Weibull_distribution#Related_distributions
    d = (x - x0) / l
    r_value = (k / l) * (d ** (k - 1)) * np.exp(-1 * (d ** k))
    # guard against nans
    r_value[x < x0] = 0
    return r_value


# define the options to try, and their initial values
fits = [
    FitOption("Normal", stats.norm.pdf, (3, 1)),
    FitOption("Lognormal", lognorm, (3, 0.1, 0.1)),
    # FitOption("Betaprime", stats.betaprime.pdf, (1, 1, 3)),
    # FitOption("Chi-square", stats.chi2.pdf, (1, 3)),
    FitOption("Gamma", stats.gamma.pdf, (6, 1, 1)),
    # FitOption("Weibull", stats.weibull_min.pdf, (1, 3, 1)),
    FitOption("Weibull", weibull, (0, 2, 3.6)),
]

# ======================================================================================
#
# Make the plot
#
# ======================================================================================
fig, ax = bpl.subplots()

# plot the observed PDF
ax.plot(
    pdf_radii,
    pdf_stacked,
    lw=4,
    zorder=15,
    label=f"Observed, N={len(catalog)}",
)
# then plot the fits
for fit in fits:
    fit.plot_fit(ax, pdf_radii)
    print(fit.name, fit.params)

ax.set_xscale("log")
ax.set_limits(0.1, 20, 0)
ax.set_xticks([0.1, 1, 10])
ax.set_xticklabels(["0.1", "1", "10"])
ax.add_labels("$R_{eff}$ [pc]", "Normalized Density")
ax.legend(frameon=False, fontsize=14, loc=2)


fig.savefig(plot_name)
