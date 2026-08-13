

from __future__ import annotations

import inspect
import logging
import string

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from iminuit import Minuit, cost
from iminuit.util import ValueView
from numpy.polynomial.polynomial import Polynomial
from scipy.stats import chi2

import pygama.math.binned_fitting as pgb
import pygama.math.distributions as pgf
import pygama.math.histogram as pgh
from pygama.math.histogram import get_i_local_maxima
from pygama.math.least_squares import fit_simple_scaling
from pygama.pargen.utils import convert_to_minuit, return_nans

from __future__ import annotations

import copy
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iminuit import Minuit, cost
from iminuit.util import ValueView

import pygama.pargen.energy_cal as pgc
from pygama.math.distributions import gauss_on_step, hpge_peak

def get_survival_fraction(
    energy: np.ndarray,
    cut_param: np.ndarray,
    cut_val: float,
    peak: float,
    eres_pars: float,
    fit_range: tuple = None,
    high_cut: float = None,
    pars: ValueView = None,
    data_mask: np.ndarray = None,
    mode: str = "greater",
    func=hpge_peak,
    fix_step=True,
    display=0,
):
    """
    Function for calculating the survival fraction of a cut
    using a fit to the surviving and failing energy distributions.

    Parameters
    ----------

    energy: array
        array of energies
    cut_param: array
        array of the cut parameter for the survival fraction calculation, should have the same length as energy
    cut_val: float
        value of the cut parameter to be used for the survival fraction calculation
    peak: float
        energy of the peak to be fitted
    eres_pars: float
        energy resolution parameter for the peak
    fit_range: tuple
        range of the fit in keV
    high_cut: float
        upper value for the cut parameter to have a range in the cut value
    pars: iMinuit ValueView
        initial parameters for the fit
    data_mask: array
        mask for the data to be used in the fit
    mode: str
        mode of the cut, either "greater" or "less"
    func: function
        function to be used in the fit
    fix_step: bool
        option to fix the step parameters in the fit
    display: int
        option to display the fit if greater than 1

    Returns
    -------

    sf: float
        survival fraction
    err: float
        error on the survival fraction
    values: iMinuit ValueView
        values of the parameters of the fit
    errors: iMinuit ValueView
        errors on the parameters of the fit
    """
    if data_mask is None:
        data_mask = np.full(len(cut_param), True, dtype=bool)

    if not isinstance(energy, np.ndarray):
        energy = np.array(energy)
        
    if not isinstance(cut_param, np.ndarray):
        cut_param = np.array(cut_param)

    if fit_range is None:
        fit_range = (np.nanmin(energy), np.nanmax(energy))

    nan_idxs = np.isnan(cut_param)
    if high_cut is not None:
        idxs = (cut_param > cut_val) & (cut_param < high_cut) & data_mask
    else:
        if mode == "greater":
            idxs = (cut_param > cut_val) & data_mask
        elif mode == "less":
            idxs = (cut_param < cut_val) & data_mask
        else:
            raise ValueError("mode not recognised")

    if pars is None:
        (pars, errs, cov, _, func, _, _, _) = pgc.unbinned_staged_energy_fit(
            energy,
            func,
            guess_func=energy_guess,
            bounds_func=get_bounds,
            guess_kwargs={"peak": peak, "eres": eres_pars},
            fit_range=fit_range,
        )

    guess_pars_surv = copy.deepcopy(pars)

    # add update guess here for n_sig and n_bkg
    guess_pars_surv = update_guess(func, guess_pars_surv, energy[(~nan_idxs) & (idxs)])

    parguess = {
        "x_lo": pars["x_lo"],
        "x_hi": pars["x_hi"],
        "mu": pars["mu"],
        "sigma": pars["sigma"],
        "hstep1": pars["hstep"],
        "hstep2": pars["hstep"],
        "n_sig": pars["n_sig"],
        "n_bkg": pars["n_bkg"],
        "epsilon_sig": guess_pars_surv["n_sig"] / pars["n_sig"],
        "epsilon_bkg": guess_pars_surv["n_bkg"] / pars["n_bkg"],
    }

    bounds = {
        "n_sig": (0, pars["n_sig"] + pars["n_bkg"]),
        "epsilon_sig": (0, 1),
        "n_bkg": (0, pars["n_bkg"] + pars["n_sig"]),
        "epsilon_bkg": (0, 1),
        "hstep1": (-1, 1),
        "hstep2": (-1, 1),
    }

    if func == hpge_peak:
        parguess.update({"htail": pars["htail"], "tau": pars["tau"]})

    if func == hpge_peak:
        lh = cost.ExtendedUnbinnedNLL(
            energy[(~nan_idxs) & (idxs)], pass_pdf_hpge
        ) + cost.ExtendedUnbinnedNLL(energy[(~nan_idxs) & (~idxs)], fail_pdf_hpge)
    elif func == gauss_on_step:
        lh = cost.ExtendedUnbinnedNLL(
            energy[(~nan_idxs) & (idxs)], pass_pdf_gos
        ) + cost.ExtendedUnbinnedNLL(energy[(~nan_idxs) & (~idxs)], fail_pdf_gos)

    else:
        raise ValueError("Unknown func")

    m = Minuit(lh, **parguess)
    fixed = ["x_lo", "x_hi", "n_sig", "n_bkg", "mu", "sigma"]  # "hstep"
    if func == hpge_peak:
        fixed += ["tau", "htail"]
    if fix_step is True:
        fixed += ["hstep1", "hstep2"]

    m.fixed[fixed] = True
    for arg, val in bounds.items():
        m.limits[arg] = val

    m.simplex().migrad()
    m.hesse()

    sf = m.values["epsilon_sig"] * 100
    err = m.errors["epsilon_sig"] * 100

    if display > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        bins = np.arange(1552, 1612, 1)
        ax1.hist(energy[(~nan_idxs) & (idxs)], bins=bins, histtype="step")

        ax2.hist(energy[(~nan_idxs) & (~idxs)], bins=bins, histtype="step")

        if func == hpge_peak:
            ax1.plot(bins, pass_pdf_hpge(bins, **m.values.to_dict())[1])
            ax2.plot(bins, fail_pdf_hpge(bins, **m.values.to_dict())[1])
        elif func == gauss_on_step:
            ax1.plot(bins, pass_pdf_gos(bins, **m.values.to_dict())[1])
            ax2.plot(bins, fail_pdf_gos(bins, **m.values.to_dict())[1])

        plt.show()

    return sf, err, m.values, m.errors


def unbinned_staged_energy_fit(
    energy,
    func,
    gof_range=None,
    fit_range=None,
    guess=None,
    guess_func=get_hpge_energy_peak_par_guess,
    bounds_func=get_hpge_energy_bounds,
    fixed_func=get_hpge_energy_fixed,
    guess_kwargs=None,
    bounds_kwargs=None,
    fixed_kwargs=None,
    tol=None,
    tail_weight=0,
    allow_tail_drop=True,
    bin_width=None,
    lock_guess=False,
    p_val_threshold=10e-20,
    display=0,
):
    """
    Unbinned fit to energy. This is different to the default fitting as
    it will try different fitting methods and choose the best. This is necessary for the lower statistics.
    """

    if fit_range is None:
        fit_range = (np.nanmin(energy), np.nanmax(energy))

    if gof_range is None:
        gof_range = fit_range

    if bin_width is None:
        init_bin_width = (
            2
            * (np.nanpercentile(energy, 75) - np.nanpercentile(energy, 25))
            * len(energy) ** (-1 / 3)
        )
        init_hist, init_bins, _ = pgh.get_hist(
            energy, dx=init_bin_width, range=fit_range
        )
        try:
            _, init_sigma, _ = pgh.get_gaussian_guess(init_hist, init_bins)
        except IndexError:
            init_hist, init_bins, _ = pgh.get_hist(
                energy, dx=init_bin_width / 2, range=fit_range
            )
            try:
                _, init_sigma, _ = pgh.get_gaussian_guess(init_hist, init_bins)
            except IndexError:
                init_sigma = np.nanstd(energy)
        bin_width = 2 * (init_sigma) * len(energy) ** (-1 / 3)

    gof_hist, gof_bins, gof_var = pgh.get_hist(energy, range=gof_range, dx=bin_width)
    # remove remaining when average counts < 1
    gof_hist, gof_bins, gof_var = average_counts_check(gof_hist, gof_bins, gof_var)
    # sum bins with counts < 5
    gof_hist, gof_bins, gof_var = sum_bins(gof_hist, gof_bins, gof_var)

    if guess is not None:
        if not isinstance(guess, ValueView):
            x0 = convert_to_minuit(guess, func)
        if lock_guess is True:
            x0 = guess
            x0["x_lo"] = fit_range[0]
            x0["x_hi"] = fit_range[1]
        else:
            x0["x_lo"] = fit_range[0]
            x0["x_hi"] = fit_range[1]
            x1 = guess_func(
                energy,
                func,
                fit_range,
                bin_width=bin_width,
                **guess_kwargs if guess_kwargs is not None else {},
            )
            for arg, val in x1.items():
                if arg not in x0:
                    x0[arg] = val
            if len(x0) == len(x1):
                cs, _ = pgb.goodness_of_fit(
                    gof_hist, gof_bins, None, func.pdf_norm, x0, method="Pearson"
                )
                cs2, _ = pgb.goodness_of_fit(
                    gof_hist, gof_bins, None, func.pdf_norm, x1, method="Pearson"
                )
                if cs >= cs2:
                    x0 = x1
            else:
                x0 = x1
    else:
        if func == pgf.hpge_peak:
            x0_notail = guess_func(
                energy,
                pgf.gauss_on_step,
                fit_range,
                bin_width=bin_width,
                **guess_kwargs if guess_kwargs is not None else {},
            )
            c = cost.ExtendedUnbinnedNLL(energy, pgf.gauss_on_step.pdf_ext)
            m = Minuit(c, *x0_notail)
            bounds = bounds_func(
                pgf.gauss_on_step,
                x0_notail,
                **bounds_kwargs if bounds_kwargs is not None else {},
            )
            for arg, val in bounds.items():
                m.limits[arg] = val
            fixed, mask = fixed_func(
                pgf.gauss_on_step,
                **fixed_kwargs if fixed_kwargs is not None else {},
            )
            m.fixed[fixed] = True
            m.simplex().migrad()
            m.hesse()
            x0 = guess_func(
                energy,
                func,
                fit_range,
                bin_width=bin_width,
                **guess_kwargs if guess_kwargs is not None else {},
            )
            cs = pgb.goodness_of_fit(
                gof_hist,
                gof_bins,
                gof_var,
                pgf.gauss_on_step.get_pdf,
                m.values,
                method="Pearson",
                scale_bins=True,
            )
            cs = (cs[0], cs[1] + len(np.where(mask)[0]))
            p_val = chi2.sf(cs[0], cs[1])
            if m.valid and (p_val > 0):
                for arg in x0_notail.to_dict():
                    x0[arg] = x0_notail[arg]

        else:
            x0 = guess_func(
                energy,
                func,
                fit_range,
                bin_width=bin_width,
                **guess_kwargs if guess_kwargs is not None else {},
            )

    if (func == pgf.hpge_peak) and allow_tail_drop is True:
        fit_no_tail = unbinned_staged_energy_fit(
            energy,
            func=pgf.gauss_on_step,
            gof_range=gof_range,
            fit_range=fit_range,
            guess=None,
            guess_func=guess_func,
            bounds_func=bounds_func,
            fixed_func=fixed_func,
            guess_kwargs=guess_kwargs,
            bounds_kwargs=bounds_kwargs,
            fixed_kwargs=fixed_kwargs,
            tol=tol,
            tail_weight=None,
            allow_tail_drop=False,
            bin_width=bin_width,
        )

        c = cost.ExtendedUnbinnedNLL(energy, func.pdf_ext) + TailPrior(
            energy, func, tail_weight=tail_weight
        )
    else:
        c = cost.ExtendedUnbinnedNLL(energy, func.pdf_ext)

    fixed, mask = fixed_func(func, **fixed_kwargs if fixed_kwargs is not None else {})
    bounds = bounds_func(func, x0, **bounds_kwargs if bounds_kwargs is not None else {})

    # try without simplex
    m = Minuit(c, *x0)
    if tol is not None:
        m.tol = tol
    m.fixed[fixed] = True
    for arg, val in bounds.items():
        m.limits[arg] = val
    m.migrad()
    m.hesse()

    valid1 = (
        m.valid
        & (~np.isnan(np.array(m.errors)[mask]).any())
        & (~(np.array(m.errors)[mask] == 0).all())
    )

    cs = pgb.goodness_of_fit(
        gof_hist,
        gof_bins,
        gof_var,
        func.get_pdf,
        m.values,
        method="Pearson",
        scale_bins=True,
    )
    cs = (cs[0], cs[1] + len(np.where(mask)[0]))

    fit1 = (m.values, m.errors, m.covariance, cs, func, mask, valid1, m)

    # Now try with simplex
    m2 = Minuit(c, *x0)
    if tol is not None:
        m2.tol = tol
    m2.fixed[fixed] = True
    for arg, val in bounds.items():
        m2.limits[arg] = val
    m2.simplex().migrad()
    m2.hesse()

    valid2 = (
        m2.valid
        & (~np.isnan(np.array(m2.errors)[mask]).any())
        & (~(np.array(m2.errors)[mask] == 0).all())
    )

    cs2 = pgb.goodness_of_fit(
        gof_hist,
        gof_bins,
        gof_var,
        func.get_pdf,
        m2.values,
        method="Pearson",
        scale_bins=True,
    )
    cs2 = (cs2[0], cs2[1] + len(np.where(mask)[0]))

    fit2 = (m2.values, m2.errors, m2.covariance, cs2, func, mask, valid2, m2)

    frac_errors1 = np.sum(np.abs(np.array(m.errors)[mask] / np.array(m.values)[mask]))
    frac_errors2 = np.sum(np.abs(np.array(m2.errors)[mask] / np.array(m2.values)[mask]))

    if display > 1:
        hist, bins, _ = pgh.get_hist(energy, range=fit_range, dx=bin_width)
        bin_cs = (bins[:-1] + bins[1:]) / 2

        m_fit = func.get_pdf(bin_cs, *m.values) * np.diff(bin_cs)[0]
        m2_fit = func.get_pdf(bin_cs, *m2.values) * np.diff(bin_cs)[0]
        guess_fit = func.get_pdf(bin_cs, *x0) * np.diff(bin_cs)[0]
        plt.figure()
        plt.step(bin_cs, hist, label="hist")
        plt.plot(bin_cs, guess_fit, label="Guess")
        plt.plot(bin_cs, m_fit, label=f"Fit 1: {cs}")
        plt.plot(bin_cs, m2_fit, label=f"Fit 2: {cs2}")
        plt.legend()
        plt.show()

    if valid1 is False and valid2 is False:
        log.debug("Extra simplex needed")
        m = Minuit(c, *x0)
        if tol is not None:
            m.tol = tol
        m.fixed[fixed] = True
        for arg, val in bounds.items():
            m.limits[arg] = val
        m.simplex().simplex().migrad()
        m.hesse()
        cs = pgb.goodness_of_fit(
            gof_hist,
            gof_bins,
            gof_var,
            func.get_pdf,
            m.values,
            method="Pearson",
            scale_bins=True,
        )
        cs = (cs[0], cs[1] + len(np.where(mask)[0]))
        valid3 = (
            m.valid
            & (~np.isnan(np.array(m.errors)[mask]).any())
            & (~(np.array(m.errors)[mask] == 0).all())
        )
        if valid3 is False:
            try:
                m.minos()
                valid3 = (
                    m.valid
                    & (~np.isnan(np.array(m.errors)[mask]).any())
                    & (~(np.array(m.errors)[mask] == 0).all())
                )
            except Exception:
                raise RuntimeError

        fit = (m.values, m.errors, m.covariance, cs, func, mask, valid3, m)

    elif valid2 is False:
        fit = fit1

    elif valid1 is False:
        fit = fit2

    elif cs[0] * 1.05 < cs2[0]:
        fit = fit1

    elif cs2[0] * 1.05 < cs[0]:
        fit = fit2

    elif frac_errors1 < frac_errors2:
        fit = fit1

    elif frac_errors1 > frac_errors2:
        fit = fit2

    else:
        raise RuntimeError

    if (func == pgf.hpge_peak) and allow_tail_drop is True:
        p_val = chi2.sf(fit[3][0], fit[3][1])
        p_val_no_tail = chi2.sf(fit_no_tail[3][0], fit_no_tail[3][1])
        if (
            (p_val_no_tail > p_val)
            or ((fit[0]["htail"] < fit[1]["htail"]) & (p_val_no_tail > p_val_threshold))
            or (
                (fit[0]["htail"] < fit[1]["htail"])
                & (p_val_no_tail < p_val_threshold)
                & (p_val < p_val_threshold)
            )
        ):
            debug_string = f'dropping tail tail val : {fit[0]["htail"]} tail err : {fit[1]["htail"]} '
            debug_string += f"p_val no tail: : {p_val_no_tail} p_val with tail: {p_val}"
            log.debug(debug_string)

            if display > 0:
                m_fit = pgf.gauss_on_step.get_pdf(bin_cs, *fit_no_tail[0])
                m_fit_tail = pgf.hpge_peak.get_pdf(bin_cs, *fit[0])
                plt.figure()
                plt.step(bin_cs, hist, where="mid", label="hist")
                plt.plot(
                    bin_cs,
                    m_fit * np.diff(bin_cs)[0],
                    label=f"Drop tail: {p_val_no_tail}",
                )
                plt.plot(
                    bin_cs,
                    m_fit_tail * np.diff(bin_cs)[0],
                    label=f"Drop tail: {p_val}",
                )
                plt.legend()
                plt.show()

            fit = fit_no_tail
    return fit

def get_areas_fracs(
    params: np.array,
    area_frac_idxs: np.array,
    frac_flag: bool,
    area_flag: bool,
    one_area_flag: bool,
) -> Tuple[np.array, np.array]:
    r"""
    Grab the value(s) of either the fraction or the areas passed in the params array from the :func:`SumDists` call.
    If :func:`SumDists` is in "fracs" mode, then this grabs `f` from the params array and returns fracs = [f, 1-f] and areas of unity.
    If :func:`SumDists` is in "areas" mode, then this grabs `s, b` from the params array and returns unity fracs and areas = [s, b]
    If :func:`SumDists` is in "one_area" mode, then this grabs `s` from the params array and returns unity fracs and areas = [s, 1]

    Parameters
    ----------
    params
        An array containing the shape values from a :func:`SumDists` call
    area_frac_idxs
        An array containing the indices of either the fracs or the areas present in the params array
    frac_flag
        A boolean telling if :func:`SumDists` is in fracs mode or not
    area_flag
        A boolean telling if :func:`SumDists` is in areas mode or not
    one_area_flag
        A boolean telling if :func:`SumDists` is to apply only area to one distribution

    Returns
    -------
    fracs, areas
        Values of the fractions and the areas to post-multiply the sum of the distributions with
    """
    if frac_flag:
        fracs = np.array([params[area_frac_idxs[0]], 1 - params[area_frac_idxs[0]]])
        areas = np.array([1, 1])
    elif area_flag:
        fracs = np.array([1, 1])
        areas = np.array([*params[area_frac_idxs]])
    elif one_area_flag:
        fracs = np.array([1, 1])
        areas = np.array([*params[area_frac_idxs], 1])

    else:
        fracs = np.array([1, 1])
        areas = np.array([1, 1])

    return fracs, areas
def get_pdf(self, x, *params):
    """
    Returns the specified sum of all distributions' :func:`get_pdf` methods.
    """
    pdfs = self.dists
    params = np.array(params)

    fracs, areas = get_areas_fracs(
        params,
        self.area_frac_idxs,
        self.frac_flag,
        self.area_flag,
        self.one_area_flag,
    )

    if self.components:
        return areas[0] * fracs[0] * pdfs[0].get_pdf(
            x, *params[self.par_idxs[0]]
        ), areas[1] * fracs[1] * pdfs[1].get_pdf(x, *params[self.par_idxs[1]])

    else:
        # This is faster than list comprehension
        probs = areas[0] * fracs[0] * pdfs[0].get_pdf(
            x, *params[self.par_idxs[0]]
        ) + areas[1] * fracs[1] * pdfs[1].get_pdf(x, *params[self.par_idxs[1]])
        return probs

def get_hpge_energy_peak_par_guess(
    energy, func, fit_range=None, bin_width=None, mode_guess=None
):
    """
    Get parameter guesses for func fit to peak in hist

    Parameters
    ----------
    energy : array
        An array of energy values in the range around the peak for guessing.
    func : function
        The function to be fit to the peak in the histogram.
    fit_range : tuple, optional
        A tuple specifying the range around the peak to perform the fit. If not provided, the entire range of energy values will be used.
    bin_width : float, optional
        The width of the bins in the histogram. Default is 1.
    mode_guess : float, optional
        A guess for the mode (mu) parameter of the function. If not provided, it will be estimated from the data.

    Returns
    -------
    ValueView
        A ValueView object from iminuit containing the parameter guesses for the function fit.

    Notes
    -----
    This function calculates parameter guesses for fitting a function to a peak in a histogram. It uses various methods to estimate the parameters based on the provided energy values and the selected function.

    If the function is 'gauss_on_step', the following parameters will be estimated:
    - n_sig: Number of signal events in the peak.
    - mu: Mean of the peak.
    - sigma: Standard deviation of the peak.
    - n_bkg: Number of background events.
    - hstep: Height of the step between the peak and the background.
    - x_lo: Lower bound of the fit range.
    - x_hi: Upper bound of the fit range.

    If the function is 'hpge_peak', the following parameters will be estimated:
    - n_sig: Number of signal events in the peak.
    - mu: Mean of the peak.
    - sigma: Standard deviation of the peak.
    - htail: Height of the tail component.
    - tau: Decay constant of the tail component.
    - n_bkg: Number of background events.
    - hstep: Height of the step between the peak and the background.
    - x_lo: Lower bound of the fit range.
    - x_hi: Upper bound of the fit range.

    If the provided function is not implemented, an error will be raised.

    Examples
    --------
    >>> energy = [1, 2, 3, 4, 5]
    >>> func = pgf.gauss_on_step
    >>> fit_range = (2, 4)
    >>> bin_width = 0.5
    >>> mode_guess = 3.5
    >>> get_hpge_energy_peak_par_guess(energy, func, fit_range, bin_width, mode_guess)
    {'n_sig': 3, 'mu': 3.5, 'sigma': 0.5, 'n_bkg': 2, 'hstep': 0.5, 'x_lo': 2, 'x_hi': 4}
    """
    if fit_range is None:
        fit_range = (np.nanmin(energy), np.nanmax(energy))

    energy = energy[(energy >= fit_range[0]) & (energy <= fit_range[1])]
    if bin_width is None:
        init_bin_width = (
            2
            * (np.nanpercentile(energy, 75) - np.nanpercentile(energy, 25))
            * len(energy) ** (-1 / 3)
        )
        init_hist, init_bins, _ = pgh.get_hist(
            energy, dx=init_bin_width, range=fit_range
        )
        try:
            _, init_sigma, _ = pgh.get_gaussian_guess(init_hist, init_bins)
        except IndexError:
            init_hist, init_bins, _ = pgh.get_hist(
                energy, dx=init_bin_width / 2, range=fit_range
            )
            try:
                _, init_sigma, _ = pgh.get_gaussian_guess(init_hist, init_bins)
            except IndexError:
                init_sigma = np.nanstd(energy)
        bin_width = 2 * (init_sigma) * len(energy) ** (-1 / 3)

    hist, bins, var = pgh.get_hist(energy, dx=bin_width, range=fit_range)

    if (
        func == pgf.gauss_on_step
        or func == pgf.hpge_peak
        or func == pgf.gauss_on_uniform
        or func == pgf.gauss_on_linear
    ):
        # get mu and height from a gauss fit, also sigma as fallback
        pars, cov = pgb.gauss_mode_width_max(
            hist, bins, var, mode_guess=mode_guess, n_bins=5
        )

        bin_centres = pgh.get_bin_centers(bins)
        if pars is None:
            log.info("get_hpge_energy_peak_par_guess: gauss_mode_width_max failed")
            i_0 = np.argmax(hist)
            mu = bin_centres[i_0]
            height = hist[i_0]
            sigma_guess = None
        else:
            mu = mode_guess if mode_guess is not None else pars[0]
            sigma_guess = pars[1]
            height = pars[2]
        # get bg and step from edges of hist
        bg = np.mean(hist[-10:])
        step = bg - np.mean(hist[:10])
        # get sigma from fwfm with f = 1/sqrt(e)
        try:
            sigma = pgh.get_fwfm(
                0.6065,
                hist,
                bins,
                var,
                mx=height,
                bl=bg - step / 2,
                method="interpolate",
            )[0]
            if (
                sigma <= 0
                or abs(sigma / sigma_guess) > 5
                or sigma > (fit_range[1] - fit_range[0]) / 2
            ):
                raise ValueError
        except ValueError:
            try:
                sigma = pgh.get_fwfm(
                    0.6065,
                    hist,
                    bins,
                    var,
                    mx=height,
                    bl=bg - step / 2,
                    method="fit_slopes",
                )[0]
            except RuntimeError:
                sigma = -1
            if (
                sigma <= 0
                or sigma > (fit_range[1] - fit_range[0]) / 2
                or abs(sigma / sigma_guess) > 5
            ):
                if (
                    sigma_guess is not None
                    and sigma_guess > 0
                    and sigma_guess < (fit_range[1] - fit_range[0]) / 2
                ):
                    sigma = sigma_guess
                else:
                    (_, sigma, _) = pgh.get_gaussian_guess(hist, bins)
                    if (
                        sigma is not None
                        and sigma_guess > 0
                        and sigma_guess < (fit_range[1] - fit_range[0]) / 2
                    ):
                        pass
                    else:
                        log.info(
                            "get_hpge_energy_peak_par_guess: sigma estimation failed"
                        )
                        return {}
        # now compute amp and return
        n_sig = np.sum(
            hist[(bin_centres > mu - 3 * sigma) & (bin_centres < mu + 3 * sigma)]
        )
        n_bkg = np.sum(hist) - n_sig

        parguess = {
            "n_sig": n_sig,
            "mu": mu,
            "sigma": sigma,
            "n_bkg": n_bkg,
            "x_lo": bins[0],
            "x_hi": bins[-1],
        }

        if func == pgf.gauss_on_linear:
            # bg1 = np.mean(hist[-10:])
            # bg2 = np.mean(hist[:10])
            # m = (bg1 - bg2) / (bins[-5] - bins[5])
            # b = bg1 - m * bins[-5]
            parguess["m"] = 0
            parguess["b"] = 1

        elif func == pgf.gauss_on_step or func == pgf.hpge_peak:
            hstep = step / (bg + np.mean(hist[:10]))
            parguess["hstep"] = hstep

            if func == pgf.hpge_peak:
                sigma = sigma * 0.8  # roughly remove some amount due to tail
                # for now hard-coded
                htail = 1.0 / 5
                tau = sigma / 2
                parguess["sigma"] = sigma
                parguess["htail"] = htail
                parguess["tau"] = tau

        for name, guess in parguess.items():
            if np.isnan(guess):
                parguess[name] = 0

    else:
        log.error(f"get_hpge_energy_peak_par_guess not implemented for {func.__name__}")
        return return_nans(func)

    return convert_to_minuit(parguess, func).values


def get_fwhm(
    hist: np.ndarray,
    bins: np.ndarray,
    var: Optional[np.ndarray] = None,
    mx: Optional[Union[float, tuple[float, float]]] = None,
    dmx: Optional[float] = 0,
    bl: Optional[Union[float, tuple[float, float]]] = 0,
    dbl: Optional[float] = 0,
    method: str = "bins_over_f",
    n_slope: int = 3,
) -> tuple[float, float]:
    """Convenience function for the FWHM of data in a histogram

    Typically used by sending slices around a peak. Searches about argmax(hist)
    for the peak to fall by [fraction] from mx to bl

    Parameters
    ----------
    fraction
        The fractional amplitude at which to evaluate the full width
    hist
        The histogram data array containing the peak
    bins
        An array of bin edges for the histogram
    var
        An array of histogram variances. Used with the 'fit_slopes' method
    mx
        The value to use for the max of the peak. If None, np.amax(hist) is
        used.
    dmx
        The uncertainty in mx
    bl
        Used to specify an offset from which to estimate the FWFM.
    dbl
        The uncertainty in the bl
    method
        'bins_over_f' : the simplest method: just take the difference in the bin
            centers that are over [fraction] of max. Only works for high stats and
            FWFM/bin_width >> 1
        'interpolate' : interpolate between the bins that cross the [fraction]
            line.  Works well for high stats and a reasonable number of bins.
            Uncertainty incorporates var, if provided.
        'fit_slopes' : fit over n_slope bins in the vicinity of the FWFM and
            interpolate to get the fractional crossing point. Works okay even
            when stats are moderate but requires enough bins that dx traversed
            by n_slope bins is approximately linear. Incorporates bin variances
            in fit and final uncertainties if provided.
    n_slope
        Number of bins in the vicinity of the FWFM used to interpolate the fractional
        crossing point with the 'fit_slopes' method

    Returns
    -------
    fwhm, dfwhm
        fwfm: the full width at half of the maximum above bl
        dfwfm: the uncertainty in fwhm


    See Also
    --------
    get_fwfm
        Function that computes the FWFM
    """
    if len(bins) == len(hist):
        log.warning(
            "note: this function has been updated to require bins rather than bin_centers. Don't trust this result"
        )
    return get_fwfm(0.5, hist, bins, var, mx, dmx, bl, dbl, method, n_slope)


def get_fwfm(
    fraction: float,
    hist: np.ndarray,
    bins: np.ndarray,
    var: Optional[np.ndarray] = None,
    mx: Optional[Union[float, tuple[float, float]]] = None,
    dmx: Optional[float] = 0,
    bl: Optional[Union[float, tuple[float, float]]] = 0,
    dbl: Optional[float] = 0,
    method: str = "bins_over_f",
    n_slope: int = 3,
) -> tuple[float, float]:
    """
    Estimate the full width at some fraction of the max of data in a histogram

    Typically used by sending slices around a peak. Searches about argmax(hist)
    for the peak to fall by [fraction] from mx to bl

    Parameters
    ----------
    fraction
        The fractional amplitude at which to evaluate the full width
    hist
        The histogram data array containing the peak
    bins
        An array of bin edges for the histogram
    var
        An array of histogram variances. Used with the 'fit_slopes' method
    mx
        The value to use for the max of the peak. If None, np.amax(hist) is
        used.
    dmx
        The uncertainty in mx
    bl
        Used to specify an offset from which to estimate the FWFM.
    dbl
        The uncertainty in the bl
    method
        'bins_over_f' : the simplest method: just take the difference in the bin
            centers that are over [fraction] of max. Only works for high stats and
            FWFM/bin_width >> 1
        'interpolate' : interpolate between the bins that cross the [fraction]
            line.  Works well for high stats and a reasonable number of bins.
            Uncertainty incorporates var, if provided.
        'fit_slopes' : fit over n_slope bins in the vicinity of the FWFM and
            interpolate to get the fractional crossing point. Works okay even
            when stats are moderate but requires enough bins that dx traversed
            by n_slope bins is approximately linear. Incorporates bin variances
            in fit and final uncertainties if provided.
    n_slope
        Number of bins in the vicinity of the FWFM used to interpolate the fractional
        crossing point with the 'fit_slopes' method

    Returns
    -------
    fwfm, dfwfm
        fwfm: the full width at [fraction] of the maximum above bl
        dfwfm: the uncertainty in fwfm

    Examples
    --------
    >>> import pygama.analysis.histograms as pgh
    >>> from numpy.random import normal
    >>> hist, bins, var = pgh.get_hist(normal(size=10000), bins=100, range=(-5,5))
    >>> pgh.get_fwfm(0.5, hist, bins, var, method='bins_over_f')
    (2.2, 0.15919638684132664) # may vary

    >>> pgh.get_fwfm(0.5, hist, bins, var, method='interpolate')
    (2.2041666666666666, 0.09790931254396479) # may vary

    >>> pgh.get_fwfm(0.5, hist, bins, var, method='fit_slopes')
    (2.3083363869003466, 0.10939486522749278) # may vary
    """
    # find bins over [fraction]
    if mx is None:
        mx = np.amax(hist)
        if var is not None and dmx == 0:
            dmx = np.sqrt(var[np.argmax(hist)])
    idxs_over_f = hist > (bl + fraction * (mx - bl))

    # argmax will return the index of the first occurrence of a maximum
    # so we can use it to find the first and last time idxs_over_f is "True"
    bin_lo = np.argmax(idxs_over_f)
    bin_hi = len(idxs_over_f) - np.argmax(idxs_over_f[::-1])
    bin_centers = get_bin_centers(bins)

    # precalc dheight: uncertainty in height used as the threshold
    dheight2 = (fraction * dmx) ** 2 + ((1 - fraction) * dbl) ** 2

    if method == "bins_over_f":
        # the simplest method: just take the difference in the bin centers
        fwfm = bin_centers[bin_hi] - bin_centers[bin_lo]

        # compute rough uncertainty as [bin width] (+) [dheight / slope]
        dx = bin_centers[bin_lo] - bin_centers[bin_lo - 1]
        dy = hist[bin_lo] - hist[bin_lo - 1]
        if dy == 0:
            dy = (hist[bin_lo + 1] - hist[bin_lo - 2]) / 3
        dfwfm2 = dx**2 + dheight2 * (dx / dy) ** 2
        dx = bin_centers[bin_hi + 1] - bin_centers[bin_hi]
        dy = hist[bin_hi] - hist[bin_hi + 1]
        if dy == 0:
            dy = (hist[bin_hi - 1] - hist[bin_hi + 2]) / 3
        dfwfm2 += dx**2 + dheight2 * (dx / dy) ** 2
        return fwfm, np.sqrt(dfwfm2)

    elif method == "interpolate":
        # interpolate between the two bins that cross the [fraction] line
        # works well for high stats
        if bin_lo < 1 or bin_hi >= len(hist) - 1:
            raise ValueError(f"Can't interpolate ({bin_lo}, {bin_hi})")

        val_f = bl + fraction * (mx - bl)

        # x_lo
        dx = bin_centers[bin_lo] - bin_centers[bin_lo - 1]
        dhf = val_f - hist[bin_lo - 1]
        dh = hist[bin_lo] - hist[bin_lo - 1]
        x_lo = bin_centers[bin_lo - 1] + dx * dhf / dh
        # uncertainty
        dx2_lo = 0
        if var is not None:
            dx2_lo = (dhf / dh) ** 2 * var[bin_lo] + ((dh - dhf) / dh) ** 2 * var[
                bin_lo - 1
            ]
            dx2_lo *= (dx / dh) ** 2
        dd_dh = -dx / dh

        # x_hi
        dx = bin_centers[bin_hi + 1] - bin_centers[bin_hi]
        dhf = hist[bin_hi] - val_f
        dh = hist[bin_hi] - hist[bin_hi + 1]
        if dh == 0:
            raise ValueError("Interpolation failed, dh == 0")
        x_hi = bin_centers[bin_hi] + dx * dhf / dh
        if x_hi < x_lo:
            raise ValueError("Interpolation produced negative fwfm")
        # uncertainty
        dx2_hi = 0
        if var is not None:
            dx2_hi = (dhf / dh) ** 2 * var[bin_hi + 1] + ((dh - dhf) / dh) ** 2 * var[
                bin_hi
            ]
            dx2_hi *= (dx / dh) ** 2
        dd_dh += dx / dh

        return x_hi - x_lo, np.sqrt(dx2_lo + dx2_hi + dd_dh**2 * dheight2)

    elif method == "fit_slopes":
        # evaluate the [fraction] point on a line fit to n_slope bins near the crossing.
        # works okay even when stats are moderate
        val_f = bl + fraction * (mx - bl)

        # x_lo
        i_0 = bin_lo - int(np.floor(n_slope / 2))
        if i_0 < 0:
            raise RuntimeError("Fit slopes failed")
        i_n = i_0 + n_slope
        wts = (
            None if var is None else 1 / np.sqrt(var[i_0:i_n])
        )  # fails for any var = 0
        wts = [w if w != np.inf else 0 for w in wts]

        try:
            (m, b), cov = np.polyfit(
                bin_centers[i_0:i_n], hist[i_0:i_n], 1, w=wts, cov="unscaled"
            )
        except np.linalg.LinAlgError:
            raise RuntimeError("LinAlgError in x_lo")
        x_lo = (val_f - b) / m
        # uncertainty
        dxl2 = (
            cov[0, 0] / m**2
            + (cov[1, 1] + dheight2) / (val_f - b) ** 2
            + 2 * cov[0, 1] / (val_f - b) / m
        )
        dxl2 *= x_lo**2

        # x_hi
        i_0 = bin_hi - int(np.floor(n_slope / 2)) + 1
        if i_0 == len(hist):
            raise RuntimeError("Fit slopes failed")

        i_n = i_0 + n_slope
        wts = None if var is None else 1 / np.sqrt(var[i_0:i_n])
        wts = [w if w != np.inf else 0 for w in wts]
        try:
            (m, b), cov = np.polyfit(
                bin_centers[i_0:i_n], hist[i_0:i_n], 1, w=wts, cov="unscaled"
            )
        except np.linalg.LinAlgError:
            raise RuntimeError("LinAlgError in x_hi")
        x_hi = (val_f - b) / m
        if x_hi < x_lo:
            raise RuntimeError("Fit slopes produced negative fwfm")

        # uncertainty
        dxh2 = (
            cov[0, 0] / m**2
            + (cov[1, 1] + dheight2) / (val_f - b) ** 2
            + 2 * cov[0, 1] / (val_f - b) / m
        )
        dxh2 *= x_hi**2

        return x_hi - x_lo, np.sqrt(dxl2 + dxh2)

    else:
        raise NameError(f"Unrecognized method {method}")

def get_gaussian_guess(
    hist: np.ndarray, bins: np.ndarray
) -> tuple[float, float, float]:
    """
    given a hist, gives guesses for mu, sigma, and amplitude

    Parameters
    ----------
    hist
        Array of histogrammed data
    bins
        Array of histogram bins

    Returns
    -------
    guess_mu
        guess for the mu parameter of a Gaussian
    guess_sigma
        guess for the sigma parameter of a Gaussian
    guess_area
        guess for the A parameter of a Gaussian
    """
    if len(bins) == len(hist):
        log.warning(
            "note: this function has been updated to require bins rather than bin_centers. Don't trust this result"
        )

    max_idx = np.argmax(hist)
    guess_mu = (bins[max_idx] + bins[max_idx]) / 2  # bin center
    guess_amp = hist[max_idx]

    # find 50% amp bounds on both sides for a FWHM guess
    guess_sigma = get_fwhm(hist, bins)[0] / 2.355  # FWHM to sigma
    guess_area = guess_amp * guess_sigma * np.sqrt(2 * np.pi)

    return (guess_mu, guess_sigma, guess_area)



import pygama.pargen.survival_fractions as sf
fit_range = (2580, 2645)
peak = 2614.5
cut_value = 0.965
sf_val, sf_err, values, errors = sf.get_survival_fraction(
    energy=E_clean,
    cut_param=aoe_clean,
    cut_val=cut_value,
    peak=peak,        # DEP peak
    fit_range=fitrange,
    mode="greater",
    func=sf.hpge_peak,
    pars=pars_manual,
    display=2,
    eres_pars=1
)