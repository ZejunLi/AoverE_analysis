import os
import gc
import pickle
from turtle import left, right
from typing import Dict, Iterable, Optional, Tuple
from scipy.optimize import curve_fit
import math
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import re
def get_channel(index=0):
    return "ch001" if index == 0 else "ch002"

def gauss(x, A, mu, sigma, c):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + c

def get_pathlist(info, dataset):
    mapping = {
        "cal": info["cal_pathlist"],
        "phy_bg": info["phy_bg_pathlist"],
        "phy_signal": info["phy_signal_pathlist"],
    }
    if dataset not in mapping:
        raise ValueError(f"[ERROR] unknown dataset: {dataset}")
    return mapping[dataset]

def read_array_from_info(
    info,
    lh5,
    dataset="cal",
    channel="ch001",
    table="hit",
    field="trapEftp_cal",
    is_valid_waveform=False,
    copy_array=True,
):
    pathlist = get_pathlist(info, dataset)
    if len(pathlist) == 0:
        raise ValueError(f"[ERROR] {dataset}_pathlist is empty for run {info['run']}")

    group = f"{channel}/{table}"
    obj = safe_lh5_read(lh5, group, pathlist)
    ak_obj = obj.view_as("ak")

    fields = list(ak_obj.fields)

    if field not in fields:
        del obj, ak_obj
        gc.collect()
        raise KeyError(
            f"[ERROR] field '{field}' not found in {group} for run {info['run']}. "
            f"Available fields: {fields}"
        )

    arr = ak.to_numpy(ak_obj[field])

    if is_valid_waveform:
        valid_field = "is_valid_waveform"

        if valid_field not in fields:
            del obj, ak_obj
            gc.collect()
            raise KeyError(
                f"[ERROR] field '{valid_field}' not found in {group} for run {info['run']}. "
                f"Available fields: {fields}"
            )

        is_valid = ak.to_numpy(ak_obj[valid_field]).astype(bool)
        arr = arr[is_valid]

    if copy_array:
        arr = arr.copy()

    del obj, ak_obj
    gc.collect()
    return arr

def safe_lh5_read(lh5, key, files):
    """
    lh5.read sometimes returns (obj, n_rows), sometimes obj.
    This wrapper always returns the object itself.
    """
    out = lh5.read(key, files)
    return out[0] if isinstance(out, tuple) else out

def read_array_from_run(
    run_db,
    run,
    lh5,
    dataset="cal",
    channel="ch001",
    table="hit",
    field="trapEftp_cal",
    is_valid_waveform=False,
    copy_array=True,
):
    if run not in run_db:
        raise KeyError(f"[ERROR] {run} not found in run_db")
    return read_array_from_info(
        run_db[run],
        lh5,
        dataset=dataset,
        channel=channel,
        table=table,
        field=field,
        is_valid_waveform=is_valid_waveform,
        copy_array=copy_array,
    )

def plot_trapeftp(
    run_db,
    run,
    lh5,
    index=0,
    energy_field="trapEftp_cal",
    low_cut_field=None,
    show_low_cut=None,   # None / "true" / "false" / "both"
    bins=400,
    energy_range=None,   # e.g. (1500, 1650)
    yscale="linear",     # "linear" or "log"
    dataset='cal',
    title=None,
    save_fig=None,
    window_nsigma=3.0,
    log_style=True,
    show_plot=True,
):
    """
    Plot calibration energy spectrum directly from cal dataset.

    Parameters
    ----------
    run_db : dict
    run : str
    lh5 : lh5 store/reader
    index : int
        0 -> ch001, 1 -> ch002
    energy_field : str
        usually "trapEftp_cal"
    low_cut_field : str or None
        e.g. "A_max_mw_o_E_Low_Cut"
    show_low_cut : None / "true" / "false" / "both"
        If not None, overlay subsets selected by low_cut_field.
    bins : int
    energy_range : tuple or None
        Histogram range in keV
    yscale : str
        "linear" or "log"
    title : str or None
    save_fig : str or None
    show_plot : bool
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if run not in run_db:
        raise KeyError(f"[ERROR] {run} not found in run_db")

    channel = get_channel(index=index)
    info = run_db[run]

    E_cal = read_array_from_info(
        info,
        lh5,
        dataset="cal",
        channel=channel,
        table="hit",
        field=energy_field,
    )
    E_cal = np.asarray(E_cal)
    E_cal = E_cal[np.isfinite(E_cal)]

    low_cut_bool = None
    if show_low_cut is not None:
        if low_cut_field is None:
            raise ValueError("[ERROR] low_cut_field must be provided when show_low_cut is not None")

        low_cut = read_array_from_info(
            info,
            lh5,
            dataset="cal",
            channel=channel,
            table="hit",
            field=low_cut_field,
        )
        low_cut_bool = np.asarray(low_cut).astype(bool)

        if len(low_cut_bool) != len(read_array_from_info(
            info,
            lh5,
            dataset="cal",
            channel=channel,
            table="hit",
            field=energy_field,
        )):
            raise ValueError("[ERROR] low_cut array length does not match energy array length")

        # apply same finite mask as E_cal
        E_raw = read_array_from_info(
            info,
            lh5,
            dataset=dataset,
            channel=channel,
            table="hit",
            field=energy_field,
        )
        finite_mask = np.isfinite(np.asarray(E_raw))
        low_cut_bool = low_cut_bool[finite_mask]

    if energy_range is None:
        e_min = float(np.nanmin(E_cal))
        e_max = float(np.nanmax(E_cal))
        energy_range = (e_min, e_max)

    counts, edges = np.histogram(E_cal, bins=bins, range=energy_range)
    centers = 0.5 * (edges[:-1] + edges[1:])

    plt.figure(figsize=(8, 5))
    plt.step(centers, counts, where="mid", label=f"all (n={counts.sum()})", linewidth=1.8)

    if show_low_cut is not None:
        if show_low_cut not in {"true", "false", "both"}:
            raise ValueError("[ERROR] show_low_cut must be one of: None, 'true', 'false', 'both'")

        if show_low_cut in {"true", "both"}:
            counts_true, _ = np.histogram(E_cal[low_cut_bool], bins=bins, range=energy_range)
            plt.step(
                centers, counts_true, where="mid",
                label=f"{low_cut_field}=True (n={counts_true.sum()})",
                linewidth=1.5
            )

        if show_low_cut in {"false", "both"}:
            counts_false, _ = np.histogram(E_cal[~low_cut_bool], bins=bins, range=energy_range)
            plt.step(
                centers, counts_false, where="mid",
                label=f"{low_cut_field}=False (n={counts_false.sum()})",
                linewidth=1.5
            )

    plt.xlabel("Energy [keV]")
    plt.ylabel("Counts")
    plt.yscale(yscale)

    if title is None:
        title = f"{run} | {channel} | {dataset} | {energy_field}"
    plt.title(title)

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if log_style == True:
        plt.yscale("log")
    if save_fig is not None:
        plt.savefig(save_fig, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_fig}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

    return {
        "run": run,
        "channel": channel,
        "energy_field": energy_field,
        "low_cut_field": low_cut_field,
        "bins": bins,
        "energy_range": energy_range,
        "counts": counts,
        "edges": edges,
        "centers": centers,
    }

def survival_compton(n_pass: float, n_total: float) -> dict:
    """
    Compton/continuum survival efficiency:
        eps = n_pass / n_total
        sigma = sqrt(eps/n_total * (1-eps))
    """
    if n_total <= 0:
        return _safe_nan_dict()

    eps = n_pass / n_total
    sigma = math.sqrt(max(eps * (1.0 - eps) / n_total, 0.0))
    return {
        "eps": eps,
        "sigma": sigma,
        "n_pass": n_pass,
        "n_total": n_total,
    }

def survival_fep_sep(
    n_peak_pass: float,
    n_peak_total: float,
    n_bkg_pass: float,
    n_bkg_total: float,
) -> dict:
    """
    FEP/SEP survival efficiency with pass/fail Gaussian propagation.

    Definitions:
        n_peak_pass = n_+^f
        n_peak_fail = n_-^f = n_peak_total - n_peak_pass

        n_bkg_pass  = n_+^b
        n_bkg_fail  = n_-^b = n_bkg_total - n_bkg_pass

    eps = (n_+^f - n_+^b) /
          [(n_+^f - n_+^b) + (n_-^f - n_-^b)]
    """

    n_peak_fail = n_peak_total - n_peak_pass
    n_bkg_fail = n_bkg_total - n_bkg_pass

    signal_pass = n_peak_pass - n_bkg_pass
    signal_fail = n_peak_fail - n_bkg_fail
    signal_total = signal_pass + signal_fail

    if signal_total <= 0:
        return _safe_nan_dict(
            signal_pass=signal_pass,
            signal_fail=signal_fail,
            signal_total=signal_total,
        )

    eps = signal_pass / signal_total

    # second version:
    # a = n_+^f, b = n_-^f, c = n_+^b, d = n_-^b
    a = n_peak_pass
    b = n_peak_fail
    c = n_bkg_pass
    d = n_bkg_fail

    numerator_var = (b - d)**2 * (a + c) + (a - c)**2 * (b + d)
    sigma = math.sqrt(max(numerator_var / signal_total**4, 0.0))

    return {
        "eps": eps,
        "sigma": sigma,

        "signal_pass": signal_pass,
        "signal_fail": signal_fail,
        "signal_total": signal_total,

        "n_peak_pass": n_peak_pass,
        "n_peak_fail": n_peak_fail,
        "n_peak_total": n_peak_total,

        "n_bkg_pass": n_bkg_pass,
        "n_bkg_fail": n_bkg_fail,
        "n_bkg_total": n_bkg_total,
    } 
    
def survival_dep(n_dep_pass: float, n_dep_total: float,
                 nI_pass: float, nI_total: float,
                 nII_pass: float, nII_total: float,
                 nIII_pass: float, nIII_total: float) -> dict:
    """
    DEP survival efficiency:
        n_bkg      = nI + 2*nII - nIII
        n_bkg_pass = nI_pass + 2*nII_pass - nIII_pass
        eps        = (n_dep_pass - n_bkg_pass) / (n_dep_total - n_bkg)

    Uncertainty from standard propagation with independent Poisson counts.
    """
    
    # fail counts
    n_dep_fail = n_dep_total - n_dep_pass
    nI_fail    = nI_total - nI_pass
    nII_fail   = nII_total - nII_pass
    nIII_fail  = nIII_total - nIII_pass
    n_bkg_pass = nI_pass + 2.0 * nII_pass - nIII_pass
    n_bkg_fail = nI_fail + 2.0 * nII_fail - nIII_fail
    n_bkg_total = n_bkg_pass + n_bkg_fail
    # background-subtracted signal counts
    s_pass = n_dep_pass - nI_pass - 2.0 * nII_pass + nIII_pass
    s_fail = n_dep_fail - nI_fail - 2.0 * nII_fail + nIII_fail

    s_total = s_pass + s_fail

    if s_total <= 0:
        return _safe_nan_dict(
            signal_pass=s_pass,
            signal_fail=s_fail,
            signal_total=s_total,
        )

    eps = s_pass / s_total
    sigma = math.sqrt(
        s_fail**2 * (n_dep_pass + nI_pass + 4.0 * nII_pass + nIII_pass)
        + s_pass**2 * (n_dep_fail + nI_fail + 4.0 * nII_fail + nIII_fail)
    ) / (s_total**2)


    #debug session
    raw_dep_sf = n_dep_pass / n_dep_total
    raw_I_sf = nI_pass / nI_total
    raw_II_sf = nII_pass / nII_total
    raw_III_sf = nIII_pass / nIII_total

    n_bkg_total = nI_total + 2.0 * nII_total - nIII_total
    n_bkg_pass = nI_pass + 2.0 * nII_pass - nIII_pass

    print("raw DEP SF =", raw_dep_sf)
    print("raw I SF   =", raw_I_sf)
    print("raw II SF  =", raw_II_sf)
    print("raw III SF =", raw_III_sf)

    print("bkg total =", n_bkg_total)
    print("bkg pass  =", n_bkg_pass)
    print("signal total - bkg total =", n_dep_total - n_bkg_total)
    print("signal pass -bkg pass =", n_dep_pass - n_bkg_pass)
    print("dep total = ",n_dep_total)
    print("dep pass = ",n_dep_pass)
    print("dep fail = ",n_dep_fail)
    return {
        "eps": eps,
        "sigma": sigma,
        "signal_pass": s_pass,
        "signal_fail": s_fail,
        "signal_total": s_total,
        "n_bkg_pass": n_bkg_pass,
        "n_bkg_fail": n_bkg_fail,
        "n_bkg_total": n_bkg_total,
    }

def _safe_nan_dict(**kwargs):
    out = {"eps": float("nan"), "sigma": float("nan")}
    out.update(kwargs)
    return out

def find_peak_with_fit(
    arr,
    search_range,
    bins=400,
    fit_half_width=6.0,
    window_nsigma=3.0,
    do_plot=False,
    title=None,
):
    """
    Fit a peak with a pure Gaussian.
    Only return fitted mu and sigma.
    No analysis window is defined here.
    """
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]

    counts, edges = np.histogram(arr, bins=bins, range=search_range)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if counts.sum() == 0:
        raise ValueError(f"Empty histogram in search range {search_range}")

    coarse_idx = np.argmax(counts)
    coarse_mu = centers[coarse_idx]

    fit_mask = (centers > coarse_mu - fit_half_width) & (centers < coarse_mu + fit_half_width)
    x = centers[fit_mask]
    y = counts[fit_mask]

    if len(x) < 5 or y.sum() == 0:
        raise ValueError(f"Not enough points to fit peak near {coarse_mu:.2f} keV")

    p0 = [max(float(np.max(y)), 1.0), coarse_mu, 2.0, 0.0]
    bounds = (
        [0.0, coarse_mu - fit_half_width, 0.3, 0.0],
        [np.inf, coarse_mu + fit_half_width, 15.0, np.inf],
    )

    popt, pcov = curve_fit(gauss, x, y, p0=p0, bounds=bounds, maxfev=20000)
    A, mu, sigma, c = popt

    sigma = abs(float(sigma))
    fwhm = 2.355 * sigma
    nsigma = window_nsigma
    left = mu - nsigma * sigma
    right = mu + nsigma * sigma
    result = {
        "peak_center": float(mu),
        "sigma": float(sigma),
        "fwhm": float(fwhm),
        "coarse_center": float(coarse_mu),
        "popt": popt,
        "pcov": pcov,
        "counts": counts,
        "edges": edges,
        "centers": centers,
        "search_range": tuple(search_range),
        "fit_half_width": float(fit_half_width),
        "window_bounds": (float(left), float(right)),
    }
    if do_plot == True:
        xfit = np.linspace(left, right, 1000)
        yfit = gauss(xfit, *popt)

        # 4.5 sigma window


        plt.figure(figsize=(8, 5))
        plt.step(centers, counts, where="mid", label="hist", linewidth=1.8)
        plt.plot(xfit, yfit, linewidth=2, label=f"fit: μ={mu:.2f}, σ={sigma:.2f}")
        plt.axvline(mu, linestyle="--", linewidth=1.4, label=f"peak @ {mu:.2f}")

        # 👉 new lines
        plt.axvline(left, linestyle=":", color="red", label=f"μ-{window_nsigma}σ")
        plt.axvline(right, linestyle=":", color="red", label=f"μ+{window_nsigma}σ")

        plt.xlabel("Energy [keV]")
        plt.ylabel("Counts")
        if title is not None:
            plt.title(title)
        plt.grid(alpha=0.3)
        plt.legend()
        # 👉 replace vertical lines with this
        plt.axvspan(left, right, alpha=0.2, color="red", label=f"±{window_nsigma}σ window")
        plt.tight_layout()
        plt.show()
       
    return result

def get_peak_window_from_sigma(mu, sigma, nsigma=3.0):
    return (float(mu - nsigma * sigma), float(mu + nsigma * sigma))

def count_in_window(E, mask, lo, hi):
    m = (E >= lo) & (E < hi)
    return float(np.count_nonzero(m & mask)), float(np.count_nonzero(m))

def find_cal_peaks_for_run(
    run_db,
    run,
    lh5,
    index=0,
    bins=400,
    is_valid_waveform=False,
    energy_field="trapEftp_cal",
    low_cut_field="A_max_mw_o_E_Low_Cut",
    dep_search_range=(1540, 1610),
    fep_search_range=(2585, 2635),
    dep_fit_half_width=6.0,
    fep_fit_half_width=6.0,
    window_nsigma=3.0,
    do_plot=False,
    plot_low_cut=False,
    logstyle = False
):
    if run not in run_db:
        raise KeyError(f"[ERROR] {run} not found in run_db")

    info = run_db[run]
    channel = get_channel(index=index)

    E_cal = read_array_from_info(
        info, lh5,
        dataset="cal",
        channel=channel,
        table="hit",
        field=energy_field,
        is_valid_waveform=is_valid_waveform,
    )
    E_cal = np.asarray(E_cal)

    low_cut = read_array_from_info(
        info, lh5,
        dataset="cal",
        channel=channel,
        table="hit",
        field=low_cut_field,
        is_valid_waveform=is_valid_waveform,
    )
    pass_mask = np.asarray(low_cut).astype(bool)

    if len(E_cal) != len(pass_mask):
        raise ValueError(
            f"[ERROR] length mismatch: E_cal={len(E_cal)}, pass_mask={len(pass_mask)}"
        )

    dep_fit = find_peak_with_fit(
        arr=E_cal,
        search_range=dep_search_range,
        bins=bins,
        fit_half_width=dep_fit_half_width,
        window_nsigma=window_nsigma,
        do_plot=do_plot,
        title=f"{run} | {channel} | DEP",
    )

    fep_fit = find_peak_with_fit(
        arr=E_cal,
        search_range=fep_search_range,
        bins=bins,
        fit_half_width=fep_fit_half_width,
        window_nsigma=window_nsigma,
        do_plot=do_plot,
        title=f"{run} | {channel} | FEP",
    )

    dep_mu = dep_fit["peak_center"]
    dep_sigma = dep_fit["sigma"]

    w_dep = window_nsigma * dep_sigma   # 👉 唯一宽度

    dep_lo = dep_mu - w_dep
    dep_hi = dep_mu + w_dep

    # I, II, III（等宽平移）
    I_lo   = dep_lo - w_dep
    I_hi   = dep_lo

    II_lo  = dep_hi
    II_hi  = dep_hi + w_dep

    III_lo = II_hi
    III_hi = II_hi + w_dep
    #counts in dep window and sidebands
    n_dep_pass, n_dep_total = count_in_window(E_cal, pass_mask, dep_lo, dep_hi)
    nI_pass, nI_total       = count_in_window(E_cal, pass_mask, I_lo, I_hi)
    nII_pass, nII_total     = count_in_window(E_cal, pass_mask, II_lo, II_hi)
    nIII_pass, nIII_total   = count_in_window(E_cal, pass_mask, III_lo, III_hi)
    
    # dep survival with sideband subtraction
    dep_survival = survival_dep(
        n_dep_pass=n_dep_pass,
        n_dep_total=n_dep_total,
        nI_pass=nI_pass,
        nI_total=nI_total,
        nII_pass=nII_pass,
        nII_total=nII_total,
        nIII_pass=nIII_pass,
        nIII_total=nIII_total,
    )
    dep_survival.update({
        "DEP_window": (dep_lo, dep_hi),
        "I_window": (I_lo, I_hi),
        "II_window": (II_lo, II_hi),
        "III_window": (III_lo, III_hi),
        "n_dep_total": n_dep_total,
        "nI_total": nI_total,
        "nII_total": nII_total,
        "nIII_total": nIII_total,
        "nI_pass": nI_pass,
        "nII_pass": nII_pass,
        "nIII_pass": nIII_pass,
    })
    # FEP survival with simple continuum subtraction using the same sidebands as DEP
    fep_mu = fep_fit["peak_center"]
    fep_sigma = fep_fit["sigma"]

    w_fep = window_nsigma * fep_sigma

    fep_lo = fep_mu - w_fep
    fep_hi = fep_mu + w_fep

    # sidebands（左右对称）
    left_lo  = fep_lo - w_fep
    left_hi  = fep_lo

    right_lo = fep_hi
    right_hi = fep_hi + w_fep
    
    # counts in fep window and sidebands
    n_fep_pass, n_fep_total = count_in_window(E_cal, pass_mask, fep_lo, fep_hi)

    n_left_pass, n_left_total   = count_in_window(E_cal, pass_mask, left_lo, left_hi)
    n_right_pass, n_right_total = count_in_window(E_cal, pass_mask, right_lo, right_hi)

    n_bkg_pass  = n_left_pass + n_right_pass
    n_bkg_total = n_left_total + n_right_total
    
    # fep survival with sideband subtraction
    fep_survival = survival_fep_sep(
        n_peak_pass=n_fep_pass,
        n_peak_total=n_fep_total,
        n_bkg_pass=n_bkg_pass,
        n_bkg_total=n_bkg_total,
    )
    fep_survival.update({
        "signal_window": (fep_lo, fep_hi),
        "left_sideband": (left_lo, left_hi),
        "right_sideband": (right_lo, right_hi),
        "n_left_total": n_left_total,
        "n_right_total": n_right_total,
        "n_left_pass": n_left_pass,
        "n_right_pass": n_right_pass,
    })
    dep_survival.update({
    "DEP_window": (dep_lo, dep_hi),
    "I_window": (I_lo, I_hi),
    "II_window": (II_lo, II_hi),
    "III_window": (III_lo, III_hi),
    "n_dep_total": n_dep_total,
    "n_dep_pass": n_dep_pass,
    "nI_total": nI_total,
    "nII_total": nII_total,
    "nIII_total": nIII_total,
    "nI_pass": nI_pass,
    "nII_pass": nII_pass,
    "nIII_pass": nIII_pass,
    })
    # if plot_low_cut:
    #     for label, fit, surv, search_range in [
    #         ("DEP", dep_fit, dep_survival, [dep_search_range[0], dep_search_range[1] + 40]),
    #         ("FEP", fep_fit, fep_survival, fep_search_range),
    #     ]:
    #         counts_all, edges = np.histogram(E_cal, bins=bins, range=search_range)
    #         counts_pass, _ = np.histogram(E_cal[pass_mask], bins=bins, range=search_range)
    #         centers = 0.5 * (edges[:-1] + edges[1:])

    #         plt.figure(figsize=(8, 5))

    #         plt.step(centers, counts_all, where="mid", label="all", linewidth=1.8)
    #         plt.step(centers, counts_pass, where="mid", label="low-cut accepted", linewidth=1.5)

    #         mu = fit["peak_center"]
    #         plt.axvline(mu, ls="--", label=f"μ={mu:.2f}")

    #         if label == "DEP":
    #             n_signal = surv["n_dep_total"]
    #             n_bkg = surv["n_bkg_total"]

    #             plt.axvspan(
    #                 *surv["DEP_window"],
    #                 alpha=0.18,
    #                 label=f"DEP: signal n={n_signal:.0f}, bkg n={n_bkg:.0f}",
    #                 color="red",
    #          )
    #             plt.axvspan(
    #                 *surv["I_window"],
    #                 alpha=0.10,
    #                 label=f"I: $n_I={surv['nI_total']:.0f}$, $n_I^+={surv['nI_pass']:.0f}$",
    #                 color="blue",
    #             )

    #             plt.axvspan(
    #                 *surv["II_window"],
    #                 alpha=0.10,
    #                 label=f"II: $n_{{II}}={surv['nII_total']:.0f}$, $n_{{II}}^+={surv['nII_pass']:.0f}$",
    #                 color="green",
    #             )

    #             plt.axvspan(
    #                 *surv["III_window"],
    #                 alpha=0.10,
    #                 label=f"III: $n_{{III}}={surv['nIII_total']:.0f}$, $n_{{III}}^+={surv['nIII_pass']:.0f}$",
    #                 color="orange",
    #             )
    #         else:
    #             n_signal = surv["n_peak_total"]
    #             n_signal_pass = surv["n_peak_pass"]
    #             n_bkg = surv["n_bkg_total"]
    #             n_bkg_pass = surv["n_bkg_pass"]

    #             plt.axvspan(
    #                 *surv["signal_window"],
    #                 alpha=0.18,
    #                 label=f"FEP: $n_{{FEP}}={surv['n_peak_total']:.0f}$, $n_{{FEP}}^+={surv['n_peak_pass']:.0f}$",
    #                 color="red",
    #             )

    #             plt.axvspan(
    #                 *surv["left_sideband"],
    #                 alpha=0.10,
    #                 label=f"left: $n_L={surv['n_left_total']:.0f}$, $n_L^+={surv['n_left_pass']:.0f}$",
    #                 color="blue",
    #             )

    #             plt.axvspan(
    #                 *surv["right_sideband"],
    #                 alpha=0.10,
    #                 label=f"right: $n_R={surv['n_right_total']:.0f}$, $n_R^+={surv['n_right_pass']:.0f}$",
    #                 color="blue",
    #             )
    #         plt.xlabel("Energy [keV]")
    #         plt.ylabel("Counts")
    #         # plt.yscale("log")
    #         plt.title(
    #             f"{run} | {channel} | {label} calibration peak\n"
    #             f"low-cut survival_efficiency = {surv['eps']:.3f} ± {surv['sigma']:.3f}"
    #         )
    #         if logstyle == True:
    #             plt.yscale("log")
    #         plt.grid(alpha=0.3)
    #         plt.legend()
    #         plt.tight_layout()
    #         plt.show()
    
    if plot_low_cut:

        def _fill_count_region(
            centers,
            counts_all,
            counts_pass,
            lo,
            hi,
            label,
            color,
            n_total,
            n_pass,
            logstyle=False,
        ):
            """
            Fill the histogram counts that are inside [lo, hi).

            Light fill  = total counts in this region
            Darker fill = accepted/pass counts in this region
            """
            region_mask = (centers >= lo) & (centers < hi)

            # For log scale, filling down to 0 is problematic
            y0 = 0.5 if logstyle else 0.0

            plt.fill_between(
                centers,
                y0,
                counts_all,
                where=region_mask,
                step="mid",
                alpha=0.18,
                color=color,
                label=f"{label} total: n={n_total:.0f}",
            )

            plt.fill_between(
                centers,
                y0,
                counts_pass,
                where=region_mask,
                step="mid",
                alpha=0.45,
                color=color,
                label=f"{label} accepted: $n^+$={n_pass:.0f}",
            )

            # draw exact count boundaries
            plt.axvline(lo, color=color, ls=":", linewidth=1.2)
            plt.axvline(hi, color=color, ls=":", linewidth=1.2)

        for label, fit, surv, search_range in [
            ("DEP", dep_fit, dep_survival, [dep_search_range[0], dep_search_range[1] + 40]),
            ("FEP", fep_fit, fep_survival, fep_search_range),
        ]:
            counts_all, edges = np.histogram(E_cal, bins=bins, range=search_range)
            counts_pass, _ = np.histogram(E_cal[pass_mask], bins=bins, range=search_range)
            centers = 0.5 * (edges[:-1] + edges[1:])

            plt.figure(figsize=(9, 5))

            # Draw full spectra as outlines
            plt.step(
                centers,
                counts_all,
                where="mid",
                label="all spectrum",
                linewidth=1.8,
                color="black",
            )
            plt.step(
                centers,
                counts_pass,
                where="mid",
                label="low-cut accepted spectrum",
                linewidth=1.5,
                color="gray",
            )

            mu = fit["peak_center"]
            plt.axvline(mu, ls="--", color="black", label=f"μ={mu:.2f} keV")

            if label == "DEP":

                _fill_count_region(
                    centers,
                    counts_all,
                    counts_pass,
                    surv["DEP_window"][0],
                    surv["DEP_window"][1],
                    label="DEP signal",
                    color="red",
                    n_total=surv["n_dep_total"],
                    n_pass=surv["n_dep_pass"],
                    logstyle=logstyle,
                )

                _fill_count_region(
                    centers,
                    counts_all,
                    counts_pass,
                    surv["I_window"][0],
                    surv["I_window"][1],
                    label="Region I",
                    color="blue",
                    n_total=surv["nI_total"],
                    n_pass=surv["nI_pass"],
                    logstyle=logstyle,
                )

                _fill_count_region(
                    centers,
                    counts_all,
                    counts_pass,
                    surv["II_window"][0],
                    surv["II_window"][1],
                    label="Region II",
                    color="green",
                    n_total=surv["nII_total"],
                    n_pass=surv["nII_pass"],
                    logstyle=logstyle,
                )

                _fill_count_region(
                    centers,
                    counts_all,
                    counts_pass,
                    surv["III_window"][0],
                    surv["III_window"][1],
                    label="Region III",
                    color="orange",
                    n_total=surv["nIII_total"],
                    n_pass=surv["nIII_pass"],
                    logstyle=logstyle,
                )

                # extra_text = (
                #     f"bkg total = I + 2II - III = {surv['n_bkg_total']:.0f}"
                # )

            else:

                _fill_count_region(
                    centers,
                    counts_all,
                    counts_pass,
                    surv["signal_window"][0],
                    surv["signal_window"][1],
                    label="FEP signal",
                    color="red",
                    n_total=surv["n_peak_total"],
                    n_pass=surv["n_peak_pass"],
                    logstyle=logstyle,
                )

                _fill_count_region(
                    centers,
                    counts_all,
                    counts_pass,
                    surv["left_sideband"][0],
                    surv["left_sideband"][1],
                    label="Left sideband",
                    color="blue",
                    n_total=surv["n_left_total"],
                    n_pass=surv["n_left_pass"],
                    logstyle=logstyle,
                )

                _fill_count_region(
                    centers,
                    counts_all,
                    counts_pass,
                    surv["right_sideband"][0],
                    surv["right_sideband"][1],
                    label="Right sideband",
                    color="green",
                    n_total=surv["n_right_total"],
                    n_pass=surv["n_right_pass"],
                    logstyle=logstyle,
                )

                # extra_text = (
                #     f"bkg total = L + R = {surv['n_bkg_total']:.0f}"
                # )

            plt.xlabel("Energy [keV]")
            plt.ylabel("Counts")

            plt.title(
                f"{run} | {channel} | {label} calibration peak\n"
                f"low-cut survival_efficiency = {surv['eps']:.3f} ± {surv['sigma']:.3f}"
            )

            # plt.text(
            #     0.02,
            #     0.95,
            #     extra_text,
            #     transform=plt.gca().transAxes,
            #     va="top",
            #     ha="left",
            #     fontsize=10,
            #     bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
            # )

            if logstyle:
                plt.yscale("log")
                plt.ylim(bottom=0.5)

            plt.grid(alpha=0.3)
            plt.legend(fontsize=8, ncols=1)
            plt.tight_layout()
            plt.show()
    return {
        "run": run,
        "channel": channel,
        "energy_field": energy_field,
        "low_cut_field": low_cut_field,
        "n_cal_total": int(len(E_cal)),
        "n_cal_pass": int(np.count_nonzero(pass_mask)),
        "window_nsigma": window_nsigma,
        "DEP": {
            "mu": dep_fit["peak_center"],
            "sigma": dep_fit["sigma"],
            "window": (dep_lo, dep_hi),
            "counts": {
                "dep": (n_dep_pass, n_dep_total),
                "I": (nI_pass, nI_total),
                "II": (nII_pass, nII_total),
                "III": (nIII_pass, nIII_total),
            },
            "survival": {
                "eps": dep_survival["eps"],
                "sigma": dep_survival["sigma"],
            },
        },
        "FEP": {
            "mu": fep_fit["peak_center"],
            "sigma": fep_fit["sigma"],
            "window": (fep_lo, fep_hi),
            "counts": {
                "peak": (n_fep_pass, n_fep_total),
                "left": (n_left_pass, n_left_total),
                "right": (n_right_pass, n_right_total),
            },
            "survival": {
                "eps": fep_survival["eps"],
                "sigma": fep_survival["sigma"],
            },
        },
    }

def get_peak_db_for_run(
    run_db,
    run,
    lh5,
    index=0,
    bins=400,
    energy_field="trapEftp_cal",
    low_cut_field="A_max_mw_o_E_Low_Cut",
    is_valid_waveform=False,
    do_plot=False,
    plot_low_cut=False,
    window_nsigma=3.0,
):
    peak_db = find_cal_peaks_for_run(
        run_db=run_db,
        run=run,
        lh5=lh5,
        index=index,
        bins=bins,
        energy_field=energy_field,
        low_cut_field=low_cut_field,
        window_nsigma=window_nsigma,
        is_valid_waveform=is_valid_waveform,
        do_plot=do_plot,
        plot_low_cut=plot_low_cut,
    )
    return peak_db

def build_energy_masks_from_run(
    run_db,
    run,
    lh5,
    channel="ch001",
    energy_field="trapEftp_cal",
    dep_min=1587,
    dep_max=1597,
    fep_min=2610,
    fep_max=2620,
    roi_min=1839,
    roi_max=2239,
    use_fitted_peaks=True,
    peak_db=None,
    index=None,
    do_peak_plot=False,
    is_valid_waveform=False,
    window_nsigma=3.0,
):
    """
    Read energy arrays and build masks once.

    If use_fitted_peaks=True, DEP/FEP windows are obtained from calibration-spectrum
    peak fits for this run. Otherwise the fixed dep_min/dep_max and fep_min/fep_max
    arguments are used.
    """
    if index is None:
        index = 0 if channel == "ch001" else 1

    E_cal = read_array_from_run(
        run_db, run, lh5, dataset="cal", channel=channel, table="hit", field=energy_field,is_valid_waveform=is_valid_waveform
    )
    E_bg = read_array_from_run(
        run_db, run, lh5, dataset="phy_bg", channel=channel, table="hit", field=energy_field,is_valid_waveform=is_valid_waveform
    )
    E_signal = read_array_from_run(
        run_db, run, lh5, dataset="phy_signal", channel=channel, table="hit", field=energy_field,is_valid_waveform=is_valid_waveform
    )

    if use_fitted_peaks:
        if peak_db is None:
            peak_db = get_peak_db_for_run(
                run_db=run_db,
                run=run,
                lh5=lh5,
                index=index,
                energy_field=energy_field,
                do_plot=do_peak_plot,
            )
        dep_mu = peak_db["DEP"]["mu"]
        dep_sigma = peak_db["DEP"]["sigma"]
        fep_mu = peak_db["FEP"]["mu"]
        fep_sigma = peak_db["FEP"]["sigma"]

        dep_min, dep_max = get_peak_window_from_sigma(dep_mu, dep_sigma, nsigma=window_nsigma)
        fep_min, fep_max = get_peak_window_from_sigma(fep_mu, fep_sigma, nsigma=window_nsigma)

    DEP_mask = (E_cal > dep_min) & (E_cal < dep_max)
    FEP_mask = (E_cal > fep_min) & (E_cal < fep_max)
    Energy_mask_bg = (E_bg > roi_min) & (E_bg < roi_max)
    Energy_mask_signal = (E_signal > roi_min) & (E_signal < roi_max)

    return {
        "run": run,
        "channel": channel,
        "energy_field": energy_field,
        "peak_db": peak_db,
        "windows": {
            "dep": (dep_min, dep_max),
            "fep": (fep_min, fep_max),
            "roi": (roi_min, roi_max),
        },
        "masks": {
            "DEP_cal": DEP_mask,
            "FEP_cal": FEP_mask,
            "ROI_bg": Energy_mask_bg,
            "ROI_signal": Energy_mask_signal,
        },
        "energies": {
            "cal": E_cal,
            "phy_bg": E_bg,
            "phy_signal": E_signal,
        },
    }
    
# =========================================================
# 1. generic pickle helpers
# =========================================================

def save_pkl(obj, filename: str):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)
    print(f"saved to {filename}")

def load_pkl(filename: str):
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    print(f"loaded from {filename}")
    return obj


# =========================================================
# 2. field reading helpers
# =========================================================
def get_available_fields(lh5, group, files):
    obj = safe_lh5_read(lh5, group, files)
    ak_obj = obj.view_as("ak")
    fields = list(ak_obj.fields)
    del obj, ak_obj
    gc.collect()
    return fields

def build_aoe_lowcut_event_db_from_masks(
    run_db,
    run,
    lh5,
    mask_db,
    aoe_field: str = "A_max_mw_o_E_Corrected",
    low_cut_field: str = "A_max_mw_o_E_Low_Cut",
    is_valid_waveform=False,
):
    """
    Read A/E and stored low-cut boolean arrays, then apply the already-built
    energy masks from mask_db.
    """
    channel = mask_db["channel"]

    aoe_cal = read_array_from_run(
        run_db, run, lh5, dataset="cal", channel=channel, table="hit", field=aoe_field,is_valid_waveform=is_valid_waveform
    )
    aoe_bg = read_array_from_run(
        run_db, run, lh5, dataset="phy_bg", channel=channel, table="hit", field=aoe_field,is_valid_waveform=is_valid_waveform
    )
    aoe_signal = read_array_from_run(
        run_db, run, lh5, dataset="phy_signal", channel=channel, table="hit", field=aoe_field,is_valid_waveform=is_valid_waveform
    )

    low_cal = read_array_from_run(
        run_db, run, lh5, dataset="cal", channel=channel, table="hit", field=low_cut_field,is_valid_waveform=is_valid_waveform
    )
    low_bg = read_array_from_run(
        run_db, run, lh5, dataset="phy_bg", channel=channel, table="hit", field=low_cut_field,is_valid_waveform=is_valid_waveform
    )
    low_signal = read_array_from_run(
        run_db, run, lh5, dataset="phy_signal", channel=channel, table="hit", field=low_cut_field,is_valid_waveform=is_valid_waveform
    )

    dep_mask = mask_db["masks"]["DEP_cal"]
    fep_mask = mask_db["masks"]["FEP_cal"]
    roi_bg_mask = mask_db["masks"]["ROI_bg"]
    roi_signal_mask = mask_db["masks"]["ROI_signal"]

    out = {
        "run": run,
        "channel": channel,
        "aoe_field": aoe_field,
        "low_cut_field": low_cut_field,
        "selections": {
            "DEP": np.asarray(aoe_cal[dep_mask]),
            "FEP": np.asarray(aoe_cal[fep_mask]),
            "ROI_BG": np.asarray(aoe_bg[roi_bg_mask]),
            "ROI_Signal": np.asarray(aoe_signal[roi_signal_mask]),
        },
        "low_cut_masks": {
            "DEP": np.asarray(low_cal[dep_mask]).astype(bool),
            "FEP": np.asarray(low_cal[fep_mask]).astype(bool),
            "ROI_BG": np.asarray(low_bg[roi_bg_mask]).astype(bool),
            "ROI_Signal": np.asarray(low_signal[roi_signal_mask]).astype(bool),
        },
    }

    del aoe_cal, aoe_bg, aoe_signal
    del low_cal, low_bg, low_signal
    del dep_mask, fep_mask, roi_bg_mask, roi_signal_mask
    gc.collect()

    return out

# =========================================================
# 3. event db builders
# =========================================================

def _clean_event_pair(aoe, low_bool):
    aoe = np.asarray(aoe)
    low_bool = np.asarray(low_bool).astype(bool)

    finite_mask = np.isfinite(aoe)
    aoe = aoe[finite_mask]
    low_bool = low_bool[finite_mask]

    n_total = int(len(aoe))
    n_true = int(np.count_nonzero(low_bool))
    n_false = int(n_total - n_true)

    return {
        "aoe": aoe,
        "low_cut_true": low_bool,
        "n_total": n_total,
        "n_low_cut_true": n_true,
        "n_low_cut_false": n_false,
        "fraction_low_cut_true": (n_true / n_total) if n_total > 0 else np.nan,
        "fraction_low_cut_false": (n_false / n_total) if n_total > 0 else np.nan,
    }

def build_lowcut_event_db_for_run(
    run_db,
    window,
    mw,
    lh5,
    is_valid_waveform=False,
    index: int = 0,
    dep_min: float = 1587,
    dep_max: float = 1597,
    fep_min: float = 2610,
    fep_max: float = 2620,
    roi_min: float = 1839,
    roi_max: float = 2239,
    aoe_field: str = "A_max_mw_o_E_Corrected",
    window_nsigma: float = 3.0,
    low_cut_field: str = "A_max_mw_o_E_Low_Cut",
    energy_field: str = "trapEftp_cal",
    use_fitted_peaks: bool = True,
    alpha: float = 1.0,
    do_peak_plot: bool = False,
):
    """
    Build an event-level db for one run and one channel.

    This stores the selected A/E arrays and the corresponding precomputed
    low-cut boolean arrays. It is meant to be saved as pkl before plotting.
    """
    run = f"s{window}_{mw}"
    if run not in run_db:
        raise KeyError(f"[ERROR] {run} not found")

    channel = get_channel(index=index)
    
    mask_db = build_energy_masks_from_run(
        run_db=run_db,
        run=run,
        lh5=lh5,
        is_valid_waveform=is_valid_waveform,
        channel=channel,
        energy_field=energy_field,
        dep_min=dep_min,
        dep_max=dep_max,
        fep_min=fep_min,
        fep_max=fep_max,
        roi_min=roi_min,
        roi_max=roi_max,
        use_fitted_peaks=use_fitted_peaks,
        peak_db=None,
        index=index,
        do_peak_plot=do_peak_plot,
        window_nsigma=window_nsigma,
    )

    raw_db = build_aoe_lowcut_event_db_from_masks(
        run_db=run_db,
        run=run,
        lh5=lh5,
        mask_db=mask_db,
        aoe_field=aoe_field,
        low_cut_field=low_cut_field,
    )
    
    sig_pass = raw_db["low_cut_masks"]["ROI_Signal"].sum()
    sig_total = raw_db["low_cut_masks"]["ROI_Signal"].size

    bg_pass = raw_db["low_cut_masks"]["ROI_BG"].sum()
    bg_total = raw_db["low_cut_masks"]["ROI_BG"].size

    alpha = alpha # 只有 bg 和 signal 已经同 livetime/exposure 才用 1

    compton_npass = sig_pass - alpha * bg_pass
    compton_total = sig_total - alpha * bg_total

    compton_sf_signal = survival_compton(sig_pass, sig_total)
    compton_sf_bg_subtracted = survival_compton(compton_npass, compton_total)
    roi_bg_sf = survival_compton(
    raw_db["low_cut_masks"]["ROI_BG"].sum(),
    raw_db["low_cut_masks"]["ROI_BG"].size,
    )
    event_db = {
        "meta": {   
            "run": run,
            "channel": channel,
            "aoe_field": aoe_field,
            "low_cut_field": low_cut_field,
            "energy_field": energy_field,
            "windows": mask_db["windows"],
            "peak_db": mask_db["peak_db"],
            "use_fitted_peaks": use_fitted_peaks,
            "window_nsigma": window_nsigma,
            "survival_fep": mask_db["peak_db"]["FEP"]["survival"] if mask_db["peak_db"] is not None else None,
            "survival_dep": mask_db["peak_db"]["DEP"]["survival"] if mask_db["peak_db"] is not None else None,
            "survival_compton_signal": compton_sf_signal,
            "survival_compton_bg_subtracted": compton_sf_bg_subtracted,
        },
        "events": {
            "DEP": _clean_event_pair(raw_db["selections"]["DEP"], raw_db["low_cut_masks"]["DEP"]),
            "FEP": _clean_event_pair(raw_db["selections"]["FEP"], raw_db["low_cut_masks"]["FEP"]),
            "ROI_BG": _clean_event_pair(raw_db["selections"]["ROI_BG"], raw_db["low_cut_masks"]["ROI_BG"]),
            "ROI_Signal": _clean_event_pair(raw_db["selections"]["ROI_Signal"], raw_db["low_cut_masks"]["ROI_Signal"]),
        },
    }

    return event_db

def build_all_lowcut_event_db(
    run_db,
    lh5,
    window_nsigma: float = 3.0,
    indices: Iterable[int] = (0, 1),
    aoe_field: str = "A_max_mw_o_E_Corrected",
    low_cut_field: str = "A_max_mw_o_E_Low_Cut",
    energy_field: str = "trapEftp_cal",
    use_fitted_peaks: bool = True,
    do_peak_plot: bool = False,
    is_valid_waveform: bool = False,
):
    """
    Build event-level low-cut db for all runs and requested channels.
    """
    all_db = {}

    for run in run_db:
        all_db[run] = {}
        parts = run.split("_")
        window = parts[0][1:]      # '100ns'
        mw = parts[1]              # 'mw3'

        for index in indices:
            channel = get_channel(index=index)
            try:
                event_db = build_lowcut_event_db_for_run(
                    run_db=run_db,
                    window=window,
                    mw=mw,
                    lh5=lh5,
                    index=index,
                    aoe_field=aoe_field,
                    low_cut_field=low_cut_field,
                    energy_field=energy_field,
                    use_fitted_peaks=use_fitted_peaks,
                    window_nsigma=window_nsigma,
                    do_peak_plot=do_peak_plot,
                    is_valid_waveform=is_valid_waveform,
                )
                all_db[run][channel] = {
                    "status": "ok",
                    "event_db": event_db,
                }
                print(f"[OK] {run} | {channel}")
            except Exception as e:
                all_db[run][channel] = {
                    "status": "fail",
                    "error": str(e),
                }
                print(f"[FAIL] {run} | {channel}: {e}")

    return all_db

# =========================================================
# 4. hist builders from stored event db
# =========================================================

def make_hist_with_boolean(arr, bool_mask, bins: int = 400, value_range: Tuple[float, float] = (0.75, 1.25)):
    arr = np.asarray(arr)
    bool_mask = np.asarray(bool_mask).astype(bool)

    finite_mask = np.isfinite(arr)
    arr = arr[finite_mask]
    bool_mask = bool_mask[finite_mask]

    counts, edges = np.histogram(arr, bins=bins, range=value_range)
    counts_true, _ = np.histogram(arr[bool_mask], bins=bins, range=value_range)
    centers = 0.5 * (edges[:-1] + edges[1:])

    n_total = int(len(arr))
    n_true = int(np.count_nonzero(bool_mask))

    return {
        "counts": counts,
        "counts_true": counts_true,
        "edges": edges,
        "centers": centers,
        "n_total": n_total,
        "n_true": n_true,
        "fraction_true": (n_true / n_total) if n_total > 0 else np.nan,
    }

def make_hist_db_from_event_db(
    event_db,
    bins: int = 200,
    value_range: Tuple[float, float] = (0.75, 1.25),
):
    hist_db = {
        "meta": {
            **event_db["meta"],
            "bins": bins,
            "value_range": value_range,
            "survival_fep": event_db["meta"].get("survival_fep", None),
            "survival_dep": event_db["meta"].get("survival_dep", None),
            "survival_compton_signal": event_db["meta"].get("survival_compton_signal", None),
            "survival_compton_bg_subtracted": event_db["meta"].get("survival_compton_bg_subtracted", None),
        }
    }

    for key in ["DEP", "FEP", "ROI_BG", "ROI_Signal"]:
        hist_db[key] = make_hist_with_boolean(
            event_db["events"][key]["aoe"],
            event_db["events"][key]["low_cut_true"],
            bins=bins,
            value_range=value_range,
        )

    return hist_db

# =========================================================
# 5. summary helpers
# =========================================================

def summarize_lowcut_event_db(event_db):
    out = {
        "meta": event_db["meta"],
        "summary": {},
    }
    for key in ["DEP", "FEP", "ROI_BG", "ROI_Signal"]:
        item = event_db["events"][key]
        out["summary"][key] = {
            "n_total": item["n_total"],
            "n_low_cut_true": item["n_low_cut_true"],
            "n_low_cut_false": item["n_low_cut_false"],
            "fraction_low_cut_true": item["fraction_low_cut_true"],
        }
    return out

def summarize_all_lowcut_event_db(all_db):
    summary = {}
    for run, run_dict in all_db.items():
        summary[run] = {}
        for channel, res in run_dict.items():
            if res["status"] != "ok":
                summary[run][channel] = {
                    "status": "fail",
                    "error": res.get("error", "unknown error"),
                }
                continue

            event_db = res["event_db"]
            row = {"status": "ok"}
            for key in ["DEP", "FEP", "ROI_BG", "ROI_Signal"]:
                item = event_db["events"][key]
                row[f"{key}_n_total"] = item["n_total"]
                row[f"{key}_n_true"] = item["n_low_cut_true"]
                row[f"{key}_fraction_true"] = item["fraction_low_cut_true"]
            summary[run][channel] = row
    return summary

# =========================================================
# 6. plotting from stored event db
# =========================================================

def plot_lowcut_hist_from_event_db(
    event_db,
    bins: int = 200,
    value_range: Tuple[float, float] = (0.75, 1.25),
    ylim_top: float = 1e6,
    show_labels=("DEP", "FEP", "ROI_BG", "ROI_Signal"),
    save_fig: Optional[str] = None,
    show_plot: bool = True,
):
    hist_db = make_hist_db_from_event_db(
        event_db,
        bins=bins,
        value_range=value_range,
    )

    label_map = {
        "DEP": "DEP",
        "FEP": "FEP",
        "ROI_BG": "ROI BG",
        "ROI_Signal": "ROI Signal",
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    # plt.title(f"A/E low-cut accepted Events| {event_db['meta']['run']} | {event_db['meta']['channel']}")
    plt.title(f"A/E low-cut accepted Events| sp01 | r059 | IC | {event_db['meta']['run']}")
    for key in show_labels:
        centers = hist_db[key]["centers"]
        total = hist_db[key]["counts"]
        passed = hist_db[key]["counts_true"]

        plt.step(
            centers,
            total,
            where="mid",
            label=f"{label_map.get(key, key)} total"
        )
        ax_inset = ax.inset_axes([0.23,0.1,0.3,0.3])
        ax_inset.set_yscale("log")
        ax_inset.plot([], [])
        ax_inset.step(
            centers,
            total,
            where="mid",
            label=f"{label_map.get(key, key)} total"
        )

        ax_inset.set_xlim(-10, 10)
        ax_inset.set_ylim(0.5e1,3e3)
        ax_inset.set_title("Zoomed Inset")
        ax_inset.set_xlabel(event_db["meta"]["aoe_field"])
        ax_inset.set_ylabel("Counts")
        ax_inset.tick_params(labelsize=8)
        
        # plt.fill_between(
        #     centers,
        #     np.maximum(passed, 1e-12),
        #     step="mid",
        #     alpha=0.30,
        #     label=(
        #     f"{label_map.get(key, key)} accepted events"
        #     + (
        #         f", survival = {event_db['meta'][{'DEP': 'survival_dep', 'FEP': 'survival_fep', 'ROI_Signal': 'survival_compton_signal'}[key]]['eps']:.3%}"
        #         if key in {"DEP", "FEP", "ROI_Signal"}
        #         else ""
        #         )
        #     )
        #  ),

    plt.xlabel(event_db["meta"]["aoe_field"])
    plt.ylabel("Counts")
    plt.yscale("log")
    plt.ylim(1, ylim_top)
    plt.grid(alpha=0.3)
    # plt.legend(fontsize=9)
    plt.tight_layout()
    if save_fig is not None:
        plt.savefig(save_fig)
        print(f"Figure saved to {save_fig}")
    if show_plot:
        plt.show()
    else:        plt.close()
    return hist_db

def plot_lowcut_hist_from_pkl(
    filename: str,
    bins: int = 200,
    value_range: Tuple[float, float] = (0.75, 1.25),
    ylim_top: float = 1e6,
    aoe_field: str = "A_max_mw_o_E_Corrected",
    energy_field: str = "trapEftp_cal",
    low_cut_field: str = "A_max_mw_o_E_Low_Cut",
    show_labels=("DEP", "FEP", "ROI_BG", "ROI_Signal"),
):
    event_db = load_pkl(filename)
    return plot_lowcut_hist_from_event_db(
        event_db=event_db,
        aoe_field=aoe_field,
        energy_field=energy_field,
        low_cut_field=low_cut_field,
        bins=bins,
        value_range=value_range,
        ylim_top=ylim_top,
        show_labels=show_labels,
    )