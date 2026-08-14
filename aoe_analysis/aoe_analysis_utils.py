
import os
import re
import gc
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import awkward as ak
from scipy.optimize import curve_fit


# =========================================================
# 1. run_db
# =========================================================
def build_run_db(
    runs,
    filepath,
    pnum_cal="00",
    run_cal="037",
    pnum_phy_bg="00",
    run_phy_bg="038",
    pnum_phy_signal="05",
    run_phy_signal="059",
):
    """
    Build a run database from folder structure.

    Expected run name format:
        s100ns_mw3
        s200ns_mw7
        ...

    Returns
    -------
    db : dict
        Nested metadata for each run.
    """
    db = {}
    missing = []

    for run in runs:
        m = re.match(r"s(\d+)ns_mw(\d+)", run)
        if not m:
            missing.append(f"{run}: invalid run name format")
            continue

        run_path = os.path.join(filepath, run)

        phy_bg_path = os.path.join(run_path, "phy", f"p{pnum_phy_bg}", f"r{run_phy_bg}")
        phy_signal_path = os.path.join(run_path, "phy", f"p{pnum_phy_signal}", f"r{run_phy_signal}")
        cal_path = os.path.join(run_path, "cal", f"p{pnum_cal}", f"r{run_cal}")

        missing_this_run = []
        for label, path in [
            ("run folder", run_path),
            ("phy_bg path", phy_bg_path),
            ("phy_signal path", phy_signal_path),
            ("cal path", cal_path),
        ]:
            if not os.path.exists(path):
                missing_this_run.append(f"{label} missing: {path}")

        if missing_this_run:
            missing.append(f"{run}\n  " + "\n  ".join(missing_this_run))
            continue

        db[run] = {
            "run": run,
            "time_ns": int(m.group(1)),
            "mw": int(m.group(2)),
            "interval": f"{m.group(1)}ns",
            "mw_label": f"mw{m.group(2)}",
            "cal_path": cal_path,
            "phy_bg_path": phy_bg_path,
            "phy_signal_path": phy_signal_path,
            "cal_pathlist": sorted(os.path.join(cal_path, f) for f in os.listdir(cal_path)),
            "phy_bg_pathlist": sorted(os.path.join(phy_bg_path, f) for f in os.listdir(phy_bg_path)),
            "phy_signal_pathlist": sorted(os.path.join(phy_signal_path, f) for f in os.listdir(phy_signal_path)),
        }

    if missing:
        print("[WARN] Some runs were skipped:")
        for msg in missing:
            print(msg)

    return db


# =========================================================
# 2. basic helpers
# =========================================================
def safe_lh5_read(lh5, key, files):
    """
    lh5.read sometimes returns (obj, n_rows), sometimes obj.
    This wrapper always returns the object itself.
    """
    out = lh5.read(key, files)
    return out[0] if isinstance(out, tuple) else out


def get_pathlist(info, dataset):
    mapping = {
        "cal": info["cal_pathlist"],
        "phy_bg": info["phy_bg_pathlist"],
        "phy_signal": info["phy_signal_pathlist"],
    }
    if dataset not in mapping:
        raise ValueError(f"[ERROR] unknown dataset: {dataset}")
    return mapping[dataset]


def get_channel(index=0):
    return "ch001" if index == 0 else "ch002"


def get_group_and_pathlist(info, key):
    mapping = {
        "cal_ch001": ("ch001/hit", info["cal_pathlist"]),
        "cal_ch002": ("ch002/hit", info["cal_pathlist"]),
        "phy_bg_ch001": ("ch001/hit", info["phy_bg_pathlist"]),
        "phy_bg_ch002": ("ch002/hit", info["phy_bg_pathlist"]),
        "phy_signal_ch001": ("ch001/hit", info["phy_signal_pathlist"]),
        "phy_signal_ch002": ("ch002/hit", info["phy_signal_pathlist"]),
    }
    return mapping.get(key, (None, None))


# =========================================================
# 3. field inspection / reading
# =========================================================
def get_available_fields(lh5, group, files):
    obj = safe_lh5_read(lh5, group, files)
    ak_obj = obj.view_as("ak")
    fields = list(ak_obj.fields)
    del obj, ak_obj
    gc.collect()
    return fields


def read_array_from_info(
    info,
    lh5,
    dataset="cal",
    channel="ch001",
    table="hit",
    field="trapEftp_cal",
    copy_array=True,
):
    """
    Read a whole field array from one dataset of one run.
    """
    pathlist = get_pathlist(info, dataset)
    if len(pathlist) == 0:
        raise ValueError(f"[ERROR] {dataset}_pathlist is empty for run {info['run']}")

    group = f"{channel}/{table}"
    obj = safe_lh5_read(lh5, group, pathlist)
    ak_obj = obj.view_as("ak")

    if field not in ak_obj.fields:
        fields = list(ak_obj.fields)
        del obj, ak_obj
        gc.collect()
        raise KeyError(
            f"[ERROR] field '{field}' not found in {group} for run {info['run']}. "
            f"Available fields: {fields}"
        )

    arr = ak.to_numpy(ak_obj[field])
    if copy_array:
        arr = arr.copy()

    del obj, ak_obj
    gc.collect()
    return arr


def read_array_from_run(
    run_db,
    run,
    lh5,
    dataset="cal",
    channel="ch001",
    table="hit",
    field="trapEftp_cal",
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
        copy_array=copy_array,
    )


def get_field_array_from_file(lh5, group, file, field, copy_array=True):
    """
    Read one field from one file only.
    Useful for low-memory loops over many files.
    """
    obj = safe_lh5_read(lh5, group, [file])
    if obj is None:
        return None

    ak_obj = obj.view_as("ak")
    if field not in ak_obj.fields:
        fields = list(ak_obj.fields)
        del obj, ak_obj
        gc.collect()
        print(f"[ERROR] field '{field}' not found in {file}")
        print("available fields:", fields)
        return None

    arr = ak.to_numpy(ak_obj[field])
    if copy_array:
        arr = arr.copy()

    del obj, ak_obj
    gc.collect()
    return arr




# =========================================================
# 4. peak finding on calibration spectrum
# =========================================================
def gauss_lin(x, A, mu, sigma, m, c):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + m * x + c


def find_peak_with_fit(
    arr,
    search_range,
    bins=400,
    fit_half_width=7.0,
    window_scale=1.5,
    do_plot=False,
    title=None,
):
    """
    Find a peak by:
    1) coarse histogram search
    2) local Gaussian + linear background fit
    3) define ROI window from fitted FWHM
    """
    import matplotlib.pyplot as plt

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

    p0 = [max(float(np.max(y) - np.min(y)), 1.0), coarse_mu, 2.0, 0.0, float(np.min(y))]
    bounds = (
        [0.0, coarse_mu - fit_half_width, 0.3, -np.inf, -np.inf],
        [np.inf, coarse_mu + fit_half_width, 15.0, np.inf, np.inf],
    )

    popt, pcov = curve_fit(gauss_lin, x, y, p0=p0, bounds=bounds, maxfev=20000)
    A, mu, sigma, m, c = popt
    sigma = abs(float(sigma))
    fwhm = 2.355 * sigma
    peak_min = mu - window_scale * fwhm
    peak_max = mu + window_scale * fwhm

    result = {
        "peak_center": float(mu),
        "sigma": sigma,
        "fwhm": float(fwhm),
        "peak_min": float(peak_min),
        "peak_max": float(peak_max),
        "coarse_center": float(coarse_mu),
        "popt": popt,
        "pcov": pcov,
        "counts": counts,
        "edges": edges,
        "centers": centers,
        "search_range": tuple(search_range),
        "fit_half_width": float(fit_half_width),
    }

    if do_plot:
        xx = np.linspace(x.min(), x.max(), 1000)
        plt.figure(figsize=(7, 4))
        plt.step(centers, counts, where="mid", label="hist")
        plt.plot(xx, gauss_lin(xx, *popt), label=f"fit: mu={mu:.3f}, FWHM={fwhm:.3f}")
        plt.axvline(peak_min, ls="--", label=f"window min = {peak_min:.2f}")
        plt.axvline(peak_max, ls="--", label=f"window max = {peak_max:.2f}")
        plt.xlabel("Energy (keV)")
        plt.ylabel("Counts")
        if title:
            plt.title(title)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return result
def get_bg_subtracted_peak_counts(
    E,
    peak_center,
    signal_half_width=4.0,
    sideband_width=8.0,
    sideband_gap=2.0,
):
    """
    Use symmetric sidebands to estimate background under a peak.

    Parameters
    ----------
    E : array-like
        Energy array
    peak_center : float
        Peak center in keV
    signal_half_width : float
        Half-width of signal window
    sideband_width : float
        Width of each sideband
    sideband_gap : float
        Gap between signal window and each sideband
    """
    import numpy as np

    E = np.asarray(E)
    E = E[np.isfinite(E)]

    sig_lo = peak_center - signal_half_width
    sig_hi = peak_center + signal_half_width

    left_lo = sig_lo - sideband_gap - sideband_width
    left_hi = sig_lo - sideband_gap
    right_lo = sig_hi + sideband_gap
    right_hi = sig_hi + sideband_gap + sideband_width

    sig_mask = (E >= sig_lo) & (E <= sig_hi)
    left_mask = (E >= left_lo) & (E <= left_hi)
    right_mask = (E >= right_lo) & (E <= right_hi)

    n_signal_raw = int(np.count_nonzero(sig_mask))
    n_left_raw = int(np.count_nonzero(left_mask))
    n_right_raw = int(np.count_nonzero(right_mask))
    n_side_raw = n_left_raw + n_right_raw

    signal_width = sig_hi - sig_lo
    side_width = (left_hi - left_lo) + (right_hi - right_lo)

    if side_width <= 0:
        raise ValueError("Invalid sideband widths")

    bg_density = n_side_raw / side_width
    bg_expected_in_signal = bg_density * signal_width
    n_signal_bg_sub = n_signal_raw - bg_expected_in_signal

    return {
        "peak_center": float(peak_center),
        "signal_window": (float(sig_lo), float(sig_hi)),
        "left_sideband": (float(left_lo), float(left_hi)),
        "right_sideband": (float(right_lo), float(right_hi)),
        "n_signal_raw": n_signal_raw,
        "n_left_raw": n_left_raw,
        "n_right_raw": n_right_raw,
        "n_side_raw": n_side_raw,
        "bg_density_per_keV": float(bg_density),
        "bg_expected_in_signal": float(bg_expected_in_signal),
        "n_signal_bg_sub": float(n_signal_bg_sub),
    }


def get_bg_subtracted_survival_fraction(
    E,
    pass_mask,
    peak_center,
    signal_half_width=4.0,
    sideband_width=8.0,
    sideband_gap=2.0,
):
    """
    Compute raw and background-subtracted survival fraction for one peak.
    """
    import numpy as np

    E = np.asarray(E)
    pass_mask = np.asarray(pass_mask).astype(bool)

    if len(E) != len(pass_mask):
        raise ValueError(
            f"Length mismatch: len(E)={len(E)}, len(pass_mask)={len(pass_mask)}"
        )

    all_db = get_bg_subtracted_peak_counts(
        E=E,
        peak_center=peak_center,
        signal_half_width=signal_half_width,
        sideband_width=sideband_width,
        sideband_gap=sideband_gap,
    )

    pass_db = get_bg_subtracted_peak_counts(
        E=E[pass_mask],
        peak_center=peak_center,
        signal_half_width=signal_half_width,
        sideband_width=sideband_width,
        sideband_gap=sideband_gap,
    )

    fraction_raw = (
        pass_db["n_signal_raw"] / all_db["n_signal_raw"]
        if all_db["n_signal_raw"] > 0 else np.nan
    )

    fraction_bg_sub = (
        pass_db["n_signal_bg_sub"] / all_db["n_signal_bg_sub"]
        if all_db["n_signal_bg_sub"] > 0 else np.nan
    )

    return {
        "peak_center": float(peak_center),
        "all": all_db,
        "pass": pass_db,
        "fraction_raw": float(fraction_raw) if np.isfinite(fraction_raw) else np.nan,
        "fraction_bg_sub": float(fraction_bg_sub) if np.isfinite(fraction_bg_sub) else np.nan,
    }
def find_cal_peaks_for_run(
    run_db,
    run,
    lh5,
    index=0,
    bins=400,
    energy_field="trapEftp_cal",
    low_cut_field="A_max_mw_o_E_Low_Cut",
    dep_search_range=(1540, 1645),
    fep_search_range=(2585, 2635),
    dep_fit_half_width=7.0,
    fep_fit_half_width=4.0,
    dep_window_scale=1.5,
    fep_window_scale=1.5,
    dep_signal_half_width=4.0,
    fep_signal_half_width=4.0,
    sideband_width=8.0,
    sideband_gap=2.0,
    invert_low_cut=False,
    do_plot=False,
    plot_low_cut=False,
):
    if run not in run_db:
        raise KeyError(f"[ERROR] {run} not found in run_db")

    info = run_db[run]
    channel = get_channel(index=index)

    E_cal = read_array_from_info(
        info,
        lh5,
        dataset="cal",
        channel=channel,
        table="hit",
        field=energy_field,
    )
    E_cal = np.asarray(E_cal)

    low_cut = read_array_from_info(
        info,
        lh5,
        dataset="cal",
        channel=channel,
        table="hit",
        field=low_cut_field,
    )
    pass_mask = np.asarray(low_cut).astype(bool)

    if invert_low_cut:
        pass_mask = ~pass_mask

    if len(E_cal) != len(pass_mask):
        raise ValueError(
            f"[ERROR] length mismatch: E_cal={len(E_cal)}, pass_mask={len(pass_mask)}"
        )

    dep_fit = find_peak_with_fit(
        arr=E_cal,
        search_range=dep_search_range,
        bins=bins,
        fit_half_width=dep_fit_half_width,
        window_scale=dep_window_scale,
        do_plot=do_plot,
        title=f"{run} | {channel} | DEP",
    )

    fep_fit = find_peak_with_fit(
        arr=E_cal,
        search_range=fep_search_range,
        bins=bins,
        fit_half_width=fep_fit_half_width,
        window_scale=fep_window_scale,
        do_plot=do_plot,
        title=f"{run} | {channel} | FEP",
    )

    dep_survival = get_bg_subtracted_survival_fraction(
        E=E_cal,
        pass_mask=pass_mask,
        peak_center=dep_fit["peak_center"],
        signal_half_width=dep_signal_half_width,
        sideband_width=sideband_width,
        sideband_gap=sideband_gap,
    )

    fep_survival = get_bg_subtracted_survival_fraction(
        E=E_cal,
        pass_mask=pass_mask,
        peak_center=fep_fit["peak_center"],
        signal_half_width=fep_signal_half_width,
        sideband_width=sideband_width,
        sideband_gap=sideband_gap,
    )

    return {
        "run": run,
        "channel": channel,
        "energy_field": energy_field,
        "low_cut_field": low_cut_field,
        "invert_low_cut": bool(invert_low_cut),
        "n_cal_total": int(len(E_cal)),
        "DEP": {
            "fit": dep_fit,
            "survival": dep_survival,
        },
        "FEP": {
            "fit": fep_fit,
            "survival": fep_survival,
        },
    }
_peak_cache = {}

def get_peak_db_for_run(
    run_db,
    run,
    lh5,
    index=0,
    bins=400,
    energy_field="trapEftp_cal",
    low_cut_field="A_max_mw_o_E_Low_Cut",
    invert_low_cut=False,
    do_plot=False,
    use_cache=True,
    plot_low_cut=False,
):
    cache_key = (run, index, bins, energy_field, low_cut_field, invert_low_cut)

    if use_cache and cache_key in _peak_cache:
        return _peak_cache[cache_key]

    peak_db = find_cal_peaks_for_run(
        run_db=run_db,
        run=run,
        lh5=lh5,
        index=index,
        bins=bins,
        energy_field=energy_field,
        low_cut_field=low_cut_field,
        invert_low_cut=invert_low_cut,
        do_plot=do_plot,
        plot_low_cut=plot_low_cut,
    )

    if use_cache:
        _peak_cache[cache_key] = peak_db

    return peak_db

def plot_peak_fit_result(result, title=None):
    import matplotlib.pyplot as plt

    centers = result["centers"]
    counts = result["counts"]
    popt = result["popt"]
    x = np.linspace(centers.min(), centers.max(), 1000)

    plt.figure(figsize=(7, 4))
    plt.step(centers, counts, where="mid", label="hist")
    plt.plot(x, gauss_lin(x, *popt), label="gauss+lin fit")
    plt.axvline(result["peak_center"], ls="--", label=f"mu={result['peak_center']:.2f}")
    plt.axvline(result["peak_min"], ls=":")
    plt.axvline(result["peak_max"], ls=":")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Counts")
    if title:
        plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cal_spectrum_with_peaks(
    run_db,
    run,
    lh5,
    peak_db,
    index=0,
    energy_field="trapEftp_cal",
    bins=500,
    xlim=(1450, 2700),
    figsize=(9, 4),
):
    import matplotlib.pyplot as plt

    info = run_db[run]
    channel = get_channel(index=index)
    E_cal = read_array_from_info(
        info,
        lh5,
        dataset="cal",
        channel=channel,
        table="hit",
        field=energy_field,
    )

    dep = peak_db["DEP"]
    fep = peak_db["FEP"]

    plt.figure(figsize=figsize)
    plt.hist(E_cal, bins=bins, range=xlim, histtype="step", label="cal spectrum")
    plt.axvline(dep["peak_center"], ls="--", label=f"DEP center = {dep['peak_center']:.2f}")
    plt.axvline(dep["peak_min"], ls=":", label="DEP window")
    plt.axvline(dep["peak_max"], ls=":")
    plt.axvline(fep["peak_center"], ls="--", label=f"FEP center = {fep['peak_center']:.2f}")
    plt.axvline(fep["peak_min"], ls=":", label="FEP window")
    plt.axvline(fep["peak_max"], ls=":")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Counts")
    plt.title(f"{run} | {channel}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_dep_fep_masks(E_cal, peak_db):
    dep = peak_db["DEP"]
    fep = peak_db["FEP"]
    dep_mask = (E_cal > dep["peak_min"]) & (E_cal < dep["peak_max"])
    fep_mask = (E_cal > fep["peak_min"]) & (E_cal < fep["peak_max"])
    return dep_mask, fep_mask

# =========================================================
# 4. plotting / mask building
# =========================================================
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
        run_db, run, lh5, dataset="cal", channel=channel, table="hit", field=energy_field
    )
    E_bg = read_array_from_run(
        run_db, run, lh5, dataset="phy_bg", channel=channel, table="hit", field=energy_field
    )
    E_signal = read_array_from_run(
        run_db, run, lh5, dataset="phy_signal", channel=channel, table="hit", field=energy_field
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
                use_cache=True,
            )
        dep_min = peak_db["DEP"]["peak_min"]
        dep_max = peak_db["DEP"]["peak_max"]
        fep_min = peak_db["FEP"]["peak_min"]
        fep_max = peak_db["FEP"]["peak_max"]

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


def build_aoe_selections_from_masks(
    run_db,
    run,
    lh5,
    mask_db,
    aoe_field="A_max_mw_o_E_Corrected",
):
    """
    Use existing energy masks to slice A/E arrays.
    """
    channel = mask_db["channel"]

    aoe_cal = read_array_from_run(
        run_db, run, lh5, dataset="cal", channel=channel, table="hit", field=aoe_field
    )
    aoe_bg = read_array_from_run(
        run_db, run, lh5, dataset="phy_bg", channel=channel, table="hit", field=aoe_field
    )
    aoe_signal = read_array_from_run(
        run_db, run, lh5, dataset="phy_signal", channel=channel, table="hit", field=aoe_field
    )

    DEP_mask = mask_db["masks"]["DEP_cal"]
    FEP_mask = mask_db["masks"]["FEP_cal"]
    ROI_bg_mask = mask_db["masks"]["ROI_bg"]
    ROI_signal_mask = mask_db["masks"]["ROI_signal"]

    return {
        "run": run,
        "channel": channel,
        "aoe_field": aoe_field,
        "selections": {
            "DEP": aoe_cal[DEP_mask],
            "FEP": aoe_cal[FEP_mask],
            "ROI_bg": aoe_bg[ROI_bg_mask],
            "ROI_signal": aoe_signal[ROI_signal_mask],
        },
    }


def make_hist(arr, bins=400, value_range=(0.75, 1.25)):
    counts, edges = np.histogram(arr, bins=bins, range=value_range)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "counts": counts,
        "edges": edges,
        "centers": centers,
        "n_total": len(arr),
    }


def plot_aoe_for_run_lowmem(
    run_db,
    window,
    mw,
    lh5,
    index=0,
    dep_min=1587,
    dep_max=1597,
    fep_min=2610,
    fep_max=2620,
    roi_min=1839,
    roi_max=2239,
    binnum=200,
    aoe_field="A_max_mw_o_E_Corrected",
    energy_field="trapEftp_cal",
    ylim_top=1e6,
    return_hist=True,
    use_fitted_peaks=True,
    do_peak_plot=False,
):
    """
    Low-memory version:
    - for each A/E plot, first find DEP/FEP on calibration data
    - then build dynamic energy masks
    - then read AoE and histogram the selected samples
    """
    import matplotlib.pyplot as plt

    run = f"s{window}_{mw}"
    if run not in run_db:
        raise KeyError(f"[ERROR] {run} not found")

    info = run_db[run]
    channel = get_channel(index=index)

    peak_db = None
    if use_fitted_peaks:
        peak_db = get_peak_db_for_run(
            run_db=run_db,
            run=run,
            lh5=lh5,
            index=index,
            energy_field=energy_field,
            do_plot=do_peak_plot,
            use_cache=True,
        )
        dep_min = peak_db["DEP"]["peak_min"]
        dep_max = peak_db["DEP"]["peak_max"]
        fep_min = peak_db["FEP"]["peak_min"]
        fep_max = peak_db["FEP"]["peak_max"]

    E_cal = read_array_from_info(info, lh5, dataset="cal", channel=channel, table="hit", field=energy_field)
    E_bg = read_array_from_info(info, lh5, dataset="phy_bg", channel=channel, table="hit", field=energy_field)
    E_signal = read_array_from_info(info, lh5, dataset="phy_signal", channel=channel, table="hit", field=energy_field)

    DEP_mask = (E_cal > dep_min) & (E_cal < dep_max)
    FEP_mask = (E_cal > fep_min) & (E_cal < fep_max)
    Energy_mask_bg = (E_bg > roi_min) & (E_bg < roi_max)
    Energy_mask_signal = (E_signal > roi_min) & (E_signal < roi_max)

    aoe_cal = read_array_from_info(info, lh5, dataset="cal", channel=channel, table="hit", field=aoe_field)
    aoe_bg = read_array_from_info(info, lh5, dataset="phy_bg", channel=channel, table="hit", field=aoe_field)
    aoe_signal = read_array_from_info(info, lh5, dataset="phy_signal", channel=channel, table="hit", field=aoe_field)

    aoe_dep = aoe_cal[DEP_mask]
    aoe_FEP = aoe_cal[FEP_mask]
    aoe_roi_bg = aoe_bg[Energy_mask_bg]
    aoe_roi_signal = aoe_signal[Energy_mask_signal]

    hist_db = {
        "DEP": make_hist(aoe_dep, bins=binnum, value_range=(0.75, 1.25)),
        "FEP": make_hist(aoe_FEP, bins=binnum, value_range=(0.75, 1.25)),
        "ROI_BG": make_hist(aoe_roi_bg, bins=binnum, value_range=(0.75, 1.25)),
        "ROI_Signal": make_hist(aoe_roi_signal, bins=binnum, value_range=(0.75, 1.25)),
        "meta": {
            "run": run,
            "channel": channel,
            "peak_db": peak_db,
            "windows": {
                "dep": (dep_min, dep_max),
                "fep": (fep_min, fep_max),
                "roi": (roi_min, roi_max),
            },
        },
    }

    plt.figure(figsize=(8, 6))
    plt.title(f"A/E {run} {channel}")

    plt.step(hist_db["DEP"]["centers"], hist_db["DEP"]["counts"], where="mid", label="DEP")
    plt.step(hist_db["FEP"]["centers"], hist_db["FEP"]["counts"], where="mid", label="FEP")
    plt.step(hist_db["ROI_BG"]["centers"], hist_db["ROI_BG"]["counts"], where="mid", label="ROI BG")
    plt.step(hist_db["ROI_Signal"]["centers"], hist_db["ROI_Signal"]["counts"], where="mid", label="ROI Signal")

    plt.xlabel(aoe_field)
    plt.ylabel("Counts")
    plt.yscale("log")
    plt.ylim(1, ylim_top)
    plt.grid(alpha=0.3)
    plt.legend()

    cal_path = info["cal_path"]
    phy_bg_path = info["phy_bg_path"]
    phy_signal_path = info["phy_signal_path"]

    cal_tag = f"{os.path.basename(os.path.dirname(cal_path))}/{os.path.basename(cal_path)}"
    phy_bg_tag = f"{os.path.basename(os.path.dirname(phy_bg_path))}/{os.path.basename(phy_bg_path)}"
    phy_signal_tag = f"{os.path.basename(os.path.dirname(phy_signal_path))}/{os.path.basename(phy_signal_path)}"

    if use_fitted_peaks and peak_db is not None:
        dep_info = peak_db["DEP"]
        fep_info = peak_db["FEP"]

        info_text = (
            f"cal: {cal_tag}\n"
            f"phy bg: {phy_bg_tag}\n"
            f"phy signal: {phy_signal_tag}\n"
            f"DEP peak: {dep_info['peak_center']:.2f} keV\n"
            f"DEP mask: [{dep_info['peak_min']:.2f}, {dep_info['peak_max']:.2f}]\n"
            f"FEP peak: {fep_info['peak_center']:.2f} keV\n"
            f"FEP mask: [{fep_info['peak_min']:.2f}, {fep_info['peak_max']:.2f}]\n"
            f"ROI: [{roi_min:.0f}, {roi_max:.0f}]"
        )
    else:
        info_text = (
            f"cal: {cal_tag}\n"
            f"phy bg: {phy_bg_tag}\n"
            f"phy signal: {phy_signal_tag}\n"
            f"DEP mask: [{dep_min:.2f}, {dep_max:.2f}] (default)\n"
            f"FEP mask: [{fep_min:.2f}, {fep_max:.2f}] (default)\n"
            f"ROI: [{roi_min:.0f}, {roi_max:.0f}]"
        )

    plt.gca().text(
        0.02, 0.98,
        info_text,
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75)
    )

    plt.tight_layout()
    plt.show()

    del E_cal, E_bg, E_signal
    del aoe_cal, aoe_bg, aoe_signal
    del DEP_mask, FEP_mask, Energy_mask_bg, Energy_mask_signal
    gc.collect()

    if return_hist:
        return hist_db
    return None



# =========================================================
# 5. A/E cuts
# =========================================================
def build_cut_db_fixed(
    all_aoe_results,
    low_cut=0.9,
    high_cut=1.1,
):
    """
    Build a cut database with fixed low/high cuts.

    Returns
    -------
    cut_db[run][channel] = {
        "low_cut": ...,
        "high_cut": ...,
    }
    """
    cut_db = {}

    for run, run_dict in all_aoe_results.items():
        cut_db[run] = {}

        for channel, res in run_dict.items():
            if res["status"] != "ok":
                cut_db[run][channel] = {
                    "low_cut": np.nan,
                    "high_cut": np.nan,
                }
                continue

            cut_db[run][channel] = {
                "low_cut": float(low_cut),
                "high_cut": float(high_cut),
            }

    return cut_db

def build_cut_db_from_dep_fraction(
    all_aoe_results,
    low_fraction=0.90,
    high_fraction=0.99,
):
    """
    Build cuts from DEP A/E distribution.

    low_cut  = DEP lower quantile at (1 - low_fraction)
    high_cut = DEP upper quantile at high_fraction
    """
    cut_db = {}

    for run, run_dict in all_aoe_results.items():
        cut_db[run] = {}

        for channel, res in run_dict.items():
            if res["status"] != "ok":
                cut_db[run][channel] = {
                    "low_cut": np.nan,
                    "high_cut": np.nan,
                    "n_dep": 0,
                }
                continue

            dep = np.asarray(res["aoe_db"]["selections"]["DEP"])
            dep = dep[np.isfinite(dep)]

            if len(dep) == 0:
                cut_db[run][channel] = {
                    "low_cut": np.nan,
                    "high_cut": np.nan,
                    "n_dep": 0,
                }
                continue

            low_cut = np.quantile(dep, 1 - low_fraction)
            high_cut = np.quantile(dep, high_fraction)

            cut_db[run][channel] = {
                "low_cut": float(low_cut),
                "high_cut": float(high_cut),
                "n_dep": int(len(dep)),
            }

    return cut_db
def compute_survival_fraction_db(all_aoe_results, cut_db):
    """
    Apply low/high cuts and compute survival fractions.

    Returns
    -------
    sf_db[run][channel][label] = {
        "n_total": ...,
        "n_survive": ...,
        "fraction": ...
    }
    """
    labels = ["DEP", "FEP", "ROI_bg", "ROI_signal"]
    sf_db = {}

    for run, run_dict in all_aoe_results.items():
        sf_db[run] = {}

        for channel, res in run_dict.items():
            if res["status"] != "ok":
                sf_db[run][channel] = {
                    "status": "fail",
                    "error": res.get("error", "unknown error"),
                }
                continue

            low_cut = cut_db[run][channel]["low_cut"]
            high_cut = cut_db[run][channel]["high_cut"]

            out = {
                "status": "ok",
                "low_cut": low_cut,
                "high_cut": high_cut,
            }

            for label in labels:
                arr = np.asarray(res["aoe_db"]["selections"][label])
                arr = arr[np.isfinite(arr)]

                n_total = len(arr)

                if n_total == 0 or not np.isfinite(low_cut) or not np.isfinite(high_cut):
                    out[label] = {
                        "n_total": int(n_total),
                        "n_survive": 0,
                        "fraction": np.nan,
                    }
                    continue

                mask = (arr >= low_cut) & (arr <= high_cut)
                n_survive = np.count_nonzero(mask)

                out[label] = {
                    "n_total": int(n_total),
                    "n_survive": int(n_survive),
                    "fraction": n_survive / n_total,
                }

            sf_db[run][channel] = out

    return sf_db
import matplotlib.pyplot as plt

def plot_hist_with_cuts(
    all_aoe_results,
    cut_db,
    run,
    channel,
    ylim_top=1e6,
    show_labels=("DEP", "FEP", "ROI_BG", "ROI_Signal"),
):
    """
    Plot stored histograms and overlay low/high cut lines.
    """
    res = all_aoe_results[run][channel]

    if res["status"] != "ok":
        print(f"[FAIL] {run} | {channel}: {res['error']}")
        return

    hist_db = res["hist_db"]
    low_cut = cut_db[run][channel]["low_cut"]
    high_cut = cut_db[run][channel]["high_cut"]

    plt.figure(figsize=(8, 6))

    label_map = {
        "DEP": "DEP",
        "FEP": "FEP",
        "ROI_BG": "ROI BG",
        "ROI_Signal": "ROI Signal",
    }

    for key in show_labels:
        plt.step(
            hist_db[key]["centers"],
            hist_db[key]["counts"],
            where="mid",
            label=label_map.get(key, key),
        )

    if np.isfinite(low_cut):
        plt.axvline(low_cut, ls="--", label=f"low cut = {low_cut:.3f}")
    if np.isfinite(high_cut):
        plt.axvline(high_cut, ls="--", label=f"high cut = {high_cut:.3f}")

    aoe_field = hist_db["meta"].get("aoe_field", "A/E")
    plt.xlabel(aoe_field)
    plt.ylabel("Counts")
    plt.yscale("log")
    plt.ylim(1, ylim_top)
    plt.title(f"{run} | {channel}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def make_fraction_summary(sf_db):
    """
    Flatten FEP / ROI survival fractions into an easy-to-read dict.
    """
    summary = {}

    for run, run_dict in sf_db.items():
        summary[run] = {}

        for channel, res in run_dict.items():
            if res["status"] != "ok":
                summary[run][channel] = {
                    "status": "fail",
                    "error": res.get("error", "unknown error"),
                }
                continue

            summary[run][channel] = {
                "low_cut": res["low_cut"],
                "high_cut": res["high_cut"],
                "DEP_fraction": res["DEP"]["fraction"],
                "FEP_fraction": res["FEP"]["fraction"],
                "ROI_bg_fraction": res["ROI_bg"]["fraction"],
                "ROI_signal_fraction": res["ROI_signal"]["fraction"],
                "FEP_n_total": res["FEP"]["n_total"],
                "ROI_bg_n_total": res["ROI_bg"]["n_total"],
                "ROI_signal_n_total": res["ROI_signal"]["n_total"],
            }

    return summary
# =========================================================
# 6. save / load
# =========================================================
def save_summary(summary, filename="summary.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(summary, f)
    print(f"saved to {filename}")


def load_summary(filename="summary.pkl"):
    with open(filename, "rb") as f:
        summary = pickle.load(f)
    print(f"loaded from {filename}")
    return summary
