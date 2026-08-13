
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erfc


def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def left_exgauss_tail(x, A, mu_tail, sigma_tail, tau):
    """
    Left-sided exponentially modified Gaussian tail.
    """
    lam = 1.0 / tau

    return (
        A * lam / 2.0
        * np.exp(
            lam * (x - mu_tail)
            + 0.5 * (lam * sigma_tail) ** 2
        )
        * erfc(
            (x - mu_tail + lam * sigma_tail**2)
            / (np.sqrt(2) * sigma_tail)
        )
    )


def aoe_peak_plus_slow_tail(
    x,
    A_peak,
    mu_peak,
    sigma_peak,
    A_tail,
    mu_tail,
    sigma_tail,
    tau,
    C,
):
    peak = gaussian(x, A_peak, mu_peak, sigma_peak)
    tail = left_exgauss_tail(x, A_tail, mu_tail, sigma_tail, tau)
    bg = C * np.ones_like(x)

    return peak + tail + bg


def estimate_tail_counts_from_fit(
    centers,
    popt,
    threshold=1.0,
    side="above",
):
    """
    Estimate fitted slow-tail counts using histogram bin centers.

    Since the model is fitted to counts-per-bin, this returns a sum over bins,
    not a continuous density integral.
    """
    centers = np.asarray(centers, dtype=float)

    A_peak, mu_peak, sigma_peak, A_tail, mu_tail, sigma_tail, tau, C = popt

    y_peak = gaussian(centers, A_peak, mu_peak, sigma_peak)
    y_tail = left_exgauss_tail(centers, A_tail, mu_tail, sigma_tail, tau)
    y_bg = C * np.ones_like(centers)
    y_total = y_peak + y_tail + y_bg

    finite = (
        np.isfinite(centers)
        & np.isfinite(y_peak)
        & np.isfinite(y_tail)
        & np.isfinite(y_bg)
        & np.isfinite(y_total)
    )

    if side == "above":
        region = finite & (centers > threshold)
    elif side == "below":
        region = finite & (centers < threshold)
    else:
        raise ValueError("side must be 'above' or 'below'")

    all_region = finite

    n_tail_region = np.sum(y_tail[region])
    n_peak_region = np.sum(y_peak[region])
    n_bg_region = np.sum(y_bg[region])
    n_total_region = np.sum(y_total[region])

    n_tail_total = np.sum(y_tail[all_region])
    n_peak_total = np.sum(y_peak[all_region])
    n_bg_total = np.sum(y_bg[all_region])
    n_model_total = np.sum(y_total[all_region])

    return {
        "threshold": float(threshold),
        "side": side,

        "n_tail_region": float(n_tail_region),
        "n_peak_region": float(n_peak_region),
        "n_bg_region": float(n_bg_region),
        "n_total_region": float(n_total_region),

        "n_tail_total": float(n_tail_total),
        "n_peak_total": float(n_peak_total),
        "n_bg_total": float(n_bg_total),
        "n_model_total": float(n_model_total),

        # useful fractions
        "tail_region_fraction_of_tail": (
            float(n_tail_region / n_tail_total) if n_tail_total > 0 else np.nan
        ),
        "tail_region_fraction_of_total": (
            float(n_tail_region / n_model_total) if n_model_total > 0 else np.nan
        ),
        "tail_region_fraction_of_region": (
            float(n_tail_region / n_total_region) if n_total_region > 0 else np.nan
        ),
        "tail_to_peak_total": (
            float(n_tail_total / n_peak_total) if n_peak_total > 0 else np.nan
        ),
    }


def plot_tail_fit_components(
    x_fit,
    y_fit,
    yerr,
    popt,
    fit_range=(0.75, 1.08),
    threshold=1.0,
    ylim=(1, 1e4),
    title="A/E fit: peak + slow low-A/E tail",
):
    A_peak, mu_peak, sigma_peak, A_tail, mu_tail, sigma_tail, tau, C = popt

    x_plot = np.linspace(fit_range[0], fit_range[1], 1200)

    y_total = aoe_peak_plus_slow_tail(x_plot, *popt)
    y_peak = gaussian(x_plot, A_peak, mu_peak, sigma_peak)
    y_tail = left_exgauss_tail(x_plot, A_tail, mu_tail, sigma_tail, tau)
    y_bg = C * np.ones_like(x_plot)

    plt.figure(figsize=(8, 5))

    plt.errorbar(
        x_fit,
        y_fit,
        yerr=yerr,
        fmt=".",
        markersize=3,
        alpha=0.5,
        label="histogram",
    )

    plt.plot(x_plot, y_total, linewidth=2, label="total fit")
    plt.plot(x_plot, y_peak, "--", label="normal A/E peak")
    plt.plot(x_plot, y_tail, "--", label="slow low-A/E tail")
    plt.plot(x_plot, y_bg, ":", label="constant background")

    # Shade slow-tail leakage above threshold
    mask_above = x_plot > threshold
    plt.fill_between(
        x_plot[mask_above],
        1e-300,
        y_tail[mask_above],
        alpha=0.3,
        label=f"slow-tail part above {threshold:.2f}",
    )

    plt.axvline(mu_peak, linestyle="--", alpha=0.5, label=f"mu_peak = {mu_peak:.4f}")
    plt.axvline(mu_tail, linestyle="--", alpha=0.5, label=f"mu_tail = {mu_tail:.4f}")
    plt.axvline(threshold, linestyle=":", alpha=0.8, label=f"threshold = {threshold:.2f}")

    plt.xlim(fit_range)
    if ylim is not None:
        plt.ylim(*ylim)

    plt.xlabel("A/E")
    plt.ylabel("Counts")
    plt.title(title)
    plt.yscale("log")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
