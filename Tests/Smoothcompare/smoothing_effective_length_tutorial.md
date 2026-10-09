# Repeated Moving-Average Smoothing: Effective Length

## Baby tutorial

Suppose a waveform is sampled every $\Delta t$, and you smooth it using a moving average of $M$ samples:

$
y[i] = \frac{1}{M}\sum_{k=0}^{M-1}x[i-k].
$

This replaces every sample with the average of $M$ neighboring samples. If the sampling period is $10\,\mathrm{ns}$, a 10-sample window is usually described as a $100\,\mathrm{ns}$ moving average.

Applying the same moving average several times is **not** equivalent to simply multiplying the window length by the number of passes. The rectangular kernels are convolved:

- one pass gives a rectangular kernel;
- two passes give a triangular kernel;
- additional passes give increasingly rounded kernels.

This is the discrete counterpart of the [Irwin–Hall distribution](https://en.wikipedia.org/wiki/Irwin%E2%80%93Hall_distribution), which describes the sum of independent uniform variables. Its central result for us is that the variances of the individual uniform distributions add.

## Effective-length definition

The phrase *effective length* is ambiguous unless we define it. Here, it means:

> The length of one rectangular moving-average window having the same temporal variance as the repeated filter.

For one discrete $M$-sample boxcar, the variance of the sample positions is

$
\sigma_1^2 = \frac{M^2-1}{12}
\quad\text{samples}^2.
$

After $N$ identical smoothing passes, variances add:

$
\sigma_N^2 = N\frac{M^2-1}{12}.
$

An equivalent boxcar containing $M_\mathrm{eff}$ samples must satisfy

$
\frac{M_\mathrm{eff}^2-1}{12}
=N\frac{M^2-1}{12}.
$

Therefore,

$
\boxed{M_\mathrm{eff}=\sqrt{N(M^2-1)+1}}
$

and the effective time length is

$
\boxed{L_\mathrm{eff}=M_\mathrm{eff}\,\Delta t}.
$

For a sufficiently large window, the discrete correction becomes negligible:

$
L_\mathrm{eff}\approx L\sqrt{N}.
$

For example, $100\,\mathrm{ns}\times5$ is approximately

$
100\sqrt{5}=223.6\,\mathrm{ns},
$

not $500\,\mathrm{ns}$.

## Sampling rate and sampling period

These terms are reciprocals:

$
\Delta t=\frac{1}{f_s}.
$

For example,

$
f_s=100\,\mathrm{MHz}
\quad\Longleftrightarrow\quad
\Delta t=10\,\mathrm{ns}.
$

The number of samples in a requested window is

$
M=\operatorname{round}\!\left(\frac{L_\mathrm{requested}}{\Delta t}\right).
$

The actual implemented window is then

$
L_\mathrm{actual}=M\Delta t.
$

## Python converter

```python
import numpy as np


def smoothing_lengths(
    window_length_ns,
    n_smoothings,
    sampling_rate_hz,
    sampling_rate_uncertainty_hz=0.0,
):
    """Calculate characteristic lengths of repeated moving-average smoothing.

    Parameters
    ----------
    window_length_ns : float
        Requested duration of each moving-average window in ns.
    n_smoothings : int
        Number of identical moving-average passes.
    sampling_rate_hz : float
        Sampling frequency in Hz; for 10 ns sampling, use 100e6.
    sampling_rate_uncertainty_hz : float, optional
        One-standard-deviation uncertainty of the physical sampling clock.
        Leave this at zero if the uncertainty is unknown or negligible.

    Returns
    -------
    dict
        All lengths and uncertainties are returned in ns.
    """
    if window_length_ns <= 0:
        raise ValueError("window_length_ns must be positive")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    if sampling_rate_uncertainty_hz < 0:
        raise ValueError("sampling_rate_uncertainty_hz cannot be negative")
    if int(n_smoothings) != n_smoothings or n_smoothings < 1:
        raise ValueError("n_smoothings must be a positive integer")

    n_smoothings = int(n_smoothings)
    sampling_period_ns = 1e9 / sampling_rate_hz

    # A digital window must contain an integer number of samples.
    window_samples = max(
        1,
        int(np.rint(window_length_ns / sampling_period_ns)),
    )
    actual_window_ns = window_samples * sampling_period_ns
    rounding_error_ns = actual_window_ns - window_length_ns

    # Equivalent single boxcar, matched by temporal variance.
    effective_samples = np.sqrt(
        n_smoothings * (window_samples**2 - 1) + 1
    )
    effective_length_ns = effective_samples * sampling_period_ns

    # Exact support of N convolved M-tap filters.
    support_samples = n_smoothings * (window_samples - 1) + 1
    support_span_ns = (support_samples - 1) * sampling_period_ns
    group_delay_ns = support_span_ns / 2

    # Resolution associated with rounding a requested duration to samples.
    # This is a resolution scale, not automatically a statistical error.
    sensitivity = (
        n_smoothings * actual_window_ns / effective_length_ns
    )
    effective_quantization_half_step_ns = (
        sensitivity * sampling_period_ns / 2
    )
    effective_quantization_std_ns = (
        sensitivity * sampling_period_ns / np.sqrt(12)
    )

    # Genuine propagated clock uncertainty. For fixed sample counts, all
    # physical times are proportional to 1 / sampling_rate_hz.
    relative_clock_uncertainty = (
        sampling_rate_uncertainty_hz / sampling_rate_hz
    )
    effective_clock_uncertainty_ns = (
        effective_length_ns * relative_clock_uncertainty
    )
    support_clock_uncertainty_ns = (
        support_span_ns * relative_clock_uncertainty
    )
    delay_clock_uncertainty_ns = (
        group_delay_ns * relative_clock_uncertainty
    )

    return {
        "sampling_period_ns": sampling_period_ns,
        "window_samples": window_samples,
        "actual_window_ns": actual_window_ns,
        "rounding_error_ns": rounding_error_ns,
        "effective_samples": effective_samples,
        "effective_length_ns": effective_length_ns,
        "support_samples": support_samples,
        "support_span_ns": support_span_ns,
        "group_delay_ns": group_delay_ns,
        "effective_quantization_half_step_ns": (
            effective_quantization_half_step_ns
        ),
        "effective_quantization_std_ns": (
            effective_quantization_std_ns
        ),
        "effective_clock_uncertainty_ns": (
            effective_clock_uncertainty_ns
        ),
        "support_clock_uncertainty_ns": support_clock_uncertainty_ns,
        "delay_clock_uncertainty_ns": delay_clock_uncertainty_ns,
    }
```

## First example: $100\,\mathrm{ns}\times5$

For a waveform sampled every $10\,\mathrm{ns}$, the sampling rate is $100\,\mathrm{MHz}$:

```python
result = smoothing_lengths(
    window_length_ns=100,
    n_smoothings=5,
    sampling_rate_hz=100e6,
)

for name, value in result.items():
    print(f"{name:42s}: {value:.4f}")
```

The important results are approximately:

| Output | Result | Meaning |
| --- | ---: | --- |
| `sampling_period_ns` | $10\,\mathrm{ns}$ | Separation between neighboring waveform samples |
| `window_samples` | 10 | Samples used in each moving average |
| `actual_window_ns` | $100\,\mathrm{ns}$ | Window actually implemented after rounding |
| `effective_samples` | 22.271 | Equivalent boxcar width in samples |
| `effective_length_ns` | $222.71\,\mathrm{ns}$ | Main effective-length result |
| `support_samples` | 46 | Number of kernel coefficients after five convolutions |
| `support_span_ns` | $450\,\mathrm{ns}$ | Time from the first to the last kernel sample |
| `group_delay_ns` | $225\,\mathrm{ns}$ | Delay of a causal implementation |
| `rounding_error_ns` | $0\,\mathrm{ns}$ | Requested 100 ns is exactly 10 samples |

## What every output means

### `sampling_period_ns`

This is the time between adjacent samples:

$
\Delta t=10^9/f_s\quad\text{in ns}.
$

It is $10\,\mathrm{ns}$ for $100\,\mathrm{MHz}$, $4\,\mathrm{ns}$ for $250\,\mathrm{MHz}$, and $1\,\mathrm{ns}$ for $1\,\mathrm{GHz}$.

### `window_samples`

This is the integer number of samples averaged during each pass. Digital filters cannot use, for example, 10.3 samples, so the requested duration must be rounded.

### `actual_window_ns`

This is the window the code can actually implement:

$
L_\mathrm{actual}=M\Delta t.
$

For a requested $103\,\mathrm{ns}$ window with $10\,\mathrm{ns}$ sampling, $M=10$, so the actual window is $100\,\mathrm{ns}$.

### `rounding_error_ns`

This is a signed, deterministic difference:

$
e_\mathrm{round}=L_\mathrm{actual}-L_\mathrm{requested}.
$

In the 103 ns example it is $-3\,\mathrm{ns}$. It is not a random uncertainty because, after choosing $M$, the implemented window is known exactly in sample units.

### `effective_samples` and `effective_length_ns`

These describe how broadly the repeated filter weights are distributed around their center. They are the most appropriate outputs when comparing smoothing strength between different $(M,N)$ configurations.

For example, approximately equal effective lengths should produce approximately equal broadening, although their detailed kernel shapes can still differ.

### `support_samples`

After convolution, the number of nonzero kernel coefficients is

$
K=N(M-1)+1.
$

For $M=10$ and $N=5$, this gives $K=46$ samples.

### `support_span_ns`

This is the distance between the first and last kernel sample:

$
T_\mathrm{support}=(K-1)\Delta t=N(M-1)\Delta t.
$

It is much larger than the variance-matched effective length. It answers “over what complete range can an input sample influence the output?” rather than “how strong is the smoothing?”

Some software may informally call $K\Delta t$ the support duration. This tutorial uses the physically clearer first-to-last-sample span, $(K-1)\Delta t$. State your convention when reporting it.

### `group_delay_ns`

A causal symmetric finite-impulse-response filter delays features by the center of its kernel:

$
T_\mathrm{delay}=\frac{T_\mathrm{support}}{2}.
$

For offline analysis, centered filtering can compensate for this delay. Near waveform boundaries, however, padding or truncation choices still matter.

## Understanding the reported uncertainty-like quantities

### 1. Sampling alone does not create a statistical uncertainty

If the filter is exactly 10 samples wide and the clock is exactly known, its width is exactly 10 samples. There is no statistical error bar to calculate merely because the data are sampled.

### 2. `effective_quantization_half_step_ns`

If a user requests an arbitrary continuous duration and it is rounded to the nearest sample, the raw window has a resolution of approximately

$
\pm\frac{\Delta t}{2}.
$

The converter propagates that half-sample scale to the effective length. This is a **discretization bound or resolution scale**, not a measured confidence interval.

If the requested length is already exactly an integer number of samples, use the explicit `rounding_error_ns`; it may be zero even though the general resolution scale is nonzero.

### 3. `effective_quantization_std_ns`

If the unknown rounding offset is deliberately modeled as uniformly distributed between $-\Delta t/2$ and $+\Delta t/2$, its standard deviation is

$
u_L=\frac{\Delta t}{\sqrt{12}}.
$

The code propagates this to $L_\mathrm{eff}$. Only interpret it as a $1\sigma$ uncertainty when that uniform-rounding model is appropriate.

### 4. `effective_clock_uncertainty_ns`

This is a genuine propagated uncertainty if the sampling frequency was measured as

$
f_s\pm u_f.
$

For a fixed number of samples, every physical time is proportional to $1/f_s$. Therefore,

$
\frac{u_t}{t}=\frac{u_f}{f_s},
$

or

$
u(L_\mathrm{eff})
=L_\mathrm{eff}\frac{u_f}{f_s}.
$

The support and group-delay clock uncertainties are calculated in the same way.

Example with a $100\,\mathrm{MHz}$ clock whose standard uncertainty is $1\,\mathrm{kHz}$:

```python
result = smoothing_lengths(
    window_length_ns=100,
    n_smoothings=5,
    sampling_rate_hz=100e6,
    sampling_rate_uncertainty_hz=1e3,
)

print(result["effective_length_ns"])
print(result["effective_clock_uncertainty_ns"])
```

The relative clock uncertainty is $10^{-5}$, so the effective length is

$
222.71\,\mathrm{ns}\pm0.00223\,\mathrm{ns}.
$

Do not enter the sampling frequency itself as its uncertainty. If only $f_s$ is known, leave `sampling_rate_uncertainty_hz=0` and report the sample rounding separately.

## Comparing several smoothing configurations

```python
setups = [
    (100, 5),
    (300, 5),
    (1000, 5),
]

for window_ns, repetitions in setups:
    result = smoothing_lengths(
        window_length_ns=window_ns,
        n_smoothings=repetitions,
        sampling_rate_hz=100e6,
    )

    print(
        f"{window_ns} ns x {repetitions}: "
        f"effective = {result['effective_length_ns']:.2f} ns, "
        f"support span = {result['support_span_ns']:.2f} ns"
    )
```

At $100\,\mathrm{MHz}$, the results are:

| Setup | Effective length | Full support span | Group delay |
| --- | ---: | ---: | ---: |
| $100\,\mathrm{ns}\times5$ | $222.71\,\mathrm{ns}$ | $450\,\mathrm{ns}$ | $225\,\mathrm{ns}$ |
| $300\,\mathrm{ns}\times5$ | $670.52\,\mathrm{ns}$ | $1450\,\mathrm{ns}$ | $725\,\mathrm{ns}$ |
| $1000\,\mathrm{ns}\times5$ | $2235.98\,\mathrm{ns}$ | $4950\,\mathrm{ns}$ | $2475\,\mathrm{ns}$ |

## Which number should I use?

| Question | Use |
| --- | --- |
| How strongly does this setup smooth compared with another setup? | `effective_length_ns` |
| Over what total time range does the kernel extend? | `support_span_ns` |
| How much does a causal filter shift a pulse? | `group_delay_ns` |
| What window did the digital implementation actually use? | `actual_window_ns` |
| How far was the requested duration rounded? | `rounding_error_ns` |
| What is the resolution imposed by whole samples? | `effective_quantization_half_step_ns` |
| What error comes from an uncertain physical clock? | `effective_clock_uncertainty_ns` |

For plot labels comparing smoothing configurations, the recommended quantity is the variance-matched effective length:

```python
label = (
    f"{window_ns} ns x {repetitions} "
    f"(effective {result['effective_length_ns']:.1f} ns)"
)
```

## Important assumptions

The formulas above assume:

1. every smoothing pass uses the same normalized rectangular window;
2. the passes are applied sequentially;
3. the sampling interval is constant;
4. effective width is defined by matching the second central moment;
5. boundary padding effects are ignored.

If your implementation uses different window lengths in different passes, the variances still add. For window sizes $M_1,M_2,\ldots,M_N$, use

$
M_\mathrm{eff}
=\sqrt{1+\sum_{j=1}^{N}(M_j^2-1)}.
$

This lets the same method compare mixed smoothing chains such as $100\,\mathrm{ns}$, followed by $300\,\mathrm{ns}$, followed by $500\,\mathrm{ns}$.

## Repeated Moving-Average Filters and the Irwin–Hall Distribution

A moving-average filter replaces each waveform sample by the average of (M) neighbouring samples. Its normalized discrete kernel is

[
h_M[n]=
\begin{cases}
1/M, & n=0,\ldots,M-1,\
0, & \text{otherwise}.
\end{cases}
]

Because all positions inside the window have equal weight, this kernel has the same mathematical form as a discrete uniform probability distribution. Applying the moving average (N) times is equivalent to convolving this kernel with itself (N) times:

\underbrace{h_Mh_M\cdots*h_M}_{N\ \mathrm{times}}.
]

The continuous counterpart of this repeated convolution is the Irwin–Hall distribution, which describes the sum of (N) independent uniformly distributed variables. A single uniform distribution is rectangular; convolving two produces a triangular shape, and further convolutions produce progressively smoother, bell-shaped kernels. For sufficiently large (N), the kernel approaches a Gaussian shape as a consequence of the central limit theorem.

This interpretation provides a convenient measure of the total smoothing scale. The variance of an (M)-point discrete uniform kernel is

[
\sigma_M^2=\frac{M^2-1}{12}.
]

Since variances add under convolution, the variance after (N) identical moving-average passes is

N\frac{M^2-1}{12}.
]

An equivalent single boxcar width can therefore be defined by matching its variance to that of the repeated kernel:

\sqrt{N(M^2-1)+1}
\approx M\sqrt{N}.
]

For a sampling interval (\Delta t), the corresponding effective smoothing time is

\Delta t\sqrt{N(M^2-1)+1}.
]

For example, a (100,\mathrm{ns}) moving average applied five times to a waveform sampled every (10,\mathrm{ns}) has (M=10) and

10,\mathrm{ns}\sqrt{5(10^2-1)+1}
\approx222.7,\mathrm{ns}.
]

The frequency-selective behaviour follows from the Fourier transform of the rectangular kernel. The magnitude response of one (M)-sample moving average is

\left|
\frac{\sin(\pi fM\Delta t)}
{M\sin(\pi f\Delta t)}
\right|.
]

This is a sinc-like low-pass response. Its first zero occurs at

[
f_0=\frac{1}{M\Delta t}=\frac{1}{L},
]

where (L=M\Delta t) is the duration of one averaging window. A signal component whose period is (T=L) completes one full oscillation inside the averaging window, causing its positive and negative contributions to cancel. Components with periods much longer than (L) are largely preserved, whereas components with periods comparable to or shorter than (L) are increasingly suppressed.

After (N) passes, the frequency response becomes

[
H_{M,N}(f)=H_M(f)^N.
]

Repeated smoothing therefore does not introduce new zero frequencies, but it strengthens the attenuation between them and produces a narrower low-frequency passband.

The Irwin–Hall picture and the frequency-response picture provide complementary interpretations. In the time domain, repeated moving averages produce a broader and smoother effective kernel. In the frequency domain, this corresponds to stronger suppression of short-period, high-frequency waveform components. A moving-average filter should nevertheless be understood as having a characteristic smoothing scale rather than a sharp cutoff frequency, because its sinc-like response contains a gradual transition and sidelobes.