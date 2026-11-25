# Changelog

## 0.3.0

Use this release with `sci-core 0.3` and matching Sci crate releases when exchanging shared types.

### Breaking changes

- Move signal-processing functions into family modules, for example `convolve` to `convolution::convolve`. See the [migration guide](../../MIGRATING.md) for replacement paths.
- Remove `choose_conv_method` and `ConvolutionMethod`. Select `convolve`, `fftconvolve`, or `oaconvolve` explicitly for your workload.
- Rename `closest_STFT_dual_window` to `closest_stft_dual_window`.
- Replace `freqresp`, `dfreqresp`, and `sosfreqz` with `freqs`, `freqz`, and `freqz_sos`; replace `check_COLA` and `check_NOLA` with `check_cola` and `check_nola` in their family modules.

### Fixed

- Support two-tap Remez filter designs that previously failed.

### Improvements

- Speed up Whittaker–Henderson smoothing and reduce memory use for typical inputs, retaining a fallback for difficult systems.
- Speed up short DPSS windows, second-order-section filtering, short cosine windows, FFT convolution, and Welch/coherence analysis.
- Reduce temporary allocations in signal validation, window construction, and filter design.
- Upgrade `pm-remez` to 0.3.5 and select its nalgebra backend to avoid a duplicate faer dependency stack. Remez exchange iterations remain bounded; the backend eigenvalue solver does not expose an iteration limit.
- Include the bundled SciPy-derived license notice in the published package.

## 0.2.0

- Upgrade to shared `sci-core 0.2` types. Update cooperating Sci crates together; `sci-core 0.1` types are incompatible.
- Accelerate direct `convolve` by evaluating only products retained by the requested output mode, preserving summation order and the existing work limit.

## 0.1.0

- Initial release of signal processing, convolution, spectral analysis, and filter design.
