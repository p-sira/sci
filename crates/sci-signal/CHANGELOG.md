# Changelog

## 0.3.0

- Process second-order filter sections in one output buffer and evaluate short cosine windows with a harmonic recurrence. Use real half-spectrum FFTs in convolution and share spectral passes in Welch/coherence; check representative input modes against independent references.

- Solve DPSS windows of order below 128 directly from their symmetric
  tridiagonal matrix with Faer QR eigenvectors rather than a redundant dense
  eigendecomposition. Keep the original nalgebra solver for larger windows
  and numerical fallback, including periodic length 256; preserve ordering,
  parity, normalization, and truncation. The native DPSS benchmark improved
  approximately 1.45×.
- Remove choose_conv_method and ConvolutionMethod (SciPy timing calibration); rename closest_STFT_dual_window to closest_stft_dual_window.

- Remove freqresp, dfreqresp, sosfreqz, check_COLA and check_NOLA aliases; use freqs, freqz, freqz_sos, check_cola and check_nola.
- Validate computed real and complex outputs by borrowing their storage. Build mirrored windows and conjugate roots without temporary vector clones, and retain polynomial workspace storage.
- Upgrade private `pm-remez` to 0.3.5 and reuse nalgebra 0.35 for Remez
  eigenvalues, removing the duplicate faer stack and its default dependencies.
- Solve two-tap Remez designs directly, avoiding the backend's constant-proxy
  failure. Preserve symmetric taps, sample-rate units, and gain/weight scaling.
- Document that the nalgebra backend does not expose an eigenvalue iteration
  limit; exchange iterations remain capped at 100 with degree-24 proxies.

- Use a scaled banded Cholesky solve for Whittaker–Henderson smoothing,
  retaining a dense Cholesky fallback for doubtful pivots or nonfinite
  intermediate results. Reduce normal-equation work to O(n · order²) and
  memory to O(n · order); the native smoothing benchmark improved about 16×.

## 0.2.0

Changes since the repository tag `v0.1.0`.

- Require `sci-core 0.2` and update Sci dependency requirements to matching releases. Shared error and numerical types are incompatible with those from `sci-core 0.1`.
- Accelerate direct `convolve` by evaluating only products retained by the requested output mode, preserving summation order and the existing work limit.

## 0.1.0

- Initial release of signal processing, convolution, spectral analysis, and filter design.
