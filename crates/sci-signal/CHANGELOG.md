# Changelog

## 0.2.1

- Upgrade private `pm-remez` to 0.3.5 and reuse nalgebra 0.35 for Remez
  eigenvalues, removing the duplicate faer stack and its default dependencies.
- Solve two-tap Remez designs directly, avoiding the backend's constant-proxy
  failure. Preserve symmetric taps, sample-rate units, and gain/weight scaling.
- Document that the nalgebra backend does not expose an eigenvalue iteration
  limit; exchange iterations remain capped at 100 with degree-24 proxies.

## 0.2.0

Changes since the repository tag `v0.1.0`.

- Require `sci-core 0.2` and update Sci dependency requirements to matching releases. Shared error and numerical types are incompatible with those from `sci-core 0.1`.
- Accelerate direct `convolve` by evaluating only products retained by the requested output mode, preserving summation order and the existing work limit.

## 0.1.0

- Initial release of signal processing, convolution, spectral analysis, and filter design.
