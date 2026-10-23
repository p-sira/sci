# Changelog

## 0.2.0

Changes since the repository tag `v0.1.0`.

- Require `sci-core 0.2` and update Sci dependency requirements to matching releases. Shared error and numerical types are incompatible with those from `sci-core 0.1`.
- Accelerate direct `convolve` by evaluating only products retained by the requested output mode, preserving summation order and the existing work limit.

## 0.1.0

- Initial release of signal processing, convolution, spectral analysis, and filter design.
