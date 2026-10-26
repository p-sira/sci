# Changelog

## 0.3.0

- Remove the real selector from next_fast_len and prev_fast_len. Both use an 11-smooth size heuristic based on RustFFT guidance, with no promise of optimal CPU-specific performance.
- Update workspace dependencies for the new minor releases.

## Unreleased

- The default `hankel` feature enables `fht`, `ifht`, and `fhtoffset` with the private Rust gamma backend. All transforms build without XSF, C++ compilation, or libclang. Use `default-features = false` to omit FFTLog. The Sci facade selects this capability with `fft-hankel`.

## 0.2.0

Changes since the repository tag `v0.1.0`.

- Require `sci-core 0.2` and update Sci dependency requirements to matching releases. Shared error and numerical types are incompatible with those from `sci-core 0.1`.

## 0.1.0

- Initial release of Fourier transforms, real transforms, and periodic operators.
