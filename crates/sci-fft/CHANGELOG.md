# Changelog

## 0.2.1

- Inline private complex conversion and periodic multiplier helpers.

## Unreleased

- The default `hankel` feature enables `fht`, `ifht`, and `fhtoffset` with the private Rust gamma backend. All transforms build without XSF, C++ compilation, or libclang. Use `default-features = false` to omit FFTLog. The Sci facade selects this capability with `fft-hankel`.

## 0.2.0

Changes since the repository tag `v0.1.0`.

- Require `sci-core 0.2` and update Sci dependency requirements to matching releases. Shared error and numerical types are incompatible with those from `sci-core 0.1`.

## 0.1.0

- Initial release of Fourier transforms, real transforms, and periodic operators.
