# Changelog

## 0.3.0

Use this release with `sci-core 0.3` and matching Sci crate releases when exchanging shared types.

### Breaking changes

- Move interpolation functions into family modules, for example `lagrange` to `polynomial::lagrange`. See the [migration guide](../../MIGRATING.md).
- Replace `splrep` and `splprep` with `make_splrep` and `make_splprep`.

### Added

- Support `f32` and `f64` barycentric, Krogh, Lagrange, and PCHIP interpolation.

### Improvements

- Speed up B-spline interpolation construction for well-conditioned systems while retaining a fallback for ill-conditioned inputs.
- Speed up multidimensional interpolation with singleton axes, tensor-product surface evaluation, and repeated spline derivatives.

## 0.2.0

- Upgrade to shared `sci-core 0.2` types. Update cooperating Sci crates together; `sci-core 0.1` types are incompatible.

## 0.1.0

- Initial release of interpolation and shared spline APIs.
