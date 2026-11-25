# Changelog

## 0.3.0

Use this release with `sci-core 0.3` and matching Sci crate releases when exchanging shared types.

### Breaking changes

- Move integration functions into family modules, for example `simpson` to `sampled::simpson`. See the [migration guide](../../MIGRATING.md).

### Added

- Support `f32` and `f64` sampled-data integration and fixed quadrature, including callbacks in the selected precision. Gaussian nodes and weights are generated in `f64` and rounded to the selected type.

### Improvements

- Reduce dependencies: integration no longer pulls in `sci-special` or `nalgebra`.

## 0.2.0

- Upgrade to shared `sci-core 0.2` types. Update cooperating Sci crates together; `sci-core 0.1` types are incompatible.
- Improve weights and error coefficients for uniform Newton–Cotes rules of orders 1 through 12.

## 0.1.0

- Initial release of numerical quadrature and sampled-data integration.
