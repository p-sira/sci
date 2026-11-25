# Changelog

## 0.3.0

Use this release with `sci-core 0.3` and matching Sci crate releases when exchanging shared types.

### Breaking changes

- Move filter functions into family modules, for example `kalman_predict` to `kalman::kalman_predict`. See the [migration guide](../../MIGRATING.md).

### Improvements

- Reduce particle-filter work when likelihoods are not requested and speed up validation of positive-definite covariance matrices.
- Reduce particle-filter dependencies by consolidating propagation and weighted moments on Smelly 0.2.

## 0.2.0

- Upgrade to shared `sci-core 0.2` types. Update cooperating Sci crates together; `sci-core 0.1` types are incompatible.
- Use nalgebra 0.35 for filtering and covariance validation.

## 0.1.0

- Initial release of state-estimation filters and covariance validation.
