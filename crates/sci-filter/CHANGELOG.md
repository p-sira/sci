# Changelog

## 0.3.0

- Update shared Sci dependency requirements to the new minor versions.
- Upgrade Smelly to 0.2 and reuse its particle propagation and weighted moments;
  remove kalman_filters and its random dependencies. Preserve native log-weight
  normalization, effective sample size, resampling boundaries, and callback errors.

## 0.2.0

Changes since the repository tag `v0.1.0`.

- Require `sci-core 0.2` and update Sci dependency requirements to matching releases. Shared error and numerical types are incompatible with those from `sci-core 0.1`.
- Use the shared nalgebra 0.35 dependency for filtering and covariance validation.
- Add a regression test for covariance eigensolver work exhaustion.

## 0.1.0

- Initial release of state-estimation filters and covariance validation.
