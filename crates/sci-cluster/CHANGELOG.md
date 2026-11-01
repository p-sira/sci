# Changelog

## 0.3.0

- Remove runtime dependencies on `sci-stats` and `sci-spatial` by retaining
  equivalent bounded population standard deviation and Euclidean condensed
  distance kernels, with parity tests for numerical edge cases.
- Remove py_vq and maxRstat aliases; use vq and max_rstat.
- Update workspace dependencies for the new minor releases.

## 0.2.0

Changes since the repository tag `v0.1.0`.

- Require `sci-core 0.2` and update Sci dependency requirements to matching releases. Shared error and numerical types are incompatible with those from `sci-core 0.1`.

## 0.1.0

- Initial release of hierarchical clustering, vector quantization, and clustering utilities.
