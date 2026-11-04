# Changelog

## 0.3.0

- Compute directed Hausdorff distances with a prevalidated sequential `hypot`
  kernel, avoiding per-pair unit-weight allocations and repeated coordinate
  validation. Preserve exhaustive search, tie indices, and overflow behavior;
  the native benchmark improved approximately 2.2–2.5×.
- Update shared Sci dependency requirements to the new minor versions.
- Borrow explicit distance weights until normalization requires owned storage.

## 0.2.0

Changes since the repository tag `v0.1.0`.

- Require `sci-core 0.2` and update Sci dependency requirements to matching releases. Shared error and numerical types are incompatible with those from `sci-core 0.1`.

## 0.1.0

- Initial release of spatial geometry, distances, and coordinate transformations.
