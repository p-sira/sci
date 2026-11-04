# Changelog

## 0.3.0

- Validate `pdist`/`cdist` point sets once, reuse pairwise unit weights, and
  compute short, moderately scaled Euclidean distances with one square root
  instead of repeated `hypot` calls. Keep `hypot` for extreme or longer
  vectors, preserving work limits and errors; the fast path can differ by a
  few final rounding bits. Native Euclidean benchmarks improved about 8.5×
  for `pdist` and 6.2× for `cdist`.
- Update shared Sci dependency requirements to the new minor versions.
- Borrow explicit distance weights until normalization requires owned storage.

## 0.2.0

Changes since the repository tag `v0.1.0`.

- Require `sci-core 0.2` and update Sci dependency requirements to matching releases. Shared error and numerical types are incompatible with those from `sci-core 0.1`.

## 0.1.0

- Initial release of spatial geometry, distances, and coordinate transformations.
