# Changelog

## 0.3.0

- Reuse unit-weight distance inputs without allocating normalization buffers for each cosine/correlation pair; validate against the original weighted formulas.

- Compute directed Hausdorff distances with a prevalidated sequential `hypot`
  kernel, avoiding per-pair unit-weight allocations and repeated coordinate
  validation. Preserve exhaustive search, tie indices, and overflow behavior;
  the native benchmark improved approximately 2.2–2.5×.
- Reuse barycentric scratch vectors across simplex candidates in `tsearch`
  instead of reallocating them for each trial. Preserve first-simplex ties,
  edge tolerance, and numerical validation; the native `tsearch` benchmark
  improved approximately 2×.
- Update shared Sci dependency requirements to the new minor versions.
- Borrow explicit distance weights until normalization requires owned storage.

- Validate `pdist`/`cdist` point sets once, reuse pairwise unit weights, and
  compute short, moderately scaled Euclidean distances with one square root
  instead of repeated `hypot` calls. Keep `hypot` for extreme or longer
  vectors, preserving work limits and errors; the fast path can differ by a
  few final rounding bits. Native Euclidean benchmarks improved about 8.5×
  for `pdist` and 6.2× for `cdist`.

## 0.2.0

Changes since the repository tag `v0.1.0`.

- Require `sci-core 0.2` and update Sci dependency requirements to matching releases. Shared error and numerical types are incompatible with those from `sci-core 0.1`.

## 0.1.0

- Initial release of spatial geometry, distances, and coordinate transformations.
