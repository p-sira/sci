# Changelog

## 0.3.0

Use this release with `sci-core 0.3` and matching Sci crate releases when exchanging shared types.

### Breaking changes

- Replace `py_vq` with `vq` and `maxRstat` with `max_rstat`.

### Improvements

- Speed up `kmeans2` centroid assignment and reuse iteration storage while preserving first-centroid tie handling.
- Speed up hierarchical partitioning, maximum-cluster cuts, and leader lookup, including datasets with tied merge heights.
- Reduce dependencies: clustering no longer pulls in `sci-stats` or `sci-spatial`.

## 0.2.0

- Upgrade to shared `sci-core 0.2` types. Update cooperating Sci crates together; `sci-core 0.1` types are incompatible.

## 0.1.0

- Initial release of hierarchical clustering, vector quantization, and clustering utilities.
