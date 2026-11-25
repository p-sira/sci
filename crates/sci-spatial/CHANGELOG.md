# Changelog

## 0.3.0

Use this release with `sci-core 0.3` and matching Sci crate releases when exchanging shared types.

### Breaking changes

- Move spatial functions into family modules, for example `pdist` to `distance::pdist`. See the [migration guide](../../MIGRATING.md).

### Fixed

- Improve cosine and correlation distance accuracy for nearly parallel vectors.

### Improvements

- Speed up `pdist`, `cdist`, cosine/correlation distances, and directed Hausdorff distances by reducing repeated validation and allocation.
- Speed up ordinary Euclidean distances while retaining overflow-resistant evaluation for extreme inputs. Results can differ by a few final rounding bits.
- Reduce temporary allocations in `tsearch` while preserving first-simplex tie handling and edge tolerance.

## 0.2.0

- Upgrade to shared `sci-core 0.2` types. Update cooperating Sci crates together; `sci-core 0.1` types are incompatible.

## 0.1.0

- Initial release of spatial geometry, distances, and coordinate transformations.
