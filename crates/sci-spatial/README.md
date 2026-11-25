# Sci-Spatial

Euclidean geometry, spatial queries, and pairwise distances.

Distance metrics and pairwise operations live under `sci_spatial::distance`;
geometric transforms and point location live under `sci_spatial::transform`.

Part of [Sci](https://github.com/p-sira/sci), a unified scientific computing library for Rust.

Points are rows of `Vec<Vec<f64>>`, interoperable with `sci_linalg`.
Coordinates must be finite. Vector and matrix work is limited to 1,000,000
entries and pairwise work to 10,000,000 coordinate comparisons. Overflow
returns an error; mathematically undefined distances return NaN as documented.

Euclidean distance is `d(a,b) = sqrt(Σ (a[i] - b[i])²)`. Pairwise
functions return rows for the first point set and columns for the second.
Metric-specific normalization, weights, and covariance conventions are
documented on each metric; there is no implicit coordinate scaling.

## Acknowledgments

We thank `nalgebra` for its private matrix operations.
