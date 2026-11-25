# Sci-Interpolate

Scalar polynomial, B-spline, surface, and multidimensional interpolation.

The public namespaces are `polynomial`, `spline`, `multidimensional`, and
`fitting`. Shared `Spline`, `ParametricSpline`, and `SurfaceSpline` types remain
available from the crate root.

Polynomial and PCHIP interpolation accept `T: sci_core::Float`; spline and fitting algorithms use `f64`.

Part of [Sci](https://github.com/p-sira/sci), a unified scientific computing library for Rust.

Functions use slices and shared spline types. Polynomial coefficients use
descending powers, interoperating with signal APIs. Parametric curves support
up to three coordinates; regular grids up to eight axes. Each function
documents supported methods, boundary conditions, and work limits.

Polynomial coefficients are ordered from highest to lowest power.
Spline coefficients use increasing basis index, with degree distinct from
order (`order = degree + 1`). Evaluation is within the documented knot or
grid domain; extrapolation and duplicate-coordinate behavior are specified
by each constructor rather than inferred globally.

## Acknowledgments

We thank `nalgebra`, `splinefit`, and `delaunator` for
their interpolation and geometric building blocks.
