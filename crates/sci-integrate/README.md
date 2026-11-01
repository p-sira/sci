# Sci-Integrate

One-dimensional integration of samples and fixed-order Gaussian quadrature.

Sample-based rules live under `sci_integrate::sampled`, and fixed Gaussian
quadrature and Newton–Cotes weights under `sci_integrate::quadrature`.

Sampled rules and fixed Gaussian quadrature accept `T: sci_core::Float`; Newton–Cotes weights use `f64`.

Part of [Sci](https://github.com/p-sira/sci), a unified scientific computing library for Rust.

Functions accept slices or scalar callbacks; sampled inputs have at most
1,000,001 entries. Adaptive quadrature,
infinite intervals, multidimensional integration, and ODE solvers are absent.

Sampled rules integrate ordinates over their supplied spacing. Callback
rules evaluate a bounded number of points on the requested finite interval;
Gaussian rules use the documented fixed order and map the reference
interval `[-1, 1]` to the integration interval.

The small, bounded quadrature solvers are implemented directly in Rust without
the general-purpose linear algebra or special-function crates.
