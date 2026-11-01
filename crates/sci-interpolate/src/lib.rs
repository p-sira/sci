//! # Sci-Interpolate
//!
//! Scalar polynomial, B-spline, surface, and multidimensional interpolation.
//!
//! Polynomial and PCHIP interpolation accept `T: sci_core::Float`; other
//! constructors and spline algorithms currently accept `f64`.
//!
//! Part of [Sci](https://github.com/p-sira/sci), a unified scientific computing library for Rust.
//!
//! Functions use slices and shared spline types. Polynomial coefficients use
//! descending powers, interoperating with signal APIs. Parametric curves support
//! up to three coordinates; regular grids up to eight axes. Each function
//! documents supported methods, boundary conditions, and work limits.
//!
//! Polynomial coefficients are ordered from highest to lowest power.
//! Spline coefficients use increasing basis index, with degree distinct from
//! order (`order = degree + 1`). Evaluation is within the documented knot or
//! grid domain; extrapolation and duplicate-coordinate behavior are specified
//! by each constructor rather than inferred globally.
//!
//! ## Acknowledgments
//!
//! We thank `nalgebra`, `splinefit`, and `delaunator` for
//! their interpolation and geometric building blocks.

#![forbid(unsafe_code)]
#![deny(missing_docs, rustdoc::broken_intra_doc_links)]

use sci_core as error;
pub use sci_core::{Error, ParametricSpline, Result, Spline, SurfaceSpline};
#[path = "mod.rs"]
mod implementation;
/// Polynomial interpolation and rational approximation.
pub mod polynomial {
    pub use crate::fitting_impl::approximate_taylor_polynomial;
    pub use crate::implementation::{
        barycentric_interpolate, krogh_interpolate, lagrange, pade, pchip_interpolate,
    };
}
mod backend;
#[path = "fitting.rs"]
mod fitting_impl;
/// Spline construction, fitting, and smoothing.
pub mod fitting {
    pub use crate::fitting_impl::{
        bisplrep, generate_knots, make_interp_spline, make_lsq_spline, make_smoothing_spline,
        make_splprep, make_splrep,
    };
}
/// Interpolation on regular grids, scattered points, and surfaces.
pub mod multidimensional;
/// Spline evaluation, differentiation, and integration.
pub mod spline;
