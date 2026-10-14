//! # Interpolation
//!
//! Scalar polynomial, B-spline, surface, and multidimensional interpolation.
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
pub use implementation::*;
mod backend;
mod fitting;
mod multidimensional;
mod spline;
pub use fitting::*;
pub use multidimensional::*;
pub use spline::*;
