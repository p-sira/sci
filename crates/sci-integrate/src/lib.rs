//! # Numerical integration
//!
//! One-dimensional integration of samples and fixed-order Gaussian quadrature.
//!
//! Functions accept slices or scalar callbacks; sampled inputs have at most
//! 1,000,001 entries. Adaptive quadrature,
//! infinite intervals, multidimensional integration, and ODE solvers are absent.
//!
//! Sampled rules integrate ordinates over their supplied spacing. Callback
//! rules evaluate a bounded number of points on the requested finite interval;
//! Gaussian rules use the documented fixed order and map the reference
//! interval `[-1, 1]` to the integration interval.
//!
//! ## Acknowledgments
//!
//! We thank `nalgebra` for its numerical linear algebra.

#![forbid(unsafe_code)]
#![deny(missing_docs, rustdoc::broken_intra_doc_links)]

use sci_core as error;
pub use sci_core::{Error, Result};
use sci_special as special;
#[path = "mod.rs"]
mod implementation;
pub use implementation::*;
