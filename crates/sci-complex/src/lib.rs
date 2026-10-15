//! # Sci-Complex
//!
//! Complex arithmetic and elementary mathematical functions.
//!
//! Part of [Sci](https://github.com/p-sira/sci), a unified scientific computing library for Rust.
//!
//! Scalar functional operations on [`Complex`], shared with `sci_fft` and
//! `sci_special`. Components use `f64`; angles use radians. Elementary
//! functions use principal branches and a private Rust backend.
//!
//! Array broadcasting, operators, and backend types are not part of this API.
//!
//! # Numerical behavior
//!
//! Operations take constant work, except [`powi`] and [`powu`] (at most 32 squaring steps).
//! IEEE values are returned directly rather than converted to errors; only
//! [`from_polar`] validates a restricted domain. Overflow, underflow, and
//! cancellation can occur. Finite inverse functions use rationalized distances,
//! `log1p`, and scaled large-input limits to preserve small branch components
//! and avoid intermediate overflow. Branch sensitivity and rounding remain.
//! Nonfinite inverse inputs retain the backend's IEEE behavior; callers should
//! not infer componentwise results from a nonfinite magnitude or angle.
//!
//! # Definitions
//!
//! For `z = x + iy`, `abs(z) = sqrt(x² + y²)`, `conjugate(z) = x - iy`,
//! and `to_polar(z) = (abs(z), atan2(y, x))`. Polar angles are in
//! `(-π, π]`; `from_polar(r, θ) = r(cos θ + i sin θ)` accepts finite
//! `r >= 0` and finite `θ`. Inverse elementary functions use their principal
//! complex branches.
//!
//! # Example
//!
//! ```
//! use sci_complex::{self as complex, Complex};
//! let z = Complex { re: 3.0, im: 4.0 };
//! assert_eq!(complex::abs(z), 5.0);
//! assert_eq!(complex::conjugate(z), Complex { re: 3.0, im: -4.0 });
//! let (radius, phase) = complex::to_polar(z);
//! let restored = complex::from_polar(radius, phase)?;
//! assert!((restored.re - z.re).abs() < 1e-14);
//! # Ok::<(), sci_core::Error>(())
//! ```
//!
//! ## Acknowledgments
//!
//! We thank `num-complex`, which provides private
//! complex-number representations, and the Rust standard library.

#![forbid(unsafe_code)]
#![deny(missing_docs, rustdoc::broken_intra_doc_links)]

pub use sci_core::{Error, Result};
#[path = "mod.rs"]
mod implementation;
pub use implementation::*;
