//! # Sci-Special
//!
//! Special mathematical functions.
//!
//! Elliptic integrals, scalar gamma/error/Bessel functions, polynomial evaluators,
//! and stable reductions accept `T: sci_core::Float`. Sequence and wave APIs
//! currently use `f64`. Specialized kernels generally evaluate in binary64 and
//! convert to the requested type; elliptic integrals compute in the selected type.
//!
//! Part of [Sci](https://github.com/p-sira/sci), a unified scientific computing library for Rust.
//!
//! Gamma, Bessel, elliptic, probability, polynomial, and combinatorial functions
//! with floating-point scalars and shared Sci types. Specialized algorithms use private
//! pure Rust kernels and ellip backends; elementary formulas use the standard library.
//! Use `value.is_finite()` to check scalar results for NaN and infinity.
//! Complex gamma/error/Bessel/Hankel/Airy/Fresnel APIs are available under
//! `complex` using shared generic complex values and errors.
//! There is no broadcasting, implicit complex-input overloading, exact integer
//! mode, or derivative-axis output for array functions. Polynomial constructors
//! return descending coefficients. Read each function's documented real domain,
//! output ordering, and bounded order/size limits.
//! Scalar functions return documented IEEE values; fallible integrals and
//! sequences return the shared [`crate::Result`].
//!
//! Elliptic integrals use Ellip's notation: `ellipf(phi, m)` computes `F(phi, m)`,
//! with radians and elliptic parameter `m = k²`; incomplete third-kind arguments
//! are `(phi, n, m)`. Supported principal values are documented per function.
//! SciPy is an independent reference; complete SciPy parity is not required.

//! ## Examples
//!
//! ```
//! use sci_special as special;
//! assert_eq!(special::gamma(5.0_f64), 24.0);
//! let k = special::elliptic::ellipk(0.5_f64)?;
//! assert!((k - 1.8540746773013719).abs() < 1e-14);
//! # Ok::<(), sci_core::Error>(())
//! ```
//!
//! ## References
//!
//! - [NIST DLMF, gamma functions](https://dlmf.nist.gov/5).
//! - [NIST DLMF, elliptic integrals](https://dlmf.nist.gov/19).
//!
//! ## Acknowledgments
//!
//! We thank `ellip`, `nalgebra`, `num-complex`, `complex-bessel`, and `lambert_w`
//! for their specialized numerical algorithms. Retained third-party license
//! notices accompany the private translated kernels in `LICENSES`.

#![forbid(unsafe_code)]
#![deny(missing_docs, rustdoc::broken_intra_doc_links)]

pub use sci_core::{Error, Result};
mod backend;

pub mod complex;
#[path = "mod.rs"]
mod implementation;
pub use implementation::*;
