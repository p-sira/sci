//! # Sci-Special
//!
//! Special mathematical functions.
//!
//! Part of [Sci](https://github.com/p-sira/sci), a unified scientific computing library for Rust.
//!
//! Gamma, Bessel, elliptic, probability, polynomial, and combinatorial functions
//! with `f64` scalars and shared Sci types. Specialized algorithms use private
//! special, xsf, and ellip backends; elementary formulas use the standard library.
//! [`geterr`] and [`seterr`] use explicit immutable error policies.
//! There is no broadcasting, general complex-input overloading, exact integer
//! mode, or derivative-axis output for array functions. Polynomial constructors
//! return descending coefficients. Read each function's documented real domain,
//! output ordering, and bounded order/size limits.
//! Scalar functions return documented IEEE values; fallible integrals and
//! sequences return the shared [`crate::Result`].
//!
//! ## Examples
//!
//! ```
//! use sci_special as special;
//! assert_eq!(special::gamma(5.0), 24.0);
//! let k = special::elliptic::ellipk(0.5)?;
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
//! We thank `ellip`, `nalgebra`, `num-complex`, `special`,
//! and `xsf` for their specialized numerical algorithms.

#![forbid(unsafe_code)]
#![deny(missing_docs, rustdoc::broken_intra_doc_links)]

use sci_core as error;
pub use sci_core::{Error, Result};
mod complex {
    pub use sci_complex::*;
    pub(crate) fn native(z: Complex) -> num_complex::Complex64 {
        num_complex::Complex64::new(z.re, z.im)
    }
    pub(crate) fn owned(z: num_complex::Complex64) -> Complex {
        Complex { re: z.re, im: z.im }
    }
}
#[path = "mod.rs"]
mod implementation;
pub use implementation::*;
