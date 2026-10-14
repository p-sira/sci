//! # Linear algebra
//!
//! Operations on `f64` vectors and matrices through [`VectorInput`] and
//! [`MatrixInput`]. Inputs may use shared slices/rows, ndarray arrays, nalgebra
//! matrices, or faer matrices, including read-only and mutable views. Different
//! libraries may be mixed within a call. Entry ordering follows logical indices,
//! including transposed, column-major, and strided storage; inputs are never mutated.
//! Results use shared row vectors, matching `sci_differentiate::jacobian`,
//! and can be copied back through the input traits' conversion methods.
//!
//! Faer is the private numerical backend for products, decompositions, and
//! solves. Matrix functions use bounded native assembly with faer kernels.
//! Input types select storage only.
//!
//! ## Features
//!
//! The `faer` feature (enabled by default) exposes faer input adapters and
//! conversions. `nalgebra` and `ndarray` enable their respective adapters and
//! optional dependencies. All three can be enabled together. With default
//! features disabled, shared slices/rows still use faer for computations.
//! The umbrella `sci` crate forwards these as `linalg-faer`, `linalg-nalgebra`,
//! and `linalg-ndarray`; its defaults enable all three.
//! The original dot/product helpers propagate IEEE values; added numerical
//! algorithms validate finite inputs and report shared errors.
//!
//! ## Input conventions
//!
//! Shared rows are borrowed; library matrices are copied into shared rows once
//! at the API boundary. Required real vector arguments also accept library
//! vectors; optional vectors retain slice signatures. Typed BLAS/LAPACK function
//! pointers retain shared slice/row signatures. Complex inputs to `cdf2rdf`
//! retain shared complex vectors/rows. A `0 × n` input with `n > 0` is accepted
//! by basic products, transpose, and matrix collections, but rejected by APIs
//! that convert to shared rows, since those rows cannot retain its column count.
//! Shared output rows have the same limitation. Library-to-library conversions
//! preserve empty axes. Untyped empty matrix literals/collections may need a
//! type annotation; generic function items used as pointers need a concrete
//! signature. See [`MatrixInput`] and [`VectorInput`] for runnable examples.
//!
//! Dense algorithm input axes are bounded at 256, constructors at 1,000,000
//! output entries. Input conversions and basic helpers bound each axis and the
//! entry count at 1,000,000. Faer uses bounded
//! algorithm-specific sweep limits. Tighter limits and signature adaptations
//! are documented per function.
//!
//! Matrix products use `C[i,j] = Σ A[i,k] B[k,j]`. Solvers interpret vectors
//! in the same logical row and column order as their matrices; no implicit
//! transposition or conjugation is performed. Singular systems, invalid
//! decompositions, and nonfinite algorithm outputs are returned as [`Error`].
//!
//! ## Examples
//!
//! ```
//! use sci_linalg as linalg;
//! let jacobian = sci_differentiate::jacobian(
//!     |x| vec![x[0] * x[1], x[0] + x[1]], &[2.0, 3.0], 1e-5,
//! )?;
//! let direction = linalg::matvec(&jacobian, &[1.0, 0.0])?;
//! assert!((direction[0] - 3.0).abs() < 1e-8);
//! # Ok::<(), sci_core::Error>(())
//! ```
//!
//! ## Acknowledgments
//!
//! We thank `dyn-stack`, `faer`, `num-complex`, and the
//! optional `nalgebra` and `ndarray` interoperability libraries.

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
use sci_special as special;
#[path = "mod.rs"]
mod implementation;
pub use implementation::*;
