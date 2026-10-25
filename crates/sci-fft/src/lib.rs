//! # Sci-FFT
//!
//! Fourier transforms on shared real and complex data.
//!
//! Part of [Sci](https://github.com/p-sira/sci), a unified scientific computing library for Rust.
//!
//! Forward transforms are unnormalized; inverses divide by transformed lengths.
//! The multidimensional and extended transform functions validate finite data and bound work to 1,048,576 entries and
//! 16 axes. Multidimensional input uses flat row-major storage plus explicit
//! shapes and axes. Two-dimensional wrappers accept shared row matrices.
//! Periodic differential, Hilbert, Tilbert, hyperbolic, and shift operators
//! accept finite real samples and reuse the shared Fourier kernels. Their sign,
//! endpoint, and Nyquist conventions are documented per function.
//! Signal processing reuses [`fft`], [`ifft`], and [`rfft`] from this crate.
//! The default `hankel` feature adds FFTLog transforms and their private Rust
//! gamma backend. All transforms compile without XSF, C++ compilation, or
//! libclang. Disable default features to omit FFTLog. Through the Sci
//! facade, enable `fft-hankel` to include FFTLog.
//!
//! For a length-`n` sequence, the forward transform is
//! `X[k] = Σ x[j] exp(-2π i j k/n)` and the inverse is
//! `x[j] = (1/n) Σ X[k] exp(2π i j k/n)`. Real transforms use the same
//! forward sign and return the nonnegative-frequency half-spectrum.
//!
//! ## Acknowledgments
//!
//! We thank `num-complex`, `rustfft`, `realfft`, `rustdct`,
//! and `special` for their valuable numerical building blocks.

#![forbid(unsafe_code)]
#![deny(missing_docs, rustdoc::broken_intra_doc_links)]

use sci_core as error;
pub use sci_core::{Error, Result};
mod complex {
    pub use sci_complex::*;
    #[inline]
    pub(crate) fn native(z: Complex) -> num_complex::Complex64 {
        num_complex::Complex64::new(z.re, z.im)
    }
    #[inline]
    pub(crate) fn owned(z: num_complex::Complex64) -> Complex {
        Complex { re: z.re, im: z.im }
    }
}
#[path = "mod.rs"]
mod implementation;
pub use implementation::*;
