//! # Sci-FFT
//!
//! Fourier transforms on shared real and complex data.
//!
//! Fourier, cosine/sine, and periodic operators use [`FftFloat`] (`f32` or
//! `f64`) with native precision. FFTLog Hankel transforms currently use `f64`.
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
//! Signal processing reuses [`fourier::fft`], [`fourier::ifft`], and
//! [`real::rfft`] from this crate.
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
    pub(crate) fn native<T: sci_core::Float>(z: Complex<T>) -> num_complex::Complex<T> {
        num_complex::Complex::new(z.re, z.im)
    }
    pub(crate) fn owned<T: sci_core::Float>(z: num_complex::Complex<T>) -> Complex<T> {
        Complex { re: z.re, im: z.im }
    }
}
#[path = "mod.rs"]
mod implementation;
pub use implementation::{FftFloat, RealTransformType};

/// Complex Fourier transforms, including multidimensional transforms.
pub mod fourier {
    pub use crate::implementation::{fft, fft2, fftn, ifft, ifft2, ifftn};
}

/// Real Fourier and discrete cosine/sine transforms.
pub mod real {
    pub use crate::implementation::{
        RealTransformType, dct, dctn, dst, dstn, hfft, hfft2, hfftn, idct, idctn, idst, idstn,
        ihfft, ihfft2, ihfftn, irfft, irfft2, irfftn, rfft, rfft2, rfftn,
    };
}

/// FFTLog Hankel transforms (requires the `hankel` feature).
#[cfg(feature = "hankel")]
pub mod hankel {
    pub use crate::implementation::{fht, fhtoffset, ifht};
}

/// Frequency grids, shifts, fast-length helpers, and periodic operators.
pub mod utilities {
    pub use crate::implementation::{
        cc_diff, cs_diff, diff, fftfreq, fftshift, hilbert, ifftshift, ihilbert, itilbert,
        next_fast_len, prev_fast_len, rfftfreq, sc_diff, shift, ss_diff, tilbert,
    };
}
