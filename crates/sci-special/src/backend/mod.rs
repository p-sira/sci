//! Private pure Rust numerical kernels.
#![allow(unused_imports, clippy::excessive_precision)]

/// Static error string, as in Ellip.
pub type StrErr = &'static str;

#[allow(unused_imports)]
mod cephes64;
mod real;
mod utils;
pub use real::*;

mod elementary;
pub use elementary::*;
mod elliptic;
pub use elliptic::*;
mod boxcox;
pub use boxcox::*;
mod expint;
pub use expint::*;
mod error;
pub use error::*;
mod gamma_extra;
pub use gamma_extra::*;
pub(crate) mod bessel;
pub(crate) mod complex;
pub use bessel::{cyl_hankel_1, cyl_hankel_1e, cyl_hankel_2, cyl_hankel_2e};
pub use bessel::{
    iv as cyl_bessel_i, ive as cyl_bessel_ie, kv as cyl_bessel_k, kve as cyl_bessel_ke,
};
#[allow(dead_code)] // Private trait methods are also used by the independent unit tests.
mod faddeeva;
mod spherical;
pub use complex::wofz;
pub use spherical::*;

mod lambert;
pub use lambert::*;
mod orthogonal;
pub use orthogonal::*;

mod distributions;
pub use distributions::*;

mod wright;
pub use wright::wrightomega;

mod kelvin;
pub use kelvin::*;
mod complex_integrals;

mod legendre;
pub use legendre::*;

mod double_double;
mod ratios;
pub(crate) use ratios::iv_ratio_pair;
mod hypergeometric;
pub use hypergeometric::hyp0f1;

mod struve;
pub use struve::*;

mod kolmogorov;
pub use kolmogorov::*;

mod statistics;
pub use statistics::*;

mod smirnov;
pub use smirnov::*;

mod kummer;
pub(crate) mod legacy;
mod specfun;
