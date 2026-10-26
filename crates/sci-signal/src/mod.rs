mod advanced_design;
mod convolution;
mod design;
mod filtering;
mod fir;
mod multidimensional;
mod peaks;
mod representations;
mod spectral;
mod splines;
mod systems;
mod utilities;
mod waveforms;
/// Symmetric and periodic taper windows.
pub mod windows;

pub use advanced_design::*;
pub use convolution::*;
pub use design::*;
pub use filtering::*;
pub use fir::*;
pub use multidimensional::*;
pub use peaks::*;
pub use representations::*;
pub use sci_fft::{fft, ifft, rfft};
pub use spectral::*;
pub use splines::*;
pub use systems::*;
pub use utilities::*;
pub use waveforms::*;
pub use windows::get_window;

use crate::{Error, Result};

// Limits apply before allocation and expensive numerical work.
const MAX_SAMPLES: usize = 1_048_576;
const MAX_WORK: usize = 16_777_216;

fn size(n: usize) -> Result<()> {
    if n > MAX_SAMPLES {
        Err(Error::InvalidInput("signal length exceeds 1048576"))
    } else {
        Ok(())
    }
}
fn work(a: usize, b: usize) -> Result<()> {
    if a.checked_mul(b).is_none_or(|n| n > MAX_WORK) {
        Err(Error::InvalidInput(
            "numerical work exceeds 16777216 operations",
        ))
    } else {
        Ok(())
    }
}
fn samples(x: &[f64]) -> Result<()> {
    size(x.len())?;
    crate::error::is_finite_slice(x)
}
fn nonempty(x: &[f64]) -> Result<()> {
    samples(x)?;
    if x.is_empty() {
        Err(Error::InvalidInput("signal must not be empty"))
    } else {
        Ok(())
    }
}
fn positive(x: f64) -> Result<()> {
    if x.is_finite() && x > 0. {
        Ok(())
    } else {
        Err(Error::InvalidInput("parameter must be finite and positive"))
    }
}
fn checked(x: Vec<f64>) -> Result<Vec<f64>> {
    check_real(&x)?;
    Ok(x)
}
fn check_real(x: &[f64]) -> Result<()> {
    if x.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(Error::NonFiniteEvaluation)
    }
}
fn complex_checked(x: Vec<crate::complex::Complex>) -> Result<Vec<crate::complex::Complex>> {
    check_complex(&x)?;
    Ok(x)
}
fn check_complex(x: &[crate::complex::Complex]) -> Result<()> {
    if x.iter().all(|v| v.re.is_finite() && v.im.is_finite()) {
        Ok(())
    } else {
        Err(Error::NonFiniteEvaluation)
    }
}
