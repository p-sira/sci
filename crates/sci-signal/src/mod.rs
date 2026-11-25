/// Advanced IIR prototypes and filter design utilities.
pub mod advanced_design;
/// Convolution, correlation, deconvolution, and sample-rate conversion.
pub mod convolution;
/// Classical IIR filter design and transfer-function transformations.
pub mod design;
/// Time-domain digital filtering and frequency-response functions.
pub mod filtering;
/// FIR least-squares, window, and equiripple filter design.
pub mod fir;
/// Two-dimensional filtering, convolution, and Hilbert transforms.
pub mod multidimensional;
/// Peak and relative-extremum detection and characterization.
pub mod peaks;
/// Filter representations, partial fractions, and second-order sections.
pub mod representations;
/// Fourier analysis, spectra, spectrograms, and short-time transforms.
pub mod spectral;
/// Spline interpolation, smoothing, and spline-based filters.
pub mod splines;
/// State-space models, simulation, and discretization.
pub mod systems;
/// Signal envelopes, overlap-add convolution, and auxiliary utilities.
pub mod utilities;
/// Signal waveforms, chirps, trends, and pulse generators.
pub mod waveforms;
/// Symmetric and periodic taper windows.
pub mod windows;

// Shared sibling-module dependencies stay private to the implementation.
use advanced_design::{BesselNorm, bessel, ellip};
use convolution::convolve;
use design::{
    BandType, IirFamily, Zpk, buttord, cheb1ord, cheb2ord, iirfilter, normalize, tf2zpk, zpk2tf,
};
use filtering::{TransferFunction, firwin, freqs, freqz, freqz_sos};
use multidimensional::sepfir2d;
use spectral::{fft, ifft, resample, rfft};
use windows::get_window;

use crate::{Error, Mode, Result};

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
