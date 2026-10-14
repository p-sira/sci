use crate::error::{finite, finite_slice};
use crate::{Error, Result, complex::Complex};
use std::f64::consts::{LN_2, TAU};

fn samples(x: &[f64]) -> Result<()> {
    super::length(x.len())?;
    finite_slice(x)
}
fn period(p: f64) -> Result<()> {
    if !p.is_finite() || p <= 0.0 {
        Err(Error::InvalidInput("period must be finite and positive"))
    } else {
        Ok(())
    }
}
fn multiplier(
    x: &[f64],
    period_value: f64,
    mut symbol: impl FnMut(usize, f64, usize) -> Result<Complex>,
) -> Result<Vec<f64>> {
    samples(x)?;
    period(period_value)?;
    let n = x.len();
    let mut bins = crate::rfft(x);
    for (k, z) in bins.iter_mut().enumerate() {
        let w = if k == 0 {
            0.0
        } else {
            finite(k as f64 * (TAU / period_value))?
        };
        let m = symbol(k, w, n)?;
        let result = crate::complex::multiply(*z, m);
        z.re = finite(result.re)?;
        z.im = finite(result.im)?;
    }
    crate::irfft(&bins, Some(n))
}
fn real(re: f64) -> Result<Complex> {
    Ok(Complex {
        re: finite(re)?,
        im: 0.0,
    })
}
fn imaginary(im: f64) -> Result<Complex> {
    Ok(Complex {
        re: 0.0,
        im: finite(im)?,
    })
}
fn log_cosh(x: f64) -> f64 {
    x.abs() + (-2.0 * x.abs()).exp().ln_1p() - LN_2
}
fn log_sinh(x: f64) -> f64 {
    x.abs() + (-(-2.0 * x.abs()).exp_m1()).ln() - LN_2
}
fn ratio(a: f64, b: f64, sinh_a: bool, sinh_b: bool) -> Result<f64> {
    finite(a)?;
    finite(b)?;
    if sinh_b && b == 0.0 {
        return Err(Error::InvalidInput("zero hyperbolic denominator"));
    }
    if sinh_a && a == 0.0 {
        return Ok(0.0);
    }
    let log_a = if sinh_a { log_sinh(a) } else { log_cosh(a) };
    let log_b = if sinh_b { log_sinh(b) } else { log_cosh(b) };
    let sign_a = if sinh_a { a.signum() } else { 1.0 };
    let sign_b = if sinh_b { b.signum() } else { 1.0 };
    finite((log_a - log_b).exp() * sign_a * sign_b)
}

/// Differentiates/integrates a periodic sequence with Fourier symbol `(i*2πk/period)^order`.
///
/// ## Parameters
///
/// `input` is one period sampled uniformly, with period in the original independent-variable units. The zero Fourier mode is handled explicitly. No Python kernel cache or axis batching is exposed. `order` is signed; negative values integrate with zero integration constant. Order zero copies samples. Nonzero orders remove the mean; odd orders discard even Nyquist.
/// ## Domain
///
/// Finite nonempty real samples, at most 1,048,576 entries; finite positive period and finite operator parameters. NaN/infinity are rejected. Order is bounded to −64..=64.
/// ## Returns
///
/// A same-length real derivative or zero-mean integral, in input units divided by independent-variable units to the order.
/// ## Errors
///
/// Invalid values, dimensions, work limits, or nonfinite arithmetic return shared errors.
/// ## Numerical behavior
///
/// Uses shared real FFT/inverse FFT, O(n log n) work and O(n) storage. Frequency multipliers follow the SciPy convention; arithmetic overflow returns an error. Periodic extension can cause endpoint ringing. Odd imaginary multipliers discard even Nyquist; returned values retain original sample ordering.
/// # Examples
///
/// ```
/// let x=sci_fft::diff(&[1.0,2.0],0,std::f64::consts::TAU)?;
/// assert_eq!(x,vec![1.0,2.0]);
/// # Ok::<(),sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy diff](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.diff.html).
pub fn diff(input: &[f64], order: i32, period_value: f64) -> Result<Vec<f64>> {
    if !(-64..=64).contains(&order) {
        return Err(Error::InvalidInput("periodic derivative order limit"));
    }
    samples(input)?;
    period(period_value)?;
    if order == 0 {
        return Ok(input.to_vec());
    }
    multiplier(input, period_value, |k, w, n| {
        if k == 0 || n.is_multiple_of(2) && k == n / 2 && order.rem_euclid(2) == 1 {
            return real(0.0);
        }
        let power = finite(w.powi(order))?;
        match order.rem_euclid(4) {
            0 => real(power),
            1 => imaginary(power),
            2 => real(-power),
            _ => imaginary(-power),
        }
    })
}

/// Applies the periodic Hilbert Fourier symbol `1*i*sign(k)`, with zero DC and even Nyquist.
///
/// ## Parameters
///
/// `input` is one period sampled uniformly, with period in the original independent-variable units. The zero Fourier mode is handled explicitly. No Python kernel cache or axis batching is exposed. This follows FFTPACK’s Hilbert sign convention; the imaginary part of signal::hilbert has the opposite sign.
/// ## Domain
///
/// Finite nonempty real samples, at most 1,048,576 entries; no extra parameters. NaN/infinity are rejected.
/// ## Returns
///
/// A same-length real Hilbert component.
/// ## Errors
///
/// Invalid values, dimensions, work limits, or nonfinite arithmetic return shared errors.
/// ## Numerical behavior
///
/// Uses shared real FFT/inverse FFT, O(n log n) work and O(n) storage. Frequency multipliers follow the SciPy convention; arithmetic overflow returns an error. Periodic extension can cause endpoint ringing. Odd imaginary multipliers discard even Nyquist; returned values retain original sample ordering.
/// # Examples
///
/// ```
/// assert_eq!(sci_fft::hilbert(&[1.0,1.0])?,vec![0.0,0.0]);
/// # Ok::<(),sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy hilbert](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.hilbert.html).
pub fn hilbert(input: &[f64]) -> Result<Vec<f64>> {
    multiplier(input, TAU, |k, _, n| {
        if k == 0 || n.is_multiple_of(2) && k == n / 2 {
            real(0.0)
        } else {
            imaginary(1.0)
        }
    })
}

/// Applies the periodic Hilbert Fourier symbol `-1*i*sign(k)`, with zero DC and even Nyquist.
///
/// ## Parameters
///
/// `input` is one period sampled uniformly, with period in the original independent-variable units. The zero Fourier mode is handled explicitly. No Python kernel cache or axis batching is exposed. This is the inverse of [`hilbert`] on sequences with zero DC and even Nyquist. It has the same sign as the imaginary component of `signal::hilbert`.
/// ## Domain
///
/// Finite nonempty real samples, at most 1,048,576 entries; no extra parameters. NaN/infinity are rejected.
/// ## Returns
///
/// A same-length real Hilbert component.
/// ## Errors
///
/// Invalid values, dimensions, work limits, or nonfinite arithmetic return shared errors.
/// ## Numerical behavior
///
/// Uses shared real FFT/inverse FFT, O(n log n) work and O(n) storage. Frequency multipliers follow the SciPy convention; arithmetic overflow returns an error. Periodic extension can cause endpoint ringing. Odd imaginary multipliers discard even Nyquist; returned values retain original sample ordering.
/// # Examples
///
/// ```
/// assert_eq!(sci_fft::ihilbert(&[1.0,1.0])?,vec![0.0,0.0]);
/// # Ok::<(),sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy ihilbert](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.ihilbert.html).
pub fn ihilbert(input: &[f64]) -> Result<Vec<f64>> {
    multiplier(input, TAU, |k, _, n| {
        if k == 0 || n.is_multiple_of(2) && k == n / 2 {
            real(0.0)
        } else {
            imaginary(-1.0)
        }
    })
}

/// Applies the Tilbert Fourier symbol `i*coth(2πkh/period)`.
///
/// ## Parameters
///
/// `input` is one period sampled uniformly, with period in the original independent-variable units. The zero Fourier mode is handled explicitly. No Python kernel cache or axis batching is exposed. `h` has independent-variable units; DC and even Nyquist are discarded.
/// ## Domain
///
/// Finite nonempty real samples, at most 1,048,576 entries; finite positive period and finite operator parameters. NaN/infinity are rejected. h must be nonzero.
/// ## Returns
///
/// A same-length real Tilbert component.
/// ## Errors
///
/// Invalid values, dimensions, work limits, or nonfinite arithmetic return shared errors.
/// ## Numerical behavior
///
/// Uses shared real FFT/inverse FFT, O(n log n) work and O(n) storage. Frequency multipliers follow the SciPy convention; arithmetic overflow returns an error. Periodic extension can cause endpoint ringing. Odd imaginary multipliers discard even Nyquist; returned values retain original sample ordering.
/// # Examples
///
/// ```
/// let x=sci_fft::tilbert(&[1.0,1.0],1.0,std::f64::consts::TAU)?;
/// assert_eq!(x,vec![0.0,0.0]);
/// # Ok::<(),sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy tilbert](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.tilbert.html).
pub fn tilbert(input: &[f64], h: f64, period_value: f64) -> Result<Vec<f64>> {
    finite_slice(&[h])?;
    if h == 0.0 {
        return Err(Error::InvalidInput("Tilbert height must be nonzero"));
    }
    multiplier(input, period_value, |k, w, n| {
        if k == 0 || n.is_multiple_of(2) && k == n / 2 {
            real(0.0)
        } else {
            imaginary(1.0 / finite(w * h)?.tanh())
        }
    })
}

/// Applies the inverse Tilbert Fourier symbol `−i*tanh(2πkh/period)`.
///
/// ## Parameters
///
/// `input` is one period sampled uniformly, with period in the original independent-variable units. The zero Fourier mode is handled explicitly. No Python kernel cache or axis batching is exposed. `h` has independent-variable units; DC and even Nyquist are discarded.
/// ## Domain
///
/// Finite nonempty real samples, at most 1,048,576 entries; finite positive period and finite operator parameters. NaN/infinity are rejected. h may be zero, yielding zero output.
/// ## Returns
///
/// A same-length real Tilbert component.
/// ## Errors
///
/// Invalid values, dimensions, work limits, or nonfinite arithmetic return shared errors.
/// ## Numerical behavior
///
/// Uses shared real FFT/inverse FFT, O(n log n) work and O(n) storage. Frequency multipliers follow the SciPy convention; arithmetic overflow returns an error. Periodic extension can cause endpoint ringing. Odd imaginary multipliers discard even Nyquist; returned values retain original sample ordering.
/// # Examples
///
/// ```
/// let x=sci_fft::itilbert(&[1.0,1.0],1.0,std::f64::consts::TAU)?;
/// assert_eq!(x,vec![0.0,0.0]);
/// # Ok::<(),sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy itilbert](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.itilbert.html).
pub fn itilbert(input: &[f64], h: f64, period_value: f64) -> Result<Vec<f64>> {
    finite_slice(&[h])?;
    multiplier(input, period_value, |k, w, n| {
        if k == 0 || n.is_multiple_of(2) && k == n / 2 {
            real(0.0)
        } else {
            imaginary(-finite(w * h)?.tanh())
        }
    })
}

/// Shifts the periodic interpolant: `y(u)=x(u+a)`, using Fourier symbol `exp(i*2πka/period)`.
///
/// ## Parameters
///
/// `input` is one period sampled uniformly, with period in the original independent-variable units. The zero Fourier mode is handled explicitly. No Python kernel cache or axis batching is exposed. `a` is a signed displacement in independent-variable units. For even n, Nyquist uses the real multiplier cos(phase)+sin(phase), matching SciPy FFTPACK’s packed-kernel convention.
/// ## Domain
///
/// Finite nonempty real samples, at most 1,048,576 entries; finite positive period and finite operator parameters. NaN/infinity are rejected.
/// ## Returns
///
/// A same-length real shifted sequence; its mean is preserved.
/// ## Errors
///
/// Invalid values, dimensions, work limits, or nonfinite arithmetic return shared errors.
/// ## Numerical behavior
///
/// Uses shared real FFT/inverse FFT, O(n log n) work and O(n) storage. Frequency multipliers follow the SciPy convention; arithmetic overflow returns an error. Periodic extension can cause endpoint ringing. Odd imaginary multipliers discard even Nyquist; returned values retain original sample ordering. Large finite phases may lose argument-reduction accuracy.
/// # Examples
///
/// ```
/// assert_eq!(sci_fft::shift(&[1.0,2.0],0.0,2.0)?,vec![1.0,2.0]);
/// # Ok::<(),sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy shift](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.shift.html).
pub fn shift(input: &[f64], a: f64, period_value: f64) -> Result<Vec<f64>> {
    finite_slice(&[a])?;
    multiplier(input, period_value, |k, w, n| {
        let phase = finite(w * a)?;
        let (sin, cos) = phase.sin_cos();
        if n.is_multiple_of(2) && k == n / 2 {
            real(cos + sin)
        } else {
            Ok(Complex { re: cos, im: sin })
        }
    })
}

/// Applies the hyperbolic Fourier symbol `cosh(2πka/period)/cosh(2πkb/period)`, with DC multiplier `1`.
///
/// ## Parameters
///
/// `input` is one period sampled uniformly, with period in the original independent-variable units. The zero Fourier mode is handled explicitly. No Python kernel cache or axis batching is exposed. `a` and `b` have independent-variable units. Even Nyquist is retained.
/// ## Domain
///
/// Finite nonempty real samples, at most 1,048,576 entries; finite positive period and finite operator parameters. NaN/infinity are rejected.
/// ## Returns
///
/// A same-length real pseudo-derivative.
/// ## Errors
///
/// Invalid values, dimensions, work limits, or nonfinite arithmetic return shared errors.
/// ## Numerical behavior
///
/// Uses shared real FFT/inverse FFT, O(n log n) work and O(n) storage. Frequency multipliers follow the SciPy convention; arithmetic overflow returns an error. Periodic extension can cause endpoint ringing. Odd imaginary multipliers discard even Nyquist; returned values retain original sample ordering. Log-domain hyperbolic ratios avoid overflow in separate sinh/cosh evaluations; tiny ratios can underflow to zero.
/// # Examples
///
/// ```
/// let x=sci_fft::cc_diff(&[1.0,1.0],1.0,1.0,std::f64::consts::TAU)?;
/// assert_eq!(x,vec![1.0;2]);
/// # Ok::<(),sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy cc_diff](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.cc_diff.html).
pub fn cc_diff(input: &[f64], a: f64, b: f64, period_value: f64) -> Result<Vec<f64>> {
    finite_slice(&[a, b])?;
    multiplier(input, period_value, |k, w, _| {
        if k == 0 {
            return real(1.0);
        }
        let value = ratio(finite(w * a)?, finite(w * b)?, false, false)?;
        real(value)
    })
}

/// Applies the hyperbolic Fourier symbol `-1*i*cosh(2πka/period)/sinh(2πkb/period)`, with DC multiplier `0`.
///
/// ## Parameters
///
/// `input` is one period sampled uniformly, with period in the original independent-variable units. The zero Fourier mode is handled explicitly. No Python kernel cache or axis batching is exposed. `a` and `b` have independent-variable units. Even Nyquist is discarded.
/// ## Domain
///
/// Finite nonempty real samples, at most 1,048,576 entries; finite positive period and finite operator parameters. NaN/infinity are rejected. b must be nonzero.
/// ## Returns
///
/// A same-length real pseudo-derivative.
/// ## Errors
///
/// Invalid values, dimensions, work limits, or nonfinite arithmetic return shared errors.
/// ## Numerical behavior
///
/// Uses shared real FFT/inverse FFT, O(n log n) work and O(n) storage. Frequency multipliers follow the SciPy convention; arithmetic overflow returns an error. Periodic extension can cause endpoint ringing. Odd imaginary multipliers discard even Nyquist; returned values retain original sample ordering. Log-domain hyperbolic ratios avoid overflow in separate sinh/cosh evaluations; tiny ratios can underflow to zero.
/// # Examples
///
/// ```
/// let x=sci_fft::cs_diff(&[1.0,1.0],1.0,1.0,std::f64::consts::TAU)?;
/// assert_eq!(x,vec![0.0;2]);
/// # Ok::<(),sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy cs_diff](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.cs_diff.html).
pub fn cs_diff(input: &[f64], a: f64, b: f64, period_value: f64) -> Result<Vec<f64>> {
    finite_slice(&[a, b])?;
    if b == 0.0 {
        return Err(Error::InvalidInput(
            "hyperbolic denominator parameter must be nonzero",
        ));
    }
    multiplier(input, period_value, |k, w, n| {
        if k == 0 {
            return real(0.0);
        }
        if n.is_multiple_of(2) && k == n / 2 {
            return real(0.0);
        }
        let value = ratio(finite(w * a)?, finite(w * b)?, false, true)?;
        imaginary(-value)
    })
}

/// Applies the hyperbolic Fourier symbol `1*i*sinh(2πka/period)/cosh(2πkb/period)`, with DC multiplier `0`.
///
/// ## Parameters
///
/// `input` is one period sampled uniformly, with period in the original independent-variable units. The zero Fourier mode is handled explicitly. No Python kernel cache or axis batching is exposed. `a` and `b` have independent-variable units. Even Nyquist is discarded.
/// ## Domain
///
/// Finite nonempty real samples, at most 1,048,576 entries; finite positive period and finite operator parameters. NaN/infinity are rejected.
/// ## Returns
///
/// A same-length real pseudo-derivative.
/// ## Errors
///
/// Invalid values, dimensions, work limits, or nonfinite arithmetic return shared errors.
/// ## Numerical behavior
///
/// Uses shared real FFT/inverse FFT, O(n log n) work and O(n) storage. Frequency multipliers follow the SciPy convention; arithmetic overflow returns an error. Periodic extension can cause endpoint ringing. Odd imaginary multipliers discard even Nyquist; returned values retain original sample ordering. Log-domain hyperbolic ratios avoid overflow in separate sinh/cosh evaluations; tiny ratios can underflow to zero.
/// # Examples
///
/// ```
/// let x=sci_fft::sc_diff(&[1.0,1.0],1.0,1.0,std::f64::consts::TAU)?;
/// assert_eq!(x,vec![0.0;2]);
/// # Ok::<(),sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy sc_diff](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.sc_diff.html).
pub fn sc_diff(input: &[f64], a: f64, b: f64, period_value: f64) -> Result<Vec<f64>> {
    finite_slice(&[a, b])?;
    multiplier(input, period_value, |k, w, n| {
        if k == 0 {
            return real(0.0);
        }
        if n.is_multiple_of(2) && k == n / 2 {
            return real(0.0);
        }
        let value = ratio(finite(w * a)?, finite(w * b)?, true, false)?;
        imaginary(value)
    })
}

/// Applies the hyperbolic Fourier symbol `sinh(2πka/period)/sinh(2πkb/period)`, with DC multiplier `a/b`.
///
/// ## Parameters
///
/// `input` is one period sampled uniformly, with period in the original independent-variable units. The zero Fourier mode is handled explicitly. No Python kernel cache or axis batching is exposed. `a` and `b` have independent-variable units. Even Nyquist is retained.
/// ## Domain
///
/// Finite nonempty real samples, at most 1,048,576 entries; finite positive period and finite operator parameters. NaN/infinity are rejected. b must be nonzero.
/// ## Returns
///
/// A same-length real pseudo-derivative.
/// ## Errors
///
/// Invalid values, dimensions, work limits, or nonfinite arithmetic return shared errors.
/// ## Numerical behavior
///
/// Uses shared real FFT/inverse FFT, O(n log n) work and O(n) storage. Frequency multipliers follow the SciPy convention; arithmetic overflow returns an error. Periodic extension can cause endpoint ringing. Odd imaginary multipliers discard even Nyquist; returned values retain original sample ordering. Log-domain hyperbolic ratios avoid overflow in separate sinh/cosh evaluations; tiny ratios can underflow to zero.
/// # Examples
///
/// ```
/// let x=sci_fft::ss_diff(&[1.0,1.0],1.0,1.0,std::f64::consts::TAU)?;
/// assert_eq!(x,vec![1.0;2]);
/// # Ok::<(),sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy ss_diff](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.ss_diff.html).
pub fn ss_diff(input: &[f64], a: f64, b: f64, period_value: f64) -> Result<Vec<f64>> {
    finite_slice(&[a, b])?;
    if b == 0.0 {
        return Err(Error::InvalidInput(
            "hyperbolic denominator parameter must be nonzero",
        ));
    }
    multiplier(input, period_value, |k, w, n| {
        if k == 0 {
            return real(a / b);
        }
        let _ = n;
        let value = ratio(finite(w * a)?, finite(w * b)?, true, true)?;
        real(value)
    })
}
