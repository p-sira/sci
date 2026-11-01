use super::{checked, nonempty, positive, samples, size};
use crate::{Error, Result};
use std::f64::consts::{PI, TAU};

/// Evaluates a 2π-periodic ramp, rising from -1 to +1 over fraction `width` and falling over the remainder.
///
/// ## Parameters
///
/// `pub fn sawtooth(t: f64, width: f64) -> f64`
///
/// t is phase in radians; width=0 is a falling ramp and width=1 a rising ramp. Discontinuities use the start of the new period. Scalar adaptation of SciPy.
///
/// ## Domain
///
/// NaN propagates. Nonfinite phase or width/duty outside `[0,1]` returns NaN.
///
/// ## Returns
///
/// Output ordering, units and shapes follow the conventions above.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic. Cancellation and poorly conditioned filters or polynomial fits can reduce accuracy; no uniform relative-error guarantee is made. FFT routines use RustFFT, polynomial fits and DPSS use bounded private nalgebra solvers.
///
/// # Examples
///
/// ```
/// use sci_signal::waveforms::*;
/// let _result = sawtooth(0.,0.5);
/// ```
///
/// # References
///
/// [SciPy sawtooth definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sawtooth.html).
pub fn sawtooth(t: f64, width: f64) -> f64 {
    if !t.is_finite() || !width.is_finite() || !(0. ..=1.).contains(&width) {
        return f64::NAN;
    }
    let phase = t.rem_euclid(TAU) / TAU;
    if phase < width {
        2. * phase / width - 1.
    } else {
        1. - 2. * (phase - width) / (1. - width)
    }
}
/// Evaluates `+1` for `t mod 2π < 2π duty` and `-1` otherwise.
///
/// ## Parameters
///
/// `pub fn square(t: f64, duty: f64) -> f64`
///
/// t is phase in radians. duty=0 and duty=1 yield constant -1 and +1. Scalar adaptation of SciPy.
///
/// ## Domain
///
/// NaN propagates. Nonfinite phase or width/duty outside `[0,1]` returns NaN.
///
/// ## Returns
///
/// Output ordering, units and shapes follow the conventions above.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic. Cancellation and poorly conditioned filters or polynomial fits can reduce accuracy; no uniform relative-error guarantee is made. FFT routines use RustFFT, polynomial fits and DPSS use bounded private nalgebra solvers.
///
/// # Examples
///
/// ```
/// use sci_signal::waveforms::*;
/// let _result = square(0.,0.5);
/// ```
///
/// # References
///
/// [SciPy square definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.square.html).
pub fn square(t: f64, duty: f64) -> f64 {
    if !t.is_finite() || !duty.is_finite() || !(0. ..=1.).contains(&duty) {
        return f64::NAN;
    }
    if t.rem_euclid(TAU) < duty * TAU {
        1.
    } else {
        -1.
    }
}

/// Instantaneous-frequency law for [`chirp`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChirpMethod {
    /// `f(t) = f0 + (f1 - f0)t/t1`.
    Linear,
    /// `f(t) = f0 + (f1 - f0)(t/t1)^2`, with vertex at zero.
    Quadratic,
    /// `f(t) = f0 (f1/f0)^(t/t1)`.
    Logarithmic,
    /// `f(t) = f0 f1 t1 / ((f0 - f1)t + f1 t1)`.
    Hyperbolic,
}

/// Evaluates `cos(2π integral_0^t f(u) du + phase π/180)`.
///
/// ## Parameters
///
/// `pub fn chirp(t: f64, f0: f64, t1: f64, f1: f64, method: ChirpMethod, phase: f64) -> Result<f64>`
///
/// The method selects the frequency law documented on ChirpMethod; frequencies are cycles/time, t and t1 have matching time units, phase is degrees. Quadratic chirps use vertex_zero=true; output is real. Returns a scalar.
///
/// ## Domain
///
/// Fallible functions reject NaN and infinity in inputs. Input slices may be empty only where the definition is meaningful.
///
/// ## Returns
///
/// Output ordering, units and shapes follow the conventions above.
///
/// ## Errors
///
/// Returns the shared Error for invalid domains or shapes, singular evaluation, or nonfinite results. Sample vectors are limited to 1048576 entries and direct numerical work to 16777216 operations; smaller algorithm limits above take precedence.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic. Cancellation and poorly conditioned filters or polynomial fits can reduce accuracy; no uniform relative-error guarantee is made. FFT routines use RustFFT, polynomial fits and DPSS use bounded private nalgebra solvers.
///
/// # Examples
///
/// ```
/// use sci_signal::waveforms::*;
/// let _result = chirp(0.,1.,1.,2.,ChirpMethod::Linear,0.)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy chirp definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.chirp.html).
pub fn chirp(t: f64, f0: f64, t1: f64, f1: f64, method: ChirpMethod, phase: f64) -> Result<f64> {
    if ![t, f0, t1, f1, phase].iter().all(|v| v.is_finite()) || t1 == 0. {
        return Err(Error::InvalidInput(
            "chirp parameters must be finite and t1 nonzero",
        ));
    }
    let cycles = match method {
        ChirpMethod::Linear => f0 * t + (f1 - f0) * t * t / (2. * t1),
        ChirpMethod::Quadratic => f0 * t + (f1 - f0) * t.powi(3) / (3. * t1 * t1),
        ChirpMethod::Logarithmic => {
            if f0 == 0. || f1 == 0. || f0.is_sign_positive() != f1.is_sign_positive() {
                return Err(Error::InvalidInput(
                    "logarithmic frequencies must have the same nonzero sign",
                ));
            }
            if f0 == f1 {
                f0 * t
            } else {
                let a = (f1 / f0).ln() / t1;
                f0 * (a * t).exp_m1() / a
            }
        }
        ChirpMethod::Hyperbolic => {
            if f0 == 0. || f1 == 0. {
                return Err(Error::InvalidInput(
                    "hyperbolic frequencies must be nonzero",
                ));
            }
            if f0 == f1 {
                f0 * t
            } else {
                let singular = -f1 * t1 / (f0 - f1);
                -singular * f0 * (1. - t / singular).abs().ln()
            }
        }
    };
    match (TAU * cycles + phase * PI / 180.).cos() {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}

/// Evaluates `cos(2π integral_0^t P(u) du + phase π/180)`.
///
/// ## Parameters
///
/// `pub fn sweep_poly(t: f64, coefficients: &[f64], phase: f64) -> Result<f64>`
///
/// coefficients specify the frequency polynomial in descending powers, frequencies in cycles/time, phase in degrees. Returns a scalar.
///
/// ## Domain
///
/// Fallible functions reject NaN and infinity in inputs. Input slices may be empty only where the definition is meaningful.
///
/// ## Returns
///
/// Output ordering, units and shapes follow the conventions above.
///
/// ## Errors
///
/// Returns the shared Error for invalid domains or shapes, singular evaluation, or nonfinite results. Sample vectors are limited to 1048576 entries and direct numerical work to 16777216 operations; smaller algorithm limits above take precedence.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic. Cancellation and poorly conditioned filters or polynomial fits can reduce accuracy; no uniform relative-error guarantee is made. FFT routines use RustFFT, polynomial fits and DPSS use bounded private nalgebra solvers.
///
/// # Examples
///
/// ```
/// use sci_signal::waveforms::*;
/// let _result = sweep_poly(0.,&[1.,2.],0.)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy sweep_poly definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sweep_poly.html).
pub fn sweep_poly(t: f64, coefficients: &[f64], phase: f64) -> Result<f64> {
    nonempty(coefficients)?;
    if !t.is_finite() || !phase.is_finite() {
        return Err(Error::InvalidInput("time and phase must be finite"));
    }
    let n = coefficients.len();
    let integral = coefficients
        .iter()
        .enumerate()
        .fold(0., |v, (i, &c)| v * t + c / (n - i) as f64)
        * t;
    match (TAU * integral + phase * PI / 180.).cos() {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}

/// Evaluates `exp(-a t²) exp(2π i fc t)`, with `a=-(π fc bandwidth)²/(4 ln(10^(reference_db/20)))`.
///
/// ## Parameters
///
/// `pub fn gausspulse(t: f64, fc: f64, bandwidth: f64, reference_db: f64) -> Result<(f64,f64,f64)>`
///
/// Time is in seconds, carrier fc in Hz, bandwidth is fractional bandwidth, reference_db is a negative amplitude level. Returns (in-phase, quadrature, envelope); cutoff-time dispatch is unsupported.
///
/// ## Domain
///
/// Fallible functions reject NaN and infinity in inputs. Input slices may be empty only where the definition is meaningful.
///
/// ## Returns
///
/// Output ordering, units and shapes follow the conventions above.
///
/// ## Errors
///
/// Returns the shared Error for invalid domains or shapes, singular evaluation, or nonfinite results. Sample vectors are limited to 1048576 entries and direct numerical work to 16777216 operations; smaller algorithm limits above take precedence.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic. Cancellation and poorly conditioned filters or polynomial fits can reduce accuracy; no uniform relative-error guarantee is made. FFT routines use RustFFT, polynomial fits and DPSS use bounded private nalgebra solvers.
///
/// # Examples
///
/// ```
/// use sci_signal::waveforms::*;
/// let _result = gausspulse(0.,5.,0.5,-6.)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy gausspulse definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.gausspulse.html).
pub fn gausspulse(t: f64, fc: f64, bandwidth: f64, reference_db: f64) -> Result<(f64, f64, f64)> {
    if !t.is_finite()
        || !fc.is_finite()
        || fc < 0.
        || !reference_db.is_finite()
        || reference_db >= 0.
    {
        return Err(Error::InvalidInput(
            "finite time, nonnegative carrier and negative reference dB required",
        ));
    }
    positive(bandwidth)?;
    let a = -(PI * fc * bandwidth).powi(2) / (4. * (10_f64.powf(reference_db / 20.)).ln());
    let envelope = (-a * t * t).exp();
    let (s, c) = (TAU * fc * t).sin_cos();
    if !envelope.is_finite() || !s.is_finite() || !c.is_finite() {
        return Err(Error::NonFiniteEvaluation);
    }
    Ok((envelope * c, envelope * s, envelope))
}

/// Evaluates the Gaussian B-spline approximation `exp(-x²/(2σ²))/sqrt(2πσ²)`, `σ²=(order+1)/12`.
///
/// ## Parameters
///
/// `pub fn gauss_spline(x: f64, order: usize) -> f64`
///
/// x is in knot-spacing units; order is a nonnegative integer. Returns a scalar.
///
/// ## Domain
///
/// NaN propagates; infinities follow the documented equation.
///
/// ## Returns
///
/// Output ordering, units and shapes follow the conventions above.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic. Cancellation and poorly conditioned filters or polynomial fits can reduce accuracy; no uniform relative-error guarantee is made. FFT routines use RustFFT, polynomial fits and DPSS use bounded private nalgebra solvers.
///
/// # Examples
///
/// ```
/// use sci_signal::waveforms::*;
/// let _result = gauss_spline(0.,3);
/// ```
///
/// # References
///
/// [SciPy gauss_spline definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.gauss_spline.html).
pub fn gauss_spline(x: f64, order: usize) -> f64 {
    let variance = (order as f64 + 1.) / 12.;
    (-x * x / (2. * variance)).exp() / (TAU * variance).sqrt()
}

/// Returns the Kronecker impulse `x[i] = 1` when `i=index`, and zero otherwise.
///
/// ## Parameters
///
/// `pub fn unit_impulse(length: usize, index: usize) -> Result<Vec<f64>>`
///
/// One-dimensional length and explicit zero-based index replace SciPy shape and midpoint dispatch. Returns length samples.
///
/// ## Domain
///
/// Fallible functions reject NaN and infinity in inputs. Input slices may be empty only where the definition is meaningful.
///
/// ## Returns
///
/// Output ordering, units and shapes follow the conventions above.
///
/// ## Errors
///
/// Returns the shared Error for invalid domains or shapes, singular evaluation, or nonfinite results. Sample vectors are limited to 1048576 entries and direct numerical work to 16777216 operations; smaller algorithm limits above take precedence.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic. Cancellation and poorly conditioned filters or polynomial fits can reduce accuracy; no uniform relative-error guarantee is made. FFT routines use RustFFT, polynomial fits and DPSS use bounded private nalgebra solvers.
///
/// # Examples
///
/// ```
/// use sci_signal::waveforms::*;
/// let _result = unit_impulse(3,1)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy unit_impulse definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.unit_impulse.html).
pub fn unit_impulse(length: usize, index: usize) -> Result<Vec<f64>> {
    size(length)?;
    if index >= length {
        return Err(Error::InvalidInput("impulse index out of range"));
    }
    let mut result = vec![0.; length];
    result[index] = 1.;
    Ok(result)
}

/// Computes `z = mean(exp(2π i events/period))` and returns `(|z|, arg(z))`.
///
/// ## Parameters
///
/// `pub fn vectorstrength(events: &[f64], period: f64) -> Result<(f64,f64)>`
///
/// Events and positive scalar period share time units. Phase is in radians in `[-π,π]`; undefined zero-vector phase follows atan2. Multiple periods require separate calls.
///
/// ## Domain
///
/// Fallible functions reject NaN and infinity in inputs. Input slices may be empty only where the definition is meaningful.
///
/// ## Returns
///
/// Output ordering, units and shapes follow the conventions above.
///
/// ## Errors
///
/// Returns the shared Error for invalid domains or shapes, singular evaluation, or nonfinite results. Sample vectors are limited to 1048576 entries and direct numerical work to 16777216 operations; smaller algorithm limits above take precedence.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic. Cancellation and poorly conditioned filters or polynomial fits can reduce accuracy; no uniform relative-error guarantee is made. FFT routines use RustFFT, polynomial fits and DPSS use bounded private nalgebra solvers.
///
/// # Examples
///
/// ```
/// use sci_signal::waveforms::*;
/// let _result = vectorstrength(&[0.,1.],2.)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy vectorstrength definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.vectorstrength.html).
pub fn vectorstrength(events: &[f64], period: f64) -> Result<(f64, f64)> {
    nonempty(events)?;
    positive(period)?;
    let (re, im) = events.iter().fold((0., 0.), |(re, im), &t| {
        let (s, c) = (TAU * (t.rem_euclid(period) / period)).sin_cos();
        (re + c, im + s)
    });
    Ok((re.hypot(im) / events.len() as f64, im.atan2(re)))
}

/// Subtracts the least-squares line `a+b i` or the sample mean.
///
/// ## Parameters
///
/// `pub fn detrend(x: &[f64], linear: bool) -> Result<Vec<f64>>`
///
/// linear=true removes a line; false removes a constant. Returns the input shape. No breakpoint or multidimensional dispatch. A singleton becomes zero.
///
/// ## Domain
///
/// Fallible functions reject NaN and infinity in inputs. Input slices may be empty only where the definition is meaningful.
///
/// ## Returns
///
/// Output ordering, units and shapes follow the conventions above.
///
/// ## Errors
///
/// Returns the shared Error for invalid domains or shapes, singular evaluation, or nonfinite results. Sample vectors are limited to 1048576 entries and direct numerical work to 16777216 operations; smaller algorithm limits above take precedence.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic. Cancellation and poorly conditioned filters or polynomial fits can reduce accuracy; no uniform relative-error guarantee is made. FFT routines use RustFFT, polynomial fits and DPSS use bounded private nalgebra solvers.
///
/// # Examples
///
/// ```
/// use sci_signal::waveforms::*;
/// let _result = detrend(&[1.,2.,3.],true)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy detrend definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.detrend.html).
pub fn detrend(x: &[f64], linear: bool) -> Result<Vec<f64>> {
    samples(x)?;
    if x.is_empty() {
        return Ok(vec![]);
    }
    let n = x.len() as f64;
    let center = (n - 1.) / 2.;
    let mean = x.iter().map(|v| v / n).sum::<f64>();
    let slope = if linear && x.len() > 1 {
        let numerator = x
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as f64 - center) * (v - mean))
            .sum::<f64>();
        numerator / (n * (n * n - 1.) / 12.)
    } else {
        0.
    };
    checked(
        x.iter()
            .enumerate()
            .map(|(i, &v)| v - mean - slope * (i as f64 - center))
            .collect(),
    )
}
