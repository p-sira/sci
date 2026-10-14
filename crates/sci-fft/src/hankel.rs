//! FFTLog composition using private XSF gamma and RustFFT backends.
//!
//! Coefficient and grid adaptations follow SciPy; license retained in
//! `LICENSES/scipy-fftlog.txt`.
use super::{irfft, length, rfft};
use crate::error::{finite, finite_slice};
use crate::{
    Error, Result,
    complex::{native, owned},
};
use std::f64::consts::{LN_2, PI};

fn parameters(dln: f64, mu: f64, offset: f64, bias: f64) -> Result<(f64, f64)> {
    finite_slice(&[dln, mu, offset, bias])?;
    if dln == 0.0 {
        return Err(Error::InvalidInput("logarithmic spacing must be nonzero"));
    }
    Ok((
        finite((mu + 1.0 + bias) / 2.0)?,
        finite((mu + 1.0 - bias) / 2.0)?,
    ))
}
fn gamma(x: f64, y: f64) -> Result<num_complex::Complex64> {
    let z = xsf::loggamma(num_complex::Complex64::new(x, y));
    finite(z.re)?;
    finite(z.im)?;
    Ok(z)
}
fn scale(value: f64, exponent: f64) -> Result<f64> {
    let factor = finite(finite(exponent)?.exp())?;
    if factor == 0.0 {
        return Err(Error::NumericalFailure("FFTLog bias scaling underflow"));
    }
    finite(value * factor)
}
fn transform(
    input: &[f64],
    dln: f64,
    mu: f64,
    offset: f64,
    bias: f64,
    inverse: bool,
) -> Result<Vec<f64>> {
    length(input.len())?;
    finite_slice(input)?;
    let (xp, xm) = parameters(dln, mu, offset, bias)?;
    let n = input.len();
    let center = (n as f64 - 1.0) / 2.0;
    let mut biased = Vec::with_capacity(n);
    for (j, &v) in input.iter().enumerate() {
        let grid = (j as f64 - center) * dln;
        biased.push(if bias == 0.0 {
            v
        } else {
            scale(
                v,
                if inverse {
                    bias * (grid + offset)
                } else {
                    -bias * grid
                },
            )?
        });
    }
    let mut spectrum = rfft(&biased);
    for (k, v) in spectrum.iter_mut().enumerate() {
        let mut coefficient = if k == 0 {
            let value = finite((LN_2 * bias).exp() * xsf::pow_rising(xm, xp - xm))?;
            if value == 0.0 {
                return Err(Error::NumericalFailure("singular FFTLog coefficients"));
            }
            num_complex::Complex64::new(value, 0.0)
        } else {
            let y = finite((PI * k as f64 / n as f64) / dln)?;
            let zp = gamma(xp, y)?;
            let zm = gamma(xm, y)?;
            let log = num_complex::Complex64::new(
                finite(zp.re - zm.re + LN_2 * bias)?,
                finite(zp.im + zm.im + 2.0 * y * (LN_2 - offset))?,
            );
            let c = log.exp();
            finite(c.re)?;
            finite(c.im)?;
            if c.norm() == 0.0 {
                return Err(Error::NumericalFailure("FFTLog coefficient underflow"));
            }
            c
        };
        if n.is_multiple_of(2) && k == n / 2 {
            coefficient.im = 0.0;
        }
        if inverse && coefficient.re == 0.0 && coefficient.im == 0.0 {
            return Err(Error::NumericalFailure(
                "singular inverse FFTLog coefficients",
            ));
        }
        let result = if inverse {
            native(*v) / coefficient.conj()
        } else {
            native(*v) * coefficient
        };
        finite(result.re)?;
        finite(result.im)?;
        *v = owned(result);
    }
    let mut out = irfft(&spectrum, Some(n))?;
    out.reverse();
    if bias != 0.0 {
        for (j, v) in out.iter_mut().enumerate() {
            let grid = (j as f64 - center) * dln;
            *v = scale(
                *v,
                if inverse {
                    bias * grid
                } else {
                    -bias * (grid + offset)
                },
            )?;
        }
    }
    Ok(out)
}

/// Approximates `A(k)=∫₀∞ a(r) J_μ(kr) k dr` using the periodic FFTLog discretization.
///
/// ## Parameters
///
/// `input[j]=a(r_c exp((j−(n−1)/2)dln))`. Output grids are
/// `k[j]=exp(offset)/r[n−1−j]`, where `offset=ln(k_c r_c)` and `bias` is the
/// power-law exponent used to reduce periodic endpoint mismatch. The integral
/// uses k dr, following SciPy; it differs from the common r dr convention.
/// Captures one sequence; SciPy leading-axis batching is omitted.
/// ## Domain
///
/// Finite real input with 1..=1,048,576 entries; finite nonzero logarithmic
/// spacing, finite real order, offset, and bias. Negative spacing describes a
/// descending grid. Nonfinite values, singular DC coefficients, coefficient
/// underflow, and underflow/overflow in bias factors are rejected.
/// ## Returns
///
/// A same-length real sequence in ascending input-index order on the output grid.
/// ## Errors
///
/// Invalid inputs/work limits, singular gamma ratios, or nonfinite arithmetic.
/// SciPy's singular-transform warning and coefficient replacement are adapted
/// to explicit shared errors.
/// ## Numerical behavior
///
/// Private RustFFT and XSF backends perform O(n log n) Fourier work and O(n)
/// complex log-gamma evaluations. Gamma coefficients use the principal log-gamma
/// branch and the SciPy FFTLog convention. Even Nyquist coefficients are projected
/// onto the real axis. Periodic extension, logarithmic discretization, endpoint
/// ringing, and f64 roundoff limit continuous-integral accuracy. This is not an
/// adaptive quadrature and supplies no integral error estimate.
/// # Examples
///
/// ```
/// let offset=sci_fft::fhtoffset(0.1,0.0,0.0,0.0)?;
/// let y=sci_fft::fht(&[1.0,2.0,3.0,4.0],0.1,0.0,offset,0.0)?;
/// assert_eq!(y.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy fht](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fht.html);
/// Hamilton (2000), [FFTLog](https://doi.org/10.1046/j.1365-8711.2000.03171.x).
pub fn fht(input: &[f64], dln: f64, mu: f64, offset: f64, bias: f64) -> Result<Vec<f64>> {
    transform(input, dln, mu, offset, bias, false)
}

/// Inverts the periodic FFTLog mapping of [`fht`]: `a = FHT⁻¹(A)`.
///
/// ## Parameters
///
/// Input is the output-grid sequence of [`fht`]; dln, order, offset, and bias must
/// match that transform. Uses reciprocal conjugate coefficients and inverse bias
/// factors; output follows the original r-grid ordering. No batching is provided.
/// ## Domain
///
/// The finite input, work, gamma, and scaling domain of [`fht`]. Zero inverse
/// coefficients, including a zero real Nyquist coefficient, return an error.
/// ## Returns
///
/// A same-length real approximation to the original input sequence.
/// ## Errors
///
/// Invalid inputs, singular coefficients, underflowed bias factors, or overflow.
/// ## Numerical behavior
///
/// O(n log n) work through shared private backends. Small coefficients and large
/// bias factors amplify roundoff; discretization and periodic ringing remain.
/// No continuous-integral accuracy estimate or convergence control is supplied.
/// # Examples
///
/// ```
/// let offset=sci_fft::fhtoffset(0.1,0.0,0.0,0.0)?;
/// let y=sci_fft::fht(&[1.0,2.0,3.0,4.0],0.1,0.0,offset,0.0)?;
/// let x=sci_fft::ifht(&y,0.1,0.0,offset,0.0)?;
/// assert!((x[0]-1.0).abs()<1e-12);
/// # Ok::<(),sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy ifht](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.ifht.html);
/// Hamilton (2000), [FFTLog](https://doi.org/10.1046/j.1365-8711.2000.03171.x).
pub fn ifht(input: &[f64], dln: f64, mu: f64, offset: f64, bias: f64) -> Result<Vec<f64>> {
    transform(input, dln, mu, offset, bias, true)
}

/// Chooses the nearest FFTLog offset satisfying the low-ringing phase condition.
///
/// ## Parameters
///
/// `dln`, `mu`, and `bias` follow [`fht`]. `initial` is the desired log(k_c r_c).
/// With `y=π/(2dln)`, `x±=(μ+1±bias)/2`, define
/// `phase=(ln(2)−initial)/dln+(Im logΓ(x++iy)+Im logΓ(x−+iy))/π`.
/// Returns `initial+(phase−round_ties_even(phase))*dln`.
/// ## Domain
///
/// Finite arguments and nonzero dln; finite complex log-gamma values are required.
/// NaN/infinity and overflowed phase arithmetic are rejected.
/// ## Returns
///
/// A logarithmic grid offset within |dln|/2 of initial, up to f64 roundoff.
/// ## Errors
///
/// Invalid inputs, gamma evaluation failure, or nonfinite arithmetic.
/// ## Numerical behavior
///
/// Two private XSF complex log-gamma evaluations. Reduces even-length Nyquist
/// phase mismatch; it does not remove all ringing or guarantee integral accuracy.
/// Extreme phases can lose fractional bits before rounding.
/// # Examples
///
/// ```
/// let offset=sci_fft::fhtoffset(0.1,2.0,0.5,0.0)?;
/// assert!((offset-0.5454581477676637).abs()<1e-12);
/// # Ok::<(),sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy fhtoffset](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fhtoffset.html);
/// Hamilton (2000), [FFTLog](https://doi.org/10.1046/j.1365-8711.2000.03171.x).
pub fn fhtoffset(dln: f64, mu: f64, initial: f64, bias: f64) -> Result<f64> {
    let (xp, xm) = parameters(dln, mu, initial, bias)?;
    let y = finite((PI / 2.0) / dln)?;
    let zp = gamma(xp, y)?;
    let zm = gamma(xm, y)?;
    let phase = finite((LN_2 - initial) / dln + (zp.im + zm.im) / PI)?;
    finite(initial + (phase - phase.round_ties_even()) * dln)
}
