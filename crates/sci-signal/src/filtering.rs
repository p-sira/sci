use super::{
    Mode, checked, nonempty, positive, samples, size,
    windows::{self, Window},
    work,
};
use crate::{
    Error, Result,
    complex::{Complex, native, owned},
};
use num_complex::Complex64;
use std::f64::consts::{PI, TAU};

/// Numerator and denominator coefficient vectors, in that order.
pub type TransferFunction = (Vec<f64>, Vec<f64>);

fn coefficients(b: &[f64], a: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
    nonempty(b)?;
    nonempty(a)?;
    if a[0] == 0. {
        return Err(Error::InvalidInput("a[0] must be nonzero"));
    }
    let n = a.len().max(b.len());
    let mut bn = vec![0.; n];
    let mut an = vec![0.; n];
    for (out, &v) in bn.iter_mut().zip(b) {
        *out = v / a[0];
    }
    for (out, &v) in an.iter_mut().zip(a) {
        *out = v / a[0];
    }
    Ok((checked(bn)?, checked(an)?))
}

/// Applies `sum_j a[j] y[k-j] = sum_j b[j] x[k-j]` with zero initial state.
///
/// ## Parameters
///
/// `pub fn lfilter(b:&[f64],a:&[f64],x:&[f64])->Result<Vec<f64>>`
///
/// Coefficients are ascending delays z^-j, normalized by a`[0]`. Uses transposed direct form II. Returns one sample per input.
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
/// use sci_signal::filtering::*;
/// let _result = lfilter(&[0.5,0.5],&[1.],&[1.,2.])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy lfilter definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html).
pub fn lfilter(b: &[f64], a: &[f64], x: &[f64]) -> Result<Vec<f64>> {
    let (b, a) = coefficients(b, a)?;
    Ok(filter_state(&b, &a, x, &vec![0.; b.len() - 1])?.0)
}

/// Applies the lfilter recurrence with explicit transposed direct-form II state.
///
/// ## Parameters
///
/// `pub fn lfilter_with_state(b:&[f64],a:&[f64],x:&[f64],initial:&[f64])->Result<(Vec<f64>,Vec<f64>)>`
///
/// Initial state has max(len(a),len(b))-1 values. Returns (output samples, final state), suitable for streaming consecutive chunks.
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
/// use sci_signal::filtering::*;
/// let _result = lfilter_with_state(&[0.5,0.5],&[1.],&[1.,2.],&[0.])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy lfilter_with_state definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html).
pub fn lfilter_with_state(
    b: &[f64],
    a: &[f64],
    x: &[f64],
    initial: &[f64],
) -> Result<(Vec<f64>, Vec<f64>)> {
    let (b, a) = coefficients(b, a)?;
    filter_state(&b, &a, x, initial)
}
fn filter_state(b: &[f64], a: &[f64], x: &[f64], initial: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
    samples(x)?;
    samples(initial)?;
    work(x.len(), b.len())?;
    if initial.len() != b.len() - 1 {
        return Err(Error::DimensionMismatch);
    }
    let mut state = initial.to_vec();
    let mut output = Vec::with_capacity(x.len());
    for &v in x {
        let y = b[0] * v + state.first().copied().unwrap_or(0.);
        for j in 0..state.len() {
            state[j] = state.get(j + 1).copied().unwrap_or(0.) + b[j + 1] * v - a[j + 1] * y;
        }
        output.push(y);
    }
    Ok((checked(output)?, checked(state)?))
}

/// Constructs state satisfying `zi = A zi + B` for a unit step.
///
/// ## Parameters
///
/// `pub fn lfilter_zi(b:&[f64],a:&[f64])->Result<Vec<f64>>`
///
/// Coefficients are ascending delays, normalized by a`[0]`. Returns max(len(a),len(b))-1 state values. A pole at z=1 is an error; no stability guarantee is made.
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
/// use sci_signal::filtering::*;
/// let _result = lfilter_zi(&[0.5,0.5],&[1.])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy lfilter_zi definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter_zi.html).
pub fn lfilter_zi(b: &[f64], a: &[f64]) -> Result<Vec<f64>> {
    let (b, a) = coefficients(b, a)?;
    let denominator = match a.iter().sum::<f64>() {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }?;
    if denominator == 0. {
        return Err(Error::NumericalFailure("filter has a pole at DC"));
    }
    let gain = match match b.iter().sum::<f64>() {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }? / denominator
    {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }?;
    let mut state = vec![0.; b.len() - 1];
    let mut tail = 0.;
    for j in (1..b.len()).rev() {
        tail += b[j] - a[j] * gain;
        state[j - 1] = tail;
    }
    checked(state)
}

/// Constructs state `zi[m] = sum_j (b[m+j+1] past_x[j] - a[m+j+1] past_y[j])`.
///
/// ## Parameters
///
/// `pub fn lfiltic(b:&[f64],a:&[f64],past_y:&[f64],past_x:&[f64])->Result<Vec<f64>>`
///
/// Coefficients are normalized by a`[0]`. History is newest first; missing history is zero padded. Returns transposed direct-form II state.
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
/// use sci_signal::filtering::*;
/// let _result = lfiltic(&[0.5,0.5],&[1.],&[1.],&[2.])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy lfiltic definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfiltic.html).
pub fn lfiltic(b: &[f64], a: &[f64], past_y: &[f64], past_x: &[f64]) -> Result<Vec<f64>> {
    let (b, a) = coefficients(b, a)?;
    samples(past_y)?;
    samples(past_x)?;
    work(b.len(), b.len())?;
    let mut state = vec![0.; b.len() - 1];
    for (m, out) in state.iter_mut().enumerate() {
        for j in 0..b.len() - m - 1 {
            *out += b[m + j + 1] * past_x.get(j).copied().unwrap_or(0.)
                - a[m + j + 1] * past_y.get(j).copied().unwrap_or(0.);
        }
    }
    checked(state)
}

/// Applies a filter forward and backward, yielding magnitude response `|H(exp(iω))|²`.
///
/// ## Parameters
///
/// `pub fn filtfilt(b:&[f64],a:&[f64],x:&[f64],padlen:usize)->Result<Vec<f64>>`
///
/// Uses odd reflected padding of explicit padlen and step-steady initial states. Returns input length; padlen=0 disables extension. Gustafsson and other padding methods are unsupported.
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
/// use sci_signal::filtering::*;
/// let _result = filtfilt(&[0.5,0.5],&[1.],&[1.,2.,3.,4.],1)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy filtfilt definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html).
pub fn filtfilt(b: &[f64], a: &[f64], x: &[f64], padlen: usize) -> Result<Vec<f64>> {
    nonempty(x)?;
    size(padlen)?;
    if padlen >= x.len() {
        return Err(Error::InvalidInput(
            "padlen must be smaller than signal length",
        ));
    }
    let len = x.len() + 2 * padlen;
    size(len)?;
    let zi = lfilter_zi(b, a)?;
    let mut extended = Vec::with_capacity(len);
    for i in (1..=padlen).rev() {
        extended.push(2. * x[0] - x[i]);
    }
    extended.extend_from_slice(x);
    for i in 1..=padlen {
        extended.push(2. * x[x.len() - 1] - x[x.len() - 1 - i]);
    }
    let initial: Vec<_> = zi.iter().map(|v| v * extended[0]).collect();
    let mut forward = lfilter_with_state(b, a, &extended, &initial)?.0;
    forward.reverse();
    let initial: Vec<_> = zi.iter().map(|v| v * forward[0]).collect();
    let mut backward = lfilter_with_state(b, a, &forward, &initial)?.0;
    backward.reverse();
    Ok(backward[padlen..padlen + x.len()].to_vec())
}

fn validate_sos(sos: &[[f64; 6]]) -> Result<()> {
    if sos.is_empty() {
        return Err(Error::InvalidInput("SOS must not be empty"));
    }
    size(sos.len())?;
    if sos
        .iter()
        .any(|row| row[3] != 1. || row.iter().any(|v| !v.is_finite()))
    {
        return Err(Error::InvalidInput(
            "SOS coefficients must be finite with a0 = 1",
        ));
    }
    Ok(())
}
/// Applies a cascade `H(z)=product_s B_s(z)/A_s(z)` from zero state.
///
/// ## Parameters
///
/// `pub fn sosfilt(sos:&[[f64;6]],x:&[f64])->Result<Vec<f64>>`
///
/// Each row is `[b0,b1,b2,a0,a1,a2]`, with a0=1. Returns input length. Explicit SOS streaming state is unsupported; lfilter_with_state supports individual sections.
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
/// use sci_signal::filtering::*;
/// let _result = sosfilt(&[[1.,0.,0.,1.,0.,0.]],&[1.,2.])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy sosfilt definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html).
pub fn sosfilt(sos: &[[f64; 6]], x: &[f64]) -> Result<Vec<f64>> {
    validate_sos(sos)?;
    samples(x)?;
    work(sos.len(), x.len())?;
    sos.iter()
        .try_fold(x.to_vec(), |x, row| lfilter(&row[..3], &row[3..], &x))
}
/// Constructs each section unit-step state, scaled by all preceding sections DC gains.
///
/// ## Parameters
///
/// `pub fn sosfilt_zi(sos:&[[f64;6]])->Result<Vec<[f64;2]>>`
///
/// Returns one `[z0,z1]` state per `[b0,b1,b2,1,a1,a2]` section. A DC pole is an error.
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
/// use sci_signal::filtering::*;
/// let _result = sosfilt_zi(&[[1.,0.,0.,1.,0.,0.]])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy sosfilt_zi definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt_zi.html).
pub fn sosfilt_zi(sos: &[[f64; 6]]) -> Result<Vec<[f64; 2]>> {
    validate_sos(sos)?;
    let mut scale = 1.;
    let mut result = Vec::with_capacity(sos.len());
    for row in sos {
        let zi = lfilter_zi(&row[..3], &row[3..])?;
        let state = [zi[0] * scale, zi[1] * scale];
        if !state.iter().all(|v| v.is_finite()) {
            return Err(Error::NonFiniteEvaluation);
        }
        result.push(state);
        scale *= row[..3].iter().sum::<f64>() / row[3..].iter().sum::<f64>();
    }
    Ok(result)
}
/// Applies the SOS cascade forward and backward, giving squared magnitude and zero phase.
///
/// ## Parameters
///
/// `pub fn sosfiltfilt(sos:&[[f64;6]],x:&[f64],padlen:usize)->Result<Vec<f64>>`
///
/// Rows are `[b0,b1,b2,1,a1,a2]`. Uses explicit odd reflected padlen and scaled unit-step states. Returns input length.
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
/// use sci_signal::filtering::*;
/// let _result = sosfiltfilt(&[[1.,0.,0.,1.,0.,0.]],&[1.,2.,3.],1)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy sosfiltfilt definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html).
pub fn sosfiltfilt(sos: &[[f64; 6]], x: &[f64], padlen: usize) -> Result<Vec<f64>> {
    validate_sos(sos)?;
    nonempty(x)?;
    if padlen >= x.len() {
        return Err(Error::InvalidInput(
            "padlen must be smaller than signal length",
        ));
    }
    size(x.len() + 2 * padlen)?;
    work(sos.len(), x.len() + 2 * padlen)?;
    let zi = sosfilt_zi(sos)?;
    let apply = |mut input: Vec<f64>| -> Result<Vec<f64>> {
        let scale = input[0];
        for (row, state) in sos.iter().zip(&zi) {
            input = lfilter_with_state(
                &row[..3],
                &row[3..],
                &input,
                &[state[0] * scale, state[1] * scale],
            )?
            .0;
        }
        Ok(input)
    };
    let mut extended = Vec::new();
    for i in (1..=padlen).rev() {
        extended.push(2. * x[0] - x[i]);
    }
    extended.extend_from_slice(x);
    for i in 1..=padlen {
        extended.push(2. * x[x.len() - 1] - x[x.len() - 1 - i]);
    }
    let mut y = apply(extended)?;
    y.reverse();
    let mut y = apply(y)?;
    y.reverse();
    Ok(y[padlen..padlen + x.len()].to_vec())
}
/// Expands `B(z)=product B_s(z)` and `A(z)=product A_s(z)`.
///
/// ## Parameters
///
/// `pub fn sos2tf(sos:&[[f64;6]])->Result<TransferFunction>`
///
/// Rows are `[b0,b1,b2,1,a1,a2]`. Returns (numerator, denominator), each length 2*sections+1 in ascending delay order.
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
/// use sci_signal::filtering::*;
/// let _result = sos2tf(&[[1.,0.,0.,1.,0.,0.]])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy sos2tf definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sos2tf.html).
pub fn sos2tf(sos: &[[f64; 6]]) -> Result<TransferFunction> {
    validate_sos(sos)?;
    work(sos.len(), sos.len() * 6)?;
    let mut b = vec![1.];
    let mut a = vec![1.];
    for row in sos {
        b = super::convolve(&b, &row[..3], Mode::Full)?;
        a = super::convolve(&a, &row[3..], Mode::Full)?;
    }
    Ok((b, a))
}
fn response(b: &[f64], a: &[f64], frequencies: &[f64], digital: bool) -> Result<Vec<Complex>> {
    nonempty(b)?;
    nonempty(a)?;
    samples(frequencies)?;
    work(frequencies.len(), a.len() + b.len())?;
    if a.iter().all(|v| *v == 0.) {
        return Err(Error::InvalidInput("denominator cannot be zero"));
    }
    let poly = |c: &[f64], z: Complex64| c.iter().fold(Complex64::new(0., 0.), |v, &a| v * z + a);
    let output = frequencies
        .iter()
        .map(|&w| {
            let z = if digital {
                Complex64::from_polar(1., -w)
            } else {
                Complex64::new(0., w)
            };
            let (num, den) = if digital {
                (
                    poly(&b.iter().rev().copied().collect::<Vec<_>>(), z),
                    poly(&a.iter().rev().copied().collect::<Vec<_>>(), z),
                )
            } else {
                (poly(b, z), poly(a, z))
            };
            owned(num / den)
        })
        .collect();
    super::complex_checked(output)
}
/// Evaluates `H(ω) = sum_j b[j] exp(-iωj) / sum_j a[j] exp(-iωj)`.
///
/// ## Parameters
///
/// `pub fn freqz(b:&[f64],a:&[f64],frequencies:&[f64])->Result<Vec<Complex>>`
///
/// frequencies are explicit angular frequencies in radians/sample, preserved in caller order. Returns only the complex responses; count-based grids, batching, and plotting are unsupported.
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
/// use sci_signal::filtering::*;
/// let _result = freqz(&[1.],&[1.],&[0.,1.])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy freqz definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqz.html).
pub fn freqz(b: &[f64], a: &[f64], frequencies: &[f64]) -> Result<Vec<Complex>> {
    response(b, a, frequencies, true)
}
/// Evaluates `H(ω) = B(iω)/A(iω)`.
///
/// ## Parameters
///
/// `pub fn freqs(b:&[f64],a:&[f64],frequencies:&[f64])->Result<Vec<Complex>>`
///
/// Coefficients are descending powers of the Laplace variable; angular frequencies are radians/time. Returns complex responses in caller frequency order. Automatic grids are unsupported.
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
/// use sci_signal::filtering::*;
/// let _result = freqs(&[1.],&[1.,1.],&[0.,1.])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy freqs definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqs.html).
pub fn freqs(b: &[f64], a: &[f64], frequencies: &[f64]) -> Result<Vec<Complex>> {
    response(b, a, frequencies, false)
}
/// Evaluates `H(ω) = product_s B_s(exp(iω))/A_s(exp(iω))`.
///
/// ## Parameters
///
/// `pub fn freqz_sos(sos:&[[f64;6]],frequencies:&[f64])->Result<Vec<Complex>>`
///
/// Rows are `[b0,b1,b2,1,a1,a2]`; frequencies are radians/sample. Returns complex responses in input frequency order.
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
/// use sci_signal::filtering::*;
/// let _result = freqz_sos(&[[1.,0.,0.,1.,0.,0.]],&[0.,1.])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy freqz_sos definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqz_sos.html).
pub fn freqz_sos(sos: &[[f64; 6]], frequencies: &[f64]) -> Result<Vec<Complex>> {
    validate_sos(sos)?;
    samples(frequencies)?;
    work(sos.len() * 6, frequencies.len())?;
    let mut output = vec![Complex64::new(1., 0.); frequencies.len()];
    for row in sos {
        for (value, h) in output
            .iter_mut()
            .zip(freqz(&row[..3], &row[3..], frequencies)?)
        {
            *value *= native(h);
        }
    }
    super::complex_checked(output.into_iter().map(owned).collect())
}

/// Computes `-d arg H(ω)/dω = Re(B1/B - A1/A)`, where `C1=sum_j j c[j] exp(-iωj)`.
///
/// ## Parameters
///
/// `pub fn group_delay(b:&[f64],a:&[f64],frequencies:&[f64])->Result<Vec<f64>>`
///
/// Coefficients are ascending delays and frequencies radians/sample. Returns delay in samples in caller frequency order. Singular frequencies return an error rather than SciPy warning-and-zero behavior.
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
/// use sci_signal::filtering::*;
/// let _result = group_delay(&[1.],&[1.],&[0.,1.])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy group_delay definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.group_delay.html).
pub fn group_delay(b: &[f64], a: &[f64], frequencies: &[f64]) -> Result<Vec<f64>> {
    nonempty(b)?;
    nonempty(a)?;
    samples(frequencies)?;
    work(frequencies.len(), a.len() + b.len())?;
    let moment = |c: &[f64], w: f64| {
        let (sum, weighted) = c.iter().enumerate().fold(
            (Complex64::new(0., 0.), Complex64::new(0., 0.)),
            |(s, m), (i, &v)| {
                let term = Complex64::from_polar(v, -w * i as f64);
                (s + term, m + term * i as f64)
            },
        );
        if sum.norm() == 0. {
            return Err(Error::NumericalFailure(
                "group delay undefined at a zero or pole",
            ));
        }
        Ok((weighted / sum).re)
    };
    checked(
        frequencies
            .iter()
            .map(|&w| Ok(moment(b, w)? - moment(a, w)?))
            .collect::<Result<_>>()?,
    )
}

fn notch_peak(frequency: f64, q: f64, fs: f64, peak: bool) -> Result<TransferFunction> {
    positive(fs)?;
    positive(q)?;
    if !frequency.is_finite() || frequency <= 0. || frequency >= fs / 2. {
        return Err(Error::InvalidInput(
            "frequency must lie between zero and Nyquist",
        ));
    }
    let w = TAU * frequency / fs;
    let bw = w / q;
    if bw >= PI {
        return Err(Error::InvalidInput("bandwidth must be less than Nyquist"));
    }
    let gain = 1. / (1. + (bw / 2.).tan());
    let b = if peak {
        vec![1. - gain, 0., gain - 1.]
    } else {
        vec![gain, -2. * gain * w.cos(), gain]
    };
    Ok((
        checked(b)?,
        checked(vec![1., -2. * gain * w.cos(), 2. * gain - 1.])?,
    ))
}
/// Designs a second-order notch with `g=1/(1+tan(ω0/(2Q)))`, `b=g[1,-2cos(ω0),1]`, `a=[1,-2g cos(ω0),2g-1]`.
///
/// ## Parameters
///
/// `pub fn iirnotch(frequency:f64,q:f64,fs:f64)->Result<TransferFunction>`
///
/// Frequency and fs share units (usually Hz); Q is center frequency divided by -3 dB bandwidth. Returns (b,a) in ascending delay order.
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
/// use sci_signal::filtering::*;
/// let _result = iirnotch(10.,30.,100.)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy iirnotch definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html).
pub fn iirnotch(frequency: f64, q: f64, fs: f64) -> Result<TransferFunction> {
    notch_peak(frequency, q, fs, false)
}
/// Designs a second-order peak filter with `g=1/(1+tan(ω0/(2Q)))`, `b=(1-g)[1,0,-1]`, `a=[1,-2g cos(ω0),2g-1]`.
///
/// ## Parameters
///
/// `pub fn iirpeak(frequency:f64,q:f64,fs:f64)->Result<TransferFunction>`
///
/// Frequency and fs share units; Q is center frequency divided by -3 dB bandwidth. Returns (b,a) in ascending delay order.
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
/// use sci_signal::filtering::*;
/// let _result = iirpeak(10.,30.,100.)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy iirpeak definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirpeak.html).
pub fn iirpeak(frequency: f64, q: f64, fs: f64) -> Result<TransferFunction> {
    notch_peak(frequency, q, fs, true)
}

/// Computes the empirical Kaiser beta: `0.1102(A-8.7)` above 50 dB, `0.5842(A-21)^0.4+0.07886(A-21)` above 21 dB, zero otherwise.
///
/// ## Parameters
///
/// `pub fn kaiser_beta(attenuation:f64)->f64`
///
/// A is attenuation in dB; negative values and NaN return zero. Returns dimensionless beta.
///
/// ## Domain
///
/// NaN and negative attenuation return zero; positive infinity returns infinity.
///
/// ## Returns
///
/// Returns dimensionless beta.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic. Cancellation and poorly conditioned filters or polynomial fits can reduce accuracy; no uniform relative-error guarantee is made. FFT routines use RustFFT, polynomial fits and DPSS use bounded private nalgebra solvers.
///
/// # Examples
///
/// ```
/// use sci_signal::filtering::*;
/// let _result = kaiser_beta(60.);
/// ```
///
/// # References
///
/// [SciPy kaiser_beta definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.kaiser_beta.html).
pub fn kaiser_beta(attenuation: f64) -> f64 {
    let a = attenuation;
    if a > 50. {
        0.1102 * (a - 8.7)
    } else if a > 21. {
        0.5842 * (a - 21.).powf(0.4) + 0.07886 * (a - 21.)
    } else {
        0.
    }
}
/// Estimates attenuation `A=2.285 (numtaps-1) π width + 7.95`.
///
/// ## Parameters
///
/// `pub fn kaiser_atten(numtaps:usize,width:f64)->Result<f64>`
///
/// width is transition bandwidth as a fraction of Nyquist. Returns attenuation in dB.
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
/// use sci_signal::filtering::*;
/// let _result = kaiser_atten(31,0.1)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy kaiser_atten definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.kaiser_atten.html).
pub fn kaiser_atten(numtaps: usize, width: f64) -> Result<f64> {
    if numtaps == 0 {
        return Err(Error::InvalidInput("numtaps must be positive"));
    }
    positive(width)?;
    match 2.285 * (numtaps - 1) as f64 * PI * width + 7.95 {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}
/// Estimates `numtaps=ceil((|ripple|-7.95)/(2.285 π width)+1)` and Kaiser beta.
///
/// ## Parameters
///
/// `pub fn kaiserord(ripple:f64,width:f64)->Result<(usize,f64)>`
///
/// ripple is amplitude rejection in dB, at least 8 in magnitude; width is a positive fraction of Nyquist. Returns (tap count, beta).
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
/// use sci_signal::filtering::*;
/// let _result = kaiserord(60.,0.1)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy kaiserord definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.kaiserord.html).
pub fn kaiserord(ripple: f64, width: f64) -> Result<(usize, f64)> {
    positive(width)?;
    if !ripple.is_finite() || ripple.abs() < 8. {
        return Err(Error::InvalidInput(
            "ripple magnitude must be at least 8 dB",
        ));
    }
    let taps = ((ripple.abs() - 7.95) / (2.285 * PI * width) + 1.).ceil();
    if !taps.is_finite() || taps > super::MAX_SAMPLES as f64 {
        return Err(Error::InvalidInput("filter length exceeds work limit"));
    }
    Ok((taps as usize, kaiser_beta(ripple)))
}

/// Designs a linear-phase FIR `h[i]=w[i] sum_b (f_hi sinc(f_hi t)-f_lo sinc(f_lo t))`, `t=i-(numtaps-1)/2`.
///
/// ## Parameters
///
/// `pub fn firwin(numtaps:usize,cutoffs:&[f64],pass_zero:bool,window:&Window,scale:bool,fs:f64)->Result<Vec<f64>>`
///
/// Cutoffs and fs share units. pass_zero starts with a passband at DC; successive cutoffs alternate bands. Window is symmetric. scale normalizes the first passband at DC, Nyquist, or its midpoint. Returns ascending-delay taps. Width-based dispatch is unsupported.
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
/// use sci_signal::filtering::*;
/// use sci_signal::windows;
/// let _result = firwin(5,&[0.2],true,&windows::Window::Hamming,true,2.)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy firwin definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html).
pub fn firwin(
    numtaps: usize,
    cutoffs: &[f64],
    pass_zero: bool,
    window: &Window,
    scale: bool,
    fs: f64,
) -> Result<Vec<f64>> {
    positive(fs)?;
    nonempty(cutoffs)?;
    size(numtaps)?;
    work(numtaps, cutoffs.len())?;
    if numtaps == 0
        || cutoffs[0] <= 0.
        || cutoffs[cutoffs.len() - 1] >= fs / 2.
        || cutoffs.windows(2).any(|w| w[0] >= w[1])
    {
        return Err(Error::InvalidInput(
            "positive taps and strictly increasing cutoffs inside Nyquist required",
        ));
    }
    let pass_nyquist = pass_zero == (cutoffs.len().is_multiple_of(2));
    if pass_nyquist && numtaps.is_multiple_of(2) {
        return Err(Error::InvalidInput("even tap count cannot pass Nyquist"));
    }
    let mut edges = Vec::new();
    if pass_zero {
        edges.push(0.);
    }
    edges.extend(cutoffs.iter().map(|v| v / (fs / 2.)));
    if pass_nyquist {
        edges.push(1.);
    }
    let taper = windows::get_window(window, numtaps, false)?;
    let center = (numtaps - 1) as f64 / 2.;
    let mut h: Vec<_> = (0..numtaps)
        .map(|i| {
            let t = i as f64 - center;
            edges
                .as_chunks::<2>()
                .0
                .iter()
                .map(|band| {
                    band[1] * crate::special::elementary::sinc(band[1] * t)
                        - band[0] * crate::special::elementary::sinc(band[0] * t)
                })
                .sum::<f64>()
                * taper[i]
        })
        .collect();
    if scale {
        let f = if edges[0] == 0. {
            0.
        } else if edges[1] == 1. {
            1.
        } else {
            (edges[0] + edges[1]) / 2.
        };
        let gain = h
            .iter()
            .enumerate()
            .map(|(i, &v)| v * (PI * f * (i as f64 - center)).cos())
            .sum::<f64>();
        if gain == 0. {
            return Err(Error::NumericalFailure("FIR normalization is singular"));
        }
        h.iter_mut().for_each(|v| *v /= gain);
    }
    checked(h)
}

/// Returns the median of each zero-padded neighborhood.
///
/// ## Parameters
///
/// `pub fn medfilt(x:&[f64],kernel:usize)->Result<Vec<f64>>`
///
/// kernel is a positive odd length. Returns input length; padding uses zeros even if kernel exceeds input.
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
/// use sci_signal::filtering::*;
/// let _result = medfilt(&[1.,4.,2.],3)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy medfilt definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt.html).
pub fn medfilt(x: &[f64], kernel: usize) -> Result<Vec<f64>> {
    order_filter(x, kernel, kernel / 2)
}
/// Returns the rank-th smallest sample in a zero-padded centered neighborhood.
///
/// ## Parameters
///
/// `pub fn order_filter(x:&[f64],kernel:usize,rank:usize)->Result<Vec<f64>>`
///
/// kernel is a positive odd full neighborhood length; rank is zero based. This adapts SciPy domain masks to a one-dimensional all-ones mask. Returns input length.
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
/// use sci_signal::filtering::*;
/// let _result = order_filter(&[1.,4.,2.],3,1)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy order_filter definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.order_filter.html).
pub fn order_filter(x: &[f64], kernel: usize, rank: usize) -> Result<Vec<f64>> {
    samples(x)?;
    size(kernel)?;
    work(x.len(), kernel)?;
    if kernel == 0 || kernel.is_multiple_of(2) || rank >= kernel {
        return Err(Error::InvalidInput(
            "kernel must be odd and rank smaller than kernel",
        ));
    }
    let half = kernel / 2;
    let mut buffer = vec![0.; kernel];
    let mut out = Vec::with_capacity(x.len());
    for i in 0..x.len() {
        for (j, v) in buffer.iter_mut().enumerate() {
            *v = (i + j)
                .checked_sub(half)
                .and_then(|k| x.get(k))
                .copied()
                .unwrap_or(0.);
        }
        buffer.select_nth_unstable_by(rank, f64::total_cmp);
        out.push(buffer[rank]);
    }
    Ok(out)
}

/// Computes `mean + (1-noise/variance)(x-mean)` when local variance exceeds noise, and the local mean otherwise.
///
/// ## Parameters
///
/// `pub fn wiener(x:&[f64],kernel:usize,noise:Option<f64>)->Result<Vec<f64>>`
///
/// kernel is positive odd and uses zero padding. noise=None estimates mean local variance. Returns input length. Zero-variance neighborhoods return their mean, including zero noise.
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
/// use sci_signal::filtering::*;
/// let _result = wiener(&[1.,4.,2.],3,None)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy wiener definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.wiener.html).
pub fn wiener(x: &[f64], kernel: usize, noise: Option<f64>) -> Result<Vec<f64>> {
    nonempty(x)?;
    size(kernel)?;
    work(x.len(), kernel)?;
    if kernel == 0 || kernel.is_multiple_of(2) || noise.is_some_and(|v| !v.is_finite() || v < 0.) {
        return Err(Error::InvalidInput(
            "kernel must be positive odd and noise nonnegative",
        ));
    }
    let mut means = Vec::with_capacity(x.len());
    let mut variances = Vec::with_capacity(x.len());
    for i in 0..x.len() {
        let local: Vec<_> = (0..kernel)
            .map(|j| {
                (i + j)
                    .checked_sub(kernel / 2)
                    .and_then(|k| x.get(k))
                    .copied()
                    .unwrap_or(0.)
            })
            .collect();
        let mean = local.iter().sum::<f64>() / kernel as f64;
        let var = match local.iter().map(|v| v * v).sum::<f64>() / kernel as f64 - mean * mean {
            value if value.is_finite() => Ok(value),
            _ => Err(Error::NonFiniteEvaluation),
        }?;
        means.push(mean);
        variances.push(var.max(0.));
    }
    let noise = match noise.unwrap_or_else(|| variances.iter().sum::<f64>() / x.len() as f64) {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }?;
    checked(
        x.iter()
            .zip(means)
            .zip(variances)
            .map(|((&v, m), s)| {
                if s <= noise || s == 0. {
                    m
                } else {
                    m + (1. - noise / s) * (v - m)
                }
            })
            .collect(),
    )
}

/// Finds least-squares polynomial derivative weights satisfying `sum_i c[i] x_i^j = derivative! / delta^derivative` for j=derivative, zero otherwise.
///
/// ## Parameters
///
/// `pub fn savgol_coeffs(window:usize,degree:usize,derivative:usize,delta:f64)->Result<Vec<f64>>`
///
/// window is the sample count, degree is polynomial degree <=32, derivative is derivative order, delta is positive sample spacing. Returns convolution-order centered weights; even windows center between samples. SciPy pos and use dispatch are unsupported.
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
/// use sci_signal::filtering::*;
/// let _result = savgol_coeffs(5,2,0,1.)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy savgol_coeffs definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_coeffs.html).
pub fn savgol_coeffs(
    window: usize,
    degree: usize,
    derivative: usize,
    delta: f64,
) -> Result<Vec<f64>> {
    validate_polynomial(window, degree, delta)?;
    if derivative > degree {
        return Ok(vec![0.; window]);
    }
    let projection = polynomial_projection(window, degree)?;
    polynomial_weights(&projection, window, derivative, delta)
}

fn validate_polynomial(window: usize, degree: usize, delta: f64) -> Result<()> {
    positive(delta)?;
    if window == 0 || window > 4095 || degree >= window || degree > 32 {
        return Err(Error::InvalidInput(
            "Savitzky-Golay requires 1..4095 samples and degree <= 32 smaller than window",
        ));
    }
    work(window, (degree + 1).pow(2))
}

fn polynomial_projection(window: usize, degree: usize) -> Result<nalgebra::DMatrix<f64>> {
    let scale = (window as f64 / 2.).max(1.);
    let center = (window - 1) as f64 / 2.;
    let a = nalgebra::DMatrix::from_fn(window, degree + 1, |i, j| {
        ((i as f64 - center) / scale).powi(j as i32)
    });
    nalgebra::linalg::SVD::try_new(a, true, true, f64::EPSILON, 100_000)
        .ok_or(Error::NumericalFailure("polynomial SVD did not converge"))?
        .pseudo_inverse(1e-13)
        .map_err(|_| Error::NumericalFailure("polynomial fit is singular"))
}

fn derivative_scale(window: usize, derivative: usize, delta: f64) -> f64 {
    let scale = (window as f64 / 2.).max(1.);
    (1..=derivative).map(|j| j as f64).product::<f64>() / (delta * scale).powi(derivative as i32)
}

fn polynomial_weights(
    projection: &nalgebra::DMatrix<f64>,
    window: usize,
    derivative: usize,
    delta: f64,
) -> Result<Vec<f64>> {
    let factor = derivative_scale(window, derivative, delta);
    checked(
        (0..window)
            .rev()
            .map(|i| projection[(derivative, i)] * factor)
            .collect(),
    )
}

fn polynomial_edge(
    coefficients: &nalgebra::DVector<f64>,
    window: usize,
    position: usize,
    derivative: usize,
    delta: f64,
) -> f64 {
    let scale = (window as f64 / 2.).max(1.);
    let coordinate = (position as f64 - (window - 1) as f64 / 2.) / scale;
    let mut value = 0.0;
    for j in (derivative..coefficients.len()).rev() {
        let factor = (j - derivative + 1..=j).map(|k| k as f64).product::<f64>();
        value = value * coordinate + coefficients[j] * factor;
    }
    value / (delta * scale).powi(derivative as i32)
}
/// Applies centered least-squares polynomial derivative weights with polynomial interpolation at the edges.
///
/// ## Parameters
///
/// `pub fn savgol_filter(x:&[f64],window:usize,degree:usize,derivative:usize,delta:f64)->Result<Vec<f64>>`
///
/// Only SciPy mode=interp and odd windows are supported. A single bounded SVD supplies the centered weights and both boundary polynomial fits. Derivative order and delta specify output units per delta^derivative. Returns input length; derivative greater than degree returns zeros.
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
/// use sci_signal::filtering::*;
/// let _result = savgol_filter(&[1.,2.,3.,4.,5.],3,1,0,1.)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy savgol_filter definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html).
pub fn savgol_filter(
    x: &[f64],
    window: usize,
    degree: usize,
    derivative: usize,
    delta: f64,
) -> Result<Vec<f64>> {
    nonempty(x)?;
    if window.is_multiple_of(2) || window > x.len() {
        return Err(Error::InvalidInput(
            "interpolated Savitzky-Golay requires odd window <= signal length",
        ));
    }
    validate_polynomial(window, degree, delta)?;
    work(x.len(), window * (degree + 1))?;
    if derivative > degree {
        return Ok(vec![0.; x.len()]);
    }
    // One pseudoinverse supplies the centered weights and both edge fits.
    let projection = polynomial_projection(window, degree)?;
    let weights = polynomial_weights(&projection, window, derivative, delta)?;
    let fit = |samples: &[f64]| &projection * nalgebra::DVector::from_column_slice(samples);
    let left = fit(&x[..window]);
    let right = fit(&x[x.len() - window..]);
    let half = window / 2;
    let mut result = vec![0.; x.len()];
    for (i, out) in result.iter_mut().enumerate() {
        if i < half {
            *out = polynomial_edge(&left, window, i, derivative, delta);
        } else if i + half >= x.len() {
            *out = polynomial_edge(&right, window, i - (x.len() - window), derivative, delta);
        } else {
            *out = x[i - half..=i + half]
                .iter()
                .zip(weights.iter().rev())
                .map(|(&v, &c)| v * c)
                .sum();
        }
    }
    checked(result)
}
