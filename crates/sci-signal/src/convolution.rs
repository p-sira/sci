use super::{checked, nonempty, size, work};
use crate::{
    Error, Result,
    complex::{Complex, native, owned},
};

pub use sci_core::Mode;

fn range(n: usize, m: usize, mode: Mode) -> (usize, usize) {
    match mode {
        Mode::Full => (0, n + m - 1),
        Mode::Same => ((m - 1) / 2, n),
        Mode::Valid => (n.min(m) - 1, n.abs_diff(m) + 1),
    }
}

/// Computes zero-extended linear convolution `y[k] = sum_j x[j] h[k-j]`.
///
/// ## Parameters
///
/// `pub fn convolve(x: &[f64], h: &[f64], mode: Mode) -> Result<Vec<f64>>`
///
/// Real one-dimensional slices. Full output has n+m-1 elements, Same has n elements centered at floor((m-1)/2), and Valid has |n-m|+1 elements.
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
/// Direct binary64 accumulation visits only products retained by the requested
/// output mode, in increasing input-index order. Cropping preserves the full
/// convolution's summation order. Cancellation, intermediate overflow and
/// underflow remain possible; no uniform relative-error guarantee is made.
/// The conservative n*m work limit still applies before evaluating products.
///
/// # Examples
///
/// ```
/// use sci_signal::convolution::*;
/// let _result = convolve(&[1.,2.], &[1.,-1.], Mode::Full)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy convolve definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html).
pub fn convolve(x: &[f64], h: &[f64], mode: Mode) -> Result<Vec<f64>> {
    nonempty(x)?;
    nonempty(h)?;
    work(x.len(), h.len())?;
    size(x.len() + h.len() - 1)?;
    let (start, len) = range(x.len(), h.len(), mode);
    let mut y = vec![0.; len];
    for (i, &a) in x.iter().enumerate() {
        let first = start.saturating_sub(i);
        let end = (start + len).saturating_sub(i).min(h.len());
        if first < end {
            let output = &mut y[i + first - start..i + end - start];
            for (value, &b) in output.iter_mut().zip(&h[first..end]) {
                *value += a * b;
            }
        }
    }
    checked(y)
}

/// Computes real cross-correlation `r[k] = sum_j x[j] y[j-k]`.
///
/// ## Parameters
///
/// `pub fn correlate(x: &[f64], y: &[f64], mode: Mode) -> Result<Vec<f64>>`
///
/// Same and Valid crop the full correlation using the convolution convention; full lags run from -(m-1) through n-1. Complex and multidimensional inputs are unsupported.
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
/// use sci_signal::convolution::*;
/// let _result = correlate(&[1.,2.], &[1.], Mode::Full)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy correlate definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html).
pub fn correlate(x: &[f64], y: &[f64], mode: Mode) -> Result<Vec<f64>> {
    super::samples(y)?;
    let reversed: Vec<_> = y.iter().rev().copied().collect();
    convolve(x, &reversed, mode)
}

/// Returns integer sample displacements for cross-correlation.
///
/// ## Parameters
///
/// `pub fn correlation_lags(n: usize, m: usize, mode: Mode) -> Result<Vec<isize>>`
///
/// Full lags are -(m-1)..n-1. Valid selects complete overlap. SciPy Same selects n lags centered on the full lag vector; its offset differs from correlate for odd n and even m.
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
/// use sci_signal::convolution::*;
/// let _result = correlation_lags(3,2,Mode::Full)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy correlation_lags definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlation_lags.html).
pub fn correlation_lags(n: usize, m: usize, mode: Mode) -> Result<Vec<isize>> {
    if n == 0 || m == 0 {
        return Err(Error::InvalidInput("lengths must be positive"));
    }
    size(n)?;
    size(m)?;
    size(n + m - 1)?;
    // SciPy's same-mode lag convention differs from convolution cropping for
    // odd first lengths and even second lengths.
    let (start, len) = if mode == Mode::Same {
        let mid = (n + m - 1) / 2;
        (mid - n / 2, n)
    } else {
        range(n, m, mode)
    };
    Ok((start..start + len)
        .map(|i| i as isize - (m - 1) as isize)
        .collect())
}

/// Computes linear convolution `y = IFFT(FFT(x) FFT(h))` with zero padding.
///
/// ## Parameters
///
/// `pub fn fftconvolve(x: &[f64], h: &[f64], mode: Mode) -> Result<Vec<f64>>`
///
/// Mode cropping agrees with convolve. Transform length is the next power of two of n+m-1; RustFFT is private.
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
/// use sci_signal::convolution::*;
/// let _result = fftconvolve(&[1.,2.], &[1.,-1.], Mode::Full)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy fftconvolve definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html).
pub fn fftconvolve(x: &[f64], h: &[f64], mode: Mode) -> Result<Vec<f64>> {
    nonempty(x)?;
    nonempty(h)?;
    let full = x.len() + h.len() - 1;
    let n = full.next_power_of_two();
    size(n)?;
    let padded = |values: &[f64]| {
        let mut buffer = vec![Complex::default(); n];
        for (v, &re) in buffer.iter_mut().zip(values) {
            v.re = re;
        }
        super::fft(&buffer)
    };
    let a = padded(x);
    let b = padded(h);
    let product: Vec<_> = a
        .into_iter()
        .zip(b)
        .map(|(a, b)| owned(native(a) * native(b)))
        .collect();
    let output = super::ifft(&product);
    let (start, len) = range(x.len(), h.len(), mode);
    checked(output[start..start + len].iter().map(|v| v.re).collect())
}

/// Divides coefficient polynomials so that `x = convolve(h, quotient) + remainder`.
///
/// ## Parameters
///
/// `pub fn deconvolve(x: &[f64], h: &[f64]) -> Result<(Vec<f64>, Vec<f64>)>`
///
/// Coefficients are ordered by descending powers. Returns (quotient, remainder), with remainder padded to the input length. A longer divisor returns an empty quotient.
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
/// use sci_signal::convolution::*;
/// let _result = deconvolve(&[1.,3.,2.], &[1.,1.])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy deconvolve definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.deconvolve.html).
pub fn deconvolve(x: &[f64], h: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
    nonempty(x)?;
    nonempty(h)?;
    if h[0] == 0. {
        return Err(Error::InvalidInput(
            "leading divisor coefficient must be nonzero",
        ));
    }
    if h.len() > x.len() {
        return Ok((vec![], x.to_vec()));
    }
    let n = x.len() - h.len() + 1;
    work(n, h.len())?;
    let mut remainder = x.to_vec();
    let mut quotient = vec![0.; n];
    for i in 0..n {
        quotient[i] = remainder[i] / h[0];
        for (j, &v) in h.iter().enumerate() {
            remainder[i + j] -= quotient[i] * v;
        }
    }
    Ok((checked(quotient)?, checked(remainder)?))
}

/// Computes `y[k] = sum_j x[j] h[k down - j up]` with zero extension.
///
/// ## Parameters
///
/// `pub fn upfirdn(h: &[f64], x: &[f64], up: usize, down: usize) -> Result<Vec<f64>>`
///
/// up and down are positive integer rates. Returns ceil(((n-1)up+len(h))/down) samples without allocating a zero-inserted input. Only constant-zero padding is supported.
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
/// use sci_signal::convolution::*;
/// let _result = upfirdn(&[1.,1.], &[1.,2.], 2,1)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy upfirdn definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.upfirdn.html).
pub fn upfirdn(h: &[f64], x: &[f64], up: usize, down: usize) -> Result<Vec<f64>> {
    nonempty(h)?;
    nonempty(x)?;
    if up == 0 || down == 0 {
        return Err(Error::InvalidInput("rates must be positive"));
    }
    let full = (x.len() - 1)
        .checked_mul(up)
        .and_then(|n| n.checked_add(h.len()))
        .ok_or(Error::InvalidInput("resampling length overflow"))?;
    let len = (full - 1) / down + 1;
    size(len)?;
    work(x.len(), h.len())?;
    let mut output = vec![0.; len];
    for (i, &v) in x.iter().enumerate() {
        for (j, &c) in h.iter().enumerate() {
            let k = i * up + j;
            if k.is_multiple_of(down) {
                output[k / down] += v * c;
            }
        }
    }
    checked(output)
}

/// Resamples `y[k]=sum_j x[j] h[k down-j up]` with centered lowpass filtering.
///
/// ## Parameters
///
/// Positive integer up and down are reduced by their greatest common divisor.
/// `filter=None` designs a `20 max(up,down)+1` tap Kaiser-beta-5 lowpass,
/// matching SciPy's default. A supplied filter uses unscaled ascending-delay
/// FIR taps. Padding is constant zero; phase alignment removes filter delay.
///
/// ## Domain
///
/// Samples and taps must be nonempty and finite. Equal reduced rates return a
/// copy of the input. Odd symmetric taps are recommended for zero phase.
///
/// ## Returns
///
/// Returns ceil(input length * up / down) real samples. Output sample spacing
/// is down/up times the input spacing.
///
/// ## Errors
///
/// Returns shared errors for invalid rates, shapes, nonfinite arithmetic,
/// lengths above 1048576, or direct filter work above 16777216 operations.
/// Other SciPy pad modes, multidimensional axes and window dispatch are unsupported.
///
/// ## Numerical behavior
///
/// Uses explicit FIR design and integer phase selection; arithmetic is f64.
/// Finite filtering approximates ideal interpolation and suppresses aliasing.
/// End transients depend on the zero extension.
///
/// # Examples
///
/// ```
/// let x=sci_signal::convolution::resample_poly(&[1.,2.,3.],2,1,None)?;
/// assert_eq!(x.len(),6);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy resample_poly](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html).
pub fn resample_poly(
    x: &[f64],
    up: usize,
    down: usize,
    filter: Option<&[f64]>,
) -> Result<Vec<f64>> {
    nonempty(x)?;
    if up == 0 || down == 0 {
        return Err(Error::InvalidInput("rates must be positive"));
    }
    if let Some(filter) = filter {
        nonempty(filter)?;
    }
    let (mut a, mut b) = (up, down);
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    let up = up / a;
    let down = down / a;
    size(up)?;
    size(down)?;
    if up == 1 && down == 1 {
        return Ok(x.to_vec());
    }
    let count = x
        .len()
        .checked_mul(up)
        .ok_or(Error::InvalidInput("resampling length overflow"))?
        .div_ceil(down);
    size(count)?;
    let rate = up.max(down);
    let taps = if let Some(filter) = filter {
        filter.to_vec()
    } else {
        super::firwin(
            20 * rate + 1,
            &[1. / rate as f64],
            true,
            &super::windows::Window::Kaiser(5.),
            true,
            2.,
        )?
    };
    let half = (taps.len() - 1) / 2;
    let pre = down - half % down;
    let remove = (half + pre) / down;
    let initial_len = pre + taps.len();
    let full = (x.len() - 1)
        .checked_mul(up)
        .and_then(|n| n.checked_add(initial_len))
        .ok_or(Error::InvalidInput("resampling length overflow"))?;
    let required = (count + remove - 1)
        .checked_mul(down)
        .and_then(|n| n.checked_add(1))
        .ok_or(Error::InvalidInput("resampling padding overflow"))?;
    let post = required.saturating_sub(full);
    size(initial_len + post)?;
    let mut h = vec![0.; initial_len + post];
    for (out, &tap) in h[pre..].iter_mut().zip(&taps) {
        *out = tap * up as f64;
    }
    let result = upfirdn(&h, x, up, down)?;
    Ok(result[remove..remove + count].to_vec())
}

/// Downsamples by q after applying a Hamming-windowed lowpass FIR.
///
/// ## Parameters
///
/// q is a positive integer factor; order determines order+1 FIR taps with
/// cutoff Nyquist/q. zero_phase centers the filter using polyphase delay
/// removal; false retains its causal delay. This supports SciPy ftype="fir"
/// only, with explicit order. IIR decimation requires filter and downsample calls.
///
/// ## Domain
///
/// Input must be nonempty and finite, q>1, order positive and <=1048575.
///
/// ## Returns
///
/// Returns ceil(input length/q) real samples; spacing is q times the input.
///
/// ## Errors
///
/// Returns shared errors for invalid parameters, overflow, or signal/filter
/// lengths above 1048576 and direct work above 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses finite f64 FIR coefficients. Filter length controls stopband accuracy;
/// zero extension causes transients at both ends in zero-phase mode.
///
/// # Examples
///
/// ```
/// let x=sci_signal::convolution::decimate(&[1.,2.,3.,4.],2,8,true)?;
/// assert_eq!(x.len(),2);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy decimate](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.decimate.html).
pub fn decimate(x: &[f64], q: usize, order: usize, zero_phase: bool) -> Result<Vec<f64>> {
    nonempty(x)?;
    size(q)?;
    size(order)?;
    if q < 2 || order == 0 || order >= super::MAX_SAMPLES {
        return Err(Error::InvalidInput(
            "FIR decimation requires q>1 and order in 1..1048575",
        ));
    }
    let taps = super::firwin(
        order + 1,
        &[1. / q as f64],
        true,
        &super::windows::Window::Hamming,
        true,
        2.,
    )?;
    if zero_phase {
        resample_poly(x, 1, q, Some(&taps))
    } else {
        let mut result = upfirdn(&taps, x, 1, q)?;
        result.truncate(x.len().div_ceil(q));
        Ok(result)
    }
}
