use super::{checked, complex_checked, nonempty, positive, samples, size, work};
use crate::{
    Error, Result,
    complex::{Complex, native, owned},
};
use num_complex::Complex64;
use std::f64::consts::TAU;

/// Returns the analytic signal `x + i H(x)` by doubling positive DFT frequencies and removing negative frequencies.
///
/// ## Parameters
///
/// `pub fn hilbert(x:&[f64])->Result<Vec<Complex>>`
///
/// Real input, original transform length, DC and even-length Nyquist bins unchanged. Returns one shared Complex per sample; no axis or length dispatch.
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
/// use sci_signal::*;
/// let _result = hilbert(&[1.,2.,3.])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy hilbert definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html).
pub fn hilbert(x: &[f64]) -> Result<Vec<Complex>> {
    nonempty(x)?;
    let input: Vec<_> = x.iter().map(|&re| Complex { re, im: 0. }).collect();
    let mut spectrum = super::fft(&input);
    let n = x.len();
    for (i, v) in spectrum.iter_mut().enumerate().skip(1) {
        if i < n.div_ceil(2) {
            v.re *= 2.;
            v.im *= 2.;
        } else if !(n.is_multiple_of(2) && i == n / 2) {
            *v = Complex::default();
        }
    }
    complex_checked(super::ifft(&spectrum))
}
/// Resamples a periodic signal by truncating or zero padding its DFT with Nyquist splitting/merging.
///
/// ## Parameters
///
/// `pub fn resample(x:&[f64],length:usize)->Result<Vec<f64>>`
///
/// Returns length equally spaced real samples over the original duration. Input and output periods coincide; no spectral window, timestamp, or frequency-domain dispatch.
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
/// use sci_signal::*;
/// let _result = resample(&[1.,2.,3.],5)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy resample definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html).
pub fn resample(x: &[f64], length: usize) -> Result<Vec<f64>> {
    nonempty(x)?;
    size(length)?;
    if length == 0 {
        return Err(Error::InvalidInput("output length must be positive"));
    }
    let spectrum = super::rfft(x);
    let m = x.len().min(length);
    let mut output = vec![Complex::default(); length];
    let scale = length as f64 / x.len() as f64;
    for i in 0..=m / 2 {
        let mut v = native(spectrum[i]) * scale;
        if m.is_multiple_of(2) && x.len() != length && i == m / 2 {
            v *= if length < x.len() { 2. } else { 0.5 };
        }
        output[i] = owned(v);
        if i > 0 && i < length - i {
            output[length - i] = owned(v.conj());
        }
    }
    checked(super::ifft(&output).into_iter().map(|v| v.re).collect())
}

/// Returns spiral points `z[k] = a w^(-k)` for k=0..length-1.
///
/// ## Parameters
///
/// `pub fn czt_points(length:usize,w:Option<Complex>,a:Complex)->Result<Vec<Complex>>`
///
/// Shared Complex parameters, nonzero a and w. w=None uses exp(-2πi/length). Returns positive-length points in contour order.
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
/// use sci_signal::*;
/// let _result = czt_points(3,None,sci_complex::Complex{re:1.,im:0.})?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy czt_points definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.czt_points.html).
pub fn czt_points(length: usize, w: Option<Complex>, a: Complex) -> Result<Vec<Complex>> {
    size(length)?;
    if length == 0 {
        return Err(Error::InvalidInput("CZT length must be positive"));
    }
    let w = w
        .map(native)
        .unwrap_or_else(|| Complex64::from_polar(1., -TAU / length as f64));
    let a = native(a);
    if ![w.re, w.im, a.re, a.im].iter().all(|v| v.is_finite()) || w.norm() == 0. || a.norm() == 0. {
        return Err(Error::InvalidInput(
            "CZT a and w must be finite and nonzero",
        ));
    }
    complex_checked((0..length).map(|k| owned(a / w.powu(k as u32))).collect())
}
/// Evaluates the z-transform `X[k]=sum_j x[j] (a w^(-k))^(-j)`.
///
/// ## Parameters
///
/// `pub fn czt(x:&[Complex],length:usize,w:Option<Complex>,a:Complex)->Result<Vec<Complex>>`
///
/// Shared Complex input, length output points, w=None selects equally spaced unit-circle points. a and w must be finite and nonzero. Returns length Complex values.
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
/// use sci_signal::*;
/// let _result = czt(&[sci_complex::Complex{re:1.,im:0.}],3,None,sci_complex::Complex{re:1.,im:0.})?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy czt definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.czt.html).
pub fn czt(x: &[Complex], length: usize, w: Option<Complex>, a: Complex) -> Result<Vec<Complex>> {
    size(x.len())?;
    size(length)?;
    work(x.len(), length)?;
    if x.is_empty() || x.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
        return Err(Error::InvalidInput("CZT input must be nonempty and finite"));
    }
    let points = czt_points(length, w, a)?;
    complex_checked(
        points
            .into_iter()
            .map(|z| {
                let inv = Complex64::new(1., 0.) / native(z);
                owned(
                    x.iter()
                        .rev()
                        .fold(Complex64::new(0., 0.), |value, &sample| {
                            value * inv + native(sample)
                        }),
                )
            })
            .collect(),
    )
}
/// Evaluates `X[k]=sum_j x[j] exp(-2π i f[k] j/fs)` on a uniformly spaced frequency interval.
///
/// ## Parameters
///
/// `pub fn zoom_fft(x:&[Complex],start:f64,stop:f64,length:usize,fs:f64,endpoint:bool)->Result<Vec<Complex>>`
///
/// start, stop and fs share frequency units. endpoint determines whether stop is included when length>1. Returns length Complex values; a singleton evaluates start.
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
/// use sci_signal::*;
/// let _result = zoom_fft(&[sci_complex::Complex{re:1.,im:0.}],0.,1.,3,2.,false)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy zoom_fft definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.zoom_fft.html).
pub fn zoom_fft(
    x: &[Complex],
    start: f64,
    stop: f64,
    length: usize,
    fs: f64,
    endpoint: bool,
) -> Result<Vec<Complex>> {
    positive(fs)?;
    if !start.is_finite() || !stop.is_finite() || length == 0 {
        return Err(Error::InvalidInput(
            "finite frequencies and positive output length required",
        ));
    }
    let denominator = if endpoint && length > 1 {
        length - 1
    } else {
        length
    };
    let w = owned(Complex64::from_polar(
        1.,
        -TAU * (stop - start) / (fs * denominator as f64),
    ));
    let a = owned(Complex64::from_polar(1., TAU * start / fs));
    czt(x, length, Some(w), a)
}

/// Frequency bins and real power values, in that order.
pub type PowerSpectrum = (Vec<f64>, Vec<f64>);
/// Frequency bins and complex cross-power values, in that order.
pub type CrossSpectrum = (Vec<f64>, Vec<Complex>);

/// Spectral power normalization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scaling {
    /// Divide squared transform by `fs * sum(window^2)`, giving power per Hz.
    Density,
    /// Divide by `sum(window)^2`, giving power per frequency bin.
    Spectrum,
}

fn spectral_parameters(
    x: &[f64],
    window: &[f64],
    overlap: usize,
    nfft: usize,
    fs: f64,
) -> Result<usize> {
    nonempty(x)?;
    nonempty(window)?;
    positive(fs)?;
    size(nfft)?;
    if overlap >= window.len() || nfft < window.len() || x.len() < window.len() {
        return Err(Error::InvalidInput(
            "requires overlap < window length <= signal length and FFT length",
        ));
    }
    let count = 1 + (x.len() - window.len()) / (window.len() - overlap);
    work(count, nfft)?;
    Ok(count)
}
fn spectral_scale(window: &[f64], fs: f64, scaling: Scaling) -> Result<f64> {
    let denominator = match scaling {
        Scaling::Density => fs * window.iter().map(|v| v * v).sum::<f64>(),
        Scaling::Spectrum => window.iter().sum::<f64>().powi(2),
    };
    if denominator == 0. || !denominator.is_finite() {
        return Err(Error::InvalidInput(
            "spectral window normalization must be finite and nonzero",
        ));
    }
    Ok(1. / denominator)
}
fn segment(x: &[f64], window: &[f64], nfft: usize, detrend: bool) -> Vec<Complex> {
    let mean = if detrend {
        x.iter().map(|v| v / x.len() as f64).sum::<f64>()
    } else {
        0.
    };
    let mut values = vec![0.; nfft];
    for ((v, &x), &w) in values.iter_mut().zip(x).zip(window) {
        *v = (x - mean) * w;
    }
    super::rfft(&values)
}
fn bins(nfft: usize, fs: f64) -> Vec<f64> {
    (0..=nfft / 2)
        .map(|k| k as f64 * (fs / nfft as f64))
        .collect()
}
fn doubling(k: usize, nfft: usize) -> f64 {
    if k == 0 || (nfft.is_multiple_of(2) && k == nfft / 2) {
        1.
    } else {
        2.
    }
}

/// Estimates one-sided spectral power `d[k] |FFT(w (x-mean))[k]|² / normalization`.
///
/// ## Parameters
///
/// `pub fn periodogram(x:&[f64],window:&[f64],nfft:usize,fs:f64,detrend:bool,scaling:Scaling)->Result<PowerSpectrum>`
///
/// Window length equals input length; nfft zero pads. fs is samples/time; bins run from DC to floor(nfft/2)*fs/nfft. detrend removes the mean. Scaling selects power/Hz or power/bin. Returns (frequencies, power).
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
/// use sci_signal::*;
/// let _result = periodogram(&[1.,2.,3.],&[1.;3],3,1.,true,Scaling::Density)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy periodogram definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html).
pub fn periodogram(
    x: &[f64],
    window: &[f64],
    nfft: usize,
    fs: f64,
    detrend: bool,
    scaling: Scaling,
) -> Result<PowerSpectrum> {
    if window.len() != x.len() {
        return Err(Error::DimensionMismatch);
    }
    spectral_parameters(x, window, 0, nfft, fs)?;
    let scale = spectral_scale(window, fs, scaling)?;
    let power = segment(x, window, nfft, detrend)
        .iter()
        .enumerate()
        .map(|(k, z)| native(*z).norm_sqr() * scale * doubling(k, nfft))
        .collect();
    Ok((bins(nfft, fs), checked(power)?))
}
/// Averages cross-periodograms `d[k] conj(FFT(w x))[k] FFT(w y)[k] / normalization`.
///
/// ## Parameters
///
/// Accepts two real signals and explicit window, overlap, FFT length, sample rate, detrending, and scaling parameters.
///
/// Equal-length real signals, explicit window and overlap count. Optional mean removal, one-sided scaling, incomplete segments dropped. Returns (frequency bins, Complex cross-power), using conjugate(x)*y ordering.
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
/// use sci_signal::*;
/// let _result = csd(&[1.,2.,3.],&[3.,2.,1.],&[1.;3],0,3,1.,true,Scaling::Density)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy csd definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.csd.html).
// Keep the two input signals and explicit SciPy spectral parameters together.
#[allow(clippy::too_many_arguments)]
pub fn csd(
    x: &[f64],
    y: &[f64],
    window: &[f64],
    overlap: usize,
    nfft: usize,
    fs: f64,
    detrend: bool,
    scaling: Scaling,
) -> Result<CrossSpectrum> {
    if x.len() != y.len() {
        return Err(Error::DimensionMismatch);
    }
    samples(y)?;
    let count = spectral_parameters(x, window, overlap, nfft, fs)?;
    let scale = spectral_scale(window, fs, scaling)? / count as f64;
    let mut output = vec![Complex64::new(0., 0.); nfft / 2 + 1];
    for i in 0..count {
        let start = i * (window.len() - overlap);
        let a = segment(&x[start..start + window.len()], window, nfft, detrend);
        let b = segment(&y[start..start + window.len()], window, nfft, detrend);
        for (k, v) in output.iter_mut().enumerate() {
            *v += native(a[k]).conj() * native(b[k]) * scale * doubling(k, nfft);
        }
    }
    Ok((
        bins(nfft, fs),
        complex_checked(output.into_iter().map(owned).collect())?,
    ))
}
/// Averages modified segment periodograms using arithmetic means.
///
/// ## Parameters
///
/// `pub fn welch(x:&[f64],window:&[f64],overlap:usize,nfft:usize,fs:f64,detrend:bool,scaling:Scaling)->Result<PowerSpectrum>`
///
/// Explicit window, overlap count, nfft, positive fs, optional mean detrending, and Scaling. Drops incomplete final segments. Returns (frequency bins, mean power); only real one-sided mean averaging is supported.
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
/// use sci_signal::*;
/// let _result = welch(&[1.,2.,3.,4.],&[1.;2],1,2,1.,true,Scaling::Density)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy welch definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html).
pub fn welch(
    x: &[f64],
    window: &[f64],
    overlap: usize,
    nfft: usize,
    fs: f64,
    detrend: bool,
    scaling: Scaling,
) -> Result<PowerSpectrum> {
    let (f, p) = csd(x, x, window, overlap, nfft, fs, detrend, scaling)?;
    Ok((f, p.into_iter().map(|v| v.re).collect()))
}
/// Estimates magnitude-squared coherence `|Pxy|²/(Pxx Pyy)`.
///
/// ## Parameters
///
/// `pub fn coherence(x:&[f64],y:&[f64],window:&[f64],overlap:usize,nfft:usize,fs:f64)->Result<PowerSpectrum>`
///
/// Uses equal-length real signals, arithmetic Welch averaging, mean detrending, explicit window and overlap, positive fs. Returns (frequencies, dimensionless coherence). Zero-power bins yield NaN.
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
/// Normalizes the cross-spectrum magnitude by each auto-spectrum square root
/// before squaring, avoiding overflow/underflow in their squared ratio. Welch
/// spectra still must be representable; zero auto-spectrum bins return NaN.
/// FFT rounding and spectral cancellation can reduce accuracy.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result = coherence(&[1.,2.,3.,4.],&[4.,3.,2.,1.],&[1.;2],1,2,1.)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy coherence definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html).
pub fn coherence(
    x: &[f64],
    y: &[f64],
    window: &[f64],
    overlap: usize,
    nfft: usize,
    fs: f64,
) -> Result<PowerSpectrum> {
    let (f, pxy) = csd(x, y, window, overlap, nfft, fs, true, Scaling::Density)?;
    let (_, pxx) = welch(x, window, overlap, nfft, fs, true, Scaling::Density)?;
    let (_, pyy) = welch(y, window, overlap, nfft, fs, true, Scaling::Density)?;
    // Zero-power bins have undefined coherence, matching SciPy's NaN.
    Ok((
        f,
        pxy.iter()
            .zip(pxx)
            .zip(pyy)
            .map(|((z, a), b)| {
                if a == 0. || b == 0. {
                    Ok(f64::NAN)
                } else {
                    crate::error::finite((z.re.hypot(z.im) / a.sqrt() / b.sqrt()).powi(2))
                }
            })
            .collect::<Result<_>>()?,
    ))
}

/// Frequencies, segment-center times, and power rows indexed `[frequency][time]`.
pub type Spectrogram = (Vec<f64>, Vec<f64>, Vec<Vec<f64>>);
/// Frequencies, segment-center times, and complex rows indexed `[frequency][time]`.
pub type ShortTimeSpectrum = (Vec<f64>, Vec<f64>, Vec<Vec<Complex>>);

/// Computes one-sided segment periodograms without averaging.
///
/// ## Parameters
///
/// `pub fn spectrogram(x:&[f64],window:&[f64],overlap:usize,nfft:usize,fs:f64,detrend:bool,scaling:Scaling)->Result<Spectrogram>`
///
/// Explicit window and overlap, nfft, positive fs, optional mean detrending and Scaling. No boundary extension; drops incomplete final segments. Returns (frequencies, center times, power`[frequency]``[time]`); times have reciprocal-fs units.
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
/// use sci_signal::*;
/// let _result = spectrogram(&[1.,2.,3.,4.],&[1.;2],1,2,1.,true,Scaling::Density)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy spectrogram definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html).
pub fn spectrogram(
    x: &[f64],
    window: &[f64],
    overlap: usize,
    nfft: usize,
    fs: f64,
    detrend: bool,
    scaling: Scaling,
) -> Result<Spectrogram> {
    let count = spectral_parameters(x, window, overlap, nfft, fs)?;
    let scale = spectral_scale(window, fs, scaling)?;
    let mut power = vec![vec![0.; count]; nfft / 2 + 1];
    for i in 0..count {
        let start = i * (window.len() - overlap);
        let spectrum = segment(&x[start..start + window.len()], window, nfft, detrend);
        for (k, row) in power.iter_mut().enumerate() {
            row[i] =
                crate::error::finite(native(spectrum[k]).norm_sqr() * scale * doubling(k, nfft))?;
        }
    }
    let times = (0..count)
        .map(|i| (i * (window.len() - overlap)) as f64 / fs + window.len() as f64 / (2. * fs))
        .collect();
    Ok((bins(nfft, fs), times, power))
}
/// Computes segment transforms `Z[k,t]=FFT(w x_t)[k]/sum(w)`.
///
/// ## Parameters
///
/// `pub fn stft(x:&[f64],window:&[f64],overlap:usize,nfft:usize,fs:f64)->Result<ShortTimeSpectrum>`
///
/// Real input, one-sided output, no detrending, spectrum scaling. Extends both ends by floor(window.len()/2) zeros and pads the last frame, matching SciPy boundary=zeros,padded=true. Returns (frequencies,times,Z`[frequency]``[time]`); times begin at zero.
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
/// use sci_signal::*;
/// let _result = stft(&[1.,2.,3.,4.],&[1.;2],1,2,1.)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy stft definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html).
pub fn stft(
    x: &[f64],
    window: &[f64],
    overlap: usize,
    nfft: usize,
    fs: f64,
) -> Result<ShortTimeSpectrum> {
    nonempty(x)?;
    nonempty(window)?;
    positive(fs)?;
    size(nfft)?;
    if overlap >= window.len() || nfft < window.len() || window.len() > x.len() {
        return Err(Error::InvalidInput(
            "requires overlap < window length <= signal and FFT lengths",
        ));
    }
    let hop = window.len() - overlap;
    let half = window.len() / 2;
    let extended_len = x.len() + 2 * half;
    let remainder = (extended_len - window.len()) % hop;
    let padding = (hop - remainder) % hop;
    size(extended_len + padding)?;
    let mut extended = vec![0.; extended_len + padding];
    extended[half..half + x.len()].copy_from_slice(x);
    let count = spectral_parameters(&extended, window, overlap, nfft, fs)?;
    let gain = window.iter().sum::<f64>();
    if !gain.is_finite() || gain == 0. {
        return Err(Error::InvalidInput(
            "STFT window sum must be finite and nonzero",
        ));
    }
    let mut output = vec![vec![Complex::default(); count]; nfft / 2 + 1];
    for i in 0..count {
        let spectrum = segment(
            &extended[i * hop..i * hop + window.len()],
            window,
            nfft,
            false,
        );
        for (k, row) in output.iter_mut().enumerate() {
            row[i] = owned(native(spectrum[k]) / gain);
        }
    }
    for row in &output {
        super::check_complex(row)?;
    }
    Ok((
        bins(nfft, fs),
        (0..count).map(|i| i as f64 * hop as f64 / fs).collect(),
        output,
    ))
}
/// Reconstructs samples by `x[n]=sum_t w[n-tH] IFFT(Z_t)[n-tH] sum(w) / sum_t w[n-tH]²`.
///
/// ## Parameters
///
/// `pub fn istft(spectrum:&[Vec<Complex>],window:&[f64],overlap:usize,nfft:usize,fs:f64)->Result<(Vec<f64>,Vec<f64>)>`
///
/// Input shape is `[frequency]``[time]`, with nfft/2+1 rows. Uses spectrum scaling and removes floor(window.len()/2) boundary samples from each side. Returns (sample times, real samples); trailing padded samples remain. NOLA failure is an error.
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
/// use sci_signal::*;
/// let _result = istft(&stft(&[1.,2.,3.,4.],&[1.;2],1,2,1.)?.2,&[1.;2],1,2,1.)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy istft definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.istft.html).
pub fn istft(
    spectrum: &[Vec<Complex>],
    window: &[f64],
    overlap: usize,
    nfft: usize,
    fs: f64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    nonempty(window)?;
    positive(fs)?;
    size(nfft)?;
    if nfft < window.len()
        || overlap >= window.len()
        || spectrum.len() != nfft / 2 + 1
        || spectrum.is_empty()
        || spectrum[0].is_empty()
    {
        return Err(Error::InvalidInput(
            "invalid one-sided STFT shape or window",
        ));
    }
    let count = spectrum[0].len();
    size(count)?;
    work(count, nfft)?;
    if spectrum.iter().any(|row| row.len() != count) {
        return Err(Error::DimensionMismatch);
    }
    let hop = window.len() - overlap;
    let len = window.len() + hop * (count - 1);
    size(len)?;
    let gain = window.iter().sum::<f64>();
    if !gain.is_finite() || gain == 0. {
        return Err(Error::InvalidInput(
            "STFT window sum must be finite and nonzero",
        ));
    }
    let mut output = vec![0.; len];
    let mut normalization = vec![0.; len];
    for i in 0..count {
        let mut full = vec![Complex::default(); nfft];
        for k in 0..=nfft / 2 {
            full[k] = owned(native(spectrum[k][i]) * gain);
            if k > 0 && k < nfft - k {
                full[nfft - k] = owned(native(full[k]).conj());
            }
        }
        let values = super::ifft(&full);
        for j in 0..window.len() {
            output[i * hop + j] += values[j].re * window[j];
            normalization[i * hop + j] += window[j] * window[j];
        }
    }
    let half = window.len() / 2;
    let mut result = Vec::with_capacity(len - 2 * half);
    for i in half..len - half {
        if normalization[i] <= 1e-10 {
            return Err(Error::NumericalFailure(
                "STFT window fails nonzero overlap-add",
            ));
        }
        result.push(output[i] / normalization[i]);
    }
    let result = checked(result)?;
    Ok(((0..result.len()).map(|i| i as f64 / fs).collect(), result))
}

fn overlap_bins(window: &[f64], overlap: usize, squared: bool) -> Result<Vec<f64>> {
    nonempty(window)?;
    if overlap >= window.len() {
        return Err(Error::InvalidInput("overlap must be smaller than window"));
    }
    let mut bins = vec![0.; window.len() - overlap];
    for (i, &v) in window.iter().enumerate() {
        let k = i % bins.len();
        bins[k] += if squared { v * v } else { v };
    }
    checked(bins)
}
/// Tests whether overlap-bin window sums are constant within tolerance of their median.
///
/// ## Parameters
///
/// `pub fn check_cola(window:&[f64],overlap:usize,tolerance:f64)->Result<bool>`
///
/// overlap is sample count, tolerance is positive absolute tolerance. Returns a boolean.
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
/// use sci_signal::*;
/// let _result = check_cola(&[1.;4],2,1e-10)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy check_cola definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.check_cola.html).
pub fn check_cola(window: &[f64], overlap: usize, tolerance: f64) -> Result<bool> {
    positive(tolerance)?;
    let mut bins = overlap_bins(window, overlap, false)?;
    bins.sort_by(f64::total_cmp);
    let n = bins.len();
    let median = if n % 2 == 1 {
        bins[n / 2]
    } else {
        (bins[n / 2 - 1] + bins[n / 2]) / 2.
    };
    Ok(bins.iter().all(|v| (v - median).abs() < tolerance))
}
/// Tests whether every overlap-bin squared-window sum exceeds tolerance.
///
/// ## Parameters
///
/// `pub fn check_nola(window:&[f64],overlap:usize,tolerance:f64)->Result<bool>`
///
/// overlap is sample count, tolerance is positive absolute threshold. Returns a boolean.
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
/// use sci_signal::*;
/// let _result = check_nola(&[1.;4],2,1e-10)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy check_nola definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.check_nola.html).
pub fn check_nola(window: &[f64], overlap: usize, tolerance: f64) -> Result<bool> {
    positive(tolerance)?;
    Ok(overlap_bins(window, overlap, true)?
        .iter()
        .all(|v| *v > tolerance))
}

/// Evaluates classical Lomb-Scargle power `0.5 ((sum y cos)^2/sum cos² + (sum y sin)^2/sum sin²)` after phase rotation.
///
/// ## Parameters
///
/// `pub fn lombscargle(times:&[f64],x:&[f64],frequencies:&[f64],precenter:bool,normalize:bool)->Result<Vec<f64>>`
///
/// times are sample times, frequencies positive radians/time. precenter removes the mean. normalize divides by the energy of the optionally centered data divided by two. Returns powers in frequency order; floating means, weights, and amplitude mode are unsupported.
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
/// use sci_signal::*;
/// let _result = lombscargle(&[0.,1.,2.],&[1.,2.,1.],&[1.],false,false)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy lombscargle definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lombscargle.html).
pub fn lombscargle(
    times: &[f64],
    x: &[f64],
    frequencies: &[f64],
    precenter: bool,
    normalize: bool,
) -> Result<Vec<f64>> {
    nonempty(times)?;
    nonempty(x)?;
    samples(frequencies)?;
    if times.len() != x.len() {
        return Err(Error::DimensionMismatch);
    }
    work(times.len(), frequencies.len())?;
    if frequencies.iter().any(|v| *v <= 0.) {
        return Err(Error::InvalidInput(
            "Lomb-Scargle frequencies must be positive angular frequencies",
        ));
    }
    let mean = if precenter {
        x.iter().sum::<f64>() / x.len() as f64
    } else {
        0.
    };
    let energy = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
    if normalize && energy == 0. {
        return Err(Error::InvalidInput("cannot normalize zero-power data"));
    }
    let mut output = Vec::with_capacity(frequencies.len());
    for &w in frequencies {
        let (s, c) = times.iter().fold((0., 0.), |(s, c), &t| {
            let (a, b) = (2. * w * t).sin_cos();
            (s + a, c + b)
        });
        let tau = s.atan2(c) / 2.;
        let (yc, ys, cc, ss) =
            times
                .iter()
                .zip(x)
                .fold((0., 0., 0., 0.), |(yc, ys, cc, ss), (&t, &v)| {
                    let (s, c) = (w * t - tau).sin_cos();
                    (
                        yc + (v - mean) * c,
                        ys + (v - mean) * s,
                        cc + c * c,
                        ss + s * s,
                    )
                });
        if cc == 0. || ss == 0. {
            return Err(Error::NumericalFailure("degenerate Lomb-Scargle design"));
        }
        let power = 0.5 * (yc * yc / cc + ys * ys / ss);
        output.push(if normalize {
            2. * power / energy
        } else {
            power
        });
    }
    checked(output)
}
