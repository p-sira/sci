//! Window samples for filter design and spectral analysis.
//!
//! `symmetric = true` evaluates an M-point symmetric taper; false evaluates
//! M+1 symmetric points and discards the last. Lengths 0 and 1 return an empty
//! vector and `[1]`, respectively. Windows are peak-normalized according to
//! SciPy, rather than energy-normalized, unless explicitly stated.

use super::{checked, positive, size, work};
use crate::{Error, Result, complex::owned};
use std::f64::consts::{PI, TAU};

fn length(m: usize, symmetric: bool) -> Result<usize> {
    size(m)?;
    let n = m + usize::from(!symmetric && m > 1);
    size(n)?;
    Ok(n)
}
fn evaluate(m: usize, symmetric: bool, f: impl Fn(f64, f64) -> f64) -> Result<Vec<f64>> {
    let n = length(m, symmetric)?;
    if m <= 1 {
        return Ok(vec![1.; m]);
    }
    checked((0..m).map(|i| f(i as f64, n as f64)).collect())
}

macro_rules! simple_window {
    ($name:ident,$equation:literal,$function:expr) => {
        #[doc=concat!("Evaluates the window `",$equation,"`.",
        "\n\n# Parameters and conventions\n`m` is the output length; `symmetric` selects filter-design symmetry or periodic truncation. `n` in the equation is the extended length.",
        "\n\n# Domain and special values\nLengths zero and one yield `[]` and `[1]`.",
        "\n\n# Return values\nReturns `m` real weights in sample order; no energy normalization.",
        "\n\n# Errors\nRejects lengths above 1048576 and arithmetic overflow.",
        "\n\n# Numerical behavior\nDirect f64 evaluation; symmetric even lengths need not attain one. O(m) work.",
        "\n\n# Examples\n```\nassert_eq!(sci_signal::windows::",stringify!($name),"(1, true)?, vec![1.]);\n# Ok::<(), sci_core::Error>(())\n```",
        "\n\n# References\n[SciPy window definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.",stringify!($name),".html).")]
        pub fn $name(m: usize,symmetric: bool)->Result<Vec<f64>> {evaluate(m,symmetric,$function)}
    }
}
simple_window!(boxcar, "w[i] = 1", |_, _| 1.);
simple_window!(bartlett, "w[i] = 1 - |2i/(n-1)-1|", |i, n| 1.
    - (2. * i / (n - 1.) - 1.).abs());
simple_window!(triang, "w[i] = 1 - |(i-(n-1)/2)/ceil(n/2)|", |i, n| 1.
    - ((i - (n - 1.) / 2.) / (n / 2.).ceil()).abs());
simple_window!(cosine, "w[i] = sin(pi (i+1/2)/n)", |i, n| (PI * (i + 0.5)
    / n)
    .sin());
simple_window!(lanczos, "w[i] = sinc(2i/(n-1)-1)", |i, n| {
    crate::special::sinc(2. * i / (n - 1.) - 1.)
});
simple_window!(
    barthann,
    "w[i] = 0.62 - 0.48|i/(n-1)-1/2| + 0.38 cos(2 pi (i/(n-1)-1/2))",
    |i, n| {
        let x = (i / (n - 1.) - 0.5).abs();
        0.62 - 0.48 * x + 0.38 * (TAU * x).cos()
    }
);
simple_window!(
    bohman,
    "w[i] = (1-|x|)cos(pi |x|)+sin(pi |x|)/pi, x=2i/(n-1)-1",
    |i, n| {
        let x = (2. * i / (n - 1.) - 1.).abs();
        if x >= 1. {
            0.
        } else {
            (1. - x) * (PI * x).cos() + (PI * x).sin() / PI
        }
    }
);
simple_window!(
    parzen,
    "w[i] = 1-6x^2+6x^3 for x<=1/2; 2(1-x)^3 otherwise, x=|i-(n-1)/2|/(n/2)",
    |i, n| {
        let x = (i - (n - 1.) / 2.).abs() / (n / 2.);
        if x <= 0.5 {
            1. - 6. * x * x + 6. * x.powi(3)
        } else {
            2. * (1. - x).powi(3)
        }
    }
);

/// Evaluates `w[i]=sum_k coefficients[k] cos(k(-π+2π i/(n-1)))`.
///
/// ## Parameters
///
/// `pub fn general_cosine(m: usize, coefficients: &[f64], symmetric: bool) -> Result<Vec<f64>>`
///
/// Coefficients are alternating-sign cosine amplitudes using SciPy centered convention. symmetric uses M points; false truncates M+1. Returns m weights without rescaling coefficients.
///
/// ## Domain
///
/// Fallible functions reject NaN and infinity in inputs. Input slices may be empty only where the definition is meaningful. Window lengths zero and one yield `[]` and `[1]`, unless a stricter domain is stated.
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
/// use sci_signal::windows::*;
/// let _result = general_cosine(5,&[0.5,0.5],true)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy general_cosine definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.general_cosine.html).
pub fn general_cosine(m: usize, coefficients: &[f64], symmetric: bool) -> Result<Vec<f64>> {
    super::nonempty(coefficients)?;
    work(m, coefficients.len())?;
    evaluate(m, symmetric, |i, n| {
        let angle = -PI + TAU * i / (n - 1.);
        coefficients
            .iter()
            .enumerate()
            .map(|(k, &a)| a * (k as f64 * angle).cos())
            .sum()
    })
}
macro_rules! cosine_window {
    ($name:ident,$a:expr) => {
        #[doc=concat!("Evaluates `w[i] = sum_k a[k] cos(k(-pi + 2 pi i/(n-1)))` with coefficients `",stringify!($a),"`.",
        "\n\n# Parameters and conventions\n`m` is output length; `symmetric` uses M points when true and truncates M+1 when false.",
        "\n\n# Domain and special values\nZero and one lengths return `[]` and `[1]`.",
        "\n\n# Return values\nReturns `m` weights, normalized to the continuous window peak.",
        "\n\n# Errors\nLengths above 1048576 or numerical work above 16777216 operations are rejected.",
        "\n\n# Numerical behavior\nDirect f64 cosine summation; even symmetric windows may not attain one.",
        "\n\n# Examples\n```\nassert_eq!(sci_signal::windows::",stringify!($name),"(1,true)?,vec![1.]);\n# Ok::<(),sci_core::Error>(())\n```",
        "\n\n# References\n[SciPy definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.",stringify!($name),".html).")]
        pub fn $name(m:usize,symmetric:bool)->Result<Vec<f64>> {general_cosine(m,&$a,symmetric)}
    }
}
cosine_window!(hann, [0.5, 0.5]);
cosine_window!(hamming, [0.54, 0.46]);
cosine_window!(blackman, [0.42, 0.5, 0.08]);
cosine_window!(blackmanharris, [0.35875, 0.48829, 0.14128, 0.01168]);
cosine_window!(nuttall, [0.3635819, 0.4891775, 0.1365995, 0.0106411]);
cosine_window!(
    flattop,
    [
        0.21557895,
        0.41663158,
        0.277263158,
        0.083578947,
        0.006947368
    ]
);

/// Evaluates `w[i]=alpha-(1-alpha) cos(2π i/(n-1))`.
///
/// ## Parameters
///
/// `pub fn general_hamming(m: usize, alpha: f64, symmetric: bool) -> Result<Vec<f64>>`
///
/// alpha is a finite real coefficient. symmetric=true uses M points, false truncates M+1. Returns m weights.
///
/// ## Domain
///
/// Fallible functions reject NaN and infinity in inputs. Input slices may be empty only where the definition is meaningful. Window lengths zero and one yield `[]` and `[1]`, unless a stricter domain is stated.
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
/// use sci_signal::windows::*;
/// let _result = general_hamming(5,0.54,true)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy general_hamming definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.general_hamming.html).
pub fn general_hamming(m: usize, alpha: f64, symmetric: bool) -> Result<Vec<f64>> {
    if !alpha.is_finite() {
        return Err(Error::InvalidInput("alpha must be finite"));
    }
    general_cosine(m, &[alpha, 1. - alpha], symmetric)
}
/// Evaluates `w[i]=exp(-0.5 ((i-(n-1)/2)/std)²)`.
///
/// ## Parameters
///
/// `pub fn gaussian(m:usize,std:f64,symmetric:bool)->Result<Vec<f64>>`
///
/// std is a positive standard deviation in samples; symmetric=false truncates M+1 points. Returns m weights.
///
/// ## Domain
///
/// Fallible functions reject NaN and infinity in inputs. Input slices may be empty only where the definition is meaningful. Window lengths zero and one yield `[]` and `[1]`, unless a stricter domain is stated.
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
/// use sci_signal::windows::*;
/// let _result = gaussian(5,1.,true)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy gaussian definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.gaussian.html).
pub fn gaussian(m: usize, std: f64, symmetric: bool) -> Result<Vec<f64>> {
    positive(std)?;
    evaluate(m, symmetric, |i, n| {
        (-0.5 * ((i - (n - 1.) / 2.) / std).powi(2)).exp()
    })
}
/// Evaluates `w[i]=exp(-0.5 |(i-(n-1)/2)/sigma|^(2p))`.
///
/// ## Parameters
///
/// `pub fn general_gaussian(m:usize,p:f64,sigma:f64,symmetric:bool)->Result<Vec<f64>>`
///
/// p and sigma are positive, sigma in samples. symmetric=false truncates M+1 points. Returns m weights.
///
/// ## Domain
///
/// Fallible functions reject NaN and infinity in inputs. Input slices may be empty only where the definition is meaningful. Window lengths zero and one yield `[]` and `[1]`, unless a stricter domain is stated.
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
/// use sci_signal::windows::*;
/// let _result = general_gaussian(5,1.5,2.,true)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy general_gaussian definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.general_gaussian.html).
pub fn general_gaussian(m: usize, p: f64, sigma: f64, symmetric: bool) -> Result<Vec<f64>> {
    positive(p)?;
    positive(sigma)?;
    evaluate(m, symmetric, |i, n| {
        (-0.5 * ((i - (n - 1.) / 2.) / sigma).abs().powf(2. * p)).exp()
    })
}
/// Evaluates `w[i]=exp(-|i-center|/tau)`.
///
/// ## Parameters
///
/// `pub fn exponential(m:usize,center:Option<f64>,tau:f64,symmetric:bool)->Result<Vec<f64>>`
///
/// tau is positive and measured in samples. center=None selects (n-1)/2. Explicit centers require symmetric=false; periodic mode uses n=M+1. Returns m weights.
///
/// ## Domain
///
/// Fallible functions reject NaN and infinity in inputs. Input slices may be empty only where the definition is meaningful. Window lengths zero and one yield `[]` and `[1]`, unless a stricter domain is stated.
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
/// use sci_signal::windows::*;
/// let _result = exponential(5,None,1.,true)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy exponential definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.exponential.html).
pub fn exponential(m: usize, center: Option<f64>, tau: f64, symmetric: bool) -> Result<Vec<f64>> {
    positive(tau)?;
    if center.is_some_and(|c| !c.is_finite()) || (symmetric && center.is_some()) {
        return Err(Error::InvalidInput(
            "symmetric exponential requires default center",
        ));
    }
    evaluate(m, symmetric, |i, n| {
        (-(i - center.unwrap_or((n - 1.) / 2.)).abs() / tau).exp()
    })
}
/// Evaluates a cosine-tapered boxcar with taper fraction alpha.
///
/// ## Parameters
///
/// `pub fn tukey(m:usize,alpha:f64,symmetric:bool)->Result<Vec<f64>>`
///
/// alpha<=0 yields boxcar; alpha>=1 yields Hann. Other finite alpha values taper both ends by alpha/2 of the interval. symmetric=false truncates M+1 points. Returns m weights.
///
/// ## Domain
///
/// Fallible functions reject NaN and infinity in inputs. Input slices may be empty only where the definition is meaningful. Window lengths zero and one yield `[]` and `[1]`, unless a stricter domain is stated.
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
/// use sci_signal::windows::*;
/// let _result = tukey(5,0.5,true)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy tukey definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.tukey.html).
pub fn tukey(m: usize, alpha: f64, symmetric: bool) -> Result<Vec<f64>> {
    if !alpha.is_finite() {
        return Err(Error::InvalidInput("alpha must be finite"));
    }
    if alpha <= 0. {
        return boxcar(m, symmetric);
    }
    if alpha >= 1. {
        return hann(m, symmetric);
    }
    evaluate(m, symmetric, |i, n| {
        let x = i / (n - 1.);
        if x < alpha / 2. {
            0.5 * (1. + (PI * (2. * x / alpha - 1.)).cos())
        } else if x > 1. - alpha / 2. {
            0.5 * (1. + (PI * (2. * x / alpha - 2. / alpha + 1.)).cos())
        } else {
            1.
        }
    })
}
/// Evaluates `w[i]=I0(beta sqrt(1-(2i/(n-1)-1)²))/I0(beta)`.
///
/// ## Parameters
///
/// `pub fn kaiser(m:usize,beta:f64,symmetric:bool)->Result<Vec<f64>>`
///
/// beta is finite and dimensionless. symmetric=false truncates M+1 points. Returns m weights. Scaled Bessel functions avoid overflow for large beta.
///
/// ## Domain
///
/// Fallible functions reject NaN and infinity in inputs. Input slices may be empty only where the definition is meaningful. Window lengths zero and one yield `[]` and `[1]`, unless a stricter domain is stated.
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
/// use sci_signal::windows::*;
/// let _result = kaiser(5,4.,true)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy kaiser definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.kaiser.html).
pub fn kaiser(m: usize, beta: f64, symmetric: bool) -> Result<Vec<f64>> {
    if !beta.is_finite() {
        return Err(Error::InvalidInput("beta must be finite"));
    }
    let b = beta.abs();
    evaluate(m, symmetric, |i, n| {
        let x = 2. * i / (n - 1.) - 1.;
        let a = b * (1. - x * x).max(0.).sqrt();
        crate::special::i0e(a) / crate::special::i0e(b) * (a - b).exp()
    })
}
/// Evaluates KBD weights `sqrt(cumsum(kaiser(M/2+1))/sum(kaiser(M/2+1)))` mirrored about the midpoint.
///
/// ## Parameters
///
/// `pub fn kaiser_bessel_derived(m:usize,beta:f64,symmetric:bool)->Result<Vec<f64>>`
///
/// m must be even, symmetric must be true, beta finite. Returns m samples satisfying paired square sums approximately one. Length zero returns empty.
///
/// ## Domain
///
/// Fallible functions reject NaN and infinity in inputs. Input slices may be empty only where the definition is meaningful. Window lengths zero and one yield `[]` and `[1]`, unless a stricter domain is stated.
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
/// use sci_signal::windows::*;
/// let _result = kaiser_bessel_derived(4,4.,true)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy kaiser_bessel_derived definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.kaiser_bessel_derived.html).
pub fn kaiser_bessel_derived(m: usize, beta: f64, symmetric: bool) -> Result<Vec<f64>> {
    if !symmetric || !m.is_multiple_of(2) {
        return Err(Error::InvalidInput("KBD requires symmetric even length"));
    }
    size(m)?;
    if m == 0 {
        return Ok(vec![]);
    }
    let taper = kaiser(m / 2 + 1, beta, true)?;
    let sum = taper.iter().sum::<f64>();
    let mut cumulative = 0.;
    let mut result = Vec::with_capacity(m);
    for &v in taper.iter().take(m / 2) {
        cumulative += v;
        result.push((cumulative / sum).sqrt());
    }
    result.extend(result.clone().into_iter().rev());
    checked(result)
}
/// Constructs the Dolph-Chebyshev window from `T_(n-1)(beta cos(π k/n))`, `beta=cosh(acosh(10^(|attenuation|/20))/(n-1))`.
///
/// ## Parameters
///
/// `pub fn chebwin(m:usize,attenuation:f64,symmetric:bool)->Result<Vec<f64>>`
///
/// Attenuation is in dB with magnitude <=600; inverse-transform rearrangement uses SciPy conventions. symmetric=false truncates M+1 points. Returns m weights normalized to the sampled maximum. Low attenuation can have nonmonotone noise bandwidth.
///
/// ## Domain
///
/// Fallible functions reject NaN and infinity in inputs. Input slices may be empty only where the definition is meaningful. Window lengths zero and one yield `[]` and `[1]`, unless a stricter domain is stated.
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
/// use sci_signal::windows::*;
/// let _result = chebwin(5,60.,true)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy chebwin definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.chebwin.html).
pub fn chebwin(m: usize, attenuation: f64, symmetric: bool) -> Result<Vec<f64>> {
    if !attenuation.is_finite() || attenuation.abs() > 600. {
        return Err(Error::InvalidInput(
            "attenuation must be finite and at most 600 dB",
        ));
    }
    let n = length(m, symmetric)?;
    if m <= 1 {
        return Ok(vec![1.; m]);
    }
    let order = (n - 1) as f64;
    let beta = (10_f64.powf(attenuation.abs() / 20.).acosh() / order).cosh();
    let spectrum: Vec<_> = (0..n)
        .map(|k| {
            let x = beta * (PI * k as f64 / n as f64).cos();
            let p = if x > 1. {
                (order * x.acosh()).cosh()
            } else if x < -1. {
                (if n.is_multiple_of(2) { -1. } else { 1. }) * (order * (-x).acosh()).cosh()
            } else {
                (order * x.acos()).cos()
            };
            owned(num_complex::Complex64::from_polar(
                p,
                if n.is_multiple_of(2) {
                    PI * k as f64 / n as f64
                } else {
                    0.
                },
            ))
        })
        .collect();
    let f = super::fft(&spectrum);
    let mut result = Vec::with_capacity(n);
    if n % 2 == 1 {
        for k in (1..n.div_ceil(2)).rev() {
            result.push(f[k].re);
        }
        result.extend(f.iter().take(n.div_ceil(2)).map(|v| v.re));
    } else {
        for k in (1..=n / 2).rev() {
            result.push(f[k].re);
        }
        result.extend(f.iter().take(n / 2 + 1).skip(1).map(|v| v.re));
    }
    let max = result.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    result.iter_mut().for_each(|v| *v /= max);
    result.truncate(m);
    checked(result)
}
/// Evaluates the Taylor cosine series `w[i]=(1+2 sum_k F_k cos(2π k(i-n/2+1/2)/n))/scale`.
///
/// ## Parameters
///
/// `pub fn taylor(m:usize,nbar:usize,sll:f64,normalize:bool,symmetric:bool)->Result<Vec<f64>>`
///
/// nbar controls nearly constant sidelobes (1..128); sll is positive suppression <=600 dB. normalize uses continuous midpoint peak; otherwise DC series coefficient is one. symmetric=false truncates M+1 points. Returns m weights.
///
/// ## Domain
///
/// Fallible functions reject NaN and infinity in inputs. Input slices may be empty only where the definition is meaningful. Window lengths zero and one yield `[]` and `[1]`, unless a stricter domain is stated.
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
/// use sci_signal::windows::*;
/// let _result = taylor(5,4,30.,true,true)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy taylor definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.taylor.html).
pub fn taylor(
    m: usize,
    nbar: usize,
    sll: f64,
    normalize: bool,
    symmetric: bool,
) -> Result<Vec<f64>> {
    positive(sll)?;
    if nbar == 0 || nbar > 128 || sll > 600. {
        return Err(Error::InvalidInput(
            "nbar must be 1..128 and sll at most 600 dB",
        ));
    }
    work(m, nbar)?;
    let a = 10_f64.powf(sll / 20.).acosh() / PI;
    let s2 = (nbar as f64).powi(2) / (a * a + (nbar as f64 - 0.5).powi(2));
    let coefficients: Vec<_> = (1..nbar)
        .map(|k| {
            let k2 = (k as f64).powi(2);
            let numerator = (1..nbar)
                .map(|j| 1. - k2 / s2 / (a * a + (j as f64 - 0.5).powi(2)))
                .product::<f64>();
            let denominator = 2.
                * (1..nbar)
                    .filter(|&j| j != k)
                    .map(|j| 1. - k2 / (j as f64).powi(2))
                    .product::<f64>();
            (if k % 2 == 1 { 1. } else { -1. }) * numerator / denominator
        })
        .collect();
    let scale = if normalize {
        1. + 2. * coefficients.iter().sum::<f64>()
    } else {
        1.
    };
    evaluate(m, symmetric, |i, n| {
        (1. + 2.
            * coefficients
                .iter()
                .enumerate()
                .map(|(k, &a)| a * (TAU * (k + 1) as f64 * (i - n / 2. + 0.5) / n).cos())
                .sum::<f64>())
            / scale
    })
}

/// Computes the highest eigenvectors of the Slepian tridiagonal matrix with diagonal `((n-1-2i)/2)² cos(2π NW/n)` and off-diagonal `i(n-i)/2`.
///
/// ## Parameters
///
/// `pub fn dpss(m:usize,nw:f64,k:usize,symmetric:bool)->Result<Vec<Vec<f64>>>`
///
/// m=2..256, 0<NW<m/2, k=1..m. Returns k rows of m samples, ordered by decreasing eigenvalue, L2-normalized before periodic truncation. Even tapers have positive sums, odd tapers start positive. This supports SciPy Kmax=k,norm=2; concentration ratios and singleton approximate normalization are unsupported.
///
/// ## Domain
///
/// Fallible functions reject NaN and infinity in inputs. Input slices may be empty only where the definition is meaningful. Window lengths zero and one yield `[]` and `[1]`, unless a stricter domain is stated.
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
/// use sci_signal::windows::*;
/// let _result = dpss(5,1.,2,true)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy dpss definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.dpss.html).
pub fn dpss(m: usize, nw: f64, k: usize, symmetric: bool) -> Result<Vec<Vec<f64>>> {
    if !(2..=256).contains(&m)
        || k == 0
        || k > m
        || !nw.is_finite()
        || nw <= 0.
        || nw >= m as f64 / 2.
    {
        return Err(Error::InvalidInput(
            "DPSS requires 2..256 samples, 1..m tapers, 0 < NW < m/2",
        ));
    }
    let n = length(m, symmetric)?;
    let w = nw / n as f64;
    let mut matrix = nalgebra::DMatrix::zeros(n, n);
    for i in 0..n {
        matrix[(i, i)] = ((n as f64 - 1. - 2. * i as f64) / 2.).powi(2) * (TAU * w).cos();
        if i > 0 {
            let v = (i * (n - i)) as f64 / 2.;
            matrix[(i, i - 1)] = v;
            matrix[(i - 1, i)] = v;
        }
    }
    let eig = nalgebra::linalg::SymmetricEigen::try_new(matrix, 1e-14, 100_000)
        .ok_or(Error::NumericalFailure("DPSS eigensolver did not converge"))?;
    let mut indices: Vec<_> = (0..n).collect();
    indices.sort_by(|&a, &b| eig.eigenvalues[b].total_cmp(&eig.eigenvalues[a]));
    indices
        .into_iter()
        .take(k)
        .enumerate()
        .map(|(order, j)| {
            let mut v: Vec<_> = eig.eigenvectors.column(j).iter().copied().collect();
            let sign = if order.is_multiple_of(2) {
                v.iter().sum::<f64>()
            } else {
                v.iter().copied().find(|v| v.abs() > 1e-7).unwrap_or(1.)
            };
            if sign < 0. {
                v.iter_mut().for_each(|x| *x = -*x);
            }
            v.truncate(m);
            checked(v)
        })
        .collect()
}

/// A statically typed window specification for [`get_window`].
#[derive(Clone, Debug, PartialEq)]
pub enum Window {
    /// Constant weights.
    Boxcar,
    /// Raised cosine with coefficients 0.5, 0.5.
    Hann,
    /// Raised cosine with coefficients 0.54, 0.46.
    Hamming,
    /// Three-term Blackman cosine sum.
    Blackman,
    /// Four-term Blackman-Harris cosine sum.
    BlackmanHarris,
    /// Four-term Nuttall cosine sum.
    Nuttall,
    /// Five-term flat-top cosine sum.
    FlatTop,
    /// Triangular, zero endpoints.
    Bartlett,
    /// Triangular, nonzero endpoints.
    Triang,
    /// Bartlett-Hann taper.
    Barthann,
    /// Bohman taper.
    Bohman,
    /// Cubic Parzen taper.
    Parzen,
    /// Sine taper.
    Cosine,
    /// Normalized sinc taper.
    Lanczos,
    /// Kaiser beta.
    Kaiser(f64),
    /// Gaussian standard deviation in samples.
    Gaussian(f64),
    /// General Gaussian exponent and width.
    GeneralGaussian(f64, f64),
    /// Tukey taper fraction.
    Tukey(f64),
    /// Exponential decay time in samples, default midpoint center.
    Exponential(f64),
    /// Dolph-Chebyshev attenuation in dB.
    Chebwin(f64),
    /// Taylor sidelobe count, suppression in dB, and peak normalization.
    Taylor(usize, f64, bool),
    /// Generalized Hamming alpha.
    GeneralHamming(f64),
    /// General cosine coefficients.
    GeneralCosine(Vec<f64>),
}
/// Dispatches a typed Window to its mathematical taper definition.
///
/// ## Parameters
///
/// `pub fn get_window(window:&Window,m:usize,periodic:bool)->Result<Vec<f64>>`
///
/// periodic=true truncates M+1 symmetric points, corresponding to SciPy fftbins=true. Returns m weights. String aliases, device selection, DPSS singleton, and KBD dispatch are unsupported; DPSS and KBD have explicit functions.
///
/// ## Domain
///
/// Fallible functions reject NaN and infinity in inputs. Input slices may be empty only where the definition is meaningful. Window lengths zero and one yield `[]` and `[1]`, unless a stricter domain is stated.
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
/// use sci_signal::windows::*;
/// let _result = get_window(&Window::Hann,5,true)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy get_window definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.get_window.html).
pub fn get_window(window: &Window, m: usize, periodic: bool) -> Result<Vec<f64>> {
    let s = !periodic;
    match window {
        Window::Boxcar => boxcar(m, s),
        Window::Hann => hann(m, s),
        Window::Hamming => hamming(m, s),
        Window::Blackman => blackman(m, s),
        Window::BlackmanHarris => blackmanharris(m, s),
        Window::Nuttall => nuttall(m, s),
        Window::FlatTop => flattop(m, s),
        Window::Bartlett => bartlett(m, s),
        Window::Triang => triang(m, s),
        Window::Barthann => barthann(m, s),
        Window::Bohman => bohman(m, s),
        Window::Parzen => parzen(m, s),
        Window::Cosine => cosine(m, s),
        Window::Lanczos => lanczos(m, s),
        Window::Kaiser(b) => kaiser(m, *b, s),
        Window::Gaussian(a) => gaussian(m, *a, s),
        Window::GeneralGaussian(p, a) => general_gaussian(m, *p, *a, s),
        Window::Tukey(a) => tukey(m, *a, s),
        Window::Exponential(t) => exponential(m, None, *t, s),
        Window::Chebwin(a) => chebwin(m, *a, s),
        Window::Taylor(n, a, norm) => taylor(m, *n, *a, *norm, s),
        Window::GeneralHamming(a) => general_hamming(m, *a, s),
        Window::GeneralCosine(a) => general_cosine(m, a, s),
    }
}
