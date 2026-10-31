use super::{TransferFunction, checked, complex_checked, nonempty, positive, samples, work};
use crate::{
    Error, Result,
    complex::{Complex, native, owned},
};
use num_complex::Complex64;
use std::f64::consts::{PI, TAU};

/// A transfer function `H(v)=gain product(v-zero)/product(v-pole)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Zpk {
    /// Finite zeros, in caller or prototype order.
    pub zeros: Vec<Complex>,
    /// Finite poles, in caller or prototype order.
    pub poles: Vec<Complex>,
    /// Real multiplicative gain.
    pub gain: f64,
}
/// Passband arrangement for classical IIR filter design.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BandType {
    /// Pass frequencies below one cutoff.
    Lowpass,
    /// Pass frequencies above one cutoff.
    Highpass,
    /// Pass frequencies between two cutoffs.
    Bandpass,
    /// Reject frequencies between two cutoffs.
    Bandstop,
}
pub(super) fn order(n: usize) -> Result<()> {
    if n > 64 {
        Err(Error::InvalidInput("filter order exceeds 64"))
    } else {
        Ok(())
    }
}
pub(super) fn validate(zpk: &Zpk) -> Result<usize> {
    if zpk.zeros.len() > zpk.poles.len() || zpk.poles.len() > 128 || !zpk.gain.is_finite() {
        return Err(Error::InvalidInput(
            "ZPK must be proper, finite, and have at most 128 poles",
        ));
    }
    if zpk
        .zeros
        .iter()
        .chain(&zpk.poles)
        .any(|z| !z.re.is_finite() || !z.im.is_finite())
    {
        return Err(Error::InvalidInput("roots must be finite"));
    }
    Ok(zpk.poles.len() - zpk.zeros.len())
}
fn result(zpk: Zpk) -> Result<Zpk> {
    if !zpk.gain.is_finite() {
        return Err(Error::NonFiniteEvaluation);
    }
    super::check_complex(&zpk.zeros)?;
    super::check_complex(&zpk.poles)?;
    Ok(zpk)
}
fn sorted_result(mut zpk: Zpk) -> Result<Zpk> {
    let compare = |a: &Complex, b: &Complex| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im));
    zpk.zeros.sort_by(compare);
    zpk.poles.sort_by(compare);
    result(zpk)
}
fn product(roots: &[Complex], f: impl Fn(Complex64) -> Complex64) -> Complex64 {
    roots
        .iter()
        .fold(Complex64::new(1., 0.), |p, &z| p * f(native(z)))
}

/// Returns Butterworth prototype poles `p[j]=-exp(iπ(2j-n+1)/(2n))`, no zeros, and unit gain.
///
/// ## Parameters
///
/// n=0..64; analog lowpass cutoff is 1 radian/time. Pole order follows SciPy.
///
/// ## Domain
///
/// Inputs must be finite. ZPK inputs have at most 128 poles and no more zeros than poles. No implicit selection of frequencies, units, branches or output representations is performed.
///
/// ## Returns
///
/// Output shapes and units follow the conventions above. Transformed roots are sorted by real then imaginary components.
///
/// ## Errors
///
/// Returns shared Error for invalid domains, non-real coefficients, singular transformations, solver failure, or arithmetic overflow. Exact response zeros give negative infinity dB in Bode functions; response poles are errors.
///
/// ## Numerical behavior
///
/// Uses explicit f64 prototype formulas and rational transformations. Polynomial factorization uses the private nalgebra Schur solver with at most 100000 iterations. High-order coefficient representations and repeated roots are ill-conditioned; use root responses when available. No uniform relative accuracy guarantee is made.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result = buttap(3)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy buttap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.buttap.html).
pub fn buttap(n: usize) -> Result<Zpk> {
    order(n)?;
    let poles = (0..n)
        .map(|j| {
            let m = 2. * j as f64 - n as f64 + 1.;
            owned(-Complex64::from_polar(1., PI * m / (2. * n as f64)))
        })
        .collect();
    Ok(Zpk {
        zeros: vec![],
        poles,
        gain: 1.,
    })
}
/// Returns Chebyshev I prototype `p[j]=-sinh(mu+iπ(2j-n+1)/(2n))`, `mu=asinh(1/epsilon)/n`, `epsilon=sqrt(10^(ripple/10)-1)`.
///
/// ## Parameters
///
/// n=0..64; ripple is positive passband ripple in dB. Gain is product(-p), divided by sqrt(1+epsilon²) for even order. Cutoff is 1 radian/time.
///
/// ## Domain
///
/// Inputs must be finite. ZPK inputs have at most 128 poles and no more zeros than poles. No implicit selection of frequencies, units, branches or output representations is performed.
///
/// ## Returns
///
/// Output shapes and units follow the conventions above. Transformed roots are sorted by real then imaginary components.
///
/// ## Errors
///
/// Returns shared Error for invalid domains, non-real coefficients, singular transformations, solver failure, or arithmetic overflow. Exact response zeros give negative infinity dB in Bode functions; response poles are errors.
///
/// ## Numerical behavior
///
/// Uses explicit f64 prototype formulas and rational transformations. Polynomial factorization uses the private nalgebra Schur solver with at most 100000 iterations. High-order coefficient representations and repeated roots are ill-conditioned; use root responses when available. No uniform relative accuracy guarantee is made.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result = cheb1ap(3,1.)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy cheb1ap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheb1ap.html).
pub fn cheb1ap(n: usize, ripple: f64) -> Result<Zpk> {
    order(n)?;
    positive(ripple)?;
    if n == 0 {
        return result(Zpk {
            zeros: vec![],
            poles: vec![],
            gain: 10_f64.powf(-ripple / 20.),
        });
    }
    let epsilon = (10_f64.powf(ripple / 10.) - 1.).sqrt();
    let mu = (1. / epsilon).asinh() / n as f64;
    let poles: Vec<_> = (0..n)
        .map(|j| {
            let m = 2. * j as f64 - n as f64 + 1.;
            owned(-Complex64::new(mu, PI * m / (2. * n as f64)).sinh())
        })
        .collect();
    let gain = product(&poles, |p| -p).re
        / if n.is_multiple_of(2) {
            (1. + epsilon * epsilon).sqrt()
        } else {
            1.
        };
    result(Zpk {
        zeros: vec![],
        poles,
        gain,
    })
}
/// Returns Chebyshev II prototype poles obtained by reciprocating the sinh/cosh-scaled Butterworth poles.
///
/// ## Parameters
///
/// n=0..64; attenuation is positive stopband suppression in dB. Zeros are i/sin(πm/(2n)) for nonzero odd-step m. Gain is Re(product(-p)/product(-z)); cutoff is 1 radian/time.
///
/// ## Domain
///
/// Inputs must be finite. ZPK inputs have at most 128 poles and no more zeros than poles. No implicit selection of frequencies, units, branches or output representations is performed.
///
/// ## Returns
///
/// Output shapes and units follow the conventions above. Transformed roots are sorted by real then imaginary components.
///
/// ## Errors
///
/// Returns shared Error for invalid domains, non-real coefficients, singular transformations, solver failure, or arithmetic overflow. Exact response zeros give negative infinity dB in Bode functions; response poles are errors.
///
/// ## Numerical behavior
///
/// Uses explicit f64 prototype formulas and rational transformations. Polynomial factorization uses the private nalgebra Schur solver with at most 100000 iterations. High-order coefficient representations and repeated roots are ill-conditioned; use root responses when available. No uniform relative accuracy guarantee is made.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result = cheb2ap(3,40.)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy cheb2ap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheb2ap.html).
pub fn cheb2ap(n: usize, attenuation: f64) -> Result<Zpk> {
    order(n)?;
    positive(attenuation)?;
    if n == 0 {
        return Ok(Zpk {
            zeros: vec![],
            poles: vec![],
            gain: 1.,
        });
    }
    let epsilon = 1. / (10_f64.powf(attenuation / 10.) - 1.).sqrt();
    let mu = (1. / epsilon).asinh() / n as f64;
    let mut zeros = vec![];
    let mut poles = vec![];
    for j in 0..n {
        let m = 2. * j as f64 - n as f64 + 1.;
        let angle = PI * m / (2. * n as f64);
        if m != 0. {
            zeros.push(Complex {
                re: 0.,
                im: 1. / angle.sin(),
            });
        }
        let p = -Complex64::from_polar(1., angle);
        poles.push(owned(
            Complex64::new(1., 0.) / Complex64::new(mu.sinh() * p.re, mu.cosh() * p.im),
        ));
    }
    let gain = (product(&poles, |p| -p) / product(&zeros, |z| -z)).re;
    result(Zpk { zeros, poles, gain })
}
/// Substitutes `s -> s/wo`, scaling roots by wo and gain by `wo^(poles-zeros)`.
///
/// ## Parameters
///
/// wo is positive angular cutoff in radians/time; returns analog roots sorted by real then imaginary components.
///
/// ## Domain
///
/// Inputs must be finite. ZPK inputs have at most 128 poles and no more zeros than poles. No implicit selection of frequencies, units, branches or output representations is performed.
///
/// ## Returns
///
/// Output shapes and units follow the conventions above. Transformed roots are sorted by real then imaginary components.
///
/// ## Errors
///
/// Returns shared Error for invalid domains, non-real coefficients, singular transformations, solver failure, or arithmetic overflow. Exact response zeros give negative infinity dB in Bode functions; response poles are errors.
///
/// ## Numerical behavior
///
/// Uses explicit f64 prototype formulas and rational transformations. Polynomial factorization uses the private nalgebra Schur solver with at most 100000 iterations. High-order coefficient representations and repeated roots are ill-conditioned; use root responses when available. No uniform relative accuracy guarantee is made.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result = lp2lp_zpk(&buttap(3)?,2.)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy lp2lp_zpk](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lp2lp_zpk.html).
pub fn lp2lp_zpk(zpk: &Zpk, wo: f64) -> Result<Zpk> {
    let degree = validate(zpk)?;
    positive(wo)?;
    sorted_result(Zpk {
        zeros: zpk.zeros.iter().map(|&z| owned(native(z) * wo)).collect(),
        poles: zpk.poles.iter().map(|&p| owned(native(p) * wo)).collect(),
        gain: zpk.gain * wo.powi(degree as i32),
    })
}
/// Substitutes `s -> wo/s`, reciprocating roots, adding zeros at zero, and adjusting gain.
///
/// ## Parameters
///
/// wo is positive angular cutoff in radians/time. Nonzero roots are required by this finite-root representation.
///
/// ## Domain
///
/// Inputs must be finite. ZPK inputs have at most 128 poles and no more zeros than poles. No implicit selection of frequencies, units, branches or output representations is performed.
///
/// ## Returns
///
/// Output shapes and units follow the conventions above. Transformed roots are sorted by real then imaginary components.
///
/// ## Errors
///
/// Returns shared Error for invalid domains, non-real coefficients, singular transformations, solver failure, or arithmetic overflow. Exact response zeros give negative infinity dB in Bode functions; response poles are errors.
///
/// ## Numerical behavior
///
/// Uses explicit f64 prototype formulas and rational transformations. Polynomial factorization uses the private nalgebra Schur solver with at most 100000 iterations. High-order coefficient representations and repeated roots are ill-conditioned; use root responses when available. No uniform relative accuracy guarantee is made.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result = lp2hp_zpk(&buttap(3)?,2.)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy lp2hp_zpk](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lp2hp_zpk.html).
pub fn lp2hp_zpk(zpk: &Zpk, wo: f64) -> Result<Zpk> {
    let degree = validate(zpk)?;
    positive(wo)?;
    if zpk
        .zeros
        .iter()
        .chain(&zpk.poles)
        .any(|z| z.re == 0. && z.im == 0.)
    {
        return Err(Error::InvalidInput(
            "highpass transformation cannot invert zero roots",
        ));
    }
    let mut zeros: Vec<_> = zpk.zeros.iter().map(|&z| owned(wo / native(z))).collect();
    zeros.extend(vec![Complex::default(); degree]);
    sorted_result(Zpk {
        zeros,
        poles: zpk.poles.iter().map(|&p| owned(wo / native(p))).collect(),
        gain: zpk.gain * (product(&zpk.zeros, |z| -z) / product(&zpk.poles, |p| -p)).re,
    })
}
fn canonical_sqrt(v: Complex64) -> Complex64 {
    Complex64::new(v.re, if v.im == 0. { 0. } else { v.im }).sqrt()
}
fn band_roots(roots: &[Complex], wo: f64, bw: f64, invert: bool) -> Vec<Complex> {
    let half: Vec<_> = roots
        .iter()
        .map(|&z| {
            if invert {
                (bw / 2.) / native(z)
            } else {
                native(z) * (bw / 2.)
            }
        })
        .collect();
    half.iter()
        .map(|&v| owned(v + canonical_sqrt(v * v - wo * wo)))
        .chain(
            half.iter()
                .map(|&v| owned(v - canonical_sqrt(v * v - wo * wo))),
        )
        .collect()
}
/// Substitutes `s -> (s²+wo²)/(bw s)`; roots become `v ± sqrt(v²-wo²)`, `v=bw root/2`.
///
/// ## Parameters
///
/// wo is center angular frequency and bw is positive angular bandwidth. Adds relative-degree zeros at zero and scales gain by bw^degree. Roots are sorted by real then imaginary components. Input has at most 64 poles.
///
/// ## Domain
///
/// Inputs must be finite. ZPK inputs have at most 128 poles and no more zeros than poles. No implicit selection of frequencies, units, branches or output representations is performed.
///
/// ## Returns
///
/// Output shapes and units follow the conventions above. Transformed roots are sorted by real then imaginary components.
///
/// ## Errors
///
/// Returns shared Error for invalid domains, non-real coefficients, singular transformations, solver failure, or arithmetic overflow. Exact response zeros give negative infinity dB in Bode functions; response poles are errors.
///
/// ## Numerical behavior
///
/// Uses explicit f64 prototype formulas and rational transformations. Polynomial factorization uses the private nalgebra Schur solver with at most 100000 iterations. High-order coefficient representations and repeated roots are ill-conditioned; use root responses when available. No uniform relative accuracy guarantee is made.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result = lp2bp_zpk(&buttap(3)?,2.,1.)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy lp2bp_zpk](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lp2bp_zpk.html).
pub fn lp2bp_zpk(zpk: &Zpk, wo: f64, bw: f64) -> Result<Zpk> {
    let degree = validate(zpk)?;
    positive(wo)?;
    positive(bw)?;
    if zpk.poles.len() > 64 {
        return Err(Error::InvalidInput(
            "band transform doubles order beyond 128",
        ));
    }
    let mut zeros = band_roots(&zpk.zeros, wo, bw, false);
    zeros.extend(vec![Complex::default(); degree]);
    sorted_result(Zpk {
        zeros,
        poles: band_roots(&zpk.poles, wo, bw, false),
        gain: zpk.gain * bw.powi(degree as i32),
    })
}
/// Substitutes `s -> bw s/(s²+wo²)`; roots become `v ± sqrt(v²-wo²)`, `v=bw/(2 root)`.
///
/// ## Parameters
///
/// wo and bw are positive angular frequencies; adds zeros at ±i wo, and gain is multiplied by Re(product(-z)/product(-p)). Nonzero roots and at most 64 poles are required.
///
/// ## Domain
///
/// Inputs must be finite. ZPK inputs have at most 128 poles and no more zeros than poles. No implicit selection of frequencies, units, branches or output representations is performed.
///
/// ## Returns
///
/// Output shapes and units follow the conventions above. Transformed roots are sorted by real then imaginary components.
///
/// ## Errors
///
/// Returns shared Error for invalid domains, non-real coefficients, singular transformations, solver failure, or arithmetic overflow. Exact response zeros give negative infinity dB in Bode functions; response poles are errors.
///
/// ## Numerical behavior
///
/// Uses explicit f64 prototype formulas and rational transformations. Polynomial factorization uses the private nalgebra Schur solver with at most 100000 iterations. High-order coefficient representations and repeated roots are ill-conditioned; use root responses when available. No uniform relative accuracy guarantee is made.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result = lp2bs_zpk(&buttap(3)?,2.,1.)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy lp2bs_zpk](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lp2bs_zpk.html).
pub fn lp2bs_zpk(zpk: &Zpk, wo: f64, bw: f64) -> Result<Zpk> {
    let degree = validate(zpk)?;
    positive(wo)?;
    positive(bw)?;
    if zpk.poles.len() > 64
        || zpk
            .zeros
            .iter()
            .chain(&zpk.poles)
            .any(|z| z.re == 0. && z.im == 0.)
    {
        return Err(Error::InvalidInput(
            "bandstop transform requires order <=64 and nonzero roots",
        ));
    }
    let mut zeros = band_roots(&zpk.zeros, wo, bw, true);
    zeros.extend(vec![Complex { re: 0., im: wo }; degree]);
    zeros.extend(vec![Complex { re: 0., im: -wo }; degree]);
    sorted_result(Zpk {
        zeros,
        poles: band_roots(&zpk.poles, wo, bw, true),
        gain: zpk.gain * (product(&zpk.zeros, |z| -z) / product(&zpk.poles, |p| -p)).re,
    })
}
/// Substitutes `s -> 2 fs (z-1)/(z+1)`, mapping each root r to `(2fs+r)/(2fs-r)`.
///
/// ## Parameters
///
/// fs is positive samples/time. Adds relative-degree zeros at -1; gain is multiplied by Re(product(2fs-z)/product(2fs-p)). Performs no prewarping.
///
/// ## Domain
///
/// Inputs must be finite. ZPK inputs have at most 128 poles and no more zeros than poles. No implicit selection of frequencies, units, branches or output representations is performed.
///
/// ## Returns
///
/// Output shapes and units follow the conventions above. Transformed roots are sorted by real then imaginary components.
///
/// ## Errors
///
/// Returns shared Error for invalid domains, non-real coefficients, singular transformations, solver failure, or arithmetic overflow. Exact response zeros give negative infinity dB in Bode functions; response poles are errors.
///
/// ## Numerical behavior
///
/// Uses explicit f64 prototype formulas and rational transformations. Polynomial factorization uses the private nalgebra Schur solver with at most 100000 iterations. High-order coefficient representations and repeated roots are ill-conditioned; use root responses when available. No uniform relative accuracy guarantee is made.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result = bilinear_zpk(&buttap(3)?,10.)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy bilinear_zpk](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.bilinear_zpk.html).
pub fn bilinear_zpk(zpk: &Zpk, fs: f64) -> Result<Zpk> {
    let degree = validate(zpk)?;
    positive(fs)?;
    let rate = 2. * fs;
    if zpk.poles.iter().any(|z| z.re == rate && z.im == 0.) {
        return Err(Error::InvalidInput("pole maps to infinity"));
    }
    let mut zeros: Vec<_> = zpk
        .zeros
        .iter()
        .map(|&z| owned((rate + native(z)) / (rate - native(z))))
        .collect();
    zeros.extend(vec![Complex { re: -1., im: 0. }; degree]);
    sorted_result(Zpk {
        zeros,
        poles: zpk
            .poles
            .iter()
            .map(|&p| owned((rate + native(p)) / (rate - native(p))))
            .collect(),
        gain: zpk.gain * (product(&zpk.zeros, |z| rate - z) / product(&zpk.poles, |p| rate - p)).re,
    })
}
pub(super) fn polynomial(roots: &[Complex]) -> Result<Vec<f64>> {
    let mut c = vec![Complex64::new(1., 0.)];
    for &z in roots {
        let z = native(z);
        c.push(Complex64::new(0., 0.));
        for i in (1..c.len()).rev() {
            let previous = c[i - 1];
            c[i] -= z * previous;
        }
    }
    let scale = c.iter().map(|v| v.norm()).fold(1., f64::max);
    if c.iter().any(|v| v.im.abs() > 1e-10 * scale) {
        return Err(Error::InvalidInput(
            "roots must define real polynomial coefficients",
        ));
    }
    checked(c.iter().map(|v| v.re).collect())
}
/// Expands `B(v)=gain product(v-zero)` and `A(v)=product(v-pole)`.
///
/// ## Parameters
///
/// Returns (numerator,denominator) in descending polynomial order. Digital proper transfer coefficients may require numerator leading-zero padding before lfilter: lfilter interprets coefficients as ascending delays. Roots must define real coefficients to relative imaginary tolerance 1e-10.
///
/// ## Domain
///
/// Inputs must be finite. ZPK inputs have at most 128 poles and no more zeros than poles. No implicit selection of frequencies, units, branches or output representations is performed.
///
/// ## Returns
///
/// Output shapes and units follow the conventions above. Transformed roots are sorted by real then imaginary components.
///
/// ## Errors
///
/// Returns shared Error for invalid domains, non-real coefficients, singular transformations, solver failure, or arithmetic overflow. Exact response zeros give negative infinity dB in Bode functions; response poles are errors.
///
/// ## Numerical behavior
///
/// Uses explicit f64 prototype formulas and rational transformations. Polynomial factorization uses the private nalgebra Schur solver with at most 100000 iterations. High-order coefficient representations and repeated roots are ill-conditioned; use root responses when available. No uniform relative accuracy guarantee is made.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result = zpk2tf(&buttap(3)?)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy zpk2tf](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.zpk2tf.html).
pub fn zpk2tf(zpk: &Zpk) -> Result<TransferFunction> {
    validate(zpk)?;
    let mut b = polynomial(&zpk.zeros)?;
    for v in &mut b {
        *v *= zpk.gain;
    }
    Ok((checked(b)?, polynomial(&zpk.poles)?))
}
pub(super) fn roots(coefficients: &[f64]) -> Result<Vec<Complex>> {
    let start = coefficients
        .iter()
        .position(|v| *v != 0.)
        .unwrap_or(coefficients.len());
    if start == coefficients.len() {
        return Ok(vec![]);
    }
    let c = &coefficients[start..];
    let n = c.len() - 1;
    if n == 0 {
        return Ok(vec![]);
    }
    if n > 128 {
        return Err(Error::InvalidInput("polynomial degree exceeds 128"));
    }
    let mut companion = nalgebra::DMatrix::zeros(n, n);
    for i in 0..n {
        companion[(0, i)] = -c[i + 1] / c[0];
        if i > 0 {
            companion[(i, i - 1)] = 1.;
        }
    }
    if companion.iter().any(|v| !v.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    let schur = nalgebra::linalg::Schur::try_new(companion, 1e-14, 100_000)
        .ok_or(Error::NumericalFailure("polynomial eigensolver failed"))?;
    let mut values: Vec<_> = schur
        .complex_eigenvalues()
        .iter()
        .copied()
        .map(owned)
        .collect();
    values.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im)));
    complex_checked(values)
}
/// Divides numerator and denominator by the first nonzero denominator coefficient.
///
/// ## Parameters
///
/// Returns (numerator,denominator) in descending powers. Removes leading denominator zeros and numerator coefficients with normalized magnitude <=1e-14, retaining at least one numerator coefficient, following SciPy warning-trimming behavior without warnings.
///
/// ## Domain
///
/// Inputs must be finite. ZPK inputs have at most 128 poles and no more zeros than poles. No implicit selection of frequencies, units, branches or output representations is performed.
///
/// ## Returns
///
/// Output shapes and units follow the conventions above. Transformed roots are sorted by real then imaginary components.
///
/// ## Errors
///
/// Returns shared Error for invalid domains, non-real coefficients, singular transformations, solver failure, or arithmetic overflow. Exact response zeros give negative infinity dB in Bode functions; response poles are errors.
///
/// ## Numerical behavior
///
/// Uses explicit f64 prototype formulas and rational transformations. Polynomial factorization uses the private nalgebra Schur solver with at most 100000 iterations. High-order coefficient representations and repeated roots are ill-conditioned; use root responses when available. No uniform relative accuracy guarantee is made.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result = normalize(&[0.,2.],&[0.,2.,4.])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy normalize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.normalize.html).
pub fn normalize(b: &[f64], a: &[f64]) -> Result<TransferFunction> {
    nonempty(b)?;
    nonempty(a)?;
    let start = a
        .iter()
        .position(|v| *v != 0.)
        .ok_or(Error::InvalidInput("denominator cannot be zero"))?;
    let scale = a[start];
    let denominator = checked(a[start..].iter().map(|v| v / scale).collect())?;
    let numerator = checked(b.iter().map(|v| v / scale).collect())?;
    let first = numerator
        .iter()
        .position(|v| v.abs() > 1e-14)
        .unwrap_or(numerator.len() - 1);
    Ok((numerator[first..].to_vec(), denominator))
}
/// Factors `H(v)=B(v)/A(v)` using companion-matrix eigenvalues.
///
/// ## Parameters
///
/// Input coefficients are descending powers. Returns zeros,poles,gain in Zpk, with roots sorted by real then imaginary components. Requires a proper transfer function, degree <=128. SciPy normalization trimming applies. Repeated roots may be inaccurate.
///
/// ## Domain
///
/// Inputs must be finite. ZPK inputs have at most 128 poles and no more zeros than poles. No implicit selection of frequencies, units, branches or output representations is performed.
///
/// ## Returns
///
/// Output shapes and units follow the conventions above. Transformed roots are sorted by real then imaginary components.
///
/// ## Errors
///
/// Returns shared Error for invalid domains, non-real coefficients, singular transformations, solver failure, or arithmetic overflow. Exact response zeros give negative infinity dB in Bode functions; response poles are errors.
///
/// ## Numerical behavior
///
/// Uses explicit f64 prototype formulas and rational transformations. Polynomial factorization uses the private nalgebra Schur solver with at most 100000 iterations. High-order coefficient representations and repeated roots are ill-conditioned; use root responses when available. No uniform relative accuracy guarantee is made.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result = tf2zpk(&[1.],&[1.,1.])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy tf2zpk](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.tf2zpk.html).
pub fn tf2zpk(b: &[f64], a: &[f64]) -> Result<Zpk> {
    let (b, a) = normalize(b, a)?;
    if b.len() > a.len() {
        return Err(Error::InvalidInput("transfer function must be proper"));
    }
    result(Zpk {
        zeros: roots(&b)?,
        poles: roots(&a)?,
        gain: b[0] / a[0],
    })
}
/// Factors each second-order section and collects its zeros and poles, multiplying gains.
///
/// ## Parameters
///
/// SOS rows are `[b0,b1,b2,1,a1,a2]`. Returns zeros and poles sorted within each section. At most 64 sections are supported; section leading zeros use the SciPy polynomial convention.
///
/// ## Domain
///
/// Inputs must be finite. ZPK inputs have at most 128 poles and no more zeros than poles. No implicit selection of frequencies, units, branches or output representations is performed.
///
/// ## Returns
///
/// Output shapes and units follow the conventions above. Transformed roots are sorted by real then imaginary components.
///
/// ## Errors
///
/// Returns shared Error for invalid domains, non-real coefficients, singular transformations, solver failure, or arithmetic overflow. Exact response zeros give negative infinity dB in Bode functions; response poles are errors.
///
/// ## Numerical behavior
///
/// Uses explicit f64 prototype formulas and rational transformations. Polynomial factorization uses the private nalgebra Schur solver with at most 100000 iterations. High-order coefficient representations and repeated roots are ill-conditioned; use root responses when available. No uniform relative accuracy guarantee is made.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result = sos2zpk(&[[1.,0.,0.,1.,-0.5,0.1]])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy sos2zpk](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sos2zpk.html).
pub fn sos2zpk(sos: &[[f64; 6]]) -> Result<Zpk> {
    // Validate using the shared SOS response without expanding the cascade.
    if sos.len() > 64 {
        return Err(Error::InvalidInput("at most 64 SOS sections supported"));
    }
    super::freqz_sos(sos, &[])?;
    let mut zeros = vec![];
    let mut poles = vec![];
    let mut gain = 1.;
    for row in sos {
        let section = tf2zpk(&row[..3], &row[3..])?;
        zeros.extend(section.zeros);
        poles.extend(section.poles);
        gain *= section.gain;
    }
    result(Zpk { zeros, poles, gain })
}
fn zpk_response(zpk: &Zpk, frequencies: &[f64], digital: bool) -> Result<Vec<Complex>> {
    validate(zpk)?;
    samples(frequencies)?;
    work(frequencies.len(), zpk.zeros.len() + zpk.poles.len())?;
    complex_checked(
        frequencies
            .iter()
            .map(|&w| {
                let v = if digital {
                    Complex64::from_polar(1., w)
                } else {
                    Complex64::new(0., w)
                };
                owned(zpk.gain * product(&zpk.zeros, |z| v - z) / product(&zpk.poles, |p| v - p))
            })
            .collect(),
    )
}
/// Evaluates `H(ω)=gain product(iω-zero)/product(iω-pole)`.
///
/// ## Parameters
///
/// Frequencies are explicit radians/time and preserved in caller order. Returns shared Complex responses only.
///
/// ## Domain
///
/// Inputs must be finite. ZPK inputs have at most 128 poles and no more zeros than poles. No implicit selection of frequencies, units, branches or output representations is performed.
///
/// ## Returns
///
/// Output shapes and units follow the conventions above. Transformed roots are sorted by real then imaginary components.
///
/// ## Errors
///
/// Returns shared Error for invalid domains, non-real coefficients, singular transformations, solver failure, or arithmetic overflow. Exact response zeros give negative infinity dB in Bode functions; response poles are errors.
///
/// ## Numerical behavior
///
/// Uses explicit f64 prototype formulas and rational transformations. Polynomial factorization uses the private nalgebra Schur solver with at most 100000 iterations. High-order coefficient representations and repeated roots are ill-conditioned; use root responses when available. No uniform relative accuracy guarantee is made.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result = freqs_zpk(&buttap(3)?,&[0.,1.])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy freqs_zpk](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqs_zpk.html).
pub fn freqs_zpk(zpk: &Zpk, frequencies: &[f64]) -> Result<Vec<Complex>> {
    zpk_response(zpk, frequencies, false)
}
/// Evaluates `H(ω)=gain product(exp(iω)-zero)/product(exp(iω)-pole)`.
///
/// ## Parameters
///
/// Frequencies are explicit radians/sample and preserved in caller order. Returns shared Complex responses only.
///
/// ## Domain
///
/// Inputs must be finite. ZPK inputs have at most 128 poles and no more zeros than poles. No implicit selection of frequencies, units, branches or output representations is performed.
///
/// ## Returns
///
/// Output shapes and units follow the conventions above. Transformed roots are sorted by real then imaginary components.
///
/// ## Errors
///
/// Returns shared Error for invalid domains, non-real coefficients, singular transformations, solver failure, or arithmetic overflow. Exact response zeros give negative infinity dB in Bode functions; response poles are errors.
///
/// ## Numerical behavior
///
/// Uses explicit f64 prototype formulas and rational transformations. Polynomial factorization uses the private nalgebra Schur solver with at most 100000 iterations. High-order coefficient representations and repeated roots are ill-conditioned; use root responses when available. No uniform relative accuracy guarantee is made.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result = freqz_zpk(&bilinear_zpk(&buttap(3)?,10.)?,&[0.,1.])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy freqz_zpk](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqz_zpk.html).
pub fn freqz_zpk(zpk: &Zpk, frequencies: &[f64]) -> Result<Vec<Complex>> {
    zpk_response(zpk, frequencies, true)
}

pub(super) fn design(
    prototype: Zpk,
    cutoffs: &[f64],
    band: BandType,
    fs: Option<f64>,
) -> Result<TransferFunction> {
    nonempty(cutoffs)?;
    let bands = matches!(band, BandType::Bandpass | BandType::Bandstop);
    if cutoffs.len() != if bands { 2 } else { 1 }
        || cutoffs.iter().any(|v| *v <= 0.)
        || (bands && cutoffs[0] >= cutoffs[1])
    {
        return Err(Error::InvalidInput(
            "cutoff shape must match band and frequencies be increasing positive",
        ));
    }
    let warped = if let Some(fs) = fs {
        positive(fs)?;
        if cutoffs.iter().any(|v| *v >= fs / 2.) {
            return Err(Error::InvalidInput("digital cutoffs must be below Nyquist"));
        }
        cutoffs
            .iter()
            .map(|v| 2. * fs * (PI * v / fs).tan())
            .collect::<Vec<_>>()
    } else {
        cutoffs.to_vec()
    };
    let zpk = match band {
        BandType::Lowpass => lp2lp_zpk(&prototype, warped[0])?,
        BandType::Highpass => lp2hp_zpk(&prototype, warped[0])?,
        BandType::Bandpass => lp2bp_zpk(
            &prototype,
            (warped[0] * warped[1]).sqrt(),
            warped[1] - warped[0],
        )?,
        BandType::Bandstop => lp2bs_zpk(
            &prototype,
            (warped[0] * warped[1]).sqrt(),
            warped[1] - warped[0],
        )?,
    };
    if let Some(fs) = fs {
        zpk2tf(&bilinear_zpk(&zpk, fs)?)
    } else {
        zpk2tf(&zpk)
    }
}
/// Designs the Butterworth filter with prototype `|H(iω)|²=1/(1+ω^(2n))` and cutoff transformations.
///
/// ## Parameters
///
/// n=0..64, band controls cutoff count (one for low/high, two for bandpass/stop). fs=Some(rate) means digital cutoffs in rate units with bilinear prewarping; fs=None means analog cutoffs in radians/time. Returns (b,a), SciPy output=ba; digital coefficients are ascending delays.
///
/// ## Domain
///
/// Inputs must be finite. ZPK inputs have at most 128 poles and no more zeros than poles. No implicit selection of frequencies, units, branches or output representations is performed.
///
/// ## Returns
///
/// Output shapes and units follow the conventions above. Transformed roots are sorted by real then imaginary components.
///
/// ## Errors
///
/// Returns shared Error for invalid domains, non-real coefficients, singular transformations, solver failure, or arithmetic overflow. Exact response zeros give negative infinity dB in Bode functions; response poles are errors.
///
/// ## Numerical behavior
///
/// Uses explicit f64 prototype formulas and rational transformations. Polynomial factorization uses the private nalgebra Schur solver with at most 100000 iterations. High-order coefficient representations and repeated roots are ill-conditioned; use root responses when available. No uniform relative accuracy guarantee is made.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result = butter(3,&[0.2],BandType::Lowpass,Some(2.))?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy butter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html).
pub fn butter(
    n: usize,
    cutoffs: &[f64],
    band: BandType,
    fs: Option<f64>,
) -> Result<TransferFunction> {
    design(buttap(n)?, cutoffs, band, fs)
}
/// Designs the Chebyshev I filter with `|H(iω)|²=1/(1+epsilon² T_n(ω)²)` and cutoff transformations.
///
/// ## Parameters
///
/// n=0..64, positive ripple in dB. Band sets one or two cutoffs; fs=Some(rate) selects digital cutoffs in rate units, fs=None analog radians/time. Returns (b,a), SciPy output=ba; digital coefficients are ascending delays.
///
/// ## Domain
///
/// Inputs must be finite. ZPK inputs have at most 128 poles and no more zeros than poles. No implicit selection of frequencies, units, branches or output representations is performed.
///
/// ## Returns
///
/// Output shapes and units follow the conventions above. Transformed roots are sorted by real then imaginary components.
///
/// ## Errors
///
/// Returns shared Error for invalid domains, non-real coefficients, singular transformations, solver failure, or arithmetic overflow. Exact response zeros give negative infinity dB in Bode functions; response poles are errors.
///
/// ## Numerical behavior
///
/// Uses explicit f64 prototype formulas and rational transformations. Polynomial factorization uses the private nalgebra Schur solver with at most 100000 iterations. High-order coefficient representations and repeated roots are ill-conditioned; use root responses when available. No uniform relative accuracy guarantee is made.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result = cheby1(3,1.,&[0.2],BandType::Lowpass,Some(2.))?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy cheby1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheby1.html).
pub fn cheby1(
    n: usize,
    ripple: f64,
    cutoffs: &[f64],
    band: BandType,
    fs: Option<f64>,
) -> Result<TransferFunction> {
    design(cheb1ap(n, ripple)?, cutoffs, band, fs)
}
/// Designs the Chebyshev II filter with `|H(iω)|²=1/(1+1/(epsilon² T_n(1/ω)²))` and cutoff transformations.
///
/// ## Parameters
///
/// n=0..64, positive attenuation in dB. Band sets one or two cutoffs; fs=Some(rate) selects digital cutoffs in rate units, fs=None analog radians/time. Returns (b,a), SciPy output=ba; digital coefficients are ascending delays.
///
/// ## Domain
///
/// Inputs must be finite. ZPK inputs have at most 128 poles and no more zeros than poles. No implicit selection of frequencies, units, branches or output representations is performed.
///
/// ## Returns
///
/// Output shapes and units follow the conventions above. Transformed roots are sorted by real then imaginary components.
///
/// ## Errors
///
/// Returns shared Error for invalid domains, non-real coefficients, singular transformations, solver failure, or arithmetic overflow. Exact response zeros give negative infinity dB in Bode functions; response poles are errors.
///
/// ## Numerical behavior
///
/// Uses explicit f64 prototype formulas and rational transformations. Polynomial factorization uses the private nalgebra Schur solver with at most 100000 iterations. High-order coefficient representations and repeated roots are ill-conditioned; use root responses when available. No uniform relative accuracy guarantee is made.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result = cheby2(3,40.,&[0.2],BandType::Lowpass,Some(2.))?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy cheby2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheby2.html).
pub fn cheby2(
    n: usize,
    attenuation: f64,
    cutoffs: &[f64],
    band: BandType,
    fs: Option<f64>,
) -> Result<TransferFunction> {
    design(cheb2ap(n, attenuation)?, cutoffs, band, fs)
}

/// Computes analog gain `20 log10 |H(iω)|` and unwrapped phase `arg H(iω)`.
///
/// ## Parameters
///
/// Coefficients are descending powers, explicit angular frequencies in radians/time. Returns (magnitude in dB, phase in degrees) in caller order; frequencies are not repeated in the return value. Phase unwrapping requires a suitable ordered grid.
///
/// ## Domain
///
/// Inputs must be finite. ZPK inputs have at most 128 poles and no more zeros than poles. No implicit selection of frequencies, units, branches or output representations is performed.
///
/// ## Returns
///
/// Output shapes and units follow the conventions above. Transformed roots are sorted by real then imaginary components.
///
/// ## Errors
///
/// Returns shared Error for invalid domains, non-real coefficients, singular transformations, solver failure, or arithmetic overflow. Exact response zeros give negative infinity dB in Bode functions; response poles are errors.
///
/// ## Numerical behavior
///
/// Uses explicit f64 prototype formulas and rational transformations. Polynomial factorization uses the private nalgebra Schur solver with at most 100000 iterations. High-order coefficient representations and repeated roots are ill-conditioned; use root responses when available. No uniform relative accuracy guarantee is made.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result = bode(&[1.],&[1.,1.],&[0.,1.])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy bode](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.bode.html).
pub fn bode(b: &[f64], a: &[f64], frequencies: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
    let h = super::freqs(b, a, frequencies)?;
    bode_values(h)
}
/// Computes digital gain `20 log10 |H(exp(iω))|` and unwrapped phase.
///
/// ## Parameters
///
/// Coefficients are ascending delays, explicit angular frequencies in radians/sample. Returns (magnitude in dB, phase in degrees); caller retains frequencies and sample interval. Phase unwrapping requires an ordered grid.
///
/// ## Domain
///
/// Inputs must be finite. ZPK inputs have at most 128 poles and no more zeros than poles. No implicit selection of frequencies, units, branches or output representations is performed.
///
/// ## Returns
///
/// Output shapes and units follow the conventions above. Transformed roots are sorted by real then imaginary components.
///
/// ## Errors
///
/// Returns shared Error for invalid domains, non-real coefficients, singular transformations, solver failure, or arithmetic overflow. Exact response zeros give negative infinity dB in Bode functions; response poles are errors.
///
/// ## Numerical behavior
///
/// Uses explicit f64 prototype formulas and rational transformations. Polynomial factorization uses the private nalgebra Schur solver with at most 100000 iterations. High-order coefficient representations and repeated roots are ill-conditioned; use root responses when available. No uniform relative accuracy guarantee is made.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result = dbode(&[1.],&[1.,-0.5],&[0.,1.])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy dbode](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.dbode.html).
pub fn dbode(b: &[f64], a: &[f64], frequencies: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
    let h = super::freqz(b, a, frequencies)?;
    bode_values(h)
}
fn bode_values(h: Vec<Complex>) -> Result<(Vec<f64>, Vec<f64>)> {
    let mut previous = 0.;
    let mut offset = 0.;
    let mut phase = vec![];
    let mut magnitude = vec![];
    for (i, z) in h.iter().enumerate() {
        let angle = z.im.atan2(z.re);
        if i > 0 {
            let delta = angle - previous;
            if delta > PI {
                offset -= TAU;
            } else if delta < -PI {
                offset += TAU;
            }
        }
        previous = angle;
        phase.push((angle + offset) * 180. / PI);
        magnitude.push(20. * z.re.hypot(z.im).log10());
    }
    // Exact transfer zeros have -infinity dB, intentionally preserved.
    if phase.iter().any(|v| !v.is_finite()) || magnitude.iter().any(|v| v.is_nan()) {
        return Err(Error::NonFiniteEvaluation);
    }
    Ok((magnitude, phase))
}

/// Classical IIR prototype family supported by [`iirfilter`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IirFamily {
    /// Maximally flat magnitude.
    Butterworth,
    /// Equiripple passband; parameter is positive ripple in dB.
    ChebyshevI(f64),
    /// Equiripple stopband; parameter is positive attenuation in dB.
    ChebyshevII(f64),
    /// Equiripple passband and stopband, parameters are ripple and attenuation in dB.
    Elliptic(f64, f64),
    /// Maximally flat delay, with explicit prototype normalization.
    Bessel(super::BesselNorm),
}

/// Designs `H` by transforming a classical lowpass prototype.
///
/// ## Parameters
///
/// `n` is order <=64. `cutoffs`, `band`, and `fs` follow [`butter`].
/// `family` selects Butterworth, Chebyshev I/II, elliptic, or Bessel.
/// Digital cutoffs are in sample-rate units; analog cutoffs are radians/time.
///
/// ## Domain
///
/// Inputs must be finite, cutoffs positive and increasing, and digital
/// cutoffs below Nyquist. Bandpass/stop requires two cutoffs; low/high one.
///
/// ## Returns
///
/// Returns numerator and denominator vectors; digital coefficients are
/// ascending delays and analog coefficients are descending powers.
///
/// ## Errors
///
/// Returns shared errors for invalid parameters or numerical overflow.
///
/// ## Numerical behavior
///
/// Prewarps digital cutoffs and uses the bilinear transformation. Coefficient
/// representations of high-order filters may be poorly conditioned.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _ = iirfilter(3,&[0.2],BandType::Lowpass,Some(2.),IirFamily::Butterworth)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy iirfilter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirfilter.html).
pub fn iirfilter(
    n: usize,
    cutoffs: &[f64],
    band: BandType,
    fs: Option<f64>,
    family: IirFamily,
) -> Result<TransferFunction> {
    match family {
        IirFamily::Butterworth => butter(n, cutoffs, band, fs),
        IirFamily::ChebyshevI(r) => cheby1(n, r, cutoffs, band, fs),
        IirFamily::ChebyshevII(r) => cheby2(n, r, cutoffs, band, fs),
        IirFamily::Elliptic(r, s) => super::ellip(n, r, s, cutoffs, band, fs),
        IirFamily::Bessel(norm) => super::bessel(n, cutoffs, band, fs, norm),
    }
}

fn minimum_order(
    wp: f64,
    ws: f64,
    gpass: f64,
    gstop: f64,
    fs: Option<f64>,
    family: usize,
) -> Result<(usize, f64)> {
    positive(wp)?;
    positive(ws)?;
    positive(gpass)?;
    positive(gstop)?;
    if wp == ws || gstop <= gpass {
        return Err(Error::InvalidInput(
            "distinct edges and gstop > gpass required",
        ));
    }
    let (p, s) = if let Some(fs) = fs {
        positive(fs)?;
        if wp >= fs / 2. || ws >= fs / 2. {
            return Err(Error::InvalidInput("edges must be below Nyquist"));
        }
        ((PI * wp / fs).tan(), (PI * ws / fs).tan())
    } else {
        (wp, ws)
    };
    let low = p < s;
    let ratio = if low { s / p } else { p / s };
    let gp = 10_f64.powf(gpass / 10.) - 1.;
    let gs = 10_f64.powf(gstop / 10.) - 1.;
    let numerator = (gs / gp).sqrt().acosh();
    let n = if family == 0 {
        ((gs / gp).ln() / (2. * ratio.ln())).ceil()
    } else {
        (numerator / ratio.acosh()).ceil()
    };
    if !n.is_finite() || !(1. ..=64.).contains(&n) {
        return Err(Error::InvalidInput("required order outside 1..64"));
    }
    let cutoff = match family {
        0 => {
            if low {
                p / gp.powf(1. / (2. * n))
            } else {
                p * gp.powf(1. / (2. * n))
            }
        }
        1 => p,
        _ => {
            if low {
                p * (numerator / n).cosh()
            } else {
                p / (numerator / n).cosh()
            }
        }
    };
    let cutoff = if let Some(fs) = fs {
        fs / PI * cutoff.atan()
    } else {
        cutoff
    };
    Ok((
        n as usize,
        match cutoff {
            value if value.is_finite() => Ok(value),
            _ => Err(Error::NonFiniteEvaluation),
        }?,
    ))
}
macro_rules! order_function {
    ($name:ident,$family:expr,$equation:literal) => {
        #[doc=concat!("Computes the minimum lowpass/highpass order `",$equation,"`.",
        "\n\n# Parameters and conventions\nScalar wp and ws are passband and stopband edges. gpass and gstop are positive amplitude losses in dB, gstop>gpass. fs=Some(rate) means digital frequency units; None means analog radians/time. Bandpass and bandstop order selection are unsupported.",
        "\n\n# Domain and special values\nEdges must be positive, finite and distinct; digital edges below Nyquist. The frequency ratio r is larger warped edge divided by smaller. Gp=10^(gpass/10)-1, Gs=10^(gstop/10)-1.",
        "\n\n# Return values\nReturns (order, critical frequency), suitable for the matching design function, in caller units. Critical frequency preserves the specified passband loss.",
        "\n\n# Errors\nReturns shared errors for invalid parameters, required order outside 1..64 or nonfinite arithmetic.",
        "\n\n# Numerical behavior\nClosed-form f64 evaluation with tangent prewarping for digital edges; cancellation at tiny dB differences can limit accuracy.",
        "\n\n# Examples\n```\nlet (n,w)=sci_signal::",stringify!($name),"(0.2,0.4,1.,40.,Some(2.))?;\nassert!(n>0 && w>0.);\n# Ok::<(),sci_core::Error>(())\n```",
        "\n\n# References\n[SciPy order selection](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.",stringify!($name),".html).")]
        pub fn $name(wp:f64,ws:f64,gpass:f64,gstop:f64,fs:Option<f64>)->Result<(usize,f64)>{minimum_order(wp,ws,gpass,gstop,fs,$family)}
    }
}
order_function!(buttord, 0, "ceil(log(Gs/Gp)/(2 log(r)))");
order_function!(cheb1ord, 1, "ceil(acosh(sqrt(Gs/Gp))/acosh(r))");
order_function!(cheb2ord, 2, "ceil(acosh(sqrt(Gs/Gp))/acosh(r))");

macro_rules! transform_function {
    ($name:ident,$root_function:ident,$equation:literal,$($parameter:ident),+) => {
        #[doc=concat!("Transforms `H(s)=B(s)/A(s)` by `",$equation,"`.",
        "\n\n# Parameters and conventions\nNumerator b and denominator a use descending powers. wo is a positive center/cutoff angular frequency; bw is a positive angular bandwidth where present. fs is positive samples/time for bilinear. This adapts SciPy to proper real transfer functions and explicitly normalized outputs.",
        "\n\n# Domain and special values\nCoefficients and parameters must be finite, the denominator nonzero, and numerator degree <= denominator degree. Polynomial degree is at most 128; band transforms require input degree <=64. Roots mapping to infinity are rejected.",
        "\n\n# Return values\nReturns normalized (numerator,denominator). Analog coefficients use descending powers; bilinear output uses ascending digital delays. Leading normalized numerator coefficients <=1e-14 are trimmed following tf2zpk.",
        "\n\n# Errors\nReturns shared errors for invalid domains, work limits, eigensolver failure, non-real coefficients or arithmetic overflow.",
        "\n\n# Numerical behavior\nReuses root conversion, bounded private nalgebra Schur factorization, and the shared ZPK transformation. Repeated roots and high-degree polynomial coefficients can be ill-conditioned; prefer root transformations when possible.",
        "\n\n# Examples\n```\n",$("let ",stringify!($parameter),"=2.;\n"),+,"let _ = sci_signal::",stringify!($name),"(&[1.],&[1.,1.]",$(",",stringify!($parameter)),+, ")?;\n# Ok::<(),sci_core::Error>(())\n```",
        "\n\n# References\n[SciPy transformation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.",stringify!($name),".html).")]
        pub fn $name(b:&[f64],a:&[f64],$($parameter:f64),+)->Result<TransferFunction>{
            zpk2tf(&$root_function(&tf2zpk(b,a)?,$($parameter),+)?)
        }
    }
}
transform_function!(lp2lp, lp2lp_zpk, "s -> s/wo", wo);
transform_function!(lp2hp, lp2hp_zpk, "s -> wo/s", wo);
transform_function!(lp2bp, lp2bp_zpk, "s -> (s²+wo²)/(bw s)", wo, bw);
transform_function!(lp2bs, lp2bs_zpk, "s -> bw s/(s²+wo²)", wo, bw);
transform_function!(bilinear, bilinear_zpk, "s -> 2fs (z-1)/(z+1)", fs);
