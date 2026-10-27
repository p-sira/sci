use super::design::{design, roots, validate};
use super::{
    BandType, IirFamily, TransferFunction, Zpk, buttord, cheb1ord, cheb2ord, checked, iirfilter,
    nonempty, positive, size,
};
use crate::special::{ellipj, ellipk, ellipkinc, ellipkm1};
use crate::{
    Error, Result,
    complex::{Complex, native, owned},
};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Normalization of the reverse Bessel polynomial.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BesselNorm {
    /// Unit asymptotic phase midpoint (SciPy default).
    Phase,
    /// Unit group delay at zero.
    Delay,
    /// Half-power magnitude at one radian/time.
    Magnitude,
}
fn fact(n: usize) -> f64 {
    (1..=n).fold(1., |v, k| v * k as f64)
}
// The scaled companion solver supplies initial guesses. Refine zeros of
// exp(z) K_(n+1/2)(z), whose zeros are those of theta_n(z), avoiding
// cancellation in high-order polynomial evaluation (as in SciPy besselap).
fn refine_bessel_pole(n: usize, mut z: Complex64) -> Result<Complex64> {
    if !z.re.is_finite() || !z.im.is_finite() || z == Complex64::new(0., 0.) {
        return Err(Error::NonFiniteEvaluation);
    }
    let scaled_k = |order, argument| {
        complex_bessel::besselk_scaled(order, argument)
            .map_err(|_| Error::NumericalFailure("scaled Bessel K evaluation failed"))
    };
    for _ in 0..16 {
        let value = scaled_k(n as f64 + 0.5, z)?;
        let derivative =
            value - 0.5 * (scaled_k(n as f64 - 0.5, z)? + scaled_k(n as f64 + 1.5, z)?);
        let step = value / derivative;
        if !step.re.is_finite() || !step.im.is_finite() {
            return Err(Error::NonFiniteEvaluation);
        }
        z -= step;
        if step.norm() <= 8. * f64::EPSILON * z.norm() {
            return Ok(z);
        }
    }
    Err(Error::NumericalFailure(
        "Bessel pole refinement did not converge",
    ))
}
fn dc_gain(p: &[Complex], z: &[Complex]) -> f64 {
    (p.iter()
        .fold(Complex64::new(1., 0.), |v, &p| v * (-native(p)))
        / z.iter()
            .fold(Complex64::new(1., 0.), |v, &z| v * (-native(z))))
    .re
}
fn specification(rp: f64, rs: f64) -> Result<f64> {
    positive(rp)?;
    positive(rs)?;
    if rs <= rp || rs > 600. {
        return Err(Error::InvalidInput(
            "require 0 < ripple < attenuation <= 600 dB",
        ));
    }
    Ok((10f64.powf(rp / 10.) - 1.) / (10f64.powf(rs / 10.) - 1.))
}

/// Constructs `H(s)=a_0/theta_n(s)` with reverse Bessel polynomial
/// `theta_n(s)=sum_(k=0)^n (2n-k)! s^k/[2^(n-k)(n-k)! k!]`.
///
/// ## Parameters
///
/// `n` is 0..12. `norm` selects unit group delay at zero (`Delay`),
/// half-power magnitude at angular frequency 1 (`Magnitude`), or unit
/// asymptotic phase midpoint (`Phase`). Frequencies use radians per unit time.
///
/// ## Domain
///
/// Order zero returns no zeros or poles and gain 1. The discrete parameters
/// do not admit NaN or infinity.
///
/// ## Returns
///
/// Returns ZPK with no finite zeros, poles sorted by real then imaginary
/// parts, and gain scaled to the selected normalization.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for orders above 12,
/// [`Error::NumericalFailure`] if the Schur solver or pole refinement fails,
/// and [`Error::NonFiniteEvaluation`] for nonfinite numerical results.
///
/// ## Numerical behavior
///
/// Uses a private Schur solver with at most 100000 iterations for initial
/// polynomial roots, followed by at most 16 Newton steps per pole using the
/// exponentially scaled complex Bessel K backend. This avoids cancellation
/// in polynomial refinement. Conjugate pairs are exactly symmetric.
/// Magnitude normalization uses at most 64 bracketing and 80 bisection steps.
/// No uniform relative accuracy guarantee is made.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _ = besselap(3,BesselNorm::Phase)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy besselap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.besselap.html).
pub fn besselap(n: usize, norm: BesselNorm) -> Result<Zpk> {
    if n > 12 {
        return Err(Error::InvalidInput("Bessel order exceeds 12"));
    }
    if n == 0 {
        return Ok(Zpk {
            zeros: vec![],
            poles: vec![],
            gain: 1.,
        });
    }
    let coefficients: Vec<_> = (0..=n)
        .rev()
        .map(|k| fact(2 * n - k) / (2f64.powi((n - k) as i32) * fact(n - k) * fact(k)))
        .collect();
    let gain = coefficients[n];
    let phase_scale = gain.powf(-1. / n as f64);
    let scaled: Vec<_> = coefficients
        .iter()
        .enumerate()
        .map(|(j, c)| c * phase_scale.powi(j as i32))
        .collect();
    let mut poles = roots(&scaled)?;
    // Schur returns conjugate pairs with identical real components; roots()
    // sorts each pair together. Refine one member and reflect it exactly.
    let mut i = 0;
    while i < poles.len() {
        let real = poles[i].im == 0.;
        let mut p = refine_bessel_pole(n, native(poles[i]) / phase_scale)?;
        if real {
            p.im = 0.;
            poles[i] = owned(p);
            i += 1;
        } else {
            poles[i] = owned(p);
            poles[i + 1] = owned(p.conj());
            i += 2;
        }
    }
    poles.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im)));
    let scale = match norm {
        BesselNorm::Delay => 1.,
        BesselNorm::Phase => gain.powf(-1. / n as f64),
        BesselNorm::Magnitude => {
            let magnitude = |w: f64| {
                gain / poles
                    .iter()
                    .map(|&p| (Complex64::new(0., w) - native(p)).norm())
                    .product::<f64>()
            };
            let (mut lo, mut hi) = (0., 1.);
            for _ in 0..64 {
                if magnitude(hi) <= 0.5f64.sqrt() {
                    break;
                }
                hi *= 2.;
            }
            for _ in 0..80 {
                let mid = (hi + lo) / 2.;
                if magnitude(mid) > 0.5f64.sqrt() {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            1. / ((hi + lo) / 2.)
        }
    };
    for p in &mut poles {
        p.re *= scale;
        p.im *= scale;
    }
    let out = Zpk {
        zeros: vec![],
        poles,
        gain: gain * scale.powi(n as i32),
    };
    validate(&out)?;
    Ok(out)
}

/// Designs a Bessel filter by transforming `theta_n(s)` to the requested band.
///
/// ## Parameters
///
/// `n` and `norm` follow besselap; cutoffs and band follow butter. `fs=None` is analog radians/time, otherwise cutoffs use sample-rate units. Returns descending analog or ascending-delay digital coefficients.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic; cancellation and ill conditioning can reduce accuracy.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _ = bessel(3,&[0.2],BandType::Lowpass,Some(2.),BesselNorm::Phase)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy bessel](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.bessel.html).
pub fn bessel(
    n: usize,
    cutoffs: &[f64],
    band: BandType,
    fs: Option<f64>,
    norm: BesselNorm,
) -> Result<TransferFunction> {
    design(besselap(n, norm)?, cutoffs, band, fs)
}

/// Constructs the elliptic analog prototype `|H(i omega)|^2=1/(1+epsilon^2 R_n(omega)^2)`,
/// where R_n is the degree-n elliptic rational function and epsilon^2=10^(rp/10)-1.
///
/// ## Parameters
///
/// `n` is 0..64; `rp` and `rs` are passband ripple and stopband attenuation in dB, with 0 < rp < rs <=600. Cutoff is one radian/time at the first -rp dB crossing. Returns zeros, poles, and real gain in SciPy prototype ordering.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses shared elliptic functions and a rapidly converging nome series; extreme ripple ratios and parameters near elliptic singularities can fail explicitly.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _ = ellipap(4,1.,40.)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy ellipap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ellipap.html).
pub fn ellipap(n: usize, rp: f64, rs: f64) -> Result<Zpk> {
    super::design::order(n)?;
    let m1 = specification(rp, rs)?;
    let eps2 = 10f64.powf(rp / 10.) - 1.;
    if n == 0 {
        return Ok(Zpk {
            zeros: vec![],
            poles: vec![],
            gain: 10f64.powf(-rp / 20.),
        });
    }
    if n == 1 {
        let p = -1. / eps2.sqrt();
        return Ok(Zpk {
            zeros: vec![],
            poles: vec![Complex { re: p, im: 0. }],
            gain: -p,
        });
    }
    let k1 = ellipk(m1)?;
    let q = (-PI * ellipkm1(m1)? / (k1 * n as f64)).exp();
    let num = (0..=7).map(|j| q.powi(j * (j + 1))).sum::<f64>();
    let den = 1. + 2. * (1..=8).map(|j| q.powi(j * j)).sum::<f64>();
    let m = 16. * q * (num / den).powi(4);
    let k = ellipk(m)?;
    let r = ellipkinc((1. / eps2.sqrt()).atan(), 1. - m1)?;
    let v = k * r / (n as f64 * k1);
    let (sv, cv, dv, _) = ellipj(v, 1. - m);
    let mut zeros = vec![];
    let mut poles = vec![];
    for j in ((1 - n % 2)..n).step_by(2) {
        let (s, c, d, _) = ellipj(j as f64 * k / n as f64, m);
        if s.abs() > 2e-16 {
            zeros.push(Complex {
                re: 0.,
                im: 1. / (m.sqrt() * s),
            });
        }
        let den = 1. - (d * sv).powi(2);
        poles.push(Complex {
            re: -c * d * sv * cv / den,
            im: -s * dv / den,
        });
    }
    for i in 0..zeros.len() {
        let z = zeros[i];
        zeros.push(Complex {
            re: z.re,
            im: -z.im,
        });
    }
    for i in 0..poles.len() {
        let p = poles[i];
        if p.im.abs() > 1e-15 {
            poles.push(Complex {
                re: p.re,
                im: -p.im,
            });
        }
    }
    let mut gain = dc_gain(&poles, &zeros);
    if n.is_multiple_of(2) {
        gain /= (1. + eps2).sqrt();
    }
    let out = Zpk { zeros, poles, gain };
    validate(&out)?;
    Ok(out)
}

/// Designs an elliptic filter by transforming the equiripple lowpass prototype.
///
/// ## Parameters
///
/// `n`, `rp`, `rs` follow ellipap; cutoffs, band, and fs follow butter. Returns numerator/denominator, with digital ascending delays or analog descending powers.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic; cancellation and ill conditioning can reduce accuracy.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _ = ellip(3,1.,40.,&[0.2],BandType::Lowpass,Some(2.))?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy ellip](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ellip.html).
pub fn ellip(
    n: usize,
    rp: f64,
    rs: f64,
    cutoffs: &[f64],
    band: BandType,
    fs: Option<f64>,
) -> Result<TransferFunction> {
    design(ellipap(n, rp, rs)?, cutoffs, band, fs)
}

/// Finds minimum elliptic order `ceil(K(m0) K(1-m1)/(K(1-m0) K(m1)))`.
///
/// ## Parameters
///
/// Scalar wp/ws select lowpass when wp<ws and highpass otherwise; gpass/gstop are dB. fs=None is analog; digital edges are below fs/2. Returns (order, wp), suitable for ellip. Two-edge band selection is unsupported.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic; cancellation and ill conditioning can reduce accuracy.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _ = ellipord(0.2,0.3,1.,40.,Some(2.))?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy ellipord](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ellipord.html).
pub fn ellipord(wp: f64, ws: f64, gpass: f64, gstop: f64, fs: Option<f64>) -> Result<(usize, f64)> {
    positive(wp)?;
    positive(ws)?;
    let m1 = specification(gpass, gstop)?;
    if wp == ws {
        return Err(Error::InvalidInput("distinct edges required"));
    }
    let (p, s) = if let Some(fs) = fs {
        positive(fs)?;
        if wp >= fs / 2. || ws >= fs / 2. {
            return Err(Error::InvalidInput("edges above Nyquist"));
        }
        ((PI * wp / fs).tan(), (PI * ws / fs).tan())
    } else {
        (wp, ws)
    };
    let m0 = (p.min(s) / p.max(s)).powi(2);
    let n = (ellipk(m0)? * ellipkm1(m1)? / (ellipk(1. - m0)? * ellipk(m1)?)).ceil() as usize;
    super::design::order(n)?;
    Ok((n, wp))
}

/// Evaluates the continuous filter-order objective for an analog bandstop candidate.
///
/// ## Parameters
///
/// Replaces passband\[index\] by candidate; index is 0 or 1. Requires pass\[0\]<stop\[0\]<stop\[1\]<pass\[1\]. gpass/gstop are dB. Family uses only its variant; ripple payloads are ignored in favor of these specifications. Returns a fractional order.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic; cancellation and ill conditioning can reduce accuracy.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _ = band_stop_obj(1.,0,[0.8,4.],[2.,3.],1.,40.,IirFamily::Butterworth)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy band_stop_obj](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.band_stop_obj.html).
pub fn band_stop_obj(
    candidate: f64,
    index: usize,
    mut passband: [f64; 2],
    stopband: [f64; 2],
    gpass: f64,
    gstop: f64,
    family: IirFamily,
) -> Result<f64> {
    if index > 1 {
        return Err(Error::InvalidInput("index must be 0 or 1"));
    }
    passband[index] = candidate;
    for v in passband.into_iter().chain(stopband) {
        positive(v)?;
    }
    if !(passband[0] < stopband[0] && stopband[0] < stopband[1] && stopband[1] < passband[1]) {
        return Err(Error::InvalidInput("invalid bandstop ordering"));
    }
    let m1 = specification(gpass, gstop)?;
    let nat = stopband
        .iter()
        .map(|s| (s * (passband[0] - passband[1]) / (s * s - passband[0] * passband[1])).abs())
        .fold(f64::INFINITY, f64::min);
    let value = match family {
        IirFamily::Butterworth => (1. / m1).ln() / (2. * nat.ln()),
        IirFamily::ChebyshevI(_) | IirFamily::ChebyshevII(_) => {
            (1. / m1.sqrt()).acosh() / nat.acosh()
        }
        IirFamily::Elliptic(_, _) => {
            let m0 = 1. / nat.powi(2);
            ellipk(m0)? * ellipkm1(m1)? / (ellipk(1. - m0)? * ellipk(m1)?)
        }
        IirFamily::Bessel(_) => {
            return Err(Error::InvalidInput("Bessel has no minimum-order objective"));
        }
    };
    if !value.is_finite() {
        return Err(Error::NonFiniteEvaluation);
    }
    Ok(value)
}

/// Designs a minimum-order lowpass or highpass IIR satisfying magnitude specifications.
///
/// ## Parameters
///
/// Scalar wp/ws, gpass/gstop in dB, and fs follow ellipord. Supports Butterworth, both Chebyshev types, and elliptic; family payloads are replaced by gpass/gstop. Returns transfer coefficients; bandpass/stop and Bessel are unsupported.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic; cancellation and ill conditioning can reduce accuracy.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _ = iirdesign(0.2,0.3,1.,40.,Some(2.),IirFamily::Elliptic(1.,40.))?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy iirdesign](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirdesign.html).
pub fn iirdesign(
    wp: f64,
    ws: f64,
    gpass: f64,
    gstop: f64,
    fs: Option<f64>,
    family: IirFamily,
) -> Result<TransferFunction> {
    let (n, wn) = match family {
        IirFamily::Butterworth => buttord(wp, ws, gpass, gstop, fs)?,
        IirFamily::ChebyshevI(_) => cheb1ord(wp, ws, gpass, gstop, fs)?,
        IirFamily::ChebyshevII(_) => cheb2ord(wp, ws, gpass, gstop, fs)?,
        IirFamily::Elliptic(_, _) => ellipord(wp, ws, gpass, gstop, fs)?,
        IirFamily::Bessel(_) => {
            return Err(Error::InvalidInput("minimum Bessel order unsupported"));
        }
    };
    let family = match family {
        IirFamily::ChebyshevI(_) => IirFamily::ChebyshevI(gpass),
        IirFamily::ChebyshevII(_) => IirFamily::ChebyshevII(gstop),
        IirFamily::Elliptic(_, _) => IirFamily::Elliptic(gpass, gstop),
        v => v,
    };
    iirfilter(
        n,
        &[wn],
        if wp < ws {
            BandType::Lowpass
        } else {
            BandType::Highpass
        },
        fs,
        family,
    )
}

/// Designs `H(z)=(bx +/- cx z^-N)/(1 +/- ax z^-N)` for uniformly spaced notches or peaks.
///
/// ## Parameters
///
/// w0 and fs use identical frequency units; fs/w0 must be an integer N. q is quality factor >1 for a stable supported bandwidth. peak selects peaks instead of notches; pass_zero controls DC. Returns ascending-delay coefficients of length N+1.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic; cancellation and ill conditioning can reduce accuracy.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _ = iircomb(20.,30.,false,200.,false)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy iircomb](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iircomb.html).
pub fn iircomb(w0: f64, q: f64, peak: bool, fs: f64, pass_zero: bool) -> Result<TransferFunction> {
    positive(w0)?;
    positive(q)?;
    positive(fs)?;
    if w0 >= fs / 2. || q <= 1. {
        return Err(Error::InvalidInput("w0 below Nyquist and Q > 1 required"));
    }
    let n = (fs / w0).round() as usize;
    size(n.saturating_add(1))?;
    if n < 3 || (w0 - fs / n as f64).abs() / fs > 1e-14 {
        return Err(Error::InvalidInput("fs must be divisible by w0"));
    }
    let beta = (PI / (2. * q)).tan();
    let ax = (1. - beta) / (1. + beta);
    let (g0, g) = if peak { (0., 1.) } else { (1., 0.) };
    let sign = if (peak && pass_zero) || (!peak && !pass_zero) {
        -1.
    } else {
        1.
    };
    let mut b = vec![0.; n + 1];
    let mut a = b.clone();
    b[0] = (g0 + g * beta) / (1. + beta);
    b[n] = sign * (g0 - g * beta) / (1. + beta);
    a[0] = 1.;
    a[n] = sign * ax;
    Ok((checked(b)?, checked(a)?))
}

/// Selects logarithmically spaced analog response frequencies from numerator zeros and denominator poles.
///
/// ## Parameters
///
/// b/a are descending-power real coefficients; n is output length. Returns angular frequencies in radians/time using SciPy decade rounding. Root-input dispatch is unsupported.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic; cancellation and ill conditioning can reduce accuracy.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _ = findfreqs(&[1.,0.],&[1.,8.,25.],9)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy findfreqs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.findfreqs.html).
pub fn findfreqs(b: &[f64], a: &[f64], n: usize) -> Result<Vec<f64>> {
    size(n)?;
    nonempty(a)?;
    if a.iter().all(|v| *v == 0.) {
        return Err(Error::InvalidInput("denominator must be nonzero"));
    }
    let mut poles = roots(a)?;
    let zeros = roots(b)?;
    if poles.is_empty() {
        poles.push(Complex { re: -1000., im: 0. });
    }
    let ez: Vec<_> = poles
        .into_iter()
        .filter(|p| p.im >= 0.)
        .chain(
            zeros
                .into_iter()
                .filter(|z| native(*z).norm() < 1e5 && z.im >= 0.),
        )
        .collect();
    let mut hi: f64 = 0.;
    let mut lo = f64::INFINITY;
    for z in ez {
        let integ = if native(z).norm() < 1e-10 { 1. } else { 0. };
        hi = hi.max(3. * (z.re + integ).abs() + 1.5 * z.im);
        lo = lo.min((z.re + integ).abs() + 2. * z.im);
    }
    let hi = (hi.log10() + 0.5).round_ties_even();
    let lo = ((0.1 * lo).log10() - 0.5).round_ties_even();
    checked(
        (0..n)
            .map(|i| {
                10f64.powf(
                    lo + if n > 1 {
                        (hi - lo) * i as f64 / (n - 1) as f64
                    } else {
                        0.
                    },
                )
            })
            .collect(),
    )
}

/// Constructs a gammatone auditory filter, with FIR impulse `t^(order-1) exp(-2 pi bw t) cos(2 pi freq t)`.
///
/// ## Parameters
///
/// freq/fs are frequency and sample rate in Hz; freq lies in (0,fs/2). fir=true uses order 1..24 and explicit numtaps, returning (b,\[1\]). fir=false uses the fixed eighth-order IIR, requiring order=4 and numtaps=0. Bandwidth is 1.019 times the equivalent rectangular bandwidth.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic; cancellation and ill conditioning can reduce accuracy.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _ = gammatone(440.,true,4,16,16000.)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy gammatone](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.gammatone.html).
pub fn gammatone(
    freq: f64,
    fir: bool,
    order: usize,
    numtaps: usize,
    fs: f64,
) -> Result<TransferFunction> {
    positive(freq)?;
    positive(fs)?;
    if freq >= fs / 2. {
        return Err(Error::InvalidInput("center must be below Nyquist"));
    }
    let bw = 1.019 * (freq / 9.26449 + 24.7);
    if fir {
        if !(1..=24).contains(&order) || numtaps == 0 {
            return Err(Error::InvalidInput("invalid FIR order or length"));
        }
        size(numtaps)?;
        let scale = 2. * (2. * PI * bw).powi(order as i32) / fact(order - 1) / fs;
        return Ok((
            checked(
                (0..numtaps)
                    .map(|i| {
                        let t = i as f64 / fs;
                        scale
                            * t.powi(order as i32 - 1)
                            * (-2. * PI * bw * t).exp()
                            * (2. * PI * freq * t).cos()
                    })
                    .collect(),
            )?,
            vec![1.],
        ));
    }
    if order != 4 || numtaps != 0 {
        return Err(Error::InvalidInput("IIR requires order=4, numtaps=0"));
    }
    let t = 1. / fs;
    let fr = 2. * PI * freq * t;
    let bt = 2. * PI * bw * t;
    let e = (-bt).exp();
    let c = fr.cos();
    let g1 = -2. * Complex64::from_polar(t, 2. * fr);
    let g2 = 2. * Complex64::from_polar(t * e, fr);
    let g5 = Complex64::from_polar(1., 2. * fr);
    let g3 = (3. + 2f64.powf(1.5)).sqrt() * fr.sin();
    let g4 = (3. - 2f64.powf(1.5)).sqrt() * fr.sin();
    let g =
        ((g1 + g2 * (c - g4)) * (g1 + g2 * (c + g4)) * (g1 + g2 * (c - g3)) * (g1 + g2 * (c + g3))
            / (-2. * e * e - 2. * g5 + 2. * (1. + g5) * e).powu(4))
        .norm();
    let bin = [1., -4., 6., -4., 1.];
    let b = (0..5)
        .map(|k| bin[k] * t.powi(4) * (k as f64 * fr).cos() * e.powi(k as i32) / g)
        .collect();
    let a = vec![
        1.,
        -8. * c * e,
        4. * (4. + 3. * (2. * fr).cos()) * e.powi(2),
        -8. * (6. * c + (3. * fr).cos()) * e.powi(3),
        2. * (18. + 16. * (2. * fr).cos() + (4. * fr).cos()) * e.powi(4),
        -8. * (6. * c + (3. * fr).cos()) * e.powi(5),
        4. * (4. + 3. * (2. * fr).cos()) * e.powi(6),
        -8. * c * e.powi(7),
        e.powi(8),
    ];
    Ok((checked(b)?, checked(a)?))
}

#[cfg(test)]
#[path = "tests/bessel_refinement.rs"]
mod bessel_refinement_tests;
