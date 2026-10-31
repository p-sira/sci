//! Hypergeometric functions reusing Bessel kernels and shared precision arithmetic.
use crate::backend::{StrErr, double_double::Dd};
use sci_core::Float;

const ERROR: StrErr =
    "hyp0f1: Requires finite arguments and a parameter outside the nonpositive integers.";

fn validate(v: f64) -> Result<(), StrErr> {
    if !v.is_finite() || (v <= 0.0 && v == v.floor()) {
        Err(ERROR)
    } else {
        Ok(())
    }
}
/// Double-double power series for real `z`. Returns `None` when the series
/// is outside its range or loses too much precision to cancellation.
/// Full-precision anchors retain terms to 1e-32 and require a 1e-26 relative
/// roundoff estimate; ordinary scalar evaluation keeps its existing thresholds.
pub(super) fn series_real<const FULL_PRECISION: bool>(v: f64, z: f64) -> Option<Dd> {
    let norm = z.abs();
    if norm > 100.0 && !(v > 0.0 && norm / v < 16.0) {
        return None;
    }
    let mut real = Dd::from(1.0);
    // Positive and negative terms are accumulated separately, so every
    // accumulation is the cheaper same-sign double-double addition; one full
    // addition combines them at the end.
    let (mut positive, mut negative) = (real, Dd::from(0.0));
    let mut peak: f64 = 1.0;
    for k in 1..=4096 {
        // term_k = term_(k-1) z / ((v + k - 1) k); v + k - 1 is exact in
        // double-double and the integer factor k needs no promotion.
        let kf = k as f64;
        let denominator = Dd::sum(v, kf - 1.0).mul_f64(kf);
        real = real.mul_f64(z).div_dd(denominator);
        if real.hi >= 0.0 {
            positive = positive.add_same_sign(real);
        } else {
            negative = negative.add_same_sign(real);
        }
        let term = real.hi.abs();
        let magnitude = (positive.hi + negative.hi).abs();
        peak = peak.max(term);
        if !magnitude.is_finite() {
            return None;
        }
        // Once the next ratio |z| / ((v + k) (k + 1)) is below one half, the
        // remaining tail is at most twice the current term, so terms below
        // 1e-20 of the sum are beyond binary64 resolution.
        let tail_bounded = (v + kf) * (kf + 1.0) > 2.0 * norm;
        let tolerance = if FULL_PRECISION { 1e-32 } else { 1e-20 };
        if (tail_bounded && term <= tolerance * magnitude)
            || term <= 1e-32 * magnitude
            || term == 0.0
        {
            let sum = positive + negative;
            let accuracy = if FULL_PRECISION { 1e-26 } else { 1e-13 };
            if peak * 1e-30 > accuracy * sum.hi.abs() {
                return None;
            }
            return Some(sum);
        }
    }
    None
}

// Certify compensated binary64 summation using an absolute-term forward-error
// bound. The same bound works for alternating terms, provided the final result
// is not too small relative to the term magnitudes; cancellation-sensitive
// cases fall through to the existing double-double/Bessel kernels.
fn compensated_series(v: f64, z: f64) -> Option<f64> {
    if v <= 0.0 || !(-10.0..=100.0).contains(&z) {
        return None;
    }
    let (mut term, mut sum, mut correction, mut weighted) = (1.0, 1.0, 0.0, 2.0);
    for k in 1..=256 {
        let kf = k as f64;
        term *= z / ((v + (kf - 1.0)) * kf);
        let adjusted = term - correction;
        let next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
        // Four rounded operations per term, and two eps for compensated sum.
        weighted += (4.0 * kf + 2.0) * term.abs();
        if !sum.is_finite() {
            return None;
        }
        if (v + kf) * (kf + 1.0) > 2.0 * z.abs() && term.abs() <= 0.125 * f64::EPSILON * sum.abs() {
            // gamma_(4k) <= (4k)*eps/(1-1024*eps), since k <= 256.
            let error = f64::EPSILON * weighted / (1.0 - 1024.0 * f64::EPSILON) + 2.0 * term.abs();
            return (error <= 16.0 * f64::EPSILON * sum.abs()).then_some(sum);
        }
    }
    None
}

fn real_kernel(v: f64, z: f64) -> Result<f64, StrErr> {
    validate(v)?;
    if !z.is_finite() {
        return Err(ERROR);
    }
    if z == 0.0 {
        return Ok(1.0);
    }
    // Large negative arguments with |z| > v have strongly alternating
    // terms. The strict binary64 error bound usually rejects this region;
    // go directly to the higher-precision series to avoid a redundant pass.
    let cancellation_prone = v > 0.0 && z < -3.0 && v < -z;
    if !cancellation_prone && let Some(value) = compensated_series(v, z) {
        return Ok(value);
    }
    if let Some(value) = series_real::<false>(v, z) {
        return Ok(value.hi);
    }
    let root = z.abs().sqrt();
    let bessel = if z > 0.0 {
        crate::backend::bessel::ive(v - 1.0, 2.0 * root)?
    } else {
        crate::backend::cyl_bessel_j(v - 1.0, 2.0 * root)?
    };
    if bessel == 0.0 {
        return Err("hyp0f1: Bessel kernel underflow prevents reliable rescaling.");
    }
    let logarithm = crate::backend::cephes64::lgam(v)
        + (1.0 - v) * root.ln()
        + if z > 0.0 { 2.0 * root } else { 0.0 };
    Ok((logarithm + bessel.abs().ln()).exp() * crate::backend::gammasgn(v)? * bessel.signum())
}

// Taylor continuation across the real axis from z*f''+v*f'-f=0.
// Keeping powers of the transverse displacement inside the recurrence avoids
// subtracting almost equal Bessel phases to obtain a small imaginary component.

/// Confluent hypergeometric limit function 0F1(;v;z) for real arguments.
///
/// Requires finite arguments; nonpositive integer `v` is a pole. Uses
/// double-double series near zero and exponentially scaled Bessel kernels
/// elsewhere, combining prefactors logarithmically to avoid intermediate overflow.
pub fn hyp0f1<T: Float>(v: T, z: T) -> Result<T, StrErr> {
    let v = v.to_f64().ok_or(ERROR)?;
    let z = z.to_f64().ok_or(ERROR)?;
    crate::backend::utils::finish(real_kernel(v, z)?, ERROR)
}

#[cfg(test)]
mod tests {
    use super::{compensated_series, real_kernel};

    #[test]
    fn negative_argument_shortcut_matches_independent_high_precision() {
        // mpmath 1.3.0, mp.dps = 80, hyper([], [v], z), rounded to f64.
        // The cases include both sides of the shortcut threshold, small v,
        // values close to zeros, and large alternating-series cancellation.
        const REFERENCES: &[(f64, f64, f64)] = &[
            (0.125, -3.01, -2.8993537576671815),
            (0.125, -5.0, 2.2799868130002063),
            (0.125, -10.0, 5.143622558744288),
            (0.125, -30.0, 4.3540733947902295),
            (0.5, -3.00001, -0.9484413659322454),
            (0.5, -5.551652475612764, -9.468420460028252e-17),
            (0.5, -5.55165, -1.0506828067508704e-6),
            (0.5, -9.0, 0.960170286650366),
            (0.5, -25.0, -0.8390715290764524),
            (1.0, -3.0, -0.37492595624331004),
            (1.0, -7.617815585, -1.1286788840317782e-10),
            (1.0, -7.6178, -1.9214742434014665e-6),
            (1.0, -12.0, 0.29896586116360446),
            (1.0, -50.0, 0.15055578737182204),
            (2.5, -3.01, 0.2127112447528678),
            (2.5, -5.0, 0.0031146110444027386),
            (2.5, -10.0, -0.074445380648445),
            (2.5, -15.0, 0.0010271017807084596),
            (2.5, -50.0, 0.0011351770111090956),
            (5.0, -3.01, 0.5292643399033946),
            (5.0, -5.001, 0.33129163930217),
            (5.0, -8.0, 0.14602211172333576),
            (5.0, -25.0, -0.008432743146317128),
            (5.0, -50.0, 0.0004564044063749215),
            (10.0, -3.01, 0.7369299995911754),
            (10.0, -11.0, 0.3125290020302696),
            (10.0, -40.0, 0.002932256211165023),
            (0.1, -100.0, -2.968942964377085),
            (0.5, -100.0, 0.40808206181339196),
            (2.5, -100.0, -0.0027182609945775797),
            (10.0, -100.0, 4.5405815286664117e-5),
            (3.25, 0.5, 1.1631890479552405),
            (3.25, 10.0, 11.228357500393701),
            (3.25, 100.0, 548442.9914829632),
            (-0.3, -3.01, -0.835042183928332),
            (-2.3, -5.0, 7.25048701318888),
            (-0.3, 4.0, -94.14213052614711),
            (-2.3, 1.0, 0.47197024339118),
        ];
        for &(v, z, expected) in REFERENCES {
            let actual = real_kernel(v, z).unwrap();
            let public = crate::hyp0f1(v, z);
            assert!(
                (actual - expected).abs() <= 3e-12 * expected.abs().max(1.0),
                "0F1(;{v};{z}): got {actual}, reference {expected}"
            );
            assert_eq!(actual.to_bits(), public.to_bits());
        }
    }

    #[test]
    fn negative_argument_shortcut_respects_recurrence_and_poles() {
        // Contiguous relation from the defining Pochhammer series:
        // F(;b-1;z) = F(;b;z) + z F(;b+1;z)/(b(b-1)).
        for &v in &[0.5, 1.125, 2.5, 5.0, 10.0, 25.0] {
            for &z in &[-100.0, -25.0, -10.0, -3.0001, -3.0, -2.9999, 2.0, 20.0] {
                let lhs = real_kernel(v - 1.0, z).unwrap();
                let rhs = real_kernel(v, z).unwrap()
                    + z / (v * (v - 1.0)) * real_kernel(v + 1.0, z).unwrap();
                assert!(
                    (lhs - rhs).abs() <= 2e-10 * lhs.abs().max(rhs.abs()).max(1.0),
                    "0F1 recurrence v={v}, z={z}: {lhs} vs {rhs}"
                );
            }
        }
        for &v in &[0.0, -1.0, -2.0, -3.0] {
            assert!(real_kernel(v, -10.0).is_err());
            assert!(crate::hyp0f1(v, -10.0).is_nan());
        }
        assert!(real_kernel(1.0, f64::NAN).is_err());
    }

    #[test]
    fn negative_shortcut_preserves_any_certified_binary64_result() {
        // The expensive fallback must agree whenever the previous, stricter
        // binary64 bound could certify the result, even outside benchmark inputs.
        let parameters = [0.01, 0.05, 0.125, 0.5, 1.0, 1.5, 2.5, 3.0, 5.0, 10.0, 25.0];
        let arguments = [
            -3.0000001, -3.001, -3.01, -3.1, -3.25, -3.5, -4.0, -5.0, -7.5, -10.0, -20.0, -50.0,
            -100.0,
        ];
        for v in parameters {
            for z in arguments {
                if v < -z {
                    let candidate = real_kernel(v, z).unwrap();
                    if let Some(certified) = compensated_series(v, z) {
                        assert!(
                            (candidate - certified).abs()
                                <= 4e-13 * candidate.abs().max(certified.abs()).max(1.0),
                            "0F1(;{v};{z})={candidate}, certified={certified}"
                        );
                    }
                }
            }
        }
    }
}
