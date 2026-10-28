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
fn series_real(v: f64, z: f64) -> Option<f64> {
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
        if (tail_bounded && term <= 1e-20 * magnitude) || term <= 1e-32 * magnitude || term == 0.0 {
            let sum = positive + negative;
            if peak * 1e-30 > 1e-13 * sum.hi.abs() {
                return None;
            }
            return Some(sum.hi);
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
    if let Some(value) = series_real(v, z) {
        return Ok(value);
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
