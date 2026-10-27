//! Hypergeometric functions reusing Bessel kernels and shared precision arithmetic.
use crate::backend::{StrErr, double_double::Dd};
use num_complex::{Complex, Complex64};
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

fn series(v: f64, z: Complex64) -> Option<Complex64> {
    let norm = z.norm();
    if norm > 100.0 && !(v > 0.0 && norm / v < 16.0) {
        return None;
    }
    let mut real = Dd::from(1.0);
    let mut imag = Dd::from(0.0);
    let mut sr = real;
    let mut si = imag;
    let mut peak: f64 = 1.0;
    for k in 1..=4096 {
        // Form v+(k-1) in extended precision, retaining small parameters.
        let denominator = Dd::from(v) + Dd::from((k - 1) as f64);
        let fr = Dd::from(z.re) / denominator / Dd::from(k as f64);
        let fi = Dd::from(z.im) / denominator / Dd::from(k as f64);
        let next = real * fr - imag * fi;
        imag = imag * fr + real * fi;
        real = next;
        sr = sr + real;
        si = si + imag;
        let term = real.hi.hypot(imag.hi);
        let magnitude = sr.hi.hypot(si.hi);
        peak = peak.max(term);
        if !magnitude.is_finite() {
            return None;
        }
        if term <= 1e-32 * magnitude || term == 0.0 {
            if peak * 1e-30 > 1e-13 * magnitude {
                return None;
            }
            return Some(Complex64::new(sr.hi, si.hi));
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
    if let Some(value) = series(v, Complex64::new(z, 0.0)) {
        return Ok(value.re);
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
