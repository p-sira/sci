//! Complex Riemann and Hurwitz zeta for moderate arguments.
//!
//! The Euler–Maclaurin continuation follows DLMF 25.11.4 and 25.11.28,
//! with the principal logarithm for each `(q + k)^(-s)`. Re(q)>0 avoids
//! the Hurwitz branch cut.
//! A finite work budget and a cancellation check prevent misleading answers
//! where binary64 summation cannot retain the requested relative accuracy.

use super::StrErr;
use num_complex::Complex64;

// B_(2k)/(2k)!, represented by reciprocals. These are the Cephes coefficients
// in zeta.c (Cephes Math Library, Stephen L. Moshier), also given in DLMF 24.2.
const BERNOULLI_DENOMINATORS: [f64; 12] = [
    12.0,
    -720.0,
    30_240.0,
    -1_209_600.0,
    47_900_160.0,
    -1.892_437_580_318_379_2e9,
    7.472_424_96e10,
    -2.950_130_727_918_164e12,
    1.164_678_281_435_006_7e14,
    -4.597_978_722_407_473e15,
    1.815_210_540_194_354_6e17,
    -7.166_165_256_175_667e18,
];

fn finite(z: Complex64) -> bool {
    z.re.is_finite() && z.im.is_finite()
}

/// Evaluates ζ(s) on the supported complex domain.
///
/// See [`hurwitz_zeta`] for the continuation, numerical range and errors.
pub(crate) fn riemann_zeta(s: Complex64) -> Result<Complex64, StrErr> {
    hurwitz_zeta(s, Complex64::new(1.0, 0.0))
}

/// Evaluates the analytic continuation of ζ(s,q)=Σ_(n≥0)(n+q)^(-s).
///
/// Requires 0≤Re(s)≤30, |Im(s)|≤32, 0<Re(q)≤64, |Im(q)|≤32 and finite
/// components, excluding the pole at s=1. The principal logarithm determines
/// complex powers. Evaluations with an insufficient cancellation/error margin,
/// nonfinite intermediates, or a nonfinite result return an error. This bounded
/// binary64 implementation does not promise accuracy at zeros, severe
/// cancellation, or outside these ranges.
pub(crate) fn hurwitz_zeta(s: Complex64, q: Complex64) -> Result<Complex64, StrErr> {
    if !finite(s) || !finite(q) {
        return Err("zeta: Arguments must be finite.");
    }
    if !(0.0..=30.0).contains(&s.re)
        || s.im.abs() > 32.0
        || q.re <= 0.0
        || q.re > 64.0
        || q.im.abs() > 32.0
    {
        return Err("zeta: Arguments exceed the supported domain.");
    }
    if s.re == 1.0 && s.im == 0.0 {
        return Err("zeta: Pole at s=1.");
    }
    // Bernoulli polynomial B_1(q) gives the exact continuation at s=0.
    if s.re == 0.0 && s.im == 0.0 {
        return Ok(Complex64::new(0.5, 0.0) - q);
    }

    // Keeping the starting point well beyond |s| makes the 12-term
    // Euler–Maclaurin expansion strongly decreasing in the supported range.
    let count = (2.0 * s.norm()).ceil().max(16.0) as usize;
    let mut sum = Complex64::new(0.0, 0.0);
    let mut compensation = Complex64::new(0.0, 0.0);
    let mut magnitudes = 0.0;
    for k in 0..count {
        let term = (q + k as f64).powc(-s);
        if !finite(term) {
            return Err("zeta: Nonfinite series term.");
        }
        // Componentwise compensated summation of the finite Dirichlet prefix.
        let corrected = term - compensation;
        let next = sum + corrected;
        compensation = (next - sum) - corrected;
        sum = next;
        magnitudes += term.norm();
    }

    let a = q + count as f64;
    let reciprocal = Complex64::new(1.0, 0.0) / a;
    let power = a.powc(-s);
    let integral = power * a / (s - 1.0);
    let midpoint = power * 0.5;
    if !finite(integral) || !finite(midpoint) {
        return Err("zeta: Nonfinite Euler–Maclaurin tail.");
    }
    magnitudes += integral.norm() + midpoint.norm();
    let mut tail = integral + midpoint;
    let mut factor = s * reciprocal;
    let mut last = Complex64::new(0.0, 0.0);
    for (j, denominator) in BERNOULLI_DENOMINATORS.iter().enumerate() {
        last = (power * factor) / *denominator;
        if !finite(last) {
            return Err("zeta: Nonfinite Euler–Maclaurin correction.");
        }
        tail += last;
        magnitudes += last.norm();
        factor *= (s + (2 * j + 1) as f64) * (s + (2 * j + 2) as f64) * reciprocal * reciprocal;
    }

    let result = sum + tail;
    if !finite(result) || !magnitudes.is_finite() {
        return Err("zeta: Evaluation exceeded binary64 range.");
    }
    // Individual complex powers, accumulation and final cancellation introduce
    // an error proportional to the absolute sum of intermediate magnitudes.
    // The factor covers power/log arithmetic and the bounded prefix length.
    // Reject a poorly conditioned answer instead of returning spurious zeros.
    let scale = result.norm();
    if 128.0 * f64::EPSILON * magnitudes > 1e-10 * scale || last.norm() > 1e-12 * scale {
        return Err("zeta: Insufficient precision after cancellation.");
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_rejects_nonfinite_inputs_without_facade_filtering() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(riemann_zeta(Complex64::new(2.0, bad)).is_err());
            assert!(hurwitz_zeta(Complex64::new(2.0, 0.1), Complex64::new(bad, 1.0),).is_err());
        }
    }
}
