//! Principal complex dilogarithm and cancellation-resistant logarithmic products.
//!
//! Spence's convention is `spence(z) = Li_2(1-z)`. Its cut is the negative
//! real z-axis; signed imaginary zeros select the corresponding bank.
//! The dilogarithm is reduced with reflection, inversion, and Landen identities
//! before a rapidly convergent power or Bernoulli series is evaluated.

use crate::backend::{
    StrErr,
    complex::{finite, reciprocal},
};
use core::f64::consts::PI;
use num_complex::Complex64;

const ZETA_TWO: f64 = PI * PI / 6.0;

/// Unlike `1.0-z` via generic complex arithmetic, explicitly negating the
/// imaginary component preserves the signed zero that chooses a cut bank.
fn one_minus(z: Complex64) -> Complex64 {
    Complex64::new(1.0 - z.re, -z.im)
}

/// Principal complex log, scaling large magnitudes before `hypot` so two
/// finite components close to `f64::MAX` do not spuriously overflow.
fn complex_log(z: Complex64) -> Complex64 {
    let scale = z.re.abs().max(z.im.abs());
    let logarithm = if scale > f64::MAX / 2.0 {
        scale.ln() + (z.re / scale).hypot(z.im / scale).ln()
    } else {
        z.norm().ln()
    };
    Complex64::new(logarithm, z.im.atan2(z.re))
}

/// Log(1-z) with full relative precision when `z` is close to zero.
/// `atan2` also retains the bank of a negative-real cut and signed zero.
fn log_one_minus(z: Complex64) -> Complex64 {
    if z.re.abs() <= 0.5 && z.im.abs() <= 0.5 {
        Complex64::new(
            0.5 * (-2.0 * z.re + z.re * z.re + z.im * z.im).ln_1p(),
            (-z.im).atan2(1.0 - z.re),
        )
    } else {
        complex_log(one_minus(z))
    }
}

/// `Li_2(z)` for |z|<=1/2, with a finite series and an explicit work bound.
fn dilog_power(z: Complex64) -> Result<Complex64, StrErr> {
    let mut sum = z;
    let mut power = z;
    for k in 2..=128 {
        power *= z;
        let term = power / (k * k) as f64;
        sum += term;
        if term.norm() <= f64::EPSILON * 0.25 * sum.norm() {
            return Ok(sum);
        }
    }
    Err("spence: Power series did not converge.")
}

/// `Li_2(1-exp(-t)) = t - t²/4 + sum B_{2k} t^(2k+1)/(2k+1)!`.
/// Here |t| is small after the argument reductions. Bernoulli coefficients
/// are exact rationals rounded once to binary64, in ascending order.
fn dilog_bernoulli(t: Complex64) -> Complex64 {
    const COEFFICIENTS: [f64; 18] = [
        2.7777777777777778e-2,
        -2.7777777777777778e-4,
        4.724111866969010e-6,
        -9.185773074661964e-8,
        1.8978869988971001e-9,
        -4.064761645144226e-11,
        8.921691020456452e-13,
        -1.9939295860721074e-14,
        4.518980029619918e-16,
        -1.0356517612181247e-17,
        2.395218621026187e-19,
        -5.581785874325009e-21,
        1.3091507554183213e-22,
        -3.0874198024267403e-24,
        7.315975652702203e-26,
        -1.740845657234001e-27,
        4.1576356446139e-29,
        -9.962148488284622e-31,
    ];
    let square = t * t;
    let poly = COEFFICIENTS
        .iter()
        .rev()
        .fold(Complex64::new(0.0, 0.0), |acc, &c| acc * square + c);
    t - square * 0.25 + t * square * poly
}

/// Principal Li2 with its cut along [1,+infinity); at the cut, signed zero
/// in the input imaginary component determines which limiting value is used.
fn dilog(z: Complex64) -> Result<Complex64, StrErr> {
    if z == Complex64::new(0.0, 0.0) {
        return Ok(z);
    }
    if z == Complex64::new(1.0, 0.0) {
        return Ok(Complex64::new(ZETA_TWO, 0.0));
    }
    if z.norm() <= 0.5 {
        return dilog_power(z);
    }
    let other = one_minus(z);
    if other.norm() <= 0.5 {
        // Euler reflection. Evaluate the smaller Li2 argument as a power
        // series; the logarithms use principal branches on both banks.
        return Ok(ZETA_TWO - complex_log(z) * log_one_minus(z) - dilog_power(other)?);
    }
    if z.norm() > 1.0 {
        // Li2(z) + Li2(1/z) = -pi²/6 - log(-z)²/2.
        // Unlike log(z), log(-z) chooses the correct bank at z>1 ±i0.
        let log = complex_log(-z);
        return Ok(-ZETA_TWO - log * log * 0.5 - dilog(reciprocal(z))?);
    }
    if z.re < 0.0 {
        // Landen identity; the transformed argument has |.|<1.
        let log = log_one_minus(z);
        return Ok(-0.5 * log * log - dilog(z / (z - 1.0))?);
    }

    // The remaining lens has Re(z)>=0, |z|<=1 and |1-z|>=1/2.
    // Thus |-log(1-z)| is small and the Bernoulli series converges rapidly.
    Ok(dilog_bernoulli(-log_one_minus(z)))
}

/// Spence's integral `Li2(1-z)`, on both sides of its negative-real cut.
pub(crate) fn spence(z: Complex64) -> Result<Complex64, StrErr> {
    finite(z)?;
    if z == Complex64::new(0.0, 0.0) {
        return Ok(Complex64::new(ZETA_TWO, 0.0));
    }
    let value = if z.norm() <= 0.5 {
        // Preserve the input when 1-z would round to one. Euler reflection:
        // Li2(1-z) = pi²/6 - log(z)*log(1-z) - Li2(z).
        ZETA_TWO - complex_log(z) * log_one_minus(z) - dilog_power(z)?
    } else {
        dilog(one_minus(z))?
    };
    finite(value)
}

/// Computes x log(y) on the principal logarithm branch, with 0 log(y)=0.
pub(crate) fn xlogy(x: Complex64, y: Complex64) -> Result<Complex64, StrErr> {
    finite(x)?;
    finite(y)?;
    if x == Complex64::new(0.0, 0.0) {
        return Ok(Complex64::new(0.0, 0.0));
    }
    if y == Complex64::new(0.0, 0.0) {
        return Err("xlogy: Logarithm has a pole at zero.");
    }
    let logarithm = if (y.re - 1.0).abs() <= 0.5 && y.im.abs() <= 0.5 {
        // Sterbenz-exact subtraction near one lets the shared log1p
        // kernel retain the O(Im(y)^2) real part of log(1+i Im(y)).
        log_one_minus(one_minus(y))
    } else {
        complex_log(y)
    };
    finite(x * logarithm)
}

/// Computes x log(1+y), using log1p to retain sub-ulp complex increments.
pub(crate) fn xlog1py(x: Complex64, y: Complex64) -> Result<Complex64, StrErr> {
    finite(x)?;
    finite(y)?;
    if x == Complex64::new(0.0, 0.0) {
        return Ok(Complex64::new(0.0, 0.0));
    }
    if y == Complex64::new(-1.0, 0.0) {
        return Err("xlog1py: Logarithm has a pole at minus one.");
    }
    finite(x * log_one_minus(-y))
}
