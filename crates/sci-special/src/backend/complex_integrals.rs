//! Integral identities reusing the complex exponential and error functions.
use crate::backend::{
    StrErr,
    complex::{finish, finite, promote},
};
use core::f64::consts::{FRAC_PI_2, PI};
use num_complex::{Complex, Complex64};
#[allow(unused_imports)]
use sci_core::Float;

type Pair = (Complex64, Complex64);

const EULER: f64 = 0.577_215_664_901_532_9;
const EXPINT_TERMS: usize = 512;
const EXPINT_FRACTION_TERMS: usize = 1500;

fn expint_argument<T: Float>(z: Complex<T>) -> Result<Complex64, StrErr> {
    let z = finite(promote(z)?)?;
    if z.re == 0.0 && z.im == 0.0 {
        return Err("complex expint: Zero is a logarithmic singularity.");
    }
    // The continued fraction and the complex exponential are evaluated in
    // binary64. Very large phases require a different argument reduction.
    if z.norm() > 1.0e6 {
        return Err("complex expint: Argument magnitude exceeds 1e6.");
    }
    Ok(z)
}

/// Ei(z) = gamma + log(z) + sum(z^k/(k*k!)). The right half-plane
/// avoids exponentially large alternating terms; near the imaginary axis
/// the continued fraction is selected before this series loses precision.
fn ei_series(z: Complex64) -> Result<Complex64, StrErr> {
    let mut term = z;
    let mut sum = term;
    for k in 2..=EXPINT_TERMS {
        let k = k as f64;
        term *= z * ((k - 1.0) / (k * k));
        sum += term;
        if term.norm() <= 2.0 * f64::EPSILON * sum.norm() {
            return Ok(z.ln() + EULER + sum);
        }
    }
    Err("complex expint: Ei series failed to converge.")
}

fn exp1_series(z: Complex64) -> Result<Complex64, StrErr> {
    let mut term = -z;
    let mut sum = term;
    for k in 2..=EXPINT_TERMS {
        let k = k as f64;
        term *= -z * ((k - 1.0) / (k * k));
        sum += term;
        if term.norm() <= 2.0 * f64::EPSILON * sum.norm() {
            return Ok(-z.ln() - EULER - sum);
        }
    }
    Err("complex expint: E1 series failed to converge.")
}

/// E1(z) = exp(-z)/(z+1 - 1²/(z+3 - 2²/(z+5 - ...))).
/// Modified Lentz iteration, with explicit convergence and work bounds.
/// Applied only at arguments sufficiently far from the negative cut.
fn exp1_fraction(z: Complex64) -> Result<Complex64, StrErr> {
    let mut f = z + 1.0;
    let mut c = f;
    let mut d = Complex64::new(0.0, 0.0);
    for k in 1..=EXPINT_FRACTION_TERMS {
        let a = -((k * k) as f64);
        let b = z + (2 * k + 1) as f64;
        d = b + a * d;
        c = b + a / c;
        if d.norm() < 1e-300 || c.norm() < 1e-300 {
            return Err("complex expint: Ill-conditioned continued fraction.");
        }
        d = Complex64::new(1.0, 0.0) / d;
        let delta = c * d;
        f *= delta;
        if !f.re.is_finite() || !f.im.is_finite() {
            return Err("complex expint: Continued fraction overflow.");
        }
        if (delta - 1.0).norm() < 3e-15 {
            return Ok((-z).exp() / f);
        }
    }
    Err("complex expint: Continued fraction failed to converge.")
}

/// Optimally truncated Poincare expansion for |z| >= 36 away from the
/// logarithmic cut. Here Ei(z) ~ exp(z)/z * sum(k!/z^k).
fn ei_asymptotic(z: Complex64) -> Result<Complex64, StrErr> {
    let inv = Complex64::new(1.0, 0.0) / z;
    let mut sum = Complex64::new(1.0, 0.0);
    let mut term = sum;
    for k in 1..=200 {
        let next = term * (k as f64 * inv);
        if next.norm() > term.norm() {
            break;
        }
        term = next;
        sum += term;
        if term.norm() <= 2.0 * f64::EPSILON * sum.norm() {
            break;
        }
    }
    // exp(z-log(z)) retains values where exp(Re z) itself overflows but
    // exp(z)/z is finite. The +/- i*pi term is required near the imaginary axis.
    let constant = if z.im == 0.0 {
        Complex64::new(0.0, 0.0)
    } else {
        Complex64::new(0.0, PI.copysign(z.im))
    };
    let result = (z - z.ln()).exp() * sum + constant;
    if result.re.is_finite() && result.im.is_finite() {
        Ok(result)
    } else {
        Err("complex expint: Ei overflows binary64.")
    }
}

/// Integrate Ei'(x+w)=exp(x+w)/(x+w) along w=i*y. This avoids the
/// spurious +/- pi residual of an asymptotic expansion close to the
/// positive real axis, including when |y| is far below machine epsilon.
fn ei_near_positive_axis(x: f64, y: f64) -> Result<Complex64, StrErr> {
    let on_axis = if x <= 40.0 {
        ei_series(Complex64::new(x, 0.0))?.re
    } else {
        ei_asymptotic(Complex64::new(x, 0.0))?.re
    };
    let factor = (x - x.ln()).exp();
    if !factor.is_finite() {
        return Err("complex expint: Ei overflows binary64.");
    }
    let w = Complex64::new(0.0, y);
    let mut power = w;
    let mut correction = w;
    let mut coeff = 1.0;
    let mut inverse_factorial = 1.0;
    for k in 1..=96 {
        inverse_factorial /= k as f64;
        coeff = inverse_factorial - coeff / x;
        power *= w;
        let term = power * (coeff / (k + 1) as f64);
        correction += term;
        if term.norm() <= f64::EPSILON * correction.norm() {
            break;
        }
    }
    let result = Complex64::new(on_axis, 0.0) + factor * correction;
    if result.re.is_finite() && result.im.is_finite() {
        Ok(result)
    } else {
        Err("complex expint: Ei overflows binary64.")
    }
}

fn ei_right(z: Complex64) -> Result<Complex64, StrErr> {
    debug_assert!(z.re >= 0.0);
    if z.re >= 22.0 && z.im.abs() <= 8.0 {
        return ei_near_positive_axis(z.re, z.im);
    }
    if z.im.abs() >= 2.0 && z.norm() >= 9.0 {
        let e1 = exp1_fraction(-z)?;
        return Ok(-e1 + Complex64::new(0.0, PI.copysign(z.im)));
    }
    if z.norm() > 36.0 {
        return ei_asymptotic(z);
    }
    ei_series(z)
}

fn exp1_right(z: Complex64) -> Result<Complex64, StrErr> {
    debug_assert!(z.re >= 0.0);
    if z.im == 0.0 {
        return Ok(Complex64::new(crate::backend::expint::exp1_real(z.re), 0.0));
    }
    if z.norm() < 3.0 {
        exp1_series(z)
    } else {
        exp1_fraction(z)
    }
}

/// Complex exponential integral E1(z), on the principal branch.
/// Negative-real signed imaginary zeros select the upper/lower bank.
/// Zero is a logarithmic pole and returns an error. All loops are bounded.
pub fn exp1<T: Float>(z: Complex<T>) -> Result<Complex<T>, StrErr> {
    let z = expint_argument(z)?;
    let value = if z.re >= 0.0 {
        exp1_right(z)?
    } else {
        let reflected = ei_right(-z)?;
        -reflected - Complex64::new(0.0, PI.copysign(z.im))
    };
    finish(value)
}

/// Complex exponential integral Ei(z), analytic from the positive real axis.
/// On the negative real cut the value is real; either open bank differs by
/// +/- i*pi. Zero is a logarithmic pole and returns an error.
pub fn expi<T: Float>(z: Complex<T>) -> Result<Complex<T>, StrErr> {
    let z = expint_argument(z)?;
    let value = if z.re >= 0.0 {
        ei_right(z)?
    } else {
        let e1 = exp1_right(-z)?;
        if z.im == 0.0 {
            Complex64::new(-e1.re, -0.0)
        } else {
            -e1 + Complex64::new(0.0, PI.copysign(z.im))
        }
    };
    finish(value)
}

fn near_real(z: Complex64) -> bool {
    z.im.abs() <= 1e-4 && z.im.abs() <= 1e-4 * z.re.abs()
}

fn fresnel_series(z: Complex64) -> Pair {
    let q = z * z * FRAC_PI_2;
    let square = -q * q;
    let mut ct = z;
    let mut st = z * q / 3.0;
    let mut c = ct;
    let mut s = st;
    for k in 1..=32 {
        let k = k as f64;
        ct *= square * ((4.0 * k - 3.0) / (4.0 * k + 1.0)) / ((2.0 * k - 1.0) * (2.0 * k));
        st *= square * ((4.0 * k - 1.0) / (4.0 * k + 3.0)) / ((2.0 * k) * (2.0 * k + 1.0));
        c += ct;
        s += st;
        if ct.norm() <= f64::EPSILON * c.norm() && st.norm() <= f64::EPSILON * s.norm() {
            break;
        }
    }
    (s, c)
}

fn real_fresnel(z: Complex64) -> Result<Pair, StrErr> {
    let (s, c) = crate::backend::fresnel(z.re)?;
    let zero = 0.0_f64.copysign(z.im);
    let mut result = (Complex64::new(s, zero), Complex64::new(c, zero));
    if z.im == 0.0 {
        return Ok(result);
    }
    // Above 2^53 every representable x is even, so x*x/2 is an even integer.
    let phase = if z.re.abs() >= 9_007_199_254_740_992.0 {
        0.0
    } else {
        0.5 * z.re * z.re
    };
    let mut coefficient = Complex64::new(
        crate::backend::cephes64::cospi(phase),
        crate::backend::cephes64::sinpi(phase),
    );
    let mut previous = Complex64::new(0.0, 0.0);
    let mut power = Complex64::new(0.0, z.im);
    for n in 0..12 {
        result.0 += power * (coefficient.im / (n + 1) as f64);
        result.1 += power * (coefficient.re / (n + 1) as f64);
        // Fold y^n into each coefficient to avoid overflow of x^n when
        // a large real argument has a correspondingly tiny imaginary part.
        let next = Complex64::new(0.0, PI)
            * ((z.re * z.im) * coefficient + (z.im * z.im) * previous)
            / (n + 1) as f64;
        previous = coefficient;
        coefficient = next;
        power = Complex64::new(-power.im, power.re);
    }
    Ok(result)
}

/// Complex Fresnel integrals (S, C), with phase pi*t²/2.
///
/// Real and imaginary axes reuse the real Fresnel kernel. Elsewhere the error
/// function identity is combined with a series near zero to retain small S.
pub fn fresnel<T: Float>(z: Complex<T>) -> Result<(Complex<T>, Complex<T>), StrErr> {
    let z = promote(z)?;
    let (s, c) = if z.im == 0.0 {
        real_fresnel(z)?
    } else if z.re == 0.0 {
        let (s, c) = crate::backend::fresnel(z.im)?;
        (Complex64::new(z.re, -s), Complex64::new(z.re, c))
    } else {
        finite(z)?;
        if z.norm() < 0.8 {
            fresnel_series(z)
        } else if near_real(z) && (z.re * z.im).abs() <= 1e-3 {
            real_fresnel(z)?
        } else if near_real(Complex64::new(z.im, -z.re)) && (z.re * z.im).abs() <= 1e-3 {
            let (s, c) = real_fresnel(Complex64::new(z.im, -z.re))?;
            (Complex64::new(s.im, -s.re), Complex64::new(-c.im, c.re))
        } else {
            let scale = PI.sqrt() * 0.5;
            let a = Complex64::new(0.5, 0.5)
                * crate::backend::complex::erf(Complex64::new(scale, -scale) * z)?;
            let b = Complex64::new(0.5, -0.5)
                * crate::backend::complex::erf(Complex64::new(scale, scale) * z)?;
            (Complex64::new(0.0, -0.5) * (a - b), 0.5 * (a + b))
        }
    };
    Ok((finish(s)?, finish(c)?))
}
