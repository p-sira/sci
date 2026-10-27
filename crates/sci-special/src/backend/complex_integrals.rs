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

fn near_real(z: Complex64) -> bool {
    z.im.abs() <= 1e-4 && z.im.abs() <= 1e-4 * z.re.abs()
}

fn real_integrals(z: Complex64, hyperbolic: bool) -> Result<Pair, StrErr> {
    let (s, c) = if hyperbolic {
        crate::backend::shichi(z.re)?
    } else {
        crate::backend::sici(z.re)?
    };
    let mut result = (
        Complex64::new(s, 0.0_f64.copysign(z.im)),
        Complex64::new(
            c,
            if z.re < 0.0 {
                PI.copysign(z.im)
            } else {
                0.0_f64.copysign(z.im)
            },
        ),
    );
    if z.im == 0.0 {
        return Ok(result);
    }
    // Divide the sin/cos (or sinh/cosh) Taylor series by x+d and integrate.
    let (mut s, mut c) = if hyperbolic {
        (z.re.sinh(), z.re.cosh())
    } else {
        z.re.sin_cos()
    };
    let sign = if hyperbolic { 1.0 } else { -1.0 };
    let mut gs = 0.0;
    let mut gc = 0.0;
    let delta = Complex64::new(0.0, z.im);
    let mut power = delta;
    for n in 0..8 {
        gs = (s - gs) / z.re;
        gc = (c - gc) / z.re;
        result.0 += power * (gs / (n + 1) as f64);
        result.1 += power * (gc / (n + 1) as f64);
        (s, c) = (c / (n + 1) as f64, sign * s / (n + 1) as f64);
        power *= delta;
    }
    Ok(result)
}

// One factorial series serves both the circular and hyperbolic integrals.
fn integral_series(z: Complex64, hyperbolic: bool) -> Pair {
    let square = if hyperbolic { z * z } else { -z * z };
    let mut odd = z;
    let mut even = square * 0.5;
    let mut s = odd;
    let mut c = even * 0.5;
    for k in 1..=64 {
        let n = 2.0 * k as f64;
        odd *= square / (n * (n + 1.0));
        even *= square / ((n + 1.0) * (n + 2.0));
        let ds = odd / (n + 1.0);
        let dc = even / (n + 2.0);
        s += ds;
        c += dc;
        if ds.norm() <= f64::EPSILON * s.norm() && dc.norm() <= f64::EPSILON * c.norm() {
            break;
        }
    }
    (s, c + z.ln() + 0.5772156649015328606)
}

fn integrals(z: Complex64, hyperbolic: bool) -> Result<Pair, StrErr> {
    if z == Complex64::new(0.0, 0.0) {
        return Ok((z, Complex64::new(f64::NEG_INFINITY, z.im)));
    }
    if z.im == 0.0 {
        return real_integrals(z, hyperbolic);
    }
    finite(z)?;
    if z.norm() < 0.8 {
        return Ok(integral_series(z, hyperbolic));
    }
    if near_real(z) {
        return real_integrals(z, hyperbolic);
    }
    let rotated = Complex64::new(z.im, -z.re);
    if near_real(rotated) {
        let (s, c) = real_integrals(rotated, !hyperbolic)?;
        return Ok((Complex64::new(-s.im, s.re), c + z.ln() - rotated.ln()));
    }
    let argument = if hyperbolic {
        z
    } else {
        Complex64::new(-z.im, z.re)
    };
    let positive = crate::backend::complex::expi(argument)?;
    let negative = crate::backend::complex::expi(-argument)?;
    let mut c = (positive + negative) * 0.5;
    let s = if hyperbolic {
        let correction = Complex64::new(0.0, FRAC_PI_2.copysign(z.im));
        c += correction;
        (positive - negative) * 0.5 - correction
    } else {
        if z.re == 0.0 {
            c.im += FRAC_PI_2.copysign(z.im);
        } else if z.re < 0.0 {
            c.im += PI.copysign(z.im);
        }
        let mut s = Complex64::new(0.0, -0.5) * (positive - negative);
        if z.re != 0.0 {
            s.re -= FRAC_PI_2.copysign(z.re);
        }
        s
    };
    Ok((s, c))
}

macro_rules! integral_api {
    ($($name:ident, $hyperbolic:literal, $doc:literal;)+) => {$(
        #[doc = $doc]
        pub fn $name<T: Float>(z: Complex<T>) -> Result<(Complex<T>, Complex<T>), StrErr> {
            let (s,c) = integrals(promote(z)?, $hyperbolic)?;
            Ok((finish(s)?, finish(c)?))
        }
    )+};
}
integral_api! {
    sici, false, "Complex sine and cosine integrals (Si, Ci). Ci has a negative-real branch cut, retaining signed imaginary zero; Ci(0) = -infinity.";
    shichi, true, "Complex hyperbolic sine and cosine integrals (Shi, Chi). Chi has a negative-real branch cut, retaining signed imaginary zero; Chi(0) = -infinity.";
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

// B_(2k)/(2k+1)!, in ascending order. Reflection and inversion keep the
// logarithm in a small disk, so twelve terms suffice in double precision.
const DILOG_COEFFICIENTS: [f64; 12] = [
    2.7777777777777777777777777777778e-2,
    -2.777777777777777777777777777778e-4,
    4.7241118669690098261526832955404e-6,
    -9.1857730746619635508524397413286e-8,
    1.8978869988970999072009173019274e-9,
    -4.0647616451442255268059093862920e-11,
    8.9216910204564525552179873167527e-13,
    -1.9939295860721075687236443477938e-14,
    4.5189800296199181916504765528556e-16,
    -1.0356517612181247014483411542219e-17,
    2.3952186210261867457402837430010e-19,
    -5.5817858743250093362830745056254e-21,
];

fn dilog(z: Complex64) -> Complex64 {
    const ZETA_2: f64 = 1.6449340668482264365;
    if z == Complex64::new(0.0, 0.0) {
        return z;
    }
    if z == Complex64::new(1.0, 0.0) {
        return Complex64::new(ZETA_2, 0.0);
    }
    if z.norm() > 1.0 {
        let logarithm = (-z).ln();
        return -dilog(crate::backend::complex::reciprocal(z))
            - ZETA_2
            - 0.5 * logarithm * logarithm;
    }
    if z.re > 0.5 {
        let complement = Complex64::new(1.0 - z.re, -z.im);
        return ZETA_2 - z.ln() * complement.ln() - dilog(complement);
    }
    let u = -crate::backend::complex::log1p_kernel(-z);
    let square = u * u;
    let series = DILOG_COEFFICIENTS
        .iter()
        .rev()
        .fold(Complex64::new(0.0, 0.0), |sum, &coefficient| {
            sum * square + coefficient
        });
    u - square * 0.25 + u * square * series
}

/// Complex Spence function Li₂(1-z), with a cut on the negative real axis.
/// Requires finite arguments; signed imaginary zero selects the side of the cut.
pub fn spence<T: Float>(z: Complex<T>) -> Result<Complex<T>, StrErr> {
    let z = finite(promote(z)?)?;
    // Near zero, reflect before forming 1-z so tiny imaginary parts survive.
    let value = if z == Complex64::new(0.0, 0.0) {
        Complex64::new(1.6449340668482264365, 0.0)
    } else if z.norm() < 0.5 {
        1.6449340668482264365 - dilog(z) - z.ln() * crate::backend::complex::log1p(-z)?
    } else {
        dilog(Complex64::new(1.0 - z.re, -z.im))
    };
    finish(value)
}
