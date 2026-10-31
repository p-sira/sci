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
