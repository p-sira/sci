//! Real and complex Bessel, Hankel, and Airy functions.
//!
//! General orders and complex arguments reuse the pure Rust AMOS translation in
//! `complex-bessel`. Specialized real orders zero and one use xsf's Cephes kernels.
//! Elliptic parameters elsewhere in the crate use m = k², as in Ellip.
use crate::backend::{
    StrErr,
    complex::{finish, promote},
    utils::real_kernel,
};
use num_complex::{Complex, Complex64};
#[allow(unused_imports)]
use sci_core::Float;

type Kernel = fn(f64, Complex64) -> Result<Complex64, complex_bessel::Error>;

pub(crate) fn backend_error(error: complex_bessel::Error) -> StrErr {
    match error {
        complex_bessel::Error::InvalidInput => "Bessel/Airy: Invalid or nonfinite argument.",
        complex_bessel::Error::Overflow => {
            "Bessel/Airy: Overflow; use an exponentially scaled function."
        }
        complex_bessel::Error::TotalPrecisionLoss => {
            "Bessel/Airy: Argument too large for meaningful precision."
        }
        complex_bessel::Error::ConvergenceFailure => "Bessel/Airy: Iteration did not converge.",
    }
}

fn complex_bessel<T: Float>(v: T, z: Complex<T>, kernel: Kernel) -> Result<Complex<T>, StrErr> {
    let z = promote(z)?;
    let v = v.to_f64().ok_or("Bessel: Invalid order.")?;
    if !v.is_finite() {
        return Err("Bessel: Order must be finite.");
    }
    let value = kernel(v, z).map_err(backend_error)?;
    finish(value)
}

macro_rules! bessel_complex {
    ($($name:ident, $kernel:ident, $doc:literal;)+) => {$(
        #[doc = $doc]
        pub fn $name<T: Float>(v: T, z: Complex<T>) -> Result<Complex<T>, StrErr> {
            complex_bessel(v,z,complex_bessel::$kernel)
        }
    )+};
}
bessel_complex! {
    cyl_bessel_j, besselj, "Complex J_v(z), with a negative-real branch cut for noninteger order.";
    cyl_bessel_je, besselj_scaled, "Complex exp(-|Im(z)|)*J_v(z).";
    cyl_bessel_y, bessely, "Complex Y_v(z), with a branch cut on the negative real axis.";
    cyl_bessel_ye, bessely_scaled, "Complex exp(-|Im(z)|)*Y_v(z).";
    cyl_bessel_i, besseli, "Complex modified Bessel I_v(z).";
    cyl_bessel_ie, besseli_scaled, "Complex exp(-|Re(z)|)*I_v(z).";
    cyl_bessel_k, besselk, "Complex modified Bessel K_v(z); negative orders use K_-v = K_v.";
    cyl_bessel_ke, besselk_scaled, "Complex exp(z)*K_v(z).";
    cyl_hankel_1, hankel1, "Hankel function H_v^(1)(z).";
    cyl_hankel_1e, hankel1_scaled, "Scaled Hankel function exp(-i*z)*H_v^(1)(z).";
    cyl_hankel_2, hankel2, "Hankel function H_v^(2)(z).";
    cyl_hankel_2e, hankel2_scaled, "Scaled Hankel function exp(i*z)*H_v^(2)(z).";
}

/// Tuple (Ai, Ai′, Bi, Bi′), preserving xsf's output ordering.
pub type Airy<T> = (T, T, T, T);

fn airy_values(z: Complex64, scaled: bool) -> Result<Airy<Complex64>, StrErr> {
    let (ai, aip, bi, bip) = if scaled {
        (
            complex_bessel::airy_scaled(z),
            complex_bessel::airyprime_scaled(z),
            complex_bessel::biry_scaled(z),
            complex_bessel::biryprime_scaled(z),
        )
    } else {
        (
            complex_bessel::airy(z),
            complex_bessel::airyprime(z),
            complex_bessel::biry(z),
            complex_bessel::biryprime(z),
        )
    };
    Ok((
        ai.map_err(backend_error)?,
        aip.map_err(backend_error)?,
        bi.map_err(backend_error)?,
        bip.map_err(backend_error)?,
    ))
}

/// Complex Airy functions (Ai, Ai′, Bi, Bi′).
pub fn airy<T: Float>(z: Complex<T>) -> Result<Airy<Complex<T>>, StrErr> {
    let (ai, aip, bi, bip) = airy_values(promote(z)?, false)?;
    Ok((finish(ai)?, finish(aip)?, finish(bi)?, finish(bip)?))
}
/// Scaled complex Airy functions (Ai*exp(ζ), Ai′*exp(ζ), Bi*exp(-|Re ζ|), Bi′*exp(-|Re ζ|)),
/// where ζ = 2*z*sqrt(z)/3.
pub fn airye<T: Float>(z: Complex<T>) -> Result<Airy<Complex<T>>, StrErr> {
    let (ai, aip, bi, bip) = airy_values(promote(z)?, true)?;
    Ok((finish(ai)?, finish(aip)?, finish(bi)?, finish(bip)?))
}

pub(crate) fn airy_real<T: Float>(x: T, scaled: bool) -> Result<Airy<T>, StrErr> {
    if !x.is_finite() {
        return Err("airy: Argument must be finite.");
    }
    let x = x.to_f64().ok_or("airy: Invalid argument.")?;
    if scaled && x < 0.0 {
        return Err("airye: Real scaled Ai requires x >= 0; use complex::airye for negative x.");
    }
    if !scaled && (-10.0..=10.0).contains(&x) {
        let (a, b, c, d) = crate::backend::cephes64::airy(x);
        return Ok((
            crate::backend::utils::finish(a, "airy: Undefined result.")?,
            crate::backend::utils::finish(b, "airy: Undefined result.")?,
            crate::backend::utils::finish(c, "airy: Undefined result.")?,
            crate::backend::utils::finish(d, "airy: Undefined result.")?,
        ));
    }
    let (a, b, c, d) = airy_values(Complex64::new(x, 0.0), scaled)?;
    Ok((
        crate::backend::utils::finish(a.re, "airy: Undefined result.")?,
        crate::backend::utils::finish(b.re, "airy: Undefined result.")?,
        crate::backend::utils::finish(c.re, "airy: Undefined result.")?,
        crate::backend::utils::finish(d.re, "airy: Undefined result.")?,
    ))
}

fn modified_real(v: f64, x: f64, scaled: bool, second: bool) -> f64 {
    if !v.is_finite() || x.is_nan() {
        return f64::NAN;
    }
    if second && x < 0.0 {
        return f64::NAN;
    }
    if !second && x < 0.0 && v != v.floor() {
        return f64::NAN;
    }
    let sign = if x < 0.0 && v % 2.0 != 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    if x == 0.0 {
        if second || (v < 0.0 && v != v.floor()) {
            return f64::INFINITY;
        }
        return if v == 0.0 { 1.0 } else { 0.0 };
    }
    if x.is_infinite() {
        return if second || scaled {
            0.0
        } else {
            sign * f64::INFINITY
        };
    }
    if x > 1e9_f64.max(100.0 * v * v) {
        let mu = 4.0 * v * v;
        let mut sum = 1.0;
        let mut term = 1.0;
        for k in 1..=12 {
            let odd = (2 * k - 1) as f64;
            term *= ((mu - odd * odd) / x) / (8.0 * k as f64);
            if !second {
                term = -term;
            }
            sum += term;
            if term.abs() <= f64::EPSILON * sum.abs() {
                break;
            }
        }
        let prefactor = if second {
            core::f64::consts::FRAC_PI_2.sqrt() / x.sqrt()
        } else {
            1.0 / (2.0 * core::f64::consts::PI).sqrt() / x.sqrt()
        };
        if scaled {
            return sign * prefactor * sum;
        }
        return if second { 0.0 } else { sign * f64::INFINITY };
    }
    let kernel: Kernel = match (second, scaled) {
        (true, true) => complex_bessel::besselk_scaled,
        (true, false) => complex_bessel::besselk,
        (false, true) => complex_bessel::besseli_scaled,
        (false, false) => complex_bessel::besseli,
    };
    match kernel(v, Complex64::new(x, 0.0)) {
        Ok(z) => sign * z.re,
        Err(complex_bessel::Error::Overflow) => {
            if !second && !scaled && x > 700.0 {
                match complex_bessel::besseli_scaled(v, Complex64::new(x, 0.0)) {
                    Ok(z) => {
                        let e = (0.5 * x).exp();
                        sign * (z.re * e) * e
                    }
                    Err(_) => sign * f64::INFINITY,
                }
            } else if second || !scaled {
                sign * f64::INFINITY
            } else {
                f64::NAN
            }
        }
        _ => f64::NAN,
    }
}
real_kernel! {
    /// Real modified Bessel I_v(x). Negative x requires integer v.
    iv(v,x) { modified_real(v,x,false,false) }
}
real_kernel! {
    /// Scaled modified Bessel exp(-|x|)*I_v(x). Negative x requires integer v.
    ive(v,x) { modified_real(v,x,true,false) }
}
real_kernel! {
    /// Real modified Bessel K_v(x), for x >= 0. K_-v = K_v.
    kv(v,x) { modified_real(v,x,false,true) }
}
real_kernel! {
    /// Scaled modified Bessel exp(x)*K_v(x), for x >= 0.
    kve(v,x) { modified_real(v,x,true,true) }
}

pub(crate) fn real_y(v: f64, x: f64) -> f64 {
    if !v.is_finite() || !x.is_finite() || x < 0.0 {
        return f64::NAN;
    }
    if v.abs() < 10000.0 {
        return crate::backend::cephes64::yv(v, x);
    }
    // Avoid converting huge orders into machine integers or performing a
    // recurrence whose running time is proportional to the order.
    match complex_bessel::bessely(v, Complex64::new(x, 0.0)) {
        Ok(z) => z.re,
        Err(_) => f64::NAN,
    }
}
