//! Real and complex Bessel, Hankel, and Airy functions.
//!
//! Real modified I uses Cephes/Temme kernels. Complex arguments and general K
//! reuse the pure Rust AMOS translation in `complex-bessel`. Specialized real
//! orders zero and one use XSF's Cephes kernels.
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
    // `frem` is speculatable in LLVM, so keep the libm `fmod` call behind an
    // explicit branch rather than a short-circuiting condition.
    let sign = if x < 0.0 {
        if v % 2.0 != 0.0 { -1.0 } else { 1.0 }
    } else {
        1.0
    };
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
    // DLMF 10.39.1–2: elementary half-integer values also avoid the
    // cancellation-sensitive general AMOS transition at moderate x.
    if v.abs() == 0.5 {
        if second {
            let prefactor = (core::f64::consts::FRAC_PI_2 / x).sqrt();
            return if scaled {
                prefactor
            } else {
                prefactor * (-x).exp()
            };
        }
        let prefactor = (2.0 * core::f64::consts::PI).sqrt().recip() / x.sqrt();
        let difference = if v > 0.0 {
            -(-2.0 * x).exp_m1()
        } else {
            1.0 + (-2.0 * x).exp()
        };
        let value = prefactor * difference;
        return if scaled {
            value
        } else {
            let half = (0.5 * x).exp();
            (value * half) * half
        };
    }
    if !second {
        return sign * crate::backend::cephes64::iv::iv(v, x, scaled);
    }
    if x > 1e9_f64.max(100.0 * v * v) {
        let mu = 4.0 * v * v;
        let mut sum = 1.0;
        let mut term = 1.0;
        for k in 1..=12 {
            let odd = (2 * k - 1) as f64;
            term *= ((mu - odd * odd) / x) / (8.0 * k as f64);
            sum += term;
            if term.abs() <= f64::EPSILON * sum.abs() {
                break;
            }
        }
        return if scaled {
            core::f64::consts::FRAC_PI_2.sqrt() / x.sqrt() * sum
        } else {
            0.0
        };
    }
    let kernel: Kernel = if scaled {
        complex_bessel::besselk_scaled
    } else {
        complex_bessel::besselk
    };
    match kernel(v, Complex64::new(x, 0.0)) {
        Ok(z) => z.re,
        Err(complex_bessel::Error::Overflow) => f64::INFINITY,
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

bessel_complex! {
    cyl_hankel_1, hankel1, "Hankel function H_v^(1)(z).";
    cyl_hankel_1e, hankel1_scaled, "Scaled Hankel function exp(-i*z)*H_v^(1)(z).";
    cyl_hankel_2, hankel2, "Hankel function H_v^(2)(z).";
    cyl_hankel_2e, hankel2_scaled, "Scaled Hankel function exp(i*z)*H_v^(2)(z).";
}
