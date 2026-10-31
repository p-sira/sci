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
    // The Cephes real Airy kernels are accurate through this moderate positive
    // interval. Apply the real exponential factors once to all four outputs;
    // this avoids four separate AMOS complex-Bessel evaluations for airye(x).
    // Beyond it, keep the natively scaled kernels to avoid cancellation and
    // overflow in the unscaled Ai/Bi values.
    if scaled && x <= 20.0 {
        let (ai, aip, bi, bip) = crate::backend::cephes64::airy(x);
        let zeta = (2.0 / 3.0) * x * x.sqrt();
        let exp = zeta.exp();
        let inv_exp = (-zeta).exp();
        return Ok((
            crate::backend::utils::finish(ai * exp, "airye: Undefined result.")?,
            crate::backend::utils::finish(aip * exp, "airye: Undefined result.")?,
            crate::backend::utils::finish(bi * inv_exp, "airye: Undefined result.")?,
            crate::backend::utils::finish(bip * inv_exp, "airye: Undefined result.")?,
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

#[cfg(test)]
mod airy_tests {
    use super::airy_real;

    #[test]
    fn scaled_real_airy_matches_high_precision_at_kernel_transitions() {
        // mpmath 1.4, 70-digit Ai, Ai', Bi, Bi' with exp(±2x^(3/2)/3).
        let references: [(f64, [f64; 4]); 14] = [
            (
                0.0,
                [
                    0.3550280538878172,
                    -0.2588194037928068,
                    0.6149266274460007,
                    0.44828835735382636,
                ],
            ),
            (
                0.5,
                [
                    0.29327715912994736,
                    -0.28469116209194257,
                    0.6748924111156302,
                    0.4302209614637694,
                ],
            ),
            (
                1e-12,
                [
                    0.3550280538875584,
                    -0.2588194037928068,
                    0.614926627446449,
                    0.4482883573538264,
                ],
            ),
            (
                2.09,
                [
                    0.22801909727853855,
                    -0.35295656709963385,
                    0.49299852416324267,
                    0.6328540952588387,
                ],
            ),
            (
                2.1,
                [
                    0.22778653656129074,
                    -0.3532931777439597,
                    0.492202880303034,
                    0.6340057173130742,
                ],
            ),
            (
                4.0,
                [
                    0.1970948026430665,
                    -0.40568791157876305,
                    0.4048094678892981,
                    0.7817739306272438,
                ],
            ),
            (
                8.3203353,
                [
                    0.16539804113168443,
                    -0.48193818011893056,
                    0.33368593640111104,
                    0.9522113574366132,
                ],
            ),
            (
                8.33,
                [
                    0.16535122268088864,
                    -0.482073267100383,
                    0.33358641257956495,
                    0.9524985171426074,
                ],
            ),
            (
                12.0,
                [
                    0.15119256068463708,
                    -0.5268505009124518,
                    0.3039054138807329,
                    1.0463290385080119,
                ],
            ),
            (
                19.999,
                [
                    0.1332420614603509,
                    -0.5975158764105255,
                    0.2671056874514886,
                    1.1911403614781674,
                ],
            ),
            (
                20.0,
                [
                    0.13324040735181362,
                    -0.5975232737163193,
                    0.2671023248341684,
                    1.1911553993720082,
                ],
            ),
            (
                20.0001,
                [
                    0.13324024194659163,
                    -0.5975240134318844,
                    0.26710198858411147,
                    1.1911569031298575,
                ],
            ),
            (
                30.0,
                [
                    0.12045939663973669,
                    -0.6607833325212036,
                    0.24122445526882688,
                    1.3192228350679096,
                ],
            ),
            (
                100.0,
                [
                    0.08919692093633041,
                    -0.8921920625040315,
                    0.1784310111708354,
                    1.7838637549628087,
                ],
            ),
        ];
        for (x, expected) in references {
            let (ai, aip, bi, bip) = airy_real(x, true).unwrap();
            for (actual, reference) in [ai, aip, bi, bip].into_iter().zip(expected) {
                assert!(
                    (actual - reference).abs() <= 1e-12 * reference.abs(),
                    "airye({x}) = {actual}, expected {reference}"
                );
            }
            // W[Ai,Bi] = 1/pi; opposing exponential scalings cancel.
            assert!(
                (ai * bip - aip * bi - core::f64::consts::FRAC_1_PI).abs() < 1e-12,
                "Wronskian at x={x}"
            );
        }
    }
}
