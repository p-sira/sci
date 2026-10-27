//! Complex special functions. Branch cuts follow xsf and retain signed imaginary zero.
//!
//! Adapted from xsf/loggamma.h, trig.h, expint.h, and lambertw.h.
//! Original authors: Josh Wilson (log-Gamma and trigonometry), Yosef Meller
//! (Lambert W), and Zhang and Jin (exponential integrals); see LICENSES.
use crate::backend::StrErr;
use core::f64::consts::{E, PI};
use num_complex::{Complex, Complex64};
#[allow(unused_imports)]
use sci_core::Float;

pub use crate::backend::complex_integrals::{fresnel, shichi, sici, spence};

pub(crate) fn promote<T: Float>(z: Complex<T>) -> Result<Complex64, StrErr> {
    if z.re.is_nan() || z.im.is_nan() {
        return Err("complex: Arguments cannot be NAN.");
    }
    Ok(Complex64::new(
        z.re.to_f64().ok_or("complex: Invalid argument.")?,
        z.im.to_f64().ok_or("complex: Invalid argument.")?,
    ))
}
pub(crate) fn finish<T: Float>(z: Complex64) -> Result<Complex<T>, StrErr> {
    if z.re.is_nan() || z.im.is_nan() {
        return Err("complex: Undefined result.");
    }
    Ok(Complex::new(
        T::from(z.re).ok_or("complex: Invalid result.")?,
        T::from(z.im).ok_or("complex: Invalid result.")?,
    ))
}
pub(crate) fn finite(z: Complex64) -> Result<Complex64, StrErr> {
    if z.re.is_finite() && z.im.is_finite() {
        Ok(z)
    } else {
        Err("complex: Arguments must be finite.")
    }
}
fn polynomial(z: Complex64, coefficients: &[f64]) -> Complex64 {
    coefficients
        .iter()
        .fold(Complex64::new(0.0, 0.0), |value, &coefficient| {
            value * z + coefficient
        })
}

fn trig_pi(z: Complex64, cosine: bool) -> Complex64 {
    let s = crate::backend::cephes64::sinpi(z.re);
    let c = crate::backend::cephes64::cospi(z.re);
    let y = PI * z.im;
    let (a, b) = if cosine {
        (c, -s * 1.0_f64.copysign(y))
    } else {
        (s, c * 1.0_f64.copysign(y))
    };
    if y.abs() < 700.0 {
        return if cosine {
            Complex64::new(c * y.cosh(), -s * y.sinh())
        } else {
            Complex64::new(s * y.cosh(), c * y.sinh())
        };
    }
    let e = (y.abs() / 2.0).exp();
    let scaled = |v: f64| {
        if e.is_infinite() {
            if v == 0.0 {
                v
            } else {
                f64::INFINITY.copysign(v)
            }
        } else {
            (0.5 * v * e) * e
        }
    };
    Complex64::new(scaled(a), scaled(b))
}

/// sin(pi*z), with exact real-axis reduction and overflow-aware hyperbolic factors.
pub fn sinpi<T: Float>(z: Complex<T>) -> Result<Complex<T>, StrErr> {
    finish(trig_pi(finite(promote(z)?)?, false))
}
/// cos(pi*z), with exact real-axis reduction and overflow-aware hyperbolic factors.
pub fn cospi<T: Float>(z: Complex<T>) -> Result<Complex<T>, StrErr> {
    finish(trig_pi(finite(promote(z)?)?, true))
}

const STIRLING: [f64; 8] = [
    -2.955065359477124183e-2,
    6.4102564102564102564e-3,
    -1.9175269175269175269e-3,
    8.4175084175084175084e-4,
    -5.952380952380952381e-4,
    7.9365079365079365079e-4,
    -2.7777777777777777778e-3,
    8.3333333333333333333e-2,
];
const TAYLOR: [f64; 23] = [
    -4.3478266053040259361e-2,
    4.5454556293204669442e-2,
    -4.7619070330142227991e-2,
    5.000004769810169364e-2,
    -5.2631679379616660734e-2,
    5.5555767627403611102e-2,
    -5.8823978658684582339e-2,
    6.2500955141213040742e-2,
    -6.6668705882420468033e-2,
    7.1432946295361336059e-2,
    -7.6932516411352191473e-2,
    8.3353840546109004025e-2,
    -9.0954017145829042233e-2,
    1.0009945751278180853e-1,
    -1.1133426586956469049e-1,
    1.2550966952474304242e-1,
    -1.4404989676884611812e-1,
    1.6955717699740818995e-1,
    -2.0738555102867398527e-1,
    2.7058080842778454788e-1,
    -4.0068563438653142847e-1,
    8.2246703342411321824e-1,
    -5.7721566490153286061e-1,
];
fn stirling(z: Complex64) -> Complex64 {
    let reciprocal = reciprocal(z);
    (z - 0.5) * z.ln() - z
        + 0.918938533204672742
        + reciprocal * polynomial(reciprocal / z, &STIRLING)
}
fn taylor(z: Complex64) -> Complex64 {
    let z = z - 1.0;
    z * polynomial(z, &TAYLOR)
}
fn recurrence(mut z: Complex64) -> Complex64 {
    let mut flips = 0;
    let mut previous_sign = false;
    let mut product = z;
    z += 1.0;
    while z.re <= 7.0 {
        product *= z;
        let negative = product.im.is_sign_negative();
        if negative && !previous_sign {
            flips += 1;
        }
        previous_sign = negative;
        z += 1.0;
    }
    stirling(z) - product.ln() - Complex64::new(0.0, flips as f64 * 2.0 * PI)
}
fn loggamma_kernel(z: Complex64) -> Complex64 {
    if z.re > 7.0 || z.im.abs() > 7.0 {
        return stirling(z);
    }
    if (z - 1.0).norm() < 0.2 {
        return taylor(z);
    }
    if (z - 2.0).norm() < 0.2 {
        return log_near_one(z - 1.0) + taylor(z - 1.0);
    }
    if z.re < 0.1 {
        let phase = (2.0 * PI).copysign(z.im) * (0.5 * z.re + 0.25).floor();
        return Complex64::new(1.1447298858494001741434262, phase)
            - trig_pi(z, false).ln()
            - loggamma_kernel(1.0 - z);
    }
    if !z.im.is_sign_negative() {
        recurrence(z)
    } else {
        recurrence(z.conj()).conj()
    }
}
fn pole(z: Complex64) -> bool {
    z.im == 0.0 && z.re <= 0.0 && z.re == z.re.floor()
}

/// Analytic log-Gamma with a branch cut on the negative real axis.
/// This is the continuous log-Gamma branch, whose imaginary part may exceed pi.
/// Nonpositive real integers are poles and return an error.
pub fn loggamma<T: Float>(z: Complex<T>) -> Result<Complex<T>, StrErr> {
    let z = finite(promote(z)?)?;
    if pole(z) {
        return Err("loggamma: Nonpositive integers are poles.");
    }
    finish(loggamma_kernel(z))
}
/// Complex Gamma function. Nonpositive real integers are poles.
pub fn gamma<T: Float>(z: Complex<T>) -> Result<Complex<T>, StrErr> {
    let z = finite(promote(z)?)?;
    if pole(z) {
        return Err("gamma: Nonpositive integers are poles.");
    }
    if z.re <= -9007199254740992.0 {
        return finish(Complex64::new(0.0, 0.0_f64.copysign(z.im)));
    }
    finish(exp_kernel(loggamma_kernel(z)))
}
/// Complex reciprocal Gamma, including zeros at nonpositive real integers.
pub fn rgamma<T: Float>(z: Complex<T>) -> Result<Complex<T>, StrErr> {
    let z = finite(promote(z)?)?;
    if pole(z) {
        return finish(Complex64::new(0.0, 0.0));
    }
    finish(exp_kernel(-loggamma_kernel(z)))
}

fn exp1_kernel(z: Complex64) -> Complex64 {
    const EULER: f64 = 0.5772156649015328606;
    if z.norm() == 0.0 {
        return Complex64::new(f64::INFINITY, 0.0);
    }
    let mut result;
    if z.norm() < 5.0 || (z.re < -2.0 * z.im.abs() && z.norm() < 40.0) {
        let mut sum = Complex64::new(1.0, 0.0);
        let mut term = sum;
        for k in 1..501 {
            let k = k as f64;
            term = -term * z * k / ((k + 1.0) * (k + 1.0));
            sum += term;
            if term.norm() < sum.norm() * 1e-15 {
                break;
            }
        }
        if z.re <= 0.0 && z.im == 0.0 {
            result = -EULER - (-z).ln() + z * sum - Complex64::new(0.0, PI.copysign(z.im));
        } else {
            result = -EULER - z.ln() + z * sum;
        }
    } else {
        let mut d = reciprocal(z);
        let mut delta = d;
        let mut sum = d;
        for k in 1..501 {
            let k = k as f64;
            d = reciprocal(d * k + 1.0);
            delta *= d - 1.0;
            sum += delta;
            d = reciprocal(d * k + z);
            delta *= z * d - 1.0;
            sum += delta;
            if delta.norm() <= sum.norm() * 1e-15 && k > 20.0 {
                break;
            }
        }
        result = (-z).exp() * sum;
        if z.re <= 0.0 && z.im == 0.0 {
            result.im -= PI.copysign(z.im);
        }
    }
    result
}
/// Complex exponential integral E₁(z), with a branch cut on the negative real axis.
pub fn exp1<T: Float>(z: Complex<T>) -> Result<Complex<T>, StrErr> {
    finish(exp1_kernel(finite(promote(z)?)?))
}
/// Complex exponential integral Ei(z), with a branch cut on the negative real axis.
pub fn expi<T: Float>(z: Complex<T>) -> Result<Complex<T>, StrErr> {
    let z = finite(promote(z)?)?;
    let mut result = -exp1_kernel(-z);
    if z.im > 0.0 {
        result.im += PI;
    } else if z.im < 0.0 {
        result.im -= PI;
    } else if z.re > 0.0 {
        result.im += PI.copysign(z.im);
    }
    finish(result)
}

/// Lambert W on integer branch k, satisfying W(z)*exp(W(z)) = z.
/// Uses xsf's branch-aware initial approximations and Halley iteration.
/// Returns an error if iteration does not converge in 100 steps.
pub fn lambertw<T: Float>(z: Complex<T>, k: i32) -> Result<Complex<T>, StrErr> {
    let z = finite(promote(z)?)?;
    if z == Complex64::new(0.0, 0.0) {
        return finish(if k == 0 {
            z
        } else {
            Complex64::new(f64::NEG_INFINITY, 0.0)
        });
    }
    if z == Complex64::new(-1.0 / E, 0.0) && (k == 0 || k == -1) {
        return finish(Complex64::new(-1.0, 0.0));
    }
    if z == Complex64::new(1.0, 0.0) && k == 0 {
        return finish(Complex64::new(0.56714329040978387299997, 0.0));
    }
    let asymptotic = || {
        let w = z.ln() + Complex64::new(0.0, 2.0 * PI * k as f64);
        w - w.ln()
    };
    let mut w = if k == 0 {
        if (z + 1.0 / E).norm() < 0.3 {
            let p = (2.0 * (E * z + 1.0)).sqrt();
            -1.0 + p - p * p / 3.0
        } else if -1.0 < z.re && z.re < 1.5 && z.im.abs() < 1.0 && -2.5 * z.im.abs() - 0.2 < z.re {
            z * polynomial(z, &[12.85106382978723404255, 12.34042553191489361902, 1.0])
                / polynomial(z, &[32.53191489361702127660, 14.34042553191489361702, 1.0])
        } else {
            asymptotic()
        }
    } else if k == -1 && z.norm() <= 1.0 / E && z.im == 0.0 && z.re < 0.0 {
        Complex64::new((-z.re).ln(), 0.0)
    } else {
        asymptotic()
    };
    if w.re < -600.0 {
        let logarithm = z.ln() + Complex64::new(0.0, 2.0 * PI * k as f64);
        let real_lower = k == -1 && z.re < 0.0 && z.im == 0.0;
        if real_lower {
            w.im = -0.0;
        }
        for _ in 0..100 {
            let residual = w + w.ln() - logarithm;
            let mut next = w - residual / (1.0 + reciprocal(w));
            if real_lower {
                next.im = -0.0;
            }
            if (next - w).norm() <= 4.0 * f64::EPSILON * next.norm() {
                return finish(next);
            }
            w = next;
        }
        return Err("lambertw: Logarithmic iteration did not converge.");
    }
    for _ in 0..100 {
        let next = if w.re >= 0.0 {
            let residual = w - z * (-w).exp();
            w - residual / (w + 1.0 - (w + 2.0) * residual / (2.0 * w + 2.0))
        } else {
            let ew = w.exp();
            let wew = w * ew;
            let residual = wew - z;
            w - residual / (wew + ew - (w + 2.0) * residual / (2.0 * w + 2.0))
        };
        if (next - w).norm() <= 4.0 * f64::EPSILON * next.norm().max(1e-300) {
            return finish(next);
        }
        w = next;
    }
    Err("lambertw: Iteration did not converge.")
}

fn log_near_one(z: Complex64) -> Complex64 {
    if (z.re - 1.0).abs() < 0.5 && z.im.abs() < 0.5 {
        Complex64::new(
            0.5 * ((z.re - 1.0) * (z.re + 1.0) + z.im * z.im).ln_1p(),
            z.im.atan2(z.re),
        )
    } else {
        z.ln()
    }
}

pub use crate::backend::bessel::{
    airy, airye, cyl_bessel_i, cyl_bessel_ie, cyl_bessel_j, cyl_bessel_je, cyl_bessel_k,
    cyl_bessel_ke, cyl_bessel_y, cyl_bessel_ye, cyl_hankel_1, cyl_hankel_1e, cyl_hankel_2,
    cyl_hankel_2e,
};

macro_rules! complex_error {
    ($($name:ident, $method:ident, $doc:literal;)+) => {$(
        #[doc = $doc]
        pub fn $name<T: Float>(z: Complex<T>) -> Result<Complex<T>,StrErr> {
            use crate::backend::faddeeva::ComplexErrorFunctions;
            finish(promote(z)?.$method())
        }
    )+};
}
complex_error! {
    erf, erf, "Complex error function erf(z).";
    erfc, erfc, "Complex complementary error function erfc(z), evaluated without subtractive cancellation.";
    erfcx, erfcx, "Complex scaled complementary error function exp(z²)*erfc(z).";
    erfi, erfi, "Complex imaginary error function -i*erf(i*z).";
    dawsn, dawson, "Complex Dawson integral exp(-z²)*∫₀ᶻ exp(t²) dt.";
    wofz, w, "Faddeeva function w(z) = exp(-z²)*erfc(-i*z).";
}

/// Standard normal CDF for complex arguments.
pub fn ndtr<T: Float>(z: Complex<T>) -> Result<Complex<T>, StrErr> {
    use crate::backend::faddeeva::ComplexErrorFunctions;
    finish(0.5 * (-promote(z)? * core::f64::consts::FRAC_1_SQRT_2).erfc())
}

/// Natural logarithm of 1+z, accurate near z = 0.
pub fn log1p<T: Float>(z: Complex<T>) -> Result<Complex<T>, StrErr> {
    let z = promote(z)?;
    finish(log1p_kernel(z))
}

/// Logarithm of the complex standard normal CDF, on its principal branch.
pub fn log_ndtr<T: Float>(z: Complex<T>) -> Result<Complex<T>, StrErr> {
    use crate::backend::faddeeva::ComplexErrorFunctions;
    let z = finite(promote(z)?)?;
    if z.re >= 0.0 {
        let delta = -0.5 * (z * core::f64::consts::FRAC_1_SQRT_2).erfc();
        if delta.norm() < 0.5 {
            return finish(log1p_kernel(delta));
        }
    }
    let z = -z * core::f64::consts::FRAC_1_SQRT_2;
    let square = Complex64::new((z.im - z.re) * (z.re + z.im), -2.0 * z.re * z.im);
    let value = square + (0.5 * z.erfcx()).ln();
    let mut phase = value.im % (2.0 * PI);
    if phase > PI {
        phase -= 2.0 * PI;
    } else if phase < -PI {
        phase += 2.0 * PI;
    }
    finish(Complex64::new(value.re, phase))
}

const DIGAMMA_BERNOULLI: [f64; 16] = [
    0.166666666666666667,
    -0.0333333333333333333,
    0.0238095238095238095,
    -0.0333333333333333333,
    0.0757575757575757576,
    -0.253113553113553114,
    1.16666666666666667,
    -7.09215686274509804,
    54.9711779448621554,
    -529.124242424242424,
    6192.12318840579710,
    -86580.2531135531136,
    1425517.16666666667,
    -27298231.0678160920,
    601580873.900642368,
    -15116315767.0921569,
];
pub(crate) fn digamma_series(z: Complex64, root: f64, root_value: f64) -> Complex64 {
    let z = z - root;
    let mut result = Complex64::new(root_value, 0.0);
    let mut coefficient = Complex64::new(-1.0, 0.0);
    for n in 1..100 {
        coefficient *= -z;
        let term = coefficient * crate::backend::cephes64::zeta((n + 1) as f64, root);
        result += term;
        if term.norm() <= f64::EPSILON * result.norm() {
            break;
        }
    }
    result
}
fn digamma_kernel(mut z: Complex64) -> Complex64 {
    if (z + 0.504083008264455409).norm() < 0.3 {
        return digamma_series(z, -0.504083008264455409, 7.2897639029768949e-17);
    }
    let mut result = Complex64::new(0.0, 0.0);
    if z.re < 0.0 && z.im.abs() < 16.0 {
        result = -PI * trig_pi(z, true) / trig_pi(z, false);
        z = 1.0 - z;
    }
    if z.norm() < 0.5 {
        result -= reciprocal(z);
        z += 1.0;
    }
    if (z - 1.4616321449683623).norm() < 0.5 {
        return result + digamma_series(z, 1.4616321449683623, -9.2412655217294275e-17);
    }
    while z.norm() < 16.0 {
        result -= reciprocal(z);
        z += 1.0;
    }
    let reciprocal = reciprocal(z);
    let square = reciprocal * reciprocal;
    result += z.ln() - 0.5 * reciprocal;
    let mut factor = Complex64::new(1.0, 0.0);
    for (k, b) in DIGAMMA_BERNOULLI.iter().enumerate() {
        factor *= square;
        let term = -b * factor / (2 * (k + 1)) as f64;
        result += term;
        if term.norm() < f64::EPSILON * result.norm() {
            break;
        }
    }
    result
}
/// Complex logarithmic derivative of Gamma. Nonpositive real integers are poles.
pub fn digamma<T: Float>(z: Complex<T>) -> Result<Complex<T>, StrErr> {
    let z = finite(promote(z)?)?;
    if pole(z) {
        return Err("digamma: Nonpositive integers are poles.");
    }
    finish(digamma_kernel(z))
}

/// Complex Riemann zeta, using xsf's Euler-Maclaurin expansion and reflection.
/// Returns an error at the pole z = 1 or when convergence cannot be established.
pub fn riemann_zeta<T: Float>(z: Complex<T>) -> Result<Complex<T>, StrErr> {
    let z = finite(promote(z)?)?;
    finish(zeta_kernel(z)?)
}

fn zeta_kernel(z: Complex64) -> Result<Complex64, StrErr> {
    if z.im == 0.0 {
        return Ok(Complex64::new(
            crate::backend::cephes64::riemann_zeta(z.re),
            0.0,
        ));
    }
    if z.re < 0.0 {
        let log_factor = z * core::f64::consts::LN_2
            + (z - 1.0) * PI.ln()
            + trig_pi(z / 2.0, false).ln()
            + loggamma_kernel(1.0 - z);
        return Ok(log_factor.exp() * zeta_kernel(1.0 - z)?);
    }
    if z.im.abs() > 400000.0 {
        return Err("riemann_zeta: Argument too large for the Euler-Maclaurin expansion.");
    }
    let n = (z.im.abs() / 4.0).max(50.0) as usize;
    let mut sum = Complex64::new(0.0, 0.0);
    for i in 1..n {
        let term = (-z * (i as f64).ln()).exp();
        sum += term;
        if z.re > 1.0 && negligible(term, sum) {
            return Ok(sum);
        }
    }
    let n = n as f64;
    let b = (-z * n.ln()).exp();
    sum += b * (0.5 + n * reciprocal(z - 1.0));
    let mut factor = b * z / n;
    for (k, &c) in ZETA_COEFFICIENTS.iter().enumerate() {
        let term = c * factor;
        sum += term;
        if negligible(term, sum) {
            return Ok(sum);
        }
        let k = (k + 1) as f64;
        factor *= ((z + (2.0 * k - 1.0)) / n) * ((z + 2.0 * k) / n);
    }
    Err("riemann_zeta: Euler-Maclaurin expansion did not converge.")
}

const ZETA_COEFFICIENTS: [f64; 50] = [
    0.08333333333333333,
    -0.0013888888888888885,
    3.306878306878307e-05,
    -8.267195767195772e-07,
    2.0876756987868077e-08,
    -5.284190138687492e-10,
    1.3382536530684685e-11,
    -3.3896802963225777e-13,
    8.586062056277834e-15,
    -2.1748686985580553e-16,
    5.509002828360214e-18,
    -1.3954464685812542e-19,
    3.534707039629459e-21,
    -8.953517427037558e-23,
    2.2679524523376822e-24,
    -5.744790668872196e-26,
    1.455172475614867e-27,
    -3.685994940665297e-29,
    9.336734257095098e-31,
    -2.3650224157006313e-32,
    5.990671762482108e-34,
    -1.5174548844682973e-35,
    3.843758125454187e-37,
    -9.736353072646742e-39,
    2.466247044200688e-40,
    -6.247076741820735e-42,
    1.5824030244644812e-43,
    -4.0082736859489476e-45,
    1.0153075855569534e-46,
    -2.5718041582418896e-48,
    6.514456035233828e-50,
    -1.6501309906896474e-51,
    4.179830628539502e-53,
    -1.058763466770292e-54,
    2.68187919126076e-56,
    -6.793279351107359e-58,
    1.7207577616681163e-59,
    -4.358730329348934e-61,
    1.1040792903684714e-62,
    -2.7966655133781322e-64,
    7.084036501679429e-66,
    -1.7944074082892046e-67,
    4.545287063611153e-69,
    -1.1513346631982138e-70,
    2.9163647710923684e-72,
    -7.387238263497318e-74,
    1.871209311763781e-75,
    -4.7398285577617395e-77,
    1.2006125993354636e-78,
    -3.0411872415143097e-80,
];

macro_rules! spherical_complex {
    ($($name:ident,$jac:ident,$kind:literal,$doc:literal;)+)=>{$(
        #[doc=$doc]
        pub fn $name<T:Float>(n:i32,z:Complex<T>)->Result<Complex<T>,StrErr> {
            finish(crate::backend::spherical::complex_value(n,finite(promote(z)?)?,$kind)?)
        }
        #[doc=concat!("Complex derivative of [`",stringify!($name),"`].")]
        pub fn $jac<T:Float>(n:i32,z:Complex<T>)->Result<Complex<T>,StrErr> {
            let z=finite(promote(z)?)?;
            if n<0 { return Err("sph_bessel: Requires n >= 0."); }
            if z==Complex64::new(0.0,0.0) {
                return if $kind==0 || $kind==2 { finish(Complex64::new(if n==1 {1.0/3.0}else{0.0},0.0)) }
                    else { Err("sph_bessel: Undefined derivative at zero.") };
            }
            let result=if $kind==3 && n==1 {
                let r=reciprocal(z);
                -core::f64::consts::FRAC_PI_2*(-z).exp()*r*(1.0+2.0*r+2.0*r*r)
            } else if ($kind==0 || $kind==2) && z.norm()<0.5 {
                crate::backend::spherical::complex_derivative(n,z,$kind==2)?
            } else if n==0 {
                let next=crate::backend::spherical::complex_value(1,z,$kind)?;
                if $kind==2 {next}else{-next}
            } else {
                let previous=crate::backend::spherical::complex_value(n-1,z,$kind)?;
                let current=crate::backend::spherical::complex_value(n,z,$kind)?;
                if $kind==3 {-previous-(n as f64+1.0)/z*current}
                else {previous-(n as f64+1.0)/z*current}
            };
            if !result.re.is_finite() || !result.im.is_finite() { return Err("sph_bessel: Derivative overflow."); }
            finish(result)
        }
    )+};
}
spherical_complex! {
    sph_bessel_j,sph_bessel_j_jac,0,"Complex spherical Bessel j_n(z), for integer n >= 0.";
    sph_bessel_y,sph_bessel_y_jac,1,"Complex spherical Bessel y_n(z), for integer n >= 0.";
    sph_bessel_i,sph_bessel_i_jac,2,"Complex modified spherical Bessel i_n(z), for integer n >= 0.";
    sph_bessel_k,sph_bessel_k_jac,3,"Complex modified spherical Bessel k_n(z), for integer n >= 0.";
}

pub(crate) fn log1p_kernel(z: Complex64) -> Complex64 {
    if z.norm() < 0.5 {
        Complex64::new(
            0.5 * ((2.0 + z.re) * z.re + z.im * z.im).ln_1p(),
            z.im.atan2(1.0 + z.re),
        )
    } else {
        (z + 1.0).ln()
    }
}

pub(crate) fn reciprocal(z: Complex64) -> Complex64 {
    if z.re.abs() >= z.im.abs() {
        let r = z.im / z.re;
        let d = 1.0 + r * r;
        Complex64::new((1.0 / z.re) / d, (-r / z.re) / d)
    } else {
        let r = z.re / z.im;
        let d = 1.0 + r * r;
        Complex64::new((r / z.im) / d, (-1.0 / z.im) / d)
    }
}

fn exp_kernel(z: Complex64) -> Complex64 {
    if z.im == 0.0 {
        return Complex64::new(z.re.exp(), 0.0_f64.copysign(z.im));
    }
    let e = (z.re / 2.0).exp();
    let scaled = |v: f64| {
        if v == 0.0 {
            v
        } else if e.is_infinite() {
            f64::INFINITY.copysign(v)
        } else {
            (v * e) * e
        }
    };
    Complex64::new(scaled(z.im.cos()), scaled(z.im.sin()))
}

fn negligible(term: Complex64, sum: Complex64) -> bool {
    term.re.abs() <= f64::EPSILON * sum.re.abs() && term.im.abs() <= f64::EPSILON * sum.im.abs()
}

/// Wright omega, solving w + log(w) = z with the unwinding convention of xsf.
/// The branch cuts run from -1 +/- i*pi to the left. Exact cut arguments use
/// the unwinding number ceil((Im(z)-pi)/(2*pi)), with the floating-point pi constant.
pub fn wrightomega<T: Float>(z: Complex<T>) -> Result<Complex<T>, StrErr> {
    finish(crate::backend::wright::complex_omega(promote(z)?))
}

pub use crate::backend::hypergeometric::complex_hyp0f1 as hyp0f1;
