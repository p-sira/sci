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

pub use crate::backend::complex_integrals::fresnel;

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

/// Analytic log-gamma, real on the positive axis and cut on the negative axis.
pub(crate) fn loggamma(z: Complex64) -> Result<Complex64, StrErr> {
    finite(z)?;
    if pole(z) {
        return Err("loggamma: Nonpositive integers are poles.");
    }
    finish(loggamma_kernel(z))
}

/// Entire reciprocal gamma; evaluate the negative logarithm before exponentiating.
pub(crate) fn rgamma(z: Complex64) -> Result<Complex64, StrErr> {
    finite(z)?;
    if pole(z) {
        return Ok(Complex64::new(0.0, 0.0));
    }
    finish(exp_kernel(-loggamma_kernel(z)))
}

/// Logarithmic gamma derivative, using derivatives of the shared expansions.
pub(crate) fn digamma(mut z: Complex64) -> Result<Complex64, StrErr> {
    finite(z)?;
    if pole(z) {
        return Err("digamma: Nonpositive integers are poles.");
    }
    if z.im == 0.0 {
        // Reuse the real kernel's rational approximation near its real zeros.
        return Ok(Complex64::new(super::digamma(z.re)?, z.im));
    }
    for (root, value, radius) in [
        (-0.5040830082644554, 7.289763902976895e-17, 0.3),
        (1.4616321449683623, -9.241265521729428e-17, 0.5),
    ] {
        if (z - root).norm() < radius {
            return finish(digamma_series(z, root, value));
        }
    }
    let mut correction = Complex64::new(0.0, 0.0);
    if z.re < 0.5 {
        // psi(z) = psi(1-z) - pi*cot(pi*z). Exponentially scaled
        // cotangent avoids overflowing sinh/cosh for large imaginary parts.
        let exponent = -2.0 * PI * z.im.abs();
        let t = exponent.exp();
        let one_minus_t = -exponent.exp_m1();
        let sin = super::cephes64::sinpi(z.re);
        let cos = super::cephes64::cospi(z.re);
        let twice_sqrt_t = 2.0 * (0.5 * exponent).exp();
        let b = twice_sqrt_t * sin;
        let scale = one_minus_t.max(b.abs());
        let a_scaled = one_minus_t / scale;
        let b_scaled = b / scale;
        let denominator = scale * (a_scaled * a_scaled + b_scaled * b_scaled);
        let cot = Complex64::new(
            twice_sqrt_t * cos * b_scaled / denominator,
            -(a_scaled * (1.0 + t)).copysign(z.im) / denominator,
        );
        correction = -PI * cot;
        z = 1.0 - z;
    }
    // At most eight shifts after reflection. Differentiating the shared
    // Stirling correction gives -B_(2k)/(2k*z^(2k)).
    for _ in 0..8 {
        if z.re >= 8.0 {
            break;
        }
        correction -= reciprocal(z);
        z += 1.0;
    }
    let r = reciprocal(z);
    let r2 = r * r;
    let derivative = STIRLING
        .iter()
        .enumerate()
        .fold(Complex64::new(0.0, 0.0), |sum, (i, &coefficient)| {
            sum * r2 + (2 * i as i32 - 15) as f64 * coefficient
        });
    finish(correction + z.ln() - 0.5 * r + r2 * derivative)
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

macro_rules! complex_error {
    ($($name:ident, $method:ident, $doc:literal;)+) => {$(
        #[doc = $doc]
        pub fn $name<T: Float>(z: Complex<T>) -> Result<Complex<T>,StrErr> {
            use crate::backend::faddeeva::ComplexErrorFunctions;
            finish(promote(z)?.$method())
        }
    )+};
}

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

complex_error! {
    erf, erf, "Complex error function erf(z).";
    erfc, erfc, "Complex complementary error function erfc(z), evaluated without subtractive cancellation.";
    erfcx, erfcx, "Complex scaled complementary error function exp(z²)*erfc(z).";
    erfi, erfi, "Complex imaginary error function -i*erf(i*z).";
    dawsn, dawson, "Complex Dawson integral exp(-z²)*integral(exp(t²),t=0..z).";
    wofz, w, "Faddeeva function w(z) = exp(-z²)*erfc(-i*z).";
}

#[cfg(test)]
mod tests {
    use super::gamma;
    use num_complex::Complex64;

    fn close_relative(actual: f64, expected: f64, label: &str) {
        // These small components are normal binary64 numbers; an absolute
        // tolerance would incorrectly accept a result rounded to zero.
        const RELATIVE_TOLERANCE: f64 = 5e-12;
        let error = (actual - expected).abs();
        assert!(
            actual.is_finite() && error <= RELATIVE_TOLERANCE * expected.abs(),
            "{label}: actual={actual:.17e}, expected={expected:.17e}, error={error:.3e}"
        );
    }

    #[test]
    fn gamma_reflection_large_imaginary_argument_matches_high_precision_reference() {
        // Independently evaluated using mpmath 1.4.1 with mp.dps = 100:
        //   gamma(mpc(-1/4, 223))
        // At |Im(z)| = 223, pi*|Im(z)| > 700, which forces trig_pi's
        // split-exponential sine branch in loggamma_kernel's reflection.
        // Retain >60 decimal digits of each reference component; parsing to
        // f64 avoids discarding precision in the checked-in source.
        let reference = Complex64::new(
            "4.115951299497563629161749458169556969749545930356014677603747239974421362126394940741e-155"
                .parse::<f64>()
                .unwrap(),
            "3.208883445780643210060165297550080145591842073427260538804176359867594973054817465413e-154"
                .parse::<f64>()
                .unwrap(),
        );

        let upper = gamma(Complex64::new(-0.25, 223.0)).unwrap();
        close_relative(upper.re, reference.re, "upper gamma real");
        close_relative(upper.im, reference.im, "upper gamma imaginary");

        let lower = gamma(Complex64::new(-0.25, -223.0)).unwrap();
        close_relative(lower.re, reference.re, "lower gamma real");
        close_relative(lower.im, -reference.im, "lower gamma imaginary");
        // In addition to the independent reference, test the conjugate symmetry.
        close_relative(lower.re, upper.re, "gamma conjugate real");
        close_relative(lower.im, -upper.im, "gamma conjugate imaginary");
    }

    #[test]
    fn gamma_rejects_nonfinite_arguments_and_real_nonpositive_integer_poles() {
        for z in [
            Complex64::new(f64::NAN, 0.0),
            Complex64::new(0.5, f64::NAN),
            Complex64::new(f64::INFINITY, 0.25),
            Complex64::new(-0.25, f64::NEG_INFINITY),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(-6.0, -0.0),
        ] {
            assert!(gamma(z).is_err(), "expected Gamma domain error for {z:?}");
        }
    }
}
