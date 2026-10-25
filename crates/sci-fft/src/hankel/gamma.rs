//! Private Rust translation of SciPy/XSF's Hare log-gamma algorithm and
//! Cephes Pochhammer recurrence. See LICENSES/scipy-gamma.txt.
//!
//! Complex log-gamma uses Taylor/Stirling expansions, at most eight recurrence
//! steps, and one reflection. The real ratio reuses the workspace's `special`
//! backend; its recurrence is capped at 64 steps before a log-gamma fallback.
use crate::{Error, Result};
use num_complex::Complex64 as C;
use std::f64::consts::{LN_2, PI, TAU};

fn checked(z: C) -> Result<C> {
    if z.re.is_finite() && z.im.is_finite() {
        Ok(z)
    } else {
        Err(Error::NonFiniteEvaluation)
    }
}

fn polynomial(z: C, coefficients: &[f64]) -> C {
    coefficients
        .iter()
        .fold(C::new(0.0, 0.0), |v, &c| v * z + c)
}

fn stirling(z: C) -> C {
    let r = 1.0 / z;
    let coefficients = [
        -2.955_065_359_477_124e-2,
        6.410_256_410_256_41e-3,
        -1.917_526_917_526_917_6e-3,
        8.417_508_417_508_417e-4,
        -5.952_380_952_380_952e-4,
        7.936_507_936_507_937e-4,
        -2.777_777_777_777_778e-3,
        8.333_333_333_333_333e-2,
    ];
    (z - 0.5) * z.ln() - z + 0.918_938_533_204_672_7 + r * polynomial(r / z, &coefficients)
}

fn taylor(z: C) -> C {
    let coefficients = [
        -4.347_826_605_304_026e-2,
        4.545_455_629_320_467e-2,
        -4.761_907_033_014_223e-2,
        5.000_004_769_810_169e-2,
        -5.263_167_937_961_666e-2,
        5.555_576_762_740_361e-2,
        -5.882_397_865_868_459e-2,
        6.250_095_514_121_304e-2,
        -6.666_870_588_242_047e-2,
        7.143_294_629_536_134e-2,
        -7.693_251_641_135_22e-2,
        8.335_384_054_610_9e-2,
        -9.095_401_714_582_904e-2,
        1.000_994_575_127_818e-1,
        -1.113_342_658_695_647e-1,
        1.255_096_695_247_430_4e-1,
        -1.440_498_967_688_462e-1,
        1.695_571_769_974_082e-1,
        -2.073_855_510_286_74e-1,
        2.705_808_084_277_845e-1,
        -4.006_856_343_865_314_3e-1,
        8.224_670_334_241_132e-1,
        -5.772_156_649_015_329e-1,
    ];
    let w = z - 1.0;
    w * polynomial(w, &coefficients)
}

// Range reduction before multiplication by pi preserves integer/half-integer
// zeros and avoids phase loss when reflecting large negative real parts.
fn sinpi(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let r = x.abs() % 2.0;
    if r < 0.5 {
        sign * (PI * r).sin()
    } else if r > 1.5 {
        sign * (PI * (r - 2.0)).sin()
    } else {
        -sign * (PI * (r - 1.0)).sin()
    }
}

fn cospi(x: f64) -> f64 {
    let r = x.abs() % 2.0;
    if r == 0.5 {
        0.0
    } else if r < 1.0 {
        -(PI * (r - 0.5)).sin()
    } else {
        (PI * (r - 1.5)).sin()
    }
}

fn log_sinpi(z: C) -> C {
    let s = sinpi(z.re);
    let c = cospi(z.re);
    let y = PI * z.im;
    if y.abs() < 350.0 {
        C::new(s * y.cosh(), c * y.sinh()).ln()
    } else {
        C::new(y.abs() - LN_2, (c * z.im.signum()).atan2(s))
    }
}

fn right_half_plane(z: C) -> C {
    if z.re > 7.0 || z.im.abs() > 7.0 {
        return stirling(z);
    }
    if (z - 1.0).norm() < 0.2 {
        return taylor(z);
    }
    if (z - 2.0).norm() < 0.2 {
        let w = z - 2.0;
        let log = C::new(
            0.5 * (2.0 * w.re + w.norm_sqr()).ln_1p(),
            w.im.atan2(1.0 + w.re),
        );
        return log + taylor(z - 1.0);
    }
    // Sum principal logs instead of logging a product. This preserves the
    // analytic branch through every recurrence without tracking product wraps.
    let mut shifted = z;
    let mut correction = C::new(0.0, 0.0);
    for _ in 0..8 {
        if shifted.re > 7.0 {
            break;
        }
        correction += shifted.ln();
        shifted += 1.0;
    }
    stirling(shifted) - correction
}

pub(super) fn loggamma(x: f64, y: f64) -> Result<C> {
    let z = checked(C::new(x, y))?;
    if y == 0.0 && x <= 0.0 && x == x.floor() {
        return Err(Error::NumericalFailure("gamma pole in FFTLog coefficients"));
    }
    let result = if x < 0.1 {
        let phase = TAU.copysign(y) * (0.5 * x + 0.25).floor();
        C::new(PI.ln(), phase) - log_sinpi(z) - right_half_plane(1.0 - z)
    } else {
        right_half_plane(z)
    };
    checked(result)
}

fn pole(x: f64) -> bool {
    x <= 0.0 && x == x.floor()
}

pub(super) fn rising(a: f64, mut m: f64) -> Result<f64> {
    let mut r = 1.0;
    for _ in 0..64 {
        if m >= 1.0 && a + m != 1.0 {
            m -= 1.0;
            r *= a + m;
        } else if m <= -1.0 && a + m != 0.0 {
            r /= a + m;
            m += 1.0;
        } else {
            break;
        }
        if !r.is_finite() || r == 0.0 {
            return crate::error::finite(r);
        }
    }
    let value = if m == 0.0 {
        r
    } else if a > 1e4 && m.abs() <= 1.0 {
        // Cephes' gamma-ratio expansion avoids subtracting nearby large logs.
        r * a.powf(m)
            * (1.0
                + m * (m - 1.0) / (2.0 * a)
                + m * (m - 1.0) * (m - 2.0) * (3.0 * m - 1.0) / (24.0 * a * a)
                + m * m * (m - 1.0) * (m - 1.0) * (m - 2.0) * (m - 3.0) / (48.0 * a * a * a))
    } else if pole(a + m) && !pole(a) {
        return Err(Error::NumericalFailure("gamma pole in FFTLog coefficients"));
    } else if pole(a) && !pole(a + m) {
        0.0
    } else {
        let ((num, ns), (den, ds)) = if pole(a) && pole(a + m) {
            let n = ::special::Gamma::ln_gamma(1.0 - a);
            let d = ::special::Gamma::ln_gamma(1.0 - a - m);
            let sign = if m % 2.0 == 0.0 { 1 } else { -1 };
            ((n.0, sign), (d.0, 1))
        } else {
            (
                ::special::Gamma::ln_gamma(a + m),
                ::special::Gamma::ln_gamma(a),
            )
        };
        r * (num - den).exp() * f64::from(ns * ds)
    };
    crate::error::finite(value)
}
