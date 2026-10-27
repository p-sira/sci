//! Wright omega, adapted from Piers Lawrence's BSD-licensed Algorithm 917.
//! See LICENSES/wright.txt and xsf/wright.h.
use crate::backend::utils::real_kernel;
use core::f64::consts::PI;
use num_complex::Complex64;
#[allow(unused_imports)]
use sci_core::Float;

real_kernel! {
    /// Real Wright omega: the positive solution of w + log(w) = x.
    ///
    /// Evaluates W_0(exp(x)) without overflowing exp(x) for large positive x.
    wrightomega(x) { real_omega(x) }
}

fn real_omega(x: f64) -> f64 {
    if x == f64::NEG_INFINITY {
        return 0.0;
    }
    if x < -50.0 {
        return x.exp();
    }
    if x > 1e20 {
        return x;
    }
    let mut w = if x < -2.0 {
        x.exp()
    } else if x < 1.0 {
        (2.0 * (x - 1.0) / 3.0).exp()
    } else {
        let logarithm = x.ln();
        x - logarithm + logarithm / x
    };
    // Two fourth-order Fritsch-Shafer-Crowley corrections, as in xsf.
    for _ in 0..2 {
        let r = (x - w) - w.ln();
        let wp1 = w + 1.0;
        let t = 2.0 * wp1 * (wp1 + 2.0 * r / 3.0);
        let correction = r / wp1 * (t - r) / (t - 2.0 * r);
        w *= 1.0 + correction;
    }
    w
}

fn asymptotic(z: Complex64, logarithm: Complex64) -> Complex64 {
    let mut term = logarithm / z;
    let mut w = z - logarithm + term;
    term /= z;
    w += term * (0.5 * logarithm - 1.0);
    term /= z;
    w + term * (logarithm * logarithm / 3.0 - 1.5 * logarithm + 1.0)
}

pub(crate) fn complex_omega(mut z: Complex64) -> Complex64 {
    let (x, y) = (z.re, z.im);
    if x == f64::NEG_INFINITY && y > -PI && y <= PI {
        return Complex64::new(
            if y.abs() <= PI / 2.0 { 0.0 } else { -0.0 },
            0.0_f64.copysign(y),
        );
    }
    if x.is_infinite() || y.is_infinite() {
        return z;
    }
    if y == 0.0 {
        return Complex64::new(real_omega(x), y);
    }
    if x == -1.0 && y.abs() == PI {
        return Complex64::new(-1.0, 0.0);
    }
    let i = Complex64::new(0.0, 1.0);
    let mut w;
    if -2.0 < x && x <= 1.0 && 1.0 < y && y < 2.0 * PI {
        let p = if y == PI && x < -1.0 {
            Complex64::new(0.0, -(-2.0 * (x + 1.0)).sqrt())
        } else {
            (2.0 * (z + 1.0 - i * PI)).conj().sqrt().conj()
        };
        w = -1.0 + (i + (1.0 / 3.0 + (-i / 36.0 + (1.0 / 270.0 + i * p / 4320.0) * p) * p) * p) * p;
    } else if -2.0 < x && x <= 1.0 && -2.0 * PI < y && y < -1.0 {
        let p = if y == -PI && x < -1.0 {
            Complex64::new(0.0, -(-2.0 * (x + 1.0)).sqrt())
        } else {
            (2.0 * (z + 1.0 + i * PI)).conj().sqrt().conj()
        };
        w = -1.0 + (-i + (1.0 / 3.0 + (i / 36.0 + (1.0 / 270.0 - i * p / 4320.0) * p) * p) * p) * p;
    } else if x <= -2.0 && -PI < y && y <= PI {
        let p = z.exp();
        w = (1.0 + (-1.0 + (1.5 + (-8.0 / 3.0 + 125.0 * p / 24.0) * p) * p) * p) * p;
        if w == Complex64::new(0.0, 0.0) {
            return w;
        }
    } else if (-2.0 < x && x <= 1.0 && y.abs() <= 1.0)
        || (-2.0 < x && (x - 1.0) * (x - 1.0) + y * y <= PI * PI)
    {
        let p = z - 1.0;
        w = 0.5
            + 0.5 * z
            + (1.0 / 16.0 + (-1.0 / 192.0 + (-1.0 / 3072.0 + 13.0 * p / 61440.0) * p) * p) * p * p;
    } else {
        let t = if x <= -1.05 && PI < y && y - PI <= -0.75 * (x + 1.0) {
            z - i * PI
        } else if x <= -1.05 && 0.75 * (x + 1.0) < y + PI && y + PI <= 0.0 {
            z + i * PI
        } else {
            z
        };
        let logarithm = if t != z { (-t).ln() } else { t.ln() };
        w = asymptotic(t, logarithm);
        if z.norm() > 1e50 {
            return w;
        }
    }
    // Near an exact branch point the Puiseux polynomial is already accurate;
    // iterating would amplify rounding in the residual by 1/|w+1|.
    if y.abs() == PI && x < -1.0 && -1.0 - x < 1e-4 {
        return w;
    }
    let mut sign = 1.0;
    if x <= -0.99 && ((y - PI).abs() <= 0.01 || (y + PI).abs() <= 0.01) {
        sign = -1.0;
        // By Sterbenz's lemma these subtractions are exact in this region;
        // xsf's directed-rounding two_sum corrections are consequently zero.
        z.im = if (y - PI).abs() <= 0.01 {
            y - PI
        } else {
            y + PI
        };
    }
    w *= sign;
    for _ in 0..2 {
        let r = z - sign * w - w.ln();
        let wp1 = sign * w + 1.0;
        let t = 2.0 * wp1 * (wp1 + 2.0 * r / 3.0);
        let correction = r / wp1 * (t - r) / (t - 2.0 * r);
        w *= 1.0 + correction;
    }
    sign * w
}
