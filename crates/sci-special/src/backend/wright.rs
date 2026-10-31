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
