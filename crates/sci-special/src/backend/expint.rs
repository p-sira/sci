//! Exponential integrals, ported from xsf/expint.h (Zhang and Jin, 1996).
use crate::backend::utils::real_kernel;
#[allow(unused_imports)]
use sci_core::Float;
const EULER: f64 = 0.57721566490153286060651209008240243;

pub(crate) fn exp1_real(x: f64) -> f64 {
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x < 0.0 {
        return f64::NAN;
    }
    if x <= 1.0 {
        let (mut sum, mut term) = (1.0, 1.0);
        for k in 1..26 {
            let k = k as f64;
            term = -term * k * x / ((k + 1.0) * (k + 1.0));
            sum += term;
            if term.abs() <= sum.abs() * 1e-15 {
                break;
            }
        }
        return -EULER - x.ln() + x * sum;
    }
    let m = 20 + (80.0 / x) as i32;
    let mut t = 0.0;
    for k in (1..=m).rev() {
        let k = k as f64;
        t = k / (1.0 + k / (x + t));
    }
    (-x).exp() / (x + t)
}

real_kernel! {
    /// Exponential integral E₁(x) = ∫ₓ^∞ exp(-t)/t dt, for x >= 0.
    exp1(x) { exp1_real(x) }
}
real_kernel! {
    /// Real exponential integral Ei(x), with a pole at zero.
    expi(x) {
        if x == 0.0 { f64::NEG_INFINITY }
        else if x < 0.0 { -exp1_real(-x) }
        else if x <= 40.0 {
            let (mut sum, mut term) = (1.0, 1.0);
            for k in 1..=100 {
                let k = k as f64;
                term = term * k * x / ((k + 1.0) * (k + 1.0));
                sum += term;
                if (term / sum).abs() <= 1e-15 { break; }
            }
            EULER + x.ln() + x * sum
        } else if x.is_infinite() { f64::INFINITY }
        else {
            let (mut sum, mut term) = (1.0, 1.0);
            for k in 1..=20 { term *= k as f64 / x; sum += term; }
            if x > 709.782712893384 { (x - x.ln()).exp() * sum } else { x.exp() / x * sum }
        }
    }
}
