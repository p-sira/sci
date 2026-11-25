//! Real Struve functions, using shared Bessel and extended-precision kernels.
//!
//! Series selection follows the mathematical expansions in DLMF 11.2, 11.4,
//! and 11.6 and SciPy's error-estimation strategy (see LICENSES/struve.txt).
use crate::backend::{StrErr, double_double::Dd, utils::real_kernel};
use core::f64::consts::PI;
#[allow(unused_imports)]
use sci_core::Float;

#[derive(Clone, Copy)]
struct Estimate {
    value: f64,
    error: f64,
}
impl Estimate {
    fn acceptable(self, tolerance: f64) -> bool {
        !self.value.is_nan()
            && self.error.is_finite()
            && (self.error <= tolerance * self.value.abs() || self.error < 1e-300)
    }
}

fn restore(sum: f64, logarithm: f64, sign: f64) -> f64 {
    if sum == 0.0 {
        0.0
    } else {
        (sum.abs().ln() + logarithm).exp().copysign(sum) * sign
    }
}

fn power(v: f64, x: f64, modified: bool) -> Option<Estimate> {
    let mut logarithm = (v + 1.0) * (x.ln() - core::f64::consts::LN_2)
        - crate::backend::cephes64::lgam(v + 1.5)
        + (2.0 / PI.sqrt()).ln();
    let sign = crate::backend::gammasgn(v + 1.5).ok()?;
    let mut term = Dd::from(1.0);
    let mut sum = term;
    let mut peak: f64 = 1.0;
    for k in 1..=10000 {
        let k = k as f64;
        let factor = (Dd::from(x * 0.5) / Dd::from(k + 0.5))
            * (Dd::from(x * 0.5) / (Dd::from(v) + Dd::from(k + 0.5)));
        term = term * if modified { factor } else { -factor };
        sum = sum + term;
        peak = peak.max(term.hi.abs());
        if !sum.hi.is_finite() {
            return None;
        }
        if term.hi.abs() <= 1e-32 * sum.hi.abs() || term.hi == 0.0 {
            let residual = term.hi.abs() + peak * 1e-30;
            let relative = residual / sum.hi.abs();
            if !relative.is_finite() {
                return None;
            }
            let value = restore(sum.hi, logarithm, sign);
            let error = if value.is_infinite() && relative < 1e-12 {
                0.0
            } else {
                restore(residual, logarithm, 1.0)
            };
            return Some(Estimate { value, error });
        }
        if peak > 1e150 {
            term = term * Dd::from(1e-150);
            sum = sum * Dd::from(1e-150);
            peak *= 1e-150;
            logarithm += 150.0 * core::f64::consts::LN_10;
        }
    }
    None
}

fn asymptotic(v: f64, x: f64, modified: bool) -> Option<Estimate> {
    if x < v.max(12.0) {
        return None;
    }
    let logarithm = (v - 1.0) * (x.ln() - core::f64::consts::LN_2)
        - crate::backend::cephes64::lgam(v + 0.5)
        - 0.5 * PI.ln();
    let sign = crate::backend::gammasgn(v + 0.5).ok()? * if modified { -1.0 } else { 1.0 };
    let mut sum = 1.0;
    let mut term: f64 = 1.0;
    let mut omitted = 1.0;
    for k in 1..=10000 {
        let odd = (2 * k - 1) as f64;
        let next = term * (odd / x) * ((odd - 2.0 * v) / x) * if modified { 1.0 } else { -1.0 };
        omitted = next.abs();
        if omitted > term.abs() {
            break;
        }
        sum += next;
        term = next;
        if omitted <= f64::EPSILON * sum.abs() {
            break;
        }
    }
    let correction = restore(sum, logarithm, sign);
    let error = restore(omitted, logarithm, 1.0).abs() + correction.abs() * 4.0 * f64::EPSILON;
    let base = if modified {
        crate::backend::bessel::iv(v, x).ok()?
    } else {
        crate::backend::cyl_bessel_y(v, x).ok()?
    };
    Some(Estimate {
        value: base + correction,
        error,
    })
}

fn bessel_series(v: f64, x: f64, modified: bool) -> Option<Estimate> {
    if (!modified && v < 0.0) || x > v.abs() + 20.0 {
        return None;
    }
    let mut coefficient = (x / (2.0 * PI)).sqrt();
    let mut sum = Dd::from(0.0);
    let mut peak: f64 = 0.0;
    for k in 0..10000 {
        let k = k as f64;
        let order = k + v + 0.5;
        let bessel = if modified {
            crate::backend::bessel::iv(order, x).ok()?
        } else {
            crate::backend::cyl_bessel_j(order, x).ok()?
        };
        let term = coefficient * bessel / (k + 0.5);
        sum = sum + Dd::from(term);
        peak = peak.max(term.abs());
        if !sum.hi.is_finite() {
            return None;
        }
        if term.abs() <= f64::EPSILON * sum.hi.abs() || term == 0.0 {
            return Some(Estimate {
                value: sum.hi,
                error: term.abs() + peak * 4.0 * f64::EPSILON,
            });
        }
        coefficient *= (x * 0.5 / (k + 1.0)) * if modified { -1.0 } else { 1.0 };
    }
    None
}

fn struve(v: f64, x: f64, modified: bool) -> Result<f64, StrErr> {
    if !v.is_finite() || !x.is_finite() || v.abs() > 1_000_000.0 {
        return Err("Struve functions require finite arguments and |order| <= 1,000,000.");
    }
    if x < 0.0 {
        if v != v.trunc() {
            return Err("Struve functions at negative x require integer order.");
        }
        return Ok(struve(v, -x, modified)? * if v % 2.0 == 0.0 { -1.0 } else { 1.0 });
    }
    let half = -v - 0.5;
    if half >= 0.0 && half == half.trunc() {
        return if modified {
            crate::backend::bessel::iv(-v, x)
        } else {
            Ok(crate::backend::cyl_bessel_j(-v, x)? * if half % 2.0 == 0.0 { 1.0 } else { -1.0 })
        };
    }
    if x == 0.0 {
        return Ok(if v < -1.0 {
            crate::backend::gammasgn(v + 1.5)? * f64::INFINITY
        } else if v == -1.0 {
            2.0 / PI
        } else {
            0.0
        });
    }
    let mut best: Option<Estimate> = None;
    // Lazy candidates avoid evaluating expensive Bessel series when an earlier
    // expansion already meets the requested accuracy.
    type Expansion = fn(f64, f64, bool) -> Option<Estimate>;
    // L_v's positive defining series has no cancellation for v > -1/2.
    // Prefer it in the moderate range: a terminating algebraic asymptotic
    // correction can still omit an exponentially small contribution.
    let candidates: [Expansion; 3] = if modified && v > -0.5 && x <= 100.0 {
        [power, asymptotic, bessel_series]
    } else {
        [asymptotic, power, bessel_series]
    };
    for candidate in candidates {
        if let Some(estimate) = candidate(v, x, modified) {
            if estimate.acceptable(1e-12) {
                return Ok(estimate.value);
            }
            if !estimate.value.is_nan() && best.is_none_or(|b| estimate.error < b.error) {
                best = Some(estimate);
            }
        }
    }
    if let Some(estimate) = best.filter(|e| e.acceptable(1e-8)) {
        return Ok(estimate.value);
    }
    Err("Struve function could not be evaluated with sufficient precision.")
}

real_kernel! {
/// Real Struve H_v(x). Negative x requires integer order; arguments must be finite and |v| <= 1,000,000.
/// Returns an error when the series estimates indicate excessive precision loss.
    struve_h(v,x) { struve(v,x,false)? }
}
real_kernel! {
/// Real modified Struve L_v(x). Negative x requires integer order; arguments must be finite and |v| <= 1,000,000.
/// Returns an error when the series estimates indicate excessive precision loss.
    struve_l(v,x) { struve(v,x,true)? }
}
