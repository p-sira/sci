//! Limiting two-sided Kolmogorov distribution, its density, and inverses.
use crate::backend::{StrErr, utils::real_kernel};
use core::f64::consts::{LN_2, PI};
#[allow(unused_imports)]
use sci_core::Float;

// Log CDF (x <= 0.82) or log survival (x > 0.82), with the log density.
// Successive series terms are exp(-c k(k+1)) or exp(-c (k^2 - 1)), so each
// term follows from the previous one by a geometric ratio instead of an
// exponential per term.
fn series(x: f64) -> (f64, f64) {
    if x <= 0.82 {
        let a = (PI / x) * (PI / x) / 8.0;
        if !a.is_finite() {
            return (f64::NEG_INFINITY, f64::NEG_INFINITY);
        }
        // exp(-a (odd^2 - 1)) = q^(k(k+1)/2) for odd = 2k + 1, q = exp(-8a).
        let q = (-8.0 * a).exp();
        let (mut term, mut ratio) = (q, q);
        let mut sum = 1.0;
        let mut weighted = 2.0 * a - 1.0;
        for k in 1..=32 {
            let odd = (2 * k + 1) as f64;
            sum += term;
            weighted += (2.0 * a * odd * odd - 1.0) * term;
            if term <= f64::EPSILON * sum {
                break;
            }
            ratio *= q;
            term *= ratio;
        }
        let log_x = x.ln();
        let logcdf = 0.5 * (2.0 * PI).ln() - log_x - a + sum.ln();
        let logpdf = 0.5 * (2.0 * PI).ln() - 2.0 * log_x - a + weighted.ln();
        (logcdf, logpdf)
    } else {
        let a = 2.0 * x * x;
        if !a.is_finite() {
            return (f64::NEG_INFINITY, f64::NEG_INFINITY);
        }
        // exp(-a (k^2 - 1)) for k >= 2 starts at q^3 and the ratio between
        // consecutive magnitudes is q^(2k+1), with q = exp(-a).
        let q = (-a).exp();
        let q2 = q * q;
        let (mut magnitude, mut ratio) = (q2 * q, q2 * q2 * q);
        let mut sum = 1.0;
        let mut weighted = 1.0;
        for k in 2..=32 {
            let square = (k * k) as f64;
            let term = if k % 2 == 0 { -magnitude } else { magnitude };
            sum += term;
            weighted += square * term;
            if magnitude <= f64::EPSILON * sum {
                break;
            }
            magnitude *= ratio;
            ratio *= q2;
        }
        let logsf = LN_2 - a + sum.ln();
        let logpdf = (8.0 * x).ln() - a + weighted.ln();
        (logsf, logpdf)
    }
}

// Log CDF, log survival, log density. Keeping both tails in logarithmic form
// preserves inverse probabilities and densities beyond probability underflow.
fn logarithms(x: f64) -> (f64, f64, f64) {
    if x <= 0.0 {
        return (f64::NEG_INFINITY, 0.0, f64::NEG_INFINITY);
    }
    // Below this bound both the CDF and density round to zero in f64.
    if x < 0.01 {
        return (f64::NEG_INFINITY, 0.0, f64::NEG_INFINITY);
    }
    if x == f64::INFINITY {
        return (0.0, f64::NEG_INFINITY, f64::NEG_INFINITY);
    }
    let (log_tail, logpdf) = series(x);
    if x <= 0.82 {
        if log_tail == f64::NEG_INFINITY {
            return (f64::NEG_INFINITY, 0.0, f64::NEG_INFINITY);
        }
        (log_tail, (-log_tail.exp()).ln_1p(), logpdf)
    } else {
        if log_tail == f64::NEG_INFINITY && logpdf == f64::NEG_INFINITY {
            return (0.0, f64::NEG_INFINITY, f64::NEG_INFINITY);
        }
        ((-log_tail.exp()).ln_1p(), log_tail, logpdf)
    }
}

/// Logarithm of the requested tail (CDF if `lower`) and log density, forming
/// the complementary tail only when the series computes the other one.
fn tail(x: f64, lower: bool) -> (f64, f64) {
    if x < 0.01 || x == f64::INFINITY {
        let (cdf, sf, pdf) = logarithms(x);
        return (if lower { cdf } else { sf }, pdf);
    }
    let (log_tail, logpdf) = series(x);
    if lower == (x <= 0.82) {
        (log_tail, logpdf)
    } else {
        ((-log_tail.exp()).ln_1p(), logpdf)
    }
}

fn inverse(mut p: f64, mut lower: bool) -> Result<f64, StrErr> {
    if !(0.0..=1.0).contains(&p) {
        return Err("Kolmogorov inverse requires 0 <= p <= 1.");
    }
    if p == 0.0 {
        return Ok(if lower { 0.0 } else { f64::INFINITY });
    }
    if p == 1.0 {
        return Ok(if lower { f64::INFINITY } else { 0.0 });
    }
    if p > 0.5 {
        p = 1.0 - p;
        lower = !lower;
    }
    let target = p.ln();
    // Leading-term inverses of the two tail series start the iteration.
    let initial = if lower {
        PI / (-8.0 * target).sqrt()
    } else {
        ((LN_2 - target) * 0.5).sqrt()
    };
    // The requested log tail is monotone in x, so the sign of each residual
    // moves one end of the bracket [left, right] (initially [0, inf));
    // Newton steps that leave it are replaced by bisection or doubling.
    let (mut left, mut right) = (0.0, f64::INFINITY);
    let mut x = initial;
    for _ in 0..128 {
        let (log_tail, logpdf) = tail(x, lower);
        let value = log_tail - target;
        if value == 0.0 {
            return Ok(x);
        }
        if value.is_nan() {
            return Err("Zero finder encountered a nonfinite value.");
        }
        if (value < 0.0) == lower {
            left = x;
        } else {
            right = x;
        }
        if right - left <= 4.0 * f64::EPSILON * x.abs().max(1.0) {
            return Ok(x);
        }
        let slope = (logpdf - log_tail).exp() * if lower { 1.0 } else { -1.0 };
        let step = value / slope;
        let next = x - step;
        x = if value.is_finite() && next > left && next < right {
            if step.abs() <= 2.0 * f64::EPSILON * next.abs() {
                return Ok(next);
            }
            next
        } else if right.is_finite() {
            0.5 * (left + right)
        } else {
            2.0 * x
        };
    }
    Err("Zero finder did not converge.")
}

real_kernel! {
/// Survival function of the limiting two-sided Kolmogorov distribution.
/// Returns one for x <= 0 and zero at positive infinity.
    kolmogorov(x) { let (cdf,sf,_) = logarithms(x); if x<=0.82 {-cdf.exp_m1()} else {sf.exp()} }
}
real_kernel! {
/// Inverse Kolmogorov survival probability, for 0 <= p <= 1.
    kolmogi(p) { inverse(p,false)? }
}
