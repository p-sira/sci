//! Limiting two-sided Kolmogorov distribution, its density, and inverses.
use crate::backend::{
    StrErr,
    utils::{bracketed_root, real_kernel},
};
use core::f64::consts::{LN_2, PI};
#[allow(unused_imports)]
use sci_core::Float;

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
    if x <= 0.82 {
        let a = (PI / x) * (PI / x) / 8.0;
        if !a.is_finite() {
            return (f64::NEG_INFINITY, 0.0, f64::NEG_INFINITY);
        }
        let mut sum = 1.0;
        let mut weighted = 2.0 * a - 1.0;
        for k in 1..=32 {
            let odd = (2 * k + 1) as f64;
            let term = (-a * (odd * odd - 1.0)).exp();
            sum += term;
            weighted += (2.0 * a * odd * odd - 1.0) * term;
            if term <= f64::EPSILON * sum {
                break;
            }
        }
        let logcdf = 0.5 * (2.0 * PI).ln() - x.ln() - a + sum.ln();
        let logpdf = 0.5 * (2.0 * PI).ln() - 2.0 * x.ln() - a + weighted.ln();
        (logcdf, (-logcdf.exp()).ln_1p(), logpdf)
    } else {
        let a = 2.0 * x * x;
        if !a.is_finite() {
            return (0.0, f64::NEG_INFINITY, f64::NEG_INFINITY);
        }
        let mut sum = 1.0;
        let mut weighted = 1.0;
        for k in 2..=32 {
            let square = (k * k) as f64;
            let term = (-a * (square - 1.0)).exp() * if k % 2 == 0 { -1.0 } else { 1.0 };
            sum += term;
            weighted += square * term;
            if term.abs() <= f64::EPSILON * sum {
                break;
            }
        }
        let logsf = LN_2 - a + sum.ln();
        let logpdf = (8.0 * x).ln() - a + weighted.ln();
        ((-logsf.exp()).ln_1p(), logsf, logpdf)
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
    let initial = if lower {
        PI / (-8.0 * target).sqrt()
    } else {
        ((LN_2 - target) * 0.5).sqrt()
    };
    let left = if lower { 0.5 * initial } else { 0.0 };
    let mut right = initial.max(2.0);
    if !lower {
        for _ in 0..16 {
            if logarithms(right).1 <= target {
                break;
            }
            right *= 2.0;
        }
    }
    bracketed_root(left, right, |x| {
        let (cdf, sf, pdf) = logarithms(x);
        let tail = if lower { cdf } else { sf };
        let slope = (pdf - tail).exp() * if lower { 1.0 } else { -1.0 };
        Ok((tail - target, slope))
    })
}

real_kernel! {
/// Survival function of the limiting two-sided Kolmogorov distribution.
/// Returns one for x <= 0 and zero at positive infinity.
    kolmogorov(x) { let (cdf,sf,_) = logarithms(x); if x<=0.82 {-cdf.exp_m1()} else {sf.exp()} }
}
real_kernel! {
/// CDF of the limiting two-sided Kolmogorov distribution.
/// Returns zero for x <= 0 and one at positive infinity.
    kolmogc(x) { let (cdf,sf,_) = logarithms(x); if x<=0.82 {cdf.exp()} else {-sf.exp_m1()} }
}
real_kernel! {
/// Derivative of the Kolmogorov survival function (the negative density).
    kolmogp(x) { -logarithms(x).2.exp() }
}
real_kernel! {
/// Inverse Kolmogorov survival probability, for 0 <= p <= 1.
    kolmogi(p) { inverse(p,false)? }
}
real_kernel! {
/// Inverse Kolmogorov CDF, for 0 <= p <= 1.
    kolmogci(p) { inverse(p,true)? }
}
