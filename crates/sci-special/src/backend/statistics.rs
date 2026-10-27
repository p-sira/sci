//! Probability tables and distribution utilities with caller-owned storage.
use super::{cephes64, utils::real_kernel};
real_kernel! {
    /// Normal mean giving CDF probability p at x, for positive finite standard deviation.
    nrdtrimn(p, sd, x) {
        if p <= 0.0 || p >= 1.0 || !sd.is_finite() || sd <= 0.0 { f64::NAN }
        else { x - sd*cephes64::ndtri(p) }
    }
}
real_kernel! {
    /// Positive normal standard deviation giving CDF probability p at x.
    /// Inconsistent inputs and the indeterminate p=1/2,x=mean case return an error.
    nrdtrisd(mean, p, x) {
        if p <= 0.0 || p >= 1.0 { f64::NAN } else {
            let sd=(x-mean)/cephes64::ndtri(p);
            if sd>0.0 {sd} else {f64::NAN}
        }
    }
}

fn student_tail(df: f64, x: f64) -> f64 {
    if df <= 0.0 || !df.is_finite() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.5;
    }
    let ratio = x.abs() / df.sqrt();
    // Use both beta coordinates: squaring x itself can overflow unnecessarily.
    let z = 1.0 / (1.0 + ratio * ratio);
    0.5 * cephes64::incbet(0.5 * df, 0.5, z)
}
real_kernel! {
    /// Student's t CDF with finite positive degrees of freedom.
    stdtr(df,x) {
        let tail=student_tail(df,x);
        if x>0.0 {1.0-tail} else {tail}
    }
}
real_kernel! {
    /// Student's t survival function with finite positive degrees of freedom.
    stdtrc(df,x) {
        let tail=student_tail(df,x);
        if x<0.0 {1.0-tail} else {tail}
    }
}
real_kernel! {
    /// Student's t quantile for finite df>0 and probability in [0,1].
    stdtrit(df,p) {
        if !df.is_finite() || df<=0.0 || !(0.0..=1.0).contains(&p) {f64::NAN}
        else if p==0.5 {0.0}
        else {
            let tail=p.min(1.0-p);
            let z=crate::backend::gamma_extra::inverse_beta(df*0.5,0.5,2.0*tail);
            let magnitude=df.sqrt()*((1.0-z)/z).sqrt();
            if p<0.5 {-magnitude} else {magnitude}
        }
    }
}

fn tukey(x: f64, lambda: f64) -> f64 {
    if !lambda.is_finite() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.5;
    }
    if lambda > 0.0 && x.abs() >= 1.0 / lambda {
        return if x < 0.0 { 0.0 } else { 1.0 };
    }
    if lambda == 0.0 {
        return if x < 0.0 {
            x.exp() / (1.0 + x.exp())
        } else {
            1.0 / (1.0 + (-x).exp())
        };
    }
    if !x.is_finite() {
        return if x < 0.0 { 0.0 } else { 1.0 };
    }
    // Solve the smaller tail in log probability, preserving subnormal results.
    let mut left = -746.0;
    let mut right = -core::f64::consts::LN_2;
    for _ in 0..64 {
        let logp = 0.5 * (left + right);
        let p = logp.exp();
        let lp = lambda * logp;
        let lq = lambda * (-p).ln_1p();
        let value = if lambda.abs() < 0.5 {
            (lp.exp_m1() - lq.exp_m1()) / lambda
        } else {
            (lp.exp() - lq.exp()) / lambda
        };
        if value < -x.abs() {
            left = logp;
        } else {
            right = logp;
        }
    }
    let logp = 0.5 * (left + right);
    if x < 0.0 { logp.exp() } else { -logp.exp_m1() }
}
real_kernel! {
    /// Tukey-lambda CDF, evaluated without replacing small nonzero lambda by zero.
    tukeylambdacdf(x,lambda) {tukey(x,lambda)}
}
