//! Probability tables and distribution utilities with caller-owned storage.
use super::{cephes64, utils::real_kernel};

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
    if x.is_nan() || !lambda.is_finite() {
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
            let exp_x = x.exp();
            exp_x / (1.0 + exp_x)
        } else {
            1.0 / (1.0 + (-x).exp())
        };
    }
    if !x.is_finite() {
        return if x < 0.0 { 0.0 } else { 1.0 };
    }
    // The quantile map is strictly increasing, and is linear for lambda=1,2.
    // The former 64-step log-space bisection was the dominant cost for the
    // CDF; these exact reductions also avoid a loss of precision near zero.
    if lambda == 1.0 {
        return 0.5 * (1.0 + x);
    }
    if lambda == 2.0 {
        return 0.5 + x;
    }

    // Solve only the smaller tail in log probability, so the opposite tail
    // remains accurate even when p is subnormal. Newton's derivative of
    // Q_lambda(p) with respect to log(p) is
    // p^lambda + p*(1-p)^(lambda-1), which is positive for every real lambda.
    // Safeguard each Newton step by the original monotone bracket and retain
    // bisection if a difficult tail does not converge quickly.
    let magnitude = x.abs();
    let mut left = -746.0;
    let mut right = -core::f64::consts::LN_2;
    let mut logp = if magnitude < 35.0 {
        -(1.0 + magnitude.exp()).ln()
    } else {
        -magnitude
    }
    .clamp(left, right);
    let mut converged = false;
    for _ in 0..12 {
        let p = logp.exp();
        let lp = lambda * logp;
        let lq = lambda * (-p).ln_1p();
        let (value, a, b) = if lambda.abs() < 0.5 {
            let a = lp.exp_m1();
            let b = lq.exp_m1();
            ((a - b) / lambda, 1.0 + a, 1.0 + b)
        } else {
            let a = lp.exp();
            let b = lq.exp();
            ((a - b) / lambda, a, b)
        };
        let error = value + magnitude;
        if error < 0.0 {
            left = logp;
        } else {
            right = logp;
        }
        let derivative = a + (p / (1.0 - p)) * b;
        let delta = error / derivative;
        if delta.is_finite() && delta.abs() <= 4.0 * f64::EPSILON * (1.0 + logp.abs()) {
            converged = true;
            break;
        }
        let candidate = logp - delta;
        logp = if candidate > left && candidate < right {
            candidate
        } else {
            0.5 * (left + right)
        };
    }
    // The bracket remains valid even after an overflowing Newton estimate.
    // Continue with bisection only if Newton failed to narrow it enough.
    if !converged {
        for _ in 0..64 {
            logp = 0.5 * (left + right);
            let p = logp.exp();
            let lp = lambda * logp;
            let lq = lambda * (-p).ln_1p();
            let value = if lambda.abs() < 0.5 {
                (lp.exp_m1() - lq.exp_m1()) / lambda
            } else {
                (lp.exp() - lq.exp()) / lambda
            };
            if value < -magnitude {
                left = logp;
            } else {
                right = logp;
            }
        }
        logp = 0.5 * (left + right);
    }
    if x < 0.0 { logp.exp() } else { -logp.exp_m1() }
}
real_kernel! {
    /// Tukey-lambda CDF, evaluated without replacing small nonzero lambda by zero.
    tukeylambdacdf(x,lambda) {tukey(x,lambda)}
}
