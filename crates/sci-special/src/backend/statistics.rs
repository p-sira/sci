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
            x.exp() / (1.0 + x.exp())
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

#[cfg(test)]
mod tukey_tests {
    use super::tukey;

    fn close(actual: f64, expected: f64, rtol: f64, atol: f64) {
        assert!(
            actual.is_finite() && (actual - expected).abs() <= atol + rtol * expected.abs(),
            "actual={actual:.17e}, reference={expected:.17e}, rtol={rtol:e}, atol={atol:e}"
        );
    }

    #[test]
    fn tukey_quantiles_match_independent_high_precision_roots() {
        // 85-digit decimal bisection of the defining quantile
        // Q_lambda(p) = (p^lambda - (1-p)^lambda) / lambda.
        for (x, lambda, reference) in [
            (-2.5, -0.4, 0.161_611_776_379_049_35),
            (-1.25, -0.1, 0.240_932_564_535_160_28),
            (-1.75, 0.2, 0.098_782_303_656_846_74),
            (0.3, 0.75, 0.625_710_681_194_703_4),
            (-0.6, 1.5, 0.058_168_226_949_238_65),
            (1.9, 0.5, 0.997_617_762_444_227_9),
            (0.6, 1.5, 0.941_831_773_050_761_4),
            (3.0, -1.0, 0.767_591_879_243_998_2),
            (0.5, 0.3, 0.651_027_827_782_848_4),
            (-0.5, 0.3, 0.348_972_172_217_151_65),
            (0.005, 1e-6, 0.501_249_998_262_272_6),
            (5.0, -0.5, 0.920_309_670_436_742_8),
        ] {
            close(tukey(x, lambda), reference, 3e-13, 2e-15);
            close(tukey(-x, lambda), 1.0 - reference, 3e-13, 2e-15);
        }
    }

    #[test]
    fn tukey_special_cases_and_numerical_boundaries() {
        for x in [-0.9, -0.5, -0.125, 0.125, 0.5, 0.9] {
            close(tukey(x, 1.0), (1.0 + x) / 2.0, 0.0, 0.0);
        }
        for x in [-0.45, -0.125, 0.125, 0.45] {
            close(tukey(x, 2.0), x + 0.5, 0.0, 0.0);
        }
        for x in [-10.0_f64, -1.0, 0.5, 10.0] {
            let logistic = if x < 0.0 {
                x.exp() / (1.0 + x.exp())
            } else {
                1.0 / (1.0 + (-x).exp())
            };
            close(tukey(x, 0.0), logistic, 0.0, 0.0);
        }
        for lambda in [-2.0, -0.5, -1e-12, 0.0, 1e-12, 0.5, 1.0, 2.0] {
            close(tukey(0.0, lambda), 0.5, 0.0, 0.0);
            assert_eq!(tukey(f64::NEG_INFINITY, lambda), 0.0);
            assert_eq!(tukey(f64::INFINITY, lambda), 1.0);
            assert!(tukey(f64::NAN, lambda).is_nan());
        }
        assert!(tukey(0.1, f64::NAN).is_nan());
        assert!(tukey(0.1, f64::INFINITY).is_nan());
        assert_eq!(tukey(-2.0, 1.0), 0.0);
        assert_eq!(tukey(2.0, 1.0), 1.0);
        assert_eq!(tukey(-0.5, 2.0), 0.0);
        assert_eq!(tukey(0.5, 2.0), 1.0);
    }

    #[test]
    fn tukey_inverts_defining_quantile_across_tails() {
        // This independently constructs x from a prescribed probability,
        // rather than differentiating or reusing the implementation's root.
        for lambda in [-2.0, -0.5, -0.001, 0.25, 0.5, 1.5] {
            for p in [1e-6_f64, 0.01, 0.25] {
                let q = 1.0 - p;
                let x = (p.powf(lambda) - q.powf(lambda)) / lambda;
                close(tukey(x, lambda), p, 3e-10, 1e-14);
                close(tukey(-x, lambda), q, 3e-10, 1e-14);
            }
        }
        // Negative lambda has unbounded support. Solving in log(p) avoids
        // losing representable probabilities in extremely distant tails.
        for lambda in [-1.0, -0.5] {
            let p = 1e-300_f64;
            let x = (p.powf(lambda) - (1.0 - p).powf(lambda)) / lambda;
            close(tukey(x, lambda), p, 3e-11, 0.0);
        }
    }
}
