//! Exponential integrals, ported from xsf/expint.h (Zhang and Jin, 1996).
use crate::backend::utils::real_kernel;
#[allow(unused_imports)]
use sci_core::Float;
const EULER: f64 = 0.57721566490153286060651209008240243;

// The positive Ei series uses successive ratios x*k/(k+1)^2. Precompute
// the x-independent factor once rather than dividing in each iteration.
const EI_SERIES_RATIOS: [f64; 100] = {
    let mut ratios = [0.0; 100];
    let mut i = 0;
    while i < ratios.len() {
        let k = (i + 1) as f64;
        ratios[i] = k / ((k + 1.0) * (k + 1.0));
        i += 1;
    }
    ratios
};

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
    // exp(-x) / (x + t) rounds to zero beyond this point in binary64.
    // This also handles +infinity without forming infinity/infinity below.
    if x > 745.0 {
        return 0.0;
    }
    let m = 20 + (80.0 / x) as i32;
    let mut t = 0.0;
    for k in (1..=m).rev() {
        let k = k as f64;
        // Algebraically identical to k / (1 + k / (x + t)), with one
        // division per term. In this branch, k <= 100 and x <= 745, so
        // the product cannot overflow.
        let denominator = x + t;
        t = k * denominator / (denominator + k);
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
            for &ratio in &EI_SERIES_RATIOS {
                term *= x * ratio;
                sum += term;
                // All terms are nonnegative and sum is positive for 0 < x <= 40.
                if term <= sum * 1e-15 { break; }
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

#[cfg(test)]
mod tests {
    use super::{exp1_real, expi};

    #[test]
    fn positive_ei_series_matches_high_precision_references() {
        // mpmath Ei evaluated at the exact binary64 inputs, 80 decimal digits.
        // Include the smallest subnormal and values adjacent to Ei's zero,
        // where absolute rather than relative accuracy is appropriate.
        for (x, expected) in [
            (f64::from_bits(1), -743.8628562564797),
            (1e-300, -690.1983122333122),
            (1e-20, -45.474486194979384),
            (0.01, -4.017929465426669),
            (0.1, -1.6228128139692766),
            (0.25, -0.5425432646619137),
            (0.37250741078136657, -2.6748041020008385e-16),
            (0.3725074107813666, -5.119698936555685e-17),
            (0.3725074107813667, 1.6508643146897012e-16),
            (0.5, 0.4542199048631736),
            (1.0, 1.8951178163559368),
            (2.0, 4.95423435600189),
            (5.0, 40.18527535580318),
            (10.0, 2492.2289762418777),
            (20.0, 25615652.664056588),
            (40.0, 6039718263611242.0),
        ] {
            let actual = expi(x).unwrap();
            assert!(
                (actual - expected).abs() <= 16.0 * f64::EPSILON * expected.abs().max(1.0),
                "Ei({x})={actual:e}, reference={expected:e}"
            );
        }
    }

    // Independent copy of the original two-division backward continued
    // fraction, kept to detect changes in rounding, truncation and endpoints.
    fn original_exp1(x: f64) -> f64 {
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
            return -super::EULER - x.ln() + x * sum;
        }
        let m = 20 + (80.0 / x) as i32;
        let mut t = 0.0;
        for k in (1..=m).rev() {
            let k = k as f64;
            t = k / (1.0 + k / (x + t));
        }
        (-x).exp() / (x + t)
    }

    #[test]
    fn exp1_cf_matches_previous_recurrence_and_endpoints() {
        for x in [
            -1.0,
            -0.0,
            0.0,
            1.0,
            f64::from_bits(1.0_f64.to_bits() + 1),
            2.0,
            10.0,
            40.0,
            100.0,
            700.0,
            744.0,
            745.0,
            746.0,
            f64::INFINITY,
            f64::NAN,
        ] {
            let expected = original_exp1(x);
            let actual = exp1_real(x);
            if expected.is_nan() {
                assert!(actual.is_nan(), "x={x:?}: expected NaN, got {actual:?}");
            } else if expected.is_infinite() || expected == 0.0 {
                assert_eq!(actual.to_bits(), expected.to_bits(), "x={x:?}");
            } else {
                assert!(
                    (actual - expected).abs() <= 16.0 * f64::EPSILON * expected.abs(),
                    "x={x:?}: {actual:e} versus previous {expected:e}"
                );
            }
        }

        // Dense near-one and logarithmically spaced samples exercise every
        // change in the original m = 20 + floor(80/x) depth schedule.
        for i in 1..=4096 {
            let x = 1.0 + (i as f64) / 4096.0;
            let expected = original_exp1(x);
            let actual = exp1_real(x);
            assert!(
                (actual - expected).abs() <= 16.0 * f64::EPSILON * expected.abs(),
                "x={x}: {actual:e} versus previous {expected:e}"
            );
        }
        for i in 0..=2048 {
            let x = (745.0_f64.ln() * (i as f64) / 2048.0).exp();
            let expected = original_exp1(x);
            let actual = exp1_real(x);
            let tolerance = 16.0 * f64::EPSILON * expected.abs() + 2.0 * f64::from_bits(1);
            assert!(
                (actual - expected).abs() <= tolerance,
                "x={x}: {actual:e} versus previous {expected:e}"
            );
        }
    }
}
