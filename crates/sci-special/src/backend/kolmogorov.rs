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
    let Some((a, sum, weighted)) = sums(x) else {
        return (f64::NEG_INFINITY, f64::NEG_INFINITY);
    };
    if x <= 0.82 {
        let log_x = x.ln();
        let logcdf = 0.5 * (2.0 * PI).ln() - log_x - a + sum.ln();
        let logpdf = 0.5 * (2.0 * PI).ln() - 2.0 * log_x - a + weighted.ln();
        (logcdf, logpdf)
    } else {
        let logsf = LN_2 - a + sum.ln();
        let logpdf = (8.0 * x).ln() - a + weighted.ln();
        (logsf, logpdf)
    }
}

/// Log of the tail computed by `series` and the hazard ratio density/tail.
/// The ratio follows from the series sums directly, so the Newton iteration
/// of the inverse needs one logarithm per step instead of three logarithms
/// and an exponential.
fn series_ratio(x: f64) -> (f64, f64) {
    let Some((a, sum, weighted)) = sums(x) else {
        return (f64::NEG_INFINITY, f64::NAN);
    };
    if x <= 0.82 {
        (
            0.5 * (2.0 * PI).ln() - a + (sum / x).ln(),
            weighted / (sum * x),
        )
    } else {
        (LN_2 - a + sum.ln(), 4.0 * x * weighted / sum)
    }
}

/// The exponent `a` and the tail and density sums of `series`, or `None`
/// when the exponent overflows.
fn sums(x: f64) -> Option<(f64, f64, f64)> {
    if x <= 0.82 {
        let a = (PI / x) * (PI / x) / 8.0;
        if !a.is_finite() {
            return None;
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
        Some((a, sum, weighted))
    } else {
        let a = 2.0 * x * x;
        if !a.is_finite() {
            return None;
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
        Some((a, sum, weighted))
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

/// Logarithm of the requested tail (CDF if `lower`) and the ratio of the
/// density to that tail, forming the complementary tail only when the series
/// computes the other one.
fn tail(x: f64, lower: bool) -> (f64, f64) {
    if x < 0.01 || x == f64::INFINITY {
        let (cdf, sf, pdf) = logarithms(x);
        let log_tail = if lower { cdf } else { sf };
        return (log_tail, (pdf - log_tail).exp());
    }
    let (log_tail, ratio) = series_ratio(x);
    if lower == (x <= 0.82) {
        (log_tail, ratio)
    } else {
        let complement = (-log_tail.exp()).ln_1p();
        (complement, ratio * (log_tail - complement).exp())
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
        // The lower-tail leading term is
        //   p = 4 sqrt(a/pi) exp(-a),  a = pi^2/(8 x^2).
        // Hence a - ln(a)/2 = -ln(p) + ln(4/sqrt(pi)).  The old
        // estimate a=-ln(p) ignored the 1/x prefactor, often starting
        // above the 0.82 series transition and costing extra Newton steps.
        // One scalar Newton correction at a=rhs greatly improves the seed;
        // the safeguarded inverse iteration below remains authoritative.
        // ln(4/sqrt(pi)), evaluated once at binary64 precision.
        const LOG_SCALE: f64 = 0.813_929_418_195_190_5;
        let rhs = -target + LOG_SCALE;
        let a = rhs + (0.5 * rhs.ln()) / (1.0 - 0.5 / rhs);
        PI / (8.0 * a).sqrt()
    } else {
        ((LN_2 - target) * 0.5).sqrt()
    };
    // The requested log tail is monotone in x, so the sign of each residual
    // moves one end of the bracket [left, right] (initially [0, inf));
    // Newton steps that leave it are replaced by bisection or doubling.
    let (mut left, mut right) = (0.0, f64::INFINITY);
    let mut x = initial;
    // Size of the previous Newton step, for the convergence estimate below.
    let mut previous_step = f64::NAN;
    for _ in 0..128 {
        let (log_tail, ratio) = tail(x, lower);
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
        let slope = if lower { ratio } else { -ratio };
        let step = value / slope;
        let next = x - step;
        x = if value.is_finite() && next > left && next < right {
            if step.abs() <= 2.0 * f64::EPSILON * next.abs() {
                return Ok(next);
            }
            // Newton converges quadratically, e_(n+1) ≈ C e_n², and two
            // consecutive steps estimate C ≈ |step| / |previous|². Once the
            // predicted error left by this step is below a quarter ulp, the
            // confirming evaluation that would only reproduce `next` is skipped.
            let predicted = step.abs().powi(3) / (previous_step * previous_step);
            if step.abs() <= 1e-6 * next.abs() && predicted <= 0.25 * f64::EPSILON * next.abs() {
                return Ok(next);
            }
            previous_step = step;
            next
        } else {
            previous_step = f64::NAN;
            if right.is_finite() {
                0.5 * (left + right)
            } else {
                2.0 * x
            }
        };
    }
    Err("Zero finder did not converge.")
}

// The forward survival probability needs neither logarithmic complements nor
// the density used by the inverse. Split the exponential in the subnormal
// tail, so exp(-a) cannot underflow before multiplication by two.
fn survival(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.01 {
        return 1.0;
    }
    let Some((a, sum, _)) = sums(x) else {
        return 0.0;
    };
    if x <= 0.82 {
        1.0 - (2.0 * PI).sqrt() / x * (-a).exp() * sum
    } else if a < 700.0 {
        2.0 * (-a).exp() * sum
    } else {
        let half_exponent = (-0.5 * a).exp();
        (2.0 * half_exponent) * half_exponent * sum
    }
}

real_kernel! {
/// Survival function of the limiting two-sided Kolmogorov distribution.
/// Returns one for x <= 0 and zero at positive infinity.
    kolmogorov(x) { survival(x) }
}
real_kernel! {
/// Inverse Kolmogorov survival probability, for 0 <= p <= 1.
    kolmogi(p) { inverse(p,false)? }
}

#[cfg(test)]
mod tests {
    use super::{logarithms, series, series_ratio, tail};
    use core::f64::consts::PI;

    fn close(actual: f64, expected: f64, label: &str) {
        let tolerance = 3e-13 * expected.abs().max(1e-300);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}: actual {actual}, expected {expected}, tolerance {tolerance}"
        );
    }

    // Independent theta and alternating Gaussian representations of the
    // Kolmogorov limiting law, evaluated term-by-term rather than through
    // the recurrence ratios used by the production series.
    fn reference(x: f64) -> (f64, f64) {
        if x <= 0.82 {
            let mut cdf = 0.0;
            let mut pdf = 0.0;
            for k in 1..=64 {
                let odd = (2 * k - 1) as f64;
                let exponent = -odd * odd * PI * PI / (8.0 * x * x);
                let term = exponent.exp();
                cdf += term;
                pdf += (-2.0 * exponent - 1.0) * term;
            }
            let prefactor = (2.0 * PI).sqrt() / x;
            (prefactor * cdf, prefactor * pdf / x)
        } else {
            let mut sf = 0.0;
            let mut pdf = 0.0;
            for k in 1..=64 {
                let parity = if k % 2 == 0 { -1.0 } else { 1.0 };
                let term = (-2.0 * (k * k) as f64 * x * x).exp();
                sf += parity * term;
                pdf += parity * (k * k) as f64 * term;
            }
            (2.0 * sf, 8.0 * x * pdf)
        }
    }

    #[test]
    fn private_log_series_matches_independent_distribution_and_density() {
        for x in [0.25, 0.4, 0.7, 0.819, 0.82, 0.821, 1.0, 1.5, 2.0, 3.0] {
            let (tail_probability, pdf) = reference(x);
            let (log_tail, log_pdf) = series(x);
            close(log_tail, tail_probability.ln(), "series log probability");
            close(log_pdf, pdf.ln(), "series log density");
            let (ratio_tail, hazard) = series_ratio(x);
            close(ratio_tail, log_tail, "log probability / hazard path");
            close(hazard, pdf / tail_probability, "pdf / tail ratio");

            let (logcdf, logsf, logarithmic_pdf) = logarithms(x);
            close(logarithmic_pdf, log_pdf, "logarithmic density");
            if x <= 0.82 {
                close(logcdf, log_tail, "logarithmic lower tail");
                close(logsf, (-tail_probability).ln_1p(), "logarithmic survival");
            } else {
                close(logsf, log_tail, "logarithmic upper tail");
                close(
                    logcdf,
                    (-tail_probability).ln_1p(),
                    "logarithmic cumulative",
                );
            }
            for lower in [false, true] {
                let (log_probability, ratio) = tail(x, lower);
                let expected = if lower { logcdf } else { logsf };
                close(log_probability, expected, "requested log tail");
                close(ratio, pdf / expected.exp(), "requested density / tail");
            }
        }
    }

    #[test]
    fn private_log_tails_keep_extreme_limits_without_spurious_probabilities() {
        for x in [-1.0, 0.0, 0.005] {
            let (cdf, sf, density) = logarithms(x);
            assert_eq!(cdf, f64::NEG_INFINITY);
            assert_eq!(sf, 0.0);
            assert_eq!(density, f64::NEG_INFINITY);
            let (log_lower, _) = tail(x, true);
            let (log_upper, _) = tail(x, false);
            assert_eq!(log_lower, f64::NEG_INFINITY);
            assert_eq!(log_upper, 0.0);
        }
        for x in [f64::INFINITY, f64::MAX] {
            let (cdf, sf, density) = logarithms(x);
            assert_eq!(cdf, 0.0);
            assert_eq!(sf, f64::NEG_INFINITY);
            assert_eq!(density, f64::NEG_INFINITY);
        }
        // At x=19, survival itself is subnormal, but its logarithm and
        // hazard have precise leading-term analytical expressions.
        let x = 19.0;
        let (log_tail, log_pdf) = series(x);
        let a = -2.0 * x * x;
        close(log_tail, (2.0_f64).ln() + a, "subnormal tail logarithm");
        close(log_pdf, (8.0 * x).ln() + a, "subnormal density logarithm");
        let (_, hazard) = series_ratio(x);
        close(hazard, 4.0 * x, "subnormal hazard");
    }

    #[test]
    fn inverse_matches_independent_series_on_exact_seeded_benchmark_inputs() {
        // The 256 `kolmogi` values in benches/special_scipy.rs use seed 165.
        // Verify the resulting quantile against independently evaluated
        // Gaussian/theta series, not the inverse's recurrence-ratio kernel.
        let mut state = 165_u64
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        for _ in 0..256 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let v = ((state >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0;
            let p = 0.5 * (v + 1.0).clamp(1e-6, 2.0 - 1e-6);
            let x = super::inverse(p, false).unwrap();
            let (tail, _) = reference(x);
            if x <= 0.82 {
                // Use the CDF directly to avoid subtracting two nearly
                // equal numbers in the upper survival tail.
                close(tail, 1.0 - p, "inverse lower-tail reference");
            } else {
                close(tail, p, "inverse survival reference");
            }
        }
        // Independent SciPy fixture; this exercises the unreflected upper
        // survival branch in addition to the seeded comparisons above.
        close(
            super::inverse(0.3, false).unwrap(),
            0.973_063_375_332_372_6,
            "SciPy inverse quantile",
        );
    }
}
