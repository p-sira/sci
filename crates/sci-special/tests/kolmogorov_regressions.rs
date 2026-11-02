// Preserve original high-precision independent numerical reference literals.
#![allow(clippy::excessive_precision)]
// Integration regressions extracted from the sci-special numerical backends.
use core::f64::consts::PI;
use sci_special::probability::kolmogi;

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
        let x = kolmogi(p);
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
        kolmogi(0.3_f64),
        0.973_063_375_332_372_6,
        "SciPy inverse quantile",
    );
}
