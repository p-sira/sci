// Preserve original high-precision independent numerical reference literals.
#![allow(clippy::excessive_precision)]
// Integration regressions extracted from the sci-special numerical backends.
use sci_special::probability::tklmbda as tukey;

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
        (-1.25, -0.1, 0.240_932_564_535_160_3),
        (-1.75, 0.2, 0.098_782_303_656_846_74),
        (0.3, 0.75, 0.625_710_681_194_703_4),
        (-0.6, 1.5, 0.058_168_226_949_238_65),
        (1.9, 0.5, 0.997_617_762_444_228),
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
