//! Public noncentral Student-t regressions, including large Poisson centers.
#![allow(clippy::excessive_precision)]

use sci_special::probability::{nctdtr, nctdtrinc, nctdtrit};

#[test]
fn noncentral_t_cdf_matches_independent_scipy_references() {
    // Values from scipy.special.nctdtr (SciPy 1.17.0), independent of the
    // Poisson-series gamma-ratio implementation. Include the CodSpeed input,
    // tiny df, both noncentrality signs, and centers up to nc^2/2 = 5000.
    let cases: [(f64, f64, f64, f64); 13] = [
        (3.0, 0.5, 0.3, 0.41207163208530162),
        (3.0, -0.5, 0.3, 0.77968132058883288),
        (1.0, 5.0, 3.0, 0.11384628383771248),
        (30.0, 5.0, 3.0, 0.02960387195367457),
        (0.01, 3.0, 0.5, 0.0086217579821215393),
        (0.01, -3.0, 0.5, 0.99869611408736025),
        (100.0, 12.0, 9.0, 0.0055282341711740124),
        (10000.0, 12.0, 9.0, 0.00137598231690598),
        (10000.0, -12.0, -9.0, 0.99862401768309406),
        (0.01, 40.0, 50.0, 0.025544543619969717),
        (10000.0, 80.0, 81.0, 0.80674274169681792),
        (10000.0, -80.0, -81.0, 0.19325725830318211),
        (5.0, 100.0, 100.0, 0.41594115592792102),
    ];
    for (df, nc, t, expected) in cases {
        let actual = nctdtr(df, nc, t);
        // The existing noncentral-t series has incomplete-beta and tail
        // cancellation errors; these bounds test the result, not exact bits.
        let tolerance = 2e-8 + 2e-6 * expected.abs();
        assert!(
            actual.is_finite() && (actual - expected).abs() <= tolerance,
            "nctdtr({df}, {nc}, {t}) = {actual:.17e}; SciPy = {expected:.17e}; tolerance = {tolerance:e}"
        );
    }
}

#[test]
fn noncentral_t_reflection_and_inverse_roundtrips() {
    // The t<0 branch uses F(-t; -nc) = 1-F(t; nc).
    // Large |nc| exercises recurrence reseeding in both mixture directions.
    for (df, nc, t) in [
        (0.01, 0.5, 0.3),
        (3.0, 5.0, 2.0),
        (30.0, -5.0, 3.0),
        (10000.0, 80.0, 81.0),
        (5.0, 100.0, 100.0),
    ] {
        let upper = nctdtr(df, nc, t);
        let reflected = nctdtr(df, -nc, -t);
        assert!(
            upper.is_finite() && reflected.is_finite() && (upper + reflected - 1.0).abs() <= 2e-12,
            "reflection failed at df={df}, nc={nc}, t={t}: {upper}, {reflected}"
        );
    }

    // Independent SciPy inverse references for the benchmark neighborhood.
    let nc = nctdtrinc(3.0, 0.3, 0.5);
    let t = nctdtrit(3.0, 0.5, 0.3);
    assert!((nc - 0.9944526759482717).abs() < 5e-6, "nc = {nc}");
    assert!((t - (-0.026514677628187912)).abs() < 5e-6, "t = {t}");
    assert!((nctdtr(3.0, nc, 0.5) - 0.3).abs() < 2e-8);
    assert!((nctdtr(3.0, 0.5, t) - 0.3).abs() < 2e-8);
}

#[test]
fn noncentral_t_domain_and_endpoint_behavior() {
    for value in [
        nctdtr(0.009, 0.5, 1.0),
        nctdtr(10001.0, 0.5, 1.0),
        nctdtr(3.0, 100.0001, 1.0),
        nctdtr(3.0, -100.0001, 1.0),
        nctdtr(3.0, f64::NAN, 1.0),
        nctdtr(3.0, 0.5, f64::NAN),
    ] {
        assert!(value.is_nan());
    }
    assert_eq!(nctdtr(3.0, 10.0, f64::INFINITY), 1.0);
    assert_eq!(nctdtr(3.0, 10.0, f64::NEG_INFINITY), 0.0);
    assert_eq!(nctdtr(3.0, 0.0, 0.0), 0.5);
    assert!((nctdtr(3.0, 0.5, 0.0) - 0.3085375387259869).abs() < 2e-15);
}
