//! High-precision regressions for close positive gamma arguments.
#![allow(clippy::excessive_precision)] // Retain independently computed decimal reference digits.

use sci::special::gamma::gamma_ratio;

#[test]
fn nearby_large_gamma_ratios_match_independent_high_precision_values() {
    // Values from 160-digit mpmath loggamma ratios, with the arguments treated
    // as exact binary64 numbers (sci-testing PR #2 cases).
    // The previous lgamma subtraction lost approximately 1e-11 relatively.
    for (a, b, expected) in [
        (
            5282.794500823718,
            5282.781059552735,
            1.12212040582781590889680085448054,
        ),
        (
            5199.6996996996995,
            5199.639139139139,
            1.67895495389959890820388587453931,
        ),
        (
            4754.874031883128,
            4754.830209521545,
            1.44923504163566045929956427261742,
        ),
        (
            5382.168620564661,
            5382.350185306083,
            0.21018208602924076518958593302782,
        ),
        (5000.0, 5000.25, 0.11892294132167753041150584677780),
        (1000.0, 999.5, 31.610916330809688929165901552548),
        (
            8.354343210711544,
            8.098576230796418,
            1.68728464215593772979605662534427,
        ),
    ] {
        let actual = gamma_ratio::<f64>(a, b);
        assert!(
            (actual / expected - 1.0).abs() < 2e-14,
            "gamma_ratio({a:?}, {b:?}) = {actual:.17e}, expected {expected:.17e}"
        );
        let inverse = gamma_ratio(b, a);
        assert!(
            (actual * inverse - 1.0).abs() < 3e-14,
            "gamma_ratio reciprocity failed for {a:?}, {b:?}"
        );
    }
}

#[test]
fn nearby_gamma_recurrence_and_float32_are_preserved() {
    for b in [8.0_f64, 128.0, 10_000.0, 1e10] {
        let forward = gamma_ratio(b + 1.0, b);
        let backward = gamma_ratio(b, b + 1.0);
        assert!((forward / b - 1.0).abs() < 3e-14, "b={b}");
        assert!((backward * b - 1.0).abs() < 3e-14, "b={b}");
    }
    let actual = gamma_ratio(5000.0_f32, 5000.25_f32);
    let expected = 0.11892294132167753_f32;
    assert!((actual / expected - 1.0).abs() < 3e-6);
}
