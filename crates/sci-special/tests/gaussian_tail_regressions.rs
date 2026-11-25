//! Independent high-precision references for representable Gaussian tails.

use sci_special::{elementary::erfc, probability::ndtr};

fn close_tail(actual: f64, expected: f64) {
    // A relative tolerance covers normal and large subnormal results; one
    // minimum subnormal covers rounding in the smallest nonzero values.
    if expected == 0.0 {
        assert_eq!(actual, 0.0);
    } else {
        assert!(actual > 0.0);
        assert!(
            (actual - expected).abs() <= (expected * 3e-13).max(f64::from_bits(1)),
            "actual={actual:e}, expected={expected:e}"
        );
    }
}

#[test]
fn normal_cdf_preserves_subnormal_probabilities() {
    // mpmath, 100 decimal digits: erfc(-x/sqrt(2))/2, rounded to f64.
    for (x, expected) in [
        (-37.0, 5.725571222524577e-300),
        (-37.00000000000001, 5.7255712225230705e-300),
        (-37.6, 1.074811249587044e-309),
        (-37.6996996996997, 2.511781271177e-311),
        (-37.7, 2.4834853102776e-311),
        (-38.0, 2.88542835e-316),
        (-38.4, 6.4e-323),
        (-38.48, f64::from_bits(1)),
        (-38.5, 0.0),
        (-f64::MAX, 0.0),
        (f64::NEG_INFINITY, 0.0),
    ] {
        close_tail(ndtr(x), expected);
    }
    assert_eq!(ndtr(f64::INFINITY), 1.0);
    assert!(ndtr(f64::NAN).is_nan());
}

#[test]
fn complementary_error_function_preserves_subnormal_values() {
    // mpmath, 100 decimal digits: erfc(x), rounded to f64.
    for (x, expected) in [
        (26.5, 2.2109076642637343e-307),
        (26.7, 5.253110413596e-312),
        (27.0, 5.23705e-319),
        (27.2, 1e-323),
        (27.22, f64::from_bits(1)),
        (27.23, 0.0),
        (f64::MAX, 0.0),
        (f64::INFINITY, 0.0),
    ] {
        close_tail(erfc(x), expected);
        assert_eq!(erfc(-x), 2.0);
    }
    assert!(erfc(f64::NAN).is_nan());
}
