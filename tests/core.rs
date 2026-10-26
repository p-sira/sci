mod common;
use sci::Error;

#[test]
fn shared_errors_preserve_messages_and_have_no_source() {
    use std::error::Error as _;
    for (error, message) in [
        (
            Error::InvalidStep,
            "step must be finite and greater than zero",
        ),
        (
            Error::InvalidInput("invalid parameter"),
            "invalid parameter",
        ),
        (Error::DimensionMismatch, "input dimensions do not agree"),
        (
            Error::InconsistentOutputDimension {
                positive: 3,
                negative: 2,
            },
            "output dimension changed from 2 to 3",
        ),
        (
            Error::NonFiniteEvaluation,
            "evaluation returned a nonfinite value",
        ),
        (Error::NumericalFailure("solver failed"), "solver failed"),
    ] {
        assert_eq!(error.to_string(), message);
        assert_eq!(error.clone(), error);
        assert!(error.source().is_none());
    }
}

#[test]
fn finite_checks_preserve_ieee_values_and_reject_nonfinite_inputs() {
    for value in [-0.0_f64, 0.0, f64::from_bits(1), -f64::MAX, f64::MAX] {
        assert_eq!(sci_core::finite(value).unwrap().to_bits(), value.to_bits());
        assert_eq!(sci_core::finite_slice(&[value]), Ok(()));
    }
    assert_eq!(sci_core::finite_slice::<f64>(&[]), Ok(()));
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(sci_core::finite(bad), Err(Error::NonFiniteEvaluation));
        for values in [[bad, 1.0], [1.0, bad]] {
            assert_eq!(
                sci_core::finite_slice(&values),
                Err(Error::InvalidInput("coordinates must be finite"))
            );
        }
    }
}

#[test]
fn fixture_comparisons_preserve_bounds_and_special_values() {
    use common::matches;
    assert!(matches(1.125, 1., 0.0625, 0.0625));
    assert!(!matches(1.125 + f64::EPSILON, 1., 0.0625, 0.0625));
    assert!(!matches(0., 1e-100, 0., 1e-12));
    assert!(matches(f64::NAN, f64::NAN, 0., 0.));
    assert!(!matches(0., f64::NAN, 1., 1.));
    assert!(!matches(f64::NAN, 0., 1., 1.));
    for infinity in [f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches(infinity, infinity, 0., 0.));
        assert!(!matches(-infinity, infinity, 1., 1.));
        assert!(!matches(infinity, 0., 1., 1.));
    }
    // Ordinary equality permits both zero signs; contract tests check bits.
    assert!(matches(-0., 0., 0., 0.));
}
