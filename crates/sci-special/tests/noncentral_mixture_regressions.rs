//! Relative accuracy of positive noncentral-distribution mixtures in small tails.

use sci_special::probability::{chndtr, ncfdtr, nctdtr};

#[test]
fn noncentral_cdfs_retain_probability_mass_far_below_poisson_mode() {
    // Independent 70–80 digit mpmath references: sum the positive beta or
    // gamma mixtures from j=0 through the Poisson mode plus 400 terms.
    // The dominant CDF terms lie far below the dominant Poisson weights.
    let cases = [
        (chndtr(2.0, 5.0, 400.0), 9.35193547998677e-80),
        (chndtr(10.0, 5.0, 400.0), 1.529767355355516e-65),
        (ncfdtr(5.0, 30.0, 400.0, 0.5), 1.198_065_154_470_286e-74),
        (ncfdtr(5.0, 30.0, 400.0, 2.0), 1.1926527035872753e-54),
        (nctdtr(30.0, 20.0, 0.5), 1.1575844338629773e-84),
        (nctdtr(30.0, 20.0, 2.0), 9.514_285_506_949_65e-68),
        (nctdtr(10_000.0, 30.0, 2.0), 8.775469923073085e-173),
    ];
    for (actual, expected) in cases {
        assert!(
            (actual / expected - 1.0).abs() < 2e-10,
            "actual {actual:e}, expected {expected:e}"
        );
    }
    assert_eq!(chndtr(0.0, 5.0, 400.0), 0.0);
    assert_eq!(ncfdtr(5.0, 30.0, 400.0, 0.0), 0.0);
}
