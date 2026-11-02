// Preserve original high-precision independent numerical reference literals.
#![allow(clippy::excessive_precision)]
// Integration regressions extracted from the sci-special numerical backends.
use sci_special::{combinatorics::binom, gamma::gammasgn as sign_gamma};

fn assert_relative(actual: f64, expected: f64, relative_tolerance: f64) {
    assert!(
        actual.is_finite()
            && (actual - expected).abs() <= relative_tolerance * expected.abs().max(1.0),
        "binom: got {actual:.17e}, expected {expected:.17e}, rtol={relative_tolerance:e}"
    );
}

#[test]
fn binom_bounded_fractional_arguments_match_exact_gamma_half_integer_values() {
    // For half-integer arguments, the sqrt(pi) factors cancel exactly.
    // Values below are independent rational evaluations of Gamma ratios.
    for (n, k, expected) in [
        (1.5, 0.5, 1.5),
        (2.5, 0.5, 1.875),
        (3.5, 0.5, 2.1875),
        (4.5, 1.5, 6.5625),
        (
            10.0,
            0.5,
            2.0 * 4.0_f64.powi(10) / (std::f64::consts::PI * 184756.0),
        ),
    ] {
        assert_relative(binom(n, k), expected, 6e-14);
        assert_relative(binom(n, n - k), expected, 6e-14);
    }
}

#[test]
fn binom_bounded_fractional_arguments_obey_recurrences_and_symmetry() {
    for (n, k) in [
        (1.2_f64, 0.3),
        (2.7, 0.6),
        (4.125, 1.25),
        (12.5, 3.75),
        (29.5, 12.25),
    ] {
        let value = binom(n, k);
        assert_relative(binom(n, n - k), value, 1e-12);
        assert_relative(binom(n, k + 1.0), value * (n - k) / (k + 1.0), 1e-12);
    }
    // Existing signed and pole paths must remain untouched by the
    // positive-argument fast path.
    assert_relative(binom(0.5_f64, 2.0), -0.125, 2e-14);
    assert_relative(binom(0.1_f64, 2.0), -0.045, 2e-14);
    assert_relative(binom(30.0_f64, 15.0), 155117520.0, 2e-14);
    assert_relative(binom(2.0_f64, 12.0), 0.0, 0.0);
    assert!(binom(-3.0_f64, 2.0).is_nan());
}

#[test]
fn binom_seeded_cases_match_100_digit_gamma_references() {
    // Independently computed with mpmath at 100 decimal digits, treating
    // n and k as exact binary64 values and n-k at arbitrary precision.
    // Indices are from the benchmark's fixed (seed 8, seed 9) sequence;
    // the n<1 case also checks the retained general-beta fallback.
    for (n, k, reference) in [
        (
            15.990223953834528,
            10.321218532934989,
            "6713.18982494697864623107385750211826870114555812464024624959410409239341446866887814",
        ),
        (
            18.463499181702076,
            3.095743195849104,
            "1021.17259154394229533013294316947146555051393852909399628256408494735210698760707714",
        ),
        (
            0.09935884396120476,
            0.05067905159747982,
            "1.00379197716347086154031807812627649896523765424537176430885407529346603267795366854",
        ),
        (
            26.34572709026411,
            4.406417431487581,
            "29933.8479301254403024943219609438703156023254840901823550776501824382542107748611688",
        ),
        (
            15.061315477248781,
            4.232291738267407,
            "1715.95907738636030818785699339486878050358299933238476706294387922196516307070950558",
        ),
        (
            28.27617055736016,
            12.430054578325967,
            "39595701.864916808560415119087182910499797153602495802943786715250907119168336140177",
        ),
    ] {
        let expected: f64 = reference.parse().unwrap();
        assert_relative(binom(n, k), expected, 2e-13);
    }
}

#[test]
fn sign_gamma_handles_negative_intervals_and_large_exact_parity() {
    for (x, expected) in [
        (-0.5, -1.0),
        (-1.5, 1.0),
        (-2.5, -1.0),
        (-3.5, 1.0),
        (-(2.0_f64.powi(52) - 0.5), 1.0),
        (-(2.0_f64.powi(52) - 1.5), -1.0),
        (0.5, 1.0),
    ] {
        assert_eq!(sign_gamma(x), expected, "x = {x}");
    }
    for x in [-2.0, -1.0, f64::NEG_INFINITY, f64::NAN] {
        assert!(sign_gamma(x).is_nan(), "x = {x}");
    }
    assert_eq!(sign_gamma(0.0_f64).to_bits(), 1.0_f64.to_bits());
    assert_eq!(sign_gamma(-0.0_f64).to_bits(), (-1.0_f64).to_bits());
}
