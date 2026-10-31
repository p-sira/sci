//! Independent closed-form and high-precision boundary references for the
//! public Beta, Riemann zeta, Hurwitz zeta, and zetac functions.
//!
//! Public beta/ betaln intentionally restrict arguments to finite positive
//! reals. The negative-integer-pole cases in the private Cephes beta kernel
//! are therefore not exposed by these public functions.

use sci::special as s;
use std::f64::consts::PI;

fn close(actual: f64, expected: f64, relative: f64, absolute: f64) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= absolute + relative * expected.abs(),
        "actual={actual:.17e}, reference={expected:.17e}, rtol={relative:e}, atol={absolute:e}"
    );
}

fn integer_beta(a: u32, b: u32) -> f64 {
    // B(a,b) = (b-1)! / [a(a+1)...(a+b-1)] for positive integers.
    // The intermediate products in these tests fit exactly into f64's range.
    let numerator = (1..b).map(|k| k as f64).product::<f64>();
    let denominator = (a..a + b).map(|k| k as f64).product::<f64>();
    numerator / denominator
}

#[test]
fn beta_integer_factorials_and_logarithms() {
    for (a, b) in [
        (1, 1),
        (1, 9),
        (2, 3),
        (3, 7),
        (7, 3),
        (12, 15),
        (171, 2),
        (200, 3),
    ] {
        let expected = integer_beta(a, b);
        let (a, b) = (a as f64, b as f64);
        close(s::beta(a, b), expected, 2e-12, 0.0);
        close(s::beta(b, a), expected, 2e-12, 0.0);
        close(s::betaln(a, b), expected.ln(), 2e-12, 3e-13);
        close(s::betaln(b, a), expected.ln(), 2e-12, 3e-13);
    }

    // B(1/2,1/2) = π and B(1/2,1) = 2 exactly.
    close(s::beta(0.5, 0.5), PI, 4e-14, 0.0);
    close(s::betaln(0.5, 0.5), PI.ln(), 4e-14, 0.0);
    close(s::beta(0.5, 1.0), 2.0, 4e-14, 0.0);
    close(s::betaln(0.5, 1.0), 2.0_f64.ln(), 4e-14, 0.0);
}

#[test]
fn beta_extreme_positive_arguments_preserve_closed_forms_and_asymptotics() {
    // B(a,1)=1/a and B(a,2)=1/[a(a+1)] even when a+b rounds to a.
    // Both orders exercise the argument-swap path.
    for a in [1e-8, 1.0, 1e4, 1e9, 1e20, 1e100, 1e308] {
        // At a=10^4, log Gamma(a)+log Gamma(1)-log Gamma(a+1)
        // subtracts two numbers near 8.2e4 to produce ln(1/a).
        // This gives ~5e-12 relative error here; independent Python
        // math.lgamma subtraction has ~9.6e-12 error for the same
        // identity. Keep the larger allowance local to this input.
        let rtol = if a == 1e4 { 1.5e-11 } else { 3e-12 };
        close(s::beta(a, 1.0), 1.0 / a, rtol, 0.0);
        close(s::beta(1.0, a), 1.0 / a, rtol, 0.0);
        close(s::betaln(a, 1.0), -a.ln(), 3e-12, 2e-13);
    }
    for a in [3.0, 200.0, 1e7, 1e9] {
        let expected = (1.0 / a) / (a + 1.0);
        close(s::beta(a, 2.0), expected, 3e-12, 0.0);
        close(s::beta(2.0, a), expected, 3e-12, 0.0);
        close(s::betaln(a, 2.0), expected.ln(), 3e-12, 2e-13);
    }
    // The gamma-ratio expansion: B(a,1/2) =
    // sqrt(pi/a) (1 + 1/(8a) + O(a^-2)).
    for a in [1e8, 1e10] {
        let expected = (PI / a).sqrt() * (1.0 + 1.0 / (8.0 * a));
        close(s::beta(a, 0.5), expected, 3e-12, 0.0);
        close(s::betaln(a, 0.5), expected.ln(), 3e-12, 1e-12);
    }

    // beta(huge,2) underflows whereas log-beta remains a finite log.
    let huge = 1e308;
    close(s::betaln(huge, 2.0), -2.0 * huge.ln(), 3e-12, 3e-12);
    assert_eq!(s::beta(huge, 2.0), 0.0);

    // The public contract explicitly rejects poles and all nonfinite inputs.
    for (a, b) in [
        (0.0, 2.0),
        (-1.0, 2.0),
        (2.0, -3.0),
        (f64::NAN, 2.0),
        (2.0, f64::NAN),
        (f64::INFINITY, 2.0),
        (2.0, f64::NEG_INFINITY),
    ] {
        assert!(s::beta(a, b).is_nan(), "beta({a},{b})");
        assert!(s::betaln(a, b).is_nan(), "betaln({a},{b})");
    }
}

#[test]
fn riemann_zeta_even_and_negative_integer_closed_forms() {
    // Euler's evaluations ζ(2n) and Bernoulli polynomial values ζ(-n).
    for (x, zeta) in [
        (0.0, -0.5),
        (-1.0, -1.0 / 12.0),
        (-2.0, 0.0),
        (-3.0, 1.0 / 120.0),
        (-4.0, 0.0),
        (-5.0, -1.0 / 252.0),
        (-7.0, 1.0 / 240.0),
        (-9.0, -1.0 / 132.0),
        (-11.0, 691.0 / 32760.0),
        (2.0, PI * PI / 6.0),
        (4.0, PI.powi(4) / 90.0),
        (6.0, PI.powi(6) / 945.0),
        (8.0, PI.powi(8) / 9450.0),
    ] {
        close(s::zeta(x, 1.0), zeta, 6e-13, 3e-14);
        close(s::zetac(x), zeta - 1.0, 6e-13, 3e-14);
    }
    // Trivial zeros of the reflection sine factor, including a large |s|.
    for x in [-20.0, -100.0, -200.0] {
        assert_eq!(s::zeta(x, 1.0), 0.0);
        assert_eq!(s::zetac(x), -1.0);
    }
    assert_eq!(s::zeta(1.0_f64, 1.0), f64::INFINITY);
    assert_eq!(s::zetac(1.0_f64), f64::INFINITY);
    assert_eq!(s::zetac(f64::INFINITY), 0.0);
    assert_eq!(s::zeta(f64::INFINITY, 1.0), 1.0);
    for x in [f64::NAN, f64::NEG_INFINITY] {
        assert!(s::zeta(x, 1.0).is_nan());
        assert!(s::zetac(x).is_nan());
    }
}

#[test]
fn zeta_near_zero_negative_transition_matches_high_precision_references() {
    // 85-digit mpmath ζ(s) evaluated at the *exact binary64 inputs* below.
    // The interval (-0.01, 0) uses a separate Taylor approximation because
    // numerical evaluation of the reflection formula is ill-conditioned.
    for (s_value, expected) in [
        (-1e-8, -0.499_999_990_810_614_75),
        (-0.005, -0.495_430_262_313_341_3),
        (-0.009999, -0.490_910_840_777_564_97),
        (-0.01, -0.490_909_941_605_337_13),
        (-0.010001, -0.490_909_042_435_056_8),
        (-0.125, -0.399_069_668_945_045_03),
        (-1.5, -0.025_485_201_889_833_036),
        (-31.5, 750_878_449.993_701),
        (-50.5, 2.399_094_238_135_732_2e24),
    ] {
        close(s::zeta(s_value, 1.0), expected, 6e-12, 4e-14);
        close(s::zetac(s_value), expected - 1.0, 6e-12, 4e-14);
    }
    // ζ'(0)=-ln(2π)/2, independently constraining the Taylor branch's slope.
    let epsilon = 1e-8;
    let estimate = (s::zeta(-epsilon, 1.0) - s::zeta(0.0, 1.0)) / -epsilon;
    close(estimate, -(2.0 * PI).ln() / 2.0, 4e-7, 0.0);
}

#[test]
fn zeta_reflection_overflow_factoring_preserves_a_finite_answer() {
    // Around negative even integers, sin(πs/2) is tiny. The unfactored
    // reflection intermediate overflows for |s| > ~258 even while ζ(s) is
    // representable. These are independent 85-digit mpmath results for the
    // exact binary64 inputs. The sine's relative condition number near its
    // zero justifies a looser tolerance than elsewhere.
    for (x, expected) in [
        (-258.000001, 3.338_208_446_571_125_3e299),
        (-258.00001, 3.338_320_121_969_053e300),
        (-260.0000001, -5.694_105_025_215_884e301),
    ] {
        close(s::zeta(x, 1.0), expected, 5e-7, 0.0);
        close(s::zetac(x), expected, 5e-7, 0.0);
    }
    // ζ(-259) = -B_260/260 = 8.7601563446229215...e306:
    // a finite Bernoulli value (checked with 100-digit mpmath).
    // The next odd negative integer does overflow: ζ(-261) =
    // -B_262/262 = -1.505795433605468...e310.
    close(s::zetac(-259.0), 8.760_156_344_622_921e306, 2e-12, 0.0);
    close(s::zeta(-259.0, 1.0), 8.760_156_344_622_921e306, 2e-12, 0.0);
    assert_eq!(s::zetac(-261.0), f64::NEG_INFINITY);
}

#[test]
fn zetac_positive_interval_boundaries_match_independent_series() {
    for (x, expected) in [
        (1.5, 1.612_375_348_685_488_4),
        (3.5, 0.126_733_867_317_056_66),
        (5.5, 0.025_204_579_954_685_693),
        (10.5, 0.000_700_842_641_736_155_2),
        (50.5, 6.280_369_842_777_336e-16),
        (75.5, 1.871_696_065_287_417e-23),
    ] {
        // These constants were independently computed to 85 digits using
        // the Dirichlet/Euler-Maclaurin definition of ζ(x)-1.
        close(s::zetac(x), expected, 6e-12, 0.0);
        if x <= 10.5 {
            close(s::zeta(x, 1.0), 1.0 + expected, 6e-12, 0.0);
        }
    }
    // Beyond the tabulated range but below the deliberate underflow cutoff,
    // 2^-x dominates the positive Dirichlet series; for x=126 the omitted
    // 3^-x term is < 1e-22 times the leading term.
    close(s::zetac(126.0_f64), 2.0_f64.powi(-126), 6e-12, 0.0);
    assert_eq!(s::zetac(127.0_f64), 0.0);
    assert_eq!(s::zetac(128.0_f64), 0.0);
}

#[test]
fn hurwitz_zeta_half_shift_and_negative_half_shift_closed_forms() {
    // ζ(s,1/2)=(2^s-1)ζ(s), plus ζ(s,q+1)=ζ(s,q)-q^-s.
    // Even orders avoid introducing another implementation of Riemann ζ.
    for (order, riemann) in [
        (2.0, PI.powi(2) / 6.0),
        (4.0, PI.powi(4) / 90.0),
        (6.0, PI.powi(6) / 945.0),
    ] {
        let half = (2.0_f64.powf(order) - 1.0) * riemann;
        close(s::zeta(order, 0.5), half, 6e-12, 0.0);
        close(s::zeta(order, 1.5), half - 0.5_f64.powf(-order), 6e-12, 0.0);
        close(s::zeta(order, 2.0), riemann - 1.0, 6e-12, 0.0);
        close(
            s::zeta(order, -0.5),
            half + (-0.5_f64).powf(-order),
            6e-12,
            0.0,
        );
    }
    // q>>1 selects the Hurwitz Euler-Maclaurin limiting expansion.
    let q = 1e10;
    close(s::zeta(2.0_f64, q), 1.0 / q + 0.5 / (q * q), 3e-12, 0.0);
    assert!(s::zeta(2.5_f64, -0.5).is_nan()); // noninteger powers of negative q
    assert!(s::zeta(0.0_f64, 0.5).is_nan()); // q!=1: this real API requires s>1
    assert_eq!(s::zeta(1.0_f64, 2.0), f64::INFINITY);
}
