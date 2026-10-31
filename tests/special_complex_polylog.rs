//! Independent SciPy 1.18.1 complex references (NumPy 2.5.4), computed with
//! `scipy.special.{spence,xlogy,xlog1py}` and stored as complex components.
//! Both branch-cut banks and cancellation-sensitive inputs are included.
// The source reference literals intentionally retain their full independently
// generated decimal representation (including representations of constants).
#![allow(clippy::excessive_precision, clippy::approx_constant)]

use sci::{complex::Complex, special as s};

fn z(re: f64, im: f64) -> Complex<f64> {
    Complex { re, im }
}

fn close(actual: Complex<f64>, expected: (f64, f64), relative: f64, absolute: f64) {
    for (actual, expected) in [(actual.re, expected.0), (actual.im, expected.1)] {
        assert!(
            actual.is_finite() && (actual - expected).abs() <= relative * expected.abs() + absolute,
            "actual={actual:.17e}, expected={expected:.17e}, rel={relative}, abs={absolute}"
        );
    }
}

#[test]
fn complex_spence_matches_scipy_on_both_sides_of_cut_and_across_reductions() {
    let cases = [
        ((0.0, 0.0), (1.64493406684822641, 0.0)),
        ((1.0, 0.0), (0.0, 0.0)),
        ((-1.0, 0.0), (2.46740110027233994, -2.17758609030360217)),
        ((-1.0, -0.0), (2.46740110027233994, 2.17758609030360217)),
        ((-0.1, 0.0), (1.96199910130556843, -0.299425760685589293)),
        ((-0.1, -0.0), (1.96199910130556843, 0.299425760685589293)),
        ((0.5, 0.5), (0.453985269150295578, -0.643767332889268573)),
        ((0.5, -0.5), (0.453985269150295578, 0.643767332889268573)),
        ((2.0, 0.0), (-0.822467033424113092, -0.0)),
        ((2.0, -0.0), (-0.822467033424113092, -0.0)),
        ((-4.0, 2.0), (0.570577168990651806, -4.72433840776538538)),
        ((0.3, 0.7), (0.562719767433781493, -0.970033357331281931)),
        (
            (-100.0, 100.0),
            (-11.1470270597106484, -11.7086750457524520),
        ),
        ((0.1, 6.0), (-1.80925369111792667, -3.26491742835579668)),
        (
            (0.5, 0.8660254037844386),
            (0.274155677808037845, -1.01494160640965347),
        ),
        (
            (0.5, -0.8660254037844386),
            (0.274155677808037845, 1.01494160640965347),
        ),
        (
            (1e-12, 1e-12),
            (1.64493406681915655, -2.74990493622789122e-11),
        ),
        ((1e8, 1e5), (-171.305682118286569, -0.0184206752879353536)),
        ((-2.0, 1e-12), (2.32018042331205177, -3.45139229522297208)),
        ((-2.0, -1e-12), (2.32018042331205177, 3.45139229522297208)),
        (
            (0.999999999999, 1e-6),
            (7.49977878279858357e-13, -1.00000000000038874e-6),
        ),
        ((30.0, 30.0), (-8.26697539192787900, -3.01072699869159344)),
        (
            (-1e-30, 0.0),
            (1.64493406684822641, -3.14159265358979339e-30),
        ),
        (
            (-1e-30, -0.0),
            (1.64493406684822641, 3.14159265358979339e-30),
        ),
    ];
    for ((re, im), expected) in cases {
        let actual = s::complex::spence(z(re, im)).unwrap();
        // The imaginary parts are compared relatively even when 1e-30.
        close(
            actual,
            expected,
            2e-13,
            if expected.1 == 0.0 { 1e-14 } else { 0.0 },
        );
    }
}

#[test]
fn complex_xlogy_matches_scipy_with_signed_zero_and_large_dynamic_range() {
    let cases = [
        (((0.0, 0.0), (0.0, 0.0)), (0.0, 0.0)),
        (((0.0, 0.0), (-2.0, -0.0)), (0.0, 0.0)),
        (
            ((1.0, 2.0), (-1.0, 0.0)),
            (-6.28318530717958623, 3.14159265358979312),
        ),
        (
            ((1.0, 2.0), (-1.0, -0.0)),
            (6.28318530717958623, -3.14159265358979312),
        ),
        (
            ((1.0, 2.0), (1e-250, 1e-250)),
            (-576.870495985026423, -1149.81400115306542),
        ),
        (
            ((1.0, 2.0), (-3.0, 4.0)),
            (-2.81915695874226113, 5.43317326045638183),
        ),
        (
            ((-2.0, 1.0), (1.0, 1e-12)),
            (-1.00000000000100003e-12, -1.99999999999949994e-12),
        ),
        (
            ((1e-30, 1e-30), (1e300, 1e300)),
            (6.90336703325096274e-28, 6.91907499651891134e-28),
        ),
        (((1.0, 1.0), (1.0, 0.0)), (0.0, 0.0)),
    ];
    for (((a, b), (c, d)), expected) in cases {
        close(
            s::complex::xlogy(z(a, b), z(c, d)).unwrap(),
            expected,
            3e-14,
            0.0,
        );
    }
}

#[test]
fn complex_xlog1py_matches_scipy_with_sub_ulp_increments() {
    let cases = [
        (((0.0, 0.0), (-1.0, 0.0)), (0.0, 0.0)),
        (((0.0, 0.0), (-1.0, -0.0)), (0.0, 0.0)),
        (((1.0, 2.0), (1e-200, 1e-200)), (-1e-200, 3e-200)),
        (
            ((1.0, 2.0), (-1.0, 1e-300)),
            (-693.917120551803464, -1379.98025946963253),
        ),
        (
            ((1.0, 2.0), (-1.5, 0.0)),
            (-6.97633248773953163, 1.75529829246990254),
        ),
        (
            ((1.0, 2.0), (-1.5, -0.0)),
            (5.59003812661964083, -4.52788701470968391),
        ),
        (
            ((1.0, 2.0), (-1e-15, 1e-15)),
            (-3.00000000000000221e-15, -9.99999999999999092e-16),
        ),
        (
            ((-2.0, 1.0), (0.4, -0.7)),
            (-0.432440415555829405, 1.37533923027992988),
        ),
        (
            ((1e-30, 1e-30), (1e300, -1e300)),
            (6.91907499651891134e-28, 6.90336703325096274e-28),
        ),
        (
            ((1.0, 1.0), (1.0, 0.0)),
            (0.693147180559945286, 0.693147180559945286),
        ),
    ];
    for (((a, b), (c, d)), expected) in cases {
        close(
            s::complex::xlog1py(z(a, b), z(c, d)).unwrap(),
            expected,
            5e-14,
            0.0,
        );
    }
}

#[test]
fn complex_polylog_domains_and_f32() {
    let zero = z(0.0, 0.0);
    assert!(s::complex::spence(z(f64::NAN, 0.0)).is_err());
    assert!(s::complex::spence(z(0.0, f64::INFINITY)).is_err());
    for f in [s::complex::xlogy, s::complex::xlog1py] {
        assert!(f(z(f64::NAN, 0.0), zero).is_err());
        assert!(f(z(1.0, 0.0), z(f64::INFINITY, 0.0)).is_err());
    }
    assert!(s::complex::xlogy(z(1.0, 2.0), zero).is_err());
    assert!(s::complex::xlog1py(z(1.0, 2.0), z(-1.0, -0.0)).is_err());
    // A zero prefactor has an exact finite value at a logarithmic pole.
    assert_eq!(s::complex::xlogy(zero, zero).unwrap(), zero);
    assert_eq!(s::complex::xlog1py(zero, z(-1.0, 0.0)).unwrap(), zero);

    let xf32 = Complex {
        re: 1.0_f32,
        im: 2.0_f32,
    };
    let cut_upper = Complex {
        re: -1.0_f32,
        im: 0.0_f32,
    };
    let cut_lower = Complex {
        re: -1.0_f32,
        im: -0.0_f32,
    };
    for (arg, expected_im) in [
        (cut_upper, -2.177586090303602),
        (cut_lower, 2.177586090303602),
    ] {
        let value = s::complex::spence(arg).unwrap();
        assert!((value.re as f64 - 2.46740110027234).abs() < 2e-7);
        assert!((value.im as f64 - expected_im).abs() < 2e-7);
    }
    let result = s::complex::xlogy(xf32, cut_lower).unwrap();
    assert!((result.re as f64 - 2.0 * std::f64::consts::PI).abs() < 5e-7);
    assert!((result.im as f64 + std::f64::consts::PI).abs() < 5e-7);
    let result = s::complex::xlog1py(
        xf32,
        Complex {
            re: 1e-20_f32,
            im: -1e-20_f32,
        },
    )
    .unwrap();
    assert!((result.re as f64 - 3e-20).abs() / 3e-20 < 2e-6);
    assert!((result.im as f64 - 1e-20).abs() / 1e-20 < 2e-6);

    // Independent log modulus identity: log|1+it| = log1p(t²)/2.
    // A direct `ln(hypot(1,t))` loses this 5e-21 component entirely.
    let t = 1e-10_f64;
    let near_one = s::complex::xlogy(z(1.0, 0.0), z(1.0, t)).unwrap();
    close(near_one, (0.5 * (t * t).ln_1p(), t.atan()), 2e-15, 0.0);

    // Both finite components can approach MAX even when their hypotenuse
    // overflows. Their complex logarithm and the dilogarithm remain finite.
    let huge = z(f64::MAX, f64::MAX);
    let log = s::complex::xlogy(z(1.0, 0.0), huge).unwrap();
    close(
        log,
        (
            f64::MAX.ln() + 0.5 * 2.0_f64.ln(),
            std::f64::consts::FRAC_PI_4,
        ),
        2e-15,
        0.0,
    );
    let value = s::complex::spence(huge).unwrap();
    let expected = z(-std::f64::consts::PI.powi(2) / 6.0, 0.0);
    let expected = sci::complex::add(
        expected,
        sci::complex::multiply(z(-0.5, 0.0), sci::complex::multiply(log, log)),
    );
    close(value, (expected.re, expected.im), 2e-15, 0.0);
}
