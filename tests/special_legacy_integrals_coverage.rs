//! Public API regression tests for the integral and Struve boundary kernels.
//! The comparisons below are mathematical identities rather than values
//! produced by another path through the implementation under test.

use sci::special as s;
use std::f64::consts::PI;

fn close(actual: f64, expected: f64, atol: f64, rtol: f64) {
    assert!(
        actual.is_finite()
            && expected.is_finite()
            && (actual - expected).abs() <= atol + rtol * expected.abs(),
        "actual={actual:.17e}, expected={expected:.17e}, atol={atol:e}, rtol={rtol:e}"
    );
}

#[test]
fn airy_integrals_preserve_reflection_and_reject_nonfinite_arguments() {
    for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(s::itairy(x).iter().all(|v| v.is_nan()));
    }
    for x in [0.0, -0.0] {
        assert!(s::itairy(x).iter().all(|v| *v == 0.0));
    }

    // Reflection swaps positive- and negative-argument Ai/Bi integrals,
    // including each side of the series/Chebyshev/quadrature/asymptotic cuts.
    for x in [
        0.125,
        3.9,
        4.0001,
        8.5,
        9.150025219244185,
        10.0001,
        15.0,
        16.0001,
        20.0,
    ] {
        let [ai, bi, ai_negative, bi_negative] = s::itairy(x);
        let reflected = s::itairy(-x);
        for (actual, expected) in reflected
            .into_iter()
            .zip([-ai_negative, -bi_negative, -ai, -bi])
        {
            close(actual, expected, 2e-13, 2e-13);
        }
    }

    // Ai is positive on the positive real axis and its integral tends to 1/3.
    // This catches the severe cancellation formerly seen around x = 9.15.
    let ai_at_nine = s::itairy(9.150025219244185)[0];
    close(ai_at_nine, 0.33333333282847294, 3e-14, 2e-13);
}

#[test]
fn airy_integrals_differentiate_to_both_airy_branches() {
    // d/dx ∫_0^x Ai(±t)dt = Ai(±x), analogously for Bi.
    // Points exercise the series, Chebyshev, quadrature and asymptotic paths;
    // none sits on a switch, where central differences mix algorithms.
    for x in [0.75, 3.0, 5.5, 8.3, 11.5, 14.0, 19.0] {
        let h = 0.0002;
        let plus = s::itairy(x + h);
        let minus = s::itairy(x - h);
        let pos = s::airy(x);
        let neg = s::airy(-x);
        let expected = [pos[0], pos[2], neg[0], neg[2]];
        for (i, target) in expected.into_iter().enumerate() {
            let numerical_derivative = (plus[i] - minus[i]) / (2.0 * h);
            // The central-difference truncation is O(h²), including the
            // rapidly increasing Bi contribution at positive x.
            close(numerical_derivative, target, 3e-4, 3e-4);
        }
    }
}

#[test]
fn integrated_struve_is_even_and_its_derivative_is_h_zero() {
    for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(s::itstruve0(x).is_nan());
        assert!(s::itmodstruve0(x).is_nan());
        assert!(s::it2struve0(x).is_nan());
    }
    assert_eq!(s::itstruve0(0.0), 0.0);
    let small = 1e-4;
    // H_0(t) = 2t/π + O(t³); hence ∫H_0 = x²/π + O(x⁴).
    close(s::itstruve0(small), small * small / PI, 0.0, 2e-9);

    // Covers binary64 series, Chebyshev interpolation, composite quadrature,
    // and large-argument asymptotics with a true evenness property.
    for x in [
        0.5, 9.5, 10.0001, 20.0, 23.9999, 24.0001, 60.0, 127.0, 129.0, 200.0,
    ] {
        let positive = s::itstruve0(x);
        close(s::itstruve0(-x), positive, 0.0, 0.0);
    }

    for x in [0.5, 8.0, 12.0, 20.0, 32.0] {
        let h = 0.0002;
        let derivative = (s::itstruve0(x + h) - s::itstruve0(x - h)) / (2.0 * h);
        close(derivative, s::struve(0.0_f64, x), 1e-4, 1e-4);
    }
}

#[test]
fn struve_and_modified_struve_preserve_real_domain_parity_and_half_orders() {
    for x in [0.5, 3.0, 12.0] {
        // H_n(-x) = (-1)^(n+1) H_n(x), likewise for L_n.
        for order in [0, 1, 2] {
            let v = f64::from(order);
            let parity = if order % 2 == 0 { -1.0 } else { 1.0 };
            close(s::struve(v, -x), parity * s::struve(v, x), 1e-12, 1e-12);
            close(
                s::modstruve(v, -x),
                parity * s::modstruve(v, x),
                1e-12,
                1e-12,
            );
        }
    }

    // Exact half-integer identities connect the Struve and Bessel families.
    for x in [0.75, 2.25, 7.0] {
        close(s::struve(-0.5_f64, x), s::jv(0.5, x), 2e-12, 2e-12);
        close(s::struve(-1.5_f64, x), -s::jv(1.5, x), 2e-12, 2e-12);
        close(s::modstruve(-0.5_f64, x), s::iv(0.5, x), 2e-12, 2e-12);
        close(s::modstruve(-1.5_f64, x), s::iv(1.5, x), 2e-12, 2e-12);
    }

    for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(s::struve(0.0_f64, x).is_nan());
        assert!(s::modstruve(0.0_f64, x).is_nan());
    }
    for v in [f64::NAN, f64::INFINITY, 1_000_001.0, 0.5] {
        assert!(s::struve(v, -2.0_f64).is_nan());
        assert!(s::modstruve(v, -2.0_f64).is_nan());
    }
    let functions: [fn(f64, f64) -> f64; 2] = [s::struve::<f64>, s::modstruve::<f64>];
    for f in functions {
        close(f(-1.0, 0.0), 2.0 / PI, 1e-15, 1e-15);
        assert_eq!(f(-2.0, 0.0), f64::NEG_INFINITY);
        assert_eq!(f(0.0, 0.0), 0.0);
    }
}

#[test]
fn other_bessel_and_struve_integrals_respect_signs_and_antiderivatives() {
    for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for (a, b) in [s::itj0y0(x), s::it2j0y0(x), s::iti0k0(x), s::it2i0k0(x)] {
            assert!(a.is_nan() && b.is_nan());
        }
    }
    let odd_integrals: [fn(f64) -> (f64, f64); 2] = [s::itj0y0, s::iti0k0];
    let even_integrals: [fn(f64) -> (f64, f64); 2] = [s::it2j0y0, s::it2i0k0];
    for x in [1.0, 5.0] {
        for f in odd_integrals {
            let positive = f(x);
            let negative = f(-x);
            close(negative.0, -positive.0, 0.0, 0.0);
            assert!(negative.1.is_nan());
        }
        for f in even_integrals {
            let positive = f(x);
            let negative = f(-x);
            close(negative.0, positive.0, 0.0, 0.0);
            assert!(negative.1.is_nan());
        }
    }

    // Fundamental-theorem checks for the two components of it2i0k0.
    // The chosen points cross the composite and 16/12/8-node transitions.
    for x in [4.5, 6.5, 9.5, 16.0] {
        let h = 0.001;
        let plus = s::it2i0k0(x + h);
        let minus = s::it2i0k0(x - h);
        close(
            (plus.0 - minus.0) / (2.0 * h),
            (s::i0(x) - 1.0) / x,
            2e-5,
            2e-4,
        );
        close((plus.1 - minus.1) / (2.0 * h), -s::k0(x) / x, 1e-12, 2e-4);
    }
    // The I0-integral uses a separate bounded fallback near overflow.
    // Laplace's endpoint expansion of the defining integrals gives
    //   ∫_0^x (I0(t)-1)/t dt ~ (I0(x)/x) [1 + 3/(2x) + O(x^-2)],
    //   ∫_x^∞ K0(t)/t dt    ~ (K0(x)/x) [1 - 3/(2x) + O(x^-2)].
    // Ratios keep both comparisons in a numerically safe range at x=700
    // despite an enormous I0 integral and an almost-underflowing K0 tail.
    let (integral, tail) = s::it2i0k0(700.0);
    close(
        integral / s::i0(700.0) * 700.0,
        1.0 + 1.5 / 700.0,
        0.0,
        2e-4,
    );
    close(tail / s::k0(700.0) * 700.0, 1.0 - 1.5 / 700.0, 0.0, 2e-4);

    for x in [0.25, 2.0, 8.0] {
        let h = 0.0002;
        close(s::itmodstruve0(-x), s::itmodstruve0(x), 0.0, 0.0);
        let derivative = (s::itmodstruve0(x + h) - s::itmodstruve0(x - h)) / (2.0 * h);
        close(derivative, s::modstruve(0.0_f64, x), 5e-5, 3e-4);
    }
    for x in [0.25, 2.0, 8.0] {
        let h = 0.0002;
        close(s::it2struve0(-x) + s::it2struve0(x), PI, 2e-12, 2e-12);
        let derivative = (s::it2struve0(x + h) - s::it2struve0(x - h)) / (2.0 * h);
        close(derivative, -s::struve(0.0_f64, x) / x, 5e-5, 3e-4);
    }
}
