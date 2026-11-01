mod common;

use common::finite_close;
use sci::special::{self as s, elliptic};
use std::f64::consts::{FRAC_PI_2, PI};

fn jacobi_relations((sn, cn, dn, am): (f64, f64, f64, f64), m: f64, tol: f64) {
    // Jacobi's algebraic identities and its defining amplitude.
    finite_close(sn * sn + cn * cn, 1.0, tol, tol);
    finite_close(dn * dn + m * sn * sn, 1.0, tol, tol);
    finite_close(sn, am.sin(), tol, tol);
    finite_close(cn, am.cos(), tol, tol);
}

#[test]
fn jacobi_zero_parameter_reduces_to_trigonometry() {
    // Exact m=0 formulas remain valid through many ordinary trigonometric
    // periods. The amplitude is unwrapped: am(u|0) = u, not arcsin(sin u).
    for u in [-8.0 * PI, -7.3, -PI, -0.25, 0.0, 0.25, PI, 7.3, 8.0 * PI] {
        let actual = s::elliptic::ellipj(u, 0.0);
        finite_close(actual.0, u.sin(), 1e-14, 1e-14);
        finite_close(actual.1, u.cos(), 1e-14, 1e-14);
        finite_close(actual.2, 1.0, 1e-14, 1e-14);
        finite_close(actual.3, u, 1e-14, 1e-14);
        jacobi_relations(actual, 0.0, 2e-14);
    }

    // For m << 1, the independently derived first-order perturbation is
    // am(u|m) = u - m*(u - sin(u)cos(u))/4 + O(m²).
    for m in [1e-12, 5e-10] {
        for u in [-4.1_f64, -0.8, 0.0, 0.8, 4.1] {
            let actual = s::elliptic::ellipj(u, m);
            let first_order_am = u - 0.25 * m * (u - u.sin() * u.cos());
            finite_close(actual.3, first_order_am, 2e-14, 2e-14);
            jacobi_relations(actual, m, 2e-13);
        }
    }
}

#[test]
fn jacobi_unit_parameter_reduces_to_hyperbolic_functions() {
    // sn(u|1)=tanh(u), cn(u|1)=dn(u|1)=sech(u), and
    // am(u|1)=atan(sinh(u)). Avoid u where cosh/exp overflow.
    for u in [-8.0_f64, -3.0, -0.5, 0.0, 0.5, 3.0, 8.0] {
        let actual = s::elliptic::ellipj(u, 1.0);
        let sech = 1.0 / u.cosh();
        finite_close(actual.0, u.tanh(), 3e-14, 3e-14);
        finite_close(actual.1, sech, 3e-14, 3e-14);
        finite_close(actual.2, sech, 3e-14, 3e-14);
        finite_close(actual.3, u.sinh().atan(), 3e-14, 3e-14);
        jacobi_relations(actual, 1.0, 4e-14);
    }

    // The near-one expansion converges continuously to the exact m=1
    // hyperbolic solution for bounded u. O(1-m) changes are below 1e-8
    // here, while the algebraic Jacobi identities remain much tighter.
    for m in [1.0 - 5e-11, 1.0 - 1e-12] {
        for u in [-3.0_f64, -0.7, 0.0, 0.7, 3.0] {
            let actual = s::elliptic::ellipj(u, m);
            finite_close(actual.0, u.tanh(), 2e-8, 2e-8);
            finite_close(actual.1, 1.0 / u.cosh(), 2e-8, 2e-8);
            finite_close(actual.2, 1.0 / u.cosh(), 2e-8, 2e-8);
            finite_close(actual.3, u.sinh().atan(), 2e-8, 2e-8);
            jacobi_relations(actual, m, 1e-10);
        }
    }
}

#[test]
fn jacobi_unit_parameter_remains_finite_far_beyond_hyperbolic_overflow() {
    // The exact m=1 identities hold for all finite u, including magnitudes
    // where cosh(u)*sinh(u) overflows and exp(u) may overflow/underflow.
    for u in [
        -1_000.0_f64,
        -740.0,
        -400.0,
        -356.0,
        -350.0,
        350.0,
        356.0,
        400.0,
        740.0,
        1_000.0,
    ] {
        let (sn, cn, dn, am) = elliptic::ellipj(u, 1.0);
        let t = u.tanh();
        let e = (-u.abs()).exp();
        let sech = 2.0 * e / (1.0 + e * e);
        assert!(sn.is_finite() && cn.is_finite() && dn.is_finite() && am.is_finite());
        finite_close(sn, t, 1e-14, 1e-14);
        finite_close(cn, sech, 1e-14, 0.0);
        finite_close(dn, sech, 1e-14, 0.0);
        finite_close(am, t.atan2(sech), 1e-14, 1e-14);
        jacobi_relations((sn, cn, dn, am), 1.0, 2e-14);
    }

    // As soon as the near-m=1 perturbation ceases to be uniformly small,
    // evaluate by the AGM so that periodicity and the elliptic identities
    // are maintained at large u rather than forcing a hyperbolic limit.
    let m = 1.0 - 1e-11;
    let k = elliptic::ellipk(m).unwrap();
    let sample = elliptic::ellipj(0.4, m);
    for period in [-10_i32, -2, 2, 10] {
        let u = 0.4 + 2.0 * f64::from(period) * k;
        let actual = elliptic::ellipj(u, m);
        let sign = if period % 2 == 0 { 1.0 } else { -1.0 };
        finite_close(actual.0, sign * sample.0, 1e-9, 1e-9);
        finite_close(actual.1, sign * sample.1, 1e-9, 1e-9);
        finite_close(actual.2, sample.2, 1e-9, 1e-9);
        jacobi_relations(actual, m, 2e-10);
    }
}

#[test]
fn jacobi_amplitude_inverts_an_independent_elliptic_integral() {
    // ellipf comes from the independent Ellip implementation, not Cephes'
    // AGM Jacobi kernel. This checks the unwrapped amplitude quantitatively.
    for m in [0.2, 0.5, 0.9, 0.99] {
        for u in [-3.4_f64, -1.3, -0.2, 0.0, 0.2, 1.3, 3.4] {
            let actual = s::elliptic::ellipj(u, m);
            finite_close(elliptic::ellipf(actual.3, m).unwrap(), u, 3e-11, 3e-11);
            jacobi_relations(actual, m, 3e-12);
        }
    }

    // Differentiate sn, cn, dn and am numerically, then check the Jacobi
    // differential system. Symmetric differences have O(h²) truncation.
    let h = 1e-5;
    for m in [0.2, 0.7, 0.99] {
        for u in [-1.5_f64, 0.4, 2.6] {
            let (sn, cn, dn, _) = s::elliptic::ellipj(u, m);
            let plus = s::elliptic::ellipj(u + h, m);
            let minus = s::elliptic::ellipj(u - h, m);
            finite_close((plus.0 - minus.0) / (2.0 * h), cn * dn, 2e-9, 2e-9);
            finite_close((plus.1 - minus.1) / (2.0 * h), -sn * dn, 2e-9, 2e-9);
            finite_close((plus.2 - minus.2) / (2.0 * h), -m * sn * cn, 2e-9, 2e-9);
            finite_close((plus.3 - minus.3) / (2.0 * h), dn, 2e-9, 2e-9);
        }
    }
}

#[test]
fn jacobi_quarter_periods_have_correct_dn_and_amplitude() {
    // At odd multiples of the independently computed K(m), cn=0,
    // |sn|=1 and dn=sqrt(1-m). These are cancellation-sensitive arguments
    // that exercise the Cephes alternative dn formula near quarter periods.
    for m in [0.2, 0.5, 0.95, 0.999999] {
        let k = elliptic::ellipk(m).unwrap();
        for multiple in [-3_i32, -1, 1, 3, 5] {
            let u = f64::from(multiple) * k;
            let (sn, cn, dn, am) = s::elliptic::ellipj(u, m);
            let phase = f64::from(multiple) * FRAC_PI_2;
            finite_close(sn, phase.sin(), 2e-11, 2e-11);
            finite_close(cn, 0.0, 2e-11, 0.0);
            finite_close(dn, (1.0 - m).sqrt(), 2e-11, 2e-11);
            finite_close(am, phase, 2e-11, 2e-11);
            jacobi_relations((sn, cn, dn, am), m, 3e-11);
        }
    }
}

#[test]
fn jacobi_reflection_and_multiperiod_shifts_preserve_signs() {
    // sn and cn change sign after 2K, dn remains unchanged and the
    // unwrapped amplitude advances by pi. Check multiple periods both ways.
    for m in [0.4, 0.7, 0.95] {
        let k = elliptic::ellipk(m).unwrap();
        let base = s::elliptic::ellipj(0.37, m);
        for periods in [-4_i32, -2, -1, 0, 1, 2, 4] {
            let u = 0.37 + 2.0 * f64::from(periods) * k;
            let result = s::elliptic::ellipj(u, m);
            let sign = if periods % 2 == 0 { 1.0 } else { -1.0 };
            finite_close(result.0, sign * base.0, 2e-11, 2e-11);
            finite_close(result.1, sign * base.1, 2e-11, 2e-11);
            finite_close(result.2, base.2, 2e-11, 2e-11);
            finite_close(result.3, base.3 + f64::from(periods) * PI, 2e-11, 2e-11);
            jacobi_relations(result, m, 2e-11);
        }
        for u in [0.2_f64, 1.1, 3.0] {
            let positive = s::elliptic::ellipj(u, m);
            let negative = s::elliptic::ellipj(-u, m);
            finite_close(negative.0, -positive.0, 2e-12, 2e-12);
            finite_close(negative.1, positive.1, 2e-12, 2e-12);
            finite_close(negative.2, positive.2, 2e-12, 2e-12);
            finite_close(negative.3, -positive.3, 2e-12, 2e-12);
        }
    }
}

#[test]
fn jacobi_rejects_invalid_modulus_and_nonfinite_arguments() {
    for m in [-f64::INFINITY, -1e-3, f64::NAN, 1.0 + 1e-3, f64::INFINITY] {
        let (sn, cn, dn, am) = s::elliptic::ellipj(0.5, m);
        assert!(
            sn.is_nan() && cn.is_nan() && dn.is_nan() && am.is_nan(),
            "m={m}"
        );
    }
    for u in [f64::NAN, f64::NEG_INFINITY, f64::INFINITY] {
        for m in [0.0, 0.5, 1.0] {
            let (sn, cn, dn, am) = s::elliptic::ellipj(u, m);
            assert!(
                sn.is_nan() && cn.is_nan() && dn.is_nan() && am.is_nan(),
                "u={u}, m={m}"
            );
        }
    }
}
