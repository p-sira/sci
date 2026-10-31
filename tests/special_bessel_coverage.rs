//! Independent Bessel-family identities and boundaries for the pure-Rust backend.
//! In particular, the large-order J_v checks exercise the uniform Airy
//! transition as well as the regular oscillatory/asymptotic branches.
use sci::special as s;
use std::f64::consts::{FRAC_2_PI, PI};

fn close(actual: f64, expected: f64, rtol: f64, atol: f64) {
    assert!(
        actual.is_finite()
            && expected.is_finite()
            && (actual - expected).abs() <= atol.max(rtol * expected.abs()),
        "actual={actual:.17e}, expected={expected:.17e}, rtol={rtol}, atol={atol}"
    );
}

#[test]
fn cylindrical_j_half_integer_values_and_integer_reflections() {
    // J_{+/-1/2} and J_{3/2} have elementary trigonometric forms.
    // The larger arguments exercise the Hankel branch rather than the series.
    for x in [0.2_f64, 2.5, 14.0, 38.0, 120.0] {
        let prefactor = (2.0 / (PI * x)).sqrt();
        close(s::bessel::jv(0.5, x), prefactor * x.sin(), 2e-11, 2e-14);
        close(s::bessel::jv(-0.5, x), prefactor * x.cos(), 2e-11, 2e-14);
        close(
            s::bessel::jv(1.5, x),
            prefactor * (x.sin() / x - x.cos()),
            2e-11,
            2e-14,
        );
        close(s::bessel::yv(0.5, x), -prefactor * x.cos(), 2e-11, 2e-14);
        close(s::bessel::yv(-0.5, x), prefactor * x.sin(), 2e-11, 2e-14);
    }
    for n in [0_i32, 1, 2, 3, 12, 23] {
        let parity = if n % 2 == 0 { 1.0 } else { -1.0 };
        for x in [0.7_f64, 5.0, 35.0] {
            let value = s::bessel::jv(n as f64, x);
            close(s::bessel::jv(-(n as f64), x), parity * value, 2e-11, 1e-14);
            close(s::bessel::jv(n as f64, -x), parity * value, 2e-11, 1e-14);
            close(s::bessel::jve(n as f64, x), value, 2e-14, 1e-14);
        }
    }
}

#[test]
fn cylindrical_j_contiguous_relations_across_order_and_argument_regimes() {
    // J_{v-1}(x) + J_{v+1}(x) = (2v/x) J_v(x).
    // Small x with v large uses scaled seeds; the oscillatory arguments
    // exercise the continued-fraction and Hankel paths.
    for (v, x) in [
        (2.5_f64, 0.75_f64),
        (9.25, 6.0),
        (24.5, 12.0),
        (45.25, 30.0),
        (2.25, 80.0),
        (75.25, 110.0),
    ] {
        let value = s::bessel::jv(v, x);
        let contiguous = x * (s::bessel::jv(v - 1.0, x) + s::bessel::jv(v + 1.0, x)) / (2.0 * v);
        close(value, contiguous, 2e-9, 2e-13);
    }
    // Independent SciPy-generated values from the previously checked fixture,
    // covering the high-order, short-backward-recurrence scaling case.
    close(
        s::bessel::jv(499.0, 90.0),
        6.302_694_314_321_84e-309,
        3e-9,
        1e-321,
    );
    close(
        s::bessel::jv(600.0, 550.0),
        1.835_655_281_295_801_3e-8,
        3e-9,
        1e-14,
    );
    close(
        s::bessel::jv(600.0, 1000.0),
        -0.016_761_874_430_871_49,
        3e-9,
        1e-13,
    );
}

#[test]
fn cylindrical_j_large_order_turning_point_and_domain_edges() {
    // |x-n|/cbrt(n) <= 0.7 selects the J_n(n+O(n^(1/3)))
    // Airy-transition expansion, missing from the previous coverage.
    for x in [595.0_f64, 600.0, 605.0] {
        let j = s::bessel::jv(600.0, x);
        // The adjacent orders straddle the turning point. Their contiguous
        // recurrence is independent of the one-order Airy approximation.
        close(
            j,
            x * (s::bessel::jv(599.0, x) + s::bessel::jv(601.0, x)) / 1200.0,
            2e-5,
            5e-10,
        );
    }
    // Below and above the turning region the large-order uniform expansion
    // uses different signs of its Airy variable.
    for x in [450.0_f64, 550.0, 700.0, 900.0] {
        let j = s::bessel::jv(600.0, x);
        close(
            j,
            x * (s::bessel::jv(599.0, x) + s::bessel::jv(601.0, x)) / 1200.0,
            2e-5,
            1e-300,
        );
    }
    // For x/n^2 > 0.3 the kernel selects the large-argument Hankel expansion.
    // W[J_v,Y_v] = J_(v+1) Y_v - J_v Y_(v+1) = 2/(pi*x).
    // Unlike a simple boundedness check, this tests the amplitude and phase
    // against the independent second-kind Bessel implementation.
    let x = 200_000.0_f64;
    let j600 = s::bessel::jv(600.0, x);
    let j601 = s::bessel::jv(601.0, x);
    let y600 = s::bessel::yv(600.0, x);
    let y601 = s::bessel::yv(601.0, x);
    close(j601 * y600 - j600 * y601, 2.0 / (PI * x), 2e-6, 1e-12);
    close(
        j600,
        x * (s::bessel::jv(599.0, x) + j601) / 1200.0,
        2e-6,
        1e-12,
    );
    assert!(s::bessel::jv(-600.5_f64, 550.0).is_nan());
    assert!(s::bessel::jv(0.5_f64, -2.0).is_nan());
    assert!(s::bessel::jv(f64::INFINITY, 1.0).is_nan());
    assert!(s::bessel::jv(1.0, f64::NAN).is_nan());
    assert!(s::bessel::jv(-0.5_f64, 0.0).is_infinite());
    assert!(s::bessel::jv(-1.5_f64, 0.0).is_infinite());
    assert_eq!(s::bessel::jv(4.0, 0.0), 0.0);
}

#[test]
fn spherical_four_kinds_match_elementary_seeds_and_recurrences() {
    for x in [0.03_f64, 0.8, 2.5, 8.0, 30.0] {
        let (sin, cos) = x.sin_cos();
        close(s::bessel::spherical_jn(0, x, false), sin / x, 2e-12, 1e-14);
        close(s::bessel::spherical_yn(0, x, false), -cos / x, 2e-12, 1e-14);
        close(
            s::bessel::spherical_in(0, x, false),
            x.sinh() / x,
            2e-12,
            1e-14,
        );
        close(
            s::bessel::spherical_kn(0, x, false),
            PI * (-x).exp() / (2.0 * x),
            2e-12,
            1e-14,
        );
        close(
            s::bessel::spherical_jn(0, x, true),
            -s::bessel::spherical_jn(1, x, false),
            2e-11,
            1e-13,
        );
        close(
            s::bessel::spherical_yn(0, x, true),
            -s::bessel::spherical_yn(1, x, false),
            2e-11,
            1e-13,
        );
        close(
            s::bessel::spherical_in(0, x, true),
            s::bessel::spherical_in(1, x, false),
            2e-11,
            1e-13,
        );
        close(
            s::bessel::spherical_kn(0, x, true),
            -s::bessel::spherical_kn(1, x, false),
            2e-11,
            1e-13,
        );
        for n in 1_i64..=5 {
            let k = (2 * n + 1) as f64 / x;
            let j = s::bessel::spherical_jn(n, x, false);
            let y = s::bessel::spherical_yn(n, x, false);
            let i = s::bessel::spherical_in(n, x, false);
            let modified_k = s::bessel::spherical_kn(n, x, false);
            close(
                s::bessel::spherical_jn(n - 1, x, false) + s::bessel::spherical_jn(n + 1, x, false),
                k * j,
                2e-8,
                1e-11,
            );
            close(
                s::bessel::spherical_yn(n - 1, x, false) + s::bessel::spherical_yn(n + 1, x, false),
                k * y,
                2e-8,
                1e-10,
            );
            close(
                s::bessel::spherical_in(n - 1, x, false) - s::bessel::spherical_in(n + 1, x, false),
                k * i,
                2e-8,
                1e-11,
            );
            close(
                s::bessel::spherical_kn(n + 1, x, false) - s::bessel::spherical_kn(n - 1, x, false),
                k * modified_k,
                2e-8,
                1e-11,
            );
            close(
                s::bessel::spherical_jn(n, x, true),
                s::bessel::spherical_jn(n - 1, x, false) - (n + 1) as f64 * j / x,
                2e-8,
                1e-11,
            );
            close(
                s::bessel::spherical_yn(n, x, true),
                s::bessel::spherical_yn(n - 1, x, false) - (n + 1) as f64 * y / x,
                2e-8,
                1e-10,
            );
            close(
                s::bessel::spherical_in(n, x, true),
                s::bessel::spherical_in(n - 1, x, false) - (n + 1) as f64 * i / x,
                2e-8,
                1e-11,
            );
            close(
                s::bessel::spherical_kn(n, x, true),
                -s::bessel::spherical_kn(n - 1, x, false) - (n + 1) as f64 * modified_k / x,
                2e-8,
                1e-11,
            );
        }
    }
}

#[test]
fn spherical_reflections_zero_poles_and_large_order() {
    for n in 0_i64..=4 {
        let parity = if n % 2 == 0 { 1.0 } else { -1.0 };
        for x in [0.6_f64, 2.0, 5.0] {
            close(
                s::bessel::spherical_jn(n, -x, false),
                parity * s::bessel::spherical_jn(n, x, false),
                2e-12,
                1e-13,
            );
            close(
                s::bessel::spherical_in(n, -x, false),
                parity * s::bessel::spherical_in(n, x, false),
                2e-12,
                1e-13,
            );
            close(
                s::bessel::spherical_yn(n, -x, false),
                -parity * s::bessel::spherical_yn(n, x, false),
                2e-12,
                1e-13,
            );
            // The real-domain continuation of modified k to negative x.
            close(
                s::bessel::spherical_kn(n, -x, false),
                -PI * s::bessel::spherical_in(n, x, false)
                    - parity * s::bessel::spherical_kn(n, x, false),
                2e-12,
                1e-12,
            );
        }
    }
    assert_eq!(s::bessel::spherical_kn(1, -1.0, false), 0.0);
    for fun in [
        s::bessel::spherical_jn as fn(i64, f64, bool) -> f64,
        s::bessel::spherical_yn,
        s::bessel::spherical_in,
        s::bessel::spherical_kn,
    ] {
        assert!(fun(-1, 1.0, false).is_nan());
        assert!(fun(4097, 1.0, false).is_nan());
        assert!(fun(2, f64::NAN, false).is_nan());
    }
    assert_eq!(s::bessel::spherical_jn(0, 0.0, false), 1.0);
    assert_eq!(s::bessel::spherical_jn(2, 0.0, false), 0.0);
    assert_eq!(s::bessel::spherical_in(0, 0.0, false), 1.0);
    assert_eq!(s::bessel::spherical_in(2, 0.0, false), 0.0);
    close(s::bessel::spherical_jn(1, 0.0, true), 1.0 / 3.0, 0.0, 0.0);
    close(s::bessel::spherical_in(1, 0.0, true), 1.0 / 3.0, 0.0, 0.0);
    assert!(s::bessel::spherical_yn(2, 0.0_f64, false).is_infinite());
    assert!(s::bessel::spherical_kn(2, 0.0_f64, false).is_infinite());
    assert_eq!(s::bessel::spherical_jn(300, 1e-6, false), 0.0);
    assert_eq!(s::bessel::spherical_jn(300, 1e-6, true), 0.0);
    // Above the bounded Miller recurrence, use the half-integer Bessel fallback.
    // j_n(x) = sqrt(pi/(2*x)) J_(n+1/2)(x). This compares the fallback
    // (complex AMOS) with the independent real Cephes J_v implementation.
    let x = 240.0_f64;
    let expected = (PI / (2.0 * x)).sqrt() * s::bessel::jv(260.5, x);
    close(
        s::bessel::spherical_jn(260, x, false),
        expected,
        3e-7,
        2e-14,
    );
}

#[test]
fn modified_cylindrical_bessel_half_orders_and_domain_boundaries() {
    for x in [0.2_f64, 2.0, 16.0, 40.0] {
        let prefactor = (2.0 / (PI * x)).sqrt();
        let i_pos = prefactor * x.sinh();
        let i_neg = prefactor * x.cosh();
        let k = (PI / (2.0 * x)).sqrt() * (-x).exp();
        close(s::bessel::iv(0.5, x), i_pos, 3e-11, 2e-14);
        close(s::bessel::iv(-0.5, x), i_neg, 3e-11, 2e-14);
        close(s::bessel::ive(0.5, x), (-x).exp() * i_pos, 3e-11, 2e-14);
        close(s::bessel::kv(0.5, x), k, 3e-11, 2e-14);
        close(s::bessel::kv(-0.5, x), k, 3e-11, 2e-14);
        close(
            s::bessel::kve(0.5, x),
            (PI / (2.0 * x)).sqrt(),
            3e-11,
            2e-14,
        );
        for n in [0_i32, 1, 2, 3] {
            let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
            close(
                s::bessel::iv(n as f64, -x),
                sign * s::bessel::iv(n as f64, x),
                3e-11,
                1e-14,
            );
        }
    }
    assert!(s::bessel::iv(0.5_f64, -2.0).is_nan());
    assert!(s::bessel::iv(-0.5_f64, 0.0).is_infinite());
    assert_eq!(s::bessel::iv(0.0, 0.0), 1.0);
    assert_eq!(s::bessel::iv(-3.0, 0.0), 0.0);
    assert!(s::bessel::kv(0.0_f64, 0.0).is_infinite());
    assert!(s::bessel::kv(1.0_f64, -1.0).is_nan());
    assert!(s::bessel::yv(1.5_f64, -1.0).is_nan());
}

#[test]
fn struve_half_integer_reductions_reflections_and_derivatives() {
    // H_{-1/2}=J_{1/2}, H_{-3/2}=-J_{3/2}, and L_{-k-1/2}=I_{k+1/2}.
    for x in [0.3_f64, 2.0, 7.0, 16.0] {
        close(
            s::bessel::struve(-0.5, x),
            s::bessel::jv(0.5, x),
            3e-11,
            2e-13,
        );
        close(
            s::bessel::struve(-1.5, x),
            -s::bessel::jv(1.5, x),
            3e-11,
            2e-13,
        );
        close(
            s::bessel::modstruve(-0.5, x),
            s::bessel::iv(0.5, x),
            3e-11,
            2e-13,
        );
        close(
            s::bessel::modstruve(-1.5, x),
            s::bessel::iv(1.5, x),
            3e-11,
            2e-13,
        );
    }
    for x in [0.7_f64, 4.0, 12.0] {
        for n in [0_i32, 1, 2, 3] {
            let parity = if n % 2 == 0 { -1.0 } else { 1.0 };
            close(
                s::bessel::struve(n as f64, -x),
                parity * s::bessel::struve(n as f64, x),
                3e-11,
                2e-13,
            );
            close(
                s::bessel::modstruve(n as f64, -x),
                parity * s::bessel::modstruve(n as f64, x),
                3e-11,
                2e-13,
            );
        }
        // dH_0/dx = H_{-1} and dL_0/dx = L_{-1}.
        let h = 1e-4;
        let dh = (s::bessel::struve(0.0, x + h) - s::bessel::struve(0.0, x - h)) / (2.0 * h);
        let dl = (s::bessel::modstruve(0.0, x + h) - s::bessel::modstruve(0.0, x - h)) / (2.0 * h);
        close(dh, s::bessel::struve(-1.0, x), 2e-6, 5e-9);
        close(dl, s::bessel::modstruve(-1.0, x), 2e-6, 5e-9);
    }
    close(s::bessel::struve(-1.0, 0.0), FRAC_2_PI, 0.0, 0.0);
    close(s::bessel::modstruve(-1.0, 0.0), FRAC_2_PI, 0.0, 0.0);
    // Independently tabulated large-argument values; these reach the
    // asymptotic/Bessel-based candidates rather than the small-x series.
    close(
        s::bessel::struve(0.0_f64, 50.0),
        -0.085_337_674_826_119_02,
        2e-10,
        1e-12,
    );
    close(
        s::bessel::modstruve(0.0_f64, 50.0),
        2.932_553_783_849_337e20,
        2e-10,
        1e-12,
    );
    assert_eq!(s::bessel::struve(0.0, 0.0), 0.0);
    assert_eq!(s::bessel::modstruve(0.0, 0.0), 0.0);
    assert!(s::bessel::struve(-1.25_f64, 0.0).is_infinite());
    assert!(s::bessel::modstruve(-1.25_f64, 0.0).is_infinite());
    assert!(s::bessel::struve(0.5_f64, -1.0).is_nan());
    assert!(s::bessel::modstruve(0.5_f64, -1.0).is_nan());
    assert!(s::bessel::struve(0.0, f64::INFINITY).is_nan());
    assert!(s::bessel::modstruve(1_000_001.0_f64, 2.0).is_nan());
}
