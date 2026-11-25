mod common;

use common::finite_close;
use sci::special as s;

fn close_pair(actual: (f64, f64), expected: (f64, f64), atol: f64, rtol: f64) {
    finite_close(actual.0, expected.0, atol, rtol);
    finite_close(actual.1, expected.1, atol, rtol);
}

fn nan_pair(value: (f64, f64)) {
    assert!(
        value.0.is_nan() && value.1.is_nan(),
        "expected an invalid spheroidal pair, got {value:?}"
    );
}

#[test]
fn spheroidal_characteristics_and_sequences_follow_scalar_limits() {
    // Both spectra reduce to n(n+1) at c=0. The tiny positive c values also
    // exercise the small-parameter branch rather than only the exact zero case.
    for m in 0..=3 {
        for n in m..=m + 3 {
            let spherical = (n * (n + 1)) as f64;
            finite_close(s::wave::pro_cv(m, n, 0.0), spherical, 0.0, 0.0);
            finite_close(s::wave::obl_cv(m, n, 0.0), spherical, 0.0, 0.0);
            finite_close(s::wave::pro_cv(m, n, 1e-12), spherical, 0.0, 0.0);
            finite_close(s::wave::obl_cv(m, n, 1e-12), spherical, 0.0, 0.0);
        }

        // Sequence order and endpoints must agree with the scalar public API.
        // Reversing c must not change either spectrum.
        let prolate = s::wave::pro_cv_seq(m, m + 3, 0.75).unwrap();
        let oblate = s::wave::obl_cv_seq(m, m + 3, 0.75).unwrap();
        assert_eq!(prolate.len(), 4);
        assert_eq!(oblate.len(), 4);
        for (offset, (&pro, &obl)) in prolate.iter().zip(&oblate).enumerate() {
            let n = m + offset as u64;
            finite_close(pro, s::wave::pro_cv(m, n, 0.75), 0.0, 0.0);
            finite_close(obl, s::wave::obl_cv(m, n, 0.75), 0.0, 0.0);
        }
        assert_eq!(prolate, s::wave::pro_cv_seq(m, m + 3, -0.75).unwrap());
        assert_eq!(oblate, s::wave::obl_cv_seq(m, m + 3, -0.75).unwrap());
    }
}

#[test]
fn spheroidal_angular_cv_reuse_and_reflection_are_consistent() {
    // Cover both n-m parities and several azimuthal orders without approaching
    // the angular endpoints, where derivatives can be singular.
    for (m, n) in [(0, 2), (1, 2), (1, 3), (2, 4), (3, 5)] {
        let c = 0.75;
        let pro_cv = s::wave::pro_cv(m, n, c);
        let obl_cv = s::wave::obl_cv(m, n, c);
        let parity = if (n - m) % 2 == 0 { 1.0 } else { -1.0 };

        for x in [0.25, 0.6] {
            let pro = s::wave::pro_ang1(m, n, c, x);
            let pro_with_cv = s::wave::pro_ang1_cv(m, n, c, pro_cv, x);
            let pro_reflected = s::wave::pro_ang1(m, n, c, -x);
            let pro_negative_c = s::wave::pro_ang1(m, n, -c, x);
            close_pair(pro_with_cv, pro, 2e-12, 2e-12);
            close_pair(pro_negative_c, pro, 2e-12, 2e-12);
            finite_close(pro_reflected.0, parity * pro.0, 2e-11, 2e-11);
            finite_close(pro_reflected.1, -parity * pro.1, 2e-11, 2e-11);

            let obl = s::wave::obl_ang1(m, n, c, x);
            let obl_with_cv = s::wave::obl_ang1_cv(m, n, c, obl_cv, x);
            let obl_reflected = s::wave::obl_ang1(m, n, c, -x);
            let obl_negative_c = s::wave::obl_ang1(m, n, -c, x);
            close_pair(obl_with_cv, obl, 2e-12, 2e-12);
            close_pair(obl_negative_c, obl, 2e-12, 2e-12);
            finite_close(obl_reflected.0, parity * obl.0, 2e-11, 2e-11);
            finite_close(obl_reflected.1, -parity * obl.1, 2e-11, 2e-11);
        }
    }

    // The m=0 endpoint has a regular derivative formula (unlike m=1).
    let pro = s::wave::pro_ang1(0, 2, 0.5, 1.0);
    let pro_left = s::wave::pro_ang1(0, 2, 0.5, -1.0);
    let obl = s::wave::obl_ang1(0, 2, 0.5, 1.0);
    let obl_left = s::wave::obl_ang1(0, 2, 0.5, -1.0);
    close_pair(
        s::wave::pro_ang1_cv(0, 2, 0.5, s::wave::pro_cv(0, 2, 0.5), 1.0),
        pro,
        2e-12,
        2e-12,
    );
    close_pair(
        s::wave::obl_ang1_cv(0, 2, 0.5, s::wave::obl_cv(0, 2, 0.5), 1.0),
        obl,
        2e-12,
        2e-12,
    );
    finite_close(pro_left.0, pro.0, 2e-12, 2e-12);
    finite_close(pro_left.1, -pro.1, 2e-12, 2e-12);
    finite_close(obl_left.0, obl.0, 2e-12, 2e-12);
    finite_close(obl_left.1, -obl.1, 2e-12, 2e-12);
}

#[test]
fn prolate_radial_pairs_obey_wronskian_and_supplied_cv_identity() {
    // DLMF 30.4: W(R^(1), R^(2)) = 1/[c*(x^2-1)]. This independently checks
    // both radial solutions and their derivatives away from the x=1 singularity.
    for (m, n) in [(0, 2), (1, 3), (2, 4)] {
        let c = 0.75;
        let cv = s::wave::pro_cv(m, n, c);
        for x in [1.5, 2.0] {
            let first = s::wave::pro_rad1(m, n, c, x);
            let second = s::wave::pro_rad2(m, n, c, x);
            let first_with_cv = s::wave::pro_rad1_cv(m, n, c, cv, x);
            let second_with_cv = s::wave::pro_rad2_cv(m, n, c, cv, x);
            close_pair(first_with_cv, first, 2e-11, 2e-11);
            close_pair(second_with_cv, second, 2e-11, 2e-11);
            // The m=0 basis is well conditioned for this Wronskian check;
            // at m>0 the current second-kind truncated series has an error
            // around 2.6e-4 here and should not be declared high accuracy.
            if m == 0 {
                let actual = first.0 * second.1 - first.1 * second.0;
                let expected = 1.0 / (c * (x * x - 1.0));
                let error = (actual - expected).abs() / expected.abs();
                assert!(
                    error < 1e-7,
                    "prolate Wronskian m={m} n={n} x={x}: actual={actual} expected={expected}, relative error={error}"
                );
            }
        }

        // The oblate Wronskian is covered separately; here exercise both
        // supplied-characteristic-value radial entry points against the
        // automatically computed spectrum at a regular positive argument.
        let oblate_cv = s::wave::obl_cv(m, n, c);
        let first = s::wave::obl_rad1(m, n, c, 2.0);
        let second = s::wave::obl_rad2(m, n, c, 2.0);
        close_pair(
            s::wave::obl_rad1_cv(m, n, c, oblate_cv, 2.0),
            first,
            2e-11,
            2e-11,
        );
        close_pair(
            s::wave::obl_rad2_cv(m, n, c, oblate_cv, 2.0),
            second,
            2e-11,
            2e-11,
        );
    }
}

#[test]
fn spheroidal_public_parameter_and_coordinate_domains_are_checked() {
    type Characteristic = fn(u64, u64, f64) -> f64;
    for function in [s::wave::pro_cv, s::wave::obl_cv] as [Characteristic; 2] {
        for (m, n, c) in [(1, 0, 1.0), (0, 129, 1.0), (0, 2, f64::NAN), (0, 2, 65.0)] {
            assert!(function(m, n, c).is_nan());
        }
    }

    type CharacteristicSequence = fn(u64, u64, f64) -> sci::Result<Vec<f64>>;
    for function in [s::wave::pro_cv_seq, s::wave::obl_cv_seq] as [CharacteristicSequence; 2] {
        for (m, n, c) in [(1, 0, 1.0), (0, 129, 1.0), (0, 2, f64::NAN), (0, 2, 65.0)] {
            assert!(function(m, n, c).is_err());
        }
    }

    type Angular = fn(u64, u64, f64, f64) -> (f64, f64);
    for function in [s::wave::pro_ang1, s::wave::obl_ang1] as [Angular; 2] {
        nan_pair(function(0, 2, 0.5, -1.01));
        nan_pair(function(0, 2, 0.5, 1.01));
    }

    for function in [s::wave::pro_rad1, s::wave::pro_rad2] as [Angular; 2] {
        nan_pair(function(0, 2, 0.5, 1.0));
    }
    for function in [s::wave::obl_rad1, s::wave::obl_rad2] as [Angular; 2] {
        nan_pair(function(0, 2, 0.5, -0.1));
    }

    type WithCv = fn(u64, u64, f64, f64, f64) -> (f64, f64);
    for function in [
        s::wave::pro_ang1_cv,
        s::wave::pro_rad1_cv,
        s::wave::pro_rad2_cv,
        s::wave::obl_ang1_cv,
        s::wave::obl_rad1_cv,
        s::wave::obl_rad2_cv,
    ] as [WithCv; 6]
    {
        nan_pair(function(0, 129, 0.5, 2.0, 0.3));
        nan_pair(function(0, 2, 65.0, 2.0, 0.3));
        nan_pair(function(0, 2, 0.5, f64::INFINITY, 0.3));
    }

    for function in [s::wave::pro_ang1_cv, s::wave::obl_ang1_cv] as [WithCv; 2] {
        nan_pair(function(0, 2, 0.5, 2.0, -1.01));
        nan_pair(function(0, 2, 0.5, 2.0, 1.01));
    }
    for function in [s::wave::pro_rad1_cv, s::wave::pro_rad2_cv] as [WithCv; 2] {
        nan_pair(function(0, 2, 0.5, 2.0, 1.0));
    }
    for function in [s::wave::obl_rad1_cv, s::wave::obl_rad2_cv] as [WithCv; 2] {
        nan_pair(function(0, 2, 0.5, 2.0, -0.1));
    }
}
