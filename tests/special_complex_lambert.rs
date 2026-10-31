//! Independent SciPy-reference validation of complex Lambert W branches.
use std::collections::BTreeMap;

use sci::{complex::Complex, special::complex::lambertw};

#[test]
fn lambertw_matches_scipy_on_multiple_sheets() {
    // The checked-in grid is stratified to avoid redundant Cartesian-product
    // points. For a full numerical sweep, generate the exhaustive fixture and
    // pass its path via SCI_LAMBERTW_FIXTURE without modifying tracked files.
    let csv = std::env::var_os("SCI_LAMBERTW_FIXTURE")
        .map(|path| std::fs::read_to_string(path).expect("read Lambert W reference fixture"))
        .unwrap_or_else(|| include_str!("fixtures/complex_lambert.csv").to_owned());
    let mut checked = 0;
    let mut per_sheet = BTreeMap::new();
    let mut branch_points = BTreeMap::new();
    for (line_number, line) in csv.lines().enumerate().skip(1) {
        let parts: Vec<_> = line.split(',').collect();
        assert_eq!(parts.len(), 6, "line {line_number}");
        let k: i32 = parts[1].parse().unwrap();
        let re: f64 = parts[2].parse().unwrap();
        let im: f64 = parts[3].parse().unwrap();
        let want_re: f64 = parts[4].parse().unwrap();
        let want_im: f64 = parts[5].parse().unwrap();
        let near_branch_point = (re + core::f64::consts::E.recip()).hypot(im) < 1e-5;
        *per_sheet.entry((parts[0], k)).or_insert(0usize) += 1;
        if near_branch_point {
            *branch_points.entry((parts[0], k)).or_insert(0usize) += 1;
        }
        let (got_re, got_im, reltol) = if parts[0] == "f32" {
            let z = Complex {
                re: re as f32,
                im: im as f32,
            };
            let w = lambertw(z, k).unwrap_or_else(|error| {
                panic!("line {line_number}: ({k}, {z:?}) failed: {error:?}")
            });
            (
                f64::from(w.re),
                f64::from(w.im),
                if near_branch_point { 5e-4 } else { 2e-5 },
            )
        } else {
            let z = Complex { re, im };
            let w = lambertw(z, k).unwrap_or_else(|error| {
                panic!("line {line_number}: ({k}, {z:?}) failed: {error:?}")
            });
            (w.re, w.im, if near_branch_point { 2e-8 } else { 1e-9 })
        };
        for (got, want, component) in [(got_re, want_re, "re"), (got_im, want_im, "im")] {
            assert!(
                (got - want).abs() <= reltol * want.abs().max(1e-10),
                "line {line_number} {component}: k={k}, z=({re},{im}), got={got:.17e}, want={want:.17e}"
            );
        }
        checked += 1;
    }
    assert!(
        checked >= 1700,
        "expected a broad independent reference grid, got {checked}"
    );
    for precision in ["f32", "f64"] {
        for k in [-5, -2, -1, 0, 1, 2, 5] {
            let sheet = (precision, k);
            assert!(
                per_sheet.get(&sheet).copied().unwrap_or(0) >= 120,
                "missing broad {precision} reference coverage for branch {k}"
            );
            assert!(
                branch_points.get(&sheet).copied().unwrap_or(0) >= 6,
                "missing branch-point {precision} coverage for branch {k}"
            );
        }
    }
}

#[test]
fn lambertw_reports_singularities_and_invalid_inputs() {
    for k in [-2, -1, 1, 2] {
        assert!(
            lambertw(
                Complex {
                    re: 0.0_f64,
                    im: 0.0
                },
                k
            )
            .is_err()
        );
    }
    for z in [
        Complex {
            re: f64::NAN,
            im: 0.0,
        },
        Complex {
            re: 1.0,
            im: f64::INFINITY,
        },
    ] {
        assert!(lambertw(z, 0).is_err());
    }
    assert!(
        lambertw(
            Complex {
                re: 1.0_f64,
                im: 0.0
            },
            1_000_001
        )
        .is_err()
    );
}

#[test]
fn branch_one_below_negative_cut_uses_puiseux_near_double_root() {
    // Mac ARM and Windows MSVC exposed a nonconvergent Halley path for this
    // exact float32 input. SciPy 1.18.1 evaluates the k=1 lower-cut value as
    // -1.000109403208048 + 0.0002484450061818578i.
    let z = Complex {
        re: -0.36787945_f32,
        im: -1e-8_f32,
    };
    let actual = lambertw(z, 1).unwrap();
    assert!((f64::from(actual.re) + 1.000109403208048).abs() < 2e-6);
    assert!((f64::from(actual.im) - 0.0002484450061818578).abs() < 5e-8);

    let z = Complex {
        re: -0.36787944117144233_f64,
        im: -1e-12_f64,
    };
    let actual = lambertw(z, 1).unwrap();
    assert!((actual.re + 1.00000164871204).abs() < 2e-9);
    assert!((actual.im - 1.648732309854558e-6).abs() < 3e-9);

    // The branch discontinuity matters: k=1 above the negative cut is not
    // the same near-real solution; it remains on the next logarithmic sheet.
    let above = lambertw(
        Complex {
            re: -0.36787945_f32,
            im: 1e-8_f32,
        },
        1,
    )
    .unwrap();
    assert!(above.im > 5.0);
}

#[test]
fn exact_double_root_has_analytic_value_on_both_meeting_sheets() {
    // SciPy's finite-value fixture skips this exact f64 point, where its
    // implementation can return NaN. The identity W(-1/e)=-1 is exact.
    let z = Complex {
        re: -1.0_f64 / std::f64::consts::E,
        im: 0.0,
    };
    for k in [0, -1] {
        let w = lambertw(z, k).unwrap();
        assert!((w.re + 1.0).abs() <= 1e-14, "branch {k}: {w:?}");
        assert_eq!(w.im, 0.0);
    }
}
