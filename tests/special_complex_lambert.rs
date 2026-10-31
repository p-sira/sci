//! Independent SciPy-reference validation of complex Lambert W branches.
use sci::{complex::Complex, special::complex::lambertw};

#[test]
fn lambertw_matches_scipy_on_multiple_sheets() {
    let csv = include_str!("fixtures/complex_lambert.csv");
    let mut checked = 0;
    for (line_number, line) in csv.lines().enumerate().skip(1) {
        let parts: Vec<_> = line.split(',').collect();
        assert_eq!(parts.len(), 6, "line {line_number}");
        let k: i32 = parts[1].parse().unwrap();
        let re: f64 = parts[2].parse().unwrap();
        let im: f64 = parts[3].parse().unwrap();
        let want_re: f64 = parts[4].parse().unwrap();
        let want_im: f64 = parts[5].parse().unwrap();
        let near_branch_point = (re + core::f64::consts::E.recip()).hypot(im) < 1e-5;
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
        checked >= 3500,
        "expected many reference cases, got {checked}"
    );
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
