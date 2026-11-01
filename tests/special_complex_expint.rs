//! Complex E1 and Ei checked against independent SciPy reference fixtures.
use sci::{
    complex::Complex,
    special::{complex, integrals},
};

fn close(actual: f64, expected: f64, tolerance: f64, context: &str) {
    if expected == 0.0 {
        assert_eq!(
            actual, 0.0,
            "{context}: expected exact zero, got {actual:e}"
        );
    } else {
        assert!(
            actual.is_finite()
                && (actual - expected).abs() <= tolerance * expected.abs().max(1e-44),
            "{context}: actual {actual:.17e}, expected {expected:.17e}"
        );
    }
}

#[test]
fn expint_matches_independent_complex_references_in_both_precisions() {
    let fixture = include_str!("fixtures/special_complex_expint.csv");
    let mut checked_f64 = 0;
    let mut checked_f32 = 0;
    for (index, line) in fixture.lines().enumerate().skip(1) {
        let row: Vec<_> = line.split(',').collect();
        assert_eq!(row.len(), 7, "line {index}");
        let re: f64 = row[1].parse().unwrap();
        let im: f64 = row[2].parse().unwrap();
        let e1_re: f64 = row[3].parse().unwrap();
        let e1_im: f64 = row[4].parse().unwrap();
        let ei_re: f64 = row[5].parse().unwrap();
        let ei_im: f64 = row[6].parse().unwrap();
        let (a, b, tolerance) = if row[0] == "f32" {
            let z = Complex {
                re: re as f32,
                im: im as f32,
            };
            let e1 = complex::exp1(z).unwrap_or_else(|error| panic!("E1 {z:?}: {error:?}"));
            let ei = complex::expi(z).unwrap_or_else(|error| panic!("Ei {z:?}: {error:?}"));
            checked_f32 += 1;
            (
                (f64::from(e1.re), f64::from(e1.im)),
                (f64::from(ei.re), f64::from(ei.im)),
                4e-6,
            )
        } else {
            let z = Complex { re, im };
            let e1 = complex::exp1(z).unwrap_or_else(|error| panic!("E1 {z:?}: {error:?}"));
            let ei = complex::expi(z).unwrap_or_else(|error| panic!("Ei {z:?}: {error:?}"));
            checked_f64 += 1;
            ((e1.re, e1.im), (ei.re, ei.im), 4e-12)
        };
        for (got, expected, name) in [
            (a.0, e1_re, "E1 real"),
            (a.1, e1_im, "E1 imaginary"),
            (b.0, ei_re, "Ei real"),
            (b.1, ei_im, "Ei imaginary"),
        ] {
            close(
                got,
                expected,
                tolerance,
                &format!("line {index} {name}, z={re}+{im}i"),
            );
        }
    }
    assert!(checked_f64 >= 70 && checked_f32 >= 65);
}

#[test]
fn signed_zero_selects_e1_cut_and_ei_real_axis_value() {
    for re in [-40., -20., -1., -0.1] {
        let upper = complex::exp1(Complex { re, im: 0.0_f64 }).unwrap();
        let lower = complex::exp1(Complex { re, im: -0.0_f64 }).unwrap();
        let ordinary = complex::expi(Complex { re, im: -0.0_f64 }).unwrap();
        assert_eq!(upper.im, -std::f64::consts::PI);
        assert_eq!(lower.im, std::f64::consts::PI);
        assert_eq!(upper.re, lower.re);
        assert_eq!(ordinary.im, 0.0);
    }
}

#[test]
fn real_facade_and_derivative_identities() {
    for x in [0.05, 0.5, 1., 10., 30., 100.] {
        let z = Complex { re: x, im: 0.0_f64 };
        let a = complex::exp1(z).unwrap();
        let b = complex::expi(z).unwrap();
        close(a.re, integrals::exp1(x), 4e-12, "E1 real interop");
        close(b.re, integrals::expi(x), 4e-12, "Ei real interop");
    }
    let z = Complex {
        re: 1.0_f64,
        im: 0.8,
    };
    let h = 1e-5;
    let neg = num_complex_exp(z.re, z.im, true);
    for (kernel, expected) in [
        (
            complex::exp1 as fn(Complex<f64>) -> sci::Result<Complex<f64>>,
            (-neg.0, -neg.1),
        ),
        (
            complex::expi as fn(Complex<f64>) -> sci::Result<Complex<f64>>,
            num_complex_exp(z.re, z.im, false),
        ),
    ] {
        let up = kernel(Complex {
            re: z.re + h,
            im: z.im,
        })
        .unwrap();
        let down = kernel(Complex {
            re: z.re - h,
            im: z.im,
        })
        .unwrap();
        close(
            (up.re - down.re) / (2.0 * h),
            expected.0,
            2e-9,
            "derivative real",
        );
        close(
            (up.im - down.im) / (2.0 * h),
            expected.1,
            2e-9,
            "derivative imaginary",
        );
    }
}

fn num_complex_exp(re: f64, im: f64, negate: bool) -> (f64, f64) {
    let (a, b) = if negate { (-re, -im) } else { (re, im) };
    let scale = a.exp();
    // exp(±z)/z, computed using the complex reciprocal explicitly.
    let inverse = 1.0 / (re * re + im * im);
    (
        scale * (b.cos() * re + b.sin() * im) * inverse,
        scale * (b.sin() * re - b.cos() * im) * inverse,
    )
}

#[test]
fn poles_invalid_inputs_and_bounded_domain_report_errors() {
    for z in [
        Complex {
            re: 0.0_f64,
            im: 0.0,
        },
        Complex { re: 0.0, im: -0.0 },
        Complex {
            re: f64::NAN,
            im: 1.0,
        },
        Complex {
            re: 1.0,
            im: f64::INFINITY,
        },
        Complex {
            re: 1.01e6,
            im: 0.0,
        },
    ] {
        assert!(complex::exp1(z).is_err(), "E1 must reject {z:?}");
        assert!(complex::expi(z).is_err(), "Ei must reject {z:?}");
    }
    assert!(
        complex::expi(Complex {
            re: 720.0_f64,
            im: 0.0
        })
        .is_err()
    );
}
