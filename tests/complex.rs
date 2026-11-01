mod common;
use common::assert_nonzero_close;

use sci::{
    Error,
    complex::{self as c, Complex},
    fft,
};

#[test]
fn every_public_function_matches_independent_references() {
    let inputs: Vec<_> = common::lines(include_str!("fixtures/complex_inputs.csv")).collect();
    let expected: Vec<_> = common::lines(include_str!("fixtures/complex_expected.csv")).collect();
    assert_eq!(inputs.len(), expected.len());
    let mut covered = std::collections::BTreeSet::new();
    for (input, expected) in inputs.into_iter().zip(expected) {
        let row: Vec<_> = common::fields(input);
        let reference: Vec<_> = common::fields(expected);
        assert_eq!(row[0], reference[0]);
        let z = Complex {
            re: row[2].parse().unwrap(),
            im: row[3].parse().unwrap(),
        };
        let w = Complex {
            re: row[4].parse().unwrap(),
            im: row[5].parse().unwrap(),
        };
        let scalar = |re| Complex { re, im: 0.0 };
        let result = match row[1] {
            "add" => c::add(z, w),
            "subtract" => c::subtract(z, w),
            "multiply" => c::multiply(z, w),
            "divide" => c::divide(z, w),
            "negate" => c::negate(z),
            "conjugate" => c::conjugate(z),
            "reciprocal" => c::reciprocal(z),
            "real" => scalar(c::real(z)),
            "imag" => scalar(c::imag(z)),
            "abs" => scalar(c::abs(z)),
            "norm_sqr" => scalar(c::norm_sqr(z)),
            "arg" => scalar(c::arg(z)),
            "to_polar" => {
                let (re, im) = c::to_polar(z);
                Complex { re, im }
            }
            "from_polar" => c::from_polar(z.re, z.im).unwrap(),
            "exp" => c::exp(z),
            "ln" => c::ln(z),
            "sqrt" => c::sqrt(z),
            "sin" => c::sin(z),
            "cos" => c::cos(z),
            "tan" => c::tan(z),
            "sinh" => c::sinh(z),
            "cosh" => c::cosh(z),
            "tanh" => c::tanh(z),
            "asin" => c::asin(z),
            "acos" => c::acos(z),
            "atan" => c::atan(z),
            "asinh" => c::asinh(z),
            "acosh" => c::acosh(z),
            "atanh" => c::atanh(z),
            "powf" => c::powf(z, w.re),
            "powc" => c::powc(z, w),
            "powi" => c::powi(z, row[4].parse().unwrap()),
            name => panic!("unknown function {name}"),
        };
        covered.insert(row[1]);
        assert_nonzero_close(result.re, reference[1].parse().unwrap(), 3e-14, 1e-6, input);
        assert_nonzero_close(result.im, reference[2].parse().unwrap(), 3e-14, 1e-6, input);
        if row[0].starts_with("large_") || row[0].starts_with("tangent_") {
            for (actual, expected) in [result.re, result.im]
                .into_iter()
                .zip(reference[1..].iter().map(|x| x.parse::<f64>().unwrap()))
            {
                if expected == 0.0 {
                    assert_eq!(
                        actual.is_sign_negative(),
                        expected.is_sign_negative(),
                        "{input}"
                    );
                }
            }
        }
    }
    assert_eq!(covered.len(), 32);
}

#[test]
fn domains_and_signed_zero_conventions() {
    for (radius, phase) in [
        (-1.0, 0.0),
        (f64::NAN, 0.0),
        (f64::INFINITY, 0.0),
        (1.0, f64::NAN),
        (1.0, f64::INFINITY),
    ] {
        assert!(matches!(
            c::from_polar(radius, phase),
            Err(Error::InvalidInput(_))
        ));
    }
    let z: Complex = Complex { re: 0.0, im: 0.0 };
    for result in [
        c::divide(Complex { re: 1.0, im: 2.0 }, z),
        c::reciprocal(z),
        c::powc(z, Complex { re: -1.0, im: 0.0 }),
    ] {
        assert!(result.re.is_nan() && result.im.is_nan());
    }
    for im in [0.0_f64, -0.0] {
        let z = Complex { re: -2.0, im };
        assert_eq!(c::ln(z).im.is_sign_negative(), im.is_sign_negative());
        assert_eq!(c::sqrt(z).im.is_sign_negative(), im.is_sign_negative());
        assert_eq!(c::asin(z).im.is_sign_negative(), im.is_sign_negative());
        assert_eq!(c::acos(z).im.is_sign_negative(), !im.is_sign_negative());
        assert_eq!(c::acosh(z).im.is_sign_negative(), im.is_sign_negative());
        assert_eq!(c::atanh(z).im.is_sign_negative(), im.is_sign_negative());
        let zero = Complex { re: 0.0, im };
        assert_eq!(
            c::conjugate(zero).im.is_sign_negative(),
            !im.is_sign_negative()
        );
        assert_eq!(c::sqrt(zero).im.is_sign_negative(), im.is_sign_negative());
        assert_eq!(c::imag(zero).is_sign_negative(), im.is_sign_negative());
    }
}

fn complex_close(a: Complex, b: Complex) {
    assert_nonzero_close(a.re, b.re, 3e-14, 1e-6, "identity real");
    assert_nonzero_close(a.im, b.im, 3e-14, 1e-6, "identity imaginary");
}

#[test]
fn identities_inverse_round_trips_and_fft_interoperability() {
    for z in [
        Complex { re: 0.3, im: 0.4 },
        Complex {
            re: -1.2,
            im: -0.75,
        },
        Complex { re: 2.0, im: 1.0 },
    ] {
        complex_close(c::exp(c::ln(z)), z);
        complex_close(c::multiply(c::sqrt(z), c::sqrt(z)), z);
        complex_close(c::sin(c::asin(z)), z);
        complex_close(c::cos(c::acos(z)), z);
        complex_close(c::tan(c::atan(z)), z);
        complex_close(c::sinh(c::asinh(z)), z);
        complex_close(c::cosh(c::acosh(z)), z);
        complex_close(c::tanh(c::atanh(z)), z);
        complex_close(
            c::multiply(z, c::reciprocal(z)),
            Complex { re: 1.0, im: 0.0 },
        );
        complex_close(c::powi(z, 3), c::multiply(c::multiply(z, z), z));
        let (r, theta) = c::to_polar(z);
        complex_close(c::from_polar(r, theta).unwrap(), z);
    }
    let input = [Complex { re: 1.0, im: 0.0 }, Complex { re: 0.0, im: 1.0 }];
    let spectrum = fft::fourier::fft(&input);
    let energy: f64 = spectrum.iter().copied().map(c::norm_sqr).sum();
    assert_nonzero_close(energy, 4.0, 3e-14, 1e-6, "Parseval");
}

#[test]
fn inverse_round_trips_preserve_tiny_components() {
    for z in [
        Complex {
            re: 1e-300,
            im: -1e-300,
        },
        Complex {
            re: -1e-20,
            im: 1e-20,
        },
        Complex {
            re: 1e-8,
            im: -1e-8,
        },
    ] {
        for actual in [
            c::sin(c::asin(z)),
            c::tan(c::atan(z)),
            c::sinh(c::asinh(z)),
            c::tanh(c::atanh(z)),
        ] {
            assert_nonzero_close(actual.re, z.re, 3e-14, 1e-6, "tiny inverse real round trip");
            assert_nonzero_close(
                actual.im,
                z.im,
                3e-14,
                1e-6,
                "tiny inverse imaginary round trip",
            );
        }
    }
}

#[test]
fn real_powers_reuse_roots_and_preserve_axis_components() {
    for z in [
        Complex { re: -2.0, im: 0.0 },
        Complex { re: -2.0, im: -0.0 },
        Complex {
            re: 1e308_f64,
            im: 1e-300,
        },
        Complex {
            re: 1e308_f64,
            im: 1e308,
        },
    ] {
        assert_eq!(c::powf(z, 0.5), c::sqrt(z));
        assert_eq!(c::powf(z, -0.5), c::reciprocal(c::sqrt(z)));
        assert_eq!(c::powf(z, 1.0), z);
        for exponent in [-0.5, 0.0, 0.5, 1.0, 1.5] {
            assert_eq!(
                c::powc(
                    z,
                    Complex {
                        re: exponent,
                        im: 0.0
                    }
                ),
                c::powf(z, exponent)
            );
        }
    }
    let z = c::powf(
        Complex {
            re: 1e308_f64,
            im: 1e-300,
        },
        1.5,
    );
    assert!(z.re.is_infinite());
    assert_nonzero_close(z.im, 1.5e-146, 3e-14, 1e-6, "near-axis power derivative");
}

#[test]
fn products_preserve_components_and_exact_cancellation_residuals() {
    let z = c::multiply(
        Complex {
            re: 1e308_f64,
            im: 1e308,
        },
        Complex { re: 2.0, im: 2.0 },
    );
    assert_eq!(z.re, 0.0);
    assert_eq!(z.im, f64::INFINITY);
    assert_eq!(
        c::norm_sqr(Complex {
            re: 1.2e-162,
            im: 1.2e-162
        }),
        f64::from_bits(1)
    );
    let eps = f64::EPSILON;
    let z = c::multiply(
        Complex {
            re: 1.0 + eps,
            im: 1.0,
        },
        Complex {
            re: 1.0 - eps,
            im: 1.0,
        },
    );
    assert_eq!(z.re, -eps * eps);
    assert_eq!(z.im, 2.0);
}
