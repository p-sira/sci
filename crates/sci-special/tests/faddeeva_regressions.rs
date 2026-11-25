// Preserve original high-precision independent numerical reference literals.
#![allow(clippy::excessive_precision)]
// High-precision complex error-function references via the public Sci API.
use sci_complex::Complex;

fn assert_components(actual: Complex<f64>, expected: Complex<f64>) {
    let tolerance = 1.0e-13;
    for (actual, expected) in [(actual.re, expected.re), (actual.im, expected.im)] {
        assert!(
            ((actual - expected) / expected).abs() < tolerance,
            "actual={actual:.17e}, independent reference={expected:.17e}"
        );
    }
}

#[test]
fn test_erfi_complex() {
    assert_components(
        sci_special::complex::erfi(Complex {
            re: 1.234_f64,
            im: 0.5678,
        })
        .unwrap(),
        Complex {
            re: 1.081032284405373149432716643834106923212,
            im: 1.926775520840916645838949402886591180834,
        },
    );
}

#[test]
fn test_erfcx_complex() {
    assert_components(
        sci_special::complex::erfcx(Complex {
            re: 1.234_f64,
            im: 0.5678,
        })
        .unwrap(),
        Complex {
            re: 0.3382187479799972294747793561190487832579,
            im: -0.1116077470811648467464927471872945833154,
        },
    );
}
