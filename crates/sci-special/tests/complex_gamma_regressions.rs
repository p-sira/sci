// Preserve original high-precision independent numerical reference literals.
#![allow(clippy::excessive_precision)]
// Integration regressions extracted from the sci-special numerical backends.
use sci_special::complex::{Complex, gamma};
fn z(re: f64, im: f64) -> Complex<f64> {
    Complex { re, im }
}

fn close_relative(actual: f64, expected: f64, label: &str) {
    // These small components are normal binary64 numbers; an absolute
    // tolerance would incorrectly accept a result rounded to zero.
    const RELATIVE_TOLERANCE: f64 = 5e-12;
    let error = (actual - expected).abs();
    assert!(
        actual.is_finite() && error <= RELATIVE_TOLERANCE * expected.abs(),
        "{label}: actual={actual:.17e}, expected={expected:.17e}, error={error:.3e}"
    );
}

#[test]
fn gamma_reflection_large_imaginary_argument_matches_high_precision_reference() {
    // Independently evaluated using mpmath 1.4.1 with mp.dps = 100:
    //   gamma(mpc(-1/4, 223))
    // At |Im(z)| = 223, pi*|Im(z)| > 700, which forces trig_pi's
    // split-exponential sine branch in loggamma_kernel's reflection.
    // Retain >60 decimal digits of each reference component; parsing to
    // f64 avoids discarding precision in the checked-in source.
    let reference = z(
        "4.115951299497563629161749458169556969749545930356014677603747239974421362126394940741e-155"
            .parse::<f64>()
            .unwrap(),
        "3.208883445780643210060165297550080145591842073427260538804176359867594973054817465413e-154"
            .parse::<f64>()
            .unwrap(),
    );

    let upper = gamma(z(-0.25, 223.0)).unwrap();
    close_relative(upper.re, reference.re, "upper gamma real");
    close_relative(upper.im, reference.im, "upper gamma imaginary");

    let lower = gamma(z(-0.25, -223.0)).unwrap();
    close_relative(lower.re, reference.re, "lower gamma real");
    close_relative(lower.im, -reference.im, "lower gamma imaginary");
    // In addition to the independent reference, test the conjugate symmetry.
    close_relative(lower.re, upper.re, "gamma conjugate real");
    close_relative(lower.im, -upper.im, "gamma conjugate imaginary");
}

#[test]
fn gamma_rejects_nonfinite_arguments_and_real_nonpositive_integer_poles() {
    for z in [
        z(f64::NAN, 0.0),
        z(0.5, f64::NAN),
        z(f64::INFINITY, 0.25),
        z(-0.25, f64::NEG_INFINITY),
        z(0.0, 0.0),
        z(-1.0, 0.0),
        z(-6.0, -0.0),
    ] {
        assert!(gamma(z).is_err(), "expected Gamma domain error for {z:?}");
    }
}
