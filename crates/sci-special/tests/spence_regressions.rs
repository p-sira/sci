// Preserve original high-precision independent numerical reference literals.
#![allow(clippy::excessive_precision)]
// Integration regressions extracted from the sci-special numerical backends.
use sci_special::complex::{Complex, spence};

fn z(re: f64, im: f64) -> Complex<f64> {
    Complex { re, im }
}

#[test]
fn dilogarithm_reflection_and_unit_argument_match_independent_values() {
    let at_one = spence(z(0.0, 0.0)).unwrap();
    assert_eq!(
        at_one.re,
        core::f64::consts::PI * core::f64::consts::PI / 6.0
    );
    assert_eq!(at_one.im, 0.0);

    // SciPy 1.18.1: spence(0.25 - 0.1i) = Li_2(0.75 + 0.1i).
    // Exercise the dilogarithm reflection path through the public Spence API
    // and catch incorrect logarithm branches.
    let got = spence(z(0.25, -0.1)).unwrap();
    let want = z(0.964_578_687_021_021, 0.182_669_341_597_261_73);
    assert!((got.re - want.re).hypot(got.im - want.im) <= 2e-14 * want.re.hypot(want.im));
}
