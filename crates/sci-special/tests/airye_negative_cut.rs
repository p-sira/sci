//! Independent lower-cut regression for the exponentially scaled Airy family.

use sci_complex::Complex;
use sci_special::complex::{airy, airye};

fn z(re: f64, im: f64) -> Complex {
    Complex { re, im }
}

#[test]
fn airye_negative_real_signed_zero_matches_analytic_continuation() {
    // High-precision independent reference: Ai(x) and Bi(x) are real for
    // real x; at x = -1.25, zeta = (2/3) z sqrt(z) tends to +/- i theta,
    // theta = (2/3) 1.25^(3/2). Thus Ai and Ai' scale by exp(+i theta)
    // below the cut and exp(-i theta) above, while Bi and Bi' stay real.
    // Constants from 65-digit mpmath Airy evaluations and this identity.
    let lower_expected = [
        z(0.3101938067761736, 0.4174051998222041),
        z(0.08295739713699708, 0.11162972365107177),
        z(-0.045867468727426906, 0.0),
        z(0.5998141935575834, 0.0),
    ];
    let upper = airye(z(-1.25, 0.0)).unwrap();
    let lower = airye(z(-1.25, -0.0)).unwrap();
    for (i, ((lo, hi), expected)) in lower.iter().zip(&upper).zip(&lower_expected).enumerate() {
        let tol = 2e-13;
        assert!(
            (lo.re - expected.re).abs() < tol,
            "lower component {i}: {lo:?}"
        );
        assert!(
            (lo.im - expected.im).abs() < tol,
            "lower component {i}: {lo:?}"
        );
        assert!(
            (lo.re - hi.re).abs() < tol,
            "conjugate real {i}: {lo:?}, {hi:?}"
        );
        assert!(
            (lo.im + hi.im).abs() < tol,
            "conjugate imag {i}: {lo:?}, {hi:?}"
        );
    }
}

#[test]
fn airye_cut_limits_and_unscaled_airy_remain_consistent() {
    for x in [-0.5, -1.25, -3.5, -12.0] {
        let upper = airye(z(x, 0.0)).unwrap();
        let lower = airye(z(x, -0.0)).unwrap();
        let unscaled = airy(z(x, 0.0)).unwrap();
        let near_lower = airye(z(x, -1e-12)).unwrap();
        for i in 0..4 {
            let tol = 2e-11 * upper[i].re.hypot(upper[i].im).max(1.0);
            assert!(
                (lower[i].re - upper[i].re).abs() < tol,
                "x={x}, component {i}"
            );
            assert!(
                (lower[i].im + upper[i].im).abs() < tol,
                "x={x}, component {i}"
            );
            assert!(
                (lower[i].re - near_lower[i].re).abs() < tol,
                "x={x}, component {i}"
            );
            assert!(
                (lower[i].im - near_lower[i].im).abs() < tol,
                "x={x}, component {i}"
            );
            if i >= 2 {
                // Re(zeta)=0 on the negative real axis, so Bi's scaling is one.
                assert!(
                    (lower[i].re - unscaled[i].re).abs() < tol,
                    "x={x}, Bi component {i}"
                );
                assert!(lower[i].im.abs() < tol, "x={x}, Bi imaginary component {i}");
            }
        }
    }
}
