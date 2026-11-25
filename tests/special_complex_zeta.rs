//! Independent binary64 references for the public complex Riemann and Hurwitz
//! zeta APIs. Riemann values are from scipy.special.zeta (SciPy 1.18.1),
//! except the near-pole case, where SciPy loses ~3e-7 in the real part and
//! mpmath 1.4.1 at 70 dps supplies the reference. SciPy's two-argument zeta
//! rejects complex arguments; Hurwitz values also use 70-dps mpmath,
//! with inputs converted from the exact binary64 components used below.

// Preserve full independent-reference decimal representations for auditability.
#![allow(clippy::excessive_precision)]

use sci::complex::Complex;
use sci::special as s;

fn c(re: f64, im: f64) -> Complex {
    Complex { re, im }
}

fn close(actual: Complex, reference: Complex, tolerance: f64) {
    let scale = reference.re.hypot(reference.im).max(1e-300);
    for (got, want) in [(actual.re, reference.re), (actual.im, reference.im)] {
        assert!(
            got.is_finite() && (got - want).abs() <= tolerance * (want.abs() + 0.01 * scale),
            "got {actual:?}, reference {reference:?}, tolerance={tolerance:e}"
        );
    }
}

#[test]
fn riemann_zeta_matches_scipy_complex_references() {
    // (Re(s), Im(s), Re(ζ(s)), Im(ζ(s)))
    const REFERENCES: &[(f64, f64, f64, f64)] = &[
        (0.5, 0.0, -1.4603545088095866, 0.0),
        (0.5, 20.0, 0.42991386043784746, -1.0642914430805916),
        (2.0, 3.0, 0.79802198514627565, -0.11374430805293852),
        (4.0, 0.0, 1.0823232337111381, 0.0),
        (0.001, 0.05, -0.49841074112528205, -0.045922109056198333),
        (1.0, 0.1, 0.57726412640686964, -9.992718073212018),
        // Near s=1: Laurent constant γ is ~0.5772156649. SciPy's real
        // component here is inaccurate by 3.18e-7 (mpmath 70 dps).
        (1.0, 1e-5, 0.5772156649020174, -99999.99999927183),
        (7.5, 31.0, 0.994951978671306, -0.002763274766518369),
        (27.5, 4.0, 0.9999999950862486, -1.9001534509920548e-9),
        (0.0, 2.0, 0.3147257640420996, -0.23167964875052077),
        (0.0, 0.0, -0.5, 0.0),
        (30.0, 0.0, 1.0000000009313275, 0.0),
        (2.0, -31.0, 0.7547548874326079, 0.0803679045678366),
    ];
    for &(sr, si, zr, zi) in REFERENCES {
        let actual = s::complex::zeta(c(sr, si)).unwrap();
        close(actual, c(zr, zi), 2e-11);
    }
}

#[test]
fn hurwitz_zeta_matches_high_precision_complex_references() {
    // (Re(s), Im(s), Re(q), Im(q), Re(ζ(s,q)), Im(ζ(s,q)))
    const REFERENCES: &[(f64, f64, f64, f64, f64, f64)] = &[
        (2.0, 0.0, 1.5, 0.0, 0.9348022005446793, 0.0),
        (2.0, 3.0, 1.5, 0.3, -0.301247137078625, -0.6542040595109239),
        (0.5, 0.5, 2.0, 1.0, -1.8175629440121062, -1.794607829368408),
        (0.0, 0.0, 2.0, 1.0, -1.5, -1.0),
        (1.0, 0.5, 0.25, 0.0, 3.3053509634790106, 0.5421975917399197),
        (
            0.5,
            20.0,
            2.0,
            -1.0,
            -0.0013424613999705464,
            -0.0012991811804499796,
        ),
        (
            20.0,
            0.5,
            0.75,
            0.1,
            -228.2733318419403,
            -166.43663971952867,
        ),
        (0.1, 1.0, 4.0, 0.0, -2.110872923716671, 0.9227428278604085),
        (
            4.5,
            3.2,
            3.0,
            -2.0,
            -0.0002522545118451087,
            -0.00038003470769719453,
        ),
        (
            10.0,
            10.0,
            10.0,
            5.0,
            -3.615463064159707e-9,
            3.045675258828946e-9,
        ),
        (
            29.0,
            31.0,
            0.1,
            0.1,
            1.597165517957176e35,
            2.5889369838870343e34,
        ),
        (0.0, 10.0, 0.1, 0.1, 1954.8094263363821, 1677.3378240372781),
        (0.1, 30.0, 3.0, 2.0, 27666378.68541216, -30650161.50314899),
    ];
    for &(sr, si, qr, qi, zr, zi) in REFERENCES {
        let actual = s::complex::hurwitz_zeta(c(sr, si), c(qr, qi)).unwrap();
        close(actual, c(zr, zi), 3e-11);
    }
}

#[test]
fn zeta_shift_half_argument_and_conjugation() {
    for &(sr, si, qr, qi) in &[
        (0.5, 2.0, 0.5, 0.75),
        (2.5, 5.0, 2.0, -1.5),
        (0.25, 12.0, 3.0, 1.0),
        // Large Re(s) with tiny q gives a cancellation-sensitive subtraction
        // in the shift identity; use a moderately sized order here.
        (4.0, -3.0, 0.25, 0.25),
    ] {
        let order = c(sr, si);
        let q = c(qr, qi);
        let first = s::complex::hurwitz_zeta(order, q).unwrap();
        let next = s::complex::hurwitz_zeta(order, c(qr + 1.0, qi)).unwrap();
        let expected = sci::complex::subtract(first, sci::complex::powc(q, c(-sr, -si)));
        close(next, expected, 4e-11);

        let conjugate = s::complex::hurwitz_zeta(c(sr, -si), c(qr, -qi)).unwrap();
        close(conjugate, c(first.re, -first.im), 2e-11);
    }

    // Hurwitz's half-argument identity: ζ(s,1/2)=(2^s-1)ζ(s).
    let order = c(2.0, 3.0);
    let actual = s::complex::hurwitz_zeta(order, c(0.5, 0.0)).unwrap();
    let z = s::complex::zeta(order).unwrap();
    let factor = sci::complex::subtract(sci::complex::powc(c(2.0, 0.0), order), c(1.0, 0.0));
    let expected = sci::complex::multiply(factor, z);
    close(actual, expected, 4e-11);

    // Preserve real-facade interoperability on its supported s>1 domain.
    for &(order, q) in &[(2.0, 0.5), (2.5, 1.5), (5.0, 2.0), (15.0, 1.0)] {
        let result = s::complex::hurwitz_zeta(c(order, 0.0), c(q, 0.0)).unwrap();
        close(result, c(s::zeta::zeta(order, q), 0.0), 3e-11);
    }
}

#[test]
fn complex_zeta_f32_public_conversion() {
    let z = s::complex::zeta(Complex {
        re: 2.0_f32,
        im: 3.0,
    })
    .unwrap();
    assert!((z.re - 0.798021985_f32).abs() < 3e-6);
    assert!((z.im - (-0.113744308_f32)).abs() < 3e-6);

    let z = s::complex::hurwitz_zeta(
        Complex {
            re: 2.0_f32,
            im: 0.0,
        },
        Complex {
            re: 1.5_f32,
            im: 0.0,
        },
    )
    .unwrap();
    assert!((z.re - 0.9348022_f32).abs() < 3e-6);
    assert!(z.im.abs() < 3e-6);
}

#[test]
fn complex_zeta_rejects_invalid_poles_overflow_and_cancellation() {
    for order in [
        c(f64::NAN, 0.0),
        c(2.0, f64::INFINITY),
        c(1.0, 0.0),
        c(-0.1, 0.5),
        c(30.1, 0.5),
        c(2.0, 32.1),
    ] {
        assert!(s::complex::zeta(order).is_err(), "order={order:?}");
    }
    for q in [
        c(0.0, 0.0),
        c(-2.0, 0.0),
        c(65.0, 0.0),
        c(1.0, 33.0),
        c(f64::NAN, 1.0),
        c(1.0, f64::NEG_INFINITY),
    ] {
        assert!(s::complex::hurwitz_zeta(c(2.0, 0.5), q).is_err(), "q={q:?}");
    }
    assert!(s::complex::hurwitz_zeta(c(30.0, 0.0), c(1e-20, 0.0)).is_err());

    // A point near the first nontrivial Riemann zero is too ill-conditioned
    // for relative-accuracy certification with ordinary binary64 summation.
    assert!(s::complex::zeta(c(0.5, 14.13472514173469)).is_err());
    // A complex-Hurwitz cancellation example: individual terms swamp the
    // true ~2e-26 result (mpmath, 70 dps).
    assert!(
        s::complex::hurwitz_zeta(
            c(2.5632884854256686, 31.838047872185875),
            c(0.011126059374215555, -17.753046721913535),
        )
        .is_err()
    );
}
