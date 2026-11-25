//! Preserve Sci's public domain boundaries when standalone crates support a
//! broader domain, and exercise real Airy conversion/fallback behavior.

use sci_core::Error;
use sci_special::{bessel, orthogonal, wave};

#[test]
fn standalone_polynomial_evaluators_retain_sci_degree_limits() {
    // Standalone orthogonal accepts degrees above 4096; Sci deliberately
    // retains its historical non-fallible NaN contract for those degrees.
    for n in [-1, 4097, i32::MAX] {
        for value in [
            orthogonal::eval_chebyc(n, 0.3),
            orthogonal::eval_chebys(n, 0.3),
            orthogonal::eval_sh_chebyt(n, 0.3),
            orthogonal::eval_sh_chebyu(n, 0.3),
            orthogonal::eval_sh_legendre(n, 0.3),
            orthogonal::eval_sh_jacobi(n, 2.5, 1.5, 0.3),
        ] {
            assert!(value.is_nan(), "invalid degree {n} should return NaN");
        }
    }
    assert_eq!(orthogonal::eval_chebyc(0, 0.3), 2.0);
    assert_eq!(orthogonal::eval_chebys(0, 0.3), 1.0);
    assert_eq!(orthogonal::eval_sh_chebyt(0, 0.3), 1.0);
    assert_eq!(orthogonal::eval_sh_chebyu(0, 0.3), 1.0);
    assert_eq!(orthogonal::eval_sh_legendre(0, 0.3), 1.0);
    assert_eq!(orthogonal::eval_sh_jacobi(0, 2.5, 1.5, 0.3), 1.0);
}

#[test]
fn standalone_quadrature_adapters_retain_sci_order_limits() {
    // The standalone quadrature routines support orders above Sci's limit.
    // Reject invalid orders *before* forwarding and allocating node arrays.
    for n in [0, 257, usize::MAX] {
        for result in [
            orthogonal::roots_jacobi(n, 0.5, 0.25),
            orthogonal::roots_genlaguerre(n, 0.5),
            orthogonal::roots_sh_jacobi(n, 2.5, 1.5),
        ] {
            assert!(matches!(result, Err(Error::InvalidInput(_))));
        }
    }
    assert_eq!(orthogonal::roots_jacobi(1, 0.5, 0.25).unwrap().0.len(), 1);
    assert_eq!(orthogonal::roots_genlaguerre(1, 0.5).unwrap().0.len(), 1);
    assert_eq!(orthogonal::roots_sh_jacobi(1, 2.5, 1.5).unwrap().0.len(), 1);
}

#[test]
fn standalone_bessel_zero_adapters_retain_sci_order_limits() {
    for n in [257, u32::MAX] {
        for result in [
            bessel::jn_zeros(n, 1),
            bessel::jnp_zeros(n, 1),
            bessel::yn_zeros(n, 1),
            bessel::ynp_zeros(n, 1),
        ] {
            assert!(matches!(result, Err(Error::InvalidInput(_))));
        }
        assert!(matches!(
            bessel::jnyn_zeros(n, 1),
            Err(Error::InvalidInput(_))
        ));
    }
    // Invalid count takes precedence over invalid order, as in Sci 0.2.x.
    assert!(matches!(
        bessel::jn_zeros(257, 0),
        Err(Error::InvalidInput("zero count must be in 1..=256"))
    ));
}

#[test]
fn real_airy_adapters_preserve_negative_scaled_convention_and_nan() {
    // On negative x the standalone scaled Airy kernel rejects the input;
    // Sci's documented real convention supplies Bi/Bi' and NaN-scaled Ai.
    let x = -1.25;
    let scaled = wave::airye(x);
    let unscaled = wave::airy(x);
    assert!(scaled[0].is_nan() && scaled[1].is_nan());
    // Independently evaluated 65-digit Airy references at x=-1.25.
    assert!((scaled[2] - (-0.045867468727426906)).abs() < 2e-14);
    assert!((scaled[3] - 0.5998141935575834).abs() < 2e-14);
    for i in 2..4 {
        assert!(
            (scaled[i] - unscaled[i]).abs() <= 1e-12 * unscaled[i].abs().max(1.0),
            "negative real scaled Airy component {i}: {:?} vs {:?}",
            scaled,
            unscaled
        );
    }
    for values in [wave::airy(f64::NAN), wave::airye(f64::NAN)] {
        assert!(values.iter().all(|value| value.is_nan()));
    }
}
