//! Inverse and scaled real error functions and logarithmic normal probabilities.
use crate::backend::{cephes64, utils::real_kernel};
#[allow(unused_imports)]
use sci_core::Float;

real_kernel! {
    /// Inverse error function, for -1 <= y <= 1. Endpoints give signed infinity.
    erfinv(y) {
        if y.abs() < 1e-7 { y / core::f64::consts::FRAC_2_SQRT_PI }
        else if y.abs() <= 1.0 {
            // Use symmetry so 1+y does not discard precision in the negative tail.
            (-cephes64::ndtri(0.5 * (1.0 - y.abs())) * core::f64::consts::FRAC_1_SQRT_2).copysign(y)
        } else { f64::NAN }
    }
}
real_kernel! {
    /// Inverse complementary error function, for 0 <= y <= 2.
    erfcinv(y) { -cephes64::ndtri(0.5 * y) * core::f64::consts::FRAC_1_SQRT_2 }
}

pub(crate) fn erfcx_real(x: f64) -> f64 {
    if x < 0.0 {
        if x * x > (f64::MAX / 2.0).ln() {
            return f64::INFINITY;
        }
        return 2.0 * (x * x).exp() - erfcx_real(-x);
    }
    if x < 1.0 {
        return (x * x).exp() * cephes64::erfc(x);
    }
    cephes64::ndtr::erfc_scaled(x)
}
real_kernel! {
    /// Scaled complementary error function exp(x²)*erfc(x), without overflow for positive x.
    erfcx(x) { erfcx_real(x) }
}
real_kernel! {
    /// Imaginary error function erfi(x) = 2/sqrt(pi)*exp(x²)*Dawson(x).
    erfi(x) {
        if x.abs() > 27.0 { f64::INFINITY.copysign(x) }
        else { { let e = (0.5 * x * x).exp(); (core::f64::consts::FRAC_2_SQRT_PI * cephes64::dawsn(x) * e) * e } }
    }
}
real_kernel! {
    /// Logarithm of the standard normal CDF, accurate in both tails.
    log_ndtr(x) {
        let t = x * core::f64::consts::FRAC_1_SQRT_2;
        if x < -1.0 { (erfcx_real(-t) / 2.0).ln() - t * t }
        else { (-cephes64::erfc(t) / 2.0).ln_1p() }
    }
}
real_kernel! {
    /// Inverse of log_ndtr, for y <= 0; supports logarithmic probabilities below underflow.
    ndtri_exp(y) {
        let mut x=cephes64::ndtri::ndtri_exp(y);
        if x < -8.0 && x.is_finite() {
            // The original rational tail fit is tuned to |x| <= 64. Refine
            // logarithmically outside that interval without exponentiating y.
            for _ in 0..2 {
                let t= -x*core::f64::consts::FRAC_1_SQRT_2;
                let scaled=erfcx_real(t);
                let residual=(scaled/2.0).ln()-t*t-y;
                if !residual.is_finite() { break; }
                x-=residual/(core::f64::consts::FRAC_2_SQRT_PI*core::f64::consts::FRAC_1_SQRT_2/scaled);
            }
        }
        x
    }
}

real_kernel! {
    /// Voigt profile: convolution of a Gaussian of width sigma and a Cauchy
    /// distribution of half-width gamma. Requires sigma,gamma >= 0.
    voigt_profile(x,sigma,gamma) {
        if sigma<0.0 || gamma<0.0 { f64::NAN }
        else if sigma==0.0 {
            if gamma==0.0 { if x==0.0 { f64::INFINITY } else { 0.0 } }
            else { gamma/(core::f64::consts::PI*(x*x+gamma*gamma)) }
        } else if gamma==0.0 {
            (-0.5*(x/sigma)*(x/sigma)).exp()/(sigma*(2.0*core::f64::consts::PI).sqrt())
        } else {
            let z=num_complex::Complex64::new(x/(sigma*core::f64::consts::SQRT_2),gamma/(sigma*core::f64::consts::SQRT_2));
            crate::backend::faddeeva::w_with_relerror(z,0.0).re/(sigma*(2.0*core::f64::consts::PI).sqrt())
        }
    }
}
