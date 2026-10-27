//! Real special functions backed by double-precision Cephes kernels.
use crate::backend::{
    StrErr,
    utils::{finish, real_kernel},
};
use sci_core::Float;

/// Gamma function, with signed poles at zero. Negative integer arguments are outside the domain.
///
/// Returns an error for NaN arguments or an undefined real result.
pub fn gamma<T: Float>(x: T) -> Result<T, StrErr> {
    if x.is_nan() {
        return Err("gamma: Arguments cannot be NAN.");
    }
    let x = x
        .to_f64()
        .ok_or("gamma: Cannot represent argument as f64.")?;
    if x == 0.0 {
        return finish(f64::INFINITY.copysign(x), "gamma: Undefined result.");
    }
    finish(
        crate::backend::cephes64::gamma(x),
        "gamma: Result is undefined in the real domain.",
    )
}

real_kernel! {
/// Natural logarithm of the absolute Gamma function.
    gammaln(x) { crate::backend::cephes64::lgam(x) }
}

real_kernel! {
/// Reciprocal Gamma function; nonpositive integers are zeros.
    rgamma(x) { crate::backend::cephes64::rgamma(x) }
}

real_kernel! {
/// Logarithmic derivative of Gamma; nonpositive integers are poles.
    digamma(x) {
        if (x+0.504083008264455409).abs()<0.3 {
            crate::backend::complex::digamma_series(num_complex::Complex64::new(x,0.0),-0.504083008264455409,7.2897639029768949e-17).re
        } else { crate::backend::cephes64::psi(x) }
    }
}

real_kernel! {
/// Beta function B(a,b), continued to real arguments where defined.
    beta(a, b) { crate::backend::cephes64::beta(a, b) }
}

real_kernel! {
/// Natural logarithm of the absolute beta function.
    betaln(a, b) { crate::backend::cephes64::lbeta(a, b) }
}

real_kernel! {
/// Regularized lower incomplete Gamma P(a,x), for a >= 0 and x >= 0.
    gammainc(a, x) { crate::backend::cephes64::igam(a, x) }
}

real_kernel! {
/// Regularized upper incomplete Gamma Q(a,x), for a >= 0 and x >= 0.
    gammaincc(a, x) { crate::backend::cephes64::igamc(a, x) }
}

real_kernel! {
/// Inverse of P(a,x) with respect to x, for a > 0 and 0 <= p <= 1.
    gammaincinv(a, p) { crate::backend::cephes64::igami(a, p) }
}

real_kernel! {
/// Inverse of Q(a,x) with respect to x, for a > 0 and 0 <= p <= 1.
    gammainccinv(a, p) { crate::backend::cephes64::igamci(a, p) }
}

real_kernel! {
/// Regularized incomplete beta I_x(a,b), for a,b > 0 and 0 <= x <= 1.
    betainc(a, b, x) { crate::backend::cephes64::incbet(a, b, x) }
}

/// Inverse regularized incomplete beta, for a,b > 0 and 0 <= p <= 1.
///
/// Returns an error for NaN arguments or an undefined real result.
pub fn betaincinv<T: Float>(a: T, b: T, p: T) -> Result<T, StrErr> {
    if a.is_nan() || b.is_nan() || p.is_nan() {
        return Err("betaincinv: Arguments cannot be NAN.");
    }
    let a = a
        .to_f64()
        .ok_or("betaincinv: Cannot represent argument as f64.")?;
    let b = b
        .to_f64()
        .ok_or("betaincinv: Cannot represent argument as f64.")?;
    let p = p
        .to_f64()
        .ok_or("betaincinv: Cannot represent argument as f64.")?;
    if a <= 0.0 || b <= 0.0 || !(0.0..=1.0).contains(&p) {
        return Err("betaincinv: Requires a,b > 0 and 0 <= p <= 1.");
    }
    finish(
        crate::backend::gamma_extra::inverse_beta(a, b, p),
        "betaincinv: Result is undefined in the real domain.",
    )
}

real_kernel! {
/// Real error function.
    erf(x) { crate::backend::cephes64::erf(x) }
}

real_kernel! {
/// Complementary error function, evaluated without subtractive cancellation.
    erfc(x) { crate::backend::cephes64::erfc(x) }
}

real_kernel! {
/// Standard normal cumulative distribution function.
    ndtr(x) { crate::backend::cephes64::ndtr(x) }
}

real_kernel! {
/// Standard normal quantile, for 0 <= p <= 1.
    ndtri(p) { crate::backend::cephes64::ndtri(p) }
}

real_kernel! {
/// Dawson integral exp(-x²) ∫₀ˣ exp(t²) dt.
    dawsn(x) { crate::backend::cephes64::dawsn(x) }
}

/// Bessel J of order zero.
///
/// Returns an error for NaN arguments or an undefined real result.
pub fn cyl_bessel_j0<T: Float>(x: T) -> Result<T, StrErr> {
    if x.is_nan() {
        return Err("cyl_bessel_j0: Arguments cannot be NAN.");
    }
    let x = x
        .to_f64()
        .ok_or("cyl_bessel_j0: Cannot represent argument as f64.")?;
    finish(
        crate::backend::cephes64::j0(x),
        "cyl_bessel_j0: Result is undefined in the real domain.",
    )
}

/// Bessel J of order one.
///
/// Returns an error for NaN arguments or an undefined real result.
pub fn cyl_bessel_j1<T: Float>(x: T) -> Result<T, StrErr> {
    if x.is_nan() {
        return Err("cyl_bessel_j1: Arguments cannot be NAN.");
    }
    let x = x
        .to_f64()
        .ok_or("cyl_bessel_j1: Cannot represent argument as f64.")?;
    if x.abs() < 1e-100 {
        return finish(0.5 * x, "Bessel: Undefined result.");
    }
    finish(
        crate::backend::cephes64::j1(x),
        "cyl_bessel_j1: Result is undefined in the real domain.",
    )
}

real_kernel! {
/// Bessel Y of order zero, for x >= 0.
    cyl_bessel_y0(x) { crate::backend::cephes64::y0(x) }
}

real_kernel! {
/// Bessel Y of order one, for x >= 0.
    cyl_bessel_y1(x) { crate::backend::cephes64::y1(x) }
}

/// Modified Bessel I of order zero.
///
/// Returns an error for NaN arguments or an undefined real result.
pub fn cyl_bessel_i0<T: Float>(x: T) -> Result<T, StrErr> {
    if x.is_nan() {
        return Err("cyl_bessel_i0: Arguments cannot be NAN.");
    }
    let x = x
        .to_f64()
        .ok_or("cyl_bessel_i0: Cannot represent argument as f64.")?;
    if x.is_infinite() {
        return finish(f64::INFINITY, "Bessel: Undefined result.");
    }
    finish(
        crate::backend::cephes64::i0(x),
        "cyl_bessel_i0: Result is undefined in the real domain.",
    )
}

/// Modified Bessel I of order one.
///
/// Returns an error for NaN arguments or an undefined real result.
pub fn cyl_bessel_i1<T: Float>(x: T) -> Result<T, StrErr> {
    if x.is_nan() {
        return Err("cyl_bessel_i1: Arguments cannot be NAN.");
    }
    let x = x
        .to_f64()
        .ok_or("cyl_bessel_i1: Cannot represent argument as f64.")?;
    if x.is_infinite() {
        return finish(f64::INFINITY.copysign(x), "Bessel: Undefined result.");
    }
    finish(
        crate::backend::cephes64::i1(x),
        "cyl_bessel_i1: Result is undefined in the real domain.",
    )
}

/// exp(-|x|) times modified Bessel I of order zero.
///
/// Returns an error for NaN arguments or an undefined real result.
pub fn cyl_bessel_i0e<T: Float>(x: T) -> Result<T, StrErr> {
    if x.is_nan() {
        return Err("cyl_bessel_i0e: Arguments cannot be NAN.");
    }
    let x = x
        .to_f64()
        .ok_or("cyl_bessel_i0e: Cannot represent argument as f64.")?;
    if x == f64::INFINITY {
        return finish(0.0, "Bessel: Undefined result.");
    }
    if x == f64::NEG_INFINITY {
        return finish(0.0, "Bessel: Undefined result.");
    }
    finish(
        crate::backend::cephes64::i0e(x),
        "cyl_bessel_i0e: Result is undefined in the real domain.",
    )
}

/// exp(-|x|) times modified Bessel I of order one.
///
/// Returns an error for NaN arguments or an undefined real result.
pub fn cyl_bessel_i1e<T: Float>(x: T) -> Result<T, StrErr> {
    if x.is_nan() {
        return Err("cyl_bessel_i1e: Arguments cannot be NAN.");
    }
    let x = x
        .to_f64()
        .ok_or("cyl_bessel_i1e: Cannot represent argument as f64.")?;
    if x == f64::INFINITY {
        return finish(0.0, "Bessel: Undefined result.");
    }
    if x == f64::NEG_INFINITY {
        return finish(0.0_f64.copysign(x), "Bessel: Undefined result.");
    }
    finish(
        crate::backend::cephes64::i1e(x),
        "cyl_bessel_i1e: Result is undefined in the real domain.",
    )
}

real_kernel! {
/// Modified Bessel K of order zero, for x >= 0.
    cyl_bessel_k0(x) { crate::backend::cephes64::k0(x) }
}

real_kernel! {
/// Modified Bessel K of order one, for x >= 0.
    cyl_bessel_k1(x) { crate::backend::cephes64::k1(x) }
}

/// exp(x) times modified Bessel K of order zero, for x >= 0.
///
/// Returns an error for NaN arguments or an undefined real result.
pub fn cyl_bessel_k0e<T: Float>(x: T) -> Result<T, StrErr> {
    if x.is_nan() {
        return Err("cyl_bessel_k0e: Arguments cannot be NAN.");
    }
    let x = x
        .to_f64()
        .ok_or("cyl_bessel_k0e: Cannot represent argument as f64.")?;
    if x == f64::INFINITY {
        return finish(0.0, "Bessel: Undefined result.");
    }
    finish(
        crate::backend::cephes64::k0e(x),
        "cyl_bessel_k0e: Result is undefined in the real domain.",
    )
}

/// exp(x) times modified Bessel K of order one, for x >= 0.
///
/// Returns an error for NaN arguments or an undefined real result.
pub fn cyl_bessel_k1e<T: Float>(x: T) -> Result<T, StrErr> {
    if x.is_nan() {
        return Err("cyl_bessel_k1e: Arguments cannot be NAN.");
    }
    let x = x
        .to_f64()
        .ok_or("cyl_bessel_k1e: Cannot represent argument as f64.")?;
    if x == f64::INFINITY {
        return finish(0.0, "Bessel: Undefined result.");
    }
    finish(
        crate::backend::cephes64::k1e(x),
        "cyl_bessel_k1e: Result is undefined in the real domain.",
    )
}

real_kernel! {
/// Real Bessel J_v(x). Negative x requires integer order.
    cyl_bessel_j(v, x) { if v.is_finite() && x.is_finite() { crate::backend::cephes64::jv(v,x) } else { f64::NAN } }
}

real_kernel! {
/// Real Bessel Y_v(x), for x >= 0.
    cyl_bessel_y(v, x) { crate::backend::bessel::real_y(v,x) }
}

real_kernel! {
/// Spence integral Li₂(1-x), for x >= 0.
    spence(x) { crate::backend::cephes64::spence(x) }
}

/// Hurwitz zeta, for s > 1; q may be negative only for integer s.
///
/// Returns an error for NaN arguments or an undefined real result.
pub fn zeta<T: Float>(s: T, q: T) -> Result<T, StrErr> {
    if s.is_nan() || q.is_nan() {
        return Err("zeta: Arguments cannot be NAN.");
    }
    let s = s
        .to_f64()
        .ok_or("zeta: Cannot represent argument as f64.")?;
    let q = q
        .to_f64()
        .ok_or("zeta: Cannot represent argument as f64.")?;
    if q == 1.0 {
        return finish(
            crate::backend::cephes64::riemann_zeta(s),
            "zeta: Undefined result.",
        );
    }
    finish(
        crate::backend::cephes64::zeta(s, q),
        "zeta: Result is undefined in the real domain.",
    )
}

real_kernel! {
/// Riemann zeta minus one, evaluated without cancellation.
    zetac(x) { crate::backend::cephes64::zetac(x) }
}

real_kernel! {
/// Real Riemann zeta function.
    riemann_zeta(x) { crate::backend::cephes64::riemann_zeta(x) }
}

real_kernel! {
/// log Gamma(1+x), accurate near x = 0.
    lgam1p(x) { crate::backend::cephes64::lgam1p(x) }
}

real_kernel! {
/// log(1+x) - x, accurate near zero. Requires x >= -1.
    log1pmx(x) { crate::backend::cephes64::log1pmx(x) }
}

real_kernel! {
/// Chi-squared CDF, for v > 0 and x >= 0.
    chdtr(v, x) { crate::backend::cephes64::chdtr(v, x) }
}

real_kernel! {
/// Chi-squared survival function, for v > 0 and x >= 0.
    chdtrc(v, x) { crate::backend::cephes64::chdtrc(v, x) }
}

real_kernel! {
/// Inverse chi-squared survival function, for v > 0 and 0 <= p <= 1.
    chdtri(v, p) { crate::backend::cephes64::chdtri(v, p) }
}

real_kernel! {
/// F-distribution CDF, for a,b > 0 and x >= 0.
    fdtr(a, b, x) { crate::backend::cephes64::fdtr(a, b, x) }
}

real_kernel! {
/// F-distribution survival function, for a,b > 0 and x >= 0.
    fdtrc(a, b, x) { if !a.is_finite() || !b.is_finite() { f64::NAN } else {crate::backend::cephes64::fdtrc(a, b, x)} }
}

real_kernel! {
/// Inverse F-distribution CDF, for a,b > 0 and 0 <= p <= 1.
    fdtri(a, b, p) { crate::backend::cephes64::fdtri(a, b, p) }
}

/// Gamma distribution CDF, with rate a > 0, shape b > 0, and x >= 0.
///
/// Returns an error for NaN arguments or an undefined real result.
pub fn gdtr<T: Float>(a: T, b: T, x: T) -> Result<T, StrErr> {
    if a.is_nan() || b.is_nan() || x.is_nan() {
        return Err("gdtr: Arguments cannot be NAN.");
    }
    let a = a
        .to_f64()
        .ok_or("gdtr: Cannot represent argument as f64.")?;
    let b = b
        .to_f64()
        .ok_or("gdtr: Cannot represent argument as f64.")?;
    let x = x
        .to_f64()
        .ok_or("gdtr: Cannot represent argument as f64.")?;
    if a <= 0.0 || b <= 0.0 {
        return Err("gdtr: Rate and shape must be positive.");
    }
    finish(
        crate::backend::cephes64::gdtr(a, b, x),
        "gdtr: Result is undefined in the real domain.",
    )
}

/// Gamma distribution survival function, with rate a > 0, shape b > 0, and x >= 0.
///
/// Returns an error for NaN arguments or an undefined real result.
pub fn gdtrc<T: Float>(a: T, b: T, x: T) -> Result<T, StrErr> {
    if a.is_nan() || b.is_nan() || x.is_nan() {
        return Err("gdtrc: Arguments cannot be NAN.");
    }
    let a = a
        .to_f64()
        .ok_or("gdtrc: Cannot represent argument as f64.")?;
    let b = b
        .to_f64()
        .ok_or("gdtrc: Cannot represent argument as f64.")?;
    let x = x
        .to_f64()
        .ok_or("gdtrc: Cannot represent argument as f64.")?;
    if a <= 0.0 || b <= 0.0 {
        return Err("gdtrc: Rate and shape must be positive.");
    }
    finish(
        crate::backend::cephes64::gdtrc(a, b, x),
        "gdtrc: Result is undefined in the real domain.",
    )
}

real_kernel! {
/// Poisson CDF, for k >= 0 and mean m >= 0.
    pdtr(k, m) { crate::backend::cephes64::pdtr(k, m) }
}

real_kernel! {
/// Poisson survival function, for k >= 0 and mean m >= 0.
    pdtrc(k, m) { crate::backend::cephes64::pdtrc(k, m) }
}

/// Airy functions (Ai, Ai′, Bi, Bi′).
pub fn airy<T: Float>(x: T) -> Result<crate::backend::bessel::Airy<T>, StrErr> {
    crate::backend::bessel::airy_real(x, false)
}

/// Exponentially scaled real Airy functions (Ai, Ai′, Bi, Bi′), for x >= 0.
pub fn airye<T: Float>(x: T) -> Result<crate::backend::bessel::Airy<T>, StrErr> {
    crate::backend::bessel::airy_real(x, true)
}

/// Real exp(-|Im(x)|)*J_v(x), identical to J_v(x).
pub fn cyl_bessel_je<T: Float>(v: T, x: T) -> Result<T, StrErr> {
    cyl_bessel_j(v, x)
}
/// Real exp(-|Im(x)|)*Y_v(x), identical to Y_v(x).
pub fn cyl_bessel_ye<T: Float>(v: T, x: T) -> Result<T, StrErr> {
    cyl_bessel_y(v, x)
}

/// Fresnel integrals (S, C), in that order.
pub fn fresnel<T: Float>(x: T) -> Result<(T, T), StrErr> {
    if x.is_nan() {
        return Err("fresnel: Arguments cannot be NAN.");
    }
    let x = x
        .to_f64()
        .ok_or("fresnel: Cannot represent argument as f64.")?;
    let (a, b) = crate::backend::cephes64::fresnl(x);
    Ok((
        finish(a, "fresnel: Undefined real result.")?,
        finish(b, "fresnel: Undefined real result.")?,
    ))
}

/// Sine and cosine integrals (Si, Ci); Ci is the real part on the negative axis.
pub fn sici<T: Float>(x: T) -> Result<(T, T), StrErr> {
    if x.is_nan() {
        return Err("sici: Arguments cannot be NAN.");
    }
    let x = x
        .to_f64()
        .ok_or("sici: Cannot represent argument as f64.")?;
    let (a, b) = crate::backend::cephes64::sici(x);
    Ok((
        finish(a, "sici: Undefined real result.")?,
        finish(b, "sici: Undefined real result.")?,
    ))
}

/// Hyperbolic sine and cosine integrals (Shi, Chi).
pub fn shichi<T: Float>(x: T) -> Result<(T, T), StrErr> {
    if x.is_nan() {
        return Err("shichi: Arguments cannot be NAN.");
    }
    let x = x
        .to_f64()
        .ok_or("shichi: Cannot represent argument as f64.")?;
    let (a, b) = crate::backend::cephes64::shichi(x);
    Ok((
        finish(a, "shichi: Undefined real result.")?,
        finish(b, "shichi: Undefined real result.")?,
    ))
}

/// Generalized exponential integral E_n(x), for n >= 0 and x >= 0.
pub fn expn<T: Float>(n: i32, x: T) -> Result<T, StrErr> {
    if n < 0 || x.is_nan() {
        return Err("expn: Requires n >= 0 and a non-NaN argument.");
    }
    finish(
        crate::backend::cephes64::expn(n, x.to_f64().ok_or("expn: Invalid argument.")?),
        "expn: Requires x >= 0.",
    )
}
