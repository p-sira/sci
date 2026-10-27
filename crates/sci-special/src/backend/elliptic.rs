//! Elliptic integrals reuse Ellip's pure Rust implementations.
use crate::backend::{StrErr, utils::finish};
use sci_core::Float;

pub use ellip::{ellipe, ellipeinc, ellipk, elliprc, elliprd, elliprf, elliprg, elliprj};

/// Incomplete elliptic integral of the first kind F(phi | m).
pub fn ellipkinc<T: Float>(phi: T, m: T) -> Result<T, StrErr> {
    ellip::ellipf(phi, m)
}

/// Complete elliptic integral K(1-p), for the complementary parameter p >= 0.
/// Retains precision when p is too small for 1-p to be represented.
pub fn ellipkm1<T: Float>(p: T) -> Result<T, StrErr> {
    if p.is_nan() || p < T::zero() {
        return Err("ellipkm1: Requires a nonnegative complementary parameter.");
    }
    if p == T::zero() {
        return Ok(T::infinity());
    }
    if p.is_infinite() {
        return Ok(T::zero());
    }
    ellip::elliprf(T::zero(), p, T::one())
}

/// Jacobian elliptic functions (sn, cn, dn, amplitude), for 0 <= m <= 1.
pub fn ellipj<T: Float>(u: T, m: T) -> Result<(T, T, T, T), StrErr> {
    if !u.is_finite() || !m.is_finite() {
        return Err("ellipj: Arguments must be finite.");
    }
    let u = u.to_f64().ok_or("ellipj: Invalid argument.")?;
    let m = m.to_f64().ok_or("ellipj: Invalid argument.")?;
    let (sn, cn, dn, ph) = crate::backend::cephes64::ellpj(u, m);
    Ok((
        finish(sn, "ellipj: Requires 0 <= m <= 1.")?,
        finish(cn, "ellipj: Undefined result.")?,
        finish(dn, "ellipj: Undefined result.")?,
        finish(ph, "ellipj: Undefined result.")?,
    ))
}
