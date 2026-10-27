//! Elliptic integrals reuse Ellip's pure Rust implementations.
use crate::backend::{StrErr, utils::finish};
use sci_core::Float;

pub use ellip::{ellipe, ellipeinc, ellipk, elliprc, elliprd, elliprf, elliprg, elliprj};

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
