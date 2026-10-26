//! # Elliptic integrals
//!
//! Legendre and Carlson integrals evaluated by ellip in `f64` precision.
//! The parameter convention is `m = k²`; amplitudes are radians.
//! These wrappers restrict inputs to finite values and nonsingular real domains.
//! The narrower domains make errors predictable across backend versions.
//!
//! ## Parameter conventions
//!
//! Legendre integrals use the elliptic parameter `m`, the characteristic `n`
//! for third-kind integrals, and the amplitude `phi` in radians for incomplete
//! integrals. Carlson integrals use real arguments `x`, `y`, and `z`;
//! `p` is the additional third-kind argument. No modulus-to-parameter
//! conversion is performed.
//!
//! ## Domain
//!
//! These functions evaluate ordinary real integrals without principal-value
//! continuation. Some backend functions support larger real or principal-value
//! domains; Sci accepts only the finite domains documented below.
//! NaN and infinity inputs are rejected. Numerical evaluation errors use the
//! shared error type; a nonfinite backend result is an evaluation error.
//!
//! ## Numerical behavior
//!
//! Evaluation uses the private ellip backend in binary64 precision. Accuracy
//! depends on the arguments; relative error can grow near zeros and singularities.
//! Sci provides no configurable tolerance or iteration limit for these wrappers.

use crate::{Error, Result};

/// Computes K(m) = ∫₀^(π/2) (1 − m sin²θ)^(-1/2) dθ.
///
/// ## Parameters
///
/// - `m`: Elliptic parameter `m = k²`, where `k` is the modulus.
///
/// ## Domain
///
/// All inputs must be finite. Required domain: m < 1.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] outside the stated domain,
/// [`Error::NumericalFailure`] if the backend cannot evaluate the integral, and
/// [`Error::NonFiniteEvaluation`] if the result is nonfinite.
///
/// # Examples
///
/// ```
/// let value = sci_special::elliptic::ellipk(0.0)?;
/// assert!((value - (std::f64::consts::FRAC_PI_2)).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [NIST DLMF, Chapter 19](https://dlmf.nist.gov/19).
pub fn ellipk(m: f64) -> Result<f64> {
    if !(m.is_finite() && (m < 1.0)) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::ellipk(m).map_err(Error::NumericalFailure)?;
    crate::error::finite(value)
}

/// Computes E(m) = ∫₀^(π/2) √(1 − m sin²θ) dθ.
///
/// ## Parameters
///
/// - `m`: Elliptic parameter `m = k²`, where `k` is the modulus.
///
/// ## Domain
///
/// All inputs must be finite. Required domain: m ≤ 1.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] outside the stated domain,
/// [`Error::NumericalFailure`] if the backend cannot evaluate the integral, and
/// [`Error::NonFiniteEvaluation`] if the result is nonfinite.
///
/// # Examples
///
/// ```
/// let value = sci_special::elliptic::ellipe(0.0)?;
/// assert!((value - (std::f64::consts::FRAC_PI_2)).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [NIST DLMF, Chapter 19](https://dlmf.nist.gov/19).
pub fn ellipe(m: f64) -> Result<f64> {
    if !(m.is_finite() && (m <= 1.0)) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::ellipe(m).map_err(Error::NumericalFailure)?;
    crate::error::finite(value)
}

/// Computes Π(n,m) = ∫₀^(π/2) [(1 − n sin²θ)√(1 − m sin²θ)]⁻¹ dθ.
///
/// ## Parameters
///
/// - `n`: Characteristic of a third-kind integral.
/// - `m`: Elliptic parameter `m = k²`, where `k` is the modulus.
///
/// ## Domain
///
/// All inputs must be finite. Required domain: n < 1 and m < 1.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] outside the stated domain,
/// [`Error::NumericalFailure`] if the backend cannot evaluate the integral, and
/// [`Error::NonFiniteEvaluation`] if the result is nonfinite.
///
/// # Examples
///
/// ```
/// let value = sci_special::elliptic::ellippi(0.0, 0.0)?;
/// assert!((value - (std::f64::consts::FRAC_PI_2)).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [NIST DLMF, Chapter 19](https://dlmf.nist.gov/19).
pub fn ellippi(n: f64, m: f64) -> Result<f64> {
    if !(n.is_finite() && m.is_finite() && (n < 1.0 && m < 1.0)) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::ellippi(n, m).map_err(Error::NumericalFailure)?;
    crate::error::finite(value)
}

/// Computes D(m) = ∫₀^(π/2) sin²θ / √(1 − m sin²θ) dθ.
///
/// ## Parameters
///
/// - `m`: Elliptic parameter `m = k²`, where `k` is the modulus.
///
/// ## Domain
///
/// All inputs must be finite. Required domain: m < 1.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] outside the stated domain,
/// [`Error::NumericalFailure`] if the backend cannot evaluate the integral, and
/// [`Error::NonFiniteEvaluation`] if the result is nonfinite.
///
/// # Examples
///
/// ```
/// let value = sci_special::elliptic::ellipd(0.0)?;
/// assert!((value - (std::f64::consts::FRAC_PI_4)).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [NIST DLMF, Chapter 19](https://dlmf.nist.gov/19).
pub fn ellipd(m: f64) -> Result<f64> {
    if !(m.is_finite() && (m < 1.0)) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::ellipd(m).map_err(Error::NumericalFailure)?;
    crate::error::finite(value)
}

/// Computes F(φ,m) = ∫₀^φ (1 − m sin²θ)^(-1/2) dθ.
///
/// ## Parameters
///
/// - `phi`: Amplitude in radians.
/// - `m`: Elliptic parameter `m = k²`, where `k` is the modulus.
///
/// ## Domain
///
/// All inputs must be finite. Required domain: m < 1, with any finite amplitude φ.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] outside the stated domain,
/// [`Error::NumericalFailure`] if the backend cannot evaluate the integral, and
/// [`Error::NonFiniteEvaluation`] if the result is nonfinite.
///
/// ## Numerical behavior
///
/// Uses the private elliptic-integral backend with amplitude reduction and
/// Carlson symmetric integrals. Work is independent of the amplitude magnitude;
/// backend duplication iterations are bounded. Floating-point rounding affects
/// large-amplitude reduction and values near the singular parameter m=1.
/// No uniform relative-accuracy guarantee is made. This preserves Sci's finite
/// real domain m<1; SciPy's additional m=1 and restricted m>1 cases are unsupported.
///
/// # Examples
///
/// ```
/// let value = sci_special::elliptic::ellipkinc(0.5, 0.0)?;
/// assert!((value - (0.5)).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [NIST DLMF, Chapter 19](https://dlmf.nist.gov/19).
pub fn ellipkinc(phi: f64, m: f64) -> Result<f64> {
    if !(phi.is_finite() && m.is_finite() && (m < 1.0)) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::ellipf(phi, m).map_err(Error::NumericalFailure)?;
    crate::error::finite(value)
}

/// Computes E(φ,m) = ∫₀^φ √(1 − m sin²θ) dθ.
///
/// ## Parameters
///
/// - `phi`: Amplitude in radians.
/// - `m`: Elliptic parameter `m = k²`, where `k` is the modulus.
///
/// ## Domain
///
/// All inputs must be finite. Required domain: m ≤ 1, with any finite amplitude φ.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] outside the stated domain,
/// [`Error::NumericalFailure`] if the backend cannot evaluate the integral, and
/// [`Error::NonFiniteEvaluation`] if the result is nonfinite.
///
/// # Examples
///
/// ```
/// let value = sci_special::elliptic::ellipeinc(0.5, 0.0)?;
/// assert!((value - (0.5)).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [NIST DLMF, Chapter 19](https://dlmf.nist.gov/19).
pub fn ellipeinc(phi: f64, m: f64) -> Result<f64> {
    if !(phi.is_finite() && m.is_finite() && (m <= 1.0)) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::ellipeinc(phi, m).map_err(Error::NumericalFailure)?;
    crate::error::finite(value)
}

/// Computes Π(φ,n,m) = ∫₀^φ [(1 − n sin²θ)√(1 − m sin²θ)]⁻¹ dθ.
///
/// ## Parameters
///
/// - `phi`: Amplitude in radians.
/// - `n`: Characteristic of a third-kind integral.
/// - `m`: Elliptic parameter `m = k²`, where `k` is the modulus.
///
/// ## Domain
///
/// All inputs must be finite. Required domain: m < 1, n < 1, with any finite amplitude φ.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] outside the stated domain,
/// [`Error::NumericalFailure`] if the backend cannot evaluate the integral, and
/// [`Error::NonFiniteEvaluation`] if the result is nonfinite.
///
/// # Examples
///
/// ```
/// let value = sci_special::elliptic::ellippiinc(0.5, 0.0, 0.0)?;
/// assert!((value - (0.5)).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [NIST DLMF, Chapter 19](https://dlmf.nist.gov/19).
pub fn ellippiinc(phi: f64, n: f64, m: f64) -> Result<f64> {
    if !(phi.is_finite() && n.is_finite() && m.is_finite() && (m < 1.0 && n < 1.0)) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::ellippiinc(phi, n, m).map_err(Error::NumericalFailure)?;
    crate::error::finite(value)
}

/// Computes D(φ,m) = ∫₀^φ sin²θ / √(1 − m sin²θ) dθ.
///
/// ## Parameters
///
/// - `phi`: Amplitude in radians.
/// - `m`: Elliptic parameter `m = k²`, where `k` is the modulus.
///
/// ## Domain
///
/// All inputs must be finite. Required domain: m < 1, with any finite amplitude φ.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] outside the stated domain,
/// [`Error::NumericalFailure`] if the backend cannot evaluate the integral, and
/// [`Error::NonFiniteEvaluation`] if the result is nonfinite.
///
/// # Examples
///
/// ```
/// let value = sci_special::elliptic::ellipdinc(0.0, 0.0)?;
/// assert!((value - (0.0)).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [NIST DLMF, Chapter 19](https://dlmf.nist.gov/19).
pub fn ellipdinc(phi: f64, m: f64) -> Result<f64> {
    if !(phi.is_finite() && m.is_finite() && (m < 1.0)) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::ellipdinc(phi, m).map_err(Error::NumericalFailure)?;
    crate::error::finite(value)
}

/// Computes R_F(x,y,z) = ½ ∫₀^∞ [(t+x)(t+y)(t+z)]^(-1/2) dt.
///
/// ## Parameters
///
/// - `x`: Real Carlson argument.
/// - `y`: Real Carlson argument.
/// - `z`: Real Carlson argument.
///
/// ## Domain
///
/// All inputs must be finite. Required domain: x,y,z ≥ 0 with at most one zero.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] outside the stated domain,
/// [`Error::NumericalFailure`] if the backend cannot evaluate the integral, and
/// [`Error::NonFiniteEvaluation`] if the result is nonfinite.
///
/// # Examples
///
/// ```
/// let value = sci_special::elliptic::elliprf(1.0, 1.0, 1.0)?;
/// assert!((value - (1.0)).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [NIST DLMF, Chapter 19](https://dlmf.nist.gov/19).
pub fn elliprf(x: f64, y: f64, z: f64) -> Result<f64> {
    if !(x.is_finite()
        && y.is_finite()
        && z.is_finite()
        && (x >= 0.0
            && y >= 0.0
            && z >= 0.0
            && ((x == 0.0) as u8 + (y == 0.0) as u8 + (z == 0.0) as u8 <= 1)))
    {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::elliprf(x, y, z).map_err(Error::NumericalFailure)?;
    crate::error::finite(value)
}

/// Computes R_D(x,y,z) = R_J(x,y,z,z).
///
/// ## Parameters
///
/// - `x`: Real Carlson argument.
/// - `y`: Real Carlson argument.
/// - `z`: Real Carlson argument.
///
/// ## Domain
///
/// All inputs must be finite. Required domain: x,y ≥ 0, x+y > 0, and z > 0.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] outside the stated domain,
/// [`Error::NumericalFailure`] if the backend cannot evaluate the integral, and
/// [`Error::NonFiniteEvaluation`] if the result is nonfinite.
///
/// # Examples
///
/// ```
/// let value = sci_special::elliptic::elliprd(1.0, 1.0, 1.0)?;
/// assert!((value - (1.0)).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [NIST DLMF, Chapter 19](https://dlmf.nist.gov/19).
pub fn elliprd(x: f64, y: f64, z: f64) -> Result<f64> {
    if !(x.is_finite()
        && y.is_finite()
        && z.is_finite()
        && (x >= 0.0 && y >= 0.0 && x + y > 0.0 && z > 0.0))
    {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::elliprd(x, y, z).map_err(Error::NumericalFailure)?;
    crate::error::finite(value)
}

/// Computes Carlson's symmetric elliptic integral R_G(x,y,z).
///
/// ## Parameters
///
/// - `x`: Real Carlson argument.
/// - `y`: Real Carlson argument.
/// - `z`: Real Carlson argument.
///
/// ## Domain
///
/// All inputs must be finite. Required domain: x,y,z ≥ 0.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] outside the stated domain,
/// [`Error::NumericalFailure`] if the backend cannot evaluate the integral, and
/// [`Error::NonFiniteEvaluation`] if the result is nonfinite.
///
/// # Examples
///
/// ```
/// let value = sci_special::elliptic::elliprg(1.0, 1.0, 1.0)?;
/// assert!((value - (1.0)).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [NIST DLMF, Chapter 19](https://dlmf.nist.gov/19).
pub fn elliprg(x: f64, y: f64, z: f64) -> Result<f64> {
    if !(x.is_finite() && y.is_finite() && z.is_finite() && (x >= 0.0 && y >= 0.0 && z >= 0.0)) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::elliprg(x, y, z).map_err(Error::NumericalFailure)?;
    crate::error::finite(value)
}

/// Computes R_J(x,y,z,p) = (3/2) ∫₀^∞ [(t+p)√((t+x)(t+y)(t+z))]⁻¹ dt.
///
/// ## Parameters
///
/// - `x`: Real Carlson argument.
/// - `y`: Real Carlson argument.
/// - `z`: Real Carlson argument.
/// - `p`: Additional Carlson third-kind argument.
///
/// ## Domain
///
/// All inputs must be finite. Required domain: x,y,z ≥ 0 with at most one zero and p > 0.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] outside the stated domain,
/// [`Error::NumericalFailure`] if the backend cannot evaluate the integral, and
/// [`Error::NonFiniteEvaluation`] if the result is nonfinite.
///
/// # Examples
///
/// ```
/// let value = sci_special::elliptic::elliprj(1.0, 1.0, 1.0, 1.0)?;
/// assert!((value - (1.0)).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [NIST DLMF, Chapter 19](https://dlmf.nist.gov/19).
pub fn elliprj(x: f64, y: f64, z: f64, p: f64) -> Result<f64> {
    if !(x.is_finite()
        && y.is_finite()
        && z.is_finite()
        && p.is_finite()
        && (x >= 0.0
            && y >= 0.0
            && z >= 0.0
            && ((x == 0.0) as u8 + (y == 0.0) as u8 + (z == 0.0) as u8 <= 1)
            && p > 0.0))
    {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::elliprj(x, y, z, p).map_err(Error::NumericalFailure)?;
    crate::error::finite(value)
}

/// Computes R_C(x,y) = R_F(x,y,y).
///
/// ## Parameters
///
/// - `x`: Real Carlson argument.
/// - `y`: Real Carlson argument.
///
/// ## Domain
///
/// All inputs must be finite. Required domain: x ≥ 0 and y > 0.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] outside the stated domain,
/// [`Error::NumericalFailure`] if the backend cannot evaluate the integral, and
/// [`Error::NonFiniteEvaluation`] if the result is nonfinite.
///
/// # Examples
///
/// ```
/// let value = sci_special::elliptic::elliprc(1.0, 1.0)?;
/// assert!((value - (1.0)).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [NIST DLMF, Chapter 19](https://dlmf.nist.gov/19).
pub fn elliprc(x: f64, y: f64) -> Result<f64> {
    if !(x.is_finite() && y.is_finite() && (x >= 0.0 && y > 0.0)) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::elliprc(x, y).map_err(Error::NumericalFailure)?;
    crate::error::finite(value)
}

/// Computes (K(m), E(m)), evaluated together.
///
/// ## Parameters
///
/// - `m`: Elliptic parameter `m = k²`, where `k` is the modulus.
///
/// ## Domain
///
/// Finite `m < 1`; `m = k²` is the parameter.
///
/// ## Returns
///
/// The pair `(K(m), E(m))`, in first-kind, second-kind order.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid arguments,
/// [`Error::NumericalFailure`] on backend failure, and
/// [`Error::NonFiniteEvaluation`] if evaluation overflows.
///
/// # Examples
///
/// ```
/// let value = sci_special::elliptic::ellipke(0.0)?;
/// let (k, e) = value; assert!((k-e).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [NIST DLMF, elliptic integrals](https://dlmf.nist.gov/19).
pub fn ellipke(m: f64) -> Result<(f64, f64)> {
    if !(m.is_finite() && m < 1.0) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::ellipke(m).map_err(Error::NumericalFailure)?;
    Ok((
        crate::error::finite(value.0)?,
        crate::error::finite(value.1)?,
    ))
}

/// Computes ∫₀^(π/2) (a cos²θ + b sin²θ)/((cos²θ+p sin²θ)√(cos²θ+kc² sin²θ)) dθ.
///
/// ## Parameters
///
/// - `kc`: Complementary modulus; `kc² = 1 - m`. Its sign does not affect the integral.
/// - `p`: Bulirsch parameter `p = 1 - n`, where `n` is the characteristic.
/// - `a`: Coefficient of the cosine-squared term.
/// - `b`: Coefficient of the sine-squared term.
///
/// ## Domain
///
/// Finite arguments, nonzero complementary modulus `kc`, and `p > 0`. Negative `kc` gives the same value as positive `kc`. Principal values are excluded.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid arguments,
/// [`Error::NumericalFailure`] on backend failure, and
/// [`Error::NonFiniteEvaluation`] if evaluation overflows.
///
/// # Examples
///
/// ```
/// let value = sci_special::elliptic::cel(1.0, 1.0, 1.0, 1.0)?;
/// assert!((value-std::f64::consts::FRAC_PI_2).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [NIST DLMF, elliptic integrals](https://dlmf.nist.gov/19).
pub fn cel(kc: f64, p: f64, a: f64, b: f64) -> Result<f64> {
    if !(kc.is_finite() && kc != 0.0 && p.is_finite() && p > 0.0 && a.is_finite() && b.is_finite())
    {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::cel(kc, p, a, b).map_err(Error::NumericalFailure)?;
    crate::error::finite(value)
}

/// Computes K(1−kc²).
///
/// ## Parameters
///
/// - `kc`: Complementary modulus; `kc² = 1 - m`. Its sign does not affect the integral.
///
/// ## Domain
///
/// Finite, nonzero complementary modulus `kc`.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid arguments,
/// [`Error::NumericalFailure`] on backend failure, and
/// [`Error::NonFiniteEvaluation`] if evaluation overflows.
///
/// # Examples
///
/// ```
/// let value = sci_special::elliptic::cel1(1.0)?;
/// assert!((value-std::f64::consts::FRAC_PI_2).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [NIST DLMF, elliptic integrals](https://dlmf.nist.gov/19).
pub fn cel1(kc: f64) -> Result<f64> {
    if !(kc.is_finite() && kc != 0.0) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::cel1(kc).map_err(Error::NumericalFailure)?;
    crate::error::finite(value)
}

/// Computes cel(kc,1,a,b).
///
/// ## Parameters
///
/// - `kc`: Complementary modulus; `kc² = 1 - m`. Its sign does not affect the integral.
/// - `a`: Coefficient of the cosine-squared term.
/// - `b`: Coefficient of the sine-squared term.
///
/// ## Domain
///
/// Finite `a`, `b`, and nonzero complementary modulus `kc`.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid arguments,
/// [`Error::NumericalFailure`] on backend failure, and
/// [`Error::NonFiniteEvaluation`] if evaluation overflows.
///
/// # Examples
///
/// ```
/// let value = sci_special::elliptic::cel2(1.0, 1.0, 1.0)?;
/// assert!((value-std::f64::consts::FRAC_PI_2).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [NIST DLMF, elliptic integrals](https://dlmf.nist.gov/19).
pub fn cel2(kc: f64, a: f64, b: f64) -> Result<f64> {
    if !(kc.is_finite() && kc != 0.0 && a.is_finite() && b.is_finite()) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::cel2(kc, a, b).map_err(Error::NumericalFailure)?;
    crate::error::finite(value)
}

/// Computes Π(1−p,1−kc²) = cel(kc,p,1,1).
///
/// ## Parameters
///
/// - `kc`: Complementary modulus; `kc² = 1 - m`. Its sign does not affect the integral.
/// - `p`: Bulirsch parameter `p = 1 - n`, where `n` is the characteristic.
///
/// ## Domain
///
/// Finite, nonzero complementary modulus `kc` and `p > 0`. Principal values are excluded.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid arguments,
/// [`Error::NumericalFailure`] on backend failure, and
/// [`Error::NonFiniteEvaluation`] if evaluation overflows.
///
/// # Examples
///
/// ```
/// let value = sci_special::elliptic::cel3(1.0, 1.0)?;
/// assert!((value-std::f64::consts::FRAC_PI_2).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [NIST DLMF, elliptic integrals](https://dlmf.nist.gov/19).
pub fn cel3(kc: f64, p: f64) -> Result<f64> {
    if !(kc.is_finite() && kc != 0.0 && p.is_finite() && p > 0.0) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::cel3(kc, p).map_err(Error::NumericalFailure)?;
    crate::error::finite(value)
}

/// Computes F(arctan(x),1−kc²).
///
/// ## Parameters
///
/// - `x`: Tangent amplitude; the upper integration angle is `atan(x)` in radians.
/// - `kc`: Complementary modulus; `kc² = 1 - m`. Its sign does not affect the integral.
///
/// ## Domain
///
/// Finite tangent amplitude `x` and nonzero complementary modulus `kc`.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid arguments,
/// [`Error::NumericalFailure`] on backend failure, and
/// [`Error::NonFiniteEvaluation`] if evaluation overflows.
///
/// # Examples
///
/// ```
/// let value = sci_special::elliptic::el1(0.5, 1.0)?;
/// assert!((value-0.5_f64.atan()).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [NIST DLMF, elliptic integrals](https://dlmf.nist.gov/19).
pub fn el1(x: f64, kc: f64) -> Result<f64> {
    if !(x.is_finite() && kc.is_finite() && kc != 0.0) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::el1(x, kc).map_err(Error::NumericalFailure)?;
    crate::error::finite(value)
}

/// Computes ∫₀^arctan(x) (a cos²θ+b sin²θ)/√(cos²θ+kc² sin²θ) dθ.
///
/// ## Parameters
///
/// - `x`: Tangent amplitude; the upper integration angle is `atan(x)` in radians.
/// - `kc`: Complementary modulus; `kc² = 1 - m`. Its sign does not affect the integral.
/// - `a`: Coefficient of the cosine-squared term.
/// - `b`: Coefficient of the sine-squared term.
///
/// ## Domain
///
/// All arguments finite; `kc` is a nonzero complementary modulus, `x` is a tangent amplitude.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid arguments,
/// [`Error::NumericalFailure`] on backend failure, and
/// [`Error::NonFiniteEvaluation`] if evaluation overflows.
///
/// # Examples
///
/// ```
/// let value = sci_special::elliptic::el2(0.5, 1.0, 1.0, 1.0)?;
/// assert!((value-0.5_f64.atan()).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [NIST DLMF, elliptic integrals](https://dlmf.nist.gov/19).
pub fn el2(x: f64, kc: f64, a: f64, b: f64) -> Result<f64> {
    if !(x.is_finite() && kc.is_finite() && kc != 0.0 && a.is_finite() && b.is_finite()) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::el2(x, kc, a, b).map_err(Error::NumericalFailure)?;
    crate::error::finite(value)
}

/// Computes Π(1−p; arctan(x),1−kc²).
///
/// ## Parameters
///
/// - `x`: Tangent amplitude; the upper integration angle is `atan(x)` in radians.
/// - `kc`: Complementary modulus; `kc² = 1 - m`. Its sign does not affect the integral.
/// - `p`: Bulirsch parameter `p = 1 - n`, where `n` is the characteristic.
///
/// ## Domain
///
/// All arguments finite, `kc != 0`, and `p >= 0`. No poles are crossed; principal values are excluded.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid arguments,
/// [`Error::NumericalFailure`] on backend failure, and
/// [`Error::NonFiniteEvaluation`] if evaluation overflows.
///
/// # Examples
///
/// ```
/// let value = sci_special::elliptic::el3(0.5, 1.0, 1.0)?;
/// assert!((value-0.5_f64.atan()).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [NIST DLMF, elliptic integrals](https://dlmf.nist.gov/19).
pub fn el3(x: f64, kc: f64, p: f64) -> Result<f64> {
    if !(x.is_finite() && kc.is_finite() && kc != 0.0 && p.is_finite() && p >= 0.0) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::el3(x, kc, p).map_err(Error::NumericalFailure)?;
    crate::error::finite(value)
}

/// Computes Z(φ,m) = E(φ,m) − E(m)F(φ,m)/K(m).
///
/// ## Parameters
///
/// - `phi`: Amplitude in radians.
/// - `m`: Elliptic parameter `m = k²`, where `k` is the modulus.
///
/// ## Domain
///
/// Finite amplitude `phi` in radians and finite parameter `m < 1`.
///
/// ## Returns
///
/// The real value of the Jacobi zeta function defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid arguments,
/// [`Error::NumericalFailure`] on backend failure, and
/// [`Error::NonFiniteEvaluation`] if evaluation overflows.
///
/// # Examples
///
/// ```
/// let value = sci_special::elliptic::jacobi_zeta(0.5, 0.0)?;
/// assert!(value.abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [NIST DLMF, elliptic integrals](https://dlmf.nist.gov/19).
pub fn jacobi_zeta(phi: f64, m: f64) -> Result<f64> {
    if !(phi.is_finite() && m.is_finite() && m < 1.0) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::jacobi_zeta(phi, m).map_err(Error::NumericalFailure)?;
    crate::error::finite(value)
}

/// Computes Λ₀(φ,m) = F(φ,1−m)/K(1−m) + 2 K(m) Z(φ,1−m)/π.
///
/// ## Parameters
///
/// - `phi`: Amplitude in radians.
/// - `m`: Elliptic parameter `m = k²`, where `k` is the modulus.
///
/// ## Domain
///
/// Finite amplitude `phi` in radians and parameter `m` in `[0,1)`.
///
/// ## Returns
///
/// The real value of Heuman’s lambda function defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid arguments,
/// [`Error::NumericalFailure`] on backend failure, and
/// [`Error::NonFiniteEvaluation`] if evaluation overflows.
///
/// # Examples
///
/// ```
/// let value = sci_special::elliptic::heuman_lambda(0.5, 0.0)?;
/// assert!((value-0.5_f64.sin()).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [NIST DLMF, elliptic integrals](https://dlmf.nist.gov/19).
pub fn heuman_lambda(phi: f64, m: f64) -> Result<f64> {
    if !(phi.is_finite() && m.is_finite() && (0.0..1.0).contains(&m)) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::heuman_lambda(phi, m).map_err(Error::NumericalFailure)?;
    crate::error::finite(value)
}

/// Computes Π(n;φ,m), using Bulirsch's algorithm.
///
/// ## Parameters
///
/// - `phi`: Amplitude in radians.
/// - `n`: Characteristic of a third-kind integral.
/// - `m`: Elliptic parameter `m = k²`, where `k` is the modulus.
///
/// ## Domain
///
/// Finite radian amplitude `phi`, characteristic `n < 1`, and parameter `m < 1`.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid arguments,
/// [`Error::NumericalFailure`] on backend failure, and
/// [`Error::NonFiniteEvaluation`] if evaluation overflows.
///
/// # Examples
///
/// ```
/// let value = sci_special::elliptic::ellippiinc_bulirsch(0.5, 0.0, 0.0)?;
/// assert!((value-0.5).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [NIST DLMF, elliptic integrals](https://dlmf.nist.gov/19).
pub fn ellippiinc_bulirsch(phi: f64, n: f64, m: f64) -> Result<f64> {
    if !(phi.is_finite() && n.is_finite() && n < 1.0 && m.is_finite() && m < 1.0) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::ellippiinc_bulirsch(phi, n, m).map_err(Error::NumericalFailure)?;
    crate::error::finite(value)
}

/// Computes K(1−p) accurately near the logarithmic singularity.
///
/// ## Parameters
///
/// - `p`: Complementary parameter `p = 1 - m`.
///
/// ## Domain
///
/// `p = 1−m` is the **complementary parameter**, and must be finite and positive.
/// The Carlson identity K(1−p) = R_F(0,p,1) avoids subtracting a tiny `p` from one.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for nonpositive or nonfinite `p`.
/// Numerical failures follow [`elliprf`].
///
/// # Examples
///
/// ```
/// let k = sci_special::elliptic::ellipkm1(1e-20)?;
/// assert!(k > 20.0);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [NIST DLMF, Carlson integrals](https://dlmf.nist.gov/19.25).
pub fn ellipkm1(p: f64) -> Result<f64> {
    if !p.is_finite() || p <= 0.0 {
        return Err(Error::InvalidInput(
            "complementary parameter must be finite and positive",
        ));
    }
    elliprf(0.0, p, 1.0)
}
