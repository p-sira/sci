//! # Elliptic integrals
//!
//! Legendre, Carlson, and Bulirsch integrals use Ellip's notation and argument
//! ordering. Evaluation runs in the selected `sci_core::Float` precision.
//!
//! ## Parameters and conventions
//!
//! `m` is the elliptic parameter (`m = k²`), `n` is the third-kind characteristic,
//! and `phi` is an amplitude in radians. The incomplete Legendre integrals are
//! `F(phi, m)`, `E(phi, m)`, `Π(phi, n, m)`, and `D(phi, m)`; [`ellipf`] evaluates
//! `F`. [`ellipkinc`] remains an alias. Carlson arguments are `x`, `y`, `z`, and
//! `p`. Bulirsch uses the complementary modulus `kc`, with `kc² = 1 − m`, the
//! tangent amplitude `x = tan(phi)`, and `p = 1 − n`. These parameters are
//! dimensionless. Negative `m` is supported even though it has no real modulus.
//!
//! ## Domain and special values
//!
//! Sci accepts finite inputs and finite real results. NaN and infinity inputs
//! return [`crate::Error::InvalidInput`]; divergent poles or overflow may return a
//! backend error or [`crate::Error::NonFiniteEvaluation`]. Ellip's supported Cauchy
//! principal values are available for third-kind integrals and [`elliprc`].
//! No complex continuation is performed. Each function documents its real domain.
//!
//! SciPy supplies independent references on shared domains. Full SciPy parity is
//! not a design requirement; Ellip defines the elliptic notation and supported
//! algorithms, including real and principal-value cases beyond the old wrappers.
//!
//! ## Numerical behavior
//!
//! The checked Ellip algorithms use series, transformations, and bounded
//! duplication or Bulirsch iterations. Sci exposes no tolerance or iteration
//! controls. Computation stays in `T`; no binary64 promotion is imposed here.
//! Relative accuracy can deteriorate near zeros and poles, and large-amplitude
//! reduction is limited by the selected floating-point precision.
//!
//! ## References
//!
//! - [Ellip 1.2.3](https://docs.rs/ellip/1.2.3/ellip/).
//! - [NIST DLMF, Chapter 19](https://dlmf.nist.gov/19).

use crate::{Error, Result};
use sci_core::Float;

/// Computes K(m) = ∫₀^(π/2) dθ / √(1 − m sin²θ).
///
/// ## Parameters
///
/// - `m`: Elliptic parameter `m = k²`, where `k` is the modulus.
///
/// ## Domain and special values
///
/// Finite `m < 1`. The pole at `m = 1` is rejected; negative parameters are supported.
///
/// NaN and infinity inputs return an error; a nonfinite result is rejected.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] when a wrapper input check fails,
/// [`Error::NumericalFailure`] if Ellip rejects the domain or cannot evaluate the integral, and
/// [`Error::NonFiniteEvaluation`] if the result is nonfinite.
///
/// ## Numerical behavior
///
/// Evaluates in `T` using Ellip's checked algorithms. No tolerance or iteration
/// controls are exposed. Relative accuracy can degrade near zeros or poles;
/// amplitude reduction, where applicable, is limited by floating-point precision.
/// See the module documentation for shared numerical conventions.
///
/// ## Examples
///
/// ```
/// let value = sci_special::elliptic::ellipk(0.0)?;
/// assert!((value - (std::f64::consts::FRAC_PI_2)).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [Ellip `ellipk`](https://docs.rs/ellip/1.2.3/ellip/fn.ellipk.html).
/// - [NIST DLMF, Chapter 19](https://dlmf.nist.gov/19).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn ellipk<T: Float>(m: T) -> Result<T> {
    if !(m.is_finite() && (m < 1.0)) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::ellipk(m).map_err(Error::NumericalFailure)?;
    match value {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}

/// Computes E(m) = ∫₀^(π/2) √(1 − m sin²θ) dθ.
///
/// ## Parameters
///
/// - `m`: Elliptic parameter `m = k²`, where `k` is the modulus.
///
/// ## Domain and special values
///
/// Finite `m <= 1`; `E(1) = 1`. Negative parameters are supported.
///
/// NaN and infinity inputs return an error; a nonfinite result is rejected.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] when a wrapper input check fails,
/// [`Error::NumericalFailure`] if Ellip rejects the domain or cannot evaluate the integral, and
/// [`Error::NonFiniteEvaluation`] if the result is nonfinite.
///
/// ## Numerical behavior
///
/// Evaluates in `T` using Ellip's checked algorithms. No tolerance or iteration
/// controls are exposed. Relative accuracy can degrade near zeros or poles;
/// amplitude reduction, where applicable, is limited by floating-point precision.
/// See the module documentation for shared numerical conventions.
///
/// ## Examples
///
/// ```
/// let value = sci_special::elliptic::ellipe(0.0)?;
/// assert!((value - (std::f64::consts::FRAC_PI_2)).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [Ellip `ellipe`](https://docs.rs/ellip/1.2.3/ellip/fn.ellipe.html).
/// - [NIST DLMF, Chapter 19](https://dlmf.nist.gov/19).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn ellipe<T: Float>(m: T) -> Result<T> {
    if !(m.is_finite() && (m <= 1.0)) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::ellipe(m).map_err(Error::NumericalFailure)?;
    match value {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}

/// Computes Π(n, m) = ∫₀^(π/2) dθ / [(1 − n sin²θ)√(1 − m sin²θ)].
///
/// ## Parameters
///
/// - `n`: Characteristic of a third-kind integral.
/// - `m`: Elliptic parameter `m = k²`, where `k` is the modulus.
///
/// ## Domain and special values
///
/// Finite `m < 1` and `n != 1`. For `n > 1`, returns the Cauchy principal value across the
/// interior pole.
///
/// NaN and infinity inputs return an error; a nonfinite result is rejected.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] when a wrapper input check fails,
/// [`Error::NumericalFailure`] if Ellip rejects the domain or cannot evaluate the integral, and
/// [`Error::NonFiniteEvaluation`] if the result is nonfinite.
///
/// ## Numerical behavior
///
/// Evaluates in `T` using Ellip's checked algorithms. No tolerance or iteration
/// controls are exposed. Relative accuracy can degrade near zeros or poles;
/// amplitude reduction, where applicable, is limited by floating-point precision.
/// See the module documentation for shared numerical conventions.
///
/// ## Examples
///
/// ```
/// let value = sci_special::elliptic::ellippi(0.0, 0.0)?;
/// assert!((value - (std::f64::consts::FRAC_PI_2)).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [Ellip `ellippi`](https://docs.rs/ellip/1.2.3/ellip/fn.ellippi.html).
/// - [NIST DLMF, Chapter 19](https://dlmf.nist.gov/19).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn ellippi<T: Float>(n: T, m: T) -> Result<T> {
    if !(n.is_finite() && m.is_finite() && (n != 1.0 && m < 1.0)) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::ellippi(n, m).map_err(Error::NumericalFailure)?;
    match value {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}

/// Computes D(m) = ∫₀^(π/2) sin²θ dθ / √(1 − m sin²θ).
///
/// ## Parameters
///
/// - `m`: Elliptic parameter `m = k²`, where `k` is the modulus.
///
/// ## Domain and special values
///
/// Finite `m < 1`; `D(0) = π/4`. The singular endpoint `m = 1` is rejected.
///
/// NaN and infinity inputs return an error; a nonfinite result is rejected.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] when a wrapper input check fails,
/// [`Error::NumericalFailure`] if Ellip rejects the domain or cannot evaluate the integral, and
/// [`Error::NonFiniteEvaluation`] if the result is nonfinite.
///
/// ## Numerical behavior
///
/// Evaluates in `T` using Ellip's checked algorithms. No tolerance or iteration
/// controls are exposed. Relative accuracy can degrade near zeros or poles;
/// amplitude reduction, where applicable, is limited by floating-point precision.
/// See the module documentation for shared numerical conventions.
///
/// ## Examples
///
/// ```
/// let value = sci_special::elliptic::ellipd(0.0)?;
/// assert!((value - (std::f64::consts::FRAC_PI_4)).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [Ellip `ellipd`](https://docs.rs/ellip/1.2.3/ellip/fn.ellipd.html).
/// - [NIST DLMF, Chapter 19](https://dlmf.nist.gov/19).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn ellipd<T: Float>(m: T) -> Result<T> {
    if !(m.is_finite() && (m < 1.0)) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::ellipd(m).map_err(Error::NumericalFailure)?;
    match value {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}

/// Computes F(phi, m) = ∫₀^phi dθ / √(1 − m sin²θ).
///
/// ## Parameters
///
/// - `phi`: Amplitude in radians.
/// - `m`: Elliptic parameter `m = k²`, where `k` is the modulus.
///
/// ## Domain and special values
///
/// Finite arguments. For `m < 1`, any real amplitude is accepted. For `m >= 1`, the path must
/// stay before the first pole or nonreal interval: `abs(phi) < π/2` and `m sin²(phi) < 1`.
/// Negative amplitudes give the odd continuation.
///
/// NaN and infinity inputs return an error; a nonfinite result is rejected.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] when a wrapper input check fails,
/// [`Error::NumericalFailure`] if Ellip rejects the domain or cannot evaluate the integral, and
/// [`Error::NonFiniteEvaluation`] if the result is nonfinite.
///
/// ## Numerical behavior
///
/// Evaluates in `T` using Ellip's checked algorithms. No tolerance or iteration
/// controls are exposed. Relative accuracy can degrade near zeros or poles;
/// amplitude reduction, where applicable, is limited by floating-point precision.
/// See the module documentation for shared numerical conventions.
///
/// ## Examples
///
/// ```
/// let phi = 0.5_f64;
/// let value = sci_special::elliptic::ellipf(phi, 1.0)?;
/// assert!((value - phi.sin().atanh()).abs() < 1e-14);
/// // A restricted real path also exists for m > 1.
/// assert!(sci_special::elliptic::ellipf(0.4_f64, 2.0)?.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [Ellip `ellipf`](https://docs.rs/ellip/1.2.3/ellip/fn.ellipf.html).
/// - [NIST DLMF, Chapter 19](https://dlmf.nist.gov/19).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn ellipf<T: Float>(phi: T, m: T) -> Result<T> {
    if !(phi.is_finite() && m.is_finite()) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::ellipf(phi, m).map_err(Error::NumericalFailure)?;
    match value {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}

/// Compatibility name for [`ellipf`], with identical arguments, domain, and errors.
/// The mathematical notation is `F(phi, m)`.
pub use ellipf as ellipkinc;

/// Computes E(phi, m) = ∫₀^phi √(1 − m sin²θ) dθ.
///
/// ## Parameters
///
/// - `phi`: Amplitude in radians.
/// - `m`: Elliptic parameter `m = k²`, where `k` is the modulus.
///
/// ## Domain and special values
///
/// Finite arguments. For `m <= 1`, any real amplitude is accepted. For `m > 1`, the
/// integration path must stay real, within the first turning point (`abs(phi) <=
/// asin(1/sqrt(m))`). Backend failures at limiting endpoints are returned as errors.
///
/// NaN and infinity inputs return an error; a nonfinite result is rejected.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] when a wrapper input check fails,
/// [`Error::NumericalFailure`] if Ellip rejects the domain or cannot evaluate the integral, and
/// [`Error::NonFiniteEvaluation`] if the result is nonfinite.
///
/// ## Numerical behavior
///
/// Evaluates in `T` using Ellip's checked algorithms. No tolerance or iteration
/// controls are exposed. Relative accuracy can degrade near zeros or poles;
/// amplitude reduction, where applicable, is limited by floating-point precision.
/// See the module documentation for shared numerical conventions.
///
/// ## Examples
///
/// ```
/// let value = sci_special::elliptic::ellipeinc(0.5_f64, 0.0)?;
/// assert!((value - (0.5)).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [Ellip `ellipeinc`](https://docs.rs/ellip/1.2.3/ellip/fn.ellipeinc.html).
/// - [NIST DLMF, Chapter 19](https://dlmf.nist.gov/19).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn ellipeinc<T: Float>(phi: T, m: T) -> Result<T> {
    if !(phi.is_finite() && m.is_finite()) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::ellipeinc(phi, m).map_err(Error::NumericalFailure)?;
    match value {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}

/// Computes Π(phi, n, m) = ∫₀^phi dθ / [(1 − n sin²θ)√(1 − m sin²θ)].
///
/// ## Parameters
///
/// - `phi`: Amplitude in radians.
/// - `n`: Characteristic of a third-kind integral.
/// - `m`: Elliptic parameter `m = k²`, where `k` is the modulus.
///
/// ## Domain and special values
///
/// Finite arguments. For `m < 1`, real amplitudes and characteristics are delegated to Ellip,
/// including Cauchy principal values when the path crosses an interior third-kind pole. A
/// pole at the endpoint (`n sin²(phi) = 1`) is rejected. For `m >= 1`, only Ellip’s real
/// special cases are supported; nonreal paths and backend failures return errors.
///
/// NaN and infinity inputs return an error; a nonfinite result is rejected.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] when a wrapper input check fails,
/// [`Error::NumericalFailure`] if Ellip rejects the domain or cannot evaluate the integral, and
/// [`Error::NonFiniteEvaluation`] if the result is nonfinite.
///
/// ## Numerical behavior
///
/// Evaluates in `T` using Ellip's checked algorithms. No tolerance or iteration
/// controls are exposed. Relative accuracy can degrade near zeros or poles;
/// amplitude reduction, where applicable, is limited by floating-point precision.
/// See the module documentation for shared numerical conventions.
///
/// ## Examples
///
/// ```
/// let value = sci_special::elliptic::ellippiinc(0.5_f64, 0.0, 0.0)?;
/// assert!((value - (0.5)).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [Ellip `ellippiinc`](https://docs.rs/ellip/1.2.3/ellip/fn.ellippiinc.html).
/// - [NIST DLMF, Chapter 19](https://dlmf.nist.gov/19).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn ellippiinc<T: Float>(phi: T, n: T, m: T) -> Result<T> {
    if !(phi.is_finite() && m.is_finite() && n.is_finite()) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::ellippiinc(phi, n, m).map_err(Error::NumericalFailure)?;
    match value {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}

/// Computes D(phi, m) = ∫₀^phi sin²θ dθ / √(1 − m sin²θ).
///
/// ## Parameters
///
/// - `phi`: Amplitude in radians.
/// - `m`: Elliptic parameter `m = k²`, where `k` is the modulus.
///
/// ## Domain and special values
///
/// Finite arguments. For `m < 1`, any real amplitude is accepted. For `m >= 1`, the
/// integration path must remain real and nonsingular; the endpoint condition alone does not
/// permit crossing a nonreal interval. Backend domain and convergence errors are propagated.
///
/// NaN and infinity inputs return an error; a nonfinite result is rejected.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] when a wrapper input check fails,
/// [`Error::NumericalFailure`] if Ellip rejects the domain or cannot evaluate the integral, and
/// [`Error::NonFiniteEvaluation`] if the result is nonfinite.
///
/// ## Numerical behavior
///
/// Evaluates in `T` using Ellip's checked algorithms. No tolerance or iteration
/// controls are exposed. Relative accuracy can degrade near zeros or poles;
/// amplitude reduction, where applicable, is limited by floating-point precision.
/// See the module documentation for shared numerical conventions.
///
/// ## Examples
///
/// ```
/// let value = sci_special::elliptic::ellipdinc(0.0_f64, 0.0)?;
/// assert!((value - (0.0)).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [Ellip `ellipdinc`](https://docs.rs/ellip/1.2.3/ellip/fn.ellipdinc.html).
/// - [NIST DLMF, Chapter 19](https://dlmf.nist.gov/19).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn ellipdinc<T: Float>(phi: T, m: T) -> Result<T> {
    if !(phi.is_finite() && m.is_finite()) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::ellipdinc(phi, m).map_err(Error::NumericalFailure)?;
    match value {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}

/// Computes R_F(x, y, z) = ½ ∫₀^∞ dt / √[(t+x)(t+y)(t+z)].
///
/// ## Parameters
///
/// - `x`: Real Carlson argument.
/// - `y`: Real Carlson argument.
/// - `z`: Real Carlson argument.
///
/// ## Domain and special values
///
/// Finite nonnegative arguments with at most one zero. Symmetric in all three arguments.
///
/// NaN and infinity inputs return an error; a nonfinite result is rejected.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] when a wrapper input check fails,
/// [`Error::NumericalFailure`] if Ellip rejects the domain or cannot evaluate the integral, and
/// [`Error::NonFiniteEvaluation`] if the result is nonfinite.
///
/// ## Numerical behavior
///
/// Evaluates in `T` using Ellip's checked algorithms. No tolerance or iteration
/// controls are exposed. Relative accuracy can degrade near zeros or poles;
/// amplitude reduction, where applicable, is limited by floating-point precision.
/// See the module documentation for shared numerical conventions.
///
/// ## Examples
///
/// ```
/// let value = sci_special::elliptic::elliprf(1.0_f64, 1.0, 1.0)?;
/// assert!((value - (1.0)).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [Ellip `elliprf`](https://docs.rs/ellip/1.2.3/ellip/fn.elliprf.html).
/// - [NIST DLMF, Chapter 19](https://dlmf.nist.gov/19).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn elliprf<T: Float>(x: T, y: T, z: T) -> Result<T> {
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
    match value {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}

/// Computes R_D(x, y, z) = (3/2) ∫₀^∞ dt / [(t+z)√((t+x)(t+y)(t+z))].
///
/// ## Parameters
///
/// - `x`: Real Carlson argument.
/// - `y`: Real Carlson argument.
/// - `z`: Real Carlson argument.
///
/// ## Domain and special values
///
/// Finite `x, y >= 0`, at least one positive, and `z > 0`. Symmetric in `x` and `y`; the last
/// argument is distinguished.
///
/// NaN and infinity inputs return an error; a nonfinite result is rejected.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] when a wrapper input check fails,
/// [`Error::NumericalFailure`] if Ellip rejects the domain or cannot evaluate the integral, and
/// [`Error::NonFiniteEvaluation`] if the result is nonfinite.
///
/// ## Numerical behavior
///
/// Evaluates in `T` using Ellip's checked algorithms. No tolerance or iteration
/// controls are exposed. Relative accuracy can degrade near zeros or poles;
/// amplitude reduction, where applicable, is limited by floating-point precision.
/// See the module documentation for shared numerical conventions.
///
/// ## Examples
///
/// ```
/// let value = sci_special::elliptic::elliprd(1.0_f64, 1.0, 1.0)?;
/// assert!((value - (1.0)).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [Ellip `elliprd`](https://docs.rs/ellip/1.2.3/ellip/fn.elliprd.html).
/// - [NIST DLMF, Chapter 19](https://dlmf.nist.gov/19).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn elliprd<T: Float>(x: T, y: T, z: T) -> Result<T> {
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
    match value {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}

/// Computes R_G(x, y, z) = ¼ ∫₀^∞ [x/(t+x) + y/(t+y) + z/(t+z)] t dt / √[(t+x)(t+y)(t+z)].
///
/// ## Parameters
///
/// - `x`: Real Carlson argument.
/// - `y`: Real Carlson argument.
/// - `z`: Real Carlson argument.
///
/// ## Domain and special values
///
/// Finite nonnegative arguments, including multiple zeros by continuity; `R_G(0,0,0) = 0`.
/// Symmetric in all three arguments.
///
/// NaN and infinity inputs return an error; a nonfinite result is rejected.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] when a wrapper input check fails,
/// [`Error::NumericalFailure`] if Ellip rejects the domain or cannot evaluate the integral, and
/// [`Error::NonFiniteEvaluation`] if the result is nonfinite.
///
/// ## Numerical behavior
///
/// Evaluates in `T` using Ellip's checked algorithms. No tolerance or iteration
/// controls are exposed. Relative accuracy can degrade near zeros or poles;
/// amplitude reduction, where applicable, is limited by floating-point precision.
/// See the module documentation for shared numerical conventions.
///
/// ## Examples
///
/// ```
/// let value = sci_special::elliptic::elliprg(1.0_f64, 1.0, 1.0)?;
/// assert!((value - (1.0)).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [Ellip `elliprg`](https://docs.rs/ellip/1.2.3/ellip/fn.elliprg.html).
/// - [NIST DLMF, Chapter 19](https://dlmf.nist.gov/19).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn elliprg<T: Float>(x: T, y: T, z: T) -> Result<T> {
    if !(x.is_finite() && y.is_finite() && z.is_finite() && (x >= 0.0 && y >= 0.0 && z >= 0.0)) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::elliprg(x, y, z).map_err(Error::NumericalFailure)?;
    match value {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}

/// Computes R_J(x, y, z, p) = (3/2) ∫₀^∞ dt / [(t+p)√((t+x)(t+y)(t+z))].
///
/// ## Parameters
///
/// - `x`: Real Carlson argument.
/// - `y`: Real Carlson argument.
/// - `z`: Real Carlson argument.
/// - `p`: Additional Carlson third-kind argument.
///
/// ## Domain and special values
///
/// Finite nonnegative `x, y, z` with at most one zero, and finite `p != 0`. For `p < 0`,
/// returns the Cauchy principal value. Symmetric in the first three arguments.
///
/// NaN and infinity inputs return an error; a nonfinite result is rejected.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] when a wrapper input check fails,
/// [`Error::NumericalFailure`] if Ellip rejects the domain or cannot evaluate the integral, and
/// [`Error::NonFiniteEvaluation`] if the result is nonfinite.
///
/// ## Numerical behavior
///
/// Evaluates in `T` using Ellip's checked algorithms. No tolerance or iteration
/// controls are exposed. Relative accuracy can degrade near zeros or poles;
/// amplitude reduction, where applicable, is limited by floating-point precision.
/// See the module documentation for shared numerical conventions.
///
/// ## Examples
///
/// ```
/// let value = sci_special::elliptic::elliprj(1.0_f64, 1.0, 1.0, 1.0)?;
/// assert!((value - (1.0)).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [Ellip `elliprj`](https://docs.rs/ellip/1.2.3/ellip/fn.elliprj.html).
/// - [NIST DLMF, Chapter 19](https://dlmf.nist.gov/19).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn elliprj<T: Float>(x: T, y: T, z: T, p: T) -> Result<T> {
    if !(x.is_finite()
        && y.is_finite()
        && z.is_finite()
        && p.is_finite()
        && (x >= 0.0
            && y >= 0.0
            && z >= 0.0
            && ((x == 0.0) as u8 + (y == 0.0) as u8 + (z == 0.0) as u8 <= 1)
            && p != 0.0))
    {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::elliprj(x, y, z, p).map_err(Error::NumericalFailure)?;
    match value {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}

/// Computes R_C(x, y) = ½ ∫₀^∞ dt / [(t+y)√(t+x)] = R_F(x,y,y) for y > 0.
///
/// ## Parameters
///
/// - `x`: Real Carlson argument.
/// - `y`: Real Carlson argument.
///
/// ## Domain and special values
///
/// Finite `x >= 0` and `y != 0`. For `y < 0`, returns the Cauchy principal value of the
/// defining integral, rather than a complex square-root continuation.
///
/// NaN and infinity inputs return an error; a nonfinite result is rejected.
///
/// ## Returns
///
/// The real value of the integral defined above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] when a wrapper input check fails,
/// [`Error::NumericalFailure`] if Ellip rejects the domain or cannot evaluate the integral, and
/// [`Error::NonFiniteEvaluation`] if the result is nonfinite.
///
/// ## Numerical behavior
///
/// Evaluates in `T` using Ellip's checked algorithms. No tolerance or iteration
/// controls are exposed. Relative accuracy can degrade near zeros or poles;
/// amplitude reduction, where applicable, is limited by floating-point precision.
/// See the module documentation for shared numerical conventions.
///
/// ## Examples
///
/// ```
/// let value = sci_special::elliptic::elliprc(1.0_f64, -2.0)?;
/// let expected = (1.0 / 3.0_f64.sqrt()).atanh() / 3.0_f64.sqrt();
/// assert!((value - expected).abs() < 1e-14); // Cauchy principal value
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [Ellip `elliprc`](https://docs.rs/ellip/1.2.3/ellip/fn.elliprc.html).
/// - [NIST DLMF, Chapter 19](https://dlmf.nist.gov/19).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn elliprc<T: Float>(x: T, y: T) -> Result<T> {
    if !(x.is_finite() && y.is_finite() && (x >= 0.0 && y != 0.0)) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::elliprc(x, y).map_err(Error::NumericalFailure)?;
    match value {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}

/// Computes (K(m), E(m)), with K(m) = ∫₀^(π/2) dθ/√(1−m sin²θ) and E(m) = ∫₀^(π/2) √(1−m sin²θ) dθ.
///
/// ## Parameters
///
/// - `m`: Elliptic parameter `m = k²`, where `k` is the modulus.
///
/// ## Domain and special values
///
/// Finite `m < 1`. Both values are evaluated together and returned in first-kind, second-kind order.
///
/// NaN and infinity inputs return an error; a nonfinite result is rejected.
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
/// ## Numerical behavior
///
/// Evaluates in `T` using Ellip's checked algorithms. No tolerance or iteration
/// controls are exposed. Relative accuracy can degrade near zeros or poles;
/// amplitude reduction, where applicable, is limited by floating-point precision.
/// See the module documentation for shared numerical conventions.
///
/// ## Examples
///
/// ```
/// let value = sci_special::elliptic::ellipke(0.0_f64)?;
/// let (k, e) = value; assert!((k-e).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [Ellip `ellipke`](https://docs.rs/ellip/1.2.3/ellip/fn.ellipke.html).
/// - [NIST DLMF, elliptic integrals](https://dlmf.nist.gov/19).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn ellipke<T: Float>(m: T) -> Result<(T, T)> {
    if !(m.is_finite() && m < 1.0) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::ellipke(m).map_err(Error::NumericalFailure)?;
    Ok((
        match value.0 {
            value if value.is_finite() => Ok(value),
            _ => Err(Error::NonFiniteEvaluation),
        }?,
        match value.1 {
            value if value.is_finite() => Ok(value),
            _ => Err(Error::NonFiniteEvaluation),
        }?,
    ))
}

/// Computes cel(kc, p, a, b) = ∫₀^(π/2) (a cos²θ + b sin²θ) dθ / [(cos²θ + p sin²θ)√(cos²θ + kc² sin²θ)].
///
/// ## Parameters
///
/// - `kc`: Complementary modulus; `kc² = 1 - m`. Its sign does not affect the integral.
/// - `p`: Bulirsch parameter `p = 1 - n`, where `n` is the characteristic.
/// - `a`: Coefficient of the cosine-squared term.
/// - `b`: Coefficient of the sine-squared term.
///
/// ## Domain and special values
///
/// Finite arguments, `kc != 0`, and `p != 0`. For `p < 0`, returns the Cauchy principal
/// value. The sign of `kc` does not affect the value.
///
/// NaN and infinity inputs return an error; a nonfinite result is rejected.
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
/// ## Numerical behavior
///
/// Evaluates in `T` using Ellip's checked algorithms. No tolerance or iteration
/// controls are exposed. Relative accuracy can degrade near zeros or poles;
/// amplitude reduction, where applicable, is limited by floating-point precision.
/// See the module documentation for shared numerical conventions.
///
/// ## Examples
///
/// ```
/// let value = sci_special::elliptic::cel(1.0, 1.0, 1.0, 1.0)?;
/// assert!((value-std::f64::consts::FRAC_PI_2).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [Ellip `cel`](https://docs.rs/ellip/1.2.3/ellip/fn.cel.html).
/// - [NIST DLMF, elliptic integrals](https://dlmf.nist.gov/19).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn cel<T: Float>(kc: T, p: T, a: T, b: T) -> Result<T> {
    if !(kc.is_finite() && kc != 0.0 && p.is_finite() && p != 0.0 && a.is_finite() && b.is_finite())
    {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::cel(kc, p, a, b).map_err(Error::NumericalFailure)?;
    match value {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}

/// Computes cel1(kc) = cel(kc,1,1,1) = K(1−kc²).
///
/// ## Parameters
///
/// - `kc`: Complementary modulus; `kc² = 1 - m`. Its sign does not affect the integral.
///
/// ## Domain and special values
///
/// Finite `kc != 0`; either sign is accepted.
///
/// NaN and infinity inputs return an error; a nonfinite result is rejected.
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
/// ## Numerical behavior
///
/// Evaluates in `T` using Ellip's checked algorithms. No tolerance or iteration
/// controls are exposed. Relative accuracy can degrade near zeros or poles;
/// amplitude reduction, where applicable, is limited by floating-point precision.
/// See the module documentation for shared numerical conventions.
///
/// ## Examples
///
/// ```
/// let value = sci_special::elliptic::cel1(1.0)?;
/// assert!((value-std::f64::consts::FRAC_PI_2).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [Ellip `cel1`](https://docs.rs/ellip/1.2.3/ellip/fn.cel1.html).
/// - [NIST DLMF, elliptic integrals](https://dlmf.nist.gov/19).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn cel1<T: Float>(kc: T) -> Result<T> {
    if !(kc.is_finite() && kc != 0.0) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::cel1(kc).map_err(Error::NumericalFailure)?;
    match value {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}

/// Computes cel2(kc, a, b) = cel(kc,1,a,b).
///
/// ## Parameters
///
/// - `kc`: Complementary modulus; `kc² = 1 - m`. Its sign does not affect the integral.
/// - `a`: Coefficient of the cosine-squared term.
/// - `b`: Coefficient of the sine-squared term.
///
/// ## Domain and special values
///
/// Finite `a`, `b`, and `kc != 0`; either sign of `kc` is accepted.
///
/// NaN and infinity inputs return an error; a nonfinite result is rejected.
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
/// ## Numerical behavior
///
/// Evaluates in `T` using Ellip's checked algorithms. No tolerance or iteration
/// controls are exposed. Relative accuracy can degrade near zeros or poles;
/// amplitude reduction, where applicable, is limited by floating-point precision.
/// See the module documentation for shared numerical conventions.
///
/// ## Examples
///
/// ```
/// let value = sci_special::elliptic::cel2(1.0, 1.0, 1.0)?;
/// assert!((value-std::f64::consts::FRAC_PI_2).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [Ellip `cel2`](https://docs.rs/ellip/1.2.3/ellip/fn.cel2.html).
/// - [NIST DLMF, elliptic integrals](https://dlmf.nist.gov/19).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn cel2<T: Float>(kc: T, a: T, b: T) -> Result<T> {
    if !(kc.is_finite() && kc != 0.0 && a.is_finite() && b.is_finite()) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::cel2(kc, a, b).map_err(Error::NumericalFailure)?;
    match value {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}

/// Computes cel3(kc, p) = cel(kc,p,1,1) = Π(1−p, 1−kc²).
///
/// ## Parameters
///
/// - `kc`: Complementary modulus; `kc² = 1 - m`. Its sign does not affect the integral.
/// - `p`: Bulirsch parameter `p = 1 - n`, where `n` is the characteristic.
///
/// ## Domain and special values
///
/// Finite `kc != 0` and `p != 0`. For `p < 0`, returns the Cauchy principal value.
///
/// NaN and infinity inputs return an error; a nonfinite result is rejected.
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
/// ## Numerical behavior
///
/// Evaluates in `T` using Ellip's checked algorithms. No tolerance or iteration
/// controls are exposed. Relative accuracy can degrade near zeros or poles;
/// amplitude reduction, where applicable, is limited by floating-point precision.
/// See the module documentation for shared numerical conventions.
///
/// ## Examples
///
/// ```
/// let value = sci_special::elliptic::cel3(1.0, 1.0)?;
/// assert!((value-std::f64::consts::FRAC_PI_2).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [Ellip `cel3`](https://docs.rs/ellip/1.2.3/ellip/fn.cel3.html).
/// - [NIST DLMF, elliptic integrals](https://dlmf.nist.gov/19).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn cel3<T: Float>(kc: T, p: T) -> Result<T> {
    if !(kc.is_finite() && kc != 0.0 && p.is_finite() && p != 0.0) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::cel3(kc, p).map_err(Error::NumericalFailure)?;
    match value {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}

/// Computes el1(x, kc) = F(atan(x), 1−kc²).
///
/// ## Parameters
///
/// - `x`: Tangent amplitude; the upper integration angle is `atan(x)` in radians.
/// - `kc`: Complementary modulus; `kc² = 1 - m`. Its sign does not affect the integral.
///
/// ## Domain and special values
///
/// Finite `x` and finite `kc != 0`. The amplitude is `atan(x)` in `(-π/2, π/2)`; either sign
/// of `kc` is accepted.
///
/// NaN and infinity inputs return an error; a nonfinite result is rejected.
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
/// ## Numerical behavior
///
/// Evaluates in `T` using Ellip's checked algorithms. No tolerance or iteration
/// controls are exposed. Relative accuracy can degrade near zeros or poles;
/// amplitude reduction, where applicable, is limited by floating-point precision.
/// See the module documentation for shared numerical conventions.
///
/// ## Examples
///
/// ```
/// let value = sci_special::elliptic::el1(0.5, 1.0)?;
/// assert!((value-0.5_f64.atan()).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [Ellip `el1`](https://docs.rs/ellip/1.2.3/ellip/fn.el1.html).
/// - [NIST DLMF, elliptic integrals](https://dlmf.nist.gov/19).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn el1<T: Float>(x: T, kc: T) -> Result<T> {
    if !(x.is_finite() && kc.is_finite() && kc != 0.0) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::el1(x, kc).map_err(Error::NumericalFailure)?;
    match value {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}

/// Computes el2(x, kc, a, b) = ∫₀^atan(x) (a cos²θ + b sin²θ) dθ / √(cos²θ + kc² sin²θ).
///
/// ## Parameters
///
/// - `x`: Tangent amplitude; the upper integration angle is `atan(x)` in radians.
/// - `kc`: Complementary modulus; `kc² = 1 - m`. Its sign does not affect the integral.
/// - `a`: Coefficient of the cosine-squared term.
/// - `b`: Coefficient of the sine-squared term.
///
/// ## Domain and special values
///
/// Finite arguments and `kc != 0`. The amplitude is `atan(x)`; either sign of `kc` is accepted.
///
/// NaN and infinity inputs return an error; a nonfinite result is rejected.
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
/// ## Numerical behavior
///
/// Evaluates in `T` using Ellip's checked algorithms. No tolerance or iteration
/// controls are exposed. Relative accuracy can degrade near zeros or poles;
/// amplitude reduction, where applicable, is limited by floating-point precision.
/// See the module documentation for shared numerical conventions.
///
/// ## Examples
///
/// ```
/// let value = sci_special::elliptic::el2(0.5, 1.0, 1.0, 1.0)?;
/// assert!((value-0.5_f64.atan()).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [Ellip `el2`](https://docs.rs/ellip/1.2.3/ellip/fn.el2.html).
/// - [NIST DLMF, elliptic integrals](https://dlmf.nist.gov/19).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn el2<T: Float>(x: T, kc: T, a: T, b: T) -> Result<T> {
    if !(x.is_finite() && kc.is_finite() && kc != 0.0 && a.is_finite() && b.is_finite()) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::el2(x, kc, a, b).map_err(Error::NumericalFailure)?;
    match value {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}

/// Computes el3(x, kc, p) = Π(atan(x), 1−p, 1−kc²).
///
/// ## Parameters
///
/// - `x`: Tangent amplitude; the upper integration angle is `atan(x)` in radians.
/// - `kc`: Complementary modulus; `kc² = 1 - m`. Its sign does not affect the integral.
/// - `p`: Bulirsch parameter `p = 1 - n`, where `n` is the characteristic.
///
/// ## Domain and special values
///
/// Finite arguments with `kc != 0` and `1 + p*x² != 0`. For `p < 0`, Ellip requires `abs(kc)
/// <= 1`. If `1 + p*x² < 0`, returns the Cauchy principal value; a negative `p` is also
/// allowed before the pole.
///
/// NaN and infinity inputs return an error; a nonfinite result is rejected.
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
/// ## Numerical behavior
///
/// Evaluates in `T` using Ellip's checked algorithms. No tolerance or iteration
/// controls are exposed. Relative accuracy can degrade near zeros or poles;
/// amplitude reduction, where applicable, is limited by floating-point precision.
/// See the module documentation for shared numerical conventions.
///
/// ## Examples
///
/// ```
/// let value = sci_special::elliptic::el3(0.5, 1.0, 1.0)?;
/// assert!((value-0.5_f64.atan()).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [Ellip `el3`](https://docs.rs/ellip/1.2.3/ellip/fn.el3.html).
/// - [NIST DLMF, elliptic integrals](https://dlmf.nist.gov/19).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn el3<T: Float>(x: T, kc: T, p: T) -> Result<T> {
    if !(x.is_finite()
        && kc.is_finite()
        && kc != 0.0
        && p.is_finite()
        && (1.0 + p * x * x != 0.0)
        && (p >= 0.0 || kc.abs() <= 1.0))
    {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::el3(x, kc, p).map_err(Error::NumericalFailure)?;
    match value {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}

/// Computes Z(phi, m) = E(phi,m) − E(m)F(phi,m)/K(m).
///
/// ## Parameters
///
/// - `phi`: Amplitude in radians.
/// - `m`: Elliptic parameter `m = k²`, where `k` is the modulus.
///
/// ## Domain and special values
///
/// Finite `phi` and `m <= 1`. At `m = 1`, Ellip evaluates the limiting branch directly; the
/// quotient formula is interpreted by continuity.
///
/// NaN and infinity inputs return an error; a nonfinite result is rejected.
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
/// ## Numerical behavior
///
/// Evaluates in `T` using Ellip's checked algorithms. No tolerance or iteration
/// controls are exposed. Relative accuracy can degrade near zeros or poles;
/// amplitude reduction, where applicable, is limited by floating-point precision.
/// See the module documentation for shared numerical conventions.
///
/// ## Examples
///
/// ```
/// let value = sci_special::elliptic::jacobi_zeta(0.5_f64, 0.0)?;
/// assert!(value.abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [Ellip `jacobi_zeta`](https://docs.rs/ellip/1.2.3/ellip/fn.jacobi_zeta.html).
/// - [NIST DLMF, elliptic integrals](https://dlmf.nist.gov/19).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn jacobi_zeta<T: Float>(phi: T, m: T) -> Result<T> {
    if !(phi.is_finite() && m.is_finite() && m <= 1.0) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::jacobi_zeta(phi, m).map_err(Error::NumericalFailure)?;
    match value {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}

/// Computes Λ₀(phi, m) = F(phi,1−m)/K(1−m) + 2 K(m) Z(phi,1−m)/π.
///
/// ## Parameters
///
/// - `phi`: Amplitude in radians.
/// - `m`: Elliptic parameter `m = k²`, where `k` is the modulus.
///
/// ## Domain and special values
///
/// Finite `phi` and `0 <= m < 1`. The formula at `m = 0` is interpreted by continuity:
/// `Λ₀(phi,0) = sin(phi)` on the principal amplitude interval.
///
/// NaN and infinity inputs return an error; a nonfinite result is rejected.
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
/// ## Numerical behavior
///
/// Evaluates in `T` using Ellip's checked algorithms. No tolerance or iteration
/// controls are exposed. Relative accuracy can degrade near zeros or poles;
/// amplitude reduction, where applicable, is limited by floating-point precision.
/// See the module documentation for shared numerical conventions.
///
/// ## Examples
///
/// ```
/// let value = sci_special::elliptic::heuman_lambda(0.5, 0.0)?;
/// assert!((value-0.5_f64.sin()).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [Ellip `heuman_lambda`](https://docs.rs/ellip/1.2.3/ellip/fn.heuman_lambda.html).
/// - [NIST DLMF, elliptic integrals](https://dlmf.nist.gov/19).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn heuman_lambda<T: Float>(phi: T, m: T) -> Result<T> {
    if !(phi.is_finite() && m.is_finite() && (0.0..1.0).contains(&m)) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = ellip::heuman_lambda(phi, m).map_err(Error::NumericalFailure)?;
    match value {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}

/// Computes Π(phi, n, m) = ∫₀^phi dθ / [(1 − n sin²θ)√(1 − m sin²θ)], evaluated with Bulirsch’s algorithm.
///
/// ## Parameters
///
/// - `phi`: Amplitude in radians.
/// - `n`: Characteristic of a third-kind integral.
/// - `m`: Elliptic parameter `m = k²`, where `k` is the modulus.
///
/// ## Domain and special values
///
/// The finite-input domain and principal-value conventions match `ellippiinc`. Ellip
/// delegates cases outside the direct Bulirsch regime to its general third-kind algorithm.
///
/// NaN and infinity inputs return an error; a nonfinite result is rejected.
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
/// ## Numerical behavior
///
/// Evaluates in `T` using Ellip's checked algorithms. No tolerance or iteration
/// controls are exposed. Relative accuracy can degrade near zeros or poles;
/// amplitude reduction, where applicable, is limited by floating-point precision.
/// See the module documentation for shared numerical conventions.
///
/// ## Examples
///
/// ```
/// let value = sci_special::elliptic::ellippiinc_bulirsch(0.5_f64, 0.0, 0.0)?;
/// assert!((value-0.5).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [Ellip `ellippiinc_bulirsch`](https://docs.rs/ellip/1.2.3/ellip/fn.ellippiinc_bulirsch.html).
/// - [NIST DLMF, elliptic integrals](https://dlmf.nist.gov/19).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn ellippiinc_bulirsch<T: BulirschFloat>(phi: T, n: T, m: T) -> Result<T> {
    if !(phi.is_finite() && m.is_finite() && n.is_finite()) {
        return Err(Error::InvalidInput(
            "elliptic integral arguments are outside the real domain",
        ));
    }
    let value = T::third_kind(phi, n, m)?;
    match value {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}

/// Computes K(1−p) = R_F(0,p,1).
///
/// ## Parameters
///
/// - `p`: Complementary parameter `p = 1 - m`.
///
/// ## Domain and special values
///
/// Finite `p > 0`. Here `p = 1−m` is the complementary parameter, not the characteristic or
/// complementary modulus. Passing it directly avoids rounding `1−p` to one.
///
/// NaN and infinity inputs return an error; a nonfinite result is rejected.
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
/// ## Numerical behavior
///
/// Evaluates in `T` using Ellip's checked algorithms. No tolerance or iteration
/// controls are exposed. Relative accuracy can degrade near zeros or poles;
/// amplitude reduction, where applicable, is limited by floating-point precision.
/// See the module documentation for shared numerical conventions.
///
/// ## Examples
///
/// ```
/// let k = sci_special::elliptic::ellipkm1(1e-20)?;
/// assert!(k > 20.0);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [Ellip `elliprf`](https://docs.rs/ellip/1.2.3/ellip/fn.elliprf.html).
/// - [NIST DLMF, Carlson integrals](https://dlmf.nist.gov/19.25).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn ellipkm1<T: Float>(p: T) -> Result<T> {
    if !p.is_finite() || p <= 0.0 {
        return Err(Error::InvalidInput(
            "complementary parameter must be finite and positive",
        ));
    }
    elliprf(0.0, p, 1.0)
}

/// Floating-point types supported by the specialized Bulirsch algorithm.
///
/// This dispatch keeps backend precision-control traits private. The standard
/// `f32` and `f64` implementations evaluate directly in their own precision.
pub trait BulirschFloat: Float {
    /// Evaluates the incomplete third-kind integral; used by [`ellippiinc_bulirsch`].
    #[doc(hidden)]
    fn third_kind(phi: Self, n: Self, m: Self) -> Result<Self>;
}

macro_rules! bulirsch_float {
    ($($t:ty),*) => {$(
        impl BulirschFloat for $t {
            fn third_kind(phi: Self, n: Self, m: Self) -> Result<Self> {
                ellip::ellippiinc_bulirsch(phi, n, m).map_err(Error::NumericalFailure)
            }
        }
    )*};
}
bulirsch_float!(f32, f64);
