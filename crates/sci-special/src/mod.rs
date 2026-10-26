/// Computes Γ(x), the analytic continuation of the factorial.
///
/// ## Domain
///
/// Real x away from nonpositive integer poles. Poles and overflow follow the backend's IEEE behavior; NaN inputs return NaN.
///
/// # Examples
///
/// ```
/// assert!((sci_special::gamma(5.0) - (24.0)).abs() < 1e-12);
/// ```
///
/// # References
///
/// - [NIST DLMF](https://dlmf.nist.gov/5.2).
pub fn gamma(x: f64) -> f64 {
    ::special::Gamma::gamma(x)
}

/// Computes ln |Γ(x)|.
///
/// ## Domain
///
/// Real x away from poles. The sign of Γ is discarded. Poles return infinity and NaN propagates.
///
/// # Examples
///
/// ```
/// assert!((sci_special::gammaln(1.0) - (0.0)).abs() < 1e-12);
/// ```
///
/// # References
///
/// - [NIST DLMF](https://dlmf.nist.gov/5.2).
pub fn gammaln(x: f64) -> f64 {
    ::special::Gamma::ln_gamma(x).0
}

/// Computes ψ(x) = d ln Γ(x) / dx.
///
/// ## Domain
///
/// Real x away from nonpositive integer poles. Negative integer poles return
/// NaN, positive infinity returns infinity, and NaN propagates. Positive finite
/// arguments use `special`; other arguments use a nonrecursive reflection backend.
///
/// # Examples
///
/// ```
/// assert!((sci_special::digamma(1.0) - (-0.5772156649015329)).abs() < 1e-12);
/// ```
///
/// # References
///
/// - [NIST DLMF](https://dlmf.nist.gov/5.15).
pub fn digamma(x: f64) -> f64 {
    if x.is_finite() && x > 0.0 {
        ::special::Gamma::digamma(x)
    } else {
        xsf::digamma(x)
    }
}

/// Computes ψ₁(x) = d ψ(x) / dx.
///
/// ## Domain
///
/// Real x away from nonpositive integer poles. Poles diverge to infinity;
/// positive infinity returns zero and NaN propagates.
///
/// # Examples
///
/// ```
/// assert!((sci_special::trigamma(1.0) - (1.6449340668482264)).abs() < 1e-12);
/// ```
///
/// # References
///
/// - [NIST DLMF](https://dlmf.nist.gov/5.15).
pub fn trigamma(x: f64) -> f64 {
    if x.is_finite() && x > 0.0 {
        ::special::Gamma::trigamma(&x)
    } else {
        xsf::polygamma(1, x)
    }
}

/// Computes erf(x) = (2/√π) ∫₀ˣ exp(−t²) dt.
///
/// ## Domain
///
/// All real values. Infinite inputs return ±1 and NaN propagates.
///
/// # Examples
///
/// ```
/// assert!((sci_special::erf(0.0) - (0.0)).abs() < 1e-12);
/// ```
///
/// # References
///
/// - [NIST DLMF](https://dlmf.nist.gov/7.2).
pub fn erf(x: f64) -> f64 {
    ::special::Error::error(x)
}

/// Computes erfc(x) = 1 − erf(x).
///
/// ## Domain
///
/// All real values. At −∞ and +∞ the results are 2 and 0, respectively; NaN propagates.
///
/// # Examples
///
/// ```
/// assert!((sci_special::erfc(0.0) - (1.0)).abs() < 1e-12);
/// ```
///
/// # References
///
/// - [NIST DLMF](https://dlmf.nist.gov/7.2).
pub fn erfc(x: f64) -> f64 {
    ::special::Error::compl_error(x)
}

/// Computes the inverse of erf.
///
/// ## Domain
///
/// x in `[-1, 1]`. Endpoints return signed infinity; values outside the interval return NaN.
///
/// # Examples
///
/// ```
/// assert!((sci_special::erfinv(0.0) - (0.0)).abs() < 1e-12);
/// ```
///
/// # References
///
/// - [NIST DLMF](https://dlmf.nist.gov/7.17).
pub fn erfinv(x: f64) -> f64 {
    if x == 1.0 {
        f64::INFINITY
    } else if x == -1.0 {
        f64::NEG_INFINITY
    } else if x.abs() < 1.0 {
        ::special::Error::inv_error(x)
    } else {
        f64::NAN
    }
}

/// Computes B(a,b) = Γ(a)Γ(b)/Γ(a+b).
///
/// ## Domain
///
/// Finite positive a and b. Invalid inputs return NaN; overflow and underflow follow IEEE arithmetic.
///
/// # Examples
///
/// ```
/// assert!((sci_special::beta(1.0, 1.0) - (1.0)).abs() < 1e-12);
/// ```
///
/// # References
///
/// - [NIST DLMF](https://dlmf.nist.gov/5.12).
pub fn beta(a: f64, b: f64) -> f64 {
    betaln(a, b).exp()
}

/// Computes ln B(a,b).
///
/// ## Domain
///
/// Finite positive a and b. Invalid inputs return NaN.
///
/// # Examples
///
/// ```
/// assert!((sci_special::betaln(1.0, 1.0) - (0.0)).abs() < 1e-12);
/// ```
///
/// # References
///
/// - [NIST DLMF](https://dlmf.nist.gov/5.12).
pub fn betaln(a: f64, b: f64) -> f64 {
    if a.is_finite() && b.is_finite() && a > 0.0 && b > 0.0 {
        ::special::Beta::ln_beta(a, b)
    } else {
        f64::NAN
    }
}

/// Computes Iₓ(a,b), the regularized incomplete beta integral.
///
/// ## Domain
///
/// Finite positive a and b; x in `[0,1]`. Invalid inputs return NaN.
///
/// # Examples
///
/// ```
/// assert!((sci_special::betainc(1.0, 1.0, 0.5) - (0.5)).abs() < 1e-12);
/// ```
///
/// # References
///
/// - [NIST DLMF](https://dlmf.nist.gov/8.17).
pub fn betainc(a: f64, b: f64, x: f64) -> f64 {
    if a.is_finite() && b.is_finite() && a > 0.0 && b > 0.0 && (0.0..=1.0).contains(&x) {
        xsf::betainc(a, b, x)
    } else {
        f64::NAN
    }
}

/// Computes the inverse regularized incomplete beta function.
///
/// Returns `x` such that `Iₓ(a,b) = probability`.
///
/// ## Domain
///
/// Shape parameters `a` and `b` must be finite and positive; probability
/// must be in `[0,1]`. Invalid inputs return NaN. Endpoints return zero and one.
///
/// # Examples
///
/// ```
/// assert!((sci_special::betaincinv(1.0, 1.0, 0.5) - 0.5).abs() < 1e-12);
/// ```
///
/// # References
///
/// - [NIST DLMF, incomplete beta functions](https://dlmf.nist.gov/8.17).
pub fn betaincinv(a: f64, b: f64, probability: f64) -> f64 {
    if a.is_finite() && b.is_finite() && a > 0.0 && b > 0.0 && (0.0..=1.0).contains(&probability) {
        xsf::betaincinv(a, b, probability)
    } else {
        f64::NAN
    }
}

/// Computes regularized lower incomplete gamma `P(a,x) = γ(a,x) / Γ(a)`.
///
/// Here `γ(a,x) = ∫₀ˣ t^(a−1) exp(−t) dt`. Argument ordering follows SciPy.
///
/// ## Domain
///
/// `a` must be finite and positive; `x` must be nonnegative. At zero the
/// result is zero and at positive infinity it is one. Invalid inputs return NaN.
///
/// # Examples
///
/// ```
/// assert!((sci_special::gammainc(1.0, 1.0) - (1.0 - (-1.0_f64).exp())).abs() < 1e-12);
/// ```
///
/// # References
///
/// - [NIST DLMF, incomplete gamma functions](https://dlmf.nist.gov/8.2).
pub fn gammainc(a: f64, x: f64) -> f64 {
    if !a.is_finite() || a <= 0.0 || x.is_nan() || x < 0.0 {
        f64::NAN
    } else if x == f64::INFINITY {
        1.0
    } else {
        xsf::gammainc(a, x)
    }
}

/// Computes the principal real Lambert W branch W₀(x), satisfying W exp(W) = x.
///
/// ## Parameters and conventions
///
/// `x` is a dimensionless real argument. This function selects the real branch
/// with W₀(x) ≥ −1.
///
/// ## Domain and special values
///
/// x ≥ −1/e. Inputs below the branch point return NaN; positive infinity returns infinity.
/// NaN propagates, and signed zero is preserved.
///
/// ## Return value
///
/// Returns W₀(x), with W₀(−1/e) = −1 and W₀(0) = 0.
///
/// ## Numerical behavior
///
/// For |x| ≤ 10⁻³, a degree-seven Taylor expansion of
/// W₀(x) = ∑ₙ₌₁∞ (−n)ⁿ⁻¹ xⁿ/n! avoids cancellation in the backend near zero.
/// The omitted terms contribute less than 6 × 10⁻²⁰ relative error in this
/// interval before floating-point rounding. Other inputs use the private
/// `special` backend's piecewise rational approximation.
///
/// # Examples
///
/// ```
/// assert!((sci_special::lambertw(0.0) - (0.0)).abs() < 1e-12);
/// ```
///
/// # References
///
/// - [NIST DLMF](https://dlmf.nist.gov/4.13).
pub fn lambertw(x: f64) -> f64 {
    if x.abs() <= 1e-3 {
        x * (1.0
            + x * (-1.0
                + x * (1.5
                    + x * (-8.0 / 3.0
                        + x * (125.0 / 24.0 + x * (-54.0 / 5.0 + x * (16807.0 / 720.0)))))))
    } else {
        ::special::LambertW::lambert_w0(x)
    }
}

/// Computes the lower real Lambert W branch W₋₁(x).
///
/// ## Domain
///
/// x in `[-1/e,0]`. Inputs outside the interval return NaN; at zero the result is −∞.
///
/// # Examples
///
/// ```
/// assert!((sci_special::lambertwm1(-0.36787944117144233) - (-1.0)).abs() < 1e-12);
/// ```
///
/// # References
///
/// - [NIST DLMF](https://dlmf.nist.gov/4.13).
pub fn lambertwm1(x: f64) -> f64 {
    ::special::LambertW::lambert_wm1(x)
}

/// Elliptic integral functions with explicit domain errors.
///
/// Legendre integrals use the parameter `m = k²`, not the modulus `k`.
/// Angles are radians. Each function forwards numerical evaluation to ellip
/// and maps its errors into [`crate::Error`].
pub mod elliptic;

// Re-export Sci functions, never backend traits or native types.
pub use elliptic::*;
mod scipy;
pub use scipy::*;
mod waves;
pub use waves::*;
mod polynomials;
pub use polynomials::*;
mod sequences;
pub use sequences::*;
mod elementary;
pub use elementary::*;
mod inverses;
pub use inverses::*;
mod zeros;
pub use zeros::*;
mod ellipsoidal;
pub use ellipsoidal::*;

mod combinatorics;
pub use combinatorics::*;
