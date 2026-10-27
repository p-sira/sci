/// Computes Γ(x), the analytic continuation of the factorial.
///
/// ## Domain
///
/// Real x away from nonpositive integer poles. Poles and overflow follow the backend's IEEE behavior; NaN inputs return NaN.
///
/// ## Examples
///
/// ```
/// assert!((sci_special::gamma(5.0_f64) - (24.0)).abs() < 1e-12);
/// ```
///
/// ## References
///
/// - [NIST DLMF](https://dlmf.nist.gov/5.2).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn gamma<T: sci_core::Float>(x: T) -> T {
    crate::backend::gamma(x).unwrap_or(T::nan())
}

/// Computes ln |Γ(x)|.
///
/// ## Domain
///
/// Real x away from poles. The sign of Γ is discarded. Poles return infinity and NaN propagates.
///
/// ## Examples
///
/// ```
/// assert!((sci_special::gammaln(1.0_f64) - (0.0)).abs() < 1e-12);
/// ```
///
/// ## References
///
/// - [NIST DLMF](https://dlmf.nist.gov/5.2).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn gammaln<T: sci_core::Float>(x: T) -> T {
    crate::backend::gammaln(x).unwrap_or(T::nan())
}

/// Computes ψ(x) = d ln Γ(x) / dx.
///
/// ## Domain
///
/// Real x away from nonpositive integer poles. Negative integer poles return
/// NaN, positive infinity returns infinity, and NaN propagates. Positive finite
/// arguments use `special`; other arguments use a nonrecursive reflection backend.
///
/// ## Examples
///
/// ```
/// assert!((sci_special::digamma(1.0_f64) - (-0.5772156649015329)).abs() < 1e-12);
/// ```
///
/// ## References
///
/// - [NIST DLMF](https://dlmf.nist.gov/5.15).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn digamma<T: sci_core::Float>(x: T) -> T {
    crate::backend::digamma(x).unwrap_or(T::nan())
}

/// Computes ψ₁(x) = d ψ(x) / dx.
///
/// ## Domain
///
/// Real x away from nonpositive integer poles. Poles diverge to infinity;
/// positive infinity returns zero and NaN propagates.
///
/// ## Examples
///
/// ```
/// assert!((sci_special::trigamma(1.0) - (1.6449340668482264)).abs() < 1e-12);
/// ```
///
/// ## References
///
/// - [NIST DLMF](https://dlmf.nist.gov/5.15).
pub fn trigamma(x: f64) -> f64 {
    crate::backend::legacy::polygamma(1, x)
}

/// Computes erf(x) = (2/√π) ∫₀ˣ exp(−t²) dt.
///
/// ## Domain
///
/// All real values. Infinite inputs return ±1 and NaN propagates.
///
/// ## Examples
///
/// ```
/// assert!((sci_special::erf(0.0_f64) - (0.0)).abs() < 1e-12);
/// ```
///
/// ## References
///
/// - [NIST DLMF](https://dlmf.nist.gov/7.2).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn erf<T: sci_core::Float>(x: T) -> T {
    crate::backend::erf(x).unwrap_or(T::nan())
}

/// Computes erfc(x) = 1 − erf(x).
///
/// ## Domain
///
/// All real values. At −∞ and +∞ the results are 2 and 0, respectively; NaN propagates.
///
/// ## Examples
///
/// ```
/// assert!((sci_special::erfc(0.0_f64) - (1.0)).abs() < 1e-12);
/// ```
///
/// ## References
///
/// - [NIST DLMF](https://dlmf.nist.gov/7.2).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn erfc<T: sci_core::Float>(x: T) -> T {
    crate::backend::erfc(x).unwrap_or(T::nan())
}

/// Computes the inverse of erf.
///
/// ## Domain
///
/// x in `[-1, 1]`. Endpoints return signed infinity; values outside the interval return NaN.
///
/// ## Examples
///
/// ```
/// assert!((sci_special::erfinv(0.0_f64) - (0.0)).abs() < 1e-12);
/// ```
///
/// ## References
///
/// - [NIST DLMF](https://dlmf.nist.gov/7.17).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn erfinv<T: sci_core::Float>(x: T) -> T {
    crate::backend::erfinv(x).unwrap_or(T::nan())
}

/// Computes B(a,b) = Γ(a)Γ(b)/Γ(a+b).
///
/// ## Domain
///
/// Finite positive a and b. Invalid inputs return NaN; overflow and underflow follow IEEE arithmetic.
///
/// ## Examples
///
/// ```
/// assert!((sci_special::beta(1.0, 1.0) - (1.0)).abs() < 1e-12);
/// ```
///
/// ## References
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
/// ## Examples
///
/// ```
/// assert!((sci_special::betaln(1.0, 1.0) - (0.0)).abs() < 1e-12);
/// ```
///
/// ## References
///
/// - [NIST DLMF](https://dlmf.nist.gov/5.12).
pub fn betaln(a: f64, b: f64) -> f64 {
    if a.is_finite() && b.is_finite() && a > 0.0 && b > 0.0 {
        crate::backend::betaln(a, b).unwrap_or(f64::NAN)
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
/// ## Examples
///
/// ```
/// assert!((sci_special::betainc(1.0_f64, 1.0, 0.5) - (0.5)).abs() < 1e-12);
/// ```
///
/// ## References
///
/// - [NIST DLMF](https://dlmf.nist.gov/8.17).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn betainc<T: sci_core::Float>(a: T, b: T, x: T) -> T {
    if a.is_finite() && b.is_finite() && a > 0.0 && b > 0.0 && (0.0..=1.0).contains(&x) {
        crate::backend::betainc(a, b, x).unwrap_or(T::nan())
    } else {
        T::nan()
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
/// ## Examples
///
/// ```
/// assert!((sci_special::betaincinv(1.0_f64, 1.0, 0.5) - 0.5).abs() < 1e-12);
/// ```
///
/// ## References
///
/// - [NIST DLMF, incomplete beta functions](https://dlmf.nist.gov/8.17).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn betaincinv<T: sci_core::Float>(a: T, b: T, probability: T) -> T {
    if a.is_finite() && b.is_finite() && a > 0.0 && b > 0.0 && (0.0..=1.0).contains(&probability) {
        crate::backend::betaincinv(a, b, probability).unwrap_or(T::nan())
    } else {
        T::nan()
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
/// ## Examples
///
/// ```
/// assert!((sci_special::gammainc(1.0_f64, 1.0) - (1.0 - (-1.0_f64).exp())).abs() < 1e-12);
/// ```
///
/// ## References
///
/// - [NIST DLMF, incomplete gamma functions](https://dlmf.nist.gov/8.2).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn gammainc<T: sci_core::Float>(a: T, x: T) -> T {
    if !a.is_finite() || a <= 0.0 || x.is_nan() || x < 0.0 {
        T::nan()
    } else if x == T::infinity() {
        1.0
    } else {
        crate::backend::gammainc(a, x).unwrap_or(T::nan())
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
/// Uses `lambert_w` 2.0.8's piecewise rational approximations and near-zero
/// Taylor expansion. Evaluation uses binary64 and converts back to `T`.
/// Within `1e-6` above the rounded `−1/e`, Sci retains a fifth-degree Puiseux
/// expansion with a correction for the rounded branch point; upstream still
/// loses accuracy there. No Sci-specific near-zero Taylor correction is needed.
/// The backend targets approximately 50 bits of accuracy; conditioning worsens
/// near `−1/e`, where small input changes cause large changes in the result.
/// There is no configurable tolerance or iteration budget.
///
/// ## Examples
///
/// ```
/// assert!((sci_special::lambertw(0.0_f64) - (0.0)).abs() < 1e-12);
/// ```
///
/// ## References
///
/// - [NIST DLMF](https://dlmf.nist.gov/4.13).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn lambertw<T: sci_core::Float>(x: T) -> T {
    crate::backend::lambertw0(x).unwrap_or(T::nan())
}

/// Computes W₋₁(x), the real solution of `w exp(w) = x` with `w <= −1`.
///
/// ## Parameters and conventions
///
/// `x` is a dimensionless real argument. This selects the lower real branch.
///
/// ## Domain and special values
///
/// `−1/e <= x <= 0`. Both signed zeros return negative infinity; the rounded
/// branch point returns `−1`. NaN, infinity, and other out-of-domain inputs
/// return NaN. There is no complex continuation.
///
/// ## Returns
///
/// The lower real inverse of `w exp(w)` in the requested floating-point type.
///
/// ## Numerical behavior
///
/// Uses `lambert_w` 2.0.8 in binary64 and converts back to `T`. The backend
/// targets approximately 50 bits of accuracy. The inverse is ill-conditioned
/// near `−1/e`; relative accuracy is also limited by the representation of tiny
/// negative inputs. Within `1e-6` above the rounded `−1/e`, Sci retains a
/// fifth-degree Puiseux expansion with a correction for the rounded branch point;
/// independent 90-digit inverse references verify this region. There is no
/// configurable tolerance or iteration control.
///
/// ## Examples
///
/// ```
/// let w = sci_special::lambertwm1(-0.1_f64);
/// assert!(w < -1.0);
/// assert!((w * w.exp() + 0.1).abs() < 1e-14);
/// ```
///
/// ## References
///
/// - [NIST DLMF, Lambert W](https://dlmf.nist.gov/4.13).
/// - [lambert_w 2.0.8](https://docs.rs/lambert_w/2.0.8/lambert_w/).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn lambertwm1<T: sci_core::Float>(x: T) -> T {
    crate::backend::lambertwm1(x).unwrap_or(T::nan())
}

/// Elliptic integral functions with explicit domain errors.
///
/// Legendre integrals follow Ellip: `F(phi, m)`, `E(phi, m)`, and `Π(phi, n, m)`.
/// The elliptic parameter is `m = k²`; [`elliptic::ellipf`] evaluates `F`.
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
mod native;
pub use native::*;
