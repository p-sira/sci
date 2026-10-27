//! Private adapters for the broad SciPy special-function families.

use crate::{Error, Result};

/// Row-major value and derivative matrices, sharing the linear-algebra representation.
pub type ValueDerivativeMatrices = (Vec<Vec<f64>>, Vec<Vec<f64>>);

/// Arithmetic–geometric mean, the common limit of a←(a+b)/2 and b←√(ab).
///
/// ## Parameters and domain
///
/// Finite real `a` and `b` of the same sign; opposite signs return NaN.
/// If either input is zero, the result is zero. Two negative inputs give
/// the negative of the mean of their absolute values. Nonfinite inputs return NaN.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::agm(1.0_f64,2.0);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy agm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.agm.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn agm<T: sci_core::Float>(a: T, b: T) -> T {
    crate::backend::agm(a, b).unwrap_or(T::nan())
}

/// Airy equation y″−xy=0 solutions, ordered `[Ai,Ai′,Bi,Bi′]`.
///
/// ## Parameters and domain
///
/// All real x; large positive x may overflow Bi.
///
/// ## Returns
///
/// The array `[Ai(x), Ai′(x), Bi(x), Bi′(x)]`, with derivatives with respect to `x`.
/// Nonfinite/backend failures can produce NaNs; positive `x` can overflow the Bi components.
///
/// ## Numerical behavior
///
/// Uses binary64 complex-Bessel Airy kernels. Positive large arguments can
/// underflow Ai and overflow Bi; use `airye` when scaled values are needed.
///
/// ## Examples
///
/// ```
/// let value = sci_special::airy(0.5);
/// assert!(value.iter().all(|v| v.is_finite()));
/// ```
///
/// ## References
///
/// - [SciPy airy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.airy.html).
pub fn airy(x: f64) -> [f64; 4] {
    crate::backend::legacy::airy(x)
}

/// Scaled Airy values `[Ai exp(ζ),Ai′ exp(ζ),Bi exp(−abs(ζ)),Bi′ exp(−abs(ζ))]`, ζ=2x^(3/2)/3.
///
/// ## Parameters and domain
///
/// Real x; scaled Ai and Ai′ return NaN for negative x.
///
/// ## Returns
///
/// The array of four scaled values in `[Ai, Ai′, Bi, Bi′]` order. Derivatives
/// are scaled derivatives of the original functions, not derivatives of the scaled
/// functions. For negative `x`, Ai entries are NaN; Bi entries are unscaled.
///
/// ## Numerical behavior
///
/// Uses binary64 complex-Bessel Airy kernels. For real `x > 0`, the scale
/// exponent is `zeta = 2*x^(3/2)/3`; the Bi scale for `x < 0` is one.
/// Backend failures yield NaNs. Scaling extends the usable positive-argument range.
///
/// ## Examples
///
/// ```
/// let value = sci_special::airye(0.5);
/// assert!(value.iter().all(|v| v.is_finite()));
/// ```
///
/// ## References
///
/// - [SciPy airye](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.airye.html).
pub fn airye(x: f64) -> [f64; 4] {
    crate::backend::legacy::airy_scaled(x)
}

/// Integrals from 0 to x of Ai(t), Bi(t), Ai(−t), and Bi(−t), in that order.
///
/// ## Parameters and domain
///
/// Finite real `x`, including negative upper limits. A negative upper limit
/// reverses the oriented integral; output order remains `[∫Ai(t), ∫Bi(t), ∫Ai(−t), ∫Bi(−t)]`.
///
/// ## Returns
///
/// The four oriented integrals in the order stated above. Nonfinite inputs
/// produce `[NaN; 4]`; the Bi integral can overflow for large positive upper limits.
///
/// ## Examples
///
/// ```
/// let value = sci_special::itairy(0.5);
/// assert!(value.iter().all(|v| v.is_finite()));
/// ```
///
/// ## References
///
/// - [SciPy itairy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.itairy.html).
///
/// ## Numerical behavior and work limits
///
/// An integrated Airy power series with double-double accumulation is used for
/// `abs(x) <= 10`; cached 32-point Gaussian rules cover `10 < abs(x) <= 16`.
/// Translated asymptotic expansions apply for larger magnitudes. Negative inputs
/// use reflection and orientation identities. Large oscillatory arguments remain
/// limited by argument reduction and asymptotic truncation. This corrects the
/// early termination in SciPy's legacy series; it is not a uniform error bound.
pub fn itairy(x: f64) -> [f64; 4] {
    crate::backend::legacy::airy_integrals(x)
}

/// Jacobi elliptic functions (sn,cn,dn,amplitude), with u=F(amplitude,m).
///
/// ## Parameters and domain
///
/// Finite real `u` and parameter `m` in `[0,1]`; `m = k²`.
/// The returned amplitude `phi` is in radians and satisfies `u = F(phi, m)`.
///
/// ## Returns
///
/// Returns `(sn, cn, dn, phi)`, where `sn = sin(phi)`, `cn = cos(phi)`,
/// and `dn = sqrt(1 − m*sn²)`. Invalid inputs return four NaNs.
///
/// ## Numerical behavior
///
/// Uses the binary64 Cephes arithmetic–geometric mean algorithm, with special
/// approximations near `m = 0` and `m = 1`; large `u` loses phase precision.
/// These are Jacobi functions, separate from Ellip’s integral algorithms.
///
/// ## Examples
///
/// ```
/// let value = sci_special::ellipj(0.5,0.4);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy ellipj](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipj.html).
pub fn ellipj(u: f64, m: f64) -> (f64, f64, f64, f64) {
    crate::backend::legacy::ellipj(u, m)
}

/// Generalized binomial coefficient Γ(n+1)/(Γ(k+1)Γ(n−k+1)).
///
/// ## Parameters and domain
///
/// Real n,k; negative integer n returns NaN; singular cases follow SciPy.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::binom(5.0_f64,2.0);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy binom](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.binom.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn binom<T: sci_core::Float>(n: T, k: T) -> T {
    crate::backend::binom(n, k).unwrap_or(T::nan())
}

/// Scaled complementary error function exp(x²) erfc(x), evaluated without forming the exponential.
///
/// ## Parameters and domain
///
/// All real x; large negative x overflows.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::erfcx(0.5_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy erfcx](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erfcx.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn erfcx<T: sci_core::Float>(x: T) -> T {
    crate::backend::erfcx(x).unwrap_or(T::nan())
}

/// Imaginary error function 2/√π ∫₀ˣ exp(t²) dt.
///
/// ## Parameters and domain
///
/// All real x, with signed infinity on overflow.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::erfi(0.5_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy erfi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erfi.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn erfi<T: sci_core::Float>(x: T) -> T {
    crate::backend::erfi(x).unwrap_or(T::nan())
}

/// Inverse complementary error function, satisfying erfc(result)=p.
///
/// ## Parameters and domain
///
/// p in `[0,2]`; endpoints return +∞ and −∞; outside returns NaN.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::erfcinv(0.3_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy erfcinv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erfcinv.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn erfcinv<T: sci_core::Float>(p: T) -> T {
    crate::backend::erfcinv(p).unwrap_or(T::nan())
}

/// Dawson integral exp(−x²) ∫₀ˣ exp(t²) dt.
///
/// ## Parameters and domain
///
/// All real x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::dawsn(0.5_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy dawsn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.dawsn.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn dawsn<T: sci_core::Float>(x: T) -> T {
    crate::backend::dawsn(x).unwrap_or(T::nan())
}

/// Sign of Γ(x), with zero at nonpositive integer poles.
///
/// ## Parameters and domain
///
/// Real x; NaN propagates.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::gammasgn(-0.5_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy gammasgn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gammasgn.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn gammasgn<T: sci_core::Float>(x: T) -> T {
    crate::backend::gammasgn(x).unwrap_or(T::nan())
}

/// Regularized upper incomplete gamma Q(a,x)=Γ(a,x)/Γ(a).
///
/// ## Parameters and domain
///
/// a>0 and x≥0; x=+∞ gives zero.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::gammaincc(2.5_f64,3.0);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy gammaincc](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gammaincc.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn gammaincc<T: sci_core::Float>(a: T, x: T) -> T {
    crate::backend::gammaincc(a, x).unwrap_or(T::nan())
}

/// Inverse of P(a,x) with respect to x.
///
/// ## Parameters and domain
///
/// a>0 and p in `[0,1]`; endpoints give zero and +∞.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::gammaincinv(2.5_f64,0.3);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy gammaincinv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gammaincinv.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn gammaincinv<T: sci_core::Float>(a: T, p: T) -> T {
    crate::backend::gammaincinv(a, p).unwrap_or(T::nan())
}

/// Inverse of Q(a,x) with respect to x.
///
/// ## Parameters and domain
///
/// a>0 and q in `[0,1]`; endpoints give +∞ and zero.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::gammainccinv(2.5_f64,0.3);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy gammainccinv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gammainccinv.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn gammainccinv<T: sci_core::Float>(a: T, q: T) -> T {
    crate::backend::gammainccinv(a, q).unwrap_or(T::nan())
}

/// Principal real logarithm of Γ(x).
///
/// ## Parameters and domain
///
/// x>0; negative x returns NaN, unlike gammaln.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::loggamma(2.5_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy loggamma](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.loggamma.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn loggamma<T: sci_core::Float>(x: T) -> T {
    crate::backend::loggamma(x).unwrap_or(T::nan())
}

/// Reciprocal gamma function 1/Γ(x), continued to zero at the poles.
///
/// ## Parameters and domain
///
/// All real x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::rgamma(-2.0_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy rgamma](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.rgamma.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn rgamma<T: sci_core::Float>(x: T) -> T {
    crate::backend::rgamma(x).unwrap_or(T::nan())
}

/// Order-n derivative of digamma, ψ^(n)(x).
///
/// ## Parameters and domain
///
/// Real x away from nonpositive integer poles; derivative order 0≤n≤1024.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::polygamma(2,2.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy polygamma](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.polygamma.html).
pub fn polygamma(n: u32, x: f64) -> f64 {
    if n > 1024 {
        return f64::NAN;
    }
    crate::backend::legacy::polygamma(n, x)
}

/// Log multivariate gamma: d(d−1)ln(π)/4 + Σⱼ₌₁ᵈ ln Γ(a+(1−j)/2).
///
/// ## Parameters and domain
///
/// 1≤d≤4096 and a>(d−1)/2.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::multigammaln(3.0,2);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy multigammaln](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.multigammaln.html).
pub fn multigammaln(a: f64, d: u32) -> f64 {
    if d == 0 || d > 4096 || a <= (d as f64 - 1.0) / 2.0 {
        return f64::NAN;
    }
    crate::backend::legacy::multigammaln(a, d)
}

/// Cylindrical Bessel function J_v(x), a solution of x²y″+xy′+(x²−v²)y=0.
///
/// ## Parameters and domain
///
/// Real order v; negative x requires integer v for a real value.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::jv(0.5_f64,2.0);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy jv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jv.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn jv<T: sci_core::Float>(v: T, x: T) -> T {
    crate::backend::cyl_bessel_j(v, x).unwrap_or(T::nan())
}

/// Exponentially scaled J_v(x); for real x this equals J_v(x).
///
/// ## Parameters and domain
///
/// Real order v; negative x requires integer v.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::jve(0.5_f64,2.0);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy jve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jve.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn jve<T: sci_core::Float>(v: T, x: T) -> T {
    crate::backend::cyl_bessel_je(v, x).unwrap_or(T::nan())
}

/// Cylindrical Bessel function Y_v(x), the second real Bessel solution.
///
/// ## Parameters and domain
///
/// Real v and positive x; x=0 is singular.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::yv(0.5_f64,2.0);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy yv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.yv.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn yv<T: sci_core::Float>(v: T, x: T) -> T {
    crate::backend::cyl_bessel_y(v, x).unwrap_or(T::nan())
}

/// Exponentially scaled Y_v(x); real positive x leaves the value unchanged.
///
/// ## Parameters and domain
///
/// Real v and positive x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::yve(0.5_f64,2.0);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy yve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.yve.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn yve<T: sci_core::Float>(v: T, x: T) -> T {
    crate::backend::cyl_bessel_ye(v, x).unwrap_or(T::nan())
}

/// Modified Bessel function I_v(x), solving x²y″+xy′−(x²+v²)y=0.
///
/// ## Parameters and domain
///
/// Real v; negative x requires integer v.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::iv(0.5_f64,2.0);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy iv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.iv.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn iv<T: sci_core::Float>(v: T, x: T) -> T {
    crate::backend::cyl_bessel_i(v, x).unwrap_or(T::nan())
}

/// Scaled modified Bessel function exp(−abs(x)) I_v(x).
///
/// ## Parameters and domain
///
/// Real v; negative x requires integer v.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::ive(0.5_f64,2.0);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy ive](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ive.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn ive<T: sci_core::Float>(v: T, x: T) -> T {
    crate::backend::cyl_bessel_ie(v, x).unwrap_or(T::nan())
}

/// Modified Bessel function K_v(x), decaying on the positive real axis.
///
/// ## Parameters and domain
///
/// Real v and positive x; zero is singular.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::kv(0.5_f64,2.0);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy kv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kv.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn kv<T: sci_core::Float>(v: T, x: T) -> T {
    crate::backend::cyl_bessel_k(v, x).unwrap_or(T::nan())
}

/// Scaled modified Bessel function exp(x) K_v(x).
///
/// ## Parameters and domain
///
/// Real v and positive x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::kve(0.5_f64,2.0);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy kve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kve.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn kve<T: sci_core::Float>(v: T, x: T) -> T {
    crate::backend::cyl_bessel_ke(v, x).unwrap_or(T::nan())
}

/// Bessel function J₀(x).
///
/// ## Parameters and domain
///
/// All real x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::j0(2.0_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy j0](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.j0.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn j0<T: sci_core::Float>(x: T) -> T {
    crate::backend::cyl_bessel_j0(x).unwrap_or(T::nan())
}

/// Bessel function J₁(x).
///
/// ## Parameters and domain
///
/// All real x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::j1(2.0_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy j1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.j1.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn j1<T: sci_core::Float>(x: T) -> T {
    crate::backend::cyl_bessel_j1(x).unwrap_or(T::nan())
}

/// Bessel function Y₀(x).
///
/// ## Parameters and domain
///
/// Positive x; zero returns −∞ and negative x returns NaN.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::y0(2.0_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy y0](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.y0.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn y0<T: sci_core::Float>(x: T) -> T {
    crate::backend::cyl_bessel_y0(x).unwrap_or(T::nan())
}

/// Bessel function Y₁(x).
///
/// ## Parameters and domain
///
/// Positive x; zero returns −∞ and negative x returns NaN.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::y1(2.0_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy y1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.y1.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn y1<T: sci_core::Float>(x: T) -> T {
    crate::backend::cyl_bessel_y1(x).unwrap_or(T::nan())
}

/// Modified Bessel function I₀(x).
///
/// ## Parameters and domain
///
/// All real x; large abs(x) overflows.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::i0(2.0_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy i0](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.i0.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn i0<T: sci_core::Float>(x: T) -> T {
    crate::backend::cyl_bessel_i0(x).unwrap_or(T::nan())
}

/// Modified Bessel function I₁(x).
///
/// ## Parameters and domain
///
/// All real x; large abs(x) overflows.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::i1(2.0_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy i1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.i1.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn i1<T: sci_core::Float>(x: T) -> T {
    crate::backend::cyl_bessel_i1(x).unwrap_or(T::nan())
}

/// Scaled modified Bessel function exp(−abs(x)) I₀(x).
///
/// ## Parameters and domain
///
/// All real x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::i0e(2.0_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy i0e](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.i0e.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn i0e<T: sci_core::Float>(x: T) -> T {
    crate::backend::cyl_bessel_i0e(x).unwrap_or(T::nan())
}

/// Scaled modified Bessel function exp(−abs(x)) I₁(x).
///
/// ## Parameters and domain
///
/// All real x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::i1e(2.0_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy i1e](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.i1e.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn i1e<T: sci_core::Float>(x: T) -> T {
    crate::backend::cyl_bessel_i1e(x).unwrap_or(T::nan())
}

/// Modified Bessel function K₀(x).
///
/// ## Parameters and domain
///
/// Positive x; zero returns +∞.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::k0(2.0_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy k0](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.k0.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn k0<T: sci_core::Float>(x: T) -> T {
    crate::backend::cyl_bessel_k0(x).unwrap_or(T::nan())
}

/// Modified Bessel function K₁(x).
///
/// ## Parameters and domain
///
/// Positive x; zero returns +∞.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::k1(2.0_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy k1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.k1.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn k1<T: sci_core::Float>(x: T) -> T {
    crate::backend::cyl_bessel_k1(x).unwrap_or(T::nan())
}

/// Scaled modified Bessel function exp(x) K₀(x).
///
/// ## Parameters and domain
///
/// Positive x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::k0e(2.0_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy k0e](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.k0e.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn k0e<T: sci_core::Float>(x: T) -> T {
    crate::backend::cyl_bessel_k0e(x).unwrap_or(T::nan())
}

/// Scaled modified Bessel function exp(x) K₁(x).
///
/// ## Parameters and domain
///
/// Positive x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::k1e(2.0_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy k1e](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.k1e.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn k1e<T: sci_core::Float>(x: T) -> T {
    crate::backend::cyl_bessel_k1e(x).unwrap_or(T::nan())
}

/// Weighted Bessel integral ∫₀¹ x^lambda J_v(2ax) dx.
///
/// ## Parameters and domain
///
/// a≥0 and lambda+v>−1 for the convergent integral.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::besselpoly(1.0,0.5,2.0);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy besselpoly](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.besselpoly.html).
pub fn besselpoly(a: f64, lambda: f64, v: f64) -> f64 {
    crate::backend::legacy::besselpoly(a, lambda, v)
}

/// Integrals (∫₀ˣ J₀(t)dt, ∫₀ˣ Y₀(t)dt).
///
/// ## Parameters and domain
///
/// Nonnegative x for both real components.
///
/// ## Returns
///
/// The pair `(∫₀ˣ J₀(t) dt, ∫₀ˣ Y₀(t) dt)`, with the integrable Y₀
/// singularity interpreted as a limit. At zero both entries are zero.
///
/// ## Examples
///
/// ```
/// let value = sci_special::itj0y0(2.0);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy itj0y0](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.itj0y0.html).
pub fn itj0y0(x: f64) -> (f64, f64) {
    crate::backend::legacy::it1j0y0(x)
}

/// Integrals (∫₀ˣ (1−J₀(t))/t dt, ∫ₓ^∞ Y₀(t)/t dt).
///
/// ## Parameters and domain
///
/// Positive x.
///
/// ## Returns
///
/// The pair `(∫₀ˣ (1−J₀(t))/t dt, ∫ₓ^∞ Y₀(t)/t dt)`. The first
/// integrand has a removable endpoint value; the second integral is improper.
///
/// ## Examples
///
/// ```
/// let value = sci_special::it2j0y0(2.0);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy it2j0y0](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.it2j0y0.html).
pub fn it2j0y0(x: f64) -> (f64, f64) {
    crate::backend::legacy::it2j0y0(x)
}

/// Integrals (∫₀ˣ I₀(t)dt, ∫₀ˣ K₀(t)dt).
///
/// ## Parameters and domain
///
/// Nonnegative x.
///
/// ## Returns
///
/// The pair `(∫₀ˣ I₀(t) dt, ∫₀ˣ K₀(t) dt)`. The logarithmic K₀
/// singularity is integrable. The I₀ integral can overflow at large positive `x`.
///
/// ## Examples
///
/// ```
/// let value = sci_special::iti0k0(2.0);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy iti0k0](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.iti0k0.html).
pub fn iti0k0(x: f64) -> (f64, f64) {
    crate::backend::legacy::it1i0k0(x)
}

/// Integrals (∫₀ˣ (I₀(t)−1)/t dt, ∫ₓ^∞ K₀(t)/t dt).
///
/// ## Parameters and domain
///
/// Positive x.
///
/// ## Returns
///
/// The pair `(∫₀ˣ (I₀(t)−1)/t dt, ∫ₓ^∞ K₀(t)/t dt)`. The first
/// integrand has a removable endpoint value; the second diverges at zero.
///
/// ## Examples
///
/// ```
/// let value = sci_special::it2i0k0(2.0);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy it2i0k0](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.it2i0k0.html).
pub fn it2i0k0(x: f64) -> (f64, f64) {
    crate::backend::legacy::it2i0k0(x)
}

/// Struve function H_v(x), the inhomogeneous Bessel solution.
///
/// ## Parameters and domain
///
/// Real v and x; negative x requires integer v for real continuation.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::struve(0.5_f64,2.0);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy struve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.struve.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn struve<T: sci_core::Float>(v: T, x: T) -> T {
    crate::backend::struve_h(v, x).unwrap_or(T::nan())
}

/// Modified Struve function L_v(x).
///
/// ## Parameters and domain
///
/// Real v and x; negative x requires integer v.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::modstruve(0.5_f64,2.0);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy modstruve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.modstruve.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn modstruve<T: sci_core::Float>(v: T, x: T) -> T {
    crate::backend::struve_l(v, x).unwrap_or(T::nan())
}

/// Integral ∫₀ˣ H₀(t) dt.
///
/// ## Parameters and domain
///
/// Nonnegative x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Numerical behavior and work limits
///
/// Uses the integrated defining series with double-double accumulation and at
/// most 127 updates for `abs(x) <= 24`, composite 32-point Gaussian quadrature
/// on intervals at most four units wide for `24 < abs(x) <= 128`, and the
/// translated asymptotic expansion beyond 128. The integral is even in `x`.
/// Nonfinite inputs return NaN. Large-argument asymptotic accuracy is limited.
///
/// ## Examples
///
/// ```
/// let value = sci_special::itstruve0(2.0);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy itstruve0](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.itstruve0.html).
///
/// ## Numerical behavior and work limits
///
/// For `abs(x) <= 128`, bounded Gaussian quadrature evaluates the defining integral using
/// the Struve kernel. Larger arguments use the translated asymptotic expansion. The result
/// follows the defining integral where SciPy's large-argument implementation disagrees.
pub fn itstruve0(x: f64) -> f64 {
    crate::backend::legacy::itstruve0(x)
}

/// Integral ∫ₓ^∞ H₀(t)/t dt.
///
/// ## Parameters and domain
///
/// Positive x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::it2struve0(2.0);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy it2struve0](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.it2struve0.html).
pub fn it2struve0(x: f64) -> f64 {
    crate::backend::legacy::it2struve0(x)
}

/// Integral ∫₀ˣ L₀(t) dt.
///
/// ## Parameters and domain
///
/// Nonnegative x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::itmodstruve0(2.0);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy itmodstruve0](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.itmodstruve0.html).
pub fn itmodstruve0(x: f64) -> f64 {
    crate::backend::legacy::itmodstruve0(x)
}

/// Confluent hypergeometric limit function Σₖ x^k/((b)ₖ k!).
///
/// ## Parameters and domain
///
/// Real b,x; b must avoid nonpositive integer poles.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::hyp0f1(2.5_f64,0.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy hyp0f1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hyp0f1.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn hyp0f1<T: sci_core::Float>(b: T, x: T) -> T {
    crate::backend::hyp0f1(b, x).unwrap_or(T::nan())
}

/// Kummer function ₁F₁(a;b;x)=Σₖ (a)ₖ x^k/((b)ₖ k!).
///
/// ## Parameters and domain
///
/// Real a,b,x away from denominator poles; terminating cancellations follow SciPy.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Accuracy limitations
///
/// Parameter work bounds do not imply uniform accuracy. Cancellation and
/// continuation can lose digits or return NaN; difficult large-parameter cases
/// require independent validation.
/// For `a > 0`, `b < a`, `-8 <= b <= 64` and `-1000 <= x < 0`,
/// outward Taylor continuation avoids cancellation of the defining series.
/// Integer `a-b` uses a normalized Laguerre recurrence with exponential scaling
/// instead, since that solution is recessive on the negative ray.
/// More negative `b`, other parameter regions, proximity to poles/zeros and
/// work-limit exhaustion remain accuracy limitations; there is no uniform
/// certification of the whole parameter box.
///
/// ## Examples
///
/// ```
/// let value = sci_special::hyp1f1(1.5,2.5,0.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy hyp1f1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hyp1f1.html).
///
/// ## Numerical behavior and work limits
///
/// Finite real parameters and argument; `abs(a) <= 100000`. Continuation uses
/// double-double arithmetic, at most 256 origin-series terms and 8192 Taylor
/// steps of at most 128 terms each. The integer-difference recurrence takes at
/// most 100008 steps. Failed convergence, nonfinite continuation intermediates
/// or exhausted continuation work return NaN. Other regions use bounded series
/// and asymptotic expansions that can lose relative accuracy under cancellation.
///
/// [NIST DLMF §13.29(ii)](https://dlmf.nist.gov/13.29#ii) describes outward
/// integration; [§13.2](https://dlmf.nist.gov/13.2) gives the defining equation.
pub fn hyp1f1(a: f64, b: f64, x: f64) -> f64 {
    crate::backend::legacy::hyp1f1(a, b, x)
}

/// Gauss function `₂F₁(a,b;c;x) = Σₖ (a)ₖ(b)ₖ xᵏ / [(c)ₖ k!]`.
/// The series defines the value for `abs(x) < 1`; real continuation is used
/// where implemented. Here `(a)ₖ = Γ(a+k)/Γ(a)` is the rising factorial.
///
/// ## Parameters and domain
///
/// Real parameters; real continuation generally requires x≤1; denominator poles may be singular.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Accuracy limitations
///
/// Parameter work bounds do not imply uniform accuracy. Cancellation and
/// continuation can lose digits or return NaN; difficult large-parameter cases
/// require independent validation.
///
/// ## Examples
///
/// ```
/// let value = sci_special::hyp2f1(0.5,1.0,2.5,0.3);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy hyp2f1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hyp2f1.html).
///
/// ## Numerical behavior and work limits
///
/// Finite real inputs with `abs(a), abs(b), abs(c) <= 100000`. Series and continuation
/// iterations are bounded; unsupported real continuations return NaN.
pub fn hyp2f1(a: f64, b: f64, c: f64, x: f64) -> f64 {
    crate::backend::legacy::hyp2f1(a, b, c, x)
}

/// Tricomi function `U(a,b,x)`, solving `x y″ + (b−x)y′ − a y = 0`.
/// For `a > 0` and `x > 0`, `U(a,b,x) = ∫₀^∞ exp(−xt)t^(a−1)(1+t)^(b−a−1) dt / Γ(a)`;
/// other supported real parameters use continuation of this solution.
///
/// ## Parameters and domain
///
/// Real a,b and positive x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Accuracy limitations
///
/// Parameter work bounds do not imply uniform accuracy. Cancellation and
/// continuation can lose digits or return NaN; difficult large-parameter cases
/// require independent validation.
///
/// ## Examples
///
/// ```
/// let value = sci_special::hyperu(1.5,2.5,0.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy hyperu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hyperu.html).
///
/// ## Numerical behavior and work limits
///
/// Finite real inputs with `abs(a), abs(b) <= 100000` and `x >= 0`. Series, integral, and
/// recurrence algorithms have finite work limits; cancellation limits accuracy.
pub fn hyperu(a: f64, b: f64, x: f64) -> f64 {
    crate::backend::legacy::hypu(a, b, x)
}

/// Exponential integral E₁(x)=∫ₓ^∞ exp(−t)/t dt.
///
/// ## Parameters and domain
///
/// x≥0; zero gives +∞; negative x returns NaN.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::exp1(0.5_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy exp1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.exp1.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn exp1<T: sci_core::Float>(x: T) -> T {
    crate::backend::exp1(x).unwrap_or(T::nan())
}

/// Exponential integral Ei(x), defined by the real Cauchy principal value.
///
/// ## Parameters and domain
///
/// All real x; zero gives −∞.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::expi(0.5_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy expi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expi.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn expi<T: sci_core::Float>(x: T) -> T {
    crate::backend::expi(x).unwrap_or(T::nan())
}

/// Generalized exponential integral E_n(x)=∫₁^∞ exp(−xt)/t^n dt.
///
/// ## Parameters and domain
///
/// Nonnegative integer n and x≥0; divergent endpoint cases give +∞.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::expn(3,0.5_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy expn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expn.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn expn<T: sci_core::Float>(n: u32, x: T) -> T {
    crate::backend::expn(n as i32, x).unwrap_or(T::nan())
}

/// Sine and cosine integrals (Si(x),Ci(x)); Ci uses the real principal value on the negative axis.
///
/// ## Parameters and domain
///
/// All real x; Ci(0)=−∞.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::sici(0.5);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy sici](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sici.html).
pub fn sici(x: f64) -> (f64, f64) {
    crate::backend::legacy::sici(x)
}

/// Hyperbolic sine and cosine integrals (Shi(x),Chi(x)), with real principal values.
///
/// ## Parameters and domain
///
/// All real x; Chi(0)=−∞.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::shichi(0.5);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy shichi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.shichi.html).
pub fn shichi(x: f64) -> (f64, f64) {
    crate::backend::legacy::shichi(x)
}

/// Fresnel integrals (S,C) of sin(πt²/2) and cos(πt²/2) from 0 to x.
///
/// ## Parameters and domain
///
/// All real x; the return order is S then C.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::fresnel(0.5);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy fresnel](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.fresnel.html).
pub fn fresnel(x: f64) -> (f64, f64) {
    crate::backend::legacy::fresnel(x)
}

/// Spence dilogarithm Li₂(1−x)=−∫₁ˣ ln(t)/(t−1) dt.
///
/// ## Parameters and domain
///
/// x≥0 for real values.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::spence(0.5_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy spence](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.spence.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn spence<T: sci_core::Float>(x: T) -> T {
    crate::backend::spence(x).unwrap_or(T::nan())
}

/// Hurwitz zeta ζ(s,q)=Σₖ₌₀^∞ (k+q)^(−s).
///
/// ## Parameters and domain
///
/// s>1 and q>0 for the defining series; poles and real continuation follow SciPy.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::zeta(2.5_f64,1.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy zeta](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.zeta.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn zeta<T: sci_core::Float>(s: T, q: T) -> T {
    crate::backend::zeta(s, q).unwrap_or(T::nan())
}

/// Riemann zeta minus one, ζ(x)−1.
///
/// ## Parameters and domain
///
/// Real x away from the pole at one.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::zetac(2.5_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy zetac](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.zetac.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn zetac<T: sci_core::Float>(x: T) -> T {
    crate::backend::zetac(x).unwrap_or(T::nan())
}

/// Real Wright omega, solving w+ln(w)=x with w≥0.
///
/// ## Parameters and domain
///
/// All real x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::wrightomega(0.5_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy wrightomega](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.wrightomega.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn wrightomega<T: sci_core::Float>(x: T) -> T {
    crate::backend::wrightomega(x).unwrap_or(T::nan())
}

/// Wright generalized Bessel function Σₖ x^k/(k! Γ(ak+b)).
///
/// ## Parameters and domain
///
/// Nonnegative a,b,x; invalid arguments return NaN.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::wright_bessel(0.5,1.5,0.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy wright_bessel](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.wright_bessel.html).
///
/// ## Numerical behavior and work limits
///
/// The logarithmically scaled positive defining series uses at most 100000 terms. Finite
/// nonnegative parameters are required; failure to converge within the budget returns NaN.
/// True overflow returns infinity.
pub fn wright_bessel(a: f64, b: f64, x: f64) -> f64 {
    crate::backend::legacy::wright_bessel(a, b, x)
}

/// Natural logarithm of the Wright generalized Bessel function, evaluated in scaled form.
///
/// ## Parameters and domain
///
/// Nonnegative a,b,x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::log_wright_bessel(0.5,1.5,0.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy log_wright_bessel](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.log_wright_bessel.html).
///
/// ## Numerical behavior and work limits
///
/// The logarithmically scaled positive defining series uses at most 100000 terms. Finite
/// nonnegative parameters are required; failure to converge within the budget returns NaN.
/// Summation retains large logarithms without exponentiating the final result.
pub fn log_wright_bessel(a: f64, b: f64, x: f64) -> f64 {
    crate::backend::legacy::log_wright_bessel(a, b, x)
}

/// Convolution of a normal density with standard deviation sigma and a Cauchy density with half-width gamma.
///
/// ## Parameters and domain
///
/// sigma≥0 and gamma≥0; both zero give +∞ at x=0 and zero elsewhere.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::voigt_profile(0.5_f64,1.0,0.3);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy voigt_profile](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.voigt_profile.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn voigt_profile<T: sci_core::Float>(x: T, sigma: T, gamma: T) -> T {
    crate::backend::voigt_profile(x, sigma, gamma).unwrap_or(T::nan())
}

/// Binomial CDF P(X≤floor(k)), X~Binomial(n,p).
///
/// ## Parameters and domain
///
/// n≥0, k≥0, p in `[0,1]`.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::bdtr(2.0_f64,5,0.3);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy bdtr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.bdtr.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn bdtr<T: sci_core::Float>(k: T, n: i32, p: T) -> T {
    crate::backend::bdtr(k, n, p).unwrap_or(T::nan())
}

/// Binomial survival P(X>floor(k)), computed without subtracting the CDF.
///
/// ## Parameters and domain
///
/// n≥0, k≥0, p in `[0,1]`.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::bdtrc(2.0_f64,5,0.3);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy bdtrc](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.bdtrc.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn bdtrc<T: sci_core::Float>(k: T, n: i32, p: T) -> T {
    crate::backend::bdtrc(k, n, p).unwrap_or(T::nan())
}

/// Inverse binomial CDF with respect to success probability p.
///
/// ## Parameters and domain
///
/// 0≤k<n, n≥1, y in `[0,1]`.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::bdtri(2.0_f64,5,0.3);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy bdtri](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.bdtri.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn bdtri<T: sci_core::Float>(k: T, n: i32, y: T) -> T {
    crate::backend::bdtri(k, n, y).unwrap_or(T::nan())
}

/// Chi-square CDF P(df/2,x/2).
///
/// ## Parameters and domain
///
/// df>0 and x≥0.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::chdtr(3.0_f64,2.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy chdtr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.chdtr.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn chdtr<T: sci_core::Float>(df: T, x: T) -> T {
    crate::backend::chdtr(df, x).unwrap_or(T::nan())
}

/// Chi-square survival Q(df/2,x/2).
///
/// ## Parameters and domain
///
/// df>0 and x≥0.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::chdtrc(3.0_f64,2.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy chdtrc](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.chdtrc.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn chdtrc<T: sci_core::Float>(df: T, x: T) -> T {
    crate::backend::chdtrc(df, x).unwrap_or(T::nan())
}

/// Inverse chi-square survival function.
///
/// ## Parameters and domain
///
/// df>0 and q in `[0,1]`.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::chdtri(3.0_f64,0.3);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy chdtri](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.chdtri.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn chdtri<T: sci_core::Float>(df: T, q: T) -> T {
    crate::backend::chdtri(df, q).unwrap_or(T::nan())
}

/// F-distribution CDF I_(dfn*x/(dfd+dfn*x))(dfn/2,dfd/2).
///
/// ## Parameters and domain
///
/// Positive degrees of freedom and x≥0.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::fdtr(3.0_f64,5.0,2.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy fdtr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.fdtr.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn fdtr<T: sci_core::Float>(dfn: T, dfd: T, x: T) -> T {
    crate::backend::fdtr(dfn, dfd, x).unwrap_or(T::nan())
}

/// F-distribution survival function, evaluated directly in the upper tail.
///
/// ## Parameters and domain
///
/// Positive degrees of freedom and x≥0.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::fdtrc(3.0_f64,5.0,2.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy fdtrc](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.fdtrc.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn fdtrc<T: sci_core::Float>(dfn: T, dfd: T, x: T) -> T {
    crate::backend::fdtrc(dfn, dfd, x).unwrap_or(T::nan())
}

/// Inverse F-distribution CDF with respect to x.
///
/// ## Parameters and domain
///
/// Positive degrees of freedom and p in `[0,1]`.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::fdtri(3.0_f64,5.0,0.3);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy fdtri](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.fdtri.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn fdtri<T: sci_core::Float>(dfn: T, dfd: T, p: T) -> T {
    crate::backend::fdtri(dfn, dfd, p).unwrap_or(T::nan())
}

/// Gamma CDF P(b,a*x), with rate a and shape b.
///
/// ## Parameters and domain
///
/// Positive a,b and nonnegative x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::gdtr(1.5_f64,2.5,0.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy gdtr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gdtr.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn gdtr<T: sci_core::Float>(a: T, b: T, x: T) -> T {
    crate::backend::gdtr(a, b, x).unwrap_or(T::nan())
}

/// Gamma survival Q(b,a*x), with rate a and shape b.
///
/// ## Parameters and domain
///
/// Positive a,b and nonnegative x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::gdtrc(1.5_f64,2.5,0.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy gdtrc](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gdtrc.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn gdtrc<T: sci_core::Float>(a: T, b: T, x: T) -> T {
    crate::backend::gdtrc(a, b, x).unwrap_or(T::nan())
}

/// Inverse of gdtr with respect to shape b.
///
/// ## Parameters and domain
///
/// a,x>0 and p in (0,1).
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::gdtrib(1.5,0.3,0.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy gdtrib](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gdtrib.html).
pub fn gdtrib(a: f64, p: f64, x: f64) -> f64 {
    crate::backend::legacy::gdtrib(a, p, x)
}

/// Negative-binomial CDF for k failures before n successes, I_p(n,k+1).
///
/// ## Parameters and domain
///
/// k≥0, n≥1, p in `[0,1]`.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::nbdtr(2,5,0.3_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy nbdtr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.nbdtr.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn nbdtr<T: sci_core::Float>(k: i32, n: i32, p: T) -> T {
    crate::backend::nbdtr(k, n, p).unwrap_or(T::nan())
}

/// Negative-binomial survival probability.
///
/// ## Parameters and domain
///
/// k≥0, n≥1, p in `[0,1]`.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::nbdtrc(2,5,0.3_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy nbdtrc](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.nbdtrc.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn nbdtrc<T: sci_core::Float>(k: i32, n: i32, p: T) -> T {
    crate::backend::nbdtrc(k, n, p).unwrap_or(T::nan())
}

/// Inverse negative-binomial CDF with respect to p.
///
/// ## Parameters and domain
///
/// k≥0, n≥1, y in `[0,1]`.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::nbdtri(2,5,0.3_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy nbdtri](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.nbdtri.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn nbdtri<T: sci_core::Float>(k: i32, n: i32, y: T) -> T {
    crate::backend::nbdtri(k, n, y).unwrap_or(T::nan())
}

/// Standard normal CDF Φ(x)=erfc(−x/√2)/2.
///
/// ## Parameters and domain
///
/// All real x; infinite endpoints give zero and one.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::ndtr(0.5_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy ndtr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ndtr.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn ndtr<T: sci_core::Float>(x: T) -> T {
    crate::backend::ndtr(x).unwrap_or(T::nan())
}

/// Inverse standard normal CDF Φ⁻¹(p).
///
/// ## Parameters and domain
///
/// p in `[0,1]`; endpoints give signed infinity.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::ndtri(0.3_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy ndtri](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ndtri.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn ndtri<T: sci_core::Float>(p: T) -> T {
    crate::backend::ndtri(p).unwrap_or(T::nan())
}

/// Log standard normal CDF ln Φ(x), stable in the far negative tail.
///
/// ## Parameters and domain
///
/// All real x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::log_ndtr(-30.0_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy log_ndtr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.log_ndtr.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn log_ndtr<T: sci_core::Float>(x: T) -> T {
    crate::backend::log_ndtr(x).unwrap_or(T::nan())
}

/// Inverse log standard normal CDF, Φ⁻¹(exp(y)), without underflow.
///
/// ## Parameters and domain
///
/// y≤0; −∞ gives −∞ and zero gives +∞.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::ndtri_exp(-100.0_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy ndtri_exp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ndtri_exp.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn ndtri_exp<T: sci_core::Float>(y: T) -> T {
    crate::backend::ndtri_exp(y).unwrap_or(T::nan())
}

/// Poisson CDF P(X≤floor(k)), X~Poisson(m).
///
/// ## Parameters and domain
///
/// k≥0 and m≥0.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::pdtr(2.0_f64,1.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy pdtr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pdtr.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn pdtr<T: sci_core::Float>(k: T, m: T) -> T {
    crate::backend::pdtr(k, m).unwrap_or(T::nan())
}

/// Poisson survival P(X>floor(k)).
///
/// ## Parameters and domain
///
/// k≥0 and m≥0.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::pdtrc(2.0_f64,1.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy pdtrc](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pdtrc.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn pdtrc<T: sci_core::Float>(k: T, m: T) -> T {
    crate::backend::pdtrc(k, m).unwrap_or(T::nan())
}

/// Inverse Poisson CDF with respect to mean m.
///
/// ## Parameters and domain
///
/// k≥0 and p in `[0,1]`.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::pdtri(2,0.3_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy pdtri](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pdtri.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn pdtri<T: sci_core::Float>(k: i32, p: T) -> T {
    crate::backend::pdtri(k, p).unwrap_or(T::nan())
}

/// Student t CDF with df degrees of freedom.
/// `F(t)=P(T_df ≤ t)` for the standard, zero-location Student distribution.
///
/// ## Parameters and domain
///
/// df>0; all real t.
/// NaN inputs and invalid degrees of freedom return NaN; infinite t gives
/// zero or one, and positive infinite df gives the standard-normal limit.
///
/// ## Numerical behavior
///
/// Uses elementary Cauchy and df=2 formulas, with reciprocal tails to avoid
/// cancellation and overflow. Other degrees use private XSF incomplete-beta
/// evaluation. For `|t|≤sqrt(df)` and finite
/// positive df, evaluates `F(t)=1/2+sign(t) I_q(1/2,df/2)/2`,
/// `q=(t/sqrt(df))²/(1+(t/sqrt(df))²)`, preserving variation near the median.
/// The backend tail formula handles other arguments. Binary64 rounding,
/// underflow, and extreme shape parameters limit accuracy.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::stdtr(3.0,0.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy stdtr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.stdtr.html).
pub fn stdtr(df: f64, t: f64) -> f64 {
    if df.is_nan() || df <= 0.0 || t.is_nan() {
        f64::NAN
    } else if t == f64::NEG_INFINITY {
        0.0
    } else if t == f64::INFINITY {
        1.0
    } else if t == 0.0 {
        0.5
    } else if df == 1.0 {
        if t.abs() <= 1.0 {
            0.5 + t.atan() / std::f64::consts::PI
        } else {
            let tail = (1.0 / t.abs()).atan() / std::f64::consts::PI;
            if t < 0.0 { tail } else { 1.0 - tail }
        }
    } else if df == 2.0 {
        let norm = t.hypot(std::f64::consts::SQRT_2);
        if t.abs() <= std::f64::consts::SQRT_2 {
            0.5 + 0.5 * (t / norm)
        } else {
            let tail = ((1.0 / norm) / (1.0 + t.abs() / norm)) / norm;
            if t < 0.0 { tail } else { 1.0 - tail }
        }
    } else if df.is_finite() && t.abs() <= df.sqrt() {
        let ratio = (t / df.sqrt()).powi(2);
        let probability =
            0.5 * crate::backend::betainc(0.5, 0.5 * df, ratio / (1.0 + ratio)).unwrap_or(f64::NAN);
        0.5 + probability.copysign(t)
    } else {
        crate::backend::stdtr(df, t).unwrap_or(f64::NAN)
    }
}

/// Inverse Student t CDF with respect to t.
/// Solves `P(T_df ≤ t)=p` for the standard, zero-location Student distribution.
///
/// ## Parameters and domain
///
/// df>0 and p in `[0,1]`.
/// NaN and invalid parameters return NaN; endpoint probabilities return signed
/// infinities and the median returns zero.
///
/// ## Numerical behavior
///
/// Uses elementary Cauchy and df=2 formulas, with reciprocal tangents and
/// separate square roots for extreme probabilities. Other degrees use private
/// XSF incomplete-beta inversion. For `0.25 < p < 0.75`,
/// `q=I⁻¹_(|2p−1|)(1/2,df/2)` and `t=sign(p−1/2) sqrt(df*q/(1−q))`
/// avoid subtracting an inverse-beta value close to one. If q exceeds 1/2,
/// or for other probabilities, the backend tail formula avoids cancellation
/// in `1−q` at small degrees of freedom. Binary64 rounding and backend convergence
/// limit accuracy, particularly for extreme parameters.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::stdtrit(3.0,0.3);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy stdtrit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.stdtrit.html).
pub fn stdtrit(df: f64, p: f64) -> f64 {
    if df.is_nan() || df <= 0.0 || !(0.0..=1.0).contains(&p) {
        f64::NAN
    } else if p == 0.0 {
        f64::NEG_INFINITY
    } else if p == 1.0 {
        f64::INFINITY
    } else if p == 0.5 {
        0.0
    } else if df == 1.0 {
        if p < 0.25 {
            -1.0 / (std::f64::consts::PI * p).tan()
        } else if p > 0.75 {
            1.0 / (std::f64::consts::PI * (1.0 - p)).tan()
        } else {
            (std::f64::consts::PI * (p - 0.5)).tan()
        }
    } else if df == 2.0 {
        ((p - 0.5) * std::f64::consts::SQRT_2 / p.sqrt()) / (1.0 - p).sqrt()
    } else if df.is_finite() && p > 0.25 && p < 0.75 {
        let q =
            crate::backend::betaincinv(0.5, 0.5 * df, (2.0 * p - 1.0).abs()).unwrap_or(f64::NAN);
        if q <= 0.5 {
            (df * q / (1.0 - q)).sqrt().copysign(p - 0.5)
        } else {
            crate::backend::stdtrit(df, p).unwrap_or(f64::NAN)
        }
    } else {
        crate::backend::stdtrit(df, p).unwrap_or(f64::NAN)
    }
}

/// Limiting two-sided Kolmogorov survival probability 2Σₖ₌₁^∞ (−1)^(k−1) exp(−2k²x²).
///
/// ## Parameters and domain
///
/// x≥0; zero gives one.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::kolmogorov(0.5_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy kolmogorov](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kolmogorov.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn kolmogorov<T: sci_core::Float>(x: T) -> T {
    crate::backend::kolmogorov(x).unwrap_or(T::nan())
}

/// Inverse limiting Kolmogorov survival probability.
///
/// ## Parameters and domain
///
/// q in `[0,1]`.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::kolmogi(0.3_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy kolmogi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kolmogi.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn kolmogi<T: sci_core::Float>(q: T) -> T {
    crate::backend::kolmogi(q).unwrap_or(T::nan())
}

/// One-sided Kolmogorov–Smirnov survival probability for sample size n.
///
/// ## Parameters and domain
///
/// n≥1 and x in `[0,1]`.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::smirnov(5,0.3);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy smirnov](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.smirnov.html).
pub fn smirnov(n: i32, x: f64) -> f64 {
    crate::backend::legacy::smirnov(n, x)
}

/// Inverse one-sided Kolmogorov–Smirnov survival probability.
///
/// ## Parameters and domain
///
/// n≥1 and q in `[0,1]`.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::smirnovi(5,0.3);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy smirnovi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.smirnovi.html).
pub fn smirnovi(n: i32, q: f64) -> f64 {
    crate::backend::legacy::smirnovi(n, q)
}

/// Owen's T: (1/(2π)) ∫₀ᵃ exp(−h²(1+t²)/2)/(1+t²) dt.
///
/// ## Parameters and domain
///
/// All finite real h,a; infinite limits follow SciPy.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::owens_t(0.5,0.3);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy owens_t](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.owens_t.html).
pub fn owens_t(h: f64, a: f64) -> f64 {
    crate::backend::legacy::owens_t(h, a)
}

/// Tukey lambda CDF, inverse of (p^lambda−(1−p)^lambda)/lambda.
///
/// ## Parameters and domain
///
/// All real x and lambda, including the logistic limit lambda=0.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::tklmbda(0.5_f64,0.3);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy tklmbda](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.tklmbda.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn tklmbda<T: sci_core::Float>(x: T, lambda: T) -> T {
    crate::backend::tukeylambdacdf(x, lambda).unwrap_or(T::nan())
}

/// Parabolic cylinder function and derivative (D_v(x),D′_v(x)).
///
/// ## Parameters and domain
///
/// Real order v and real x.
///
/// ## Returns
///
/// The pair `(D_v(x), D′_v(x))`, with derivative taken with respect to `x`.
/// Both entries are NaN on invalid inputs or backend failure.
///
/// ## Numerical behavior
///
/// Solves `y″ + (v + 1/2 − x²/4)y = 0` using bounded binary64 series,
/// asymptotics, and order recurrences. Finite inputs require `abs(v) <= 4096`.
/// The recessive normalization is `D_v(x) ~ x^v exp(−x²/4)` as `x → +∞`.
/// Accuracy is limited near zeros and in dominant/recessive transition regions.
///
/// ## Examples
///
/// ```
/// let value = sci_special::pbdv(0.5,1.0);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy pbdv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pbdv.html).
///
/// ## Numerical behavior and work limits
///
/// Finite real inputs with `abs(v) <= 4096`. Bounded forward/backward recurrences return
/// `(value, derivative)`; larger orders or nonfinite inputs return NaN pairs.
pub fn pbdv(v: f64, x: f64) -> (f64, f64) {
    crate::backend::legacy::pbdv(v, x)
}

/// Parabolic cylinder function and derivative (V_v(x),V′_v(x)).
///
/// ## Parameters and domain
///
/// Real order v and real x.
///
/// ## Returns
///
/// The pair `(V_v(x), V′_v(x))`, with derivative taken with respect to `x`.
/// Both entries are NaN on invalid inputs or backend failure.
///
/// ## Numerical behavior
///
/// Uses the real standard parabolic-cylinder V solution of
/// `y″ + (v + 1/2 − x²/4)y = 0`, with the Zhang–Jin/SciPy normalization.
/// Binary64 series, asymptotics, and bounded order recurrences require finite
/// inputs and `abs(v) <= 4096`. Relative accuracy degrades near zeros.
///
/// ## Examples
///
/// ```
/// let value = sci_special::pbvv(0.5,1.0);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy pbvv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pbvv.html).
///
/// ## Numerical behavior and work limits
///
/// Finite real inputs with `abs(v) <= 4096`. Bounded forward/backward recurrences return
/// `(value, derivative)`; larger orders or nonfinite inputs return NaN pairs.
pub fn pbvv(v: f64, x: f64) -> (f64, f64) {
    crate::backend::legacy::pbvv(v, x)
}

/// Parabolic cylinder function (W(a,x),W′(a,x)).
///
/// ## Parameters and domain
///
/// abs(a)≤5 and abs(x)≤5 for the supported algorithm.
///
/// ## Returns
///
/// The pair `(W(a,x), W′(a,x))`, with derivative taken with respect to `x`.
/// Out-of-range or nonfinite inputs return `(NaN, NaN)`.
///
/// ## Numerical behavior
///
/// Solves `y″ + (x²/4 − a)y = 0` in the real standard W normalization
/// (DLMF 12.14). Finite `abs(a), abs(x) <= 5` are required. The even and odd
/// power series use 202 coefficient slots and 200 recurrence updates; there
/// is no configurable tolerance. Accuracy is parameter dependent.
///
/// ## Examples
///
/// ```
/// let value = sci_special::pbwa(0.5,1.0);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy pbwa](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pbwa.html).
pub fn pbwa(a: f64, x: f64) -> (f64, f64) {
    crate::backend::legacy::pbwa(a, x)
}

/// Even Mathieu characteristic value a_m(q).
///
/// ## Parameters and domain
///
/// Nonnegative integer order m and real q.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Definition and numerical behavior
///
/// Solves `y″ + [a − 2q cos(2z)]y = 0`. The characteristic value selects the periodic
/// even solution. A tridiagonal Fourier recurrence uses at most 251 terms,
/// 128 Sturm bisections, and four inverse iterations. Work limits do not
/// guarantee uniform accuracy near degeneracies.
///
/// [NIST DLMF, Mathieu functions](https://dlmf.nist.gov/28).
///
/// ## Examples
///
/// ```
/// let value = sci_special::mathieu_a(2,0.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy mathieu_a](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.mathieu_a.html).
/// ## Work limits
///
/// Orders must satisfy m≤128 and finite abs(q)≤100.
/// Inputs outside these bounds return NaN without entering native work buffers.
pub fn mathieu_a(m: u32, q: f64) -> f64 {
    if m > 128 || !q.is_finite() || q.abs() > 100.0 {
        return f64::NAN;
    }
    crate::backend::legacy::mathieu_a(m, q)
}

/// Odd Mathieu characteristic value b_m(q).
///
/// ## Parameters and domain
///
/// Positive integer order m and real q.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Definition and numerical behavior
///
/// Solves `y″ + [a − 2q cos(2z)]y = 0`. The characteristic value selects the periodic
/// odd solution. A tridiagonal Fourier recurrence uses at most 251 terms,
/// 128 Sturm bisections, and four inverse iterations. Work limits do not
/// guarantee uniform accuracy near degeneracies.
///
/// [NIST DLMF, Mathieu functions](https://dlmf.nist.gov/28).
///
/// ## Examples
///
/// ```
/// let value = sci_special::mathieu_b(2,0.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy mathieu_b](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.mathieu_b.html).
/// ## Work limits
///
/// Orders must satisfy m≤128 and finite abs(q)≤100.
/// Inputs outside these bounds return NaN without entering native work buffers.
pub fn mathieu_b(m: u32, q: f64) -> f64 {
    if m > 128 || !q.is_finite() || q.abs() > 100.0 {
        return f64::NAN;
    }
    crate::backend::legacy::mathieu_b(m, q)
}

/// Even Mathieu function (ce_m(x,q), derivative), with the SciPy normalization.
///
/// ## Parameters and domain
///
/// Nonnegative integer m and real q; x is in degrees.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Definition and numerical behavior
///
/// Solves `y″ + [a − 2q cos(2z)]y = 0`. The input `x` is in degrees,
/// with `z = x*π/180`, while the returned derivative is **with respect to z
/// in radians**. Fourier coefficients have squared integral π over `[0,2π]`
/// (at `q = 0`, `ce_0 = 1/sqrt(2)`). A bounded tridiagonal Fourier recurrence
/// supplies the characteristic value and coefficients; large arguments lose
/// phase precision. Returns `(value, dvalue/dz)`.
///
/// [NIST DLMF, Mathieu functions](https://dlmf.nist.gov/28).
///
/// ## Examples
///
/// ```
/// let value = sci_special::mathieu_cem(2,0.5,30.0);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy mathieu_cem](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.mathieu_cem.html).
/// ## Work limits
///
/// Orders must satisfy m≤128 and finite abs(q)≤100. The coordinate must be finite.
/// Inputs outside these bounds return NaN without entering native work buffers.
pub fn mathieu_cem(m: u32, q: f64, x: f64) -> (f64, f64) {
    if m > 128 || !q.is_finite() || q.abs() > 100.0 || !x.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    crate::backend::legacy::mathieu_cem(m, q, x)
}

/// Odd Mathieu function (se_m(x,q), derivative), with the SciPy normalization.
///
/// ## Parameters and domain
///
/// Positive integer m and real q; x is in degrees.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Definition and numerical behavior
///
/// Solves `y″ + [a − 2q cos(2z)]y = 0`. The input `x` is in degrees,
/// with `z = x*π/180`, while the returned derivative is **with respect to z
/// in radians**. Fourier coefficients have squared integral π over `[0,2π]`
/// (at `q = 0`, `ce_0 = 1/sqrt(2)`). A bounded tridiagonal Fourier recurrence
/// supplies the characteristic value and coefficients; large arguments lose
/// phase precision. Returns `(value, dvalue/dz)`.
///
/// [NIST DLMF, Mathieu functions](https://dlmf.nist.gov/28).
///
/// ## Examples
///
/// ```
/// let value = sci_special::mathieu_sem(2,0.5,30.0);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy mathieu_sem](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.mathieu_sem.html).
/// ## Work limits
///
/// Orders must satisfy m≤128 and finite abs(q)≤100. The coordinate must be finite.
/// Inputs outside these bounds return NaN without entering native work buffers.
pub fn mathieu_sem(m: u32, q: f64, x: f64) -> (f64, f64) {
    if m > 128 || !q.is_finite() || q.abs() > 100.0 || !x.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    crate::backend::legacy::mathieu_sem(m, q, x)
}

/// Even modified Mathieu function of the first kind and derivative.
///
/// ## Parameters and domain
///
/// Nonnegative integer m, positive q, and real x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Definition and numerical behavior
///
/// Solves `y″ − [a − 2q cosh(2x)]y = 0`. The coordinate `x` is dimensionless;
/// the returned pair is `(value, dvalue/dx)` in the real standard modified
/// Mathieu normalization used by SciPy. Bessel-product recurrences are bounded;
/// large `abs(x)` can overflow intermediate hyperbolic/exponential factors.
///
/// [NIST DLMF, Mathieu functions](https://dlmf.nist.gov/28).
///
/// ## Examples
///
/// ```
/// let value = sci_special::mathieu_modcem1(2,0.5,0.3);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy mathieu_modcem1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.mathieu_modcem1.html).
/// ## Work limits
///
/// Orders must satisfy m≤128 and finite abs(q)≤100. The coordinate must be finite.
/// Inputs outside these bounds return NaN without entering native work buffers.
pub fn mathieu_modcem1(m: u32, q: f64, x: f64) -> (f64, f64) {
    if m > 128 || !q.is_finite() || q.abs() > 100.0 || !x.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    crate::backend::legacy::mathieu_modcem1(m, q, x)
}

/// Even modified Mathieu function of the second kind and derivative.
///
/// ## Parameters and domain
///
/// Nonnegative integer m, positive q, and real x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Definition and numerical behavior
///
/// Solves `y″ − [a − 2q cosh(2x)]y = 0`. The coordinate `x` is dimensionless;
/// the returned pair is `(value, dvalue/dx)` in the real standard modified
/// Mathieu normalization used by SciPy. Bessel-product recurrences are bounded;
/// large `abs(x)` can overflow intermediate hyperbolic/exponential factors.
///
/// [NIST DLMF, Mathieu functions](https://dlmf.nist.gov/28).
///
/// ## Examples
///
/// ```
/// let value = sci_special::mathieu_modcem2(2,0.5,0.3);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy mathieu_modcem2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.mathieu_modcem2.html).
/// ## Work limits
///
/// Orders must satisfy m≤128 and finite abs(q)≤100. The coordinate must be finite.
/// Inputs outside these bounds return NaN without entering native work buffers.
pub fn mathieu_modcem2(m: u32, q: f64, x: f64) -> (f64, f64) {
    if m > 128 || !q.is_finite() || q.abs() > 100.0 || !x.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    crate::backend::legacy::mathieu_modcem2(m, q, x)
}

/// Odd modified Mathieu function of the first kind and derivative.
///
/// ## Parameters and domain
///
/// Positive integer m, positive q, and real x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Definition and numerical behavior
///
/// Solves `y″ − [a − 2q cosh(2x)]y = 0`. The coordinate `x` is dimensionless;
/// the returned pair is `(value, dvalue/dx)` in the real standard modified
/// Mathieu normalization used by SciPy. Bessel-product recurrences are bounded;
/// large `abs(x)` can overflow intermediate hyperbolic/exponential factors.
///
/// [NIST DLMF, Mathieu functions](https://dlmf.nist.gov/28).
///
/// ## Examples
///
/// ```
/// let value = sci_special::mathieu_modsem1(2,0.5,0.3);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy mathieu_modsem1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.mathieu_modsem1.html).
/// ## Work limits
///
/// Orders must satisfy m≤128 and finite abs(q)≤100. The coordinate must be finite.
/// Inputs outside these bounds return NaN without entering native work buffers.
pub fn mathieu_modsem1(m: u32, q: f64, x: f64) -> (f64, f64) {
    if m > 128 || !q.is_finite() || q.abs() > 100.0 || !x.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    crate::backend::legacy::mathieu_modsem1(m, q, x)
}

/// Odd modified Mathieu function of the second kind and derivative.
///
/// ## Parameters and domain
///
/// Positive integer m, positive q, and real x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Definition and numerical behavior
///
/// Solves `y″ − [a − 2q cosh(2x)]y = 0`. The coordinate `x` is dimensionless;
/// the returned pair is `(value, dvalue/dx)` in the real standard modified
/// Mathieu normalization used by SciPy. Bessel-product recurrences are bounded;
/// large `abs(x)` can overflow intermediate hyperbolic/exponential factors.
///
/// [NIST DLMF, Mathieu functions](https://dlmf.nist.gov/28).
///
/// ## Examples
///
/// ```
/// let value = sci_special::mathieu_modsem2(2,0.5,0.3);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy mathieu_modsem2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.mathieu_modsem2.html).
/// ## Work limits
///
/// Orders must satisfy m≤128 and finite abs(q)≤100. The coordinate must be finite.
/// Inputs outside these bounds return NaN without entering native work buffers.
pub fn mathieu_modsem2(m: u32, q: f64, x: f64) -> (f64, f64) {
    if m > 128 || !q.is_finite() || q.abs() > 100.0 || !x.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    crate::backend::legacy::mathieu_modsem2(m, q, x)
}

/// Real part of J₀(x exp(3πi/4)), the Kelvin ber function.
///
/// ## Parameters and domain
///
/// Real x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::ber(0.5_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy ber](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ber.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn ber<T: sci_core::Float>(x: T) -> T {
    crate::backend::ber(x).unwrap_or(T::nan())
}

/// Imaginary part of J₀(x exp(3πi/4)), the Kelvin bei function.
///
/// ## Parameters and domain
///
/// Real x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::bei(0.5_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy bei](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.bei.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn bei<T: sci_core::Float>(x: T) -> T {
    crate::backend::bei(x).unwrap_or(T::nan())
}

/// Real part of K₀(x exp(πi/4)), the Kelvin ker function.
///
/// ## Parameters and domain
///
/// Positive x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::ker(0.5_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy ker](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ker.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn ker<T: sci_core::Float>(x: T) -> T {
    crate::backend::ker(x).unwrap_or(T::nan())
}

/// Imaginary part of K₀(x exp(πi/4)), the Kelvin kei function.
///
/// ## Parameters and domain
///
/// Positive x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::kei(0.5_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy kei](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kei.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn kei<T: sci_core::Float>(x: T) -> T {
    crate::backend::kei(x).unwrap_or(T::nan())
}

/// Derivative of the Kelvin ber function with respect to x.
///
/// ## Parameters and domain
///
/// Real x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::berp(0.5_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy berp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.berp.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn berp<T: sci_core::Float>(x: T) -> T {
    crate::backend::berp(x).unwrap_or(T::nan())
}

/// Derivative of the Kelvin bei function with respect to x.
///
/// ## Parameters and domain
///
/// Real x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::beip(0.5_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy beip](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.beip.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn beip<T: sci_core::Float>(x: T) -> T {
    crate::backend::beip(x).unwrap_or(T::nan())
}

/// Derivative of the Kelvin ker function with respect to x.
///
/// ## Parameters and domain
///
/// Positive x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::kerp(0.5_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy kerp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kerp.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn kerp<T: sci_core::Float>(x: T) -> T {
    crate::backend::kerp(x).unwrap_or(T::nan())
}

/// Derivative of the Kelvin kei function with respect to x.
///
/// ## Parameters and domain
///
/// Positive x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::keip(0.5_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy keip](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.keip.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn keip<T: sci_core::Float>(x: T) -> T {
    crate::backend::keip(x).unwrap_or(T::nan())
}

/// First nt positive zeros of ber in ascending order.
///
/// ## Parameters and domain
///
/// nt must be positive and at most 4096.
///
/// ## Returns
///
/// An owned vector of zeros in the order described above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] when the requested count is outside the stated domain.
///
/// ## Examples
///
/// ```
/// let value = sci_special::ber_zeros(3)?;
/// assert_eq!(value.len(), 3);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy ber_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ber_zeros.html).
pub fn ber_zeros(nt: usize) -> Result<Vec<f64>> {
    if !(nt <= 4096 && nt > 0) {
        return Err(Error::InvalidInput(
            "requested special-function sequence is too large or empty",
        ));
    }
    crate::backend::legacy::ber_zeros(nt).map_err(Error::NumericalFailure)
}

/// First nt positive zeros of bei in ascending order.
///
/// ## Parameters and domain
///
/// nt must be positive and at most 4096.
///
/// ## Returns
///
/// An owned vector of zeros in the order described above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] when the requested count is outside the stated domain.
///
/// ## Examples
///
/// ```
/// let value = sci_special::bei_zeros(3)?;
/// assert_eq!(value.len(), 3);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy bei_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.bei_zeros.html).
pub fn bei_zeros(nt: usize) -> Result<Vec<f64>> {
    if !(nt <= 4096 && nt > 0) {
        return Err(Error::InvalidInput(
            "requested special-function sequence is too large or empty",
        ));
    }
    crate::backend::legacy::bei_zeros(nt).map_err(Error::NumericalFailure)
}

/// First nt positive zeros of ker in ascending order.
///
/// ## Parameters and domain
///
/// nt must be positive and at most 4096.
///
/// ## Returns
///
/// An owned vector of zeros in the order described above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] when the requested count is outside the stated domain.
///
/// ## Examples
///
/// ```
/// let value = sci_special::ker_zeros(3)?;
/// assert_eq!(value.len(), 3);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy ker_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ker_zeros.html).
pub fn ker_zeros(nt: usize) -> Result<Vec<f64>> {
    if !(nt <= 4096 && nt > 0) {
        return Err(Error::InvalidInput(
            "requested special-function sequence is too large or empty",
        ));
    }
    crate::backend::legacy::ker_zeros(nt).map_err(Error::NumericalFailure)
}

/// First nt positive zeros of kei in ascending order.
///
/// ## Parameters and domain
///
/// nt must be positive and at most 4096.
///
/// ## Returns
///
/// An owned vector of zeros in the order described above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] when the requested count is outside the stated domain.
///
/// ## Examples
///
/// ```
/// let value = sci_special::kei_zeros(3)?;
/// assert_eq!(value.len(), 3);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy kei_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kei_zeros.html).
pub fn kei_zeros(nt: usize) -> Result<Vec<f64>> {
    if !(nt <= 4096 && nt > 0) {
        return Err(Error::InvalidInput(
            "requested special-function sequence is too large or empty",
        ));
    }
    crate::backend::legacy::kei_zeros(nt).map_err(Error::NumericalFailure)
}

/// First nt positive zeros of berp in ascending order.
///
/// ## Parameters and domain
///
/// nt must be positive and at most 4096.
///
/// ## Returns
///
/// An owned vector of zeros in the order described above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] when the requested count is outside the stated domain.
///
/// ## Examples
///
/// ```
/// let value = sci_special::berp_zeros(3)?;
/// assert_eq!(value.len(), 3);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy berp_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.berp_zeros.html).
pub fn berp_zeros(nt: usize) -> Result<Vec<f64>> {
    if !(nt <= 4096 && nt > 0) {
        return Err(Error::InvalidInput(
            "requested special-function sequence is too large or empty",
        ));
    }
    crate::backend::legacy::berp_zeros(nt).map_err(Error::NumericalFailure)
}

/// First nt positive zeros of beip in ascending order.
///
/// ## Parameters and domain
///
/// nt must be positive and at most 4096.
///
/// ## Returns
///
/// An owned vector of zeros in the order described above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] when the requested count is outside the stated domain.
///
/// ## Examples
///
/// ```
/// let value = sci_special::beip_zeros(3)?;
/// assert_eq!(value.len(), 3);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy beip_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.beip_zeros.html).
pub fn beip_zeros(nt: usize) -> Result<Vec<f64>> {
    if !(nt <= 4096 && nt > 0) {
        return Err(Error::InvalidInput(
            "requested special-function sequence is too large or empty",
        ));
    }
    crate::backend::legacy::beip_zeros(nt).map_err(Error::NumericalFailure)
}

/// First nt positive zeros of kerp in ascending order.
///
/// ## Parameters and domain
///
/// nt must be positive and at most 4096.
///
/// ## Returns
///
/// An owned vector of zeros in the order described above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] when the requested count is outside the stated domain.
///
/// ## Examples
///
/// ```
/// let value = sci_special::kerp_zeros(3)?;
/// assert_eq!(value.len(), 3);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy kerp_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kerp_zeros.html).
pub fn kerp_zeros(nt: usize) -> Result<Vec<f64>> {
    if !(nt <= 4096 && nt > 0) {
        return Err(Error::InvalidInput(
            "requested special-function sequence is too large or empty",
        ));
    }
    crate::backend::legacy::kerp_zeros(nt).map_err(Error::NumericalFailure)
}

// kei'(x) = Im[-a K_1(a x)], a = exp(i pi/4). Removing the
// positive exp(-x/sqrt(2)) factor avoids underflow for long sequences.
/// First nt positive zeros of keip in ascending order.
///
/// ## Parameters and domain
///
/// nt must be positive and at most 4096.
///
/// ## Returns
///
/// An owned vector of zeros in the order described above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] when the requested count is outside the stated domain.
/// Returns [`Error::NumericalFailure`] if a refined kei derivative zero fails to converge.
///
/// ## Numerical behavior
///
/// The kei derivative zeros use `kei'(x) = Im[-a K_1(a x)]`, where
/// `a = exp(i pi/4)`, with the private scaled complex Bessel K backend and
/// at most 32 bracketing steps and 64 safeguarded Newton iterations per zero. Scaling avoids exponential underflow.
/// This avoids cancellation in the Kelvin series near x=9.4; SciPy's
/// `keip_zeros` can differ by about 4e-10 there. Accuracy remains limited
/// by binary64 argument reduction for large zeros.
///
/// ## Examples
///
/// ```
/// let value = sci_special::keip_zeros(3)?;
/// assert_eq!(value.len(), 3);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [DLMF Kelvin definition, Eq. 10.61.2](https://dlmf.nist.gov/10.61.E2).
/// - [DLMF Bessel K derivatives, §10.29](https://dlmf.nist.gov/10.29).
/// - [SciPy keip_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.keip_zeros.html).
pub fn keip_zeros(nt: usize) -> Result<Vec<f64>> {
    if !(nt <= 4096 && nt > 0) {
        return Err(Error::InvalidInput(
            "requested special-function sequence is too large or empty",
        ));
    }
    crate::backend::legacy::keip_zeros(nt).map_err(Error::NumericalFailure)
}

/// First nt positive zeros of ber,bei,ker,kei,berp,beip,kerp,keip, in that order.
///
/// ## Parameters and domain
///
/// nt must be positive and at most 4096.
///
/// ## Returns
///
/// An array of eight vectors, each containing `nt` positive zeros, in the
/// function order listed above.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] when the requested count is outside the stated domain.
/// Returns [`Error::NumericalFailure`] if a refined kei derivative zero fails to converge.
///
/// ## Numerical behavior
///
/// The kei derivative zeros use `kei'(x) = Im[-a K_1(a x)]`, where
/// `a = exp(i pi/4)`, with the private scaled complex Bessel K backend and
/// at most 32 bracketing steps and 64 safeguarded Newton iterations per zero. Scaling avoids exponential underflow.
/// This avoids cancellation in the Kelvin series near x=9.4; SciPy's
/// `keip_zeros` can differ by about 4e-10 there. All Kelvin zero families share this bounded solver and scaled Bessel identities.
///
/// ## Examples
///
/// ```
/// let value = sci_special::kelvin_zeros(3)?;
/// assert_eq!(value[0].len(), 3);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [DLMF Kelvin definition, Eq. 10.61.2](https://dlmf.nist.gov/10.61.E2).
/// - [DLMF Bessel K derivatives, §10.29](https://dlmf.nist.gov/10.29).
/// - [SciPy kelvin_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kelvin_zeros.html).
pub fn kelvin_zeros(nt: usize) -> Result<[Vec<f64>; 8]> {
    if !(nt <= 4096 && nt > 0) {
        return Err(Error::InvalidInput(
            "requested special-function sequence is too large or empty",
        ));
    }
    Ok([
        crate::backend::legacy::ber_zeros(nt).map_err(Error::NumericalFailure)?,
        crate::backend::legacy::bei_zeros(nt).map_err(Error::NumericalFailure)?,
        crate::backend::legacy::ker_zeros(nt).map_err(Error::NumericalFailure)?,
        crate::backend::legacy::kei_zeros(nt).map_err(Error::NumericalFailure)?,
        crate::backend::legacy::berp_zeros(nt).map_err(Error::NumericalFailure)?,
        crate::backend::legacy::beip_zeros(nt).map_err(Error::NumericalFailure)?,
        crate::backend::legacy::kerp_zeros(nt).map_err(Error::NumericalFailure)?,
        crate::backend::legacy::keip_zeros(nt).map_err(Error::NumericalFailure)?,
    ])
}

/// Associated Legendre function P_v^m(x), including the Condon–Shortley phase (−1)^m.
///
/// ## Parameters and domain
///
/// Integer order m, real degree v, and x in `[−1,1]`.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::lpmv(1,2.5,0.3);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy lpmv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.lpmv.html).
///
/// ## Numerical behavior and work limits
///
/// Finite real inputs with `abs(v) <= 100000`, `abs(m) <= 4096`, and `x` in `[-1,1]`.
/// Larger orders or degrees return NaN before recurrence work.
pub fn lpmv(m: i32, v: f64, x: f64) -> f64 {
    crate::backend::legacy::pmv(m, v, x)
}

/// Legendre polynomial P_n(x), using value-only semantics (no derivative axis).
///
/// ## Parameters and domain
///
/// Nonnegative integer degree n and real x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::legendre_p(3,0.3_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy legendre_p](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.legendre_p.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn legendre_p<T: sci_core::Float>(n: i32, x: T) -> T {
    crate::backend::eval_legendre(n, x).unwrap_or(T::nan())
}

/// Unnormalized associated Legendre P_n^m(x) with branch-cut convention 2 and Condon–Shortley phase.
///
/// ## Parameters and domain
///
/// Nonnegative degree n, integer m with abs(m)≤n, and real x in `[−1,1]`.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::assoc_legendre_p(3,1,0.3);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy assoc_legendre_p](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.assoc_legendre_p.html).
pub fn assoc_legendre_p(n: i32, m: i32, x: f64) -> f64 {
    crate::backend::legacy::assoc_legendre_p(n, m, x)
}

/// Spherical-normalized associated Legendre value, the theta factor in a spherical harmonic.
///
/// ## Parameters and domain
///
/// Nonnegative degree n, abs(m)≤n, theta in radians.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
///
/// ## Examples
///
/// ```
/// let value = sci_special::sph_legendre_p(3,1,0.3);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy sph_legendre_p](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sph_legendre_p.html).
pub fn sph_legendre_p(n: i32, m: i32, theta: f64) -> f64 {
    crate::backend::legacy::sph_legendre_p(n, m, theta)
}

/// Legendre functions Q₀ through Q_n and their derivatives, as two vectors.
///
/// ## Parameters and domain
///
/// n≤4096; real x away from ±1.
///
/// ## Returns
///
/// The pair `(values, derivatives)`, each with `n + 1` entries for degrees
/// `0..=n`. Derivatives are with respect to `x`.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] when the requested degrees, orders, or output size exceed
/// the documented limits. Values of `x` are passed through to the backend,
/// including singular and nonfinite values.
///
/// ## Examples
///
/// ```
/// let value = sci_special::lqn(3,0.3)?;
/// let _ = value;
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy lqn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.lqn.html).
pub fn lqn(n: usize, x: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    if !(n <= 4096) {
        return Err(Error::InvalidInput(
            "requested special-function sequence is too large or empty",
        ));
    }
    let (mut values, mut derivatives) = crate::backend::legacy::legendre_q_all(n.max(1), x);
    values.truncate(n + 1);
    derivatives.truncate(n + 1);
    Ok((values, derivatives))
}

/// Associated Legendre Q_m^n and derivatives, as order-major matrices.
///
/// ## Parameters and domain
///
/// m,n≤4096 with (m+1)(n+1)≤1048576; real x away from ±1.
///
/// ## Returns
///
/// The pair `(values, derivatives)`, each an `(m + 1) × (n + 1)` matrix with
/// order rows and degree columns. Derivatives are with respect to `x`.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] when the requested degrees, orders, or output size exceed
/// the documented limits. Values of `x` are passed through to the backend,
/// including singular and nonfinite values.
///
/// Returns [`Error::DimensionMismatch`] if backend output has an unexpected shape.
///
/// ## Examples
///
/// ```
/// let value = sci_special::lqmn(2,3,0.3)?;
/// assert_eq!(value.0.len(), 3);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy lqmn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.lqmn.html).
pub fn lqmn(m: usize, n: usize, x: f64) -> Result<ValueDerivativeMatrices> {
    if !(m <= 4096 && n <= 4096 && (m + 1) * (n + 1) <= 1_048_576) {
        return Err(Error::InvalidInput(
            "requested special-function sequence is too large or empty",
        ));
    }
    let (values, derivatives) = crate::backend::legacy::assoc_legendre_q_all(n.max(1), m.max(1), x);
    let crop = |matrix: Vec<Vec<f64>>| -> Result<Vec<Vec<f64>>> {
        if matrix.len() < n + 1 || matrix.iter().take(n + 1).any(|row| row.len() < m + 1) {
            return Err(Error::DimensionMismatch);
        }
        Ok((0..=m)
            .map(|order| matrix.iter().take(n + 1).map(|row| row[order]).collect())
            .collect())
    };
    Ok((crop(values)?, crop(derivatives)?))
}

/// Logistic sigmoid 1/(1+exp(−x)), evaluated without overflow.
///
/// ## Parameters and domain
///
/// All real values; infinite endpoints give zero and one.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// ## Examples
///
/// ```
/// assert!(sci_special::expit(0.5).is_finite());
/// ```
///
/// ## References
///
/// - [SciPy expit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expit.html).
pub fn expit(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Log odds ln(p/(1−p)).
///
/// ## Parameters and domain
///
/// p in `[0,1]`; endpoints give signed infinity and outside gives NaN.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// ## Examples
///
/// ```
/// assert!(sci_special::logit(0.3).is_finite());
/// ```
///
/// ## References
///
/// - [SciPy logit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logit.html).
pub fn logit(p: f64) -> f64 {
    p.ln() - (-p).ln_1p()
}

/// Stable ln(1+exp(x)).
///
/// ## Parameters and domain
///
/// All real values.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// ## Examples
///
/// ```
/// assert!(sci_special::softplus(0.5).is_finite());
/// ```
///
/// ## References
///
/// - [SciPy softplus](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softplus.html).
pub fn softplus(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// Stable log sigmoid −ln(1+exp(−x)).
///
/// ## Parameters and domain
///
/// All real values.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// ## Examples
///
/// ```
/// assert!(sci_special::log_expit(-100.0).is_finite());
/// ```
///
/// ## References
///
/// - [SciPy log_expit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.log_expit.html).
pub fn log_expit(x: f64) -> f64 {
    -softplus(-x)
}

/// Relative exponential (exp(x)−1)/x with removable value one at zero.
///
/// ## Parameters and domain
///
/// All real values.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// ## Examples
///
/// ```
/// assert!(sci_special::exprel(0.000001).is_finite());
/// ```
///
/// ## References
///
/// - [SciPy exprel](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.exprel.html).
pub fn exprel(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else if x == f64::INFINITY {
        f64::INFINITY
    } else {
        x.exp_m1() / x
    }
}

/// Normalized sinc sin(πx)/(πx), with value one at zero.
///
/// ## Parameters and domain
///
/// Finite real x; infinite inputs return NaN.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// ## Examples
///
/// ```
/// assert!(sci_special::sinc(0.5).is_finite());
/// ```
///
/// ## References
///
/// - [SciPy sinc](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sinc.html).
pub fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let y = std::f64::consts::PI * x;
        y.sin() / y
    }
}

/// cos(x)−1, stable near zero.
///
/// ## Parameters and domain
///
/// Finite real x in radians.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// ## Examples
///
/// ```
/// assert!(sci_special::cosm1(0.000001).is_finite());
/// ```
///
/// ## References
///
/// - [SciPy cosm1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.cosm1.html).
pub fn cosm1(x: f64) -> f64 {
    -2.0 * (0.5 * x).sin().powi(2)
}

/// Sine of an angle in degrees.
///
/// ## Parameters and domain
///
/// Finite real x.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// ## Examples
///
/// ```
/// assert!(sci_special::sindg(30.0).is_finite());
/// ```
///
/// ## References
///
/// - [SciPy sindg](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sindg.html).
pub fn sindg(x: f64) -> f64 {
    let r = x.rem_euclid(360.0);
    if r <= 90.0 {
        r.to_radians().sin()
    } else if r <= 180.0 {
        (180.0 - r).to_radians().sin()
    } else if r <= 270.0 {
        -(r - 180.0).to_radians().sin()
    } else {
        -(360.0 - r).to_radians().sin()
    }
}

/// Cosine of an angle in degrees.
///
/// ## Parameters and domain
///
/// Finite real x.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// ## Examples
///
/// ```
/// assert!(sci_special::cosdg(30.0).is_finite());
/// ```
///
/// ## References
///
/// - [SciPy cosdg](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.cosdg.html).
pub fn cosdg(x: f64) -> f64 {
    sindg(x.rem_euclid(360.0) + 90.0)
}

/// Tangent of an angle in degrees.
///
/// ## Parameters and domain
///
/// Finite real x away from odd multiples of 90°.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// ## Examples
///
/// ```
/// assert!(sci_special::tandg(30.0).is_finite());
/// ```
///
/// ## References
///
/// - [SciPy tandg](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.tandg.html).
pub fn tandg(x: f64) -> f64 {
    sindg(x) / cosdg(x)
}

/// Cotangent of an angle in degrees.
///
/// ## Parameters and domain
///
/// Finite real x away from multiples of 180°.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// ## Examples
///
/// ```
/// assert!(sci_special::cotdg(30.0).is_finite());
/// ```
///
/// ## References
///
/// - [SciPy cotdg](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.cotdg.html).
pub fn cotdg(x: f64) -> f64 {
    1.0 / tandg(x)
}

/// Converts signed degrees, minutes, and seconds to radians.
///
/// ## Parameters and domain
///
/// Finite real components; signs are added algebraically.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// ## Examples
///
/// ```
/// assert!(sci_special::radian(30.0,20.0,10.0).is_finite());
/// ```
///
/// ## References
///
/// - [SciPy radian](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.radian.html).
pub fn radian(d: f64, m: f64, s: f64) -> f64 {
    (d + (m + s / 60.0) / 60.0).to_radians()
}

/// Entropy contribution −x ln x.
///
/// ## Parameters and domain
///
/// Positive x; zero gives zero and negative x gives −∞.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// ## Examples
///
/// ```
/// assert!(sci_special::entr(0.3).is_finite());
/// ```
///
/// ## References
///
/// - [SciPy entr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.entr.html).
pub fn entr(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x > 0.0 {
        -x * x.ln()
    } else if x == 0.0 {
        0.0
    } else {
        f64::NEG_INFINITY
    }
}

/// Relative entropy contribution `x ln(x/y)`.
///
/// ## Parameters and conventions
/// `x` and `y` are nonnegative masses; logarithms are natural.
/// ## Domain and special values
/// Positive x,y; x=0 with y≥0 gives zero; other domains give +∞.
/// NaN propagates. Infinite arguments follow IEEE arithmetic.
/// ## Returns
/// Scalar contribution in the units of x; it may be negative.
/// ## Numerical behavior
/// Constant work. Nearby arguments use `log1p((x-y)/y)` to retain their
/// difference. Extreme ratios use separate logarithms to avoid ratio range
/// errors. True overflow/underflow and binary64 rounding remain.
/// ## Examples
/// ```
/// assert!(sci_special::rel_entr(0.3,0.5).is_finite());
/// ```
/// ## References
/// [SciPy rel_entr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.rel_entr.html).
pub fn rel_entr(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        f64::NAN
    } else if x > 0.0 && y > 0.0 {
        x * log_ratio(x, y)
    } else if x == 0.0 && y >= 0.0 {
        0.0
    } else {
        f64::INFINITY
    }
}

// Positive arguments; preserve nearby differences before taking logarithms.
fn log_ratio(x: f64, y: f64) -> f64 {
    let ratio = x / y;
    if (0.5..=1.5).contains(&ratio) {
        ((x - y) / y).ln_1p()
    } else if ratio.is_normal() {
        ratio.ln()
    } else {
        x.ln() - y.ln()
    }
}

/// Generalized Kullback–Leibler contribution `x ln(x/y)-x+y`.
///
/// ## Parameters and conventions
/// `x` and `y` are nonnegative masses; logarithms are natural.
/// ## Domain and special values
/// Positive x,y; x=0 with y≥0 returns y; other domains give +∞.
/// NaN propagates. Infinite arguments follow IEEE arithmetic.
/// ## Returns
/// Nonnegative scalar contribution for finite valid arguments, in mass units.
/// ## Numerical behavior
/// Near equal arguments, a series for `(1+d)ln(1+d)-d`, d=(x-y)/y,
/// avoids cancellation, using at most 65 terms for |d|≤0.5. Other ratios use
/// scaled logarithms and fused multiply-add. True overflow/underflow and
/// binary64 rounding remain. Work and storage are constant.
/// ## Examples
/// ```
/// assert!(sci_special::kl_div(0.3,0.5) > 0.0);
/// ```
/// ## References
/// [SciPy kl_div](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kl_div.html).
pub fn kl_div(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        f64::NAN
    } else if x == 0.0 && y >= 0.0 {
        y
    } else if x > 0.0 && y > 0.0 {
        if !x.is_finite() || !y.is_finite() {
            return rel_entr(x, y) - x + y;
        }
        let delta = x - y;
        let d = delta / y;
        if d.abs() <= 0.5 {
            // ((1+d) ln(1+d) - d) / d² = Σ (-d)^k/((k+1)(k+2)).
            // The bounded series avoids subtracting nearly equal terms and
            // also retains a positive result when x and y differ by one ulp.
            let mut coefficient = 0.5;
            let mut power = 1.0;
            for k in 1..=64 {
                power *= -d;
                let next = coefficient + power / ((k + 1) * (k + 2)) as f64;
                if next == coefficient {
                    break;
                }
                coefficient = next;
            }
            (coefficient * d) * delta
        } else {
            x.mul_add(log_ratio(x, y) - 1.0, y)
        }
    } else {
        f64::INFINITY
    }
}

/// Huber loss r²/2 for abs(r)≤delta, and delta*(abs(r)−delta/2) otherwise.
///
/// ## Parameters and domain
///
/// delta≥0; negative delta gives +∞ and NaN propagates.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// ## Examples
///
/// ```
/// assert!(sci_special::huber(1.0,2.0).is_finite());
/// ```
///
/// ## References
///
/// - [SciPy huber](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.huber.html).
pub fn huber(delta: f64, r: f64) -> f64 {
    if delta < 0.0 {
        f64::INFINITY
    } else if r.abs() <= delta {
        0.5 * r * r
    } else {
        delta * (r.abs() - 0.5 * delta)
    }
}

/// Smooth Huber loss `delta²(√(1+(r/delta)²)-1)`.
///
/// ## Parameters and conventions
/// `r` is the residual; delta≥0 is its scale, in the same units.
/// ## Domain and special values
/// Zero delta gives zero; negative delta and NaN inputs give NaN.
/// Infinite delta gives r²/2; infinite residual with positive delta gives +∞.
/// These retain Sci's negative-delta and infinite-delta conventions; SciPy
/// 1.18.1 returns +∞ and NaN, respectively, for those delta values.
/// ## Returns
/// Nonnegative scalar loss in squared residual units.
/// ## Numerical behavior
/// Constant work. Rationalized formulas normalize by the larger of delta and
/// |r| before evaluating a hypot, avoiding cancellation and an overflowing
/// denominator. True overflow/underflow and binary64 rounding remain.
/// ## Examples
/// ```
/// assert_eq!(sci_special::pseudo_huber(1e308, 1.0), 0.5);
/// ```
/// ## References
/// [SciPy pseudo_huber](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pseudo_huber.html).
pub fn pseudo_huber(delta: f64, r: f64) -> f64 {
    if delta.is_nan() || r.is_nan() || delta < 0.0 {
        f64::NAN
    } else if delta == 0.0 {
        0.0
    } else if delta.is_infinite() {
        0.5 * r * r
    } else if r.is_infinite() {
        f64::INFINITY
    } else {
        let r = r.abs();
        if r <= delta {
            (r / (1.0_f64.hypot(r / delta) + 1.0)) * r
        } else {
            let ratio = delta / r;
            (delta / (1.0_f64.hypot(ratio) + ratio)) * r
        }
    }
}

/// x ln y with zero when x=0, including y=0.
///
/// ## Parameters and domain
///
/// Real x and y≥0; negative y gives NaN unless x=0.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// ## Examples
///
/// ```
/// assert!(sci_special::xlogy(0.0,0.0).is_finite());
/// ```
///
/// ## References
///
/// - [SciPy xlogy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.xlogy.html).
pub fn xlogy(x: f64, y: f64) -> f64 {
    if x == 0.0 && !y.is_nan() {
        0.0
    } else {
        x * y.ln()
    }
}

/// x ln(1+y) with zero when x=0, evaluated stably near y=0.
///
/// ## Parameters and domain
///
/// Real x and y≥−1; NaN propagates.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// ## Examples
///
/// ```
/// assert!(sci_special::xlog1py(0.3,0.000001).is_finite());
/// ```
///
/// ## References
///
/// - [SciPy xlog1py](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.xlog1py.html).
pub fn xlog1py(x: f64, y: f64) -> f64 {
    if x == 0.0 && !y.is_nan() {
        0.0
    } else {
        x * y.ln_1p()
    }
}

/// x^y−1, evaluated stably when the result is small.
///
/// ## Parameters and domain
///
/// Real-valued power domain: positive x, or negative x with integer y.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// ## Examples
///
/// ```
/// assert!(sci_special::powm1(1.000001,0.3).is_finite());
/// ```
///
/// ## References
///
/// - [SciPy powm1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.powm1.html).
pub fn powm1(x: f64, y: f64) -> f64 {
    if x > 0.0 {
        (y * x.ln()).exp_m1()
    } else {
        x.powf(y) - 1.0
    }
}

/// Computes `B(x, λ) = (x^λ − 1)/λ`, with `B(x, 0) = ln(x)`.
///
/// ## Parameters and conventions
///
/// `x` is dimensionless; `lambda` is the real power parameter λ.
///
/// ## Domain and special values
///
/// `x >= 0`. Zero returns `−1/λ` for positive λ and `−∞` otherwise.
/// Negative x and NaN inputs return NaN. Infinite arguments follow IEEE
/// arithmetic; indeterminate combinations can return NaN.
///
/// ## Returns
///
/// The scalar Box–Cox transform. This infallible API reports domain errors as NaN.
///
/// ## Numerical behavior
///
/// Uses the private Rust kernel's logarithmic limit for tiny λ and scaled
/// exponential for large powers, preserving representable results when x^λ
/// overflows. `boxcox1p` also preserves tiny x using `log1p`. Constant work;
/// true overflow, underflow, and binary64 rounding remain. Correct rounding
/// is not guaranteed. Endpoint values are handled before the limiting formula.
///
/// ## Examples
///
/// ```
/// assert_eq!(sci_special::boxcox(4.0_f64, 0.0), 4.0_f64.ln());
/// ```
///
/// ## References
///
/// [SciPy boxcox](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.boxcox.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn boxcox<T: sci_core::Float>(x: T, lambda: T) -> T {
    // Preserve the exact endpoint even when the backend takes its tiny-lambda
    // logarithmic limit, which is valid only for strictly positive x.
    if x == 0.0 && lambda > 0.0 {
        -1.0 / lambda
    } else {
        crate::backend::boxcox(x, lambda).unwrap_or(T::nan())
    }
}

/// Computes `((1+x)^λ − 1)/λ`, with `ln(1+x)` at `λ = 0`.
///
/// ## Parameters and conventions
///
/// `x` is dimensionless; `lambda` is the real power parameter λ.
///
/// ## Domain and special values
///
/// `x >= −1`. At x = −1, returns `−1/λ` for positive λ and `−∞`
/// otherwise. Smaller x and NaN inputs return NaN. Infinite arguments follow
/// IEEE arithmetic; indeterminate combinations can return NaN.
///
/// ## Returns
///
/// The scalar Box–Cox transform. This infallible API reports domain errors as NaN.
///
/// ## Numerical behavior
///
/// Uses the private Rust kernel's logarithmic limit for tiny λ and scaled
/// exponential for large powers, preserving representable results when x^λ
/// overflows. `boxcox1p` also preserves tiny x using `log1p`. Constant work;
/// true overflow, underflow, and binary64 rounding remain. Correct rounding
/// is not guaranteed. Endpoint values are handled before the limiting formula.
///
/// ## Examples
///
/// ```
/// assert_eq!(sci_special::boxcox1p(3.0_f64, 0.0), 4.0_f64.ln());
/// ```
///
/// ## References
///
/// [SciPy boxcox1p](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.boxcox1p.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn boxcox1p<T: sci_core::Float>(x: T, lambda: T) -> T {
    if x == -1.0 && lambda > 0.0 {
        -1.0 / lambda
    } else {
        crate::backend::boxcox1p(x, lambda).unwrap_or(T::nan())
    }
}

/// Inverse Box–Cox transformation (1+lambda*y)^(1/lambda).
///
/// ## Parameters and domain
///
/// 1+lambda*y≥0; lambda=0 uses exp(y).
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// ## Examples
///
/// ```
/// assert!(sci_special::inv_boxcox(0.5,0.3).is_finite());
/// ```
///
/// ## References
///
/// - [SciPy inv_boxcox](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.inv_boxcox.html).
pub fn inv_boxcox(y: f64, lambda: f64) -> f64 {
    if lambda == 0.0 {
        y.exp()
    } else {
        ((lambda * y).ln_1p() / lambda).exp()
    }
}

/// Inverse Box–Cox transformation minus one, stable near zero.
///
/// ## Parameters and domain
///
/// 1+lambda*y≥0; lambda=0 uses expm1(y).
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// ## Examples
///
/// ```
/// assert!(sci_special::inv_boxcox1p(0.000001,0.3).is_finite());
/// ```
///
/// ## References
///
/// - [SciPy inv_boxcox1p](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.inv_boxcox1p.html).
pub fn inv_boxcox1p(y: f64, lambda: f64) -> f64 {
    if lambda == 0.0 {
        y.exp_m1()
    } else {
        ((lambda * y).ln_1p() / lambda).exp_m1()
    }
}

/// Complementary regularized incomplete beta 1−I_x(a,b)=I_(1−x)(b,a).
///
/// ## Parameters and domain
///
/// Finite positive a,b and x in `[0,1]`; the upper tail is evaluated directly.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// ## Examples
///
/// ```
/// assert!(sci_special::betaincc(2.5,3.5,0.7).is_finite());
/// ```
///
/// ## References
///
/// - [SciPy betaincc](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.betaincc.html).
pub fn betaincc(a: f64, b: f64, x: f64) -> f64 {
    super::betainc(b, a, 1.0 - x)
}

/// Inverse complementary regularized incomplete beta with respect to x.
///
/// ## Parameters and domain
///
/// Finite positive a,b and q in `[0,1]`.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// ## Examples
///
/// ```
/// assert!(sci_special::betainccinv(2.5,3.5,0.3).is_finite());
/// ```
///
/// ## References
///
/// - [SciPy betainccinv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.betainccinv.html).
pub fn betainccinv(a: f64, b: f64, q: f64) -> f64 {
    1.0 - super::betaincinv(b, a, q)
}

/// Normal mean solving Φ((x−mean)/sd)=p.
///
/// ## Parameters and domain
///
/// sd>0 and p in (0,1).
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// ## Examples
///
/// ```
/// assert!(sci_special::nrdtrimn(0.3,0.5,1.5).is_finite());
/// ```
///
/// ## References
///
/// - [SciPy nrdtrimn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.nrdtrimn.html).
pub fn nrdtrimn(p: f64, sd: f64, x: f64) -> f64 {
    if sd > 0.0 {
        x - sd * ndtri(p)
    } else {
        f64::NAN
    }
}

/// Normal standard deviation solving Φ((x−mean)/sd)=p.
///
/// ## Parameters and domain
///
/// p in (0,1) excluding 1/2, and x−mean must have the same sign as Φ⁻¹(p).
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// ## Examples
///
/// ```
/// assert!(sci_special::nrdtrisd(1.5,0.3,0.5).is_finite());
/// ```
///
/// ## References
///
/// - [SciPy nrdtrisd](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.nrdtrisd.html).
pub fn nrdtrisd(mean: f64, p: f64, x: f64) -> f64 {
    let sd = (x - mean) / ndtri(p);
    if sd > 0.0 { sd } else { f64::NAN }
}

/// Inverse gamma CDF P(b,a*x)=p with respect to rate a.
///
/// ## Parameters and domain
///
/// b,x>0 and p in `[0,1]`.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// ## Examples
///
/// ```
/// assert!(sci_special::gdtria(0.3,2.5,0.5).is_finite());
/// ```
///
/// ## References
///
/// - [SciPy gdtria](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gdtria.html).
pub fn gdtria(p: f64, b: f64, x: f64) -> f64 {
    if x > 0.0 {
        gammaincinv(b, p) / x
    } else {
        f64::NAN
    }
}

/// Inverse gamma CDF P(b,a*x)=p with respect to x.
///
/// ## Parameters and domain
///
/// a,b>0 and p in `[0,1]`.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// ## Examples
///
/// ```
/// assert!(sci_special::gdtrix(1.5,2.5,0.3).is_finite());
/// ```
///
/// ## References
///
/// - [SciPy gdtrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gdtrix.html).
pub fn gdtrix(a: f64, b: f64, p: f64) -> f64 {
    if a > 0.0 {
        gammaincinv(b, p) / a
    } else {
        f64::NAN
    }
}

/// Inverse chi-square CDF with respect to degrees of freedom.
///
/// ## Parameters and domain
///
/// x>0 and p in (0,1).
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// ## Examples
///
/// ```
/// assert!(sci_special::chdtriv(0.3,2.5).is_finite());
/// ```
///
/// ## References
///
/// - [SciPy chdtriv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.chdtriv.html).
pub fn chdtriv(p: f64, x: f64) -> f64 {
    gdtrib(0.5, p, x) * 2.0
}

/// Integer-order Bessel function J_n(x), an alias of jv.
///
/// ## Parameters and domain
///
/// Integer order n and real x.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// ## Examples
///
/// ```
/// assert!(sci_special::jn(2,0.5).is_finite());
/// ```
///
/// ## References
///
/// - [SciPy jn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jn.html).
pub fn jn(n: i32, x: f64) -> f64 {
    jv(f64::from(n), x)
}

/// Integer-order Bessel function Y_n(x).
///
/// ## Parameters and domain
///
/// Integer order n and positive x.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// ## Examples
///
/// ```
/// assert!(sci_special::yn(2,0.5).is_finite());
/// ```
///
/// ## References
///
/// - [SciPy yn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.yn.html).
pub fn yn(n: i32, x: f64) -> f64 {
    yv(f64::from(n), x)
}

/// Integer-order modified Bessel function K_n(x).
///
/// ## Parameters and domain
///
/// Integer order n and positive x.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// ## Examples
///
/// ```
/// assert!(sci_special::kn(2,0.5).is_finite());
/// ```
///
/// ## References
///
/// - [SciPy kn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kn.html).
pub fn kn(n: i32, x: f64) -> f64 {
    kv(f64::from(n), x)
}
