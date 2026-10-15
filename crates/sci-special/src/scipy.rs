//! Private adapters for the broad SciPy special-function families.

use crate::{Error, Result};

/// Row-major value and derivative matrices, sharing the linear-algebra representation.
pub type ValueDerivativeMatrices = (Vec<Vec<f64>>, Vec<Vec<f64>>);

/// Arithmetic–geometric mean, the common limit of a←(a+b)/2 and b←√(ab).
///
/// ## Parameters and domain
///
/// Nonnegative a and b; opposite signs return NaN.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::agm(1.0,2.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy agm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.agm.html).
pub fn agm(a: f64, b: f64) -> f64 {
    xsf::agm(a, b)
}

/// Airy equation y″−xy=0 solutions, ordered `[Ai,Ai′,Bi,Bi′]`.
///
/// ## Parameters and domain
///
/// All real x; large positive x may overflow Bi.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::airy(0.5);
/// assert!(value.iter().all(|v| v.is_finite()));
/// ```
///
/// # References
///
/// - [SciPy airy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.airy.html).
pub fn airy(x: f64) -> [f64; 4] {
    xsf::airy(x)
}

/// Scaled Airy values `[Ai exp(ζ),Ai′ exp(ζ),Bi exp(−abs(ζ)),Bi′ exp(−abs(ζ))]`, ζ=2x^(3/2)/3.
///
/// ## Parameters and domain
///
/// Real x; scaled Ai and Ai′ return NaN for negative x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::airye(0.5);
/// assert!(value.iter().all(|v| v.is_finite()));
/// ```
///
/// # References
///
/// - [SciPy airye](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.airye.html).
pub fn airye(x: f64) -> [f64; 4] {
    xsf::airy_scaled(x)
}

/// Integrals from 0 to x of Ai(t), Bi(t), Ai(−t), and Bi(−t), in that order.
///
/// ## Parameters and domain
///
/// Nonnegative x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::itairy(0.5);
/// assert!(value.iter().all(|v| v.is_finite()));
/// ```
///
/// # References
///
/// - [SciPy itairy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.itairy.html).
pub fn itairy(x: f64) -> [f64; 4] {
    xsf::airy_integrals(x)
}

/// Jacobi elliptic functions (sn,cn,dn,amplitude), with u=F(amplitude,m).
///
/// ## Parameters and domain
///
/// Finite u and m in `[0,1]`.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::ellipj(0.5,0.4);
/// let _ = value;
/// ```
///
/// # References
///
/// - [SciPy ellipj](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipj.html).
pub fn ellipj(u: f64, m: f64) -> (f64, f64, f64, f64) {
    xsf::ellipj(u, m)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::binom(5.0,2.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy binom](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.binom.html).
pub fn binom(n: f64, k: f64) -> f64 {
    xsf::binom(n, k)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::erfcx(0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy erfcx](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erfcx.html).
pub fn erfcx(x: f64) -> f64 {
    xsf::erfcx(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::erfi(0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy erfi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erfi.html).
pub fn erfi(x: f64) -> f64 {
    xsf::erfi(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::erfcinv(0.3);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy erfcinv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erfcinv.html).
pub fn erfcinv(p: f64) -> f64 {
    xsf::erfcinv(p)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::dawsn(0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy dawsn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.dawsn.html).
pub fn dawsn(x: f64) -> f64 {
    xsf::dawsn(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::gammasgn(-0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy gammasgn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gammasgn.html).
pub fn gammasgn(x: f64) -> f64 {
    xsf::gammasgn(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::gammaincc(2.5,3.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy gammaincc](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gammaincc.html).
pub fn gammaincc(a: f64, x: f64) -> f64 {
    xsf::gammaincc(a, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::gammaincinv(2.5,0.3);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy gammaincinv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gammaincinv.html).
pub fn gammaincinv(a: f64, p: f64) -> f64 {
    xsf::gammaincinv(a, p)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::gammainccinv(2.5,0.3);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy gammainccinv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gammainccinv.html).
pub fn gammainccinv(a: f64, q: f64) -> f64 {
    xsf::gammainccinv(a, q)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::loggamma(2.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy loggamma](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.loggamma.html).
pub fn loggamma(x: f64) -> f64 {
    xsf::loggamma(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::rgamma(-2.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy rgamma](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.rgamma.html).
pub fn rgamma(x: f64) -> f64 {
    xsf::rgamma(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::polygamma(2,2.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy polygamma](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.polygamma.html).
pub fn polygamma(n: u32, x: f64) -> f64 {
    if n > 1024 {
        return f64::NAN;
    }
    xsf::polygamma(n, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::multigammaln(3.0,2);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy multigammaln](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.multigammaln.html).
pub fn multigammaln(a: f64, d: u32) -> f64 {
    if d == 0 || d > 4096 || a <= (d as f64 - 1.0) / 2.0 {
        return f64::NAN;
    }
    xsf::multigammaln(a, d)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::jv(0.5,2.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy jv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jv.html).
pub fn jv(v: f64, x: f64) -> f64 {
    xsf::bessel_j(v, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::jve(0.5,2.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy jve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jve.html).
pub fn jve(v: f64, x: f64) -> f64 {
    xsf::bessel_je(v, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::yv(0.5,2.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy yv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.yv.html).
pub fn yv(v: f64, x: f64) -> f64 {
    xsf::bessel_y(v, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::yve(0.5,2.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy yve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.yve.html).
pub fn yve(v: f64, x: f64) -> f64 {
    xsf::bessel_ye(v, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::iv(0.5,2.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy iv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.iv.html).
pub fn iv(v: f64, x: f64) -> f64 {
    xsf::bessel_i(v, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::ive(0.5,2.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy ive](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ive.html).
pub fn ive(v: f64, x: f64) -> f64 {
    xsf::bessel_ie(v, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::kv(0.5,2.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy kv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kv.html).
pub fn kv(v: f64, x: f64) -> f64 {
    xsf::bessel_k(v, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::kve(0.5,2.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy kve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kve.html).
pub fn kve(v: f64, x: f64) -> f64 {
    xsf::bessel_ke(v, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::j0(2.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy j0](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.j0.html).
pub fn j0(x: f64) -> f64 {
    xsf::bessel_j0(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::j1(2.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy j1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.j1.html).
pub fn j1(x: f64) -> f64 {
    xsf::bessel_j1(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::y0(2.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy y0](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.y0.html).
pub fn y0(x: f64) -> f64 {
    xsf::bessel_y0(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::y1(2.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy y1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.y1.html).
pub fn y1(x: f64) -> f64 {
    xsf::bessel_y1(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::i0(2.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy i0](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.i0.html).
pub fn i0(x: f64) -> f64 {
    xsf::bessel_i0(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::i1(2.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy i1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.i1.html).
pub fn i1(x: f64) -> f64 {
    xsf::bessel_i1(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::i0e(2.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy i0e](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.i0e.html).
pub fn i0e(x: f64) -> f64 {
    xsf::bessel_i0e(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::i1e(2.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy i1e](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.i1e.html).
pub fn i1e(x: f64) -> f64 {
    xsf::bessel_i1e(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::k0(2.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy k0](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.k0.html).
pub fn k0(x: f64) -> f64 {
    xsf::bessel_k0(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::k1(2.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy k1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.k1.html).
pub fn k1(x: f64) -> f64 {
    xsf::bessel_k1(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::k0e(2.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy k0e](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.k0e.html).
pub fn k0e(x: f64) -> f64 {
    xsf::bessel_k0e(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::k1e(2.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy k1e](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.k1e.html).
pub fn k1e(x: f64) -> f64 {
    xsf::bessel_k1e(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::besselpoly(1.0,0.5,2.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy besselpoly](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.besselpoly.html).
pub fn besselpoly(a: f64, lambda: f64, v: f64) -> f64 {
    xsf::besselpoly(a, lambda, v)
}

/// Integrals (∫₀ˣ J₀(t)dt, ∫₀ˣ Y₀(t)dt).
///
/// ## Parameters and domain
///
/// Nonnegative x for both real components.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::itj0y0(2.0);
/// let _ = value;
/// ```
///
/// # References
///
/// - [SciPy itj0y0](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.itj0y0.html).
pub fn itj0y0(x: f64) -> (f64, f64) {
    xsf::it1j0y0(x)
}

/// Integrals (∫₀ˣ (1−J₀(t))/t dt, ∫ₓ^∞ Y₀(t)/t dt).
///
/// ## Parameters and domain
///
/// Positive x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::it2j0y0(2.0);
/// let _ = value;
/// ```
///
/// # References
///
/// - [SciPy it2j0y0](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.it2j0y0.html).
pub fn it2j0y0(x: f64) -> (f64, f64) {
    xsf::it2j0y0(x)
}

/// Integrals (∫₀ˣ I₀(t)dt, ∫₀ˣ K₀(t)dt).
///
/// ## Parameters and domain
///
/// Nonnegative x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::iti0k0(2.0);
/// let _ = value;
/// ```
///
/// # References
///
/// - [SciPy iti0k0](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.iti0k0.html).
pub fn iti0k0(x: f64) -> (f64, f64) {
    xsf::it1i0k0(x)
}

/// Integrals (∫₀ˣ (I₀(t)−1)/t dt, ∫ₓ^∞ K₀(t)/t dt).
///
/// ## Parameters and domain
///
/// Positive x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::it2i0k0(2.0);
/// let _ = value;
/// ```
///
/// # References
///
/// - [SciPy it2i0k0](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.it2i0k0.html).
pub fn it2i0k0(x: f64) -> (f64, f64) {
    xsf::it2i0k0(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::struve(0.5,2.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy struve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.struve.html).
pub fn struve(v: f64, x: f64) -> f64 {
    xsf::struve_h(v, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::modstruve(0.5,2.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy modstruve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.modstruve.html).
pub fn modstruve(v: f64, x: f64) -> f64 {
    xsf::struve_l(v, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::itstruve0(2.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy itstruve0](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.itstruve0.html).
pub fn itstruve0(x: f64) -> f64 {
    xsf::itstruve0(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::it2struve0(2.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy it2struve0](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.it2struve0.html).
pub fn it2struve0(x: f64) -> f64 {
    xsf::it2struve0(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::itmodstruve0(2.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy itmodstruve0](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.itmodstruve0.html).
pub fn itmodstruve0(x: f64) -> f64 {
    xsf::itmodstruve0(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::hyp0f1(2.5,0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy hyp0f1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hyp0f1.html).
pub fn hyp0f1(b: f64, x: f64) -> f64 {
    xsf::hyp0f1(b, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::hyp1f1(1.5,2.5,0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy hyp1f1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hyp1f1.html).
pub fn hyp1f1(a: f64, b: f64, x: f64) -> f64 {
    xsf::hyp1f1(a, b, x)
}

/// Gauss hypergeometric function ₂F₁(a,b;c;x).
///
/// ## Parameters and domain
///
/// Real parameters; real continuation generally requires x≤1; denominator poles may be singular.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::hyp2f1(0.5,1.0,2.5,0.3);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy hyp2f1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hyp2f1.html).
pub fn hyp2f1(a: f64, b: f64, c: f64, x: f64) -> f64 {
    xsf::hyp2f1(a, b, c, x)
}

/// Tricomi function U(a,b,x), the second confluent hypergeometric solution.
///
/// ## Parameters and domain
///
/// Real a,b and positive x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::hyperu(1.5,2.5,0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy hyperu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hyperu.html).
pub fn hyperu(a: f64, b: f64, x: f64) -> f64 {
    xsf::hypu(a, b, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::exp1(0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy exp1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.exp1.html).
pub fn exp1(x: f64) -> f64 {
    xsf::exp1(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::expi(0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy expi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expi.html).
pub fn expi(x: f64) -> f64 {
    xsf::expi(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::expn(3,0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy expn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expn.html).
pub fn expn(n: u32, x: f64) -> f64 {
    xsf::expn(n, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::sici(0.5);
/// let _ = value;
/// ```
///
/// # References
///
/// - [SciPy sici](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sici.html).
pub fn sici(x: f64) -> (f64, f64) {
    xsf::sici(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::shichi(0.5);
/// let _ = value;
/// ```
///
/// # References
///
/// - [SciPy shichi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.shichi.html).
pub fn shichi(x: f64) -> (f64, f64) {
    xsf::shichi(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::fresnel(0.5);
/// let _ = value;
/// ```
///
/// # References
///
/// - [SciPy fresnel](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.fresnel.html).
pub fn fresnel(x: f64) -> (f64, f64) {
    xsf::fresnel(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::spence(0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy spence](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.spence.html).
pub fn spence(x: f64) -> f64 {
    xsf::spence(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::zeta(2.5,1.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy zeta](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.zeta.html).
pub fn zeta(s: f64, q: f64) -> f64 {
    xsf::zeta(s, q)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::zetac(2.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy zetac](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.zetac.html).
pub fn zetac(x: f64) -> f64 {
    xsf::zetac(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::wrightomega(0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy wrightomega](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.wrightomega.html).
pub fn wrightomega(x: f64) -> f64 {
    xsf::wrightomega(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::wright_bessel(0.5,1.5,0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy wright_bessel](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.wright_bessel.html).
pub fn wright_bessel(a: f64, b: f64, x: f64) -> f64 {
    xsf::wright_bessel(a, b, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::log_wright_bessel(0.5,1.5,0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy log_wright_bessel](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.log_wright_bessel.html).
pub fn log_wright_bessel(a: f64, b: f64, x: f64) -> f64 {
    xsf::log_wright_bessel(a, b, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::voigt_profile(0.5,1.0,0.3);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy voigt_profile](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.voigt_profile.html).
pub fn voigt_profile(x: f64, sigma: f64, gamma: f64) -> f64 {
    xsf::voigt_profile(x, sigma, gamma)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::bdtr(2.0,5,0.3);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy bdtr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.bdtr.html).
pub fn bdtr(k: f64, n: i32, p: f64) -> f64 {
    xsf::bdtr(k, n, p)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::bdtrc(2.0,5,0.3);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy bdtrc](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.bdtrc.html).
pub fn bdtrc(k: f64, n: i32, p: f64) -> f64 {
    xsf::bdtrc(k, n, p)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::bdtri(2.0,5,0.3);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy bdtri](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.bdtri.html).
pub fn bdtri(k: f64, n: i32, y: f64) -> f64 {
    xsf::bdtri(k, n, y)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::chdtr(3.0,2.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy chdtr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.chdtr.html).
pub fn chdtr(df: f64, x: f64) -> f64 {
    xsf::chdtr(df, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::chdtrc(3.0,2.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy chdtrc](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.chdtrc.html).
pub fn chdtrc(df: f64, x: f64) -> f64 {
    xsf::chdtrc(df, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::chdtri(3.0,0.3);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy chdtri](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.chdtri.html).
pub fn chdtri(df: f64, q: f64) -> f64 {
    xsf::chdtri(df, q)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::fdtr(3.0,5.0,2.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy fdtr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.fdtr.html).
pub fn fdtr(dfn: f64, dfd: f64, x: f64) -> f64 {
    xsf::fdtr(dfn, dfd, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::fdtrc(3.0,5.0,2.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy fdtrc](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.fdtrc.html).
pub fn fdtrc(dfn: f64, dfd: f64, x: f64) -> f64 {
    xsf::fdtrc(dfn, dfd, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::fdtri(3.0,5.0,0.3);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy fdtri](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.fdtri.html).
pub fn fdtri(dfn: f64, dfd: f64, p: f64) -> f64 {
    xsf::fdtri(dfn, dfd, p)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::gdtr(1.5,2.5,0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy gdtr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gdtr.html).
pub fn gdtr(a: f64, b: f64, x: f64) -> f64 {
    xsf::gdtr(a, b, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::gdtrc(1.5,2.5,0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy gdtrc](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gdtrc.html).
pub fn gdtrc(a: f64, b: f64, x: f64) -> f64 {
    xsf::gdtrc(a, b, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::gdtrib(1.5,0.3,0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy gdtrib](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gdtrib.html).
pub fn gdtrib(a: f64, p: f64, x: f64) -> f64 {
    xsf::gdtrib(a, p, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::nbdtr(2,5,0.3);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy nbdtr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.nbdtr.html).
pub fn nbdtr(k: i32, n: i32, p: f64) -> f64 {
    xsf::nbdtr(k, n, p)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::nbdtrc(2,5,0.3);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy nbdtrc](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.nbdtrc.html).
pub fn nbdtrc(k: i32, n: i32, p: f64) -> f64 {
    xsf::nbdtrc(k, n, p)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::nbdtri(2,5,0.3);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy nbdtri](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.nbdtri.html).
pub fn nbdtri(k: i32, n: i32, y: f64) -> f64 {
    xsf::nbdtri(k, n, y)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::ndtr(0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy ndtr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ndtr.html).
pub fn ndtr(x: f64) -> f64 {
    xsf::ndtr(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::ndtri(0.3);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy ndtri](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ndtri.html).
pub fn ndtri(p: f64) -> f64 {
    xsf::ndtri(p)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::log_ndtr(-30.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy log_ndtr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.log_ndtr.html).
pub fn log_ndtr(x: f64) -> f64 {
    xsf::log_ndtr(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::ndtri_exp(-100.0);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy ndtri_exp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ndtri_exp.html).
pub fn ndtri_exp(y: f64) -> f64 {
    xsf::ndtri_exp(y)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::pdtr(2.0,1.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy pdtr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pdtr.html).
pub fn pdtr(k: f64, m: f64) -> f64 {
    xsf::pdtr(k, m)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::pdtrc(2.0,1.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy pdtrc](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pdtrc.html).
pub fn pdtrc(k: f64, m: f64) -> f64 {
    xsf::pdtrc(k, m)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::pdtri(2,0.3);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy pdtri](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pdtri.html).
pub fn pdtri(k: i32, p: f64) -> f64 {
    xsf::pdtri(k, p)
}

/// Student t CDF with df degrees of freedom.
///
/// ## Parameters and domain
///
/// df>0; all real t.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::stdtr(3.0,0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy stdtr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.stdtr.html).
pub fn stdtr(df: f64, t: f64) -> f64 {
    xsf::stdtr(df, t)
}

/// Inverse Student t CDF with respect to t.
///
/// ## Parameters and domain
///
/// df>0 and p in `[0,1]`.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::stdtrit(3.0,0.3);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy stdtrit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.stdtrit.html).
pub fn stdtrit(df: f64, p: f64) -> f64 {
    xsf::stdtri(df, p)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::kolmogorov(0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy kolmogorov](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kolmogorov.html).
pub fn kolmogorov(x: f64) -> f64 {
    xsf::kolmogorov(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::kolmogi(0.3);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy kolmogi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kolmogi.html).
pub fn kolmogi(q: f64) -> f64 {
    xsf::kolmogi(q)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::smirnov(5,0.3);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy smirnov](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.smirnov.html).
pub fn smirnov(n: i32, x: f64) -> f64 {
    xsf::smirnov(n, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::smirnovi(5,0.3);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy smirnovi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.smirnovi.html).
pub fn smirnovi(n: i32, q: f64) -> f64 {
    xsf::smirnovi(n, q)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::owens_t(0.5,0.3);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy owens_t](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.owens_t.html).
pub fn owens_t(h: f64, a: f64) -> f64 {
    xsf::owens_t(h, a)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::tklmbda(0.5,0.3);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy tklmbda](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.tklmbda.html).
pub fn tklmbda(x: f64, lambda: f64) -> f64 {
    xsf::tukeylambdacdf(x, lambda)
}

/// Parabolic cylinder function and derivative (D_v(x),D′_v(x)).
///
/// ## Parameters and domain
///
/// Real order v and real x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::pbdv(0.5,1.0);
/// let _ = value;
/// ```
///
/// # References
///
/// - [SciPy pbdv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pbdv.html).
pub fn pbdv(v: f64, x: f64) -> (f64, f64) {
    xsf::pbdv(v, x)
}

/// Parabolic cylinder function and derivative (V_v(x),V′_v(x)).
///
/// ## Parameters and domain
///
/// Real order v and real x.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::pbvv(0.5,1.0);
/// let _ = value;
/// ```
///
/// # References
///
/// - [SciPy pbvv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pbvv.html).
pub fn pbvv(v: f64, x: f64) -> (f64, f64) {
    xsf::pbvv(v, x)
}

/// Parabolic cylinder function (W(a,x),W′(a,x)).
///
/// ## Parameters and domain
///
/// abs(a)≤5 and abs(x)≤5 for the supported algorithm.
///
/// ## Returns
///
/// Returns IEEE values: NaN outside the real domain, infinity at divergent poles or overflow, and zero on underflow. NaN inputs propagate unless a limiting case specifies otherwise.
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::pbwa(0.5,1.0);
/// let _ = value;
/// ```
///
/// # References
///
/// - [SciPy pbwa](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pbwa.html).
pub fn pbwa(a: f64, x: f64) -> (f64, f64) {
    xsf::pbwa(a, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::mathieu_a(2,0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
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
    xsf::mathieu_a(m, q)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::mathieu_b(2,0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
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
    xsf::mathieu_b(m, q)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::mathieu_cem(2,0.5,30.0);
/// let _ = value;
/// ```
///
/// # References
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
    xsf::mathieu_cem(m, q, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::mathieu_sem(2,0.5,30.0);
/// let _ = value;
/// ```
///
/// # References
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
    xsf::mathieu_sem(m, q, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::mathieu_modcem1(2,0.5,0.3);
/// let _ = value;
/// ```
///
/// # References
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
    xsf::mathieu_modcem1(m, q, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::mathieu_modcem2(2,0.5,0.3);
/// let _ = value;
/// ```
///
/// # References
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
    xsf::mathieu_modcem2(m, q, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::mathieu_modsem1(2,0.5,0.3);
/// let _ = value;
/// ```
///
/// # References
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
    xsf::mathieu_modsem1(m, q, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::mathieu_modsem2(2,0.5,0.3);
/// let _ = value;
/// ```
///
/// # References
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
    xsf::mathieu_modsem2(m, q, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::ber(0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy ber](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ber.html).
pub fn ber(x: f64) -> f64 {
    xsf::ber(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::bei(0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy bei](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.bei.html).
pub fn bei(x: f64) -> f64 {
    xsf::bei(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::ker(0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy ker](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ker.html).
pub fn ker(x: f64) -> f64 {
    xsf::ker(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::kei(0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy kei](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kei.html).
pub fn kei(x: f64) -> f64 {
    xsf::kei(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::berp(0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy berp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.berp.html).
pub fn berp(x: f64) -> f64 {
    xsf::berp(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::beip(0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy beip](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.beip.html).
pub fn beip(x: f64) -> f64 {
    xsf::beip(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::kerp(0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy kerp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kerp.html).
pub fn kerp(x: f64) -> f64 {
    xsf::kerp(x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::keip(0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy keip](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.keip.html).
pub fn keip(x: f64) -> f64 {
    xsf::keip(x)
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
/// # Examples
///
/// ```
/// let value = sci_special::ber_zeros(3)?;
/// assert_eq!(value.len(), 3);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [SciPy ber_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ber_zeros.html).
pub fn ber_zeros(nt: usize) -> Result<Vec<f64>> {
    if !(nt <= 4096 && nt > 0) {
        return Err(Error::InvalidInput(
            "requested special-function sequence is too large or empty",
        ));
    }
    Ok(xsf::ber_zeros(nt))
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
/// # Examples
///
/// ```
/// let value = sci_special::bei_zeros(3)?;
/// assert_eq!(value.len(), 3);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [SciPy bei_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.bei_zeros.html).
pub fn bei_zeros(nt: usize) -> Result<Vec<f64>> {
    if !(nt <= 4096 && nt > 0) {
        return Err(Error::InvalidInput(
            "requested special-function sequence is too large or empty",
        ));
    }
    Ok(xsf::bei_zeros(nt))
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
/// # Examples
///
/// ```
/// let value = sci_special::ker_zeros(3)?;
/// assert_eq!(value.len(), 3);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [SciPy ker_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ker_zeros.html).
pub fn ker_zeros(nt: usize) -> Result<Vec<f64>> {
    if !(nt <= 4096 && nt > 0) {
        return Err(Error::InvalidInput(
            "requested special-function sequence is too large or empty",
        ));
    }
    Ok(xsf::ker_zeros(nt))
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
/// # Examples
///
/// ```
/// let value = sci_special::kei_zeros(3)?;
/// assert_eq!(value.len(), 3);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [SciPy kei_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kei_zeros.html).
pub fn kei_zeros(nt: usize) -> Result<Vec<f64>> {
    if !(nt <= 4096 && nt > 0) {
        return Err(Error::InvalidInput(
            "requested special-function sequence is too large or empty",
        ));
    }
    Ok(xsf::kei_zeros(nt))
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
/// # Examples
///
/// ```
/// let value = sci_special::berp_zeros(3)?;
/// assert_eq!(value.len(), 3);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [SciPy berp_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.berp_zeros.html).
pub fn berp_zeros(nt: usize) -> Result<Vec<f64>> {
    if !(nt <= 4096 && nt > 0) {
        return Err(Error::InvalidInput(
            "requested special-function sequence is too large or empty",
        ));
    }
    Ok(xsf::berp_zeros(nt))
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
/// # Examples
///
/// ```
/// let value = sci_special::beip_zeros(3)?;
/// assert_eq!(value.len(), 3);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [SciPy beip_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.beip_zeros.html).
pub fn beip_zeros(nt: usize) -> Result<Vec<f64>> {
    if !(nt <= 4096 && nt > 0) {
        return Err(Error::InvalidInput(
            "requested special-function sequence is too large or empty",
        ));
    }
    Ok(xsf::beip_zeros(nt))
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
/// # Examples
///
/// ```
/// let value = sci_special::kerp_zeros(3)?;
/// assert_eq!(value.len(), 3);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [SciPy kerp_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kerp_zeros.html).
pub fn kerp_zeros(nt: usize) -> Result<Vec<f64>> {
    if !(nt <= 4096 && nt > 0) {
        return Err(Error::InvalidInput(
            "requested special-function sequence is too large or empty",
        ));
    }
    Ok(xsf::kerp_zeros(nt))
}

// kei'(x) = Im[-a K_1(a x)], a = exp(i pi/4). Removing the
// positive exp(-x/sqrt(2)) factor avoids underflow for long sequences.
fn refine_keip_zero(mut x: f64) -> Result<f64> {
    use num_complex::Complex64;
    use std::f64::consts::FRAC_1_SQRT_2;
    let a = Complex64::new(FRAC_1_SQRT_2, FRAC_1_SQRT_2);
    for _ in 0..16 {
        let z = a * x;
        let phase = Complex64::from_polar(1., -z.im);
        let k0 = xsf::bessel_ke(0., z);
        let k1 = xsf::bessel_ke(1., z);
        let value = (-a * phase * k1).im;
        // K_1'(z) = -K_0(z) - K_1(z)/z. Include the derivative
        // of the removed real exponential factor in the scaled function.
        let derivative = (a * a * phase * (k0 + k1 / z)).im + FRAC_1_SQRT_2 * value;
        let step = value / derivative;
        if !step.is_finite() || step.abs() > 1. {
            return Err(Error::NumericalFailure("Kelvin zero refinement failed"));
        }
        x -= step;
        if step.abs() <= 8. * f64::EPSILON * x {
            return Ok(x);
        }
    }
    Err(Error::NumericalFailure(
        "Kelvin zero refinement did not converge",
    ))
}

fn stable_keip_zeros(nt: usize) -> Result<Vec<f64>> {
    use std::f64::consts::{PI, SQRT_2};
    (0..nt)
        .map(|index| refine_keip_zero(4.93 + index as f64 * PI * SQRT_2))
        .collect()
}

#[cfg(test)]
#[path = "tests/kelvin_refinement.rs"]
mod kelvin_refinement_tests;

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
/// at most 16 Newton iterations per zero. Scaling avoids exponential underflow.
/// This avoids cancellation in the Kelvin series near x=9.4; SciPy's
/// `keip_zeros` can differ by about 4e-10 there. Accuracy remains limited
/// by binary64 argument reduction for large zeros.
///
/// # Examples
///
/// ```
/// let value = sci_special::keip_zeros(3)?;
/// assert_eq!(value.len(), 3);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
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
    stable_keip_zeros(nt)
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
/// at most 16 Newton iterations per zero. Scaling avoids exponential underflow.
/// This avoids cancellation in the Kelvin series near x=9.4; SciPy's
/// `keip_zeros` can differ by about 4e-10 there. Other Kelvin families retain
/// the backend's series/asymptotic accuracy limitations.
///
/// # Examples
///
/// ```
/// let value = sci_special::kelvin_zeros(3)?;
/// assert_eq!(value[0].len(), 3);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
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
        xsf::ber_zeros(nt),
        xsf::bei_zeros(nt),
        xsf::ker_zeros(nt),
        xsf::kei_zeros(nt),
        xsf::berp_zeros(nt),
        xsf::beip_zeros(nt),
        xsf::kerp_zeros(nt),
        stable_keip_zeros(nt)?,
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::lpmv(1,2.5,0.3);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy lpmv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.lpmv.html).
pub fn lpmv(m: i32, v: f64, x: f64) -> f64 {
    xsf::pmv(m, v, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::legendre_p(3,0.3);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy legendre_p](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.legendre_p.html).
pub fn legendre_p(n: i32, x: f64) -> f64 {
    xsf::legendre_p(n, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::assoc_legendre_p(3,1,0.3);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy assoc_legendre_p](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.assoc_legendre_p.html).
pub fn assoc_legendre_p(n: i32, m: i32, x: f64) -> f64 {
    xsf::assoc_legendre_p(n, m, x)
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
/// No NumPy broadcasting or output buffers are provided.
///
/// # Examples
///
/// ```
/// let value = sci_special::sph_legendre_p(3,1,0.3);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy sph_legendre_p](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sph_legendre_p.html).
pub fn sph_legendre_p(n: i32, m: i32, theta: f64) -> f64 {
    xsf::sph_legendre_p(n, m, theta)
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
/// # Examples
///
/// ```
/// let value = sci_special::lqn(3,0.3)?;
/// let _ = value;
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [SciPy lqn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.lqn.html).
pub fn lqn(n: usize, x: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    if !(n <= 4096) {
        return Err(Error::InvalidInput(
            "requested special-function sequence is too large or empty",
        ));
    }
    let (mut values, mut derivatives) = xsf::legendre_q_all(n.max(1), x);
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
/// # Examples
///
/// ```
/// let value = sci_special::lqmn(2,3,0.3)?;
/// assert_eq!(value.0.len(), 3);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// - [SciPy lqmn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.lqmn.html).
pub fn lqmn(m: usize, n: usize, x: f64) -> Result<ValueDerivativeMatrices> {
    if !(m <= 4096 && n <= 4096 && (m + 1) * (n + 1) <= 1_048_576) {
        return Err(Error::InvalidInput(
            "requested special-function sequence is too large or empty",
        ));
    }
    let (values, derivatives) = xsf::assoc_legendre_q_all(n.max(1), m.max(1), x);
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

/// Real cube root of x.
///
/// ## Parameters and domain
///
/// All real values; signed zero and infinities are preserved.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// # Examples
///
/// ```
/// assert!(sci_special::cbrt(0.5).is_finite());
/// ```
///
/// # References
///
/// - [SciPy cbrt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.cbrt.html).
pub fn cbrt(x: f64) -> f64 {
    x.cbrt()
}

/// Base-two exponential 2^x.
///
/// ## Parameters and domain
///
/// All real values; overflow gives +∞.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// # Examples
///
/// ```
/// assert!(sci_special::exp2(0.5).is_finite());
/// ```
///
/// # References
///
/// - [SciPy exp2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.exp2.html).
pub fn exp2(x: f64) -> f64 {
    x.exp2()
}

/// Base-ten exponential 10^x.
///
/// ## Parameters and domain
///
/// All real values; overflow gives +∞.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// # Examples
///
/// ```
/// assert!(sci_special::exp10(0.5).is_finite());
/// ```
///
/// # References
///
/// - [SciPy exp10](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.exp10.html).
pub fn exp10(x: f64) -> f64 {
    10.0_f64.powf(x)
}

/// exp(x)−1, avoiding cancellation near zero.
///
/// ## Parameters and domain
///
/// All real values.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// # Examples
///
/// ```
/// assert!(sci_special::expm1(0.000001).is_finite());
/// ```
///
/// # References
///
/// - [SciPy expm1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expm1.html).
pub fn expm1(x: f64) -> f64 {
    x.exp_m1()
}

/// ln(1+x), avoiding cancellation near zero.
///
/// ## Parameters and domain
///
/// x≥−1; −1 gives −∞ and smaller inputs give NaN.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// # Examples
///
/// ```
/// assert!(sci_special::log1p(0.000001).is_finite());
/// ```
///
/// # References
///
/// - [SciPy log1p](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.log1p.html).
pub fn log1p(x: f64) -> f64 {
    x.ln_1p()
}

/// Logistic sigmoid 1/(1+exp(−x)), evaluated without overflow.
///
/// ## Parameters and domain
///
/// All real values; infinite endpoints give zero and one.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// # Examples
///
/// ```
/// assert!(sci_special::expit(0.5).is_finite());
/// ```
///
/// # References
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
/// # Examples
///
/// ```
/// assert!(sci_special::logit(0.3).is_finite());
/// ```
///
/// # References
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
/// # Examples
///
/// ```
/// assert!(sci_special::softplus(0.5).is_finite());
/// ```
///
/// # References
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
/// # Examples
///
/// ```
/// assert!(sci_special::log_expit(-100.0).is_finite());
/// ```
///
/// # References
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
/// # Examples
///
/// ```
/// assert!(sci_special::exprel(0.000001).is_finite());
/// ```
///
/// # References
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
/// # Examples
///
/// ```
/// assert!(sci_special::sinc(0.5).is_finite());
/// ```
///
/// # References
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
/// # Examples
///
/// ```
/// assert!(sci_special::cosm1(0.000001).is_finite());
/// ```
///
/// # References
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
/// # Examples
///
/// ```
/// assert!(sci_special::sindg(30.0).is_finite());
/// ```
///
/// # References
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
/// # Examples
///
/// ```
/// assert!(sci_special::cosdg(30.0).is_finite());
/// ```
///
/// # References
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
/// # Examples
///
/// ```
/// assert!(sci_special::tandg(30.0).is_finite());
/// ```
///
/// # References
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
/// # Examples
///
/// ```
/// assert!(sci_special::cotdg(30.0).is_finite());
/// ```
///
/// # References
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
/// # Examples
///
/// ```
/// assert!(sci_special::radian(30.0,20.0,10.0).is_finite());
/// ```
///
/// # References
///
/// - [SciPy radian](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.radian.html).
pub fn radian(d: f64, m: f64, s: f64) -> f64 {
    (d + (m + s / 60.0) / 60.0).to_radians()
}

/// Rounds to the nearest integer, with ties going to the even integer.
///
/// ## Parameters and domain
///
/// All real values; signed zero and infinities are preserved.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// # Examples
///
/// ```
/// assert!(sci_special::round(2.5).is_finite());
/// ```
///
/// # References
///
/// - [SciPy round](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.round.html).
pub fn round(x: f64) -> f64 {
    x.round_ties_even()
}

/// Entropy contribution −x ln x.
///
/// ## Parameters and domain
///
/// Positive x; zero gives zero and negative x gives −∞.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// # Examples
///
/// ```
/// assert!(sci_special::entr(0.3).is_finite());
/// ```
///
/// # References
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
/// # Examples
/// ```
/// assert!(sci_special::rel_entr(0.3,0.5).is_finite());
/// ```
/// # References
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
/// # Examples
/// ```
/// assert!(sci_special::kl_div(0.3,0.5) > 0.0);
/// ```
/// # References
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
/// # Examples
///
/// ```
/// assert!(sci_special::huber(1.0,2.0).is_finite());
/// ```
///
/// # References
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
/// # Examples
/// ```
/// assert_eq!(sci_special::pseudo_huber(1e308, 1.0), 0.5);
/// ```
/// # References
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
/// # Examples
///
/// ```
/// assert!(sci_special::xlogy(0.0,0.0).is_finite());
/// ```
///
/// # References
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
/// # Examples
///
/// ```
/// assert!(sci_special::xlog1py(0.3,0.000001).is_finite());
/// ```
///
/// # References
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
/// # Examples
///
/// ```
/// assert!(sci_special::powm1(1.000001,0.3).is_finite());
/// ```
///
/// # References
///
/// - [SciPy powm1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.powm1.html).
pub fn powm1(x: f64, y: f64) -> f64 {
    if x > 0.0 {
        (y * x.ln()).exp_m1()
    } else {
        x.powf(y) - 1.0
    }
}

/// Box–Cox transformation (x^lambda−1)/lambda, with ln(x) at lambda=0.
///
/// ## Parameters and domain
///
/// x≥0; zero gives −∞ when lambda≤0 and −1/lambda otherwise.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// # Examples
///
/// ```
/// assert!(sci_special::boxcox(2.0,0.3).is_finite());
/// ```
///
/// # References
///
/// - [SciPy boxcox](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.boxcox.html).
pub fn boxcox(x: f64, lambda: f64) -> f64 {
    if x < 0.0 {
        f64::NAN
    } else if lambda == 0.0 {
        x.ln()
    } else {
        (lambda * x.ln()).exp_m1() / lambda
    }
}

/// Box–Cox transformation of 1+x, stable near zero.
///
/// ## Parameters and domain
///
/// x≥−1; endpoint behavior follows boxcox.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// # Examples
///
/// ```
/// assert!(sci_special::boxcox1p(0.000001,0.3).is_finite());
/// ```
///
/// # References
///
/// - [SciPy boxcox1p](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.boxcox1p.html).
pub fn boxcox1p(x: f64, lambda: f64) -> f64 {
    if x < -1.0 {
        f64::NAN
    } else if lambda == 0.0 {
        x.ln_1p()
    } else {
        (lambda * x.ln_1p()).exp_m1() / lambda
    }
}

/// Inverse Box–Cox transformation (1+lambda*y)^(1/lambda).
///
/// ## Parameters and domain
///
/// 1+lambda*y≥0; lambda=0 uses exp(y).
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// # Examples
///
/// ```
/// assert!(sci_special::inv_boxcox(0.5,0.3).is_finite());
/// ```
///
/// # References
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
/// # Examples
///
/// ```
/// assert!(sci_special::inv_boxcox1p(0.000001,0.3).is_finite());
/// ```
///
/// # References
///
/// - [SciPy inv_boxcox1p](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.inv_boxcox1p.html).
pub fn inv_boxcox1p(y: f64, lambda: f64) -> f64 {
    if lambda == 0.0 {
        y.exp_m1()
    } else {
        ((lambda * y).ln_1p() / lambda).exp_m1()
    }
}

/// Alias for digamma, ψ(x).
///
/// ## Parameters and domain
///
/// Same domain and endpoint behavior as digamma.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// # Examples
///
/// ```
/// assert!(sci_special::psi(2.5).is_finite());
/// ```
///
/// # References
///
/// - [SciPy psi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.psi.html).
pub fn psi(x: f64) -> f64 {
    super::digamma(x)
}

/// Complementary regularized incomplete beta 1−I_x(a,b)=I_(1−x)(b,a).
///
/// ## Parameters and domain
///
/// Finite positive a,b and x in `[0,1]`; the upper tail is evaluated directly.
/// NaN propagates; divergent values and overflow use IEEE infinities.
///
/// # Examples
///
/// ```
/// assert!(sci_special::betaincc(2.5,3.5,0.7).is_finite());
/// ```
///
/// # References
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
/// # Examples
///
/// ```
/// assert!(sci_special::betainccinv(2.5,3.5,0.3).is_finite());
/// ```
///
/// # References
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
/// # Examples
///
/// ```
/// assert!(sci_special::nrdtrimn(0.3,0.5,1.5).is_finite());
/// ```
///
/// # References
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
/// # Examples
///
/// ```
/// assert!(sci_special::nrdtrisd(1.5,0.3,0.5).is_finite());
/// ```
///
/// # References
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
/// # Examples
///
/// ```
/// assert!(sci_special::gdtria(0.3,2.5,0.5).is_finite());
/// ```
///
/// # References
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
/// # Examples
///
/// ```
/// assert!(sci_special::gdtrix(1.5,2.5,0.3).is_finite());
/// ```
///
/// # References
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
/// # Examples
///
/// ```
/// assert!(sci_special::chdtriv(0.3,2.5).is_finite());
/// ```
///
/// # References
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
/// # Examples
///
/// ```
/// assert!(sci_special::jn(2,0.5).is_finite());
/// ```
///
/// # References
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
/// # Examples
///
/// ```
/// assert!(sci_special::yn(2,0.5).is_finite());
/// ```
///
/// # References
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
/// # Examples
///
/// ```
/// assert!(sci_special::kn(2,0.5).is_finite());
/// ```
///
/// # References
///
/// - [SciPy kn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kn.html).
pub fn kn(n: i32, x: f64) -> f64 {
    kv(f64::from(n), x)
}
