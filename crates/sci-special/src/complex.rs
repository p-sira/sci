//! Complex special functions using Sci's shared [`Complex`] type.
//!
//! Arguments and results are dimensionless. Functions compute in binary64 and
//! convert components back to `T`; this is not arbitrary precision. Real-axis
//! branch cuts preserve the sign of the imaginary zero. Inputs must be finite;
//! overflow and backend convergence/precision failures return shared Sci errors.
//! Bessel and Airy algorithms use bounded AMOS series and recurrences with
//! parameter-dependent accuracy; error functions use bounded Faddeeva rational
//! approximations. No user-adjustable work budget or tolerance is exposed.
//!
//! These functions complement the real scalar facade. Arithmetic functions live
//! in `sci_complex`; no backend types are exposed here.
use crate::{Error, Result};
pub use sci_complex::Complex;
use sci_core::Float;

pub(crate) fn native(z: Complex) -> num_complex::Complex64 {
    num_complex::Complex64::new(z.re, z.im)
}
pub(crate) fn owned(z: num_complex::Complex64) -> Complex {
    Complex { re: z.re, im: z.im }
}
fn input<T: Float>(z: Complex<T>) -> Result<num_complex::Complex64> {
    if !z.re.is_finite() || !z.im.is_finite() {
        return Err(Error::InvalidInput(
            "complex special functions require finite components",
        ));
    }
    let value = num_complex::Complex64::new(
        z.re.to_f64().ok_or(Error::NonFiniteEvaluation)?,
        z.im.to_f64().ok_or(Error::NonFiniteEvaluation)?,
    );
    if !value.re.is_finite() || !value.im.is_finite() {
        return Err(Error::NonFiniteEvaluation);
    }
    Ok(value)
}
fn output<T: Float>(z: num_complex::Complex64) -> Result<Complex<T>> {
    let result = Complex {
        re: T::from(z.re).ok_or(Error::NonFiniteEvaluation)?,
        im: T::from(z.im).ok_or(Error::NonFiniteEvaluation)?,
    };
    if !result.re.is_finite() || !result.im.is_finite() {
        return Err(Error::NonFiniteEvaluation);
    }
    Ok(result)
}

/// Computes complex Lambert W on the integer branch `k`, solving `w exp(w) = z`.
///
/// ## Parameters and conventions
/// `z` is a dimensionless complex argument, and `k = 0` selects the principal
/// branch. Positive and negative integers select the other logarithmic sheets.
/// The scalar real branches are also available in `elementary`.
///
/// ## Domain and special values
/// Both input components must be finite, and `|k| <= 1_000_000`. At `z = 0`,
/// only the principal branch is finite and returns signed complex zero.
/// The branches `k=0` and `k=-1` meet at `z=-1/e`, where both return `-1`.
/// The sign of the imaginary component distinguishes the banks of a cut.
///
/// ## Returns
/// A complex value in the same floating-point type as `z`.
///
/// ## Errors
/// Nonfinite input, an unsupported branch index, a singularity, failed
/// convergence, or nonfinite output produces a shared `Error`.
///
/// ## Numerical behavior
/// Uses bounded Halley refinement (at most 100 steps) from rational, Puiseux,
/// or asymptotic initial approximations. The iteration checks its equation
/// residual; accuracy may degrade close to the coalescing branches at `-1/e`.
/// Computation is in binary64 before conversion to `T`.
///
/// ## Examples
/// ```
/// let z = sci_complex::Complex { re: 1.0_f64, im: 0.0 };
/// let w = sci_special::complex::lambertw(z, 0)?;
/// assert!((w.re - 0.5671432904097838).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF §4.13](https://dlmf.nist.gov/4.13).
pub fn lambertw<T: Float>(z: Complex<T>, k: i32) -> Result<Complex<T>> {
    output(
        crate::backend::complex_lambert::lambertw(input(z)?, k).map_err(Error::NumericalFailure)?,
    )
}

/// Computes Spence's integral `Li₂(1-z)` on its principal complex branch.
///
/// ## Parameters and conventions
/// `z` is dimensionless and uses Sci's shared complex representation.
/// `Li₂(w)=-∫₀ʷ ln(1-t)/t dt` defines the dilogarithm.
///
/// ## Domain and special values
/// Finite input is required; the branch cut is the negative real `z` axis.
/// Signed imaginary zero selects the corresponding cut bank.
/// `spence(1)=0`, and `spence(0)=π²/6`.
///
/// ## Returns
/// A complex value with the same floating-point type as the input.
///
/// ## Errors
/// Nonfinite input, failed series convergence, or nonfinite output produces a shared Sci error.
///
/// ## Numerical behavior
/// Binary64 evaluation uses bounded power and Bernoulli series, reflection,
/// Landen, and inversion identities. Branch-cut cancellation may lose precision.
///
/// ## Examples
/// ```
/// let z = sci_complex::Complex { re: 1.0_f64, im: 0.0 };
/// assert_eq!(sci_special::complex::spence(z)?.re, 0.0);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF §25.12](https://dlmf.nist.gov/25.12).
pub fn spence<T: Float>(z: Complex<T>) -> Result<Complex<T>> {
    output(crate::backend::complex_polylog::spence(input(z)?).map_err(Error::NumericalFailure)?)
}

/// Computes `x ln(y)` with the principal complex logarithm, defining `0 ln(y)=0`.
///
/// ## Parameters and conventions
/// `x,y` are dimensionless complex arguments; `ln` uses the principal branch.
///
/// ## Domain and special values
/// Finite components are required. The value at `x=0` is exactly zero, even
/// for `y=0`. For nonzero `x`, `y=0` is a logarithmic singularity.
/// Signed imaginary zero selects the negative-real cut bank of `ln(y)`.
///
/// ## Returns
/// A complex value in the same float type as `x` and `y`.
///
/// ## Errors
/// Invalid components, a logarithmic singularity, or nonfinite results.
///
/// ## Numerical behavior
/// Computes the principal complex logarithm in binary64 before multiplying.
///
/// ## Examples
/// ```
/// let x = sci_complex::Complex { re: 0.0_f64, im: 0.0 };
/// assert_eq!(sci_special::complex::xlogy(x,x)?,x);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF §4.2](https://dlmf.nist.gov/4.2).
pub fn xlogy<T: Float>(x: Complex<T>, y: Complex<T>) -> Result<Complex<T>> {
    output(
        crate::backend::complex_polylog::xlogy(input(x)?, input(y)?)
            .map_err(Error::NumericalFailure)?,
    )
}

/// Computes `x ln(1+y)` stably for small complex `y`, defining `0 ln(1+y)=0`.
///
/// ## Parameters and conventions
/// Both arguments are dimensionless; the logarithm uses its principal branch.
///
/// ## Domain and special values
/// Finite complex inputs are required. The nonzero-`x` logarithm has a pole at
/// `y=-1`; signed imaginary zero selects the cut bank below `y=-1`.
///
/// ## Returns
/// A complex result in the common floating-point type.
///
/// ## Errors
/// Nonfinite input/output and logarithmic poles produce shared Sci errors.
///
/// ## Numerical behavior
/// A binary64 `log1p`-style expansion protects sub-ulp complex increments.
///
/// ## Examples
/// ```
/// let x = sci_complex::Complex { re: 1.0_f64, im: 0.0 };
/// let y = sci_complex::Complex { re: 1e-16_f64, im: 0.0 };
/// assert!((sci_special::complex::xlog1py(x,y)?.re - 1e-16).abs() < 1e-28);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF §4.2](https://dlmf.nist.gov/4.2).
pub fn xlog1py<T: Float>(x: Complex<T>, y: Complex<T>) -> Result<Complex<T>> {
    output(
        crate::backend::complex_polylog::xlog1py(input(x)?, input(y)?)
            .map_err(Error::NumericalFailure)?,
    )
}

/// Computes `E₁(z) = ∫₁∞ exp(−zt)/t dt` on the principal branch.
///
/// ## Parameters and conventions
/// `z` is dimensionless; integration defines `E₁` for `Re(z)>0`, with
/// analytic continuation elsewhere.
///
/// ## Domain and special values
/// Finite inputs are required. The negative real axis is a branch cut:
/// signed imaginary zero chooses its upper or lower bank. Zero is a pole.
///
/// ## Returns
/// A complex value in the input precision.
///
/// ## Errors
/// Nonfinite or unsupported arguments, nonconvergence and nonfinite results
/// return shared Sci errors.
///
/// ## Numerical behavior
/// Uses bounded binary64 series, continued fractions and asymptotic kernels.
/// Large arguments and cancellation near the cut can reduce accuracy.
///
/// ## Examples
/// ```
/// let z = sci_complex::Complex { re: 1.0_f64, im: 0.0 };
/// assert!((sci_special::complex::exp1(z)?.re - 0.2193839343955203).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF §6.2](https://dlmf.nist.gov/6.2).
pub fn exp1<T: Float>(z: Complex<T>) -> Result<Complex<T>> {
    output(crate::backend::complex::exp1(input(z)?).map_err(Error::NumericalFailure)?)
}

/// Computes `Ei(z)` with the standard principal-value continuation.
///
/// ## Parameters and conventions
/// On the positive real axis `Ei(x) = PV ∫₋∞ˣ exp(t)/t dt`.
/// Complex values use analytic continuation with the negative-real cut.
///
/// ## Domain and special values
/// Requires finite components and nonzero argument; signed imaginary zero
/// selects the side of the negative-real cut.
///
/// ## Returns
/// A complex result in the input float precision.
///
/// ## Errors
/// Unsupported magnitudes, divergence, nonfinite results, and failed series
/// convergence return a shared Sci error.
///
/// ## Numerical behavior
/// Binary64 evaluation uses bounded Taylor, continued fraction and asymptotic
/// branches. Accuracy worsens near Stokes boundaries and severe cancellation.
///
/// ## Examples
/// ```
/// let z = sci_complex::Complex { re: 1.0_f64, im: 0.0 };
/// assert!((sci_special::complex::expi(z)?.re - 1.8951178163559368).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF §6.2](https://dlmf.nist.gov/6.2).
pub fn expi<T: Float>(z: Complex<T>) -> Result<Complex<T>> {
    output(crate::backend::complex::expi(input(z)?).map_err(Error::NumericalFailure)?)
}

/// Computes the analytic continuation of Riemann's `ζ(s)=Σₙ₌₁∞ n^(-s)`.
///
/// ## Parameters and conventions
/// `s` is a dimensionless complex order; the sum converges absolutely for
/// `Re(s)>1` and is analytically continued elsewhere.
///
/// ## Domain and special values
/// This bounded implementation supports `0≤Re(s)≤30`, `|Im(s)|≤32`, excluding
/// the pole at `s=1`. It returns `ζ(0)=-1/2` exactly.
///
/// ## Returns
/// The complex value in the same floating-point type as the input.
///
/// ## Errors
/// Unsupported or nonfinite inputs, a pole, loss of certified binary64
/// accuracy after cancellation, or nonfinite output return shared Sci errors.
///
/// ## Numerical behavior
/// Uses a compensated Dirichlet prefix and 12 Euler–Maclaurin corrections,
/// with a bounded work budget and a cancellation check. Complex zeros and
/// large oscillatory arguments can be ill-conditioned and return errors.
///
/// ## Examples
/// ```
/// let s = sci_complex::Complex { re: 2.0_f64, im: 0.0 };
/// assert!((sci_special::complex::zeta(s)?.re - std::f64::consts::PI.powi(2)/6.0).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF §25.2](https://dlmf.nist.gov/25.2).
pub fn zeta<T: Float>(s: Complex<T>) -> Result<Complex<T>> {
    output(crate::backend::complex_zeta::riemann_zeta(input(s)?).map_err(Error::NumericalFailure)?)
}

/// Computes the Hurwitz zeta `ζ(s,q)=Σₙ₌₀∞ (q+n)^(-s)` by analytic continuation.
///
/// ## Parameters and conventions
/// Both `s` and `q` use shared complex numbers. Complex powers use the
/// principal logarithm. `q=1` recovers the Riemann zeta function.
///
/// ## Domain and special values
/// Requires `0≤Re(s)≤30`, `|Im(s)|≤32`, `0<Re(q)≤64` and `|Im(q)|≤32`.
/// The pole at `s=1` is excluded; `ζ(0,q)=1/2-q`.
///
/// ## Returns
/// A complex value in the common input floating-point type.
///
/// ## Errors
/// Nonfinite input/output, unsupported argument ranges, poles, excessive
/// cancellation or insufficient convergence accuracy return shared Sci errors.
///
/// ## Numerical behavior
/// Evaluates in binary64 by a bounded Euler–Maclaurin method with compensated
/// summation and an error guard. Severe cancellation near zeros can cause an
/// error rather than returning an unverified approximation.
///
/// ## Examples
/// ```
/// let z = sci_complex::Complex { re: 2.0_f64, im: 0.0 };
/// let q = sci_complex::Complex { re: 1.0_f64, im: 0.0 };
/// assert!((sci_special::complex::hurwitz_zeta(z,q)?.re - 1.6449340668482264).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF §25.11](https://dlmf.nist.gov/25.11).
pub fn hurwitz_zeta<T: Float>(s: Complex<T>, q: Complex<T>) -> Result<Complex<T>> {
    output(
        crate::backend::complex_zeta::hurwitz_zeta(input(s)?, input(q)?)
            .map_err(Error::NumericalFailure)?,
    )
}

/// Computes Γ(z) = ∫₀∞ t^(z−1) exp(−t) dt for Re(z)>0, continued meromorphically.
///
/// ## Parameters and conventions
/// `z` is a dimensionless complex argument using shared Sci components.
///
/// ## Domain and special values
/// Requires finite components. Nonpositive real integers are poles and return an error; there is no branch cut.
///
/// ## Returns
/// The complex value with components in the input float type.
///
/// ## Errors
/// [`Error::InvalidInput`] for nonfinite input; [`Error::NumericalFailure`] for
/// a backend domain or numerical failure; [`Error::NonFiniteEvaluation`] for
/// a nonfinite result or conversion overflow.
///
/// ## Numerical behavior
/// Computes in binary64 using the private log-gamma recurrence and reflection kernel, then converts to `T`.
/// Accuracy can worsen near zeros and where exponential growth overflows.
/// Work is bounded by the private algorithm; correct rounding is not guaranteed.
///
/// ## Examples
/// ```
/// let value = sci_special::complex::gamma(sci_complex::Complex { re: 1.0_f64, im: 0.5 })?;
/// assert!(value.re.is_finite() && value.im.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF §5.2](https://dlmf.nist.gov/5.2).
pub fn gamma<T: Float>(z: Complex<T>) -> Result<Complex<T>> {
    output(crate::backend::complex::gamma(input(z)?).map_err(Error::NumericalFailure)?)
}

/// Computes L(z), defined by exp(L(z)) = Γ(z) and real L(x) for x > 0.
///
/// ## Parameters and conventions
/// `z` is dimensionless and uses shared Sci complex components. Computation is in
/// binary64, converted back to `T`.
///
/// ## Domain and special values
/// Requires finite components. The branch cut is the nonpositive real axis. Nonpositive
/// real integers are poles and return an error. Signed imaginary zero selects the upper
/// or lower bank at negative nonintegers. The imaginary part is unwrapped: this is not
/// the principal logarithm of the computed Γ(z).
///
/// ## Returns
/// The complex value with components in the input float type.
///
/// ## Errors
/// [`Error::InvalidInput`] for NaN or infinite inputs; [`Error::NumericalFailure`]
/// for a pole or undefined backend result; [`Error::NonFiniteEvaluation`] for
/// result overflow or conversion overflow. Underflow is allowed.
///
/// ## Numerical behavior
/// Reuses the private gamma kernel: Taylor expansions near 1 and 2, at most eight
/// recurrence steps, reflection and an eight-term Stirling correction. Avoids forming
/// Γ(z), so large finite logarithms can remain representable after Γ overflows.
/// Relative component accuracy can deteriorate near zeros and branch boundaries.
/// Correct rounding is not guaranteed.
///
/// ## Examples
/// ```
/// let value = sci_special::complex::loggamma(sci_complex::Complex { re: 1000.0_f64, im: 0.5 })?;
/// assert!(value.re.is_finite() && value.im.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF §5.2](https://dlmf.nist.gov/5.2).
pub fn loggamma<T: Float>(z: Complex<T>) -> Result<Complex<T>> {
    output(crate::backend::complex::loggamma(input(z)?).map_err(Error::NumericalFailure)?)
}

/// Computes 1/Γ(z), the entire reciprocal gamma function.
///
/// ## Parameters and conventions
/// `z` is dimensionless and uses shared Sci complex components. Computation is in
/// binary64, converted back to `T`.
///
/// ## Domain and special values
/// Requires finite components. All finite complex arguments are accepted. Nonpositive
/// real integers are zeros and return exactly complex zero. There is no branch cut.
///
/// ## Returns
/// The complex value with components in the input float type.
///
/// ## Errors
/// [`Error::InvalidInput`] for NaN or infinite inputs; [`Error::NumericalFailure`]
/// for an undefined backend result; [`Error::NonFiniteEvaluation`] for
/// result overflow or conversion overflow. Underflow is allowed.
///
/// ## Numerical behavior
/// Exponentiates the negative analytic log-gamma rather than dividing by Γ(z),
/// retaining small representable values when Γ overflows. Reuses bounded Taylor,
/// reflection, recurrence and Stirling kernels. Very small results underflow to zero;
/// relative accuracy can deteriorate near zeros and at large arguments. Correct
/// rounding is not guaranteed.
///
/// ## Examples
/// ```
/// let value = sci_special::complex::rgamma(sci_complex::Complex { re: 172.0_f64, im: 0.5 })?;
/// assert!(value.re.is_finite() && value.im.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF §5.2](https://dlmf.nist.gov/5.2).
pub fn rgamma<T: Float>(z: Complex<T>) -> Result<Complex<T>> {
    output(crate::backend::complex::rgamma(input(z)?).map_err(Error::NumericalFailure)?)
}

/// Computes ψ(z) = Γ′(z)/Γ(z) = d log Γ(z)/dz.
///
/// ## Parameters and conventions
/// `z` is dimensionless and uses shared Sci complex components. Computation is in
/// binary64, converted back to `T`.
///
/// ## Domain and special values
/// Requires finite components. Nonpositive real integers are poles and return an error.
/// The function is meromorphic with no branch cut; signed imaginary zero represents its
/// real-axis continuation.
///
/// ## Returns
/// The complex value with components in the input float type.
///
/// ## Errors
/// [`Error::InvalidInput`] for NaN or infinite inputs; [`Error::NumericalFailure`]
/// for a pole or undefined backend result; [`Error::NonFiniteEvaluation`] for
/// result overflow or conversion overflow. Underflow is allowed.
///
/// ## Numerical behavior
/// Reuses the real digamma approximation on the real axis and up to 99 Taylor terms
/// near the two zeros closest to the origin. Elsewhere uses scaled cotangent
/// reflection, at most eight recurrence steps and the derivative of the shared
/// eight-term Stirling correction. Accuracy can deteriorate near other zeros and poles.
/// Correct rounding is not guaranteed.
///
/// ## Examples
/// ```
/// let value = sci_special::complex::digamma(sci_complex::Complex { re: 1.0_f64, im: 0.5 })?;
/// assert!(value.re.is_finite() && value.im.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF §5.5](https://dlmf.nist.gov/5.5).
pub fn digamma<T: Float>(z: Complex<T>) -> Result<Complex<T>> {
    output(crate::backend::complex::digamma(input(z)?).map_err(Error::NumericalFailure)?)
}

/// Computes erf(z) = (2/√π) ∫₀ᶻ exp(−t²) dt.
///
/// ## Parameters and conventions
/// `z` is a dimensionless complex argument using shared Sci components.
///
/// ## Domain and special values
/// Requires finite components. Entire; no branch cuts or finite singularities.
///
/// ## Returns
/// The complex value with components in the input float type.
///
/// ## Errors
/// [`Error::InvalidInput`] for nonfinite input; [`Error::NumericalFailure`] for
/// a backend domain or numerical failure; [`Error::NonFiniteEvaluation`] for
/// a nonfinite result or conversion overflow.
///
/// ## Numerical behavior
/// Computes in binary64 using the private Faddeeva approximation, then converts to `T`.
/// Accuracy can worsen near zeros and where exponential growth overflows.
/// Work is bounded by the private algorithm; correct rounding is not guaranteed.
///
/// ## Examples
/// ```
/// let value = sci_special::complex::erf(sci_complex::Complex { re: 1.0_f64, im: 0.5 })?;
/// assert!(value.re.is_finite() && value.im.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF §7.2](https://dlmf.nist.gov/7.2).
pub fn erf<T: Float>(z: Complex<T>) -> Result<Complex<T>> {
    output(crate::backend::complex::erf(input(z)?).map_err(Error::NumericalFailure)?)
}

/// Computes erfc(z) = 1 − erf(z).
///
/// ## Parameters and conventions
/// `z` is a dimensionless complex argument using shared Sci components.
///
/// ## Domain and special values
/// Requires finite components. Entire; no branch cuts or finite singularities.
///
/// ## Returns
/// The complex value with components in the input float type.
///
/// ## Errors
/// [`Error::InvalidInput`] for nonfinite input; [`Error::NumericalFailure`] for
/// a backend domain or numerical failure; [`Error::NonFiniteEvaluation`] for
/// a nonfinite result or conversion overflow.
///
/// ## Numerical behavior
/// Computes in binary64 using the private Faddeeva approximation, then converts to `T`.
/// Accuracy can worsen near zeros and where exponential growth overflows.
/// Work is bounded by the private algorithm; correct rounding is not guaranteed.
///
/// ## Examples
/// ```
/// let value = sci_special::complex::erfc(sci_complex::Complex { re: 1.0_f64, im: 0.5 })?;
/// assert!(value.re.is_finite() && value.im.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF §7.2](https://dlmf.nist.gov/7.2).
pub fn erfc<T: Float>(z: Complex<T>) -> Result<Complex<T>> {
    output(crate::backend::complex::erfc(input(z)?).map_err(Error::NumericalFailure)?)
}

/// Computes w(z) = exp(−z²) erfc(−iz), the Faddeeva function.
///
/// ## Parameters and conventions
/// `z` is a dimensionless complex argument using shared Sci components.
///
/// ## Domain and special values
/// Requires finite components. Entire; no branch cuts or finite singularities.
///
/// ## Returns
/// The complex value with components in the input float type.
///
/// ## Errors
/// [`Error::InvalidInput`] for nonfinite input; [`Error::NumericalFailure`] for
/// a backend domain or numerical failure; [`Error::NonFiniteEvaluation`] for
/// a nonfinite result or conversion overflow.
///
/// ## Numerical behavior
/// Computes in binary64 using the private Faddeeva approximation, then converts to `T`.
/// Accuracy can worsen near zeros and where exponential growth overflows.
/// Work is bounded by the private algorithm; correct rounding is not guaranteed.
///
/// ## Examples
/// ```
/// let value = sci_special::complex::wofz(sci_complex::Complex { re: 1.0_f64, im: 0.5 })?;
/// assert!(value.re.is_finite() && value.im.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF §7.2](https://dlmf.nist.gov/7.2).
pub fn wofz<T: Float>(z: Complex<T>) -> Result<Complex<T>> {
    output(crate::backend::complex::wofz(input(z)?).map_err(Error::NumericalFailure)?)
}

/// Computes erfcx(z) = exp(z²) erfc(z) = w(iz).
///
/// ## Parameters and conventions
/// `z` is dimensionless and uses Sci's shared complex type. Here
/// `erfc(z) = 1 − (2/√π) ∫₀ᶻ exp(−t²) dt` and `w` is [`wofz`].
///
/// ## Domain and special values
/// Entire, with no branch cuts or finite singularities. Requires finite input
/// components; NaN and infinity return an error. Underflow to zero is allowed.
///
/// ## Returns
/// The complex value with components in the input float type.
///
/// ## Errors
/// [`Error::InvalidInput`] for nonfinite input; [`Error::NumericalFailure`] for
/// an undefined backend result or exhausted series budget;
/// [`Error::NonFiniteEvaluation`] for overflow or conversion overflow.
///
/// ## Numerical behavior
/// Evaluates the Faddeeva function at `iz`, avoiding an overflowing `exp(z²)`
/// multiplied by an underflowing `erfc(z)`. Large positive real arguments
/// therefore retain their algebraically small values.
/// Computes in binary64, then converts to `T`. The underlying Faddeeva kernel
/// uses rational approximations and bounded series (at most 128 terms per sum,
/// or 52 for table-based sums). No user tolerance is exposed. Relative component
/// accuracy can deteriorate near zeros; correct rounding is not guaranteed.
///
/// ## Examples
/// ```
/// let value = sci_special::complex::erfcx(sci_complex::Complex { re: 30.0_f64, im: 0.5 })?;
/// assert!(value.re.is_finite() && value.im.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF §7.2](https://dlmf.nist.gov/7.2).
pub fn erfcx<T: Float>(z: Complex<T>) -> Result<Complex<T>> {
    output(crate::backend::complex::erfcx(input(z)?).map_err(Error::NumericalFailure)?)
}

/// Computes erfi(z) = −i erf(iz) = (2/√π) ∫₀ᶻ exp(t²) dt.
///
/// ## Parameters and conventions
/// `z` is dimensionless and uses Sci's shared complex type. Integrals run from
/// zero to `z`; the integrands are entire, so the path does not affect the value.
///
/// ## Domain and special values
/// Entire, with no branch cuts or finite singularities. Requires finite input
/// components; NaN and infinity return an error. Underflow to zero is allowed.
///
/// ## Returns
/// The complex value with components in the input float type.
///
/// ## Errors
/// [`Error::InvalidInput`] for nonfinite input; [`Error::NumericalFailure`] for
/// an undefined backend result or exhausted series budget;
/// [`Error::NonFiniteEvaluation`] for overflow or conversion overflow.
///
/// ## Numerical behavior
/// Reuses the private error-function kernel with a rotated argument. Small-argument
/// Taylor expansions preserve tiny values; exponential growth can overflow.
/// Computes in binary64, then converts to `T`. The underlying Faddeeva kernel
/// uses rational approximations and bounded series (at most 128 terms per sum,
/// or 52 for table-based sums). No user tolerance is exposed. Relative component
/// accuracy can deteriorate near zeros; correct rounding is not guaranteed.
///
/// ## Examples
/// ```
/// let value = sci_special::complex::erfi(sci_complex::Complex { re: 1.0_f64, im: 0.5 })?;
/// assert!(value.re.is_finite() && value.im.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF §7.2](https://dlmf.nist.gov/7.2).
pub fn erfi<T: Float>(z: Complex<T>) -> Result<Complex<T>> {
    output(crate::backend::complex::erfi(input(z)?).map_err(Error::NumericalFailure)?)
}

/// Computes Dawson’s integral D(z) = exp(−z²) ∫₀ᶻ exp(t²) dt.
///
/// ## Parameters and conventions
/// `z` is dimensionless and uses Sci's shared complex type. Integrals run from
/// zero to `z`; the integrands are entire, so the path does not affect the value.
///
/// ## Domain and special values
/// Entire, with no branch cuts or finite singularities. Requires finite input
/// components; NaN and infinity return an error. Underflow to zero is allowed.
///
/// ## Returns
/// The complex value with components in the input float type.
///
/// ## Errors
/// [`Error::InvalidInput`] for nonfinite input; [`Error::NumericalFailure`] for
/// an undefined backend result or exhausted series budget;
/// [`Error::NonFiniteEvaluation`] for overflow or conversion overflow.
///
/// ## Numerical behavior
/// Reuses the private Dawson kernel with near-origin and near-axis Taylor
/// expansions and Faddeeva identities elsewhere. It avoids forming an overflowing
/// `erfi(z)` on the large real axis. Exponential growth off that axis can overflow.
/// Computes in binary64, then converts to `T`. The underlying Faddeeva kernel
/// uses rational approximations and bounded series (at most 128 terms per sum,
/// or 52 for table-based sums). No user tolerance is exposed. Relative component
/// accuracy can deteriorate near zeros; correct rounding is not guaranteed.
///
/// ## Examples
/// ```
/// let value = sci_special::complex::dawsn(sci_complex::Complex { re: 30.0_f64, im: 0.5 })?;
/// assert!(value.re.is_finite() && value.im.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF §7.2](https://dlmf.nist.gov/7.2).
pub fn dawsn<T: Float>(z: Complex<T>) -> Result<Complex<T>> {
    output(crate::backend::complex::dawsn(input(z)?).map_err(Error::NumericalFailure)?)
}

/// Computes J_v(z), solving z²w″+zw′+(z²−v²)w=0.
///
/// ## Parameters and conventions
/// `v` is the real order and `z` the dimensionless complex argument.
/// The returned normalization is unscaled J_v(z).
///
/// ## Domain and special values
/// Requires finite order and components. The negative real axis is the
/// principal branch cut for noninteger orders; signed imaginary zero selects its side.
/// Singularities at zero return a backend error or a nonfinite-result error.
///
/// ## Returns
/// The complex value in the input float type.
///
/// ## Errors
/// [`Error::InvalidInput`] for nonfinite inputs; [`Error::NumericalFailure`]
/// for AMOS domain, overflow, precision-loss, or convergence failures;
/// [`Error::NonFiniteEvaluation`] for nonfinite results or conversion overflow.
///
/// ## Numerical behavior
/// Reuses the private Rust AMOS backend in binary64, with bounded series and
/// recurrences. Large orders/arguments and points near zeros can lose accuracy;
/// no uniform relative-error or correctly-rounded guarantee is provided.
/// Scaling avoids some exponential overflow but cannot prevent every failure.
///
/// ## Examples
/// ```
/// let value = sci_special::complex::jv(0.5_f64, sci_complex::Complex { re: 1.0, im: 0.5 })?;
/// assert!(value.re.is_finite() && value.im.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF, Bessel functions](https://dlmf.nist.gov/10).
pub fn jv<T: Float>(v: T, z: Complex<T>) -> Result<Complex<T>> {
    bessel(v, z, complex_bessel::besselj, complex_bessel::besselj)
}

/// Computes exp(−|Im z|) J_v(z).
///
/// ## Parameters and conventions
/// `v` is the real order and `z` the dimensionless complex argument.
/// The returned normalization is exp(−|Im z|) times J_v(z).
///
/// ## Domain and special values
/// Requires finite order and components. The negative real axis is the
/// principal branch cut for noninteger orders; signed imaginary zero selects its side.
/// Singularities at zero return a backend error or a nonfinite-result error.
///
/// ## Returns
/// The complex value in the input float type.
///
/// ## Errors
/// [`Error::InvalidInput`] for nonfinite inputs; [`Error::NumericalFailure`]
/// for AMOS domain, overflow, precision-loss, or convergence failures;
/// [`Error::NonFiniteEvaluation`] for nonfinite results or conversion overflow.
///
/// ## Numerical behavior
/// Reuses the private Rust AMOS backend in binary64, with bounded series and
/// recurrences. Large orders/arguments and points near zeros can lose accuracy;
/// no uniform relative-error or correctly-rounded guarantee is provided.
/// Scaling avoids some exponential overflow but cannot prevent every failure.
///
/// ## Examples
/// ```
/// let value = sci_special::complex::jve(0.5_f64, sci_complex::Complex { re: 1.0, im: 0.5 })?;
/// assert!(value.re.is_finite() && value.im.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF, Bessel functions](https://dlmf.nist.gov/10).
pub fn jve<T: Float>(v: T, z: Complex<T>) -> Result<Complex<T>> {
    bessel(
        v,
        z,
        complex_bessel::besselj_scaled,
        complex_bessel::besselj_scaled,
    )
}

/// Computes Y_v(z), the second cylindrical Bessel solution.
///
/// ## Parameters and conventions
/// `v` is the real order and `z` the dimensionless complex argument.
/// The returned normalization is unscaled Y_v(z).
///
/// ## Domain and special values
/// Requires finite order and components. The negative real axis is the
/// principal branch cut for all orders; signed imaginary zero selects its side.
/// Singularities at zero return a backend error or a nonfinite-result error.
///
/// ## Returns
/// The complex value in the input float type.
///
/// ## Errors
/// [`Error::InvalidInput`] for nonfinite inputs; [`Error::NumericalFailure`]
/// for AMOS domain, overflow, precision-loss, or convergence failures;
/// [`Error::NonFiniteEvaluation`] for nonfinite results or conversion overflow.
///
/// ## Numerical behavior
/// Reuses the private Rust AMOS backend in binary64, with bounded series and
/// recurrences. Large orders/arguments and points near zeros can lose accuracy;
/// no uniform relative-error or correctly-rounded guarantee is provided.
/// Scaling avoids some exponential overflow but cannot prevent every failure.
///
/// ## Examples
/// ```
/// let value = sci_special::complex::yv(0.5_f64, sci_complex::Complex { re: 1.0, im: 0.5 })?;
/// assert!(value.re.is_finite() && value.im.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF, Bessel functions](https://dlmf.nist.gov/10).
pub fn yv<T: Float>(v: T, z: Complex<T>) -> Result<Complex<T>> {
    bessel(v, z, complex_bessel::bessely, complex_bessel::bessely)
}

/// Computes exp(−|Im z|) Y_v(z).
///
/// ## Parameters and conventions
/// `v` is the real order and `z` the dimensionless complex argument.
/// The returned normalization is exp(−|Im z|) times Y_v(z).
///
/// ## Domain and special values
/// Requires finite order and components. The negative real axis is the
/// principal branch cut for all orders; signed imaginary zero selects its side.
/// Singularities at zero return a backend error or a nonfinite-result error.
///
/// ## Returns
/// The complex value in the input float type.
///
/// ## Errors
/// [`Error::InvalidInput`] for nonfinite inputs; [`Error::NumericalFailure`]
/// for AMOS domain, overflow, precision-loss, or convergence failures;
/// [`Error::NonFiniteEvaluation`] for nonfinite results or conversion overflow.
///
/// ## Numerical behavior
/// Reuses the private Rust AMOS backend in binary64, with bounded series and
/// recurrences. Large orders/arguments and points near zeros can lose accuracy;
/// no uniform relative-error or correctly-rounded guarantee is provided.
/// Scaling avoids some exponential overflow but cannot prevent every failure.
///
/// ## Examples
/// ```
/// let value = sci_special::complex::yve(0.5_f64, sci_complex::Complex { re: 1.0, im: 0.5 })?;
/// assert!(value.re.is_finite() && value.im.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF, Bessel functions](https://dlmf.nist.gov/10).
pub fn yve<T: Float>(v: T, z: Complex<T>) -> Result<Complex<T>> {
    bessel(
        v,
        z,
        complex_bessel::bessely_scaled,
        complex_bessel::bessely_scaled,
    )
}

/// Computes I_v(z), solving z²w″+zw′−(z²+v²)w=0.
///
/// ## Parameters and conventions
/// `v` is the real order and `z` the dimensionless complex argument.
/// The returned normalization is unscaled I_v(z).
///
/// ## Domain and special values
/// Requires finite order and components. The negative real axis is the
/// principal branch cut for noninteger orders; signed imaginary zero selects its side.
/// Singularities at zero return a backend error or a nonfinite-result error.
///
/// ## Returns
/// The complex value in the input float type.
///
/// ## Errors
/// [`Error::InvalidInput`] for nonfinite inputs; [`Error::NumericalFailure`]
/// for AMOS domain, overflow, precision-loss, or convergence failures;
/// [`Error::NonFiniteEvaluation`] for nonfinite results or conversion overflow.
///
/// ## Numerical behavior
/// Reuses the private Rust AMOS backend in binary64, with bounded series and
/// recurrences. Large orders/arguments and points near zeros can lose accuracy;
/// no uniform relative-error or correctly-rounded guarantee is provided.
/// Scaling avoids some exponential overflow but cannot prevent every failure.
///
/// ## Examples
/// ```
/// let value = sci_special::complex::iv(0.5_f64, sci_complex::Complex { re: 1.0, im: 0.5 })?;
/// assert!(value.re.is_finite() && value.im.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF, Bessel functions](https://dlmf.nist.gov/10).
pub fn iv<T: Float>(v: T, z: Complex<T>) -> Result<Complex<T>> {
    bessel(v, z, complex_bessel::besseli, complex_bessel::besseli)
}

/// Computes exp(−|Re z|) I_v(z).
///
/// ## Parameters and conventions
/// `v` is the real order and `z` the dimensionless complex argument.
/// The returned normalization is exp(−|Re z|) times I_v(z).
///
/// ## Domain and special values
/// Requires finite order and components. The negative real axis is the
/// principal branch cut for noninteger orders; signed imaginary zero selects its side.
/// Singularities at zero return a backend error or a nonfinite-result error.
///
/// ## Returns
/// The complex value in the input float type.
///
/// ## Errors
/// [`Error::InvalidInput`] for nonfinite inputs; [`Error::NumericalFailure`]
/// for AMOS domain, overflow, precision-loss, or convergence failures;
/// [`Error::NonFiniteEvaluation`] for nonfinite results or conversion overflow.
///
/// ## Numerical behavior
/// Reuses the private Rust AMOS backend in binary64, with bounded series and
/// recurrences. Large orders/arguments and points near zeros can lose accuracy;
/// no uniform relative-error or correctly-rounded guarantee is provided.
/// Scaling avoids some exponential overflow but cannot prevent every failure.
///
/// ## Examples
/// ```
/// let value = sci_special::complex::ive(0.5_f64, sci_complex::Complex { re: 1.0, im: 0.5 })?;
/// assert!(value.re.is_finite() && value.im.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF, Bessel functions](https://dlmf.nist.gov/10).
pub fn ive<T: Float>(v: T, z: Complex<T>) -> Result<Complex<T>> {
    bessel(
        v,
        z,
        complex_bessel::besseli_scaled,
        complex_bessel::besseli_scaled,
    )
}

/// Computes K_v(z), the decaying modified Bessel solution.
///
/// ## Parameters and conventions
/// `v` is the real order and `z` the dimensionless complex argument.
/// The returned normalization is unscaled K_v(z).
///
/// ## Domain and special values
/// Requires finite order and components. The negative real axis is the
/// principal branch cut for all orders; signed imaginary zero selects its side.
/// Singularities at zero return a backend error or a nonfinite-result error.
///
/// ## Returns
/// The complex value in the input float type.
///
/// ## Errors
/// [`Error::InvalidInput`] for nonfinite inputs; [`Error::NumericalFailure`]
/// for AMOS domain, overflow, precision-loss, or convergence failures;
/// [`Error::NonFiniteEvaluation`] for nonfinite results or conversion overflow.
///
/// ## Numerical behavior
/// Reuses the private Rust AMOS backend in binary64, with bounded series and
/// recurrences. Large orders/arguments and points near zeros can lose accuracy;
/// no uniform relative-error or correctly-rounded guarantee is provided.
/// Scaling avoids some exponential overflow but cannot prevent every failure.
///
/// ## Examples
/// ```
/// let value = sci_special::complex::kv(0.5_f64, sci_complex::Complex { re: 1.0, im: 0.5 })?;
/// assert!(value.re.is_finite() && value.im.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF, Bessel functions](https://dlmf.nist.gov/10).
pub fn kv<T: Float>(v: T, z: Complex<T>) -> Result<Complex<T>> {
    bessel(v, z, complex_bessel::besselk, complex_bessel::besselk)
}

/// Computes exp(z) K_v(z).
///
/// ## Parameters and conventions
/// `v` is the real order and `z` the dimensionless complex argument.
/// The returned normalization is exp(z) times K_v(z), including its complex phase.
///
/// ## Domain and special values
/// Requires finite order and components. The negative real axis is the
/// principal branch cut for all orders; signed imaginary zero selects its side.
/// Singularities at zero return a backend error or a nonfinite-result error.
///
/// ## Returns
/// The complex value in the input float type.
///
/// ## Errors
/// [`Error::InvalidInput`] for nonfinite inputs; [`Error::NumericalFailure`]
/// for AMOS domain, overflow, precision-loss, or convergence failures;
/// [`Error::NonFiniteEvaluation`] for nonfinite results or conversion overflow.
///
/// ## Numerical behavior
/// Reuses the private Rust AMOS backend in binary64, with bounded series and
/// recurrences. Large orders/arguments and points near zeros can lose accuracy;
/// no uniform relative-error or correctly-rounded guarantee is provided.
/// Scaling avoids some exponential overflow but cannot prevent every failure.
///
/// ## Examples
/// ```
/// let value = sci_special::complex::kve(0.5_f64, sci_complex::Complex { re: 1.0, im: 0.5 })?;
/// assert!(value.re.is_finite() && value.im.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF, Bessel functions](https://dlmf.nist.gov/10).
pub fn kve<T: Float>(v: T, z: Complex<T>) -> Result<Complex<T>> {
    bessel(
        v,
        z,
        complex_bessel::besselk_scaled,
        complex_bessel::besselk_scaled,
    )
}

/// Computes H_v⁽¹⁾(z) = J_v(z) + i Y_v(z).
///
/// ## Parameters and conventions
/// `v` is the real order and `z` the dimensionless complex argument.
/// The returned normalization is unscaled H_v⁽¹⁾(z).
///
/// ## Domain and special values
/// Requires finite order and components. The negative real axis is the
/// principal branch cut for all orders; signed imaginary zero selects its side.
/// Singularities at zero return a backend error or a nonfinite-result error.
///
/// ## Returns
/// The complex value in the input float type.
///
/// ## Errors
/// [`Error::InvalidInput`] for nonfinite inputs; [`Error::NumericalFailure`]
/// for AMOS domain, overflow, precision-loss, or convergence failures;
/// [`Error::NonFiniteEvaluation`] for nonfinite results or conversion overflow.
///
/// ## Numerical behavior
/// Reuses the private Rust AMOS backend in binary64, with bounded series and
/// recurrences. Large orders/arguments and points near zeros can lose accuracy;
/// no uniform relative-error or correctly-rounded guarantee is provided.
/// Scaling avoids some exponential overflow but cannot prevent every failure.
///
/// ## Examples
/// ```
/// let value = sci_special::complex::hankel1(0.5_f64, sci_complex::Complex { re: 1.0, im: 0.5 })?;
/// assert!(value.re.is_finite() && value.im.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF, Bessel functions](https://dlmf.nist.gov/10).
pub fn hankel1<T: Float>(v: T, z: Complex<T>) -> Result<Complex<T>> {
    bessel(v, z, complex_bessel::hankel1, complex_bessel::hankel2)
}

/// Computes exp(−iz) H_v⁽¹⁾(z).
///
/// ## Parameters and conventions
/// `v` is the real order and `z` the dimensionless complex argument.
/// The returned normalization is exp(−iz) times H_v⁽¹⁾(z).
///
/// ## Domain and special values
/// Requires finite order and components. The negative real axis is the
/// principal branch cut for all orders; signed imaginary zero selects its side.
/// Singularities at zero return a backend error or a nonfinite-result error.
///
/// ## Returns
/// The complex value in the input float type.
///
/// ## Errors
/// [`Error::InvalidInput`] for nonfinite inputs; [`Error::NumericalFailure`]
/// for AMOS domain, overflow, precision-loss, or convergence failures;
/// [`Error::NonFiniteEvaluation`] for nonfinite results or conversion overflow.
///
/// ## Numerical behavior
/// Reuses the private Rust AMOS backend in binary64, with bounded series and
/// recurrences. Large orders/arguments and points near zeros can lose accuracy;
/// no uniform relative-error or correctly-rounded guarantee is provided.
/// Scaling avoids some exponential overflow but cannot prevent every failure.
///
/// ## Examples
/// ```
/// let value = sci_special::complex::hankel1e(0.5_f64, sci_complex::Complex { re: 1.0, im: 0.5 })?;
/// assert!(value.re.is_finite() && value.im.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF, Bessel functions](https://dlmf.nist.gov/10).
pub fn hankel1e<T: Float>(v: T, z: Complex<T>) -> Result<Complex<T>> {
    bessel(
        v,
        z,
        complex_bessel::hankel1_scaled,
        complex_bessel::hankel2_scaled,
    )
}

/// Computes H_v⁽²⁾(z) = J_v(z) − i Y_v(z).
///
/// ## Parameters and conventions
/// `v` is the real order and `z` the dimensionless complex argument.
/// The returned normalization is unscaled H_v⁽²⁾(z).
///
/// ## Domain and special values
/// Requires finite order and components. The negative real axis is the
/// principal branch cut for all orders; signed imaginary zero selects its side.
/// Singularities at zero return a backend error or a nonfinite-result error.
///
/// ## Returns
/// The complex value in the input float type.
///
/// ## Errors
/// [`Error::InvalidInput`] for nonfinite inputs; [`Error::NumericalFailure`]
/// for AMOS domain, overflow, precision-loss, or convergence failures;
/// [`Error::NonFiniteEvaluation`] for nonfinite results or conversion overflow.
///
/// ## Numerical behavior
/// Reuses the private Rust AMOS backend in binary64, with bounded series and
/// recurrences. Large orders/arguments and points near zeros can lose accuracy;
/// no uniform relative-error or correctly-rounded guarantee is provided.
/// Scaling avoids some exponential overflow but cannot prevent every failure.
///
/// ## Examples
/// ```
/// let value = sci_special::complex::hankel2(0.5_f64, sci_complex::Complex { re: 1.0, im: 0.5 })?;
/// assert!(value.re.is_finite() && value.im.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF, Bessel functions](https://dlmf.nist.gov/10).
pub fn hankel2<T: Float>(v: T, z: Complex<T>) -> Result<Complex<T>> {
    bessel(v, z, complex_bessel::hankel2, complex_bessel::hankel1)
}

/// Computes exp(iz) H_v⁽²⁾(z).
///
/// ## Parameters and conventions
/// `v` is the real order and `z` the dimensionless complex argument.
/// The returned normalization is exp(iz) times H_v⁽²⁾(z).
///
/// ## Domain and special values
/// Requires finite order and components. The negative real axis is the
/// principal branch cut for all orders; signed imaginary zero selects its side.
/// Singularities at zero return a backend error or a nonfinite-result error.
///
/// ## Returns
/// The complex value in the input float type.
///
/// ## Errors
/// [`Error::InvalidInput`] for nonfinite inputs; [`Error::NumericalFailure`]
/// for AMOS domain, overflow, precision-loss, or convergence failures;
/// [`Error::NonFiniteEvaluation`] for nonfinite results or conversion overflow.
///
/// ## Numerical behavior
/// Reuses the private Rust AMOS backend in binary64, with bounded series and
/// recurrences. Large orders/arguments and points near zeros can lose accuracy;
/// no uniform relative-error or correctly-rounded guarantee is provided.
/// Scaling avoids some exponential overflow but cannot prevent every failure.
///
/// ## Examples
/// ```
/// let value = sci_special::complex::hankel2e(0.5_f64, sci_complex::Complex { re: 1.0, im: 0.5 })?;
/// assert!(value.re.is_finite() && value.im.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF, Bessel functions](https://dlmf.nist.gov/10).
pub fn hankel2e<T: Float>(v: T, z: Complex<T>) -> Result<Complex<T>> {
    bessel(
        v,
        z,
        complex_bessel::hankel2_scaled,
        complex_bessel::hankel1_scaled,
    )
}

/// Computes complex Airy functions and derivatives.
///
/// Returns `[Ai(z), Ai′(z), Bi(z), Bi′(z)]`, the two standard solutions of
/// `w″ − zw = 0` and their derivatives with respect to `z`.
///
/// ## Parameters and conventions
/// `z` is dimensionless; output ordering is value, derivative for Ai, then Bi.
///
/// ## Domain and special values
/// Requires finite components. The unscaled functions are entire.
/// Scaling uses the principal square root, with signed imaginary zero selecting
/// the side of the negative real axis.
///
/// ## Returns
/// Four shared complex values in the input precision, in the ordering above.
///
/// ## Errors
/// [`Error::InvalidInput`] for nonfinite input; [`Error::NumericalFailure`]
/// for AMOS overflow, precision-loss, or convergence failure;
/// [`Error::NonFiniteEvaluation`] for nonfinite results or conversion overflow.
///
/// ## Numerical behavior
/// Uses bounded AMOS series/continuations in binary64 and converts to `T`.
/// Accuracy is parameter dependent, especially near zeros and for large `z`.
/// Scaling reduces exponential overflow without guaranteeing finite results.
///
/// ## Examples
/// ```
/// let values = sci_special::complex::airy(sci_complex::Complex { re: 1.0_f64, im: 0.5 })?;
/// assert!(values.iter().all(|z| z.re.is_finite() && z.im.is_finite()));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF §9.2](https://dlmf.nist.gov/9.2).
pub fn airy<T: Float>(z: Complex<T>) -> Result<[Complex<T>; 4]> {
    let z = input(z)?;
    Ok([
        output(
            complex_bessel::airy(z)
                .map_err(|e| Error::NumericalFailure(crate::backend::bessel::backend_error(e)))?,
        )?,
        output(
            complex_bessel::airyprime(z)
                .map_err(|e| Error::NumericalFailure(crate::backend::bessel::backend_error(e)))?,
        )?,
        output(
            complex_bessel::biry(z)
                .map_err(|e| Error::NumericalFailure(crate::backend::bessel::backend_error(e)))?,
        )?,
        output(
            complex_bessel::biryprime(z)
                .map_err(|e| Error::NumericalFailure(crate::backend::bessel::backend_error(e)))?,
        )?,
    ])
}

/// Computes exponentially scaled complex Airy functions and derivatives.
///
/// Returns `[Ai(z), Ai′(z), Bi(z), Bi′(z)]`, the two standard solutions of
/// `w″ − zw = 0` and their derivatives with respect to `z`.
/// Ai and Ai′ are multiplied by `exp(ζ)`; Bi and Bi′ by `exp(−|Re ζ|)`,
/// with `ζ = (2/3) z sqrt(z)` using the principal square root. These are scaled
/// derivatives of the original functions, not derivatives of the scaled values.
///
/// ## Parameters and conventions
/// `z` is dimensionless; output ordering is value, derivative for Ai, then Bi.
///
/// ## Domain and special values
/// Requires finite components. The unscaled functions are entire.
/// Scaling uses the principal square root, with signed imaginary zero selecting
/// the side of the negative real axis.
///
/// ## Returns
/// Four shared complex values in the input precision, in the ordering above.
///
/// ## Errors
/// [`Error::InvalidInput`] for nonfinite input; [`Error::NumericalFailure`]
/// for AMOS overflow, precision-loss, or convergence failure;
/// [`Error::NonFiniteEvaluation`] for nonfinite results or conversion overflow.
///
/// ## Numerical behavior
/// Uses bounded AMOS series/continuations in binary64 and converts to `T`.
/// Accuracy is parameter dependent, especially near zeros and for large `z`.
/// Scaling reduces exponential overflow without guaranteeing finite results.
///
/// ## Examples
/// ```
/// let values = sci_special::complex::airye(sci_complex::Complex { re: 1.0_f64, im: 0.5 })?;
/// assert!(values.iter().all(|z| z.re.is_finite() && z.im.is_finite()));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF §9.2](https://dlmf.nist.gov/9.2).
pub fn airye<T: Float>(z: Complex<T>) -> Result<[Complex<T>; 4]> {
    let z = input(z)?;
    Ok([
        output(
            complex_bessel::airy_scaled(z)
                .map_err(|e| Error::NumericalFailure(crate::backend::bessel::backend_error(e)))?,
        )?,
        output(
            complex_bessel::airyprime_scaled(z)
                .map_err(|e| Error::NumericalFailure(crate::backend::bessel::backend_error(e)))?,
        )?,
        output(
            complex_bessel::biry_scaled(z)
                .map_err(|e| Error::NumericalFailure(crate::backend::bessel::backend_error(e)))?,
        )?,
        output(
            complex_bessel::biryprime_scaled(z)
                .map_err(|e| Error::NumericalFailure(crate::backend::bessel::backend_error(e)))?,
        )?,
    ])
}

/// Computes the complex Fresnel integrals `S(z)=∫₀ᶻ sin(πt²/2)dt` and
/// `C(z)=∫₀ᶻ cos(πt²/2)dt`.
///
/// ## Parameters and conventions
/// `z` is dimensionless; the quadratic phase uses `π/2`, with no additional
/// rescaling. The ordered result is `(S, C)`.
///
/// ## Domain and special values
/// Both integrals are entire; finite input components are required.
///
/// ## Returns
/// The pair `(S(z), C(z))` with shared complex values in the input precision.
///
/// ## Errors
/// [`Error::InvalidInput`] for nonfinite inputs; [`Error::NumericalFailure`]
/// for backend failures; [`Error::NonFiniteEvaluation`] for nonfinite results.
///
/// ## Numerical behavior
/// Computes in binary64 using bounded defining series near zero and Faddeeva
/// identities elsewhere. Near-axis expansions preserve small components.
/// Cancellation near zeros and exponential overflow can limit accuracy.
///
/// ## Examples
/// ```
/// let (s, c) = sci_special::complex::fresnel(sci_complex::Complex { re: 1.0_f64, im: 0.5 })?;
/// assert!(s.re.is_finite() && c.im.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF §7.2](https://dlmf.nist.gov/7.2).
pub fn fresnel<T: Float>(z: Complex<T>) -> Result<(Complex<T>, Complex<T>)> {
    let (s, c) = crate::backend::complex::fresnel(input(z)?).map_err(Error::NumericalFailure)?;
    Ok((output(s)?, output(c)?))
}

type BesselKernel = fn(
    f64,
    num_complex::Complex64,
) -> core::result::Result<num_complex::Complex64, complex_bessel::Error>;
fn bessel<T: Float>(
    v: T,
    z: Complex<T>,
    kernel: BesselKernel,
    reflected: BesselKernel,
) -> Result<Complex<T>> {
    if !v.is_finite() {
        return Err(Error::InvalidInput("Bessel order must be finite"));
    }
    let mut z = input(z)?;
    // AMOS treats either zero sign as the upper bank. Schwarz reflection
    // restores the lower bank, interchanging Hankel kinds where necessary.
    let lower = z.re < 0.0 && z.im == 0.0 && z.im.is_sign_negative();
    let kernel = if lower {
        z.im = 0.0;
        reflected
    } else {
        kernel
    };
    let v = v
        .to_f64()
        .filter(|v| v.is_finite())
        .ok_or(Error::NonFiniteEvaluation)?;
    let value = kernel(v, z)
        .map_err(|e| Error::NumericalFailure(crate::backend::bessel::backend_error(e)))?;
    output(if lower { value.conj() } else { value })
}
