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
