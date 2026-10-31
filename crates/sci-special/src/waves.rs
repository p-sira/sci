//! Private wave-function and polynomial evaluation adapters.

/// Evaluates the classical polynomial `P_n^(alpha,beta)(x)`.
///
/// P_n^(α,β)(x) = (−1)ⁿ/(2ⁿ n!) (1−x)⁻α (1+x)⁻β dⁿ/dxⁿ [(1−x)^(n+α)(1+x)^(n+β)].
///
/// ## Parameters and conventions
///
/// `n` is the integer degree and `x` is a dimensionless real evaluation point.
/// `alpha` and `beta` are real shape parameters greater than `−1` in the orthogonality domain. The polynomial continuation also evaluates other parameters when its recurrence is defined.
/// The normalization is the classical one in the definition above; no unit-norm
/// rescaling is applied.
///
/// ## Domain and special values
///
/// Supported degrees are `0 <= n <= 4096`; other degrees return NaN before
/// recurrence work. NaN arguments return NaN. Finite real `x` need not lie in the
/// orthogonality interval. Overflow may return signed infinity; polynomial
/// cancellation and indeterminate parameter combinations can produce NaN.
///
/// ## Returns
///
/// The scalar polynomial value converted to the requested `sci_core::Float` type.
///
/// ## Numerical behavior
///
/// Uses a binary64 recurrence with work proportional to the degree and bounded by
/// 4096 steps. Conversion to `T` does not add precision. Relative accuracy can
/// worsen near roots and for large degrees or parameters. Integer-degree evaluation
/// is supported; noninteger polynomial degrees are not part of this API.
///
/// ## Mathematical reference
///
/// [NIST DLMF, classical orthogonal polynomials](https://dlmf.nist.gov/18).
///
/// ## Examples
///
/// ```
/// let value = sci_special::eval_jacobi(3,0.5_f64,0.3,0.2);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy eval_jacobi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_jacobi.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn eval_jacobi<T: sci_core::Float>(n: i32, alpha: T, beta: T, x: T) -> T {
    if !(0..=4096).contains(&n) {
        return T::nan();
    }
    crate::backend::eval_jacobi(n, alpha, beta, x).unwrap_or(T::nan())
}

/// Evaluates the classical polynomial `P_n(x)`.
///
/// P_n(x) = (1/(2ⁿ n!)) dⁿ/dxⁿ (x²−1)ⁿ, with P_n(1) = 1.
///
/// ## Parameters and conventions
///
/// `n` is the integer degree and `x` is a dimensionless real evaluation point.
///
/// The normalization is the classical one in the definition above; no unit-norm
/// rescaling is applied.
///
/// ## Domain and special values
///
/// Supported degrees are `0 <= n <= 4096`; other degrees return NaN before
/// recurrence work. NaN arguments return NaN. Finite real `x` need not lie in the
/// orthogonality interval. Overflow may return signed infinity; polynomial
/// cancellation and indeterminate parameter combinations can produce NaN.
///
/// ## Returns
///
/// The scalar polynomial value converted to the requested `sci_core::Float` type.
///
/// ## Numerical behavior
///
/// Uses a binary64 recurrence with work proportional to the degree and bounded by
/// 4096 steps. Conversion to `T` does not add precision. Relative accuracy can
/// worsen near roots and for large degrees or parameters. Integer-degree evaluation
/// is supported; noninteger polynomial degrees are not part of this API.
///
/// ## Mathematical reference
///
/// [NIST DLMF, classical orthogonal polynomials](https://dlmf.nist.gov/18).
///
/// ## Examples
///
/// ```
/// let value = sci_special::eval_legendre(3,0.2_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy eval_legendre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_legendre.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn eval_legendre<T: sci_core::Float>(n: i32, x: T) -> T {
    if !(0..=4096).contains(&n) {
        return T::nan();
    }
    crate::backend::eval_legendre(n, x).unwrap_or(T::nan())
}

/// Evaluates the classical polynomial `C_n^alpha(x)`.
///
/// (1−2xt+t²)^(−alpha) = Σₙ C_n^alpha(x)tⁿ.
///
/// ## Parameters and conventions
///
/// `n` is the integer degree and `x` is a dimensionless real evaluation point.
/// `alpha` is the real Gegenbauer parameter; orthogonality requires `alpha > −1/2`. The recurrence also evaluates other finite parameters where defined.
/// The normalization is the classical one in the definition above; no unit-norm
/// rescaling is applied.
///
/// ## Domain and special values
///
/// Supported degrees are `0 <= n <= 4096`; other degrees return NaN before
/// recurrence work. NaN arguments return NaN. Finite real `x` need not lie in the
/// orthogonality interval. Overflow may return signed infinity; polynomial
/// cancellation and indeterminate parameter combinations can produce NaN.
///
/// ## Returns
///
/// The scalar polynomial value converted to the requested `sci_core::Float` type.
///
/// ## Numerical behavior
///
/// Uses a binary64 recurrence with work proportional to the degree and bounded by
/// 4096 steps. Conversion to `T` does not add precision. Relative accuracy can
/// worsen near roots and for large degrees or parameters. Integer-degree evaluation
/// is supported; noninteger polynomial degrees are not part of this API.
///
/// ## Mathematical reference
///
/// [NIST DLMF, classical orthogonal polynomials](https://dlmf.nist.gov/18).
///
/// ## Examples
///
/// ```
/// let value = sci_special::eval_gegenbauer(3,0.5_f64,0.2);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy eval_gegenbauer](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_gegenbauer.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn eval_gegenbauer<T: sci_core::Float>(n: i32, alpha: T, x: T) -> T {
    if !(0..=4096).contains(&n) {
        return T::nan();
    }
    crate::backend::eval_gegenbauer(n, alpha, x).unwrap_or(T::nan())
}

/// Evaluates the classical polynomial `L_n^alpha(x)`.
///
/// L_n^alpha(x) = x^(−alpha) exp(x)/n! dⁿ/dxⁿ [exp(−x)x^(n+alpha)].
///
/// ## Parameters and conventions
///
/// `n` is the integer degree and `x` is a dimensionless real evaluation point.
/// `alpha` is real; the polynomial orthogonality domain is `alpha > −1`. Evaluation uses polynomial continuation rather than powers in the defining formula.
/// The normalization is the classical one in the definition above; no unit-norm
/// rescaling is applied.
///
/// ## Domain and special values
///
/// Supported degrees are `0 <= n <= 4096`; other degrees return NaN before
/// recurrence work. NaN arguments return NaN. Finite real `x` need not lie in the
/// orthogonality interval. Overflow may return signed infinity; polynomial
/// cancellation and indeterminate parameter combinations can produce NaN.
///
/// ## Returns
///
/// The scalar polynomial value converted to the requested `sci_core::Float` type.
///
/// ## Numerical behavior
///
/// Uses a binary64 recurrence with work proportional to the degree and bounded by
/// 4096 steps. Conversion to `T` does not add precision. Relative accuracy can
/// worsen near roots and for large degrees or parameters. Integer-degree evaluation
/// is supported; noninteger polynomial degrees are not part of this API.
///
/// ## Mathematical reference
///
/// [NIST DLMF, classical orthogonal polynomials](https://dlmf.nist.gov/18).
///
/// ## Examples
///
/// ```
/// let value = sci_special::eval_genlaguerre(3,0.5_f64,0.2);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy eval_genlaguerre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_genlaguerre.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn eval_genlaguerre<T: sci_core::Float>(n: i32, alpha: T, x: T) -> T {
    if !(0..=4096).contains(&n) {
        return T::nan();
    }
    crate::backend::eval_genlaguerre(n, alpha, x).unwrap_or(T::nan())
}

/// Evaluates the classical polynomial `L_n(x)`.
///
/// L_n(x) = exp(x)/n! dⁿ/dxⁿ [exp(−x)xⁿ] = L_n^0(x).
///
/// ## Parameters and conventions
///
/// `n` is the integer degree and `x` is a dimensionless real evaluation point.
///
/// The normalization is the classical one in the definition above; no unit-norm
/// rescaling is applied.
///
/// ## Domain and special values
///
/// Supported degrees are `0 <= n <= 4096`; other degrees return NaN before
/// recurrence work. NaN arguments return NaN. Finite real `x` need not lie in the
/// orthogonality interval. Overflow may return signed infinity; polynomial
/// cancellation and indeterminate parameter combinations can produce NaN.
///
/// ## Returns
///
/// The scalar polynomial value converted to the requested `sci_core::Float` type.
///
/// ## Numerical behavior
///
/// Uses a binary64 recurrence with work proportional to the degree and bounded by
/// 4096 steps. Conversion to `T` does not add precision. Relative accuracy can
/// worsen near roots and for large degrees or parameters. Integer-degree evaluation
/// is supported; noninteger polynomial degrees are not part of this API.
///
/// ## Mathematical reference
///
/// [NIST DLMF, classical orthogonal polynomials](https://dlmf.nist.gov/18).
///
/// ## Examples
///
/// ```
/// let value = sci_special::eval_laguerre(3,0.2_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy eval_laguerre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_laguerre.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn eval_laguerre<T: sci_core::Float>(n: i32, x: T) -> T {
    if !(0..=4096).contains(&n) {
        return T::nan();
    }
    crate::backend::eval_laguerre(n, x).unwrap_or(T::nan())
}

/// Evaluates the classical polynomial `T_n(x)`.
///
/// T_n(cos(theta)) = cos(n*theta), with T_0 = 1 and T_1 = x.
///
/// ## Parameters and conventions
///
/// `n` is the integer degree and `x` is a dimensionless real evaluation point.
///
/// The normalization is the classical one in the definition above; no unit-norm
/// rescaling is applied.
///
/// ## Domain and special values
///
/// Supported degrees are `0 <= n <= 4096`; other degrees return NaN before
/// recurrence work. NaN arguments return NaN. Finite real `x` need not lie in the
/// orthogonality interval. Overflow may return signed infinity; polynomial
/// cancellation and indeterminate parameter combinations can produce NaN.
///
/// ## Returns
///
/// The scalar polynomial value converted to the requested `sci_core::Float` type.
///
/// ## Numerical behavior
///
/// Uses binary64 arithmetic: a factorized degree-ten polynomial, a short
/// recurrence for other small degrees, and binary doubling inside (-1,1)
/// for degrees at least 64. Work is bounded by 4096 steps. Conversion to
/// `T` does not add precision. Relative accuracy can worsen near roots and
/// for large degrees or parameters. Only integer degrees are supported.
///
/// ## Mathematical reference
///
/// [NIST DLMF, classical orthogonal polynomials](https://dlmf.nist.gov/18).
///
/// ## Examples
///
/// ```
/// let value = sci_special::eval_chebyt(3,0.2_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy eval_chebyt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_chebyt.html).
#[inline]
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn eval_chebyt<T: sci_core::Float>(n: i32, x: T) -> T {
    if !(0..=4096).contains(&n) {
        return T::nan();
    }
    crate::backend::eval_chebyt(n, x).unwrap_or(T::nan())
}

/// Evaluates the classical polynomial `U_n(x)`.
///
/// U_n(cos(theta)) = sin((n+1)*theta)/sin(theta), continued at the endpoints; U_n(1) = n+1.
///
/// ## Parameters and conventions
///
/// `n` is the integer degree and `x` is a dimensionless real evaluation point.
///
/// The normalization is the classical one in the definition above; no unit-norm
/// rescaling is applied.
///
/// ## Domain and special values
///
/// Supported degrees are `0 <= n <= 4096`; other degrees return NaN before
/// recurrence work. NaN arguments return NaN. Finite real `x` need not lie in the
/// orthogonality interval. Overflow may return signed infinity; polynomial
/// cancellation and indeterminate parameter combinations can produce NaN.
///
/// ## Returns
///
/// The scalar polynomial value converted to the requested `sci_core::Float` type.
///
/// ## Numerical behavior
///
/// Uses binary64 arithmetic: a factorized degree-ten polynomial, a short
/// recurrence for other small degrees, and binary doubling inside (-1,1)
/// for degrees at least 64. Work is bounded by 4096 steps. Conversion to
/// `T` does not add precision. Relative accuracy can worsen near roots and
/// for large degrees or parameters. Only integer degrees are supported.
///
/// ## Mathematical reference
///
/// [NIST DLMF, classical orthogonal polynomials](https://dlmf.nist.gov/18).
///
/// ## Examples
///
/// ```
/// let value = sci_special::eval_chebyu(3,0.2_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy eval_chebyu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_chebyu.html).
#[inline]
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn eval_chebyu<T: sci_core::Float>(n: i32, x: T) -> T {
    if !(0..=4096).contains(&n) {
        return T::nan();
    }
    crate::backend::eval_chebyu(n, x).unwrap_or(T::nan())
}

/// Evaluates the classical polynomial `H_n(x)`.
///
/// H_n(x) = (−1)ⁿ exp(x²) dⁿ/dxⁿ exp(−x²), with leading coefficient 2ⁿ.
///
/// ## Parameters and conventions
///
/// `n` is the integer degree and `x` is a dimensionless real evaluation point.
///
/// The normalization is the classical one in the definition above; no unit-norm
/// rescaling is applied.
///
/// ## Domain and special values
///
/// Supported degrees are `0 <= n <= 4096`; other degrees return NaN before
/// recurrence work. NaN arguments return NaN. Finite real `x` need not lie in the
/// orthogonality interval. Overflow may return signed infinity; polynomial
/// cancellation and indeterminate parameter combinations can produce NaN.
///
/// ## Returns
///
/// The scalar polynomial value converted to the requested `sci_core::Float` type.
///
/// ## Numerical behavior
///
/// Uses a binary64 recurrence with work proportional to the degree and bounded by
/// 4096 steps. Conversion to `T` does not add precision. Relative accuracy can
/// worsen near roots and for large degrees or parameters. Integer-degree evaluation
/// is supported; noninteger polynomial degrees are not part of this API.
///
/// ## Mathematical reference
///
/// [NIST DLMF, classical orthogonal polynomials](https://dlmf.nist.gov/18).
///
/// ## Examples
///
/// ```
/// let value = sci_special::eval_hermite(3,0.2_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy eval_hermite](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_hermite.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn eval_hermite<T: sci_core::Float>(n: i32, x: T) -> T {
    if !(0..=4096).contains(&n) {
        return T::nan();
    }
    crate::backend::eval_hermite(n, x).unwrap_or(T::nan())
}

/// Evaluates the classical polynomial `He_n(x)`.
///
/// He_n(x) = (−1)ⁿ exp(x²/2) dⁿ/dxⁿ exp(−x²/2), with leading coefficient one.
///
/// ## Parameters and conventions
///
/// `n` is the integer degree and `x` is a dimensionless real evaluation point.
///
/// The normalization is the classical one in the definition above; no unit-norm
/// rescaling is applied.
///
/// ## Domain and special values
///
/// Supported degrees are `0 <= n <= 4096`; other degrees return NaN before
/// recurrence work. NaN arguments return NaN. Finite real `x` need not lie in the
/// orthogonality interval. Overflow may return signed infinity; polynomial
/// cancellation and indeterminate parameter combinations can produce NaN.
///
/// ## Returns
///
/// The scalar polynomial value converted to the requested `sci_core::Float` type.
///
/// ## Numerical behavior
///
/// Uses a binary64 recurrence with work proportional to the degree and bounded by
/// 4096 steps. Conversion to `T` does not add precision. Relative accuracy can
/// worsen near roots and for large degrees or parameters. Integer-degree evaluation
/// is supported; noninteger polynomial degrees are not part of this API.
///
/// ## Mathematical reference
///
/// [NIST DLMF, classical orthogonal polynomials](https://dlmf.nist.gov/18).
///
/// ## Examples
///
/// ```
/// let value = sci_special::eval_hermitenorm(3,0.2_f64);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy eval_hermitenorm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_hermitenorm.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn eval_hermitenorm<T: sci_core::Float>(n: i32, x: T) -> T {
    if !(0..=4096).contains(&n) {
        return T::nan();
    }
    crate::backend::eval_hermitenorm(n, x).unwrap_or(T::nan())
}

/// Computes the prolate spheroidal characteristic value `lambda_mn(c)`.
///
/// It is the eigenvalue for the solution regular at both endpoints of
/// `(1−x²) S″ − 2x S′ + [lambda − sigma*c²*x² − m²/(1−x²)] S = 0`,
/// where `sigma = 1`. At `c = 0`, `lambda = n*(n+1)`.
///
/// ## Parameters and domain
///
/// Finite dimensionless `c` with `abs(c) <= 64`, and integers `0 <= m <= n <= 128`.
/// The sign of `c` is ignored; kernels use `abs(c)`.
///
/// ## Returns
///
/// The binary64 characteristic value; invalid inputs or numerical failure return NaN.
///
/// ## Numerical behavior and work limits
///
/// Orders and parameters are checked before entering fixed recurrence buffers.
/// Characteristic values use bounded eigenvalue bisection (at most 128 iterations
/// per value); supplied-`cv` variants skip that eigenvalue calculation. Series and
/// recurrences use binary64. Accuracy is parameter dependent and can deteriorate
/// near endpoints, zeros, and radial continuation transitions; the supported work
/// bounds are not uniform accuracy guarantees.
///
/// ## Mathematical reference
///
/// [NIST DLMF, spheroidal wave functions](https://dlmf.nist.gov/30).
///
/// ## Examples
///
/// ```
/// let value = sci_special::pro_cv(1,3,0.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy pro_cv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pro_cv.html).
pub fn pro_cv(m: u64, n: u64, c: f64) -> f64 {
    if m > n || n > 128 || (!c.is_finite() || c.abs() > 64.0) {
        return f64::NAN;
    }
    crate::backend::legacy::prolate_segv(m, n, c)
}

/// Computes the prolate spheroidal angular first-kind function and its derivative.
///
/// The angular function solves
/// `(1−x²) S″ − 2x S′ + [lambda − sigma*c²*x² − m²/(1−x²)] S = 0`,
/// with `sigma = 1` and the regular eigenvalue `lambda_mn(c)`.
///
/// Normalization follows the real Legendre/Bessel-series convention used by
/// the Zhang–Jin/SciPy spheroidal algorithms; no unit-norm rescaling is applied.
///
/// ## Parameters and domain
///
/// Finite dimensionless `c` with `abs(c) <= 64`, and integers `0 <= m <= n <= 128`.
/// The sign of `c` is ignored; kernels use `abs(c)`.
/// Use finite `abs(x) < 1`; the backend accepts endpoints but endpoint derivatives
/// can be singular or nonfinite, so no finite endpoint result is guaranteed.
///
/// ## Returns
///
/// The pair `(value, derivative)` in binary64, with derivative taken with respect
/// to `x`. Invalid inputs or backend failure return `(NaN, NaN)`; individual
/// components may also overflow or become nonfinite near singularities.
///
/// ## Numerical behavior and work limits
///
/// Orders and parameters are checked before entering fixed recurrence buffers.
/// Characteristic values use bounded eigenvalue bisection (at most 128 iterations
/// per value); supplied-`cv` variants skip that eigenvalue calculation. Series and
/// recurrences use binary64. Accuracy is parameter dependent and can deteriorate
/// near endpoints, zeros, and radial continuation transitions; the supported work
/// bounds are not uniform accuracy guarantees.
///
/// ## Mathematical reference
///
/// [NIST DLMF, spheroidal wave functions](https://dlmf.nist.gov/30).
///
/// ## Examples
///
/// ```
/// let value = sci_special::pro_ang1(1,3,0.5,0.3);
/// assert!(value.0.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy pro_ang1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pro_ang1.html).
pub fn pro_ang1(m: u64, n: u64, c: f64, x: f64) -> (f64, f64) {
    if m > n || n > 128 || (!c.is_finite() || c.abs() > 64.0) {
        return (f64::NAN, f64::NAN);
    }
    crate::backend::legacy::prolate_aswfa_nocv(m, n, c, x)
}

/// Computes the prolate spheroidal angular first-kind function and its derivative.
///
/// The angular function solves
/// `(1−x²) S″ − 2x S′ + [lambda − sigma*c²*x² − m²/(1−x²)] S = 0`,
/// with `sigma = 1` and the regular eigenvalue `lambda_mn(c)`.
///
/// Normalization follows the real Legendre/Bessel-series convention used by
/// the Zhang–Jin/SciPy spheroidal algorithms; no unit-norm rescaling is applied.
///
/// ## Parameters and domain
///
/// Finite dimensionless `c` with `abs(c) <= 64`, and integers `0 <= m <= n <= 128`.
/// The sign of `c` is ignored; kernels use `abs(c)`.
/// Use finite `abs(x) < 1`; the backend accepts endpoints but endpoint derivatives
/// can be singular or nonfinite, so no finite endpoint result is guaranteed.
/// `cv` is a finite supplied characteristic value for the same `(m, n, c)`.
/// Sci checks finiteness but does not verify that it is the correct eigenvalue.
///
/// ## Returns
///
/// The pair `(value, derivative)` in binary64, with derivative taken with respect
/// to `x`. Invalid inputs or backend failure return `(NaN, NaN)`; individual
/// components may also overflow or become nonfinite near singularities.
///
/// ## Numerical behavior and work limits
///
/// Orders and parameters are checked before entering fixed recurrence buffers.
/// Characteristic values use bounded eigenvalue bisection (at most 128 iterations
/// per value); supplied-`cv` variants skip that eigenvalue calculation. Series and
/// recurrences use binary64. Accuracy is parameter dependent and can deteriorate
/// near endpoints, zeros, and radial continuation transitions; the supported work
/// bounds are not uniform accuracy guarantees.
///
/// ## Mathematical reference
///
/// [NIST DLMF, spheroidal wave functions](https://dlmf.nist.gov/30).
///
/// ## Examples
///
/// ```
/// let value = sci_special::pro_ang1_cv(1,3,0.5,12.0,0.3);
/// assert!(value.0.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy pro_ang1_cv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pro_ang1_cv.html).
pub fn pro_ang1_cv(m: u64, n: u64, c: f64, cv: f64, x: f64) -> (f64, f64) {
    if m > n || n > 128 || (!c.is_finite() || c.abs() > 64.0) {
        return (f64::NAN, f64::NAN);
    }
    crate::backend::legacy::prolate_aswfa(m, n, c, cv, x)
}

/// Computes the prolate spheroidal radial first-kind function and its derivative.
///
/// The radial function solves
/// `(x²−sigma) R″ + 2x R′ + [c²*x² − lambda − sigma*m²/(x²−sigma)] R = 0`,
/// where `sigma = 1` and `lambda = lambda_mn(c)`.
///
/// Normalization follows the real Legendre/Bessel-series convention used by
/// the Zhang–Jin/SciPy spheroidal algorithms; no unit-norm rescaling is applied.
///
/// ## Parameters and domain
///
/// Finite dimensionless `c` with `abs(c) <= 64`, and integers `0 <= m <= n <= 128`.
/// The sign of `c` is ignored; kernels use `abs(c)`.
/// Use finite `x > 1`. Radial values at `c = 0` are not guaranteed;
/// use positive `abs(c)` for the normalized radial solutions.
///
/// ## Returns
///
/// The pair `(value, derivative)` in binary64, with derivative taken with respect
/// to `x`. Invalid inputs or backend failure return `(NaN, NaN)`; individual
/// components may also overflow or become nonfinite near singularities.
///
/// ## Numerical behavior and work limits
///
/// Orders and parameters are checked before entering fixed recurrence buffers.
/// Characteristic values use bounded eigenvalue bisection (at most 128 iterations
/// per value); supplied-`cv` variants skip that eigenvalue calculation. Series and
/// recurrences use binary64. Accuracy is parameter dependent and can deteriorate
/// near endpoints, zeros, and radial continuation transitions; the supported work
/// bounds are not uniform accuracy guarantees.
///
/// ## Mathematical reference
///
/// [NIST DLMF, spheroidal wave functions](https://dlmf.nist.gov/30).
///
/// ## Examples
///
/// ```
/// let value = sci_special::pro_rad1(1,3,0.5,2.0);
/// assert!(value.0.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy pro_rad1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pro_rad1.html).
pub fn pro_rad1(m: u64, n: u64, c: f64, x: f64) -> (f64, f64) {
    if m > n || n > 128 || (!c.is_finite() || c.abs() > 64.0) {
        return (f64::NAN, f64::NAN);
    }
    crate::backend::legacy::prolate_radial1_nocv(m, n, c, x)
}

/// Computes the prolate spheroidal radial first-kind function and its derivative.
///
/// The radial function solves
/// `(x²−sigma) R″ + 2x R′ + [c²*x² − lambda − sigma*m²/(x²−sigma)] R = 0`,
/// where `sigma = 1` and `lambda = lambda_mn(c)`.
///
/// Normalization follows the real Legendre/Bessel-series convention used by
/// the Zhang–Jin/SciPy spheroidal algorithms; no unit-norm rescaling is applied.
///
/// ## Parameters and domain
///
/// Finite dimensionless `c` with `abs(c) <= 64`, and integers `0 <= m <= n <= 128`.
/// The sign of `c` is ignored; kernels use `abs(c)`.
/// Use finite `x > 1`. Radial values at `c = 0` are not guaranteed;
/// use positive `abs(c)` for the normalized radial solutions.
/// `cv` is a finite supplied characteristic value for the same `(m, n, c)`.
/// Sci checks finiteness but does not verify that it is the correct eigenvalue.
///
/// ## Returns
///
/// The pair `(value, derivative)` in binary64, with derivative taken with respect
/// to `x`. Invalid inputs or backend failure return `(NaN, NaN)`; individual
/// components may also overflow or become nonfinite near singularities.
///
/// ## Numerical behavior and work limits
///
/// Orders and parameters are checked before entering fixed recurrence buffers.
/// Characteristic values use bounded eigenvalue bisection (at most 128 iterations
/// per value); supplied-`cv` variants skip that eigenvalue calculation. Series and
/// recurrences use binary64. Accuracy is parameter dependent and can deteriorate
/// near endpoints, zeros, and radial continuation transitions; the supported work
/// bounds are not uniform accuracy guarantees.
///
/// ## Mathematical reference
///
/// [NIST DLMF, spheroidal wave functions](https://dlmf.nist.gov/30).
///
/// ## Examples
///
/// ```
/// let value = sci_special::pro_rad1_cv(1,3,0.5,12.0,2.0);
/// assert!(value.0.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy pro_rad1_cv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pro_rad1_cv.html).
pub fn pro_rad1_cv(m: u64, n: u64, c: f64, cv: f64, x: f64) -> (f64, f64) {
    if m > n || n > 128 || (!c.is_finite() || c.abs() > 64.0) {
        return (f64::NAN, f64::NAN);
    }
    crate::backend::legacy::prolate_radial1(m, n, c, cv, x)
}

/// Computes the prolate spheroidal radial second-kind function and its derivative.
///
/// The radial function solves
/// `(x²−sigma) R″ + 2x R′ + [c²*x² − lambda − sigma*m²/(x²−sigma)] R = 0`,
/// where `sigma = 1` and `lambda = lambda_mn(c)`.
///
/// Normalization follows the real Legendre/Bessel-series convention used by
/// the Zhang–Jin/SciPy spheroidal algorithms; no unit-norm rescaling is applied.
///
/// ## Parameters and domain
///
/// Finite dimensionless `c` with `abs(c) <= 64`, and integers `0 <= m <= n <= 128`.
/// The sign of `c` is ignored; kernels use `abs(c)`.
/// Use finite `x > 1`. Radial values at `c = 0` are not guaranteed;
/// use positive `abs(c)` for the normalized radial solutions.
///
/// ## Returns
///
/// The pair `(value, derivative)` in binary64, with derivative taken with respect
/// to `x`. Invalid inputs or backend failure return `(NaN, NaN)`; individual
/// components may also overflow or become nonfinite near singularities.
///
/// ## Numerical behavior and work limits
///
/// Orders and parameters are checked before entering fixed recurrence buffers.
/// Characteristic values use bounded eigenvalue bisection (at most 128 iterations
/// per value); supplied-`cv` variants skip that eigenvalue calculation. Series and
/// recurrences use binary64. Accuracy is parameter dependent and can deteriorate
/// near endpoints, zeros, and radial continuation transitions; the supported work
/// bounds are not uniform accuracy guarantees.
///
/// ## Mathematical reference
///
/// [NIST DLMF, spheroidal wave functions](https://dlmf.nist.gov/30).
///
/// ## Examples
///
/// ```
/// let value = sci_special::pro_rad2(1,3,0.5,2.0);
/// assert!(value.0.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy pro_rad2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pro_rad2.html).
pub fn pro_rad2(m: u64, n: u64, c: f64, x: f64) -> (f64, f64) {
    if m > n || n > 128 || (!c.is_finite() || c.abs() > 64.0) {
        return (f64::NAN, f64::NAN);
    }
    crate::backend::legacy::prolate_radial2_nocv(m, n, c, x)
}

/// Computes the prolate spheroidal radial second-kind function and its derivative.
///
/// The radial function solves
/// `(x²−sigma) R″ + 2x R′ + [c²*x² − lambda − sigma*m²/(x²−sigma)] R = 0`,
/// where `sigma = 1` and `lambda = lambda_mn(c)`.
///
/// Normalization follows the real Legendre/Bessel-series convention used by
/// the Zhang–Jin/SciPy spheroidal algorithms; no unit-norm rescaling is applied.
///
/// ## Parameters and domain
///
/// Finite dimensionless `c` with `abs(c) <= 64`, and integers `0 <= m <= n <= 128`.
/// The sign of `c` is ignored; kernels use `abs(c)`.
/// Use finite `x > 1`. Radial values at `c = 0` are not guaranteed;
/// use positive `abs(c)` for the normalized radial solutions.
/// `cv` is a finite supplied characteristic value for the same `(m, n, c)`.
/// Sci checks finiteness but does not verify that it is the correct eigenvalue.
///
/// ## Returns
///
/// The pair `(value, derivative)` in binary64, with derivative taken with respect
/// to `x`. Invalid inputs or backend failure return `(NaN, NaN)`; individual
/// components may also overflow or become nonfinite near singularities.
///
/// ## Numerical behavior and work limits
///
/// Orders and parameters are checked before entering fixed recurrence buffers.
/// Characteristic values use bounded eigenvalue bisection (at most 128 iterations
/// per value); supplied-`cv` variants skip that eigenvalue calculation. Series and
/// recurrences use binary64. Accuracy is parameter dependent and can deteriorate
/// near endpoints, zeros, and radial continuation transitions; the supported work
/// bounds are not uniform accuracy guarantees.
///
/// ## Mathematical reference
///
/// [NIST DLMF, spheroidal wave functions](https://dlmf.nist.gov/30).
///
/// ## Examples
///
/// ```
/// let value = sci_special::pro_rad2_cv(1,3,0.5,12.0,2.0);
/// assert!(value.0.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy pro_rad2_cv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pro_rad2_cv.html).
pub fn pro_rad2_cv(m: u64, n: u64, c: f64, cv: f64, x: f64) -> (f64, f64) {
    if m > n || n > 128 || (!c.is_finite() || c.abs() > 64.0) {
        return (f64::NAN, f64::NAN);
    }
    crate::backend::legacy::prolate_radial2(m, n, c, cv, x)
}

/// Computes the oblate spheroidal characteristic value `lambda_mn(c)`.
///
/// It is the eigenvalue for the solution regular at both endpoints of
/// `(1−x²) S″ − 2x S′ + [lambda − sigma*c²*x² − m²/(1−x²)] S = 0`,
/// where `sigma = −1`. At `c = 0`, `lambda = n*(n+1)`.
///
/// ## Parameters and domain
///
/// Finite dimensionless `c` with `abs(c) <= 64`, and integers `0 <= m <= n <= 128`.
/// The sign of `c` is ignored; kernels use `abs(c)`.
///
/// ## Returns
///
/// The binary64 characteristic value; invalid inputs or numerical failure return NaN.
///
/// ## Numerical behavior and work limits
///
/// Orders and parameters are checked before entering fixed recurrence buffers.
/// Characteristic values use bounded eigenvalue bisection (at most 128 iterations
/// per value); supplied-`cv` variants skip that eigenvalue calculation. Series and
/// recurrences use binary64. Accuracy is parameter dependent and can deteriorate
/// near endpoints, zeros, and radial continuation transitions; the supported work
/// bounds are not uniform accuracy guarantees.
///
/// ## Mathematical reference
///
/// [NIST DLMF, spheroidal wave functions](https://dlmf.nist.gov/30).
///
/// ## Examples
///
/// ```
/// let value = sci_special::obl_cv(1,3,0.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy obl_cv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.obl_cv.html).
pub fn obl_cv(m: u64, n: u64, c: f64) -> f64 {
    if m > n || n > 128 || (!c.is_finite() || c.abs() > 64.0) {
        return f64::NAN;
    }
    crate::backend::legacy::oblate_segv(m, n, c)
}

/// Computes the oblate spheroidal angular first-kind function and its derivative.
///
/// The angular function solves
/// `(1−x²) S″ − 2x S′ + [lambda − sigma*c²*x² − m²/(1−x²)] S = 0`,
/// with `sigma = −1` and the regular eigenvalue `lambda_mn(c)`.
///
/// Normalization follows the real Legendre/Bessel-series convention used by
/// the Zhang–Jin/SciPy spheroidal algorithms; no unit-norm rescaling is applied.
///
/// ## Parameters and domain
///
/// Finite dimensionless `c` with `abs(c) <= 64`, and integers `0 <= m <= n <= 128`.
/// The sign of `c` is ignored; kernels use `abs(c)`.
/// Use finite `abs(x) < 1`; the backend accepts endpoints but endpoint derivatives
/// can be singular or nonfinite, so no finite endpoint result is guaranteed.
///
/// ## Returns
///
/// The pair `(value, derivative)` in binary64, with derivative taken with respect
/// to `x`. Invalid inputs or backend failure return `(NaN, NaN)`; individual
/// components may also overflow or become nonfinite near singularities.
///
/// ## Numerical behavior and work limits
///
/// Orders and parameters are checked before entering fixed recurrence buffers.
/// Characteristic values use bounded eigenvalue bisection (at most 128 iterations
/// per value); supplied-`cv` variants skip that eigenvalue calculation. Series and
/// recurrences use binary64. Accuracy is parameter dependent and can deteriorate
/// near endpoints, zeros, and radial continuation transitions; the supported work
/// bounds are not uniform accuracy guarantees.
///
/// ## Mathematical reference
///
/// [NIST DLMF, spheroidal wave functions](https://dlmf.nist.gov/30).
///
/// ## Examples
///
/// ```
/// let value = sci_special::obl_ang1(1,3,0.5,0.3);
/// assert!(value.0.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy obl_ang1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.obl_ang1.html).
pub fn obl_ang1(m: u64, n: u64, c: f64, x: f64) -> (f64, f64) {
    if m > n || n > 128 || (!c.is_finite() || c.abs() > 64.0) {
        return (f64::NAN, f64::NAN);
    }
    crate::backend::legacy::oblate_aswfa_nocv(m, n, c, x)
}

/// Computes the oblate spheroidal angular first-kind function and its derivative.
///
/// The angular function solves
/// `(1−x²) S″ − 2x S′ + [lambda − sigma*c²*x² − m²/(1−x²)] S = 0`,
/// with `sigma = −1` and the regular eigenvalue `lambda_mn(c)`.
///
/// Normalization follows the real Legendre/Bessel-series convention used by
/// the Zhang–Jin/SciPy spheroidal algorithms; no unit-norm rescaling is applied.
///
/// ## Parameters and domain
///
/// Finite dimensionless `c` with `abs(c) <= 64`, and integers `0 <= m <= n <= 128`.
/// The sign of `c` is ignored; kernels use `abs(c)`.
/// Use finite `abs(x) < 1`; the backend accepts endpoints but endpoint derivatives
/// can be singular or nonfinite, so no finite endpoint result is guaranteed.
/// `cv` is a finite supplied characteristic value for the same `(m, n, c)`.
/// Sci checks finiteness but does not verify that it is the correct eigenvalue.
///
/// ## Returns
///
/// The pair `(value, derivative)` in binary64, with derivative taken with respect
/// to `x`. Invalid inputs or backend failure return `(NaN, NaN)`; individual
/// components may also overflow or become nonfinite near singularities.
///
/// ## Numerical behavior and work limits
///
/// Orders and parameters are checked before entering fixed recurrence buffers.
/// Characteristic values use bounded eigenvalue bisection (at most 128 iterations
/// per value); supplied-`cv` variants skip that eigenvalue calculation. Series and
/// recurrences use binary64. Accuracy is parameter dependent and can deteriorate
/// near endpoints, zeros, and radial continuation transitions; the supported work
/// bounds are not uniform accuracy guarantees.
///
/// ## Mathematical reference
///
/// [NIST DLMF, spheroidal wave functions](https://dlmf.nist.gov/30).
///
/// ## Examples
///
/// ```
/// let value = sci_special::obl_ang1_cv(1,3,0.5,12.0,0.3);
/// assert!(value.0.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy obl_ang1_cv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.obl_ang1_cv.html).
pub fn obl_ang1_cv(m: u64, n: u64, c: f64, cv: f64, x: f64) -> (f64, f64) {
    if m > n || n > 128 || (!c.is_finite() || c.abs() > 64.0) {
        return (f64::NAN, f64::NAN);
    }
    crate::backend::legacy::oblate_aswfa(m, n, c, cv, x)
}

/// Computes the oblate spheroidal radial first-kind function and its derivative.
///
/// The radial function solves
/// `(x²−sigma) R″ + 2x R′ + [c²*x² − lambda − sigma*m²/(x²−sigma)] R = 0`,
/// where `sigma = −1` and `lambda = lambda_mn(c)`.
///
/// Normalization follows the real Legendre/Bessel-series convention used by
/// the Zhang–Jin/SciPy spheroidal algorithms; no unit-norm rescaling is applied.
///
/// ## Parameters and domain
///
/// Finite dimensionless `c` with `abs(c) <= 64`, and integers `0 <= m <= n <= 128`.
/// The sign of `c` is ignored; kernels use `abs(c)`.
/// Use finite `x >= 0`. Radial values at `c = 0` are not guaranteed;
/// use positive `abs(c)` for the normalized radial solutions.
///
/// ## Returns
///
/// The pair `(value, derivative)` in binary64, with derivative taken with respect
/// to `x`. Invalid inputs or backend failure return `(NaN, NaN)`; individual
/// components may also overflow or become nonfinite near singularities.
///
/// ## Numerical behavior and work limits
///
/// Orders and parameters are checked before entering fixed recurrence buffers.
/// Characteristic values use bounded eigenvalue bisection (at most 128 iterations
/// per value); supplied-`cv` variants skip that eigenvalue calculation. Series and
/// recurrences use binary64. Accuracy is parameter dependent and can deteriorate
/// near endpoints, zeros, and radial continuation transitions; the supported work
/// bounds are not uniform accuracy guarantees.
///
/// ## Mathematical reference
///
/// [NIST DLMF, spheroidal wave functions](https://dlmf.nist.gov/30).
///
/// ## Examples
///
/// ```
/// let value = sci_special::obl_rad1(1,3,0.5,2.0);
/// assert!(value.0.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy obl_rad1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.obl_rad1.html).
pub fn obl_rad1(m: u64, n: u64, c: f64, x: f64) -> (f64, f64) {
    if m > n || n > 128 || (!c.is_finite() || c.abs() > 64.0) {
        return (f64::NAN, f64::NAN);
    }
    crate::backend::legacy::oblate_radial1_nocv(m, n, c, x)
}

/// Computes the oblate spheroidal radial first-kind function and its derivative.
///
/// The radial function solves
/// `(x²−sigma) R″ + 2x R′ + [c²*x² − lambda − sigma*m²/(x²−sigma)] R = 0`,
/// where `sigma = −1` and `lambda = lambda_mn(c)`.
///
/// Normalization follows the real Legendre/Bessel-series convention used by
/// the Zhang–Jin/SciPy spheroidal algorithms; no unit-norm rescaling is applied.
///
/// ## Parameters and domain
///
/// Finite dimensionless `c` with `abs(c) <= 64`, and integers `0 <= m <= n <= 128`.
/// The sign of `c` is ignored; kernels use `abs(c)`.
/// Use finite `x >= 0`. Radial values at `c = 0` are not guaranteed;
/// use positive `abs(c)` for the normalized radial solutions.
/// `cv` is a finite supplied characteristic value for the same `(m, n, c)`.
/// Sci checks finiteness but does not verify that it is the correct eigenvalue.
///
/// ## Returns
///
/// The pair `(value, derivative)` in binary64, with derivative taken with respect
/// to `x`. Invalid inputs or backend failure return `(NaN, NaN)`; individual
/// components may also overflow or become nonfinite near singularities.
///
/// ## Numerical behavior and work limits
///
/// Orders and parameters are checked before entering fixed recurrence buffers.
/// Characteristic values use bounded eigenvalue bisection (at most 128 iterations
/// per value); supplied-`cv` variants skip that eigenvalue calculation. Series and
/// recurrences use binary64. Accuracy is parameter dependent and can deteriorate
/// near endpoints, zeros, and radial continuation transitions; the supported work
/// bounds are not uniform accuracy guarantees.
///
/// ## Mathematical reference
///
/// [NIST DLMF, spheroidal wave functions](https://dlmf.nist.gov/30).
///
/// ## Examples
///
/// ```
/// let value = sci_special::obl_rad1_cv(1,3,0.5,12.0,2.0);
/// assert!(value.0.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy obl_rad1_cv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.obl_rad1_cv.html).
pub fn obl_rad1_cv(m: u64, n: u64, c: f64, cv: f64, x: f64) -> (f64, f64) {
    if m > n || n > 128 || (!c.is_finite() || c.abs() > 64.0) {
        return (f64::NAN, f64::NAN);
    }
    crate::backend::legacy::oblate_radial1(m, n, c, cv, x)
}

/// Computes the oblate spheroidal radial second-kind function and its derivative.
///
/// The radial function solves
/// `(x²−sigma) R″ + 2x R′ + [c²*x² − lambda − sigma*m²/(x²−sigma)] R = 0`,
/// where `sigma = −1` and `lambda = lambda_mn(c)`.
///
/// Normalization follows the real Legendre/Bessel-series convention used by
/// the Zhang–Jin/SciPy spheroidal algorithms; no unit-norm rescaling is applied.
///
/// ## Parameters and domain
///
/// Finite dimensionless `c` with `abs(c) <= 64`, and integers `0 <= m <= n <= 128`.
/// The sign of `c` is ignored; kernels use `abs(c)`.
/// Use finite `x >= 0`. Radial values at `c = 0` are not guaranteed;
/// use positive `abs(c)` for the normalized radial solutions.
///
/// ## Returns
///
/// The pair `(value, derivative)` in binary64, with derivative taken with respect
/// to `x`. Invalid inputs or backend failure return `(NaN, NaN)`; individual
/// components may also overflow or become nonfinite near singularities.
///
/// ## Numerical behavior and work limits
///
/// Orders and parameters are checked before entering fixed recurrence buffers.
/// Characteristic values use bounded eigenvalue bisection (at most 128 iterations
/// per value); supplied-`cv` variants skip that eigenvalue calculation. Series and
/// recurrences use binary64. Accuracy is parameter dependent and can deteriorate
/// near endpoints, zeros, and radial continuation transitions; the supported work
/// bounds are not uniform accuracy guarantees.
///
/// ## Mathematical reference
///
/// [NIST DLMF, spheroidal wave functions](https://dlmf.nist.gov/30).
///
/// ## Examples
///
/// ```
/// let value = sci_special::obl_rad2(1,3,0.5,2.0);
/// assert!(value.0.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy obl_rad2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.obl_rad2.html).
pub fn obl_rad2(m: u64, n: u64, c: f64, x: f64) -> (f64, f64) {
    if m > n || n > 128 || (!c.is_finite() || c.abs() > 64.0) {
        return (f64::NAN, f64::NAN);
    }
    crate::backend::legacy::oblate_radial2_nocv(m, n, c, x)
}

/// Computes the oblate spheroidal radial second-kind function and its derivative.
///
/// The radial function solves
/// `(x²−sigma) R″ + 2x R′ + [c²*x² − lambda − sigma*m²/(x²−sigma)] R = 0`,
/// where `sigma = −1` and `lambda = lambda_mn(c)`.
///
/// Normalization follows the real Legendre/Bessel-series convention used by
/// the Zhang–Jin/SciPy spheroidal algorithms; no unit-norm rescaling is applied.
///
/// ## Parameters and domain
///
/// Finite dimensionless `c` with `abs(c) <= 64`, and integers `0 <= m <= n <= 128`.
/// The sign of `c` is ignored; kernels use `abs(c)`.
/// Use finite `x >= 0`. Radial values at `c = 0` are not guaranteed;
/// use positive `abs(c)` for the normalized radial solutions.
/// `cv` is a finite supplied characteristic value for the same `(m, n, c)`.
/// Sci checks finiteness but does not verify that it is the correct eigenvalue.
///
/// ## Returns
///
/// The pair `(value, derivative)` in binary64, with derivative taken with respect
/// to `x`. Invalid inputs or backend failure return `(NaN, NaN)`; individual
/// components may also overflow or become nonfinite near singularities.
///
/// ## Numerical behavior and work limits
///
/// Orders and parameters are checked before entering fixed recurrence buffers.
/// Characteristic values use bounded eigenvalue bisection (at most 128 iterations
/// per value); supplied-`cv` variants skip that eigenvalue calculation. Series and
/// recurrences use binary64. Accuracy is parameter dependent and can deteriorate
/// near endpoints, zeros, and radial continuation transitions; the supported work
/// bounds are not uniform accuracy guarantees.
///
/// ## Mathematical reference
///
/// [NIST DLMF, spheroidal wave functions](https://dlmf.nist.gov/30).
///
/// ## Examples
///
/// ```
/// let value = sci_special::obl_rad2_cv(1,3,0.5,12.0,2.0);
/// assert!(value.0.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy obl_rad2_cv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.obl_rad2_cv.html).
pub fn obl_rad2_cv(m: u64, n: u64, c: f64, cv: f64, x: f64) -> (f64, f64) {
    if m > n || n > 128 || (!c.is_finite() || c.abs() > 64.0) {
        return (f64::NAN, f64::NAN);
    }
    crate::backend::legacy::oblate_radial2(m, n, c, cv, x)
}
