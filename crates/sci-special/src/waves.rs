//! Private wave-function and polynomial evaluation adapters.

/// Jacobi polynomial P_n^(alpha,beta)(x).
///
/// ## Parameters and domain
///
/// Integer degree n≥0; real x and alpha,beta>−1.
/// Unsupported orders return NaN. Scalar outputs propagate IEEE overflow and domain errors.
///
/// # Examples
///
/// ```
/// let value = sci_special::eval_jacobi(3,0.5_f64,0.3,0.2);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy eval_jacobi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_jacobi.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn eval_jacobi<T: sci_core::Float>(n: i32, alpha: T, beta: T, x: T) -> T {
    if !(0..=4096).contains(&n) {
        return T::nan();
    }
    crate::backend::eval_jacobi(n, alpha, beta, x).unwrap_or(T::nan())
}

/// Legendre polynomial P_n(x).
///
/// ## Parameters and domain
///
/// Integer degree n≥0 and real x.
/// Unsupported orders return NaN. Scalar outputs propagate IEEE overflow and domain errors.
///
/// # Examples
///
/// ```
/// let value = sci_special::eval_legendre(3,0.2_f64);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy eval_legendre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_legendre.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn eval_legendre<T: sci_core::Float>(n: i32, x: T) -> T {
    if !(0..=4096).contains(&n) {
        return T::nan();
    }
    crate::backend::eval_legendre(n, x).unwrap_or(T::nan())
}

/// Gegenbauer polynomial C_n^(alpha)(x).
///
/// ## Parameters and domain
///
/// Integer degree n≥0, real alpha and real x.
/// Unsupported orders return NaN. Scalar outputs propagate IEEE overflow and domain errors.
///
/// # Examples
///
/// ```
/// let value = sci_special::eval_gegenbauer(3,0.5_f64,0.2);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy eval_gegenbauer](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_gegenbauer.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn eval_gegenbauer<T: sci_core::Float>(n: i32, alpha: T, x: T) -> T {
    if !(0..=4096).contains(&n) {
        return T::nan();
    }
    crate::backend::eval_gegenbauer(n, alpha, x).unwrap_or(T::nan())
}

/// Generalized Laguerre polynomial L_n^(alpha)(x).
///
/// ## Parameters and domain
///
/// Integer degree n≥0, alpha>−1, and real x.
/// Unsupported orders return NaN. Scalar outputs propagate IEEE overflow and domain errors.
///
/// # Examples
///
/// ```
/// let value = sci_special::eval_genlaguerre(3,0.5_f64,0.2);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy eval_genlaguerre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_genlaguerre.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn eval_genlaguerre<T: sci_core::Float>(n: i32, alpha: T, x: T) -> T {
    if !(0..=4096).contains(&n) {
        return T::nan();
    }
    crate::backend::eval_genlaguerre(n, alpha, x).unwrap_or(T::nan())
}

/// Laguerre polynomial L_n(x).
///
/// ## Parameters and domain
///
/// Integer degree n≥0 and real x.
/// Unsupported orders return NaN. Scalar outputs propagate IEEE overflow and domain errors.
///
/// # Examples
///
/// ```
/// let value = sci_special::eval_laguerre(3,0.2_f64);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy eval_laguerre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_laguerre.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn eval_laguerre<T: sci_core::Float>(n: i32, x: T) -> T {
    if !(0..=4096).contains(&n) {
        return T::nan();
    }
    crate::backend::eval_laguerre(n, x).unwrap_or(T::nan())
}

/// Chebyshev first-kind polynomial T_n(x), defined by T_n(cos θ)=cos(nθ).
///
/// ## Parameters and domain
///
/// Integer degree n≥0 and real x.
/// Unsupported orders return NaN. Scalar outputs propagate IEEE overflow and domain errors.
///
/// # Examples
///
/// ```
/// let value = sci_special::eval_chebyt(3,0.2_f64);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy eval_chebyt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_chebyt.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn eval_chebyt<T: sci_core::Float>(n: i32, x: T) -> T {
    if !(0..=4096).contains(&n) {
        return T::nan();
    }
    crate::backend::eval_chebyt(n, x).unwrap_or(T::nan())
}

/// Chebyshev second-kind polynomial U_n(x), defined by U_n(cos θ)=sin((n+1)θ)/sin θ.
///
/// ## Parameters and domain
///
/// Integer degree n≥0 and real x.
/// Unsupported orders return NaN. Scalar outputs propagate IEEE overflow and domain errors.
///
/// # Examples
///
/// ```
/// let value = sci_special::eval_chebyu(3,0.2_f64);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy eval_chebyu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_chebyu.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn eval_chebyu<T: sci_core::Float>(n: i32, x: T) -> T {
    if !(0..=4096).contains(&n) {
        return T::nan();
    }
    crate::backend::eval_chebyu(n, x).unwrap_or(T::nan())
}

/// Physicists' Hermite polynomial H_n(x), with leading coefficient 2^n.
///
/// ## Parameters and domain
///
/// Integer degree n≥0 and real x.
/// Unsupported orders return NaN. Scalar outputs propagate IEEE overflow and domain errors.
///
/// # Examples
///
/// ```
/// let value = sci_special::eval_hermite(3,0.2_f64);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy eval_hermite](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_hermite.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn eval_hermite<T: sci_core::Float>(n: i32, x: T) -> T {
    if !(0..=4096).contains(&n) {
        return T::nan();
    }
    crate::backend::eval_hermite(n, x).unwrap_or(T::nan())
}

/// Probabilists' Hermite polynomial He_n(x), with leading coefficient one.
///
/// ## Parameters and domain
///
/// Integer degree n≥0 and real x.
/// Unsupported orders return NaN. Scalar outputs propagate IEEE overflow and domain errors.
///
/// # Examples
///
/// ```
/// let value = sci_special::eval_hermitenorm(3,0.2_f64);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy eval_hermitenorm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_hermitenorm.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn eval_hermitenorm<T: sci_core::Float>(n: i32, x: T) -> T {
    if !(0..=4096).contains(&n) {
        return T::nan();
    }
    crate::backend::eval_hermitenorm(n, x).unwrap_or(T::nan())
}

/// prolate spheroidal characteristic value λ_(mn)(c).
///
/// ## Parameters and domain
///
/// Integer 0≤m≤n≤128, and finite real c with abs(c)≤64.
/// Unsupported orders return NaN. Scalar outputs propagate IEEE overflow and domain errors.
///
/// # Examples
///
/// ```
/// let value = sci_special::pro_cv(1,3,0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy pro_cv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pro_cv.html).
pub fn pro_cv(m: u64, n: u64, c: f64) -> f64 {
    if m > n || n > 128 || (!c.is_finite() || c.abs() > 64.0) {
        return f64::NAN;
    }
    crate::backend::legacy::prolate_segv(m, n, c)
}

/// prolate spheroidal angular first-kind function.
///
/// ## Parameters and domain
///
/// Integer 0≤m≤n≤128 and finite c with abs(c)≤64. abs(x)<1. Returns (value, derivative with respect to x).
/// Unsupported orders return NaN. Scalar outputs propagate IEEE overflow and domain errors.
///
/// # Examples
///
/// ```
/// let value = sci_special::pro_ang1(1,3,0.5,0.3);
/// assert!(value.0.is_finite());
/// ```
///
/// # References
///
/// - [SciPy pro_ang1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pro_ang1.html).
pub fn pro_ang1(m: u64, n: u64, c: f64, x: f64) -> (f64, f64) {
    if m > n || n > 128 || (!c.is_finite() || c.abs() > 64.0) {
        return (f64::NAN, f64::NAN);
    }
    crate::backend::legacy::prolate_aswfa_nocv(m, n, c, x)
}

/// prolate spheroidal function evaluated with a supplied characteristic value cv.
///
/// ## Parameters and domain
///
/// Integer 0≤m≤n≤128 and finite c with abs(c)≤64. abs(x)<1. Returns (value, derivative with respect to x).
/// Unsupported orders return NaN. Scalar outputs propagate IEEE overflow and domain errors.
///
/// # Examples
///
/// ```
/// let value = sci_special::pro_ang1_cv(1,3,0.5,12.0,0.3);
/// assert!(value.0.is_finite());
/// ```
///
/// # References
///
/// - [SciPy pro_ang1_cv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pro_ang1_cv.html).
pub fn pro_ang1_cv(m: u64, n: u64, c: f64, cv: f64, x: f64) -> (f64, f64) {
    if m > n || n > 128 || (!c.is_finite() || c.abs() > 64.0) {
        return (f64::NAN, f64::NAN);
    }
    crate::backend::legacy::prolate_aswfa(m, n, c, cv, x)
}

/// prolate spheroidal radial first-kind function.
///
/// ## Parameters and domain
///
/// Integer 0≤m≤n≤128 and finite c with abs(c)≤64. x>1. Returns (value, derivative with respect to x).
/// Unsupported orders return NaN. Scalar outputs propagate IEEE overflow and domain errors.
///
/// # Examples
///
/// ```
/// let value = sci_special::pro_rad1(1,3,0.5,2.0);
/// assert!(value.0.is_finite());
/// ```
///
/// # References
///
/// - [SciPy pro_rad1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pro_rad1.html).
pub fn pro_rad1(m: u64, n: u64, c: f64, x: f64) -> (f64, f64) {
    if m > n || n > 128 || (!c.is_finite() || c.abs() > 64.0) {
        return (f64::NAN, f64::NAN);
    }
    crate::backend::legacy::prolate_radial1_nocv(m, n, c, x)
}

/// prolate spheroidal function evaluated with a supplied characteristic value cv.
///
/// ## Parameters and domain
///
/// Integer 0≤m≤n≤128 and finite c with abs(c)≤64. x>1. Returns (value, derivative with respect to x).
/// Unsupported orders return NaN. Scalar outputs propagate IEEE overflow and domain errors.
///
/// # Examples
///
/// ```
/// let value = sci_special::pro_rad1_cv(1,3,0.5,12.0,2.0);
/// assert!(value.0.is_finite());
/// ```
///
/// # References
///
/// - [SciPy pro_rad1_cv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pro_rad1_cv.html).
pub fn pro_rad1_cv(m: u64, n: u64, c: f64, cv: f64, x: f64) -> (f64, f64) {
    if m > n || n > 128 || (!c.is_finite() || c.abs() > 64.0) {
        return (f64::NAN, f64::NAN);
    }
    crate::backend::legacy::prolate_radial1(m, n, c, cv, x)
}

/// prolate spheroidal radial second-kind function.
///
/// ## Parameters and domain
///
/// Integer 0≤m≤n≤128 and finite c with abs(c)≤64. x>1. Returns (value, derivative with respect to x).
/// Unsupported orders return NaN. Scalar outputs propagate IEEE overflow and domain errors.
///
/// # Examples
///
/// ```
/// let value = sci_special::pro_rad2(1,3,0.5,2.0);
/// assert!(value.0.is_finite());
/// ```
///
/// # References
///
/// - [SciPy pro_rad2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pro_rad2.html).
pub fn pro_rad2(m: u64, n: u64, c: f64, x: f64) -> (f64, f64) {
    if m > n || n > 128 || (!c.is_finite() || c.abs() > 64.0) {
        return (f64::NAN, f64::NAN);
    }
    crate::backend::legacy::prolate_radial2_nocv(m, n, c, x)
}

/// prolate spheroidal function evaluated with a supplied characteristic value cv.
///
/// ## Parameters and domain
///
/// Integer 0≤m≤n≤128 and finite c with abs(c)≤64. x>1. Returns (value, derivative with respect to x).
/// Unsupported orders return NaN. Scalar outputs propagate IEEE overflow and domain errors.
///
/// # Examples
///
/// ```
/// let value = sci_special::pro_rad2_cv(1,3,0.5,12.0,2.0);
/// assert!(value.0.is_finite());
/// ```
///
/// # References
///
/// - [SciPy pro_rad2_cv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pro_rad2_cv.html).
pub fn pro_rad2_cv(m: u64, n: u64, c: f64, cv: f64, x: f64) -> (f64, f64) {
    if m > n || n > 128 || (!c.is_finite() || c.abs() > 64.0) {
        return (f64::NAN, f64::NAN);
    }
    crate::backend::legacy::prolate_radial2(m, n, c, cv, x)
}

/// oblate spheroidal characteristic value λ_(mn)(c).
///
/// ## Parameters and domain
///
/// Integer 0≤m≤n≤128, and finite real c with abs(c)≤64.
/// Unsupported orders return NaN. Scalar outputs propagate IEEE overflow and domain errors.
///
/// # Examples
///
/// ```
/// let value = sci_special::obl_cv(1,3,0.5);
/// assert!(value.is_finite());
/// ```
///
/// # References
///
/// - [SciPy obl_cv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.obl_cv.html).
pub fn obl_cv(m: u64, n: u64, c: f64) -> f64 {
    if m > n || n > 128 || (!c.is_finite() || c.abs() > 64.0) {
        return f64::NAN;
    }
    crate::backend::legacy::oblate_segv(m, n, c)
}

/// oblate spheroidal angular first-kind function.
///
/// ## Parameters and domain
///
/// Integer 0≤m≤n≤128 and finite c with abs(c)≤64. abs(x)<1. Returns (value, derivative with respect to x).
/// Unsupported orders return NaN. Scalar outputs propagate IEEE overflow and domain errors.
///
/// # Examples
///
/// ```
/// let value = sci_special::obl_ang1(1,3,0.5,0.3);
/// assert!(value.0.is_finite());
/// ```
///
/// # References
///
/// - [SciPy obl_ang1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.obl_ang1.html).
pub fn obl_ang1(m: u64, n: u64, c: f64, x: f64) -> (f64, f64) {
    if m > n || n > 128 || (!c.is_finite() || c.abs() > 64.0) {
        return (f64::NAN, f64::NAN);
    }
    crate::backend::legacy::oblate_aswfa_nocv(m, n, c, x)
}

/// oblate spheroidal function evaluated with a supplied characteristic value cv.
///
/// ## Parameters and domain
///
/// Integer 0≤m≤n≤128 and finite c with abs(c)≤64. abs(x)<1. Returns (value, derivative with respect to x).
/// Unsupported orders return NaN. Scalar outputs propagate IEEE overflow and domain errors.
///
/// # Examples
///
/// ```
/// let value = sci_special::obl_ang1_cv(1,3,0.5,12.0,0.3);
/// assert!(value.0.is_finite());
/// ```
///
/// # References
///
/// - [SciPy obl_ang1_cv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.obl_ang1_cv.html).
pub fn obl_ang1_cv(m: u64, n: u64, c: f64, cv: f64, x: f64) -> (f64, f64) {
    if m > n || n > 128 || (!c.is_finite() || c.abs() > 64.0) {
        return (f64::NAN, f64::NAN);
    }
    crate::backend::legacy::oblate_aswfa(m, n, c, cv, x)
}

/// oblate spheroidal radial first-kind function.
///
/// ## Parameters and domain
///
/// Integer 0≤m≤n≤128 and finite c with abs(c)≤64. x≥0. Returns (value, derivative with respect to x).
/// Unsupported orders return NaN. Scalar outputs propagate IEEE overflow and domain errors.
///
/// # Examples
///
/// ```
/// let value = sci_special::obl_rad1(1,3,0.5,2.0);
/// assert!(value.0.is_finite());
/// ```
///
/// # References
///
/// - [SciPy obl_rad1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.obl_rad1.html).
pub fn obl_rad1(m: u64, n: u64, c: f64, x: f64) -> (f64, f64) {
    if m > n || n > 128 || (!c.is_finite() || c.abs() > 64.0) {
        return (f64::NAN, f64::NAN);
    }
    crate::backend::legacy::oblate_radial1_nocv(m, n, c, x)
}

/// oblate spheroidal function evaluated with a supplied characteristic value cv.
///
/// ## Parameters and domain
///
/// Integer 0≤m≤n≤128 and finite c with abs(c)≤64. x≥0. Returns (value, derivative with respect to x).
/// Unsupported orders return NaN. Scalar outputs propagate IEEE overflow and domain errors.
///
/// # Examples
///
/// ```
/// let value = sci_special::obl_rad1_cv(1,3,0.5,12.0,2.0);
/// assert!(value.0.is_finite());
/// ```
///
/// # References
///
/// - [SciPy obl_rad1_cv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.obl_rad1_cv.html).
pub fn obl_rad1_cv(m: u64, n: u64, c: f64, cv: f64, x: f64) -> (f64, f64) {
    if m > n || n > 128 || (!c.is_finite() || c.abs() > 64.0) {
        return (f64::NAN, f64::NAN);
    }
    crate::backend::legacy::oblate_radial1(m, n, c, cv, x)
}

/// oblate spheroidal radial second-kind function.
///
/// ## Parameters and domain
///
/// Integer 0≤m≤n≤128 and finite c with abs(c)≤64. x≥0. Returns (value, derivative with respect to x).
/// Unsupported orders return NaN. Scalar outputs propagate IEEE overflow and domain errors.
///
/// # Examples
///
/// ```
/// let value = sci_special::obl_rad2(1,3,0.5,2.0);
/// assert!(value.0.is_finite());
/// ```
///
/// # References
///
/// - [SciPy obl_rad2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.obl_rad2.html).
pub fn obl_rad2(m: u64, n: u64, c: f64, x: f64) -> (f64, f64) {
    if m > n || n > 128 || (!c.is_finite() || c.abs() > 64.0) {
        return (f64::NAN, f64::NAN);
    }
    crate::backend::legacy::oblate_radial2_nocv(m, n, c, x)
}

/// oblate spheroidal function evaluated with a supplied characteristic value cv.
///
/// ## Parameters and domain
///
/// Integer 0≤m≤n≤128 and finite c with abs(c)≤64. x≥0. Returns (value, derivative with respect to x).
/// Unsupported orders return NaN. Scalar outputs propagate IEEE overflow and domain errors.
///
/// # Examples
///
/// ```
/// let value = sci_special::obl_rad2_cv(1,3,0.5,12.0,2.0);
/// assert!(value.0.is_finite());
/// ```
///
/// # References
///
/// - [SciPy obl_rad2_cv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.obl_rad2_cv.html).
pub fn obl_rad2_cv(m: u64, n: u64, c: f64, cv: f64, x: f64) -> (f64, f64) {
    if m > n || n > 128 || (!c.is_finite() || c.abs() > 64.0) {
        return (f64::NAN, f64::NAN);
    }
    crate::backend::legacy::oblate_radial2(m, n, c, cv, x)
}
