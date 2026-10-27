//! Complex results and bounded sequence adapters.

use crate::{Error, Result, complex::Complex};

/// First-kind Hankel function `H_v^(1)(x) = J_v(x) + i Y_v(x)`.
///
/// ## Parameters and domain
///
/// Finite real order `v` and real argument `x`, passed as `x + 0i`. Negative
/// arguments use the principal complex branch on the upper side of the
/// negative-real cut. This API does not accept a general complex argument.
///
/// ## Returns
///
/// Singularities, overflow, and invalid domains return IEEE infinities or NaNs.
/// Returned values never contain backend-specific types.
///
/// ## Numerical behavior
///
/// Uses the binary64 AMOS complex Bessel backend. Backend domain, convergence,
/// overflow, or precision-loss errors produce NaN components. Phase reduction
/// limits large-argument accuracy; the scaled forms avoid the explicit Hankel
/// exponential. No configurable tolerance or iteration control is exposed.
///
/// ## Examples
///
/// ```
/// let value = sci_special::hankel1(0.5,2.0);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy hankel1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hankel1.html).
pub fn hankel1(v: f64, x: f64) -> Complex {
    crate::complex::owned(crate::backend::legacy::hankel_1(v, x))
}

/// Scaled first-kind Hankel function `exp(−i*x) H_v^(1)(x)`.
///
/// ## Parameters and domain
///
/// Finite real order `v` and real argument `x`, passed as `x + 0i`. Negative
/// arguments use the principal complex branch on the upper side of the
/// negative-real cut. This API does not accept a general complex argument.
///
/// ## Returns
///
/// Singularities, overflow, and invalid domains return IEEE infinities or NaNs.
/// Returned values never contain backend-specific types.
///
/// ## Numerical behavior
///
/// Uses the binary64 AMOS complex Bessel backend. Backend domain, convergence,
/// overflow, or precision-loss errors produce NaN components. Phase reduction
/// limits large-argument accuracy; the scaled forms avoid the explicit Hankel
/// exponential. No configurable tolerance or iteration control is exposed.
///
/// ## Examples
///
/// ```
/// let value = sci_special::hankel1e(0.5,2.0);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy hankel1e](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hankel1e.html).
pub fn hankel1e(v: f64, x: f64) -> Complex {
    crate::complex::owned(crate::backend::legacy::hankel_1e(v, x))
}

/// Second-kind Hankel function `H_v^(2)(x) = J_v(x) − i Y_v(x)`.
///
/// ## Parameters and domain
///
/// Finite real order `v` and real argument `x`, passed as `x + 0i`. Negative
/// arguments use the principal complex branch on the upper side of the
/// negative-real cut. This API does not accept a general complex argument.
///
/// ## Returns
///
/// Singularities, overflow, and invalid domains return IEEE infinities or NaNs.
/// Returned values never contain backend-specific types.
///
/// ## Numerical behavior
///
/// Uses the binary64 AMOS complex Bessel backend. Backend domain, convergence,
/// overflow, or precision-loss errors produce NaN components. Phase reduction
/// limits large-argument accuracy; the scaled forms avoid the explicit Hankel
/// exponential. No configurable tolerance or iteration control is exposed.
///
/// ## Examples
///
/// ```
/// let value = sci_special::hankel2(0.5,2.0);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy hankel2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hankel2.html).
pub fn hankel2(v: f64, x: f64) -> Complex {
    crate::complex::owned(crate::backend::legacy::hankel_2(v, x))
}

/// Scaled second-kind Hankel function `exp(i*x) H_v^(2)(x)`.
///
/// ## Parameters and domain
///
/// Finite real order `v` and real argument `x`, passed as `x + 0i`. Negative
/// arguments use the principal complex branch on the upper side of the
/// negative-real cut. This API does not accept a general complex argument.
///
/// ## Returns
///
/// Singularities, overflow, and invalid domains return IEEE infinities or NaNs.
/// Returned values never contain backend-specific types.
///
/// ## Numerical behavior
///
/// Uses the binary64 AMOS complex Bessel backend. Backend domain, convergence,
/// overflow, or precision-loss errors produce NaN components. Phase reduction
/// limits large-argument accuracy; the scaled forms avoid the explicit Hankel
/// exponential. No configurable tolerance or iteration control is exposed.
///
/// ## Examples
///
/// ```
/// let value = sci_special::hankel2e(0.5,2.0);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy hankel2e](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hankel2e.html).
pub fn hankel2e(v: f64, x: f64) -> Complex {
    crate::complex::owned(crate::backend::legacy::hankel_2e(v, x))
}

/// Faddeeva function w(z)=exp(−z²) erfc(−iz).
///
/// ## Parameters and domain
///
/// Complex argument with f64 components; NaN and infinities follow analytic limits.
///
/// ## Returns
///
/// Singularities, overflow, and invalid domains return IEEE infinities or NaNs.
/// Returned values never contain backend-specific types.
///
/// ## Examples
///
/// ```
/// let value = sci_special::wofz(sci_complex::Complex { re: 0.5, im: 0.3 });
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy wofz](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.wofz.html).
pub fn wofz(z: Complex) -> Complex {
    crate::complex::owned(crate::backend::legacy::wofz(crate::complex::native(z)))
}

/// Kelvin values (ber+i bei, ker+i kei, berp+i beip, kerp+i keip).
///
/// ## Parameters and domain
///
/// Real x; the second-kind components require positive x.
///
/// ## Returns
///
/// Singularities, overflow, and invalid domains return IEEE infinities or NaNs.
/// Returned values never contain backend-specific types.
///
/// ## Examples
///
/// ```
/// let value = sci_special::kelvin(0.5);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy kelvin](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kelvin.html).
pub fn kelvin(x: f64) -> [Complex; 4] {
    crate::backend::legacy::kelvin(x).map(crate::complex::owned)
}

/// Modified Fresnel integrals (F,K), with the plus/minus phase convention of SciPy.
///
/// ## Parameters and domain
///
/// Real x; complex outputs use f64 components.
///
/// ## Returns
///
/// Singularities, overflow, and invalid domains return IEEE infinities or NaNs.
/// Returned values never contain backend-specific types.
///
/// ## Examples
///
/// ```
/// let value = sci_special::modfresnelp(0.5);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy modfresnelp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.modfresnelp.html).
pub fn modfresnelp(x: f64) -> (Complex, Complex) {
    {
        let (a, b) = crate::backend::legacy::modified_fresnel_plus(x);
        (crate::complex::owned(a), crate::complex::owned(b))
    }
}

/// Modified Fresnel integrals (F,K), with the plus/minus phase convention of SciPy.
///
/// ## Parameters and domain
///
/// Real x; complex outputs use f64 components.
///
/// ## Returns
///
/// Singularities, overflow, and invalid domains return IEEE infinities or NaNs.
/// Returned values never contain backend-specific types.
///
/// ## Examples
///
/// ```
/// let value = sci_special::modfresnelm(0.5);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy modfresnelm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.modfresnelm.html).
pub fn modfresnelm(x: f64) -> (Complex, Complex) {
    {
        let (a, b) = crate::backend::legacy::modified_fresnel_minus(x);
        (crate::complex::owned(a), crate::complex::owned(b))
    }
}

/// First nt complex Fresnel zeros (S zeros,C zeros), in the first quadrant.
///
/// ## Parameters and domain
///
/// nt in 1..=4096.
///
/// ## Returns
///
/// A pair `(s, c)` of complex zero vectors, each containing `nt` entries.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for sizes or parameters outside the stated domain.
///
/// ## Examples
///
/// ```
/// let value = sci_special::fresnel_zeros(3)?;
/// let _ = value;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy fresnel_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.fresnel_zeros.html).
pub fn fresnel_zeros(nt: usize) -> Result<(Vec<Complex>, Vec<Complex>)> {
    if !(nt > 0 && nt <= 4096) {
        return Err(Error::InvalidInput(
            "sequence arguments exceed the supported domain",
        ));
    }
    {
        let (a, b) = crate::backend::legacy::fresnel_zeros(nt).map_err(Error::NumericalFailure)?;
        Ok((
            a.into_iter().map(crate::complex::owned).collect(),
            b.into_iter().map(crate::complex::owned).collect(),
        ))
    }
}

/// First nt first-quadrant zeros of Fresnel S.
///
/// ## Parameters and domain
///
/// nt in 1..=4096.
///
/// ## Returns
///
/// A vector containing `nt` complex zeros of the Fresnel S integral.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for sizes or parameters outside the stated domain.
///
/// ## Examples
///
/// ```
/// let value = sci_special::fresnels_zeros(3)?;
/// let _ = value;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy fresnels_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.fresnels_zeros.html).
pub fn fresnels_zeros(nt: usize) -> Result<Vec<Complex>> {
    if !(nt > 0 && nt <= 4096) {
        return Err(Error::InvalidInput(
            "sequence arguments exceed the supported domain",
        ));
    }
    Ok(fresnel_zeros(nt)?.0)
}

/// First nt first-quadrant zeros of Fresnel C.
///
/// ## Parameters and domain
///
/// nt in 1..=4096.
///
/// ## Returns
///
/// A vector containing `nt` complex zeros of the Fresnel C integral.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for sizes or parameters outside the stated domain.
///
/// ## Examples
///
/// ```
/// let value = sci_special::fresnelc_zeros(3)?;
/// let _ = value;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy fresnelc_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.fresnelc_zeros.html).
pub fn fresnelc_zeros(nt: usize) -> Result<Vec<Complex>> {
    if !(nt > 0 && nt <= 4096) {
        return Err(Error::InvalidInput(
            "sequence arguments exceed the supported domain",
        ));
    }
    Ok(fresnel_zeros(nt)?.1)
}

/// Spherical harmonic Y_n^m(theta,phi), including the Condon–Shortley phase.
///
/// ## Parameters and domain
///
/// Degree n≤128, abs(m)≤n. theta is the polar angle and phi the azimuth, both in radians. Value-only output; derivative axes are omitted.
///
/// ## Returns
///
/// Singularities, overflow, and invalid domains return IEEE infinities or NaNs.
/// Returned values never contain backend-specific types.
///
/// ## Examples
///
/// ```
/// let value = sci_special::sph_harm_y(3,1,0.5,0.3);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy sph_harm_y](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sph_harm_y.html).
pub fn sph_harm_y(n: usize, m: isize, theta: f64, phi: f64) -> Complex {
    if n > 128 || m.unsigned_abs() > n {
        Complex {
            re: f64::NAN,
            im: f64::NAN,
        }
    } else {
        crate::complex::owned(
            crate::backend::sph_harm_y(n as u32, m as i32, theta, phi)
                .unwrap_or(num_complex::Complex64::new(f64::NAN, f64::NAN)),
        )
    }
}

/// Spherical harmonics for degrees 0..=n and orders 0..=m followed by −m..=−1.
///
/// ## Parameters and domain
///
/// n,m≤128. Rows are degrees; columns use SciPy order ordering; angles are radians. Derivative axes are omitted.
///
/// ## Returns
///
/// A complex matrix with `n + 1` degree rows and `2m + 1` order columns.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for sizes or parameters outside the stated domain.
///
/// ## Examples
///
/// ```
/// let value = sci_special::sph_harm_y_all(3,2,0.5,0.3)?;
/// let _ = value;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy sph_harm_y_all](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sph_harm_y_all.html).
pub fn sph_harm_y_all(n: usize, m: usize, theta: f64, phi: f64) -> Result<Vec<Vec<Complex>>> {
    if !(n <= 128 && m <= 128) {
        return Err(Error::InvalidInput(
            "sequence arguments exceed the supported domain",
        ));
    }
    Ok(crate::backend::legacy::sph_harm_y_all(n, m, theta, phi)
        .into_iter()
        .map(|row| row.into_iter().map(crate::complex::owned).collect())
        .collect())
}

/// All Legendre P values through degree n.
///
/// ## Parameters and domain
///
/// n≤128; returns degrees 0..=n with no derivative axis.
///
/// ## Returns
///
/// A vector with `n + 1` entries, in increasing degree order.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for sizes or parameters outside the stated domain.
///
/// ## Examples
///
/// ```
/// let value = sci_special::legendre_p_all(3,0.3)?;
/// let _ = value;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy legendre_p_all](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.legendre_p_all.html).
pub fn legendre_p_all(n: usize, x: f64) -> Result<Vec<f64>> {
    if n > 128 {
        return Err(Error::InvalidInput(
            "sequence arguments exceed the supported domain",
        ));
    }
    Ok(crate::backend::legacy::legendre_p_all(n, x))
}

/// All Legendre P values through degree n and order magnitude m.
///
/// ## Parameters and domain
///
/// n,m≤128; order columns are 0..=m then −m..=−1, and rows are degrees. Value-only semantics; derivative axes are omitted.
///
/// ## Returns
///
/// A real matrix with `n + 1` degree rows and `2m + 1` order columns.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for sizes or parameters outside the stated domain.
///
/// ## Examples
///
/// ```
/// let value = sci_special::assoc_legendre_p_all(3,2,0.3)?;
/// let _ = value;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy assoc_legendre_p_all](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.assoc_legendre_p_all.html).
pub fn assoc_legendre_p_all(n: usize, m: usize, x: f64) -> Result<Vec<Vec<f64>>> {
    if !(n <= 128 && m <= 128) {
        return Err(Error::InvalidInput(
            "sequence arguments exceed the supported domain",
        ));
    }
    Ok(crate::backend::legacy::assoc_legendre_p_all(n, m, x))
}

/// All spherical-normalized Legendre P values through degree n and order magnitude m.
///
/// ## Parameters and domain
///
/// n,m≤128; order columns are 0..=m then −m..=−1, and rows are degrees. Value-only semantics; derivative axes are omitted.
///
/// ## Returns
///
/// A real matrix with `n + 1` degree rows and `2m + 1` order columns.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for sizes or parameters outside the stated domain.
///
/// ## Examples
///
/// ```
/// let value = sci_special::sph_legendre_p_all(3,2,0.3)?;
/// let _ = value;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy sph_legendre_p_all](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sph_legendre_p_all.html).
pub fn sph_legendre_p_all(n: usize, m: usize, x: f64) -> Result<Vec<Vec<f64>>> {
    if !(n <= 128 && m <= 128) {
        return Err(Error::InvalidInput(
            "sequence arguments exceed the supported domain",
        ));
    }
    Ok(crate::backend::legacy::sph_legendre_p_all(n, m, x))
}

/// Even Mathieu Fourier coefficients, in increasing harmonic order.
///
/// ## Parameters and domain
///
/// m≤128; finite 0≤q≤100. The backend limits the number of coefficients to 251.
///
/// ## Returns
///
/// An owned vector of even Mathieu Fourier coefficients in increasing harmonic order.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for sizes or parameters outside the stated domain.
/// Returns [`Error::NumericalFailure`] if coefficient evaluation fails.
///
/// ## Examples
///
/// ```
/// let value = sci_special::mathieu_even_coef(2,0.5)?;
/// let _ = value;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy mathieu_even_coef](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.mathieu_even_coef.html).
pub fn mathieu_even_coef(m: u32, q: f64) -> Result<Vec<f64>> {
    if !(m <= 128 && q.is_finite() && (0.0..=100.0).contains(&q)) {
        return Err(Error::InvalidInput(
            "sequence arguments exceed the supported domain",
        ));
    }
    crate::backend::legacy::mathieu_even_coef(m, q).map_err(|_| {
        crate::Error::NumericalFailure("Mathieu Fourier coefficient evaluation failed")
    })
}

/// Odd Mathieu Fourier coefficients, in increasing harmonic order.
///
/// ## Parameters and domain
///
/// m≤128; finite 0≤q≤100. The backend limits the number of coefficients to 251.
///
/// ## Returns
///
/// An owned vector of odd Mathieu Fourier coefficients in increasing harmonic order.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for sizes or parameters outside the stated domain.
/// Returns [`Error::NumericalFailure`] if coefficient evaluation fails.
///
/// ## Examples
///
/// ```
/// let value = sci_special::mathieu_odd_coef(2,0.5)?;
/// let _ = value;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy mathieu_odd_coef](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.mathieu_odd_coef.html).
pub fn mathieu_odd_coef(m: u32, q: f64) -> Result<Vec<f64>> {
    if !(m <= 128 && q.is_finite() && (0.0..=100.0).contains(&q)) {
        return Err(Error::InvalidInput(
            "sequence arguments exceed the supported domain",
        ));
    }
    crate::backend::legacy::mathieu_odd_coef(m, q).map_err(|_| {
        crate::Error::NumericalFailure("Mathieu Fourier coefficient evaluation failed")
    })
}

/// Prolate spheroidal characteristic values for degrees m..=n.
///
/// ## Parameters and domain
///
/// 0≤m≤n≤128; finite real c with abs(c)≤64.
///
/// ## Returns
///
/// A vector with `n - m + 1` characteristic values, in increasing degree order.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for sizes or parameters outside the stated domain.
///
/// ## Examples
///
/// ```
/// let value = sci_special::pro_cv_seq(1,3,0.5)?;
/// let _ = value;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy pro_cv_seq](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pro_cv_seq.html).
pub fn pro_cv_seq(m: u64, n: u64, c: f64) -> Result<Vec<f64>> {
    if !(m <= n && n <= 128 && c.is_finite() && c.abs() <= 64.0) {
        return Err(Error::InvalidInput(
            "sequence arguments exceed the supported domain",
        ));
    }
    Ok((m..=n).map(|degree| super::pro_cv(m, degree, c)).collect())
}

/// Oblate spheroidal characteristic values for degrees m..=n.
///
/// ## Parameters and domain
///
/// 0≤m≤n≤128; finite real c with abs(c)≤64.
///
/// ## Returns
///
/// A vector with `n - m + 1` characteristic values, in increasing degree order.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for sizes or parameters outside the stated domain.
///
/// ## Examples
///
/// ```
/// let value = sci_special::obl_cv_seq(1,3,0.5)?;
/// let _ = value;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy obl_cv_seq](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.obl_cv_seq.html).
pub fn obl_cv_seq(m: u64, n: u64, c: f64) -> Result<Vec<f64>> {
    if !(m <= n && n <= 128 && c.is_finite() && c.abs() <= 64.0) {
        return Err(Error::InvalidInput(
            "sequence arguments exceed the supported domain",
        ));
    }
    Ok((m..=n).map(|degree| super::obl_cv(m, degree, c)).collect())
}
