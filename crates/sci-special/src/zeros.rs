//! Bracketed real zero sequences and complex Newton iteration.

use crate::{Error, Result, complex::Complex};

fn count(nt: usize) -> Result<()> {
    if nt == 0 || nt > 256 {
        Err(Error::InvalidInput("zero count must be in 1..=256"))
    } else {
        Ok(())
    }
}

fn root(function: &impl Fn(f64) -> f64, mut a: f64, mut b: f64) -> Result<f64> {
    let mut fa = function(a);
    if !fa.is_finite() || !function(b).is_finite() {
        return Err(Error::NonFiniteEvaluation);
    }
    for _ in 0..100 {
        let x = a + (b - a) * 0.5;
        let f = function(x);
        if !f.is_finite() {
            return Err(Error::NonFiniteEvaluation);
        }
        if f == 0.0 || (b - a).abs() <= 4.0 * f64::EPSILON * x.abs().max(1.0) {
            return Ok(x);
        }
        if f.is_sign_positive() == fa.is_sign_positive() {
            a = x;
            fa = f;
        } else {
            b = x;
        }
    }
    Err(Error::NumericalFailure("zero refinement did not converge"))
}

fn airy_roots(nt: usize, component: usize) -> Result<Vec<f64>> {
    count(nt)?;
    let f = |x: f64| super::airy(x)[component];
    let mut near = 0.0;
    let mut values = Vec::with_capacity(nt);
    for _ in 0..nt * 64 + 64 {
        let far = near - std::f64::consts::FRAC_PI_4 / (1.0_f64 - near).sqrt();
        if f(far).is_sign_positive() != f(near).is_sign_positive() {
            values.push(root(&f, far, near)?);
            if values.len() == nt {
                return Ok(values);
            }
        }
        near = far;
    }
    Err(Error::NumericalFailure(
        "unable to bracket requested Airy zeros",
    ))
}

/// First nt positive zeros of J_n(x), ascending.
///
/// ## Parameters and domain
///
/// 0≤n≤256 and 1≤nt≤256; zero at the origin is excluded.
///
/// ## Errors
///
/// Invalid sizes or orders return [`Error::InvalidInput`]; nonfinite evaluation
/// returns [`Error::NonFiniteEvaluation`], and failed bracketing or convergence
/// returns [`Error::NumericalFailure`]. All iterations have finite work limits.
///
/// ## Examples
///
/// ```
/// let zeros = sci_special::bessel::jn_zeros(2,3)?;
/// let _ = zeros;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy jn_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jn_zeros.html).
pub fn jn_zeros(n: u32, nt: usize) -> Result<Vec<f64>> {
    count(nt)?;
    if n > 256 {
        return Err(Error::InvalidInput("Bessel order must not exceed 256"));
    }
    ::bessel::jn_zeros(n, nt).map_err(super::external_bessel_error)
}

/// First nt positive zeros of J_n′(x), ascending.
///
/// ## Parameters and domain
///
/// 0≤n≤256 and 1≤nt≤256; zero at the origin is excluded.
///
/// ## Errors
///
/// Invalid sizes or orders return [`Error::InvalidInput`]; nonfinite evaluation
/// returns [`Error::NonFiniteEvaluation`], and failed bracketing or convergence
/// returns [`Error::NumericalFailure`]. All iterations have finite work limits.
///
/// ## Examples
///
/// ```
/// let zeros = sci_special::bessel::jnp_zeros(2,3)?;
/// let _ = zeros;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy jnp_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jnp_zeros.html).
pub fn jnp_zeros(n: u32, nt: usize) -> Result<Vec<f64>> {
    count(nt)?;
    if n > 256 {
        return Err(Error::InvalidInput("Bessel order must not exceed 256"));
    }
    ::bessel::jnp_zeros(n, nt).map_err(super::external_bessel_error)
}

/// First nt positive zeros of Y_n(x), ascending.
///
/// ## Parameters and domain
///
/// 0≤n≤256 and 1≤nt≤256; zero at the origin is excluded.
///
/// ## Errors
///
/// Invalid sizes or orders return [`Error::InvalidInput`]; nonfinite evaluation
/// returns [`Error::NonFiniteEvaluation`], and failed bracketing or convergence
/// returns [`Error::NumericalFailure`]. All iterations have finite work limits.
///
/// ## Examples
///
/// ```
/// let zeros = sci_special::bessel::yn_zeros(2,3)?;
/// let _ = zeros;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy yn_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.yn_zeros.html).
pub fn yn_zeros(n: u32, nt: usize) -> Result<Vec<f64>> {
    count(nt)?;
    if n > 256 {
        return Err(Error::InvalidInput("Bessel order must not exceed 256"));
    }
    ::bessel::yn_zeros(n, nt).map_err(super::external_bessel_error)
}

/// First nt positive zeros of Y_n′(x), ascending.
///
/// ## Parameters and domain
///
/// 0≤n≤256 and 1≤nt≤256; zero at the origin is excluded.
///
/// ## Errors
///
/// Invalid sizes or orders return [`Error::InvalidInput`]; nonfinite evaluation
/// returns [`Error::NonFiniteEvaluation`], and failed bracketing or convergence
/// returns [`Error::NumericalFailure`]. All iterations have finite work limits.
///
/// ## Examples
///
/// ```
/// let zeros = sci_special::bessel::ynp_zeros(2,3)?;
/// let _ = zeros;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy ynp_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ynp_zeros.html).
pub fn ynp_zeros(n: u32, nt: usize) -> Result<Vec<f64>> {
    count(nt)?;
    if n > 256 {
        return Err(Error::InvalidInput("Bessel order must not exceed 256"));
    }
    ::bessel::ynp_zeros(n, nt).map_err(super::external_bessel_error)
}

/// First nt positive zeros of J_n, J_n′, Y_n, and Y_n′, in that order.
///
/// ## Parameters and domain
///
/// 0≤n≤256 and 1≤nt≤256; zeros at the origin are excluded.
///
/// ## Errors
///
/// Invalid sizes or orders return [`Error::InvalidInput`]; nonfinite evaluation
/// returns [`Error::NonFiniteEvaluation`], and failed bracketing or convergence
/// returns [`Error::NumericalFailure`]. All iterations have finite work limits.
///
/// ## Examples
///
/// ```
/// let zeros = sci_special::bessel::jnyn_zeros(2,3)?;
/// let _ = zeros;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy jnyn_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jnyn_zeros.html).
pub fn jnyn_zeros(n: u32, nt: usize) -> Result<[Vec<f64>; 4]> {
    count(nt)?;
    if n > 256 {
        return Err(Error::InvalidInput("Bessel order must not exceed 256"));
    }
    ::bessel::jnyn_zeros(n, nt).map_err(super::external_bessel_error)
}

/// Positive real Y₀ zeros and Y₁ values at the zeros.
///
/// ## Parameters and domain
///
/// 1≤nt≤256. SciPy's optional complex-root mode is omitted; outputs are real f64 vectors.
///
/// ## Errors
///
/// Invalid sizes or orders return [`Error::InvalidInput`]; nonfinite evaluation
/// returns [`Error::NonFiniteEvaluation`], and failed bracketing or convergence
/// returns [`Error::NumericalFailure`]. All iterations have finite work limits.
///
/// ## Examples
///
/// ```
/// let zeros = sci_special::bessel::y0_zeros(3)?;
/// let _ = zeros;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy y0_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.y0_zeros.html).
pub fn y0_zeros(nt: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    count(nt)?;
    ::bessel::y0_zeros(nt).map_err(super::external_bessel_error)
}

/// Positive real Y1 zeros and derivatives at the zeros.
///
/// ## Parameters and domain
///
/// 1≤nt≤256. SciPy's optional complex-root mode is omitted; outputs are real f64 vectors.
///
/// ## Errors
///
/// Invalid sizes or orders return [`Error::InvalidInput`]; nonfinite evaluation
/// returns [`Error::NonFiniteEvaluation`], and failed bracketing or convergence
/// returns [`Error::NumericalFailure`]. All iterations have finite work limits.
///
/// ## Examples
///
/// ```
/// let zeros = sci_special::bessel::y1_zeros(3)?;
/// let _ = zeros;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy y1_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.y1_zeros.html).
pub fn y1_zeros(nt: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    count(nt)?;
    ::bessel::y1_zeros(nt).map_err(super::external_bessel_error)
}

/// Positive real Y₁′ zeros and Y₁ values at the zeros.
///
/// ## Parameters and domain
///
/// 1≤nt≤256. SciPy's optional complex-root mode is omitted; outputs are real f64 vectors.
///
/// ## Errors
///
/// Invalid sizes or orders return [`Error::InvalidInput`]; nonfinite evaluation
/// returns [`Error::NonFiniteEvaluation`], and failed bracketing or convergence
/// returns [`Error::NumericalFailure`]. All iterations have finite work limits.
///
/// ## Examples
///
/// ```
/// let zeros = sci_special::bessel::y1p_zeros(3)?;
/// let _ = zeros;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy y1p_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.y1p_zeros.html).
pub fn y1p_zeros(nt: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    count(nt)?;
    ::bessel::y1p_zeros(nt).map_err(super::external_bessel_error)
}

/// Negative real Ai zeros, derivative zeros, values at derivative zeros, and derivative values at zeros.
///
/// ## Parameters and domain
///
/// 1≤nt≤256; zeros are ordered from nearest to farthest from the origin.
///
/// ## Errors
///
/// Invalid sizes or orders return [`Error::InvalidInput`]; nonfinite evaluation
/// returns [`Error::NonFiniteEvaluation`], and failed bracketing or convergence
/// returns [`Error::NumericalFailure`]. All iterations have finite work limits.
///
/// ## Examples
///
/// ```
/// let zeros = sci_special::wave::ai_zeros(3)?;
/// let _ = zeros;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy ai_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ai_zeros.html).
pub fn ai_zeros(nt: usize) -> Result<[Vec<f64>; 4]> {
    let zeros = airy_roots(nt, 0)?;
    let prime = airy_roots(nt, 1)?;
    let values = prime.iter().map(|&x| super::airy(x)[0]).collect();
    let derivatives = zeros.iter().map(|&x| super::airy(x)[1]).collect();
    Ok([zeros, prime, values, derivatives])
}

/// Negative real Bi zeros, derivative zeros, values at derivative zeros, and derivative values at zeros.
///
/// ## Parameters and domain
///
/// 1≤nt≤256; zeros are ordered from nearest to farthest from the origin.
///
/// ## Errors
///
/// Invalid sizes or orders return [`Error::InvalidInput`]; nonfinite evaluation
/// returns [`Error::NonFiniteEvaluation`], and failed bracketing or convergence
/// returns [`Error::NumericalFailure`]. All iterations have finite work limits.
///
/// ## Examples
///
/// ```
/// let zeros = sci_special::wave::bi_zeros(3)?;
/// let _ = zeros;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy bi_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.bi_zeros.html).
pub fn bi_zeros(nt: usize) -> Result<[Vec<f64>; 4]> {
    let zeros = airy_roots(nt, 2)?;
    let prime = airy_roots(nt, 3)?;
    let values = prime.iter().map(|&x| super::airy(x)[2]).collect();
    let derivatives = zeros.iter().map(|&x| super::airy(x)[3]).collect();
    Ok([zeros, prime, values, derivatives])
}

/// First nt nonzero first-quadrant zeros of erf(z), refined using complex Newton iteration.
///
/// ## Parameters and domain
///
/// 1≤nt≤256; conjugate and negative-axis partners are omitted.
///
/// ## Errors
///
/// Invalid sizes or orders return [`Error::InvalidInput`]; nonfinite evaluation
/// returns [`Error::NonFiniteEvaluation`], and failed bracketing or convergence
/// returns [`Error::NumericalFailure`]. All iterations have finite work limits.
///
/// ## Examples
///
/// ```
/// let zeros = sci_special::elementary::erf_zeros(3)?;
/// let _ = zeros;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy erf_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erf_zeros.html).
pub fn erf_zeros(nt: usize) -> Result<Vec<Complex>> {
    count(nt)?;
    let mut out = Vec::with_capacity(nt);
    for k in 1..=nt {
        let k = k as f64;
        let u = (std::f64::consts::PI * (4.0 * k - 0.5)).sqrt();
        let correction = (std::f64::consts::PI * (2.0 * k - 0.25).sqrt()).ln() / (2.0 * u);
        let mut z = num_complex::Complex64::new(u / 2.0 - correction, u / 2.0 + correction);
        let mut converged = false;
        for _ in 0..100 {
            let f = crate::backend::complex::erf(z).map_err(Error::NumericalFailure)?;
            let derivative = 2.0 / std::f64::consts::PI.sqrt() * (-z * z).exp();
            let step = f / derivative;
            z -= step;
            if !z.re.is_finite() || !z.im.is_finite() {
                break;
            }
            if step.norm() < 2e-14 * z.norm() {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(Error::NumericalFailure(
                "complex error-function zero did not converge",
            ));
        }
        out.push(crate::complex::owned(z));
    }
    Ok(out)
}

/// Merged zeros of integer-order J_n and J_n′, with order, serial index, and derivative flag.
///
/// ## Parameters and domain
///
/// 1≤nt≤128. Returns `[zeros,orders,serials,flags]`, all as f64 vectors for module interoperability. The origin entry is included.
///
/// ## Errors
///
/// Invalid sizes or orders return [`Error::InvalidInput`]; nonfinite evaluation
/// returns [`Error::NonFiniteEvaluation`], and failed bracketing or convergence
/// returns [`Error::NumericalFailure`]. All iterations have finite work limits.
///
/// ## Examples
///
/// ```
/// let zeros = sci_special::bessel::jnjnp_zeros(8)?;
/// let _ = zeros;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy jnjnp_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jnjnp_zeros.html).
pub fn jnjnp_zeros(nt: usize) -> Result<[Vec<f64>; 4]> {
    count(nt)?;
    // The first merged entry is J_0'(0) = 0, with order/index 0 and flag 1.
    // Avoid the general backend's root-search setup for this exact case.
    if nt == 1 {
        return Ok([vec![0.0], vec![0.0], vec![0.0], vec![1.0]]);
    }
    ::bessel::jnjnp_zeros(nt).map_err(super::external_bessel_error)
}

#[cfg(test)]
mod tests {
    #[test]
    fn first_merged_bessel_zero_matches_original_backend_bits() {
        let actual = super::jnjnp_zeros(1).unwrap();
        let original = ::bessel::jnjnp_zeros(1).unwrap();
        for (component, (&expected, values)) in [0.0_f64, 0.0, 0.0, 1.0]
            .iter()
            .zip(actual.iter())
            .enumerate()
        {
            assert_eq!(values.len(), 1, "component {component}");
            assert_eq!(
                values[0].to_bits(),
                expected.to_bits(),
                "component {component}"
            );
            assert_eq!(values[0].to_bits(), original[component][0].to_bits());
        }
    }
}
