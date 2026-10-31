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

fn bessel_roots(n: u32, nt: usize, kind: u8) -> Result<Vec<f64>> {
    count(nt)?;
    if n > 256 {
        return Err(Error::InvalidInput("Bessel order must not exceed 256"));
    }
    let v = f64::from(n);
    let f = |x| match kind {
        0 => super::jv(v, x),
        1 => super::jvp(v, x, 1),
        2 => super::yv(v, x),
        _ => super::yvp(v, x, 1),
    };
    let mut a = v.max(0.01);
    let mut fa = f(a);
    let mut zeros = Vec::with_capacity(nt);
    for _ in 0..(nt + 256) * 32 {
        let b = a + std::f64::consts::FRAC_PI_4;
        let fb = f(b);
        if fa.is_finite()
            && fb.is_finite()
            && fa != 0.0
            && fa.is_sign_positive() != fb.is_sign_positive()
        {
            zeros.push(root(&f, a, b)?);
            if zeros.len() == nt {
                return Ok(zeros);
            }
        }
        a = b;
        fa = fb;
    }
    Err(Error::NumericalFailure(
        "unable to bracket requested Bessel zeros",
    ))
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
/// let zeros = sci_special::jn_zeros(2,3)?;
/// let _ = zeros;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy jn_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jn_zeros.html).
pub fn jn_zeros(n: u32, nt: usize) -> Result<Vec<f64>> {
    bessel_roots(n, nt, 0)
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
/// let zeros = sci_special::jnp_zeros(2,3)?;
/// let _ = zeros;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy jnp_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jnp_zeros.html).
pub fn jnp_zeros(n: u32, nt: usize) -> Result<Vec<f64>> {
    bessel_roots(n, nt, 1)
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
/// let zeros = sci_special::yn_zeros(2,3)?;
/// let _ = zeros;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy yn_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.yn_zeros.html).
pub fn yn_zeros(n: u32, nt: usize) -> Result<Vec<f64>> {
    bessel_roots(n, nt, 2)
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
/// let zeros = sci_special::ynp_zeros(2,3)?;
/// let _ = zeros;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy ynp_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ynp_zeros.html).
pub fn ynp_zeros(n: u32, nt: usize) -> Result<Vec<f64>> {
    bessel_roots(n, nt, 3)
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
/// let zeros = sci_special::jnyn_zeros(2,3)?;
/// let _ = zeros;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy jnyn_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jnyn_zeros.html).
pub fn jnyn_zeros(n: u32, nt: usize) -> Result<[Vec<f64>; 4]> {
    Ok([
        bessel_roots(n, nt, 0)?,
        bessel_roots(n, nt, 1)?,
        bessel_roots(n, nt, 2)?,
        bessel_roots(n, nt, 3)?,
    ])
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
/// let zeros = sci_special::y0_zeros(3)?;
/// let _ = zeros;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy y0_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.y0_zeros.html).
pub fn y0_zeros(nt: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    let zeros = bessel_roots(0, nt, 2)?;
    let values = zeros.iter().map(|&x| super::yv(1.0, x)).collect();
    Ok((zeros, values))
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
/// let zeros = sci_special::y1_zeros(3)?;
/// let _ = zeros;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy y1_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.y1_zeros.html).
pub fn y1_zeros(nt: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    let zeros = bessel_roots(1, nt, 2)?;
    let values = zeros.iter().map(|&x| super::yvp(1.0, x, 1)).collect();
    Ok((zeros, values))
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
/// let zeros = sci_special::y1p_zeros(3)?;
/// let _ = zeros;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy y1p_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.y1p_zeros.html).
pub fn y1p_zeros(nt: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    let zeros = bessel_roots(1, nt, 3)?;
    let values = zeros.iter().map(|&x| super::yv(1.0, x)).collect();
    Ok((zeros, values))
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
/// let zeros = sci_special::ai_zeros(3)?;
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
/// let zeros = sci_special::bi_zeros(3)?;
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
/// let zeros = sci_special::erf_zeros(3)?;
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
/// let zeros = sci_special::jnjnp_zeros(8)?;
/// let _ = zeros;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy jnjnp_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jnjnp_zeros.html).
pub fn jnjnp_zeros(nt: usize) -> Result<[Vec<f64>; 4]> {
    count(nt)?;
    if nt > 128 {
        return Err(Error::InvalidInput("merged zero count must not exceed 128"));
    }
    let mut entries = vec![(0.0, 0.0, 0.0, 1.0)];
    let cutoff = (nt as f64 * 8.0).sqrt() + 4.0;
    for n in 0..=cutoff.ceil() as u32 {
        for kind in 0..=1 {
            let roots = bessel_roots(n, nt, kind)?;
            for (index, x) in roots.into_iter().enumerate() {
                entries.push((x, f64::from(n), (index + 1) as f64, f64::from(kind)));
            }
        }
    }
    entries.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| b.3.total_cmp(&a.3)));
    // J₀′ = −J₁: sort these analytically identical roots by function first.
    let mut start = 0;
    while start < entries.len() {
        let mut end = start + 1;
        while end < entries.len() && (entries[end].0 - entries[start].0).abs() <= 1e-12 {
            end += 1;
        }
        entries[start..end].sort_by(|a, b| a.3.total_cmp(&b.3));
        start = end;
    }
    let mut out: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::with_capacity(nt));
    for (x, n, index, flag) in entries.into_iter().take(nt) {
        out[0].push(x);
        out[1].push(n);
        out[2].push(index);
        out[3].push(flag);
    }
    Ok(out)
}
