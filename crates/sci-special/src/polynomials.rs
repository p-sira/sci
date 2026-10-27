//! Polynomial coefficients and Gaussian rules use the same three-term recurrence.

use crate::{Error, Result};
use nalgebra::{DMatrix, SymmetricEigen};

pub(super) fn eigensystem(diagonal: &[f64], off: &[f64]) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
    let n = diagonal.len();
    let mut matrix = DMatrix::zeros(n, n);
    for i in 0..n {
        matrix[(i, i)] = diagonal[i];
        if i + 1 < n {
            matrix[(i, i + 1)] = off[i];
            matrix[(i + 1, i)] = off[i];
        }
    }
    if !matrix.iter().all(|x| x.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    let eigen = SymmetricEigen::try_new(matrix, 1e-15, 100_000).ok_or(Error::NumericalFailure(
        "symmetric eigensolver did not converge",
    ))?;
    let mut indices: Vec<_> = (0..n).collect();
    indices.sort_by(|&a, &b| eigen.eigenvalues[a].total_cmp(&eigen.eigenvalues[b]));
    Ok((
        indices.iter().map(|&i| eigen.eigenvalues[i]).collect(),
        indices
            .iter()
            .map(|&i| eigen.eigenvectors.column(i).iter().copied().collect())
            .collect(),
    ))
}

// Coefficients are ascending internally. The public representation is descending.
fn advance(previous: &[f64], before: &[f64], x: f64, constant: f64, subtract: f64) -> Vec<f64> {
    let mut out = vec![0.0; previous.len() + 1];
    for (i, &value) in previous.iter().enumerate() {
        out[i] += constant * value;
        out[i + 1] += x * value;
    }
    for (i, &value) in before.iter().enumerate() {
        out[i] -= subtract * value;
    }
    out
}

#[derive(Clone, Copy)]
enum Family {
    Jacobi(f64, f64),
    Laguerre(f64),
    Hermite(bool),
    Chebyt,
    Chebyu,
    Gegenbauer(f64),
}

fn coefficients(
    n: usize,
    family: Family,
    monic: bool,
    scale: f64,
    shift: f64,
    factor: f64,
) -> Result<Vec<f64>> {
    if n > 256 {
        return Err(Error::InvalidInput("polynomial degree must not exceed 256"));
    }
    let mut prev = vec![1.0];
    let mut before = vec![];
    for k in 1..=n {
        let kf = k as f64;
        let (x, c, b) = match family {
            Family::Jacobi(a, b) if k == 1 => ((a + b + 2.0) / 2.0, (a - b) / 2.0, 0.0),
            Family::Jacobi(a, b) => {
                let t = 2.0 * kf + a + b;
                let denominator = 2.0 * kf * (kf + a + b) * (t - 2.0);
                (
                    (t - 1.0) * t * (t - 2.0) / denominator,
                    (t - 1.0) * (a * a - b * b) / denominator,
                    2.0 * (kf + a - 1.0) * (kf + b - 1.0) * t / denominator,
                )
            }
            Family::Laguerre(a) => (-1.0 / kf, (2.0 * kf - 1.0 + a) / kf, (kf - 1.0 + a) / kf),
            Family::Hermite(true) => (1.0, 0.0, kf - 1.0),
            Family::Hermite(false) => (2.0, 0.0, 2.0 * (kf - 1.0)),
            Family::Chebyt if k == 1 => (1.0, 0.0, 0.0),
            Family::Chebyt | Family::Chebyu => (2.0, 0.0, 1.0),
            Family::Gegenbauer(a) => (2.0 * (kf - 1.0 + a) / kf, 0.0, (kf - 2.0 + 2.0 * a) / kf),
        };
        let next = advance(&prev, &before, x, c, b);
        before = prev;
        prev = next;
    }
    // Horner composition evaluates p(scale*x+shift) symbolically.
    let mut composed = vec![0.0];
    for &coefficient in prev.iter().rev() {
        composed = advance(&composed, &[], scale, shift, 0.0);
        composed[0] += coefficient;
    }
    composed.truncate(n + 1);
    let divisor = if monic { composed[n] } else { 1.0 / factor };
    for c in &mut composed {
        *c /= divisor;
    }
    if composed.iter().any(|x| !x.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    composed.reverse();
    Ok(composed)
}

fn quadrature(
    n: usize,
    family: Family,
    scale: f64,
    shift: f64,
    weight_scale: f64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if n == 0 || n > 256 {
        return Err(Error::InvalidInput("quadrature order must be in 1..=256"));
    }
    let mut diagonal = Vec::with_capacity(n);
    let mut off = Vec::with_capacity(n - 1);
    let mass = match family {
        Family::Jacobi(a, b) => (2.0_f64).powf(a + b + 1.0) * super::beta(a + 1.0, b + 1.0),
        Family::Laguerre(a) => super::gamma(a + 1.0),
        Family::Hermite(true) => (2.0 * std::f64::consts::PI).sqrt(),
        Family::Hermite(false) => std::f64::consts::PI.sqrt(),
        _ => unreachable!(),
    };
    for i in 0..n {
        let k = i as f64;
        diagonal.push(match family {
            Family::Jacobi(a, b) if i == 0 => (b - a) / (a + b + 2.0),
            Family::Jacobi(a, b) => (b * b - a * a) / ((2.0 * k + a + b) * (2.0 * k + a + b + 2.0)),
            Family::Laguerre(a) => 2.0 * k + a + 1.0,
            _ => 0.0,
        });
        if i > 0 {
            off.push(match family {
                Family::Jacobi(a, b) if i == 1 => {
                    (4.0 * (1.0 + a) * (1.0 + b) / ((a + b + 2.0).powi(2) * (a + b + 3.0))).sqrt()
                }
                Family::Jacobi(a, b) => {
                    let t = 2.0 * k + a + b;
                    (4.0 * k * (k + a) * (k + b) * (k + a + b) / (t * t * (t * t - 1.0))).sqrt()
                }
                Family::Laguerre(a) => (k * (k + a)).sqrt(),
                Family::Hermite(true) => k.sqrt(),
                Family::Hermite(false) => (k / 2.0).sqrt(),
                _ => unreachable!(),
            });
        }
    }
    let (nodes, vectors) = eigensystem(&diagonal, &off)?;
    let weights: Vec<_> = vectors
        .iter()
        .map(|v| v[0] * v[0] * mass * weight_scale)
        .collect();
    if weights.iter().any(|x| !x.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    Ok((
        nodes.into_iter().map(|x| scale * x + shift).collect(),
        weights,
    ))
}

/// Returns coefficients of the Jacobi P_n^(alpha,beta) polynomial in descending powers.
///
/// ## Parameters and domain
///
/// `n` is the integer degree, at most 256. `monic=true` divides by the leading
/// coefficient. Parameters must make the weight (1−x)^alpha(1+x)^beta on `[−1,1]` integrable.
/// Gegenbauer parameter zero has a vanishing polynomial for n>0 and cannot be made monic.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid parameters or degree;
/// [`Error::NonFiniteEvaluation`] if coefficients overflow or normalization is singular.
/// Coefficients are ordinary vectors and can be evaluated with Horner's rule.
///
/// ## Examples
///
/// ```
/// let coefficients = sci_special::jacobi(3,0.5,0.3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy jacobi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jacobi.html).
pub fn jacobi(n: usize, alpha: f64, beta: f64, monic: bool) -> Result<Vec<f64>> {
    if !(alpha.is_finite() && beta.is_finite() && alpha > -1.0 && beta > -1.0) {
        return Err(Error::InvalidInput(
            "orthogonal polynomial parameters are outside the domain",
        ));
    }
    coefficients(n, Family::Jacobi(alpha, beta), monic, 1.0, 0.0, 1.0)
}

/// Gaussian nodes and weights for (1−x)^alpha(1+x)^beta on `[−1,1]`.
///
/// ## Parameters and domain
///
/// `n` is the number of nodes in `1..=256`; shape parameters must define an
/// integrable positive weight. Nodes are sorted ascending. The weights integrate
/// weighted polynomials exactly through degree 2n−1, up to floating-point error.
///
/// ## Errors
///
/// Invalid orders and parameters return [`Error::InvalidInput`]. Overflow or
/// eigensolver failure returns [`Error::NonFiniteEvaluation`] or [`Error::NumericalFailure`].
/// The total weight (SciPy's optional `mu`) is available by summing the returned weights.
///
/// ## Examples
///
/// ```
/// let (nodes, weights) = sci_special::roots_jacobi(3,0.5,0.3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_jacobi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_jacobi.html).
pub fn roots_jacobi(n: usize, alpha: f64, beta: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    if !(alpha.is_finite() && beta.is_finite() && alpha > -1.0 && beta > -1.0) {
        return Err(Error::InvalidInput(
            "quadrature parameters are outside the domain",
        ));
    }
    quadrature(n, Family::Jacobi(alpha, beta), 1.0, 0.0, 1.0)
}

/// Returns coefficients of the Legendre P_n polynomial in descending powers.
///
/// ## Parameters and domain
///
/// `n` is the integer degree, at most 256. `monic=true` divides by the leading
/// coefficient. Parameters must make the weight unit weight on `[−1,1]` integrable.
/// Gegenbauer parameter zero has a vanishing polynomial for n>0 and cannot be made monic.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid parameters or degree;
/// [`Error::NonFiniteEvaluation`] if coefficients overflow or normalization is singular.
/// Coefficients are ordinary vectors and can be evaluated with Horner's rule.
///
/// ## Examples
///
/// ```
/// let coefficients = sci_special::legendre(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy legendre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.legendre.html).
pub fn legendre(n: usize, monic: bool) -> Result<Vec<f64>> {
    if !(true) {
        return Err(Error::InvalidInput(
            "orthogonal polynomial parameters are outside the domain",
        ));
    }
    coefficients(n, Family::Jacobi(0.0, 0.0), monic, 1.0, 0.0, 1.0)
}

/// Gaussian nodes and weights for unit weight on `[−1,1]`.
///
/// ## Parameters and domain
///
/// `n` is the number of nodes in `1..=256`; shape parameters must define an
/// integrable positive weight. Nodes are sorted ascending. The weights integrate
/// weighted polynomials exactly through degree 2n−1, up to floating-point error.
///
/// ## Errors
///
/// Invalid orders and parameters return [`Error::InvalidInput`]. Overflow or
/// eigensolver failure returns [`Error::NonFiniteEvaluation`] or [`Error::NumericalFailure`].
/// The total weight (SciPy's optional `mu`) is available by summing the returned weights.
///
/// ## Examples
///
/// ```
/// let (nodes, weights) = sci_special::roots_legendre(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_legendre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_legendre.html).
pub fn roots_legendre(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if !(true) {
        return Err(Error::InvalidInput(
            "quadrature parameters are outside the domain",
        ));
    }
    quadrature(n, Family::Jacobi(0.0, 0.0), 1.0, 0.0, 1.0)
}

/// Returns coefficients of the Gegenbauer C_n^alpha polynomial in descending powers.
///
/// ## Parameters and domain
///
/// `n` is the integer degree, at most 256. `monic=true` divides by the leading
/// coefficient. Parameters must make the weight (1−x²)^(alpha−1/2) on `[−1,1]` integrable.
/// Gegenbauer parameter zero has a vanishing polynomial for n>0 and cannot be made monic.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid parameters or degree;
/// [`Error::NonFiniteEvaluation`] if coefficients overflow or normalization is singular.
/// Coefficients are ordinary vectors and can be evaluated with Horner's rule.
///
/// ## Examples
///
/// ```
/// let coefficients = sci_special::gegenbauer(3,0.5,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy gegenbauer](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gegenbauer.html).
pub fn gegenbauer(n: usize, alpha: f64, monic: bool) -> Result<Vec<f64>> {
    if !(alpha.is_finite() && alpha > -0.5) {
        return Err(Error::InvalidInput(
            "orthogonal polynomial parameters are outside the domain",
        ));
    }
    coefficients(n, Family::Gegenbauer(alpha), monic, 1.0, 0.0, 1.0)
}

/// Gaussian nodes and weights for (1−x²)^(alpha−1/2) on `[−1,1]`.
///
/// ## Parameters and domain
///
/// `n` is the number of nodes in `1..=256`; shape parameters must define an
/// integrable positive weight. Nodes are sorted ascending. The weights integrate
/// weighted polynomials exactly through degree 2n−1, up to floating-point error.
///
/// ## Errors
///
/// Invalid orders and parameters return [`Error::InvalidInput`]. Overflow or
/// eigensolver failure returns [`Error::NonFiniteEvaluation`] or [`Error::NumericalFailure`].
/// The total weight (SciPy's optional `mu`) is available by summing the returned weights.
///
/// ## Examples
///
/// ```
/// let (nodes, weights) = sci_special::roots_gegenbauer(3,0.5)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_gegenbauer](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_gegenbauer.html).
pub fn roots_gegenbauer(n: usize, alpha: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    if !(alpha.is_finite() && alpha > -0.5) {
        return Err(Error::InvalidInput(
            "quadrature parameters are outside the domain",
        ));
    }
    quadrature(n, Family::Jacobi(alpha - 0.5, alpha - 0.5), 1.0, 0.0, 1.0)
}

/// Returns coefficients of the generalized Laguerre L_n^alpha polynomial in descending powers.
///
/// ## Parameters and domain
///
/// `n` is the integer degree, at most 256. `monic=true` divides by the leading
/// coefficient. Parameters must make the weight x^alpha exp(−x) on [0,∞) integrable.
/// Gegenbauer parameter zero has a vanishing polynomial for n>0 and cannot be made monic.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid parameters or degree;
/// [`Error::NonFiniteEvaluation`] if coefficients overflow or normalization is singular.
/// Coefficients are ordinary vectors and can be evaluated with Horner's rule.
///
/// ## Examples
///
/// ```
/// let coefficients = sci_special::genlaguerre(3,0.5,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy genlaguerre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.genlaguerre.html).
pub fn genlaguerre(n: usize, alpha: f64, monic: bool) -> Result<Vec<f64>> {
    if !(alpha.is_finite() && alpha > -1.0) {
        return Err(Error::InvalidInput(
            "orthogonal polynomial parameters are outside the domain",
        ));
    }
    coefficients(n, Family::Laguerre(alpha), monic, 1.0, 0.0, 1.0)
}

/// Gaussian nodes and weights for x^alpha exp(−x) on [0,∞).
///
/// ## Parameters and domain
///
/// `n` is the number of nodes in `1..=256`; shape parameters must define an
/// integrable positive weight. Nodes are sorted ascending. The weights integrate
/// weighted polynomials exactly through degree 2n−1, up to floating-point error.
///
/// ## Errors
///
/// Invalid orders and parameters return [`Error::InvalidInput`]. Overflow or
/// eigensolver failure returns [`Error::NonFiniteEvaluation`] or [`Error::NumericalFailure`].
/// The total weight (SciPy's optional `mu`) is available by summing the returned weights.
///
/// ## Examples
///
/// ```
/// let (nodes, weights) = sci_special::roots_genlaguerre(3,0.5)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_genlaguerre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_genlaguerre.html).
pub fn roots_genlaguerre(n: usize, alpha: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    if !(alpha.is_finite() && alpha > -1.0) {
        return Err(Error::InvalidInput(
            "quadrature parameters are outside the domain",
        ));
    }
    quadrature(n, Family::Laguerre(alpha), 1.0, 0.0, 1.0)
}

/// Returns coefficients of the Laguerre L_n polynomial in descending powers.
///
/// ## Parameters and domain
///
/// `n` is the integer degree, at most 256. `monic=true` divides by the leading
/// coefficient. Parameters must make the weight exp(−x) on [0,∞) integrable.
/// Gegenbauer parameter zero has a vanishing polynomial for n>0 and cannot be made monic.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid parameters or degree;
/// [`Error::NonFiniteEvaluation`] if coefficients overflow or normalization is singular.
/// Coefficients are ordinary vectors and can be evaluated with Horner's rule.
///
/// ## Examples
///
/// ```
/// let coefficients = sci_special::laguerre(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy laguerre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.laguerre.html).
pub fn laguerre(n: usize, monic: bool) -> Result<Vec<f64>> {
    if !(true) {
        return Err(Error::InvalidInput(
            "orthogonal polynomial parameters are outside the domain",
        ));
    }
    coefficients(n, Family::Laguerre(0.0), monic, 1.0, 0.0, 1.0)
}

/// Gaussian nodes and weights for exp(−x) on [0,∞).
///
/// ## Parameters and domain
///
/// `n` is the number of nodes in `1..=256`; shape parameters must define an
/// integrable positive weight. Nodes are sorted ascending. The weights integrate
/// weighted polynomials exactly through degree 2n−1, up to floating-point error.
///
/// ## Errors
///
/// Invalid orders and parameters return [`Error::InvalidInput`]. Overflow or
/// eigensolver failure returns [`Error::NonFiniteEvaluation`] or [`Error::NumericalFailure`].
/// The total weight (SciPy's optional `mu`) is available by summing the returned weights.
///
/// ## Examples
///
/// ```
/// let (nodes, weights) = sci_special::roots_laguerre(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_laguerre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_laguerre.html).
pub fn roots_laguerre(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if !(true) {
        return Err(Error::InvalidInput(
            "quadrature parameters are outside the domain",
        ));
    }
    quadrature(n, Family::Laguerre(0.0), 1.0, 0.0, 1.0)
}

/// Returns coefficients of the physicists' Hermite H_n polynomial in descending powers.
///
/// ## Parameters and domain
///
/// `n` is the integer degree, at most 256. `monic=true` divides by the leading
/// coefficient. Parameters must make the weight exp(−x²) on (−∞,∞) integrable.
/// Gegenbauer parameter zero has a vanishing polynomial for n>0 and cannot be made monic.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid parameters or degree;
/// [`Error::NonFiniteEvaluation`] if coefficients overflow or normalization is singular.
/// Coefficients are ordinary vectors and can be evaluated with Horner's rule.
///
/// ## Examples
///
/// ```
/// let coefficients = sci_special::hermite(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy hermite](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hermite.html).
pub fn hermite(n: usize, monic: bool) -> Result<Vec<f64>> {
    if !(true) {
        return Err(Error::InvalidInput(
            "orthogonal polynomial parameters are outside the domain",
        ));
    }
    coefficients(n, Family::Hermite(false), monic, 1.0, 0.0, 1.0)
}

/// Gaussian nodes and weights for exp(−x²) on (−∞,∞).
///
/// ## Parameters and domain
///
/// `n` is the number of nodes in `1..=256`; shape parameters must define an
/// integrable positive weight. Nodes are sorted ascending. The weights integrate
/// weighted polynomials exactly through degree 2n−1, up to floating-point error.
///
/// ## Errors
///
/// Invalid orders and parameters return [`Error::InvalidInput`]. Overflow or
/// eigensolver failure returns [`Error::NonFiniteEvaluation`] or [`Error::NumericalFailure`].
/// The total weight (SciPy's optional `mu`) is available by summing the returned weights.
///
/// ## Examples
///
/// ```
/// let (nodes, weights) = sci_special::roots_hermite(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_hermite](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_hermite.html).
pub fn roots_hermite(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if !(true) {
        return Err(Error::InvalidInput(
            "quadrature parameters are outside the domain",
        ));
    }
    quadrature(n, Family::Hermite(false), 1.0, 0.0, 1.0)
}

/// Returns coefficients of the probabilists' Hermite He_n polynomial in descending powers.
///
/// ## Parameters and domain
///
/// `n` is the integer degree, at most 256. `monic=true` divides by the leading
/// coefficient. Parameters must make the weight exp(−x²/2) on (−∞,∞) integrable.
/// Gegenbauer parameter zero has a vanishing polynomial for n>0 and cannot be made monic.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid parameters or degree;
/// [`Error::NonFiniteEvaluation`] if coefficients overflow or normalization is singular.
/// Coefficients are ordinary vectors and can be evaluated with Horner's rule.
///
/// ## Examples
///
/// ```
/// let coefficients = sci_special::hermitenorm(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy hermitenorm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hermitenorm.html).
pub fn hermitenorm(n: usize, monic: bool) -> Result<Vec<f64>> {
    if !(true) {
        return Err(Error::InvalidInput(
            "orthogonal polynomial parameters are outside the domain",
        ));
    }
    coefficients(n, Family::Hermite(true), monic, 1.0, 0.0, 1.0)
}

/// Gaussian nodes and weights for exp(−x²/2) on (−∞,∞).
///
/// ## Parameters and domain
///
/// `n` is the number of nodes in `1..=256`; shape parameters must define an
/// integrable positive weight. Nodes are sorted ascending. The weights integrate
/// weighted polynomials exactly through degree 2n−1, up to floating-point error.
///
/// ## Errors
///
/// Invalid orders and parameters return [`Error::InvalidInput`]. Overflow or
/// eigensolver failure returns [`Error::NonFiniteEvaluation`] or [`Error::NumericalFailure`].
/// The total weight (SciPy's optional `mu`) is available by summing the returned weights.
///
/// ## Examples
///
/// ```
/// let (nodes, weights) = sci_special::roots_hermitenorm(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_hermitenorm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_hermitenorm.html).
pub fn roots_hermitenorm(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if !(true) {
        return Err(Error::InvalidInput(
            "quadrature parameters are outside the domain",
        ));
    }
    quadrature(n, Family::Hermite(true), 1.0, 0.0, 1.0)
}

/// Returns coefficients of the Chebyshev T_n polynomial in descending powers.
///
/// ## Parameters and domain
///
/// `n` is the integer degree, at most 256. `monic=true` divides by the leading
/// coefficient. Parameters must make the weight (1−x²)^(-1/2) on `[−1,1]` integrable.
/// Gegenbauer parameter zero has a vanishing polynomial for n>0 and cannot be made monic.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid parameters or degree;
/// [`Error::NonFiniteEvaluation`] if coefficients overflow or normalization is singular.
/// Coefficients are ordinary vectors and can be evaluated with Horner's rule.
///
/// ## Examples
///
/// ```
/// let coefficients = sci_special::chebyt(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy chebyt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.chebyt.html).
pub fn chebyt(n: usize, monic: bool) -> Result<Vec<f64>> {
    if !(true) {
        return Err(Error::InvalidInput(
            "orthogonal polynomial parameters are outside the domain",
        ));
    }
    coefficients(n, Family::Chebyt, monic, 1.0, 0.0, 1.0)
}

/// Gaussian nodes and weights for (1−x²)^(-1/2) on `[−1,1]`.
///
/// ## Parameters and domain
///
/// `n` is the number of nodes in `1..=256`; shape parameters must define an
/// integrable positive weight. Nodes are sorted ascending. The weights integrate
/// weighted polynomials exactly through degree 2n−1, up to floating-point error.
///
/// ## Errors
///
/// Invalid orders and parameters return [`Error::InvalidInput`]. Overflow or
/// eigensolver failure returns [`Error::NonFiniteEvaluation`] or [`Error::NumericalFailure`].
/// The total weight (SciPy's optional `mu`) is available by summing the returned weights.
///
/// ## Examples
///
/// ```
/// let (nodes, weights) = sci_special::roots_chebyt(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_chebyt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_chebyt.html).
pub fn roots_chebyt(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if !(true) {
        return Err(Error::InvalidInput(
            "quadrature parameters are outside the domain",
        ));
    }
    quadrature(n, Family::Jacobi(-0.5, -0.5), 1.0, 0.0, 1.0)
}

/// Returns coefficients of the Chebyshev U_n polynomial in descending powers.
///
/// ## Parameters and domain
///
/// `n` is the integer degree, at most 256. `monic=true` divides by the leading
/// coefficient. Parameters must make the weight √(1−x²) on `[−1,1]` integrable.
/// Gegenbauer parameter zero has a vanishing polynomial for n>0 and cannot be made monic.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid parameters or degree;
/// [`Error::NonFiniteEvaluation`] if coefficients overflow or normalization is singular.
/// Coefficients are ordinary vectors and can be evaluated with Horner's rule.
///
/// ## Examples
///
/// ```
/// let coefficients = sci_special::chebyu(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy chebyu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.chebyu.html).
pub fn chebyu(n: usize, monic: bool) -> Result<Vec<f64>> {
    if !(true) {
        return Err(Error::InvalidInput(
            "orthogonal polynomial parameters are outside the domain",
        ));
    }
    coefficients(n, Family::Chebyu, monic, 1.0, 0.0, 1.0)
}

/// Gaussian nodes and weights for √(1−x²) on `[−1,1]`.
///
/// ## Parameters and domain
///
/// `n` is the number of nodes in `1..=256`; shape parameters must define an
/// integrable positive weight. Nodes are sorted ascending. The weights integrate
/// weighted polynomials exactly through degree 2n−1, up to floating-point error.
///
/// ## Errors
///
/// Invalid orders and parameters return [`Error::InvalidInput`]. Overflow or
/// eigensolver failure returns [`Error::NonFiniteEvaluation`] or [`Error::NumericalFailure`].
/// The total weight (SciPy's optional `mu`) is available by summing the returned weights.
///
/// ## Examples
///
/// ```
/// let (nodes, weights) = sci_special::roots_chebyu(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_chebyu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_chebyu.html).
pub fn roots_chebyu(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if !(true) {
        return Err(Error::InvalidInput(
            "quadrature parameters are outside the domain",
        ));
    }
    quadrature(n, Family::Jacobi(0.5, 0.5), 1.0, 0.0, 1.0)
}

/// Returns coefficients of the scaled Chebyshev 2T_n(x/2) polynomial in descending powers.
///
/// ## Parameters and domain
///
/// `n` is the integer degree, at most 256. `monic=true` divides by the leading
/// coefficient. Parameters must make the weight (1−x²/4)^(-1/2) on `[−2,2]` integrable.
/// Gegenbauer parameter zero has a vanishing polynomial for n>0 and cannot be made monic.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid parameters or degree;
/// [`Error::NonFiniteEvaluation`] if coefficients overflow or normalization is singular.
/// Coefficients are ordinary vectors and can be evaluated with Horner's rule.
///
/// ## Examples
///
/// ```
/// let coefficients = sci_special::chebyc(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy chebyc](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.chebyc.html).
pub fn chebyc(n: usize, monic: bool) -> Result<Vec<f64>> {
    if !(true) {
        return Err(Error::InvalidInput(
            "orthogonal polynomial parameters are outside the domain",
        ));
    }
    coefficients(n, Family::Chebyt, monic, 0.5, 0.0, 2.0)
}

/// Gaussian nodes and weights for (1−x²/4)^(-1/2) on `[−2,2]`.
///
/// ## Parameters and domain
///
/// `n` is the number of nodes in `1..=256`; shape parameters must define an
/// integrable positive weight. Nodes are sorted ascending. The weights integrate
/// weighted polynomials exactly through degree 2n−1, up to floating-point error.
///
/// ## Errors
///
/// Invalid orders and parameters return [`Error::InvalidInput`]. Overflow or
/// eigensolver failure returns [`Error::NonFiniteEvaluation`] or [`Error::NumericalFailure`].
/// The total weight (SciPy's optional `mu`) is available by summing the returned weights.
///
/// ## Examples
///
/// ```
/// let (nodes, weights) = sci_special::roots_chebyc(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_chebyc](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_chebyc.html).
pub fn roots_chebyc(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if !(true) {
        return Err(Error::InvalidInput(
            "quadrature parameters are outside the domain",
        ));
    }
    quadrature(n, Family::Jacobi(-0.5, -0.5), 2.0, 0.0, 2.0)
}

/// Returns coefficients of the scaled Chebyshev U_n(x/2) polynomial in descending powers.
///
/// ## Parameters and domain
///
/// `n` is the integer degree, at most 256. `monic=true` divides by the leading
/// coefficient. Parameters must make the weight √(1−x²/4) on `[−2,2]` integrable.
/// Gegenbauer parameter zero has a vanishing polynomial for n>0 and cannot be made monic.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid parameters or degree;
/// [`Error::NonFiniteEvaluation`] if coefficients overflow or normalization is singular.
/// Coefficients are ordinary vectors and can be evaluated with Horner's rule.
///
/// ## Examples
///
/// ```
/// let coefficients = sci_special::chebys(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy chebys](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.chebys.html).
pub fn chebys(n: usize, monic: bool) -> Result<Vec<f64>> {
    if !(true) {
        return Err(Error::InvalidInput(
            "orthogonal polynomial parameters are outside the domain",
        ));
    }
    coefficients(n, Family::Chebyu, monic, 0.5, 0.0, 1.0)
}

/// Gaussian nodes and weights for √(1−x²/4) on `[−2,2]`.
///
/// ## Parameters and domain
///
/// `n` is the number of nodes in `1..=256`; shape parameters must define an
/// integrable positive weight. Nodes are sorted ascending. The weights integrate
/// weighted polynomials exactly through degree 2n−1, up to floating-point error.
///
/// ## Errors
///
/// Invalid orders and parameters return [`Error::InvalidInput`]. Overflow or
/// eigensolver failure returns [`Error::NonFiniteEvaluation`] or [`Error::NumericalFailure`].
/// The total weight (SciPy's optional `mu`) is available by summing the returned weights.
///
/// ## Examples
///
/// ```
/// let (nodes, weights) = sci_special::roots_chebys(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_chebys](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_chebys.html).
pub fn roots_chebys(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if !(true) {
        return Err(Error::InvalidInput(
            "quadrature parameters are outside the domain",
        ));
    }
    quadrature(n, Family::Jacobi(0.5, 0.5), 2.0, 0.0, 2.0)
}

/// Returns coefficients of the shifted Chebyshev T_n(2x−1) polynomial in descending powers.
///
/// ## Parameters and domain
///
/// `n` is the integer degree, at most 256. `monic=true` divides by the leading
/// coefficient. Parameters must make the weight (x(1−x))^(-1/2) on `[0,1]` integrable.
/// Gegenbauer parameter zero has a vanishing polynomial for n>0 and cannot be made monic.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid parameters or degree;
/// [`Error::NonFiniteEvaluation`] if coefficients overflow or normalization is singular.
/// Coefficients are ordinary vectors and can be evaluated with Horner's rule.
///
/// ## Examples
///
/// ```
/// let coefficients = sci_special::sh_chebyt(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy sh_chebyt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sh_chebyt.html).
pub fn sh_chebyt(n: usize, monic: bool) -> Result<Vec<f64>> {
    if !(true) {
        return Err(Error::InvalidInput(
            "orthogonal polynomial parameters are outside the domain",
        ));
    }
    coefficients(n, Family::Chebyt, monic, 2.0, -1.0, 1.0)
}

/// Gaussian nodes and weights for (x(1−x))^(-1/2) on `[0,1]`.
///
/// ## Parameters and domain
///
/// `n` is the number of nodes in `1..=256`; shape parameters must define an
/// integrable positive weight. Nodes are sorted ascending. The weights integrate
/// weighted polynomials exactly through degree 2n−1, up to floating-point error.
///
/// ## Errors
///
/// Invalid orders and parameters return [`Error::InvalidInput`]. Overflow or
/// eigensolver failure returns [`Error::NonFiniteEvaluation`] or [`Error::NumericalFailure`].
/// The total weight (SciPy's optional `mu`) is available by summing the returned weights.
///
/// ## Examples
///
/// ```
/// let (nodes, weights) = sci_special::roots_sh_chebyt(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_sh_chebyt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_sh_chebyt.html).
pub fn roots_sh_chebyt(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if !(true) {
        return Err(Error::InvalidInput(
            "quadrature parameters are outside the domain",
        ));
    }
    quadrature(n, Family::Jacobi(-0.5, -0.5), 0.5, 0.5, 1.0)
}

/// Returns coefficients of the shifted Chebyshev U_n(2x−1) polynomial in descending powers.
///
/// ## Parameters and domain
///
/// `n` is the integer degree, at most 256. `monic=true` divides by the leading
/// coefficient. Parameters must make the weight √(x(1−x)) on `[0,1]` integrable.
/// Gegenbauer parameter zero has a vanishing polynomial for n>0 and cannot be made monic.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid parameters or degree;
/// [`Error::NonFiniteEvaluation`] if coefficients overflow or normalization is singular.
/// Coefficients are ordinary vectors and can be evaluated with Horner's rule.
///
/// ## Examples
///
/// ```
/// let coefficients = sci_special::sh_chebyu(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy sh_chebyu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sh_chebyu.html).
pub fn sh_chebyu(n: usize, monic: bool) -> Result<Vec<f64>> {
    if !(true) {
        return Err(Error::InvalidInput(
            "orthogonal polynomial parameters are outside the domain",
        ));
    }
    coefficients(n, Family::Chebyu, monic, 2.0, -1.0, 1.0)
}

/// Gaussian nodes and weights for √(x(1−x)) on `[0,1]`.
///
/// ## Parameters and domain
///
/// `n` is the number of nodes in `1..=256`; shape parameters must define an
/// integrable positive weight. Nodes are sorted ascending. The weights integrate
/// weighted polynomials exactly through degree 2n−1, up to floating-point error.
///
/// ## Errors
///
/// Invalid orders and parameters return [`Error::InvalidInput`]. Overflow or
/// eigensolver failure returns [`Error::NonFiniteEvaluation`] or [`Error::NumericalFailure`].
/// The total weight (SciPy's optional `mu`) is available by summing the returned weights.
///
/// ## Examples
///
/// ```
/// let (nodes, weights) = sci_special::roots_sh_chebyu(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_sh_chebyu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_sh_chebyu.html).
pub fn roots_sh_chebyu(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if !(true) {
        return Err(Error::InvalidInput(
            "quadrature parameters are outside the domain",
        ));
    }
    quadrature(n, Family::Jacobi(0.5, 0.5), 0.5, 0.5, 0.25)
}

/// Returns coefficients of the shifted Legendre P_n(2x−1) polynomial in descending powers.
///
/// ## Parameters and domain
///
/// `n` is the integer degree, at most 256. `monic=true` divides by the leading
/// coefficient. Parameters must make the weight unit weight on `[0,1]` integrable.
/// Gegenbauer parameter zero has a vanishing polynomial for n>0 and cannot be made monic.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid parameters or degree;
/// [`Error::NonFiniteEvaluation`] if coefficients overflow or normalization is singular.
/// Coefficients are ordinary vectors and can be evaluated with Horner's rule.
///
/// ## Examples
///
/// ```
/// let coefficients = sci_special::sh_legendre(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy sh_legendre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sh_legendre.html).
pub fn sh_legendre(n: usize, monic: bool) -> Result<Vec<f64>> {
    if !(true) {
        return Err(Error::InvalidInput(
            "orthogonal polynomial parameters are outside the domain",
        ));
    }
    coefficients(n, Family::Jacobi(0.0, 0.0), monic, 2.0, -1.0, 1.0)
}

/// Gaussian nodes and weights for unit weight on `[0,1]`.
///
/// ## Parameters and domain
///
/// `n` is the number of nodes in `1..=256`; shape parameters must define an
/// integrable positive weight. Nodes are sorted ascending. The weights integrate
/// weighted polynomials exactly through degree 2n−1, up to floating-point error.
///
/// ## Errors
///
/// Invalid orders and parameters return [`Error::InvalidInput`]. Overflow or
/// eigensolver failure returns [`Error::NonFiniteEvaluation`] or [`Error::NumericalFailure`].
/// The total weight (SciPy's optional `mu`) is available by summing the returned weights.
///
/// ## Examples
///
/// ```
/// let (nodes, weights) = sci_special::roots_sh_legendre(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_sh_legendre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_sh_legendre.html).
pub fn roots_sh_legendre(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if !(true) {
        return Err(Error::InvalidInput(
            "quadrature parameters are outside the domain",
        ));
    }
    quadrature(n, Family::Jacobi(0.0, 0.0), 0.5, 0.5, 0.5)
}

/// Returns coefficients of the shifted Jacobi G_n^(p,q) polynomial in descending powers.
///
/// ## Parameters and domain
///
/// `n` is the integer degree, at most 256. `monic=true` divides by the leading
/// coefficient. Parameters must make the weight (1−x)^(p−q) x^(q−1) on `[0,1]` integrable.
/// Gegenbauer parameter zero has a vanishing polynomial for n>0 and cannot be made monic.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] for invalid parameters or degree;
/// [`Error::NonFiniteEvaluation`] if coefficients overflow or normalization is singular.
/// Coefficients are ordinary vectors and can be evaluated with Horner's rule.
///
/// ## Examples
///
/// ```
/// let coefficients = sci_special::sh_jacobi(3,2.5,1.5,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy sh_jacobi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sh_jacobi.html).
pub fn sh_jacobi(n: usize, p: f64, q: f64, monic: bool) -> Result<Vec<f64>> {
    if !(p.is_finite() && q.is_finite() && q > 0.0 && p - q > -1.0) {
        return Err(Error::InvalidInput(
            "orthogonal polynomial parameters are outside the domain",
        ));
    }
    coefficients(
        n,
        Family::Jacobi(p - q, q - 1.0),
        monic,
        2.0,
        -1.0,
        (super::gammaln(n as f64 + 1.0) + super::gammaln(n as f64 + p)
            - super::gammaln(2.0 * n as f64 + p))
        .exp(),
    )
}

/// Gaussian nodes and weights for (1−x)^(p−q) x^(q−1) on `[0,1]`.
///
/// ## Parameters and domain
///
/// `n` is the number of nodes in `1..=256`; shape parameters must define an
/// integrable positive weight. Nodes are sorted ascending. The weights integrate
/// weighted polynomials exactly through degree 2n−1, up to floating-point error.
///
/// ## Errors
///
/// Invalid orders and parameters return [`Error::InvalidInput`]. Overflow or
/// eigensolver failure returns [`Error::NonFiniteEvaluation`] or [`Error::NumericalFailure`].
/// The total weight (SciPy's optional `mu`) is available by summing the returned weights.
///
/// ## Examples
///
/// ```
/// let (nodes, weights) = sci_special::roots_sh_jacobi(3,2.5,1.5)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_sh_jacobi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_sh_jacobi.html).
pub fn roots_sh_jacobi(n: usize, p: f64, q: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    if !(p.is_finite() && q.is_finite() && q > 0.0 && p - q > -1.0) {
        return Err(Error::InvalidInput(
            "quadrature parameters are outside the domain",
        ));
    }
    quadrature(
        n,
        Family::Jacobi(p - q, q - 1.0),
        0.5,
        0.5,
        2.0_f64.powf(-p),
    )
}
