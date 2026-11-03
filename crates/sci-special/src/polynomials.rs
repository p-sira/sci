//! Polynomial coefficients and Gaussian rules use the same three-term recurrence.

use crate::{Error, Result};

// Keep Sci's error type stable while using the standalone polynomial kernels.
pub(super) fn external_orthogonal_error(error: ::orthogonal::Error) -> Error {
    match error {
        ::orthogonal::Error::InvalidInput(message) => Error::InvalidInput(message),
        ::orthogonal::Error::DimensionMismatch => Error::DimensionMismatch,
        ::orthogonal::Error::NonFiniteEvaluation => Error::NonFiniteEvaluation,
        ::orthogonal::Error::NumericalFailure(message) => Error::NumericalFailure(message),
    }
}

// Implicit symmetric tridiagonal QL with Wilkinson shifts (EISPACK IMTQL2).
// Quadrature needs only the first component of each eigenvector (Golub-Welsch),
// while Lamé polynomials need the complete eigensystem. Keep the same shifted
// iteration for both, but accumulate a single row for the quadrature case.
// Scaling avoids overflow in shifts and Givens rotations; work is bounded.
pub(super) fn eigensystem(diagonal: &[f64], off: &[f64]) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
    let (values, rows) = tridiagonal_eigensystem(diagonal, off, true)?;
    Ok((
        values,
        (0..diagonal.len())
            .map(|i| rows.iter().map(|row| row[i]).collect())
            .collect(),
    ))
}

#[cfg(test)]
fn first_eigenvector_components(diagonal: &[f64], off: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
    let (values, mut rows) = tridiagonal_eigensystem(diagonal, off, false)?;
    Ok((values, rows.pop().unwrap()))
}

// Accumulate eigenvector rows, sorted consistently with the eigenvalues.
// With all_vectors=false, only row zero is retained, avoiding unnecessary
// O(n^3) eigenvector rotations and O(n^2) eigenvector storage.
fn tridiagonal_eigensystem(
    diagonal: &[f64],
    off: &[f64],
    all_vectors: bool,
) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
    let n = diagonal.len();
    if n == 0 || n > 256 {
        return Err(Error::InvalidInput("tridiagonal order must be in 1..=256"));
    }
    if off.len() != n - 1 {
        return Err(Error::DimensionMismatch);
    }
    if diagonal.iter().chain(off).any(|x| !x.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    let scale = diagonal
        .iter()
        .chain(off)
        .fold(0.0_f64, |value, &x| value.max(x.abs()));
    if scale == 0.0 {
        let mut first = vec![0.0; n];
        first[0] = 1.0;
        if !all_vectors {
            return Ok((vec![0.0; n], vec![first]));
        }
        let vectors = (0..n)
            .map(|j| (0..n).map(|i| f64::from(i == j)).collect())
            .collect();
        return Ok((vec![0.0; n], vectors));
    }
    let mut d: Vec<_> = diagonal.iter().map(|&x| x / scale).collect();
    let mut e: Vec<_> = off.iter().map(|&x| x / scale).collect();
    e.push(0.0);
    let mut vectors = vec![vec![0.0; n]; if all_vectors { n } else { 1 }];
    for (i, row) in vectors.iter_mut().enumerate() {
        row[i] = 1.0;
    }

    for l in 0..n {
        let mut iterations = 0;
        loop {
            let mut m = l;
            while m + 1 < n {
                if e[m].abs() <= f64::EPSILON * (d[m].abs() + d[m + 1].abs()) {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            if iterations == 128 {
                return Err(Error::NumericalFailure(
                    "symmetric tridiagonal eigensolver did not converge",
                ));
            }
            iterations += 1;

            // Wilkinson shift, choosing the sign to avoid cancellation.
            let shift = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let denominator = shift + shift.hypot(1.0).copysign(shift);
            let mut g = d[m] - d[l] + e[l] / denominator;
            let mut sine = 1.0;
            let mut cosine = 1.0;
            let mut correction = 0.0;
            for i in (l..m).rev() {
                let f = sine * e[i];
                let b = cosine * e[i];
                let radius = f.hypot(g);
                if radius == 0.0 {
                    e[i + 1] = 0.0;
                    sine = 0.0;
                    cosine = 1.0;
                } else {
                    e[i + 1] = radius;
                    sine = f / radius;
                    cosine = g / radius;
                }
                g = d[i + 1] - correction;
                let rotation = (d[i] - g) * sine + 2.0 * cosine * b;
                correction = sine * rotation;
                d[i + 1] = g + correction;
                g = cosine * rotation - b;
                for row in &mut vectors {
                    let f = row[i + 1];
                    row[i + 1] = sine * row[i] + cosine * f;
                    row[i] = cosine * row[i] - sine * f;
                }
            }
            d[l] -= correction;
            e[l] = g;
            e[m] = 0.0;
            if !d[l].is_finite() || !g.is_finite() {
                return Err(Error::NonFiniteEvaluation);
            }
        }
    }
    for value in &mut d {
        *value *= scale;
    }
    if d.iter().any(|x| !x.is_finite()) || vectors.iter().flatten().any(|x| !x.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    let mut indices: Vec<_> = (0..n).collect();
    indices.sort_by(|&a, &b| d[a].total_cmp(&d[b]));
    Ok((
        indices.iter().map(|&i| d[i]).collect(),
        vectors
            .iter()
            .map(|row| indices.iter().map(|&i| row[i]).collect())
            .collect(),
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
/// let coefficients = sci_special::orthogonal::jacobi(3,0.5,0.3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy jacobi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jacobi.html).
pub fn jacobi(n: usize, alpha: f64, beta: f64, monic: bool) -> Result<Vec<f64>> {
    ::orthogonal::jacobi(n, alpha, beta, monic).map_err(external_orthogonal_error)
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
/// let (nodes, weights) = sci_special::orthogonal::roots_jacobi(3,0.5,0.3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_jacobi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_jacobi.html).
pub fn roots_jacobi(n: usize, alpha: f64, beta: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    if n == 0 || n > 256 {
        return Err(Error::InvalidInput("quadrature order must be in 1..=256"));
    }
    ::orthogonal::roots_jacobi(n, alpha, beta).map_err(external_orthogonal_error)
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
/// let coefficients = sci_special::orthogonal::legendre(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy legendre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.legendre.html).
pub fn legendre(n: usize, monic: bool) -> Result<Vec<f64>> {
    ::orthogonal::legendre(n, monic).map_err(external_orthogonal_error)
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
/// let (nodes, weights) = sci_special::orthogonal::roots_legendre(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_legendre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_legendre.html).
pub fn roots_legendre(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if n == 0 || n > 256 {
        return Err(Error::InvalidInput("quadrature order must be in 1..=256"));
    }
    ::orthogonal::roots_legendre(n).map_err(external_orthogonal_error)
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
/// let coefficients = sci_special::orthogonal::gegenbauer(3,0.5,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy gegenbauer](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gegenbauer.html).
pub fn gegenbauer(n: usize, alpha: f64, monic: bool) -> Result<Vec<f64>> {
    ::orthogonal::gegenbauer(n, alpha, monic).map_err(external_orthogonal_error)
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
/// let (nodes, weights) = sci_special::orthogonal::roots_gegenbauer(3,0.5)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_gegenbauer](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_gegenbauer.html).
pub fn roots_gegenbauer(n: usize, alpha: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    if n == 0 || n > 256 {
        return Err(Error::InvalidInput("quadrature order must be in 1..=256"));
    }
    ::orthogonal::roots_gegenbauer(n, alpha).map_err(external_orthogonal_error)
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
/// let coefficients = sci_special::orthogonal::genlaguerre(3,0.5,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy genlaguerre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.genlaguerre.html).
pub fn genlaguerre(n: usize, alpha: f64, monic: bool) -> Result<Vec<f64>> {
    ::orthogonal::genlaguerre(n, alpha, monic).map_err(external_orthogonal_error)
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
/// let (nodes, weights) = sci_special::orthogonal::roots_genlaguerre(3,0.5)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_genlaguerre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_genlaguerre.html).
pub fn roots_genlaguerre(n: usize, alpha: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    if n == 0 || n > 256 {
        return Err(Error::InvalidInput("quadrature order must be in 1..=256"));
    }
    ::orthogonal::roots_genlaguerre(n, alpha).map_err(external_orthogonal_error)
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
/// let coefficients = sci_special::orthogonal::laguerre(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy laguerre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.laguerre.html).
pub fn laguerre(n: usize, monic: bool) -> Result<Vec<f64>> {
    ::orthogonal::laguerre(n, monic).map_err(external_orthogonal_error)
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
/// let (nodes, weights) = sci_special::orthogonal::roots_laguerre(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_laguerre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_laguerre.html).
pub fn roots_laguerre(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if n == 0 || n > 256 {
        return Err(Error::InvalidInput("quadrature order must be in 1..=256"));
    }
    ::orthogonal::roots_laguerre(n).map_err(external_orthogonal_error)
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
/// let coefficients = sci_special::orthogonal::hermite(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy hermite](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hermite.html).
pub fn hermite(n: usize, monic: bool) -> Result<Vec<f64>> {
    ::orthogonal::hermite(n, monic).map_err(external_orthogonal_error)
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
/// let (nodes, weights) = sci_special::orthogonal::roots_hermite(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_hermite](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_hermite.html).
pub fn roots_hermite(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if n == 0 || n > 256 {
        return Err(Error::InvalidInput("quadrature order must be in 1..=256"));
    }
    ::orthogonal::roots_hermite(n).map_err(external_orthogonal_error)
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
/// let coefficients = sci_special::orthogonal::hermitenorm(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy hermitenorm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hermitenorm.html).
pub fn hermitenorm(n: usize, monic: bool) -> Result<Vec<f64>> {
    ::orthogonal::hermitenorm(n, monic).map_err(external_orthogonal_error)
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
/// let (nodes, weights) = sci_special::orthogonal::roots_hermitenorm(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_hermitenorm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_hermitenorm.html).
pub fn roots_hermitenorm(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if n == 0 || n > 256 {
        return Err(Error::InvalidInput("quadrature order must be in 1..=256"));
    }
    ::orthogonal::roots_hermitenorm(n).map_err(external_orthogonal_error)
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
/// let coefficients = sci_special::orthogonal::chebyt(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy chebyt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.chebyt.html).
pub fn chebyt(n: usize, monic: bool) -> Result<Vec<f64>> {
    ::orthogonal::chebyt(n, monic).map_err(external_orthogonal_error)
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
/// let (nodes, weights) = sci_special::orthogonal::roots_chebyt(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_chebyt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_chebyt.html).
pub fn roots_chebyt(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if n == 0 || n > 256 {
        return Err(Error::InvalidInput("quadrature order must be in 1..=256"));
    }
    ::orthogonal::roots_chebyt(n).map_err(external_orthogonal_error)
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
/// let coefficients = sci_special::orthogonal::chebyu(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy chebyu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.chebyu.html).
pub fn chebyu(n: usize, monic: bool) -> Result<Vec<f64>> {
    ::orthogonal::chebyu(n, monic).map_err(external_orthogonal_error)
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
/// let (nodes, weights) = sci_special::orthogonal::roots_chebyu(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_chebyu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_chebyu.html).
pub fn roots_chebyu(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if n == 0 || n > 256 {
        return Err(Error::InvalidInput("quadrature order must be in 1..=256"));
    }
    ::orthogonal::roots_chebyu(n).map_err(external_orthogonal_error)
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
/// let coefficients = sci_special::orthogonal::chebyc(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy chebyc](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.chebyc.html).
pub fn chebyc(n: usize, monic: bool) -> Result<Vec<f64>> {
    ::orthogonal::chebyc(n, monic).map_err(external_orthogonal_error)
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
/// let (nodes, weights) = sci_special::orthogonal::roots_chebyc(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_chebyc](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_chebyc.html).
pub fn roots_chebyc(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if n == 0 || n > 256 {
        return Err(Error::InvalidInput("quadrature order must be in 1..=256"));
    }
    ::orthogonal::roots_chebyc(n).map_err(external_orthogonal_error)
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
/// let coefficients = sci_special::orthogonal::chebys(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy chebys](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.chebys.html).
pub fn chebys(n: usize, monic: bool) -> Result<Vec<f64>> {
    ::orthogonal::chebys(n, monic).map_err(external_orthogonal_error)
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
/// let (nodes, weights) = sci_special::orthogonal::roots_chebys(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_chebys](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_chebys.html).
pub fn roots_chebys(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if n == 0 || n > 256 {
        return Err(Error::InvalidInput("quadrature order must be in 1..=256"));
    }
    ::orthogonal::roots_chebys(n).map_err(external_orthogonal_error)
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
/// let coefficients = sci_special::orthogonal::sh_chebyt(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy sh_chebyt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sh_chebyt.html).
pub fn sh_chebyt(n: usize, monic: bool) -> Result<Vec<f64>> {
    ::orthogonal::sh_chebyt(n, monic).map_err(external_orthogonal_error)
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
/// let (nodes, weights) = sci_special::orthogonal::roots_sh_chebyt(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_sh_chebyt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_sh_chebyt.html).
pub fn roots_sh_chebyt(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if n == 0 || n > 256 {
        return Err(Error::InvalidInput("quadrature order must be in 1..=256"));
    }
    ::orthogonal::roots_sh_chebyt(n).map_err(external_orthogonal_error)
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
/// let coefficients = sci_special::orthogonal::sh_chebyu(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy sh_chebyu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sh_chebyu.html).
pub fn sh_chebyu(n: usize, monic: bool) -> Result<Vec<f64>> {
    ::orthogonal::sh_chebyu(n, monic).map_err(external_orthogonal_error)
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
/// let (nodes, weights) = sci_special::orthogonal::roots_sh_chebyu(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_sh_chebyu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_sh_chebyu.html).
pub fn roots_sh_chebyu(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if n == 0 || n > 256 {
        return Err(Error::InvalidInput("quadrature order must be in 1..=256"));
    }
    ::orthogonal::roots_sh_chebyu(n).map_err(external_orthogonal_error)
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
/// let coefficients = sci_special::orthogonal::sh_legendre(3,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy sh_legendre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sh_legendre.html).
pub fn sh_legendre(n: usize, monic: bool) -> Result<Vec<f64>> {
    ::orthogonal::sh_legendre(n, monic).map_err(external_orthogonal_error)
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
/// let (nodes, weights) = sci_special::orthogonal::roots_sh_legendre(3)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_sh_legendre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_sh_legendre.html).
pub fn roots_sh_legendre(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if n == 0 || n > 256 {
        return Err(Error::InvalidInput("quadrature order must be in 1..=256"));
    }
    ::orthogonal::roots_sh_legendre(n).map_err(external_orthogonal_error)
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
/// let coefficients = sci_special::orthogonal::sh_jacobi(3,2.5,1.5,false)?;
/// assert_eq!(coefficients.len(),4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy sh_jacobi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sh_jacobi.html).
pub fn sh_jacobi(n: usize, p: f64, q: f64, monic: bool) -> Result<Vec<f64>> {
    ::orthogonal::sh_jacobi(n, p, q, monic).map_err(external_orthogonal_error)
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
/// let (nodes, weights) = sci_special::orthogonal::roots_sh_jacobi(3,2.5,1.5)?;
/// assert_eq!(nodes.len(),weights.len());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy roots_sh_jacobi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_sh_jacobi.html).
pub fn roots_sh_jacobi(n: usize, p: f64, q: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    if n == 0 || n > 256 {
        return Err(Error::InvalidInput("quadrature order must be in 1..=256"));
    }
    ::orthogonal::roots_sh_jacobi(n, p, q).map_err(external_orthogonal_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadrature_first_components_match_full_eigensystem() {
        // Both paths use identical shifts and sorting; the optimization skips
        // unused eigenvector rows rather than changing eigenvalue algorithms.
        for n in [1, 2, 3, 16, 64, 128, 256] {
            let diagonal: Vec<_> = (0..n).map(|i| 0.15 * (i as f64).sin()).collect();
            let off: Vec<_> = (0..n - 1).map(|i| 0.25 + 0.01 * ((i % 7) as f64)).collect();
            let (full_values, full_vectors) = eigensystem(&diagonal, &off).unwrap();
            let (first_values, first_components) =
                first_eigenvector_components(&diagonal, &off).unwrap();
            assert_eq!(first_values, full_values, "n={n}");
            for (i, &component) in first_components.iter().enumerate() {
                assert_eq!(component, full_vectors[i][0], "n={n}, i={i}");
            }
        }
        let (values, first) = first_eigenvector_components(&[0.0; 3], &[0.0; 2]).unwrap();
        assert_eq!(values, vec![0.0; 3]);
        assert_eq!(first, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn shifted_tridiagonal_eigenpairs_have_small_residuals_and_are_orthogonal() {
        // The Toeplitz tridiagonal spectrum is 2 - 2*cos(k*pi/(n+1)).
        // In particular, the smallest eigenvalues of the 256-row case are
        // much smaller than the entries of the matrix.
        for n in [1, 2, 16, 64, 256] {
            let diagonal = vec![2.0; n];
            let off = vec![-1.0; n - 1];
            let (values, vectors) = eigensystem(&diagonal, &off).unwrap();
            assert_eq!(values.len(), n);
            assert_eq!(vectors.len(), n);
            assert!(values.windows(2).all(|pair| pair[0] <= pair[1]));
            for (k, (&value, vector)) in values.iter().zip(&vectors).enumerate() {
                let expected =
                    2.0 - 2.0 * (((k + 1) as f64 * std::f64::consts::PI) / (n + 1) as f64).cos();
                assert!(
                    (value - expected).abs() < 3e-12,
                    "n={n}, k={k}, eigenvalue={value}, expected={expected}"
                );
                let mut squared_norm = 0.0;
                for i in 0..n {
                    let left = if i > 0 { -vector[i - 1] } else { 0.0 };
                    let right = if i + 1 < n { -vector[i + 1] } else { 0.0 };
                    let residual = left + 2.0 * vector[i] + right - value * vector[i];
                    assert!(
                        residual.abs() < 3e-12,
                        "n={n}, k={k}, i={i}, residual={residual:e}"
                    );
                    squared_norm += vector[i] * vector[i];
                }
                assert!((squared_norm - 1.0).abs() < 3e-12);
            }
            // Check the full basis at modest orders and representative
            // separated/adjacent eigenpairs at the largest supported order.
            let selected: Vec<_> = if n <= 64 {
                (0..n).collect()
            } else {
                vec![0, 1, 63, 64, 127, 128, 254, 255]
            };
            for &i in &selected {
                for &j in &selected {
                    let dot = vectors[i]
                        .iter()
                        .zip(&vectors[j])
                        .map(|(x, y)| x * y)
                        .sum::<f64>();
                    assert!(
                        (dot - f64::from(i == j)).abs() < 3e-12,
                        "n={n}, i={i}, j={j}, dot={dot}"
                    );
                }
            }
        }
    }

    #[test]
    fn shifted_tridiagonal_handles_scaling_degeneracy_and_invalid_data() {
        assert!(matches!(eigensystem(&[], &[]), Err(Error::InvalidInput(_))));
        assert!(matches!(
            eigensystem(&[1.], &[1.]),
            Err(Error::DimensionMismatch)
        ));
        assert_eq!(eigensystem(&[5.], &[]).unwrap(), (vec![5.], vec![vec![1.]]));
        assert!(matches!(
            eigensystem(&[f64::NAN], &[]),
            Err(Error::NonFiniteEvaluation)
        ));
        assert!(matches!(
            eigensystem(&[1., 1.], &[f64::INFINITY]),
            Err(Error::NonFiniteEvaluation)
        ));
        let (zero_values, zero_vectors) = eigensystem(&[0.; 4], &[0.; 3]).unwrap();
        assert_eq!(zero_values, [0.; 4]);
        for (i, vector) in zero_vectors.iter().enumerate() {
            for (j, &entry) in vector.iter().enumerate() {
                assert_eq!(entry, f64::from(i == j));
            }
        }
        for scale in [1e-200, 1e200] {
            let (values, vectors) = eigensystem(&[2.0 * scale; 2], &[scale]).unwrap();
            for (value, expected) in values.iter().zip([1., 3.]) {
                assert!((value / scale - expected).abs() < 3e-14);
            }
            assert!((vectors[0][0].abs() - std::f64::consts::FRAC_1_SQRT_2).abs() < 3e-14);
        }
    }
}
