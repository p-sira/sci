//! Sparse direct/iterative solves, operators, spectra, and matrix functions.
//!
//! LU uses faer privately, with an order limit of 1,024 to bound worst-case
//! fill-in. No dense copy of the input matrix is formed. The pivoting strategy
//! differs from SciPy's SuperLU; ill-conditioned problems can have large forward
//! errors even when residuals are small. Singular factorizations and nonfinite
//! outputs return errors. General sparse operations retain crate storage limits.

use super::{Error, LIMIT, Result, SparseMatrix, entries, identity, matmul, validate};
use faer::prelude::Solve;

/// Reusable sparse LU factorization, with private backend storage.
#[derive(Debug, Clone)]
pub struct SparseLu {
    inner: faer::sparse::linalg::solvers::Lu<usize, f64>,
    order: usize,
}

impl SparseLu {
    /// Solve `A x = b` using this factorization.
    ///
    /// # Parameters and conventions
    /// `b` has length equal to the order of the factored matrix.
    /// # Domain and special values
    /// `b` must be finite; the factorization is square and nonempty.
    /// # Returns
    /// A solution vector in column order, without modifying `b`.
    /// # Errors
    /// Incompatible length, nonfinite input, or nonfinite solution.
    /// # Numerical behavior
    /// Uses the cached pivoted sparse LU factors. No iterative refinement;
    /// ill-conditioning may amplify errors. See [`splu`] for work limits.
    /// # Examples
    /// ```
    /// let lu = sci_sparse::linalg::splu(&sci_sparse::construct::identity(2)?)?;
    /// assert_eq!(lu.solve(&[2.0, 3.0])?, vec![2.0, 3.0]);
    /// # Ok::<(), sci_sparse::Error>(())
    /// ```
    /// # References
    /// [SciPy SuperLU.solve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.SuperLU.solve.html).
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>> {
        if b.len() != self.order {
            return Err(Error::DimensionMismatch);
        }
        sci_core::is_finite_slice(b)?;
        let rhs = faer::Mat::from_fn(self.order, 1, |i, _| b[i]);
        let x = self.inner.solve(&rhs);
        (0..self.order)
            .map(|i| match x[(i, 0)] {
                value if value.is_finite() => Ok(value),
                _ => Err(Error::NonFiniteEvaluation),
            })
            .collect()
    }
}

/// Factor a square matrix as `P A Q = L U` for repeated linear solves.
///
/// # Parameters and conventions
/// `a` is canonical CSR, converted privately to backend CSC storage. `P,Q` are
/// backend permutations; factors and permutations are not exposed.
/// # Domain and special values
/// Finite, square, nonsingular matrices of order `1..=1024`.
/// # Returns
/// A reusable [`SparseLu`] with a vector [`SparseLu::solve`] method.
/// # Errors
/// Invalid storage, unsupported shape/order, singularity, or backend failure.
/// # Numerical behavior
/// Sparse pivoted LU with fill-reducing ordering from faer. The order cap bounds
/// worst-case fill-in quadratically and factorization work cubically. Near
/// singularity is not diagnosed by a condition estimate. No SuperLU options.
/// # Examples
/// ```
/// let lu = sci_sparse::linalg::splu(&sci_sparse::construct::identity(2)?)?;
/// assert_eq!(lu.solve(&[1.0, 2.0])?, vec![1.0, 2.0]);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy splu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html),
/// [faer sparse LU](https://docs.rs/faer/0.24.4/faer/sparse/linalg/solvers/struct.Lu.html).
pub fn splu(a: &SparseMatrix) -> Result<SparseLu> {
    validate(a)?;
    if a.rows != a.cols || a.rows == 0 || a.rows > 1024 {
        return Err(Error::InvalidInput(
            "sparse LU requires square order 1..=1024",
        ));
    }
    let triples: Vec<_> = entries(a)
        .map(|(r, c, v)| faer::sparse::Triplet::new(r, c, v))
        .collect();
    let matrix =
        faer::sparse::SparseColMat::<usize, f64>::try_new_from_triplets(a.rows, a.cols, &triples)
            .map_err(|_| Error::NumericalFailure("sparse backend conversion failed"))?;
    let inner = matrix.sp_lu().map_err(|_| {
        Error::NumericalFailure("sparse LU failed: singular matrix or backend failure")
    })?;
    Ok(SparseLu {
        inner,
        order: a.rows,
    })
}

/// Solve `A x = b` by sparse pivoted LU.
///
/// # Parameters and conventions
/// `a` is square canonical CSR and `b` is a vector of length `a.rows`.
/// # Domain and special values
/// Finite nonsingular matrices of order `1..=1024`, and finite right sides.
/// # Returns
/// The vector `x`, in column order. SciPy matrix right sides are not supported.
/// # Errors
/// Invalid storage/shape, nonfinite input/output, singularity, or backend failure.
/// # Numerical behavior
/// Uses [`splu`] and [`SparseLu::solve`], including their accuracy/work limits.
/// # Examples
/// ```
/// assert_eq!(sci_sparse::linalg::spsolve(&sci_sparse::construct::identity(2)?, &[3.0, 4.0])?, vec![3.0, 4.0]);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy spsolve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html).
pub fn spsolve(a: &SparseMatrix, b: &[f64]) -> Result<Vec<f64>> {
    validate(a)?;
    if b.len() != a.rows {
        return Err(Error::DimensionMismatch);
    }
    sci_core::is_finite_slice(b)?;
    splu(a)?.solve(b)
}

/// Solve a triangular system `A x = b` by substitution.
///
/// # Parameters and conventions
/// `lower` selects lower/upper triangular storage. If `unit_diagonal` is true,
/// diagonal entries are treated as one, even if absent or explicitly zero.
/// `b.len() == a.rows == a.cols`; off-triangle nonzero entries are rejected.
/// # Domain and special values
/// Finite values only; a zero-order system is accepted. Nonunit diagonal
/// entries must be nonzero. Off-triangle explicit zeros are allowed.
/// # Returns
/// The solution vector in column order.
/// # Errors
/// Invalid CSR, shapes, triangular structure, zero pivots, or nonfinite arithmetic.
/// # Numerical behavior
/// O(rows + stored entries) substitution with f64 arithmetic and no pivoting.
/// Small pivots and cancellation can severely reduce accuracy.
/// # Examples
/// ```
/// let a = sci_sparse::construct::from_dense(&[vec![2.0, 0.0], vec![1.0, 3.0]])?;
/// assert_eq!(sci_sparse::linalg::spsolve_triangular(&a, &[2.0, 7.0], true, false)?, vec![1.0, 2.0]);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy spsolve_triangular](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve_triangular.html).
pub fn spsolve_triangular(
    a: &SparseMatrix,
    b: &[f64],
    lower: bool,
    unit_diagonal: bool,
) -> Result<Vec<f64>> {
    validate(a)?;
    if a.rows != a.cols || b.len() != a.rows {
        return Err(Error::DimensionMismatch);
    }
    sci_core::is_finite_slice(b)?;
    let mut x = vec![0.0; a.cols];
    for step in 0..a.rows {
        let r = if lower { step } else { a.rows - 1 - step };
        let mut rhs = b[r];
        let mut diagonal = if unit_diagonal { 1.0 } else { 0.0 };
        for p in a.indptr[r]..a.indptr[r + 1] {
            let c = a.indices[p];
            let v = a.data[p];
            if c == r {
                if !unit_diagonal {
                    diagonal = v;
                }
            } else if (lower && c > r) || (!lower && c < r) {
                if v != 0.0 {
                    return Err(Error::InvalidInput("matrix is not triangular"));
                }
            } else {
                rhs = match rhs
                    - match v * x[c] {
                        value if value.is_finite() => Ok(value),
                        _ => Err(Error::NonFiniteEvaluation),
                    }? {
                    value if value.is_finite() => Ok(value),
                    _ => Err(Error::NonFiniteEvaluation),
                }?;
            }
        }
        if diagonal == 0.0 {
            return Err(Error::NumericalFailure("zero triangular pivot"));
        }
        x[r] = match rhs / diagonal {
            value if value.is_finite() => Ok(value),
            _ => Err(Error::NonFiniteEvaluation),
        }?;
    }
    Ok(x)
}

/// Matrix norm selection for [`norm`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixNorm {
    /// `sqrt(Σ_ij |A[i,j]|²)`.
    Frobenius,
    /// Largest singular value (spectral 2-norm), axes at most 256.
    Spectral,
    /// Maximum absolute column sum.
    One,
    /// Minimum absolute column sum (SciPy `ord=-1`; not a mathematical norm).
    NegOne,
    /// Maximum absolute row sum.
    Infinity,
    /// Minimum absolute row sum (SciPy `ord=-inf`; not a mathematical norm).
    NegInfinity,
}

/// Compute a matrix norm or minimum absolute row/column sum.
///
/// # Parameters and conventions
/// `a` is canonical CSR; `order` selects the formula in [`MatrixNorm`].
/// Whole-matrix norms use this entry point; [`norm_axis`] supplies vector-axis norms.
/// # Domain and special values
/// Finite entries. Frobenius accepts empty axes and returns zero. Other orders
/// require both axes nonempty; implicit zeros contribute zero to all sums.
/// # Returns
/// A nonnegative scalar in the same units as the entries.
/// # Errors
/// Invalid CSR, unsupported empty shape, or overflow of the result/sums.
/// # Numerical behavior
/// O(rows + cols + stored entries). Frobenius uses repeated `hypot` to avoid
/// intermediate squaring overflow/underflow; induced sums use sequential f64 addition.
/// Spectral norm uses the bounded dense SVD fallback with axes <=256.
/// # Examples
/// ```
/// use sci_sparse::linalg::{norm, MatrixNorm};
/// assert_eq!(norm(&sci_sparse::construct::identity(4)?, MatrixNorm::Frobenius)?, 2.0);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy sparse norm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.norm.html).
pub fn norm(a: &SparseMatrix, order: MatrixNorm) -> Result<f64> {
    validate(a)?;
    if order == MatrixNorm::Frobenius {
        return match a.data.iter().fold(0.0_f64, |s, &v| s.hypot(v)) {
            value if value.is_finite() => Ok(value),
            _ => Err(Error::NonFiniteEvaluation),
        };
    }
    if order == MatrixNorm::Spectral {
        return Ok(svds(a, 1, true)?.singular_values[0]);
    }
    if a.rows == 0 || a.cols == 0 {
        return Err(Error::InvalidInput("induced norm requires nonempty axes"));
    }
    let column = matches!(order, MatrixNorm::One | MatrixNorm::NegOne);
    let minimum = matches!(order, MatrixNorm::NegOne | MatrixNorm::NegInfinity);
    let mut sums = vec![0.0; if column { a.cols } else { a.rows }];
    for (r, c, v) in entries(a) {
        let i = if column { c } else { r };
        sums[i] = match sums[i] + v.abs() {
            value if value.is_finite() => Ok(value),
            _ => Err(Error::NonFiniteEvaluation),
        }?;
    }
    Ok(sums
        .into_iter()
        .fold(if minimum { f64::INFINITY } else { 0.0 }, |s, v| {
            if minimum { s.min(v) } else { s.max(v) }
        }))
}

/// Compute `A^power` using matrix multiplication, with `A^0 = I`.
///
/// # Parameters and conventions
/// `a` must be square canonical CSR; `power` is a nonnegative integer up to 1024.
/// # Domain and special values
/// Finite entries; the zero-order square matrix is supported. Negative powers
/// and SciPy's dense output options are unsupported.
/// # Returns
/// Canonical CSR with exact cancellation zeros omitted after multiplication.
/// # Errors
/// Invalid/nonsquare input, power above 1024, or multiplication/finite-value limits.
/// # Numerical behavior
/// Exponentiation by squaring, at most 20 sparse products; each product retains
/// the crate's 20,000,000 scalar-product bound. Error can grow with power and conditioning.
/// # Examples
/// ```
/// let a = sci_sparse::construct::identity(3)?;
/// assert_eq!(sci_sparse::linalg::matrix_power(&a, 7)?, a);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy matrix_power](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.matrix_power.html).
pub fn matrix_power(a: &SparseMatrix, mut power: u32) -> Result<SparseMatrix> {
    validate(a)?;
    if a.rows != a.cols {
        return Err(Error::DimensionMismatch);
    }
    if power > 1024 {
        return Err(Error::InvalidInput("matrix power exceeds 1024"));
    }
    let mut out = identity(a.rows)?;
    let mut base = a.clone();
    while power > 0 {
        if power & 1 != 0 {
            out = matmul(&out, &base)?;
        }
        power >>= 1;
        if power > 0 {
            base = matmul(&base, &base)?;
        }
    }
    Ok(out)
}

/// Compute the inverse `B` satisfying `A B = I`.
///
/// # Parameters and conventions
/// `a` is finite, square, nonsingular canonical CSR.
/// # Domain and special values
/// Requires `1..=1000` rows, bounding the potentially dense inverse to 1,000,000 entries.
/// # Returns
/// CSR with exact-zero inverse entries omitted; the inverse may be dense.
/// # Errors
/// Invalid input, unsupported shape/order, singularity, backend or nonfinite solve failure.
/// # Numerical behavior
/// One sparse LU factorization followed by one solve per identity column.
/// Work and fill-in may be cubic and quadratic in the order. No condition estimate
/// or iterative refinement; solving a system directly usually has lower cost.
/// # Examples
/// ```
/// let a = sci_sparse::construct::identity(2)?;
/// assert_eq!(sci_sparse::linalg::inv(&a)?, a);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy sparse inv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.inv.html).
pub fn inv(a: &SparseMatrix) -> Result<SparseMatrix> {
    validate(a)?;
    if a.rows.checked_mul(a.cols).is_none_or(|n| n > LIMIT) {
        return Err(Error::InvalidInput("inverse may exceed 1,000,000 entries"));
    }
    let lu = splu(a)?;
    let mut coordinates = Vec::new();
    let mut rhs = vec![0.0; a.rows];
    for c in 0..a.cols {
        rhs[c] = 1.0;
        for (r, v) in lu.solve(&rhs)?.into_iter().enumerate() {
            if v != 0.0 {
                coordinates.push((r, c, v));
            }
        }
        rhs[c] = 0.0;
    }
    super::from_coo(a.rows, a.cols, &coordinates)
}

/// Return lower and upper bandwidths of the numerical nonzero pattern.
///
/// # Parameters and conventions
/// `a` is canonical CSR and may be rectangular. Lower bandwidth is
/// `max(i-j,0)` over nonzero `A[i,j]`; upper bandwidth is `max(j-i,0)`.
/// # Domain and special values
/// Finite entries; empty and all-zero matrices have bandwidths `(0,0)`.
/// Explicit zeros are ignored.
/// # Returns
/// `(lower, upper)` as nonnegative integers.
/// # Errors
/// Invalid CSR or crate limits.
/// # Numerical behavior
/// Exact index arithmetic in O(rows + stored entries) work.
/// # Examples
/// ```
/// assert_eq!(sci_sparse::linalg::spbandwidth(&sci_sparse::construct::eye(2, 3, 1)?)?, (0, 1));
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy spbandwidth](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spbandwidth.html).
pub fn spbandwidth(a: &SparseMatrix) -> Result<(usize, usize)> {
    validate(a)?;
    let (mut lower, mut upper) = (0, 0);
    for (r, c, v) in entries(a) {
        if v != 0.0 {
            lower = lower.max(r.saturating_sub(c));
            upper = upper.max(c.saturating_sub(r));
        }
    }
    Ok((lower, upper))
}

/// Determine whether `A` is lower triangular and upper triangular.
///
/// # Parameters and conventions
/// `a` is canonical CSR, including rectangular shapes; explicit zeros are ignored.
/// Lower means `A[i,j]=0` for `j>i`; upper means zero for `i>j`.
/// # Domain and special values
/// Finite entries; diagonal, empty, and all-zero matrices satisfy both conditions.
/// # Returns
/// `(is_lower, is_upper)`, following SciPy's output ordering.
/// # Errors
/// Invalid CSR or crate limits.
/// # Numerical behavior
/// Exact nonzero-pattern checks using [`spbandwidth`].
/// # Examples
/// ```
/// assert_eq!(sci_sparse::linalg::is_sptriangular(&sci_sparse::construct::identity(2)?)?, (true, true));
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy is_sptriangular](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.is_sptriangular.html).
pub fn is_sptriangular(a: &SparseMatrix) -> Result<(bool, bool)> {
    let (lower, upper) = spbandwidth(a)?;
    Ok((upper == 0, lower == 0))
}

mod ilu;
mod iterative;
mod spectral;
pub use ilu::{SparseIlu, spilu};
pub use iterative::*;
pub use spectral::*;
