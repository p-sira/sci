mod advanced;
mod backend;
mod cs;
mod dense;
mod extra;
mod functions;
mod input;
pub mod interpolative;
mod pencil;
mod principal;
mod qr;
mod riccati;
mod structure;
#[path = "structured.rs"]
mod structured_impl;

pub use advanced::*;
pub use cs::*;
pub use dense::*;
pub use extra::*;
pub use functions::*;
pub use input::{MatrixInput, VectorInput};
pub use pencil::*;
pub use principal::*;
pub use qr::*;
pub use riccati::*;
pub use structure::*;
pub use structured_impl::*;

/// Basic matrix and vector operations, matrix inverses, norms and properties.
///
/// For example, [`matmul`] computes a matrix product and [`norm`] evaluates
/// the requested matrix norm. Matrix and vector inputs use the shared
/// [`crate::MatrixInput`] and [`crate::VectorInput`] adapters.
pub mod basic {
    pub use super::{
        MatrixNorm, bandwidth, det, dot, inv, ishermitian, issymmetric, khatri_rao, matmul, matvec,
        norm, pinv, pinvh, transpose,
    };
}

/// Factorizations of dense, structured and banded matrices.
///
/// Result types accompany their associated decomposition functions.
pub mod decompositions {
    pub use super::{
        CholeskyFactor, CsResult, LdlResult, LuFactor, LuResult, QrResult, SvdResult, cho_factor,
        cholesky, cholesky_banded, cossin, hessenberg, ldl, lu, lu_factor, matrix_balance, polar,
        qr, qr_delete, qr_insert, qr_multiply, qr_update, rq, rsf2csf, schur, svd, svdvals,
    };
}

/// Linear systems, least squares, and matrix equations.
pub mod solvers {
    pub use super::{
        LeastSquaresResult, cho_solve, cho_solve_banded, lstsq, lu_solve, solve, solve_banded,
        solve_circulant, solve_continuous_are, solve_continuous_lyapunov, solve_discrete_are,
        solve_discrete_lyapunov, solve_sylvester, solve_toeplitz, solve_triangular, solveh_banded,
    };
}

/// Standard and generalized eigenvalue problems.
pub mod eigen {
    pub use super::{
        ComplexMatrix, EigenResult, GeneralEigenResult, QzResult, cdf2rdf, eig, eig_banded, eigh,
        eigh_tridiagonal, eigvals, eigvals_banded, eigvalsh, eigvalsh_tridiagonal, ordqz, qz,
    };
}

/// Matrix-valued functions and their numerical derivatives and condition estimates.
pub mod matrix_functions {
    pub use super::{
        coshm, cosm, expm, expm_cond, expm_frechet, fractional_matrix_power, funm, logm, signm,
        sinhm, sinm, sqrtm, tanhm, tanm,
    };
}

/// Structured matrix constructors and their specialized products.
pub mod structured {
    pub use super::{
        DftScale, PascalKind, block_diag, circulant, companion, convolution_matrix, dft, diagsvd,
        fiedler, fiedler_companion, hadamard, hankel, helmert, hilbert, invhilbert, invpascal,
        leslie, matmul_toeplitz, pascal, toeplitz,
    };
}

/// Orthogonal subspaces, geometric comparisons, and sketching.
pub mod subspaces {
    pub use super::{
        clarkson_woodruff_transform, null_space, orth, orthogonal_procrustes, subspace_angles,
    };
}

use crate::{Error, Result};

/// Computes the inner product `Σ x[i] y[i]`.
///
/// ## Parameters
///
/// `x` and `y` are vectors of equal length. Empty vectors have dot product zero.
/// Accepts [`VectorInput`] storage with at most 1,000,000 entries per vector.
/// IEEE values propagate through the faer reduction.
///
/// ## Errors
///
/// Returns [`Error::DimensionMismatch`] when vector lengths differ.
/// Returns [`Error::InvalidInput`] when an input exceeds the work bound.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::basic::dot(&[1.0, 2.0], &[3.0, 4.0])?, 11.0);
/// # Ok::<(), sci_core::Error>(())
/// ```
pub fn dot(x: &(impl VectorInput + ?Sized), y: &(impl VectorInput + ?Sized)) -> Result<f64> {
    let x_input = input::vector_slice(x)?;
    let x = x_input.as_ref();
    let y_input = input::vector_slice(y)?;
    let y = y_input.as_ref();
    if x.len() != y.len() {
        return Err(Error::DimensionMismatch);
    }
    Ok(faer::ColRef::from_slice(x).transpose() * faer::ColRef::from_slice(y))
}

/// Computes matrix-vector product `y[i] = Σ A[i][j] x[j]`.
///
/// ## Parameters
///
/// `matrix` accepts [`MatrixInput`]; `vector` accepts [`VectorInput`] and must
/// match its column count. Empty axes are supported; an empty shared matrix
/// represents `0 × 0`. IEEE values propagate. Each axis and input entry count
/// are at most 1,000,000.
///
/// ## Errors
///
/// Returns [`Error::DimensionMismatch`] for ragged or incompatible inputs.
/// Returns [`Error::InvalidInput`] when an input exceeds the work bound.
///
/// # Examples
///
/// ```
/// let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
/// assert_eq!(sci_linalg::basic::matvec(&a, &[2.0, 1.0])?, vec![4.0, 10.0]);
/// # Ok::<(), sci_core::Error>(())
/// ```
pub fn matvec(
    matrix: &(impl MatrixInput + ?Sized),
    vector: &(impl VectorInput + ?Sized),
) -> Result<Vec<f64>> {
    let vector_input = input::vector_slice(vector)?;
    let vector = vector_input.as_ref();
    let matrix = input::faer_input(matrix)?;
    let matrix = matrix.as_ref();
    if matrix.ncols() != vector.len() {
        return Err(Error::DimensionMismatch);
    }
    Ok((matrix * faer::ColRef::from_slice(vector))
        .iter()
        .copied()
        .collect())
}

/// Computes matrix product `C[i][j] = Σ A[i][k] B[k][j]`.
///
/// ## Parameters
///
/// Matrices accept independent [`MatrixInput`] storage. The left column count
/// must equal the right row count. Empty axes are supported; an empty shared
/// matrix represents `0 × 0`. IEEE values propagate. Each input/output axis
/// and entry count are at most 1,000,000.
/// Matrix multiplication is additionally limited to 16,777,216 scalar
/// products (`rows * inner_dimension * columns`), checked before input copying.
///
/// ## Domain
///
/// Rectangular compatible matrices within the size/work bounds. All IEEE
/// values are accepted, including NaN and infinity; arithmetic propagates them.
///
/// ## Returns
///
/// Shared rows with shape left.rows × right.columns. Output entries have
/// the product of the input units. Empty shared rows cannot retain a column
/// count for zero-row output.
///
/// ## Errors
///
/// Returns [`Error::DimensionMismatch`] for ragged or incompatible matrices.
/// Returns [`Error::InvalidInput`] when an input or output exceeds the work bound.
///
/// ## Numerical behavior
///
/// Private faer multiplication takes O(rows × inner_dimension × columns)
/// work. Compatible native storage is borrowed, including strided faer views;
/// other inputs copy directly to backend storage. Reductions use f64 and can
/// lose accuracy through cancellation or overflow.
///
/// # Examples
///
/// ```
/// let a = vec![vec![1.0, 2.0]];
/// let b = vec![vec![3.0], vec![4.0]];
/// assert_eq!(sci_linalg::basic::matmul(&a, &b)?, vec![vec![11.0]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
pub fn matmul(
    left: &(impl MatrixInput + ?Sized),
    right: &(impl MatrixInput + ?Sized),
) -> Result<Vec<Vec<f64>>> {
    let (m, k) = input::checked_shape(left)?;
    let (r, n) = input::checked_shape(right)?;
    if k != r {
        return Err(Error::DimensionMismatch);
    }
    backend::size(m, n)?;
    if m.checked_mul(k)
        .and_then(|work| work.checked_mul(n))
        .is_none_or(|work| work > 16_777_216)
    {
        return Err(Error::InvalidInput(
            "matrix multiplication work exceeds 16777216",
        ));
    }
    let left = input::faer_input(left)?;
    let right = input::faer_input(right)?;
    let product = left.as_ref() * right.as_ref();
    Ok((0..m)
        .map(|i| (0..n).map(|j| product[(i, j)]).collect())
        .collect())
}

/// Transposes a matrix: `B[j][i] = A[i][j]`.
///
/// ## Parameters
///
/// Accepts [`MatrixInput`] storage, including empty axes. Each axis and the
/// entry count are at most 1,000,000. IEEE values are copied unchanged.
///
/// ## Returns
///
/// Shared row vectors of shape columns × rows.
///
/// ## Errors
///
/// Returns [`Error::DimensionMismatch`] for ragged rows. Zero-column matrices
/// transpose to an empty matrix, whose representation cannot retain a column count.
/// Returns [`Error::InvalidInput`] when the input exceeds the work bound.
///
/// ## Numerical behavior
///
/// Copies logical entries directly into output rows in O(rows × columns) work,
/// without an intermediate backend matrix or floating-point arithmetic.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::basic::transpose(&[vec![1.0, 2.0]])?,
///            vec![vec![1.0], vec![2.0]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
pub fn transpose(matrix: &(impl MatrixInput + ?Sized)) -> Result<Vec<Vec<f64>>> {
    let (m, n) = input::checked_shape(matrix)?;
    Ok((0..n)
        .map(|j| (0..m).map(|i| matrix.matrix_entry(i, j)).collect())
        .collect())
}
