mod advanced;
mod backend;
mod cs;
mod dense;
mod dispatch;
mod extra;
mod functions;
mod input;
pub mod interpolative;
mod pencil;
mod principal;
mod qr;
mod riccati;
mod structure;
mod structured;

pub use advanced::*;
pub use cs::*;
pub use dense::*;
pub use dispatch::*;
pub use extra::*;
pub use functions::*;
pub use input::{MatrixInput, VectorInput};
pub use pencil::*;
pub use principal::*;
pub use qr::*;
pub use riccati::*;
pub use structure::*;
pub use structured::*;

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
/// assert_eq!(sci_linalg::dot(&[1.0, 2.0], &[3.0, 4.0])?, 11.0);
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
/// assert_eq!(sci_linalg::matvec(&a, &[2.0, 1.0])?, vec![4.0, 10.0]);
/// # Ok::<(), sci_core::Error>(())
/// ```
pub fn matvec(
    matrix: &(impl MatrixInput + ?Sized),
    vector: &(impl VectorInput + ?Sized),
) -> Result<Vec<f64>> {
    let vector_input = input::vector_slice(vector)?;
    let vector = vector_input.as_ref();
    let matrix = input::faer_matrix(matrix)?;
    if matrix.ncols() != vector.len() {
        return Err(Error::DimensionMismatch);
    }
    Ok((&matrix * faer::ColRef::from_slice(vector))
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
/// work. Inputs are copied into contiguous storage; reductions use f64 and
/// can lose accuracy through cancellation or overflow.
///
/// # Examples
///
/// ```
/// let a = vec![vec![1.0, 2.0]];
/// let b = vec![vec![3.0], vec![4.0]];
/// assert_eq!(sci_linalg::matmul(&a, &b)?, vec![vec![11.0]]);
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
    let left = input::faer_matrix(left)?;
    let right = input::faer_matrix(right)?;
    let product = &left * &right;
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
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::transpose(&[vec![1.0, 2.0]])?,
///            vec![vec![1.0], vec![2.0]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
pub fn transpose(matrix: &(impl MatrixInput + ?Sized)) -> Result<Vec<Vec<f64>>> {
    let matrix = input::faer_matrix(matrix)?;
    let transposed = matrix.transpose();
    Ok((0..transposed.nrows())
        .map(|i| {
            (0..transposed.ncols())
                .map(|j| transposed[(i, j)])
                .collect()
        })
        .collect())
}
