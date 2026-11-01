//! Read-only interoperability with dense matrix and vector libraries.
use super::backend::size;
use crate::{Error, Result};
use std::borrow::Cow;

/// A read-only real matrix, indexed by logical `(row, column)` coordinates.
///
/// Implemented for row vectors, fixed arrays, ndarray arrays/views, nalgebra
/// matrices/views, and faer matrices/views under the corresponding feature
/// flags. Layout and strides do not affect
/// entry ordering. Implementors must report a stable rectangular shape and
/// provide every entry within it. Algorithms never mutate their inputs.
/// The numerical backend is always faer, independently of adapter features.
///
/// Conversions copy IEEE values without arithmetic or finite-value validation.
/// All conversions bound each axis and the entry count at 1,000,000; numerical
/// operations retain their own tighter bounds and finite-value requirements.
/// Numerical calls pass logical entries directly to the private faer backend.
/// Faer matrices/views, fixed arrays, and contiguous C/F-order ndarray storage
/// can be borrowed. Other layouts and shared row vectors are copied directly
/// into backend storage, without intermediate shared rows. Algorithms that
/// scale or overwrite their working matrices still make an owned copy.
///
/// # Examples
///
/// ```
/// # #[cfg(all(feature = "ndarray", feature = "nalgebra", feature = "faer"))]
/// # {
/// use sci_linalg::{self as linalg, MatrixInput};
/// let a = ndarray::arr2(&[[2.0, 1.0], [0.0, 3.0]]);
/// let b = nalgebra::DMatrix::identity(2, 2);
/// let product = linalg::basic::matmul(&a.view(), &b)?;
/// let faer_matrix = product.to_faer()?;
/// assert_eq!(linalg::solvers::solve(&faer_matrix, &[5.0, 6.0])?, vec![1.5, 2.0]);
/// # }
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [ndarray arrays](https://docs.rs/ndarray/0.16.1/ndarray/struct.ArrayBase.html),
/// [nalgebra matrices](https://docs.rs/nalgebra/0.33.2/nalgebra/base/struct.Matrix.html),
/// and [faer matrices](https://docs.rs/faer/0.24.4/faer/).
pub trait MatrixInput {
    /// Returns `(rows, columns)`, rejecting ragged rows with
    /// [`Error::DimensionMismatch`]. This method does not impose a work bound.
    fn matrix_shape(&self) -> Result<(usize, usize)>;

    /// Reads an entry. May panic if either index is outside [`Self::matrix_shape`].
    fn matrix_entry(&self, row: usize, column: usize) -> f64;

    /// Borrows shared rows, when available, to avoid an input copy.
    /// Implementors returning `Some` must match the reported shape and entries.
    #[doc(hidden)]
    #[inline]
    fn shared_rows(&self) -> Option<&[Vec<f64>]> {
        None
    }

    /// Borrows native faer-compatible storage when available. This storage hook
    /// must match the reported shape and logical entries, including strides.
    /// Numerical backends and their decomposition objects remain private.
    #[doc(hidden)]
    #[inline]
    fn faer_view(&self) -> Option<faer::MatRef<'_, f64>> {
        None
    }

    /// Copies `A[i,j]` into owned shared row vectors, preserving entry order.
    ///
    /// ## Returns
    ///
    /// A matrix with one vector per row. IEEE values and signed zero are preserved.
    ///
    /// ## Errors
    ///
    /// Ragged rows or an axis/entry count above 1,000,000. A `0 × n` shape with
    /// `n > 0` returns [`Error::DimensionMismatch`], because shared row vectors
    /// cannot represent its column count. Other conversions preserve this shape.
    ///
    /// ## Numerical behavior
    ///
    /// O(rows × columns) copying; no rounding or backend numerical computation.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "nalgebra")]
    /// # {
    /// use sci_linalg::MatrixInput;
    /// let a = nalgebra::DMatrix::from_row_slice(1, 2, &[1.0, 2.0]);
    /// assert_eq!(a.to_matrix()?, vec![vec![1.0, 2.0]]);
    /// # }
    /// # Ok::<(), sci_core::Error>(())
    /// ```
    fn to_matrix(&self) -> Result<Vec<Vec<f64>>> {
        Ok(matrix_rows(self)?.into_owned())
    }

    /// Copies `A[i,j]` to an owned ndarray array of the same shape.
    ///
    /// Preserves empty axes and IEEE values; uses O(rows × columns) work.
    /// Returns a shared error for ragged rows or an axis/entry count above
    /// 1,000,000. See [`MatrixInput`] for conventions and examples.
    /// Requires the `ndarray` interoperability feature.
    #[cfg(feature = "ndarray")]
    fn to_ndarray(&self) -> Result<ndarray::Array2<f64>> {
        let shape = checked_shape(self)?;
        Ok(ndarray::Array2::from_shape_fn(shape, |(i, j)| {
            self.matrix_entry(i, j)
        }))
    }

    /// Copies `A[i,j]` to an owned dynamic nalgebra matrix of the same shape.
    ///
    /// Preserves empty axes and IEEE values; uses O(rows × columns) work.
    /// Returns a shared error for ragged rows or an axis/entry count above
    /// 1,000,000. See [`MatrixInput`] for conventions and examples.
    /// Requires the `nalgebra` interoperability feature.
    #[cfg(feature = "nalgebra")]
    fn to_nalgebra(&self) -> Result<nalgebra::DMatrix<f64>> {
        let (m, n) = checked_shape(self)?;
        Ok(nalgebra::DMatrix::from_fn(m, n, |i, j| {
            self.matrix_entry(i, j)
        }))
    }

    /// Copies `A[i,j]` to an owned faer matrix of the same shape.
    ///
    /// Preserves empty axes and IEEE values; uses O(rows × columns) work.
    /// Returns a shared error for ragged rows or an axis/entry count above
    /// 1,000,000. See [`MatrixInput`] for conventions and examples.
    /// Requires the `faer` interoperability feature.
    #[cfg(feature = "faer")]
    fn to_faer(&self) -> Result<faer::Mat<f64>> {
        faer_matrix(self)
    }
}

pub(super) fn checked_shape(a: &(impl MatrixInput + ?Sized)) -> Result<(usize, usize)> {
    let (m, n) = a.matrix_shape()?;
    size(m, n)?;
    Ok((m, n))
}

pub(super) fn matrix_rows(a: &(impl MatrixInput + ?Sized)) -> Result<Cow<'_, [Vec<f64>]>> {
    let (m, n) = checked_shape(a)?;
    if m == 0 && n != 0 {
        return Err(Error::DimensionMismatch);
    }
    Ok(match a.shared_rows() {
        Some(rows) => Cow::Borrowed(rows),
        None => Cow::Owned(
            (0..m)
                .map(|i| (0..n).map(|j| a.matrix_entry(i, j)).collect())
                .collect(),
        ),
    })
}

impl MatrixInput for [Vec<f64>] {
    #[inline]
    fn matrix_shape(&self) -> Result<(usize, usize)> {
        let n = self.first().map_or(0, Vec::len);
        if self.iter().any(|row| row.len() != n) {
            return Err(Error::DimensionMismatch);
        }
        Ok((self.len(), n))
    }
    #[inline]
    fn matrix_entry(&self, row: usize, column: usize) -> f64 {
        self[row][column]
    }
    #[inline]
    fn shared_rows(&self) -> Option<&[Vec<f64>]> {
        Some(self)
    }
}

impl MatrixInput for Vec<Vec<f64>> {
    #[inline]
    fn matrix_shape(&self) -> Result<(usize, usize)> {
        self.as_slice().matrix_shape()
    }
    #[inline]
    fn matrix_entry(&self, row: usize, column: usize) -> f64 {
        self[row][column]
    }
    #[inline]
    fn shared_rows(&self) -> Option<&[Vec<f64>]> {
        Some(self)
    }
}

impl<const N: usize> MatrixInput for [Vec<f64>; N] {
    #[inline]
    fn matrix_shape(&self) -> Result<(usize, usize)> {
        self.as_slice().matrix_shape()
    }
    #[inline]
    fn matrix_entry(&self, row: usize, column: usize) -> f64 {
        self[row][column]
    }
    #[inline]
    fn shared_rows(&self) -> Option<&[Vec<f64>]> {
        Some(self)
    }
}

impl<const N: usize> MatrixInput for [[f64; N]] {
    #[inline]
    fn faer_view(&self) -> Option<faer::MatRef<'_, f64>> {
        Some(faer::MatRef::from_row_major_slice(
            self.as_flattened(),
            self.len(),
            N,
        ))
    }

    #[inline]
    fn matrix_shape(&self) -> Result<(usize, usize)> {
        Ok((self.len(), N))
    }
    #[inline]
    fn matrix_entry(&self, row: usize, column: usize) -> f64 {
        self[row][column]
    }
}

impl<const M: usize, const N: usize> MatrixInput for [[f64; N]; M] {
    #[inline]
    fn faer_view(&self) -> Option<faer::MatRef<'_, f64>> {
        Some(faer::MatRef::from_row_major_slice(
            self.as_flattened(),
            self.len(),
            N,
        ))
    }

    #[inline]
    fn matrix_shape(&self) -> Result<(usize, usize)> {
        Ok((M, N))
    }
    #[inline]
    fn matrix_entry(&self, row: usize, column: usize) -> f64 {
        self[row][column]
    }
}

#[cfg(feature = "ndarray")]
impl<S: ndarray::Data<Elem = f64>> MatrixInput for ndarray::ArrayBase<S, ndarray::Ix2> {
    #[inline]
    fn matrix_shape(&self) -> Result<(usize, usize)> {
        Ok(self.dim())
    }
    #[inline]
    fn faer_view(&self) -> Option<faer::MatRef<'_, f64>> {
        let (m, n) = self.dim();
        if let Some(slice) = self.as_slice() {
            Some(faer::MatRef::from_row_major_slice(slice, m, n))
        } else if self.t().is_standard_layout() {
            Some(faer::MatRef::from_column_major_slice(
                self.as_slice_memory_order()?,
                m,
                n,
            ))
        } else {
            None
        }
    }
    #[inline]
    fn matrix_entry(&self, row: usize, column: usize) -> f64 {
        self[(row, column)]
    }
}

#[cfg(feature = "nalgebra")]
impl<R: nalgebra::Dim, C: nalgebra::Dim, S: nalgebra::Storage<f64, R, C>> MatrixInput
    for nalgebra::Matrix<f64, R, C, S>
{
    #[inline]
    fn matrix_shape(&self) -> Result<(usize, usize)> {
        Ok(self.shape())
    }
    #[inline]
    fn matrix_entry(&self, row: usize, column: usize) -> f64 {
        self[(row, column)]
    }
}

#[cfg(feature = "faer")]
impl<R: faer::Shape, C: faer::Shape> MatrixInput for faer::Mat<f64, R, C> {
    #[inline]
    fn matrix_shape(&self) -> Result<(usize, usize)> {
        Ok((self.nrows().unbound(), self.ncols().unbound()))
    }
    #[inline]
    fn matrix_entry(&self, row: usize, column: usize) -> f64 {
        self.as_ref().as_dyn()[(row, column)]
    }
    #[inline]
    fn faer_view(&self) -> Option<faer::MatRef<'_, f64>> {
        Some(self.as_ref().as_dyn().as_dyn_stride())
    }
}

#[cfg(feature = "faer")]
impl<R: faer::Shape, C: faer::Shape, RS: faer::Stride, CS: faer::Stride> MatrixInput
    for faer::MatRef<'_, f64, R, C, RS, CS>
{
    #[inline]
    fn matrix_shape(&self) -> Result<(usize, usize)> {
        Ok((self.nrows().unbound(), self.ncols().unbound()))
    }
    #[inline]
    fn matrix_entry(&self, row: usize, column: usize) -> f64 {
        self.as_ref().as_dyn()[(row, column)]
    }
    #[inline]
    fn faer_view(&self) -> Option<faer::MatRef<'_, f64>> {
        Some(self.as_ref().as_dyn().as_dyn_stride())
    }
}

#[cfg(feature = "faer")]
impl<R: faer::Shape, C: faer::Shape, RS: faer::Stride, CS: faer::Stride> MatrixInput
    for faer::MatMut<'_, f64, R, C, RS, CS>
{
    #[inline]
    fn matrix_shape(&self) -> Result<(usize, usize)> {
        Ok((self.nrows().unbound(), self.ncols().unbound()))
    }
    #[inline]
    fn matrix_entry(&self, row: usize, column: usize) -> f64 {
        self.as_ref().as_dyn()[(row, column)]
    }
    #[inline]
    fn faer_view(&self) -> Option<faer::MatRef<'_, f64>> {
        Some(self.as_ref().as_dyn().as_dyn_stride())
    }
}

impl<M: MatrixInput + ?Sized> MatrixInput for &M {
    #[inline]
    fn matrix_shape(&self) -> Result<(usize, usize)> {
        (**self).matrix_shape()
    }
    #[inline]
    fn matrix_entry(&self, row: usize, column: usize) -> f64 {
        (**self).matrix_entry(row, column)
    }
    #[inline]
    fn shared_rows(&self) -> Option<&[Vec<f64>]> {
        (**self).shared_rows()
    }
    #[inline]
    fn faer_view(&self) -> Option<faer::MatRef<'_, f64>> {
        (**self).faer_view()
    }
}

/// A read-only real vector in logical index order.
///
/// Implemented for slices/arrays, ndarray one-dimensional arrays/views, nalgebra
/// column vectors/views, and faer columns/rows/views. Noncontiguous inputs are
/// copied; shared or contiguous slices are borrowed when possible. Algorithms
/// retain their existing scalar domains and never mutate inputs. Optional
/// vector arguments retain slice signatures; use [`Self::to_vector`] for them.
///
/// # Examples
///
/// ```
/// # #[cfg(all(feature = "ndarray", feature = "nalgebra", feature = "faer"))]
/// # {
/// use sci_linalg::{self as linalg, VectorInput};
/// let x = ndarray::arr1(&[1.0, 2.0, 3.0]);
/// let y = nalgebra::Vector3::new(3.0, 2.0, 1.0);
/// assert_eq!(linalg::basic::dot(&x, &y)?, 10.0);
/// assert_eq!(x.to_faer_col()?.nrows(), 3);
/// # }
/// # Ok::<(), sci_core::Error>(())
/// ```
pub trait VectorInput {
    /// Returns the number of entries.
    fn vector_len(&self) -> usize;
    /// Reads an entry. May panic if `index >= self.vector_len()`.
    fn vector_entry(&self, index: usize) -> f64;
    /// Borrows entries in logical order when contiguous.
    /// Implementors returning `Some` must match the reported length and entries.
    #[doc(hidden)]
    #[inline]
    fn shared_slice(&self) -> Option<&[f64]> {
        None
    }

    /// Copies `x[i]` to a vector in logical order, preserving all IEEE values.
    ///
    /// ## Returns
    ///
    /// An owned vector with the same length, including empty vectors.
    ///
    /// ## Errors
    ///
    /// Lengths above 1,000,000 return [`Error::InvalidInput`].
    ///
    /// ## Numerical behavior
    ///
    /// O(length) copying with no arithmetic or finite-value checks.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "ndarray")]
    /// # {
    /// use sci_linalg::VectorInput;
    /// assert_eq!(ndarray::arr1(&[1.0, 2.0]).to_vector()?, vec![1.0, 2.0]);
    /// # }
    /// # Ok::<(), sci_core::Error>(())
    /// ```
    fn to_vector(&self) -> Result<Vec<f64>> {
        Ok(vector_slice(self)?.into_owned())
    }

    /// Copies entries in logical order to an owned ndarray vector.
    /// Preserves IEEE values, takes O(length) work, and rejects lengths above
    /// 1,000,000. See [`VectorInput`] for conventions and examples.
    /// Requires the `ndarray` interoperability feature.
    #[cfg(feature = "ndarray")]
    fn to_ndarray_vector(&self) -> Result<ndarray::Array1<f64>> {
        Ok(ndarray::Array1::from_vec(self.to_vector()?))
    }
    /// Copies entries in logical order to an owned dynamic nalgebra column.
    /// Preserves IEEE values, takes O(length) work, and rejects lengths above
    /// 1,000,000. See [`VectorInput`] for conventions and examples.
    /// Requires the `nalgebra` interoperability feature.
    #[cfg(feature = "nalgebra")]
    fn to_nalgebra_vector(&self) -> Result<nalgebra::DVector<f64>> {
        Ok(nalgebra::DVector::from_vec(self.to_vector()?))
    }
    /// Copies entries in logical order to an owned faer column.
    /// Preserves IEEE values, takes O(length) work, and rejects lengths above
    /// 1,000,000. See [`VectorInput`] for conventions and examples.
    /// Requires the `faer` interoperability feature.
    #[cfg(feature = "faer")]
    fn to_faer_col(&self) -> Result<faer::Col<f64>> {
        let x = vector_slice(self)?;
        Ok(faer::Col::from_fn(x.len(), |i| x[i]))
    }
}

pub(super) fn vector_slice(x: &(impl VectorInput + ?Sized)) -> Result<Cow<'_, [f64]>> {
    size(x.vector_len(), 1)?;
    Ok(match x.shared_slice() {
        Some(x) => Cow::Borrowed(x),
        None => Cow::Owned((0..x.vector_len()).map(|i| x.vector_entry(i)).collect()),
    })
}

impl VectorInput for [f64] {
    #[inline]
    fn vector_len(&self) -> usize {
        self.len()
    }
    #[inline]
    fn vector_entry(&self, index: usize) -> f64 {
        self[index]
    }
    #[inline]
    fn shared_slice(&self) -> Option<&[f64]> {
        Some(self)
    }
}
impl VectorInput for Vec<f64> {
    #[inline]
    fn vector_len(&self) -> usize {
        self.len()
    }
    #[inline]
    fn vector_entry(&self, index: usize) -> f64 {
        self[index]
    }
    #[inline]
    fn shared_slice(&self) -> Option<&[f64]> {
        Some(self)
    }
}
impl<const N: usize> VectorInput for [f64; N] {
    #[inline]
    fn vector_len(&self) -> usize {
        N
    }
    #[inline]
    fn vector_entry(&self, index: usize) -> f64 {
        self[index]
    }
    #[inline]
    fn shared_slice(&self) -> Option<&[f64]> {
        Some(self)
    }
}
#[cfg(feature = "ndarray")]
impl<S: ndarray::Data<Elem = f64>> VectorInput for ndarray::ArrayBase<S, ndarray::Ix1> {
    #[inline]
    fn vector_len(&self) -> usize {
        self.len()
    }
    #[inline]
    fn vector_entry(&self, index: usize) -> f64 {
        self[index]
    }
    #[inline]
    fn shared_slice(&self) -> Option<&[f64]> {
        self.as_slice()
    }
}
#[cfg(feature = "nalgebra")]
impl<R: nalgebra::Dim, S: nalgebra::Storage<f64, R>> VectorInput
    for nalgebra::Matrix<f64, R, nalgebra::U1, S>
{
    #[inline]
    fn vector_len(&self) -> usize {
        self.nrows()
    }
    #[inline]
    fn vector_entry(&self, index: usize) -> f64 {
        self[index]
    }
}
impl<V: VectorInput + ?Sized> VectorInput for &V {
    #[inline]
    fn vector_len(&self) -> usize {
        (**self).vector_len()
    }
    #[inline]
    fn vector_entry(&self, index: usize) -> f64 {
        (**self).vector_entry(index)
    }
    #[inline]
    fn shared_slice(&self) -> Option<&[f64]> {
        (**self).shared_slice()
    }
}

#[cfg(feature = "faer")]
impl<D: faer::Shape> VectorInput for faer::Col<f64, D> {
    #[inline]
    fn vector_len(&self) -> usize {
        self.nrows().unbound()
    }
    #[inline]
    fn vector_entry(&self, index: usize) -> f64 {
        self.as_ref().as_dyn_rows()[index]
    }
}

#[cfg(feature = "faer")]
impl<D: faer::Shape, S: faer::Stride> VectorInput for faer::ColRef<'_, f64, D, S> {
    #[inline]
    fn vector_len(&self) -> usize {
        self.nrows().unbound()
    }
    #[inline]
    fn vector_entry(&self, index: usize) -> f64 {
        self.as_ref().as_dyn_rows()[index]
    }
}

#[cfg(feature = "faer")]
impl<D: faer::Shape, S: faer::Stride> VectorInput for faer::ColMut<'_, f64, D, S> {
    #[inline]
    fn vector_len(&self) -> usize {
        self.nrows().unbound()
    }
    #[inline]
    fn vector_entry(&self, index: usize) -> f64 {
        self.as_ref().as_dyn_rows()[index]
    }
}

#[cfg(feature = "faer")]
impl<D: faer::Shape> VectorInput for faer::Row<f64, D> {
    #[inline]
    fn vector_len(&self) -> usize {
        self.ncols().unbound()
    }
    #[inline]
    fn vector_entry(&self, index: usize) -> f64 {
        self.as_ref().as_dyn_cols()[index]
    }
}

#[cfg(feature = "faer")]
impl<D: faer::Shape, S: faer::Stride> VectorInput for faer::RowRef<'_, f64, D, S> {
    #[inline]
    fn vector_len(&self) -> usize {
        self.ncols().unbound()
    }
    #[inline]
    fn vector_entry(&self, index: usize) -> f64 {
        self.as_ref().as_dyn_cols()[index]
    }
}

#[cfg(feature = "faer")]
impl<D: faer::Shape, S: faer::Stride> VectorInput for faer::RowMut<'_, f64, D, S> {
    #[inline]
    fn vector_len(&self) -> usize {
        self.ncols().unbound()
    }
    #[inline]
    fn vector_entry(&self, index: usize) -> f64 {
        self.as_ref().as_dyn_cols()[index]
    }
}

#[cfg(feature = "faer")]
pub(super) fn faer_matrix(a: &(impl MatrixInput + ?Sized)) -> Result<faer::Mat<f64>> {
    let (m, n) = checked_shape(a)?;
    Ok(faer::Mat::from_fn(m, n, |i, j| a.matrix_entry(i, j)))
}

/// Borrow compatible storage, owning only the fallback conversion.
pub(super) enum FaerInput<'a> {
    Borrowed(faer::MatRef<'a, f64>),
    Owned(faer::Mat<f64>),
}
impl FaerInput<'_> {
    pub(super) fn as_ref(&self) -> faer::MatRef<'_, f64> {
        match self {
            Self::Borrowed(a) => *a,
            Self::Owned(a) => a.as_ref(),
        }
    }
    pub(super) fn into_owned(self) -> faer::Mat<f64> {
        match self {
            Self::Borrowed(a) => a.to_owned(),
            Self::Owned(a) => a,
        }
    }
}
pub(super) fn faer_input(a: &(impl MatrixInput + ?Sized)) -> Result<FaerInput<'_>> {
    let (m, n) = checked_shape(a)?;
    Ok(match a.faer_view() {
        Some(view) => FaerInput::Borrowed(view),
        None => FaerInput::Owned(faer::Mat::from_fn(m, n, |i, j| a.matrix_entry(i, j))),
    })
}

impl MatrixInput for FaerInput<'_> {
    #[inline]
    fn matrix_shape(&self) -> Result<(usize, usize)> {
        let a = self.as_ref();
        Ok((a.nrows(), a.ncols()))
    }
    #[inline]
    fn matrix_entry(&self, row: usize, column: usize) -> f64 {
        self.as_ref()[(row, column)]
    }
    #[inline]
    fn faer_view(&self) -> Option<faer::MatRef<'_, f64>> {
        Some(self.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::{FaerInput, MatrixInput};

    #[test]
    fn native_matrix_entries_preserve_logical_order_and_signed_zero() {
        let entries = [[-0.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let matrix = faer::Mat::from_fn(2, 3, |i, j| entries[i][j]);
        let view = matrix.as_ref().transpose().reverse_rows();
        let expected: [[f64; 2]; 3] = [[3.0, 6.0], [2.0, 5.0], [-0.0, 4.0]];
        for input in [FaerInput::Borrowed(view), FaerInput::Owned(view.to_owned())] {
            assert_eq!(input.matrix_shape().unwrap(), (3, 2));
            for (i, row) in expected.iter().enumerate() {
                for (j, value) in row.iter().enumerate() {
                    assert_eq!(input.matrix_entry(i, j).to_bits(), value.to_bits());
                }
            }
        }
    }
}
