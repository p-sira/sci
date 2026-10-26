use super::*;
use rand::{Rng, SeedableRng};

pub(crate) fn validate_array(a: &SparseArray) -> Result<()> {
    if a.shape.is_empty()
        || a.shape.len() > 32
        || a.shape.iter().any(|&d| d > LIMIT)
        || a.data.len() > LIMIT
        || a.coordinates.len() != a.data.len()
        || a.coordinates
            .iter()
            .any(|c| c.len() != a.shape.len() || c.iter().zip(&a.shape).any(|(&i, &n)| i >= n))
        || a.coordinates.windows(2).any(|w| w[0] >= w[1])
    {
        return Err(Error::InvalidInput(
            "invalid canonical sparse array or rank/storage limit",
        ));
    }
    sci_core::is_finite_slice(&a.data)
}

/// Convert a CSR matrix to a rank-two coordinate array, `B[i,j] = A[i,j]`.
///
/// # Parameters and conventions
/// `a` is canonical CSR; explicit zeros are retained.
/// # Domain and special values
/// Empty axes are supported; all stored values must be finite.
/// # Returns
/// An owned [`SparseArray`] of shape `[a.rows, a.cols]`.
/// # Errors
/// Invalid CSR or crate storage limits.
/// # Numerical behavior
/// Exact copying in O(rows + stored entries) work.
/// # Examples
/// ```
/// let a = sci_sparse::as_array(&sci_sparse::identity(2)?)?;
/// assert_eq!(a.shape, [2, 2]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy coo_array](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.html).
pub fn as_array(a: &SparseMatrix) -> Result<SparseArray> {
    validate(a)?;
    Ok(SparseArray {
        shape: vec![a.rows, a.cols],
        coordinates: entries(a).map(|(r, c, _)| vec![r, c]).collect(),
        data: a.data.clone(),
    })
}

/// Insert a length-one axis: `B[...,0,...] = A[...]`.
///
/// # Parameters and conventions
/// `axis` indexes the resulting shape; negative axes count from its end.
/// # Domain and special values
/// Canonical finite COO arrays of rank 1..31, including empty axes.
/// # Returns
/// A rank-one-higher array, retaining explicit zeros and coordinate order.
/// # Errors
/// Invalid storage, axis, or resulting rank above 32.
/// # Numerical behavior
/// Exact index manipulation; at most 32 million coordinate components.
/// # Examples
/// ```
/// let a = sci_sparse::as_array(&sci_sparse::identity(2)?)?;
/// assert_eq!(sci_sparse::expand_dims(&a, -1)?.shape, [2,2,1]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy expand_dims](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.expand_dims.html).
pub fn expand_dims(a: &SparseArray, axis: isize) -> Result<SparseArray> {
    validate_array(a)?;
    if a.shape.len() == 32 {
        return Err(Error::InvalidInput("sparse rank exceeds 32"));
    }
    let axis = axis_index(axis, a.shape.len() + 1)?;
    let mut b = a.clone();
    b.shape.insert(axis, 1);
    for c in &mut b.coordinates {
        c.insert(axis, 0);
    }
    Ok(b)
}
fn axis_index(axis: isize, n: usize) -> Result<usize> {
    let i = if axis < 0 { axis + n as isize } else { axis };
    if i < 0 || i >= n as isize {
        Err(Error::InvalidInput("axis out of bounds"))
    } else {
        Ok(i as usize)
    }
}

/// Permute axes: `B[c[axes[0]], …] = A[c[0], …]`.
///
/// # Parameters and conventions
/// `axes` is a permutation of input axes; negative axes count from the end.
/// # Domain and special values
/// Finite canonical arrays of rank 1..32; empty axes are supported.
/// # Returns
/// An owned canonical array with permuted shape and coordinates.
/// # Errors
/// Invalid storage, repeated axes, or incorrect permutation length.
/// # Numerical behavior
/// Exact reindexing followed by O(nnz log nnz) stable sorting.
/// # Examples
/// ```
/// let a = sci_sparse::as_array(&sci_sparse::identity(2)?)?;
/// assert_eq!(sci_sparse::permute_dims(&a, &[1,0])?, a);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy permute_dims](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.permute_dims.html).
pub fn permute_dims(a: &SparseArray, axes: &[isize]) -> Result<SparseArray> {
    validate_array(a)?;
    if axes.len() != a.shape.len() {
        return Err(Error::DimensionMismatch);
    }
    let axes: Vec<_> = axes
        .iter()
        .map(|&i| axis_index(i, a.shape.len()))
        .collect::<Result<_>>()?;
    let mut sorted = axes.clone();
    sorted.sort_unstable();
    sorted.dedup();
    if sorted.len() != axes.len() {
        return Err(Error::InvalidInput("axes must form a permutation"));
    }
    let mut pairs: Vec<_> = a
        .coordinates
        .iter()
        .zip(&a.data)
        .map(|(c, &v)| (axes.iter().map(|&i| c[i]).collect::<Vec<_>>(), v))
        .collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    let (coordinates, data) = pairs.into_iter().unzip();
    Ok(SparseArray {
        shape: axes.iter().map(|&i| a.shape[i]).collect(),
        coordinates,
        data,
    })
}

/// Interchange axes `axis1` and `axis2`, preserving all element values.
///
/// # Parameters and conventions
/// Signed axes follow [`permute_dims`]; equal axes leave the array unchanged.
/// # Domain and special values
/// Finite canonical arrays of rank 1..32, including empty axes.
/// # Returns
/// An owned canonical array with the two axes exchanged.
/// # Errors
/// Invalid storage or out-of-range axes.
/// # Numerical behavior
/// Uses [`permute_dims`] with its work and ordering conventions.
/// # Examples
/// ```
/// let a = sci_sparse::as_array(&sci_sparse::identity(2)?)?;
/// assert_eq!(sci_sparse::swapaxes(&a, 0, 1)?, a);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy swapaxes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.swapaxes.html).
pub fn swapaxes(a: &SparseArray, axis1: isize, axis2: isize) -> Result<SparseArray> {
    validate_array(a)?;
    let i = axis_index(axis1, a.shape.len())?;
    let j = axis_index(axis2, a.shape.len())?;
    let mut axes: Vec<_> = (0..a.shape.len() as isize).collect();
    axes.swap(i, j);
    permute_dims(a, &axes)
}

/// CSR row pointers and column indices in a concrete integer representation.
///
/// `indptr` has `rows + 1` elements; `indices` has one entry per stored value.
/// Both retain the matrix's zero-based CSR ordering, including explicit zeros.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CsrIndices<T> {
    /// Offsets into the column-index and value arrays for each row.
    pub indptr: Vec<T>,
    /// Zero-based column indices in CSR order.
    pub indices: Vec<T>,
}

fn csr_indices<T: TryFrom<usize>>(a: &SparseMatrix) -> Result<CsrIndices<T>> {
    validate(a)?;
    let convert =
        |i| T::try_from(i).map_err(|_| Error::InvalidInput("CSR indices exceed integer range"));
    // Dimensions must also fit, even when the matrix has no stored entries.
    convert(a.rows)?;
    convert(a.cols)?;
    Ok(CsrIndices {
        indptr: a
            .indptr
            .iter()
            .copied()
            .map(convert)
            .collect::<Result<_>>()?,
        indices: a
            .indices
            .iter()
            .copied()
            .map(convert)
            .collect::<Result<_>>()?,
    })
}

/// Copies CSR row pointers and column indices as signed 32-bit integers.
///
/// # Parameters and conventions
/// `a` is a shared CSR matrix. Indices are zero-based; no data values are copied.
/// # Domain and special values
/// Canonical finite CSR under the crate's storage limits, including empty shapes.
/// # Returns
/// [`CsrIndices<i32>`] with named `indptr` and `indices` fields in original order.
/// # Errors
/// Invalid CSR or dimensions, pointers, or indices outside `0..=i32::MAX`.
/// # Numerical behavior
/// Exact checked integer conversion; O(rows + stored entries) time and storage.
/// # Examples
/// ```
/// let indices = sci_sparse::csr_indices_i32(&sci_sparse::identity(2)?)?;
/// assert_eq!(indices.indptr, vec![0_i32, 1, 2]);
/// assert_eq!(indices.indices, vec![0_i32, 1]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy CSR storage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html).
pub fn csr_indices_i32(a: &SparseMatrix) -> Result<CsrIndices<i32>> {
    csr_indices(a)
}

/// Copies CSR row pointers and column indices as signed 64-bit integers.
///
/// # Parameters and conventions
/// `a` is a shared CSR matrix. Indices are zero-based; no data values are copied.
/// # Domain and special values
/// Canonical finite CSR under the crate's storage limits, including empty shapes.
/// # Returns
/// [`CsrIndices<i64>`] with named `indptr` and `indices` fields in original order.
/// # Errors
/// Invalid CSR or dimensions, pointers, or indices outside `0..=i64::MAX`.
/// # Numerical behavior
/// Exact checked integer conversion; O(rows + stored entries) time and storage.
/// # Examples
/// ```
/// let indices = sci_sparse::csr_indices_i64(&sci_sparse::identity(1)?)?;
/// assert_eq!(indices.indptr, vec![0_i64, 1]);
/// assert_eq!(indices.indices, vec![0_i64]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy CSR storage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html).
pub fn csr_indices_i64(a: &SparseMatrix) -> Result<CsrIndices<i64>> {
    csr_indices(a)
}

/// Generate uniformly distributed sparse entries on distinct random coordinates.
///
/// # Parameters and conventions
/// `shape` sets rank/axes; `density` is the fraction of stored entries in `[0,1]`.
/// The stored count is ties-to-even rounding of `density * product(shape)`.
/// `seed` initializes private StdRng; values are uniform in `[0,1)`.
/// # Domain and special values
/// Rank 1..32, each axis <=1,000,000, volume <=20,000,000 and nnz <=1,000,000.
/// Density must be finite; zero volume yields an empty array.
/// # Returns
/// A canonical [`SparseArray`] with sampled coordinates, without replacement.
/// # Errors
/// Invalid shape/density, overflow, or work/storage limits.
/// # Numerical behavior
/// O(volume) index sampling, O(nnz log nnz) sorting. RNG streams differ from
/// NumPy; a fixed seed reproduces this implementation, not SciPy's samples.
/// # Examples
/// ```
/// assert_eq!(sci_sparse::random_array(&[4,5], 0.5, 42)?.data.len(), 10);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy random_array](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.random_array.html).
pub fn random_array(shape: &[usize], density: f64, seed: u64) -> Result<SparseArray> {
    let mut a = SparseArray {
        shape: shape.to_vec(),
        coordinates: vec![],
        data: vec![],
    };
    validate_array(&a)?;
    if !density.is_finite() || !(0.0..=1.0).contains(&density) {
        return Err(Error::InvalidInput("density must be in [0,1]"));
    }
    let volume = if shape.contains(&0) {
        0
    } else {
        shape
            .iter()
            .try_fold(1usize, |v, &d| v.checked_mul(d))
            .ok_or(Error::InvalidInput("array volume overflow"))?
    };
    if volume > WORK {
        return Err(Error::InvalidInput(
            "random sampling volume exceeds 20,000,000",
        ));
    }
    let count = (density * volume as f64).round_ties_even() as usize;
    if count > LIMIT {
        return Err(Error::InvalidInput("random storage exceeds 1,000,000"));
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut positions = rand::seq::index::sample(&mut rng, volume, count).into_vec();
    positions.sort_unstable();
    for mut p in positions {
        let mut c = vec![0; shape.len()];
        for j in (0..shape.len()).rev() {
            c[j] = p % shape[j];
            p /= shape[j];
        }
        a.coordinates.push(c);
        a.data.push(rng.random());
    }
    Ok(a)
}

/// Generate a two-dimensional sparse sample with uniform `[0,1)` values.
///
/// # Parameters and conventions
/// rows, cols set shape; density and seed follow [`random_array`]. The shared
/// CSR type replaces SciPy's format/dtype options and legacy matrix record.
/// # Domain and special values
/// Finite density in `[0,1]`, crate axis/storage limits and volume <=20,000,000.
/// # Returns
/// Canonical CSR with ties-to-even rounded density×volume stored entries.
/// # Errors
/// Invalid density, shape/volume overflow, or sampling/storage limits.
/// # Numerical behavior
/// Uses [`random_array`]; streams differ from NumPy. No custom sampler argument;
/// callers may replace sampled data with validated finite values explicitly.
/// # Examples
/// ```
/// assert_eq!(sci_sparse::random(2,3,1.,42)?.data.len(),6);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy random](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.random.html).
pub fn random(rows: usize, cols: usize, density: f64, seed: u64) -> Result<SparseMatrix> {
    let a = random_array(&[rows, cols], density, seed)?;
    from_coo(
        rows,
        cols,
        &a.coordinates
            .iter()
            .zip(&a.data)
            .map(|(c, &v)| (c[0], c[1], v))
            .collect::<Vec<_>>(),
    )
}

/// Build a sparse matrix from padded diagonal rows, `A[j-k,j]=data[d][j]`.
///
/// # Parameters and conventions
/// `offsets[d]` is the column-minus-row offset k. All data rows have equal width;
/// unlike [`diags`], entries are aligned by output column, including padding.
/// Out-of-bounds row/column positions are discarded; offsets must be unique.
/// # Domain and special values
/// Finite data, including explicit zeros; nonnegative dimensions under limits.
/// Arbitrarily large offsets produce no entries without signed overflow.
/// # Returns
/// Canonical CSR of explicit shape rows×cols.
/// # Errors
/// Invalid lengths, repeated offsets, nonfinite data, or crate storage/work limits.
/// # Numerical behavior
/// Exact placement with O(input data size) work, capped at 1,000,000 input values.
/// # Examples
/// ```
/// let a=sci_sparse::spdiags(&[vec![0.,2.,3.]],&[1],3,3)?;
/// assert_eq!(sci_sparse::to_dense(&a)?[0][1],2.);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy spdiags](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.spdiags.html).
pub fn spdiags(
    data: &[Vec<f64>],
    offsets: &[isize],
    rows: usize,
    cols: usize,
) -> Result<SparseMatrix> {
    shape(rows, cols)?;
    if data.len() > LIMIT {
        return Err(Error::InvalidInput("too many diagonal rows"));
    }
    if data.len() != offsets.len() {
        return Err(Error::DimensionMismatch);
    }
    let width = data.first().map_or(0, Vec::len);
    if data.len().checked_mul(width).is_none_or(|n| n > LIMIT) {
        return Err(Error::InvalidInput("diagonal data exceeds storage limit"));
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut t = vec![];
    for (row, &k) in data.iter().zip(offsets) {
        if row.len() != width {
            return Err(Error::DimensionMismatch);
        }
        sci_core::is_finite_slice(row)?;
        if !seen.insert(k) {
            return Err(Error::InvalidInput("repeated diagonal offset"));
        }
        for (c, &v) in row.iter().enumerate().take(cols) {
            if let Some(r) = c.checked_add_signed(k.saturating_neg())
                && r < rows
            {
                t.push((r, c, v));
            }
        }
    }
    from_coo(rows, cols, &t)
}
