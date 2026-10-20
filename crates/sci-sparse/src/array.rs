use super::*;
use rand::{Rng, SeedableRng};
use std::any::Any;

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
    sci_core::finite_slice(&a.data)
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

/// Test whether a Rust value has shared sparse array or CSR matrix storage.
///
/// This is a type predicate, independent of storage validity. Unlike Python,
/// other Rust types cannot dynamically subclass these shared records.
/// # Examples
/// ```
/// assert!(sci_sparse::issparse(&sci_sparse::identity(1)?));
/// assert!(!sci_sparse::issparse(&vec![1.0]));
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy issparse](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.issparse.html).
pub fn issparse(value: &dyn Any) -> bool {
    value.is::<SparseMatrix>() || value.is::<SparseArray>()
}

/// Supported signed storage widths for sparse indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexDtype {
    /// Signed 32-bit integers.
    I32,
    /// Signed 64-bit integers.
    I64,
}

/// Choose the smallest signed index width containing all indices and `maxval`.
///
/// # Parameters and conventions
/// `arrays` contains nonnegative integer indices; `maxval` is an optional bound.
/// Rust inputs have concrete integer storage, so values are always inspected.
/// # Domain and special values
/// Values through `i64::MAX`; arrays total at most 1,000,000 entries.
/// # Returns
/// [`IndexDtype::I32`] if every value fits, otherwise [`IndexDtype::I64`].
/// # Errors
/// Values above the signed 64-bit maximum or excessive input storage.
/// # Numerical behavior
/// Exact integer comparisons in linear time.
/// # Examples
/// ```
/// assert_eq!(sci_sparse::get_index_dtype(&[&[1,2]], None)?, sci_sparse::IndexDtype::I32);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy get_index_dtype](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.get_index_dtype.html).
pub fn get_index_dtype(arrays: &[&[u64]], maxval: Option<u64>) -> Result<IndexDtype> {
    if arrays.len() > LIMIT {
        return Err(Error::InvalidInput("too many index arrays"));
    }
    let mut count = 0usize;
    let mut max = maxval.unwrap_or(0);
    for a in arrays {
        count = count
            .checked_add(a.len())
            .ok_or(Error::InvalidInput("index count overflow"))?;
        if count > LIMIT {
            return Err(Error::InvalidInput("too many indices"));
        }
        max = max.max(a.iter().copied().max().unwrap_or(0));
    }
    if max > i64::MAX as u64 {
        return Err(Error::InvalidInput("indices exceed signed 64-bit range"));
    }
    Ok(if max <= i32::MAX as u64 {
        IndexDtype::I32
    } else {
        IndexDtype::I64
    })
}

/// Validate and copy CSR index arrays into a requested signed storage width.
///
/// # Parameters and conventions
/// `dtype` controls representability. Returned pointers and indices use `i64`;
/// an I32 request additionally guarantees lossless narrowing by the caller.
/// # Domain and special values
/// Canonical finite CSR with normal crate storage/axis limits.
/// # Returns
/// `(indptr, indices)` in original CSR order, with no matrix mutation.
/// # Errors
/// Invalid CSR, or shape/pointer/index values too large for `dtype`.
/// # Numerical behavior
/// Exact integer conversion, O(rows + nnz) work.
/// # Examples
/// ```
/// let (p,i) = sci_sparse::safely_cast_index_arrays(&sci_sparse::identity(1)?, sci_sparse::IndexDtype::I32)?;
/// assert_eq!((p,i), (vec![0,1],vec![0]));
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy safely_cast_index_arrays](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.safely_cast_index_arrays.html).
pub fn safely_cast_index_arrays(
    a: &SparseMatrix,
    dtype: IndexDtype,
) -> Result<(Vec<i64>, Vec<i64>)> {
    validate(a)?;
    let max = if dtype == IndexDtype::I32 {
        i32::MAX as usize
    } else {
        i64::MAX as usize
    };
    if a.rows > max || a.cols > max || a.data.len() > max {
        return Err(Error::InvalidInput("indices do not fit dtype"));
    }
    Ok((
        a.indptr.iter().map(|&i| i as i64).collect(),
        a.indices.iter().map(|&i| i as i64).collect(),
    ))
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
        sci_core::finite_slice(row)?;
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
