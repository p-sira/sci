//! Sparse matrices with real `f64` values and shared [`SparseMatrix`] storage.
//!
//! Construction and arithmetic use canonical CSR: sorted, unique columns per
//! row. Duplicate coordinates are summed; explicit zeros are retained unless
//! [`eliminate_zeros`] is requested. Numerical inputs reject NaN and infinity, invalid
//! storage, axes above 1,000,000, and storage above 1,000,000 entries. Arithmetic
//! rejects nonfinite results. Dense conversion is limited to 1,000,000 cells;
//! sparse multiplication to 20,000,000 scalar products and block assembly to
//! 20,000,000 total input rows. Empty axes are supported.
//!
//! Shared [`SparseArray`] COO storage supports rank 1..32 and axis operations.
//! Graph algorithms, iterative solves, spectral decompositions, matrix functions,
//! random sampling, and NPZ interchange cover the modern SciPy function inventory
//! through documented Rust adaptations. Spectral and iterative axes are bounded
//! at 256; spectral routines use dense fallbacks. Python mutable storage classes,
//! deprecated compatibility namespaces and legacy solver controls are excluded.
//! Private faer, petgraph, lapjv, RNG and ZIP backends do not appear in the API.
//! See the repository's sparse coverage report for signature and work limits.
//! NPZ reading and writing require the default-enabled `npz` feature
//! (`sparse-npz` through the Sci facade). Disable default features to omit ZIP
//! and compression dependencies while retaining sparse numerical operations.
//!
//! ```
//! use sci_sparse::{from_coo, matvec, linalg::spsolve};
//! let a = from_coo(2, 2, &[(0, 0, 2.0), (1, 1, 4.0)])?;
//! assert_eq!(matvec(&a, &[3.0, 2.0])?, vec![6.0, 8.0]);
//! assert_eq!(spsolve(&a, &[6.0, 8.0])?, vec![3.0, 2.0]);
//! # Ok::<(), sci_sparse::Error>(())
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs, rustdoc::broken_intra_doc_links)]

pub use sci_core::{Error, Matrix, Result, SparseArray, SparseMatrix};
use std::collections::BTreeMap;

mod array;
/// Algorithms for weighted graphs represented by CSR matrices.
pub mod csgraph;
#[cfg(feature = "npz")]
mod io;
/// Sparse linear algebra using finite real matrices.
pub mod linalg;
pub use array::*;
#[cfg(feature = "npz")]
pub use io::{load_npz, save_npz};

const LIMIT: usize = 1_000_000;
const WORK: usize = 20_000_000;

fn shape(rows: usize, cols: usize) -> Result<()> {
    if rows > LIMIT || cols > LIMIT {
        return Err(Error::InvalidInput("sparse axis exceeds 1,000,000"));
    }
    Ok(())
}

#[inline]
fn finite(value: f64) -> Result<f64> {
    sci_core::finite(value)
}

/// Validate CSR invariants without changing the represented matrix `A`.
///
/// # Parameters and conventions
/// `a` uses zero-based CSR storage as defined by [`SparseMatrix`].
/// # Domain and special values
/// Empty axes and explicit zeros are valid; NaN and infinity are rejected.
/// # Returns
/// `()` when storage is canonical and within the crate limits.
/// # Errors
/// Invalid pointers, unsorted/duplicate/out-of-range columns, nonfinite data,
/// or an axis or stored-entry count above 1,000,000.
/// # Numerical behavior
/// Exact structural checks in O(rows + stored entries) work.
/// # Examples
/// ```
/// sci_sparse::validate(&sci_sparse::identity(3)?)?;
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy CSR storage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html).
pub fn validate(a: &SparseMatrix) -> Result<()> {
    shape(a.rows, a.cols)?;
    if a.data.len() > LIMIT
        || a.data.len() != a.indices.len()
        || a.indptr.len() != a.rows + 1
        || a.indptr.first() != Some(&0)
        || a.indptr.last() != Some(&a.data.len())
    {
        return Err(Error::InvalidInput(
            "invalid CSR storage lengths or pointers",
        ));
    }
    for range in a.indptr.windows(2) {
        if range[0] > range[1] || range[1] > a.data.len() {
            return Err(Error::InvalidInput("invalid CSR row pointers"));
        }
        let columns = &a.indices[range[0]..range[1]];
        if columns.iter().any(|&c| c >= a.cols) || columns.windows(2).any(|w| w[0] >= w[1]) {
            return Err(Error::InvalidInput(
                "CSR columns must be in bounds, sorted, and unique",
            ));
        }
    }
    sci_core::finite_slice(&a.data)
}

fn entries(a: &SparseMatrix) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
    (0..a.rows).flat_map(move |r| {
        (a.indptr[r]..a.indptr[r + 1]).map(move |p| (r, a.indices[p], a.data[p]))
    })
}

/// Construct `A[i,j] = Σ value` over coordinate triples with the same `(i,j)`.
///
/// # Parameters and conventions
/// `rows`, `cols` give the explicit shape. `coordinates` are zero-based
/// `(row, column, value)` triples in arbitrary order.
/// # Domain and special values
/// Empty axes and explicit zeros are supported; all values must be finite.
/// # Returns
/// Canonical CSR, retaining zeros produced by duplicate cancellation.
/// # Errors
/// Out-of-range coordinates, nonfinite input/sums, or crate storage/axis limits.
/// # Numerical behavior
/// Stable coordinate sorting, then sequential f64 addition in input order for
/// duplicates. O(n log n + rows) work; at most 1,000,000 input triples.
/// # Examples
/// ```
/// let a = sci_sparse::from_coo(1, 1, &[(0, 0, 2.0), (0, 0, 3.0)])?;
/// assert_eq!(a.data, vec![5.0]);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy COO](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.html).
pub fn from_coo(
    rows: usize,
    cols: usize,
    coordinates: &[(usize, usize, f64)],
) -> Result<SparseMatrix> {
    shape(rows, cols)?;
    if coordinates.len() > LIMIT {
        return Err(Error::InvalidInput("sparse storage exceeds 1,000,000"));
    }
    for &(r, c, v) in coordinates {
        if r >= rows || c >= cols || !v.is_finite() {
            return Err(Error::InvalidInput(
                "invalid sparse coordinate or nonfinite value",
            ));
        }
    }
    let mut coordinates = coordinates.to_vec();
    coordinates.sort_by_key(|&(r, c, _)| (r, c));
    let mut a = SparseMatrix {
        rows,
        cols,
        indptr: vec![0; rows + 1],
        indices: Vec::new(),
        data: Vec::new(),
    };
    let mut previous = None;
    for (r, c, v) in coordinates {
        if previous == Some((r, c)) {
            let last = a.data.last_mut().expect("previous entry exists");
            *last = finite(*last + v)?;
        } else {
            a.indices.push(c);
            a.data.push(v);
            a.indptr[r + 1] += 1;
            previous = Some((r, c));
        }
    }
    for i in 0..rows {
        a.indptr[i + 1] += a.indptr[i];
    }
    Ok(a)
}

/// Construct `A` from CSR storage, summing duplicate columns within each row.
///
/// # Parameters and conventions
/// Shape and zero-based row pointers/column indices follow [`SparseMatrix`];
/// input columns may be unsorted or repeated.
/// # Domain and special values
/// Finite values only; empty axes and explicit zeros are accepted.
/// # Returns
/// Sorted canonical CSR with duplicate entries summed.
/// # Errors
/// Invalid pointers, lengths, indices, nonfinite data/sums, or crate limits.
/// # Numerical behavior
/// Uses [`from_coo`], including its summation order and work limits.
/// # Examples
/// ```
/// let a = sci_sparse::from_csr(1, 2, &[0, 2], &[1, 0], &[3.0, 2.0])?;
/// assert_eq!(a.indices, vec![0, 1]);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy CSR](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html).
pub fn from_csr(
    rows: usize,
    cols: usize,
    indptr: &[usize],
    indices: &[usize],
    data: &[f64],
) -> Result<SparseMatrix> {
    shape(rows, cols)?;
    if data.len() > LIMIT
        || indices.len() != data.len()
        || indptr.len() != rows + 1
        || indptr.first() != Some(&0)
        || indptr.last() != Some(&data.len())
        || indptr.windows(2).any(|w| w[0] > w[1] || w[1] > data.len())
    {
        return Err(Error::InvalidInput("invalid CSR pointers or lengths"));
    }
    let coordinates: Vec<_> = (0..rows)
        .flat_map(|r| (indptr[r]..indptr[r + 1]).map(move |p| (r, indices[p], data[p])))
        .collect();
    from_coo(rows, cols, &coordinates)
}

/// Construct `A` from CSC columns: `A[indices[p],j] = data[p]`.
///
/// # Parameters and conventions
/// `indptr` has `cols + 1` entries; `indices` are zero-based row indices.
/// Input rows may be unsorted or repeated.
/// # Domain and special values
/// Finite values, empty axes, and explicit zeros are supported.
/// # Returns
/// Canonical CSR of shape `(rows, cols)`.
/// # Errors
/// Invalid CSC storage, nonfinite sums, or crate limits, as in [`from_csr`].
/// # Numerical behavior
/// Canonicalizes columns, then transposes; O(n log n + rows + cols).
/// # Examples
/// ```
/// assert_eq!(sci_sparse::from_csc(1, 2, &[0, 1, 1], &[0], &[2.0])?.data, vec![2.0]);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy CSC](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.html).
pub fn from_csc(
    rows: usize,
    cols: usize,
    indptr: &[usize],
    indices: &[usize],
    data: &[f64],
) -> Result<SparseMatrix> {
    transpose(&from_csr(cols, rows, indptr, indices, data)?)
}
/// Construct `A[i,j] = values[i][j]`, omitting exact zeros.
///
/// # Parameters and conventions
/// `values` contains equal-length rows, using shared Sci matrix ordering.
/// # Domain and special values
/// Finite real values only; empty axes and explicit zeros are supported.
/// # Returns
/// Canonical CSR; an empty input has shape `(0,0)`.
/// # Errors
/// Ragged rows, nonfinite data, or more than 1,000,000 dense cells.
/// # Numerical behavior
/// Exact copying without rounding. Both signs of zero become implicit positive zero.
/// # Examples
/// ```
/// let a = sci_sparse::from_dense(&[vec![0.0, 3.0]])?;
/// assert_eq!(a.indices, vec![1]);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy csr_array](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html).
pub fn from_dense(values: &[Vec<f64>]) -> Result<SparseMatrix> {
    let rows = values.len();
    let cols = values.first().map_or(0, Vec::len);
    dense_size(rows, cols)?;
    let mut coordinates = Vec::new();
    for (r, row) in values.iter().enumerate() {
        if row.len() != cols {
            return Err(Error::DimensionMismatch);
        }
        for (c, &v) in row.iter().enumerate() {
            if !v.is_finite() {
                return Err(Error::InvalidInput("dense entries must be finite"));
            }
            if v != 0.0 {
                coordinates.push((r, c, v));
            }
        }
    }
    from_coo(rows, cols, &coordinates)
}

fn dense_size(rows: usize, cols: usize) -> Result<()> {
    shape(rows, cols)?;
    if rows.checked_mul(cols).is_none_or(|n| n > LIMIT) {
        return Err(Error::InvalidInput("dense output exceeds 1,000,000 cells"));
    }
    Ok(())
}

/// Return the full matrix, with missing `A[i,j] = 0`.
///
/// # Parameters and conventions
/// `a` is canonical CSR. Rows precede columns in the returned shared Sci matrix.
/// # Domain and special values
/// Finite real values only; empty axes and explicit zeros are supported.
/// # Returns
/// Owned row vectors with shape `(a.rows,a.cols)`.
/// # Errors
/// Invalid CSR, more than 1,000,000 dense cells, or a `(0,n)` shape with `n > 0`
/// (which shared row vectors cannot represent).
/// # Numerical behavior
/// Exact copying; missing entries are positive zero. O(rows × cols + stored entries).
/// # Examples
/// ```
/// let a = sci_sparse::identity(2)?;
/// assert_eq!(sci_sparse::to_dense(&a)?[0], vec![1.0, 0.0]);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy csr_array.toarray](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.toarray.html).
pub fn to_dense(a: &SparseMatrix) -> Result<Matrix> {
    validate(a)?;
    dense_size(a.rows, a.cols)?;
    if a.rows == 0 && a.cols != 0 {
        return Err(Error::InvalidInput(
            "shared rows cannot represent shape (0,n)",
        ));
    }
    let mut out = vec![vec![0.0; a.cols]; a.rows];
    for (r, c, v) in entries(a) {
        out[r][c] = v;
    }
    Ok(out)
}

/// Construct `A[i,j] = 1` when `j - i = offset`, and zero otherwise.
///
/// # Parameters and conventions
/// `rows`, `cols` are explicit dimensions; positive `offset` selects a superdiagonal.
/// # Domain and special values
/// Finite real values only; empty axes and explicit zeros are supported.
/// # Returns
/// A rectangular CSR identity diagonal, possibly empty.
/// # Errors
/// Axes above the crate limits.
/// # Numerical behavior
/// Exact values; O(rows + diagonal length) storage and work. Out-of-range offsets yield zero matrices.
/// # Examples
/// ```
/// assert_eq!(sci_sparse::eye(2, 3, 1)?.indices, vec![1, 2]);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy eye_array](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.eye_array.html).
pub fn eye(rows: usize, cols: usize, offset: isize) -> Result<SparseMatrix> {
    shape(rows, cols)?;
    let (r, c, n) = diagonal_span(rows, cols, offset);
    let coordinates: Vec<_> = (0..n).map(|i| (r + i, c + i, 1.0)).collect();
    from_coo(rows, cols, &coordinates)
}

fn diagonal_span(rows: usize, cols: usize, offset: isize) -> (usize, usize, usize) {
    let (r, c) = if offset < 0 {
        (offset.unsigned_abs(), 0)
    } else {
        (0, offset as usize)
    };
    (r, c, rows.saturating_sub(r).min(cols.saturating_sub(c)))
}

/// Construct the square identity `I[i,j] = δ(i,j)`.
///
/// # Parameters and conventions
/// `n` is the number of rows and columns.
/// # Domain and special values
/// Finite real values only; empty axes and explicit zeros are supported.
/// # Returns
/// Canonical CSR with `n` stored ones.
/// # Errors
/// `n > 1,000,000`.
/// # Numerical behavior
/// Exact; equivalent to `eye(n,n,0)`.
/// # Examples
/// ```
/// assert_eq!(sci_sparse::identity(2)?.data, vec![1.0, 1.0]);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy identity](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.identity.html).
pub fn identity(n: usize) -> Result<SparseMatrix> {
    eye(n, n, 0)
}

/// Construct `A[i,i+offsets[k]] = diagonals[k][t]` along each diagonal.
///
/// # Parameters and conventions
/// `rows`, `cols` specify shape; offsets are signed column minus row.
/// Each diagonal is unpadded and must have exactly its in-bounds length.
/// Unlike SciPy, shape is required and scalar broadcasting/truncation is unsupported.
/// # Domain and special values
/// Finite real values only; empty axes and explicit zeros are supported.
/// # Returns
/// Canonical CSR with each diagonal in increasing row order.
/// # Errors
/// Repeated offsets, a length mismatch, nonfinite values, or crate limits.
/// # Numerical behavior
/// Exact copying; sorting uses O(n log n) work for n diagonal entries.
/// Out-of-range offsets are accepted only with an empty diagonal.
/// # Examples
/// ```
/// let a = sci_sparse::diags(2, 3, &[vec![2.0, 3.0]], &[1])?;
/// assert_eq!(a.indices, vec![1, 2]);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy diags_array](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.diags_array.html).
pub fn diags(
    rows: usize,
    cols: usize,
    diagonals: &[Vec<f64>],
    offsets: &[isize],
) -> Result<SparseMatrix> {
    shape(rows, cols)?;
    if diagonals.len() != offsets.len() {
        return Err(Error::DimensionMismatch);
    }
    if offsets.len() > LIMIT {
        return Err(Error::InvalidInput("too many diagonals"));
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut coordinates = Vec::new();
    for (values, &offset) in diagonals.iter().zip(offsets) {
        if !seen.insert(offset) {
            return Err(Error::InvalidInput("repeated diagonal offset"));
        }
        let (r, c, n) = diagonal_span(rows, cols, offset);
        if values.len() != n {
            return Err(Error::DimensionMismatch);
        }
        if coordinates.len() + n > LIMIT {
            return Err(Error::InvalidInput("sparse storage exceeds 1,000,000"));
        }
        coordinates.extend(values.iter().enumerate().map(|(i, &v)| (r + i, c + i, v)));
    }
    from_coo(rows, cols, &coordinates)
}

/// Return `B[j,i] = A[i,j]`.
///
/// # Parameters and conventions
/// `a` is canonical CSR; axes are exchanged.
/// # Domain and special values
/// Finite real values only; empty axes and explicit zeros are supported.
/// # Returns
/// Canonical CSR of shape `(a.cols,a.rows)`.
/// # Errors
/// Invalid CSR or crate limits.
/// # Numerical behavior
/// Exact permutation, preserving explicit zeros and their signs. O(rows + cols + stored entries).
/// # Examples
/// ```
/// let a = sci_sparse::eye(2, 3, 1)?;
/// assert_eq!(sci_sparse::transpose(&a)?.rows, 3);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy matrix_transpose](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.matrix_transpose.html).
pub fn transpose(a: &SparseMatrix) -> Result<SparseMatrix> {
    validate(a)?;
    let mut b = SparseMatrix {
        rows: a.cols,
        cols: a.rows,
        indptr: vec![0; a.cols + 1],
        indices: vec![0; a.data.len()],
        data: vec![0.0; a.data.len()],
    };
    for &c in &a.indices {
        b.indptr[c + 1] += 1;
    }
    for i in 0..a.cols {
        b.indptr[i + 1] += b.indptr[i];
    }
    let mut next = b.indptr.clone();
    for (r, c, v) in entries(a) {
        let p = next[c];
        b.indices[p] = r;
        b.data[p] = v;
        next[c] += 1;
    }
    Ok(b)
}

/// Extract nonzero triples `(i,j,A[i,j])` in row-major order.
///
/// # Parameters and conventions
/// `a` is canonical CSR. All output indices are zero-based.
/// # Domain and special values
/// Finite real values only; empty axes and explicit zeros are supported.
/// # Returns
/// A vector of `(row,column,value)` triples; explicit zeros are excluded.
/// SciPy returns three separate arrays instead.
/// # Errors
/// Invalid CSR or crate limits.
/// # Numerical behavior
/// Exact filtering in O(rows + stored entries) work.
/// # Examples
/// ```
/// assert_eq!(sci_sparse::find(&sci_sparse::identity(1)?)?, vec![(0, 0, 1.0)]);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy find](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.find.html).
pub fn find(a: &SparseMatrix) -> Result<Vec<(usize, usize, f64)>> {
    validate(a)?;
    Ok(entries(a).filter(|&(_, _, v)| v != 0.0).collect())
}

/// Return the same matrix `A` with stored zero entries removed.
///
/// # Parameters and conventions
/// `a` is canonical CSR. Both signs of zero are removed.
/// # Domain and special values
/// Finite real values only; empty axes and explicit zeros are supported.
/// # Returns
/// Canonical CSR without explicit zeros.
/// # Errors
/// Invalid CSR or crate limits.
/// # Numerical behavior
/// Exact filtering; O(rows + stored entries) work and storage.
/// # Examples
/// ```
/// let a = sci_sparse::from_coo(1, 1, &[(0, 0, -0.0)])?;
/// assert!(sci_sparse::eliminate_zeros(&a)?.data.is_empty());
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy csr_array.eliminate_zeros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.eliminate_zeros.html).
pub fn eliminate_zeros(a: &SparseMatrix) -> Result<SparseMatrix> {
    from_coo(a.rows, a.cols, &find(a)?)
}

/// Retain `A[i,j]` where `j - i ≤ offset`; set other entries to zero.
///
/// # Parameters and conventions
/// `a` is canonical CSR; positive `offset` selects a superdiagonal.
/// # Domain and special values
/// Finite real values only; empty axes and explicit zeros are supported.
/// # Returns
/// Canonical CSR with the selected entries, including their explicit zeros.
/// # Errors
/// Invalid CSR or crate limits.
/// # Numerical behavior
/// Exact filtering; O(n log n + rows) work for n retained entries.
/// # Examples
/// ```
/// assert_eq!(sci_sparse::tril(&sci_sparse::identity(2)?, 0)?.data, vec![1.0, 1.0]);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy tril](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.tril.html).
pub fn tril(a: &SparseMatrix, offset: isize) -> Result<SparseMatrix> {
    validate(a)?;
    let coordinates: Vec<_> = entries(a)
        .filter(|&(r, c, _)| c as i128 - r as i128 <= offset as i128)
        .collect();
    from_coo(a.rows, a.cols, &coordinates)
}

/// Retain `A[i,j]` where `j - i ≥ offset`; set other entries to zero.
///
/// # Parameters and conventions
/// `a` is canonical CSR; positive `offset` selects a superdiagonal.
/// # Domain and special values
/// Finite real values only; empty axes and explicit zeros are supported.
/// # Returns
/// Canonical CSR with the selected entries, including their explicit zeros.
/// # Errors
/// Invalid CSR or crate limits.
/// # Numerical behavior
/// Exact filtering; O(n log n + rows) work for n retained entries.
/// # Examples
/// ```
/// assert_eq!(sci_sparse::triu(&sci_sparse::identity(2)?, 0)?.data, vec![1.0, 1.0]);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy triu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.triu.html).
pub fn triu(a: &SparseMatrix, offset: isize) -> Result<SparseMatrix> {
    validate(a)?;
    let coordinates: Vec<_> = entries(a)
        .filter(|&(r, c, _)| c as i128 - r as i128 >= offset as i128)
        .collect();
    from_coo(a.rows, a.cols, &coordinates)
}

/// Return `B[i,j] = factor × A[i,j]`.
///
/// # Parameters and conventions
/// `a` is canonical CSR; `factor` is a finite real scalar.
/// # Domain and special values
/// Finite real values only; empty axes and explicit zeros are supported.
/// # Returns
/// CSR with the same stored pattern, including zeros.
/// # Errors
/// Invalid CSR, nonfinite factor, or overflow.
/// # Numerical behavior
/// One rounded multiplication per stored entry. O(rows + stored entries).
/// # Examples
/// ```
/// assert_eq!(sci_sparse::scale(&sci_sparse::identity(2)?, 3.0)?.data, vec![3.0, 3.0]);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy csr_array.multiply](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.multiply.html).
pub fn scale(a: &SparseMatrix, factor: f64) -> Result<SparseMatrix> {
    validate(a)?;
    if !factor.is_finite() {
        return Err(Error::InvalidInput("scale must be finite"));
    }
    let mut b = a.clone();
    for v in &mut b.data {
        *v = finite(*v * factor)?;
    }
    Ok(b)
}

/// Return the entrywise sum `C[i,j] = A[i,j] + B[i,j]`.
///
/// # Parameters and conventions
/// `a` and `b` are canonical CSR with identical shapes.
/// # Domain and special values
/// Finite real values only; empty axes and explicit zeros are supported.
/// # Returns
/// CSR containing the union of the patterns, including cancellation zeros.
/// # Errors
/// Invalid CSR, different shapes, nonfinite sums, or output above storage limits.
/// # Numerical behavior
/// One f64 addition per overlapping coordinate. O(rows + nnz(A) + nnz(B)) work.
/// # Examples
/// ```
/// let a = sci_sparse::identity(2)?;
/// assert_eq!(sci_sparse::add(&a, &a)?.data, vec![2.0, 2.0]);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy csr_array.__add__](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.__add__.html).
pub fn add(a: &SparseMatrix, b: &SparseMatrix) -> Result<SparseMatrix> {
    pair(a, b)?;
    let mut out = SparseMatrix {
        rows: a.rows,
        cols: a.cols,
        indptr: vec![0],
        indices: Vec::new(),
        data: Vec::new(),
    };
    for r in 0..a.rows {
        let (mut i, mut j) = (a.indptr[r], b.indptr[r]);
        while i < a.indptr[r + 1] || j < b.indptr[r + 1] {
            let ac = if i < a.indptr[r + 1] {
                a.indices[i]
            } else {
                usize::MAX
            };
            let bc = if j < b.indptr[r + 1] {
                b.indices[j]
            } else {
                usize::MAX
            };
            let (c, v) = if ac < bc {
                let v = a.data[i];
                i += 1;
                (ac, v)
            } else if bc < ac {
                let v = b.data[j];
                j += 1;
                (bc, v)
            } else {
                let v = finite(a.data[i] + b.data[j])?;
                i += 1;
                j += 1;
                (ac, v)
            };
            push(&mut out, c, v)?;
        }
        out.indptr.push(out.data.len());
    }
    Ok(out)
}

fn pair(a: &SparseMatrix, b: &SparseMatrix) -> Result<()> {
    validate(a)?;
    validate(b)?;
    if (a.rows, a.cols) != (b.rows, b.cols) {
        return Err(Error::DimensionMismatch);
    }
    Ok(())
}
fn push(a: &mut SparseMatrix, c: usize, v: f64) -> Result<()> {
    if a.data.len() >= LIMIT {
        return Err(Error::InvalidInput("sparse storage exceeds 1,000,000"));
    }
    a.indices.push(c);
    a.data.push(v);
    Ok(())
}

/// Return the entrywise product `C[i,j] = A[i,j] B[i,j]`.
///
/// # Parameters and conventions
/// `a` and `b` are canonical CSR with identical shapes; this is the Hadamard product.
/// # Domain and special values
/// Finite real values only; empty axes and explicit zeros are supported.
/// # Returns
/// CSR containing the intersection of the stored patterns, including resulting zeros.
/// # Errors
/// Invalid CSR, different shapes, or nonfinite products.
/// # Numerical behavior
/// O(rows + nnz(A) + nnz(B)) work. Each common entry uses one f64 multiplication.
/// # Examples
/// ```
/// let a = sci_sparse::identity(2)?;
/// assert_eq!(sci_sparse::multiply(&a, &a)?, a);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy csr_array.multiply](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.multiply.html).
pub fn multiply(a: &SparseMatrix, b: &SparseMatrix) -> Result<SparseMatrix> {
    pair(a, b)?;
    let mut out = SparseMatrix {
        rows: a.rows,
        cols: a.cols,
        indptr: vec![0],
        indices: Vec::new(),
        data: Vec::new(),
    };
    for r in 0..a.rows {
        let (mut i, mut j) = (a.indptr[r], b.indptr[r]);
        while i < a.indptr[r + 1] && j < b.indptr[r + 1] {
            match a.indices[i].cmp(&b.indices[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    push(&mut out, a.indices[i], finite(a.data[i] * b.data[j])?)?;
                    i += 1;
                    j += 1;
                }
            }
        }
        out.indptr.push(out.data.len());
    }
    Ok(out)
}

/// Compute `y[i] = Σ_j A[i,j] x[j]`.
///
/// # Parameters and conventions
/// `a` is canonical CSR, and `x.len() == a.cols`.
/// # Domain and special values
/// Finite real values only; empty axes and explicit zeros are supported.
/// # Returns
/// A vector of length `a.rows` in row order.
/// # Errors
/// Invalid CSR, incompatible dimensions, nonfinite vector/products/sums.
/// # Numerical behavior
/// Sequential f64 accumulation by increasing column; O(rows + cols + stored entries).
/// Cancellation can reduce relative accuracy; underflow follows IEEE f64.
/// # Examples
/// ```
/// assert_eq!(sci_sparse::matvec(&sci_sparse::identity(2)?, &[3.0, 4.0])?, vec![3.0, 4.0]);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy csr_array.dot](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.dot.html).
pub fn matvec(a: &SparseMatrix, x: &[f64]) -> Result<Vec<f64>> {
    validate(a)?;
    if x.len() != a.cols {
        return Err(Error::DimensionMismatch);
    }
    sci_core::finite_slice(x)?;
    let mut out = vec![0.0; a.rows];
    for (r, c, v) in entries(a) {
        out[r] = finite(out[r] + finite(v * x[c])?)?;
    }
    Ok(out)
}

/// Compute `C[i,j] = Σ_k A[i,k] B[k,j]`.
///
/// # Parameters and conventions
/// `a`, `b` are canonical CSR, with `a.cols == b.rows`.
/// # Domain and special values
/// Finite real values only; empty axes and explicit zeros are supported.
/// # Returns
/// CSR of shape `(a.rows,b.cols)` with exact-zero results omitted.
/// # Errors
/// Invalid CSR, incompatible dimensions, nonfinite intermediate/results,
/// more than 20,000,000 scalar products, or more than 1,000,000 output entries.
/// # Numerical behavior
/// Sparse row accumulation in increasing k, with sorted output columns.
/// Cancellation can reduce relative accuracy. O(input size + products × log(output columns + 1) + output rows).
/// # Examples
/// ```
/// let a = sci_sparse::identity(2)?;
/// assert_eq!(sci_sparse::matmul(&a, &a)?, a);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy csr_array.dot](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.dot.html).
pub fn matmul(a: &SparseMatrix, b: &SparseMatrix) -> Result<SparseMatrix> {
    validate(a)?;
    validate(b)?;
    if a.cols != b.rows {
        return Err(Error::DimensionMismatch);
    }
    let mut work = 0usize;
    for &k in &a.indices {
        work += b.indptr[k + 1] - b.indptr[k];
        if work > WORK {
            return Err(Error::InvalidInput(
                "sparse product exceeds 20,000,000 scalar products",
            ));
        }
    }
    let mut out = SparseMatrix {
        rows: a.rows,
        cols: b.cols,
        indptr: vec![0],
        indices: Vec::new(),
        data: Vec::new(),
    };
    for r in 0..a.rows {
        let mut row = BTreeMap::<usize, f64>::new();
        for p in a.indptr[r]..a.indptr[r + 1] {
            let k = a.indices[p];
            for q in b.indptr[k]..b.indptr[k + 1] {
                let v = row.entry(b.indices[q]).or_default();
                *v = finite(*v + finite(a.data[p] * b.data[q])?)?;
            }
        }
        for (c, v) in row {
            if v != 0.0 {
                push(&mut out, c, v)?;
            }
        }
        out.indptr.push(out.data.len());
    }
    Ok(out)
}

fn assemble(
    rows: usize,
    cols: usize,
    blocks: &[(&SparseMatrix, usize, usize)],
) -> Result<SparseMatrix> {
    shape(rows, cols)?;
    let mut n = 0usize;
    let mut row_work = 0usize;
    for &(a, _, _) in blocks {
        shape(a.rows, a.cols)?;
        n = n
            .checked_add(a.data.len())
            .ok_or(Error::InvalidInput("sparse storage overflow"))?;
        row_work += a.rows;
        if n > LIMIT || row_work > WORK {
            return Err(Error::InvalidInput(
                "block assembly exceeds storage or 20,000,000 input rows",
            ));
        }
    }
    for &(a, _, _) in blocks {
        validate(a)?;
    }
    let mut coordinates = Vec::with_capacity(n);
    for &(a, r0, c0) in blocks {
        coordinates.extend(entries(a).map(|(r, c, v)| (r + r0, c + c0, v)));
    }
    from_coo(rows, cols, &coordinates)
}

/// Concatenate matrices horizontally: `C = [A₀ A₁ …]`.
///
/// # Parameters and conventions
/// `blocks` is a nonempty slice of CSR matrices. Row counts must agree.
/// # Domain and special values
/// Finite real values only; empty axes and explicit zeros are supported.
/// # Returns
/// Canonical CSR preserving explicit zeros.
/// # Errors
/// Invalid CSR, empty block list, incompatible shapes, or combined crate limits.
/// # Numerical behavior
/// Exact coordinate translation; O(n log n + block count + input/output rows) work for n stored entries.
/// At most 20,000,000 total input rows are scanned across blocks.
/// # Examples
/// ```
/// let a = sci_sparse::identity(2)?;
/// assert_eq!(sci_sparse::hstack(&[&a, &a])?.cols, 4);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy hstack](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.hstack.html).
pub fn hstack(blocks: &[&SparseMatrix]) -> Result<SparseMatrix> {
    if blocks.is_empty() || blocks.len() > LIMIT {
        return Err(Error::InvalidInput("expected 1..=1,000,000 blocks"));
    }
    let (rows, mut cols) = (blocks[0].rows, 0);
    let mut placed = Vec::new();
    for &a in blocks {
        shape(a.rows, a.cols)?;
        if a.rows != rows {
            return Err(Error::DimensionMismatch);
        }
        placed.push((a, 0, cols));
        cols += a.cols;
        shape(rows, cols)?;
    }
    assemble(rows, cols, &placed)
}

/// Concatenate matrices vertically: `C = [A₀; A₁; …]`.
///
/// # Parameters and conventions
/// `blocks` is a nonempty slice of CSR matrices. Column counts must agree.
/// # Domain and special values
/// Finite real values only; empty axes and explicit zeros are supported.
/// # Returns
/// Canonical CSR preserving explicit zeros.
/// # Errors
/// Invalid CSR, empty block list, incompatible shapes, or combined crate limits.
/// # Numerical behavior
/// Exact coordinate translation; O(n log n + block count + input/output rows) work for n stored entries.
/// At most 20,000,000 total input rows are scanned across blocks.
/// # Examples
/// ```
/// let a = sci_sparse::identity(2)?;
/// assert_eq!(sci_sparse::vstack(&[&a, &a])?.rows, 4);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy vstack](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.vstack.html).
pub fn vstack(blocks: &[&SparseMatrix]) -> Result<SparseMatrix> {
    if blocks.is_empty() || blocks.len() > LIMIT {
        return Err(Error::InvalidInput("expected 1..=1,000,000 blocks"));
    }
    let (mut rows, cols) = (0, blocks[0].cols);
    let mut placed = Vec::new();
    for &a in blocks {
        shape(a.rows, a.cols)?;
        if a.cols != cols {
            return Err(Error::DimensionMismatch);
        }
        placed.push((a, rows, 0));
        rows += a.rows;
        shape(rows, cols)?;
    }
    assemble(rows, cols, &placed)
}

/// Construct `C = diag(A₀,A₁,…)` with zero off-diagonal blocks.
///
/// # Parameters and conventions
/// `blocks` is a nonempty slice of CSR matrices. Shapes may differ.
/// # Domain and special values
/// Finite real values only; empty axes and explicit zeros are supported.
/// # Returns
/// Canonical CSR preserving explicit zeros.
/// # Errors
/// Invalid CSR, empty block list, incompatible shapes, or combined crate limits.
/// # Numerical behavior
/// Exact coordinate translation; O(n log n + block count + input/output rows) work for n stored entries.
/// At most 20,000,000 total input rows are scanned across blocks.
/// # Examples
/// ```
/// let a = sci_sparse::identity(2)?;
/// assert_eq!(sci_sparse::block_diag(&[&a, &a])?.rows, 4);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy block_diag](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.block_diag.html).
pub fn block_diag(blocks: &[&SparseMatrix]) -> Result<SparseMatrix> {
    if blocks.is_empty() || blocks.len() > LIMIT {
        return Err(Error::InvalidInput("expected 1..=1,000,000 blocks"));
    }
    let (mut rows, mut cols) = (0, 0);
    let mut placed = Vec::new();
    for &a in blocks {
        shape(a.rows, a.cols)?;
        placed.push((a, rows, cols));
        rows += a.rows;
        cols += a.cols;
        shape(rows, cols)?;
    }
    assemble(rows, cols, &placed)
}

/// Construct `C` by tiling a rectangular grid of matrices.
///
/// # Parameters and conventions
/// `blocks` is a nonempty rectangular grid of CSR references. All blocks in
/// one block row must have equal height, and in one block column equal width.
/// Unlike SciPy, absent (`None`) blocks are supplied as explicitly shaped zero matrices.
/// # Domain and special values
/// Finite real values only; empty axes and explicit zeros are supported.
/// # Returns
/// Canonical CSR with explicit zeros preserved.
/// # Errors
/// Invalid CSR, empty/ragged grid, incompatible block shapes, or crate limits.
/// # Numerical behavior
/// Exact coordinate translation; O(n log n + block count + input/output rows) for n stored entries.
/// At most 1,000,000 blocks and 20,000,000 total input rows are inspected.
/// # Examples
/// ```
/// let a = sci_sparse::identity(1)?;
/// assert_eq!(sci_sparse::block_array(&[vec![&a, &a]])?.cols, 2);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy block_array](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.block_array.html).
pub fn block_array(blocks: &[Vec<&SparseMatrix>]) -> Result<SparseMatrix> {
    let nc = blocks.first().map_or(0, Vec::len);
    if blocks.is_empty() || nc == 0 || blocks.len().checked_mul(nc).is_none_or(|n| n > LIMIT) {
        return Err(Error::InvalidInput(
            "expected nonempty block grid within work limit",
        ));
    }
    for row in blocks {
        if row.len() != nc {
            return Err(Error::DimensionMismatch);
        }
    }
    let mut placed = Vec::new();
    let mut rows = 0;
    let mut cols = 0;
    for row in blocks {
        let height = row[0].rows;
        cols = 0;
        for (j, &a) in row.iter().enumerate() {
            shape(a.rows, a.cols)?;
            if a.rows != height || a.cols != blocks[0][j].cols {
                return Err(Error::DimensionMismatch);
            }
            placed.push((a, rows, cols));
            cols += a.cols;
            shape(rows, cols)?;
        }
        rows += height;
        shape(rows, cols)?;
    }
    assemble(rows, cols, &placed)
}

/// Compute the Kronecker product `C[i*p+k,j*q+l] = A[i,j] B[k,l]`.
///
/// # Parameters and conventions
/// `a` has shape `(m,n)` and `b` has shape `(p,q)`; output axes are `(m*p,n*q)`.
/// # Domain and special values
/// Finite real values only; empty axes and explicit zeros are supported.
/// # Returns
/// Canonical CSR, preserving products of explicitly stored entries, including zeros.
/// # Errors
/// Invalid CSR, nonfinite products, or axis/storage limits. Stored-entry products
/// are limited to 1,000,000, including explicit zeros.
/// # Numerical behavior
/// One f64 multiplication per stored pair; O(input size + n log n + output rows) for n stored pairs.
/// # Examples
/// ```
/// let a = sci_sparse::identity(2)?;
/// assert_eq!(sci_sparse::kron(&a, &a)?, sci_sparse::identity(4)?);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy kron](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.kron.html).
pub fn kron(a: &SparseMatrix, b: &SparseMatrix) -> Result<SparseMatrix> {
    validate(a)?;
    validate(b)?;
    let rows = a
        .rows
        .checked_mul(b.rows)
        .ok_or(Error::InvalidInput("Kronecker row count overflow"))?;
    let cols = a
        .cols
        .checked_mul(b.cols)
        .ok_or(Error::InvalidInput("Kronecker column count overflow"))?;
    shape(rows, cols)?;
    if a.data
        .len()
        .checked_mul(b.data.len())
        .is_none_or(|n| n > LIMIT)
    {
        return Err(Error::InvalidInput("Kronecker storage exceeds 1,000,000"));
    }
    let mut coordinates = Vec::new();
    let b_entries: Vec<_> = entries(b).collect();
    for (i, j, v) in entries(a) {
        for &(k, l, w) in &b_entries {
            coordinates.push((i * b.rows + k, j * b.cols + l, finite(v * w)?));
        }
    }
    from_coo(rows, cols, &coordinates)
}

/// Compute the Kronecker sum `I_n ⊗ A + B ⊗ I_m`.
///
/// # Parameters and conventions
/// `a` is square `(m,m)` and `b` is square `(n,n)`; follows SciPy argument ordering.
/// # Domain and special values
/// Finite real values only; empty axes and explicit zeros are supported.
/// # Returns
/// Square CSR of order `m*n`.
/// # Errors
/// Invalid CSR, nonsquare input, nonfinite arithmetic, or crate limits.
/// # Numerical behavior
/// Uses [`kron`] and [`add`], including their limits and f64 rounding behavior.
/// # Examples
/// ```
/// let a = sci_sparse::identity(2)?;
/// assert_eq!(sci_sparse::kronsum(&a, &a)?.data, vec![2.0; 4]);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy kronsum](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.kronsum.html).
pub fn kronsum(a: &SparseMatrix, b: &SparseMatrix) -> Result<SparseMatrix> {
    validate(a)?;
    validate(b)?;
    if a.rows != a.cols || b.rows != b.cols {
        return Err(Error::DimensionMismatch);
    }
    add(&kron(&identity(b.rows)?, a)?, &kron(b, &identity(a.rows)?)?)
}

/// SciPy matrix spelling of [`block_array`], with the same signature and behavior.
pub use block_array as bmat;
/// SciPy array spelling of [`diags`], with the same signature and behavior.
pub use diags as diags_array;
/// SciPy array spelling of [`eye`], with the same signature and behavior.
pub use eye as eye_array;
/// SciPy spelling of [`transpose`], with the same signature and behavior.
pub use transpose as matrix_transpose;
