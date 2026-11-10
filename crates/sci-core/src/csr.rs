//! Canonical compressed sparse row (CSR) operations shared by Sci numerical
//! modules and file codecs. This module has no dependency on sparse solvers.

use crate::{Error, Matrix, Result, SparseMatrix, is_finite_slice};

const LIMIT: usize = 1_000_000;

fn shape(rows: usize, cols: usize) -> Result<()> {
    if rows > LIMIT || cols > LIMIT {
        return Err(Error::InvalidInput("sparse axis exceeds 1,000,000"));
    }
    Ok(())
}

fn dense_size(rows: usize, cols: usize) -> Result<()> {
    shape(rows, cols)?;
    if rows.checked_mul(cols).is_none_or(|n| n > LIMIT) {
        return Err(Error::InvalidInput("dense output exceeds 1,000,000 cells"));
    }
    Ok(())
}

/// Validates a finite canonical real CSR matrix without changing its contents.
///
/// # Parameters and conventions
/// `a` stores sorted, unique zero-based column indices within each row; row
/// pointers start at zero and end at `data.len()`.
///
/// # Domain and special values
/// Empty axes, explicit zeros (including negative zero), and subnormals are
/// valid. Each axis and stored-entry count is at most 1,000,000. NaN and
/// infinities are rejected.
///
/// # Returns
/// `Ok(())` if the storage is canonical.
///
/// # Errors
/// [`Error::InvalidInput`] for invalid axes, pointers, lengths, column order,
/// out-of-range columns, or nonfinite values.
///
/// # Numerical behavior
/// Exact O(rows + stored entries) structural checks; no floating arithmetic.
///
/// # Examples
/// ```
/// let a = sci_core::csr::from_coo(1, 2, &[(0, 1, 3.0)])?;
/// sci_core::csr::validate(&a)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
///
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
    is_finite_slice(&a.data)
}

/// Constructs a canonical finite CSR matrix from coordinate triples.
///
/// # Parameters and conventions
/// `rows`, `cols` are the explicit shape; coordinates are zero-based
/// `(row, column, value)` triples in arbitrary order.
///
/// # Domain and special values
/// Empty axes and explicit zeros are supported. Each axis and the number of
/// input triples is at most 1,000,000. Coordinates and sums must be finite.
///
/// # Returns
/// CSR with strictly increasing columns per row. Duplicate coordinates are
/// summed and explicit zeros from cancellation are retained.
///
/// # Errors
/// [`Error::InvalidInput`] on invalid shapes, coordinates, and nonfinite
/// input; [`Error::NonFiniteEvaluation`] if duplicate accumulation overflows.
///
/// # Numerical behavior
/// Stable row bucketing and per-row column sorting, followed by sequential
/// `f64` addition in original order for duplicates. O(entries + rows +
/// Σ row_entries × log(row_entries)) work. This preserves summation order and
/// signed zeros of the original Sci sparse constructor.
///
/// # Examples
/// ```
/// let a = sci_core::csr::from_coo(1, 1, &[(0, 0, 2.0), (0, 0, -2.0)])?;
/// assert_eq!(a.data, [0.0]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
/// [SciPy COO storage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.html).
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
    // Place entries into their rows in input order, then sort only within each
    // row. This avoids comparing row keys for every entry of large CSR inputs.
    let mut row_offsets = vec![0; rows + 1];
    for &(r, _, _) in coordinates {
        row_offsets[r + 1] += 1;
    }
    for r in 0..rows {
        row_offsets[r + 1] += row_offsets[r];
    }
    let mut next = row_offsets[..rows].to_vec();
    let mut grouped = vec![(0, 0.0); coordinates.len()];
    for &(r, c, v) in coordinates {
        grouped[next[r]] = (c, v);
        next[r] += 1;
    }
    let mut a: SparseMatrix = SparseMatrix {
        rows,
        cols,
        indptr: vec![0; rows + 1],
        indices: Vec::with_capacity(coordinates.len()),
        data: Vec::with_capacity(coordinates.len()),
    };
    for r in 0..rows {
        let row = &mut grouped[row_offsets[r]..row_offsets[r + 1]];
        row.sort_by_key(|&(c, _)| c); // stable: duplicate summation order matters
        let mut previous = None;
        for &(c, v) in row.iter() {
            if previous == Some(c) {
                let last = a.data.last_mut().expect("previous entry exists");
                *last = match *last + v {
                    value if value.is_finite() => Ok(value),
                    _ => Err(Error::NonFiniteEvaluation),
                }?;
            } else {
                a.indices.push(c);
                a.data.push(v);
                previous = Some(c);
            }
        }
        a.indptr[r + 1] = a.data.len();
    }
    Ok(a)
}

/// Converts a canonical finite CSR matrix into shared row-major dense storage.
///
/// # Parameters and conventions
/// `a` has zero-based canonical CSR storage; implicit entries become +0.0.
///
/// # Domain and special values
/// Axes and entry counts are at most 1,000,000, and the dense result has at
/// most 1,000,000 cells. NaN/infinity and invalid CSR are rejected. `(0,n)`
/// with `n > 0` cannot be represented in row-vector storage.
///
/// # Returns
/// A dense matrix of `a.rows` row vectors, preserving stored values exactly,
/// including explicitly stored negative zeros.
///
/// # Errors
/// [`Error::InvalidInput`] for invalid CSR, excessive cells, or `(0,n)`.
///
/// # Numerical behavior
/// O(rows × cols + stored entries) work; no floating-point arithmetic.
///
/// # Examples
/// ```
/// let a = sci_core::csr::from_coo(1, 2, &[(0, 1, -0.0)])?;
/// let dense = sci_core::csr::to_dense(&a)?;
/// assert!(dense[0][1].is_sign_negative());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
/// [SciPy CSR toarray](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.toarray.html).
pub fn to_dense(a: &SparseMatrix) -> Result<Matrix> {
    validate(a)?;
    dense_size(a.rows, a.cols)?;
    if a.rows == 0 && a.cols != 0 {
        return Err(Error::InvalidInput(
            "shared rows cannot represent shape (0,n)",
        ));
    }
    let mut out = vec![vec![0.0; a.cols]; a.rows];
    for (r, row) in out.iter_mut().enumerate() {
        for p in a.indptr[r]..a.indptr[r + 1] {
            row[a.indices[p]] = a.data[p];
        }
    }
    Ok(out)
}
