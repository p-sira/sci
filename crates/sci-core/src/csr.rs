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
/// Stable coordinate sorting followed by sequential `f64` addition in original
/// order for duplicates. O(entries log entries + rows) work. This preserves
/// summation order and signed zeros of the original Sci sparse constructor.
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
    let mut coordinates = coordinates.to_vec();
    coordinates.sort_by_key(|&(r, c, _)| (r, c));
    let mut a: SparseMatrix = SparseMatrix {
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
            *last = match *last + v {
                value if value.is_finite() => Ok(value),
                _ => Err(Error::NonFiniteEvaluation),
            }?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_coo_stable_duplicate_summation_and_signed_zero() {
        let a = from_coo(
            2,
            3,
            &[
                (1, 2, -0.0),
                (0, 1, 1e16),
                (0, 2, 3.0),
                (0, 1, -1e16),
                (0, 1, 1.0),
                (1, 0, -2.0),
                (1, 0, 2.0),
            ],
        )
        .unwrap();
        assert_eq!(a.indptr, [0, 2, 4]);
        assert_eq!(a.indices, [1, 2, 0, 2]);
        assert_eq!(a.data, [1.0, 3.0, 0.0, -0.0]);
        assert_eq!(a.data[3].to_bits(), (-0.0f64).to_bits());
        validate(&a).unwrap();
        let dense = to_dense(&a).unwrap();
        assert_eq!(dense, [vec![0., 1., 3.], vec![0., 0., -0.]]);
        assert_eq!(dense[1][2].to_bits(), (-0.0f64).to_bits());
        assert_eq!(dense[1][1].to_bits(), 0.0f64.to_bits());

        // The duplicate sum is deliberately order-sensitive in binary64.
        assert_eq!(
            from_coo(1, 1, &[(0, 0, 1e16), (0, 0, 1.), (0, 0, -1e16)])
                .unwrap()
                .data,
            [0.]
        );
        assert_eq!(from_coo(0, 0, &[]).unwrap().indptr, [0]);
    }

    #[test]
    fn canonical_coo_and_csr_errors_are_precise() {
        assert_eq!(
            from_coo(1_000_001, 0, &[]),
            Err(Error::InvalidInput("sparse axis exceeds 1,000,000"))
        );
        assert_eq!(
            from_coo(1, 1, &[(1, 0, 1.)]),
            Err(Error::InvalidInput(
                "invalid sparse coordinate or nonfinite value"
            ))
        );
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                from_coo(1, 1, &[(0, 0, value)]),
                Err(Error::InvalidInput(
                    "invalid sparse coordinate or nonfinite value"
                ))
            );
        }
        assert_eq!(
            from_coo(1, 1, &[(0, 0, 1e308), (0, 0, 1e308)]),
            Err(Error::NonFiniteEvaluation)
        );

        let mut a = from_coo(1, 2, &[(0, 1, 1.)]).unwrap();
        a.indptr = vec![1, 1];
        assert_eq!(
            validate(&a),
            Err(Error::InvalidInput(
                "invalid CSR storage lengths or pointers"
            ))
        );
        let mut bad_pointers = from_coo(2, 2, &[(0, 0, 1.)]).unwrap();
        bad_pointers.indptr = vec![0, 2, 1];
        assert_eq!(
            validate(&bad_pointers),
            Err(Error::InvalidInput("invalid CSR row pointers"))
        );
        a.indptr = vec![0, 1];
        a.indices = vec![2];
        assert_eq!(
            validate(&a),
            Err(Error::InvalidInput(
                "CSR columns must be in bounds, sorted, and unique"
            ))
        );
        a.indices = vec![1, 1];
        a.data = vec![1., 2.];
        a.indptr = vec![0, 2];
        assert_eq!(
            validate(&a),
            Err(Error::InvalidInput(
                "CSR columns must be in bounds, sorted, and unique"
            ))
        );
        a.indices = vec![0, 1];
        a.data[1] = f64::NAN;
        assert_eq!(
            validate(&a),
            Err(Error::InvalidInput("coordinates must be finite"))
        );

        let empty_rows = from_coo(0, 2, &[]).unwrap();
        assert_eq!(
            to_dense(&empty_rows),
            Err(Error::InvalidInput(
                "shared rows cannot represent shape (0,n)"
            ))
        );
        let oversized = from_coo(1001, 1000, &[]).unwrap();
        assert_eq!(
            to_dense(&oversized),
            Err(Error::InvalidInput("dense output exceeds 1,000,000 cells"))
        );
    }
}
