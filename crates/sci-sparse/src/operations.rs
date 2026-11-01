//! Validation, structural transformations, arithmetic, and sparse products.

use super::*;
use std::collections::BTreeMap;

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
/// sci_sparse::operations::validate(&sci_sparse::construct::identity(3)?)?;
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy CSR storage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html).
pub fn validate(a: &SparseMatrix) -> Result<()> {
    sci_core::csr::validate(a)
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
/// let a = sci_sparse::construct::eye(2, 3, 1)?;
/// assert_eq!(sci_sparse::operations::transpose(&a)?.rows, 3);
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
/// assert_eq!(sci_sparse::operations::find(&sci_sparse::construct::identity(1)?)?, vec![(0, 0, 1.0)]);
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
/// let a = sci_sparse::construct::from_coo(1, 1, &[(0, 0, -0.0)])?;
/// assert!(sci_sparse::operations::eliminate_zeros(&a)?.data.is_empty());
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
/// assert_eq!(sci_sparse::operations::tril(&sci_sparse::construct::identity(2)?, 0)?.data, vec![1.0, 1.0]);
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
/// assert_eq!(sci_sparse::operations::triu(&sci_sparse::construct::identity(2)?, 0)?.data, vec![1.0, 1.0]);
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
/// assert_eq!(sci_sparse::operations::scale(&sci_sparse::construct::identity(2)?, 3.0)?.data, vec![3.0, 3.0]);
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
        *v = match *v * factor {
            value if value.is_finite() => Ok(value),
            _ => Err(Error::NonFiniteEvaluation),
        }?;
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
/// let a = sci_sparse::construct::identity(2)?;
/// assert_eq!(sci_sparse::operations::add(&a, &a)?.data, vec![2.0, 2.0]);
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
                let v = match a.data[i] + b.data[j] {
                    value if value.is_finite() => Ok(value),
                    _ => Err(Error::NonFiniteEvaluation),
                }?;
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
/// let a = sci_sparse::construct::identity(2)?;
/// assert_eq!(sci_sparse::operations::multiply(&a, &a)?, a);
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
                    push(
                        &mut out,
                        a.indices[i],
                        match a.data[i] * b.data[j] {
                            value if value.is_finite() => Ok(value),
                            _ => Err(Error::NonFiniteEvaluation),
                        }?,
                    )?;
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
/// assert_eq!(sci_sparse::operations::matvec(&sci_sparse::construct::identity(2)?, &[3.0, 4.0])?, vec![3.0, 4.0]);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy csr_array.dot](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.dot.html).
pub fn matvec(a: &SparseMatrix, x: &[f64]) -> Result<Vec<f64>> {
    validate(a)?;
    if x.len() != a.cols {
        return Err(Error::DimensionMismatch);
    }
    sci_core::is_finite_slice(x)?;
    let mut out = vec![0.0; a.rows];
    for (r, c, v) in entries(a) {
        out[r] = match out[r]
            + match v * x[c] {
                value if value.is_finite() => Ok(value),
                _ => Err(Error::NonFiniteEvaluation),
            }? {
            value if value.is_finite() => Ok(value),
            _ => Err(Error::NonFiniteEvaluation),
        }?;
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
/// let a = sci_sparse::construct::identity(2)?;
/// assert_eq!(sci_sparse::operations::matmul(&a, &a)?, a);
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
                *v = match *v
                    + match a.data[p] * b.data[q] {
                        value if value.is_finite() => Ok(value),
                        _ => Err(Error::NonFiniteEvaluation),
                    }? {
                    value if value.is_finite() => Ok(value),
                    _ => Err(Error::NonFiniteEvaluation),
                }?;
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
