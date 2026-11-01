//! Sparse CSR construction, format conversion, and block assembly.

use super::*;
#[cfg(feature = "random")]
use crate::array::random_array;
use crate::operations::add;

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
/// let a = sci_sparse::construct::from_coo(1, 1, &[(0, 0, 2.0), (0, 0, 3.0)])?;
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
    sci_core::csr::from_coo(rows, cols, coordinates)
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
/// let a = sci_sparse::construct::from_csr(1, 2, &[0, 2], &[1, 0], &[3.0, 2.0])?;
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
/// assert_eq!(sci_sparse::construct::from_csc(1, 2, &[0, 1, 1], &[0], &[2.0])?.data, vec![2.0]);
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
/// let a = sci_sparse::construct::from_dense(&[vec![0.0, 3.0]])?;
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
/// let a = sci_sparse::construct::identity(2)?;
/// assert_eq!(sci_sparse::construct::to_dense(&a)?[0], vec![1.0, 0.0]);
/// # Ok::<(), sci_sparse::Error>(())
/// ```
/// # References
/// [SciPy csr_array.toarray](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.toarray.html).
pub fn to_dense(a: &SparseMatrix) -> Result<Matrix> {
    sci_core::csr::to_dense(a)
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
/// assert_eq!(sci_sparse::construct::eye(2, 3, 1)?.indices, vec![1, 2]);
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
/// assert_eq!(sci_sparse::construct::identity(2)?.data, vec![1.0, 1.0]);
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
/// let a = sci_sparse::construct::diags(2, 3, &[vec![2.0, 3.0]], &[1])?;
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
/// let a = sci_sparse::construct::identity(2)?;
/// assert_eq!(sci_sparse::construct::hstack(&[&a, &a])?.cols, 4);
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
/// let a = sci_sparse::construct::identity(2)?;
/// assert_eq!(sci_sparse::construct::vstack(&[&a, &a])?.rows, 4);
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
/// let a = sci_sparse::construct::identity(2)?;
/// assert_eq!(sci_sparse::construct::block_diag(&[&a, &a])?.rows, 4);
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
/// let a = sci_sparse::construct::identity(1)?;
/// assert_eq!(sci_sparse::construct::block_array(&[vec![&a, &a]])?.cols, 2);
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
/// let a = sci_sparse::construct::identity(2)?;
/// assert_eq!(sci_sparse::construct::kron(&a, &a)?, sci_sparse::construct::identity(4)?);
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
            coordinates.push((
                i * b.rows + k,
                j * b.cols + l,
                match v * w {
                    value if value.is_finite() => Ok(value),
                    _ => Err(Error::NonFiniteEvaluation),
                }?,
            ));
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
/// Uses [`kron`] and [`crate::operations::add`], including their limits and f64 rounding behavior.
/// # Examples
/// ```
/// let a = sci_sparse::construct::identity(2)?;
/// assert_eq!(sci_sparse::construct::kronsum(&a, &a)?.data, vec![2.0; 4]);
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

/// Generate a two-dimensional sparse sample with uniform `[0,1)` values.
///
/// # Parameters and conventions
/// rows, cols set shape; density and seed follow [`crate::array::random_array`]. The shared
/// CSR type replaces SciPy's format/dtype options and legacy matrix record.
/// # Domain and special values
/// Finite density in `[0,1]`, crate axis/storage limits and volume <=20,000,000.
/// # Returns
/// Canonical CSR with ties-to-even rounded density×volume stored entries.
/// # Errors
/// Invalid density, shape/volume overflow, or sampling/storage limits.
/// # Numerical behavior
/// Uses [`crate::array::random_array`]; streams differ from NumPy. No custom sampler argument;
/// callers may replace sampled data with validated finite values explicitly.
/// # Examples
/// ```
/// assert_eq!(sci_sparse::construct::random(2,3,1.,42)?.data.len(),6);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy random](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.random.html).
#[cfg(feature = "random")]
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
/// let a=sci_sparse::construct::spdiags(&[vec![0.,2.,3.]],&[1],3,3)?;
/// assert_eq!(sci_sparse::construct::to_dense(&a)?[0][1],2.);
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
