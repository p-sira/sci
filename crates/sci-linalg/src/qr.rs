//! QR transformations through shared finite dense factorizations.
use super::backend::*;
use super::dense::{Matrix, QrResult, qr};
use crate::{Error, Result};
use faer::{Mat, Scale};
fn reconstructed(q: &[Vec<f64>], r: &[Vec<f64>]) -> Result<Mat<f64>> {
    let q = dense(q)?;
    let r = dense(r)?;
    if q.ncols() != r.nrows() {
        return Err(Error::DimensionMismatch);
    }
    let a = q * r;
    if a.iter().any(|v| !v.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    Ok(a)
}
/// Computes the columnwise Kronecker product `C[i*p+j,k]=A[i,k] B[j,k]`.
///
/// ## Parameters
///
/// `a` is m×n and `b` is p×n, with equal column counts. Output ordering is the A row first, then the B row.
///
/// ## Domain
///
/// Inputs must be finite, rectangular, and nonempty, with axes in 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// An (m*p)×n matrix. Output may have up to 1,000,000 entries.
///
/// ## Errors
///
/// Returns a shared error for invalid shapes, indices, work bounds, or nonfinite output.
///
/// ## Numerical behavior
///
/// Uses f64 and the private Householder QR backend. No pivoting, raw/full modes, batching, or overwrite options. Factor signs are nonunique. Direct O(mpn) products; no factorization. Complex inputs are omitted.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::khatri_rao(&[vec![1.,2.]],&[vec![3.,4.],vec![5.,6.]])?,vec![vec![3.,8.],vec![5.,12.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `khatri_rao`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.khatri_rao.html).
pub fn khatri_rao(
    a: &(impl super::MatrixInput + ?Sized),
    b: &(impl super::MatrixInput + ?Sized),
) -> Result<Matrix> {
    let a_input = super::input::matrix_rows(a)?;
    let a = a_input.as_ref();
    let b_input = super::input::matrix_rows(b)?;
    let b = b_input.as_ref();
    let a = dense(a)?;
    let b = dense(b)?;
    if a.ncols() != b.ncols() {
        return Err(Error::DimensionMismatch);
    }
    let (m, p, n) = (a.nrows(), b.nrows(), a.ncols());
    size(m * p, n)?;
    output(
        (0..m * p)
            .map(|r| (0..n).map(|k| a[(r / p, k)] * b[(r % p, k)]).collect())
            .collect(),
    )
}
/// Computes the economy RQ factorization `A=R Q`, `Q Qᵀ=I`, with upper trapezoidal R.
///
/// ## Parameters
///
/// `a` is m×n. Only economy real factors are returned.
///
/// ## Domain
///
/// Inputs must be finite, rectangular, and nonempty, with axes in 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// (R,Q), shapes m×k and k×n, k=min(m,n).
///
/// ## Errors
///
/// Returns a shared error for invalid shapes, indices, work bounds, or nonfinite output.
///
/// ## Numerical behavior
///
/// Uses f64 and the private Householder QR backend. No pivoting, raw/full modes, batching, or overwrite options. Factor signs are nonunique. O(mnk); reuses QR of the reversed transpose.
///
/// # Examples
///
/// ```
/// let (r,q)=sci_linalg::rq(&[vec![2.]])?;
/// assert!((r[0][0]*q[0][0]-2.).abs()<1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `rq`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.rq.html).
pub fn rq(a: &(impl super::MatrixInput + ?Sized)) -> Result<(Matrix, Matrix)> {
    let a_input = super::input::matrix_rows(a)?;
    let a = a_input.as_ref();
    let a = dense(a)?;
    let (m, n) = (a.nrows(), a.ncols());
    let k = m.min(n);
    let reversed = Mat::from_fn(n, m, |i, j| a[(m - 1 - j, n - 1 - i)]);
    let (reversed, scale) = scaled(reversed);
    let (q, mut r) = reversed.qr_factors();
    r *= Scale(scale);
    Ok((
        rows(&Mat::from_fn(m, k, |i, j| r[(k - 1 - j, m - 1 - i)]))?,
        rows(&Mat::from_fn(k, n, |i, j| q[(n - 1 - j, k - 1 - i)]))?,
    ))
}
/// Computes economy `A=Q R` and the product `Q C` or `C Q`.
///
/// ## Parameters
///
/// `left` selects Q C (true), requiring C rows=min(m,n), or C Q (false), requiring C columns=m. `a` is m×n. No conjugation mode for these real inputs.
///
/// ## Domain
///
/// Inputs must be finite, rectangular, and nonempty, with axes in 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// (product,R), with R k×n.
///
/// ## Errors
///
/// Returns a shared error for invalid shapes, indices, work bounds, or nonfinite output.
///
/// ## Numerical behavior
///
/// Uses f64 and the private Householder QR backend. No pivoting, raw/full modes, batching, or overwrite options. Factor signs are nonunique. Computes Q explicitly. Its nonunique signs may differ from SciPy along with corresponding rows of R.
///
/// # Examples
///
/// ```
/// let (product,r)=sci_linalg::qr_multiply(&[vec![2.]],&[vec![3.]],true)?;
/// assert!((product[0][0]*r[0][0]-6.).abs()<1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `qr_multiply`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.qr_multiply.html).
pub fn qr_multiply(
    a: &(impl super::MatrixInput + ?Sized),
    c: &(impl super::MatrixInput + ?Sized),
    left: bool,
) -> Result<(Matrix, Matrix)> {
    let a_input = super::input::matrix_rows(a)?;
    let a = a_input.as_ref();
    let c_input = super::input::matrix_rows(c)?;
    let c = c_input.as_ref();
    let f = qr(a)?;
    let q = dense(&f.q)?;
    let c = dense(c)?;
    let product = if left {
        if q.ncols() != c.nrows() {
            return Err(Error::DimensionMismatch);
        }
        q * c
    } else {
        if c.ncols() != q.nrows() {
            return Err(Error::DimensionMismatch);
        }
        c * q
    };
    Ok((rows(&product)?, f.r))
}
/// Computes QR factors of the rank-one update `Q R+u vᵀ`.
///
/// ## Parameters
///
/// `q` and `r` have compatible product m×n; `u` has m entries and `v` n. Higher-rank updates and complex inputs are omitted.
///
/// ## Domain
///
/// Inputs must be finite, rectangular, and nonempty, with axes in 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// New economy [`QrResult`] factors.
///
/// ## Errors
///
/// Returns a shared error for invalid shapes, indices, work bounds, or nonfinite output.
///
/// ## Numerical behavior
///
/// Uses f64 and the private Householder QR backend. No pivoting, raw/full modes, batching, or overwrite options. Factor signs are nonunique. Reconstructs and refactors in O(mnk), rather than SciPy's specialized update. Does not test whether the supplied Q is orthonormal.
///
/// # Examples
///
/// ```
/// let f=sci_linalg::qr_update(&[vec![1.]],&[vec![2.]],&[3.],&[4.])?;
/// assert!((f.q[0][0]*f.r[0][0]-14.).abs()<1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `qr_update`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.qr_update.html).
pub fn qr_update(
    q: &(impl super::MatrixInput + ?Sized),
    r: &(impl super::MatrixInput + ?Sized),
    u: &(impl super::VectorInput + ?Sized),
    v: &(impl super::VectorInput + ?Sized),
) -> Result<QrResult> {
    let q_input = super::input::matrix_rows(q)?;
    let q = q_input.as_ref();
    let r_input = super::input::matrix_rows(r)?;
    let r = r_input.as_ref();
    let u_input = super::input::vector_slice(u)?;
    let u = u_input.as_ref();
    let v_input = super::input::vector_slice(v)?;
    let v = v_input.as_ref();
    let mut a = reconstructed(q, r)?;
    if u.len() != a.nrows() || v.len() != a.ncols() {
        return Err(Error::DimensionMismatch);
    }
    crate::error::finite_slice(u)?;
    crate::error::finite_slice(v)?;
    for (i, &u) in u.iter().enumerate() {
        for (j, &v) in v.iter().enumerate() {
            a[(i, j)] += u * v;
        }
    }
    qr(&rows(&a)?)
}
/// Computes QR factors after inserting consecutive rows or columns into `A=Q R`.
///
/// ## Parameters
///
/// `index` is a zero-based insertion position, inclusive of the axis end. `rows` chooses row insertion; `values` has p×n shape for rows or m×p for columns. At least one row/column is inserted.
///
/// ## Domain
///
/// Inputs must be finite, rectangular, and nonempty, with axes in 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// New economy [`QrResult`], with the modified axis increased by p.
///
/// ## Errors
///
/// Returns a shared error for invalid shapes, indices, work bounds, or nonfinite output.
///
/// ## Numerical behavior
///
/// Uses f64 and the private Householder QR backend. No pivoting, raw/full modes, batching, or overwrite options. Factor signs are nonunique. Reconstructs and refactors in O(mnk). The supplied Q need not be orthonormal; no rcond check on the inserted rows is offered.
///
/// # Examples
///
/// ```
/// let f=sci_linalg::qr_insert(&[vec![1.]],&[vec![2.]],&[vec![3.]],1,true)?;
/// assert_eq!((f.q.len(),f.r[0].len()),(2,1));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `qr_insert`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.qr_insert.html).
pub fn qr_insert(
    q: &(impl super::MatrixInput + ?Sized),
    r: &(impl super::MatrixInput + ?Sized),
    values: &(impl super::MatrixInput + ?Sized),
    index: usize,
    insert_rows: bool,
) -> Result<QrResult> {
    let q_input = super::input::matrix_rows(q)?;
    let q = q_input.as_ref();
    let r_input = super::input::matrix_rows(r)?;
    let r = r_input.as_ref();
    let values_input = super::input::matrix_rows(values)?;
    let values = values_input.as_ref();
    let a = reconstructed(q, r)?;
    let v = dense(values)?;
    let (m, n) = (a.nrows(), a.ncols());
    let (out_m, out_n) = if insert_rows {
        if index > m || v.ncols() != n {
            return Err(Error::DimensionMismatch);
        }
        (m + v.nrows(), n)
    } else {
        if index > n || v.nrows() != m {
            return Err(Error::DimensionMismatch);
        }
        (m, n + v.ncols())
    };
    if out_m > MAX_DIM || out_n > MAX_DIM {
        return Err(Error::InvalidInput("inserted matrix axes exceed 256"));
    }
    let out = Mat::from_fn(out_m, out_n, |i, j| {
        if insert_rows {
            if i < index {
                a[(i, j)]
            } else if i < index + v.nrows() {
                v[(i - index, j)]
            } else {
                a[(i - v.nrows(), j)]
            }
        } else if j < index {
            a[(i, j)]
        } else if j < index + v.ncols() {
            v[(i, j - index)]
        } else {
            a[(i, j - v.ncols())]
        }
    });
    qr(&rows(&out)?)
}
/// Computes QR factors after deleting consecutive rows or columns from `A=Q R`.
///
/// ## Parameters
///
/// `index` is zero-based; `count` is positive; `rows` chooses rows (true) or columns (false). The deleted range must be inside its axis and leave it nonempty.
///
/// ## Domain
///
/// Inputs must be finite, rectangular, and nonempty, with axes in 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// New economy [`QrResult`] factors.
///
/// ## Errors
///
/// Returns a shared error for invalid shapes, indices, work bounds, or nonfinite output.
///
/// ## Numerical behavior
///
/// Uses f64 and the private Householder QR backend. No pivoting, raw/full modes, batching, or overwrite options. Factor signs are nonunique. Reconstructs and refactors in O(mnk), instead of SciPy's specialized downdate. Does not test the supplied Q for orthonormality.
///
/// # Examples
///
/// ```
/// let f=sci_linalg::qr_delete(&[vec![1.,0.],vec![0.,1.]],&[vec![2.],vec![3.]],0,1,true)?;
/// assert!((f.q[0][0]*f.r[0][0]-3.).abs()<1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `qr_delete`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.qr_delete.html).
pub fn qr_delete(
    q: &(impl super::MatrixInput + ?Sized),
    r: &(impl super::MatrixInput + ?Sized),
    index: usize,
    count: usize,
    delete_rows: bool,
) -> Result<QrResult> {
    let q_input = super::input::matrix_rows(q)?;
    let q = q_input.as_ref();
    let r_input = super::input::matrix_rows(r)?;
    let r = r_input.as_ref();
    let a = reconstructed(q, r)?;
    let (m, n) = (a.nrows(), a.ncols());
    let axis = if delete_rows { m } else { n };
    if count == 0 || count >= axis || index > axis - count {
        return Err(Error::InvalidInput("invalid QR deletion range"));
    }
    let (out_m, out_n) = if delete_rows {
        (m - count, n)
    } else {
        (m, n - count)
    };
    let out = Mat::from_fn(out_m, out_n, |i, j| {
        a[(
            if delete_rows && i >= index {
                i + count
            } else {
                i
            },
            if !delete_rows && j >= index {
                j + count
            } else {
                j
            },
        )]
    });
    qr(&rows(&out)?)
}
