//! Dense real decompositions and solvers using a private faer backend.
use super::backend::*;
use super::structured::{circulant, toeplitz};
use crate::{Error, Result};
use faer::prelude::Solve;
use faer::{Mat, Scale};

pub use sci_core::Matrix;
/// Economy singular-value decomposition `A = U diag(s) Vh`.
#[derive(Clone, Debug)]
pub struct SvdResult {
    /// Left singular vectors as columns, shape m×min(m,n).
    pub u: Matrix,
    /// Nonnegative singular values in descending order.
    pub singular_values: Vec<f64>,
    /// Transposed right singular vectors, shape min(m,n)×n.
    pub vh: Matrix,
}
/// Symmetric eigendecomposition `A V = V diag(w)`.
#[derive(Clone, Debug)]
pub struct EigenResult {
    /// Eigenvalues in ascending order.
    pub eigenvalues: Vec<f64>,
    /// Orthonormal eigenvectors as columns, ordered with the eigenvalues.
    pub eigenvectors: Matrix,
}
/// Economy QR factors `A = Q R`.
#[derive(Clone, Debug)]
pub struct QrResult {
    /// Orthonormal columns, shape m×min(m,n).
    pub q: Matrix,
    /// Upper trapezoidal factor, shape min(m,n)×n.
    pub r: Matrix,
}
/// Pivoted LU factors `A = P L U`.
#[derive(Clone, Debug)]
pub struct LuResult {
    /// Permutation matrix, shape m×m, with SciPy's `A=P L U` orientation.
    pub p: Matrix,
    /// Unit lower trapezoidal factor, shape m×min(m,n).
    pub l: Matrix,
    /// Upper trapezoidal factor, shape min(m,n)×n.
    pub u: Matrix,
}
/// Reusable Cholesky factor, containing no backend types.
#[derive(Clone, Debug)]
pub struct CholeskyFactor {
    /// Triangular factor; the opposite triangle is zero.
    pub factor: Matrix,
    /// Whether `A=L Lᵀ` (true) or `A=Uᵀ U` (false).
    pub lower: bool,
}
/// Minimum-norm least-squares result.
#[derive(Clone, Debug)]
pub struct LeastSquaresResult {
    /// Solution vector, length n.
    pub solution: Vec<f64>,
    /// One residual sum of squares for m>n and full rank; otherwise empty.
    pub residuals: Vec<f64>,
    /// Effective rank after the singular-value cutoff.
    pub rank: usize,
    /// Singular values in descending order.
    pub singular_values: Vec<f64>,
}
/// Matrix norm order, without axes or batching.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixNorm {
    /// Square root of the sum of squared entries.
    Frobenius,
    /// Largest absolute column sum.
    One,
    /// Smallest absolute column sum.
    NegativeOne,
    /// Largest absolute row sum.
    Infinity,
    /// Smallest absolute row sum.
    NegativeInfinity,
    /// Largest singular value.
    Spectral,
    /// Smallest singular value.
    SmallestSingular,
    /// Sum of singular values.
    Nuclear,
}
struct SvdFactors {
    u: Option<Mat<f64>>,
    singular_values: Vec<f64>,
    v_t: Option<Mat<f64>>,
}
fn decomposition(a: Mat<f64>, vectors: bool) -> Result<SvdFactors> {
    let (a, scale) = scaled(a);
    let (u, v_t, mut singular_values) = if vectors {
        let d = a
            .thin_svd()
            .map_err(|_| Error::NumericalFailure("SVD did not converge"))?;
        (
            Some(d.U().to_owned()),
            Some(d.V().transpose().to_owned()),
            (0..a.nrows().min(a.ncols()))
                .map(|i| d.S()[i])
                .collect::<Vec<_>>(),
        )
    } else {
        (
            None,
            None,
            a.singular_values()
                .map_err(|_| Error::NumericalFailure("SVD did not converge"))?,
        )
    };
    for x in &mut singular_values {
        *x = crate::error::finite(*x * scale)?;
    }
    Ok(SvdFactors {
        u,
        v_t,
        singular_values,
    })
}
fn eigendecomposition(a: Mat<f64>) -> Result<EigenResult> {
    let (a, scale) = scaled(a);
    let d = a
        .self_adjoint_eigen(faer::Side::Lower)
        .map_err(|_| Error::NumericalFailure("symmetric eigensolver did not converge"))?;
    Ok(EigenResult {
        eigenvalues: (0..a.nrows())
            .map(|i| crate::error::finite(d.S()[i] * scale))
            .collect::<Result<_>>()?,
        eigenvectors: rows(&d.U())?,
    })
}
fn eigenvalues(a: Mat<f64>) -> Result<Vec<f64>> {
    let (a, scale) = scaled(a);
    let mut values = a
        .self_adjoint_eigenvalues(faer::Side::Lower)
        .map_err(|_| Error::NumericalFailure("symmetric eigensolver did not converge"))?;
    for value in &mut values {
        *value = crate::error::finite(*value * scale)?;
    }
    Ok(values)
}
fn rhs(b: &[f64], n: usize) -> Result<Mat<f64>> {
    if b.len() != n {
        return Err(Error::DimensionMismatch);
    }
    crate::error::finite_slice(b)?;
    Ok(Mat::from_fn(n, 1, |i, _| b[i]))
}
fn vector(a: Mat<f64>) -> Result<Vec<f64>> {
    if a.iter().any(|v| !v.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    Ok(a.iter().copied().collect())
}
fn pseudoinverse(d: &mut SvdFactors, cutoff: f64) -> Result<(Mat<f64>, usize)> {
    let mut vt = d
        .v_t
        .take()
        .ok_or(Error::NumericalFailure("missing right singular vectors"))?;
    let u =
        d.u.as_ref()
            .ok_or(Error::NumericalFailure("missing left singular vectors"))?;
    let mut rank = 0;
    for (i, &sigma) in d.singular_values.iter().enumerate() {
        let inv = if sigma > cutoff {
            rank += 1;
            1.0 / sigma
        } else {
            0.0
        };
        vt.as_mut().row_mut(i).iter_mut().for_each(|x| *x *= inv);
    }
    Ok((vt.transpose() * u.transpose(), rank))
}
/// Solves the square linear system `A x=b`.
///
/// ## Parameters
///
/// `a` is n×n and `b` has length n. Supports one real right-hand side; SciPy batching, transposition, and structure/driver hints are omitted.
///
/// ## Domain
///
/// All matrix entries and vector inputs must be finite, rectangular, and nonempty. Dense input axes are limited to 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// The solution vector of length n.
///
/// ## Errors
///
/// Returns a shared error for invalid domains or shapes, singular systems where applicable, nonconvergence, or nonfinite output.
///
/// ## Numerical behavior
///
/// Uses O(n³) private faer pivoted LU, followed by at most four O(n²)
/// refinement steps reusing the factors. Fused product residuals and compensated
/// sums reduce cancellation in `b−Ax`; only finite updates that reduce the
/// estimated remaining correction are retained, and refinement stops once a
/// correction falls below one ulp of `‖x‖∞`. Refinement is skipped when residual arithmetic
/// exceeds the finite range. Ill-conditioned systems can still lose accuracy;
/// no condition-number warning or forward-error guarantee is supplied.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::solve(&[vec![2.,0.],vec![0.,4.]],&[6.,8.])?,vec![3.,2.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `solve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve.html).
pub fn solve(
    a: &(impl super::MatrixInput + ?Sized),
    b: &(impl super::VectorInput + ?Sized),
) -> Result<Vec<f64>> {
    let b_input = super::input::vector_slice(b)?;
    let b = b_input.as_ref();
    let a_input = square_ref(a)?;
    let a = a_input.as_ref();
    let b = rhs(b, a.nrows())?;
    let lu = a.partial_piv_lu();
    if (0..a.nrows()).any(|i| lu.U()[(i, i)] == 0. || !lu.U()[(i, i)].is_finite()) {
        return Err(Error::NumericalFailure("singular linear system"));
    }
    let mut x = lu.solve(&b);
    let mut correction = solve_residual(a, &b, &x);
    if correction.iter().any(|r| !r.is_finite()) {
        return vector(x);
    }
    lu.solve_in_place(correction.as_mut());
    for _ in 0..4 {
        let norm = correction.iter().fold(0_f64, |m, r| m.max(r.abs()));
        if norm == 0. || correction.iter().any(|r| !r.is_finite()) {
            break;
        }
        let candidate = &x + &correction;
        if candidate.iter().any(|v| !v.is_finite()) {
            break;
        }
        // A correction below one ulp of ‖x‖∞ has converged; verifying it would
        // cost another residual and triangular solve without changing `x`
        // beyond rounding.
        if norm <= f64::EPSILON * x.iter().fold(0_f64, |m, v| m.max(v.abs())) {
            x = candidate;
            break;
        }
        let mut next = solve_residual(a, &b, &candidate);
        if next.iter().any(|r| !r.is_finite()) {
            break;
        }
        lu.solve_in_place(next.as_mut());
        if next.iter().any(|r| !r.is_finite())
            || next.iter().fold(0_f64, |m, r| m.max(r.abs())) >= norm
        {
            break;
        }
        x = candidate;
        correction = next;
    }
    vector(x)
}

fn solve_residual(a: faer::MatRef<'_, f64>, b: &Mat<f64>, x: &Mat<f64>) -> Mat<f64> {
    let n = a.nrows();
    let mut residual = b.to_owned();
    let mut correction = vec![0.; n];
    let sum = residual
        .col_mut(0)
        .try_as_col_major_mut()
        .expect("owned faer columns are contiguous")
        .as_slice_mut();
    // Column traversal over borrowed or owned storage; each row still
    // accumulates its products in column order.
    for j in 0..a.ncols() {
        let xj = x[(j, 0)];
        let column = a.col(j);
        let split_x = split(xj);
        for ((s, c), &aij) in sum.iter_mut().zip(&mut correction).zip(column.iter()) {
            let (product, error) = two_product(-aij, xj, split_x);
            let next = *s + product;
            let back = next - *s;
            *c += ((*s - (next - back)) + (product - back)) + error;
            *s = next;
        }
    }
    for (s, c) in sum.iter_mut().zip(correction) {
        *s += c;
    }
    #[cfg(not(target_feature = "fma"))]
    if residual.col(0).iter().any(|r| !r.is_finite()) {
        // Dekker splitting overflows only for entries near the finite limit;
        // recompute those residuals with library FMA.
        return Mat::from_fn(n, 1, |i, _| {
            let mut sum = b[(i, 0)];
            let mut correction = 0.;
            for j in 0..a.ncols() {
                let product = -a[(i, j)] * x[(j, 0)];
                let error = (-a[(i, j)]).mul_add(x[(j, 0)], -product);
                let next = sum + product;
                let back = next - sum;
                correction += ((sum - (next - back)) + (product - back)) + error;
                sum = next;
            }
            sum + correction
        });
    }
    residual
}

/// Veltkamp split `v = hi + lo` with 26-bit halves for Dekker's product.
#[inline]
fn split(v: f64) -> (f64, f64) {
    let scaled = 134_217_729.0 * v;
    let hi = scaled - (scaled - v);
    (hi, v - hi)
}

/// Returns `(fl(a·b), a·b − fl(a·b))`, given `split_b = split(b)`.
///
/// Uses hardware FMA when the target enables it; otherwise Dekker's product
/// avoids the library `fma` call that `f64::mul_add` lowers to.
#[inline]
fn two_product(a: f64, b: f64, split_b: (f64, f64)) -> (f64, f64) {
    let product = a * b;
    #[cfg(target_feature = "fma")]
    let error = {
        let _ = split_b;
        a.mul_add(b, -product)
    };
    #[cfg(not(target_feature = "fma"))]
    let error = {
        let (a_hi, a_lo) = split(a);
        let (b_hi, b_lo) = split_b;
        ((a_hi * b_hi - product) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo
    };
    (product, error)
}
/// Computes `A⁻¹` satisfying `A A⁻¹=I`.
///
/// ## Parameters
///
/// `a` is a real square matrix. No batching, overwrite options, or backend exposure.
///
/// ## Domain
///
/// All matrix entries and vector inputs must be finite, rectangular, and nonempty. Dense input axes are limited to 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// An n×n inverse.
///
/// ## Errors
///
/// Returns a shared error for invalid domains or shapes, singular systems where applicable, nonconvergence, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private faer algorithms use f64. Ill-conditioned results can lose accuracy. Iterative routines use bounded backend-specific sweep limits. O(n³); singularity detection is based on backend pivots, without a conditioning guarantee.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::inv(&[vec![2.]])?,vec![vec![0.5]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `inv`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.inv.html).
pub fn inv(a: &(impl super::MatrixInput + ?Sized)) -> Result<Matrix> {
    rows(
        &square(a)?
            .inverse_checked()
            .ok_or(Error::NumericalFailure("singular matrix"))?,
    )
}
/// Computes `det(A)=Σσ sign(σ) Πi A[i,σ(i)]`.
///
/// ## Parameters
///
/// `a` is square. This adaptation rejects empty matrices (SciPy returns 1).
///
/// ## Domain
///
/// All matrix entries and vector inputs must be finite, rectangular, and nonempty. Dense input axes are limited to 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// The signed determinant; singular matrices return zero.
///
/// ## Errors
///
/// Returns a shared error for invalid domains or shapes, singular systems where applicable, nonconvergence, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private faer algorithms use f64. Ill-conditioned results can lose accuracy. Iterative routines use bounded backend-specific sweep limits. O(n³) LU. Products can underflow to zero or overflow to an error.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::det(&[vec![2.]])?,2.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `det`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.det.html).
pub fn det(a: &(impl super::MatrixInput + ?Sized)) -> Result<f64> {
    crate::error::finite(square_ref(a)?.as_ref().determinant())
}
/// Solves `T x=b` or `Tᵀ x=b` for a triangular matrix T.
///
/// ## Parameters
///
/// `lower` selects the triangle; the opposite triangle is ignored. `unit_diagonal` treats all diagonal entries as 1. `transpose` solves the transposed system. One real RHS.
///
/// ## Domain
///
/// All matrix entries and vector inputs must be finite, rectangular, and nonempty. Dense input axes are limited to 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// The vector x.
///
/// ## Errors
///
/// Returns a shared error for invalid domains or shapes, singular systems where applicable, nonconvergence, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private faer algorithms use f64. Ill-conditioned results can lose accuracy. Iterative routines use bounded backend-specific sweep limits. O(n²) substitution on a borrowed input view. Ignored triangle and unit diagonal still require finite supplied entries.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::solve_triangular(&[vec![2.,9.],vec![1.,3.]],&[4.,8.],true,false,false)?,vec![2.,2.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `solve_triangular`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_triangular.html).
pub fn solve_triangular(
    a: &(impl super::MatrixInput + ?Sized),
    b: &(impl super::VectorInput + ?Sized),
    lower: bool,
    unit_diagonal: bool,
    transpose: bool,
) -> Result<Vec<f64>> {
    let b_input = super::input::vector_slice(b)?;
    let b = b_input.as_ref();
    let a_input = square_ref(a)?;
    let a = a_input.as_ref();
    if b.len() != a.nrows() {
        return Err(Error::DimensionMismatch);
    }
    crate::error::finite_slice(b)?;
    let mut x = b.to_vec();
    substitute(a, &mut x, lower, unit_diagonal, transpose)?;
    if x.iter().any(|v| !v.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    Ok(x)
}
/// Overwrites `x` with the solution of `op(A) x = x` for triangular A.
///
/// Small systems dominate this crate's workloads, so a direct substitution
/// over contiguous columns avoids the dispatch and recursion overhead of the
/// blocked backend kernels.
pub(super) fn substitute(
    a: faer::MatRef<'_, f64>,
    x: &mut [f64],
    lower: bool,
    unit_diagonal: bool,
    transpose: bool,
) -> Result<()> {
    let n = a.nrows();
    debug_assert_eq!(x.len(), n);
    if !unit_diagonal && (0..n).any(|i| a[(i, i)] == 0.0) {
        return Err(Error::NumericalFailure("zero triangular pivot"));
    }
    // A view with both strides negative is A = J B J for the forward view B,
    // where J reverses order: solve op(B) (J x) = J b on B, whose triangle is
    // the opposite one, instead of copying A.
    let reversed = n > 1 && a.row_stride() < 0 && a.col_stride() < 0;
    let a = if reversed {
        a.reverse_rows_and_cols()
    } else {
        a
    };
    let effective_lower = (lower != transpose) != reversed;
    if reversed {
        x.reverse();
    }
    if a.row_stride() == 1 {
        substitute_col_major(a, x, effective_lower, unit_diagonal, transpose);
    } else if a.col_stride() == 1 {
        // Row-major storage is the column-major transpose; op(A) is unchanged.
        substitute_col_major(a.transpose(), x, effective_lower, unit_diagonal, !transpose);
    } else {
        let a = a.to_owned();
        substitute_col_major(a.as_ref(), x, effective_lower, unit_diagonal, transpose);
    }
    if reversed {
        x.reverse();
    }
    Ok(())
}
fn substitute_col_major(
    a: faer::MatRef<'_, f64>,
    x: &mut [f64],
    effective_lower: bool,
    unit_diagonal: bool,
    transpose: bool,
) {
    let n = a.nrows();
    let column = |j: usize| {
        a.col(j)
            .try_as_col_major()
            .expect("unit row stride gives contiguous columns")
            .as_slice()
    };
    if transpose {
        // Row i of Aᵀ is column i of A: contiguous dot products.
        for k in 0..n {
            let i = if effective_lower { k } else { n - 1 - k };
            let c = column(i);
            let range = if effective_lower { 0..i } else { i + 1..n };
            let mut value = x[i];
            for (&aji, &xj) in c[range.clone()].iter().zip(&x[range]) {
                value -= aji * xj;
            }
            x[i] = if unit_diagonal { value } else { value / c[i] };
        }
    } else {
        // Column-oriented elimination keeps every update contiguous.
        for k in 0..n {
            let j = if effective_lower { k } else { n - 1 - k };
            let c = column(j);
            if !unit_diagonal {
                x[j] /= c[j];
            }
            let xj = x[j];
            let range = if effective_lower { j + 1..n } else { 0..j };
            for (xi, &aij) in x[range.clone()].iter_mut().zip(&c[range]) {
                *xi -= aij * xj;
            }
        }
    }
}
/// Solves `T(c,r) x=b`, with the diagonals defined by [`toeplitz`].
///
/// ## Parameters
///
/// `c` is the first column, optional `r` the first row; `r[0]` is ignored. A square real system and one RHS are required.
///
/// ## Domain
///
/// All matrix entries and vector inputs must be finite, rectangular, and nonempty. Dense input axes are limited to 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// The vector x.
///
/// ## Errors
///
/// Returns a shared error for invalid domains or shapes, singular systems where applicable, nonconvergence, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private faer algorithms use f64. Ill-conditioned results can lose accuracy. Iterative routines use bounded backend-specific sweep limits. O(n³) dense LU replaces SciPy's O(n²) Levinson recursion.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::solve_toeplitz(&[2.,0.],None,&[4.,6.])?,vec![2.,3.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `solve_toeplitz`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_toeplitz.html).
pub fn solve_toeplitz(
    c: &(impl super::VectorInput + ?Sized),
    r: Option<&[f64]>,
    b: &(impl super::VectorInput + ?Sized),
) -> Result<Vec<f64>> {
    let c_input = super::input::vector_slice(c)?;
    let c = c_input.as_ref();
    let b_input = super::input::vector_slice(b)?;
    let b = b_input.as_ref();
    if c.len() > MAX_DIM || r.is_some_and(|r| r.len() > MAX_DIM) {
        return Err(Error::InvalidInput("Toeplitz solver order exceeds 256"));
    }
    solve(&toeplitz(c, r)?, b)
}
/// Computes `Y=T(c,r) X`.
///
/// ## Parameters
///
/// `c`, optional `r` define [`toeplitz`]; `x` is a real matrix with len(r) rows. No batching, worker or FFT options.
///
/// ## Domain
///
/// All matrix entries and vector inputs must be finite, rectangular, and nonempty. Dense input axes are limited to 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// A row-major matrix with len(c) rows and x column count.
///
/// ## Errors
///
/// Returns a shared error for invalid domains or shapes, singular systems where applicable, nonconvergence, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private faer algorithms use f64. Ill-conditioned results can lose accuracy. Iterative routines use bounded backend-specific sweep limits. Uses a bounded dense product rather than SciPy's FFT multiplication, O(mnk).
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::matmul_toeplitz(&[2.,0.],None,&[vec![1.],vec![3.]])?,vec![vec![2.],vec![6.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `matmul_toeplitz`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.matmul_toeplitz.html).
pub fn matmul_toeplitz(
    c: &(impl super::VectorInput + ?Sized),
    r: Option<&[f64]>,
    x: &(impl super::MatrixInput + ?Sized),
) -> Result<Matrix> {
    let c_input = super::input::vector_slice(c)?;
    let c = c_input.as_ref();
    if c.len() > MAX_DIM || r.is_some_and(|r| r.len() > MAX_DIM) {
        return Err(Error::InvalidInput("Toeplitz product axes exceed 256"));
    }
    let t = dense(&toeplitz(c, r)?)?;
    let x = dense(x)?;
    if t.ncols() != x.nrows() {
        return Err(Error::DimensionMismatch);
    }
    rows(&(t * x))
}
/// Solves `C(c) x=b`, where `C[i,j]=c[(i-j) mod n]`.
///
/// ## Parameters
///
/// `c` is the first column; `b` is one RHS. Singular least-squares mode and SciPy tolerance options are omitted.
///
/// ## Domain
///
/// All matrix entries and vector inputs must be finite, rectangular, and nonempty. Dense input axes are limited to 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// The vector x.
///
/// ## Errors
///
/// Returns a shared error for invalid domains or shapes, singular systems where applicable, nonconvergence, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private faer algorithms use f64. Ill-conditioned results can lose accuracy. Iterative routines use bounded backend-specific sweep limits. O(n³) dense LU replaces SciPy's FFT solver; singularity detection therefore differs for nearly singular input.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::solve_circulant(&[2.,0.],&[4.,6.])?,vec![2.,3.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `solve_circulant`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_circulant.html).
pub fn solve_circulant(
    c: &(impl super::VectorInput + ?Sized),
    b: &(impl super::VectorInput + ?Sized),
) -> Result<Vec<f64>> {
    let c_input = super::input::vector_slice(c)?;
    let c = c_input.as_ref();
    let b_input = super::input::vector_slice(b)?;
    let b = b_input.as_ref();
    if c.len() > MAX_DIM {
        return Err(Error::InvalidInput("circulant solver order exceeds 256"));
    }
    solve(&circulant(c)?, b)
}
/// Computes `A=L Lᵀ` (lower) or `A=Uᵀ U` (upper).
///
/// ## Parameters
///
/// `lower` selects both the input triangle and output factor. The opposite input triangle is ignored, after a finite-input scan. No batching.
///
/// ## Domain
///
/// All matrix entries and vector inputs must be finite, rectangular, and nonempty. Dense input axes are limited to 1..=256. NaN and infinity are rejected. The selected symmetric matrix must be positive definite.
///
/// ## Returns
///
/// A triangular square factor with the opposite triangle set to zero.
///
/// ## Errors
///
/// Returns a shared error for invalid domains or shapes, singular systems where applicable, nonconvergence, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private faer algorithms use f64. Ill-conditioned results can lose accuracy. Iterative routines use bounded backend-specific sweep limits. O(n³); near-semidefinite input can fail due to roundoff.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::cholesky(&[vec![4.]],true)?,vec![vec![2.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `cholesky`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cholesky.html).
pub fn cholesky(a: &(impl super::MatrixInput + ?Sized), lower: bool) -> Result<Matrix> {
    let a = square_ref(a)?;
    let a = a.as_ref();
    let d = (if lower { a } else { a.transpose() })
        .llt(faer::Side::Lower)
        .map_err(|_| Error::NumericalFailure("matrix is not positive definite"))?;
    if lower {
        rows(&d.L())
    } else {
        rows(&d.L().transpose())
    }
}
/// Factors `A=L Lᵀ` or `A=Uᵀ U` for reuse by [`cho_solve`].
///
/// ## Parameters
///
/// Arguments match [`cholesky`]. The owned return type replaces SciPy's (factor,lower) tuple.
///
/// ## Domain
///
/// All matrix entries and vector inputs must be finite, rectangular, and nonempty. Dense input axes are limited to 1..=256. NaN and infinity are rejected. The selected symmetric matrix must be positive definite.
///
/// ## Returns
///
/// A [`CholeskyFactor`] whose unused triangle is deterministically zero.
///
/// ## Errors
///
/// Returns a shared error for invalid domains or shapes, singular systems where applicable, nonconvergence, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private faer algorithms use f64. Ill-conditioned results can lose accuracy. Iterative routines use bounded backend-specific sweep limits. Reuses [`cholesky`]; no uninitialized unused entries are exposed.
///
/// # Examples
///
/// ```
/// let f=sci_linalg::cho_factor(&[vec![4.]],true)?;
/// assert_eq!(sci_linalg::cho_solve(&f,&[8.])?,vec![2.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `cho_factor`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cho_factor.html).
pub fn cho_factor(a: &(impl super::MatrixInput + ?Sized), lower: bool) -> Result<CholeskyFactor> {
    Ok(CholeskyFactor {
        factor: cholesky(a, lower)?,
        lower,
    })
}
/// Solves `A x=b` from `A=L Lᵀ` or `A=Uᵀ U`.
///
/// ## Parameters
///
/// `factor` is owned output from [`cho_factor`]; `b` is one RHS. For manually supplied factors, diagonal entries must be positive.
///
/// ## Domain
///
/// All matrix entries and vector inputs must be finite, rectangular, and nonempty. Dense input axes are limited to 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// The vector x, without recomputing the factorization.
///
/// ## Errors
///
/// Returns a shared error for invalid domains or shapes, singular systems where applicable, nonconvergence, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private faer algorithms use f64. Ill-conditioned results can lose accuracy. Iterative routines use bounded backend-specific sweep limits. O(n²) triangular solves; the factor's opposite triangle is ignored.
///
/// # Examples
///
/// ```
/// let f=sci_linalg::cho_factor(&[vec![4.]],false)?;
/// assert_eq!(sci_linalg::cho_solve(&f,&[8.])?,vec![2.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `cho_solve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cho_solve.html).
pub fn cho_solve(
    factor: &CholeskyFactor,
    b: &(impl super::VectorInput + ?Sized),
) -> Result<Vec<f64>> {
    solve_cholesky(&factor.factor, b, factor.lower)
}
pub(super) fn solve_cholesky(
    factor: &(impl super::MatrixInput + ?Sized),
    b: &(impl super::VectorInput + ?Sized),
    lower: bool,
) -> Result<Vec<f64>> {
    let a = square_ref(factor)?;
    let a = a.as_ref();
    if (0..a.nrows()).any(|i| a[(i, i)] <= 0.0) {
        return Err(Error::InvalidInput("Cholesky diagonal must be positive"));
    }
    let y = solve_triangular(&native(&a), b, lower, false, !lower)?;
    solve_triangular(&native(&a), &y, lower, false, lower)
}

/// Computes the pivoted decomposition `A=P L U`.
///
/// ## Parameters
///
/// `a` is real and may be rectangular. Returns separate factors with SciPy's permutation orientation; permute-l and batching options are omitted.
///
/// ## Domain
///
/// All matrix entries and vector inputs must be finite, rectangular, and nonempty. Dense input axes are limited to 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// [`LuResult`] with P m×m, L m×k, U k×n, k=min(m,n).
///
/// ## Errors
///
/// Returns a shared error for invalid domains or shapes, singular systems where applicable, nonconvergence, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private faer algorithms use f64. Ill-conditioned results can lose accuracy. Iterative routines use bounded backend-specific sweep limits. O(mnk) partial pivoting. Singular matrices can still be decomposed.
///
/// # Examples
///
/// ```
/// let f=sci_linalg::lu(&[vec![2.]])?;
/// assert_eq!(f.u,vec![vec![2.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `lu`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu.html).
pub fn lu(a: &(impl super::MatrixInput + ?Sized)) -> Result<LuResult> {
    let a = dense(a)?;
    let m = a.nrows();
    let d = lu_factors(a)?;
    let perm = &d.permutation;
    let p = Mat::from_fn(m, m, |i, j| f64::from(i == perm[j]));
    Ok(LuResult {
        p: rows(&p)?,
        l: rows(&d.l)?,
        u: rows(&d.u)?,
    })
}
/// Computes an economy QR decomposition `A=Q R`, `Qᵀ Q=I`.
///
/// ## Parameters
///
/// Real rectangular `a`. Economy mode is always used; full/raw/pivoted options are omitted.
///
/// ## Domain
///
/// All matrix entries and vector inputs must be finite, rectangular, and nonempty. Dense input axes are limited to 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// [`QrResult`] with Q m×k and R k×n, k=min(m,n).
///
/// ## Errors
///
/// Returns a shared error for invalid domains or shapes, singular systems where applicable, nonconvergence, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private faer algorithms use f64. Ill-conditioned results can lose accuracy. Iterative routines use bounded backend-specific sweep limits. O(mnk) scaled Householder QR. Column signs are nonunique and may differ from SciPy.
///
/// # Examples
///
/// ```
/// let f=sci_linalg::qr(&[vec![2.]])?;
/// assert!((f.q[0][0]*f.r[0][0]-2.).abs()<1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `qr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.qr.html).
pub fn qr(a: &(impl super::MatrixInput + ?Sized)) -> Result<QrResult> {
    let (a, scale) = scaled(dense(a)?);
    let (q, mut r) = a.qr_factors();
    r *= Scale(scale);
    Ok(QrResult {
        q: rows(&q)?,
        r: rows(&r)?,
    })
}
/// Computes an economy SVD `A=U diag(s) Vh`, with descending nonnegative s.
///
/// ## Parameters
///
/// Real rectangular `a`. Only economy matrices are returned; full matrices, complex inputs, driver and overwrite options are omitted.
///
/// ## Domain
///
/// All matrix entries and vector inputs must be finite, rectangular, and nonempty. Dense input axes are limited to 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// [`SvdResult`] with U m×k, s length k, Vh k×n, k=min(m,n).
///
/// ## Errors
///
/// Returns a shared error for invalid domains or shapes, singular systems where applicable, nonconvergence, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private faer algorithms use f64. Ill-conditioned results can lose accuracy. Iterative routines use bounded backend-specific sweep limits. O(mnk) scaled bidiagonalization. Singular vector signs and bases for repeated values are nonunique.
///
/// # Examples
///
/// ```
/// let f=sci_linalg::svd(&[vec![3.,0.],vec![0.,2.]])?;
/// assert_eq!(f.singular_values,vec![3.,2.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `svd`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html).
pub fn svd(a: &(impl super::MatrixInput + ?Sized)) -> Result<SvdResult> {
    let d = decomposition(dense(a)?, true)?;
    Ok(SvdResult {
        u: rows(d.u.as_ref().unwrap())?,
        singular_values: d.singular_values,
        vh: rows(d.v_t.as_ref().unwrap())?,
    })
}
/// Computes the singular values `s[i]=sqrt(λi(Aᵀ A))`, in descending order.
///
/// ## Parameters
///
/// Real rectangular `a`; no vectors, batching or driver options.
///
/// ## Domain
///
/// All matrix entries and vector inputs must be finite, rectangular, and nonempty. Dense input axes are limited to 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// min(m,n) nonnegative singular values.
///
/// ## Errors
///
/// Returns a shared error for invalid domains or shapes, singular systems where applicable, nonconvergence, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private faer algorithms use f64. Ill-conditioned results can lose accuracy. Iterative routines use bounded backend-specific sweep limits. Uses direct scaled SVD rather than forming Aᵀ A, avoiding squared conditioning.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::svdvals(&[vec![3.,0.],vec![0.,2.]])?,vec![3.,2.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `svdvals`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svdvals.html).
pub fn svdvals(a: &(impl super::MatrixInput + ?Sized)) -> Result<Vec<f64>> {
    Ok(decomposition(dense(a)?, false)?.singular_values)
}
/// Computes the Moore–Penrose inverse `A⁺=V diag(s⁺) Uᵀ`, retaining singular values `s>atol+rtol*smax`.
///
/// ## Parameters
///
/// `atol` is finite nonnegative; optional `rtol` defaults to max(m,n)*machine epsilon. The effective rank is always returned. No batching.
///
/// ## Domain
///
/// All matrix entries and vector inputs must be finite, rectangular, and nonempty. Dense input axes are limited to 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// An n×m pseudoinverse and effective rank, in that order.
///
/// ## Errors
///
/// Returns a shared error for invalid domains or shapes, singular systems where applicable, nonconvergence, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private faer algorithms use f64. Ill-conditioned results can lose accuracy. Iterative routines use bounded backend-specific sweep limits. Rank is sensitive to values near the cutoff; tiny retained singular values can cause overflow.
///
/// # Examples
///
/// ```
/// let (p,rank)=sci_linalg::pinv(&[vec![2.]],0.,None)?;
/// assert_eq!((p,rank),(vec![vec![0.5]],1));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `pinv`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.pinv.html).
pub fn pinv(
    a: &(impl super::MatrixInput + ?Sized),
    atol: f64,
    rtol: Option<f64>,
) -> Result<(Matrix, usize)> {
    let a = dense(a)?;
    let rtol = relative(rtol, a.nrows(), a.ncols())?;
    tolerance(atol)?;
    let mut d = decomposition(a, true)?;
    let cutoff = crate::error::finite(atol + rtol * d.singular_values[0])?;
    let (p, rank) = pseudoinverse(&mut d, cutoff)?;
    Ok((rows(&p)?, rank))
}
/// Minimizes `||A x-b||₂`, choosing the minimum-norm solution when the system is rank deficient.
///
/// ## Parameters
///
/// `a` is m×n; `b` has length m. `cond` is the nonnegative relative singular-value cutoff, default machine epsilon (SciPy gelsd convention). One real RHS.
///
/// ## Domain
///
/// All matrix entries and vector inputs must be finite, rectangular, and nonempty. Dense input axes are limited to 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// [`LeastSquaresResult`]: solution, residuals, rank, and singular values. Residuals contain one sum of squares only for m>n and full column rank.
///
/// ## Errors
///
/// Returns a shared error for invalid domains or shapes, singular systems where applicable, nonconvergence, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private faer algorithms use f64. Ill-conditioned results can lose accuracy. Iterative routines use bounded backend-specific sweep limits. SVD solver; SciPy LAPACK driver selection is omitted. Residual computation may overflow.
///
/// # Examples
///
/// ```
/// let r=sci_linalg::lstsq(&[vec![1.],vec![1.]],&[1.,3.],None)?;
/// assert!((r.solution[0]-2.).abs()<1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `lstsq`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html).
pub fn lstsq(
    a: &(impl super::MatrixInput + ?Sized),
    b: &(impl super::VectorInput + ?Sized),
    cond: Option<f64>,
) -> Result<LeastSquaresResult> {
    let b_input = super::input::vector_slice(b)?;
    let b = b_input.as_ref();
    let a = dense(a)?;
    let b = rhs(b, a.nrows())?;
    let cond = tolerance(cond.unwrap_or(f64::EPSILON))?;
    let mut d = decomposition(a.clone(), true)?;
    let cutoff = crate::error::finite(cond * d.singular_values[0])?;
    let (p, rank) = pseudoinverse(&mut d, cutoff)?;
    let x = p * &b;
    let residuals = if a.nrows() > a.ncols() && rank == a.ncols() {
        vec![crate::error::finite((&a * &x - b).squared_norm_l2())?]
    } else {
        vec![]
    };
    Ok(LeastSquaresResult {
        solution: vector(x)?,
        residuals,
        rank,
        singular_values: d.singular_values,
    })
}
/// Constructs an orthonormal basis Q for `range(A)`, retaining `s>rcond*smax`.
///
/// ## Parameters
///
/// `rcond` defaults to max(m,n)*machine epsilon; must be finite nonnegative. Real rectangular matrix.
///
/// ## Domain
///
/// All matrix entries and vector inputs must be finite, rectangular, and nonempty. Dense input axes are limited to 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// An m×rank matrix whose columns span the range. Rank zero gives m empty rows.
///
/// ## Errors
///
/// Returns a shared error for invalid domains or shapes, singular systems where applicable, nonconvergence, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private faer algorithms use f64. Ill-conditioned results can lose accuracy. Iterative routines use bounded backend-specific sweep limits. Signs and bases in repeated singular subspaces may differ from SciPy; compare projectors.
///
/// # Examples
///
/// ```
/// let q=sci_linalg::orth(&[vec![1.],vec![0.]],None)?;
/// assert_eq!((q.len(),q[0].len()),(2,1));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `orth`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.orth.html).
pub fn orth(a: &(impl super::MatrixInput + ?Sized), rcond: Option<f64>) -> Result<Matrix> {
    let a = dense(a)?;
    let r = relative(rcond, a.nrows(), a.ncols())?;
    let d = decomposition(a, true)?;
    let cutoff = crate::error::finite(r * d.singular_values[0])?;
    let rank = d.singular_values.iter().filter(|&&s| s > cutoff).count();
    rows(&d.u.as_ref().unwrap().subcols(0, rank))
}
/// Constructs an orthonormal basis Z for `null(A)`, where `A Z≈0`, discarding `s<=rcond*smax`.
///
/// ## Parameters
///
/// `rcond` defaults to max(m,n)*machine epsilon and is finite nonnegative.
///
/// ## Domain
///
/// All matrix entries and vector inputs must be finite, rectangular, and nonempty. Dense input axes are limited to 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// An n×(n-rank) matrix. Full rank gives n empty rows.
///
/// ## Errors
///
/// Returns a shared error for invalid domains or shapes, singular systems where applicable, nonconvergence, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private faer algorithms use f64. Ill-conditioned results can lose accuracy. Iterative routines use bounded backend-specific sweep limits. Wide matrices are padded with zero rows before bounded SVD; basis orientation is nonunique.
///
/// # Examples
///
/// ```
/// let z=sci_linalg::null_space(&[vec![1.,0.]],None)?;
/// assert_eq!((z.len(),z[0].len()),(2,1));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `null_space`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.null_space.html).
pub fn null_space(a: &(impl super::MatrixInput + ?Sized), rcond: Option<f64>) -> Result<Matrix> {
    let a = dense(a)?;
    let (m, n) = (a.nrows(), a.ncols());
    let r = relative(rcond, m, n)?;
    // Zero rows supply the complete right basis for a wide matrix.
    let padded = Mat::from_fn(m.max(n), n, |i, j| if i < m { a[(i, j)] } else { 0.0 });
    let d = decomposition(padded, true)?;
    let cutoff = crate::error::finite(r * d.singular_values[0])?;
    let rank = d.singular_values.iter().filter(|&&s| s > cutoff).count();
    rows(&d.v_t.as_ref().unwrap().subrows(rank, n - rank).transpose())
}
/// Computes `A=U P` (right) or `A=P U` (left), where P is positive semidefinite.
///
/// ## Parameters
///
/// `right` chooses P n×n (true) or m×m (false), replacing SciPy's side string. Real rectangular input.
///
/// ## Domain
///
/// All matrix entries and vector inputs must be finite, rectangular, and nonempty. Dense input axes are limited to 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// (U,P), with U m×n. U has orthonormal columns for m>=n, or rows for m<=n.
///
/// ## Errors
///
/// Returns a shared error for invalid domains or shapes, singular systems where applicable, nonconvergence, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private faer algorithms use f64. Ill-conditioned results can lose accuracy. Iterative routines use bounded backend-specific sweep limits. O(mnk) SVD. Rank-deficient U is a nonunique orthogonal extension, following SciPy's convention.
///
/// # Examples
///
/// ```
/// let (u,p)=sci_linalg::polar(&[vec![2.]],true)?;
/// assert!((u[0][0]*p[0][0]-2.).abs()<1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `polar`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.polar.html).
pub fn polar(a: &(impl super::MatrixInput + ?Sized), right: bool) -> Result<(Matrix, Matrix)> {
    let d = decomposition(dense(a)?, true)?;
    let u = d.u.unwrap();
    let vt = d.v_t.unwrap();
    let unitary = &u * &vt;
    let s = Mat::from_fn(d.singular_values.len(), d.singular_values.len(), |i, j| {
        if i == j { d.singular_values[i] } else { 0.0 }
    });
    let p = if right {
        vt.transpose() * s * vt
    } else {
        &u * s * u.transpose()
    };
    Ok((rows(&unitary)?, rows(&p)?))
}
/// Minimizes `||A R-B||F` subject to `Rᵀ R=I`, with no centering or scaling.
///
/// ## Parameters
///
/// `a` and `b` are finite real m×n matrices of equal shape. Reflections are permitted.
///
/// ## Domain
///
/// All matrix entries and vector inputs must be finite, rectangular, and nonempty. Dense input axes are limited to 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// (R,scale), where R is n×n and scale is the sum of singular values of Aᵀ B.
///
/// ## Errors
///
/// Returns a shared error for invalid domains or shapes, singular systems where applicable, nonconvergence, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private faer algorithms use f64. Ill-conditioned results can lose accuracy. Iterative routines use bounded backend-specific sweep limits. Aᵀ B can overflow. For rank-deficient products, the orthogonal minimizer is nonunique.
///
/// # Examples
///
/// ```
/// let (r,s)=sci_linalg::orthogonal_procrustes(&[vec![2.]],&[vec![3.]])?;
/// assert_eq!((r,s),(vec![vec![1.]],6.));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `orthogonal_procrustes`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.orthogonal_procrustes.html).
pub fn orthogonal_procrustes(
    a: &(impl super::MatrixInput + ?Sized),
    b: &(impl super::MatrixInput + ?Sized),
) -> Result<(Matrix, f64)> {
    let a = dense(a)?;
    let b = dense(b)?;
    if a.shape() != b.shape() {
        return Err(Error::DimensionMismatch);
    }
    let product = a.transpose() * b;
    if product.iter().any(|x| !x.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    let d = decomposition(product, true)?;
    Ok((
        rows(&(d.u.unwrap() * d.v_t.unwrap()))?,
        crate::error::finite(d.singular_values.iter().sum())?,
    ))
}
/// Solves the standard symmetric eigenproblem `A V=V diag(w)`, `Vᵀ V=I`.
///
/// ## Parameters
///
/// `lower` selects the input triangle; the other triangle is ignored after checking finiteness. Generalized problems, subsets, complex inputs and driver options are omitted.
///
/// ## Domain
///
/// All matrix entries and vector inputs must be finite, rectangular, and nonempty. Dense input axes are limited to 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// [`EigenResult`], ascending eigenvalues and matching eigenvectors as columns.
///
/// ## Errors
///
/// Returns a shared error for invalid domains or shapes, singular systems where applicable, nonconvergence, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private faer algorithms use f64. Ill-conditioned results can lose accuracy. Iterative routines use bounded backend-specific sweep limits. O(n³); eigenvector signs and repeated-eigenvalue bases are nonunique.
///
/// # Examples
///
/// ```
/// let e=sci_linalg::eigh(&[vec![2.,0.],vec![0.,3.]],true)?;
/// assert!(e.eigenvalues.iter().zip([2.,3.]).all(|(a,b)| (a-b).abs()<1e-14));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `eigh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html).
pub fn eigh(a: &(impl super::MatrixInput + ?Sized), lower: bool) -> Result<EigenResult> {
    eigendecomposition(symmetric(a, lower)?)
}
/// Computes ascending real eigenvalues of the selected symmetric matrix A.
///
/// ## Parameters
///
/// Parameters match [`eigh`]; vectors are omitted.
///
/// ## Domain
///
/// All matrix entries and vector inputs must be finite, rectangular, and nonempty. Dense input axes are limited to 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// A vector of n eigenvalues in ascending order.
///
/// ## Errors
///
/// Returns a shared error for invalid domains or shapes, singular systems where applicable, nonconvergence, or nonfinite output.
///
/// ## Numerical behavior
///
/// Uses the scaled O(n³) faer symmetric eigenvalues-only solver without
/// computing eigenvectors. Backend sweep limits bound iteration. Results can
/// differ from [`eigh`] by roundoff, especially near zero or repeated eigenvalues.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::eigvalsh(&[vec![2.]],true)?,vec![2.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `eigvalsh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigvalsh.html).
pub fn eigvalsh(a: &(impl super::MatrixInput + ?Sized), lower: bool) -> Result<Vec<f64>> {
    eigenvalues(symmetric(a, lower)?)
}
/// Computes `A⁺=V diag(w⁺) Vᵀ`, retaining eigenvalues with `|w|>atol+rtol*max|w|`.
///
/// ## Parameters
///
/// `lower` selects the input triangle. `atol` is nonnegative; `rtol` defaults to n*machine epsilon. Rank is always returned.
///
/// ## Domain
///
/// All matrix entries and vector inputs must be finite, rectangular, and nonempty. Dense input axes are limited to 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// (pseudoinverse,rank), with a square symmetric inverse.
///
/// ## Errors
///
/// Returns a shared error for invalid domains or shapes, singular systems where applicable, nonconvergence, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private faer algorithms use f64. Ill-conditioned results can lose accuracy. Iterative routines use bounded backend-specific sweep limits. Negative eigenvalues are retained by absolute magnitude. Cutoff-near rank is sensitive to roundoff.
///
/// # Examples
///
/// ```
/// let (p,r)=sci_linalg::pinvh(&[vec![-2.]],0.,None,true)?;
/// assert_eq!((p,r),(vec![vec![-0.5]],1));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `pinvh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.pinvh.html).
pub fn pinvh(
    a: &(impl super::MatrixInput + ?Sized),
    atol: f64,
    rtol: Option<f64>,
    lower: bool,
) -> Result<(Matrix, usize)> {
    let d = eigh(a, lower)?;
    let n = d.eigenvalues.len();
    let r = relative(rtol, n, n)?;
    tolerance(atol)?;
    let largest = d.eigenvalues.iter().fold(0.0_f64, |v, &w| v.max(w.abs()));
    let cutoff = crate::error::finite(atol + r * largest)?;
    let mut rank = 0;
    let v = dense(&d.eigenvectors)?;
    let mut scaled = v.clone();
    for (i, &w) in d.eigenvalues.iter().enumerate() {
        let inv = if w.abs() > cutoff {
            rank += 1;
            1.0 / w
        } else {
            0.0
        };
        scaled
            .as_mut()
            .col_mut(i)
            .iter_mut()
            .for_each(|x| *x *= inv);
    }
    Ok((rows(&(scaled * v.transpose()))?, rank))
}
/// Solves `T V=V diag(w)` for symmetric T with diagonal d and off-diagonal e.
///
/// ## Parameters
///
/// `d` has n entries and `e` n-1. No subsets, LAPACK drivers, tolerances or batching.
///
/// ## Domain
///
/// All matrix entries and vector inputs must be finite, rectangular, and nonempty. Dense input axes are limited to 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// [`EigenResult`], ascending eigenvalues and matching vector columns.
///
/// ## Errors
///
/// Returns a shared error for invalid domains or shapes, singular systems where applicable, nonconvergence, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private faer algorithms use f64. Ill-conditioned results can lose accuracy. Iterative routines use bounded backend-specific sweep limits. Uses O(n³) dense symmetric decomposition, rather than a specialized tridiagonal backend.
///
/// # Examples
///
/// ```
/// let e=sci_linalg::eigh_tridiagonal(&[2.,2.],&[1.])?;
/// assert!((e.eigenvalues[0]-1.).abs()<1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `eigh_tridiagonal`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh_tridiagonal.html).
pub fn eigh_tridiagonal(
    d: &(impl super::VectorInput + ?Sized),
    e: &(impl super::VectorInput + ?Sized),
) -> Result<EigenResult> {
    let d_input = super::input::vector_slice(d)?;
    let d = d_input.as_ref();
    let e_input = super::input::vector_slice(e)?;
    let e = e_input.as_ref();
    eigendecomposition(tridiagonal(d, e)?)
}

fn tridiagonal(d: &[f64], e: &[f64]) -> Result<Mat<f64>> {
    let n = d.len();
    if n == 0 || n > MAX_DIM {
        return Err(Error::InvalidInput("tridiagonal order must be 1..=256"));
    }
    if e.len() != n - 1 {
        return Err(Error::DimensionMismatch);
    }
    crate::error::finite_slice(d)?;
    crate::error::finite_slice(e)?;
    Ok(Mat::from_fn(n, n, |i, j| {
        if i == j {
            d[i]
        } else if i.abs_diff(j) == 1 {
            e[i.min(j)]
        } else {
            0.0
        }
    }))
}
/// Computes ascending eigenvalues of T with diagonal d and off-diagonal e.
///
/// ## Parameters
///
/// Parameters match [`eigh_tridiagonal`]; only eigenvalues are returned.
///
/// ## Domain
///
/// All matrix entries and vector inputs must be finite, rectangular, and nonempty. Dense input axes are limited to 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// n eigenvalues in ascending order.
///
/// ## Errors
///
/// Returns a shared error for invalid domains or shapes, singular systems where applicable, nonconvergence, or nonfinite output.
///
/// ## Numerical behavior
///
/// Expands to a dense matrix and uses the scaled O(n³) faer eigenvalues-only
/// solver without computing eigenvectors. Backend sweep limits bound iteration.
/// Results can differ from [`eigh_tridiagonal`] by roundoff, especially near zero
/// or repeated eigenvalues. Tridiagonal inputs retain dense storage and work costs.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::eigvalsh_tridiagonal(&[2.],&[])?,vec![2.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `eigvalsh_tridiagonal`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigvalsh_tridiagonal.html).
pub fn eigvalsh_tridiagonal(
    d: &(impl super::VectorInput + ?Sized),
    e: &(impl super::VectorInput + ?Sized),
) -> Result<Vec<f64>> {
    let d_input = super::input::vector_slice(d)?;
    let d = d_input.as_ref();
    let e_input = super::input::vector_slice(e)?;
    let e = e_input.as_ref();
    eigenvalues(tridiagonal(d, e)?)
}
/// Computes `A=Q H Qᵀ`, where H is upper Hessenberg (zero below the first subdiagonal).
///
/// ## Parameters
///
/// `a` is square and real. Q is always returned; no batching or overwrite option.
///
/// ## Domain
///
/// All matrix entries and vector inputs must be finite, rectangular, and nonempty. Dense input axes are limited to 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// (H,Q), matching SciPy's calc_q=true output ordering.
///
/// ## Errors
///
/// Returns a shared error for invalid domains or shapes, singular systems where applicable, nonconvergence, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private faer algorithms use f64. Ill-conditioned results can lose accuracy. Iterative routines use bounded backend-specific sweep limits. O(n³) generalized Hessenberg reduction of (A,I). The orthogonal basis is nonunique.
///
/// # Examples
///
/// ```
/// let (h,q)=sci_linalg::hessenberg(&[vec![2.]])?;
/// assert_eq!((h,q),(vec![vec![2.]],vec![vec![1.]]));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `hessenberg`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.hessenberg.html).
pub fn hessenberg(a: &(impl super::MatrixInput + ?Sized)) -> Result<(Matrix, Matrix)> {
    let a = square(a)?;
    if a.nrows() == 1 {
        return Ok((rows(&a)?, vec![vec![1.0]]));
    }
    let (a, scale) = scaled(a);
    let (h, q) = real_schur(a, false)?;
    Ok((rows(&(h * Scale(scale)))?, rows(&q)?))
}
/// Computes the real Schur decomposition `A=Z T Zᵀ`, with orthogonal Z and quasi-triangular T.
///
/// ## Parameters
///
/// `a` is square and real. Only unsorted real Schur form is supported; complex output and eigenvalue selection are omitted.
///
/// ## Domain
///
/// All matrix entries and vector inputs must be finite, rectangular, and nonempty. Dense input axes are limited to 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// (T,Z), matching SciPy's ordering. Nonreal eigenvalues occupy real 2×2 blocks.
///
/// ## Errors
///
/// Returns a shared error for invalid domains or shapes, singular systems where applicable, nonconvergence, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private faer algorithms use f64. Ill-conditioned results can lose accuracy. Iterative routines use bounded backend-specific sweep limits. O(n³) generalized real Schur iteration on (A,I), with verified quasi-triangular output. Block ordering and vector signs can differ from SciPy.
///
/// # Examples
///
/// ```
/// let (t,z)=sci_linalg::schur(&[vec![2.]])?;
/// assert_eq!((t,z),(vec![vec![2.]],vec![vec![1.]]));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `schur`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.schur.html).
pub fn schur(a: &(impl super::MatrixInput + ?Sized)) -> Result<(Matrix, Matrix)> {
    let a = square(a)?;
    let (a, scale) = scaled(a);
    let (t, z) = real_schur(a, true)?;
    Ok((rows(&(t * Scale(scale)))?, rows(&z)?))
}
/// Computes the standard eigenvalues λ satisfying `det(A-λI)=0`.
///
/// ## Parameters
///
/// `a` is square and real. Complex outputs use [`sci_complex::Complex`]. Generalized and homogeneous eigenvalue modes are omitted.
///
/// ## Domain
///
/// All matrix entries and vector inputs must be finite, rectangular, and nonempty. Dense input axes are limited to 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// n complex eigenvalues in unspecified backend order, without sorting.
///
/// ## Errors
///
/// Returns a shared error for invalid domains or shapes, singular systems where applicable, nonconvergence, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private faer algorithms use f64. Ill-conditioned results can lose accuracy. Iterative routines use bounded backend-specific sweep limits. Scaled faer eigendecomposition. Eigenvalues of nonnormal matrices can be highly sensitive to roundoff.
///
/// # Examples
///
/// ```
/// let w=sci_linalg::eigvals(&[vec![0.,-1.],vec![1.,0.]])?;
/// assert!(w.iter().all(|z| (z.im.abs()-1.).abs()<1e-14));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `eigvals`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigvals.html).
pub fn eigvals(a: &(impl super::MatrixInput + ?Sized)) -> Result<Vec<crate::complex::Complex>> {
    let a = square(a)?;
    let (a, scale) = scaled(a);
    let values = a
        .eigenvalues()
        .map_err(|_| Error::NumericalFailure("eigensolver did not converge"))?;
    values
        .into_iter()
        .map(|z| {
            let z = z * scale;
            if !z.re.is_finite() || !z.im.is_finite() {
                return Err(Error::NonFiniteEvaluation);
            }
            Ok(crate::complex::owned(z))
        })
        .collect()
}
/// Computes a matrix norm: Frobenius `sqrt(Σaij²)`, absolute row/column sum extrema, singular-value extrema, or nuclear `Σs`.
///
/// ## Parameters
///
/// `order` is a [`MatrixNorm`]. SciPy vector p norms, axes, broadcasting and keepdims are omitted. A one-row matrix with Frobenius order supplies a vector Euclidean norm.
///
/// ## Domain
///
/// All matrix entries and vector inputs must be finite, rectangular, and nonempty. Dense input axes are limited to 1..=256. NaN and infinity are rejected.
///
/// ## Returns
///
/// A nonnegative scalar norm.
///
/// ## Errors
///
/// Returns a shared error for invalid domains or shapes, singular systems where applicable, nonconvergence, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private faer algorithms use f64. Ill-conditioned results can lose accuracy. Iterative routines use bounded backend-specific sweep limits. Frobenius norm uses scaled hypot accumulation. Singular-value orders require SVD; others take O(mn).
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::norm(&[vec![3.,4.]],sci_linalg::MatrixNorm::Frobenius)?,5.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `norm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.norm.html).
pub fn norm(a: &(impl super::MatrixInput + ?Sized), order: MatrixNorm) -> Result<f64> {
    let a_input = dense_ref(a)?;
    let a = a_input.as_ref();
    let value = match order {
        MatrixNorm::Frobenius => a
            .col_iter()
            .flat_map(|c| c.iter())
            .fold(0.0_f64, |v, &x| v.hypot(x)),
        MatrixNorm::One | MatrixNorm::NegativeOne => {
            let sums = (0..a.ncols()).map(|j| a.col(j).iter().map(|x| x.abs()).sum::<f64>());
            if order == MatrixNorm::One {
                sums.fold(0.0, f64::max)
            } else {
                sums.fold(f64::INFINITY, f64::min)
            }
        }
        MatrixNorm::Infinity | MatrixNorm::NegativeInfinity => {
            let sums = (0..a.nrows()).map(|i| a.row(i).iter().map(|x| x.abs()).sum::<f64>());
            if order == MatrixNorm::Infinity {
                sums.fold(0.0, f64::max)
            } else {
                sums.fold(f64::INFINITY, f64::min)
            }
        }
        MatrixNorm::Spectral | MatrixNorm::SmallestSingular | MatrixNorm::Nuclear => {
            let d = decomposition(a_input.into_owned(), false)?;
            match order {
                MatrixNorm::Spectral => d.singular_values[0],
                MatrixNorm::SmallestSingular => d.singular_values[d.singular_values.len() - 1],
                _ => d.singular_values.iter().sum(),
            }
        }
    };
    crate::error::finite(value)
}
