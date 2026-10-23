//! General eigenproblems and symmetric indefinite factorizations.
use super::{Matrix, backend::*};
use crate::{
    Error, Result,
    complex::{Complex, owned},
};
use faer::Mat;

/// Row-major complex matrix using shared Sci scalars.
pub type ComplexMatrix = Vec<Vec<Complex>>;
/// Right eigensystem satisfying `A V = V diag(w)`.
#[derive(Clone, Debug)]
pub struct GeneralEigenResult {
    /// Eigenvalues, in backend order.
    pub eigenvalues: Vec<Complex>,
    /// Unit Euclidean norm right eigenvectors, stored in columns.
    pub eigenvectors: ComplexMatrix,
}
/// Symmetric indefinite factorization `A = outer D outerᵀ`.
#[derive(Clone, Debug)]
pub struct LdlResult {
    /// Permuted unit triangular outer factor.
    pub outer: Matrix,
    /// Symmetric block diagonal factor with blocks of order one or two.
    pub d: Matrix,
    /// Row indices making `outer[permutation, :]` triangular.
    pub permutation: Vec<usize>,
}
/// Computes right eigenpairs `A v = λ v` for a real square matrix.
///
/// ## Parameters
///
/// `a` is row-major. Eigenvectors are columns; eigenvalues are unordered.
/// Generalized and left eigenvectors are omitted.
///
/// ## Domain
///
/// Finite, nonempty square matrices of order at most 256.
/// Complex conjugate pairs are supported; NaN and infinity are rejected.
///
/// ## Returns
///
/// Owned complex eigenvalues and unit-norm right eigenvectors.
///
/// ## Errors
///
/// Invalid shape, nonfinite data/output, or failed eigenvalue iteration.
///
/// ## Numerical behavior
///
/// Uses the private faer real Schur eigensolver with input scaling.
/// The backend bounds iteration by matrix order. Repeated eigenvectors are
/// nonunique and defective or nearly defective systems can be ill-conditioned.
///
/// # Examples
///
/// ```
/// let e = sci_linalg::eig(&[vec![0., -1.], vec![1., 0.]])?;
/// assert_eq!(e.eigenvalues.len(), 2);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `eig`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eig.html).
pub fn eig(a: &(impl super::MatrixInput + ?Sized)) -> Result<GeneralEigenResult> {
    let (a, scale) = scaled(square(a)?);
    let n = a.nrows();
    let f = faer::Mat::from_fn(n, n, |i, j| a[(i, j)])
        .eigen()
        .map_err(|_| Error::NumericalFailure("eigensolver did not converge"))?;
    let mut eigenvectors = vec![vec![Complex::default(); n]; n];
    let mut eigenvalues = Vec::with_capacity(n);
    for (j, value) in (0..n).map(|j| (j, f.S()[j] * scale)) {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(Error::NonFiniteEvaluation);
        }
        eigenvalues.push(owned(value));
        let length = (0..n).fold(0.0_f64, |s, i| s.hypot(f.U()[(i, j)].norm()));
        if !length.is_finite() || length == 0.0 {
            return Err(Error::NumericalFailure("invalid eigenvector"));
        }
        for (i, row) in eigenvectors.iter_mut().enumerate() {
            row[j] = owned(f.U()[(i, j)] / length);
        }
    }
    Ok(GeneralEigenResult {
        eigenvalues,
        eigenvectors,
    })
}
/// Factors a symmetric matrix as `A = L D Lᵀ` (or `U D Uᵀ`).
///
/// ## Parameters
///
/// `lower` selects the input triangle and triangular outer-factor orientation.
/// The opposite finite triangle is ignored. Permuting outer-factor rows by
/// `permutation` produces a unit lower/upper triangular matrix.
///
/// ## Domain
///
/// Finite, nonempty square matrices of order at most 256. Singular and
/// indefinite matrices are accepted. NaN and infinity are rejected.
///
/// ## Returns
///
/// An owned outer factor, block diagonal D, and final row permutation.
/// Individual pivots and signs may differ from SciPy while reconstructing A.
///
/// ## Errors
///
/// Invalid shape, nonfinite input, or numerical overflow.
///
/// ## Numerical behavior
///
/// Private faer Bunch–Kaufman pivoting, O(n³) work and O(n²) memory.
/// Scaling protects large inputs. Two-by-two pivots handle zero diagonal entries.
///
/// # Examples
///
/// ```
/// let f = sci_linalg::ldl(&[vec![0., 1.], vec![1., 0.]], true)?;
/// assert_eq!(f.d.len(), 2);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `ldl`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.ldl.html).
pub fn ldl(a: &(impl super::MatrixInput + ?Sized), lower: bool) -> Result<LdlResult> {
    let mut a = symmetric(a, lower)?;
    let n = a.nrows();
    // Reversing both axes turns an upper factor into a lower one.
    if !lower {
        a = Mat::from_fn(n, n, |i, j| a[(n - 1 - i, n - 1 - j)]);
    }
    let (a, scale) = scaled(a);
    let f = faer::Mat::from_fn(n, n, |i, j| a[(i, j)]).lblt(faer::Side::Lower);
    let permutation = f.P().arrays().0.to_vec();
    let mut outer = vec![vec![0.; n]; n];
    let mut d = vec![vec![0.; n]; n];
    for i in 0..n {
        for (j, value) in outer[permutation[i]].iter_mut().enumerate() {
            *value = f.L()[(i, j)];
        }
        d[i][i] = f.B_diag()[i] * scale;
        if i + 1 < n {
            d[i + 1][i] = f.B_subdiag()[i] * scale;
            d[i][i + 1] = d[i + 1][i];
        }
    }
    let permutation = if lower {
        permutation
    } else {
        outer = (0..n)
            .map(|i| (0..n).map(|j| outer[n - 1 - i][n - 1 - j]).collect())
            .collect();
        d = (0..n)
            .map(|i| (0..n).map(|j| d[n - 1 - i][n - 1 - j]).collect())
            .collect();
        permutation.into_iter().rev().map(|i| n - 1 - i).collect()
    };
    Ok(LdlResult {
        outer: output(outer)?,
        d: output(d)?,
        permutation,
    })
}
/// Converts conjugate eigenpairs to a real block diagonal eigensystem.
///
/// ## Parameters
///
/// `w` has n complex eigenvalues; `v` is n × n with eigenvectors in columns.
/// Nonreal eigenvalues must form adjacent conjugate pairs. Real eigenvectors
/// are required for real eigenvalues. Pairs may start with either imaginary sign.
///
/// ## Domain
///
/// Finite values, n in 1..=256. Conjugacy is checked with relative tolerance
/// 1e-10 and absolute tolerance 1e-12; NaN/infinity are rejected.
///
/// ## Returns
///
/// Tuple (D, V), satisfying `A V = V D` when input is an eigensystem of A.
/// Each complex pair yields [[re, im],[-im,re]] and columns (-Im(v), Re(v)).
///
/// ## Errors
///
/// Invalid shape, ordering, conjugacy, or nonfinite data.
///
/// ## Numerical behavior
///
/// Native linear-time scalar conversion, O(n²) storage. Conjugate-pair
/// roundoff is averaged. No eigenpair residual or independence is checked.
///
/// # Examples
///
/// ```
/// use sci_complex::Complex;
/// let (d,v) = sci_linalg::cdf2rdf(&[Complex {re:2.,im:0.}],
///     &[vec![Complex {re:1.,im:0.}]])?;
/// assert_eq!(d,vec![vec![2.]]);
/// assert_eq!(v,vec![vec![1.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `cdf2rdf`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cdf2rdf.html).
pub fn cdf2rdf(w: &[Complex], v: &[Vec<Complex>]) -> Result<(Matrix, Matrix)> {
    let v = super::principal::complex_dense(v)?;
    let n = w.len();
    if n != v.nrows() || n != v.ncols() {
        return Err(Error::DimensionMismatch);
    }
    if w.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
        return Err(Error::InvalidInput("eigenvalues must be finite"));
    }
    let close = |a: num_complex::Complex64, b: num_complex::Complex64| {
        (a - b).norm() <= 1e-12 + 1e-10 * b.norm()
    };
    let mut d = vec![vec![0.; n]; n];
    let mut r = vec![vec![0.; n]; n];
    let mut j = 0;
    while j < n {
        d[j][j] = w[j].re;
        if w[j].im == 0. {
            for i in 0..n {
                if !close(v[(i, j)], num_complex::Complex64::new(v[(i, j)].re, 0.)) {
                    return Err(Error::InvalidInput(
                        "real eigenvalue requires real eigenvector",
                    ));
                }
                r[i][j] = v[(i, j)].re;
            }
            j += 1;
        } else {
            if j + 1 >= n
                || !close(
                    crate::complex::native(w[j]),
                    crate::complex::native(w[j + 1]).conj(),
                )
            {
                return Err(Error::InvalidInput(
                    "eigenvalues must have adjacent conjugate pairs",
                ));
            }
            d[j + 1][j + 1] = w[j + 1].re;
            d[j][j + 1] = w[j].im;
            d[j + 1][j] = w[j + 1].im;
            for i in 0..n {
                if !close(v[(i, j)], v[(i, j + 1)].conj()) {
                    return Err(Error::InvalidInput(
                        "eigenvector columns must be conjugates",
                    ));
                }
                r[i][j] = -(v[(i, j)].im - v[(i, j + 1)].im) / 2.;
                r[i][j + 1] = (v[(i, j)].re + v[(i, j + 1)].re) / 2.;
            }
            j += 2;
        }
    }
    Ok((output(d)?, output(r)?))
}
/// Applies CountSketch, `Y[h(i),j] += signs[i] A[i,j]`.
///
/// ## Parameters
///
/// `rows` is the output row count. `buckets` and `signs` contain one
/// explicit row index and ±1 sign per input row. Caller-supplied draws
/// replace SciPy rng objects and make the transform reproducible.
///
/// ## Domain
///
/// Finite input with axes in 1..=256; rows in 1..=256.
/// Each bucket is less than rows; each sign is exactly ±1.
///
/// ## Returns
///
/// A rows × n dense sketch of the input matrix.
///
/// ## Errors
///
/// Invalid shape, buckets, signs, work bound, or nonfinite sum.
///
/// ## Numerical behavior
///
/// Native O(m n) accumulation. Norm-preservation probability requires
/// independent uniform buckets and signs; arbitrary supplied draws have
/// no probabilistic guarantee. Collisions accumulate in input-row order.
///
/// # Examples
///
/// ```
/// let y = sci_linalg::clarkson_woodruff_transform(
///     &[vec![1.,2.],vec![3.,4.]],1,&[0,0],&[1.,-1.])?;
/// assert_eq!(y,vec![vec![-2.,-2.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `clarkson_woodruff_transform`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.clarkson_woodruff_transform.html).
pub fn clarkson_woodruff_transform(
    a: &(impl super::MatrixInput + ?Sized),
    rows: usize,
    buckets: &[usize],
    signs: &(impl super::VectorInput + ?Sized),
) -> Result<Matrix> {
    let signs_input = super::input::vector_slice(signs)?;
    let signs = signs_input.as_ref();
    let a = dense(a)?;
    if rows == 0 || rows > MAX_DIM || buckets.len() != a.nrows() || signs.len() != a.nrows() {
        return Err(Error::DimensionMismatch);
    }
    if buckets.iter().any(|&i| i >= rows) || signs.iter().any(|&s| s != 1. && s != -1.) {
        return Err(Error::InvalidInput("invalid sketch buckets or signs"));
    }
    let mut y = vec![vec![0.; a.ncols()]; rows];
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            y[buckets[i]][j] += signs[i] * a[(i, j)];
        }
    }
    output(y)
}
/// Balances a matrix by a similarity transform `B = T⁻¹ A T`.
///
/// ## Parameters
///
/// `permute` moves isolated rows/columns to the matrix edges; `scale`
/// uses powers of two to balance off-diagonal row/column one-norms.
/// Returns the full transform, corresponding to SciPy separate=false.
///
/// ## Domain
///
/// Finite nonempty square matrices of order at most 256. Zero and singular
/// matrices are allowed; NaN/infinity are rejected.
///
/// ## Returns
///
/// Tuple (balanced matrix, transformation matrix). Permutation/order and
/// scale choices may differ from LAPACK while preserving similarity.
///
/// ## Errors
///
/// Invalid input, scaling overflow, or failure to converge within 256 sweeps.
///
/// ## Numerical behavior
///
/// Native permutation and radix-two balancing. Scaling is accepted only
/// when the off-diagonal one-norm sum improves by at least 5%.
/// Each scale adjustment is bounded at 512 powers of two.
/// Balancing can worsen eigenvector conditioning for some matrices.
///
/// # Examples
///
/// ```
/// let (b,t) = sci_linalg::matrix_balance(&[vec![1.,100.],vec![0.01,2.]],false,true)?;
/// assert_eq!(b.len(),t.len());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `matrix_balance`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.matrix_balance.html).
pub fn matrix_balance(
    a: &(impl super::MatrixInput + ?Sized),
    permute: bool,
    scale: bool,
) -> Result<(Matrix, Matrix)> {
    let mut a = square(a)?;
    let n = a.nrows();
    let mut t = Mat::identity(n, n);
    let (mut low, mut high) = (0, n);
    if permute {
        loop {
            if high <= low + 1 {
                break;
            }
            if let Some(i) = (low..high)
                .rev()
                .find(|&i| (low..high).all(|j| i == j || a[(i, j)] == 0.))
            {
                high -= 1;
                a.swap_rows(i, high);
                a.swap_columns(i, high);
                t.swap_columns(i, high);
                continue;
            }
            if let Some(i) = (low..high).find(|&i| (low..high).all(|j| i == j || a[(j, i)] == 0.)) {
                a.swap_rows(i, low);
                a.swap_columns(i, low);
                t.swap_columns(i, low);
                low += 1;
                continue;
            }
            break;
        }
    }
    if scale {
        let mut converged = false;
        for _ in 0..256 {
            let mut changed = false;
            for i in low..high {
                let row = (low..high)
                    .filter(|&j| j != i)
                    .map(|j| a[(i, j)].abs())
                    .sum::<f64>();
                let col = (low..high)
                    .filter(|&j| j != i)
                    .map(|j| a[(j, i)].abs())
                    .sum::<f64>();
                if row == 0. || col == 0. {
                    continue;
                }
                if !row.is_finite() || !col.is_finite() {
                    return Err(Error::NonFiniteEvaluation);
                }
                let exponent = ((row.log2() - col.log2()) / 2.).round().clamp(-512., 512.) as i32;
                let factor = 2.0_f64.powi(exponent);
                if row / factor + col * factor >= 0.95 * (row + col) {
                    continue;
                }
                for j in 0..n {
                    if j != i {
                        a[(i, j)] /= factor;
                        a[(j, i)] *= factor;
                    }
                    t[(j, i)] *= factor;
                }
                rows(&a)?;
                rows(&t)?;
                if t.col(i).norm_max() == 0. {
                    return Err(Error::NumericalFailure(
                        "balancing transformation underflowed",
                    ));
                }
                changed = true;
            }
            if !changed {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(Error::NumericalFailure(
                "matrix balance sweep limit reached",
            ));
        }
    }
    Ok((rows(&a)?, rows(&t)?))
}
