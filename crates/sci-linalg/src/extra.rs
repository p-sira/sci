//! Banded adapters, reusable factors, and matrix equations.
// SciPy-derived layouts/identities retain their BSD notice in LICENSES/scipy-linalg.txt.
use super::backend::*;
use super::dense::*;
use crate::{Error, Result};
use faer::Mat;
/// Reusable square LU factorization with a final row permutation.
#[derive(Clone, Debug)]
pub struct LuFactor {
    /// U on and above the diagonal, strict lower L below it; L has unit diagonal.
    pub packed: Matrix,
    /// Final zero-based permutation: row i of Pᵀ A is row `row_permutation[i]` of A.
    /// This replaces SciPy's step-by-step pivot array.
    pub row_permutation: Vec<usize>,
}
fn band(
    ab: &(impl super::MatrixInput + ?Sized),
    l: usize,
    u: usize,
) -> Result<super::input::FaerInput<'static>> {
    let ab = dense_ref(ab)?;
    let ab = ab.as_ref();
    let n = ab.ncols();
    if l >= n || u >= n || l.checked_add(u).and_then(|v| v.checked_add(1)) != Some(ab.nrows()) {
        return Err(Error::DimensionMismatch);
    }
    Ok(super::input::FaerInput::Owned(Mat::from_fn(
        n,
        n,
        |i, j| {
            if i + u >= j && j + l >= i {
                ab[(u + i - j, j)]
            } else {
                0.0
            }
        },
    )))
}
fn symmetric_band(
    ab: &(impl super::MatrixInput + ?Sized),
    lower: bool,
) -> Result<super::input::FaerInput<'static>> {
    let ab = dense_ref(ab)?;
    let ab = ab.as_ref();
    let (k, n) = (ab.nrows() - 1, ab.ncols());
    if k >= n {
        return Err(Error::DimensionMismatch);
    }
    Ok(super::input::FaerInput::Owned(Mat::from_fn(
        n,
        n,
        |i, j| {
            let (r, c) = if lower {
                (i.max(j), i.min(j))
            } else {
                (i.min(j), i.max(j))
            };
            if r.abs_diff(c) > k {
                0.0
            } else if lower {
                ab[(r - c, c)]
            } else {
                ab[(k + r - c, c)]
            }
        },
    )))
}
fn triangular_band(
    ab: &(impl super::MatrixInput + ?Sized),
    lower: bool,
) -> Result<super::input::FaerInput<'static>> {
    let mut a = symmetric_band(ab, lower)?.into_owned();
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            if (lower && i < j) || (!lower && i > j) {
                a[(i, j)] = 0.0;
            }
        }
    }
    Ok(super::input::FaerInput::Owned(a))
}

/// Factors `Pᵀ A=L U`, storing U and strict lower L in one matrix.
///
/// ## Parameters
///
/// `a` is square. [`LuFactor`] uses a final row permutation instead of SciPy's stepwise pivot array; rectangular factorization is available through [`lu`].
///
/// ## Domain
///
/// Numerical inputs must be finite, rectangular, and nonempty. Input axes are limited to 1..=256.
///
/// ## Returns
///
/// A reusable [`LuFactor`]. Singular matrices may be factored, but [`lu_solve`] fails.
///
/// ## Errors
///
/// Returns a shared error for invalid shapes, parameters, nonfinite inputs or outputs, singular systems, or failed convergence where applicable.
///
/// ## Numerical behavior
///
/// Uses f64 and the shared private backend, with bounded backend-specific SVD/eigen sweep limits. Ill-conditioned inputs can lose accuracy. O(n³) partial-pivoted LU; no singularity warning is emitted at factorization.
///
/// # Examples
///
/// ```
/// let f=sci_linalg::lu_factor(&[vec![2.]])?;
/// assert_eq!(sci_linalg::lu_solve(&f,&[6.],false)?,vec![3.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `lu_factor`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu_factor.html).
pub fn lu_factor(a: &(impl super::MatrixInput + ?Sized)) -> Result<LuFactor> {
    let a = square(a)?;
    let n = a.nrows();
    let d = lu_factors(a)?;
    let perm = &d.permutation;
    let packed = Mat::from_fn(n, n, |i, j| if i > j { d.l[(i, j)] } else { d.u[(i, j)] });
    Ok(LuFactor {
        packed: rows(&packed)?,
        row_permutation: perm.to_vec(),
    })
}
/// Solves `A x=b` or `Aᵀ x=b` from a reusable `Pᵀ A=L U` factorization.
///
/// ## Parameters
///
/// `factor` comes from [`lu_factor`]; `transpose` selects Aᵀ. One RHS of length n; manually supplied row permutations must be bijections.
///
/// ## Domain
///
/// Numerical inputs must be finite, rectangular, and nonempty. Input axes are limited to 1..=256.
///
/// ## Returns
///
/// The solution vector x.
///
/// ## Errors
///
/// Returns a shared error for invalid shapes, parameters, nonfinite inputs or outputs, singular systems, or failed convergence where applicable.
///
/// ## Numerical behavior
///
/// Uses f64 and the shared private backend, with bounded backend-specific SVD/eigen sweep limits. Ill-conditioned inputs can lose accuracy. O(n²) substitution with validation; no factorization is repeated.
///
/// # Examples
///
/// ```
/// let f=sci_linalg::lu_factor(&[vec![2.]])?;
/// assert_eq!(sci_linalg::lu_solve(&f,&[6.],true)?,vec![3.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `lu_solve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu_solve.html).
pub fn lu_solve(
    factor: &LuFactor,
    b: &(impl super::VectorInput + ?Sized),
    transpose: bool,
) -> Result<Vec<f64>> {
    let b_input = super::input::vector_slice(b)?;
    let b = b_input.as_ref();
    let a_input = square_ref(&factor.packed)?;
    let a = a_input.as_ref();
    let n = a.nrows();
    if factor.row_permutation.len() != n || b.len() != n {
        return Err(Error::DimensionMismatch);
    }
    crate::error::finite_slice(b)?;
    let mut seen = vec![false; n];
    for &p in &factor.row_permutation {
        if p >= n || seen[p] {
            return Err(Error::InvalidInput(
                "LU row permutation must be a bijection",
            ));
        }
        seen[p] = true;
    }
    let finite = |x: &[f64]| {
        if x.iter().all(|v| v.is_finite()) {
            Ok(())
        } else {
            Err(Error::NonFiniteEvaluation)
        }
    };
    if transpose {
        let mut z = b.to_vec();
        substitute(a, &mut z, false, false, true)?;
        finite(&z)?;
        substitute(a, &mut z, true, true, true)?;
        finite(&z)?;
        let mut x = vec![0.0; n];
        for (i, &p) in factor.row_permutation.iter().enumerate() {
            x[p] = z[i];
        }
        Ok(x)
    } else {
        let mut x: Vec<_> = factor.row_permutation.iter().map(|&i| b[i]).collect();
        substitute(a, &mut x, true, true, false)?;
        finite(&x)?;
        substitute(a, &mut x, false, false, false)?;
        finite(&x)?;
        Ok(x)
    }
}
/// Solves `A x=b` using general band storage `ab[u+i-j,j]=A[i,j]`.
///
/// ## Parameters
///
/// `l` and `u` are lower/upper bandwidths, each less than n. `ab` has l+u+1 rows and n columns. Unused padding is ignored after finite validation. One RHS.
///
/// ## Domain
///
/// Numerical inputs must be finite, rectangular, and nonempty. Input axes are limited to 1..=256.
///
/// ## Returns
///
/// The vector x of length n.
///
/// ## Errors
///
/// Returns a shared error for invalid shapes, parameters, nonfinite inputs or outputs, singular systems, or failed convergence where applicable.
///
/// ## Numerical behavior
///
/// Uses f64 and the shared private backend, with bounded backend-specific SVD/eigen sweep limits. Ill-conditioned inputs can lose accuracy. Expands the bounded band into a dense matrix and uses O(n³) LU, without SciPy's banded complexity.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::solve_banded(0,0,&[vec![2.,4.]],&[6.,8.])?,vec![3.,2.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `solve_banded`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_banded.html).
pub fn solve_banded(
    l: usize,
    u: usize,
    ab: &(impl super::MatrixInput + ?Sized),
    b: &(impl super::VectorInput + ?Sized),
) -> Result<Vec<f64>> {
    let b_input = super::input::vector_slice(b)?;
    let b = b_input.as_ref();
    solve(&band(ab, l, u)?, b)
}
/// Solves `A x=b` for a symmetric positive-definite band matrix.
///
/// ## Parameters
///
/// `ab` is (k+1)×n. Lower: `ab[i-j,j]=A[i,j]` for i>=j. Upper: `ab[k+i-j,j]=A[i,j]` for i<=j. `lower` selects layout, k<n.
///
/// ## Domain
///
/// Numerical inputs must be finite, rectangular, and nonempty. Input axes are limited to 1..=256. The reconstructed symmetric matrix must be positive definite.
///
/// ## Returns
///
/// The vector x of length n.
///
/// ## Errors
///
/// Returns a shared error for invalid shapes, parameters, nonfinite inputs or outputs, singular systems, or failed convergence where applicable.
///
/// ## Numerical behavior
///
/// Uses f64 and the shared private backend, with bounded backend-specific SVD/eigen sweep limits. Ill-conditioned inputs can lose accuracy. Dense expansion and O(n³) Cholesky replace a specialized banded solve.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::solveh_banded(&[vec![4.,9.]],&[8.,18.],true)?,vec![2.,2.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `solveh_banded`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solveh_banded.html).
pub fn solveh_banded(
    ab: &(impl super::MatrixInput + ?Sized),
    b: &(impl super::VectorInput + ?Sized),
    lower: bool,
) -> Result<Vec<f64>> {
    let b_input = super::input::vector_slice(b)?;
    let b = b_input.as_ref();
    cho_solve(&cho_factor(&symmetric_band(ab, lower)?, lower)?, b)
}
/// Factors a symmetric positive-definite band matrix `A=L Lᵀ` or `A=Uᵀ U`.
///
/// ## Parameters
///
/// Input band layout and `lower` match [`solveh_banded`]. Padding entries are ignored after finite validation.
///
/// ## Domain
///
/// Numerical inputs must be finite, rectangular, and nonempty. Input axes are limited to 1..=256. The selected matrix must be positive definite.
///
/// ## Returns
///
/// The Cholesky factor in the same (k+1)×n band layout; unused padding is zero.
///
/// ## Errors
///
/// Returns a shared error for invalid shapes, parameters, nonfinite inputs or outputs, singular systems, or failed convergence where applicable.
///
/// ## Numerical behavior
///
/// Uses f64 and the shared private backend, with bounded backend-specific SVD/eigen sweep limits. Ill-conditioned inputs can lose accuracy. O(n³) dense Cholesky, without SciPy's banded complexity.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::cholesky_banded(&[vec![4.,9.]],true)?,vec![vec![2.,3.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `cholesky_banded`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cholesky_banded.html).
pub fn cholesky_banded(ab: &(impl super::MatrixInput + ?Sized), lower: bool) -> Result<Matrix> {
    let factor = cholesky(&symmetric_band(ab, lower)?, lower)?;
    let (rows, n) = shape(ab)?;
    let k = rows - 1;
    Ok((0..=k)
        .map(|r| {
            (0..n)
                .map(|j| {
                    if lower {
                        if j + r < n { factor[j + r][j] } else { 0.0 }
                    } else if j + r >= k {
                        factor[j + r - k][j]
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect())
}
/// Solves `A x=b` from a banded Cholesky factor.
///
/// ## Parameters
///
/// `cb` is the output of [`cholesky_banded`]; `lower` selects layout, and `b` has n entries. One real RHS.
///
/// ## Domain
///
/// Numerical inputs must be finite, rectangular, and nonempty. Input axes are limited to 1..=256. The supplied factor must have positive diagonal.
///
/// ## Returns
///
/// The vector x of length n.
///
/// ## Errors
///
/// Returns a shared error for invalid shapes, parameters, nonfinite inputs or outputs, singular systems, or failed convergence where applicable.
///
/// ## Numerical behavior
///
/// Uses f64 and the shared private backend, with bounded backend-specific SVD/eigen sweep limits. Ill-conditioned inputs can lose accuracy. Dense triangular expansion and O(n²) substitution, without refactorization.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::cho_solve_banded(&[vec![2.,3.]],&[8.,18.],true)?,vec![2.,2.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `cho_solve_banded`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cho_solve_banded.html).
pub fn cho_solve_banded(
    cb: &(impl super::MatrixInput + ?Sized),
    b: &(impl super::VectorInput + ?Sized),
    lower: bool,
) -> Result<Vec<f64>> {
    solve_cholesky(&triangular_band(cb, lower)?, b, lower)
}
/// Solves `A V=V diag(w)` for a real symmetric band matrix.
///
/// ## Parameters
///
/// `ab` and `lower` use [`solveh_banded`] layout. Computes all eigenvalues/vectors; subsets, complex inputs, and driver options are omitted.
///
/// ## Domain
///
/// Numerical inputs must be finite, rectangular, and nonempty. Input axes are limited to 1..=256.
///
/// ## Returns
///
/// [`EigenResult`] with ascending eigenvalues and matching column eigenvectors.
///
/// ## Errors
///
/// Returns a shared error for invalid shapes, parameters, nonfinite inputs or outputs, singular systems, or failed convergence where applicable.
///
/// ## Numerical behavior
///
/// Uses f64 and the shared private backend, with bounded backend-specific SVD/eigen sweep limits. Ill-conditioned inputs can lose accuracy. Dense expansion and O(n³) symmetric eigensolver.
///
/// # Examples
///
/// ```
/// let e=sci_linalg::eig_banded(&[vec![2.,3.]],true)?;
/// assert!(e.eigenvalues.iter().zip([2.,3.]).all(|(a,b)| (a-b).abs()<1e-14));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `eig_banded`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eig_banded.html).
pub fn eig_banded(ab: &(impl super::MatrixInput + ?Sized), lower: bool) -> Result<EigenResult> {
    eigh(&symmetric_band(ab, lower)?, lower)
}
/// Computes ascending eigenvalues of a symmetric band matrix.
///
/// ## Parameters
///
/// Arguments match [`eig_banded`]; vectors are omitted from the return.
///
/// ## Domain
///
/// Numerical inputs must be finite, rectangular, and nonempty. Input axes are limited to 1..=256.
///
/// ## Returns
///
/// All n real eigenvalues in ascending order.
///
/// ## Errors
///
/// Returns a shared error for invalid shapes, parameters, nonfinite inputs or outputs, singular systems, or failed convergence where applicable.
///
/// ## Numerical behavior
///
/// Expands to a dense matrix and uses the scaled O(n³) faer eigenvalues-only
/// solver without computing eigenvectors. Backend sweep limits bound iteration.
/// Results can differ from [`eig_banded`] by roundoff, especially near zero or
/// repeated eigenvalues. Banded inputs retain dense storage and work costs.
///
/// # Examples
///
/// ```
/// let values=sci_linalg::eigvals_banded(&[vec![2.,3.]],true)?;
/// assert!(values.iter().zip([2.,3.]).all(|(a,b)| (a-b).abs()<1e-14));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `eigvals_banded`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigvals_banded.html).
pub fn eigvals_banded(ab: &(impl super::MatrixInput + ?Sized), lower: bool) -> Result<Vec<f64>> {
    eigvalsh(&symmetric_band(ab, lower)?, lower)
}
/// Computes principal angles between column spaces: `cos(θi)=si(QAᵀ QB)`.
///
/// ## Parameters
///
/// `a` and `b` have the same number of rows; ranks use [`orth`]'s default cutoff. Real inputs only.
///
/// ## Domain
///
/// Numerical inputs must be finite, rectangular, and nonempty. Input axes are limited to 1..=256.
///
/// ## Returns
///
/// min(rank(a),rank(b)) angles in descending order, in radians in [0,π/2]. A zero-rank space returns an empty vector.
///
/// ## Errors
///
/// Returns a shared error for invalid shapes, parameters, nonfinite inputs or outputs, singular systems, or failed convergence where applicable.
///
/// ## Numerical behavior
///
/// Uses f64 and the shared private backend, with bounded backend-specific SVD/eigen sweep limits. Ill-conditioned inputs can lose accuracy. Combines residual sine and cosine SVDs to preserve accuracy near zero and π/2, following Knyazev and Argentati (2002), DOI 10.1137/S1064827500377332.
///
/// # Examples
///
/// ```
/// let angles=sci_linalg::subspace_angles(&[vec![1.],vec![0.]],&[vec![0.],vec![1.]])?;
/// assert!((angles[0]-std::f64::consts::FRAC_PI_2).abs()<1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `subspace_angles`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.subspace_angles.html).
pub fn subspace_angles(
    a: &(impl super::MatrixInput + ?Sized),
    b: &(impl super::MatrixInput + ?Sized),
) -> Result<Vec<f64>> {
    let qa = orth(a, None)?;
    let qb = orth(b, None)?;
    if qa.len() != qb.len() {
        return Err(Error::DimensionMismatch);
    }
    if qa[0].is_empty() || qb[0].is_empty() {
        return Ok(vec![]);
    }
    let qa = dense(&qa)?;
    let qb = dense(&qb)?;
    let c = qa.transpose() * &qb;
    let cos = svdvals(&native(&c))?;
    let residual = if qa.ncols() >= qb.ncols() {
        qb - &qa * c
    } else {
        qa - &qb * c.transpose()
    };
    let sin = svdvals(&native(&residual))?;
    Ok(cos
        .iter()
        .rev()
        .zip(sin)
        .map(|(&c, s)| {
            if c * c >= 0.5 {
                s.clamp(0.0, 1.0).asin()
            } else {
                c.clamp(0.0, 1.0).acos()
            }
        })
        .collect())
}
/// Solves the Sylvester equation `A X+X B=Q`.
///
/// ## Parameters
///
/// `a` is m×m, `b` n×n, and `q` m×n. Only real unbatched matrices.
///
/// ## Domain
///
/// Numerical inputs must be finite, rectangular, and nonempty. Input axes are limited to 1..=256. Requires m*n<=256 and a unique solution (no eigenvalues of A and -B in common).
///
/// ## Returns
///
/// The m×n solution matrix X.
///
/// ## Errors
///
/// Returns a shared error for invalid shapes, parameters, nonfinite inputs or outputs, singular systems, or failed convergence where applicable.
///
/// ## Numerical behavior
///
/// Uses f64 and the shared private backend, with bounded backend-specific SVD/eigen sweep limits. Ill-conditioned inputs can lose accuracy. Kronecker system solved by private pivoted LU, O((mn)³), replacing SciPy's Bartels–Stewart Schur algorithm. Near resonance can lose accuracy.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::solve_sylvester(&[vec![2.]],&[vec![3.]],&[vec![10.]])?,vec![vec![2.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `solve_sylvester`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_sylvester.html).
pub fn solve_sylvester(
    a: &(impl super::MatrixInput + ?Sized),
    b: &(impl super::MatrixInput + ?Sized),
    q: &(impl super::MatrixInput + ?Sized),
) -> Result<Matrix> {
    let a = square(a)?;
    let b = square(b)?;
    let q = dense(q)?;
    let (m, n) = (a.nrows(), b.nrows());
    if q.shape() != (m, n) {
        return Err(Error::DimensionMismatch);
    }
    let d = m * n;
    if d > MAX_DIM {
        return Err(Error::InvalidInput("matrix equations require m*n <=256"));
    }
    let k = Mat::from_fn(d, d, |r, c| {
        let (i, j, p, s) = (r / n, r % n, c / n, c % n);
        (if j == s { a[(i, p)] } else { 0.0 }) + (if i == p { b[(s, j)] } else { 0.0 })
    });
    if k.iter().any(|v| !v.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    let rhs = Mat::from_fn(d, 1, |r, _| q[(r / n, r % n)]);
    let x = k
        .solve_checked(&rhs)
        .ok_or(Error::NumericalFailure("singular Sylvester equation"))?;
    rows(&Mat::from_fn(m, n, |i, j| x[(i * n + j, 0)]))
}
/// Solves the continuous Lyapunov equation `A X+X Aᵀ=Q`.
///
/// ## Parameters
///
/// `a` and `q` are equal-size square real matrices. No stability requirement is imposed; only uniqueness.
///
/// ## Domain
///
/// Numerical inputs must be finite, rectangular, and nonempty. Input axes are limited to 1..=256. n²<=256 and a unique solution are required.
///
/// ## Returns
///
/// The square solution X.
///
/// ## Errors
///
/// Returns a shared error for invalid shapes, parameters, nonfinite inputs or outputs, singular systems, or failed convergence where applicable.
///
/// ## Numerical behavior
///
/// Uses f64 and the shared private backend, with bounded backend-specific SVD/eigen sweep limits. Ill-conditioned inputs can lose accuracy. Reuses [`solve_sylvester`]; note SciPy's positive Q sign convention.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::solve_continuous_lyapunov(&[vec![-2.]],&[vec![8.]])?,vec![vec![-2.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `solve_continuous_lyapunov`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_continuous_lyapunov.html).
pub fn solve_continuous_lyapunov(
    a: &(impl super::MatrixInput + ?Sized),
    q: &(impl super::MatrixInput + ?Sized),
) -> Result<Matrix> {
    let a = square(a)?;
    solve_sylvester(&native(&a), &native(&a.transpose()), q)
}
/// Solves `A X Aᵀ-X+Q=0`, equivalently `X-A X Aᵀ=Q`.
///
/// ## Parameters
///
/// `a` and `q` are equal-size real square matrices. Method selection is omitted; uses the direct Kronecker method.
///
/// ## Domain
///
/// Numerical inputs must be finite, rectangular, and nonempty. Input axes are limited to 1..=256. n²<=256. Eigenvalue products must avoid 1 for a unique solution.
///
/// ## Returns
///
/// The square solution X.
///
/// ## Errors
///
/// Returns a shared error for invalid shapes, parameters, nonfinite inputs or outputs, singular systems, or failed convergence where applicable.
///
/// ## Numerical behavior
///
/// Uses f64 and the shared private backend, with bounded backend-specific SVD/eigen sweep limits. Ill-conditioned inputs can lose accuracy. O(n⁶) direct LU, bounded by n<=16. Near a unit eigenvalue product, accuracy deteriorates.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::solve_discrete_lyapunov(&[vec![0.5]],&[vec![3.]])?,vec![vec![4.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `solve_discrete_lyapunov`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_discrete_lyapunov.html).
pub fn solve_discrete_lyapunov(
    a: &(impl super::MatrixInput + ?Sized),
    q: &(impl super::MatrixInput + ?Sized),
) -> Result<Matrix> {
    let a = square(a)?;
    let q = square(q)?;
    if a.shape() != q.shape() {
        return Err(Error::DimensionMismatch);
    }
    let n = a.nrows();
    let d = n * n;
    if d > MAX_DIM {
        return Err(Error::InvalidInput("matrix equations require n*n <=256"));
    }
    let k = Mat::from_fn(d, d, |r, c| {
        let (i, j, p, s) = (r / n, r % n, c / n, c % n);
        (if r == c { 1.0 } else { 0.0 }) - a[(i, p)] * a[(j, s)]
    });
    if k.iter().any(|v| !v.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    let rhs = Mat::from_fn(d, 1, |r, _| q[(r / n, r % n)]);
    let x = k.solve_checked(&rhs).ok_or(Error::NumericalFailure(
        "singular discrete Lyapunov equation",
    ))?;
    rows(&Mat::from_fn(n, n, |i, j| x[(i * n + j, 0)]))
}
