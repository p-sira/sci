use super::{MatrixNorm, norm};
use crate::*;
use sci_complex::Complex;
pub use sci_linalg::{EigenResult, GeneralEigenResult, SvdResult};

fn check_shape(a: &SparseMatrix, square: bool) -> Result<()> {
    validate(a)?;
    if a.rows == 0 || a.cols == 0 || a.rows > 256 || a.cols > 256 || square && a.rows != a.cols {
        return Err(Error::InvalidInput(
            "spectral/dense fallback requires nonempty axes <=256 and a square matrix where applicable",
        ));
    }
    Ok(())
}
fn dense(a: &SparseMatrix, square: bool) -> Result<Matrix> {
    check_shape(a, square)?;
    to_dense(a)
}
fn symmetric(a: &SparseMatrix) -> Result<Matrix> {
    let d = dense(a, true)?;
    for (i, row) in d.iter().enumerate() {
        for (j, &v) in row.iter().enumerate().take(i) {
            if v != d[j][i] {
                return Err(Error::InvalidInput("matrix must be symmetric"));
            }
        }
    }
    Ok(d)
}
/// Selection of a partial spectrum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spectrum {
    /// Largest absolute values.
    LargestMagnitude,
    /// Smallest absolute values.
    SmallestMagnitude,
    /// Largest real values (real parts for a general eigensystem).
    LargestReal,
    /// Smallest real values (real parts for a general eigensystem).
    SmallestReal,
    /// Largest imaginary parts; valid only for general eigenvalues.
    LargestImaginary,
    /// Smallest imaginary parts; valid only for general eigenvalues.
    SmallestImaginary,
    /// Half from each algebraic end, with the extra value from the high end.
    BothEnds,
}
fn selection(w: &[Complex], k: usize, which: Spectrum) -> Result<Vec<usize>> {
    if k == 0 || k > w.len() {
        return Err(Error::InvalidInput(
            "requested spectrum size must be in 1..=dimension",
        ));
    }
    let mut ix: Vec<_> = (0..w.len()).collect();
    let score = |z: Complex| match which {
        Spectrum::LargestMagnitude | Spectrum::SmallestMagnitude => z.re.hypot(z.im),
        Spectrum::LargestImaginary | Spectrum::SmallestImaginary => z.im,
        _ => z.re,
    };
    ix.sort_by(|&i, &j| score(w[i]).total_cmp(&score(w[j])).then(i.cmp(&j)));
    if which == Spectrum::BothEnds {
        let lo = k / 2;
        Ok(ix[..lo]
            .iter()
            .chain(ix[w.len() - (k - lo)..].iter())
            .copied()
            .collect())
    } else if matches!(
        which,
        Spectrum::SmallestMagnitude | Spectrum::SmallestReal | Spectrum::SmallestImaginary
    ) {
        ix.truncate(k);
        Ok(ix)
    } else {
        Ok(ix[w.len() - k..].to_vec())
    }
}
fn general(a: &SparseMatrix, k: usize, which: Spectrum) -> Result<GeneralEigenResult> {
    let e = sci_linalg::eig(&dense(a, true)?)?;
    let ix = selection(&e.eigenvalues, k, which)?;
    Ok(GeneralEigenResult {
        eigenvalues: ix.iter().map(|&i| e.eigenvalues[i]).collect(),
        eigenvectors: e
            .eigenvectors
            .iter()
            .map(|r| ix.iter().map(|&j| r[j]).collect())
            .collect(),
    })
}
fn hermitian(a: &SparseMatrix, k: usize, which: Spectrum) -> Result<EigenResult> {
    if matches!(
        which,
        Spectrum::LargestImaginary | Spectrum::SmallestImaginary
    ) {
        return Err(Error::InvalidInput(
            "imaginary selection is invalid for symmetric spectra",
        ));
    }
    let e = sci_linalg::eigh(&symmetric(a)?, true)?;
    let mut ix = selection(
        &e.eigenvalues
            .iter()
            .map(|&re| Complex { re, im: 0. })
            .collect::<Vec<_>>(),
        k,
        which,
    )?;
    ix.sort_by(|&i, &j| e.eigenvalues[i].total_cmp(&e.eigenvalues[j]));
    Ok(EigenResult {
        eigenvalues: ix.iter().map(|&i| e.eigenvalues[i]).collect(),
        eigenvectors: e
            .eigenvectors
            .iter()
            .map(|r| ix.iter().map(|&j| r[j]).collect())
            .collect(),
    })
}
fn partial_svd(a: &SparseMatrix, k: usize, largest: bool) -> Result<SvdResult> {
    let s = sci_linalg::svd(&dense(a, false)?)?;
    let n = s.singular_values.len();
    if k == 0 || k > n {
        return Err(Error::InvalidInput("SVD count must be in 1..=min(shape)"));
    }
    let ix: Vec<_> = if largest {
        (0..k).rev().collect()
    } else {
        (n - k..n).rev().collect()
    };
    Ok(SvdResult {
        singular_values: ix.iter().map(|&i| s.singular_values[i]).collect(),
        u: s.u
            .iter()
            .map(|r| ix.iter().map(|&j| r[j]).collect())
            .collect(),
        vh: ix.iter().map(|&i| s.vh[i].clone()).collect(),
    })
}
fn block_eigen(a: &SparseMatrix, x: &Matrix, largest: bool) -> Result<EigenResult> {
    let n = a.rows;
    if x.len() != n
        || x.is_empty()
        || x[0].is_empty()
        || x[0].len() > n
        || x.iter().any(|r| r.len() != x[0].len())
    {
        return Err(Error::DimensionMismatch);
    }
    for r in x {
        sci_core::finite_slice(r)?;
    }
    // SciPy also falls back to a full symmetric decomposition for small blocks.
    // This bounded implementation consistently takes that branch.
    hermitian(
        a,
        x[0].len(),
        if largest {
            Spectrum::LargestReal
        } else {
            Spectrum::SmallestReal
        },
    )
}
fn exponential(a: &SparseMatrix) -> Result<SparseMatrix> {
    from_dense(&sci_linalg::expm(&dense(a, true)?)?)
}
fn exponential_action(a: &SparseMatrix, b: &[f64], times: &[f64]) -> Result<Matrix> {
    check_shape(a, true)?;
    if b.len() != a.rows {
        return Err(Error::DimensionMismatch);
    }
    sci_core::finite_slice(b)?;
    sci_core::finite_slice(times)?;
    if times.len() > 64 {
        return Err(Error::InvalidInput(
            "exponential action supports at most 64 times",
        ));
    }
    let mut scaled = to_dense(a)?;
    times
        .iter()
        .map(|&t| {
            for (r, c, v) in entries(a) {
                scaled[r][c] = finite(t * v)?;
            }
            let e = sci_linalg::expm(&scaled)?;
            let mut y = vec![0.; a.rows];
            for (i, r) in e.iter().enumerate() {
                y[i] = finite(r.iter().zip(b).map(|(x, y)| x * y).sum::<f64>())?;
            }
            Ok(y)
        })
        .collect()
}

/// Parameters for bounded Arnoldi matrix-function actions.
#[derive(Clone, Copy, Debug)]
pub struct KrylovOptions {
    /// Scalar time in `f(time * A) b`.
    pub time: f64,
    /// Relative threshold for successive-action changes and Arnoldi coupling.
    pub rtol: f64,
    /// Absolute change threshold.
    pub atol: f64,
    /// Maximum Arnoldi basis size in 1..=256, capped by matrix order.
    pub max_basis: usize,
}
impl Default for KrylovOptions {
    fn default() -> Self {
        Self {
            time: 1.,
            rtol: 1e-6,
            atol: 0.,
            max_basis: 64,
        }
    }
}
fn krylov<F: FnMut(Complex) -> Complex>(
    a: &SparseMatrix,
    b: &[f64],
    mut f: F,
    o: KrylovOptions,
) -> Result<Vec<Complex>> {
    check_shape(a, true)?;
    if b.len() != a.rows {
        return Err(Error::DimensionMismatch);
    }
    sci_core::finite_slice(b)?;
    if !o.time.is_finite()
        || !o.rtol.is_finite()
        || !o.atol.is_finite()
        || o.rtol < 0.
        || o.atol < 0.
        || o.max_basis == 0
        || o.max_basis > 256
    {
        return Err(Error::InvalidInput("invalid Krylov options"));
    }
    let beta = b.iter().fold(0f64, |s, &v| s.hypot(v));
    if beta == 0. {
        return Ok(vec![Complex { re: 0., im: 0. }; b.len()]);
    }
    finite(beta)?;
    let m = o.max_basis.min(a.rows);
    let mut q = vec![b.iter().map(|v| v / beta).collect::<Vec<_>>()];
    let mut h = vec![vec![0.; m]; m + 1];
    let matrix_scale = finite(a.data.iter().fold(0f64, |s, &v| s.hypot(v)))?;
    let mut previous: Option<Vec<Complex>> = None;
    for j in 0..m {
        let mut w = matvec(a, &q[j])?;
        // Two-pass modified Gram–Schmidt reduces loss of orthogonality.
        for _ in 0..2 {
            for i in 0..=j {
                let c = finite(q[i].iter().zip(&w).map(|(x, y)| x * y).sum::<f64>())?;
                h[i][j] = finite(h[i][j] + c)?;
                for (w, &v) in w.iter_mut().zip(&q[i]) {
                    *w = finite(*w - c * v)?;
                }
            }
        }
        h[j + 1][j] = w.iter().fold(0f64, |s, &v| s.hypot(v));
        finite(h[j + 1][j])?;
        let projected: Matrix = (0..=j)
            .map(|i| (0..=j).map(|k| finite(o.time * h[i][k])).collect())
            .collect::<Result<_>>()?;
        let fm = sci_linalg::funm(&projected, &mut f)?;
        let mut y = vec![Complex { re: 0., im: 0. }; b.len()];
        for i in 0..b.len() {
            for k in 0..=j {
                y[i].re += beta * q[k][i] * fm[k][0].re;
                y[i].im += beta * q[k][i] * fm[k][0].im;
            }
            finite(y[i].re)?;
            finite(y[i].im)?;
        }
        let happy = h[j + 1][j] <= 64. * f64::EPSILON * matrix_scale;
        if happy || j + 1 == a.rows {
            return Ok(y);
        }
        if let Some(old) = &previous {
            let diff = y
                .iter()
                .zip(old)
                .fold(0f64, |s, (v, w)| s.hypot((v.re - w.re).hypot(v.im - w.im)));
            let length = y.iter().fold(0f64, |s, v| s.hypot(v.re.hypot(v.im)));
            if diff <= o.atol.max(finite(o.rtol * length)?)
                && h[j + 1][j] <= o.atol.max(finite(o.rtol * matrix_scale)?)
            {
                return Ok(y);
            }
        }
        previous = Some(y);
        q.push(w.iter().map(|&v| v / h[j + 1][j]).collect());
    }
    Err(Error::NumericalFailure(
        "Krylov action did not converge within basis limit",
    ))
}

/// Compute k right eigenpairs satisfying `A v = λ v`.
///
/// # Parameters and conventions
/// which selects magnitude, real or imaginary extremes; BothEnds selects real extremes. Eigenvectors are unit-norm columns in returned eigenvalue order.
/// # Domain and special values
/// A real matrix may have complex eigenpairs. Repeated/defective eigenvectors are nonunique. Unless otherwise specified, nonempty axes are bounded at 256.
/// # Returns
/// Owned shared Sci output, with ordering and shape as specified above.
/// # Errors
/// Invalid CSR, shape/selection/domain, work limits, backend failure or nonfinite output.
/// # Numerical behavior
/// Private Sci/faer full real Schur decomposition followed by selection; dense fallback with axes <=256. Generalized/shift-invert and ARPACK tuning options are not exposed.
/// Accuracy deteriorates for ill-conditioned or nearly defective problems.
/// # Examples
/// ```
/// let a=sci_sparse::identity(2)?;
/// let _=sci_sparse::linalg::eigs(&a,1,sci_sparse::linalg::Spectrum::LargestMagnitude)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy eigs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html).
pub fn eigs(a: &SparseMatrix, k: usize, which: Spectrum) -> Result<GeneralEigenResult> {
    general(a, k, which)
}

/// Compute k orthonormal eigenpairs of a real symmetric matrix, `A V = V diag(w)`.
///
/// # Parameters and conventions
/// which selects spectrum extremes; output eigenvalues are ascending and eigenvectors are columns.
/// # Domain and special values
/// Both triangles must agree exactly; imaginary selection is invalid. Unless otherwise specified, nonempty axes are bounded at 256.
/// # Returns
/// Owned shared Sci output, with ordering and shape as specified above.
/// # Errors
/// Invalid CSR, shape/selection/domain, work limits, backend failure or nonfinite output.
/// # Numerical behavior
/// Private Sci/faer full symmetric eigensolver followed by selection; dense fallback with axes <=256. No generalized/shift-invert options.
/// Accuracy deteriorates for ill-conditioned or nearly defective problems.
/// # Examples
/// ```
/// let a=sci_sparse::identity(2)?;
/// let _=sci_sparse::linalg::eigsh(&a,1,sci_sparse::linalg::Spectrum::LargestReal)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy eigsh](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html).
pub fn eigsh(a: &SparseMatrix, k: usize, which: Spectrum) -> Result<EigenResult> {
    hermitian(a, k, which)
}

/// Compute a rank-k partial SVD, `A V = U diag(s)`.
///
/// # Parameters and conventions
/// largest selects largest versus smallest singular values. Values are returned ascending, U has k columns and Vᵀ has k rows.
/// # Domain and special values
/// Rectangular finite matrices; k is in 1..=min(shape). Singular values may be zero. Unless otherwise specified, nonempty axes are bounded at 256.
/// # Returns
/// Owned shared Sci output, with ordering and shape as specified above.
/// # Errors
/// Invalid CSR, shape/selection/domain, work limits, backend failure or nonfinite output.
/// # Numerical behavior
/// Private Sci/faer full SVD with selected vectors; dense fallback with axes <=256. Signs and bases of repeated singular vectors are nonunique.
/// Accuracy deteriorates for ill-conditioned or nearly defective problems.
/// # Examples
/// ```
/// let a=sci_sparse::identity(2)?;
/// let _=sci_sparse::linalg::svds(&a,1,true)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy svds](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html).
pub fn svds(a: &SparseMatrix, k: usize, largest: bool) -> Result<SvdResult> {
    partial_svd(a, k, largest)
}

/// Compute a block of symmetric eigenpairs, `A X = X diag(w)`.
///
/// # Parameters and conventions
/// initial has shape n×k and determines k; largest chooses high versus low algebraic eigenvalues. Output eigenvalues are ascending.
/// # Domain and special values
/// Symmetric finite input; finite initial columns, 1<=k<=n. Unless otherwise specified, nonempty axes are bounded at 256.
/// # Returns
/// Owned shared Sci output, with ordering and shape as specified above.
/// # Errors
/// Invalid CSR, shape/selection/domain, work limits, backend failure or nonfinite output.
/// # Numerical behavior
/// Uses the bounded full symmetric decomposition fallback (also used by SciPy for small problems). Initial values do not influence the result; B, preconditioners, constraints and iteration histories are not exposed.
/// Accuracy deteriorates for ill-conditioned or nearly defective problems.
/// # Examples
/// ```
/// let a=sci_sparse::identity(2)?;
/// let _=sci_sparse::linalg::lobpcg(&a,&vec![vec![1.],vec![0.]],true)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy lobpcg](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lobpcg.html).
pub fn lobpcg(a: &SparseMatrix, initial: &Matrix, largest: bool) -> Result<EigenResult> {
    block_eigen(a, initial, largest)
}

/// Compute the matrix exponential `exp(A) = Σ A^k/k!`.
///
/// # Parameters and conventions
/// A is square; output is canonical CSR and may be dense.
/// # Domain and special values
/// Finite matrices; a nonfinite exponential returns an error. Unless otherwise specified, nonempty axes are bounded at 256.
/// # Returns
/// Owned shared Sci output, with ordering and shape as specified above.
/// # Errors
/// Invalid CSR, shape/selection/domain, work limits, backend failure or nonfinite output.
/// # Numerical behavior
/// Delegates to existing Sci/faer scaling-and-squaring Pade implementation; dense fallback with order <=256 and its bounded scaling limits.
/// Accuracy deteriorates for ill-conditioned or nearly defective problems.
/// # Examples
/// ```
/// let a=sci_sparse::identity(2)?;
/// let _=sci_sparse::linalg::expm(&a)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy expm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm.html).
pub fn expm(a: &SparseMatrix) -> Result<SparseMatrix> {
    exponential(a)
}

/// Evaluate `y(t) = exp(t A) b` at each requested time.
///
/// # Parameters and conventions
/// b is a vector; times gives explicit times in output row order. Include 1.0 for the ordinary action; up to 64 times are accepted.
/// # Domain and special values
/// Finite square A, finite vector/time inputs. Negative times are supported. Unless otherwise specified, nonempty axes are bounded at 256.
/// # Returns
/// Owned shared Sci output, with ordering and shape as specified above.
/// # Errors
/// Invalid CSR, shape/selection/domain, work limits, backend failure or nonfinite output.
/// # Numerical behavior
/// Uses bounded Sci/faer Pade exponentials, order <=256, rather than the SciPy truncated-Taylor action algorithm. Time grids are explicit and no trace estimate is needed.
/// Accuracy deteriorates for ill-conditioned or nearly defective problems.
/// # Examples
/// ```
/// let a=sci_sparse::identity(2)?;
/// let _=sci_sparse::linalg::expm_multiply(&a,&[1.,2.],&[0.,1.])?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy expm_multiply](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm_multiply.html).
pub fn expm_multiply(a: &SparseMatrix, b: &[f64], times: &[f64]) -> Result<Matrix> {
    exponential_action(a, b, times)
}

/// Return the matrix induced 1-norm, `max_j Σ_i |A[i,j]|`.
///
/// # Parameters and conventions
/// Computes the exact norm instead of a stochastic lower estimate; this is also a valid lower bound with equality.
/// # Domain and special values
/// Finite canonical CSR; empty matrices follow norm conventions. Unless otherwise specified, nonempty axes are bounded at 256.
/// # Returns
/// Owned shared Sci output, with ordering and shape as specified above.
/// # Errors
/// Invalid CSR, shape/selection/domain, work limits, backend failure or nonfinite output.
/// # Numerical behavior
/// Linear sparse scan using the existing norm implementation, including its overflow and storage limits. Estimator iteration controls and witness vectors are omitted.
/// Accuracy deteriorates for ill-conditioned or nearly defective problems.
/// # Examples
/// ```
/// let a=sci_sparse::identity(2)?;
/// let _=sci_sparse::linalg::onenormest(&a)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy onenormest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.onenormest.html).
pub fn onenormest(a: &SparseMatrix) -> Result<f64> {
    norm(a, MatrixNorm::One)
}

/// Evaluate `f(time A)b` using an Arnoldi projection `β Q f(time H)e₁`.
///
/// # Parameters and conventions
/// `function` maps a shared complex scalar to its principal desired function
/// value. `options` selects time, tolerances and maximum basis size. Output is
/// a complex vector, supporting real matrices whose function is complex.
/// # Domain and special values
/// Finite square A of order 1..256 and finite vector b; a zero b gives zero.
/// The callback must return finite values on the projected spectrum.
/// # Returns
/// Projected action in original row order after convergence or happy breakdown.
/// # Errors
/// Invalid inputs, callback/backend failure, nonfinite values, or exhausted basis.
/// # Numerical behavior
/// Two-pass Arnoldi orthogonalization; projected functions use Sci/faer Schur
/// kernels. Coupled repeated projected eigenvalues require derivatives and
/// return an error; nearly coincident spectra can reduce accuracy. Successive-action changes are a heuristic error indicator, not a
/// rigorous bound for general functions. Basis <=256; no restarts. Unlike SciPy,
/// restart controls and matrix-structure hints are replaced by a basis limit.
/// # Examples
/// ```
/// let y=sci_sparse::linalg::funm_multiply_krylov(
///     &sci_sparse::identity(2)?, &[1.,2.], sci_complex::exp, Default::default())?;
/// assert!((y[0].re-1_f64.exp()).abs()<1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy funm_multiply_krylov](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.funm_multiply_krylov.html).
pub fn funm_multiply_krylov<F: FnMut(Complex) -> Complex>(
    a: &SparseMatrix,
    b: &[f64],
    function: F,
    options: KrylovOptions,
) -> Result<Vec<Complex>> {
    krylov(a, b, function, options)
}

/// Compute vector p-norms along an axis: `(Σ |x|^p)^(1/p)`.
///
/// # Parameters and conventions
/// axis=0 reduces rows, returning one value per column; axis=1 reduces columns.
/// p=0 counts nonzero values, ±infinity select maximum/minimum absolute value.
/// Finite negative p follows SciPy sparse-power semantics: implicit zeros do
/// not contribute, while stored zeros yield zero. Empty stored reductions yield
/// positive infinity for negative p. Output is a vector; callers reshape for keepdims.
/// # Domain and special values
/// Finite CSR input. p must not be NaN; axis is 0 or 1. Empty reduced axes
/// give zero for finite positive p or p=0, and error for negative p or ±infinity.
/// # Returns
/// Nonnegative values with units of input entries (dimensionless for p=0).
/// # Errors
/// Invalid storage/axis/p, unsupported empty reduction, or nonfinite result.
/// # Numerical behavior
/// O(rows+cols+nnz) grouping. Negative-p sparse semantics differ from dense
/// vector norms when implicit zeros are present. Each reduction is scaled by its largest
/// magnitude for p>0, or smallest nonzero magnitude for p<0, reducing overflow.
/// # Examples
/// ```
/// assert_eq!(sci_sparse::linalg::norm_axis(&sci_sparse::identity(2)?,2.,0)?,[1.,1.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy sparse norm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.norm.html).
pub fn norm_axis(a: &SparseMatrix, p: f64, axis: usize) -> Result<Vec<f64>> {
    validate(a)?;
    if p.is_nan() || axis > 1 {
        return Err(Error::InvalidInput("invalid vector norm axis/order"));
    }
    let (len, reduced) = if axis == 0 {
        (a.cols, a.rows)
    } else {
        (a.rows, a.cols)
    };
    if reduced == 0 && (p < 0. || p.is_infinite()) {
        return Err(Error::InvalidInput("empty norm reduction"));
    }
    let mut groups = vec![vec![]; len];
    for (r, c, v) in entries(a) {
        groups[if axis == 0 { c } else { r }].push(v.abs());
    }
    groups
        .iter()
        .map(|v| {
            if p == 0. {
                return Ok(v.iter().filter(|&&x| x != 0.).count() as f64);
            }
            if p == f64::INFINITY {
                return Ok(v.iter().copied().fold(0., f64::max));
            }
            if p == f64::NEG_INFINITY {
                return Ok(if v.len() < reduced {
                    0.
                } else {
                    v.iter().copied().fold(f64::INFINITY, f64::min)
                });
            }
            if v.is_empty() {
                return Ok(if p < 0. { f64::INFINITY } else { 0. });
            }
            if p < 0. && v.contains(&0.) {
                return Ok(0.);
            }
            let scale = if p > 0. {
                v.iter().copied().fold(0., f64::max)
            } else {
                v.iter().copied().fold(f64::INFINITY, f64::min)
            };
            if scale == 0. {
                return Ok(0.);
            }
            finite(
                scale
                    * v.iter()
                        .map(|&x| (x / scale).powf(p))
                        .sum::<f64>()
                        .powf(1. / p),
            )
        })
        .collect()
}
