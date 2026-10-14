//! Deterministic real interpolative decomposition.
use super::{Matrix, SvdResult, backend::*};
use crate::{Error, Result};
use faer::Mat;
/// Selects a fixed rank or a relative singular-value cutoff.
#[derive(Clone, Copy, Debug)]
pub enum Rank {
    /// Exact requested number of skeleton columns, at most min(m,n).
    Fixed(usize),
    /// Retains singular values strictly greater than epsilon times the largest.
    Relative(f64),
}
/// Interpolative decomposition `A[:,indices] ≈ B [I, projection]`.
#[derive(Clone, Debug)]
pub struct InterpolativeResult {
    /// Number of skeleton columns.
    pub rank: usize,
    /// Full column permutation; skeleton indices come first.
    pub indices: Vec<usize>,
    /// Coefficients of shape rank × (n-rank).
    pub projection: Matrix,
}
fn permutation(indices: &[usize]) -> Result<()> {
    if indices.is_empty() || indices.len() > MAX_DIM {
        return Err(Error::InvalidInput("permutation length must be in 1..=256"));
    }
    let mut seen = vec![false; indices.len()];
    for &i in indices {
        if i >= indices.len() || seen[i] {
            return Err(Error::InvalidInput("indices must be a permutation"));
        }
        seen[i] = true;
    }
    Ok(())
}
fn rank(a: &[Vec<f64>], selection: Rank) -> Result<usize> {
    let a = dense(a)?;
    match selection {
        Rank::Fixed(k) if k <= a.nrows().min(a.ncols()) => Ok(k),
        Rank::Fixed(_) => Err(Error::InvalidInput("rank exceeds the smaller matrix axis")),
        Rank::Relative(eps) => {
            if !eps.is_finite() || !(0.0..1.0).contains(&eps) {
                return Err(Error::InvalidInput("relative precision must be in [0,1)"));
            }
            let values = super::svdvals(&rows(&a)?)?;
            Ok(values.iter().filter(|&&s| s > eps * values[0]).count())
        }
    }
}
/// Computes a column interpolative approximation `A ≈ B P`.
///
/// ## Parameters
///
/// `selection` is a fixed rank or relative SVD cutoff. Pivoted QR selects
/// skeleton columns; a shared SVD least-squares solve obtains coefficients.
///
/// ## Domain
///
/// Finite real row-major matrices. Dense axes are bounded at 256.
/// NaN/infinity are rejected; zero rank is supported.
///
/// ## Returns
///
/// Rank, full column permutation, and projection coefficients.
///
/// ## Errors
///
/// Invalid shapes, rank/tolerance/permutation, nonfinite output, or failed
/// backend decomposition.
///
/// ## Numerical behavior
///
/// Uses shared faer factorizations with bounded iterations.
/// Deterministic dense computation replaces SciPy randomized/operator modes.
/// QR pivots and coefficients can differ from SciPy. A singular-value cutoff
/// does not guarantee the final ID error is at most the requested precision.
///
/// # Examples
///
/// ```
/// use sci_linalg::interpolative::{interp_decomp, Rank};
/// let id = interp_decomp(&[vec![1.,2.], vec![2.,4.]], Rank::Fixed(1))?;
/// assert_eq!(id.rank,1);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `interpolative.interp_decomp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.interpolative.interp_decomp.html).
pub fn interp_decomp(
    a: &(impl super::MatrixInput + ?Sized),
    selection: Rank,
) -> Result<InterpolativeResult> {
    let a_input = super::input::matrix_rows(a)?;
    let a = a_input.as_ref();
    let k = rank(a, selection)?;
    let a = dense(a)?;
    let (m, n) = a.shape();
    let (scaled_a, _) = scaled(a.clone());
    let f = faer::Mat::from_fn(m, n, |i, j| scaled_a[(i, j)]).col_piv_qr();
    let indices = f.P().arrays().0.to_vec();
    let skeleton = Mat::from_fn(m, k, |i, j| a[(i, indices[j])]);
    let mut projection = vec![vec![0.; n - k]; k];
    if k > 0 && k < n {
        let (inverse, _) = super::pinv(&rows(&skeleton)?, 0., None)?;
        let inverse = dense(&inverse)?;
        let remaining = Mat::from_fn(m, n - k, |i, j| a[(i, indices[k + j])]);
        projection = rows(&(inverse * remaining))?;
    }
    Ok(InterpolativeResult {
        rank: k,
        indices,
        projection,
    })
}
/// Builds `P[:,indices] = [I, projection]`.
///
/// ## Parameters
///
/// `indices` is a full permutation. `projection` has k rows and n-k columns.
/// An empty projection represents rank zero; k=n accepts k empty rows.
///
/// ## Domain
///
/// Finite real row-major matrices. Dense axes are bounded at 256.
/// NaN/infinity are rejected; zero rank is supported.
///
/// ## Returns
///
/// The k × n interpolation matrix.
///
/// ## Errors
///
/// Invalid shapes, rank/tolerance/permutation, nonfinite output, or failed
/// backend decomposition.
///
/// ## Numerical behavior
///
/// Uses shared faer factorizations with bounded iterations.
/// Deterministic dense computation replaces SciPy randomized/operator modes.
///
///
/// # Examples
///
/// ```
/// let p = sci_linalg::interpolative::reconstruct_interp_matrix(&[1,0], &[vec![0.5]])?;
/// assert_eq!(p, vec![vec![0.5,1.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `interpolative.reconstruct_interp_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.interpolative.reconstruct_interp_matrix.html).
pub fn reconstruct_interp_matrix(
    indices: &[usize],
    projection: &(impl super::MatrixInput + ?Sized),
) -> Result<Matrix> {
    let projection_input = super::input::matrix_rows(projection)?;
    let projection = projection_input.as_ref();
    permutation(indices)?;
    let n = indices.len();
    let k = projection.len();
    if k > n || projection.iter().any(|r| r.len() != n - k) {
        return Err(Error::DimensionMismatch);
    }
    for r in projection {
        crate::error::finite_slice(r)?;
    }
    let mut p = vec![vec![0.; n]; k];
    for i in 0..k {
        p[i][indices[i]] = 1.;
        for j in k..n {
            p[i][indices[j]] = projection[i][j - k];
        }
    }
    Ok(p)
}
/// Extracts skeleton columns `B = A[:,indices[..k]]`.
///
/// ## Parameters
///
/// `k` is at most the number of columns. `indices` is a full permutation.
///
/// ## Domain
///
/// Finite real row-major matrices. Dense axes are bounded at 256.
/// NaN/infinity are rejected; zero rank is supported.
///
/// ## Returns
///
/// An m × k matrix, including m empty rows when k=0.
///
/// ## Errors
///
/// Invalid shapes, rank/tolerance/permutation, nonfinite output, or failed
/// backend decomposition.
///
/// ## Numerical behavior
///
/// Uses shared faer factorizations with bounded iterations.
/// Deterministic dense computation replaces SciPy randomized/operator modes.
///
///
/// # Examples
///
/// ```
/// let b = sci_linalg::interpolative::reconstruct_skel_matrix(&[vec![1.,2.]],1,&[1,0])?;
/// assert_eq!(b,vec![vec![2.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `interpolative.reconstruct_skel_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.interpolative.reconstruct_skel_matrix.html).
pub fn reconstruct_skel_matrix(
    a: &(impl super::MatrixInput + ?Sized),
    k: usize,
    indices: &[usize],
) -> Result<Matrix> {
    let a_input = super::input::matrix_rows(a)?;
    let a = a_input.as_ref();
    let a = dense(a)?;
    permutation(indices)?;
    if indices.len() != a.ncols() || k > indices.len() {
        return Err(Error::DimensionMismatch);
    }
    Ok((0..a.nrows())
        .map(|i| (0..k).map(|j| a[(i, indices[j])]).collect())
        .collect())
}
/// Reconstructs `A = B P` from a column interpolative decomposition.
///
/// ## Parameters
///
/// `b` is the m × k skeleton; projection is k × (n-k).
/// `indices` is the full column permutation.
///
/// ## Domain
///
/// Finite real row-major matrices. Dense axes are bounded at 256.
/// NaN/infinity are rejected; zero rank is supported.
///
/// ## Returns
///
/// An m × n reconstructed matrix.
///
/// ## Errors
///
/// Invalid shapes, rank/tolerance/permutation, nonfinite output, or failed
/// backend decomposition.
///
/// ## Numerical behavior
///
/// Uses shared faer factorizations with bounded iterations.
/// Deterministic dense computation replaces SciPy randomized/operator modes.
///
///
/// # Examples
///
/// ```
/// let a = sci_linalg::interpolative::reconstruct_matrix_from_id(&[vec![2.]],&[1,0],&[vec![0.5]])?;
/// assert_eq!(a,vec![vec![1.,2.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `interpolative.reconstruct_matrix_from_id`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.interpolative.reconstruct_matrix_from_id.html).
pub fn reconstruct_matrix_from_id(
    b: &(impl super::MatrixInput + ?Sized),
    indices: &[usize],
    projection: &(impl super::MatrixInput + ?Sized),
) -> Result<Matrix> {
    let b_input = super::input::matrix_rows(b)?;
    let b = b_input.as_ref();
    let projection_input = super::input::matrix_rows(projection)?;
    let projection = projection_input.as_ref();
    let (m, k) = shape(b)?;
    if m == 0 || m > MAX_DIM || k > MAX_DIM || k != projection.len() {
        return Err(Error::DimensionMismatch);
    }
    for r in b {
        crate::error::finite_slice(r)?;
    }
    let p = reconstruct_interp_matrix(indices, projection)?;
    let mut a = vec![vec![0.; indices.len()]; m];
    for i in 0..m {
        for (j, row) in p.iter().enumerate() {
            for (h, &v) in row.iter().enumerate() {
                a[i][h] += b[i][j] * v;
            }
        }
    }
    output(a)
}
/// Computes a truncated SVD `A ≈ U diag(s) Vᵀ`.
///
/// ## Parameters
///
/// `selection` specifies retained rank or relative singular-value cutoff.
///
/// ## Domain
///
/// Finite real row-major matrices. Dense axes are bounded at 256.
/// NaN/infinity are rejected; zero rank is supported.
///
/// ## Returns
///
/// Shared SvdResult with U: m × k, s: k, vh: k × n.
/// SciPy interpolative SVD returns V; this API returns vh for interoperability.
///
/// ## Errors
///
/// Invalid shapes, rank/tolerance/permutation, nonfinite output, or failed
/// backend decomposition.
///
/// ## Numerical behavior
///
/// Uses shared faer factorizations with bounded iterations.
/// Deterministic dense computation replaces SciPy randomized/operator modes.
/// Uses full economy SVD, O(m n min(m,n)), instead of randomized ID-to-SVD.
///
/// # Examples
///
/// ```
/// use sci_linalg::interpolative::{svd,Rank};
/// assert_eq!(svd(&[vec![1.,2.],vec![2.,4.]],Rank::Fixed(1))?.singular_values.len(),1);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `interpolative.svd`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.interpolative.svd.html).
pub fn svd(a: &(impl super::MatrixInput + ?Sized), selection: Rank) -> Result<SvdResult> {
    let a_input = super::input::matrix_rows(a)?;
    let a = a_input.as_ref();
    let k = rank(a, selection)?;
    let mut f = super::svd(a)?;
    for row in &mut f.u {
        row.truncate(k);
    }
    f.singular_values.truncate(k);
    f.vh.truncate(k);
    Ok(f)
}
/// Converts `B P` into a thin singular-value decomposition.
///
/// ## Parameters
///
/// Inputs follow reconstruct_matrix_from_id. Retains k skeleton dimensions,
/// even if some singular values are zero.
///
/// ## Domain
///
/// Finite real row-major matrices. Dense axes are bounded at 256.
/// NaN/infinity are rejected; zero rank is supported.
///
/// ## Returns
///
/// Shared SvdResult with vh rather than SciPy V.
///
/// ## Errors
///
/// Invalid shapes, rank/tolerance/permutation, nonfinite output, or failed
/// backend decomposition.
///
/// ## Numerical behavior
///
/// Uses shared faer factorizations with bounded iterations.
/// Deterministic dense computation replaces SciPy randomized/operator modes.
/// Reconstructs the bounded dense matrix before SVD.
///
/// # Examples
///
/// ```
/// let f = sci_linalg::interpolative::id_to_svd(&[vec![2.]],&[1,0],&[vec![0.5]])?;
/// assert_eq!(f.singular_values.len(),1);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `interpolative.id_to_svd`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.interpolative.id_to_svd.html).
pub fn id_to_svd(
    b: &(impl super::MatrixInput + ?Sized),
    indices: &[usize],
    projection: &(impl super::MatrixInput + ?Sized),
) -> Result<SvdResult> {
    let b_input = super::input::matrix_rows(b)?;
    let b = b_input.as_ref();
    let projection_input = super::input::matrix_rows(projection)?;
    let projection = projection_input.as_ref();
    let a = reconstruct_matrix_from_id(b, indices, projection)?;
    svd(&a, Rank::Fixed(projection.len()))
}
/// Returns numerical rank `#{sᵢ > epsilon s₀}`.
///
/// ## Parameters
///
/// `epsilon` is a finite relative threshold in [0,1).
///
/// ## Domain
///
/// Finite real row-major matrices. Dense axes are bounded at 256.
/// NaN/infinity are rejected; zero rank is supported.
///
/// ## Returns
///
/// The deterministic singular-value rank; a zero matrix has rank zero.
///
/// ## Errors
///
/// Invalid shapes, rank/tolerance/permutation, nonfinite output, or failed
/// backend decomposition.
///
/// ## Numerical behavior
///
/// Uses shared faer factorizations with bounded iterations.
/// Deterministic dense computation replaces SciPy randomized/operator modes.
/// Computes the full SVD rather than SciPy randomized overestimate.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::interpolative::estimate_rank(&[vec![1.,2.],vec![2.,4.]],1e-12)?,1);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `interpolative.estimate_rank`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.interpolative.estimate_rank.html).
pub fn estimate_rank(a: &(impl super::MatrixInput + ?Sized), epsilon: f64) -> Result<usize> {
    let a_input = super::input::matrix_rows(a)?;
    let a = a_input.as_ref();
    rank(a, Rank::Relative(epsilon))
}
/// Returns spectral norm `||A||₂ = s₀`.
///
/// ## Parameters
///
/// `a` is a finite rectangular matrix. SciPy iteration/rng options are omitted.
///
/// ## Domain
///
/// Finite real row-major matrices. Dense axes are bounded at 256.
/// NaN/infinity are rejected; zero rank is supported.
///
/// ## Returns
///
/// The largest singular value, including zero for a zero matrix.
///
/// ## Errors
///
/// Invalid shapes, rank/tolerance/permutation, nonfinite output, or failed
/// backend decomposition.
///
/// ## Numerical behavior
///
/// Uses shared faer factorizations with bounded iterations.
/// Deterministic dense computation replaces SciPy randomized/operator modes.
/// Computes the full SVD instead of a randomized norm estimate.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::interpolative::estimate_spectral_norm(&[vec![3.,4.]])?,5.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `interpolative.estimate_spectral_norm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.interpolative.estimate_spectral_norm.html).
pub fn estimate_spectral_norm(a: &(impl super::MatrixInput + ?Sized)) -> Result<f64> {
    let a_input = super::input::matrix_rows(a)?;
    let a = a_input.as_ref();
    Ok(super::svdvals(a)?[0])
}
/// Returns spectral norm `||A-B||₂ = s₀(A-B)`.
///
/// ## Parameters
///
/// `a` and `b` have matching shapes. SciPy iteration/rng options are omitted.
///
/// ## Domain
///
/// Finite real row-major matrices. Dense axes are bounded at 256.
/// NaN/infinity are rejected; zero rank is supported.
///
/// ## Returns
///
/// The largest singular value of the difference.
///
/// ## Errors
///
/// Invalid shapes, rank/tolerance/permutation, nonfinite output, or failed
/// backend decomposition.
///
/// ## Numerical behavior
///
/// Uses shared faer factorizations with bounded iterations.
/// Deterministic dense computation replaces SciPy randomized/operator modes.
/// Computes the full SVD instead of randomized estimation.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::interpolative::estimate_spectral_norm_diff(&[vec![3.]],&[vec![1.]])?,2.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `interpolative.estimate_spectral_norm_diff`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.interpolative.estimate_spectral_norm_diff.html).
pub fn estimate_spectral_norm_diff(
    a: &(impl super::MatrixInput + ?Sized),
    b: &(impl super::MatrixInput + ?Sized),
) -> Result<f64> {
    let a_input = super::input::matrix_rows(a)?;
    let a = a_input.as_ref();
    let b_input = super::input::matrix_rows(b)?;
    let b = b_input.as_ref();
    let a = dense(a)?;
    let b = dense(b)?;
    if a.shape() != b.shape() {
        return Err(Error::DimensionMismatch);
    }
    estimate_spectral_norm(&rows(&(a - b))?)
}
