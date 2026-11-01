//! Matrix structure predicates.
use super::backend::{shape, tolerance};
use crate::{Error, Result};
/// Returns `(l,u)` where `l=max(i-j)` and `u=max(j-i)` over nonzero matrix entries, clipped below at zero.
///
/// ## Parameters
///
/// `a` is a real rectangular row-major matrix. No batching.
///
/// ## Domain
///
/// Finite entries are required. Empty matrices have bandwidth (0,0). At most 1,000,000 entries.
///
/// ## Returns
///
/// The lower bandwidth followed by upper bandwidth.
///
/// ## Errors
///
/// Ragged rows, nonfinite entries, or work-bound violations return a shared error.
///
/// ## Numerical behavior
///
/// O(rows × columns); exact comparisons to zero, without a numerical tolerance.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::basic::bandwidth(&[vec![1.,2.],vec![0.,3.]])?,(0,1));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `bandwidth`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.bandwidth.html).
pub fn bandwidth(a: &(impl super::MatrixInput + ?Sized)) -> Result<(usize, usize)> {
    let (m, n) = shape(a)?;
    let (mut l, mut u) = (0, 0);
    for i in 0..m {
        for j in 0..n {
            let v = a.matrix_entry(i, j);
            if !v.is_finite() {
                return Err(Error::InvalidInput("coordinates must be finite"));
            }
            if v != 0.0 {
                l = l.max(i.saturating_sub(j));
                u = u.max(j.saturating_sub(i));
            }
        }
    }
    Ok((l, u))
}
/// Tests `A=Aᵀ`, exactly or with `|Aij-Aji|<=atol+rtol*|Aji|` in both directions.
///
/// ## Parameters
///
/// `a` is square; missing tolerances contribute zero. If both tolerances are None, diagonal entries are ignored, matching SciPy. Supplying either activates allclose semantics including the diagonal.
///
/// ## Domain
///
/// Empty matrices return true. Matching infinities compare equal. Off-diagonal NaN compares false; diagonal NaN is ignored only in exact mode. Tolerances are finite and nonnegative. At most 1,000,000 entries.
///
/// ## Returns
///
/// Whether A is symmetric.
///
/// ## Errors
///
/// Ragged or nonsquare matrices, invalid tolerances, and work-bound violations return a shared error.
///
/// ## Numerical behavior
///
/// O(n²), without backend calls. No SciPy dtype or batching options. For real inputs symmetry and Hermitian symmetry coincide.
///
/// # Examples
///
/// ```
/// assert!(sci_linalg::basic::issymmetric(&[vec![1.,2.],vec![2.,3.]],None,None)?);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `issymmetric`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.issymmetric.html).
pub fn issymmetric(
    a: &(impl super::MatrixInput + ?Sized),
    atol: Option<f64>,
    rtol: Option<f64>,
) -> Result<bool> {
    let (m, n) = shape(a)?;
    if m != n {
        return Err(Error::DimensionMismatch);
    }
    let exact = atol.is_none() && rtol.is_none();
    let atol = tolerance(atol.unwrap_or(0.0))?;
    let rtol = tolerance(rtol.unwrap_or(0.0))?;
    for i in 0..n {
        for j in 0..n {
            if exact && i == j {
                continue;
            }
            let (x, y) = (a.matrix_entry(i, j), a.matrix_entry(j, i));
            if x == y {
                continue;
            }
            if !x.is_finite() || !y.is_finite() || exact || (x - y).abs() > atol + rtol * y.abs() {
                return Ok(false);
            }
        }
    }
    Ok(true)
}
/// Real-valued alias of [`issymmetric`]: Hermitian symmetry is `A=Aᵀ`.
/// Parameters, IEEE special values, errors, work limits, and conventions are identical.
/// Complex input is unsupported; SciPy's complex Hermitian test is not reproduced.
pub use issymmetric as ishermitian;
