use super::{LinearMap, spsolve_triangular};
use crate::*;

/// Threshold incomplete LU factors, used as an approximate inverse operator.
///
/// Backend factors are private; the retained triangular matrices have no
/// row/column permutation. Applying this linear map solves with the factors.
#[derive(Clone, Debug)]
pub struct SparseIlu {
    lower: SparseMatrix,
    upper: SparseMatrix,
}
impl SparseIlu {
    /// Apply the incomplete-factor inverse, `x = U⁻¹ L⁻¹ b`.
    ///
    /// # Parameters and conventions
    /// b has one component per factor row; L has an implicit unit diagonal.
    /// # Domain and special values
    /// Finite vectors; factors are nonsingular but approximate the original A.
    /// # Returns
    /// Approximate solution in original column order.
    /// # Errors
    /// Invalid length, nonfinite input/output, or triangular solve failure.
    /// # Numerical behavior
    /// Two bounded sparse triangular solves; no residual refinement.
    /// # Examples
    /// ```
    /// let lu=sci_sparse::linalg::spilu(&sci_sparse::construct::identity(2)?,1e-4,2.)?;
    /// assert_eq!(lu.solve(&[2.,3.])?,[2.,3.]);
    /// # Ok::<(), sci_core::Error>(())
    /// ```
    /// # References
    /// [SciPy spilu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spilu.html).
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>> {
        let y = spsolve_triangular(&self.lower, b, true, true)?;
        spsolve_triangular(&self.upper, &y, false, false)
    }
}
impl LinearMap for SparseIlu {
    fn rows(&self) -> usize {
        self.lower.rows
    }
    fn cols(&self) -> usize {
        self.lower.cols
    }
    fn apply(&self, x: &[f64]) -> Result<Vec<f64>> {
        self.solve(x)
    }
    fn transpose_apply(&self, x: &[f64]) -> Result<Vec<f64>> {
        let y = spsolve_triangular(&transpose(&self.upper)?, x, true, false)?;
        spsolve_triangular(&transpose(&self.lower)?, &y, false, true)
    }
}
fn retain(row: &mut [f64], start: usize, end: usize, tol: f64, cap: usize) {
    let mut ix: Vec<_> = (start..end).filter(|&i| row[i].abs() > tol).collect();
    ix.sort_by(|&i, &j| row[j].abs().total_cmp(&row[i].abs()).then(i.cmp(&j)));
    let keep: std::collections::BTreeSet<_> = ix.into_iter().take(cap).collect();
    for (i, v) in row.iter_mut().enumerate().take(end).skip(start) {
        if !keep.contains(&i) {
            *v = 0.;
        }
    }
}
/// Compute a threshold incomplete factorization `A ≈ L U` without pivoting.
///
/// # Parameters and conventions
/// drop_tol is relative to the input row's maximum magnitude for U entries,
/// and absolute for L multipliers. fill_factor sets each triangle's retained
/// row count to ceil(fill_factor * max(original row nnz,1)), capped by order.
/// Diagonal pivots are retained; L has a unit diagonal.
/// # Domain and special values
/// Finite square order 1..256, drop_tol in `[0,1]`, fill_factor in `[1,20]`.
/// Matrices requiring row exchanges may fail even when nonsingular.
/// # Returns
/// [`SparseIlu`] with a solve method and inverse [`LinearMap`] implementation.
/// # Errors
/// Invalid CSR/controls, zero pivot, or nonfinite arithmetic.
/// # Numerical behavior
/// Bounded row-wise ILUT elimination (at most 256³ multiply-adds), with
/// magnitude-based dropping and no pivoting, equilibration or SuperLU options.
/// This is an ILUT adaptation; factors and approximation differ from SuperLU.
/// Lower drop tolerances and larger fill generally improve approximation.
/// # Examples
/// ```
/// let factor=sci_sparse::linalg::spilu(&sci_sparse::construct::identity(3)?,0.,1.)?;
/// assert_eq!(factor.solve(&[1.,2.,3.])?,[1.,2.,3.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy spilu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spilu.html),
/// Saad, *Iterative Methods for Sparse Linear Systems*, second edition, §10.4.
pub fn spilu(a: &SparseMatrix, drop_tol: f64, fill_factor: f64) -> Result<SparseIlu> {
    validate(a)?;
    let n = a.rows;
    if n == 0
        || n > 256
        || n != a.cols
        || !drop_tol.is_finite()
        || !(0.0..=1.0).contains(&drop_tol)
        || !fill_factor.is_finite()
        || !(1.0..=20.0).contains(&fill_factor)
    {
        return Err(Error::InvalidInput(
            "invalid incomplete LU shape or controls",
        ));
    }
    let mut l = vec![vec![0.; n]; n];
    let mut u = vec![vec![0.; n]; n];
    for i in 0..n {
        let mut row = vec![0f64; n];
        for p in a.indptr[i]..a.indptr[i + 1] {
            row[a.indices[p]] = a.data[p];
        }
        let scale = row.iter().fold(0f64, |s, v| s.max(v.abs()));
        for j in 0..i {
            if row[j] == 0. {
                continue;
            }
            if u[j][j] == 0. {
                return Err(Error::NumericalFailure("incomplete LU has a zero pivot"));
            }
            let factor = match row[j] / u[j][j] {
                value if value.is_finite() => Ok(value),
                _ => Err(crate::Error::NonFiniteEvaluation),
            }?;
            row[j] = 0.;
            if factor.abs() <= drop_tol {
                continue;
            }
            l[i][j] = factor;
            for (k, v) in row.iter_mut().enumerate().skip(j + 1) {
                *v = match *v - factor * u[j][k] {
                    value if value.is_finite() => Ok(value),
                    _ => Err(crate::Error::NonFiniteEvaluation),
                }?;
            }
        }
        if row[i] == 0. {
            return Err(Error::NumericalFailure("incomplete LU has a zero pivot"));
        }
        let cap = (fill_factor * (a.indptr[i + 1] - a.indptr[i]).max(1) as f64).ceil() as usize;
        retain(&mut l[i], 0, i, drop_tol, cap.min(n));
        retain(
            &mut row,
            i + 1,
            n,
            drop_tol * scale,
            cap.saturating_sub(1).min(n),
        );
        l[i][i] = 1.;
        u[i] = row;
    }
    Ok(SparseIlu {
        lower: from_dense(&l)?,
        upper: from_dense(&u)?,
    })
}
