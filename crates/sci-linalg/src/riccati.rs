//! Stabilizing real algebraic Riccati equations through private Schur backends.
use super::{Matrix, backend::*, principal::complex_dense};
use crate::{Error, Result};
use faer::{Mat, Scale, prelude::Solve};
fn validate_symmetric(a: &Mat<f64>) -> Result<()> {
    if (a - a.transpose()).norm_l2() > 1e-12 * (1. + a.norm_l2()) {
        return Err(Error::InvalidInput("Q and R must be symmetric"));
    }
    Ok(())
}
fn riccati(
    a: &(impl super::MatrixInput + ?Sized),
    b: &(impl super::MatrixInput + ?Sized),
    q: &(impl super::MatrixInput + ?Sized),
    r: &(impl super::MatrixInput + ?Sized),
    discrete: bool,
) -> Result<Matrix> {
    let a = square(a)?;
    let b = dense(b)?;
    let q = square(q)?;
    let r = square(r)?;
    let n = a.nrows();
    let m = b.ncols();
    if b.nrows() != n || q.nrows() != n || r.nrows() != m {
        return Err(Error::DimensionMismatch);
    }
    if n > 128 {
        return Err(Error::InvalidInput("Riccati state order exceeds 128"));
    }
    validate_symmetric(&q)?;
    validate_symmetric(&r)?;
    let chol = r
        .llt(faer::Side::Lower)
        .ok()
        .ok_or(Error::InvalidInput("R must be positive definite"))?;
    let g = &b * chol.solve(&b.transpose());
    let h = Mat::from_fn(2 * n, 2 * n, |i, j| match (i < n, j < n) {
        (true, true) => a[(i, j)],
        (true, false) => {
            if discrete {
                0.
            } else {
                -g[(i, j - n)]
            }
        }
        (false, true) => -q[(i - n, j)],
        (false, false) => {
            if discrete {
                f64::from(i == j)
            } else {
                -a[(j - n, i - n)]
            }
        }
    });
    let pencil = if discrete {
        Mat::from_fn(2 * n, 2 * n, |i, j| match (i < n, j < n) {
            (true, true) => f64::from(i == j),
            (true, false) => g[(i, j - n)],
            (false, true) => 0.,
            (false, false) => a[(j - n, i - n)],
        })
    } else {
        Mat::identity(2 * n, 2 * n)
    };
    let f = super::ordqz(&native(&h), &native(&pencil), |alpha, beta| {
        if discrete {
            alpha.re.hypot(alpha.im) < beta.re.hypot(beta.im)
        } else {
            alpha.re * beta.re + alpha.im * beta.im < 0.
        }
    })?;
    if f.selected != n {
        return Err(Error::NumericalFailure(
            "Riccati pencil has no isolated stable subspace of the required dimension",
        ));
    }
    let z = complex_dense(&f.z)?;
    let u1 = z.as_ref().submatrix(0, 0, n, n).to_owned();
    let u2 = z.as_ref().submatrix(n, 0, n, n).to_owned();
    let x = u1
        .transpose()
        .to_owned()
        .solve_checked(&u2.transpose().to_owned())
        .ok_or(Error::NumericalFailure("singular stable Riccati subspace"))?
        .transpose()
        .to_owned();
    if x.iter().any(|z| !z.re.is_finite() || !z.im.is_finite())
        || x.iter().map(|z| z.im.abs()).fold(0., f64::max) > 1e-9 * (1. + x.norm_l2())
    {
        return Err(Error::NumericalFailure(
            "Riccati solution is nonfinite or not real",
        ));
    }
    let x = x.map(|z| z.re);
    if (&x - x.transpose()).norm_l2() > 1e-8 * (1. + x.norm_l2()) {
        return Err(Error::NumericalFailure(
            "Riccati stable subspace does not give a symmetric solution",
        ));
    }
    let x = (&x + x.transpose()) * Scale(0.5);
    let (residual, closed) = if discrete {
        let denominator = &r + b.transpose() * &x * &b;
        let gain = denominator
            .llt(faer::Side::Lower)
            .ok()
            .ok_or(Error::NumericalFailure(
                "discrete Riccati control denominator is not positive definite",
            ))?
            .solve(&(b.transpose() * &x * &a));
        (
            a.transpose() * &x * &a - &x - a.transpose() * &x * &b * &gain + &q,
            &a - &b * gain,
        )
    } else {
        (
            a.transpose() * &x + &x * &a - &x * &g * &x + &q,
            &a - &g * &x,
        )
    };
    let scale = 1.
        + q.norm_l2()
        + a.norm_l2() * x.norm_l2() * (1. + a.norm_l2())
        + g.norm_l2() * x.norm_l2().powi(2);
    if !scale.is_finite() || residual.norm_l2() > 1e-8 * scale {
        return Err(Error::NumericalFailure(
            "Riccati residual exceeds numerical bound",
        ));
    }
    let eigen = super::eigvals(&native(&closed))?;
    if eigen.iter().any(|z| {
        if discrete {
            z.re.hypot(z.im) >= 1.
        } else {
            z.re >= 0.
        }
    }) {
        return Err(Error::NumericalFailure(
            "Riccati solution is not stabilizing",
        ));
    }
    rows(&x)
}
/// Solves the stabilizing Riccati equation `Aᵀ X+X A-X B R⁻¹ Bᵀ X+Q=0`.
///
/// ## Parameters
///
/// `a` is n × n, `b` is n × m, `q` is n × n, `r` is m × m.
/// Q and R must be symmetric; R must be positive definite.
/// SciPy descriptor E, cross term S, and balancing options are omitted.
///
/// ## Domain
///
/// Finite nonempty real matrices, n ≤128, m ≤256.
/// A stable invariant subspace must be isolated and have nonsingular top block.
/// NaN/infinity are rejected.
///
/// ## Returns
///
/// Symmetric real n × n stabilizing solution X.
///
/// ## Errors
///
/// Invalid input, nonpositive R/control denominator, inseparable pencil,
/// singular subspace, nonreal/asymmetric solution, failed residual or stability.
///
/// ## Numerical behavior
///
/// Uses a Hamiltonian/symplectic pencil through private faer generalized
/// Schur stages, ordered by half-plane/unit disk. Native pencil assembly
/// and shared LU/Cholesky recover X. O(n³+m³) work; QZ sweep bounds depend
/// on order. A relative 1e-8 residual and closed-loop stability are checked.
/// Unbalanced ill-conditioned problems may fail where SciPy succeeds.
/// Positive-definite R is stricter than SciPy continuous ARE.
///
/// # Examples
///
/// ```
/// let x = sci_linalg::solve_continuous_are(&[vec![1.]], &[vec![1.]],
///     &[vec![1.]], &[vec![1.]])?;
/// assert!(x[0][0]>0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `solve_continuous_are`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_continuous_are.html).
pub fn solve_continuous_are(
    a: &(impl super::MatrixInput + ?Sized),
    b: &(impl super::MatrixInput + ?Sized),
    q: &(impl super::MatrixInput + ?Sized),
    r: &(impl super::MatrixInput + ?Sized),
) -> Result<Matrix> {
    riccati(a, b, q, r, false)
}
/// Solves the stabilizing Riccati equation `Aᵀ X A-X-Aᵀ X B (R+Bᵀ X B)⁻¹ Bᵀ X A+Q=0`.
///
/// ## Parameters
///
/// `a` is n × n, `b` is n × m, `q` is n × n, `r` is m × m.
/// Q and R must be symmetric; R must be positive definite.
/// SciPy descriptor E, cross term S, and balancing options are omitted.
///
/// ## Domain
///
/// Finite nonempty real matrices, n ≤128, m ≤256.
/// A stable invariant subspace must be isolated and have nonsingular top block.
/// NaN/infinity are rejected.
///
/// ## Returns
///
/// Symmetric real n × n stabilizing solution X.
///
/// ## Errors
///
/// Invalid input, nonpositive R/control denominator, inseparable pencil,
/// singular subspace, nonreal/asymmetric solution, failed residual or stability.
///
/// ## Numerical behavior
///
/// Uses a Hamiltonian/symplectic pencil through private faer generalized
/// Schur stages, ordered by half-plane/unit disk. Native pencil assembly
/// and shared LU/Cholesky recover X. O(n³+m³) work; QZ sweep bounds depend
/// on order. A relative 1e-8 residual and closed-loop stability are checked.
/// Unbalanced ill-conditioned problems may fail where SciPy succeeds.
/// Positive-definite R is stricter than SciPy continuous ARE.
///
/// # Examples
///
/// ```
/// let x = sci_linalg::solve_discrete_are(&[vec![1.]], &[vec![1.]],
///     &[vec![1.]], &[vec![1.]])?;
/// assert!(x[0][0]>0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `solve_discrete_are`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_discrete_are.html).
pub fn solve_discrete_are(
    a: &(impl super::MatrixInput + ?Sized),
    b: &(impl super::MatrixInput + ?Sized),
    q: &(impl super::MatrixInput + ?Sized),
    r: &(impl super::MatrixInput + ?Sized),
) -> Result<Matrix> {
    riccati(a, b, q, r, true)
}
