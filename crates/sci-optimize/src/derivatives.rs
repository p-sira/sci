//! Optimization objectives and SciPy forward-difference diagnostics.
use crate::error::{finite, finite_slice};
use crate::{Error, Result, linalg::Matrix};

fn point(x: &[f64], min: usize, max: usize) -> Result<()> {
    if !(min..=max).contains(&x.len()) {
        return Err(Error::InvalidInput("unsupported objective dimension"));
    }
    finite_slice(x)
}

fn curvature(x: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
    let n = x.len();
    let mut diagonal = vec![0.0; n];
    let mut off = Vec::with_capacity(n - 1);
    for (i, pair) in x.windows(2).enumerate() {
        diagonal[i] =
            finite(diagonal[i] + finite(1200.0 * pair[0] * pair[0] - 400.0 * pair[1] + 2.0)?)?;
        diagonal[i + 1] += 200.0;
        off.push(finite(-400.0 * pair[0])?);
    }
    Ok((diagonal, off))
}

/// Evaluates the Rosenbrock objective `R(x)=Σᵢ [100(x[i+1]−x[i]²)²+(1−x[i])²]`.
///
/// ## Parameters
///
/// `x` is one coordinate vector. SciPy array broadcasting is adapted to a slice.
/// ## Domain
///
/// Finite coordinates, at most 1,000,000 entries. Empty or singleton vectors give the empty sum zero. NaN and infinity are rejected.
/// ## Returns
///
/// The nonnegative dimensionless objective; the all-ones vector is a minimizer.
/// ## Errors
///
/// Unsupported size, nonfinite inputs, or arithmetic overflow.
/// ## Numerical behavior
///
/// O(n) direct polynomial evaluation. Fused multiply-add preserves the residual
/// `x[i+1]−x[i]²` near the curved valley; binary64 rounding still applies.
/// # Examples
///
/// ```
/// assert_eq!(sci_optimize::rosen(&[1.0,1.0])?,0.0);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy rosen](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.rosen.html).
pub fn rosen(x: &[f64]) -> Result<f64> {
    point(x, 0, 1_000_000)?;
    x.windows(2).try_fold(0.0, |sum, p| {
        finite(sum + 100.0 * (-p[0]).mul_add(p[0], p[1]).powi(2) + (1.0 - p[0]).powi(2))
    })
}

/// Computes `∇R(x)`; each pair adds `−400xᵢ(xᵢ₊₁−xᵢ²)+2(xᵢ−1)` and `200(xᵢ₊₁−xᵢ²)`.
///
/// ## Parameters
///
/// `x` follows [`rosen`]; one-dimensional input only.
/// ## Domain
///
/// Finite coordinates with 2..=1,000,000 entries. NaN/infinity and shorter vectors are rejected, matching the supported SciPy derivative domain.
/// ## Returns
///
/// A gradient in coordinate order, with length `x.len()`.
/// ## Errors
///
/// Invalid dimensions/values or nonfinite polynomial arithmetic.
/// ## Numerical behavior
///
/// O(n) analytical formula evaluated in f64 with fused multiply-add for the
/// valley residual, shared with [`rosen`]. No numerical differencing.
/// # Examples
///
/// ```
/// assert_eq!(sci_optimize::rosen_der(&[1.0,1.0])?,vec![0.0,0.0]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy rosen_der](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.rosen_der.html).
pub fn rosen_der(x: &[f64]) -> Result<Vec<f64>> {
    point(x, 2, 1_000_000)?;
    let mut out = vec![0.0; x.len()];
    for (i, p) in x.windows(2).enumerate() {
        let residual = finite((-p[0]).mul_add(p[0], p[1]))?;
        out[i] = finite(out[i] - 400.0 * p[0] * residual + 2.0 * (p[0] - 1.0))?;
        out[i + 1] = finite(out[i + 1] + 200.0 * residual)?;
    }
    Ok(out)
}

/// Computes `∇²R(x)`; each pair contributes diagonal entries `1200xᵢ²−400xᵢ₊₁+2`, `200`, and off-diagonal `−400xᵢ`.
///
/// ## Parameters
///
/// `x` follows [`rosen`]. Rows and columns follow coordinate order.
/// ## Domain
///
/// Finite coordinates with 2..=256 entries. NaN/infinity are rejected.
/// ## Returns
///
/// A symmetric `n × n` shared row matrix with only three nonzero diagonals.
/// ## Errors
///
/// Invalid dimensions/values or nonfinite curvature arithmetic.
/// ## Numerical behavior
///
/// O(n²) output storage, O(n) polynomial evaluations; shares curvature formulas with [`rosen_hess_prod`].
/// # Examples
///
/// ```
/// assert_eq!(sci_optimize::rosen_hess(&[1.0,1.0])?,vec![vec![802.0,-400.0],vec![-400.0,200.0]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy rosen_hess](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.rosen_hess.html).
pub fn rosen_hess(x: &[f64]) -> Result<Matrix> {
    point(x, 2, 256)?;
    let (d, o) = curvature(x)?;
    let mut out = vec![vec![0.0; x.len()]; x.len()];
    for (i, v) in d.into_iter().enumerate() {
        out[i][i] = v;
    }
    for (i, v) in o.into_iter().enumerate() {
        out[i][i + 1] = v;
        out[i + 1][i] = v;
    }
    Ok(out)
}

/// Computes the Hessian-vector product `p ↦ ∇²R(x)p`.
///
/// ## Parameters
///
/// `x` follows [`rosen`]; `p` is a vector of the same length and coordinate order.
/// ## Domain
///
/// Finite vectors with 2..=1,000,000 entries. NaN and infinity are rejected.
/// ## Returns
///
/// The product in coordinate order, of length `x.len()`.
/// ## Errors
///
/// Dimension mismatch, unsupported size, invalid values, or nonfinite arithmetic.
/// ## Numerical behavior
///
/// O(n) time and storage, using the tridiagonal analytical Hessian without dense allocation.
/// # Examples
///
/// ```
/// assert_eq!(sci_optimize::rosen_hess_prod(&[1.0,1.0],&[1.0,0.0])?,vec![802.0,-400.0]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy rosen_hess_prod](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.rosen_hess_prod.html).
pub fn rosen_hess_prod(x: &[f64], p: &[f64]) -> Result<Vec<f64>> {
    point(x, 2, 1_000_000)?;
    if x.len() != p.len() {
        return Err(Error::DimensionMismatch);
    }
    finite_slice(p)?;
    let (d, o) = curvature(x)?;
    let mut out = d
        .iter()
        .zip(p)
        .map(|(d, p)| finite(d * p))
        .collect::<Result<Vec<_>>>()?;
    for (i, v) in o.into_iter().enumerate() {
        out[i] = finite(out[i] + v * p[i + 1])?;
        out[i + 1] = finite(out[i + 1] + v * p[i])?;
    }
    Ok(out)
}

/// Approximates the Jacobian by `J[i,j]=(fᵢ(x+εⱼeⱼ)−fᵢ(x))/((xⱼ+εⱼ)−xⱼ)`.
///
/// ## Parameters
///
/// `function` returns an output vector; wrap scalar objectives as `vec![value]`. `epsilon` has one signed step to broadcast, or one per coordinate. Extra SciPy args are captured in the closure. Output axes are function outputs then coordinates.
/// ## Domain
///
/// Finite coordinates and nonempty outputs of 1..=512 entries; finite nonzero steps. A step must change its coordinate and yield a finite perturbed value. NaN/infinity callback values are rejected.
/// ## Returns
///
/// An `m × n` shared Jacobian matrix; scalar objectives produce one row.
/// ## Errors
///
/// Invalid steps/sizes/values, step-length mismatch, changing callback output dimension, or arithmetic overflow. Collapsed steps return an error instead of SciPy’s automatic fallback.
/// ## Numerical behavior
///
/// Exactly n+1 callback evaluations. First-order truncation and subtraction roundoff depend on step size; the denominator uses the actual representable increment. Callback execution cost is caller-controlled.
/// # Examples
///
/// ```
/// let jac=sci_optimize::approx_fprime(|x|vec![x[0]*x[0]],&[2.0],&[1e-5])?;
/// assert!((jac[0][0]-4.00001).abs()<1e-8);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy approx_fprime](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.approx_fprime.html).
pub fn approx_fprime(
    mut function: impl FnMut(&[f64]) -> Vec<f64>,
    x: &[f64],
    epsilon: &[f64],
) -> Result<Matrix> {
    point(x, 1, 512)?;
    if epsilon.len() != 1 && epsilon.len() != x.len() {
        return Err(Error::DimensionMismatch);
    }
    if epsilon.iter().any(|v| !v.is_finite() || *v == 0.0) {
        return Err(Error::InvalidStep);
    }
    let mut steps = Vec::with_capacity(x.len());
    for (i, &v) in x.iter().enumerate() {
        let next = finite(v + epsilon[if epsilon.len() == 1 { 0 } else { i }])?;
        let dx = finite(next - v)?;
        if dx == 0.0 {
            return Err(Error::InvalidStep);
        }
        steps.push((next, dx));
    }
    let baseline = function(x);
    point(&baseline, 1, 512)?;
    let mut out = vec![vec![0.0; x.len()]; baseline.len()];
    let mut perturbed = x.to_vec();
    for (j, (next, dx)) in steps.into_iter().enumerate() {
        perturbed[j] = next;
        let values = function(&perturbed);
        perturbed[j] = x[j];
        if values.len() != baseline.len() {
            return Err(Error::InconsistentOutputDimension {
                positive: values.len(),
                negative: baseline.len(),
            });
        }
        finite_slice(&values)?;
        for (i, (value, base)) in values.iter().zip(&baseline).enumerate() {
            out[i][j] = finite(finite(value - base)? / dx)?;
        }
    }
    Ok(out)
}

/// Measures analytical-gradient error `‖gradient(x)−approx_fprime(function,x)‖₂`.
///
/// ## Parameters
///
/// `function` is scalar; `gradient` returns coordinate-ordered components. `epsilon` follows [`approx_fprime`]. Implements SciPy’s `direction="all"`; random-direction checks and rng are omitted.
/// ## Domain
///
/// Finite coordinates of 1..=512 entries, finite same-length gradient, and valid signed steps. NaN/infinity evaluations are rejected.
/// ## Returns
///
/// A nonnegative Euclidean gradient discrepancy in objective units per coordinate unit.
/// ## Errors
///
/// The errors of [`approx_fprime`], gradient shape mismatch, or nonfinite evaluations/results.
/// ## Numerical behavior
///
/// Uses n+1 scalar objective evaluations and one gradient evaluation; hypot accumulation avoids squared-norm overflow. Finite-difference truncation remains part of the reported discrepancy.
/// # Examples
///
/// ```
/// let error=sci_optimize::check_grad(|x|x[0]*x[0],|x|vec![2.0*x[0]],&[2.0],&[1e-5])?;
/// assert!(error<2e-5);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy check_grad](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.check_grad.html).
pub fn check_grad(
    mut function: impl FnMut(&[f64]) -> f64,
    mut gradient: impl FnMut(&[f64]) -> Vec<f64>,
    x: &[f64],
    epsilon: &[f64],
) -> Result<f64> {
    point(x, 1, 512)?;
    let g = gradient(x);
    if g.len() != x.len() {
        return Err(Error::DimensionMismatch);
    }
    finite_slice(&g)?;
    let jac = approx_fprime(|x| vec![function(x)], x, epsilon)?;
    g.iter()
        .zip(&jac[0])
        .try_fold(0.0_f64, |norm, (a, b)| finite(norm.hypot(finite(a - b)?)))
}
