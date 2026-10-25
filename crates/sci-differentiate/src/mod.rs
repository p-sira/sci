use crate::{
    Error, Result,
    error::{finite, finite_slice},
};

const WORK_LIMIT: usize = 1_000_000;

fn validate_work(count: Option<usize>) -> Result<()> {
    if count.is_some_and(|count| count <= WORK_LIMIT) {
        Ok(())
    } else {
        Err(Error::InvalidInput("finite-difference work limit exceeded"))
    }
}

fn perturbations(point: &[f64], step: f64) -> Result<Vec<(f64, f64)>> {
    finite_slice(point)?;
    point.iter().map(|&value| neighbors(value, step)).collect()
}

fn validate_step(step: f64) -> Result<()> {
    if step.is_finite() && step > 0.0 {
        Ok(())
    } else {
        Err(Error::InvalidStep)
    }
}

fn neighbors(value: f64, step: f64) -> Result<(f64, f64)> {
    let upper = value + step;
    let lower = value - step;
    if !value.is_finite()
        || !upper.is_finite()
        || !lower.is_finite()
        || upper == value
        || lower == value
    {
        return Err(Error::InvalidInput(
            "finite-difference step must produce distinct finite neighbors",
        ));
    }
    Ok((upper, lower))
}

fn difference(upper: f64, lower: f64, coordinates: (f64, f64)) -> Result<f64> {
    difference_with_factor(upper, lower, coordinates, 1.0)
}

fn difference_with_factor(
    upper: f64,
    lower: f64,
    coordinates: (f64, f64),
    factor: f64,
) -> Result<f64> {
    finite(upper)?;
    finite(lower)?;
    let width = coordinates.0 - coordinates.1;
    let delta = upper - lower;
    finite(if width.is_finite() {
        if delta.is_finite() {
            let numerator = factor * delta;
            if numerator.is_finite() {
                numerator / width
            } else {
                (delta / width) * factor
            }
        } else {
            (upper / width - lower / width) * factor
        }
    } else {
        // Both spans may exceed the finite range; their ratio can still fit.
        let half_width = 0.5 * coordinates.0 - 0.5 * coordinates.1;
        ((0.5 * upper - 0.5 * lower) / half_width) * factor
    })
}

/// Approximates `f′(x)` by the secant `(f(x₊) − f(x₋))/(x₊ − x₋)`,
/// where `x₊ = fl(x+step)` and `x₋ = fl(x−step)` are the actual sampled coordinates.
///
/// ## Parameters
///
/// `x` is a finite evaluation point; `step` is a finite positive absolute
/// displacement, measured in the same units as `x`. The callback receives each
/// neighboring scalar once (two evaluations).
///
/// ## Returns
///
/// The slope in output units per input unit.
///
/// ## Domain
///
/// Coordinates and steps must be finite, and the step must be positive.
/// All sampled callback values and results must be finite; NaN and infinity
/// produce errors. The callback must be defined on both sides of each coordinate.
///
/// ## Numerical behavior
///
/// Uses the actual coordinate span rather than the requested `2*step`,
/// preserving linear secants when increments round asymmetrically. Symmetric
/// stencils have `O(step²)` truncation error for smooth functions. Asymmetric
/// stencils estimate the derivative at their midpoint, adding error proportional
/// to its displacement from the requested coordinate. Callback rounding still
/// increases error as the step shrinks; no error estimate or convergence
/// guarantee is supplied. See the module's work limits and SciPy adaptations.
///
/// ## Errors
///
/// Returns an error for invalid or unrepresentable perturbations, nonfinite
/// callback values, or a nonfinite result.
///
/// # References
///
/// [SciPy differentiation](https://docs.scipy.org/doc/scipy/reference/differentiate.html).
///
/// # Examples
///
/// ```
/// let slope = sci_differentiate::derivative(|x| x * x, 3.0, 1e-5)?;
/// assert!((slope - 6.0).abs() < 1e-8);
/// # Ok::<(), sci_core::Error>(())
/// ```
pub fn derivative<F>(function: F, x: f64, step: f64) -> Result<f64>
where
    F: Fn(f64) -> f64,
{
    validate_step(step)?;
    let (upper, lower) = neighbors(x, step)?;
    difference(function(upper), function(lower), (upper, lower))
}

/// Approximates `∇f[j] = ∂f/∂xⱼ` by coordinate secants
/// `(f(x[j←x₊]) − f(x[j←x₋]))/(x₊ − x₋)`, varying only coordinate `j`.
/// Here `x₊` and `x₋` are the actual rounded neighboring coordinate values.
///
/// ## Parameters
///
/// `point` contains finite coordinates. `step` is a finite positive absolute
/// displacement. The function is evaluated twice per coordinate. Returns one
/// derivative per coordinate; empty points return empty gradients.
///
/// ## Returns
///
/// A vector of `point.len()` partial derivatives in coordinate order, in output
/// units per corresponding input unit.
///
/// ## Domain
///
/// Coordinates and steps must be finite, and the step must be positive.
/// All sampled callback values and results must be finite; NaN and infinity
/// produce errors. The callback must be defined on both sides of each coordinate.
///
/// ## Numerical behavior
///
/// Uses the actual coordinate span rather than the requested `2*step`,
/// preserving linear secants when increments round asymmetrically. Symmetric
/// stencils have `O(step²)` truncation error for smooth functions. Asymmetric
/// stencils estimate the derivative at their midpoint, adding error proportional
/// to its displacement from the requested coordinate. Callback rounding still
/// increases error as the step shrinks; no error estimate or convergence
/// guarantee is supplied. See the module's work limits and SciPy adaptations.
///
/// ## Errors
///
/// Returns an error for invalid perturbations, exceeded work limits, or
/// nonfinite evaluations/results.
///
/// # References
///
/// [SciPy differentiation](https://docs.scipy.org/doc/scipy/reference/differentiate.html).
///
/// # Examples
///
/// ```
/// let gradient = sci_differentiate::gradient(
///     |x| x[0] * x[0] + 3.0 * x[1], &[2.0, 4.0], 1e-5,
/// )?;
/// assert!((gradient[0] - 4.0).abs() < 1e-8);
/// assert!((gradient[1] - 3.0).abs() < 1e-8);
/// # Ok::<(), sci_core::Error>(())
/// ```
pub fn gradient<F>(function: F, point: &[f64], step: f64) -> Result<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    validate_step(step)?;
    validate_work(point.len().checked_mul(2))?;
    let offsets = perturbations(point, step)?;
    let mut sample = point.to_vec();
    let mut result = Vec::with_capacity(point.len());
    for (index, &value) in point.iter().enumerate() {
        let (upper, lower) = offsets[index];
        sample[index] = upper;
        let positive = function(&sample);
        sample[index] = lower;
        let negative = function(&sample);
        sample[index] = value;
        result.push(difference(positive, negative, (upper, lower))?);
    }
    Ok(result)
}

/// Approximates `J[i][j] = ∂fᵢ/∂xⱼ` by coordinate secants
/// `(fᵢ(x[j←x₊]) − fᵢ(x[j←x₋]))/(x₊ − x₋)`, varying only coordinate `j`.
/// Here `x₊` and `x₋` are the actual rounded neighboring coordinate values.
///
/// ## Parameters
///
/// `point` contains finite coordinates; `step` is finite and positive.
/// Rows correspond to outputs and columns to inputs, matching
/// `sci_linalg::matvec`. For empty input the callback is evaluated once
/// to determine the number of zero-column rows. Otherwise there are exactly
/// `2 * point.len()` evaluations.
///
/// ## Returns
///
/// A row-major matrix of shape `(output.len(), point.len())`. Each entry has
/// units of its output divided by its input coordinate's units.
///
/// ## Domain
///
/// Coordinates and steps must be finite, and the step must be positive.
/// All sampled callback values and results must be finite; NaN and infinity
/// produce errors. The callback must be defined on both sides of each coordinate.
///
/// ## Numerical behavior
///
/// Uses the actual coordinate span rather than the requested `2*step`,
/// preserving linear secants when increments round asymmetrically. Symmetric
/// stencils have `O(step²)` truncation error for smooth functions. Asymmetric
/// stencils estimate the derivative at their midpoint, adding error proportional
/// to its displacement from the requested coordinate. Callback rounding still
/// increases error as the step shrinks; no error estimate or convergence
/// guarantee is supplied. See the module's work limits and SciPy adaptations.
///
/// ## Errors
///
/// Returns an error for invalid perturbations, exceeded work limits, changing output dimensions,
/// or nonfinite evaluations/results.
///
/// # References
///
/// [SciPy differentiation](https://docs.scipy.org/doc/scipy/reference/differentiate.html).
///
/// # Examples
///
/// ```
/// let jacobian = sci_differentiate::jacobian(
///     |x| vec![x[0] * x[1]], &[2.0, 3.0], 1e-5,
/// )?;
/// assert!((jacobian[0][0] - 3.0).abs() < 1e-8);
/// assert!((jacobian[0][1] - 2.0).abs() < 1e-8);
/// # Ok::<(), sci_core::Error>(())
/// ```
pub fn jacobian<F>(function: F, point: &[f64], step: f64) -> Result<Vec<Vec<f64>>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    validate_step(step)?;
    validate_work(point.len().checked_mul(2))?;
    let offsets = perturbations(point, step)?;
    if point.is_empty() {
        let values = function(point);
        validate_work(Some(values.len()))?;
        for value in &values {
            finite(*value)?;
        }
        return Ok(vec![Vec::new(); values.len()]);
    }
    let mut sample = point.to_vec();
    let mut result: Option<Vec<Vec<f64>>> = None;
    for (column, &value) in point.iter().enumerate() {
        let (upper, lower) = offsets[column];
        sample[column] = upper;
        let positive = function(&sample);
        sample[column] = lower;
        let negative = function(&sample);
        sample[column] = value;
        if positive.len() != negative.len() {
            return Err(Error::InconsistentOutputDimension {
                positive: positive.len(),
                negative: negative.len(),
            });
        }
        validate_work(positive.len().checked_mul(point.len()))?;
        let matrix = result.get_or_insert_with(|| vec![vec![0.0; point.len()]; positive.len()]);
        if matrix.len() != positive.len() {
            return Err(Error::InconsistentOutputDimension {
                positive: positive.len(),
                negative: matrix.len(),
            });
        }
        for (row, (&upper, &lower)) in positive.iter().zip(&negative).enumerate() {
            matrix[row][column] = difference(upper, lower, offsets[column])?;
        }
    }
    Ok(result.unwrap())
}

/// Approximates the Hessian `H[i][j] = ∂²f/(∂xᵢ ∂xⱼ)`.
///
/// Diagonal entries use `2[(f(x₊)−f(x))/h₊ − (f(x)−f(x₋))/h₋]/(h₊+h₋)`,
/// where `h₊ = x₊−x` and `h₋ = x−x₋` are the actual rounded coordinate increments.
/// Off-diagonal entries use `(f(++)−f(+-)−f(-+)+f(--))/(wᵢ wⱼ)`,
/// where `w` denotes the actual coordinate spans and signs select neighbors.
///
/// ## Parameters
///
/// `function` maps a coordinate slice to a scalar. `point` contains finite
/// coordinates; `step` is a finite positive absolute displacement in each
/// coordinate's units. Extra arguments can be captured in the closure.
///
/// ## Domain
///
/// The callback must be smooth and finite at the center, axial neighbors, and
/// mixed neighbors. NaN/infinite inputs, evaluations, and results are errors.
/// Empty points return an empty matrix without evaluating the callback.
///
/// ## Returns
///
/// A symmetric row-major matrix of shape `(point.len(), point.len())`, suitable
/// for `sci_linalg::matvec`. Entry `(i,j)` has units of the function output
/// divided by the product of coordinate `i` and coordinate `j` units.
///
/// ## Errors
///
/// Returns an error for invalid/unrepresentable steps, nonfinite evaluations
/// or arithmetic, or exceeded work limits. All perturbations and work bounds
/// are checked before the first callback evaluation.
///
/// ## Numerical behavior
///
/// Uses three-point divided differences and rectangular secants with exactly
/// `2n² + 1` evaluations for `n > 0`. Actual coordinate spans avoid false
/// curvature from rounded steps. Symmetric stencils have `O(step²)` truncation
/// error; diagonal asymmetry adds error proportional to `|h₊−h₋|`, and mixed
/// entries estimate curvature at their rounded rectangle's midpoint. Symmetric
/// matrix entries reuse the same estimate. Roundoff grows approximately as `1/step²`; at unit scale a step
/// near `f64::EPSILON.powf(0.25)` is a starting point, not an accuracy guarantee.
/// Large constant offsets can destroy precision. Unlike SciPy's nested adaptive
/// Jacobians, this uses direct fixed stencils without diagnostics or batching.
/// Calls are limited to 1,000,000 evaluations and output entries.
///
/// # Examples
///
/// ```
/// let h = sci_differentiate::hessian(
///     |x| x[0] * x[0] + x[0] * x[1], &[2.0, 3.0], 1e-3,
/// )?;
/// assert!((h[0][0] - 2.0).abs() < 1e-7);
/// assert!((h[0][1] - 1.0).abs() < 1e-7);
/// assert_eq!(h[0][1], h[1][0]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy Hessian definition and conventions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.differentiate.hessian.html).
pub fn hessian<F>(function: F, point: &[f64], step: f64) -> Result<Vec<Vec<f64>>>
where
    F: Fn(&[f64]) -> f64,
{
    validate_step(step)?;
    let n = point.len();
    let entries = n.checked_mul(n);
    validate_work(entries)?;
    validate_work(
        entries
            .and_then(|v| v.checked_mul(2))
            .and_then(|v| v.checked_add(1)),
    )?;
    let offsets = perturbations(point, step)?;
    if n == 0 {
        return Ok(Vec::new());
    }
    let center = finite(function(point))?;
    let mut sample = point.to_vec();
    let mut result = vec![vec![0.0; n]; n];
    for i in 0..n {
        let (upper, lower) = offsets[i];
        sample[i] = upper;
        let positive = finite(function(&sample))?;
        sample[i] = lower;
        let negative = finite(function(&sample))?;
        sample[i] = point[i];
        // Unequal three-point divided differences remove the false curvature
        // caused by rounded, asymmetric coordinate increments.
        let forward = difference(positive, center, (upper, point[i]))?;
        let backward = difference(center, negative, (point[i], lower))?;
        result[i][i] = difference_with_factor(forward, backward, (upper, lower), 2.0)?;
        for j in 0..i {
            sample[i] = upper;
            sample[j] = offsets[j].0;
            let pp = finite(function(&sample))?;
            sample[j] = offsets[j].1;
            let pm = finite(function(&sample))?;
            sample[i] = lower;
            sample[j] = offsets[j].0;
            let mp = finite(function(&sample))?;
            sample[j] = offsets[j].1;
            let mm = finite(function(&sample))?;
            sample[i] = point[i];
            sample[j] = point[j];
            let value = difference(
                difference(pp, pm, offsets[j])?,
                difference(mp, mm, offsets[j])?,
                offsets[i],
            )?;
            result[i][j] = value;
            result[j][i] = value;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn differentiates_scalar_function() {
        let value = derivative(|x| x.powi(3), 2.0, 1e-5).unwrap();
        assert!((value - 12.0).abs() < 1e-8);
    }

    #[test]
    fn calculates_gradient() {
        let value = gradient(|x| x[0].powi(2) + 3.0 * x[1], &[2.0, 4.0], 1e-5).unwrap();
        assert!((value[0] - 4.0).abs() < 1e-8);
        assert!((value[1] - 3.0).abs() < 1e-8);
    }

    #[test]
    fn calculates_jacobian() {
        let value = jacobian(|x| vec![x[0] * x[1], x[0] + x[1]], &[2.0, 3.0], 1e-5).unwrap();
        assert!((value[0][0] - 3.0).abs() < 1e-8);
        assert!((value[0][1] - 2.0).abs() < 1e-8);
        assert!((value[1][0] - 1.0).abs() < 1e-8);
        assert!((value[1][1] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn rejects_invalid_step() {
        assert_eq!(derivative(|x| x, 1.0, 0.0), Err(Error::InvalidStep));
        assert_eq!(gradient(|_| 0.0, &[], f64::NAN), Err(Error::InvalidStep));
    }

    #[test]
    fn supports_zero_dimensional_jacobian() {
        assert_eq!(
            jacobian(|_| vec![1.0, 2.0], &[], 1e-5).unwrap(),
            vec![vec![], vec![]]
        );
    }
}
