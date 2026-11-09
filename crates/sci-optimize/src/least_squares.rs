//! Dense nonlinear least squares with a rectangular trust-region dogleg step.
//!
//! SciPy's `least_squares(method="dogbox")` is the numerical reference. This
//! implementation uses a dense SVD to obtain the Gauss–Newton step and a
//! bound-constrained Cauchy step to handle singular Jacobians. Bounds may be
//! infinite. Nonfinite trial residuals shrink the trust region; initial
//! residuals and all Jacobian evaluations must be finite.

use crate::{Error, Result};
use nalgebra::{DMatrix, DVector};
use sci_core::Matrix;

const MAX_WORK: u64 = 16_777_216;

/// Stopping controls for the dense rectangular trust-region solver.
#[derive(Clone, Copy, Debug)]
pub struct DogboxOptions {
    /// Maximum trust-region attempts (including rejected steps); zero only initializes.
    pub max_iterations: u64,
    /// Stopping threshold for the infinity norm of the free gradient.
    pub gradient_tolerance: f64,
    /// Relative decrease threshold, when the local model predicts the decrease.
    pub cost_tolerance: f64,
    /// Step threshold: `||step||₂ < tolerance * (tolerance + ||point||₂)`.
    pub step_tolerance: f64,
}

impl Default for DogboxOptions {
    /// Uses 1,000 attempts and `1e-8` for each stopping tolerance.
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            gradient_tolerance: 1e-8,
            cost_tolerance: 1e-8,
            step_tolerance: 1e-8,
        }
    }
}

/// Reason for stopping the least-squares solver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DogboxStatus {
    /// The infinity norm of the gradient on free coordinates met the threshold.
    Gradient,
    /// The cost decreased by less than the relative threshold.
    Cost,
    /// The step length met the relative threshold.
    Step,
    /// Both cost and step tolerances were satisfied.
    CostAndStep,
    /// The iteration budget was exhausted.
    IterationLimit,
    /// No representable feasible improving step was found, or its model failed.
    NoProgress,
}

/// Outcome of minimizing `0.5 * sum_i(residual_i(x)^2)`.
#[derive(Clone, Debug, PartialEq)]
pub struct DogboxResult {
    /// Final feasible parameter vector.
    pub point: Vec<f64>,
    /// Half the sum of squared residuals.
    pub cost: f64,
    /// Residual vector at the final point.
    pub residuals: Vec<f64>,
    /// Unprojected `J^T residuals` gradient, in parameter order.
    pub gradient: Vec<f64>,
    /// Infinity norm of the gradient after bound-active components are excluded.
    pub optimality: f64,
    /// Per-coordinate active bounds: `-1` lower, `1` upper, `0` interior.
    pub active_mask: Vec<i8>,
    /// Number of residual callback calls (including numerical differentiation).
    pub function_evaluations: u64,
    /// Number of Jacobian evaluations (analytical or finite difference).
    pub jacobian_evaluations: u64,
    /// Trust-region attempts, including rejected steps.
    pub iterations: u64,
    /// Termination reason.
    pub status: DogboxStatus,
}

struct ResidualEvaluator<F> {
    function: F,
    calls: u64,
    expected: Option<usize>,
}

impl<F: Fn(&[f64]) -> Result<Vec<f64>>> ResidualEvaluator<F> {
    fn evaluate_trial(&mut self, point: &[f64]) -> Result<Option<Vec<f64>>> {
        let residuals = (self.function)(point)?;
        self.calls += 1;
        if residuals.is_empty() {
            return Err(Error::InvalidInput(
                "least squares needs at least one residual",
            ));
        }
        if self.expected.is_some_and(|m| residuals.len() != m) {
            return Err(Error::DimensionMismatch);
        }
        if residuals.iter().any(|v| !v.is_finite()) {
            return Ok(None);
        }
        self.expected = Some(residuals.len());
        Ok(Some(residuals))
    }

    fn evaluate(&mut self, point: &[f64]) -> Result<Vec<f64>> {
        self.evaluate_trial(point)?
            .ok_or(Error::NonFiniteEvaluation)
    }
}

fn validate(start: &[f64], lower: &[f64], upper: &[f64], o: DogboxOptions) -> Result<()> {
    let n = start.len();
    if n == 0 || n != lower.len() || n != upper.len() {
        return Err(Error::DimensionMismatch);
    }
    if [o.gradient_tolerance, o.cost_tolerance, o.step_tolerance]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.0)
    {
        return Err(Error::InvalidInput(
            "dogbox tolerances must be finite and positive",
        ));
    }
    for i in 0..n {
        if !start[i].is_finite()
            || lower[i].is_nan()
            || upper[i].is_nan()
            || lower[i] == f64::INFINITY
            || upper[i] == f64::NEG_INFINITY
            || lower[i] >= upper[i]
            || start[i] < lower[i]
            || start[i] > upper[i]
        {
            return Err(Error::InvalidInput(
                "invalid initial point or coordinate bounds",
            ));
        }
    }
    // Every admissible problem has at least one residual. Reject impossible
    // parameter budgets before allocating solver state or invoking callbacks.
    check_work(1, n, o.max_iterations)?;
    Ok(())
}

fn check_work(m: usize, n: usize, iterations: u64) -> Result<()> {
    if m.checked_mul(n).is_none_or(|v| v > MAX_WORK as usize)
        || (m as u64)
            .checked_mul(n as u64)
            .and_then(|v| iterations.checked_add(1).and_then(|i| v.checked_mul(i)))
            .is_none_or(|v| v > MAX_WORK)
    {
        return Err(Error::InvalidInput("dogbox storage or work limit exceeded"));
    }
    Ok(())
}

fn derivative<F, J>(
    evaluator: &mut ResidualEvaluator<F>,
    analytical: Option<&J>,
    point: &[f64],
    residuals: &[f64],
    lower: &[f64],
    upper: &[f64],
) -> Result<Matrix>
where
    F: Fn(&[f64]) -> Result<Vec<f64>>,
    J: Fn(&[f64]) -> Result<Matrix>,
{
    let (m, n) = (residuals.len(), point.len());
    let jac = if let Some(jacobian) = analytical {
        jacobian(point)?
    } else {
        let mut result = vec![vec![0.0; n]; m];
        let mut shifted = point.to_vec();
        for j in 0..n {
            // Prefer a step away from zero, then adjust its sign/size to bounds.
            // Stepping across zero can sample an even residual symmetrically
            // and falsely report a zero derivative at a nonstationary point.
            let scale = point[j].abs().max(1.0);
            let nominal = f64::EPSILON.sqrt() * scale;
            let right = upper[j] - point[j];
            let left = point[j] - lower[j];
            let preferred = if point[j] < 0.0 { -nominal } else { nominal };
            let step =
                if preferred >= -left && preferred <= right && (point[j] + preferred).is_finite() {
                    preferred
                } else if -preferred >= -left
                    && -preferred <= right
                    && (point[j] - preferred).is_finite()
                {
                    -preferred
                } else if right >= left {
                    nominal.min(right)
                } else {
                    -nominal.min(left)
                };
            shifted[j] = (point[j] + step).clamp(lower[j], upper[j]);
            let actual_step = shifted[j] - point[j];
            if actual_step == 0.0 || !actual_step.is_finite() {
                return Err(Error::InvalidInput(
                    "bounds too narrow for numerical Jacobian",
                ));
            }
            let shifted_residuals = evaluator.evaluate(&shifted)?;
            for i in 0..m {
                result[i][j] = (shifted_residuals[i] - residuals[i]) / actual_step;
            }
            shifted[j] = point[j];
        }
        result
    };
    if jac.len() != m || jac.iter().any(|row| row.len() != n) {
        return Err(Error::DimensionMismatch);
    }
    if jac.iter().flatten().any(|v| !v.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    Ok(jac)
}

fn gradient(jac: &[Vec<f64>], residuals: &[f64], n: usize) -> Result<Vec<f64>> {
    let mut g = vec![0.0; n];
    for (row, &residual) in jac.iter().zip(residuals) {
        for (value, &j) in g.iter_mut().zip(row) {
            *value += j * residual;
        }
    }
    if g.iter().any(|v| !v.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    Ok(g)
}

fn cost(residuals: &[f64]) -> Result<f64> {
    let value = residuals.iter().map(|v| v * v).sum::<f64>() * 0.5;
    if !value.is_finite() {
        return Err(Error::NonFiniteEvaluation);
    }
    Ok(value)
}

fn free_gradient(gradient: &[f64], mask: &[i8]) -> (Vec<f64>, Vec<usize>, f64) {
    let mut free = Vec::new();
    let mut g = Vec::new();
    let mut norm: f64 = 0.0;
    for (i, &v) in gradient.iter().enumerate() {
        if (mask[i] as f64) * v >= 0.0 {
            free.push(i);
            g.push(v);
            norm = norm.max(v.abs());
        }
    }
    (g, free, norm)
}

fn model_reduction(jac: &[Vec<f64>], free: &[usize], grad: &[f64], step: &[f64]) -> f64 {
    let linear: f64 = grad.iter().zip(step).map(|(g, s)| g * s).sum();
    let quadratic: f64 = jac
        .iter()
        .map(|row| {
            let value: f64 = free.iter().zip(step).map(|(&j, s)| row[j] * s).sum();
            value * value
        })
        .sum();
    -linear - 0.5 * quadratic
}

fn step_to_box(origin: &[f64], direction: &[f64], lo: &[f64], hi: &[f64]) -> f64 {
    let mut alpha: f64 = f64::INFINITY;
    for i in 0..origin.len() {
        if direction[i] > 0.0 {
            alpha = alpha.min(((hi[i] - origin[i]) / direction[i]).max(0.0));
        } else if direction[i] < 0.0 {
            alpha = alpha.min(((lo[i] - origin[i]) / direction[i]).max(0.0));
        }
    }
    alpha
}

// Preserve exact intersections in step coordinates. Reconstructing a limiting
// coordinate as origin + alpha * direction can round just inside the box.
fn box_point(origin: &[f64], direction: &[f64], alpha: f64, lo: &[f64], hi: &[f64]) -> Vec<f64> {
    (0..origin.len())
        .map(|i| {
            if direction[i] > 0.0 && alpha == (hi[i] - origin[i]) / direction[i] {
                hi[i]
            } else if direction[i] < 0.0 && alpha == (lo[i] - origin[i]) / direction[i] {
                lo[i]
            } else {
                (origin[i] + alpha * direction[i]).clamp(lo[i], hi[i])
            }
        })
        .collect()
}

// Valid only while the accepted point, Jacobian and free coordinates are unchanged.
struct DoglegModel {
    newton_step: Vec<f64>,
    cauchy_coefficients: Option<(f64, f64)>,
}

impl DoglegModel {
    fn new(jac: &[Vec<f64>], residuals: &[f64], free: &[usize]) -> Result<Self> {
        let (m, k) = (residuals.len(), free.len());
        let matrix = DMatrix::from_fn(m, k, |i, j| jac[i][free[j]]);
        // Impose a finite iteration budget even if a difficult SVD cannot converge.
        let svd =
            nalgebra::linalg::SVD::try_new(matrix, true, true, f64::EPSILON, 128 * m.min(k).max(1))
                .ok_or(Error::NumericalFailure("dogbox SVD did not converge"))?;
        let largest = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
        // SciPy's dense dogbox requests NumPy lstsq with rcond=-1 (machine epsilon).
        let cutoff = f64::EPSILON * largest;
        let rhs = DVector::from_iterator(m, residuals.iter().map(|r| -r));
        let newton = svd.solve(&rhs, cutoff).map_err(Error::NumericalFailure)?;

        let mut newton_step = vec![0.0; k];
        for (i, &v) in newton.iter().enumerate() {
            newton_step[i] = v;
        }
        if newton_step.iter().any(|v| !v.is_finite()) {
            return Err(Error::NonFiniteEvaluation);
        }
        Ok(Self {
            newton_step,
            cauchy_coefficients: None,
        })
    }
}

fn dogleg(
    model: &mut DoglegModel,
    jac: &[Vec<f64>],
    gradient: &[f64],
    free: &[usize],
    lower_step: &[f64],
    upper_step: &[f64],
) -> (Vec<f64>, f64) {
    let k = free.len();
    let newton_step = &model.newton_step;
    if newton_step
        .iter()
        .zip(lower_step.iter().zip(upper_step))
        .all(|(s, (lo, hi))| *s >= *lo && *s <= *hi)
    {
        let reduction = model_reduction(jac, free, gradient, newton_step);
        if reduction > 0.0 {
            return (newton_step.clone(), reduction);
        }
    }

    // Minimize the quadratic along the constrained anti-gradient ray.
    let opposite: Vec<f64> = gradient.iter().map(|g| -g).collect();
    let zero = vec![0.0; k];
    let to_bounds = step_to_box(&zero, &opposite, lower_step, upper_step);
    let (grad_sq, jg_sq) = *model.cauchy_coefficients.get_or_insert_with(|| {
        let grad_sq = gradient.iter().map(|g| g * g).sum();
        let jg_sq = jac
            .iter()
            .map(|row| {
                let val: f64 = free.iter().zip(gradient).map(|(&j, &g)| row[j] * g).sum();
                val * val
            })
            .sum();
        (grad_sq, jg_sq)
    });
    let alpha = if jg_sq > 0.0 {
        (grad_sq / jg_sq).min(to_bounds)
    } else {
        to_bounds
    };
    let cauchy = box_point(&zero, &opposite, alpha, lower_step, upper_step);
    let direction: Vec<f64> = newton_step
        .iter()
        .zip(&cauchy)
        .map(|(n, c)| n - c)
        .collect();
    let distance = step_to_box(&cauchy, &direction, lower_step, upper_step).min(1.0);
    let step = box_point(&cauchy, &direction, distance, lower_step, upper_step);

    // With rank-deficient Jacobians the dogleg path can increase the model.
    // SciPy documents the constrained Cauchy direction as the safe alternative.
    let cauchy_reduction = model_reduction(jac, free, gradient, &cauchy);
    let step_reduction = model_reduction(jac, free, gradient, &step);
    if cauchy_reduction > step_reduction {
        (cauchy, cauchy_reduction)
    } else {
        (step, step_reduction)
    }
}

fn solve<F, J>(
    function: F,
    analytical: Option<J>,
    start: &[f64],
    lower: &[f64],
    upper: &[f64],
    options: DogboxOptions,
) -> Result<DogboxResult>
where
    F: Fn(&[f64]) -> Result<Vec<f64>>,
    J: Fn(&[f64]) -> Result<Matrix>,
{
    validate(start, lower, upper, options)?;
    let n = start.len();
    let mut eval = ResidualEvaluator {
        function,
        calls: 0,
        expected: None,
    };
    let mut x = start.to_vec();
    let mut residuals = eval.evaluate(&x)?;
    check_work(residuals.len(), n, options.max_iterations)?;
    let mut jac = derivative(&mut eval, analytical.as_ref(), &x, &residuals, lower, upper)?;
    let mut njev = 1;
    let mut current_cost = cost(&residuals)?;
    let mut mask: Vec<i8> = (0..n)
        .map(|i| {
            if x[i] == lower[i] {
                -1
            } else if x[i] == upper[i] {
                1
            } else {
                0
            }
        })
        .collect();
    let mut radius = x.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    if radius == 0.0 {
        radius = 1.0;
    }
    let mut status = DogboxStatus::IterationLimit;
    let mut iterations = 0;
    let mut grad = gradient(&jac, &residuals, n)?;
    let mut model = None;

    for _ in 0..options.max_iterations {
        let (g_free, free, optimality) = free_gradient(&grad, &mask);
        if optimality <= options.gradient_tolerance {
            status = DogboxStatus::Gradient;
            break;
        }
        let low: Vec<f64> = free
            .iter()
            .map(|&i| (lower[i] - x[i]).max(-radius))
            .collect();
        let high: Vec<f64> = free
            .iter()
            .map(|&i| (upper[i] - x[i]).min(radius))
            .collect();
        let cached_model = match &mut model {
            Some(model) => model,
            slot @ None => slot.insert(DoglegModel::new(&jac, &residuals, &free)?),
        };
        let (step_free, predicted) = dogleg(cached_model, &jac, &g_free, &free, &low, &high);
        let mut step = vec![0.0; n];
        for (&i, &value) in free.iter().zip(&step_free) {
            step[i] = value;
        }
        let step_norm = step.iter().fold(0.0_f64, |a, &v| a.hypot(v));
        let step_max = step.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        if step_max == 0.0 || !step_norm.is_finite() {
            status = DogboxStatus::NoProgress;
            break;
        }
        if !predicted.is_finite() || predicted <= 0.0 {
            status = DogboxStatus::NoProgress;
            break;
        }
        let mut trial = x.clone();
        for (i, p) in trial.iter_mut().enumerate() {
            *p = (x[i] + step[i]).clamp(lower[i], upper[i]);
        }
        // Map exact step-box hits back to the original bounds before calling
        // residuals. Clipping alone cannot correct a result rounded inward.
        for (j, &i) in free.iter().enumerate() {
            if step_free[j] == low[j] && low[j] == lower[i] - x[i] {
                trial[i] = lower[i];
            } else if step_free[j] == high[j] && high[j] == upper[i] - x[i] {
                trial[i] = upper[i];
            }
        }
        if trial == x {
            status = DogboxStatus::NoProgress;
            break;
        }
        let evaluated = eval
            .evaluate_trial(&trial)?
            .and_then(|residuals| cost(&residuals).ok().map(|cost| (residuals, cost)));
        let Some((trial_residuals, trial_cost)) = evaluated else {
            // A nonfinite trial is rejected; a smaller rectangle can recover.
            radius = 0.25 * step_max;
            iterations += 1;
            continue;
        };
        let previous_cost = current_cost;
        let previous_x_norm = x.iter().fold(0.0_f64, |a, &v| a.hypot(v));
        let actual = previous_cost - trial_cost;
        let ratio = actual / predicted;
        let tr_hit = free
            .iter()
            .any(|&i| step[i].abs() >= radius * (1.0 - 8.0 * f64::EPSILON));
        if ratio < 0.25 {
            radius = 0.25 * step_max;
        } else if ratio > 0.75 && tr_hit {
            radius *= 2.0;
        }
        iterations += 1;
        if actual > 0.0 {
            x = trial;
            residuals = trial_residuals;
            current_cost = trial_cost;
            for i in 0..n {
                mask[i] = if x[i] == lower[i] {
                    -1
                } else if x[i] == upper[i] {
                    1
                } else {
                    0
                };
            }
            jac = derivative(&mut eval, analytical.as_ref(), &x, &residuals, lower, upper)?;
            njev += 1;
            grad = gradient(&jac, &residuals, n)?;
            model = None;
        }
        let ftol = actual > 0.0 && ratio > 0.25 && actual < options.cost_tolerance * previous_cost;
        let xtol = step_norm < options.step_tolerance * (options.step_tolerance + previous_x_norm);
        if (ftol || xtol) && actual > 0.0 {
            status = match (ftol, xtol) {
                (true, true) => DogboxStatus::CostAndStep,
                (true, false) => DogboxStatus::Cost,
                (false, true) => DogboxStatus::Step,
                _ => unreachable!(),
            };
            break;
        }
    }
    let (_, _, optimality) = free_gradient(&grad, &mask);
    if status == DogboxStatus::IterationLimit && optimality <= options.gradient_tolerance {
        status = DogboxStatus::Gradient;
    }
    Ok(DogboxResult {
        point: x,
        cost: current_cost,
        residuals,
        gradient: grad,
        optimality,
        active_mask: mask,
        function_evaluations: eval.calls,
        jacobian_evaluations: njev,
        iterations,
        status,
    })
}

/// Minimize `F(x) = ½ Σᵢ rᵢ(x)²` using a rectangular trust region and
/// a bound-constrained dogleg step, with a two-point numerical Jacobian.
///
/// ## Parameters and conventions
///
/// `residuals` returns a nonempty vector `r(x)`. `start`, `lower`, and `upper`
/// have the same positive length. Bounds satisfy `lower[i] < upper[i]`, with
/// infinite endpoints allowed, and `lower[i] <= start[i] <= upper[i]`.
/// `options` controls stopping and work; all three tolerances must be finite
/// and positive. Zero `max_iterations` evaluates the initial residuals and
/// Jacobian without attempting a step. Inputs and residuals are `f64`.
/// Residuals must use compatible units; cost has squared residual units.
/// Coordinates retain their supplied units and receive no automatic scaling.
///
/// ## Domain and special values
///
/// Dense real residuals and finite initial parameters. NaN bounds and equal
/// endpoints are rejected; fixed variables must be removed by the caller.
/// Initial residuals and numerical-differentiation samples must be finite.
/// Nonfinite trial residuals or overflowing trial costs shrink the trust region
/// and retry; errors from fallible callbacks always propagate unchanged.
///
/// ## Returns
///
/// [`DogboxResult`] contains the final point, cost, residuals, unprojected
/// gradient `Jᵀr`, and active bounds, all in input coordinate/residual order.
/// `optimality` is `||g_free||∞`, excluding outward gradients at active bounds.
/// A stopping status records the criterion met and does not prove a minimum.
///
/// ## Errors
///
/// Returns [`Error::DimensionMismatch`] for mismatched bounds or residual/Jacobian
/// shapes, [`Error::InvalidInput`] for invalid bounds, tolerances, unrepresentable
/// difference steps or excessive work, [`Error::NonFiniteEvaluation`] for
/// nonfinite initial/differentiation values or cost/gradient overflow, and
/// [`Error::NumericalFailure`] if the private SVD solve fails.
/// A collapsed step or unusable predicted decrease returns
/// [`DogboxStatus::NoProgress`] with the last accepted iterate.
///
/// ## Numerical behavior
///
/// The solver forms a minimum-norm Gauss–Newton step using SVD, then follows a
/// dogleg path through the box defined by the trust region and input bounds.
/// The Gauss–Newton step and Cauchy coefficients are reused across rejected
/// trials and refreshed after each accepted step.
/// Singular values at or below machine epsilon times the largest singular value
/// are discarded. Variables with outward gradients at active bounds are held
/// fixed; coordinates that hit a bound are set to that bound before evaluation.
/// Two-point differences prefer a step away from zero with magnitude
/// `sqrt(EPSILON) * max(1, |x[j]|)`, adjusted to bounds. Very narrow bounds can
/// prevent a representable step. Truncation and cancellation can limit gradient
/// accuracy, especially when the default difference step exceeds a parameter's
/// natural scale; use [`dogbox_with_jacobian`] for such problems.
///
/// Gradient stopping uses `optimality <= gradient_tolerance`. Cost stopping
/// requires a positive decrease below `cost_tolerance * previous_cost` and an
/// actual/predicted decrease ratio above `0.25`. Step stopping requires an
/// accepted step with `||step||₂ < step_tolerance * (step_tolerance + ||x||₂)`.
/// Rank-deficient problems can converge slowly. There is no global-minimum
/// guarantee. Only the linear least-squares loss is supported, with dense
/// Jacobians and no variable scaling, robust loss, or sparse/iterative solves.
///
/// At most 16,777,216 `m * n * (max_iterations + 1)` storage/work units are
/// admitted, excluding callback work. This is a dimension-based admission limit,
/// not a floating-point operation count: each dense SVD costs approximately
/// `O(m * n * min(m, n))`. SVD iterations and trust-region attempts are bounded.
/// Unlike SciPy's `max_nfev`, `max_iterations` includes rejected trust-region
/// attempts and excludes initialization and numerical-differentiation calls;
/// `function_evaluations` includes every residual callback call.
///
/// ## Examples
///
/// ```
/// use sci_optimize::least_squares::{dogbox, DogboxOptions};
/// let result = dogbox(
///     |x| vec![x[0] - 3.0, x[0] - 3.0],
///     &[0.0], &[f64::NEG_INFINITY], &[2.0], DogboxOptions::default(),
/// )?;
/// assert!((result.point[0] - 2.0).abs() < 1e-7);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// SciPy [`least_squares(method="dogbox")`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html),
/// Voglis and Lagaris (2004), and Nocedal and Wright, *Numerical Optimization*.
pub fn dogbox<F>(
    residuals: F,
    start: &[f64],
    lower: &[f64],
    upper: &[f64],
    options: DogboxOptions,
) -> Result<DogboxResult>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    try_dogbox(|x| Ok(residuals(x)), start, lower, upper, options)
}

/// Minimize `F(x) = ½ Σᵢ rᵢ(x)²` with fallible residual evaluations.
///
/// ## Parameters, domain and numerical behavior
///
/// Matches [`dogbox`], including bounds, finite-value requirements, stopping
/// rules, difference steps and work limits. `residuals` returns a shared result.
///
/// ## Returns and errors
///
/// Returns [`DogboxResult`]. Callback errors propagate unchanged, including
/// [`Error::NonFiniteEvaluation`]; only nonfinite values returned inside `Ok`
/// at trial points cause a trust-region retry. Other errors match [`dogbox`].
///
/// ## Examples
///
/// ```
/// use sci_optimize::least_squares::{try_dogbox, DogboxOptions};
/// let fit = try_dogbox(|x| Ok(vec![x[0] - 1.0]), &[0.0], &[-2.0], &[2.0],
///     DogboxOptions::default())?;
/// assert_eq!(fit.point, vec![1.0]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// See [`dogbox`] for the algorithm and independent SciPy reference.
pub fn try_dogbox<F>(
    residuals: F,
    start: &[f64],
    lower: &[f64],
    upper: &[f64],
    options: DogboxOptions,
) -> Result<DogboxResult>
where
    F: Fn(&[f64]) -> Result<Vec<f64>>,
{
    solve(
        residuals,
        None::<fn(&[f64]) -> Result<Matrix>>,
        start,
        lower,
        upper,
        options,
    )
}

/// Minimize `F(x) = ½ Σᵢ rᵢ(x)²` with an analytical `m × n` Jacobian.
///
/// ## Parameters and conventions
///
/// `jacobian(x)[i][j] = ∂r_i(x)/∂x_j`, using the shared [`Matrix`] with one
/// vector per residual row and one entry per parameter column. Entries have
/// residual units divided by the corresponding coordinate's units. Other
/// parameters match [`dogbox`].
///
/// ## Domain and special values
///
/// All Jacobian entries must be finite and its dimensions must remain `m × n`.
/// Residual and bound requirements match [`dogbox`].
///
/// ## Returns and errors
///
/// Returns [`DogboxResult`]; error variants match [`dogbox`]. Incorrect Jacobian
/// shapes return [`Error::DimensionMismatch`]; nonfinite entries return
/// [`Error::NonFiniteEvaluation`].
///
/// ## Numerical behavior
///
/// Uses the same dogleg/SVD algorithm, rank cutoff and work limits as [`dogbox`].
/// An analytical Jacobian eliminates finite-difference sampling and its step-size
/// limitations; each Jacobian call is counted separately from residual calls.
///
/// ## Examples
///
/// ```
/// use sci_optimize::least_squares::{dogbox_with_jacobian, DogboxOptions};
/// let fit = dogbox_with_jacobian(|x| vec![x[0] - 1.0], |_| vec![vec![1.0]],
///     &[0.0], &[-2.0], &[2.0], DogboxOptions::default())?;
/// assert_eq!(fit.point, vec![1.0]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// See [`dogbox`] for the algorithm and independent SciPy reference.
pub fn dogbox_with_jacobian<F, J>(
    residuals: F,
    jacobian: J,
    start: &[f64],
    lower: &[f64],
    upper: &[f64],
    options: DogboxOptions,
) -> Result<DogboxResult>
where
    F: Fn(&[f64]) -> Vec<f64>,
    J: Fn(&[f64]) -> Matrix,
{
    try_dogbox_with_jacobian(
        |x| Ok(residuals(x)),
        |x| Ok(jacobian(x)),
        start,
        lower,
        upper,
        options,
    )
}

/// Minimize `F(x) = ½ Σᵢ rᵢ(x)²` with fallible residual and Jacobian callbacks.
///
/// ## Parameters, domain and numerical behavior
///
/// Matches [`dogbox_with_jacobian`], with callbacks returning shared results.
/// The same row/column ordering, finite-value checks and work limits apply.
///
/// ## Returns and errors
///
/// Returns [`DogboxResult`]. Errors from either callback propagate unchanged;
/// invalid shapes, inputs and arithmetic use the errors listed in [`dogbox`].
///
/// ## Examples
///
/// ```
/// use sci_optimize::least_squares::{try_dogbox_with_jacobian, DogboxOptions};
/// let fit = try_dogbox_with_jacobian(|x| Ok(vec![x[0] - 1.0]),
///     |_| Ok(vec![vec![1.0]]), &[0.0], &[-2.0], &[2.0], DogboxOptions::default())?;
/// assert_eq!(fit.point, vec![1.0]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// See [`dogbox`] for the algorithm and independent SciPy reference.
pub fn try_dogbox_with_jacobian<F, J>(
    residuals: F,
    jacobian: J,
    start: &[f64],
    lower: &[f64],
    upper: &[f64],
    options: DogboxOptions,
) -> Result<DogboxResult>
where
    F: Fn(&[f64]) -> Result<Vec<f64>>,
    J: Fn(&[f64]) -> Result<Matrix>,
{
    solve(residuals, Some(jacobian), start, lower, upper, options)
}
