mod derivatives;
pub use derivatives::*;

use crate::{Error, Result, error::is_finite_slice};

const MAX_COORDINATES: usize = 1_000_000;
const MAX_WORK: u64 = 16_777_216;

fn bounded_work(coordinates: usize, iterations: u64) -> Result<()> {
    if coordinates > MAX_COORDINATES
        || iterations
            .checked_add(1)
            .and_then(|n| n.checked_mul(coordinates as u64))
            .is_none_or(|n| n > MAX_WORK)
    {
        return Err(Error::InvalidInput(
            "optimization storage or work limit exceeded",
        ));
    }
    Ok(())
}

/// Numerical controls shared by minimization functions.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Maximum completed iterations, including zero for initialization only.
    pub max_iterations: u64,
    /// Finite positive absolute stopping tolerance. Nelder–Mead uses it for
    /// simplex size and cost spread; gradient descent uses gradient norm.
    pub tolerance: f64,
    /// Finite positive step size for gradient descent; ignored by Nelder–Mead.
    pub learning_rate: f64,
}

impl Default for Options {
    /// Uses 1,000 iterations, tolerance `1e-8`, and learning rate `0.1`.
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-8,
            learning_rate: 0.1,
        }
    }
}

/// Why minimization stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// The algorithm's convergence criterion was satisfied.
    Converged,
    /// The iteration budget was exhausted.
    IterationLimit,
    /// The algorithm could not establish convergence.
    NoProgress,
}

/// Backend-independent minimization outcome.
#[derive(Clone, Debug, PartialEq)]
pub struct Minimum {
    /// Final iterate, suitable for differentiation or linear algebra.
    pub point: Vec<f64>,
    /// Objective value at the final iterate.
    pub cost: f64,
    /// Number of completed iterations.
    pub iterations: u64,
    /// Reason for stopping; inspect this before assuming convergence.
    pub status: Status,
}

struct Objective<F, G> {
    function: F,
    gradient: G,
}

impl<F, G> basin::CostFunction for Objective<F, G>
where
    F: Fn(&[f64]) -> Result<f64>,
{
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Error;
    fn cost(&self, point: &Vec<f64>) -> Result<f64> {
        is_finite_slice(point)?;
        match (self.function)(point)? {
            value if value.is_finite() => Ok(value),
            _ => Err(Error::NonFiniteEvaluation),
        }
    }
}

impl<F, G> basin::Gradient for Objective<F, G>
where
    F: Fn(&[f64]) -> Result<f64>,
    G: Fn(&[f64]) -> Result<Vec<f64>>,
{
    type Gradient = Vec<f64>;
    fn gradient(&self, point: &Vec<f64>) -> Result<Vec<f64>> {
        is_finite_slice(point)?;
        let gradient = (self.gradient)(point)?;
        if gradient.len() != point.len() {
            return Err(Error::DimensionMismatch);
        }
        if gradient.iter().any(|value| !value.is_finite()) {
            return Err(Error::NonFiniteEvaluation);
        }
        Ok(gradient)
    }
}

fn validate(start: &[f64], options: Options) -> Result<()> {
    if start.is_empty() {
        return Err(Error::InvalidInput(
            "optimization requires at least one coordinate",
        ));
    }
    bounded_work(start.len(), options.max_iterations)?;
    is_finite_slice(start)?;
    if !options.tolerance.is_finite() || options.tolerance <= 0.0 {
        return Err(Error::InvalidInput("tolerance must be finite and positive"));
    }
    Ok(())
}

fn output<S: basin::State<Param = Vec<f64>, Float = f64>>(
    result: basin::OptimizationResult<S>,
) -> Minimum {
    use basin::TerminationReason as Reason;
    let status = match result.reason {
        Reason::MaxIter => Status::IterationLimit,
        Reason::SimplexTolerance | Reason::GradientTolerance | Reason::SolverConverged => {
            Status::Converged
        }
        _ => Status::NoProgress,
    };
    Minimum {
        point: result.param().clone(),
        cost: result.cost(),
        iterations: result.iter(),
        status,
    }
}

/// Minimizes a scalar objective using Nelder–Mead.
/// Searches for a local solution of `argmin_x function(x)`.
///
/// ## Parameters
///
/// `function` maps a coordinate slice to a finite scalar cost.
/// `start` must contain at least one finite coordinate. `options` controls
/// simplex size and cost tolerance, and iteration budget.
/// The simplex stores `n*(n+1)` coordinates, limited to 1,000,000.
/// Including initialization, `(max_iterations+1)*n*(n+1)` is limited to
/// 16,777,216 coordinate-iterations; callback work is controlled by the caller.
///
/// ## Domain
///
/// Finite nonempty starting coordinates and finite positive stopping tolerance.
/// Objectives must return finite values at every sampled simplex vertex.
///
/// ## Returns
///
/// [`Minimum`] contains the final point, cost, iteration count, and stopping
/// status. An exhausted iteration budget does not establish convergence.
///
/// ## Errors
///
/// Returns an error for invalid input, exceeded storage/work bounds, or
/// nonfinite callback evaluations. Bounds are checked before callbacks.
/// Iteration exhaustion is reported by [`Minimum::status`].
///
/// ## Numerical behavior
///
/// Uses the private Basin Nelder–Mead backend with bounded simplex storage and
/// coordinate work. Simplex and cost tolerances do not guarantee a global
/// minimum; poorly scaled objectives can stagnate or lose accuracy.
///
/// # Examples
///
/// ```
/// use sci_optimize::{minimize, Options};
/// let result = minimize(|x| x[0].powi(2), &[2.0], Options::default())?;
/// assert!(result.cost < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy Nelder–Mead](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html).
pub fn minimize<F>(function: F, start: &[f64], options: Options) -> Result<Minimum>
where
    F: Fn(&[f64]) -> f64,
{
    try_minimize(|point| Ok(function(point)), start, options)
}

/// Fallible-objective variant of [`minimize`].
///
/// ## Errors
///
/// Callback errors are preserved unchanged. Input validation and finite-value
/// requirements match [`minimize`].
///
/// # Examples
///
/// ```
/// let result = sci_optimize::try_minimize(
///     |x| sci_linalg::dot(x, x), &[1.0, 2.0],
///     sci_optimize::Options::default(),
/// )?;
/// assert!(result.cost < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
pub fn try_minimize<F>(function: F, start: &[f64], options: Options) -> Result<Minimum>
where
    F: Fn(&[f64]) -> Result<f64>,
{
    validate(start, options)?;
    let simplex_size = start
        .len()
        .checked_mul(start.len() + 1)
        .ok_or(Error::InvalidInput(
            "optimization storage or work limit exceeded",
        ))?;
    bounded_work(simplex_size, options.max_iterations)?;
    let solver = basin::NelderMead::new()
        .with_absolute_simplex_size_tolerance(options.tolerance)
        .with_absolute_simplex_cost_tolerance(options.tolerance);
    let result = basin::Executor::from_start(
        Objective {
            function,
            gradient: (),
        },
        solver,
        start.to_vec(),
    )
    .max_iter(options.max_iterations)
    .run()?;
    Ok(output(result))
}

/// Minimizes using fixed-step gradient descent: `x ← x − learning_rate ∇f(x)`.
///
/// ## Parameters
///
/// The gradient returns one derivative per input coordinate. Its Euclidean
/// norm is compared to `options.tolerance`. Both callbacks must return finite
/// values. Select a learning rate appropriate to the objective's curvature.
/// At most 1,000,000 coordinates and 16,777,216 coordinate-iterations
/// (`n*(max_iterations+1)`) are allowed, excluding caller-controlled callback work.
///
/// ## Domain
///
/// Finite nonempty starting coordinates, finite callbacks, and finite positive
/// tolerance and learning rate. Gradient length must equal coordinate count.
///
/// ## Returns
///
/// [`Minimum`] records the final point, cost, iteration count and stopping
/// status; check the status before assuming convergence.
///
/// ## Errors
///
/// Returns an error for invalid controls, nonfinite evaluations, or incorrect
/// gradient dimensions. An iteration limit is a normal [`Minimum`] outcome.
///
/// ## Numerical behavior
///
/// Fixed-step iterations use the private Basin backend with bounded coordinate
/// work. The stopping norm uses hypot accumulation, avoiding overflow and
/// underflow from squaring gradients or tolerances. No extra callbacks are used.
/// Large steps can diverge and small steps can stagnate; no global
/// minimum or uniform accuracy guarantee is provided.
///
/// # Examples
///
/// ```
/// use sci_optimize::{minimize_with_gradient, Options};
/// let result = minimize_with_gradient(
///     |x| x[0].powi(2), |x| vec![2.0 * x[0]], &[3.0], Options::default(),
/// )?;
/// assert!(result.cost < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [Basin gradient descent](https://docs.rs/basin/1.15.1/basin/struct.GradientDescent.html).
pub fn minimize_with_gradient<F, G>(
    function: F,
    gradient: G,
    start: &[f64],
    options: Options,
) -> Result<Minimum>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    try_minimize_with_gradient(|x| Ok(function(x)), |x| Ok(gradient(x)), start, options)
}

/// Fallible-callback variant of [`minimize_with_gradient`].
///
/// ## Errors
///
/// Preserves callback errors and validates dimensions and finite values.
/// Controls have the same interpretation as [`minimize_with_gradient`].
///
/// # Examples
///
/// ```
/// use sci_differentiate as differentiate; use sci_linalg as linalg; use sci_optimize as optimize;
/// let cost = |x: &[f64]| linalg::dot(x, x).unwrap();
/// let result = optimize::try_minimize_with_gradient(
///     |x| linalg::dot(x, x),
///     |x| differentiate::gradient(cost, x, 1e-5),
///     &[1.0, -2.0], optimize::Options::default(),
/// )?;
/// assert!(result.cost < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
pub fn try_minimize_with_gradient<F, G>(
    function: F,
    gradient: G,
    start: &[f64],
    options: Options,
) -> Result<Minimum>
where
    F: Fn(&[f64]) -> Result<f64>,
    G: Fn(&[f64]) -> Result<Vec<f64>>,
{
    validate(start, options)?;
    if !options.learning_rate.is_finite() || options.learning_rate <= 0.0 {
        return Err(Error::InvalidInput(
            "learning rate must be finite and positive",
        ));
    }
    use basin::GradientState;
    let solver = basin::GradientDescent::new(options.learning_rate);
    let result =
        basin::Executor::from_start(Objective { function, gradient }, solver, start.to_vec())
            .stop_when(move |state| {
                state.gradient().and_then(|gradient| {
                    let norm = gradient.iter().fold(0.0_f64, |norm, &v| norm.hypot(v));
                    (norm <= options.tolerance)
                        .then_some(basin::TerminationReason::GradientTolerance)
                })
            })
            .max_iter(options.max_iterations)
            .run()?;
    Ok(output(result))
}
