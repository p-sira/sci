use super::*;
use std::cell::RefCell;

/// Predicts x⁻ = F x and P⁻ = F P Fᵀ + Q.
///
/// # Parameters and conventions
/// `prior` is an n-coordinate Gaussian; `transition` F and `process_noise` Q
/// are n×n row-major matrices. Q is the covariance per step, with squared state
/// units. Time intervals and controls must be incorporated by the caller.
///
/// # Domain and special values
/// n is 1..=32. All values must be finite. P and Q must be symmetric positive
/// semidefinite; zero covariance is allowed.
///
/// # Returns
/// An owned predicted Gaussian in the original state order.
///
/// # Errors
/// Shared errors report invalid dimensions, nonfinite inputs, invalid covariance,
/// arithmetic overflow, or an invalid computed covariance.
///
/// # Numerical behavior
/// Private Smelly f64 matrix products; covariance validation uses at most 10,000
/// eigensolver iterations and tolerates eigenvalues down to −128 ε n times the
/// largest covariance entry. No accuracy guarantee applies to ill-conditioned
/// inputs. Roundoff asymmetry within that tolerance is averaged.
///
/// # Examples
/// ```
/// use sci_filter::{Gaussian, kalman::kalman_predict};
/// let x = Gaussian { mean: vec![2.], covariance: vec![vec![1.]] };
/// assert_eq!(kalman_predict(&x, &[vec![1.]], &[vec![0.]])?, x);
/// # Ok::<(), sci_filter::Error>(())
/// ```
///
/// # References
/// [Särkkä and Svensson, Bayesian Filtering and Smoothing, ch. 6](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf).
pub fn kalman_predict(
    prior: &Gaussian,
    transition: &[Vec<f64>],
    process_noise: &[Vec<f64>],
) -> Result<Gaussian> {
    let (x, p) = gaussian(prior)?;
    let n = x.len();
    let f = matrix(transition, n, n)?;
    let q = covariance(process_noise, n)?;
    let mut filter = smelly::KalmanFilter::new(n, 1);
    filter.set_state(x).map_err(backend)?;
    filter.set_covariance(p).map_err(backend)?;
    filter.set_transition(f).map_err(backend)?;
    filter.set_process_noise(q).map_err(backend)?;
    filter.predict().map_err(backend)?;
    output(filter.state(), filter.covariance())
}

/// Updates a Gaussian with y = z − H x, S = H P Hᵀ + R,
/// K = P Hᵀ S⁻¹, x⁺ = x + K y, and P⁺ = P − K H P.
///
/// # Parameters and conventions
/// `prior` may be a predicted or initial Gaussian. `observation` z has m entries;
/// `measurement` H is m×n, and `measurement_noise` R is m×m. R has squared
/// observation units and represents independent additive noise.
///
/// # Domain and special values
/// n,m are 1..=32. Inputs must be finite; P and R symmetric positive semidefinite.
/// S must be nonsingular. Zero R is allowed for exact observations.
///
/// # Returns
/// Posterior Gaussian and innovation z − H x, in caller coordinate order.
///
/// # Errors
/// Invalid values/covariances, incompatible shapes, singular S, overflow, and
/// invalid computed covariance return shared errors. Inputs remain unchanged.
///
/// # Numerical behavior
/// Smelly solves for K with LU and uses the subtractive covariance update above,
/// averaging roundoff asymmetry. This can lose positive semidefiniteness for
/// ill-conditioned systems; detected failures are errors. Dimensions ≤32 bound
/// dense work; validation has a 10,000-iteration eigensolver limit and the
/// covariance tolerance documented by [`kalman_predict`].
///
/// # Examples
/// ```
/// use sci_filter::{Gaussian, kalman::kalman_update};
/// let x = Gaussian { mean: vec![0.], covariance: vec![vec![1.]] };
/// let u = kalman_update(&x, &[2.], &[vec![1.]], &[vec![1.]])?;
/// assert_eq!(u.estimate.mean, vec![1.]);
/// assert_eq!(u.innovation, vec![2.]);
/// # Ok::<(), sci_filter::Error>(())
/// ```
///
/// # References
/// [Särkkä and Svensson, ch. 6](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf).
pub fn kalman_update(
    prior: &Gaussian,
    observation: &[f64],
    measurement: &[Vec<f64>],
    measurement_noise: &[Vec<f64>],
) -> Result<Update> {
    let (x, p) = gaussian(prior)?;
    let n = x.len();
    let m = observation.len();
    dimension(m)?;
    vector(observation, m)?;
    let h = matrix(measurement, m, n)?;
    let r = covariance(measurement_noise, m)?;
    let mut filter = smelly::KalmanFilter::new(n, m);
    filter.set_state(x).map_err(backend)?;
    filter.set_covariance(p).map_err(backend)?;
    filter.set_measurement(h).map_err(backend)?;
    filter.set_measurement_noise(r).map_err(backend)?;
    let innovation = filter
        .update(&DVector::from_column_slice(observation))
        .map_err(backend)?;
    update_output(filter.state(), filter.covariance(), innovation)
}

fn update_output(x: &DVector<f64>, p: &DMatrix<f64>, innovation: DVector<f64>) -> Result<Update> {
    if innovation.iter().any(|v| !v.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    Ok(Update {
        estimate: output(x, p)?,
        innovation: innovation.as_slice().to_vec(),
    })
}

/// Predicts a nonlinear Gaussian approximation: x⁻ = f(x), P⁻ = J P Jᵀ + Q.
///
/// # Parameters and conventions
/// `process` returns f(x) from a state slice; `jacobian` returns J = ∂f/∂x at
/// that same prior mean. Each callback is evaluated once, process first. Q is
/// n×n additive independent noise covariance per step. Include dt and controls
/// in the callbacks; vectors and matrices retain caller state order and units.
///
/// # Domain and special values
/// n is 1..=32. Finite inputs and outputs and symmetric positive-semidefinite
/// P,Q are required. Callbacks return shared Result and must return n entries
/// or n×n entries respectively. Zero noise is allowed.
///
/// # Returns
/// Predicted Gaussian, with mean f(x) and linearized covariance.
///
/// # Errors
/// Invalid inputs are rejected before callbacks. Callback errors propagate;
/// shape, nonfinite output, overflow, and covariance errors are explicit.
///
/// # Numerical behavior
/// First-order approximation about the mean; Smelly performs covariance
/// prediction. Dense work is bounded by n≤32 and the covariance checks described
/// in [`kalman_predict`]. Strong nonlinearities can make this approximation poor.
///
/// # Examples
/// ```
/// use sci_filter::{Gaussian, kalman::extended_kalman_predict};
/// let x = Gaussian { mean: vec![2.], covariance: vec![vec![1.]] };
/// let p = extended_kalman_predict(&x, &[vec![0.]],
///     |x| Ok(vec![x[0]*x[0]]), |x| Ok(vec![vec![2.*x[0]]]))?;
/// assert_eq!(p.mean, vec![4.]);
/// assert_eq!(p.covariance, vec![vec![16.]]);
/// # Ok::<(), sci_filter::Error>(())
/// ```
///
/// # References
/// [Särkkä and Svensson, ch. 7](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf).
pub fn extended_kalman_predict<F, J>(
    prior: &Gaussian,
    process_noise: &[Vec<f64>],
    process: F,
    jacobian: J,
) -> Result<Gaussian>
where
    F: FnOnce(&[f64]) -> Result<Vec<f64>>,
    J: FnOnce(&[f64]) -> Result<Matrix>,
{
    let (x, p) = gaussian(prior)?;
    let n = x.len();
    let q = covariance(process_noise, n)?;
    let predicted = process(&prior.mean)?;
    callback_vector(&predicted, n)?;
    let f = jacobian(&prior.mean)?;
    let f = callback_matrix(&f, n, n)?;
    // Predict covariance with zero mean to avoid irrelevant F*x overflow.
    let mut filter = smelly::ExtendedKalmanFilter::new(n, 1);
    filter.set_covariance(p).map_err(backend)?;
    filter.set_transition(f).map_err(backend)?;
    filter.set_process_noise(q).map_err(backend)?;
    filter.predict().map_err(backend)?;
    output(&DVector::from_vec(predicted), filter.covariance())
}

/// Updates with nonlinear innovation y = z − h(x), using H = ∂h/∂x.
///
/// # Parameters and conventions
/// `measurement` returns m predicted observation coordinates; `jacobian` returns
/// an m×n Jacobian at the prior mean. Each runs once, measurement first.
/// `measurement_noise` is m×m independent additive covariance in squared
/// observation units. Residuals use ordinary Euclidean subtraction.
///
/// # Domain and special values
/// n,m are 1..=32. All inputs and callback outputs must be finite. P and R must
/// be symmetric positive semidefinite, and H P Hᵀ + R nonsingular.
///
/// # Returns
/// Posterior Gaussian and innovation z − h(x), in observation order.
///
/// # Errors
/// Invalid inputs are rejected before callbacks. Callback errors propagate;
/// shapes, nonfinite evaluations, singular innovations and covariance failures
/// return shared errors without changing the prior.
///
/// # Numerical behavior
/// Smelly applies the first-order Kalman update from [`kalman_update`], including
/// its subtractive covariance formula, numerical limits and roundoff tolerance.
/// Accuracy depends on local linearization and conditioning.
///
/// # Examples
/// ```
/// use sci_filter::{Gaussian, kalman::extended_kalman_update};
/// let x = Gaussian { mean: vec![2.], covariance: vec![vec![1.]] };
/// let u = extended_kalman_update(&x, &[5.], &[vec![1.]],
///     |x| Ok(vec![x[0]*x[0]]), |x| Ok(vec![vec![2.*x[0]]]))?;
/// assert_eq!(u.innovation, vec![1.]);
/// # Ok::<(), sci_filter::Error>(())
/// ```
///
/// # References
/// [Särkkä and Svensson, ch. 7](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf).
pub fn extended_kalman_update<H, J>(
    prior: &Gaussian,
    observation: &[f64],
    measurement_noise: &[Vec<f64>],
    measurement: H,
    jacobian: J,
) -> Result<Update>
where
    H: FnOnce(&[f64]) -> Result<Vec<f64>>,
    J: FnOnce(&[f64]) -> Result<Matrix>,
{
    let (x, p) = gaussian(prior)?;
    let n = x.len();
    let m = observation.len();
    dimension(m)?;
    vector(observation, m)?;
    let r = covariance(measurement_noise, m)?;
    let predicted = measurement(&prior.mean)?;
    callback_vector(&predicted, m)?;
    let h = callback_matrix(&jacobian(&prior.mean)?, m, n)?;
    let mut filter = smelly::ExtendedKalmanFilter::new(n, m);
    filter.set_state(x).map_err(backend)?;
    filter.set_covariance(p).map_err(backend)?;
    filter.set_measurement_noise(r).map_err(backend)?;
    let innovation = filter
        .update(
            &DVector::from_column_slice(observation),
            |_| DVector::from_vec(predicted),
            |_| h,
        )
        .map_err(backend)?;
    update_output(filter.state(), filter.covariance(), innovation)
}

fn callback_vector(x: &[f64], n: usize) -> Result<()> {
    vector(x, n).map_err(|e| match e {
        Error::InvalidInput(_) => Error::NonFiniteEvaluation,
        e => e,
    })
}

fn callback_matrix(x: &[Vec<f64>], m: usize, n: usize) -> Result<DMatrix<f64>> {
    matrix(x, m, n).map_err(|e| match e {
        Error::InvalidInput(_) => Error::NonFiniteEvaluation,
        e => e,
    })
}

/// Performs one scaled unscented prediction and observation update.
///
/// For c = α²(n+κ), use χ₀=x and χᵢ=x±columns(chol(cP)). Mean weights are
/// w₀=(c−n)/c, wᵢ=1/(2c); the covariance weight at zero adds 1−α²+β.
/// Transform through f, add Q, regenerate points from the predicted Gaussian,
/// then transform through h and add R. Cross covariance C gives K=C S⁻¹,
/// x⁺=x⁻+K(z−z̄), and P⁺=P⁻−K S Kᵀ.
///
/// # Parameters and conventions
/// `prior` has n coordinates; Q is n×n, observation z has m coordinates, R is
/// m×m. `points` sets the weights. `process` and `measurement` each receive a
/// state slice and return shared Result with n and m coordinates respectively.
/// Models must be pure: each is called at most 2n+1 times. Include dt/controls
/// in `process`. Noise is independent, additive, and in squared coordinate units.
///
/// # Domain and special values
/// n,m are 1..=32. Finite inputs and callback outputs are required; Q,R are
/// symmetric positive semidefinite. P and the predicted covariance must be
/// positive definite; innovation covariance must be nonsingular. Scaling obeys
/// [`SigmaPoints`]. Coordinates/residuals are Euclidean.
///
/// # Returns
/// Posterior Gaussian and innovation z − z̄, in caller coordinate order.
///
/// # Errors
/// Validation precedes callbacks. Callback errors propagate and suppress later
/// user callback evaluations. Invalid shapes/scaling, failed Cholesky, singular
/// innovation, overflow or invalid computed covariance return shared errors.
///
/// # Numerical behavior
/// Smelly regenerates sigma points after adding Q so measurement uncertainty
/// includes process noise. This differs from FilterPy's default cached-point
/// update. LU solves and subtractive covariance updates can lose accuracy;
/// small α increases cancellation. Dense work is bounded by dimensions ≤32;
/// covariance validation follows [`kalman_predict`].
///
/// # Examples
/// ```
/// use sci_filter::{Gaussian, SigmaPoints, kalman::unscented_kalman_step};
/// let x = Gaussian { mean: vec![0.], covariance: vec![vec![1.]] };
/// let u = unscented_kalman_step(&x, &[vec![0.1]], &[2.], &[vec![0.5]],
///     SigmaPoints::default(), |x| Ok(x.to_vec()), |x| Ok(x.to_vec()))?;
/// assert!(u.estimate.mean[0] > 1.);
/// # Ok::<(), sci_filter::Error>(())
/// ```
///
/// # References
/// [Särkkä and Svensson, ch. 8](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf);
/// [Smelly's UKF conventions](https://docs.rs/smelly/0.2.0/smelly/struct.UnscentedKalmanFilter.html).
pub fn unscented_kalman_step<F, H>(
    prior: &Gaussian,
    process_noise: &[Vec<f64>],
    observation: &[f64],
    measurement_noise: &[Vec<f64>],
    points: SigmaPoints,
    process: F,
    measurement: H,
) -> Result<Update>
where
    F: Fn(&[f64]) -> Result<Vec<f64>>,
    H: Fn(&[f64]) -> Result<Vec<f64>>,
{
    let (x, p) = gaussian(prior)?;
    let n = x.len();
    let m = observation.len();
    dimension(m)?;
    vector(observation, m)?;
    let q = covariance(process_noise, n)?;
    let r = covariance(measurement_noise, m)?;
    let c = points.alpha * points.alpha * (n as f64 + points.kappa);
    if !points.alpha.is_finite()
        || points.alpha <= 0.
        || !points.beta.is_finite()
        || points.beta < 0.
        || !points.kappa.is_finite()
        || !c.is_finite()
        || c < 1e-6 * n as f64
        || !(1. - points.alpha * points.alpha + points.beta).is_finite()
    {
        return Err(Error::InvalidInput("invalid sigma-point scaling"));
    }
    let mut filter = smelly::UnscentedKalmanFilter::new(
        n,
        m,
        smelly::MerweScaledSigmaPoints::new(points.alpha, points.beta, points.kappa),
    );
    filter.set_state(x).map_err(backend)?;
    filter.set_covariance(p).map_err(backend)?;
    filter.set_process_noise(q).map_err(backend)?;
    filter.set_measurement_noise(r).map_err(backend)?;
    let error = RefCell::new(None);
    let predict_result = filter.predict(|x| checked_model(&process, x, n, &error));
    if let Some(e) = error.borrow_mut().take() {
        return Err(e);
    }
    predict_result.map_err(backend)?;
    output(filter.state(), filter.covariance())?;
    let update_result = filter.update(&DVector::from_column_slice(observation), |x| {
        checked_model(&measurement, x, m, &error)
    });
    if let Some(e) = error.borrow_mut().take() {
        return Err(e);
    }
    let innovation = update_result.map_err(backend)?;
    update_output(filter.state(), filter.covariance(), innovation)
}

fn checked_model<F>(
    model: &F,
    x: &DVector<f64>,
    n: usize,
    error: &RefCell<Option<Error>>,
) -> DVector<f64>
where
    F: Fn(&[f64]) -> Result<Vec<f64>>,
{
    if error.borrow().is_some() {
        return DVector::zeros(n);
    }
    let result = if x.iter().any(|v| !v.is_finite()) {
        Err(Error::NonFiniteEvaluation)
    } else {
        model(x.as_slice()).and_then(|value| {
            callback_vector(&value, n)?;
            Ok(DVector::from_vec(value))
        })
    };
    match result {
        Ok(value) => value,
        Err(e) => {
            *error.borrow_mut() = Some(e);
            DVector::zeros(n)
        }
    }
}
