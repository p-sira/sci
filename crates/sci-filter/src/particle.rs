use super::*;

/// Weighted particles representing p(x) ≈ Σ wᵢ δ(x−xᵢ).
///
/// One state vector per row; weights may be unnormalized but must be finite,
/// nonnegative and have positive total mass. Numerical functions validate fields.
#[derive(Clone, Debug, PartialEq)]
pub struct ParticleSet {
    /// Particle coordinates, with consistent caller-defined ordering and units.
    pub states: Matrix,
    /// One nonnegative relative weight per particle.
    pub weights: Vec<f64>,
}

struct Weights {
    linear: Vec<f64>,
    log: Vec<f64>,
}

fn validated(particles: &ParticleSet) -> Result<Weights> {
    let count = particles.states.len();
    if count == 0 || count > 65_536 {
        return Err(Error::InvalidInput("particle count must be in 1..=65536"));
    }
    let n = particles.states[0].len();
    dimension(n)?;
    // All entry points obey the covariance/moment work bound, before allocation.
    if count * n * n > MAX_WORK || count * n > 1_048_576 {
        return Err(Error::InvalidInput("particle work limit exceeded"));
    }
    for state in &particles.states {
        vector(state, n)?;
    }
    if particles.weights.len() != count {
        return Err(Error::DimensionMismatch);
    }
    if particles.weights.iter().any(|w| !w.is_finite() || *w < 0.) {
        return Err(Error::InvalidInput(
            "particle weights must be finite and nonnegative",
        ));
    }
    let maximum = particles.weights.iter().copied().fold(0., f64::max);
    if maximum == 0. {
        return Err(Error::InvalidInput("particle weights have zero mass"));
    }
    let sum: f64 = particles.weights.iter().map(|w| w / maximum).sum();
    let log_sum = sum.ln();
    let log_maximum = maximum.ln();
    let (linear, log) = particles
        .weights
        .iter()
        .map(|w| {
            let relative = w / maximum;
            (
                relative / sum,
                if relative.is_normal() {
                    relative.ln() - log_sum
                } else {
                    w.ln() - log_maximum - log_sum
                },
            )
        })
        .unzip();
    Ok(Weights { linear, log })
}

// Inputs have already passed the shared validation and allocation limits.
fn filter(particles: &ParticleSet) -> Result<smelly::ParticleFilter> {
    smelly::ParticleFilter::new(DMatrix::from_fn(
        particles.states[0].len(),
        particles.states.len(),
        |r, c| particles.states[c][r],
    ))
    .map_err(backend)
}

/// Propagates each particle xᵢ⁻ = f(xᵢ), retaining its normalized weight.
///
/// # Parameters and conventions
/// `prior` contains particle rows and weights. `process` returns a shared Result
/// with one state vector per invocation, in the same units/order. Incorporate
/// controls, dt, and any sampled process noise in the callback; no noise is
/// implicitly drawn. FnMut permits a caller-owned seeded random generator.
///
/// # Domain and special values
/// 1..=65536 particles and 1..=32 coordinates; at most 1048576 coordinates and
/// 16777216 particle-count×dimension² work. Finite rectangular states and finite
/// nonnegative weights of positive mass are required. Zero weights are allowed.
///
/// # Returns
/// New particle rows in input order, with normalized weights.
///
/// # Errors
/// Validation precedes callbacks; callback errors propagate and stop later user
/// evaluations. Changed dimensions and nonfinite results are errors. Inputs are
/// unchanged; callback side effects, including RNG draws, cannot be rolled back.
///
/// # Numerical behavior
/// Private Smelly propagation with caller-supplied process noise. Weights
/// are normalized after division by their maximum to avoid overflow. Tiny
/// relative weights may underflow to zero. Each callback runs at most once per
/// particle; user callback work is outside the library's work bound.
///
/// # Examples
/// ```
/// use sci_filter::{ParticleSet, particle::particle_predict};
/// let p = ParticleSet { states: vec![vec![1.],vec![2.]], weights: vec![1.,1.] };
/// let q = particle_predict(&p, |x| Ok(vec![x[0]+1.]))?;
/// assert_eq!(q.states, vec![vec![2.],vec![3.]]);
/// # Ok::<(), sci_filter::Error>(())
/// ```
///
/// # References
/// [Särkkä and Svensson, ch. 11](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf).
pub fn particle_predict<F>(prior: &ParticleSet, process: F) -> Result<ParticleSet>
where
    F: FnMut(&[f64]) -> Result<Vec<f64>>,
{
    let weights = validated(prior)?;
    let mut filter = filter(prior)?;
    let mut process = process;
    let mut error = None;
    filter
        .predict(|x, out| {
            if error.is_some() {
                return;
            }
            match process(x).and_then(|value| {
                if value.len() != out.len() {
                    return Err(Error::DimensionMismatch);
                }
                if value.iter().any(|v| !v.is_finite()) {
                    return Err(Error::NonFiniteEvaluation);
                }
                Ok(value)
            }) {
                Ok(value) => out.copy_from_slice(&value),
                Err(e) => error = Some(e),
            }
        })
        .map_err(backend)?;
    if let Some(e) = error {
        return Err(e);
    }
    Ok(ParticleSet {
        states: filter
            .particles()
            .column_iter()
            .map(|x| x.as_slice().to_vec())
            .collect(),
        weights: weights.linear,
    })
}

/// Reweights particles using wᵢ⁺ ∝ wᵢ exp(ℓᵢ), where ℓᵢ = log p(z|xᵢ).
///
/// # Parameters and conventions
/// `log_likelihood` takes each particle state and returns a shared Result with
/// its log likelihood for the caller's observation. Observation data is captured
/// by the callback; likelihoods use natural logarithms. Called in row order,
/// once per particle, including zero-weight particles. No resampling is implicit.
///
/// # Domain and special values
/// Particle shapes, weights, and work bounds follow [`particle_predict`]. A
/// log likelihood may be finite or negative infinity (zero probability). NaN and
/// positive infinity are rejected; some positive-prior particle must survive.
///
/// # Returns
/// Original particle coordinates with normalized posterior weights.
///
/// # Errors
/// Invalid input, callback errors, NaN/+infinity, and zero posterior mass return
/// shared errors. Inputs are unchanged; callbacks stop at the first error.
///
/// # Numerical behavior
/// Native max-shift log normalization avoids likelihood underflow and preserves
/// prior ratios under enormous common log likelihood offsets. Original positive
/// priors retain finite logarithms even when normalized prior weights underflow.
/// Only final posterior weights are rounded to binary64. Very small posterior
/// probabilities may round to zero. Work is linear in particle count, with the
/// common coordinate/work bounds; no uniform relative accuracy near underflow.
///
/// # Examples
/// ```
/// use sci_filter::{ParticleSet, particle::particle_update};
/// let p = ParticleSet { states: vec![vec![0.],vec![1.]], weights: vec![1.,1.] };
/// let q = particle_update(&p, |x| Ok(if x[0]==0. { 0. } else { f64::NEG_INFINITY }))?;
/// assert_eq!(q.weights, vec![1.,0.]);
/// # Ok::<(), sci_filter::Error>(())
/// ```
///
/// # References
/// [Särkkä and Svensson, ch. 11](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf).
pub fn particle_update<F>(prior: &ParticleSet, mut log_likelihood: F) -> Result<ParticleSet>
where
    F: FnMut(&[f64]) -> Result<f64>,
{
    let mut weights = validated(prior)?;
    let mut likelihoods = Vec::with_capacity(prior.states.len());
    let mut maximum_likelihood = f64::NEG_INFINITY;
    for (state, log_weight) in prior.states.iter().zip(&weights.log) {
        let likelihood = log_likelihood(state)?;
        if likelihood.is_nan() || likelihood == f64::INFINITY {
            return Err(Error::NonFiniteEvaluation);
        }
        if log_weight.is_finite() {
            maximum_likelihood = maximum_likelihood.max(likelihood);
        }
        likelihoods.push(likelihood);
    }
    if maximum_likelihood == f64::NEG_INFINITY {
        return Err(Error::NumericalFailure(
            "observation has zero particle probability",
        ));
    }
    // Remove the common likelihood offset before combining with prior logs;
    // this preserves prior weights even when log likelihoods are around 1e308.
    let mut maximum = f64::NEG_INFINITY;
    for (log_weight, likelihood) in weights.log.iter_mut().zip(likelihoods) {
        if *log_weight != f64::NEG_INFINITY {
            *log_weight += likelihood - maximum_likelihood;
        }
        maximum = maximum.max(*log_weight);
    }
    let mut sum = 0.;
    for (weight, log_weight) in weights.linear.iter_mut().zip(weights.log) {
        *weight = (log_weight - maximum).exp();
        sum += *weight;
    }
    for weight in &mut weights.linear {
        *weight /= sum;
    }
    Ok(ParticleSet {
        states: prior.states.clone(),
        weights: weights.linear,
    })
}

/// Computes μ = Σ wᵢ xᵢ and P = Σ wᵢ(xᵢ−μ)(xᵢ−μ)ᵀ.
///
/// # Parameters and conventions
/// `particles` contains state rows and relative weights. P is the population
/// covariance of the weighted empirical distribution, with no unbiased-sample
/// correction. Means have state units, covariances squared state units.
///
/// # Domain and special values
/// Finite states and finite nonnegative positive-mass weights, with shapes and
/// work limits from [`particle_predict`]. A singleton yields zero covariance.
///
/// # Returns
/// A Gaussian summary in particle coordinate order; a non-Gaussian posterior
/// need not be well described by its first two moments.
///
/// # Errors
/// Invalid shapes, values, weights, work, overflow, or invalid computed covariance
/// return shared errors.
///
/// # Numerical behavior
/// Smelly uses normalized weighted sums and a second pass for covariance.
/// Cancellation or extreme coordinates can lose accuracy or overflow; dimensions
/// and particle-count×dimension² are bounded, as is covariance validation.
///
/// # Examples
/// ```
/// use sci_filter::{ParticleSet, particle::particle_estimate};
/// let p = ParticleSet { states: vec![vec![-1.],vec![1.]], weights: vec![1.,1.] };
/// assert_eq!(particle_estimate(&p)?.covariance, vec![vec![1.]]);
/// # Ok::<(), sci_filter::Error>(())
/// ```
///
/// # References
/// [Särkkä and Svensson, ch. 11](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf).
pub fn particle_estimate(particles: &ParticleSet) -> Result<Gaussian> {
    let weights = validated(particles)?;
    let mut filter = filter(particles)?;
    let mut weights = weights.linear.into_iter();
    filter
        .update(|_| weights.next().unwrap())
        .map_err(backend)?;
    output(&filter.state(), &filter.covariance())
}

/// Computes effective sample size N_eff = 1 / Σ normalized(wᵢ)².
///
/// # Parameters and conventions
/// `particles` is a weighted empirical state distribution. ESS is dimensionless
/// and is often compared with a chosen fraction of particle count to trigger
/// resampling. It measures weight concentration rather than estimator accuracy.
///
/// # Domain and special values
/// The shapes, finite values, positive mass, and work bounds from
/// [`particle_predict`] apply. Zero individual weights are valid.
///
/// # Returns
/// ESS in `1..=N`, up to f64 rounding; uniform weights give N.
///
/// # Errors
/// Invalid states, dimensions, weights, or work bounds return shared errors.
///
/// # Numerical behavior
/// Computes the sum of squared weights after overflow-safe max normalization; tiny
/// weights may underflow. Linear work in N with the common allocation bounds.
///
/// # Examples
/// ```
/// use sci_filter::{ParticleSet, particle::effective_sample_size};
/// let p = ParticleSet { states: vec![vec![0.],vec![1.]], weights: vec![1.,1.] };
/// assert_eq!(effective_sample_size(&p)?, 2.);
/// # Ok::<(), sci_filter::Error>(())
/// ```
///
/// # References
/// [Särkkä and Svensson, ch. 11](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf).
pub fn effective_sample_size(particles: &ParticleSet) -> Result<f64> {
    let weights = validated(particles)?;
    Ok(1. / weights.linear.iter().map(|w| w * w).sum::<f64>())
}

/// Systematically resamples at uᵢ = (i + offset)/N from the weighted CDF.
///
/// # Parameters and conventions
/// `offset` is one draw uniform on [0,1), supplied explicitly for reproducibility.
/// Particle i selects the first CDF entry strictly greater than uᵢ, skipping
/// zero-weight intervals. Uniform weights are assigned after resampling.
///
/// # Domain and special values
/// `offset` must be finite in [0,1). Particle shapes, weights and work bounds
/// follow [`particle_predict`]. Zero-weight particles are never selected.
///
/// # Returns
/// Exactly N rows in increasing CDF-selection order, each with weight 1/N.
///
/// # Errors
/// Invalid offset, states, shapes, weights, or work bounds return shared errors.
/// Inputs remain unchanged.
///
/// # Numerical behavior
/// Native linear CDF traversal uses the caller's explicit offset. Rounding near
/// CDF boundaries can change selected indices.
/// The final positive-mass particle absorbs CDF accumulation roundoff. Coordinate
/// copying is bounded by the common work and allocation limits.
///
/// # Examples
/// ```
/// use sci_filter::{ParticleSet, particle::systematic_resample};
/// let p = ParticleSet { states: vec![vec![0.],vec![1.]], weights: vec![0.,1.] };
/// assert_eq!(systematic_resample(&p,0.)?.states, vec![vec![1.],vec![1.]]);
/// # Ok::<(), sci_filter::Error>(())
/// ```
///
/// # References
/// [Douc, Cappé and Moulines, Comparison of Resampling Schemes for Particle Filtering](https://arxiv.org/abs/cs/0507025).
pub fn systematic_resample(particles: &ParticleSet, offset: f64) -> Result<ParticleSet> {
    if !offset.is_finite() || !(0. ..1.).contains(&offset) {
        return Err(Error::InvalidInput("resampling offset must be in [0,1)"));
    }
    let weights = validated(particles)?.linear;
    let count = particles.states.len();
    let last = weights.iter().rposition(|w| *w > 0.).unwrap();
    let mut states = Vec::with_capacity(count);
    let mut index = 0;
    let mut cumulative = weights[0];
    for i in 0..count {
        let u = (i as f64 + offset) / count as f64;
        while index < last && (cumulative <= u || weights[index] == 0.) {
            index += 1;
            cumulative += weights[index];
        }
        states.push(particles.states[index].clone());
    }
    Ok(ParticleSet {
        states,
        weights: vec![1. / count as f64; count],
    })
}
