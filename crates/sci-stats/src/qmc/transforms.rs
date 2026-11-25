//! Distribution transforms driven by a one- or multi-dimensional Sobol sequence.

use super::Sobol;
use crate::{Error, Result};
use nalgebra::{DMatrix, SymmetricEigen};
use sci_special::probability::ndtri;

const MAX_DIMENSIONS: usize = 64;
const MAX_OUTPUT_VALUES: usize = 1_048_576;
const SOBOL_BITS: usize = 30;
const MULTINOMIAL_CHUNK_SIZE: usize = 4096;

fn covariance_root(mean: &[f64], covariance: Option<&[Vec<f64>]>) -> Result<Option<Vec<Vec<f64>>>> {
    let Some(rows) = covariance else {
        return Ok(None);
    };
    let d = mean.len();
    if rows.len() != d || rows.iter().any(|row| row.len() != d) {
        return Err(Error::DimensionMismatch);
    }
    if rows.iter().flatten().any(|x| !x.is_finite()) {
        return Err(Error::InvalidInput("covariance entries must be finite"));
    }
    let mut matrix = DMatrix::from_fn(d, d, |i, j| rows[i][j]);
    for i in 0..d {
        if matrix[(i, i)] < 0.0 {
            return Err(Error::InvalidInput(
                "covariance must be positive semidefinite",
            ));
        }
        for j in 0..i {
            let a = matrix[(i, j)];
            let b = matrix[(j, i)];
            if (a - b).abs() > 1e-12 * a.abs().max(b.abs()) {
                return Err(Error::InvalidInput("covariance must be symmetric"));
            }
            let symmetric = 0.5 * a + 0.5 * b;
            matrix[(i, j)] = symmetric;
            matrix[(j, i)] = symmetric;
        }
    }

    // A row of the stored factor contains the coefficients of the independent
    // standard normals contributing to that output coordinate: Sigma = L L^T.
    if let Some(cholesky) = matrix.clone().cholesky() {
        let factor = cholesky.l();
        if factor.iter().any(|value| !value.is_finite()) {
            return Err(Error::NonFiniteEvaluation);
        }
        return Ok(Some(
            (0..d)
                .map(|i| (0..d).map(|j| factor[(i, j)]).collect())
                .collect(),
        ));
    }

    // Cholesky rejects singular PSD covariance matrices. For those, use an
    // eigenvalue factor, clipping only roundoff-sized negative eigenvalues.
    let eigen =
        SymmetricEigen::try_new(matrix, f64::EPSILON, 10_000).ok_or(Error::NonFiniteEvaluation)?;
    let scale = eigen
        .eigenvalues
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    if !scale.is_finite() {
        return Err(Error::NonFiniteEvaluation);
    }
    let tolerance = 1e-12 * scale;
    if eigen.eigenvalues.iter().any(|&v| v < -tolerance) {
        return Err(Error::InvalidInput(
            "covariance must be positive semidefinite",
        ));
    }
    let mut factor = vec![vec![0.0; d]; d];
    for j in 0..d {
        let root = eigen.eigenvalues[j].max(0.0).sqrt();
        for (i, row) in factor.iter_mut().enumerate() {
            row[j] = eigen.eigenvectors[(i, j)] * root;
            if !row[j].is_finite() {
                return Err(Error::NonFiniteEvaluation);
            }
        }
    }
    Ok(Some(factor))
}

/// A Sobol-driven sampler of the multivariate normal distribution `N(μ, Σ)`.
///
/// With a covariance factor `L` satisfying `Σ = L Lᵀ`, samples are
/// `x = μ + L z`, where `zⱼ = Φ⁻¹(0.5 + (1 − 10⁻¹⁰)(uⱼ − 0.5))` and `u`
/// is a Sobol point. This follows SciPy's inverse-transform convention.
///
/// ## Parameters and conventions
///
/// [`Self::new`] starts an unscrambled Sobol sequence; [`Self::scrambled`]
/// reproducibly scrambles it using `seed`. `mean` has one component per
/// coordinate; an omitted covariance uses the identity matrix. An explicit
/// covariance has one row per coordinate.
///
/// ## Domain and special values
///
/// Dimensions must be `1..=64`; mean and covariance entries must be finite,
/// and the covariance must be symmetric positive semidefinite. Singular
/// covariances, including zero covariance, are supported. The inverse-normal
/// clipping maps the Sobol origin to finite values instead of `−∞`.
///
/// ## Returns
///
/// Samples have shape `(n, d)` and the same units as the mean. Each call
/// advances the underlying Sobol sequence; `random(0)` is a no-op.
///
/// ## Errors
///
/// Invalid dimensions, covariance and oversized batches return an error.
/// Numerical overflow in the affine transform returns
/// [`Error::NonFiniteEvaluation`] without advancing the engine.
///
/// ## Numerical behavior
///
/// The Sobol net has 30 bits and at most `2³⁰` points. Each batch has at most
/// 1,048,576 scalar values. Clipping limits normal tails to approximately
/// ±6.467 standard deviations; consequently, their distribution is truncated
/// at those endpoints. Covariance factorization costs `O(d³)`; a symmetric
/// eigendecomposition handles positive semidefinite rank-deficient matrices.
/// SciPy and Sci use different seeded Sobol scrambles.
///
/// # Examples
///
/// ```
/// use sci_stats::qmc::MultivariateNormalQMC;
/// let mut sampler = MultivariateNormalQMC::new(&[1.0, -2.0], None)?;
/// let rows = sampler.random(2)?;
/// assert_eq!(rows.len(), 2);
/// assert_eq!(rows[1], vec![1.0, -2.0]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `qmc.MultivariateNormalQMC`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.MultivariateNormalQMC.html).
#[derive(Clone, Debug)]
pub struct MultivariateNormalQMC {
    mean: Vec<f64>,
    covariance_root: Option<Vec<Vec<f64>>>,
    engine: Sobol,
}

impl MultivariateNormalQMC {
    /// Construct an unscrambled normal QMC generator with optional covariance.
    ///
    /// The first Sobol point is the origin. Mean dimensions must be `1..=64`;
    /// covariance must be finite, symmetric positive semidefinite and square.
    /// See [`MultivariateNormalQMC`] for transform and error conventions.
    pub fn new(mean: &[f64], covariance: Option<&[Vec<f64>]>) -> Result<Self> {
        Self::build(mean, covariance, None)
    }

    /// Construct a reproducibly scrambled normal QMC generator.
    ///
    /// `seed` controls the Sobol digital scramble. Covariance validation,
    /// dimensions and work bounds are the same as [`Self::new`].
    pub fn scrambled(mean: &[f64], covariance: Option<&[Vec<f64>]>, seed: u64) -> Result<Self> {
        Self::build(mean, covariance, Some(seed))
    }

    fn build(mean: &[f64], covariance: Option<&[Vec<f64>]>, seed: Option<u64>) -> Result<Self> {
        if mean.is_empty() || mean.len() > MAX_DIMENSIONS || mean.iter().any(|x| !x.is_finite()) {
            return Err(Error::InvalidInput(
                "normal QMC mean needs 1..=64 finite values",
            ));
        }
        let covariance_root = covariance_root(mean, covariance)?;
        let engine = match seed {
            Some(seed) => Sobol::scrambled(mean.len(), SOBOL_BITS, seed)?,
            None => Sobol::new(mean.len(), SOBOL_BITS)?,
        };
        Ok(Self {
            mean: mean.to_vec(),
            covariance_root,
            engine,
        })
    }

    /// Number of coordinates per sample, `mean.len()`.
    pub fn dimensions(&self) -> usize {
        self.mean.len()
    }

    /// Number of samples already consumed or skipped.
    pub fn num_generated(&self) -> u128 {
        self.engine.num_generated()
    }

    /// Draw the next `n` multivariate normal samples.
    ///
    /// ## Parameters and conventions
    ///
    /// `n` is the number of output rows; zero returns an empty sample.
    /// The underlying sequence has 30 bits.
    ///
    /// ## Returns
    ///
    /// Finite row-major samples with shape `(n, dimensions())`.
    ///
    /// ## Errors
    ///
    /// Invalid sample count, exhausted sequence and numerical overflow are
    /// errors. On failure the generator's state is unchanged.
    ///
    /// ## Numerical behavior
    ///
    /// `O(nd²)` with covariance, `O(nd)` with identity covariance. The inverse
    /// normal transform uses the finite endpoint clipping described above.
    pub fn random(&mut self, n: usize) -> Result<Vec<Vec<f64>>> {
        let mut candidate = self.engine.clone();
        let uniforms = candidate.random_flat(n)?;
        let d = self.dimensions();
        let mut rows = Vec::with_capacity(n);
        let mut independent = Vec::with_capacity(d);
        for unit in uniforms.chunks_exact(d) {
            independent.clear();
            independent.extend(unit.iter().map(|&u| ndtri(0.5 + (1.0 - 1e-10) * (u - 0.5))));
            let mut sample = Vec::with_capacity(d);
            for (j, &mean) in self.mean.iter().enumerate() {
                let transformed = match &self.covariance_root {
                    Some(root) => root[j]
                        .iter()
                        .zip(&independent)
                        .fold(mean, |acc, (&coefficient, &z)| coefficient.mul_add(z, acc)),
                    None => mean + independent[j],
                };
                if !transformed.is_finite() {
                    return Err(Error::NonFiniteEvaluation);
                }
                sample.push(transformed);
            }
            rows.push(sample);
        }
        self.engine = candidate;
        Ok(rows)
    }

    /// Skip `n` normal samples using the Sobol sequence index.
    ///
    /// Returns an error if skipping would exceed `2³⁰` points and otherwise
    /// advances in `O(d log n)` work without constructing normal samples.
    pub fn fast_forward(&mut self, n: usize) -> Result<()> {
        self.engine.fast_forward(n)
    }

    /// Restart at the first normal sample, retaining the scramble and covariance.
    pub fn reset(&mut self) {
        self.engine.reset();
    }
}

/// A Sobol-driven sampler of multinomial category counts.
///
/// Each sample consists of `n_trials` independent categorical inverse-CDF
/// mappings: `cⱼ = #{t : Pⱼ₋₁ ≤ uₜ < Pⱼ}`, where `Pⱼ = Σᵢ≤ⱼ pᵢ` and `P₋₁ = 0`.
/// The `uₜ` are consecutive points from a one-dimensional Sobol sequence.
///
/// ## Parameters and conventions
///
/// `probabilities` lists categories in order; each output row contains their
/// counts. [`Self::new`] is unscrambled and [`Self::scrambled`] is reproducibly
/// seeded. The Sobol engine is consumed across repeated calls.
///
/// ## Domain and special values
///
/// `1..=64` categories with finite nonnegative probabilities summing to one
/// within `1e-12`. Zero-probability categories always have zero counts.
/// `0..=1_048_576` trials are supported; zero trials produce all-zero rows.
///
/// ## Returns
///
/// `Vec<Vec<usize>>` of shape `(n, probabilities.len())`; each row sums exactly
/// to `n_trials`. Counts have no units.
///
/// ## Errors
///
/// Invalid probabilities or trial counts, exceeding 1,048,576 sample values or
/// QMC draws per call, and exhaustion of the 30-bit Sobol sequence are errors.
/// Failure does not consume any Sobol draws.
///
/// ## Numerical behavior
///
/// Uses inverse-CDF half-open intervals to respect impossible categories even
/// at the initial Sobol point `u=0`. At a category boundary the point belongs
/// to the higher category, unlike SciPy 1.18.1's inclusive-boundary behavior.
/// This uses deterministic, highly dependent low-discrepancy draws; it is not
/// pseudorandom multinomial sampling. SciPy and Sci seeded scrambles differ.
///
/// # Examples
///
/// ```
/// use sci_stats::qmc::MultinomialQMC;
/// let mut sampler = MultinomialQMC::new(&[0.25, 0.75], 4)?;
/// let rows = sampler.random(2)?;
/// assert_eq!(rows.len(), 2);
/// assert!(rows.iter().all(|counts| counts.iter().sum::<usize>() == 4));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `qmc.MultinomialQMC`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.MultinomialQMC.html).
#[derive(Clone, Debug)]
pub struct MultinomialQMC {
    cumulative: Vec<f64>,
    n_trials: usize,
    engine: Sobol,
    samples_generated: u128,
}

impl MultinomialQMC {
    /// Construct an unscrambled multinomial QMC sampler.
    ///
    /// Probabilities must be finite, nonnegative, nonempty and sum to one
    /// within `1e-12`; the trial count is bounded by 1,048,576.
    pub fn new(probabilities: &[f64], n_trials: usize) -> Result<Self> {
        Self::build(probabilities, n_trials, None)
    }

    /// Construct a reproducibly scrambled multinomial QMC sampler.
    ///
    /// The `seed` selects the one-dimensional Sobol scramble. All input
    /// restrictions are identical to [`Self::new`].
    pub fn scrambled(probabilities: &[f64], n_trials: usize, seed: u64) -> Result<Self> {
        Self::build(probabilities, n_trials, Some(seed))
    }

    fn build(probabilities: &[f64], n_trials: usize, seed: Option<u64>) -> Result<Self> {
        if probabilities.is_empty()
            || probabilities.len() > MAX_DIMENSIONS
            || n_trials > MAX_OUTPUT_VALUES
            || probabilities.iter().any(|&p| !p.is_finite() || p < 0.0)
        {
            return Err(Error::InvalidInput(
                "invalid multinomial QMC probabilities or trials",
            ));
        }
        let mut cumulative = Vec::with_capacity(probabilities.len());
        let mut sum = 0.0;
        for &p in probabilities {
            sum += p;
            cumulative.push(sum);
        }
        if !sum.is_finite() || (sum - 1.0).abs() > 1e-12 {
            return Err(Error::InvalidInput(
                "multinomial probabilities must sum to one",
            ));
        }
        // The final interval covers all u < 1 even when probability rounding
        // produces a cumulative value a few ulps below one.
        *cumulative.last_mut().unwrap() = 1.0;
        let engine = match seed {
            Some(seed) => Sobol::scrambled(1, SOBOL_BITS, seed)?,
            None => Sobol::new(1, SOBOL_BITS)?,
        };
        Ok(Self {
            cumulative,
            n_trials,
            engine,
            samples_generated: 0,
        })
    }

    /// Number of output categories per sample.
    pub fn categories(&self) -> usize {
        self.cumulative.len()
    }

    /// Number of categorical trials per output sample.
    pub fn trials(&self) -> usize {
        self.n_trials
    }

    /// Number of multinomial samples already generated or skipped.
    pub fn num_generated(&self) -> u128 {
        self.samples_generated
    }

    /// Draw the next `n` multinomial count vectors.
    ///
    /// ## Parameters and conventions
    ///
    /// `n` is a count of multinomial samples. Each consumes `n_trials` Sobol
    /// points. `random(0)` is a no-op; when `n_trials=0`, rows are all zeros.
    ///
    /// ## Returns
    ///
    /// Row-major integer counts with shape `(n, categories())`, each summing
    /// to `n_trials`.
    ///
    /// ## Errors
    ///
    /// Returns an error if `n * categories` or `n * n_trials` exceeds
    /// 1,048,576, the Sobol sequence is exhausted, or a count overflows.
    /// On failure no draws are consumed.
    ///
    /// ## Numerical behavior
    ///
    /// The `n * n_trials` Sobol draws are mapped by cumulative probabilities
    /// in `O(n * n_trials * log(categories))` time.
    pub fn random(&mut self, n: usize) -> Result<Vec<Vec<usize>>> {
        let Some(draws) = n.checked_mul(self.n_trials) else {
            return Err(Error::InvalidInput(
                "multinomial QMC batch exceeds 1048576 values",
            ));
        };
        if n.checked_mul(self.categories())
            .is_none_or(|v| v > MAX_OUTPUT_VALUES)
            || draws > MAX_OUTPUT_VALUES
        {
            return Err(Error::InvalidInput(
                "multinomial QMC batch exceeds 1048576 values",
            ));
        }
        let count = self
            .samples_generated
            .checked_add(n as u128)
            .ok_or(Error::InvalidInput("multinomial sample counter overflow"))?;
        let mut candidate = self.engine.clone();
        let mut samples = vec![vec![0; self.categories()]; n];
        let mut uniforms = Vec::new();
        let mut remaining_draws = draws;
        let mut consumed = 0;
        while remaining_draws > 0 {
            let chunk = remaining_draws.min(MULTINOMIAL_CHUNK_SIZE);
            candidate.random_flat_into(chunk, &mut uniforms)?;
            for (offset, &point) in uniforms.iter().enumerate() {
                let category = self.cumulative.partition_point(|&p| p <= point);
                samples[(consumed + offset) / self.n_trials][category] += 1;
            }
            remaining_draws -= chunk;
            consumed += chunk;
        }
        self.engine = candidate;
        self.samples_generated = count;
        Ok(samples)
    }

    /// Skip `n` multinomial samples without materializing their trials.
    ///
    /// Skips `n * n_trials` Sobol points; returns an error if multiplication
    /// overflows or the sequence would be exhausted. Zero trials skip nothing.
    pub fn fast_forward(&mut self, n: usize) -> Result<()> {
        let draws = n
            .checked_mul(self.n_trials)
            .ok_or(Error::InvalidInput("multinomial QMC skip count overflow"))?;
        let count = self
            .samples_generated
            .checked_add(n as u128)
            .ok_or(Error::InvalidInput("multinomial sample counter overflow"))?;
        self.engine.fast_forward(draws)?;
        self.samples_generated = count;
        Ok(())
    }

    /// Restart sampling at the first Sobol point, preserving the scramble.
    pub fn reset(&mut self) {
        self.engine.reset();
        self.samples_generated = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::{MultinomialQMC, MultivariateNormalQMC, SOBOL_BITS};
    use crate::qmc::Sobol;

    #[test]
    fn normal_unscrambled_matches_scipy_inverse_transform() {
        // scipy.stats.qmc.MultivariateNormalQMC(mean=[1,-2], engine=Sobol(2,
        // scramble=False, bits=30)).random(4), SciPy 1.18.1.
        let mut sampler = MultivariateNormalQMC::new(&[1.0, -2.0], None).unwrap();
        let expected = [
            [-5.466951074732418, -8.466951074732417],
            [1.0, -2.0],
            [1.6744897501174103, -2.6744897501174103],
            [0.3255102498825898, -1.3255102498825897],
        ];
        let rows = sampler.random(4).unwrap();
        for (row, expected) in rows.iter().zip(expected) {
            for (observed, expected) in row.iter().zip(expected) {
                assert!((observed - expected).abs() < 2e-12);
            }
        }
        assert!(rows.iter().flatten().all(|v| v.is_finite()));
        assert_eq!(sampler.num_generated(), 4);
        assert!(sampler.random(0).unwrap().is_empty());
        sampler.reset();
        assert_eq!(sampler.random(4).unwrap(), rows);
    }

    #[test]
    fn normal_covariance_and_rank_deficiency() {
        let full_rank = [vec![1.0, 0.6], vec![0.6, 2.0]];
        let mut correlated = MultivariateNormalQMC::new(&[0.0, 0.0], Some(&full_rank)).unwrap();
        let mut independent = MultivariateNormalQMC::new(&[0.0, 0.0], None).unwrap();
        for (row, z) in correlated
            .random(16)
            .unwrap()
            .iter()
            .zip(independent.random(16).unwrap())
        {
            assert!((row[0] - z[0]).abs() < 1e-12);
            assert!((row[1] - (0.6 * z[0] + 1.64_f64.sqrt() * z[1])).abs() < 1e-12);
        }

        let covariance = [vec![4.0, 2.0], vec![2.0, 1.0]];
        let mut sampler = MultivariateNormalQMC::new(&[10.0, -3.0], Some(&covariance)).unwrap();
        for row in sampler.random(64).unwrap() {
            assert!((row[0] - 10.0 - 2.0 * (row[1] + 3.0)).abs() < 1e-12);
        }
        let mut zero = MultivariateNormalQMC::new(&[2.0], Some(&[vec![0.0]])).unwrap();
        assert_eq!(zero.random(8).unwrap(), vec![vec![2.0]; 8]);
        let mut independent = MultivariateNormalQMC::scrambled(&[0.0, 0.0], None, 42).unwrap();
        let first = independent.random(32).unwrap();
        independent.reset();
        assert_eq!(independent.random(32).unwrap(), first);
        let mut other = MultivariateNormalQMC::scrambled(&[0.0, 0.0], None, 43).unwrap();
        assert_ne!(first, other.random(32).unwrap());
    }

    #[test]
    fn normal_invalid_covariance_and_atomic_skip() {
        assert!(MultivariateNormalQMC::new(&[], None).is_err());
        assert!(MultivariateNormalQMC::new(&[f64::NAN], None).is_err());
        assert!(MultivariateNormalQMC::new(&[0.0], Some(&[])).is_err());
        assert!(MultivariateNormalQMC::new(&[0.0, 0.0], Some(&[vec![1.0]])).is_err());
        assert!(
            MultivariateNormalQMC::new(&[0.0, 0.0], Some(&[vec![1.0, 2.0], vec![1.0, 1.0]]))
                .is_err()
        );
        assert!(
            MultivariateNormalQMC::new(&[0.0, 0.0], Some(&[vec![1.0, 2.0], vec![2.0, 1.0]]))
                .is_err()
        );
        let mut sampler = MultivariateNormalQMC::new(&[0.0], None).unwrap();
        assert!(sampler.random(1_048_577).is_err());
        assert_eq!(sampler.num_generated(), 0);
        sampler.fast_forward(5).unwrap();
        let next = sampler.random(1).unwrap();
        let mut replay = MultivariateNormalQMC::new(&[0.0], None).unwrap();
        replay.random(5).unwrap();
        assert_eq!(next, replay.random(1).unwrap());
    }

    #[test]
    fn multinomial_counts_respect_probabilities() {
        let mut sampler = MultinomialQMC::new(&[0.0, 0.25, 0.75], 4).unwrap();
        let rows = sampler.random(32).unwrap();
        assert_eq!(rows.len(), 32);
        assert!(
            rows.iter()
                .all(|row| row[0] == 0 && row.iter().sum::<usize>() == 4)
        );
        assert_eq!(rows[0], vec![0, 1, 3]);
        assert_eq!(sampler.num_generated(), 32);
        sampler.reset();
        assert_eq!(sampler.random(32).unwrap(), rows);

        let mut certain = MultinomialQMC::new(&[1.0, 0.0], 100).unwrap();
        assert_eq!(certain.random(2).unwrap(), vec![vec![100, 0]; 2]);
        let mut degenerate = MultinomialQMC::new(&[0.5, 0.5], 0).unwrap();
        assert_eq!(degenerate.random(2).unwrap(), vec![vec![0, 0]; 2]);
        assert_eq!(degenerate.num_generated(), 2);
    }

    #[test]
    fn multinomial_chunk_boundaries_preserve_sobol_trial_order() {
        let trials = 4097;
        let mut sampler = MultinomialQMC::scrambled(&[0.2, 0.3, 0.5], trials, 91).unwrap();
        let samples = sampler.random(5).unwrap();

        let mut sobol = Sobol::scrambled(1, SOBOL_BITS, 91).unwrap();
        let draws = sobol.random(5 * trials).unwrap();
        let expected: Vec<_> = draws
            .chunks(trials)
            .map(|chunk| {
                let mut counts = vec![0_usize; 3];
                for point in chunk {
                    let category = if point[0] < 0.2 {
                        0
                    } else if point[0] < 0.5 {
                        1
                    } else {
                        2
                    };
                    counts[category] += 1;
                }
                counts
            })
            .collect();
        assert_eq!(samples, expected);
        assert_eq!(sampler.num_generated(), 5);

        let next = sampler.random(2).unwrap();
        let mut skipped = MultinomialQMC::scrambled(&[0.2, 0.3, 0.5], trials, 91).unwrap();
        skipped.fast_forward(5).unwrap();
        assert_eq!(next, skipped.random(2).unwrap());
    }

    #[test]
    fn multinomial_seed_skip_and_invalid_inputs() {
        let p = [0.3, 0.7];
        let mut seeded = MultinomialQMC::scrambled(&p, 5, 31).unwrap();
        let first = seeded.random(8).unwrap();
        seeded.reset();
        assert_eq!(seeded.random(8).unwrap(), first);
        let mut different = MultinomialQMC::scrambled(&p, 5, 32).unwrap();
        assert_ne!(different.random(8).unwrap(), first);
        let mut skipped = MultinomialQMC::new(&p, 5).unwrap();
        skipped.fast_forward(3).unwrap();
        let next = skipped.random(4).unwrap();
        let mut sequential = MultinomialQMC::new(&p, 5).unwrap();
        sequential.random(3).unwrap();
        assert_eq!(sequential.random(4).unwrap(), next);

        assert!(MultinomialQMC::new(&[], 1).is_err());
        assert!(MultinomialQMC::new(&[0.2, 0.7], 1).is_err());
        assert!(MultinomialQMC::new(&[-0.1, 1.1], 1).is_err());
        assert!(MultinomialQMC::new(&[f64::NAN, 1.0], 1).is_err());
        assert!(MultinomialQMC::new(&[0.5, 0.5], 1_048_577).is_err());
        assert!(sequential.random(1_048_577).is_err());
        assert_eq!(sequential.num_generated(), 7);
    }
}
