//! Finite discrete distributions sampled with Walker alias and guide tables.

use crate::{Error, Result};
use rand::{Rng, SeedableRng, rngs::StdRng};

const MAX_CATEGORIES: usize = 1_048_576;
const MAX_DRAWS: usize = 1_048_576;

// Scale by the largest weight before summing: two f64::MAX weights must be as
// usable as two unit weights. A compensated sum limits normalization drift.
fn normalized_weights(weights: &[f64]) -> Result<Vec<f64>> {
    if weights.is_empty() || weights.len() > MAX_CATEGORIES {
        return Err(Error::InvalidInput(
            "discrete probability vector length must be in 1..=1048576",
        ));
    }
    if weights.iter().any(|&w| !w.is_finite() || w < 0.0) {
        return Err(Error::InvalidInput(
            "discrete weights must be finite and nonnegative",
        ));
    }
    let maximum = weights.iter().copied().fold(0.0_f64, f64::max);
    if maximum == 0.0 {
        return Err(Error::InvalidInput(
            "discrete weights must contain positive mass",
        ));
    }

    let mut scaled = Vec::with_capacity(weights.len());
    let (mut total, mut correction) = (0.0_f64, 0.0_f64);
    for &weight in weights {
        let value = weight / maximum;
        scaled.push(value);
        let adjusted = value - correction;
        let next = total + adjusted;
        correction = (next - total) - adjusted;
        total = next;
    }
    for weight in &mut scaled {
        *weight /= total;
    }
    Ok(scaled)
}

fn validate_draw_count(n: usize) -> Result<()> {
    if n > MAX_DRAWS {
        Err(Error::InvalidInput(
            "discrete sample batch exceeds 1048576 draws",
        ))
    } else {
        Ok(())
    }
}

/// Walker–Vose alias sampler for a finite categorical probability distribution.
///
/// Given nonnegative weights `wᵢ` with positive total, category `i` has
/// probability `P(X=i) = wᵢ / Σⱼ wⱼ`. The alias table partitions a uniform
/// choice of `k` columns into two weighted category outcomes per column.
///
/// ## Parameters and conventions
///
/// [`DiscreteAliasUrn::new`] accepts a vector of `k` weights and a `u64` seed.
/// Category indices are zero-based, and the original weights need not sum to
/// one. The seed initializes a Rust `StdRng`, independent of NumPy's RNG.
///
/// ## Domain and special values
///
/// `1 <= k <= 1_048_576`; weights must be finite, nonnegative, and not all
/// zero. Signed zero is allowed. Zero-weight categories are never sampled.
/// NaN, infinity, and negative weights are rejected.
///
/// ## Returns
///
/// [`DiscreteAliasUrn::rvs`] returns `Vec<usize>` categories in `0..k`.
/// A zero-length draw returns an empty vector without consuming random state.
///
/// ## Errors
///
/// [`Error::InvalidInput`] reports invalid weights, lengths, or batches over
/// 1,048,576 draws. Invalid draws do not consume random state.
///
/// ## Numerical behavior
///
/// Building the table costs `O(k)` time and memory; each draw costs `O(1)`.
/// Weights are divided by their largest value before normalization so that
/// very large or subnormal finite weights do not overflow their sum. Ratios
/// below the representable `f64` range can underflow to zero. The default
/// one-column-per-category table uses the usual alias method, corresponding
/// to SciPy's `DiscreteAliasUrn` with `urn_factor=1`; seeded streams differ.
///
/// ## Examples
///
/// ```
/// use sci_stats::sampling::DiscreteAliasUrn;
/// let mut sampler = DiscreteAliasUrn::new(&[1.0, 3.0, 6.0], 42)?;
/// let draws = sampler.rvs(32)?;
/// assert!(draws.iter().all(|&category| category < 3));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// Walker (1977), *An efficient method for generating discrete random
/// variables with general distributions*, ACM TOMS 3, 253–256;
/// [SciPy `DiscreteAliasUrn`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sampling.DiscreteAliasUrn.html).
#[derive(Clone, Debug)]
pub struct DiscreteAliasUrn {
    threshold: Vec<f64>,
    alias: Vec<usize>,
    seed: u64,
    rng: StdRng,
}

impl DiscreteAliasUrn {
    /// Build the alias table from `weights` with a reproducible `seed`.
    ///
    /// ## Parameters and conventions
    ///
    /// `weights[i]` is proportional to `P(X=i)`; normalization is automatic.
    /// The seed selects a deterministic `StdRng` stream.
    ///
    /// ## Returns
    ///
    /// A stateful categorical sampler with `O(k)` preprocessing.
    ///
    /// ## Errors
    ///
    /// Returns [`Error::InvalidInput`] for empty, oversized, all-zero,
    /// negative, NaN, or infinite weight vectors.
    pub fn new(weights: &[f64], seed: u64) -> Result<Self> {
        let probabilities = normalized_weights(weights)?;
        let k = probabilities.len();
        let mut scaled: Vec<f64> = probabilities.into_iter().map(|p| p * k as f64).collect();
        let mut small = Vec::with_capacity(k);
        let mut large = Vec::with_capacity(k);
        for (index, &value) in scaled.iter().enumerate() {
            if value < 1.0 {
                small.push(index);
            } else {
                large.push(index);
            }
        }

        let mut threshold = vec![1.0; k];
        let mut alias: Vec<usize> = (0..k).collect();
        while !small.is_empty() && !large.is_empty() {
            let lower = small.pop().expect("nonempty small stack");
            let upper = large.pop().expect("nonempty large stack");
            threshold[lower] = scaled[lower];
            alias[lower] = upper;
            scaled[upper] = (scaled[upper] + scaled[lower]) - 1.0;
            if scaled[upper] < 1.0 {
                small.push(upper);
            } else {
                large.push(upper);
            }
        }

        Ok(Self {
            threshold,
            alias,
            seed,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    /// Generate `n` independent categorical samples.
    ///
    /// ## Parameters and conventions
    ///
    /// `n` is the batch length and may be zero. Calls advance the same seeded
    /// stream; use [`DiscreteAliasUrn::reset`] to replay it.
    ///
    /// ## Returns
    ///
    /// A vector of `n` zero-based category indices.
    ///
    /// ## Errors
    ///
    /// Returns [`Error::InvalidInput`] if `n > 1_048_576`; no random state is
    /// consumed when the count is invalid.
    ///
    /// ## Numerical behavior
    ///
    /// Each sample takes constant time and two independent RNG draws.
    pub fn rvs(&mut self, n: usize) -> Result<Vec<usize>> {
        validate_draw_count(n)?;
        let mut draws = Vec::with_capacity(n);
        let categories = self.threshold.len();
        for _ in 0..n {
            let column = self.rng.random_range(0..categories);
            let uniform = self.rng.random::<f64>();
            draws.push(if uniform < self.threshold[column] {
                column
            } else {
                self.alias[column]
            });
        }
        Ok(draws)
    }

    /// Restore the initial seeded sampling state without rebuilding the table.
    ///
    /// Subsequent draws repeat the same `StdRng` stream from construction.
    /// This method cannot fail and leaves the distribution unchanged.
    pub fn reset(&mut self) {
        self.rng = StdRng::seed_from_u64(self.seed);
    }
}

/// Discrete guide-table sampler and inverse CDF for finite category weights.
///
/// For nonnegative weights `wᵢ`, let `F(i) = Σⱼ₌₀ⁱ wⱼ / Σⱼ wⱼ`.
/// Inverse-CDF sampling draws `U` uniformly from `[0,1)` and returns the
/// smallest category `i` for which `F(i) >= U` (for `U > 0`). A guide table
/// locates a lower category bound for each uniform-probability interval.
///
/// ## Parameters and conventions
///
/// [`DiscreteGuideTable::new`] accepts `k` nonnegative weights and a `u64`
/// seed. Returned categories are zero-based; weights need not sum to one.
///
/// ## Domain and special values
///
/// `1 <= k <= 1_048_576`; at least one weight must be positive. NaN,
/// infinities, and negative weights are invalid. The quantile domain is the
/// closed interval `[0,1]`. By convention `ppf(0)` is the first positive-mass
/// category and `ppf(1)` the last positive-mass category. SciPy uses
/// `ppf(0)=-1` and can return a trailing zero-weight category for `ppf(1)`;
/// this Rust API always returns `usize` indices within the effective support.
///
/// ## Returns
///
/// [`DiscreteGuideTable::rvs`] returns `Vec<usize>` category samples;
/// [`DiscreteGuideTable::ppf`] maps a slice of probabilities to indices.
///
/// ## Errors
///
/// [`Error::InvalidInput`] reports invalid weights, quantiles, oversized
/// inputs, and draw counts above 1,048,576.
///
/// ## Numerical behavior
///
/// The cumulative distribution and guide contain `O(k)` entries. Setup costs
/// `O(k)`, with constant expected comparisons per draw or quantile and an
/// `O(k)` worst case for a query. Weights are scaled before summation to
/// avoid overflow. Tiny relative weights may become unrepresentable, and
/// binary64 CDF rounding affects quantiles extremely close to boundaries.
/// A guide factor of one follows SciPy's default method; random streams differ.
///
/// ## Examples
///
/// ```
/// use sci_stats::sampling::DiscreteGuideTable;
/// let mut sampler = DiscreteGuideTable::new(&[1.0, 3.0, 6.0], 7)?;
/// assert_eq!(sampler.ppf(&[0.1, 0.4, 0.9])?, vec![0, 1, 2]);
/// assert!(sampler.rvs(16)?.iter().all(|&category| category < 3));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// Chen and Asau (1974), *On generating random variates from an empirical
/// distribution*, AIIE Transactions 6, 163–166;
/// [SciPy `DiscreteGuideTable`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sampling.DiscreteGuideTable.html).
#[derive(Clone, Debug)]
pub struct DiscreteGuideTable {
    cdf: Vec<f64>,
    guide: Vec<usize>,
    first_positive: usize,
    last_positive: usize,
    seed: u64,
    rng: StdRng,
}

impl DiscreteGuideTable {
    /// Build a cumulative distribution and guide table from `weights`.
    ///
    /// ## Parameters and conventions
    ///
    /// `weights[i]` gives unnormalized mass at category `i`; `seed` creates
    /// a reproducible `StdRng`. The guide has one entry per category.
    ///
    /// ## Returns
    ///
    /// A stateful inverse-CDF sampler with `O(k)` setup and storage.
    ///
    /// ## Errors
    ///
    /// Returns [`Error::InvalidInput`] for empty, oversized, all-zero,
    /// negative, NaN, or infinite weight vectors.
    pub fn new(weights: &[f64], seed: u64) -> Result<Self> {
        let normalized = normalized_weights(weights)?;
        let k = normalized.len();
        let first_positive = weights
            .iter()
            .position(|&weight| weight > 0.0)
            .expect("validated nonzero weights");
        let last_positive = weights
            .iter()
            .rposition(|&weight| weight > 0.0)
            .expect("validated nonzero weights");

        let mut cdf = Vec::with_capacity(k);
        let (mut total, mut correction) = (0.0_f64, 0.0_f64);
        for probability in normalized {
            let adjusted = probability - correction;
            let next = total + adjusted;
            correction = (next - total) - adjusted;
            total = next;
            cdf.push(total.min(1.0));
        }
        cdf[k - 1] = 1.0;

        let mut guide = Vec::with_capacity(k);
        let mut index = 0;
        for bucket in 0..k {
            let lower = (bucket as f64) / (k as f64);
            while index + 1 < k && cdf[index] < lower {
                index += 1;
            }
            guide.push(index);
        }

        Ok(Self {
            cdf,
            guide,
            first_positive,
            last_positive,
            seed,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    fn quantile(&self, probability: f64) -> usize {
        if probability == 0.0 {
            return self.first_positive;
        }
        if probability == 1.0 {
            return self.last_positive;
        }
        // Floating-point multiplication can round a quantile just below a
        // guide boundary into the next bucket. Starting one bucket earlier
        // preserves the inverse-CDF ordering even at those boundaries.
        let bucket = ((probability * self.guide.len() as f64) as usize).min(self.guide.len() - 1);
        let mut index = self.guide[bucket.saturating_sub(1)];
        while self.cdf[index] < probability {
            index += 1;
        }
        index
    }

    /// Evaluate the discrete inverse cumulative distribution at `probabilities`.
    ///
    /// ## Parameters and conventions
    ///
    /// Quantiles must lie in `[0,1]`. Each result is the smallest index
    /// whose cumulative probability is at least the input, with endpoint
    /// conventions specified by [`DiscreteGuideTable`].
    ///
    /// ## Returns
    ///
    /// Indices in the same order as the input probabilities. An empty input
    /// returns an empty vector.
    ///
    /// ## Errors
    ///
    /// Returns [`Error::InvalidInput`] for NaN, infinity, values outside
    /// `[0,1]`, or more than 1,048,576 quantiles.
    ///
    /// ## Numerical behavior
    ///
    /// Uses a guide-table starting index followed by cumulative comparisons.
    /// A quantile exactly on a representable CDF boundary selects its first
    /// attaining category.
    pub fn ppf(&self, probabilities: &[f64]) -> Result<Vec<usize>> {
        validate_draw_count(probabilities.len())?;
        if probabilities
            .iter()
            .any(|&q| !q.is_finite() || !(0.0..=1.0).contains(&q))
        {
            return Err(Error::InvalidInput(
                "discrete quantiles must be finite and in [0,1]",
            ));
        }
        Ok(probabilities.iter().map(|&q| self.quantile(q)).collect())
    }

    /// Generate `n` independent samples using inverse-CDF lookup.
    ///
    /// ## Parameters and conventions
    ///
    /// `n` is the number of samples, and can be zero. A single seeded random
    /// stream is advanced across successive calls.
    ///
    /// ## Returns
    ///
    /// A vector of `n` zero-based category indices.
    ///
    /// ## Errors
    ///
    /// Returns [`Error::InvalidInput`] when `n > 1_048_576`, without
    /// consuming random state.
    ///
    /// ## Numerical behavior
    ///
    /// Each draw uses one uniform variate and guide-table CDF lookup.
    pub fn rvs(&mut self, n: usize) -> Result<Vec<usize>> {
        validate_draw_count(n)?;
        let mut draws = Vec::with_capacity(n);
        for _ in 0..n {
            let probability = self.rng.random::<f64>();
            draws.push(self.quantile(probability));
        }
        Ok(draws)
    }

    /// Restore the initial seeded stream without changing the distribution.
    ///
    /// Subsequent draws replay the same `StdRng` sequence from construction.
    /// This method cannot fail.
    pub fn reset(&mut self) {
        self.rng = StdRng::seed_from_u64(self.seed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_probability_vectors() {
        for weights in [
            &[][..],
            &[0.0, 0.0],
            &[-1.0, 2.0],
            &[f64::NAN, 1.0],
            &[f64::INFINITY, 1.0],
            &[f64::NEG_INFINITY, 1.0],
        ] {
            assert!(DiscreteAliasUrn::new(weights, 42).is_err());
            assert!(DiscreteGuideTable::new(weights, 42).is_err());
        }
        let oversized = vec![1.0; MAX_CATEGORIES + 1];
        assert!(DiscreteAliasUrn::new(&oversized, 42).is_err());
        assert!(DiscreteGuideTable::new(&oversized, 42).is_err());
    }

    #[test]
    fn inverse_cdf_endpoints_zero_weights_and_boundaries() {
        let guide = DiscreteGuideTable::new(&[0.0, 1.0, 0.0, 3.0, 0.0], 7).unwrap();
        let just_below = f64::from_bits(0.25_f64.to_bits() - 1);
        let just_above = f64::from_bits(0.25_f64.to_bits() + 1);
        assert_eq!(
            guide
                .ppf(&[0.0, just_below, 0.25, just_above, 0.999, 1.0])
                .unwrap(),
            [1, 1, 1, 3, 3, 3]
        );
        assert_eq!(
            DiscreteGuideTable::new(&[0.0, 0.0, 1.0], 5)
                .unwrap()
                .ppf(&[0.0, f64::MIN_POSITIVE, 1.0])
                .unwrap(),
            [2, 2, 2]
        );
        for invalid in [f64::NAN, f64::INFINITY, -0.1, 1.01] {
            assert!(guide.ppf(&[0.1, invalid]).is_err());
        }
        assert!(guide.ppf(&vec![0.5; MAX_DRAWS + 1]).is_err());
        assert_eq!(guide.ppf(&[]).unwrap(), Vec::<usize>::new());
    }

    #[test]
    fn guide_lookup_agrees_with_linear_inverse() {
        let weights = [1.0, 0.0, 4.0, 1.0, 0.0, 2.0, 0.5, 8.0];
        let guide = DiscreteGuideTable::new(&weights, 3).unwrap();
        let sum: f64 = weights.iter().sum();
        for step in 1..10_000 {
            let q = step as f64 / 10_000.0;
            let mut running = 0.0;
            let expected = weights
                .iter()
                .position(|&w| {
                    running += w / sum;
                    running >= q
                })
                .unwrap();
            assert_eq!(guide.ppf(&[q]).unwrap(), [expected], "q={q}");
        }
    }

    #[test]
    fn alias_and_guide_sample_proportions_and_zero_mass() {
        let mut alias = DiscreteAliasUrn::new(&[0.0, 1.0, 3.0, 6.0, 0.0], 42).unwrap();
        let mut guide = DiscreteGuideTable::new(&[0.0, 1.0, 3.0, 6.0, 0.0], 42).unwrap();
        for draws in [alias.rvs(40_000).unwrap(), guide.rvs(40_000).unwrap()] {
            let mut counts = [0_usize; 5];
            for index in draws {
                counts[index] += 1;
            }
            assert_eq!(counts[0], 0);
            assert_eq!(counts[4], 0);
            for (index, expected) in [(1, 0.1), (2, 0.3), (3, 0.6)] {
                assert!((counts[index] as f64 / 40_000.0 - expected).abs() < 0.015);
            }
        }
    }

    #[test]
    fn seeded_reset_and_invalid_batch_are_non_consuming() {
        let weights = [1.0, 2.0, 3.0];
        let mut alias = DiscreteAliasUrn::new(&weights, 12).unwrap();
        let initial = alias.rvs(32).unwrap();
        alias.reset();
        assert_eq!(alias.rvs(0).unwrap(), []);
        assert!(alias.rvs(MAX_DRAWS + 1).is_err());
        assert_eq!(alias.rvs(32).unwrap(), initial);
        assert_eq!(
            DiscreteAliasUrn::new(&weights, 12)
                .unwrap()
                .rvs(32)
                .unwrap(),
            initial
        );
        assert_ne!(
            DiscreteAliasUrn::new(&weights, 13)
                .unwrap()
                .rvs(32)
                .unwrap(),
            initial
        );

        let mut guide = DiscreteGuideTable::new(&weights, 12).unwrap();
        let initial = guide.rvs(32).unwrap();
        guide.reset();
        assert_eq!(guide.rvs(0).unwrap(), []);
        assert!(guide.rvs(MAX_DRAWS + 1).is_err());
        assert_eq!(guide.rvs(32).unwrap(), initial);
        assert_eq!(
            DiscreteGuideTable::new(&weights, 12)
                .unwrap()
                .rvs(32)
                .unwrap(),
            initial
        );
        assert_ne!(
            DiscreteGuideTable::new(&weights, 13)
                .unwrap()
                .rvs(32)
                .unwrap(),
            initial
        );
    }

    #[test]
    fn extreme_finite_weights_and_degenerate_distribution() {
        for weights in [
            [f64::MAX, f64::MAX, 0.0],
            [f64::MIN_POSITIVE, f64::MIN_POSITIVE, 0.0],
            [f64::from_bits(1), f64::from_bits(1), 0.0],
        ] {
            let mut alias = DiscreteAliasUrn::new(&weights, 19).unwrap();
            let mut guide = DiscreteGuideTable::new(&weights, 19).unwrap();
            for draws in [alias.rvs(512).unwrap(), guide.rvs(512).unwrap()] {
                assert!(draws.into_iter().all(|category| category < 2));
            }
            assert_eq!(guide.ppf(&[0.25, 0.5, 0.75, 1.0]).unwrap(), [0, 0, 1, 1]);
        }
        let mut alias = DiscreteAliasUrn::new(&[0.0, f64::MAX, 0.0], 19).unwrap();
        let mut guide = DiscreteGuideTable::new(&[0.0, f64::MAX, 0.0], 19).unwrap();
        assert_eq!(alias.rvs(512).unwrap(), vec![1; 512]);
        assert_eq!(guide.rvs(512).unwrap(), vec![1; 512]);
        assert_eq!(guide.ppf(&[0.0001, 1.0]).unwrap(), [1, 1]);
        assert!(DiscreteAliasUrn::new(&[f64::MAX, f64::from_bits(1)], 7).is_ok());
        assert!(DiscreteGuideTable::new(&[f64::MAX, f64::from_bits(1)], 7).is_ok());
    }
}
