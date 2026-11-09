//! Seeded strength-one Latin hypercube sampling.

use crate::{Error, Result};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

const MAX_OUTPUT_VALUES: usize = 1_048_576;

fn stratum_coordinate(stratum: usize, offset: f64, size: f64) -> f64 {
    let mut coordinate = (stratum as f64 - offset) / size;
    // Rounding can move a coordinate across either stratum boundary.
    // Enforce the assigned bin as measured by floor(size * coordinate).
    let scaled = coordinate * size;
    if scaled < (stratum - 1) as f64 || scaled >= stratum as f64 {
        coordinate = (stratum as f64 - 0.5) / size;
    }
    coordinate
}

/// Stateful generator of independently stratified Latin hypercube designs.
///
/// A batch of `n` points has exactly one point in each interval
/// `[k/n, (k+1)/n)` along each coordinate, for `k = 0, ..., n-1`.
/// Independently permuting these `n` intervals across coordinates gives a
/// strength-one Latin hypercube. With scrambling, a uniform random position
/// is drawn inside each interval; centered sampling uses its midpoint.
///
/// ## Parameters and conventions
///
/// [`LatinHypercube::new`] accepts a dimension `d` and a reproducible `u64`
/// seed, and enables within-stratum scrambling. [`LatinHypercube::centered`]
/// uses fixed midpoints but still randomly permutes the intervals according
/// to the seed. Each [`LatinHypercube::random`] call creates a new design:
/// separate batches are stratified individually, not as one combined design.
///
/// ## Domain and special values
///
/// Dimensions are in `1..=1048576`. Each batch contains at most 1,048,576
/// scalar values (`n * d`). Zero-length batches do not change the generator.
///
/// ## Returns
///
/// Each batch is a row-major `Vec<Vec<f64>>` of shape `(n, d)` with values in
/// `[0, 1)`. All coordinates are dimensionless.
///
/// ## Errors
///
/// Invalid dimensions, oversized batches, or an overflowing cumulative point
/// count return [`Error::InvalidInput`] without consuming the random state.
///
/// ## Numerical behavior
///
/// Samples follow the strength-one algorithm in SciPy's `qmc.LatinHypercube`:
/// draw a uniform offset for each coordinate, independently shuffle the
/// integer strata for each dimension, then compute `(stratum - offset)/n`.
/// Floating-point boundary rounding is corrected into the chosen stratum.
/// The generator uses `rand`'s `StdRng`, so identical seeds reproduce results
/// within a pinned `rand` version, but differ from NumPy's random stream.
/// Both drawing and skipping a batch cost `O(n*d)` time; skipping uses `O(n)`
/// scratch space. Optimized or orthogonal-array (strength-two) designs are
/// not provided.
///
/// # Examples
///
/// ```
/// use sci_stats::qmc::LatinHypercube;
/// let mut sampler = LatinHypercube::new(2, 42)?;
/// let sample = sampler.random(8)?;
/// for dimension in 0..2 {
///     let mut strata: Vec<usize> = sample.iter()
///         .map(|row| (row[dimension] * 8.0).floor() as usize).collect();
///     strata.sort_unstable();
///     assert_eq!(strata, (0..8).collect::<Vec<_>>());
/// }
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [McKay, Beckman and Conover (1979), *A Comparison of Three Methods for
/// Selecting Values of Input Variables in the Analysis of Output from a
/// Computer Code*](https://doi.org/10.1080/00401706.1979.10489755) and
/// [SciPy `qmc.LatinHypercube`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html).
#[derive(Clone, Debug)]
pub struct LatinHypercube {
    dimensions: usize,
    scramble: bool,
    seed: u64,
    rng: StdRng,
    generated: u128,
}

impl LatinHypercube {
    /// Construct a seeded, scrambled strength-one Latin hypercube generator.
    ///
    /// `d` must be in `1..=1048576`, otherwise returns [`Error::InvalidInput`].
    /// A given `(d, seed)` reproduces the same batches and random permutations.
    pub fn new(d: usize, seed: u64) -> Result<Self> {
        Self::with_scramble(d, seed, true)
    }

    /// Construct a seeded Latin hypercube with fixed stratum midpoints.
    ///
    /// Sampling uses `x = (k + 1/2)/n` for a permutation of `k = 0..n` in
    /// each dimension. The permutation is reproducible with the supplied seed.
    /// Returns [`Error::InvalidInput`] for dimensions outside `1..=1048576`.
    pub fn centered(d: usize, seed: u64) -> Result<Self> {
        Self::with_scramble(d, seed, false)
    }

    /// Construct a seeded generator, selecting random or centered offsets.
    ///
    /// If `scramble` is `true`, each point's position inside its stratum is
    /// uniform; otherwise the positions are fixed at the centers and only
    /// the stratum permutations depend on `seed`. Invalid dimensions return
    /// [`Error::InvalidInput`].
    pub fn with_scramble(d: usize, seed: u64, scramble: bool) -> Result<Self> {
        if d == 0 || d > MAX_OUTPUT_VALUES {
            return Err(Error::InvalidInput(
                "Latin hypercube dimensions must be in 1..=1048576",
            ));
        }
        Ok(Self {
            dimensions: d,
            scramble,
            seed,
            rng: StdRng::seed_from_u64(seed),
            generated: 0,
        })
    }

    /// Number of coordinates in each generated point.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Cumulative number of generated or skipped points since reset.
    pub fn num_generated(&self) -> u128 {
        self.generated
    }

    /// Generate a new independently stratified batch of `n` points.
    ///
    /// ## Parameters and conventions
    ///
    /// `n` is the number of rows; every coordinate independently occupies
    /// each of the `n` equal-width strata exactly once. A subsequent call
    /// builds a separate design, even when its size is unchanged.
    ///
    /// ## Domain and special values
    ///
    /// At most 1,048,576 scalar values (`n * d`) per batch. `n = 0` returns
    /// an empty vector without changing the RNG state or point count.
    ///
    /// ## Returns
    ///
    /// A row-major `(n, d)` matrix of dimensionless coordinates in `[0, 1)`.
    ///
    /// ## Errors
    ///
    /// Oversized batches and counter overflow return [`Error::InvalidInput`]
    /// before drawing any random values.
    ///
    /// ## Numerical behavior
    ///
    /// `O(n*d)` work and output memory. Positions in a stratum are uniform
    /// if scrambled or at its midpoint if centered.
    pub fn random(&mut self, n: usize) -> Result<Vec<Vec<f64>>> {
        let end = self.check_batch(n)?;
        if n == 0 {
            return Ok(Vec::new());
        }

        // SciPy draws all n*d jitter values before shuffling any dimension.
        let mut points = vec![vec![0.5; self.dimensions]; n];
        if self.scramble {
            for row in &mut points {
                for value in row {
                    *value = self.rng.random::<f64>();
                }
            }
        }

        let mut strata: Vec<usize> = (1..=n).collect();
        let size = n as f64;
        for axis in 0..self.dimensions {
            // SciPy starts each coordinate from the ordered stratum labels.
            for (index, stratum) in strata.iter_mut().enumerate() {
                *stratum = index + 1;
            }
            strata.shuffle(&mut self.rng);
            for (row, &stratum) in points.iter_mut().zip(&strata) {
                row[axis] = stratum_coordinate(stratum, row[axis], size);
            }
        }

        self.generated = end;
        Ok(points)
    }

    /// Consume the same random draws as generating and discarding one batch.
    ///
    /// A Latin hypercube is constructed separately for each batch, so
    /// `fast_forward(n)` is equivalent to discarding `random(n)` and does not
    /// skip within a preexisting stratification. The method performs `O(n*d)`
    /// work with `O(n)` temporary memory. `n = 0` is a no-op. Invalid batch
    /// size or counter overflow returns [`Error::InvalidInput`] without
    /// changing state.
    pub fn fast_forward(&mut self, n: usize) -> Result<()> {
        let end = self.check_batch(n)?;
        if n == 0 {
            return Ok(());
        }
        if self.scramble {
            for _ in 0..(n * self.dimensions) {
                let _: f64 = self.rng.random();
            }
        }
        let mut strata: Vec<usize> = (1..=n).collect();
        for _ in 0..self.dimensions {
            for (index, stratum) in strata.iter_mut().enumerate() {
                *stratum = index + 1;
            }
            strata.shuffle(&mut self.rng);
        }
        self.generated = end;
        Ok(())
    }

    /// Reset the random state and cumulative point count to their initial values.
    ///
    /// Repeating the same sequence of batch sizes after reset reproduces the
    /// same samples, whether scrambled or centered.
    pub fn reset(&mut self) {
        self.rng = StdRng::seed_from_u64(self.seed);
        self.generated = 0;
    }

    fn check_batch(&self, n: usize) -> Result<u128> {
        if n.checked_mul(self.dimensions)
            .is_none_or(|count| count > MAX_OUTPUT_VALUES)
        {
            return Err(Error::InvalidInput(
                "Latin hypercube batch exceeds 1048576 values",
            ));
        }
        self.generated
            .checked_add(n as u128)
            .ok_or(Error::InvalidInput("Latin hypercube point count overflow"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_stratified(points: &[Vec<f64>], dimensions: usize) {
        let n = points.len();
        assert!(n > 0);
        for row in points {
            assert_eq!(row.len(), dimensions);
            assert!(row.iter().all(|x| (0.0..1.0).contains(x)));
        }
        for axis in 0..dimensions {
            let mut strata = points
                .iter()
                .map(|row| (row[axis] * n as f64).floor() as usize)
                .collect::<Vec<_>>();
            strata.sort_unstable();
            assert_eq!(strata, (0..n).collect::<Vec<_>>());
        }
    }

    #[test]
    fn independent_batches_occupy_every_stratum() {
        let mut engine = LatinHypercube::new(7, 123).unwrap();
        for n in [1, 2, 3, 17, 256, 1024] {
            assert_stratified(&engine.random(n).unwrap(), 7);
        }
        assert_eq!(engine.num_generated(), 1303);
    }

    #[test]
    fn centered_batches_use_exact_midpoints() {
        let mut engine = LatinHypercube::centered(4, 9876).unwrap();
        for n in [1, 2, 13, 128] {
            let points = engine.random(n).unwrap();
            assert_stratified(&points, 4);
            for row in &points {
                for &x in row {
                    let scaled = x * n as f64;
                    assert!((scaled - scaled.floor() - 0.5).abs() < 1e-11);
                }
            }
        }
    }

    #[test]
    fn rounding_never_moves_coordinates_across_stratum_boundaries() {
        let almost_one = f64::from_bits(1.0_f64.to_bits() - 1);
        // Uncorrected (4 - almost_one) / 47 rounds below 3/47.
        let uncorrected = (4.0 - almost_one) / 47.0;
        assert!((uncorrected * 47.0).floor() < 3.0);
        let corrected = stratum_coordinate(4, almost_one, 47.0);
        assert_eq!((corrected * 47.0).floor(), 3.0);

        // The existing upper-bound guard remains effective for zero jitter.
        for n in [3, 47, 1000, MAX_OUTPUT_VALUES] {
            let size = n as f64;
            for stratum in [1, n / 2 + 1, n] {
                for offset in [0.0, 0.5, almost_one] {
                    let x = stratum_coordinate(stratum, offset, size);
                    assert_eq!((x * size).floor() as usize, stratum - 1);
                    assert!((0.0..1.0).contains(&x));
                }
            }
        }
    }

    #[test]
    fn seeds_and_reset_are_reproducible() {
        let mut a = LatinHypercube::new(3, 44).unwrap();
        let mut b = LatinHypercube::new(3, 44).unwrap();
        let mut c = LatinHypercube::new(3, 45).unwrap();
        let first = a.random(37).unwrap();
        assert_eq!(first, b.random(37).unwrap());
        assert_ne!(first, c.random(37).unwrap());
        let second = a.random(9).unwrap();
        assert_eq!(second, b.random(9).unwrap());
        assert_eq!(a.num_generated(), 46);
        a.reset();
        assert_eq!(a.num_generated(), 0);
        assert_eq!(first, a.random(37).unwrap());
        assert_eq!(second, a.random(9).unwrap());
    }

    #[test]
    fn skipping_matches_discarded_design_and_zero_is_noop() {
        for scramble in [true, false] {
            let mut drawn = LatinHypercube::with_scramble(5, 715, scramble).unwrap();
            let mut skipped = drawn.clone();
            skipped.fast_forward(0).unwrap();
            assert_eq!(skipped.num_generated(), 0);
            assert_eq!(drawn.random(23).unwrap().len(), 23);
            skipped.fast_forward(23).unwrap();
            assert_eq!(drawn.num_generated(), skipped.num_generated());
            assert_eq!(drawn.random(19).unwrap(), skipped.random(19).unwrap());
            skipped.reset();
            drawn.reset();
            assert_eq!(drawn.random(19).unwrap(), skipped.random(19).unwrap());
        }
    }

    #[test]
    fn invalid_requests_leave_rng_and_count_unchanged() {
        assert!(LatinHypercube::new(0, 0).is_err());
        assert!(LatinHypercube::new(MAX_OUTPUT_VALUES + 1, 0).is_err());
        let mut engine = LatinHypercube::new(4, 1234).unwrap();
        assert!(engine.random(MAX_OUTPUT_VALUES).is_err());
        assert!(engine.fast_forward(usize::MAX).is_err());
        assert_eq!(engine.num_generated(), 0);
        assert!(engine.random(0).unwrap().is_empty());
        assert_eq!(
            engine.random(11).unwrap(),
            LatinHypercube::new(4, 1234).unwrap().random(11).unwrap()
        );
    }
}
