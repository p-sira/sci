//! Bounded Bridson Poisson-disk sampling in the unit hypercube.

use std::collections::HashMap;
use std::f64::consts::TAU;

use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{Error, Result};

const MAX_DIMENSIONS: usize = 4;
const MAX_POINTS: usize = 16_384;
const MAX_OUTPUT_VALUES: usize = 1_048_576;
const MAX_CANDIDATES: usize = 256;
const MAX_ATTEMPTS: usize = 131_072;
const MAX_NEIGHBOR_PROBES: usize = 16_777_216;
const MIN_RADIUS: f64 = 1e-12;

/// Distribution of trial points around an active Poisson-disk sample.
///
/// ## Parameters and conventions
///
/// Both methods generate uniform directions on the unit sphere. [`Self::Volume`]
/// draws the distance uniformly in volume from the annulus between radii `r`
/// and `2r`. [`Self::Surface`] uses the fixed distance `1.001r`.
///
/// ## Domain and numerical behavior
///
/// Distances are dimensionless Euclidean distances in the unit hypercube.
/// Neither method changes the required minimum separation `r`. `Volume` is
/// the default. The trial-point distributions are independent of SciPy's RNG.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PoissonDiskHypersphere {
    /// Uniform volume in the shell between one and two separation radii.
    #[default]
    Volume,
    /// Uniform direction at approximately one separation radius (`1.001r`).
    Surface,
}

/// Stateful Bridson Poisson-disk sampler in the unit hypercube `[0, 1)^d`.
///
/// Starting from a uniformly random point, the sampler repeatedly chooses an
/// active point and proposes up to `k` random candidates in a surrounding
/// hyperspherical annulus. A proposal `x` is accepted only when
/// `||x - xᵢ||₂ >= r` for every earlier sample `xᵢ` and `x` lies in the cube.
/// The active pool is exhausted after unsuccessful candidates. A sparse grid
/// of cells of width `r / √d` accelerates the separation check.
///
/// ## Parameters and conventions
///
/// [`Self::new`] takes the number of coordinates `d`, minimum separation
/// `radius = r`, and a `u64` seed. The default is 30 trial candidates per
/// active point and [`PoissonDiskHypersphere::Volume`] sampling. Use
/// [`Self::with_candidates`] and [`Self::with_hypersphere`] to configure a
/// fresh engine. Returned rows are generated in order. Coordinates and
/// distances are dimensionless.
///
/// ## Domain and special values
///
/// Requires `1 <= d <= 4` and finite `1e-12 <= r`. A radius at least the
/// hypercube diameter `√d` produces exactly one sample. `random(0)` and
/// `fast_forward(0)` preserve the current state. NaN and infinite radii are
/// rejected.
///
/// ## Returns
///
/// `random(n)` returns at most `n` rows of `d` finite coordinates, each in
/// `[0, 1)`; geometric exhaustion can yield fewer rows than requested.
/// `fill_space()` returns all additional points until the search is exhausted
/// or reaches its work limits. Both methods continue the same sequence.
///
/// ## Errors
///
/// Invalid dimensions, radii, candidate counts, and requests with
/// `n*d > 1_048_576` return [`Error::InvalidInput`] without changing state.
///
/// ## Numerical behavior
///
/// Fixed-seed output is reproducible in Sci and consistent across batch
/// boundaries; SciPy 1.18.1 uses a different RNG and candidate schedule.
/// Accepted Euclidean distances are checked against `r²` in `f64`.
/// All searches have cumulative caps of 16,384 points, 131,072 candidate
/// proposals, and 16,777,216 examined grid cells. Reaching any cap ends
/// generation even if the domain has space left; [`Self::is_exhausted`]
/// reports this terminal state. This limits the high-dimensional and
/// small-radius cost of Bridson sampling.
///
/// ## Examples
///
/// ```
/// use sci_stats::qmc::PoissonDisk;
/// let mut sampler = PoissonDisk::new(2, 0.2, 42)?;
/// let points = sampler.random(20)?;
/// assert!(!points.is_empty());
/// assert!(points.iter().all(|p| p.len() == 2));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [Bridson, *Fast Poisson Disk Sampling in Arbitrary Dimensions* (2007)](https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf);
/// [SciPy `qmc.PoissonDisk`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.PoissonDisk.html).
#[derive(Clone)]
pub struct PoissonDisk {
    dimensions: usize,
    radius: f64,
    radius_squared: f64,
    cell_width: f64,
    neighbor_offsets: Vec<[i64; MAX_DIMENSIONS]>,
    candidates: usize,
    hypersphere: PoissonDiskHypersphere,
    seed: u64,
    rng: StdRng,
    points: Vec<Vec<f64>>,
    active: Vec<usize>,
    current_center: Option<(usize, usize)>,
    grid: HashMap<[i64; MAX_DIMENSIONS], usize>,
    attempts: usize,
    probes: usize,
    generated: u128,
}

impl PoissonDisk {
    /// Construct a reproducible `d`-dimensional Poisson-disk sampler.
    ///
    /// ## Parameters and conventions
    ///
    /// `radius` is the minimum Euclidean distance between samples; `seed`
    /// initializes a private `StdRng`. The default is 30 trial candidates per
    /// active sample with volume sampling in `[0, 1]^d`.
    ///
    /// ## Returns
    ///
    /// A sampler with no generated points.
    ///
    /// ## Errors
    ///
    /// Returns [`Error::InvalidInput`] for `d` outside `1..=4` or nonfinite,
    /// nonpositive, or sub-`1e-12` radii.
    ///
    /// ## Numerical behavior
    ///
    /// Radius is bounded below to keep grid indexing and squared distances
    /// reliable in `f64`. Initialization precomputes a fixed-size neighbor
    /// stencil (at most 625 grid offsets).
    pub fn new(d: usize, radius: f64, seed: u64) -> Result<Self> {
        if d == 0 || d > MAX_DIMENSIONS {
            return Err(Error::InvalidInput("PoissonDisk requires 1..=4 dimensions"));
        }
        if !radius.is_finite() || radius < MIN_RADIUS {
            return Err(Error::InvalidInput(
                "PoissonDisk requires a finite radius >= 1e-12",
            ));
        }
        let neighbor_offset = (d as f64).sqrt().ceil() as i64;
        let side = (2 * neighbor_offset + 1) as usize;
        let mut neighbor_offsets = Vec::with_capacity(side.pow(d as u32));
        for mut index in 0..side.pow(d as u32) {
            let mut offset = [0; MAX_DIMENSIONS];
            for coordinate in offset.iter_mut().take(d) {
                *coordinate = (index % side) as i64 - neighbor_offset;
                index /= side;
            }
            neighbor_offsets.push(offset);
        }
        Ok(Self {
            dimensions: d,
            radius,
            radius_squared: radius * radius,
            cell_width: radius / (d as f64).sqrt(),
            neighbor_offsets,
            candidates: 30,
            hypersphere: PoissonDiskHypersphere::Volume,
            seed,
            rng: StdRng::seed_from_u64(seed),
            points: Vec::new(),
            active: Vec::new(),
            current_center: None,
            grid: HashMap::new(),
            attempts: 0,
            probes: 0,
            generated: 0,
        })
    }

    /// Set the number of proposals tested for each active sample.
    ///
    /// ## Parameters and conventions
    ///
    /// `candidates` is an integer in `1..=256`; the default is 30. This
    /// builder resets a previously sampled engine before returning it.
    ///
    /// ## Returns
    ///
    /// The reconfigured engine, starting from its original seed.
    ///
    /// ## Errors
    ///
    /// Returns [`Error::InvalidInput`] for a zero or excessive candidate count.
    ///
    /// ## Numerical behavior
    ///
    /// Larger counts can improve coverage at increased bounded search cost.
    pub fn with_candidates(mut self, candidates: usize) -> Result<Self> {
        if candidates == 0 || candidates > MAX_CANDIDATES {
            return Err(Error::InvalidInput(
                "PoissonDisk requires 1..=256 candidates",
            ));
        }
        self.candidates = candidates;
        self.reset();
        Ok(self)
    }

    /// Choose the hypersphere distribution used for trial points.
    ///
    /// ## Parameters and conventions
    ///
    /// [`PoissonDiskHypersphere::Volume`] draws from the annulus `r..2r`;
    /// [`PoissonDiskHypersphere::Surface`] draws at distance `1.001r`.
    /// The default is `Volume`. Configuration resets the engine.
    ///
    /// ## Returns
    ///
    /// The reconfigured engine, at its original seed and index zero.
    ///
    /// ## Numerical behavior
    ///
    /// Both strategies apply the same minimum-distance rejection test.
    pub fn with_hypersphere(mut self, hypersphere: PoissonDiskHypersphere) -> Self {
        self.hypersphere = hypersphere;
        self.reset();
        self
    }

    /// Number of coordinates per sample.
    ///
    /// Returns the dimension given to [`Self::new`], always in `1..=4`.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Number of accepted points so far, including discarded skipped points.
    ///
    /// Counts accepted points only; rejected candidates do not advance it.
    /// The count is zero after [`Self::reset`].
    pub fn num_generated(&self) -> u128 {
        self.generated
    }

    /// Whether the active pool or the cumulative bounded search is exhausted.
    ///
    /// ## Returns
    ///
    /// `false` before the initial draw; `true` after all active samples have
    /// been tried or after a point, proposal, or grid-work cap is reached.
    ///
    /// ## Numerical behavior
    ///
    /// Reaching a work cap need not mean the unit hypercube is fully packed.
    pub fn is_exhausted(&self) -> bool {
        self.points.len() >= MAX_POINTS
            || self.attempts >= MAX_ATTEMPTS
            || self.probes >= MAX_NEIGHBOR_PROBES
            || (!self.points.is_empty() && self.active.is_empty() && self.current_center.is_none())
    }

    /// Draw up to `n` additional separated points in `[0, 1)^d`.
    ///
    /// ## Parameters and conventions
    ///
    /// Rows are emitted in generation order. Repeated calls continue the
    /// same seeded search; splitting a batch does not change the output.
    /// `n = 0` preserves state.
    ///
    /// ## Returns
    ///
    /// A row-major vector of up to `n` rows, each containing `d` finite
    /// coordinates. On geometric or work exhaustion, fewer than `n` rows
    /// may be returned, including an empty vector.
    ///
    /// ## Errors
    ///
    /// Returns [`Error::InvalidInput`] for `n*d > 1_048_576`; no state is
    /// consumed when the request is rejected.
    ///
    /// ## Numerical behavior
    ///
    /// Acceptance uses `f64` Euclidean distances. See [`Self`] for global
    /// resource limits and the distance threshold.
    pub fn random(&mut self, n: usize) -> Result<Vec<Vec<f64>>> {
        self.advance(n, true)
    }

    /// Continue sampling until exhausted, returning all additional points.
    ///
    /// ## Parameters and conventions
    ///
    /// Uses the current RNG state and existing active pool. It can be called
    /// after [`Self::random`] or multiple times.
    ///
    /// ## Returns
    ///
    /// Additional samples, up to the remaining 16,384-point limit. Empty
    /// when already exhausted.
    ///
    /// ## Errors
    ///
    /// Uses the same error type as [`Self::random`]; internal output size
    /// remains below the 1,048,576-value cap.
    ///
    /// ## Numerical behavior
    ///
    /// Saturation is heuristic; no more candidates survive the active search,
    /// or one of the cumulative work limits was reached.
    pub fn fill_space(&mut self) -> Result<Vec<Vec<f64>>> {
        self.advance(MAX_POINTS, true)
    }

    /// Discard up to `n` new points while advancing the same search.
    ///
    /// ## Parameters and conventions
    ///
    /// A skipped sample consumes its RNG and active-pool state exactly as in
    /// [`Self::random`]. If fewer than `n` points remain, the engine becomes
    /// exhausted and `num_generated` counts only those actually skipped.
    ///
    /// ## Returns
    ///
    /// `Ok(())` when the request was processed, including early exhaustion.
    ///
    /// ## Errors
    ///
    /// Returns [`Error::InvalidInput`] for `n*d > 1_048_576` without
    /// changing the engine.
    ///
    /// ## Numerical behavior
    ///
    /// Search still costs `O(n)` accepted samples plus rejected proposals;
    /// there is no direct random access in a Bridson sequence.
    pub fn fast_forward(&mut self, n: usize) -> Result<()> {
        self.advance(n, false).map(|_| ())
    }

    /// Restore the original seeded state and an empty sample pool.
    ///
    /// ## Returns
    ///
    /// The same engine will reproduce its prior sequence for unchanged
    /// parameters; no allocation or numerical evaluation is required.
    pub fn reset(&mut self) {
        self.rng = StdRng::seed_from_u64(self.seed);
        self.points.clear();
        self.active.clear();
        self.current_center = None;
        self.grid.clear();
        self.attempts = 0;
        self.probes = 0;
        self.generated = 0;
    }

    fn cell(&self, point: &[f64]) -> [i64; MAX_DIMENSIONS] {
        let mut cell = [0; MAX_DIMENSIONS];
        for (coordinate, value) in point.iter().enumerate() {
            cell[coordinate] = (value / self.cell_width).floor() as i64;
        }
        cell
    }

    fn add_point(&mut self, point: Vec<f64>) {
        let index = self.points.len();
        self.grid.insert(self.cell(&point), index);
        self.points.push(point);
        self.active.push(index);
        self.generated += 1;
    }

    fn sample_candidate(&mut self, center: usize) -> [f64; MAX_DIMENSIONS] {
        let mut direction = [0.0; MAX_DIMENSIONS];
        for axis in (0..self.dimensions).step_by(2) {
            let u = self.rng.random::<f64>().max(f64::MIN_POSITIVE);
            let angle = TAU * self.rng.random::<f64>();
            let amplitude = (-2.0 * u.ln()).sqrt();
            let (sine, cosine) = angle.sin_cos();
            direction[axis] = amplitude * cosine;
            if axis + 1 < self.dimensions {
                direction[axis + 1] = amplitude * sine;
            }
        }
        let norm = direction
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let radial_factor = match self.hypersphere {
            PoissonDiskHypersphere::Volume => {
                let r = self.rng.random::<f64>();
                (1.0 + ((1_u32 << self.dimensions) - 1) as f64 * r)
                    .powf(1.0 / self.dimensions as f64)
            }
            PoissonDiskHypersphere::Surface => 1.001,
        };
        let scale = self.radius * radial_factor / norm;
        direction[..self.dimensions]
            .iter_mut()
            .enumerate()
            .for_each(|(axis, value)| *value = self.points[center][axis] + *value * scale);
        direction
    }

    fn acceptable(&mut self, candidate: &[f64]) -> bool {
        if candidate
            .iter()
            .any(|&v| !v.is_finite() || !(0.0..1.0).contains(&v))
        {
            return false;
        }
        let origin = self.cell(candidate);
        for offset in &self.neighbor_offsets {
            if self.probes >= MAX_NEIGHBOR_PROBES {
                return false;
            }
            self.probes += 1;
            let mut key = origin;
            for (coordinate, difference) in key.iter_mut().zip(offset) {
                *coordinate += *difference;
            }
            if let Some(&point_index) = self.grid.get(&key) {
                let distance_squared = candidate
                    .iter()
                    .zip(&self.points[point_index])
                    .map(|(x, y)| {
                        let delta = x - y;
                        delta * delta
                    })
                    .sum::<f64>();
                if distance_squared < self.radius_squared {
                    return false;
                }
            }
        }
        true
    }

    fn advance(&mut self, n: usize, collect: bool) -> Result<Vec<Vec<f64>>> {
        if n.checked_mul(self.dimensions)
            .is_none_or(|values| values > MAX_OUTPUT_VALUES)
        {
            return Err(Error::InvalidInput(
                "PoissonDisk batch exceeds 1048576 values",
            ));
        }
        let mut output = if collect {
            Vec::with_capacity(n.min(MAX_POINTS))
        } else {
            Vec::new()
        };
        let mut added = 0;
        while added < n && !self.is_exhausted() {
            if self.points.is_empty() {
                let initial = (0..self.dimensions)
                    .map(|_| self.rng.random::<f64>())
                    .collect::<Vec<_>>();
                if collect {
                    output.push(initial.clone());
                }
                self.add_point(initial);
                added += 1;
                if self.radius >= (self.dimensions as f64).sqrt() {
                    self.active.clear();
                }
                continue;
            }

            let (center, remaining) = if let Some(current) = self.current_center {
                current
            } else {
                let center_position = self.rng.random_range(0..self.active.len());
                let center = self.active.swap_remove(center_position);
                (center, self.candidates)
            };
            self.attempts += 1;
            let candidate = self.sample_candidate(center);
            if self.acceptable(&candidate[..self.dimensions]) {
                // A productive center remains active. The proposal budget
                // counts successive failures, not all previous trials.
                self.active.push(center);
                self.current_center = None;
                if collect {
                    output.push(candidate[..self.dimensions].to_vec());
                }
                self.add_point(candidate[..self.dimensions].to_vec());
                added += 1;
            } else {
                self.current_center = (remaining > 1).then_some((center, remaining - 1));
            }
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_NEIGHBOR_PROBES, PoissonDisk, PoissonDiskHypersphere};

    fn assert_separated(points: &[Vec<f64>], radius: f64) {
        assert!(
            points
                .iter()
                .flatten()
                .all(|x| x.is_finite() && (0.0..1.0).contains(x))
        );
        for (i, first) in points.iter().enumerate() {
            for second in &points[..i] {
                let distance = first
                    .iter()
                    .zip(second)
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f64>()
                    .sqrt();
                assert!(distance + 1e-13 >= radius, "{distance} < {radius}");
            }
        }
    }

    #[test]
    fn sample_distance_bounds_and_dimension() {
        for d in 1..=4 {
            let radius = if d > 2 { 0.42 } else { 0.23 };
            let mut disk = PoissonDisk::new(d, radius, 11).unwrap();
            let mut points = disk.random(60).unwrap();
            points.extend(disk.fill_space().unwrap());
            assert!(!points.is_empty());
            assert!(points.iter().all(|point| point.len() == d));
            assert_eq!(disk.num_generated(), points.len() as u128);
            assert!(disk.is_exhausted());
            assert_separated(&points, radius);
        }
    }

    #[test]
    fn batches_reset_and_skip_follow_same_sequence() {
        let mut one = PoissonDisk::new(2, 0.12, 5).unwrap();
        let full = one.random(15).unwrap();
        assert_eq!(full.len(), 15);

        let mut split = PoissonDisk::new(2, 0.12, 5).unwrap();
        let first = split.random(4).unwrap();
        let remaining = split.random(11).unwrap();
        assert_eq!(full, [first, remaining].concat());
        let last = split.random(6).unwrap();

        let mut skipped = PoissonDisk::new(2, 0.12, 5).unwrap();
        skipped.fast_forward(15).unwrap();
        assert_eq!(skipped.num_generated(), 15);
        assert_eq!(skipped.random(6).unwrap(), last);

        split.reset();
        assert_eq!(split.num_generated(), 0);
        assert_eq!(split.random(15).unwrap(), full);

        let mut different_seed = PoissonDisk::new(2, 0.12, 6).unwrap();
        assert_ne!(different_seed.random(15).unwrap(), full);
    }

    #[test]
    fn hypersphere_and_candidate_configuration() {
        let mut volume = PoissonDisk::new(2, 0.18, 321).unwrap();
        let mut surface = PoissonDisk::new(2, 0.18, 321)
            .unwrap()
            .with_candidates(12)
            .unwrap()
            .with_hypersphere(PoissonDiskHypersphere::Surface);
        let volume_points = volume.random(12).unwrap();
        let surface_points = surface.random(12).unwrap();
        assert_ne!(volume_points, surface_points);
        assert_separated(&volume_points, 0.18);
        assert_separated(&surface_points, 0.18);
    }

    #[test]
    fn exhaustion_is_safe_and_idempotent() {
        let mut disk = PoissonDisk::new(2, 10.0, 4).unwrap();
        let single = disk.random(100).unwrap();
        assert_eq!(single.len(), 1);
        assert!(disk.is_exhausted());
        assert!(disk.fill_space().unwrap().is_empty());
        disk.fast_forward(50).unwrap();
        assert_eq!(disk.num_generated(), 1);
    }

    #[test]
    fn invalid_inputs_are_rejected_before_consuming_state() {
        assert!(PoissonDisk::new(0, 0.1, 1).is_err());
        assert!(PoissonDisk::new(5, 0.1, 1).is_err());
        for radius in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e-15] {
            assert!(PoissonDisk::new(2, radius, 1).is_err());
        }
        assert!(
            PoissonDisk::new(2, 0.1, 1)
                .unwrap()
                .with_candidates(0)
                .is_err()
        );
        assert!(
            PoissonDisk::new(2, 0.1, 1)
                .unwrap()
                .with_candidates(257)
                .is_err()
        );
        let mut disk = PoissonDisk::new(2, 0.1, 1).unwrap();
        assert!(disk.random(524_289).is_err());
        assert!(disk.fast_forward(524_289).is_err());
        assert_eq!(disk.num_generated(), 0);
        assert!(disk.random(0).unwrap().is_empty());
        assert_eq!(disk.num_generated(), 0);
    }

    #[test]
    fn probe_cap_rejects_incompletely_checked_candidates() {
        let mut disk = PoissonDisk::new(2, 0.2, 11).unwrap();
        disk.add_point(vec![0.5, 0.5]);

        assert!(!disk.acceptable(&[0.51, 0.5]));
        assert!(disk.probes < disk.neighbor_offsets.len());

        disk.probes = MAX_NEIGHBOR_PROBES - 1;
        assert!(!disk.acceptable(&[0.51, 0.5]));
        assert_eq!(disk.probes, MAX_NEIGHBOR_PROBES);
        assert!(disk.is_exhausted());
    }

    #[test]
    fn successful_center_remains_active_and_resets_failed_trial_budget() {
        let mut disk = PoissonDisk::new(2, 0.1, 21)
            .unwrap()
            .with_candidates(1)
            .unwrap();
        disk.add_point(vec![0.5, 0.5]);

        let generated = disk.random(1).unwrap();
        assert_eq!(generated.len(), 1);
        assert_eq!(disk.active.len(), 2);
        assert!(disk.active.contains(&0), "productive center was retired");
        assert!(disk.active.contains(&1));
        assert_eq!(disk.current_center, None);
    }

    #[test]
    fn center_retires_after_k_consecutive_failed_proposals() {
        let mut disk = PoissonDisk::new(1, 0.6, 23)
            .unwrap()
            .with_candidates(3)
            .unwrap();
        disk.add_point(vec![0.5]);
        // In one dimension, every candidate is at least 0.6 from the center,
        // so every trial lies outside [0, 1).
        assert!(disk.random(1).unwrap().is_empty());
        assert_eq!(disk.attempts, 3);
        assert!(disk.active.is_empty());
        assert_eq!(disk.current_center, None);
        assert!(disk.is_exhausted());
    }
}
