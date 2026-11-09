//! Shared functional interface for stateful QMC generators.

use crate::{Error, Result};

// Exact floor(x * width) for a finite binary64 x in [0, 1) and
// 1 <= width <= 2^64. The product has at most 117 significant bits.
#[inline]
fn integer_offset(x: f64, width: u128) -> u128 {
    let bits = x.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as u32;
    let fraction = bits & ((1_u64 << 52) - 1);
    let (significand, shift) = if exponent == 0 {
        (fraction, 1074)
    } else {
        (fraction | (1_u64 << 52), 1075 - exponent)
    };
    (u128::from(significand) * width)
        .checked_shr(shift)
        .unwrap_or(0)
}

/// Stateful generator of points in the unit hypercube.
///
/// For each output coordinate, an engine computes a point `x_j` in
/// `[0, 1)`. Implementations determine whether the points form a
/// low-discrepancy sequence (Sobol/Halton), a stratified design (Latin
/// hypercube), or a distance-constrained sample (Poisson disk).
///
/// ## Parameters and conventions
///
/// `random(n)` draws `n` rows and advances the generator state.
/// `fast_forward(n)` discards the next `n` rows. `reset()` returns to
/// the initial state, including any seeded scramble.
///
/// ## Domain and special values
///
/// The number of dimensions is positive. Implementations specify their
/// own dimension, period, and computational limits. Zero-length requests
/// preserve the state unless documented otherwise by the implementation.
///
/// ## Returns
///
/// Samples are row-major `Vec<Vec<f64>>`, each row containing exactly
/// `dimensions()` finite, dimensionless coordinates in `[0,1)`. Poisson-disk
/// designs can return fewer than `n` rows when their search is exhausted.
///
/// ## Errors
///
/// Invalid bounds, dimensions, counts, exhaustion, and numerical
/// failures produce errors; implementations should validate before
/// modifying generator state where practical.
///
/// ## Numerical behavior
///
/// Mapping points to integers uses the exact binary64 value of each coordinate;
/// the finite precision of these coordinates limits attainable integers in
/// very large ranges. This trait imposes
/// no guarantee on integration or discrepancy; balanced Sobol designs
/// should be drawn with `Sobol::random_base2`.
///
/// # Examples
///
/// ```
/// use sci_stats::qmc::{QmcEngine, Sobol};
/// let mut sampler = Sobol::new(2, 30)?;
/// assert_eq!(QmcEngine::integers(&mut sampler, 2, &[0,10], &[4,14], false)?,
///            vec![vec![0,10], vec![2,12]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `QMCEngine`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.QMCEngine.html).
pub trait QmcEngine {
    /// Number of coordinates per generated point.
    fn dimensions(&self) -> usize;

    /// Total number of rows consumed or skipped by the engine.
    fn num_generated(&self) -> u128;

    /// Draw `n` consecutive points from this generator.
    fn random(&mut self, n: usize) -> Result<Vec<Vec<f64>>>;

    /// Skip `n` points while preserving the sequence's position.
    fn fast_forward(&mut self, n: usize) -> Result<()>;

    /// Return to the initial state, preserving the configured seed.
    fn reset(&mut self);

    /// Map `n` samples to per-coordinate integer intervals.
    ///
    /// ## Parameters and conventions
    ///
    /// `lower` and `upper` each have length `dimensions()`. With
    /// `endpoint=false`, intervals are `[lower[j], upper[j])`; otherwise
    /// they are inclusive. The mapping is
    /// `lower[j] + floor(x[j] * (upper[j]-lower[j]+endpoint))`.
    ///
    /// ## Returns
    ///
    /// Row-major integer samples of shape `(n, dimensions())`.
    ///
    /// ## Errors
    ///
    /// Rejects mismatched bounds, invalid widths, and excessive output size.
    /// Propagates underlying generator errors. Invalid bounds are checked
    /// without consuming samples.
    ///
    /// ## Numerical behavior
    ///
    /// Inverse-CDF floors are exact for each binary64 sample and integer width,
    /// including the inclusive full `i64` interval. For power-of-two widths,
    /// binary exponent scaling gives the same exact result more cheaply.
    /// At ranges wider than 2^53, some integers cannot be reached from the
    /// finite-precision `f64` coordinates.
    fn integers(
        &mut self,
        n: usize,
        lower: &[i64],
        upper: &[i64],
        endpoint: bool,
    ) -> Result<Vec<Vec<i64>>> {
        let d = self.dimensions();
        if lower.len() != d || upper.len() != d {
            return Err(Error::DimensionMismatch);
        }
        if n.checked_mul(d).is_none_or(|v| v > 1_048_576) {
            return Err(Error::InvalidInput(
                "QMC integer batch exceeds 1048576 values",
            ));
        }
        let bounds: Vec<(i128, u128, Option<f64>)> = lower
            .iter()
            .zip(upper)
            .map(|(lo, hi)| {
                let width = *hi as i128 - *lo as i128 + i128::from(endpoint);
                if width <= 0 {
                    Err(Error::InvalidInput("invalid QMC integer bounds"))
                } else {
                    let width = width as u128;
                    let power_two_scale = width.is_power_of_two().then_some(width as f64);
                    Ok((*lo as i128, width, power_two_scale))
                }
            })
            .collect::<Result<_>>()?;
        self.random(n)?
            .into_iter()
            .map(|row| {
                if row.len() != d {
                    return Err(Error::DimensionMismatch);
                }
                row.iter()
                    .enumerate()
                    .map(|(j, &x)| {
                        if !x.is_finite() || !(0. ..1.).contains(&x) {
                            return Err(Error::NonFiniteEvaluation);
                        }
                        let (lo, width, power_two_scale) = bounds[j];
                        let offset = if let Some(scale) = power_two_scale {
                            (x * scale) as u128
                        } else {
                            integer_offset(x, width)
                        };
                        Ok((lo + offset as i128) as i64)
                    })
                    .collect()
            })
            .collect()
    }
}

impl QmcEngine for super::Sobol {
    fn dimensions(&self) -> usize {
        self.dimensions()
    }

    fn num_generated(&self) -> u128 {
        self.num_generated()
    }

    fn random(&mut self, n: usize) -> Result<Vec<Vec<f64>>> {
        self.random(n)
    }

    fn fast_forward(&mut self, n: usize) -> Result<()> {
        self.fast_forward(n)
    }

    fn reset(&mut self) {
        self.reset();
    }
}

impl QmcEngine for super::Halton {
    fn dimensions(&self) -> usize {
        self.dimensions()
    }

    fn num_generated(&self) -> u128 {
        self.num_generated()
    }

    fn random(&mut self, n: usize) -> Result<Vec<Vec<f64>>> {
        self.random(n)
    }

    fn fast_forward(&mut self, n: usize) -> Result<()> {
        self.fast_forward(n)
    }

    fn reset(&mut self) {
        self.reset();
    }
}

impl QmcEngine for super::LatinHypercube {
    fn dimensions(&self) -> usize {
        self.dimensions()
    }

    fn num_generated(&self) -> u128 {
        self.num_generated()
    }

    fn random(&mut self, n: usize) -> Result<Vec<Vec<f64>>> {
        self.random(n)
    }

    fn fast_forward(&mut self, n: usize) -> Result<()> {
        self.fast_forward(n)
    }

    fn reset(&mut self) {
        self.reset();
    }
}

impl QmcEngine for super::PoissonDisk {
    fn dimensions(&self) -> usize {
        self.dimensions()
    }

    fn num_generated(&self) -> u128 {
        self.num_generated()
    }

    fn random(&mut self, n: usize) -> Result<Vec<Vec<f64>>> {
        self.random(n)
    }

    fn fast_forward(&mut self, n: usize) -> Result<()> {
        self.fast_forward(n)
    }

    fn reset(&mut self) {
        self.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::{QmcEngine, integer_offset};
    use crate::qmc::{Halton, Sobol};

    #[test]
    fn integer_mapping_uses_exact_binary64_boundaries() {
        // 1/3 rounds below the mathematical boundary, but (1/3)*3
        // rounds to 1.0 in ordinary floating-point arithmetic.
        let third = 1.0 / 3.0;
        assert_eq!(third * 3.0, 1.0);
        assert_eq!(integer_offset(third, 3), 0);
        assert_eq!(integer_offset(f64::from_bits(third.to_bits() + 1), 3), 1);

        let mut halton = Halton::new(2).unwrap();
        assert_eq!(
            QmcEngine::integers(&mut halton, 2, &[0, 0], &[3, 3], false).unwrap(),
            vec![vec![0, 0], vec![1, 0]]
        );
    }

    #[test]
    fn integer_mapping_handles_full_i64_span_and_odd_wide_ranges() {
        let mut sobol = Sobol::new(1, 3).unwrap();
        assert_eq!(
            QmcEngine::integers(&mut sobol, 4, &[i64::MIN], &[i64::MAX], true).unwrap(),
            vec![
                vec![i64::MIN],
                vec![0],
                vec![1_i64 << 62],
                vec![-(1_i64 << 62)]
            ]
        );

        sobol.reset();
        let quarter = 1_i64 << 61;
        assert_eq!(
            QmcEngine::integers(&mut sobol, 4, &[0], &[i64::MAX], false).unwrap(),
            vec![
                vec![0],
                vec![2 * quarter - 1],
                vec![3 * quarter - 1],
                vec![quarter - 1]
            ]
        );

        assert_eq!(integer_offset(f64::from_bits(1), 1_u128 << 64), 0);
        assert_eq!(
            integer_offset(f64::from_bits(1.0_f64.to_bits() - 1), 1_u128 << 64),
            (1_u128 << 64) - 2048
        );
    }

    #[test]
    fn invalid_integer_bounds_leave_qmc_state_unchanged() {
        let mut sobol = Sobol::new(1, 3).unwrap();
        assert!(QmcEngine::integers(&mut sobol, 1, &[5], &[5], false).is_err());
        assert_eq!(sobol.num_generated(), 0);
        assert_eq!(
            QmcEngine::integers(&mut sobol, 1, &[i64::MAX], &[i64::MAX], true).unwrap(),
            vec![vec![i64::MAX]]
        );
    }
}
