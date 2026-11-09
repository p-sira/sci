//! Prime-base Halton points with optional reproducible digit scrambling.

use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{Error, Result};

const MAX_DIMENSIONS: usize = 512;
const MAX_OUTPUT_VALUES: usize = 1_048_576;
const BELOW_ONE: f64 = f64::from_bits(1.0_f64.to_bits() - 1);

fn prime_bases(d: usize) -> Vec<u32> {
    let mut primes = Vec::with_capacity(d);
    let mut candidate = 2_u32;
    while primes.len() < d {
        if primes
            .iter()
            .take_while(|&&prime| prime <= candidate / prime)
            .all(|prime| !candidate.is_multiple_of(*prime))
        {
            primes.push(candidate);
        }
        candidate += if candidate == 2 { 1 } else { 2 };
    }
    primes
}

#[derive(Clone, Copy, Debug)]
struct HaltonBase {
    prime: u32,
    inv_base: f64,
    reciprocal32: u32,
    reciprocal64: u64,
}

impl HaltonBase {
    fn new(prime: u32) -> Self {
        Self {
            prime,
            inv_base: 1.0 / f64::from(prime),
            reciprocal32: ((1_u64 << 32) / u64::from(prime)) as u32,
            reciprocal64: ((1_u128 << 64) / u128::from(prime)) as u64,
        }
    }
}

#[derive(Clone, Debug)]
struct DigitScramble {
    permutations: Vec<Vec<u16>>,
    // For an index with k base-b digits, all higher digits are zero. Their
    // shuffled values form a constant tail, so ordinary indices only require
    // O(log_b(index)) digit operations instead of all 128-bit positions.
    zero_tails: Vec<f64>,
}

fn random_permutations(base: u32, rng: &mut StdRng) -> DigitScramble {
    // A full u128 index may have up to 128 binary digits. Truncating the
    // permutations at f64's precision discards higher index digits even when
    // their contributions are distinguishable near zero.
    let mut upper = u128::MAX;
    let mut count = 0;
    while upper != 0 {
        upper /= u128::from(base);
        count += 1;
    }
    let mut positions = Vec::with_capacity(count);
    for _ in 0..count {
        let mut permutation: Vec<u16> = (0..base).map(|digit| digit as u16).collect();
        for i in (1..permutation.len()).rev() {
            permutation.swap(i, rng.random_range(0..=i));
        }
        positions.push(permutation);
    }
    let mut zero_tails = vec![0.0; count + 1];
    let inv_base = 1.0 / f64::from(base);
    let mut weights = Vec::with_capacity(count);
    let mut weight = inv_base;
    for _ in 0..count {
        weights.push(weight);
        weight *= inv_base;
    }
    for k in (0..count).rev() {
        zero_tails[k] = zero_tails[k + 1] + f64::from(positions[k][0]) * weights[k];
    }
    DigitScramble {
        permutations: positions,
        zero_tails,
    }
}

fn radical_inverse(mut index: u128, base: u32, scramble: Option<&DigitScramble>) -> f64 {
    let inv_base = 1.0 / f64::from(base);
    let mut weight = inv_base;
    let mut value = 0.0;
    let mut position = 0;
    if let Ok(mut small) = u64::try_from(index) {
        let base = u64::from(base);
        while small != 0 {
            let digit = (small % base) as usize;
            let mapped =
                scramble.map_or(digit as f64, |s| f64::from(s.permutations[position][digit]));
            value += mapped * weight;
            small /= base;
            weight *= inv_base;
            position += 1;
        }
    } else {
        let base = u128::from(base);
        while index != 0 {
            let digit = (index % base) as usize;
            let mapped =
                scramble.map_or(digit as f64, |s| f64::from(s.permutations[position][digit]));
            value += mapped * weight;
            index /= base;
            weight *= inv_base;
            position += 1;
        }
    }
    if let Some(scramble) = scramble {
        value += scramble.zero_tails[position];
    }
    value.min(BELOW_ONE)
}

// Compute each plain coordinate in the same digit-addition order as
// radical_inverse while avoiding unnecessary integer divisions.
// M=floor(2^w / b) gives q=floor(n*M / 2^w), which is either floor(n/b)
// or exactly one smaller for 0 <= n < 2^w. One remainder correction gives
// exact division without a variable integer-divide instruction.
#[inline]
fn div_rem32(n: u32, base: &HaltonBase) -> (u32, u32) {
    let mut quotient = ((u64::from(n) * u64::from(base.reciprocal32)) >> 32) as u32;
    let mut remainder = n - quotient * base.prime;
    if remainder >= base.prime {
        quotient += 1;
        remainder -= base.prime;
    }
    (quotient, remainder)
}

#[inline]
fn div_rem64(n: u64, base: &HaltonBase) -> (u64, u64) {
    let divisor = u64::from(base.prime);
    let mut quotient = ((u128::from(n) * u128::from(base.reciprocal64)) >> 64) as u64;
    let mut remainder = n - quotient * divisor;
    if remainder >= divisor {
        quotient += 1;
        remainder -= divisor;
    }
    (quotient, remainder)
}

#[inline]
fn radical_inverse_plain(mut index: u128, base: &HaltonBase) -> f64 {
    if base.prime == 2 && index <= u128::from(u32::MAX) {
        // All 32 binary fractional digits are exactly representable as f64.
        return f64::from((index as u32).reverse_bits()) * 2_f64.powi(-32);
    }
    let inv_base = base.inv_base;
    let mut weight = inv_base;
    let mut value = 0.0;
    if let Ok(mut small) = u32::try_from(index) {
        while small != 0 {
            let (quotient, digit) = div_rem32(small, base);
            value += f64::from(digit) * weight;
            small = quotient;
            weight *= inv_base;
        }
    } else if let Ok(mut small) = u64::try_from(index) {
        while small != 0 {
            let (quotient, digit) = div_rem64(small, base);
            value += digit as f64 * weight;
            small = quotient;
            weight *= inv_base;
        }
    } else {
        let base = u128::from(base.prime);
        while index != 0 {
            value += (index % base) as f64 * weight;
            index /= base;
            weight *= inv_base;
        }
    }
    value.min(BELOW_ONE)
}

/// Stateful Halton low-discrepancy sequence in the unit hypercube.
///
/// The `j`th coordinate of zero-based point `n` is the radical inverse in the
/// `(j+1)`th prime base `bⱼ`. Writing `n = Σₖ aₖ bⱼᵏ`, its value is
/// `φ_b(n) = Σₖ aₖ bⱼ⁻⁽ᵏ⁺¹⁾`. An optional randomized sequence replaces each
/// digit `aₖ` with an independently permuted digit `πⱼₖ(aₖ)` (Owen, 2017).
///
/// ## Parameters and conventions
///
/// [`Halton::new`] constructs an unscrambled sequence beginning at the origin.
/// [`Halton::scrambled`] constructs a deterministic digit-scrambled sequence
/// using a `u64` seed and Rust's `StdRng`. A dimension is a coordinate; each
/// row returned by [`Halton::random`] is a consecutive point. The seed is
/// retained across [`Halton::reset`].
///
/// ## Domain and special values
///
/// Dimensions must be `1..=512`. A zero sample count produces no points and
/// leaves the engine unchanged. Coordinates are finite values in `[0, 1)`.
///
/// ## Returns
///
/// Generates row-major `Vec<Vec<f64>>` arrays of shape `(n, d)`, with
/// dimensionless coordinates in the half-open unit hypercube.
///
/// ## Errors
///
/// Invalid dimensions, batches exceeding 1,048,576 scalar values, and a
/// generated index exceeding `u128` produce [`Error::InvalidInput`]. Errors
/// leave generator state unchanged.
///
/// ## Numerical behavior
///
/// Generation uses `O(nd log_b(index + n))` work, where `b` is each dimension's
/// prime base. Skipping is `O(1)` independent of the number of skipped points.
/// All digits of a `u128` index are scrambled, with constant zero-digit tails
/// cached to keep ordinary generation proportional to the index digit count.
/// Plain indices fitting in 32 or 64 bits use precomputed exact integer
/// reciprocal multiplication; larger indices retain 128-bit division.
/// The first coordinate uses exact binary bit reversal for 32-bit indices.
/// Floating-point rounding may still merge distinct points at very large
/// indices; values rounded to 1 are clamped to the next `f64` below 1.
/// The independent-digit scramble uses `StdRng` permutations and is reproducible
/// within Sci, but a seed does not reproduce NumPy/SciPy's random permutations.
/// Memory for permutations scales with dimensions and the prime bases.
///
/// # Examples
///
/// ```
/// use sci_stats::qmc::Halton;
/// let mut sampler = Halton::new(2)?;
/// assert_eq!(sampler.random(3)?, vec![
///     vec![0.0, 0.0],
///     vec![0.5, 1.0 / 3.0],
///     vec![0.25, 2.0 / 3.0],
/// ]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// Halton (1960), *On the efficiency of certain quasi-random sequences*,
/// [Owen (2017), A randomized Halton algorithm in R](https://arxiv.org/abs/1706.02808),
/// and [SciPy `qmc.Halton`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Halton.html).
#[derive(Clone, Debug)]
pub struct Halton {
    bases: Vec<HaltonBase>,
    permutations: Option<Vec<DigitScramble>>,
    generated: u128,
}

impl Halton {
    /// Construct the `d`-dimensional unscrambled Halton sequence.
    ///
    /// The bases are the first `d` primes, and the first point is zero in
    /// every dimension. Returns [`Error::InvalidInput`] unless `1 <= d <= 512`.
    pub fn new(d: usize) -> Result<Self> {
        if !(1..=MAX_DIMENSIONS).contains(&d) {
            return Err(Error::InvalidInput("Halton requires 1..=512 dimensions"));
        }
        Ok(Self {
            bases: prime_bases(d).into_iter().map(HaltonBase::new).collect(),
            permutations: None,
            generated: 0,
        })
    }

    /// Construct an Owen-style scrambled Halton sequence using a fixed seed.
    ///
    /// A fixed `(d, seed)` repeats the same sequence, including after reset.
    /// This uses Rust's `StdRng`, which differs from SciPy's NumPy RNG.
    /// Dimensions and errors have the same requirements as [`Halton::new`].
    pub fn scrambled(d: usize, seed: u64) -> Result<Self> {
        let mut engine = Self::new(d)?;
        let mut rng = StdRng::seed_from_u64(seed);
        engine.permutations = Some(
            engine
                .bases
                .iter()
                .map(|base| random_permutations(base.prime, &mut rng))
                .collect(),
        );
        Ok(engine)
    }

    /// Number of coordinates in each point.
    pub fn dimensions(&self) -> usize {
        self.bases.len()
    }

    /// Total points consumed, including skipped points.
    pub fn num_generated(&self) -> u128 {
        self.generated
    }

    /// Draw the next `n` points as a row-major `(n, d)` array in `[0, 1)`.
    ///
    /// `n=0` returns an empty array. Each successful call advances the same
    /// sequence. A batch with more than 1,048,576 scalar values, or one that
    /// overflows the sequence index, returns an error without advancing.
    /// The numerical behavior and precision limitations are described in
    /// [`Halton`].
    pub fn random(&mut self, n: usize) -> Result<Vec<Vec<f64>>> {
        if n.checked_mul(self.dimensions())
            .is_none_or(|count| count > MAX_OUTPUT_VALUES)
        {
            return Err(Error::InvalidInput("Halton batch exceeds 1048576 values"));
        }
        let end = self
            .generated
            .checked_add(n as u128)
            .ok_or(Error::InvalidInput("Halton sequence index overflow"))?;
        let mut samples = Vec::with_capacity(n);
        if let Some(permutations) = self.permutations.as_deref() {
            for index in self.generated..end {
                samples.push(
                    self.bases
                        .iter()
                        .zip(permutations)
                        .map(|(base, scramble)| radical_inverse(index, base.prime, Some(scramble)))
                        .collect(),
                );
            }
        } else {
            for index in self.generated..end {
                samples.push(
                    self.bases
                        .iter()
                        .map(|base| radical_inverse_plain(index, base))
                        .collect(),
                );
            }
        }
        self.generated = end;
        Ok(samples)
    }

    /// Skip the next `n` points without generating them.
    ///
    /// This advances the zero-based sequence index in constant time, unlike
    /// SciPy's implementation, which explicitly generates discarded samples.
    /// Zero is a no-op; overflow returns [`Error::InvalidInput`] without
    /// changing the current index.
    pub fn fast_forward(&mut self, n: usize) -> Result<()> {
        self.generated = self
            .generated
            .checked_add(n as u128)
            .ok_or(Error::InvalidInput("Halton sequence index overflow"))?;
        Ok(())
    }

    /// Restart from index zero, preserving any digit permutations.
    pub fn reset(&mut self) {
        self.generated = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reciprocal_division_matches_native_for_all_prime_bases() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        for prime in prime_bases(MAX_DIMENSIONS) {
            let base = HaltonBase::new(prime);
            for n in [
                0_u32,
                1,
                prime - 1,
                prime,
                prime + 1,
                u32::MAX - 1,
                u32::MAX,
            ] {
                assert_eq!(
                    div_rem32(n, &base),
                    (n / prime, n % prime),
                    "32-bit division: base={prime}, n={n}"
                );
            }
            for n in [
                0_u64,
                1,
                u64::from(prime),
                u64::from(u32::MAX),
                u64::from(u32::MAX) + 1,
                u64::MAX - 1,
                u64::MAX,
            ] {
                assert_eq!(
                    div_rem64(n, &base),
                    (n / u64::from(prime), n % u64::from(prime)),
                    "64-bit division: base={prime}, n={n}"
                );
            }
            for _ in 0..128 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let n = state;
                assert_eq!(
                    div_rem32(n as u32, &base),
                    ((n as u32) / prime, (n as u32) % prime),
                    "32-bit random division: base={prime}, n={n}"
                );
                assert_eq!(
                    div_rem64(n, &base),
                    (n / u64::from(prime), n % u64::from(prime)),
                    "64-bit random division: base={prime}, n={n}"
                );
            }
        }
    }

    #[test]
    fn plain_radical_inverse_preserves_original_bits_across_all_index_widths() {
        let fixed_indices = [
            0_u128,
            1,
            (1_u128 << 16) - 1,
            1_u128 << 16,
            (1_u128 << 32) - 1,
            1_u128 << 32,
            (1_u128 << 53) - 1,
            1_u128 << 53,
            (1_u128 << 64) - 1,
            1_u128 << 64,
            (1_u128 << 96) + 123_456_789,
            u128::MAX - 1,
            u128::MAX,
        ];
        let mut state = 0xa5a5_5a5a_1234_5678_u64;
        for prime in prime_bases(MAX_DIMENSIONS) {
            let base = HaltonBase::new(prime);
            for index in 0..1024_u128 {
                assert_eq!(
                    radical_inverse_plain(index, &base).to_bits(),
                    radical_inverse(index, prime, None).to_bits(),
                    "base={prime}, index={index}"
                );
            }
            for index in fixed_indices {
                assert_eq!(
                    radical_inverse_plain(index, &base).to_bits(),
                    radical_inverse(index, prime, None).to_bits(),
                    "base={prime}, index={index}"
                );
            }
            for _ in 0..64 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let index = u128::from(state);
                assert_eq!(
                    radical_inverse_plain(index, &base).to_bits(),
                    radical_inverse(index, prime, None).to_bits(),
                    "base={prime}, index={index}"
                );
            }
        }

        // Exercise the public batch path near every representation boundary.
        let mut engine = Halton::new(4).unwrap();
        for start in [
            0_u128,
            (1_u128 << 32) - 2,
            (1_u128 << 53) - 2,
            (1_u128 << 64) - 2,
            1_u128 << 96,
            u128::MAX - 4,
        ] {
            engine.generated = start;
            for (i, row) in engine.random(3).unwrap().iter().enumerate() {
                for (axis, (&actual, base)) in row.iter().zip(&engine.bases).enumerate() {
                    assert_eq!(
                        actual.to_bits(),
                        radical_inverse(start + i as u128, base.prime, None).to_bits(),
                        "start={start}, i={i}, axis={axis}"
                    );
                }
            }
            assert_eq!(engine.num_generated(), start + 3);
        }
    }

    #[test]
    fn van_der_corput_and_primes() {
        let bases = prime_bases(70);
        assert_eq!(&bases[..6], &[2, 3, 5, 7, 11, 13]);
        assert_eq!(bases[64], 313);
        assert_eq!(radical_inverse(0, 2, None), 0.0);
        assert_eq!(radical_inverse(1, 2, None), 0.5);
        assert_eq!(radical_inverse(2, 2, None), 0.25);
        assert_eq!(radical_inverse(3, 2, None), 0.75);
    }

    #[test]
    fn chunks_and_skips_agree() {
        let mut whole = Halton::new(12).unwrap();
        let expected = whole.random(67).unwrap();
        let mut chunks = Halton::new(12).unwrap();
        let mut actual = chunks.random(7).unwrap();
        actual.extend(chunks.random(60).unwrap());
        assert_eq!(actual, expected);
        chunks.reset();
        chunks.fast_forward(7).unwrap();
        assert_eq!(chunks.num_generated(), 7);
        assert_eq!(chunks.random(60).unwrap(), expected[7..].to_vec());
    }

    #[test]
    fn seeded_scramble_is_repeatable_and_in_range() {
        let mut one = Halton::scrambled(65, 42).unwrap();
        let mut two = Halton::scrambled(65, 42).unwrap();
        let expected = one.random(17).unwrap();
        assert_eq!(expected, two.random(17).unwrap());
        assert!(expected.iter().flatten().all(|x| (0.0..1.0).contains(x)));
        one.reset();
        assert_eq!(expected, one.random(17).unwrap());
        assert_ne!(
            expected,
            Halton::scrambled(65, 43).unwrap().random(17).unwrap()
        );
    }

    #[test]
    fn scrambled_high_index_digits_are_not_ignored() {
        let mut sampler = Halton::scrambled(1, 17).unwrap();
        let scramble = &sampler.permutations.as_ref().unwrap()[0];
        assert_eq!(scramble.permutations.len(), 128);

        // Choose low bits whose scrambled digits are zero, so the 54th
        // scrambled binary digit must remain distinguishable in f64.
        let mut low_index = 0_u128;
        for (bit, permutation) in scramble.permutations.iter().take(53).enumerate() {
            if permutation[0] != 0 {
                low_index |= 1_u128 << bit;
            }
        }
        sampler.generated = low_index;
        let low = sampler.random(1).unwrap()[0][0];
        sampler.generated = low_index | (1_u128 << 53);
        let high = sampler.random(1).unwrap()[0][0];
        assert_ne!(low, high);
        assert!((high - low).abs() > 2_f64.powi(-55));

        sampler.generated = u128::MAX - 1;
        let last = sampler.random(1).unwrap();
        assert!((0.0..1.0).contains(&last[0][0]));
        assert_eq!(sampler.num_generated(), u128::MAX);
        assert!(sampler.random(1).is_err());
        assert_eq!(sampler.num_generated(), u128::MAX);
    }

    #[test]
    fn cached_scramble_tails_match_full_digit_evaluation() {
        let mut rng = StdRng::seed_from_u64(2026);
        for base in [2, 3, 17, 313, 3671] {
            let scramble = random_permutations(base, &mut rng);
            for index in [0, 1, 1024, 1_u128 << 54, u128::MAX] {
                let mut digits = index;
                let mut weight = 1.0 / f64::from(base);
                let mut reference = 0.0;
                for permutation in &scramble.permutations {
                    reference +=
                        f64::from(permutation[(digits % u128::from(base)) as usize]) * weight;
                    digits /= u128::from(base);
                    weight /= f64::from(base);
                }
                let actual = radical_inverse(index, base, Some(&scramble));
                assert!(
                    (actual - reference.min(BELOW_ONE)).abs() <= 2.0 * f64::EPSILON,
                    "base={base}, index={index}"
                );
            }
        }
    }

    #[test]
    fn rejects_bad_dimensions_and_oversized_batches_without_advance() {
        assert!(Halton::new(0).is_err());
        assert!(Halton::new(MAX_DIMENSIONS + 1).is_err());
        let mut engine = Halton::new(2).unwrap();
        assert!(engine.random(MAX_OUTPUT_VALUES).is_err());
        assert_eq!(engine.num_generated(), 0);
        assert!(engine.random(0).unwrap().is_empty());
        engine.fast_forward(1_000_001).unwrap();
        assert_eq!(engine.num_generated(), 1_000_001);
        assert!(
            engine
                .random(3)
                .unwrap()
                .iter()
                .flatten()
                .all(|x| x.is_finite())
        );
    }
}
