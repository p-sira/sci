//! Joe–Kuo direction numbers and a base-two Sobol digital sequence.
//! The bundled table is distributed under `JOE_KUO_LICENSE.txt`.

use crate::{Error, Result};

const MAX_DIMENSIONS: usize = 21_201;
const MAX_BITS: usize = 64;
const INITIAL_VALUES: usize = 18;
const RECORD_BYTES: usize = (INITIAL_VALUES + 1) * 4;
const TABLE: &[u8; MAX_DIMENSIONS * RECORD_BYTES] = include_bytes!("sobol_directions.bin");
const MAX_OUTPUT_VALUES: usize = 1_048_576;

fn table_value(dimension: usize, column: usize) -> u32 {
    let start = dimension * RECORD_BYTES + column * 4;
    u32::from_le_bytes(TABLE[start..start + 4].try_into().unwrap())
}

fn directions(dimension: usize, bits: usize) -> Vec<u64> {
    if dimension == 0 {
        return (0..bits).map(|j| 1_u64 << (bits - j - 1)).collect();
    }

    let polynomial = table_value(dimension, 0);
    let degree = (u32::BITS - 1 - polynomial.leading_zeros()) as usize;
    let mut v = vec![0; bits];
    for (j, value) in v.iter_mut().enumerate().take(degree.min(bits)) {
        *value = u64::from(table_value(dimension, j + 1)) << (bits - j - 1);
    }
    for j in degree..bits {
        let mut value = v[j - degree] ^ (v[j - degree] >> degree);
        for k in 1..degree {
            if (polynomial >> (degree - k)) & 1 != 0 {
                value ^= v[j - k];
            }
        }
        v[j] = value;
    }
    v
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }
}

// A random nonsingular lower triangular matrix over GF(2), with the most
// significant bit representing the first row/column (Matousek's LMS scramble).
fn scramble(directions: &mut [u64], bits: usize, generator: &mut SplitMix64) {
    let original = directions.to_vec();
    directions.fill(0);
    for row in 0..bits {
        let bit = bits - row - 1;
        let diagonal = 1_u64 << bit;
        let above = if row == 0 {
            0
        } else {
            generator.next() & (u64::MAX << (bit + 1))
        };
        let mask = diagonal | above;
        for (new, &old) in directions.iter_mut().zip(&original) {
            if (old & mask).count_ones() & 1 != 0 {
                *new |= diagonal;
            }
        }
    }
}

/// Stateful Sobol sequence in the unit hypercube, using the Joe–Kuo D(6) table.
///
/// For each zero-based index `n`, the `j`th coordinate is the fixed-point binary
/// fraction `2⁻ᵇ (sⱼ ⊕ ⨁ₖ gₖ(n) vⱼₖ)`, where `g(n) = n ⊕ (n >> 1)` is the
/// binary Gray code, `vⱼₖ` are direction numbers, `b` is the bit depth, and
/// `sⱼ` is zero for an unscrambled sequence. Optional LMS scrambling applies a
/// nonsingular binary matrix and digital shift independently to each axis.
///
/// ## Parameters and conventions
///
/// Construct with [`Sobol::new`] for an unscrambled sequence or
/// [`Sobol::scrambled`] with a reproducible seed. Dimensions are coordinates;
/// output rows are consecutive points starting at index zero. The unscrambled
/// sequence begins with the origin. Each call advances the same engine.
///
/// ## Domain and special values
///
/// `1 <= d <= 21201`, `1 <= bits <= 64`, and at most `2^bits` distinct
/// fixed-point states (at high bit depths, distinct states can round to the
/// same `f64` point).
/// `random(0)` and `fast_forward(0)` leave the state unchanged. Inputs are
/// integer counts, so NaN and infinity cannot occur.
///
/// ## Returns
///
/// A generator producing `Vec<Vec<f64>>` with shape `(n, d)`. Coordinates are
/// dimensionless and in `[0, 1)` after floating-point conversion.
///
/// ## Errors
///
/// Invalid dimensions, bit depths, excessive batches, and exhaustion are
/// reported as [`Error::InvalidInput`]; no partial state is consumed on error.
///
/// ## Numerical behavior
///
/// Unscrambled samples agree with the SciPy direction-number sequence.
/// Batches are limited to 1,048,576 scalar values to bound allocation.
/// Use [`Sobol::random_base2`] to retain the balance of a base-two design;
/// arbitrary lengths or skips may weaken its low-discrepancy properties.
/// The seeded LMS scramble uses SplitMix64 to generate its binary matrices;
/// the same Rust seed does not imply the same SciPy/NumPy scramble.
/// Integer coordinates have up to 64 bits; converting to `f64` rounds when
/// `bits > 53`, and values that round to 1 are clamped to the next float below 1.
///
/// # Examples
///
/// ```
/// use sci_stats::qmc::Sobol;
/// let mut sampler = Sobol::new(2, 30)?;
/// assert_eq!(sampler.random_base2(2)?, vec![
///     vec![0.0, 0.0], vec![0.5, 0.5],
///     vec![0.75, 0.25], vec![0.25, 0.75],
/// ]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [Joe and Kuo (2008), Constructing Sobol sequences](https://doi.org/10.1137/070709359),
/// [SciPy `qmc.Sobol`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html),
/// and [Kuo's direction-number tables](https://web.maths.unsw.edu.au/~fkuo/sobol/).
#[derive(Clone, Debug)]
pub struct Sobol {
    bits: usize,
    directions: Vec<Vec<u64>>,
    shift: Vec<u64>,
    state: Vec<u64>,
    generated: u128,
}

impl Sobol {
    /// Construct an unscrambled `d`-dimensional Sobol generator with `bits` binary digits.
    ///
    /// The complete real sequence uses Joe–Kuo direction numbers and starts
    /// with the zero vector. Dimensions must be `1..=21201` and bits `1..=64`.
    /// Otherwise returns [`Error::InvalidInput`].
    pub fn new(d: usize, bits: usize) -> Result<Self> {
        if d == 0 || d > MAX_DIMENSIONS || bits == 0 || bits > MAX_BITS {
            return Err(Error::InvalidInput(
                "Sobol requires 1..=21201 dimensions and 1..=64 bits",
            ));
        }
        let directions = (0..d).map(|axis| directions(axis, bits)).collect();
        Ok(Self {
            bits,
            directions,
            shift: vec![0; d],
            state: vec![0; d],
            generated: 0,
        })
    }

    /// Construct an LMS-scrambled Sobol sequence with a deterministic `seed`.
    ///
    /// The same seed and parameters reproduce the same digital scramble within
    /// Sci; SciPy uses a different random-number stream. The bounds and errors
    /// are identical to [`Sobol::new`].
    pub fn scrambled(d: usize, bits: usize, seed: u64) -> Result<Self> {
        let mut sobol = Self::new(d, bits)?;
        let mut generator = SplitMix64(seed);
        let mask = if bits == 64 {
            u64::MAX
        } else {
            (1_u64 << bits) - 1
        };
        for axis in 0..d {
            sobol.shift[axis] = generator.next() & mask;
            scramble(&mut sobol.directions[axis], bits, &mut generator);
        }
        sobol.reset();
        Ok(sobol)
    }

    /// Number of coordinates per generated point.
    pub fn dimensions(&self) -> usize {
        self.directions.len()
    }

    /// Precision in binary digits used for the digital net.
    pub fn bits(&self) -> usize {
        self.bits
    }

    /// Total number of points consumed, including skipped points.
    pub fn num_generated(&self) -> u128 {
        self.generated
    }

    /// Generate the next `n` points in `[0,1)^d`.
    ///
    /// ## Parameters and conventions
    ///
    /// `n` is the number of new rows; use [`Sobol::random_base2`] to preserve
    /// power-of-two balance when building a design. `random(0)` returns an
    /// empty vector and retains the current state.
    ///
    /// ## Returns
    ///
    /// A row-major `(n,d)` vector of `f64` coordinates.
    ///
    /// ## Errors
    ///
    /// Returns [`Error::InvalidInput`] if the batch exceeds 1,048,576 scalar
    /// values or would pass the `2^bits` sequence limit.
    ///
    /// ## Numerical behavior
    ///
    /// Generation costs `O(nd)` time and `O(nd)` output memory. Floating-point
    /// rounding is described in [`Sobol`].
    pub fn random(&mut self, n: usize) -> Result<Vec<Vec<f64>>> {
        let d = self.dimensions();
        if n.checked_mul(d).is_none_or(|v| v > MAX_OUTPUT_VALUES) {
            return Err(Error::InvalidInput("Sobol batch exceeds 1048576 values"));
        }
        let end = self.generated + n as u128;
        if end > self.maximum() {
            return Err(Error::InvalidInput(
                "Sobol sequence exhausted at 2^bits points",
            ));
        }
        let factor = 2_f64.powi(-(self.bits as i32));
        let maximum = self.maximum();
        let below_one = f64::from_bits(1.0_f64.to_bits() - 1);
        let mut points = Vec::with_capacity(n);
        for _ in 0..n {
            points.push(
                self.state
                    .iter()
                    .map(|x| ((*x as f64) * factor).min(below_one))
                    .collect(),
            );
            self.generated += 1;
            if self.generated < maximum {
                let bit = self.generated.trailing_zeros() as usize;
                for (value, direction) in self.state.iter_mut().zip(&self.directions) {
                    *value ^= direction[bit];
                }
            }
        }
        Ok(points)
    }

    /// Generate `2^m` points while preserving base-two design balance.
    ///
    /// ## Parameters and conventions
    ///
    /// `m` is the nonnegative power of two, and the *total* generated count
    /// after the call must be a power of two. For example, call with `m=3`
    /// once to generate eight points, or twice with `m=2` to produce the same
    /// eight-point design. Use [`Sobol::random`] for arbitrary partial batches.
    ///
    /// ## Returns
    ///
    /// The next `2^m` rows, in sequence order.
    ///
    /// ## Errors
    ///
    /// Returns [`Error::InvalidInput`] for `m > bits`, unbalanced totals,
    /// generation beyond `2^bits`, or output exceeding the allocation limit.
    ///
    /// ## Numerical behavior
    ///
    /// Uses the same fixed-point generation and rounding as [`Sobol::random`].
    pub fn random_base2(&mut self, m: usize) -> Result<Vec<Vec<f64>>> {
        if m > self.bits || m >= usize::BITS as usize {
            return Err(Error::InvalidInput("invalid Sobol power-of-two batch size"));
        }
        let n = 1_usize << m;
        if !(self.generated + n as u128).is_power_of_two() {
            return Err(Error::InvalidInput("Sobol total must be a power of two"));
        }
        self.random(n)
    }

    /// Skip `n` points in logarithmic sequence-index work.
    ///
    /// ## Parameters and conventions
    ///
    /// Skipped points count towards the total sequence limit; zero is a no-op.
    ///
    /// ## Returns
    ///
    /// `Ok(())` on a successful skip.
    ///
    /// ## Errors
    ///
    /// Returns [`Error::InvalidInput`] when skipping past `2^bits` points.
    ///
    /// ## Numerical behavior
    ///
    /// The next point is reconstructed from its binary Gray code in `O(d bits)`;
    /// this gives the same result as drawing and discarding `n` points.
    pub fn fast_forward(&mut self, n: usize) -> Result<()> {
        let end = self.generated + n as u128;
        if end > self.maximum() {
            return Err(Error::InvalidInput(
                "Sobol sequence exhausted at 2^bits points",
            ));
        }
        self.generated = end;
        if end == self.maximum() {
            return Ok(());
        }
        let gray = end ^ (end >> 1);
        for ((value, shift), directions) in
            self.state.iter_mut().zip(&self.shift).zip(&self.directions)
        {
            *value = *shift;
            for (bit, &direction) in directions.iter().enumerate() {
                if (gray >> bit) & 1 != 0 {
                    *value ^= direction;
                }
            }
        }
        Ok(())
    }

    /// Restore the original sequence and scramble, including the first point.
    ///
    /// This operation cannot fail and does not regenerate the scramble.
    pub fn reset(&mut self) {
        self.state.clone_from(&self.shift);
        self.generated = 0;
    }

    fn maximum(&self) -> u128 {
        1_u128 << self.bits
    }
}
