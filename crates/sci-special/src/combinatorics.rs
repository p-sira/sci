//! Harmonic numbers and logarithmic combinatorial coefficients.

use crate::{Error, Result};

const MAX_TERMS: u64 = 1_000_000;

fn sum(values: impl Iterator<Item = f64>) -> f64 {
    let (mut total, mut correction) = (0.0, 0.0);
    for value in values {
        let adjusted = value - correction;
        let next = total + adjusted;
        if !next.is_finite() {
            return next;
        }
        correction = (next - total) - adjusted;
        total = next;
    }
    total
}

/// Computes the harmonic number H_n = Σ_{j=1}^n 1/j.
///
/// ## Parameters and conventions
/// `n` is a nonnegative integer index. The empty sum H_0 is zero;
/// this corrects Statrs's special convention H_0 = 1.
///
/// ## Domain and special values
/// Every `u64` index is accepted. H_0 = 0 and H_1 = 1 exactly.
///
/// ## Returns
/// The dimensionless harmonic number as `f64`.
///
/// ## Numerical behavior
/// Uses H_n = ψ(n+1) − ψ(1) through Sci's private special-function backend,
/// with constant work. Indices above 2^53 are rounded in conversion to `f64`;
/// backend approximation and subtraction limit accuracy at small indices.
///
/// ## Examples
/// ```
/// assert!((sci_special::combinatorics::harmonic(3) - 11.0 / 6.0).abs() < 1e-12);
/// ```
///
/// ## References
/// [NIST DLMF §5.4.14](https://dlmf.nist.gov/5.4.E14).
pub fn harmonic(n: u64) -> f64 {
    match n {
        0 => 0.0,
        1 => 1.0,
        _ => crate::gamma::digamma(n as f64 + 1.0) - crate::gamma::digamma(1.0),
    }
}

/// Computes the generalized harmonic number H_n^(m) = Σ_{j=1}^n j^(−m).
///
/// ## Parameters and conventions
/// `n` is a nonnegative integer index; `m` is a dimensionless real exponent.
/// Argument order follows Statrs. The empty sum is zero rather than Statrs's one.
///
/// ## Domain and special values
/// Requires `n <= 1_000_000` and finite `m`, including negative exponents.
/// NaN and infinite exponents are rejected even for an empty sum.
/// H_0^(m) = 0; H_n^(0) = n and H_1^(m) = 1 exactly.
///
/// ## Returns
/// The dimensionless finite sum.
///
/// ## Errors
/// [`Error::InvalidInput`] for invalid inputs or work limits;
/// [`Error::NonFiniteEvaluation`] if a term or sum overflows.
///
/// ## Numerical behavior
/// At most `n` powers, accumulated with compensated summation in binary64.
/// Underflow can discard small terms. No analytic continuation is performed.
///
/// ## Examples
/// ```
/// assert_eq!(sci_special::combinatorics::gen_harmonic(2, 2.0)?, 1.25);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF §25.11](https://dlmf.nist.gov/25.11) (finite Hurwitz-zeta sums).
pub fn gen_harmonic(n: u64, m: f64) -> Result<f64> {
    if n > MAX_TERMS || !m.is_finite() {
        return Err(Error::InvalidInput(
            "invalid harmonic exponent or term limit",
        ));
    }
    match sum((1..=n).map(|j| (j as f64).powf(-m))) {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}

/// Computes ln(n!) = ln Γ(n+1).
///
/// ## Parameters and conventions
/// `n` is a nonnegative integer; logarithms are natural.
///
/// ## Domain and special values
/// Every `u64` index is accepted. Returns zero for `n = 0` or `n = 1`.
///
/// ## Returns
/// The dimensionless logarithm, without overflowing the factorial itself.
///
/// ## Numerical behavior
/// Reuses [`crate::gamma::gammaln`] with constant work. Indices above 2^53 are rounded
/// to binary64; gamma approximation and argument rounding limit accuracy.
///
/// ## Examples
/// ```
/// assert!((sci_special::combinatorics::ln_factorial(5) - 120.0_f64.ln()).abs() < 1e-12);
/// ```
///
/// ## References
/// [NIST DLMF §5.4.1](https://dlmf.nist.gov/5.4.E1).
pub fn ln_factorial(n: u64) -> f64 {
    if n <= 1 {
        0.0
    } else {
        crate::gamma::gammaln(n as f64 + 1.0)
    }
}

/// Computes ln C(n,k) = ln(n!/(k!(n−k)!)).
///
/// ## Parameters and conventions
/// `n` and `k` are nonnegative integers; logarithms are natural.
///
/// ## Domain and special values
/// Requires `k <= n` and `min(k,n−k) <= 1_000_000`.
/// Returns zero for `k = 0` or `k = n`.
///
/// ## Returns
/// The dimensionless logarithm of the binomial coefficient.
///
/// ## Errors
/// [`Error::InvalidInput`] for `k > n` or excessive work. Unlike Statrs's
/// `−∞` for `k > n`, Sci reports the invalid domain explicitly.
///
/// ## Numerical behavior
/// Sums ln(1+(n−r)/j), j=1..r, r=min(k,n−k), with compensated summation.
/// This avoids subtracting large log-factorials for small `k`. At most
/// 1,000,000 logarithms; integer conversion above 2^53 introduces rounding.
///
/// ## Examples
/// ```
/// assert!((sci_special::combinatorics::ln_binomial(5, 2)? - 10.0_f64.ln()).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF §26.3](https://dlmf.nist.gov/26.3).
pub fn ln_binomial(n: u64, k: u64) -> Result<f64> {
    if k > n {
        return Err(Error::InvalidInput("binomial selection exceeds total"));
    }
    let r = k.min(n - k);
    if r > MAX_TERMS {
        return Err(Error::InvalidInput("binomial term limit exceeded"));
    }
    Ok(log_binomial(n, r))
}

fn log_binomial(n: u64, r: u64) -> f64 {
    sum((1..=r).map(|j| ((n - r) as f64 / j as f64).ln_1p()))
}

fn multinomial_work(n: u64, counts: &[u64]) -> Result<()> {
    if counts.len() > MAX_TERMS as usize {
        return Err(Error::InvalidInput("multinomial category limit exceeded"));
    }
    let (mut remaining, mut work) = (n, 0);
    for &count in counts {
        if count > remaining {
            return Err(Error::InvalidInput("multinomial counts must sum to total"));
        }
        remaining -= count;
        work += count.min(remaining).min(MAX_TERMS + 1);
        if work > MAX_TERMS {
            return Err(Error::InvalidInput("multinomial term limit exceeded"));
        }
    }
    if remaining != 0 {
        return Err(Error::InvalidInput("multinomial counts must sum to total"));
    }
    Ok(())
}

/// Computes `ln(n!/Π_j counts[j]!)`.
///
/// ## Parameters and conventions
/// `n` is the total and `counts` contains nonnegative integer category counts.
/// Category order does not change the mathematical result; logarithms are natural.
///
/// ## Domain and special values
/// Counts must sum to `n`. Empty counts are accepted only for `n = 0`,
/// returning zero. At most 1,000,000 categories and 1,000,000 logarithmic terms
/// across sequential binomial factors are permitted.
///
/// ## Returns
/// The dimensionless logarithm of the multinomial coefficient.
///
/// ## Errors
/// [`Error::InvalidInput`] for an inconsistent total or excessive work.
///
/// ## Numerical behavior
/// Compensated sum of [`ln_binomial`] factors, avoiding log-factorial
/// cancellation. Category ordering can affect rounding and required work.
/// Integers above 2^53 round during conversion to binary64.
///
/// ## Examples
/// ```
/// assert!((sci_special::combinatorics::ln_multinomial(4, &[2, 1, 1])? - 12.0_f64.ln()).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF §26.4](https://dlmf.nist.gov/26.4).
pub fn ln_multinomial(n: u64, counts: &[u64]) -> Result<f64> {
    multinomial_work(n, counts)?;
    let mut remaining = n;
    Ok(sum(counts.iter().map(|&count| {
        // Validation has already checked each selection and the total work.
        let r = count.min(remaining - count);
        let value = log_binomial(remaining, r);
        remaining -= count;
        value
    })))
}

/// Computes the multinomial coefficient `n!/Π_j counts[j]!` as `f64`.
///
/// ## Parameters and conventions
/// `n` is the total; `counts` contains integer category counts summing to `n`.
///
/// ## Domain and special values
/// Same domain and work limits as [`ln_multinomial`]. Empty counts with `n = 0`
/// return one. Invalid totals produce an error rather than Statrs's panic.
///
/// ## Returns
/// The dimensionless coefficient, rounded to an integer when representable.
/// Results above 2^53 cannot represent every integer exactly.
///
/// ## Errors
/// [`Error::InvalidInput`] for invalid counts or excessive work;
/// [`Error::NonFiniteEvaluation`] on overflow. Use [`ln_multinomial`] for
/// coefficients too large for binary64.
///
/// ## Numerical behavior
/// Exponentiates [`ln_multinomial`] and rounds to the nearest integer.
/// Logarithmic and exponential rounding can affect large coefficients;
/// this is not an exact integer algorithm.
///
/// ## Examples
/// ```
/// assert_eq!(sci_special::combinatorics::multinomial(4, &[2, 1, 1])?, 12.0);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF §26.4](https://dlmf.nist.gov/26.4).
pub fn multinomial(n: u64, counts: &[u64]) -> Result<f64> {
    match ln_multinomial(n, counts)?.exp().round() {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}
