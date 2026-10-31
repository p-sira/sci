//! Native scalar additions beyond the SciPy inventory.
use crate::{Error, Result};
use sci_core::Float;

/// Computes `ln(1+x) − x` without cancellation near zero.
///
/// ## Parameters and conventions
/// `x` is dimensionless; the logarithm is natural.
///
/// ## Domain and special values
/// Real `x >= −1`. At `−1` and positive infinity the result is negative
/// infinity. Values below `−1` and NaN return NaN. Both zeros return negative zero.
///
/// ## Returns
/// The nonpositive logarithmic remainder in the input precision.
///
/// ## Numerical behavior
/// Uses the defining alternating series for `|x| < 0.5` (at most 128 terms),
/// otherwise `ln_1p(x) − x`. Computation stays in `T`; underflow is possible.
///
/// ## Examples
/// ```
/// assert!((sci_special::elementary::log1pmx(1e-10_f64) + 5e-21).abs() < 1e-30);
/// ```
///
/// ## References
/// [NIST DLMF §4.6](https://dlmf.nist.gov/4.6).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn log1pmx<T: Float>(x: T) -> T {
    if x == T::infinity() {
        return T::neg_infinity();
    }
    if x.abs() >= 0.5 || x.is_nan() {
        return x.ln_1p() - x;
    }
    let mut power = -x * x;
    let mut sum = power / 2.0;
    for n in 3..=128 {
        power = -power * x;
        let term = power / T::from(n).unwrap();
        let next = sum + term;
        if next == sum {
            break;
        }
        sum = next;
    }
    sum
}

/// Computes the positive-real inverse digamma: `ψ(x) = y`, with `x > 0`.
///
/// ## Parameters and conventions
/// `y` is a dimensionless real digamma value; negative-real inverse branches
/// are excluded. The positive branch is unique and increasing.
///
/// ## Domain and special values
/// Every real `y` is accepted. Negative infinity returns zero, positive
/// infinity returns infinity, and NaN propagates. Overflow returns infinity.
///
/// ## Returns
/// The positive solution converted to the requested float type.
///
/// ## Numerical behavior
/// Reuses Sci's digamma and trigamma in binary64. Minka's initial estimate is
/// refined with at most 32 positivity-preserving Newton steps. Extreme inputs
/// use the leading asymptotic inverse; convergence failure returns NaN.
/// No arbitrary-precision or correctly-rounded guarantee is provided.
///
/// ## Examples
/// ```
/// let x = sci_special::gamma::digammainv(sci_special::gamma::digamma(2.5_f64));
/// assert!((x - 2.5).abs() < 1e-13);
/// ```
///
/// ## References
/// [Minka, Estimating a Dirichlet distribution, Appendix C](https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf).
pub fn digammainv<T: Float>(y: T) -> T {
    let y = y.to_f64().unwrap_or(f64::NAN);
    let mut x = if y >= -2.22 {
        y.exp() + 0.5
    } else {
        -1.0 / (y + 0.5772156649015329)
    };
    if !y.is_finite() || x < 1e-150 || y > 36.0 {
        return T::from(x).unwrap_or(T::nan());
    }
    for _ in 0..32 {
        let step = (crate::gamma::digamma(x) - y) / crate::gamma::trigamma(x);
        let next = if step < x { x - step } else { x * 0.5 };
        if next == x || (next - x).abs() <= 4.0 * f64::EPSILON * x {
            return T::from(next).unwrap_or(T::nan());
        }
        x = next;
    }
    T::nan()
}

/// Computes `Γ(a)/Γ(b)` for positive real arguments.
///
/// ## Parameters and conventions
/// `a` and `b` are dimensionless positive gamma arguments, in numerator and
/// denominator order. Negative arguments and pole cancellation are excluded.
///
/// ## Domain and special values
/// Both arguments must be finite and positive; invalid inputs return NaN.
/// Equal valid arguments return one. Overflow returns infinity; underflow zero.
///
/// ## Returns
/// The positive ratio, converted from binary64 to the requested float type.
///
/// ## Numerical behavior
/// Reuses [`crate::combinatorics::poch`] for representable argument differences up to 10,000
/// where recurrence cannot cross a tiny argument (bounded
/// product/reciprocal recurrence), and log-beta and log-gamma otherwise through
/// `Γ(b+d)/Γ(b) = Γ(d)/B(b,d)`.
/// This avoids overflowing either gamma value and avoids subtracting two large
/// log-gamma values for nearby arguments. Logarithmic cancellation can still
/// limit relative accuracy for extreme parameters. Recurrence work is at most
/// 10,000 steps; larger differences use the existing bounded kernels.
///
/// ## Examples
/// ```
/// assert!((sci_special::gamma::gamma_ratio(201.0_f64, 200.0) - 200.0).abs() < 1e-10);
/// ```
///
/// ## References
/// [NIST DLMF §5.12.1](https://dlmf.nist.gov/5.12.E1).
pub fn gamma_ratio<T: Float>(a: T, b: T) -> T {
    let (a, b) = (
        a.to_f64().unwrap_or(f64::NAN),
        b.to_f64().unwrap_or(f64::NAN),
    );
    if !a.is_finite() || !b.is_finite() || a <= 0.0 || b <= 0.0 {
        return T::nan();
    }
    if a == b {
        return T::one();
    }
    let d = (a - b).abs();
    if d <= 10000.0 && b + (a - b) == a && ((a >= 1.0 && b >= 1.0) || d < 1.0) {
        return T::from(crate::combinatorics::poch(b, a - b)).unwrap_or(T::nan());
    }
    let log = crate::gamma::gammaln(d) - crate::beta::betaln(a.min(b), d);
    T::from(if a > b { log.exp() } else { (-log).exp() }).unwrap_or(T::nan())
}

/// Computes `F(x) = x exp(x) E₁(x) = ∫₀∞ exp(−t)/(1+t/x) dt` for `x > 0`.
///
/// ## Parameters and conventions
/// `x` is dimensionless. This scaling includes the factor `x`, following XSF;
/// it differs from the common scaling `exp(x) E₁(x)`.
///
/// ## Domain and special values
/// Nonnegative real inputs. Zero returns zero, positive infinity returns one,
/// and negative inputs or NaN return NaN.
///
/// ## Returns
/// A value in `[0,1]`, increasing with `x`, in the input precision.
///
/// ## Numerical behavior
/// For `x <= 1`, reuses E₁ in binary64. Larger inputs use a bounded backward
/// continued fraction (at most 100 terms), evaluated in `T`, avoiding exponential
/// overflow and E₁ underflow. Correct rounding is not guaranteed.
///
/// ## Examples
/// ```
/// assert!((sci_special::integrals::scaled_exp1(1000.0_f64) - 0.9990019940238807).abs() < 1e-14);
/// ```
///
/// ## References
/// [NIST DLMF §6.9, exponential-integral continued fractions](https://dlmf.nist.gov/6.9).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn scaled_exp1<T: Float>(x: T) -> T {
    if x < 0.0 || x.is_nan() {
        return T::nan();
    }
    if x == 0.0 {
        return T::zero();
    }
    if x == T::infinity() {
        return T::one();
    }
    if x <= 1.0 {
        return x * x.exp() * crate::integrals::exp1(x);
    }
    let count = 20 + (80.0 / x).to_usize().unwrap_or(0);
    let mut tail = T::zero();
    for k in (1..=count).rev() {
        let k = T::from(k).unwrap();
        tail = k / (x + k / (1.0 + tail));
    }
    1.0 / (1.0 + tail)
}

/// Computes `Σ(i=j..k) i^(−a) / Σ(i=1..n) i^(−a)`.
///
/// ## Parameters and conventions
/// Inclusive integer indices `j`, `k`, and `n` specify a truncated power-law
/// probability. `a` is a dimensionless real exponent; logarithms are natural.
///
/// ## Domain and special values
/// Requires `1 <= j <= k <= n <= 1_000_000` and finite `a`. The full interval
/// returns one. Unlike XSF's raw sums, large negative exponents remain normalized
/// even when the individual powers overflow.
///
/// ## Returns
/// A probability in `[0,1]` in the selected float type.
///
/// ## Errors
/// [`Error::InvalidInput`] for invalid indices, nonfinite exponent, or work limit.
///
/// ## Numerical behavior
/// Scales weights by their maximum and accumulates numerator and denominator
/// with compensated summation in `T`. Work is `O(n)` and storage is constant.
/// Small relative weights can underflow; no zeta subtraction is used.
///
/// ## Examples
/// ```
/// assert_eq!(sci_special::combinatorics::normalized_gen_harmonic(2, 3, 4, 0.0_f64)?, 0.5);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
/// [NIST DLMF §25.11](https://dlmf.nist.gov/25.11) (finite Hurwitz-zeta differences).
pub fn normalized_gen_harmonic<T: Float>(j: u32, k: u32, n: u32, a: T) -> Result<T> {
    if j == 0 || k < j || n < k || n > 1_000_000 || !a.is_finite() {
        return Err(Error::InvalidInput(
            "require 1 <= j <= k <= n <= 1000000 and finite exponent",
        ));
    }
    let (mut numerator, mut denominator, mut nc, mut dc) =
        (T::zero(), T::zero(), T::zero(), T::zero());
    let anchor = if a < T::zero() { n } else { 1 };
    for i in 1..=n {
        let log = if a < T::zero() {
            // ln(i/n) = ln1p((i-n)/n), retaining adjacent large indices.
            let fraction = T::from(i).unwrap() / T::from(n).unwrap();
            if i >= n / 2 {
                ((T::from(i).unwrap() - T::from(n).unwrap()) / T::from(n).unwrap()).ln_1p()
            } else {
                fraction.ln()
            }
        } else {
            T::from(i).unwrap().ln()
        };
        let weight = if i == anchor {
            T::one()
        } else {
            (-a * log).exp()
        };
        let y = weight - dc;
        let next = denominator + y;
        dc = (next - denominator) - y;
        denominator = next;
        if (j..=k).contains(&i) {
            let y = weight - nc;
            let next = numerator + y;
            nc = (next - numerator) - y;
            numerator = next;
        }
    }
    Ok(numerator / denominator)
}

/// Computes the adjacent modified-Bessel ratio `I_v(x)/I_(v−1)(x)`.
///
/// ## Parameters and conventions
/// `v >= 1/2` is the numerator order and `x >= 0` the dimensionless argument.
/// The ratio is defined by its continuous limit at zero.
///
/// ## Domain and special values
/// Invalid domains and NaN return NaN. Signed zero `x` is preserved. Infinite
/// `x` with finite order returns one; infinite order with finite `x` returns
/// zero. Two infinite arguments have no unique limit and return NaN.
///
/// ## Returns
/// A ratio in `[0,1]`; `v=1/2` reduces to `tanh(x)`.
///
/// ## Numerical behavior
/// A scaled Perron continued fraction avoids overflowing individual Bessel
/// functions, with at most 1000 terms and private double-double accumulation.
/// The result is converted from binary64 to `T`; convergence failure returns
/// NaN. Very small complements are limited by binary64 underflow. Use
/// [`iv_ratio_c`] rather than subtracting this value from one.
///
/// ## Examples
/// ```
/// assert!((sci_special::bessel::iv_ratio(1.0_f64, 1.0) - 0.4463899658965345).abs() < 1e-14);
/// ```
///
/// ## References
/// [Gautschi and Slavik, modified Bessel function ratios (1978)](https://doi.org/10.1090/S0025-5718-1978-0470267-9).
pub fn iv_ratio<T: Float>(v: T, x: T) -> T {
    let pair = crate::backend::iv_ratio_pair(
        v.to_f64().unwrap_or(f64::NAN),
        x.to_f64().unwrap_or(f64::NAN),
    );
    T::from(pair.0).unwrap_or(T::nan())
}

/// Computes `1 − I_v(x)/I_(v−1)(x)` without subtracting a rounded ratio.
///
/// ## Parameters and conventions
/// `v` is the numerator order and `x` a dimensionless nonnegative argument.
///
/// ## Domain and special values
/// Same domain as [`iv_ratio`]. At zero `x` the result is one; infinite `x`
/// returns zero for finite order. Infinite order and finite `x` return one.
/// Invalid domains, NaN, and two infinite arguments return NaN.
///
/// ## Returns
/// The complementary ratio in `[0,1]`, retaining small values when possible.
///
/// ## Numerical behavior
/// Shares the scaled continued fraction and 1000-term limit of [`iv_ratio`].
/// The half-order uses `2 exp(−2x)/(1+exp(−2x))`. Computation uses binary64
/// with double-double accumulation; tiny results can underflow.
///
/// ## Examples
/// ```
/// let c = sci_special::bessel::iv_ratio_c(1.0_f64, 1e10);
/// assert!((c / 5e-11 - 1.0).abs() < 1e-9);
/// ```
///
/// ## References
/// [Gautschi and Slavik (1978)](https://doi.org/10.1090/S0025-5718-1978-0470267-9).
pub fn iv_ratio_c<T: Float>(v: T, x: T) -> T {
    let pair = crate::backend::iv_ratio_pair(
        v.to_f64().unwrap_or(f64::NAN),
        x.to_f64().unwrap_or(f64::NAN),
    );
    T::from(pair.1).unwrap_or(T::nan())
}

/// Solves `I_v(x)/I_(v−1)(x) = r` for nonnegative `x`.
///
/// ## Parameters and conventions
/// `v >= 1/2` is finite; `r` is a dimensionless probability in `[0,1]`.
/// The inverse selects the unique nonnegative argument.
///
/// ## Domain and special values
/// Invalid parameters and NaN return NaN. At `r=0` returns zero; at `r=1`
/// returns infinity. The half-order inverse is `atanh(r)`.
///
/// ## Returns
/// The nonnegative inverse ratio converted to the input float type.
///
/// ## Numerical behavior
/// Reuses [`iv_ratio`] and its complement. Bisection in binary64 uses at most
/// 128 evaluations, bracketed by zero and `2vr/(1−r²)`. Complement residuals
/// preserve accuracy near one. Each evaluation has a 1000-term bound.
/// Failure returns NaN; an unrepresentable upper bracket returns infinity.
/// Conditioning worsens as `r` approaches one; correct rounding is not guaranteed.
///
/// ## Examples
/// ```
/// let x = sci_special::bessel::iv_ratioinv(1.0_f64, 0.5);
/// assert!((sci_special::bessel::iv_ratio(1.0, x) - 0.5).abs() < 1e-14);
/// ```
///
/// ## References
/// [NIST DLMF §10.41](https://dlmf.nist.gov/10.41).
pub fn iv_ratioinv<T: Float>(v: T, r: T) -> T {
    let (v, r) = (
        v.to_f64().unwrap_or(f64::NAN),
        r.to_f64().unwrap_or(f64::NAN),
    );
    if !v.is_finite() || v < 0.5 || !(0.0..=1.0).contains(&r) {
        return T::nan();
    }
    if r == 0.0 {
        return T::zero();
    }
    if r == 1.0 {
        return T::infinity();
    }
    if v == 0.5 {
        return T::from(r.atanh()).unwrap_or(T::nan());
    }
    let mut high = (v * r) / (1.0 - r) * (2.0 / (1.0 + r));
    if high.is_infinite() {
        return T::infinity();
    }
    let mut low = 0.0;
    for _ in 0..128 {
        let mid = low + (high - low) * 0.5;
        let (value, complement) = crate::backend::iv_ratio_pair(v, mid);
        if value.is_nan() {
            return T::nan();
        }
        let residual = if r > 0.5 {
            (1.0 - r) - complement
        } else {
            value - r
        };
        if residual == 0.0 || high - low <= 4.0 * f64::EPSILON * mid {
            return T::from(mid).unwrap_or(T::nan());
        }
        if residual < 0.0 {
            low = mid;
        } else {
            high = mid;
        }
    }
    T::nan()
}
