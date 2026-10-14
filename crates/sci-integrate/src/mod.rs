use crate::error::{finite, finite_slice};
use crate::{Error, Result};

const MAX_SAMPLES: usize = 1_000_001;

fn validate_samples(y: &[f64]) -> Result<()> {
    if y.is_empty() || y.len() > MAX_SAMPLES {
        return Err(Error::InvalidInput("sample count must be in 1..=1000001"));
    }
    finite_slice(y)
}

fn spacings(y: &[f64], x: Option<&[f64]>, dx: f64) -> Result<Vec<f64>> {
    validate_samples(y)?;
    if let Some(x) = x {
        if x.len() != y.len() {
            return Err(Error::DimensionMismatch);
        }
        finite_slice(x)?;
        x.windows(2).map(|pair| finite(pair[1] - pair[0])).collect()
    } else {
        if !dx.is_finite() {
            return Err(Error::InvalidInput("spacing must be finite"));
        }
        Ok(vec![dx; y.len() - 1])
    }
}

// Multiply the smallest and largest magnitudes first. This avoids a spurious
// intermediate overflow/underflow when three finite factors balance each other.
fn product3(a: f64, b: f64, c: f64) -> f64 {
    let mut factors = [a, b, c];
    factors.sort_unstable_by(|a, b| a.abs().total_cmp(&b.abs()));
    (factors[0] * factors[2]) * factors[1]
}

fn sample_scale(magnitude: f64) -> f64 {
    // A power of two makes normalization exact, including for subnormal data.
    f64::from_bits(magnitude.to_bits() & 0x7ff0_0000_0000_0000).max(f64::MIN_POSITIVE)
}

fn trapezoid_interval(a: f64, b: f64, h: f64) -> f64 {
    let magnitude = a.abs().max(b.abs());
    if magnitude == 0.0 {
        return h * (a + b);
    }
    let scale = sample_scale(magnitude);
    product3(h, scale, 0.5 * (a / scale + b / scale))
}

#[derive(Default)]
struct Sum {
    value: f64,
    correction: f64,
}

impl Sum {
    fn add(&mut self, term: f64) -> Result<()> {
        let next = finite(self.value + term)?;
        self.correction = finite(
            self.correction
                + if self.value.abs() >= term.abs() {
                    (self.value - next) + term
                } else {
                    (term - next) + self.value
                },
        )?;
        self.value = next;
        Ok(())
    }

    fn total(&self) -> Result<f64> {
        finite(self.value + self.correction)
    }
}

fn accumulate(intervals: impl Iterator<Item = f64>, initial: Option<f64>) -> Result<Vec<f64>> {
    if initial.is_some_and(|v| !v.is_finite()) {
        return Err(Error::InvalidInput("initial value must be finite"));
    }
    let mut sum = Sum {
        value: initial.unwrap_or(0.0),
        correction: 0.0,
    };
    let mut result = Vec::new();
    if initial.is_some() {
        result.push(sum.value);
    }
    for value in intervals {
        sum.add(value)?;
        result.push(sum.total()?);
    }
    Ok(result)
}

/// Integrates samples using `Σ (x[i+1]-x[i]) (y[i]+y[i+1])/2`.
///
/// ## Parameters
///
/// `y` contains samples; `x` supplies matching coordinates. With `None`, `dx`
/// is the constant spacing. Coordinates are used in their supplied order.
/// ## Domain
///
/// Finite inputs with 1..=1,000,001 samples. Decreasing or repeated coordinates
/// and negative or zero `dx` are accepted. A single sample integrates to zero.
/// ## Returns
///
/// Scalar integral, with units equal to sample units times coordinate units.
/// ## Errors
///
/// Invalid sizes/nonfinite inputs, mismatched shapes, or nonfinite arithmetic.
/// ## Numerical behavior
///
/// Linear time; exact for linear data up to roundoff. Power-of-two sample
/// normalization and balanced multiplication avoid common intermediate overflow
/// and premature subnormal rounding. Neumaier summation retains small residuals
/// when intervals cancel. True or running-sum overflow remains an error.
/// # Examples
///
/// ```
/// assert_eq!(sci_integrate::trapezoid(&[0.0, 1.0, 2.0], None, 1.0)?, 2.0);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy trapezoid](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.trapezoid.html).
pub fn trapezoid(y: &[f64], x: Option<&[f64]>, dx: f64) -> Result<f64> {
    let h = spacings(y, x, dx)?;
    let mut result = Sum::default();
    for (pair, h) in y.windows(2).zip(h) {
        result.add(trapezoid_interval(pair[0], pair[1], h))?;
    }
    result.total()
}

/// Returns cumulative trapezoidal integrals `I[j] = Σ_{i<j} h[i](y[i]+y[i+1])/2`.
///
/// ## Parameters
///
/// `y`, `x`, and `dx` follow [`trapezoid`]. `initial` is either `None` or
/// `Some(0.0)`, matching SciPy's accepted initial values.
/// ## Domain
///
/// Finite inputs, 1..=1,000,001 samples; arbitrary coordinate order is allowed.
/// ## Returns
///
/// Ordered vector of length `y.len()-1`, or `y.len()` with an initial zero.
/// ## Errors
///
/// Same as [`trapezoid`], plus rejection of a nonzero initial value.
/// ## Numerical behavior
///
/// Linear time and memory, reusing the scaled intervals and compensated
/// summation of [`trapezoid`]. Each prefix is rounded separately; compensation
/// is retained for subsequent prefixes. Overflowing prefixes remain errors.
/// # Examples
///
/// ```
/// assert_eq!(sci_integrate::cumulative_trapezoid(&[0.0, 2.0], None, 1.0, Some(0.0))?, vec![0.0, 1.0]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy cumulative_trapezoid](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cumulative_trapezoid.html).
pub fn cumulative_trapezoid(
    y: &[f64],
    x: Option<&[f64]>,
    dx: f64,
    initial: Option<f64>,
) -> Result<Vec<f64>> {
    if initial.is_some_and(|v| v != 0.0) {
        return Err(Error::InvalidInput("initial must be zero or absent"));
    }
    let h = spacings(y, x, dx)?;
    accumulate(
        y.windows(2)
            .zip(h)
            .map(|(p, h)| trapezoid_interval(p[0], p[1], h)),
        initial,
    )
}

fn quadratic_half(a: f64, b: f64, c: f64, h0: f64, h1: f64) -> f64 {
    let width_scale = h0.abs().max(h1.abs());
    let r = (h0 / width_scale) / (h0 / width_scale + h1 / width_scale);
    let q = r * (h0 / h1);
    let magnitude = a.abs().max(b.abs()).max(c.abs());
    if magnitude == 0.0 {
        return 0.0;
    }
    let scale = sample_scale(magnitude);
    let (a, b, c) = (a / scale, b / scale, c / scale);
    // Factoring the difference reproduces constants even for large q.
    let difference = if b == c { 0.0 } else { q * (b - c) };
    product3(
        h0,
        scale,
        ((3.0 - r) * a + (3.0 + r) * b + difference) / 6.0,
    )
}

fn validate_quadratic(h: &[f64], increasing: bool) -> Result<()> {
    if h.iter().any(|v| *v == 0.0 || (increasing && *v < 0.0))
        || h.windows(2).any(|v| v[0] + v[1] == 0.0)
    {
        return Err(Error::InvalidInput(
            "quadratic integration requires nondegenerate intervals",
        ));
    }
    Ok(())
}

/// Integrates samples with the composite quadratic Simpson rule.
///
/// ## Parameters
///
/// `y`, `x`, and `dx` follow [`trapezoid`]. Three-point quadratic interpolants
/// are integrated in pairs; an odd last interval uses the final quadratic.
/// Equal spacing gives `h(y[0]+4y[1]+y[2])/3` per pair.
/// ## Domain
///
/// Finite inputs, 1..=1,000,001 samples. Three or more points require nonzero
/// adjacent and two-interval widths. Descending coordinates are supported.
/// One sample returns zero; two samples use the trapezoidal rule.
/// ## Returns
///
/// Scalar integral, in sample units times coordinate units.
/// ## Errors
///
/// Invalid/degenerate inputs, shape mismatch, or nonfinite arithmetic.
/// ## Numerical behavior
///
/// Linear time. Exact for quadratics; also for cubics on a uniform grid with
/// an even number of intervals. Scaled samples and balanced multiplication
/// avoid common intermediate overflow and premature subnormal rounding;
/// Neumaier summation retains cancellation residuals. Uneven grids can amplify
/// roundoff. Extreme spacing ratios, weighted cancellation, true overflow,
/// and overflowing running sums remain limitations.
/// # Examples
///
/// ```
/// assert!((sci_integrate::simpson(&[0.0, 1.0, 4.0], None, 1.0)? - 8.0/3.0).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy simpson](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.simpson.html);
/// Cartwright (2012), Simpson's Rule Cumulative Integration with Irregularly-spaced Data.
pub fn simpson(y: &[f64], x: Option<&[f64]>, dx: f64) -> Result<f64> {
    let h = spacings(y, x, dx)?;
    if y.len() <= 2 {
        return trapezoid(y, x, dx);
    }
    validate_quadratic(&h, false)?;
    let mut result = Sum::default();
    for i in (0..y.len() - 2).step_by(2) {
        result.add(quadratic_half(y[i], y[i + 1], y[i + 2], h[i], h[i + 1]))?;
        result.add(quadratic_half(y[i + 2], y[i + 1], y[i], h[i + 1], h[i]))?;
    }
    if y.len().is_multiple_of(2) {
        let i = y.len() - 3;
        result.add(quadratic_half(y[i + 2], y[i + 1], y[i], h[i + 1], h[i]))?;
    }
    result.total()
}

/// Cumulatively integrates local quadratic interpolants on each interval.
///
/// ## Parameters
///
/// `y`, `x`, and `dx` follow [`trapezoid`]. Each pair uses its three-point
/// quadratic; a final unpaired interval uses the last three points.
/// `initial`, when supplied, is prepended and added to every integral.
/// ## Domain
///
/// Finite inputs with 1..=1,000,001 samples. Explicit `x` must strictly increase;
/// uniform `dx` must be positive when at least three samples are supplied.
/// Fewer than three samples use trapezoids and allow any finite spacing.
/// ## Returns
///
/// Length `y.len()-1` without `initial`, otherwise `y.len()`.
/// ## Errors
///
/// Invalid coordinates/initial value, shape mismatch, or nonfinite arithmetic.
/// ## Numerical behavior
///
/// Linear time and memory. Partial integrals differ from repeatedly calling
/// [`simpson`] on prefixes, because the next sample informs the first interval.
/// Reuses scaled quadratic intervals and compensated summation; each prefix
/// is rounded separately. The numerical limits of [`simpson`] still apply.
/// # Examples
///
/// ```
/// let integrals = sci_integrate::cumulative_simpson(&[0.0, 1.0, 4.0], None, 1.0, Some(2.0))?;
/// assert!((integrals[1] - 7.0/3.0).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy cumulative_simpson](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cumulative_simpson.html).
pub fn cumulative_simpson(
    y: &[f64],
    x: Option<&[f64]>,
    dx: f64,
    initial: Option<f64>,
) -> Result<Vec<f64>> {
    let h = spacings(y, x, dx)?;
    if x.is_some() && h.iter().any(|v| *v <= 0.0) {
        return Err(Error::InvalidInput("coordinates must strictly increase"));
    }
    if y.len() <= 2 {
        return accumulate(
            y.windows(2)
                .zip(h)
                .map(|(p, h)| trapezoid_interval(p[0], p[1], h)),
            initial,
        );
    }
    validate_quadratic(&h, true)?;
    accumulate(
        (0..h.len()).map(|i| {
            if i % 2 == 0 && i + 1 < h.len() {
                quadratic_half(y[i], y[i + 1], y[i + 2], h[i], h[i + 1])
            } else {
                quadratic_half(y[i + 1], y[i], y[i - 1], h[i], h[i - 1])
            }
        }),
        initial,
    )
}

/// Romberg integration of `2^k+1` equally spaced samples.
///
/// ## Parameters
///
/// `y` is ordered sample data with spacing `dx`. Nested trapezoidal estimates
/// are extrapolated with `R[j,k]=(4^k R[j,k-1]-R[j-1,k-1])/(4^k-1)`.
/// ## Domain
///
/// Finite `dx` and 2..=1,000,001 finite samples; `y.len()-1` must be a power
/// of two. Negative and zero spacing are permitted.
/// ## Returns
///
/// Scalar extrapolated integral; SciPy's printed tableau is omitted.
/// ## Errors
///
/// Invalid sample count, nonfinite input, or nonfinite arithmetic.
/// ## Numerical behavior
///
/// Linear sampled work and logarithmic auxiliary storage. Balanced products
/// apply spacing before summing new samples; compensated sums retain small
/// residuals. Endpoint estimates avoid forming an overflowing total width,
/// and Richardson corrections divide before subtracting when needed. True or
/// tableau overflow still errors; subnormal rounding remains. High-order
/// extrapolation may amplify noise and does not provide an error estimate.
/// # Examples
///
/// ```
/// assert!((sci_integrate::romb(&[0.0, 1.0, 4.0, 9.0, 16.0], 1.0)? - 64.0/3.0).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy romb](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.romb.html).
pub fn romb(y: &[f64], dx: f64) -> Result<f64> {
    validate_samples(y)?;
    if !dx.is_finite() {
        return Err(Error::InvalidInput("spacing must be finite"));
    }
    let n = y.len() - 1;
    if !n.is_power_of_two() {
        return Err(Error::InvalidInput("Romberg needs 2^k+1 samples"));
    }
    let scale = sample_scale(y[0].abs().max(y[n].abs()));
    let mut previous = vec![finite(product3(
        dx,
        scale,
        0.5 * n as f64 * (y[0] / scale + y[n] / scale),
    ))?];
    let mut step = n;
    while step > 1 {
        step /= 2;
        let mut sum = Sum::default();
        sum.add(0.5 * previous[0])?;
        for i in (step..n).step_by(2 * step) {
            sum.add(product3(dx, step as f64, y[i]))?;
        }
        let mut row = vec![sum.total()?];
        let mut factor = 4.0;
        for j in 1..=previous.len() {
            let delta = row[j - 1] - previous[j - 1];
            let correction = if delta.is_finite() {
                delta / (factor - 1.0)
            } else {
                row[j - 1] / (factor - 1.0) - previous[j - 1] / (factor - 1.0)
            };
            row.push(finite(row[j - 1] + correction)?);
            factor *= 4.0;
        }
        previous = row;
    }
    Ok(*previous.last().unwrap())
}

/// Fixed-order Gauss–Legendre quadrature of `∫_a^b f(x) dx`.
///
/// ## Parameters
///
/// `f` is a scalar callback; `a`, `b` are finite bounds; `n` is the quadrature
/// order. SciPy's vectorized callback/extra arguments become closure captures.
/// ## Domain
///
/// `1 <= n <= 256`. Reversed bounds negate the integral; equal bounds return
/// zero without calling `f`. Infinite bounds and nonfinite evaluations fail.
/// ## Returns
///
/// Scalar integral; SciPy's unused second tuple entry is omitted.
/// ## Errors
///
/// Invalid bounds/order, callback overflow, or quadrature backend failure.
/// ## Numerical behavior
///
/// Uses shared [`sci_special::roots_legendre`] and exactly `n` evaluations
/// for nonzero intervals. Finite bounds map to clamped convex combinations,
/// including when their difference overflows. At most 256 callback values
/// are buffered and normalized by a power of two before compensated summation;
/// balanced products restore interval and value scales afterward. This avoids
/// common intermediate overflow and preserves subnormal constant integrals.
/// Exact for degree ≤ `2n-1` polynomials up to roundoff; ill-conditioned
/// cancellation, callback rounding, true overflow and underflow remain limits.
/// No adaptive refinement or error estimate is provided.
/// # Examples
///
/// ```
/// let value = sci_integrate::fixed_quad(|x|x*x, 0.0, 1.0, 3)?;
/// assert!((value-1.0/3.0).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy fixed_quad](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.fixed_quad.html).
pub fn fixed_quad(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> Result<f64> {
    if !a.is_finite() || !b.is_finite() || !(1..=256).contains(&n) {
        return Err(Error::InvalidInput(
            "finite bounds and order in 1..=256 required",
        ));
    }
    if a == b {
        return Ok(0.0);
    }
    let (nodes, weights) = crate::special::roots_legendre(n)?;
    let width = b - a;
    let half = 0.5 * b - 0.5 * a;
    let mut values = Vec::with_capacity(n);
    let mut magnitude = 0_f64;
    for node in nodes {
        let t = 0.5 * (node + 1.0);
        let point = if width.is_finite() {
            width.mul_add(t, a)
        } else {
            // Opposite-sign endpoints: their convex combination remains finite
            // even when subtracting them overflows.
            (1.0 - t) * a + t * b
        };
        let value = finite(f(point.clamp(a.min(b), a.max(b))))?;
        magnitude = magnitude.max(value.abs());
        values.push(value);
    }
    if magnitude == 0.0 {
        return Ok(0.0_f64.copysign(width));
    }
    let scale = sample_scale(magnitude);
    let mut sum = Sum::default();
    for (value, weight) in values.into_iter().zip(weights) {
        sum.add(weight * (value / scale))?;
    }
    let weighted = sum.total()?;
    finite(if width.is_finite() {
        product3(width, scale, 0.5 * weighted)
    } else {
        product3(half, scale, weighted)
    })
}

/// Computes closed Newton–Cotes weights and the leading error coefficient.
///
/// ## Parameters
///
/// `rn` contains `N+1` strictly increasing positions, starting at zero and
/// ending at `N`, for `1 <= N <= 12`. Use `0..=N` converted to `f64` for the
/// uniform rule; SciPy's integer shorthand and `equal` flag are omitted.
/// `∫ f ≈ h Σ w[i] f(x0+h rn[i]) + B h^(p+1) f^(p)(ξ)`.
/// ## Domain
///
/// Only finite, distinct nodes in the closed normalized interval are accepted.
/// ## Returns
///
/// `(weights, B)` in input order. `p=N+2` for even uniform `N`, otherwise `N+1`.
/// ## Errors
///
/// Invalid nodes/order, singular backend solve, or nonfinite arithmetic.
/// ## Numerical behavior
///
/// A centered Vandermonde system is solved by private nalgebra LU. Cubic work
/// is bounded by 13 nodes. Clustered nodes are ill-conditioned and may lose
/// accuracy; low orders are recommended. No condition estimate is returned.
/// # Examples
///
/// ```
/// let (weights, _) = sci_integrate::newton_cotes(&[0.0, 1.0, 2.0])?;
/// assert!((weights[1]-4.0/3.0).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy newton_cotes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.newton_cotes.html).
pub fn newton_cotes(rn: &[f64]) -> Result<(Vec<f64>, f64)> {
    if !(2..=13).contains(&rn.len()) {
        return Err(Error::InvalidInput("Newton-Cotes needs 2..=13 nodes"));
    }
    finite_slice(rn)?;
    let n = rn.len() - 1;
    if rn[0] != 0.0 || rn[n] != n as f64 || rn.windows(2).any(|p| p[1] <= p[0]) {
        return Err(Error::InvalidInput("nodes must increase from zero to N"));
    }
    let matrix = nalgebra::DMatrix::from_fn(n + 1, n + 1, |i, j| {
        (2.0 * rn[j] / n as f64 - 1.0).powi(i as i32)
    });
    let moments = nalgebra::DVector::from_iterator(
        n + 1,
        (0..=n).map(|i| {
            if i % 2 == 0 {
                n as f64 / (i + 1) as f64
            } else {
                0.0
            }
        }),
    );
    let weights = matrix
        .lu()
        .solve(&moments)
        .ok_or(Error::NumericalFailure("singular Newton-Cotes system"))?;
    let weights: Vec<f64> = weights.iter().copied().collect();
    finite_slice(&weights)?;
    let uniform = rn.iter().enumerate().all(|(i, x)| *x == i as f64);
    let power = if uniform && n.is_multiple_of(2) {
        n + 2
    } else {
        n + 1
    };
    let moment: f64 = rn
        .iter()
        .zip(&weights)
        .map(|(x, w)| w * (x / n as f64).powi(power as i32))
        .sum();
    let factorial: f64 = (1..=power).map(|i| i as f64).product();
    let error =
        (n as f64 / (power + 1) as f64 - moment) * (n as f64).powi(power as i32) / factorial;
    Ok((weights, finite(error)?))
}
