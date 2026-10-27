use crate::error::{finite, finite_slice};
use crate::{Error, Result};
use sci_core::Float;

const MAX_SAMPLES: usize = 1_000_001;

#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
fn validate_samples<T: Float>(y: &[T]) -> Result<()> {
    if y.is_empty() || y.len() > MAX_SAMPLES {
        return Err(Error::InvalidInput("sample count must be in 1..=1000001"));
    }
    finite_slice(y)
}

fn spacings<T: Float>(y: &[T], x: Option<&[T]>, dx: T) -> Result<Vec<T>> {
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
fn product3<T: Float>(a: T, b: T, c: T) -> T {
    // Orders by magnitude, with NaN last. Ties cannot change the product.
    fn smaller<T: Float>(a: T, b: T) -> bool {
        a.abs() < b.abs() || (b.is_nan() && !a.is_nan())
    }
    let (mut lo, mut mid, mut hi) = (a, b, c);
    if smaller(mid, lo) {
        std::mem::swap(&mut lo, &mut mid);
    }
    if smaller(hi, mid) {
        std::mem::swap(&mut mid, &mut hi);
    }
    if smaller(mid, lo) {
        std::mem::swap(&mut lo, &mut mid);
    }
    (lo * hi) * mid
}

#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
fn sample_scale<T: Float>(magnitude: T) -> T {
    // A power of two makes normalization exact, including for subnormal data.
    // Types no wider than binary64 (binary32, binary64) convert exactly to
    // binary64, so masking its exponent yields the same power of two without
    // decoding; the size test is resolved at compile time.
    if std::mem::size_of::<T>() <= std::mem::size_of::<f64>()
        && let Some(power) = magnitude
            .to_f64()
            .and_then(|wide| T::from(f64::from_bits(wide.to_bits() & 0x7ff0_0000_0000_0000)))
    {
        return power.max(T::min_positive_value());
    }
    let (mantissa, exponent, _) = magnitude.integer_decode();
    let leading = 63 - mantissa.max(1).leading_zeros();
    2.0.powi(i32::from(exponent) + leading as i32)
        .max(T::min_positive_value())
}

#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
fn trapezoid_interval<T: Float>(a: T, b: T, h: T) -> T {
    let magnitude = a.abs().max(b.abs());
    if magnitude == 0.0 {
        return h * (a + b);
    }
    let scale = sample_scale(magnitude);
    product3(h, scale, 0.5 * (a / scale + b / scale))
}

struct Sum<T> {
    value: T,
    correction: T,
}

impl<T: Float> Default for Sum<T> {
    fn default() -> Self {
        Self {
            value: T::zero(),
            correction: T::zero(),
        }
    }
}

impl<T: Float> Sum<T> {
    fn add(&mut self, term: T) -> Result<()> {
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

    fn total(&self) -> Result<T> {
        finite(self.value + self.correction)
    }
}

#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
fn accumulate<T: Float>(intervals: impl Iterator<Item = T>, initial: Option<T>) -> Result<Vec<T>> {
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
pub fn trapezoid<T: Float>(y: &[T], x: Option<&[T]>, dx: T) -> Result<T> {
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
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn cumulative_trapezoid<T: Float>(
    y: &[T],
    x: Option<&[T]>,
    dx: T,
    initial: Option<T>,
) -> Result<Vec<T>> {
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

#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
fn quadratic_half<T: Float>(a: T, b: T, c: T, h0: T, h1: T) -> T {
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

#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
fn validate_quadratic<T: Float>(h: &[T], increasing: bool) -> Result<()> {
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
/// assert!((sci_integrate::simpson(&[0.0_f64, 1.0, 4.0], None, 1.0)? - 8.0/3.0).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy simpson](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.simpson.html);
/// Cartwright (2012), Simpson's Rule Cumulative Integration with Irregularly-spaced Data.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn simpson<T: Float>(y: &[T], x: Option<&[T]>, dx: T) -> Result<T> {
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
/// let integrals = sci_integrate::cumulative_simpson(&[0.0_f64, 1.0, 4.0], None, 1.0, Some(2.0))?;
/// assert!((integrals[1] - 7.0/3.0).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy cumulative_simpson](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cumulative_simpson.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn cumulative_simpson<T: Float>(
    y: &[T],
    x: Option<&[T]>,
    dx: T,
    initial: Option<T>,
) -> Result<Vec<T>> {
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
/// assert!((sci_integrate::romb(&[0.0_f64, 1.0, 4.0, 9.0, 16.0], 1.0)? - 64.0/3.0).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy romb](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.romb.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn romb<T: Float>(y: &[T], dx: T) -> Result<T> {
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
        0.5 * T::from(n).unwrap() * (y[0] / scale + y[n] / scale),
    ))?];
    let mut step = n;
    while step > 1 {
        step /= 2;
        let mut sum = Sum::default();
        sum.add(0.5 * previous[0])?;
        for i in (step..n).step_by(2 * step) {
            sum.add(product3(dx, T::from(step).unwrap(), y[i]))?;
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
            factor = factor * (4.0);
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
/// Uses shared [`sci_special::roots_legendre`] to generate binary64 nodes and
/// weights, rounds them to the selected type, and performs exactly `n` evaluations
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
/// let value = sci_integrate::fixed_quad(|x|x*x, 0.0_f64, 1.0, 3)?;
/// assert!((value-1.0/3.0).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy fixed_quad](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.fixed_quad.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn fixed_quad<T: Float>(f: impl Fn(T) -> T, a: T, b: T, n: usize) -> Result<T> {
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
        let t = 0.5 * (T::from(node).unwrap() + 1.0);
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
        sum.add(T::from(weight).unwrap() * (value / scale))?;
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
/// Uniform rules use exact rational tabulations through order 12, rounded to
/// binary64. Other nodes use a centered Vandermonde system solved by private
/// nalgebra LU, with a subtracted moment for the error coefficient. Cubic work
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
    if rn.iter().enumerate().all(|(i, x)| *x == i as f64) {
        // Exact rational closed rules avoid a Vandermonde solve and the
        // cancellation in the leading moment defect for uniform nodes.
        let (factor, numerators, error): (f64, &[f64], f64) = match n {
            1 => (1.0 / 2.0, &[1.0, 1.0], -1.0 / 12.0),
            2 => (1.0 / 3.0, &[1.0, 4.0, 1.0], -1.0 / 90.0),
            3 => (3.0 / 8.0, &[1.0, 3.0, 3.0, 1.0], -3.0 / 80.0),
            4 => (2.0 / 45.0, &[7.0, 32.0, 12.0, 32.0, 7.0], -8.0 / 945.0),
            5 => (
                5.0 / 288.0,
                &[19.0, 75.0, 50.0, 50.0, 75.0, 19.0],
                -275.0 / 12096.0,
            ),
            6 => (
                1.0 / 140.0,
                &[41.0, 216.0, 27.0, 272.0, 27.0, 216.0, 41.0],
                -9.0 / 1400.0,
            ),
            7 => (
                7.0 / 17280.0,
                &[751.0, 3577.0, 1323.0, 2989.0, 2989.0, 1323.0, 3577.0, 751.0],
                -8183.0 / 518400.0,
            ),
            8 => (
                4.0 / 14175.0,
                &[
                    989.0, 5888.0, -928.0, 10496.0, -4540.0, 10496.0, -928.0, 5888.0, 989.0,
                ],
                -2368.0 / 467775.0,
            ),
            9 => (
                9.0 / 89600.0,
                &[
                    2857.0, 15741.0, 1080.0, 19344.0, 5778.0, 5778.0, 19344.0, 1080.0, 15741.0,
                    2857.0,
                ],
                -4671.0 / 394240.0,
            ),
            10 => (
                5.0 / 299376.0,
                &[
                    16067.0, 106300.0, -48525.0, 272400.0, -260550.0, 427368.0, -260550.0,
                    272400.0, -48525.0, 106300.0, 16067.0,
                ],
                -673175.0 / 163459296.0,
            ),
            11 => (
                11.0 / 87091200.0,
                &[
                    2171465.0, 13486539.0, -3237113.0, 25226685.0, -9595542.0, 15493566.0,
                    15493566.0, -9595542.0, 25226685.0, -3237113.0, 13486539.0, 2171465.0,
                ],
                -2224234463.0 / 237758976000.0,
            ),
            12 => (
                1.0 / 5255250.0,
                &[
                    1364651.0,
                    9903168.0,
                    -7587864.0,
                    35725120.0,
                    -51491295.0,
                    87516288.0,
                    -87797136.0,
                    87516288.0,
                    -51491295.0,
                    35725120.0,
                    -7587864.0,
                    9903168.0,
                    1364651.0,
                ],
                -3012.0 / 875875.0,
            ),
            _ => unreachable!("validated Newton–Cotes order"),
        };
        return Ok((numerators.iter().map(|x| factor * x).collect(), error));
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
    let power = n + 1;
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
