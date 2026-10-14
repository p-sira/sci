use crate::error::{finite, finite_slice};
use crate::{Error, Result};

fn validate(x: &[f64], y: &[f64], query: &[f64], limit: usize) -> Result<()> {
    if x.len() != y.len() {
        return Err(Error::DimensionMismatch);
    }
    if x.is_empty() || x.len() > limit || query.len() > 1_000_000 / x.len() {
        return Err(Error::InvalidInput(
            "interpolation size or work limit exceeded",
        ));
    }
    finite_slice(x)?;
    finite_slice(y)?;
    finite_slice(query)
}

fn newton(x: &[f64], y: &[f64]) -> Result<Vec<f64>> {
    let n = x.len();
    let mut starts = vec![0; n];
    for i in 1..n {
        starts[i] = if x[i] == x[i - 1] {
            starts[i - 1]
        } else {
            if x[..i].contains(&x[i]) {
                return Err(Error::InvalidInput("repeated nodes must be contiguous"));
            }
            i
        };
    }
    let mut table: Vec<f64> = starts.iter().map(|i| y[*i]).collect();
    let mut coefficients = vec![table[0]];
    let mut factorial = 1.0;
    for j in 1..n {
        factorial *= j as f64;
        for i in (j..n).rev() {
            table[i] = if x[i] == x[i - j] {
                y[starts[i] + j] / factorial
            } else {
                finite((table[i] - table[i - 1]) / finite(x[i] - x[i - j])?)?
            };
        }
        coefficients.push(table[j]);
    }
    Ok(coefficients)
}

/// Evaluates the polynomial interpolant with barycentric weights.
///
/// ## Parameters
///
/// `xi`, `yi` are matching node/value slices; `x` contains query coordinates.
/// `P(t)=Σ w[i]y[i]/(t-xi[i]) / Σ w[i]/(t-xi[i])`, where
/// `w[i]=1/Π_{j≠i}(xi[i]-xi[j])`. Only SciPy's derivative order zero is exposed.
/// ## Domain
///
/// 1..=256 distinct finite nodes in arbitrary order, finite values and queries.
/// Queries may lie outside the node interval; an empty query returns no values.
/// ## Returns
///
/// Values in query order; interpolation nodes reproduce `yi` exactly.
/// ## Errors
///
/// Shape mismatch, duplicate/nonfinite nodes, overflow, or more than 1,000,000
/// node/query products. Degenerate floating-point weights return an error.
/// ## Numerical behavior
///
/// Quadratic weight construction and linear work per query. Log-scaled weights
/// avoid product overflow. Evaluation scales distances and sample values to
/// avoid reciprocal and numerator overflow, reusing linear scratch storage.
/// Subnormal terms may underflow. High-degree interpolation, clustered nodes,
/// and extrapolation can be ill-conditioned; no accuracy estimate is provided.
/// # Examples
///
/// ```
/// let y=sci_interpolate::barycentric_interpolate(&[0.0,1.0,2.0], &[0.0,1.0,4.0], &[1.5])?;
/// assert!((y[0]-2.25).abs()<1e-14);
/// # Ok::<(),sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy barycentric_interpolate](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.barycentric_interpolate.html).
pub fn barycentric_interpolate(xi: &[f64], yi: &[f64], x: &[f64]) -> Result<Vec<f64>> {
    validate(xi, yi, x, 256)?;
    let mut logs = vec![0.0; xi.len()];
    let mut signs = vec![1.0; xi.len()];
    for i in 0..xi.len() {
        for j in 0..xi.len() {
            if i == j {
                continue;
            }
            let d = finite(xi[i] - xi[j])?;
            if d == 0.0 {
                return Err(Error::InvalidInput("barycentric nodes must be distinct"));
            }
            logs[i] -= d.abs().ln();
            signs[i] *= d.signum();
        }
    }
    let max = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = logs
        .iter()
        .zip(signs)
        .map(|(v, s)| s * (v - max).exp())
        .collect();
    if weights.contains(&0.0) {
        return Err(Error::NumericalFailure("barycentric weights underflowed"));
    }
    let value_scale = yi.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let mut terms = vec![0.0; xi.len()];
    let mut result = Vec::with_capacity(x.len());
    for t in x {
        if let Some(i) = xi.iter().position(|node| node == t) {
            result.push(yi[i]);
            continue;
        }
        let mut distance = f64::INFINITY;
        for (term, node) in terms.iter_mut().zip(xi) {
            *term = finite(t - node)?;
            distance = distance.min(term.abs());
        }
        // Multiplying both sums by the nearest distance avoids forming 1/d
        // when d is subnormal. Every distance ratio has magnitude at most one.
        for (term, weight) in terms.iter_mut().zip(&weights) {
            *term = weight * (distance / *term);
        }
        let scale = terms.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let denominator: f64 = terms.iter().map(|v| v / scale).sum();
        let numerator: f64 = if value_scale == 0.0 {
            0.0
        } else {
            terms
                .iter()
                .zip(yi)
                .map(|(v, y)| (v / scale) * (y / value_scale))
                .sum()
        };
        result.push(finite((numerator / denominator) * value_scale)?);
    }
    Ok(result)
}

/// Evaluates the Krogh/Hermite polynomial satisfying node derivatives.
///
/// ## Parameters
///
/// `xi`, `yi` have equal length. A node repeated `r` consecutive times supplies
/// `f, f', ..., f^(r-1)` in `yi`, without factorial scaling. Distinct nodes may
/// appear in arbitrary order. `x` contains query coordinates. SciPy's derivative
/// output option is omitted; the interpolant itself is evaluated.
/// ## Domain
///
/// 1..=32 finite entries and finite queries. Repeated nodes must be contiguous.
/// Extrapolation is supported; an empty query gives an empty vector.
/// ## Returns
///
/// Values in query order.
/// ## Errors
///
/// Invalid/ noncontiguous nodes, shape mismatch, nonfinite arithmetic, or more
/// than 1,000,000 node/query products.
/// ## Numerical behavior
///
/// Confluent divided differences take quadratic work; nested Newton evaluation
/// takes linear work per query. High degrees and clustered nodes lose accuracy.
/// # Examples
///
/// ```
/// let y=sci_interpolate::krogh_interpolate(&[0.0,0.0,1.0], &[0.0,0.0,1.0], &[0.5])?;
/// assert_eq!(y,vec![0.25]);
/// # Ok::<(),sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy krogh_interpolate](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.krogh_interpolate.html).
pub fn krogh_interpolate(xi: &[f64], yi: &[f64], x: &[f64]) -> Result<Vec<f64>> {
    validate(xi, yi, x, 32)?;
    let c = newton(xi, yi)?;
    x.iter()
        .map(|t| {
            let mut value = c[c.len() - 1];
            for j in (0..c.len() - 1).rev() {
                value = finite(c[j] + finite(t - xi[j])? * value)?;
            }
            Ok(value)
        })
        .collect()
}

/// Constructs the Lagrange polynomial `P(t)=Σ y[i] Π_{j≠i}(t-x[j])/(x[i]-x[j])`.
///
/// ## Parameters
///
/// `x`, `y` are matching scalar nodes/values in arbitrary order.
/// ## Domain
///
/// 1..=32 distinct finite nodes with finite values.
/// ## Returns
///
/// Descending coefficients of degree at most `x.len()-1`, retaining leading
/// zeros. This replaces SciPy's callable `poly1d` object.
/// ## Errors
///
/// Shape mismatch, duplicate/nonfinite nodes, invalid size, or overflow.
/// ## Numerical behavior
///
/// Quadratic divided differences and coefficient conversion. Monomial
/// coefficients can be inaccurate at high degrees; prefer barycentric evaluation.
/// # Examples
///
/// ```
/// assert_eq!(sci_interpolate::lagrange(&[0.0,1.0,2.0], &[0.0,1.0,4.0])?,vec![1.0,0.0,0.0]);
/// # Ok::<(),sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy lagrange](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.lagrange.html).
pub fn lagrange(x: &[f64], y: &[f64]) -> Result<Vec<f64>> {
    validate(x, y, &[], 32)?;
    for i in 0..x.len() {
        if x[..i].contains(&x[i]) {
            return Err(Error::InvalidInput("Lagrange nodes must be distinct"));
        }
    }
    let c = newton(x, y)?;
    let mut polynomial = vec![c[c.len() - 1]];
    for i in (0..c.len() - 1).rev() {
        let mut next = vec![0.0; polynomial.len() + 1];
        for (j, coefficient) in polynomial.iter().enumerate() {
            next[j] = finite(next[j] - x[i] * coefficient)?;
            next[j + 1] = finite(next[j + 1] + coefficient)?;
        }
        next[0] = finite(next[0] + c[i])?;
        polynomial = next;
    }
    polynomial.reverse();
    Ok(polynomial)
}

fn endpoint(h0: f64, h1: f64, d0: f64, d1: f64) -> Result<f64> {
    let scale = h0.max(h1);
    let a = h0 / scale;
    let b = h1 / scale;
    let slope_scale = d0.abs().max(d1.abs());
    if slope_scale == 0.0 {
        return Ok(0.0);
    }
    let normalized = d0 / slope_scale;
    let derivative = ((2.0 * a + b) / (a + b)) * normalized - (a / (a + b)) * (d1 / slope_scale);
    if derivative.signum() != d0.signum() {
        Ok(0.0)
    } else if d0.signum() != d1.signum() && derivative.abs() / 3.0 > normalized.abs() {
        finite(3.0 * d0)
    } else {
        finite(derivative * slope_scale)
    }
}

/// Evaluates the shape-preserving piecewise cubic Hermite interpolant (PCHIP).
///
/// ## Parameters
///
/// `xi`, `yi` are node/value slices; `x` is the query slice. Each interval uses
/// `P(t)=(2u³-3u²+1)y[i]+(u³-2u²+u)h d[i]+(-2u³+3u²)y[i+1]+(u³-u²)h d[i+1]`,
/// with `u=(t-xi[i])/h`. Interior derivatives are weighted harmonic secant means
/// or zero at a change in slope. Only derivative order zero is exposed.
/// ## Domain
///
/// 2..=1,000,000 finite, strictly increasing nodes with finite values and queries.
/// Queries outside the range extrapolate the first or last cubic, as in SciPy.
/// ## Returns
///
/// Values in query order; nodes reproduce `yi` exactly.
/// ## Errors
///
/// Shape mismatch, invalid coordinates, nonfinite arithmetic, or more than
/// 1,000,000 queries. No NaN or infinity inputs are accepted.
/// Sample differences and secant/endpoint derivatives must be representable,
/// even when queried values would be finite.
/// ## Numerical behavior
///
/// Linear construction, logarithmic interval lookup per query. The interpolant
/// preserves monotonicity within the data range and is continuously differentiable.
/// Endpoint derivatives use scaled slopes to avoid intermediate overflow.
/// Node matches treat both signs of zero as equal and return the stored value
/// exactly. Extrapolation can lose the shape guarantee.
/// # Examples
///
/// ```
/// assert_eq!(sci_interpolate::pchip_interpolate(&[0.0,1.0], &[0.0,2.0], &[0.5])?,vec![1.0]);
/// # Ok::<(),sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy PchipInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html);
/// Fritsch and Butland (1984), SIAM J. Sci. Comput. 5, 300–304.
pub fn pchip_interpolate(xi: &[f64], yi: &[f64], x: &[f64]) -> Result<Vec<f64>> {
    validate(xi, yi, &[], 1_000_000)?;
    if xi.len() < 2 || x.len() > 1_000_000 {
        return Err(Error::InvalidInput(
            "PCHIP needs at least two nodes and bounded queries",
        ));
    }
    finite_slice(x)?;
    let h: Vec<f64> = xi
        .windows(2)
        .map(|p| finite(p[1] - p[0]))
        .collect::<Result<_>>()?;
    if h.iter().any(|v| *v <= 0.0) {
        return Err(Error::InvalidInput("PCHIP nodes must strictly increase"));
    }
    let slopes: Vec<f64> = yi
        .windows(2)
        .zip(&h)
        .map(|(p, h)| finite((p[1] - p[0]) / h))
        .collect::<Result<_>>()?;
    let mut derivatives = vec![0.0; xi.len()];
    let last = xi.len() - 1;
    if xi.len() == 2 {
        derivatives.fill(slopes[0]);
    } else {
        derivatives[0] = endpoint(h[0], h[1], slopes[0], slopes[1])?;
        derivatives[last] = endpoint(h[last - 1], h[last - 2], slopes[last - 1], slopes[last - 2])?;
        for i in 1..last {
            if slopes[i - 1] != 0.0
                && slopes[i] != 0.0
                && slopes[i - 1].signum() == slopes[i].signum()
            {
                let scale = h[i].max(h[i - 1]);
                let next = h[i] / scale;
                let previous = h[i - 1] / scale;
                let fraction = (2.0 * next + previous) / (3.0 * (next + previous));
                let small = slopes[i - 1].abs().min(slopes[i].abs());
                let denominator = fraction * (small / slopes[i - 1].abs())
                    + (1.0 - fraction) * (small / slopes[i].abs());
                derivatives[i] = finite(slopes[i].signum() * (small / denominator))?;
            }
        }
    }
    x.iter()
        .map(|t| {
            let i = xi.partition_point(|node| node <= t).saturating_sub(1);
            if xi[i] == *t {
                return Ok(yi[i]);
            }
            let i = i.min(last - 1);
            let u = finite((t - xi[i]) / h[i])?;
            let u2 = u * u;
            let u3 = u2 * u;
            finite(
                (2.0 * u3 - 3.0 * u2 + 1.0) * yi[i]
                    + (u3 - 2.0 * u2 + u) * h[i] * derivatives[i]
                    + (-2.0 * u3 + 3.0 * u2) * yi[i + 1]
                    + (u3 - u2) * h[i] * derivatives[i + 1],
            )
        })
        .collect()
}

/// Constructs a Padé rational approximant from Taylor coefficients.
///
/// ## Parameters
///
/// `an[k]` is the coefficient of `x^k` (ascending input); `m` is denominator
/// degree; `n` is numerator degree, defaulting to `an.len()-1-m`.
/// `P(x)/Q(x)=Σ an[k]x^k+O(x^(m+n+1))`, normalized to `Q(0)=1`.
/// ## Domain
///
/// 1..=65 finite input coefficients and `m+n < an.len()`; excess coefficients
/// are ignored when `n` is explicit. Singular coefficient systems fail.
/// ## Returns
///
/// `(P, Q)` as descending coefficient vectors, replacing SciPy's `poly1d`.
/// ## Errors
///
/// Invalid degrees, nonfinite input/output, or a singular backend solve.
/// ## Numerical behavior
///
/// Private nalgebra LU solves the denominator system; work is cubic and bounded
/// by degree 64. Ill-conditioned systems may yield inaccurate coefficients.
/// # Examples
///
/// ```
/// let (p,q)=sci_interpolate::pade(&[1.0,1.0,0.5],1,None)?;
/// assert_eq!(p,vec![0.5,1.0]);
/// assert_eq!(q,vec![-0.5,1.0]);
/// # Ok::<(),sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy pade](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.pade.html).
pub fn pade(an: &[f64], m: usize, n: Option<usize>) -> Result<(Vec<f64>, Vec<f64>)> {
    if an.is_empty() || an.len() > 65 || m >= an.len() {
        return Err(Error::InvalidInput(
            "invalid Pade coefficient count or degree",
        ));
    }
    finite_slice(an)?;
    let n = n.unwrap_or(an.len() - 1 - m);
    if n >= an.len() - m {
        return Err(Error::InvalidInput(
            "Pade degrees exceed available coefficients",
        ));
    }
    let mut q = vec![1.0];
    if m > 0 {
        let a = nalgebra::DMatrix::from_fn(m, m, |i, j| {
            let k = n + 1 + i;
            if k > j { an[k - j - 1] } else { 0.0 }
        });
        let b = nalgebra::DVector::from_iterator(m, (0..m).map(|i| -an[n + 1 + i]));
        let solution = a
            .lu()
            .solve(&b)
            .ok_or(Error::NumericalFailure("singular Pade system"))?;
        q.extend(solution.iter().copied());
    }
    let mut p: Vec<f64> = (0..=n)
        .map(|k| (0..=m.min(k)).map(|j| q[j] * an[k - j]).sum())
        .collect();
    finite_slice(&p)?;
    finite_slice(&q)?;
    p.reverse();
    q.reverse();
    Ok((p, q))
}
