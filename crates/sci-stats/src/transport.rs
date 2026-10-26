use super::*;

pub(super) fn points_check(points: &[&[f64]], minimum: usize, maximum: usize) -> Result<usize> {
    if points.len() < minimum || points.len() > maximum {
        return Err(Error::InvalidInput("invalid statistical point count"));
    }
    let dimension = points[0].len();
    if !(1..=64).contains(&dimension) {
        return Err(Error::InvalidInput(
            "statistical point dimension must be 1..=64",
        ));
    }
    for row in points {
        if row.len() != dimension {
            return Err(Error::DimensionMismatch);
        }
        is_finite_slice(row)?;
    }
    Ok(dimension)
}
pub(super) fn normalized_points(points: &[&[f64]]) -> Result<(f64, Vec<Vec<f64>>)> {
    let dimension = points[0].len();
    let coordinates = (0..dimension)
        .map(|j| centered_scaled(&points.iter().map(|p| p[j]).collect::<Vec<_>>()))
        .collect::<Result<Vec<_>>>()?;
    let scale = coordinates.iter().map(|c| c.0).fold(0_f64, f64::max);
    let mut normalized = vec![vec![0.; dimension]; points.len()];
    if scale > 0. {
        for (j, (s, values)) in coordinates.iter().enumerate() {
            for (i, &value) in values.iter().enumerate() {
                normalized[i][j] = value * (s / scale);
            }
        }
    }
    Ok((scale, normalized))
}
pub(super) fn euclidean(first: &[f64], second: &[f64]) -> f64 {
    first
        .iter()
        .zip(second)
        .fold(0_f64, |norm, (a, b)| norm.hypot(a - b))
}
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Multidimensional Wasserstein-1 distance `min_π Σ_ij π_ij ||u_i−v_j||₂`.
///
/// ## Parameters and conventions
///
/// `first`,`second` contain one point per row, with equal dimension d and
/// uniform empirical masses 1/n and 1/m. Transport π has these row/column
/// marginals. Replication to L=lcm(n,m) equal-mass points reduces the problem
/// exactly to a minimum-cost linear assignment. This supports SciPy's unweighted
/// Euclidean overload; arbitrary weights and other metrics are unsupported.
///
/// ## Domain and special values
///
/// Nonempty finite rectangular point samples, 1..=64 dimensions, n,m≤256 and
/// L≤256. Duplicate points are accepted. NaN/infinity coordinates are errors.
/// Identical zero-spread point clouds return zero.
///
/// ## Returns
///
/// Nonnegative distance in common coordinate units. Coordinates of different
/// physical dimensions require caller-supplied consistent scaling.
///
/// ## Errors
///
/// Invalid domains/shapes, exceeded replicated-assignment limits, backend
/// assignment failure or unrepresentable final distance return errors.
///
/// ## Numerical behavior
///
/// Private lapjv Jonker–Volgenant backend, O(L³+L²*d) work and O(L²+(n+m)*d)
/// memory, with L³≤16777216. Native shifted/scaled coordinates and hypot norms
/// avoid large-offset cancellation and intermediate distance overflow. Assignment
/// optimality is subject to binary64 cost rounding; tiny coordinates relative
/// to the largest dimension can underflow. No Python or external solver process.
///
/// ## Examples
///
/// ```
/// let a: &[&[f64]] = &[&[0.,0.],&[1.,0.]];
/// let b: &[&[f64]] = &[&[0.,1.],&[1.,1.]];
/// assert_eq!(sci_stats::wasserstein_distance_nd(a,b)?,1.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy wasserstein_distance_nd](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance_nd.html),
/// [Jonker–Volgenant (1987)](https://doi.org/10.1007/BF02278710).
pub fn wasserstein_distance_nd(first: &[&[f64]], second: &[&[f64]]) -> Result<f64> {
    let dimension = points_check(first, 1, 256)?;
    if points_check(second, 1, 256)? != dimension {
        return Err(Error::DimensionMismatch);
    }
    let n = first.len();
    let m = second.len();
    let count = n / gcd(n, m) * m;
    if count > 256 {
        return Err(Error::InvalidInput(
            "Wasserstein replicated assignment size exceeds 256",
        ));
    }
    let pooled = first.iter().chain(second).copied().collect::<Vec<_>>();
    let (scale, points) = normalized_points(&pooled)?;
    if scale == 0. {
        return Ok(0.);
    }
    if count == 1 {
        return match euclidean(&points[0], &points[1]) * scale {
            value if value.is_finite() => Ok(value),
            _ => Err(crate::Error::NonFiniteEvaluation),
        };
    }
    let costs = ndarray::Array2::from_shape_fn((count, count), |(i, j)| {
        euclidean(&points[i / (count / n)], &points[n + j / (count / m)])
    });
    let (assignment, _) = lapjv::lapjv(&costs)
        .map_err(|_| Error::NumericalFailure("transport assignment backend failed"))?;
    if assignment.len() != count || assignment.iter().any(|&j| j >= count) {
        return Err(Error::NumericalFailure(
            "transport backend returned invalid assignment",
        ));
    }
    let cost = assignment
        .iter()
        .enumerate()
        .map(|(i, &j)| costs[(i, j)] / count as f64)
        .sum::<f64>();
    match cost * scale {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }
}
