//! Uniformity measures and scaling for supplied quasi-Monte Carlo samples.
use super::*;

/// Quadrature discrepancy convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Discrepancy {
    /// Centered squared L2 discrepancy.
    Centered,
    /// Wrap-around squared L2 discrepancy.
    WrapAround,
    /// Mixture squared L2 discrepancy.
    Mixture,
    /// Anchored L2-star discrepancy, with square root.
    L2Star,
}
/// Geometric uniformity convention (Euclidean distances).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeometricDiscrepancy {
    /// Minimum distance between any pair of sample points.
    MinimumDistance,
    /// Mean edge length in a Euclidean minimum spanning tree.
    MinimumSpanningTree,
}
fn dimensions(x: &[Vec<f64>], minimum: usize) -> Result<usize> {
    if x.len() < minimum || x.len() > MAX_SAMPLES {
        return Err(Error::InvalidInput("invalid point count"));
    }
    let d = x.first().map_or(0, Vec::len);
    if d == 0 || d > 64 || x.iter().any(|r| r.len() != d) {
        return Err(Error::DimensionMismatch);
    }
    if x.len() * d > MAX_SAMPLES {
        return Err(Error::InvalidInput("sample size exceeds limit"));
    }
    for row in x {
        sample(row, 1)?;
    }
    Ok(d)
}
fn unit(x: &[Vec<f64>], minimum: usize) -> Result<usize> {
    let d = dimensions(x, minimum)?;
    if x.iter().flatten().any(|v| !(0. ..=1.).contains(v)) {
        return Err(Error::InvalidInput("sample must lie in unit cube"));
    }
    Ok(d)
}
fn quadratic_work(n: usize, d: usize) -> Result<()> {
    if n.checked_mul(n)
        .and_then(|v| v.checked_mul(d))
        .is_none_or(|v| v > 16_777_216)
    {
        return Err(Error::InvalidInput("pairwise work exceeds 16777216"));
    }
    Ok(())
}
fn single(x: &[f64], method: Discrepancy) -> f64 {
    x.iter()
        .map(|x| {
            let a = (x - 0.5).abs();
            match method {
                Discrepancy::Centered => 1. + 0.5 * a - 0.5 * a * a,
                Discrepancy::WrapAround => 0.,
                Discrepancy::Mixture => 5. / 3. - 0.25 * a - 0.25 * a * a,
                Discrepancy::L2Star => 1. - x * x,
            }
        })
        .product()
}
fn pair(x: &[f64], y: &[f64], method: Discrepancy) -> f64 {
    x.iter()
        .zip(y)
        .map(|(x, y)| {
            let a = (x - 0.5).abs();
            let b = (y - 0.5).abs();
            let c = (x - y).abs();
            match method {
                Discrepancy::Centered => 1. + 0.5 * a + 0.5 * b - 0.5 * c,
                Discrepancy::WrapAround => 1.5 - c + c * c,
                Discrepancy::Mixture => 15. / 8. - 0.25 * a - 0.25 * b - 0.75 * c + 0.5 * c * c,
                Discrepancy::L2Star => 1. - x.max(*y),
            }
        })
        .product()
}
/// L2 quadrature discrepancy `C_d−2/n Σ A(x_i)+1/n² Σ B(x_i,x_j)`.
///
/// ## Parameters
///
/// Rows are points, columns are coordinates. CD uses C=(13/12)^d, A=Π(1+|x−1/2|/2−|x−1/2|²/2), B=Π(1+(|x−1/2|+|y−1/2|−|x−y|)/2).
/// WD uses C=−(4/3)^d, no A term, B=Π(3/2−|x−y|+|x−y|²).
/// MD uses C=(19/12)^d, A=Π(5/3−|x−1/2|/4−|x−1/2|²/4), B=Π(15/8−(|x−1/2|+|y−1/2|)/4−3|x−y|/4+|x−y|²/2).
/// L2-star takes sqrt(3^(−d)−2^(1−d)/n*ΣΠ(1−x²)+ΣΠ(1−max(x,y))/n²).
/// Iterative mode uses n+1 in denominators, for update_discrepancy.
///
/// ## Domain
///
/// Nonempty finite rectangular points in \[0,1\]^d, d≤64.
///
/// ## Returns
///
/// Dimensionless discrepancy; smaller values indicate better uniformity.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// O(n²d) direct products, limited to 16,777,216 coordinate-pair operations. Cancellation increases with dimension; no parallel workers.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// use sci_stats::qmc::{discrepancy, Discrepancy};
/// assert!(discrepancy(&[vec![0.25],vec![0.75]], Discrepancy::Centered, false)? > 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `qmc.discrepancy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.discrepancy.html).
pub fn discrepancy(points: &[Vec<f64>], method: Discrepancy, iterative: bool) -> Result<f64> {
    let d = unit(points, 1)?;
    quadratic_work(points.len(), d)?;
    let n = (points.len() + usize::from(iterative)) as f64;
    let a = points.iter().map(|x| single(x, method)).sum::<f64>();
    let mut b = 0.;
    for x in points {
        for y in points {
            b += pair(x, y, method);
        }
    }
    let d = d as i32;
    let result = match method {
        Discrepancy::Centered => (13_f64 / 12.).powi(d) - 2. * a / n + b / (n * n),
        Discrepancy::WrapAround => -(4_f64 / 3.).powi(d) + b / (n * n),
        Discrepancy::Mixture => (19_f64 / 12.).powi(d) - 2. * a / n + b / (n * n),
        Discrepancy::L2Star => (3_f64.powi(-d) - 2_f64.powi(1 - d) * a / n + b / (n * n))
            .max(0.)
            .sqrt(),
    };
    finite(result)
}
/// Updated centered discrepancy `D_new=D_initial−2*A(x_new)/(n+1)+(2ΣB(x_new,x_i)+B(x_new,x_new))/(n+1)²`.
///
/// ## Parameters
///
/// `initial` must be discrepancy(points,Centered,true). This invariant is supplied by the caller and not recomputed.
///
/// ## Domain
///
/// Finite unit-cube point matching the sample dimension; nonempty sample; finite initial value.
///
/// ## Returns
///
/// Centered discrepancy after appending the new point.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// O(nd) work; preserves the centered-discrepancy update formula without a quadratic recomputation.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// use sci_stats::qmc::{discrepancy, update_discrepancy, Discrepancy};
/// let x=vec![vec![0.25]];
/// let d=discrepancy(&x, Discrepancy::Centered, true)?;
/// assert!(update_discrepancy(&[0.75], &x, d)? > 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `qmc.update_discrepancy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.update_discrepancy.html).
pub fn update_discrepancy(new_point: &[f64], points: &[Vec<f64>], initial: f64) -> Result<f64> {
    let d = unit(points, 1)?;
    sample(new_point, 1)?;
    if new_point.len() != d {
        return Err(Error::DimensionMismatch);
    }
    for &x in new_point {
        probability_check(x)?;
    }
    finite(initial)?;
    let n = (points.len() + 1) as f64;
    let method = Discrepancy::Centered;
    let sum = points
        .iter()
        .map(|x| pair(new_point, x, method))
        .sum::<f64>();
    finite(
        initial - 2. * single(new_point, method) / n
            + (2. * sum + pair(new_point, new_point, method)) / (n * n),
    )
}
/// Affine sample scaling `y_j=lower_j+(upper_j−lower_j)x_j`, or its inverse.
///
/// ## Parameters
///
/// Rows are points; bounds specify each coordinate. Reverse maps bounded observations into the unit cube.
///
/// ## Domain
///
/// Finite rectangular nonempty points, d≤64, finite positive bound widths; forward input in \[0,1\], reverse input within bounds.
///
/// ## Returns
///
/// Matrix with identical row/column shape.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// O(nd) work; forward scaling uses a convex combination to avoid subtractive overflow.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::qmc::scale(&[vec![0.5]], &[2.], &[4.], false)?, vec![vec![3.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `qmc.scale`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.scale.html).
pub fn scale(
    points: &[Vec<f64>],
    lower: &[f64],
    upper: &[f64],
    reverse: bool,
) -> Result<Vec<Vec<f64>>> {
    let d = dimensions(points, 1)?;
    if lower.len() != d || upper.len() != d {
        return Err(Error::DimensionMismatch);
    }
    sample(lower, 1)?;
    sample(upper, 1)?;
    for j in 0..d {
        positive(upper[j] - lower[j])?;
    }
    let mut out = Vec::with_capacity(points.len());
    for row in points {
        let mut scaled = Vec::with_capacity(d);
        for j in 0..d {
            let x = row[j];
            if reverse {
                if x < lower[j] || x > upper[j] {
                    return Err(Error::InvalidInput("point outside scaling bounds"));
                }
                scaled.push(finite((x - lower[j]) / (upper[j] - lower[j]))?);
            } else {
                probability_check(x)?;
                scaled.push(finite((1. - x) * lower[j] + x * upper[j])?);
            }
        }
        out.push(scaled);
    }
    Ok(out)
}
/// Euclidean geometric discrepancy: minimum pairwise distance or mean minimum-spanning-tree edge length.
///
/// ## Parameters
///
/// Points are rows; only Euclidean metric; larger values indicate better spacing.
///
/// ## Domain
///
/// At least two distinct finite unit-cube points, d≤64. Duplicate points are rejected for MST (SciPy's sparse graph omits zero edges).
///
/// ## Returns
///
/// Dimensionless minimum distance or mean tree edge length.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// O(n²d) distance work bounded by 16,777,216; Prim's dense graph algorithm uses O(n) auxiliary storage.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// use sci_stats::qmc::{geometric_discrepancy, GeometricDiscrepancy};
/// assert_eq!(geometric_discrepancy(&[vec![0.],vec![1.]], GeometricDiscrepancy::MinimumDistance)?, 1.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `qmc.geometric_discrepancy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.geometric_discrepancy.html).
pub fn geometric_discrepancy(points: &[Vec<f64>], method: GeometricDiscrepancy) -> Result<f64> {
    let d = unit(points, 2)?;
    let n = points.len();
    quadratic_work(n, d)?;
    let distance = |i: usize, j: usize| {
        points[i]
            .iter()
            .zip(&points[j])
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    if matches!(method, GeometricDiscrepancy::MinimumDistance) {
        let mut min = f64::INFINITY;
        for i in 0..n {
            for j in 0..i {
                min = min.min(distance(i, j));
            }
        }
        return Ok(min);
    }
    let mut used = vec![false; n];
    let mut best = vec![f64::INFINITY; n];
    best[0] = 0.;
    let mut total = 0.;
    for step in 0..n {
        let i = (0..n)
            .filter(|&i| !used[i])
            .min_by(|&i, &j| best[i].total_cmp(&best[j]))
            .unwrap();
        used[i] = true;
        total += best[i];
        for j in 0..n {
            if !used[j] {
                let length = distance(i, j);
                if length == 0. {
                    return Err(Error::InvalidInput("MST sample points must be distinct"));
                }
                best[j] = best[j].min(length);
            }
        }
        if step == n - 1 {
            break;
        }
    }
    finite(total / (n - 1) as f64)
}
