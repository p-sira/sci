use super::*;
use sci_core::Matrix;
use std::collections::VecDeque;

/// Multiscale graph correlation and its supplied-permutation calibration.
#[derive(Clone, Debug, PartialEq)]
pub struct MgcResult {
    /// Smoothed maximal local correlation, dimensionless.
    pub statistic: f64,
    /// Plus-one greater-tail permutation probability.
    pub pvalue: f64,
    /// Local correlation map: row/column are increasing dense neighbor ranks in x/y.
    pub mgc_map: Matrix,
    /// Optimal one-based `(x_scale,y_scale)`; global scale is the map shape.
    pub optimal_scale: (usize, usize),
    /// Null MGC statistics in supplied-permutation order.
    pub null_distribution: Vec<f64>,
}
struct RankedDistances {
    centered: Matrix,
    ranks: Vec<Vec<usize>>,
    scales: usize,
}
fn ranked_distances(points: &[&[f64]]) -> Result<RankedDistances> {
    let n = points.len();
    let mut distances = vec![vec![0.; n]; n];
    let mut largest = 0_f64;
    for i in 0..n {
        for j in i + 1..n {
            let distance = match transport::euclidean(points[i], points[j]) {
                value if value.is_finite() => Ok(value),
                _ => Err(crate::Error::NonFiniteEvaluation),
            }?;
            distances[i][j] = distance;
            distances[j][i] = distance;
            largest = largest.max(distance);
        }
    }
    let mut ranks = vec![vec![0; n]; n];
    let mut scales = 1;
    for j in 0..n {
        let column = distances.iter().map(|row| row[j]).collect::<Vec<_>>();
        let rank = rankdata(&column, RankMethod::Dense)?;
        for i in 0..n {
            ranks[i][j] = rank[i] as usize - 1;
            scales = scales.max(ranks[i][j] + 1);
        }
        let column = if largest > 0. {
            column.iter().map(|v| v / largest).collect::<Vec<_>>()
        } else {
            column
        };
        let average = column.iter().sum::<f64>() / (n - 1) as f64;
        for i in 0..n {
            distances[i][j] = if i == j { 0. } else { column[i] - average };
        }
    }
    Ok(RankedDistances {
        centered: distances,
        ranks,
        scales,
    })
}
fn local_covariance(x: &RankedDistances, y: &RankedDistances) -> Matrix {
    let nx = x.scales;
    let ny = y.scales;
    let n = x.centered.len();
    let mut covariance = vec![vec![0.; ny]; nx];
    let mut sums_x = vec![0.; nx];
    let mut sums_y = vec![0.; ny];
    for i in 0..n {
        for j in 0..n {
            let (a, b) = (x.centered[i][j], y.centered[j][i]);
            let (k, l) = (x.ranks[i][j], y.ranks[j][i]);
            covariance[k][l] += a * b;
            sums_x[k] += a;
            sums_y[l] += b;
        }
    }
    for i in 1..nx {
        sums_x[i] += sums_x[i - 1];
    }
    for j in 1..ny {
        sums_y[j] += sums_y[j - 1];
    }
    for i in 0..nx {
        for j in 0..ny {
            if i > 0 {
                covariance[i][j] += covariance[i - 1][j];
            }
            if j > 0 {
                covariance[i][j] += covariance[i][j - 1];
            }
            if i > 0 && j > 0 {
                covariance[i][j] -= covariance[i - 1][j - 1];
            }
        }
    }
    for i in 0..nx {
        for j in 0..ny {
            covariance[i][j] -= sums_x[i] * sums_y[j] / (n * n) as f64;
        }
    }
    covariance
}
fn mgc_map(x: &RankedDistances, y: &RankedDistances) -> Matrix {
    let mut covariance = local_covariance(x, y);
    let vx = local_covariance(x, x);
    let vy = local_covariance(y, y);
    for i in 0..x.scales {
        for j in 0..y.scales {
            covariance[i][j] = if vx[i][i] <= 0. || vy[j][j] <= 0. {
                0.
            } else {
                (covariance[i][j] / vx[i][i].sqrt() / vy[j][j].sqrt()).min(1.)
            };
        }
    }
    covariance
}
fn smoothed_maximum(map: &Matrix, observations: usize) -> Result<(f64, (usize, usize))> {
    let rows = map.len();
    let columns = map[0].len();
    let global = map[rows - 1][columns - 1];
    if rows == 1 || columns == 1 {
        return Ok((global, (rows, columns)));
    }
    let n = (observations - 1) as f64;
    let shape = n * (n - 3.) / 4. - 0.5;
    let threshold =
        (2. * ppf(Distribution::Beta { a: shape, b: shape }, 1. - 0.02 / n)? - 1.).max(global);
    let mut visited = vec![false; rows * columns];
    let mut largest = Vec::new();
    for index in 0..visited.len() {
        if visited[index] || map[index / columns][index % columns] <= threshold {
            continue;
        }
        let mut component = Vec::new();
        let mut pending = VecDeque::from([index]);
        visited[index] = true;
        while let Some(index) = pending.pop_front() {
            component.push(index);
            let (i, j) = (index / columns, index % columns);
            let neighbors = [
                i.checked_sub(1).map(|r| r * columns + j),
                (i + 1 < rows).then_some((i + 1) * columns + j),
                j.checked_sub(1).map(|c| i * columns + c),
                (j + 1 < columns).then_some(i * columns + j + 1),
            ];
            for next in neighbors.into_iter().flatten() {
                if !visited[next] && map[next / columns][next % columns] > threshold {
                    visited[next] = true;
                    pending.push_back(next);
                }
            }
        }
        if component.len() > largest.len() {
            largest = component;
        }
    }
    let minimum_area = (0.02 * rows.max(columns) as f64).ceil() as usize * rows.min(columns);
    if largest.len() < minimum_area {
        return Ok((global, (rows, columns)));
    }
    let maximum = largest
        .iter()
        .map(|&i| map[i / columns][i % columns])
        .fold(f64::NEG_INFINITY, f64::max);
    if maximum < global {
        return Ok((global, (rows, columns)));
    }
    let optimal = largest
        .into_iter()
        .filter(|&i| map[i / columns][i % columns] >= maximum)
        .max()
        .ok_or(Error::NumericalFailure("empty MGC smoothing region"))?;
    Ok((maximum, (optimal / columns + 1, optimal % columns + 1)))
}
fn permuted_distances(y: &RankedDistances, permutation: &[usize]) -> RankedDistances {
    let n = permutation.len();
    let mut centered = vec![vec![0.; n]; n];
    let mut ranks = vec![vec![0; n]; n];
    for i in 0..n {
        for j in 0..n {
            centered[i][j] = y.centered[permutation[i]][permutation[j]];
            ranks[i][j] = y.ranks[permutation[i]][permutation[j]];
        }
    }
    RankedDistances {
        centered,
        ranks,
        scales: y.scales,
    }
}

/// Multiscale graph correlation, the smoothed maximum of neighborhood distance correlations.
///
/// ## Parameters and conventions
///
/// `x`,`y` are paired point rows, possibly of different dimensions. Euclidean
/// distances are column-centered (diagonal zeroed) and densely ranked. For
/// centered distances A/B and dense neighbor ranks R/S, local covariance is
/// `C_kl=Σ_ij A_ij*B_ji I(R_ij≤k)I(S_ji≤l)−E_A(k)*E_B(l)/n²`.
/// Local variances use the same covariance with A/A or B/B, and local
/// correlations are `C_kl/sqrt(V_A(k)V_B(l))`, zero at nonpositive variance.
/// Smoothing keeps the largest four-neighbor region above both global correlation
/// and a beta-null threshold, then maximizes within it if its area is large
/// enough (Shen et al. 2019). `permutations` supplies zero-based y permutations,
/// replacing SciPy RNG ownership. Point/distance callback, automatic sampling,
/// parallel workers and automatic two-sample transformation are unsupported.
///
/// ## Domain and special values
///
/// 5..=128 paired finite point rows, each rectangular with dimension 1..=64.
/// 1..=65536 full index permutations with `(B+1)*n²≤16777216`. Pairwise
/// Euclidean distances must be representable. Constants give zero correlations;
/// zero-variance map entries are zero. NaN/infinity coordinates are errors.
///
/// ## Returns
///
/// Dimensionless statistic, local map with shape equal to maximum dense ranks,
/// one-based optimal scale, null statistics in supplied order and plus-one
/// greater-tail p-value. Floating-point tail equality uses 100*epsilon*|statistic|,
/// a conservative tie adaptation of SciPy's strict comparison.
///
/// ## Errors
///
/// Invalid domains/shapes/permutations/work limits or nonfinite distances/statistics return errors.
///
/// ## Numerical behavior
///
/// Native O(n² log n+B*n²) rank/prefix-covariance/connected-component work and
/// O(n²+B) memory, plus private shared beta quantiles. Distance scaling avoids
/// product overflow while preserving raw-distance ties. Normalization and
/// threshold-boundary rounding can change selected scales near exact ties.
/// Null calibration requires independently generated uniform permutations;
/// low B produces coarse uncertain p-values. No Python runtime is used.
///
/// ## Examples
///
/// ```
/// let x: &[&[f64]] = &[&[0.],&[1.],&[2.],&[3.],&[4.]];
/// let permutations: &[&[usize]] = &[&[4,3,2,1,0],&[1,3,0,4,2]];
/// let r = sci_stats::association::multiscale_graphcorr(x,x,permutations)?;
/// assert!((r.statistic-1.).abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy multiscale_graphcorr and Shen–Priebe–Vogelstein (2019)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multiscale_graphcorr.html).
pub fn multiscale_graphcorr(
    x: &[&[f64]],
    y: &[&[f64]],
    permutations: &[&[usize]],
) -> Result<MgcResult> {
    if x.len() != y.len() {
        return Err(Error::DimensionMismatch);
    }
    transport::points_check(x, 5, 128)?;
    transport::points_check(y, 5, 128)?;
    let n = x.len();
    if permutations.is_empty()
        || permutations.len() > 65536
        || (permutations.len() + 1)
            .checked_mul(n * n)
            .is_none_or(|w| w > 16_777_216)
    {
        return Err(Error::InvalidInput("MGC permutation work limit exceeded"));
    }
    for permutation in permutations {
        if permutation.len() != n {
            return Err(Error::DimensionMismatch);
        }
        let mut seen = vec![false; n];
        for &i in *permutation {
            if i >= n || seen[i] {
                return Err(Error::InvalidInput("MGC index row must be a permutation"));
            }
            seen[i] = true;
        }
    }
    let x = ranked_distances(x)?;
    let y = ranked_distances(y)?;
    let map = mgc_map(&x, &y);
    let (statistic, optimal_scale) = smoothed_maximum(&map, n)?;
    if !statistic.is_finite() {
        return Err(crate::Error::NonFiniteEvaluation);
    }
    let mut null_distribution = Vec::with_capacity(permutations.len());
    for permutation in permutations {
        let permuted = permuted_distances(&y, permutation);
        let local = mgc_map(&x, &permuted);
        null_distribution.push(match smoothed_maximum(&local, n)?.0 {
            value if value.is_finite() => Ok(value),
            _ => Err(crate::Error::NonFiniteEvaluation),
        }?);
    }
    let tolerance = 100. * f64::EPSILON * statistic.abs();
    let extreme = null_distribution
        .iter()
        .filter(|&&v| v >= statistic - tolerance)
        .count();
    let pvalue = (extreme + 1) as f64 / (permutations.len() + 1) as f64;
    Ok(MgcResult {
        statistic,
        pvalue,
        mgc_map: map,
        optimal_scale,
        null_distribution,
    })
}
