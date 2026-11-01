//! Euclidean vector quantization and bounded Lloyd k-means.
use crate::error::is_finite_slice;
use crate::{Error, Result};

fn shape(data: &[Vec<f64>]) -> Result<usize> {
    let columns = data.first().map_or(0, Vec::len);
    if columns == 0
        || data
            .len()
            .checked_mul(columns)
            .is_none_or(|size| size > 1_000_000)
    {
        return Err(Error::InvalidInput(
            "observations must be nonempty and contain at most 1000000 entries",
        ));
    }
    for row in data {
        if row.len() != columns {
            return Err(Error::DimensionMismatch);
        }
        is_finite_slice(row)?;
    }
    Ok(columns)
}

// Keep the sign-balanced compensated mean used by the shared statistics
// implementation. Only normalized deviations with magnitude <= 1 reach this
// helper, and there are at most 1,000,000 of them, so the partial sums cannot
// overflow. Centering after scaling preserves differences at large offsets.
fn centered_mean(values: &[f64]) -> f64 {
    let mut positive = values.iter().copied().filter(|x| *x >= 0.0);
    let mut negative = values.iter().copied().filter(|x| *x < 0.0);
    let (mut sum, mut correction) = (0_f64, 0_f64);
    loop {
        let term = if sum >= 0.0 {
            negative.next().or_else(|| positive.next())
        } else {
            positive.next().or_else(|| negative.next())
        };
        let Some(term) = term else {
            return (sum + correction) / values.len() as f64;
        };
        let next = sum + term;
        correction += if sum.abs() >= term.abs() {
            (sum - next) + term
        } else {
            (term - next) + sum
        };
        sum = next;
    }
}

// Population standard deviation of an already validated column. Reuse the
// deviation buffer across columns to avoid repeated allocation in `whiten`.
fn population_std(values: &[f64], deviations: &mut Vec<f64>) -> Result<f64> {
    let origin = values[0];
    deviations.clear();
    deviations.extend(values.iter().map(|&x| x - origin));
    let mut scale = deviations.iter().fold(0_f64, |s, x| s.max(x.abs()));
    if !scale.is_finite() {
        scale = values.iter().fold(0_f64, |s, x| s.max(x.abs()));
        for (y, x) in deviations.iter_mut().zip(values) {
            *y = *x / scale;
        }
    } else if scale == 0.0 {
        return Ok(0.0);
    } else {
        for y in deviations.iter_mut() {
            *y /= scale;
        }
    }
    let center = centered_mean(deviations);
    for y in deviations.iter_mut() {
        *y -= center;
    }
    let normalized = deviations
        .iter()
        .map(|x| x.powi(2) / values.len() as f64)
        .fold(0.0, |a, b| a + b);
    match normalized.sqrt() * scale {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}

fn validate(obs: &[Vec<f64>], codes: &[Vec<f64>], iterations: usize) -> Result<()> {
    let d = shape(obs)?;
    if shape(codes)? != d {
        return Err(Error::DimensionMismatch);
    }
    if iterations == 0
        || obs
            .len()
            .checked_mul(codes.len())
            .and_then(|v| v.checked_mul(d))
            .and_then(|v| v.checked_mul(iterations))
            .is_none_or(|v| v > 10_000_000)
    {
        return Err(Error::InvalidInput(
            "vector quantization work limit exceeded",
        ));
    }
    Ok(())
}

fn assign(obs: &[Vec<f64>], codes: &[Vec<f64>]) -> Result<(Vec<usize>, Vec<f64>)> {
    let mut labels = Vec::with_capacity(obs.len());
    let mut distances = Vec::with_capacity(obs.len());
    for row in obs {
        let mut best = (0, f64::INFINITY);
        for (i, code) in codes.iter().enumerate() {
            let mut distance = 0.0_f64;
            for (a, b) in row.iter().zip(code) {
                distance = distance.hypot(match a - b {
                    value if value.is_finite() => Ok(value),
                    _ => Err(Error::NonFiniteEvaluation),
                }?);
            }
            if distance < best.1 {
                best = (i, distance);
            }
        }
        labels.push(best.0);
        distances.push(match best.1 {
            value if value.is_finite() => Ok(value),
            _ => Err(Error::NonFiniteEvaluation),
        }?);
    }
    Ok((labels, distances))
}

fn update(
    obs: &[Vec<f64>],
    codes: &[Vec<f64>],
    labels: &[usize],
    drop_empty: bool,
) -> Result<Vec<Vec<f64>>> {
    let mut counts = vec![0_usize; codes.len()];
    for label in labels {
        counts[*label] += 1;
    }
    let mut sums = vec![vec![0.0; codes[0].len()]; codes.len()];
    for (row, label) in obs.iter().zip(labels) {
        for (sum, value) in sums[*label].iter_mut().zip(row) {
            *sum = match *sum + value / counts[*label] as f64 {
                value if value.is_finite() => Ok(value),
                _ => Err(Error::NonFiniteEvaluation),
            }?;
        }
    }
    let mut result = Vec::new();
    for (i, sum) in sums.into_iter().enumerate() {
        if counts[i] > 0 {
            result.push(sum);
        } else if !drop_empty {
            result.push(codes[i].clone());
        }
    }
    Ok(result)
}

/// Assigns each observation to its nearest code: `label[i]=argmin_j ||obs[i]-code[j]||₂`.
///
/// ## Parameters
///
/// `obs` and `code_book` contain rows of equal feature dimension. Labels are
/// zero-based; equal distances choose the earliest code.
/// ## Domain
///
/// Nonempty finite matrices with nonempty rows, at most 1,000,000 entries per
/// matrix and 10,000,000 feature comparisons. NaN/infinity are rejected.
/// ## Returns
///
/// `(labels, distances)`, each of length `obs.len()`. Distances are Euclidean,
/// not squared, in the feature units. Inputs are never whitened automatically.
/// ## Errors
///
/// Invalid sizes/values, dimension mismatch, work limit, or distance overflow.
/// ## Numerical behavior
///
/// Exhaustive nearest-code search; `hypot` accumulation avoids squared overflow.
/// # Examples
///
/// ```
/// let (labels,distances)=sci_cluster::vq::vq(&[vec![1.0]],&[vec![0.0],vec![3.0]])?;
/// assert_eq!(labels,vec![0]);assert_eq!(distances,vec![1.0]);
/// # Ok::<(),sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy vq](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.vq.html).
pub fn vq(obs: &[Vec<f64>], code_book: &[Vec<f64>]) -> Result<(Vec<usize>, Vec<f64>)> {
    validate(obs, code_book, 1)?;
    assign(obs, code_book)
}

/// Scales each feature by its population standard deviation: `out[i,j]=obs[i,j]/σ[j]`.
///
/// ## Parameters
///
/// `obs` is a row-major matrix. Features are not centered; `σ` uses `ddof=0`.
/// ## Domain
///
/// A finite, nonempty matrix with nonempty rows and at most 1,000,000 entries.
/// Zero-variance columns are unchanged, matching SciPy's numerical result;
/// SciPy's runtime warning is omitted. NaN/infinity inputs are rejected.
/// ## Returns
///
/// Whitened rows with the same shape and feature order as `obs`.
/// ## Errors
///
/// Invalid/nonfinite inputs, ragged rows, or nonfinite numerical results.
/// ## Numerical behavior
///
/// Uses a private, scaled population variance with a compensated mean; two
/// passes per column plus scaling, with scratch buffers reused between columns.
/// Extreme finite inputs can overflow the variance calculation and return errors.
/// # Examples
///
/// ```
/// assert_eq!(sci_cluster::vq::whiten(&[vec![1.0],vec![3.0]])?,vec![vec![1.0],vec![3.0]]);
/// # Ok::<(),sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy whiten](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.whiten.html).
pub fn whiten(obs: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let columns = shape(obs)?;
    let mut column = Vec::with_capacity(obs.len());
    let mut deviations = Vec::with_capacity(obs.len());
    let scales: Vec<f64> = (0..columns)
        .map(|j| {
            column.clear();
            column.extend(obs.iter().map(|row| row[j]));
            let scale = match population_std(&column, &mut deviations)? {
                value if value.is_finite() => Ok(value),
                _ => Err(Error::NonFiniteEvaluation),
            }?;
            Ok(if scale == 0.0 { 1.0 } else { scale })
        })
        .collect::<Result<_>>()?;
    obs.iter()
        .map(|row| {
            row.iter()
                .zip(&scales)
                .map(|(v, s)| match v / s {
                    value if value.is_finite() => Ok(value),
                    _ => Err(Error::NonFiniteEvaluation),
                })
                .collect()
        })
        .collect()
}

/// Runs exactly `iterations` Lloyd k-means steps, minimizing squared Euclidean distortion.
///
/// ## Parameters
///
/// `obs` is row-major data; `initial` explicitly supplies starting centroid rows,
/// adapting SciPy's `minit="matrix"`. Each step assigns nearest codes then
/// replaces each code by the assigned mean. Empty clusters retain their code.
/// ## Domain
///
/// Finite, nonempty, shape-compatible matrices. Positive iteration count;
/// `rows*codes*features*iterations <= 10,000,000`, at most 1,000,000 entries
/// per matrix. No random initialization, tolerance stop, or missing-data mode.
/// ## Returns
///
/// `(centroids, labels)`: centroid shape matches `initial`, labels have one
/// zero-based entry per observation. As in SciPy, labels belong to the final
/// assignment step, preceding the final centroid update.
/// ## Errors
///
/// Invalid inputs, shape mismatch, exceeded work limit, or nonfinite arithmetic.
/// ## Numerical behavior
///
/// Deterministic Lloyd iteration converges only to a local optimum. Means sum
/// pre-divided samples; floating-point tie/order effects can affect labels.
/// # Examples
///
/// ```
/// let (codes,labels)=sci_cluster::vq::kmeans2(&[vec![0.0],vec![2.0]],&[vec![0.0]],2)?;
/// assert_eq!(codes,vec![vec![1.0]]);assert_eq!(labels,vec![0,0]);
/// # Ok::<(),sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy kmeans2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans2.html).
pub fn kmeans2(
    obs: &[Vec<f64>],
    initial: &[Vec<f64>],
    iterations: usize,
) -> Result<(Vec<Vec<f64>>, Vec<usize>)> {
    validate(obs, initial, iterations)?;
    let mut codes = initial.to_vec();
    let mut labels = Vec::new();
    for _ in 0..iterations {
        (labels, _) = assign(obs, &codes)?;
        codes = update(obs, &codes, &labels, false)?;
    }
    Ok((codes, labels))
}

/// Runs Lloyd k-means until the mean Euclidean distortion stabilizes.
///
/// ## Parameters
///
/// `obs` contains rows; `initial` supplies explicit codes. `threshold` is the
/// nonnegative absolute change in mean Euclidean distance for stopping.
/// `max_iterations` is a work bound, adapting SciPy's unbounded inner loop;
/// SciPy's `iter` restart count and random initialization are omitted.
/// Each centroid update minimizes squared distance; empty codes are removed.
/// ## Domain
///
/// Finite matrices/threshold and positive iteration bound. At most 1,000,000
/// entries per matrix and 10,000,000 total feature comparisons, including
/// one final reassignment. Nonfinite inputs are rejected.
/// ## Returns
///
/// `(centroids, mean_distance)`; removed empty codes reduce the row count.
/// ## Errors
///
/// Invalid inputs, dimension mismatch, work limit, nonfinite arithmetic, or
/// failure to meet the threshold before `max_iterations`.
/// ## Numerical behavior
///
/// Deterministic local optimization. Returned distortion is recomputed against
/// returned centroids; no global-optimum guarantee or stochastic restarts.
/// # Examples
///
/// ```
/// let (codes,d)=sci_cluster::vq::kmeans(&[vec![0.0],vec![2.0]],&[vec![0.0]],1e-8,10)?;
/// assert_eq!(codes,vec![vec![1.0]]);assert_eq!(d,1.0);
/// # Ok::<(),sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy kmeans](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans.html).
pub fn kmeans(
    obs: &[Vec<f64>],
    initial: &[Vec<f64>],
    threshold: f64,
    max_iterations: usize,
) -> Result<(Vec<Vec<f64>>, f64)> {
    if !threshold.is_finite() || threshold < 0.0 || max_iterations == 0 {
        return Err(Error::InvalidInput(
            "invalid k-means tolerance or iteration limit",
        ));
    }
    let work = max_iterations
        .checked_add(1)
        .ok_or(Error::InvalidInput("iteration limit overflow"))?;
    validate(obs, initial, work)?;
    let mut codes = initial.to_vec();
    let mut previous = f64::INFINITY;
    for _ in 0..max_iterations {
        let (labels, distances) = assign(obs, &codes)?;
        let distortion = match distances.iter().map(|v| v / obs.len() as f64).sum::<f64>() {
            value if value.is_finite() => Ok(value),
            _ => Err(Error::NonFiniteEvaluation),
        }?;
        if (previous - distortion).abs() <= threshold {
            return Ok((codes, distortion));
        }
        codes = update(obs, &codes, &labels, true)?;
        previous = distortion;
    }
    let (_, distances) = assign(obs, &codes)?;
    let distortion = match distances.iter().map(|v| v / obs.len() as f64).sum::<f64>() {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }?;
    if (previous - distortion).abs() <= threshold {
        Ok((codes, distortion))
    } else {
        Err(Error::NumericalFailure(
            "k-means did not converge within iteration limit",
        ))
    }
}
