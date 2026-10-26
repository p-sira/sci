//! Euclidean vector quantization and bounded Lloyd k-means.
use crate::error::{finite, finite_slice};
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
        finite_slice(row)?;
    }
    Ok(columns)
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
                distance = distance.hypot(finite(a - b)?);
            }
            if distance < best.1 {
                best = (i, distance);
            }
        }
        labels.push(best.0);
        distances.push(finite(best.1)?);
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
            *sum = finite(*sum + value / counts[*label] as f64)?;
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
/// Uses shared [`sci_stats::std`]; two passes per column plus scaling.
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
    let scales: Vec<f64> = (0..columns)
        .map(|j| {
            let column: Vec<f64> = obs.iter().map(|row| row[j]).collect();
            let scale = finite(crate::stats::std(&column, 0)?)?;
            Ok(if scale == 0.0 { 1.0 } else { scale })
        })
        .collect::<Result<_>>()?;
    obs.iter()
        .map(|row| {
            row.iter()
                .zip(&scales)
                .map(|(v, s)| finite(v / s))
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
        let distortion = finite(distances.iter().map(|v| v / obs.len() as f64).sum())?;
        if (previous - distortion).abs() <= threshold {
            return Ok((codes, distortion));
        }
        codes = update(obs, &codes, &labels, true)?;
        previous = distortion;
    }
    let (_, distances) = assign(obs, &codes)?;
    let distortion = finite(distances.iter().map(|v| v / obs.len() as f64).sum())?;
    if (previous - distortion).abs() <= threshold {
        Ok((codes, distortion))
    } else {
        Err(Error::NumericalFailure(
            "k-means did not converge within iteration limit",
        ))
    }
}
