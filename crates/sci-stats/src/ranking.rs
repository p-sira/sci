use super::*;

pub(super) struct TieCounts {
    pub(super) pairs: f64,
    pub(super) triples: f64,
    pub(super) variance_term: f64,
    pub(super) unique: usize,
}

pub(super) fn tie_counts(values: &[f64]) -> TieCounts {
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let mut out = TieCounts {
        pairs: 0.,
        triples: 0.,
        variance_term: 0.,
        unique: 0,
    };
    let mut start = 0;
    while start < ordered.len() {
        let mut end = start + 1;
        while end < ordered.len() && ordered[end] == ordered[start] {
            end += 1;
        }
        let k = (end - start) as f64;
        out.pairs += k * (k - 1.) / 2.;
        out.triples += k * (k - 1.) * (k - 2.);
        out.variance_term += k * (k - 1.) * (2. * k + 5.);
        out.unique += 1;
        start = end;
    }
    out
}

/// Tie handling for one-based ranks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RankMethod {
    /// Average of the ranks occupied by a tie.
    Average,
    /// Smallest rank occupied by a tie.
    Min,
    /// Largest rank occupied by a tie.
    Max,
    /// Consecutive ranks for distinct values.
    Dense,
    /// Distinct ranks, preserving input order within ties.
    Ordinal,
}
/// Percentile-of-score convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PercentileKind {
    /// Average matching one-based percentage ranks, or strict percentile if absent.
    Rank,
    /// Fraction strictly below the score.
    Strict,
    /// Fraction at or below the score.
    Weak,
    /// Average of strict and weak percentiles.
    Mean,
}
/// False discovery rate procedure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FdrMethod {
    /// Benjamini–Hochberg; independent or positively dependent tests.
    BenjaminiHochberg,
    /// Benjamini–Yekutieli; arbitrary dependence.
    BenjaminiYekutieli,
}
/// Assigns one-based ranks `r_i=1+count(x_j<x_i)`, modified for ties.
///
/// ## Parameters
///
/// `method` selects average, min, max, dense, or stable ordinal tie handling.
///
/// ## Domain
///
/// Finite observations; empty input returns empty output; signed zeros tie.
///
/// ## Returns
///
/// One rank per input observation in original order.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Sorting takes O(n log n) work; output ordering is explicit.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// use sci_stats::{ranking::rankdata, ranking::RankMethod};
/// assert_eq!(rankdata(&[3., 1., 1.], RankMethod::Average)?, vec![3., 1.5, 1.5]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `rankdata`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html).
pub fn rankdata(values: &[f64], method: RankMethod) -> Result<Vec<f64>> {
    sample(values, 0)?;
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_by(|&i, &j| values[i].partial_cmp(&values[j]).unwrap());
    let mut result = vec![0.; values.len()];
    let mut start = 0;
    let mut dense = 1;
    while start < indices.len() {
        let mut end = start + 1;
        while end < indices.len() && values[indices[end]] == values[indices[start]] {
            end += 1;
        }
        for k in start..end {
            result[indices[k]] = match method {
                RankMethod::Average => (start + 1 + end) as f64 / 2.,
                RankMethod::Min => (start + 1) as f64,
                RankMethod::Max => end as f64,
                RankMethod::Dense => dense as f64,
                RankMethod::Ordinal => (k + 1) as f64,
            };
        }
        dense += 1;
        start = end;
    }
    Ok(result)
}
/// Tie correction `1−Σ(t_j³−t_j)/(n³−n)` for tied groups of size t_j.
///
/// ## Parameters
///
/// Input contains ranks, normally produced by rankdata; ties are exact equality.
///
/// ## Domain
///
/// Finite ranks; fewer than two ranks returns one.
///
/// ## Returns
///
/// Correction factor in \[0,1\].
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Sorting takes O(n log n) work; output ordering is explicit.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::ranking::tiecorrect(&[1.5, 1.5])?, 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `tiecorrect`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tiecorrect.html).
pub fn tiecorrect(ranks: &[f64]) -> Result<f64> {
    sample(ranks, 0)?;
    if ranks.len() < 2 {
        return Ok(1.);
    }
    let mut x = ranks.to_vec();
    x.sort_by(f64::total_cmp);
    let mut start = 0;
    let mut correction = 0.;
    while start < x.len() {
        let mut end = start + 1;
        while end < x.len() && x[end] == x[start] {
            end += 1;
        }
        let t = (end - start) as f64;
        correction += t * t * t - t;
        start = end;
    }
    let n = x.len() as f64;
    Ok(1. - correction / (n * n * n - n))
}
/// Percentile rank `100*count(x≤score)/n`, with configurable strictness.
///
/// ## Parameters
///
/// Returns percent, in \[0,100\]. Rank and mean average tied positions.
///
/// ## Domain
///
/// Nonempty finite sample; infinite scores are allowed; NaN is an error.
///
/// ## Returns
///
/// Scalar percentage rank.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Counts use O(n) work; integer multiplicities are converted once to binary64.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// use sci_stats::{ranking::percentileofscore, ranking::PercentileKind};
/// assert_eq!(percentileofscore(&[1., 2., 3.], 2., PercentileKind::Weak)?, 200./3.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `percentileofscore`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.percentileofscore.html).
pub fn percentileofscore(values: &[f64], score: f64, kind: PercentileKind) -> Result<f64> {
    sample(values, 1)?;
    if score.is_nan() {
        return Err(Error::InvalidInput("score must not be NaN"));
    }
    let less = values.iter().filter(|&&x| x < score).count() as f64;
    let equal = values.iter().filter(|&&x| x == score).count() as f64;
    let count = match kind {
        PercentileKind::Strict => less,
        PercentileKind::Weak => less + equal,
        PercentileKind::Mean => less + equal / 2.,
        PercentileKind::Rank => less + if equal > 0. { (equal + 1.) / 2. } else { 0. },
    };
    Ok(100. * count / values.len() as f64)
}
/// FDR-adjusted p-values `q_(i)=min_{j≥i}(n*p_(j)/j)`, clipped to one.
///
/// ## Parameters
///
/// BY multiplies by the harmonic number H_n; BH uses multiplier one.
///
/// ## Domain
///
/// Finite probabilities in \[0,1\]; empty input returns empty output.
///
/// ## Returns
///
/// Adjusted probabilities in original input order.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Sorting takes O(n log n) work; output ordering is explicit.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// use sci_stats::{inference::false_discovery_control, inference::FdrMethod};
/// assert_eq!(false_discovery_control(&[0.01], FdrMethod::BenjaminiHochberg)?, vec![0.01]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `false_discovery_control`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html).
pub fn false_discovery_control(pvalues: &[f64], method: FdrMethod) -> Result<Vec<f64>> {
    sample(pvalues, 0)?;
    for &p in pvalues {
        probability_check(p)?;
    }
    let n = pvalues.len();
    let c = match method {
        FdrMethod::BenjaminiHochberg => 1.,
        FdrMethod::BenjaminiYekutieli => (1..=n).map(|k| 1. / k as f64).sum(),
    };
    let mut indices = (0..n).collect::<Vec<_>>();
    indices.sort_by(|&i, &j| pvalues[i].total_cmp(&pvalues[j]));
    let mut out = vec![0.; n];
    let mut bound = 1_f64;
    for i in (0..n).rev() {
        bound = bound.min(pvalues[indices[i]] * n as f64 * c / (i + 1) as f64);
        out[indices[i]] = bound;
    }
    Ok(out)
}
/// Empirical CDF `F_n(t)=count(x_i≤t)/n` and survival `1−F_n(t)`.
///
/// ## Parameters
///
/// Uncensored samples only. Duplicate observations are combined.
///
/// ## Domain
///
/// Nonempty finite sample.
///
/// ## Returns
///
/// Sorted unique quantiles and corresponding cumulative and survival probabilities.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Sorting takes O(n log n) work; output ordering is explicit.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::ranking::ecdf(&[2., 1., 2.])?.probabilities, vec![1./3., 1.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `ecdf`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ecdf.html).
pub fn ecdf(values: &[f64]) -> Result<EmpiricalCdf> {
    let x = sorted(values)?;
    let n = x.len() as f64;
    let mut quantiles = Vec::new();
    let mut probabilities = Vec::new();
    let mut i = 0;
    while i < x.len() {
        let mut j = i + 1;
        while j < x.len() && x[j] == x[i] {
            j += 1;
        }
        quantiles.push(x[i]);
        probabilities.push(j as f64 / n);
        i = j;
    }
    let survival = probabilities.iter().map(|p| 1. - p).collect();
    Ok(EmpiricalCdf {
        quantiles,
        probabilities,
        survival,
    })
}

/// Uncensored empirical distribution, ordered by distinct observation value.
#[derive(Clone, Debug, PartialEq)]
pub struct EmpiricalCdf {
    /// Sorted distinct observation values.
    pub quantiles: Vec<f64>,
    /// Right-continuous F_n at each quantile.
    pub probabilities: Vec<f64>,
    /// P_n(X>quantile), in matching order.
    pub survival: Vec<f64>,
}

/// Hyndman–Fan convention for estimating a sample CDF.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EstimatedCdfMethod {
    /// Empirical CDF: number of observations ≤y divided by n.
    InvertedCdf,
    /// Same estimated CDF as InvertedCdf, matching SciPy's inverse convention.
    AveragedInvertedCdf,
    /// Number of observations ≤y plus 0.5, divided by n and clipped to `[0,1]`.
    ClosestObservation,
    /// Linear inverse with a=0,b=1.
    InterpolatedInvertedCdf,
    /// Linear inverse with a=b=0.5.
    Hazen,
    /// Linear inverse with a=b=0.
    Weibull,
    /// Linear inverse with a=b=1, inverse of the default sample quantile.
    Linear,
    /// Linear inverse with a=b=1/3.
    MedianUnbiased,
    /// Linear inverse with a=b=3/8.
    NormalUnbiased,
}

/// Estimated sample CDF, the clipped Hyndman–Fan inverse-quantile mapping.
///
/// ## Parameters and conventions
///
/// `values` is an unweighted sample, `y` an evaluation point and `method` a
/// convention. Let J count observations≤y. Discontinuous methods use J/n,
/// except ClosestObservation uses (J+0.5)/n. Continuous methods use
/// `(j+δ−a)/(n+1−a−b)`, where j=clip(J, min(1,n−1),n−1), sorted x indices
/// are zero-based, and `δ=(y−x_(j−1))/(x_j−x_(j−1))`, or 1 for equal knots.
/// [`EstimatedCdfMethod`] lists a,b. Values outside the sample range return
/// zero/one; results are clipped to `[0,1]`. Axes, weights and masks are unsupported.
///
/// ## Domain and special values
///
/// Nonempty finite sample of at most 1048576 observations; y may be infinite
/// but not NaN. Signed zeros are tied. Linear mode with a singleton sample
/// at y=x returns NaN (a zero denominator), matching SciPy.
///
/// ## Returns
///
/// Dimensionless scalar estimated probability, generally different from [`ecdf`]
/// for continuous conventions and at ties.
///
/// ## Errors
///
/// Empty/oversized or nonfinite samples and NaN evaluation points return errors.
///
/// ## Numerical behavior
///
/// O(n log n) sorting and O(n) memory. Scaled interpolation avoids overflow
/// across opposite extreme values. Duplicate knots use SciPy's rightmost
/// search and equal-knot convention. This is an estimator, not a fitted probability model.
///
/// ## Examples
///
/// ```
/// assert_eq!(sci_stats::ranking::estimated_cdf(&[0.,2.,4.],1.,sci_stats::ranking::EstimatedCdfMethod::Linear)?, 0.25);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy estimated_cdf and Hyndman–Fan (1996)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.estimated_cdf.html).
pub fn estimated_cdf(values: &[f64], y: f64, method: EstimatedCdfMethod) -> Result<f64> {
    if y.is_nan() {
        return Err(Error::InvalidInput("CDF evaluation point must not be NaN"));
    }
    let x = sorted(values)?;
    if y < x[0] {
        return Ok(0.);
    }
    if y > x[x.len() - 1] {
        return Ok(1.);
    }
    let n = x.len() as f64;
    let count = x.partition_point(|v| *v <= y);
    let probability = match method {
        EstimatedCdfMethod::InvertedCdf | EstimatedCdfMethod::AveragedInvertedCdf => {
            count as f64 / n
        }
        EstimatedCdfMethod::ClosestObservation => (count as f64 + 0.5) / n,
        _ => {
            let (a, b) = match method {
                EstimatedCdfMethod::InterpolatedInvertedCdf => (0., 1.),
                EstimatedCdfMethod::Hazen => (0.5, 0.5),
                EstimatedCdfMethod::Weibull => (0., 0.),
                EstimatedCdfMethod::Linear => (1., 1.),
                EstimatedCdfMethod::MedianUnbiased => (1. / 3., 1. / 3.),
                EstimatedCdfMethod::NormalUnbiased => (3. / 8., 3. / 8.),
                _ => unreachable!(),
            };
            let j = count.clamp((x.len() - 1).min(1), x.len() - 1);
            let left = x[j.saturating_sub(1)];
            let right = x[j];
            let delta = if right == left {
                1.
            } else {
                let width = right - left;
                if width.is_finite() {
                    (y - left) / width
                } else {
                    let scale = left.abs().max(right.abs());
                    (y / scale - left / scale) / (right / scale - left / scale)
                }
            };
            (j as f64 + delta - a) / (n + 1. - a - b)
        }
    };
    Ok(probability.clamp(0., 1.))
}
