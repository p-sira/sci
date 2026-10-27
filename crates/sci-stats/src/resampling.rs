use super::*;

/// Statistic, p-value and null statistics from resampling.
#[derive(Clone, Debug, PartialEq)]
pub struct ResamplingResult {
    /// Observed statistic, in the callback's units.
    pub statistic: f64,
    /// Tail probability, with conventions documented by the producing function.
    pub pvalue: f64,
    /// Null statistics in deterministic enumeration or supplied-draw order.
    pub null_distribution: Vec<f64>,
}
fn resampling_probability(
    statistic: f64,
    null: &[f64],
    alternative: Alternative,
    randomized: bool,
) -> f64 {
    let tolerance = 100. * f64::EPSILON * statistic.abs();
    let adjustment = usize::from(randomized);
    let denominator = (null.len() + adjustment) as f64;
    let less = (null.iter().filter(|&&v| v <= statistic + tolerance).count() + adjustment) as f64
        / denominator;
    let greater = (null.iter().filter(|&&v| v >= statistic - tolerance).count() + adjustment)
        as f64
        / denominator;
    match alternative {
        Alternative::Less => less,
        Alternative::Greater => greater,
        Alternative::TwoSided => (2. * less.min(greater)).min(1.),
    }
}
fn resample_work(length: usize, repetitions: usize) -> Result<()> {
    if repetitions == 0
        || repetitions > MAX_SAMPLES
        || length
            .checked_mul(repetitions)
            .is_none_or(|w| w > 16_777_216)
    {
        return Err(Error::InvalidInput(
            "resampling work limit is 16777216 observations",
        ));
    }
    Ok(())
}

/// Exact independent-sample permutation test `p=P_H0(T≥T_obs)` (greater alternative).
///
/// ## Parameters and conventions
///
/// `x`,`y` are independent samples; `statistic` is a deterministic, fallible,
/// scalar callback. All partitions of pooled observations into original sizes
/// are evaluated, retaining original within-group order. The callback must be
/// invariant to within-group order. Less uses the lower tail; TwoSided is twice
/// the smaller tail, clipped at one. Equality allows 100*f64::EPSILON*|T_obs|.
/// This implements SciPy `permutation_type="independent", n_resamples=inf`;
/// randomized, paired, multi-sample and vectorized overloads are unsupported.
///
/// ## Domain and special values
///
/// At least two finite observations per sample, combined count≤20. Callback outputs must be finite.
/// Duplicate observations are distinct exchangeable subjects. At most 184756
/// partitions are evaluated, each containing at most 20 observations.
///
/// ## Returns
///
/// Observed statistic, exact p-value and one null statistic per partition in
/// lexicographic index order. Exact tails do not use a plus-one adjustment.
///
/// ## Errors
///
/// Invalid inputs/work limits, nonfinite statistics or callback errors return errors.
/// Callback errors are preserved, and no result is returned after a failed draw.
///
/// ## Numerical behavior
///
/// O(binomial(n+m,n)*(n+m)) input work and O(binomial(n+m,n)+n+m) memory;
/// callback work is additional and controlled by the caller. Counts are exact;
/// ties use SciPy's relative comparison tolerance. Null exchangeability is required.
///
/// ## Examples
///
/// ```
/// use sci_stats::{permutation_test, mean, Alternative};
/// let r = permutation_test(&[1.,2.], &[3.,4.], |x,y| Ok(mean(x)?-mean(y)?), Alternative::Less)?;
/// assert_eq!(r.null_distribution.len(), 6);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy permutation_test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).
pub fn permutation_test<F>(
    x: &[f64],
    y: &[f64],
    statistic: F,
    alternative: Alternative,
) -> Result<ResamplingResult>
where
    F: Fn(&[f64], &[f64]) -> Result<f64>,
{
    sample(x, 2)?;
    sample(y, 2)?;
    let total = x.len() + y.len();
    if total > 20 {
        return Err(Error::InvalidInput(
            "exact permutation combined sample limit is 20",
        ));
    }
    let observed = match statistic(x, y)? {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }?;
    let pooled = x.iter().chain(y).copied().collect::<Vec<_>>();
    let mut partition = (0..x.len()).collect::<Vec<_>>();
    let mut null = Vec::new();
    let mut first = Vec::with_capacity(x.len());
    let mut second = Vec::with_capacity(y.len());
    loop {
        first.clear();
        second.clear();
        let mut i = 0;
        for (j, &v) in pooled.iter().enumerate() {
            if i < partition.len() && partition[i] == j {
                first.push(v);
                i += 1;
            } else {
                second.push(v);
            }
        }
        null.push(match statistic(&first, &second)? {
            value if value.is_finite() => Ok(value),
            _ => Err(crate::Error::NonFiniteEvaluation),
        }?);
        let Some(i) = (0..partition.len())
            .rev()
            .find(|&i| partition[i] < total - partition.len() + i)
        else {
            break;
        };
        partition[i] += 1;
        for j in i + 1..partition.len() {
            partition[j] = partition[j - 1] + 1;
        }
    }
    Ok(ResamplingResult {
        statistic: observed,
        pvalue: resampling_probability(observed, &null, alternative, false),
        null_distribution: null,
    })
}

/// Monte Carlo test `p=(1+# {T_null≥T_obs})/(B+1)` (greater alternative).
///
/// ## Parameters and conventions
///
/// `values` is the observed sample; `null_samples` contains independent draws
/// from a fully specified null, each of the same length. `statistic` is a
/// deterministic fallible scalar callback. SciPy's random generator is adapted
/// to explicit supplied draws, making RNG ownership and independence the caller's
/// responsibility. Less uses the lower tail; TwoSided doubles the smaller tail.
/// Equality allows 100*f64::EPSILON*|T_obs|. No vectorized/multi-sample overload.
///
/// ## Domain and special values
///
/// Finite nonempty observations and 1..=1048576 finite equal-length null draws;
/// sample_length*draw_count≤16777216. All callback results must be finite.
///
/// ## Returns
///
/// Observed statistic, plus-one-adjusted p-value and null statistics in draw order.
///
/// ## Errors
///
/// Invalid domains, incompatible draw lengths, nonfinite results or exceeded
/// work limits return errors; callback errors propagate unchanged.
///
/// ## Numerical behavior
///
/// O(B*n) validation and callback input work, O(B) result memory. Callback work
/// is additional. Probability uncertainty is Monte Carlo uncertainty, not a
/// floating-point accuracy bound; correlated or incorrectly generated draws
/// invalidate the calibration.
///
/// ## Examples
///
/// ```
/// let draws: &[&[f64]] = &[&[0.,1.], &[1.,2.], &[2.,3.]];
/// let r = sci_stats::monte_carlo_test(&[2.,3.], draws, sci_stats::mean, sci_stats::Alternative::Greater)?;
/// assert_eq!(r.pvalue, 0.5);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy monte_carlo_test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.monte_carlo_test.html).
pub fn monte_carlo_test<F>(
    values: &[f64],
    null_samples: &[&[f64]],
    statistic: F,
    alternative: Alternative,
) -> Result<ResamplingResult>
where
    F: Fn(&[f64]) -> Result<f64>,
{
    sample(values, 1)?;
    resample_work(values.len(), null_samples.len())?;
    for row in null_samples {
        if row.len() != values.len() {
            return Err(Error::DimensionMismatch);
        }
        sample(row, 1)?;
    }
    let observed = match statistic(values)? {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }?;
    let null = null_samples
        .iter()
        .map(|row| match statistic(row)? {
            value if value.is_finite() => Ok(value),
            _ => Err(crate::Error::NonFiniteEvaluation),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(ResamplingResult {
        statistic: observed,
        pvalue: resampling_probability(observed, &null, alternative, true),
        null_distribution: null,
    })
}

/// Bootstrap confidence-interval convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BootstrapMethod {
    /// Quantiles of the bootstrap statistic distribution.
    Percentile,
    /// Reflect percentile endpoints about the observed statistic.
    Basic,
}
/// Bootstrap estimate, interval and explicit resample statistics.
#[derive(Clone, Debug, PartialEq)]
pub struct BootstrapResult {
    /// Statistic of the original sample.
    pub statistic: f64,
    /// Lower confidence endpoint, in callback units.
    pub lower: f64,
    /// Upper confidence endpoint, in callback units.
    pub upper: f64,
    /// Sample standard deviation of the bootstrap statistics, ddof=1.
    pub standard_error: f64,
    /// Statistics in the supplied resampling-index order.
    pub bootstrap_distribution: Vec<f64>,
}

/// Nonparametric bootstrap interval `[Q_((1−c)/2)(T*),Q_((1+c)/2)(T*)]` (percentile).
///
/// ## Parameters and conventions
///
/// `values` is an unweighted sample; each row of `indices` specifies one
/// with-replacement resample of the original length, using zero-based indices.
/// `statistic` is a fallible scalar callback, `confidence` is c, and `method`
/// chooses percentile or basic (`[2T−Q_high,2T−Q_low]`) intervals. Uses linear
/// quantiles. This adapts SciPy RNG ownership to a supplied index schedule;
/// independent uniform index generation is the caller's responsibility.
/// BCa, paired/multi-sample, vectorized and continuation overloads are unsupported.
///
/// ## Domain and special values
///
/// Finite nonempty observations, at least two equal-length index rows,
/// 0<confidence<1; indices must be valid. n*resamples≤16777216 and resamples≤1048576.
/// All callback statistics must be finite. Constant statistics give zero standard error.
///
/// ## Returns
///
/// Original statistic, interval endpoints, sample standard error, and statistics
/// in supplied resample order. All scalar outputs use callback units.
///
/// ## Errors
///
/// Invalid domains/indices, incompatible shapes, exceeded work limits or
/// unrepresentable endpoints return errors; callback errors propagate unchanged.
///
/// ## Numerical behavior
///
/// O(B*n+B log B) validation, resample construction and quantiles; O(B+n) memory,
/// plus callback work controlled by the caller. Percentile/basic intervals can
/// have poor coverage for biased, skewed or nonsmooth estimators. Finite B adds
/// simulation uncertainty; supplied nonuniform schedules change the estimator.
///
/// ## Examples
///
/// ```
/// let indices: &[&[usize]] = &[&[0,0,1], &[0,1,2], &[1,2,2]];
/// let r = sci_stats::bootstrap(&[1.,2.,3.], indices, sci_stats::mean, 0.9, sci_stats::BootstrapMethod::Percentile)?;
/// assert_eq!(r.statistic, 2.);
/// assert_eq!(r.bootstrap_distribution.len(), 3);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).
pub fn bootstrap<F>(
    values: &[f64],
    indices: &[&[usize]],
    statistic: F,
    confidence: f64,
    method: BootstrapMethod,
) -> Result<BootstrapResult>
where
    F: Fn(&[f64]) -> Result<f64>,
{
    sample(values, 1)?;
    resample_work(values.len(), indices.len())?;
    if indices.len() < 2 || !confidence.is_finite() || confidence <= 0. || confidence >= 1. {
        return Err(Error::InvalidInput(
            "bootstrap needs at least two draws and confidence in (0,1)",
        ));
    }
    for row in indices {
        if row.len() != values.len() {
            return Err(Error::DimensionMismatch);
        }
        if row.iter().any(|&i| i >= values.len()) {
            return Err(Error::InvalidInput("bootstrap index out of bounds"));
        }
    }
    let observed = match statistic(values)? {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }?;
    let mut resample = vec![0.; values.len()];
    let mut distribution = Vec::with_capacity(indices.len());
    for row in indices {
        for (v, &i) in resample.iter_mut().zip(*row) {
            *v = values[i];
        }
        distribution.push(match statistic(&resample)? {
            value if value.is_finite() => Ok(value),
            _ => Err(crate::Error::NonFiniteEvaluation),
        }?);
    }
    let standard_error = std(&distribution, 1)?;
    let mut lower = quantile(&distribution, (1. - confidence) / 2.)?;
    let mut upper = quantile(&distribution, 0.5 + confidence / 2.)?;
    if method == BootstrapMethod::Basic {
        (lower, upper) = (
            match observed + (observed - upper) {
                value if value.is_finite() => Ok(value),
                _ => Err(crate::Error::NonFiniteEvaluation),
            }?,
            match observed + (observed - lower) {
                value if value.is_finite() => Ok(value),
                _ => Err(crate::Error::NonFiniteEvaluation),
            }?,
        );
    }
    Ok(BootstrapResult {
        statistic: observed,
        lower,
        upper,
        standard_error,
        bootstrap_distribution: distribution,
    })
}

/// Exact Baumgartner–Weiss–Schindler two-sample rank test.
///
/// ## Parameters and conventions
///
/// `x`,`y` are independent ordinal samples; pooled average ranks are assigned
/// before permutation. For sorted group ranks R_i, define
/// `B_x=(1/n)Σ (R_i−(n+m)i/n)²/[i/(n+1)*(1−i/(n+1))*m*(n+m)/n]`
/// and symmetrically B_y. TwoSided uses `(B_x+B_y)/2`, always a greater-tail
/// test. One-sided alternatives replace squared deviations by `d*abs(d)`
/// and use `(B_x−B_y)/2`, with the requested tail. SciPy's default randomized
/// dispatch is adapted to exact independent partitions only.
///
/// ## Domain and special values
///
/// At least two finite observations per independent sample, combined count≤20. Ties use average
/// ranks. All observations equal gives zero statistic and p=1.
///
/// ## Returns
///
/// Dimensionless statistic, exact p-value and null statistics in partition order.
///
/// ## Errors
///
/// Invalid samples, nonfinite values or exceeded exact-work limits return errors.
///
/// ## Numerical behavior
///
/// Reuses [`permutation_test`], evaluating at most 184756 partitions with sorting
/// each group. O(binomial(n+m,n)*(n+m) log(n+m)) work and O(binomial(n+m,n))
/// result memory. Exact probabilities use SciPy's floating-point tie tolerance.
///
/// ## Examples
///
/// ```
/// let r = sci_stats::bws_test(&[1.,2.], &[3.,4.], sci_stats::Alternative::TwoSided)?;
/// assert!(r.statistic > 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy bws_test and Neuhäuser (2005)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bws_test.html).
pub fn bws_test(x: &[f64], y: &[f64], alternative: Alternative) -> Result<ResamplingResult> {
    sample(x, 2)?;
    sample(y, 2)?;
    if x.len() + y.len() > 20 {
        return Err(Error::InvalidInput("BWS combined sample limit is 20"));
    }
    let pooled = x.iter().chain(y).copied().collect::<Vec<_>>();
    let ranks = rankdata(&pooled, RankMethod::Average)?;
    let statistic = |a: &[f64], b: &[f64]| -> Result<f64> {
        let group = |ranks: &[f64], other: usize| -> Result<f64> {
            let ordered = sorted(ranks)?;
            let n = ranks.len() as f64;
            let m = other as f64;
            Ok(ordered
                .iter()
                .enumerate()
                .map(|(i, &r)| {
                    let position = (i + 1) as f64;
                    let d = r - (m + n) / n * position;
                    let numerator = if alternative == Alternative::TwoSided {
                        d * d
                    } else {
                        d * d.abs()
                    };
                    let f = position / (n + 1.);
                    numerator / (f * (1. - f) * m * (m + n))
                })
                .sum())
        };
        let bx = group(a, b.len())?;
        let by = group(b, a.len())?;
        Ok(if alternative == Alternative::TwoSided {
            (bx + by) / 2.
        } else {
            (bx - by) / 2.
        })
    };
    permutation_test(
        &ranks[..x.len()],
        &ranks[x.len()..],
        statistic,
        if alternative == Alternative::Less {
            Alternative::Less
        } else {
            Alternative::Greater
        },
    )
}
