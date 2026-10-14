use super::*;

/// One-dimensional sample summary returned by [`describe`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Description {
    /// Number of observations.
    pub count: usize,
    /// Smallest observation.
    pub min: f64,
    /// Largest observation.
    pub max: f64,
    /// Arithmetic mean.
    pub mean: f64,
    /// Variance using the requested ddof.
    pub variance: f64,
    /// Skewness, with the requested bias convention.
    pub skewness: f64,
    /// Fisher excess kurtosis, with the requested bias convention.
    pub kurtosis: f64,
}

fn standardized_moments(x: &[f64]) -> Result<(f64, f64, f64, f64)> {
    sample(x, 1)?;
    let (_, deviations) = centered_scaled(x)?;
    let n = x.len() as f64;
    let (mut m2, mut m3, mut m4) = (0., 0., 0.);
    for d in deviations {
        m2 += d * d / n;
        m3 += d * d * d / n;
        m4 += (d * d).powi(2) / n;
    }
    Ok((n, m2, m3, m4))
}
/// Standard deviation `sqrt(Σ(x−μ)²/(n−ddof))`.
///
/// ## Parameters
///
/// `ddof` is the degrees-of-freedom correction.
///
/// ## Domain
///
/// Nonempty finite observations and `ddof < n`.
///
/// ## Returns
///
/// `f64`: standard deviation in observation units.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Evaluated in binary64; rounding and cancellation can affect ill-conditioned data.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::std(&[1., 2., 3.], 1)?, 1.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [NumPy `std`](https://numpy.org/doc/stable/reference/generated/numpy.std.html).
pub fn std(values: &[f64], ddof: usize) -> Result<f64> {
    let (scale, normalized) = scaled_variance(values, ddof)?;
    finite(normalized.sqrt() * scale)
}

/// Standard error of the mean `sqrt(var(x,ddof)/n)`.
///
/// ## Parameters
///
/// SciPy defaults to `ddof=1`; the correction is explicit here.
///
/// ## Domain
///
/// At least two finite observations and `ddof < n`.
///
/// ## Returns
///
/// `f64`: estimated standard error.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Evaluated in binary64; rounding and cancellation can affect ill-conditioned data.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!(sci_stats::sem(&[1., 2., 3.], 1)? > 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `sem`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sem.html).
pub fn sem(values: &[f64], ddof: usize) -> Result<f64> {
    sample(values, 2)?;
    Ok(std(values, ddof)? / (values.len() as f64).sqrt())
}

/// Coefficient of variation `std(x,ddof)/mean(x)`.
///
/// ## Parameters
///
/// The mean retains its sign; no percent scaling is applied.
///
/// ## Domain
///
/// Finite observations; a zero mean returns NaN for zero variance, otherwise positive infinity.
///
/// ## Returns
///
/// `f64`: dimensionless coefficient.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Evaluated in binary64; rounding and cancellation can affect ill-conditioned data.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::variation(&[2., 2.], 0)?, 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `variation`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.variation.html).
pub fn variation(values: &[f64], ddof: usize) -> Result<f64> {
    Ok(std(values, ddof)? / mean(values)?)
}

/// Linear sample quantile: interpolate sorted observations at `h=(n−1)p`.
///
/// ## Parameters
///
/// `probability` lies in `[0,1]`. Uses SciPy method="linear" only.
///
/// ## Domain
///
/// Nonempty finite observations; endpoints return minimum and maximum.
///
/// ## Returns
///
/// `f64`: quantile in observation units.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Sorting takes O(n log n) work; convex interpolation avoids subtracting distant endpoints.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::quantile(&[1., 3.], 0.5)?, 2.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `quantile`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.quantile.html).
pub fn quantile(values: &[f64], probability: f64) -> Result<f64> {
    probability_check(probability)?;
    let x = sorted(values)?;
    Ok(interpolate(&x, probability))
}

/// Sample percentile `quantile(x, percentile/100)`.
///
/// ## Parameters
///
/// `percentile` is in `[0,100]`; linear interpolation only; no range filtering.
///
/// ## Domain
///
/// Nonempty finite data; out-of-range percentiles are errors.
///
/// ## Returns
///
/// `f64`: percentile in observation units.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Evaluated in binary64; rounding and cancellation can affect ill-conditioned data.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::scoreatpercentile(&[1., 3.], 50.)?, 2.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `scoreatpercentile`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.scoreatpercentile.html).
pub fn scoreatpercentile(values: &[f64], percentile: f64) -> Result<f64> {
    quantile(values, percentile / 100.)
}

/// Median `quantile(x,1/2)`.
///
/// ## Parameters
///
/// For even counts, averages the two middle sorted observations.
///
/// ## Domain
///
/// Nonempty finite observations.
///
/// ## Returns
///
/// `f64`: sample median.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Evaluated in binary64; rounding and cancellation can affect ill-conditioned data.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::median(&[1., 3., 2.])?, 2.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [NumPy `median`](https://numpy.org/doc/stable/reference/generated/numpy.median.html).
pub fn median(values: &[f64]) -> Result<f64> {
    quantile(values, 0.5)
}

/// Interquartile range `Q(3/4)−Q(1/4)`.
///
/// ## Parameters
///
/// Linear interpolation; no normal-distribution scaling or custom percentile range.
///
/// ## Domain
///
/// Nonempty finite observations.
///
/// ## Returns
///
/// `f64`: nonnegative range in observation units.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Evaluated in binary64; rounding and cancellation can affect ill-conditioned data.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::iqr(&[1., 2., 3., 4.])?, 1.5);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `iqr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.iqr.html).
pub fn iqr(values: &[f64]) -> Result<f64> {
    let x = sorted(values)?;
    finite(interpolate(&x, 0.75) - interpolate(&x, 0.25))
}

/// Median absolute deviation `median(|x−median(x)|)/scale`.
///
/// ## Parameters
///
/// `scale` is finite and positive; use 0.6744897501960817 for normal consistency.
///
/// ## Domain
///
/// Nonempty finite observations.
///
/// ## Returns
///
/// `f64`: scaled absolute deviation.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Evaluated in binary64; rounding and cancellation can affect ill-conditioned data.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::median_abs_deviation(&[1., 2., 9.], 1.)?, 1.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `median_abs_deviation`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html).
pub fn median_abs_deviation(values: &[f64], scale: f64) -> Result<f64> {
    positive(scale)?;
    let center = median(values)?;
    let distances = values
        .iter()
        .map(|x| (x - center).abs())
        .collect::<Vec<_>>();
    finite(median(&distances)? / scale)
}

/// Geometric mean `exp(Σ ln(x)/n)`.
///
/// ## Parameters
///
/// Unweighted one-dimensional data.
///
/// ## Domain
///
/// Nonnegative finite observations; any zero returns zero.
///
/// ## Returns
///
/// `f64`: geometric mean.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Evaluated in binary64; rounding and cancellation can affect ill-conditioned data.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!((sci_stats::gmean(&[1., 4.])? - 2.).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `gmean`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gmean.html).
pub fn gmean(values: &[f64]) -> Result<f64> {
    sample(values, 1)?;
    if values.iter().any(|x| *x < 0.) {
        return Err(Error::InvalidInput(
            "geometric mean requires nonnegative values",
        ));
    }
    if values.contains(&0.) {
        return Ok(0.);
    }
    let logs = values.iter().map(|x| x.ln()).collect::<Vec<_>>();
    finite(mean(&logs)?.exp())
}

/// Harmonic mean `n/Σ(1/x)`.
///
/// ## Parameters
///
/// Unweighted observations.
///
/// ## Domain
///
/// Nonnegative finite observations; any zero returns zero.
///
/// ## Returns
///
/// `f64`: harmonic mean.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Reciprocals are scaled by the minimum to avoid overflow for subnormal inputs.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::hmean(&[1., 1.])?, 1.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `hmean`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hmean.html).
pub fn hmean(values: &[f64]) -> Result<f64> {
    sample(values, 1)?;
    if values.iter().any(|x| *x < 0.) {
        return Err(Error::InvalidInput(
            "harmonic mean requires nonnegative values",
        ));
    }
    let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
    if minimum == 0. {
        return Ok(0.);
    }
    finite(minimum / (values.iter().map(|x| minimum / x).sum::<f64>() / values.len() as f64))
}

/// Geometric standard deviation `exp(std(ln(x),ddof))`.
///
/// ## Parameters
///
/// SciPy defaults to `ddof=1`; correction is explicit.
///
/// ## Domain
///
/// Strictly positive finite observations and `ddof < n`.
///
/// ## Returns
///
/// `f64`: dimensionless multiplicative spread.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Evaluated in binary64; rounding and cancellation can affect ill-conditioned data.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::gstd(&[2., 2.], 1)?, 1.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `gstd`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gstd.html).
pub fn gstd(values: &[f64], ddof: usize) -> Result<f64> {
    sample(values, 1)?;
    if values.iter().any(|x| *x <= 0.) {
        return Err(Error::InvalidInput(
            "geometric standard deviation requires positive values",
        ));
    }
    finite(std(&values.iter().map(|x| x.ln()).collect::<Vec<_>>(), ddof)?.exp())
}

/// Central moment `m_k=Σ(x−mean(x))^k/n`.
///
/// ## Parameters
///
/// Order is an integer from 0 through 64; normalized by n.
///
/// ## Domain
///
/// Nonempty finite data; orders zero and one return one and zero.
///
/// ## Returns
///
/// `f64`: central moment, in observation units raised to `order`.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Shifted, scaled central deviations preserve small differences at large
/// offsets and avoid unnecessary power overflow. Scaling precedes centering
/// only when subtraction overflows. Binary64 rounding and underflow remain.
/// Inputs are limited to 1,048,576 observations and order to 64.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::moment(&[1., 2., 3.], 1)?, 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `moment`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.moment.html).
pub fn moment(values: &[f64], order: usize) -> Result<f64> {
    sample(values, 1)?;
    if order > 64 {
        return Err(Error::InvalidInput("moment order exceeds 64"));
    }
    if order == 0 {
        return Ok(1.);
    }
    if order == 1 {
        return Ok(0.);
    }
    let (scale, deviations) = centered_scaled(values)?;
    if scale == 0. {
        return Ok(0.);
    }
    let terms = deviations
        .iter()
        .map(|x| x.powi(order as i32))
        .collect::<Vec<_>>();
    let mut value = mean(&terms)?;
    for _ in 0..order {
        value = finite(value * scale)?;
    }
    Ok(value)
}

/// Skewness `g₁=m₃/m₂^(3/2)`, corrected by `sqrt(n(n−1))/(n−2)` when unbiased.
///
/// ## Parameters
///
/// `bias=true` uses population moments; `false` corrects when n>2, as in SciPy.
///
/// ## Domain
///
/// Constant data returns NaN; finite nonempty observations are required.
///
/// ## Returns
///
/// `f64`: dimensionless skewness.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Moments are evaluated after scaling observations to prevent avoidable overflow.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!(sci_stats::skew(&[1., 2., 3.], true)?.abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `skew`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html).
pub fn skew(values: &[f64], bias: bool) -> Result<f64> {
    let (n, m2, m3, _) = standardized_moments(values)?;
    if m2 == 0. {
        return Ok(f64::NAN);
    }
    let g = m3 / m2.powf(1.5);
    Ok(if !bias && n > 2. {
        g * (n * (n - 1.)).sqrt() / (n - 2.)
    } else {
        g
    })
}

/// Kurtosis `m₄/m₂²`; Fisher excess subtracts three.
///
/// ## Parameters
///
/// `fisher=true` returns excess. For n>3, `bias=false` uses the k-statistic correction.
///
/// ## Domain
///
/// Constant data returns NaN; finite nonempty observations are required.
///
/// ## Returns
///
/// `f64`: dimensionless kurtosis.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Scaled moments avoid overflow; high-order statistics remain sensitive to outliers.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!((sci_stats::kurtosis(&[1., 2., 3.], true, true)? + 1.5).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `kurtosis`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html).
pub fn kurtosis(values: &[f64], fisher: bool, bias: bool) -> Result<f64> {
    let (n, m2, _, m4) = standardized_moments(values)?;
    if m2 == 0. {
        return Ok(f64::NAN);
    }
    let mut k = m4 / (m2 * m2);
    if !bias && n > 3. {
        k = ((n * n - 1.) * k - 3. * (n - 1.).powi(2)) / ((n - 2.) * (n - 3.)) + 3.;
    }
    Ok(if fisher { k - 3. } else { k })
}

/// Standardized scores `z_i=(x_i−mean(x))/std(x,ddof)`.
///
/// ## Parameters
///
/// Output preserves input order; explicit ddof; no axes or NaN omission.
///
/// ## Domain
///
/// Constant data produces NaN scores. Finite data and `ddof < n` are required.
///
/// ## Returns
///
/// `Vec<f64>`: one dimensionless score per observation.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Shifts observations before scaling when possible; opposite extremes are
/// scaled before centering. The mean and standard deviation stay in scaled
/// coordinates, avoiding intermediate overflow and subnormal rounding. Binary64
/// rounding can still affect nearly equal scores.
/// Inputs are limited to 1,048,576 observations per slice; work is O(n).
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::zscore(&[1., 2., 3.], 1)?, vec![-1., 0., 1.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `zscore`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zscore.html).
pub fn zscore(values: &[f64], ddof: usize) -> Result<Vec<f64>> {
    zmap(values, values, ddof)
}

/// Relative z-scores `(scores_i−mean(compare))/std(compare,ddof)`.
///
/// ## Parameters
///
/// `scores` and `compare` are one-dimensional slices; their lengths may differ.
/// Output follows `scores`. `ddof` is the degrees-of-freedom correction for
/// the reference spread; zero gives population normalization. SciPy axis,
/// masked-array, and NaN-omission options are unsupported.
///
/// ## Domain
///
/// `compare` must be nonempty, with `ddof < compare.len()`. Both slices must
/// contain finite values. Empty scores return an empty vector after reference
/// validation. A constant reference yields NaN for an equal score and signed
/// infinity otherwise, preserving Sci's zero-spread convention.
///
/// ## Returns
///
/// `Vec<f64>`: standardized scores in original order.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Centers and scales the reference sample, then maps scores into the same
/// coordinates. Normalized division avoids intermediate spread overflow and
/// subnormal rounding. A nonconstant reference with an unrepresentable final
/// score returns an evaluation error. Binary64 cancellation can still affect
/// scores close to the reference mean.
/// Inputs are limited to 1,048,576 observations per slice; work is O(n).
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::zmap(&[2.], &[1., 2., 3.], 1)?, vec![0.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `zmap`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zmap.html).
pub fn zmap(scores: &[f64], compare: &[f64], ddof: usize) -> Result<Vec<f64>> {
    sample(scores, 0)?;
    if ddof >= compare.len() {
        return Err(Error::InvalidInput(
            "ddof must be smaller than the observation count",
        ));
    }
    let reference = centered_sample(compare)?;
    if reference.scale == 0. {
        return Ok(scores
            .iter()
            .map(|x| (*x - reference.origin) / 0.)
            .collect());
    }
    let spread = normalized_variance(&reference.deviations, ddof).sqrt();
    scores
        .iter()
        .map(|x| {
            let delta = *x - reference.origin;
            let coordinate = if delta.is_finite() {
                delta / reference.scale
            } else {
                x / reference.scale - reference.origin / reference.scale
            };
            let score = (coordinate - reference.center) / spread;
            finite(if score.is_finite() {
                score
            } else {
                // A coordinate may overflow although division by a spread
                // above one makes the final score representable.
                let displacement = if delta.is_finite() {
                    (delta / spread) / reference.scale
                } else {
                    (x / spread) / reference.scale - (reference.origin / spread) / reference.scale
                };
                displacement - reference.center / spread
            })
        })
        .collect()
}

/// Geometric z-scores `zscore(ln(x),ddof)`.
///
/// ## Parameters
///
/// Output is ordered as input; ddof applies to log-space variance.
///
/// ## Domain
///
/// Strictly positive finite data; constant values produce NaN scores.
///
/// ## Returns
///
/// `Vec<f64>`: dimensionless log-space scores.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Applies the scaled centering used by `zscore` to binary64 logarithms.
/// Logarithm rounding can make distinct observations equal in log space;
/// this yields NaN scores when all logarithms coincide.
/// Inputs are limited to 1,048,576 observations per slice; work is O(n).
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::gzscore(&[1., 2., 4.], 1)?[1], 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `gzscore`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gzscore.html).
pub fn gzscore(values: &[f64], ddof: usize) -> Result<Vec<f64>> {
    sample(values, 1)?;
    if values.iter().any(|x| *x <= 0.) {
        return Err(Error::InvalidInput(
            "geometric scores require positive values",
        ));
    }
    zscore(&values.iter().map(|x| x.ln()).collect::<Vec<_>>(), ddof)
}

/// Removes `floor(n*proportion)` observations from each sorted tail.
///
/// ## Parameters
///
/// The proportion removed is per tail; output is sorted, unlike SciPy's unspecified order.
///
/// ## Domain
///
/// Finite data; proportion in `[0,0.5)`. Empty input returns empty output.
///
/// ## Returns
///
/// `Vec<f64>`: retained observations, ascending.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Evaluated in binary64; rounding and cancellation can affect ill-conditioned data.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::trimboth(&[4., 2., 1., 3.], 0.25)?, vec![2., 3.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `trimboth`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.trimboth.html).
pub fn trimboth(values: &[f64], proportion: f64) -> Result<Vec<f64>> {
    sample(values, 0)?;
    if !proportion.is_finite() || !(0. ..0.5).contains(&proportion) {
        return Err(Error::InvalidInput("trim proportion must be in [0,0.5)"));
    }
    let mut x = values.to_vec();
    x.sort_by(f64::total_cmp);
    let k = (x.len() as f64 * proportion).floor() as usize;
    Ok(x[k..x.len() - k].to_vec())
}

/// Trimmed mean of the middle `n−2 floor(n*proportion)` observations.
///
/// ## Parameters
///
/// Trimming proportion is per tail.
///
/// ## Domain
///
/// Nonempty finite observations; proportion in `[0,0.5)`.
///
/// ## Returns
///
/// `f64`: arithmetic mean after sorting and trimming.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Evaluated in binary64; rounding and cancellation can affect ill-conditioned data.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::trim_mean(&[1., 2., 3., 100.], 0.25)?, 2.5);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `trim_mean`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.trim_mean.html).
pub fn trim_mean(values: &[f64], proportion: f64) -> Result<f64> {
    mean(&trimboth(values, proportion)?)
}

/// Mode: observation with the greatest multiplicity.
///
/// ## Parameters
///
/// Ties choose the smallest value; +0 and −0 are one value.
///
/// ## Domain
///
/// Nonempty finite data.
///
/// ## Returns
///
/// `(f64, usize)`: modal observation and its count.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Sorting takes O(n log n) work; counts are exact integers.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::mode(&[2., 1., 2.])?, (2., 2));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `mode`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html).
pub fn mode(values: &[f64]) -> Result<(f64, usize)> {
    let x = sorted(values)?;
    let mut best = (x[0], 1);
    let mut start = 0;
    while start < x.len() {
        let mut end = start + 1;
        while end < x.len() && x[end] == x[start] {
            end += 1;
        }
        if end - start > best.1 {
            best = (x[start], end - start);
        }
        start = end;
    }
    Ok(best)
}

/// Sample summary: count, extrema, arithmetic mean, variance, skewness, and Fisher excess kurtosis.
///
/// ## Parameters
///
/// Explicit variance ddof and moment bias correction; one-dimensional data.
///
/// ## Domain
///
/// Nonempty finite data; constant data has NaN skewness and kurtosis.
///
/// ## Returns
///
/// `Description`: named summary fields.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Evaluated in binary64; rounding and cancellation can affect ill-conditioned data.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::describe(&[1., 2., 3.], 1, true)?.count, 3);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `describe`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.describe.html).
pub fn describe(values: &[f64], ddof: usize, bias: bool) -> Result<Description> {
    sample(values, 1)?;
    Ok(Description {
        count: values.len(),
        min: values.iter().copied().fold(f64::INFINITY, f64::min),
        max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        mean: mean(values)?,
        variance: variance(values, ddof)?,
        skewness: skew(values, bias)?,
        kurtosis: kurtosis(values, true, bias)?,
    })
}
