use super::*;
use sci_core::Float;

/// One-dimensional sample summary returned by [`describe`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Description<T = f64> {
    /// Number of observations.
    pub count: usize,
    /// Smallest observation.
    pub min: T,
    /// Largest observation.
    pub max: T,
    /// Arithmetic mean.
    pub mean: T,
    /// Variance using the requested ddof.
    pub variance: T,
    /// Skewness, with the requested bias convention.
    pub skewness: T,
    /// Fisher excess kurtosis, with the requested bias convention.
    pub kurtosis: T,
}

#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
fn standardized_moments<T: Float>(x: &[T]) -> Result<(T, T, T, T)> {
    sample(x, 1)?;
    let (_, deviations) = centered_scaled(x)?;
    let n = T::from(x.len()).unwrap();
    let (mut m2, mut m3, mut m4) = (0., 0., 0.);
    for d in deviations {
        m2 = m2 + (d * d / n);
        m3 = m3 + (d * d * d / n);
        m4 = m4 + ((d * d).powi(2) / n);
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
/// `T`: standard deviation in observation units.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Evaluated in the selected floating-point precision; rounding and cancellation can affect ill-conditioned data.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::descriptive::std(&[1., 2., 3.], 1)?, 1.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [NumPy `std`](https://numpy.org/doc/stable/reference/generated/numpy.std.html).
pub fn std<T: Float>(values: &[T], ddof: usize) -> Result<T> {
    let (scale, normalized) = scaled_variance(values, ddof)?;
    match normalized.sqrt() * scale {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }
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
/// `T`: estimated standard error.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Evaluated in the selected floating-point precision; rounding and cancellation can affect ill-conditioned data.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!(sci_stats::descriptive::sem(&[1., 2., 3.], 1)? > 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `sem`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sem.html).
pub fn sem<T: Float>(values: &[T], ddof: usize) -> Result<T> {
    sample(values, 2)?;
    Ok(std(values, ddof)? / (T::from(values.len()).unwrap()).sqrt())
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
/// `T`: dimensionless coefficient.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Evaluated in the selected floating-point precision; rounding and cancellation can affect ill-conditioned data.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::descriptive::variation(&[2., 2.], 0)?, 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `variation`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.variation.html).
pub fn variation<T: Float>(values: &[T], ddof: usize) -> Result<T> {
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
/// `T`: quantile in observation units.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Selection takes expected O(n) work with O(n) scratch memory. Already ordered
/// inputs are evaluated directly. Convex interpolation avoids subtracting distant endpoints.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::descriptive::quantile(&[1., 3.], 0.5)?, 2.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `quantile`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.quantile.html).
pub fn quantile<T: Float>(values: &[T], probability: T) -> Result<T> {
    probability_check(probability)?;
    sample(values, 1)?;
    // The same total order as sorted() distinguishes -0 from +0.
    // Monotone samples are common and need no allocation or selection.
    if values
        .windows(2)
        .all(|pair| total_cmp(&pair[0], &pair[1]).is_le())
    {
        return Ok(interpolate(values, probability));
    }
    if values
        .windows(2)
        .all(|pair| total_cmp(&pair[0], &pair[1]).is_ge())
    {
        let h = T::from(values.len() - 1).unwrap() * probability;
        let i = h.floor().to_usize().unwrap();
        let j = h.ceil().to_usize().unwrap();
        if i == j {
            return Ok(values[values.len() - 1 - i]);
        }
        let f = h - T::from(i).unwrap();
        return Ok((T::from(1.).unwrap() - f) * values[values.len() - 1 - i]
            + f * values[values.len() - 1 - j]);
    }
    let mut scratch = values.to_vec();
    Ok(quantile_select_in_place(&mut scratch, probability))
}

/// Quantile of a nonempty, finite scratch slice. Keep exactly interpolate()'s
/// index calculation and convex arithmetic, but select only the two neighbors.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
fn quantile_select_in_place<T: Float>(values: &mut [T], probability: T) -> T {
    let h = T::from(values.len() - 1).unwrap() * probability;
    let i = h.floor().to_usize().unwrap();
    let j = h.ceil().to_usize().unwrap();
    if i == j {
        return *values.select_nth_unstable_by(i, total_cmp).1;
    }
    // The adjacent neighbor is an extremum of the smaller partition.
    // Scanning it is cheaper than a second partitioning pass.
    let (lower, upper) = if j < values.len() - i {
        let (prefix, selected, _) = values.select_nth_unstable_by(j, total_cmp);
        (prefix.iter().copied().max_by(total_cmp).unwrap(), *selected)
    } else {
        let (_, selected, suffix) = values.select_nth_unstable_by(i, total_cmp);
        (*selected, suffix.iter().copied().min_by(total_cmp).unwrap())
    };
    let f = h - T::from(i).unwrap();
    (1. - f) * lower + f * upper
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
/// `T`: percentile in observation units.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Evaluated in the selected floating-point precision; rounding and cancellation can affect ill-conditioned data.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::descriptive::scoreatpercentile(&[1., 3.], 50.)?, 2.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `scoreatpercentile`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.scoreatpercentile.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn scoreatpercentile<T: Float>(values: &[T], percentile: T) -> Result<T> {
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
/// `T`: sample median.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Expected O(n) selection with O(n) scratch memory; the input is not changed.
/// Even counts use the same convex interpolation as `quantile(x, 0.5)`.
/// Evaluated in the selected floating-point precision; rounding and cancellation
/// can affect ill-conditioned data. Inputs are limited to 1,048,576 observations.
/// No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::descriptive::median(&[1., 3., 2.])?, 2.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [NumPy `median`](https://numpy.org/doc/stable/reference/generated/numpy.median.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn median<T: Float>(values: &[T]) -> Result<T> {
    sample(values, 1)?;
    let mut scratch = values.to_vec();
    Ok(median_select_in_place(&mut scratch))
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
/// `T`: nonnegative range in observation units.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Evaluated in the selected floating-point precision; rounding and cancellation can affect ill-conditioned data.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::descriptive::iqr(&[1., 2., 3., 4.])?, 1.5);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `iqr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.iqr.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn iqr<T: Float>(values: &[T]) -> Result<T> {
    let x = sorted(values)?;
    match interpolate(&x, 0.75) - interpolate(&x, 0.25) {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }
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
/// `T`: scaled absolute deviation.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Evaluated in the selected floating-point precision; rounding and cancellation can affect ill-conditioned data.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::descriptive::median_abs_deviation(&[1., 2., 9.], 1.)?, 1.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `median_abs_deviation`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html).
pub fn median_abs_deviation<T: Float>(values: &[T], scale: T) -> Result<T> {
    positive(scale)?;
    let center = median(values)?;
    let mut distances = values
        .iter()
        .map(|&x| (x - center).abs())
        .collect::<Vec<_>>();
    // The deviations are owned scratch storage. Select their median in place
    // rather than making a second copy inside median().
    sample(&distances, 1)?;
    match median_select_in_place(&mut distances) / scale {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }
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
/// `T`: geometric mean.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Evaluated in the selected floating-point precision; rounding and cancellation can affect ill-conditioned data.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!((sci_stats::descriptive::gmean(&[1.0_f64, 4.])? - 2.).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `gmean`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gmean.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn gmean<T: Float>(values: &[T]) -> Result<T> {
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
    match mean(&logs)?.exp() {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }
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
/// `T`: harmonic mean.
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
/// assert_eq!(sci_stats::descriptive::hmean(&[1., 1.])?, 1.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `hmean`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hmean.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn hmean<T: Float>(values: &[T]) -> Result<T> {
    sample(values, 1)?;
    if values.iter().any(|x| *x < 0.) {
        return Err(Error::InvalidInput(
            "harmonic mean requires nonnegative values",
        ));
    }
    let minimum = values.iter().copied().fold(T::infinity(), T::min);
    if minimum == 0. {
        return Ok(0.);
    }
    match minimum
        / (values
            .iter()
            .map(|&x| minimum / x)
            .fold(T::zero(), |a, b| a + b)
            / T::from(values.len()).unwrap())
    {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }
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
/// `T`: dimensionless multiplicative spread.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Evaluated in the selected floating-point precision; rounding and cancellation can affect ill-conditioned data.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::descriptive::gstd(&[2., 2.], 1)?, 1.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `gstd`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gstd.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn gstd<T: Float>(values: &[T], ddof: usize) -> Result<T> {
    sample(values, 1)?;
    if values.iter().any(|x| *x <= 0.) {
        return Err(Error::InvalidInput(
            "geometric standard deviation requires positive values",
        ));
    }
    match std(&values.iter().map(|x| x.ln()).collect::<Vec<_>>(), ddof)?.exp() {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }
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
/// `T`: central moment, in observation units raised to `order`.
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
/// assert_eq!(sci_stats::descriptive::moment(&[1., 2., 3.], 1)?, 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `moment`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.moment.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn moment<T: Float>(values: &[T], order: usize) -> Result<T> {
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
        value = match value * scale {
            value if value.is_finite() => Ok(value),
            _ => Err(crate::Error::NonFiniteEvaluation),
        }?;
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
/// `T`: dimensionless skewness.
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
/// assert!(sci_stats::descriptive::skew(&[1.0_f64, 2., 3.], true)?.abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `skew`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn skew<T: Float>(values: &[T], bias: bool) -> Result<T> {
    let (n, m2, m3, _) = standardized_moments(values)?;
    Ok(skew_from_moments(n, m2, m3, bias))
}

#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
fn skew_from_moments<T: Float>(n: T, m2: T, m3: T, bias: bool) -> T {
    if m2 == 0. {
        return T::nan();
    }
    let g = m3 / m2.powf(1.5);
    if !bias && n > 2. {
        g * (n * (n - 1.)).sqrt() / (n - 2.)
    } else {
        g
    }
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
/// `T`: dimensionless kurtosis.
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
/// assert!((sci_stats::descriptive::kurtosis(&[1.0_f64, 2., 3.], true, true)? + 1.5).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `kurtosis`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn kurtosis<T: Float>(values: &[T], fisher: bool, bias: bool) -> Result<T> {
    let (n, m2, _, m4) = standardized_moments(values)?;
    Ok(kurtosis_from_moments(n, m2, m4, fisher, bias))
}

#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
fn kurtosis_from_moments<T: Float>(n: T, m2: T, m4: T, fisher: bool, bias: bool) -> T {
    if m2 == 0. {
        return T::nan();
    }
    let mut k = m4 / (m2 * m2);
    if !bias && n > 3. {
        k = ((n * n - 1.) * k - 3. * (n - 1.).powi(2)) / ((n - 2.) * (n - 3.)) + 3.;
    }
    if fisher { k - 3. } else { k }
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
/// `Vec<T>`: one dimensionless score per observation.
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
/// assert_eq!(sci_stats::descriptive::zscore(&[1., 2., 3.], 1)?, vec![-1., 0., 1.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `zscore`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zscore.html).
pub fn zscore<T: Float>(values: &[T], ddof: usize) -> Result<Vec<T>> {
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
/// `Vec<T>`: standardized scores in original order.
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
/// assert_eq!(sci_stats::descriptive::zmap(&[2.], &[1., 2., 3.], 1)?, vec![0.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `zmap`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zmap.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn zmap<T: Float>(scores: &[T], compare: &[T], ddof: usize) -> Result<Vec<T>> {
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
                *x / reference.scale - reference.origin / reference.scale
            };
            let score = (coordinate - reference.center) / spread;
            match if score.is_finite() {
                score
            } else {
                // A coordinate may overflow although division by a spread
                // above one makes the final score representable.
                let displacement = if delta.is_finite() {
                    (delta / spread) / reference.scale
                } else {
                    (*x / spread) / reference.scale - (reference.origin / spread) / reference.scale
                };
                displacement - reference.center / spread
            } {
                value if value.is_finite() => Ok(value),
                _ => Err(crate::Error::NonFiniteEvaluation),
            }
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
/// `Vec<T>`: dimensionless log-space scores.
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
/// assert_eq!(sci_stats::descriptive::gzscore(&[1., 2., 4.], 1)?[1], 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `gzscore`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gzscore.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn gzscore<T: Float>(values: &[T], ddof: usize) -> Result<Vec<T>> {
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
/// `Vec<T>`: retained observations, ascending.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Evaluated in the selected floating-point precision; rounding and cancellation can affect ill-conditioned data.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::descriptive::trimboth(&[4., 2., 1., 3.], 0.25)?, vec![2., 3.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `trimboth`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.trimboth.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn trimboth<T: Float>(values: &[T], proportion: T) -> Result<Vec<T>> {
    sample(values, 0)?;
    if !proportion.is_finite() || !(0. ..0.5).contains(&proportion) {
        return Err(Error::InvalidInput("trim proportion must be in [0,0.5)"));
    }
    let mut x = values.to_vec();
    x.sort_by(total_cmp);
    let k = (T::from(x.len()).unwrap() * proportion)
        .floor()
        .to_usize()
        .unwrap();
    x.truncate(x.len() - k);
    x.drain(..k);
    Ok(x)
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
/// `T`: arithmetic mean after sorting and trimming.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Evaluated in the selected floating-point precision; rounding and cancellation can affect ill-conditioned data.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::descriptive::trim_mean(&[1., 2., 3., 100.], 0.25)?, 2.5);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `trim_mean`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.trim_mean.html).
pub fn trim_mean<T: Float>(values: &[T], proportion: T) -> Result<T> {
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
/// `(T, usize)`: modal observation and its count.
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
/// assert_eq!(sci_stats::descriptive::mode(&[2., 1., 2.])?, (2., 2));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `mode`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html).
pub fn mode<T: Float>(values: &[T]) -> Result<(T, usize)> {
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
/// Evaluated in the selected floating-point precision; rounding and cancellation can affect ill-conditioned data.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::descriptive::describe(&[1., 2., 3.], 1, true)?.count, 3);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `describe`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.describe.html).
pub fn describe<T: Float>(values: &[T], ddof: usize, bias: bool) -> Result<Description<T>> {
    sample(values, 1)?;
    let min = values.iter().copied().fold(T::infinity(), T::min);
    let max = values.iter().copied().fold(T::neg_infinity(), T::max);
    let mean = mean(values)?;
    let variance = variance(values, ddof)?;
    // Skewness and kurtosis require the same three standardized moments.
    // Reuse their common calculation without changing either formula.
    let (n, m2, m3, m4) = standardized_moments(values)?;
    Ok(Description {
        count: values.len(),
        min,
        max,
        mean,
        variance,
        skewness: skew_from_moments(n, m2, m3, bias),
        kurtosis: kurtosis_from_moments(n, m2, m4, true, bias),
    })
}

#[cfg(test)]
mod quantile_selection_tests {
    use super::*;

    fn check<T: Float>(values: &[T], probabilities: &[T]) {
        let input_bits: Vec<_> = values
            .iter()
            .map(|v| v.to_f64().unwrap().to_bits())
            .collect();
        let mut reference = values.to_vec();
        reference.sort_by(total_cmp);
        for &p in probabilities {
            let expected = interpolate(&reference, p).to_f64().unwrap().to_bits();
            let actual = quantile(values, p).unwrap().to_f64().unwrap().to_bits();
            assert_eq!(actual, expected, "n={}, p={:?}", values.len(), p.to_f64());

            // Exercise the selection path even when the public API recognizes
            // already sorted input and skips partitioning.
            let mut scratch = values.to_vec();
            let selected = quantile_select_in_place(&mut scratch, p)
                .to_f64()
                .unwrap()
                .to_bits();
            assert_eq!(
                selected,
                expected,
                "selection n={}, p={:?}",
                values.len(),
                p.to_f64()
            );
        }
        assert_eq!(
            values
                .iter()
                .map(|v| v.to_f64().unwrap().to_bits())
                .collect::<Vec<_>>(),
            input_bits,
            "quantile must leave its input unchanged"
        );
    }

    #[test]
    fn quantile_selection_matches_sorted_bits_f64() {
        let probabilities = [
            -0.0,
            0.0,
            f64::from_bits(1),
            f64::EPSILON,
            0.001,
            0.1,
            0.3,
            0.5,
            0.7,
            0.9,
            0.999,
            f64::from_bits(1.0_f64.to_bits() - 1),
            1.0,
        ];
        for values in [
            vec![-0.0],
            vec![0.0, -0.0],
            vec![-0.0, 0.0, -0.0, 0.0],
            vec![f64::MIN_POSITIVE, f64::from_bits(1)],
            vec![-f64::MAX, -1.0, -0.0, 0.0, 1.0, f64::MAX],
            vec![-f64::MAX, f64::MAX],
            vec![f64::MAX, f64::MAX],
            vec![3.0; 17],
        ] {
            check(&values, &probabilities);
        }
        for n in 1..=128 {
            let mut values = (0..n)
                .map(|i| ((i * 37 + 11) % 23) as f64 - 11.0)
                .collect::<Vec<_>>();
            check(&values, &probabilities);
            values.sort_by(f64::total_cmp);
            check(&values, &probabilities);
            values.reverse();
            check(&values, &probabilities);
        }
        for n in [127, 128, 255, 256, 2048, 8192] {
            let mut state = 0x245e_91b3_82ae_99dfu64;
            let values = (0..n)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 11) as f64 / (1_u64 << 53) as f64
                })
                .collect::<Vec<_>>();
            check(&values, &probabilities);
        }
    }

    #[test]
    fn quantile_selection_matches_sorted_bits_f32() {
        let probabilities = [
            -0.0_f32,
            0.0,
            f32::from_bits(1),
            f32::EPSILON,
            0.001,
            0.1,
            0.3,
            0.5,
            0.7,
            0.9,
            0.999,
            f32::from_bits(1.0_f32.to_bits() - 1),
            1.0,
        ];
        for values in [
            vec![-0.0_f32],
            vec![0.0_f32, -0.0],
            vec![-0.0_f32, 0.0, -0.0, 0.0],
            vec![f32::MIN_POSITIVE, f32::from_bits(1)],
            vec![-f32::MAX, -1.0, -0.0, 0.0, 1.0, f32::MAX],
            vec![-f32::MAX, f32::MAX],
            vec![f32::MAX, f32::MAX],
            vec![3.0_f32; 17],
        ] {
            check(&values, &probabilities);
        }
        for n in 1..=128 {
            let mut values = (0..n)
                .map(|i| ((i * 37 + 11) % 23) as f32 - 11.0)
                .collect::<Vec<_>>();
            check(&values, &probabilities);
            values.sort_by(f32::total_cmp);
            check(&values, &probabilities);
            values.reverse();
            check(&values, &probabilities);
        }
        for n in [127, 128, 255, 256, 2048, 8192] {
            let mut state = 0x245e_91b3_82ae_99dfu64;
            let values = (0..n)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    ((state >> 11) as f64 / (1_u64 << 53) as f64) as f32
                })
                .collect::<Vec<_>>();
            check(&values, &probabilities);
        }
    }

    #[test]
    fn quantile_selection_preserves_validation_and_endpoints() {
        assert!(quantile::<f64>(&[], 0.5).is_err());
        assert!(quantile::<f32>(&[], 0.5).is_err());
        for invalid in [f64::NAN, f64::NEG_INFINITY, f64::INFINITY, -0.1, 1.1] {
            assert!(quantile(&[0.0, 1.0], invalid).is_err());
        }
        for invalid in [f32::NAN, f32::NEG_INFINITY, f32::INFINITY, -0.1, 1.1] {
            assert!(quantile(&[0.0_f32, 1.0], invalid).is_err());
        }
        for invalid in [f64::NAN, f64::NEG_INFINITY, f64::INFINITY] {
            assert!(quantile(&[0.0, invalid], 0.3).is_err());
        }
        for invalid in [f32::NAN, f32::NEG_INFINITY, f32::INFINITY] {
            assert!(quantile(&[0.0_f32, invalid], 0.3).is_err());
        }
        assert!(quantile(&vec![1.0_f64; MAX_SAMPLES + 1], 0.3).is_err());
        let x: [f64; 4] = [0.0, -0.0, -1.0, 1.0];
        assert_eq!(quantile(&x, 0.0).unwrap().to_bits(), (-1.0_f64).to_bits());
        assert_eq!(quantile(&x, 1.0).unwrap().to_bits(), 1.0_f64.to_bits());
    }
}

#[cfg(test)]
mod median_selection_tests {
    use super::*;

    fn check_f64(values: &[f64]) {
        let original_bits = values.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let expected = interpolate(&sorted, 0.5).to_bits();
        let mut selected = values.to_vec();
        assert_eq!(median_select_in_place(&mut selected).to_bits(), expected);
        assert_eq!(median(values).unwrap().to_bits(), expected);
        assert_eq!(
            values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            original_bits
        );
    }

    fn check_f32(values: &[f32]) {
        let original_bits = values.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let mut sorted = values.to_vec();
        sorted.sort_by(f32::total_cmp);
        let expected = interpolate(&sorted, 0.5).to_bits();
        let mut selected = values.to_vec();
        assert_eq!(median_select_in_place(&mut selected).to_bits(), expected);
        assert_eq!(median(values).unwrap().to_bits(), expected);
        assert_eq!(
            values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            original_bits
        );
    }

    #[test]
    fn f64_median_selection_matches_sorted_bits() {
        for values in [
            vec![-0.0],
            vec![0.0, -0.0],
            vec![-0.0, -0.0],
            vec![0.0, 0.0],
            vec![0.0, -0.0, -0.0],
            vec![-0.0, 0.0, 0.0, -0.0],
            vec![-f64::MAX, f64::MAX],
            vec![-f64::MAX, -f64::MAX],
            vec![f64::MAX, f64::MAX],
            vec![f64::MIN_POSITIVE, f64::from_bits(1)],
            vec![-f64::MAX, -1.0, 0.0, 1.0, f64::MAX],
            vec![3.0, 3.0, 3.0, 3.0],
        ] {
            check_f64(&values);
        }
        for n in 1..=33 {
            let mut values = (0..n)
                .map(|i| ((i * 19 + 3) % 13 - 6) as f64)
                .collect::<Vec<_>>();
            check_f64(&values);
            values.reverse();
            check_f64(&values);
        }
    }

    #[test]
    fn f32_median_selection_matches_sorted_bits() {
        for values in [
            vec![-0.0_f32],
            vec![0.0_f32, -0.0],
            vec![-0.0_f32, -0.0],
            vec![0.0_f32, 0.0],
            vec![0.0_f32, -0.0, -0.0],
            vec![-0.0_f32, 0.0, 0.0, -0.0],
            vec![-f32::MAX, f32::MAX],
            vec![-f32::MAX, -f32::MAX],
            vec![f32::MAX, f32::MAX],
            vec![f32::MIN_POSITIVE, f32::from_bits(1)],
            vec![-f32::MAX, -1.0, 0.0, 1.0, f32::MAX],
            vec![3.0_f32, 3.0, 3.0, 3.0],
        ] {
            check_f32(&values);
        }
        for n in 1..=33 {
            let mut values = (0..n)
                .map(|i| ((i * 19 + 3) % 13 - 6) as f32)
                .collect::<Vec<_>>();
            check_f32(&values);
            values.reverse();
            check_f32(&values);
        }
    }

    #[test]
    fn median_selection_retains_input_validation() {
        assert!(median::<f64>(&[]).is_err());
        assert!(median::<f32>(&[]).is_err());
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(median(&[0.0, invalid]).is_err());
        }
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(median(&[0.0_f32, invalid]).is_err());
        }
        assert!(median(&vec![1.0_f64; MAX_SAMPLES + 1]).is_err());
    }
}

#[cfg(test)]
mod shared_moments_tests {
    use super::*;

    fn assert_same_float<T: Float>(actual: T, expected: T) {
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            // f32 and f64 convert to f64 exactly, including signed zero.
            assert_eq!(
                actual.to_f64().unwrap().to_bits(),
                expected.to_f64().unwrap().to_bits()
            );
        }
    }

    fn check_description<T: Float>(values: &[T]) {
        for ddof in [0, 1] {
            if ddof >= values.len() {
                continue;
            }
            for bias in [false, true] {
                let summary = describe(values, ddof, bias).unwrap();
                assert_eq!(summary.count, values.len());
                assert_same_float(summary.mean, mean(values).unwrap());
                assert_same_float(summary.variance, variance(values, ddof).unwrap());
                assert_same_float(summary.skewness, skew(values, bias).unwrap());
                assert_same_float(summary.kurtosis, kurtosis(values, true, bias).unwrap());
            }
        }
    }

    #[test]
    fn describe_reused_moments_match_separate_statistics_bitwise() {
        for values in [
            vec![0.0_f64],
            vec![-0.0, 0.0, -0.0, 0.0],
            vec![2.0, 2.0, 2.0, 2.0],
            vec![-11.0, -3.0, 0.0, 4.0, 5.0, 100.0],
            vec![1e-100, 3e-100, 2e-100, 1e-100],
            vec![-1e150, -1e100, 3e100, 1e150],
        ] {
            check_description(&values);
        }
        for values in [
            vec![0.0_f32],
            vec![-0.0_f32, 0.0, -0.0, 0.0],
            vec![3.0_f32; 5],
            vec![-11.0_f32, -3.0, 0.0, 4.0, 5.0, 100.0],
            vec![1e-15_f32, 3e-15, 2e-15, 1e-15],
            vec![-1e15_f32, -1e10, 3e10, 1e15],
        ] {
            check_description(&values);
        }
        assert!(describe::<f64>(&[], 0, false).is_err());
        assert!(describe(&[1.0, 2.0], 2, true).is_err());
        assert!(describe(&[1.0, f64::NAN], 0, false).is_err());
    }

    #[test]
    fn mad_in_place_matches_previous_double_copy_selection() {
        for values in [
            vec![-0.0_f64],
            vec![-0.0, 0.0, -0.0, 0.0],
            vec![1.0, 1.0, 1.0],
            vec![9.0, 1.0, 3.0, 1.0, -1.0],
            vec![-100.0, -3.0, 0.0, 4.0, 75.0, 200.0],
            vec![1e-120, -1e-120, 2e-120, 3e-120],
        ] {
            let center = median(&values).unwrap();
            let distances = values
                .iter()
                .map(|&x| (x - center).abs())
                .collect::<Vec<_>>();
            for scale in [0.5, 1.0, 1.4826] {
                let expected = median(&distances).unwrap() / scale;
                let actual = median_abs_deviation(&values, scale).unwrap();
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
        let values = [-0.0_f32, 0.0, -4.0, 2.0, 6.0];
        let center = median(&values).unwrap();
        let distances = values.map(|x| (x - center).abs());
        assert_eq!(
            median_abs_deviation(&values, 1.0_f32).unwrap().to_bits(),
            median(&distances).unwrap().to_bits()
        );
        assert!(median_abs_deviation(&[-f64::MAX, f64::MAX, f64::MAX], 1.0).is_err());
        assert!(median_abs_deviation(&[1.0, f64::NAN], 1.0).is_err());
        assert!(median_abs_deviation(&[1.0], 0.0).is_err());
    }
}
