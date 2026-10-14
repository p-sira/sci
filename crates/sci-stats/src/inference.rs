use super::*;
use statrs::distribution::{Beta, ChiSquared, ContinuousCDF, FisherSnedecor, StudentsT};

/// Direction of a statistical alternative hypothesis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Alternative {
    /// Statistic differs in either direction.
    TwoSided,
    /// First population parameter is less than the null or second parameter.
    Less,
    /// First population parameter is greater than the null or second parameter.
    Greater,
}
/// Scalar hypothesis-test result; undefined tests contain NaN fields.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TestResult {
    /// Test statistic, with the sign and normalization documented by the test.
    pub statistic: f64,
    /// Tail probability under the null, between zero and one, or NaN if undefined.
    pub pvalue: f64,
}
/// Student t-test result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TTestResult {
    /// Signed t statistic (first mean minus null or second mean).
    pub statistic: f64,
    /// Tail probability selected by the alternative.
    pub pvalue: f64,
    /// Degrees of freedom, possibly fractional for Welch's test.
    pub df: f64,
}
/// Ordinary least-squares regression with an intercept.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Regression {
    /// Slope in units of y/x.
    pub slope: f64,
    /// Intercept in units of y.
    pub intercept: f64,
    /// Pearson correlation coefficient.
    pub rvalue: f64,
    /// Tail probability for the hypothesis that slope is zero.
    pub pvalue: f64,
    /// Standard error of the slope.
    pub stderr: f64,
    /// Standard error of the intercept.
    pub intercept_stderr: f64,
}
fn paired(x: &[f64], y: &[f64], minimum: usize) -> Result<()> {
    if x.len() != y.len() {
        return Err(Error::DimensionMismatch);
    }
    sample(x, minimum)?;
    sample(y, minimum)
}
pub(super) fn t_probability(t: f64, df: f64, alternative: Alternative) -> Result<f64> {
    if t.is_nan() {
        return Ok(f64::NAN);
    }
    let d = StudentsT::new(0., 1., df)
        .map_err(|_| Error::InvalidInput("invalid t degrees of freedom"))?;
    Ok(match alternative {
        Alternative::TwoSided => 2. * d.sf(t.abs()),
        Alternative::Less => d.cdf(t),
        Alternative::Greater => d.sf(t),
    })
}
fn t_result(delta: f64, se: f64, df: f64, alternative: Alternative) -> Result<TTestResult> {
    if !delta.is_finite() || !se.is_finite() {
        return Err(Error::NonFiniteEvaluation);
    }
    let statistic = delta / se;
    Ok(TTestResult {
        statistic,
        pvalue: t_probability(statistic, df, alternative)?,
        df,
    })
}
pub(super) fn chi_result(statistic: f64, df: f64) -> Result<TestResult> {
    let pvalue = if statistic.is_nan() {
        f64::NAN
    } else {
        ChiSquared::new(df)
            .map_err(|_| Error::InvalidInput("invalid chi-square degrees of freedom"))?
            .sf(statistic)
    };
    Ok(TestResult { statistic, pvalue })
}
pub(super) fn groups_check(groups: &[&[f64]], minimum: usize) -> Result<usize> {
    if groups.len() < minimum || groups.len() > MAX_SAMPLES {
        return Err(Error::InvalidInput("invalid group count"));
    }
    let mut n = 0;
    for x in groups {
        sample(x, 1)?;
        n += x.len();
        if n > MAX_SAMPLES {
            return Err(Error::InvalidInput("total sample size exceeds 1048576"));
        }
    }
    Ok(n)
}
fn centered(x: &[f64]) -> Result<Vec<f64>> {
    centered_scaled(x).map(|(_, deviations)| deviations)
}

// If the scale ratio overflows or underflows, combine the coefficient with
// the numerator first. A zero coefficient needs no scale division.
fn rescale_ratio(value: f64, numerator: f64, denominator: f64) -> Result<f64> {
    if value == 0.0 || numerator == 0.0 {
        return Ok(0.0);
    }
    let ratio = numerator / denominator;
    finite(if ratio.is_normal() {
        value * ratio
    } else {
        (value * numerator) / denominator
    })
}
pub(super) fn correlation(x: &[f64], y: &[f64]) -> Result<f64> {
    let a = centered(x)?;
    let b = centered(y)?;
    let aa = a.iter().map(|v| v * v).sum::<f64>().sqrt();
    let bb = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    if aa == 0. || bb == 0. {
        return Ok(f64::NAN);
    }
    Ok(a.iter()
        .zip(b)
        .map(|(u, v)| (u / aa) * (v / bb))
        .sum::<f64>()
        .clamp(-1., 1.))
}
/// One-sample t-test `t=(mean(x)−μ₀)/sqrt(var(x,1)/n)`, df=n−1.
///
/// ## Parameters
///
/// `popmean` is the null mean; alternative selects the lower, upper, or two-sided tail.
///
/// ## Domain
///
/// At least two finite observations and finite null mean. Zero standard error yields NaN or signed infinity.
///
/// ## Returns
///
/// Signed statistic, selected p-value, and degrees of freedom.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Uses binary64 formulas with statrs tail probabilities. Extreme or nearly constant data can lose accuracy.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// use sci_stats::{ttest_1samp, Alternative};
/// assert_eq!(ttest_1samp(&[1., 2., 3.], 2., Alternative::TwoSided)?.pvalue, 1.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `ttest_1samp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html).
pub fn ttest_1samp(values: &[f64], popmean: f64, alternative: Alternative) -> Result<TTestResult> {
    sample(values, 2)?;
    finite(popmean)?;
    t_result(
        mean(values)? - popmean,
        sem(values, 1)?,
        (values.len() - 1) as f64,
        alternative,
    )
}
/// Paired t-test of differences `d_i=first_i−second_i` against zero.
///
/// ## Parameters
///
/// Samples are paired by position; equal lengths; alternative concerns the first minus second mean.
///
/// ## Domain
///
/// At least two finite pairs. Constant zero differences yield NaN; nonzero constant differences give infinite t.
///
/// ## Returns
///
/// Signed statistic, p-value, and n−1 degrees of freedom.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Uses binary64 formulas with statrs tail probabilities. Extreme or nearly constant data can lose accuracy.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// use sci_stats::{ttest_rel, Alternative};
/// assert!(ttest_rel(&[1., 3., 5.], &[0., 1., 2.], Alternative::Greater)?.statistic > 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `ttest_rel`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html).
pub fn ttest_rel(first: &[f64], second: &[f64], alternative: Alternative) -> Result<TTestResult> {
    paired(first, second, 2)?;
    let differences = first
        .iter()
        .zip(second)
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    ttest_1samp(&differences, 0., alternative)
}
/// Independent t-test `t=(μ₁−μ₂)/SE`, using pooled or Welch variance.
///
/// ## Parameters
///
/// Standard deviations use ddof=1. Pooled SE²=((n₁−1)s₁²+(n₂−1)s₂²)/(n₁+n₂−2)*(1/n₁+1/n₂).
/// Welch SE²=s₁²/n₁+s₂²/n₂, df=SE⁴/((s₁²/n₁)²/(n₁−1)+(s₂²/n₂)²/(n₂−1)).
///
/// ## Domain
///
/// Finite means, nonnegative finite deviations, and both counts in \[2,1048576\]. Zero spread gives NaN or infinity.
///
/// ## Returns
///
/// Signed t statistic, p-value, and degrees of freedom.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Scales standard deviations before forming pooled variances and Welch degrees
/// of freedom. With nonzero spreads, subnormal errors and overflowing mean differences use
/// normalized units for the statistic. Both zero spreads give Welch df=1;
/// identical zero-spread means give NaN and unequal means give infinite t.
/// Statrs evaluates tail probabilities; nearly constant data can lose accuracy.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// use sci_stats::{ttest_ind_from_stats, Alternative};
/// assert_eq!(ttest_ind_from_stats(1., 1., 5, 1., 2., 6, false, Alternative::TwoSided)?.pvalue, 1.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `ttest_ind_from_stats`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind_from_stats.html).
#[expect(
    clippy::too_many_arguments,
    reason = "preserve the explicit SciPy summary-statistic signature"
)]
pub fn ttest_ind_from_stats(
    mean1: f64,
    std1: f64,
    nobs1: usize,
    mean2: f64,
    std2: f64,
    nobs2: usize,
    equal_var: bool,
    alternative: Alternative,
) -> Result<TTestResult> {
    for n in [nobs1, nobs2] {
        if !(2..=MAX_SAMPLES).contains(&n) {
            return Err(Error::InvalidInput("sample count must be in [2,1048576]"));
        }
    }
    for v in [mean1, mean2, std1, std2] {
        finite(v)?;
    }
    if std1 < 0. || std2 < 0. {
        return Err(Error::InvalidInput(
            "standard deviations must be nonnegative",
        ));
    }
    let (n1, n2) = (nobs1 as f64, nobs2 as f64);
    let scale = std1.max(std2);
    if scale == 0. {
        let df = if equal_var { n1 + n2 - 2. } else { 1. };
        return t_result(mean1 - mean2, 0., df, alternative);
    }
    let (s1, s2) = (std1 / scale, std2 / scale);
    let (se, df) = if equal_var {
        let df = n1 + n2 - 2.;
        (
            ((((n1 - 1.) * s1.powi(2) + (n2 - 1.) * s2.powi(2)) / df) * (1. / n1 + 1. / n2)).sqrt(),
            df,
        )
    } else {
        let a = s1.powi(2) / n1;
        let b = s2.powi(2) / n2;
        let df = (a + b).powi(2) / (a * a / (n1 - 1.) + b * b / (n2 - 1.));
        ((a + b).sqrt(), df)
    };
    let delta = mean1 - mean2;
    let physical_se = scale * se;
    let statistic = if delta.is_finite() && physical_se.is_normal() {
        delta / physical_se
    } else {
        // Keep subnormal standard errors and overflowing mean differences in
        // normalized units until the dimensionless statistic is formed.
        let delta = if delta.is_finite() {
            delta / scale
        } else {
            mean1 / scale - mean2 / scale
        };
        delta / se
    };
    Ok(TTestResult {
        statistic,
        pvalue: t_probability(statistic, df, alternative)?,
        df,
    })
}
/// Independent two-sample t-test of equal population means.
///
/// ## Parameters
///
/// `equal_var=true` pools unbiased variances; false uses Welch degrees of freedom. No trimming or resampling.
///
/// ## Domain
///
/// Each finite sample has at least two observations; lengths may differ.
///
/// ## Returns
///
/// Signed t statistic, p-value, and degrees of freedom.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Uses binary64 formulas with statrs tail probabilities. Extreme or nearly constant data can lose accuracy.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// use sci_stats::{ttest_ind, Alternative};
/// assert_eq!(ttest_ind(&[1., 2., 3.], &[1., 2., 3.], false, Alternative::TwoSided)?.pvalue, 1.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `ttest_ind`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html).
pub fn ttest_ind(
    first: &[f64],
    second: &[f64],
    equal_var: bool,
    alternative: Alternative,
) -> Result<TTestResult> {
    sample(first, 2)?;
    sample(second, 2)?;
    ttest_ind_from_stats(
        mean(first)?,
        std(first, 1)?,
        first.len(),
        mean(second)?,
        std(second, 1)?,
        second.len(),
        equal_var,
        alternative,
    )
}
/// Pearson correlation `r=Σ(x−x̄)(y−ȳ)/sqrt(Σ(x−x̄)²Σ(y−ȳ)²)`.
///
/// ## Parameters
///
/// Tests zero correlation with the exact beta distribution Beta(n/2−1,n/2−1) on [−1,1], assuming independent normal observations.
///
/// ## Domain
///
/// Equal-length finite samples with n≥2. Constants return NaN; n=2 gives r=±1 and p=1 for every alternative, matching SciPy 1.18.1.
///
/// ## Returns
///
/// Correlation coefficient and selected tail probability.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Centered scaled observations reduce overflow; near-perfect correlation can round to ±1.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// use sci_stats::{pearsonr, Alternative};
/// assert!((pearsonr(&[1., 2., 3.], &[2., 4., 6.], Alternative::TwoSided)?.statistic - 1.).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `pearsonr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html).
pub fn pearsonr(first: &[f64], second: &[f64], alternative: Alternative) -> Result<TestResult> {
    paired(first, second, 2)?;
    let r = correlation(first, second)?;
    if r.is_nan() {
        return Ok(TestResult {
            statistic: r,
            pvalue: f64::NAN,
        });
    }
    let pvalue = if first.len() == 2 {
        1.
    } else {
        let a = first.len() as f64 / 2. - 1.;
        let d = Beta::new(a, a)
            .map_err(|_| Error::NumericalFailure("invalid correlation distribution"))?;
        match alternative {
            Alternative::TwoSided => 2. * d.cdf((1. - r.abs()) / 2.),
            Alternative::Less => d.cdf((1. + r) / 2.),
            Alternative::Greater => d.sf((1. + r) / 2.),
        }
    };
    Ok(TestResult {
        statistic: r,
        pvalue,
    })
}
/// Spearman rank correlation: Pearson r of average-tie ranks.
///
/// ## Parameters
///
/// One pair of vectors only; p-values use t=r*sqrt((n−2)/(1−r²)), df=n−2.
///
/// ## Domain
///
/// Equal finite samples n≥2; constants return NaN; n=2 returns NaN p-value.
///
/// ## Returns
///
/// Rank correlation and asymptotic tail probability.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// O(n log n) ranking; asymptotic p-values are unreliable for small samples, as documented by SciPy.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// use sci_stats::{spearmanr, Alternative};
/// assert!(spearmanr(&[1., 3., 2.], &[4., 6., 5.], Alternative::TwoSided)?.statistic > 0.99);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `spearmanr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html).
pub fn spearmanr(first: &[f64], second: &[f64], alternative: Alternative) -> Result<TestResult> {
    paired(first, second, 2)?;
    let a = rankdata(first, RankMethod::Average)?;
    let b = rankdata(second, RankMethod::Average)?;
    let r = correlation(&a, &b)?;
    let pvalue = if first.len() == 2 || r.is_nan() {
        f64::NAN
    } else {
        t_probability(
            r * ((first.len() - 2) as f64 / ((1. - r) * (1. + r))).sqrt(),
            (first.len() - 2) as f64,
            alternative,
        )?
    };
    Ok(TestResult {
        statistic: r,
        pvalue,
    })
}
/// Point-biserial correlation, equal to Pearson r for a binary predictor.
///
/// ## Parameters
///
/// Binary entries must be zero or one; returns a two-sided Pearson p-value.
///
/// ## Domain
///
/// Equal finite samples n≥2; constant predictors or observations return NaN.
///
/// ## Returns
///
/// Correlation coefficient and two-sided p-value.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Uses binary64 formulas with statrs tail probabilities. Extreme or nearly constant data can lose accuracy.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!(sci_stats::pointbiserialr(&[0., 0., 1., 1.], &[1., 2., 3., 4.])?.statistic > 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `pointbiserialr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pointbiserialr.html).
pub fn pointbiserialr(binary: &[f64], values: &[f64]) -> Result<TestResult> {
    paired(binary, values, 2)?;
    if binary.iter().any(|&x| x != 0. && x != 1.) {
        return Err(Error::InvalidInput(
            "binary observations must be zero or one",
        ));
    }
    pearsonr(binary, values, Alternative::TwoSided)
}
/// Least squares line `ŷ=b₀+b₁x`, with `b₁=Σ(x−x̄)(y−ȳ)/Σ(x−x̄)²`.
///
/// ## Parameters
///
/// Includes intercept; alternative tests the slope against zero; no weights.
///
/// ## Domain
///
/// Equal finite samples n≥2; constant x is an error. For n=2, p-value and standard errors are NaN because no residual degrees of freedom remain (SciPy 1.18.1).
///
/// ## Returns
///
/// Slope, intercept, Pearson r, p-value, and slope/intercept standard errors.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Shared shifted/scaled centering preserves differences at large offsets;
/// coefficient rescaling avoids overflowing intermediate scale ratios. Statrs
/// evaluates t probabilities. Ill-conditioned data and tails can lose accuracy.
/// Inputs are limited to 1,048,576 observations.
///
/// # Examples
///
/// ```
/// use sci_stats::{linregress, Alternative};
/// assert!((linregress(&[1., 2., 3.], &[3., 5., 7.], Alternative::TwoSided)?.slope - 2.).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `linregress`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html).
pub fn linregress(x: &[f64], y: &[f64], alternative: Alternative) -> Result<Regression> {
    paired(x, y, 2)?;
    let (sx, a) = centered_scaled(x)?;
    let (sy, b) = centered_scaled(y)?;
    let aa = a.iter().map(|v| v * v).sum::<f64>();
    let bb = b.iter().map(|v| v * v).sum::<f64>();
    if aa == 0. {
        return Err(Error::InvalidInput("regression requires nonconstant x"));
    }
    let ab = a.iter().zip(b).map(|(u, v)| u * v).sum::<f64>();
    let slope = rescale_ratio(ab / aa, sy, sx)?;
    let mx = mean(x)?;
    let intercept = finite(mean(y)? - slope * mx)?;
    let r = if bb == 0. {
        0.
    } else {
        (ab / (aa * bb).sqrt()).clamp(-1., 1.)
    };
    let (pvalue, stderr) = if x.len() == 2 {
        (f64::NAN, f64::NAN)
    } else {
        let df = (x.len() - 2) as f64;
        (
            t_probability(r * (df / ((1. - r) * (1. + r))).sqrt(), df, alternative)?,
            rescale_ratio(((1. - r * r) * bb / aa / df).max(0.).sqrt(), sy, sx)?,
        )
    };
    let intercept_stderr = if x.len() == 2 {
        f64::NAN
    } else {
        finite(stderr * mx.hypot(sx * (aa / x.len() as f64).sqrt()))?
    };
    Ok(Regression {
        slope,
        intercept,
        rvalue: r,
        pvalue,
        stderr,
        intercept_stderr,
    })
}
/// Jarque–Bera normality statistic `JB=n*(skew²/6+excess_kurtosis²/24)`.
///
/// ## Parameters
///
/// Uses population moments; asymptotic null is chi-square with two degrees of freedom.
///
/// ## Domain
///
/// Nonempty finite data; constant data returns NaN statistic and p-value.
///
/// ## Returns
///
/// Nonnegative statistic and upper-tail probability.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// The chi-square approximation requires large samples (SciPy recommends more than 2000 observations).
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!(sci_stats::jarque_bera(&[1., 2., 3., 4.])?.statistic >= 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `jarque_bera`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.jarque_bera.html).
pub fn jarque_bera(values: &[f64]) -> Result<TestResult> {
    sample(values, 1)?;
    let s = skew(values, true)?;
    let k = kurtosis(values, true, true)?;
    chi_result(values.len() as f64 * (s * s / 6. + k * k / 24.), 2.)
}
/// Cressie–Read statistic `2/(λ(λ+1))*Σ O*((O/E)^λ−1)`, with log limits at λ=0,−1.
///
/// ## Parameters
///
/// Expected frequencies are explicit; degrees of freedom k−1−ddof; sums must agree within sqrt(machine epsilon) relative tolerance.
///
/// ## Domain
///
/// At least two categories, finite nonnegative observations, positive expected counts, finite lambda and positive residual df.
///
/// ## Returns
///
/// Divergence statistic and asymptotic chi-square upper-tail p-value.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Uses binary64 formulas with statrs tail probabilities. Extreme or nearly constant data can lose accuracy.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!(sci_stats::power_divergence(&[10., 20.], &[15., 15.], 0, 0.)?.statistic > 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `power_divergence`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.power_divergence.html).
pub fn power_divergence(
    observed: &[f64],
    expected: &[f64],
    ddof: usize,
    lambda: f64,
) -> Result<TestResult> {
    paired(observed, expected, 2)?;
    if ddof >= observed.len() - 1 {
        return Err(Error::InvalidInput(
            "chi-square degrees of freedom must be positive",
        ));
    }
    finite(lambda)?;
    if observed.iter().any(|x| *x < 0.) || expected.iter().any(|x| *x <= 0.) {
        return Err(Error::InvalidInput(
            "frequencies must be nonnegative with positive expectations",
        ));
    }
    let o = finite(observed.iter().sum())?;
    let e = finite(expected.iter().sum())?;
    if (o - e).abs() > f64::EPSILON.sqrt() * o.min(e) {
        return Err(Error::InvalidInput("observed and expected totals differ"));
    }
    let mut statistic = 0.;
    for (&o, &e) in observed.iter().zip(expected) {
        statistic += if lambda == 1. {
            (o - e).powi(2) / e
        } else if lambda == 0. {
            if o == 0. { 0. } else { 2. * o * (o / e).ln() }
        } else if lambda == -1. {
            2. * e * (e / o).ln()
        } else if o == 0. {
            if lambda > -1. { 0. } else { f64::INFINITY }
        } else {
            2. * o * ((o / e).powf(lambda) - 1.) / (lambda * (lambda + 1.))
        };
    }
    if statistic.is_nan() {
        return Err(Error::NonFiniteEvaluation);
    }
    chi_result(statistic.max(0.), (observed.len() - 1 - ddof) as f64)
}
/// Pearson goodness-of-fit statistic `Σ(O−E)²/E`.
///
/// ## Parameters
///
/// Expected counts explicit; residual df=k−1−ddof; asymptotic chi-square null.
///
/// ## Domain
///
/// Finite nonnegative observations and positive expected counts with matching totals.
///
/// ## Returns
///
/// Statistic and upper-tail p-value.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Uses binary64 formulas with statrs tail probabilities. Extreme or nearly constant data can lose accuracy.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::chisquare(&[5., 5.], &[5., 5.], 0)?.pvalue, 1.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `chisquare`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html).
pub fn chisquare(observed: &[f64], expected: &[f64], ddof: usize) -> Result<TestResult> {
    power_divergence(observed, expected, ddof, 1.)
}
/// Fisher exact test: hypergeometric probability conditional on the margins of a 2×2 table.
///
/// ## Parameters
///
/// Table order is `[a,b,c,d]` in row-major order; statistic is sample odds ratio a*d/(b*c). Two-sided tail sums tables no more likely than the observed table.
///
/// ## Domain
///
/// Nonnegative integer cells and total≤65536; zero margins yield NaN odds ratio and p=1.
///
/// ## Returns
///
/// Odds ratio and exact conditional tail p-value.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Delegates to statrs. Its two-sided probability ties use a relative 1e-4 threshold; SciPy uses a tighter threshold, so near-tie tables may differ.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// use sci_stats::{fisher_exact, Alternative};
/// assert!(fisher_exact(&[6., 2., 1., 4.], Alternative::TwoSided)?.pvalue < 0.2);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `fisher_exact`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html).
pub fn fisher_exact(table: &[f64; 4], alternative: Alternative) -> Result<TestResult> {
    let mut t = [0_u64; 4];
    let mut total = 0;
    for (i, &v) in table.iter().enumerate() {
        if !v.is_finite() || !(0. ..=65536.).contains(&v) || v.fract() != 0. {
            return Err(Error::InvalidInput(
                "Fisher cells must be integers in [0,65536]",
            ));
        }
        t[i] = v as u64;
        total += t[i];
    }
    if total > 65536 {
        return Err(Error::InvalidInput("Fisher total exceeds 65536"));
    }
    let a = match alternative {
        Alternative::TwoSided => statrs::stats_tests::Alternative::TwoSided,
        Alternative::Less => statrs::stats_tests::Alternative::Less,
        Alternative::Greater => statrs::stats_tests::Alternative::Greater,
    };
    let (statistic, pvalue) = statrs::stats_tests::fishers_exact_with_odds_ratio(&t, a)
        .map_err(|_| Error::NumericalFailure("Fisher exact backend failure"))?;
    Ok(TestResult { statistic, pvalue })
}
/// One-way ANOVA `F=(SS_between/(k−1))/(SS_within/(N−k))`.
///
/// ## Parameters
///
/// Equal population variance assumption; groups may have different sizes; no Welch overload.
///
/// ## Domain
///
/// At least two nonempty finite groups, total N>k. Zero within-group spread yields NaN or infinity.
///
/// ## Returns
///
/// F statistic and upper-tail F(k−1,N−k) probability.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Uses binary64 formulas with statrs tail probabilities. Extreme or nearly constant data can lose accuracy.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!(sci_stats::f_oneway(&[&[1., 2., 3.], &[4., 5., 6.]])?.statistic > 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `f_oneway`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html).
pub fn f_oneway(groups: &[&[f64]]) -> Result<TestResult> {
    let n = groups_check(groups, 2)?;
    let k = groups.len();
    if n <= k {
        return Err(Error::InvalidInput(
            "ANOVA needs residual degrees of freedom",
        ));
    }
    let all = groups
        .iter()
        .flat_map(|x| x.iter().copied())
        .collect::<Vec<_>>();
    let center = mean(&all)?;
    let mut between = 0.;
    let mut within = 0.;
    for x in groups {
        let m = mean(x)?;
        between += x.len() as f64 * (m - center).powi(2);
        within += variance(x, 0)? * x.len() as f64;
    }
    finite(between)?;
    finite(within)?;
    let statistic = (between / (k - 1) as f64) / (within / (n - k) as f64);
    let pvalue = if statistic.is_nan() {
        f64::NAN
    } else {
        FisherSnedecor::new((k - 1) as f64, (n - k) as f64)
            .map_err(|_| Error::NumericalFailure("ANOVA backend failure"))?
            .sf(statistic)
    };
    Ok(TestResult { statistic, pvalue })
}
/// Bartlett test of equal variances: corrected log pooled-variance contrast.
///
/// ## Parameters
///
/// Statistic is `((N−k)ln(s_p²)−Σ(n_i−1)ln(s_i²))/C`,
/// C=1+(Σ1/(n_i−1)−1/(N−k))/(3(k−1)); asymptotic df=k−1.
///
/// ## Domain
///
/// At least two finite groups, each with n≥2. Zero variances produce NaN or infinity as in SciPy.
///
/// ## Returns
///
/// Bartlett statistic and chi-square upper-tail p-value.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Uses binary64 formulas with statrs tail probabilities. Extreme or nearly constant data can lose accuracy.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::bartlett(&[&[1., 2., 3.], &[4., 5., 6.]])?.pvalue, 1.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `bartlett`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bartlett.html).
pub fn bartlett(groups: &[&[f64]]) -> Result<TestResult> {
    let n = groups_check(groups, 2)?;
    let k = groups.len();
    let mut pooled = 0.;
    let mut logs = 0.;
    let mut reciprocals = 0.;
    for x in groups {
        sample(x, 2)?;
        let df = (x.len() - 1) as f64;
        let v = variance(x, 1)?;
        pooled += df * v;
        logs += df * v.ln();
        reciprocals += 1. / df;
    }
    let df = (n - k) as f64;
    let c = 1. + (reciprocals - 1. / df) / (3. * (k - 1) as f64);
    let statistic = (df * (pooled / df).ln() - logs) / c;
    chi_result(statistic, (k - 1) as f64)
}
/// Levene test: one-way ANOVA on absolute deviations from each group center.
///
/// ## Parameters
///
/// `Center::Mean` gives Levene; Median gives Brown–Forsythe. Trimmed centers are unsupported.
///
/// ## Domain
///
/// At least two finite nonempty groups with residual degrees of freedom.
///
/// ## Returns
///
/// F statistic and asymptotic upper-tail p-value.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Uses binary64 formulas with statrs tail probabilities. Extreme or nearly constant data can lose accuracy.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// use sci_stats::{levene, Center};
/// assert!(levene(&[&[1., 2., 4.], &[2., 3., 8.]], Center::Median)?.pvalue >= 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `levene`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levene.html).
pub fn levene(groups: &[&[f64]], center: Center) -> Result<TestResult> {
    groups_check(groups, 2)?;
    let transformed = groups
        .iter()
        .map(|x| {
            let c = match center {
                Center::Mean => mean(x)?,
                Center::Median => median(x)?,
            };
            Ok(x.iter().map(|v| (v - c).abs()).collect::<Vec<_>>())
        })
        .collect::<Result<Vec<_>>>()?;
    let refs = transformed.iter().map(Vec::as_slice).collect::<Vec<_>>();
    f_oneway(&refs)
}

/// Group center for robust variance tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Center {
    /// Arithmetic mean.
    Mean,
    /// Sample median.
    Median,
}
/// Kruskal–Wallis statistic `H=(12/(N(N+1))*Σ R_i²/n_i−3(N+1))/tiecorrect`.
///
/// ## Parameters
///
/// Pooled average ranks; R_i is group rank sum; asymptotic chi-square df=k−1.
///
/// ## Domain
///
/// At least two nonempty finite groups; all-equal observations are an error.
///
/// ## Returns
///
/// Tie-corrected H statistic and upper-tail probability.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// O(N log N) ranking; asymptotic p-values are unreliable with fewer than about five observations per group.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!(sci_stats::kruskal(&[&[1., 2., 3.], &[4., 5., 6.]])?.statistic > 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `kruskal`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html).
pub fn kruskal(groups: &[&[f64]]) -> Result<TestResult> {
    let n = groups_check(groups, 2)?;
    let all = groups
        .iter()
        .flat_map(|x| x.iter().copied())
        .collect::<Vec<_>>();
    let ranks = rankdata(&all, RankMethod::Average)?;
    let c = tiecorrect(&ranks)?;
    if c == 0. {
        return Err(Error::InvalidInput(
            "Kruskal requires nonidentical observations",
        ));
    }
    let mut sum = 0.;
    let mut offset = 0;
    for x in groups {
        sum += ranks[offset..offset + x.len()].iter().sum::<f64>().powi(2) / x.len() as f64;
        offset += x.len();
    }
    let n = n as f64;
    chi_result(
        ((12. * sum / (n * (n + 1.)) - 3. * (n + 1.)) / c).max(0.),
        (groups.len() - 1) as f64,
    )
}
/// Friedman repeated-measures test `Q=(12ΣR_i²/(n*k*(k+1))−3*n*(k+1))/C`.
///
/// ## Parameters
///
/// Groups are treatments; matching positions are subjects; each subject is ranked across groups; tie correction C.
///
/// ## Domain
///
/// At least three equal-length nonempty finite groups. All ties return NaN.
///
/// ## Returns
///
/// Tie-corrected statistic and asymptotic chi-square df=k−1 p-value.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// O(n*k log k) work; approximation is reliable only for sufficiently many subjects and treatments.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!(sci_stats::friedmanchisquare(&[&[1., 2., 3.], &[2., 4., 5.], &[3., 5., 6.]])?.statistic > 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `friedmanchisquare`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html).
pub fn friedmanchisquare(groups: &[&[f64]]) -> Result<TestResult> {
    groups_check(groups, 3)?;
    let n = groups[0].len();
    let k = groups.len();
    if groups.iter().any(|x| x.len() != n) {
        return Err(Error::DimensionMismatch);
    }
    let mut sums = vec![0.; k];
    let mut ties = 0.;
    for i in 0..n {
        let row = groups.iter().map(|x| x[i]).collect::<Vec<_>>();
        let r = rankdata(&row, RankMethod::Average)?;
        for j in 0..k {
            sums[j] += r[j];
        }
        ties += 1. - tiecorrect(&r)?;
    }
    let c = 1. - ties / n as f64;
    let kf = k as f64;
    let nf = n as f64;
    if c == 0. {
        return Ok(TestResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        });
    }
    chi_result(
        ((12. * sums.iter().map(|r| r * r).sum::<f64>() / (nf * kf * (kf + 1.))
            - 3. * nf * (kf + 1.))
            / c)
            .max(0.),
        (k - 1) as f64,
    )
}
/// Wilcoxon rank-sum z statistic `(R₁−n₁(N+1)/2)/sqrt(n₁*n₂*(N+1)/12)`.
///
/// ## Parameters
///
/// No tie correction or continuity correction, matching scipy.stats.ranksums. Use an appropriate tied-sample test for ties.
///
/// ## Domain
///
/// Nonempty finite independent samples; combined length≤1048576.
///
/// ## Returns
///
/// Signed z statistic and standard-normal selected tail probability.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Uses binary64 formulas with statrs tail probabilities. Extreme or nearly constant data can lose accuracy.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// use sci_stats::{ranksums, Alternative};
/// assert!(ranksums(&[1., 2.], &[3., 4.], Alternative::Less)?.pvalue < 0.1);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `ranksums`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ranksums.html).
pub fn ranksums(first: &[f64], second: &[f64], alternative: Alternative) -> Result<TestResult> {
    sample(first, 1)?;
    sample(second, 1)?;
    if first.len() + second.len() > MAX_SAMPLES {
        return Err(Error::InvalidInput("combined sample count exceeds limit"));
    }
    let all = first.iter().chain(second).copied().collect::<Vec<_>>();
    let r = rankdata(&all, RankMethod::Average)?;
    let n1 = first.len() as f64;
    let n2 = second.len() as f64;
    let statistic = (r[..first.len()].iter().sum::<f64>() - n1 * (n1 + n2 + 1.) / 2.)
        / (n1 * n2 * (n1 + n2 + 1.) / 12.).sqrt();
    let d = normal(0., 1.)?;
    let pvalue = match alternative {
        Alternative::TwoSided => 2. * d.sf(statistic.abs()),
        Alternative::Less => d.cdf(statistic),
        Alternative::Greater => d.sf(statistic),
    };
    Ok(TestResult { statistic, pvalue })
}
