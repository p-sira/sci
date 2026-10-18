use super::*;

pub(super) fn normal_result(statistic: f64, alternative: Alternative) -> Result<TestResult> {
    let pvalue = if statistic.is_nan() {
        f64::NAN
    } else {
        match alternative {
            Alternative::TwoSided => {
                2. * sf(
                    Distribution::Normal {
                        mean: 0.,
                        std_dev: 1.,
                    },
                    statistic.abs(),
                )?
            }
            Alternative::Less => cdf(
                Distribution::Normal {
                    mean: 0.,
                    std_dev: 1.,
                },
                statistic,
            )?,
            Alternative::Greater => sf(
                Distribution::Normal {
                    mean: 0.,
                    std_dev: 1.,
                },
                statistic,
            )?,
        }
    };
    Ok(TestResult { statistic, pvalue })
}

fn skew_statistic(values: &[f64]) -> Result<f64> {
    sample(values, 8)?;
    let n = values.len() as f64;
    let y = skew(values, true)? * (((n + 1.) * (n + 3.)) / (6. * (n - 2.))).sqrt();
    let beta = 3. * (n * n + 27. * n - 70.) * (n + 1.) * (n + 3.)
        / ((n - 2.) * (n + 5.) * (n + 7.) * (n + 9.));
    let w = -1. + (2. * (beta - 1.)).sqrt();
    let delta = 1. / (0.5 * w.ln()).sqrt();
    let alpha = (2. / (w - 1.)).sqrt();
    Ok(delta * (y / alpha).asinh())
}

fn kurtosis_statistic(values: &[f64]) -> Result<f64> {
    sample(values, 5)?;
    let n = values.len() as f64;
    let b = kurtosis(values, false, true)?;
    let expected = 3. * (n - 1.) / (n + 1.);
    let variance = 24. * n * (n - 2.) * (n - 3.) / ((n + 1.).powi(2) * (n + 3.) * (n + 5.));
    let x = (b - expected) / variance.sqrt();
    let beta = 6. * (n * n - 5. * n + 2.) / ((n + 7.) * (n + 9.))
        * (6. * (n + 3.) * (n + 5.) / (n * (n - 2.) * (n - 3.))).sqrt();
    let a = 6. + 8. / beta * (2. / beta + (1. + 4. / beta.powi(2)).sqrt());
    let denominator = 1. + x * (2. / (a - 4.)).sqrt();
    if denominator == 0. {
        return Ok(f64::NAN);
    }
    Ok((1. - 2. / (9. * a) - ((1. - 2. / a) / denominator).cbrt()) / (2. / (9. * a)).sqrt())
}

/// D’Agostino skewness test, with `Z = δ asinh(Y/α)`.
///
/// ## Parameters and conventions
///
/// `values` is a one-dimensional sample; `alternative` selects a standard-normal
/// tail for negative, positive, or either skewness. For biased skewness `g₁`,
/// `Y=g₁ sqrt((n+1)(n+3)/(6(n−2)))`,
/// `B=3(n²+27n−70)(n+1)(n+3)/((n−2)(n+5)(n+7)(n+9))`,
/// `W²=−1+sqrt(2(B−1))`, `δ=1/sqrt(ln(W²)/2)`, `α=sqrt(2/(W²−1))`.
/// Zero skewness gives Z=0. SciPy 1.18.1 substitutes Y=1 for exactly zero
/// skewness; Sci uses the defining continuous limit instead.
///
/// ## Domain and special values
///
/// Requires 8..=1,048,576 finite observations. Constant data returns NaN fields.
/// NaN omission, masks, axes, and resampling are unsupported.
///
/// ## Returns
///
/// Dimensionless Z and its approximate normal-null p-value.
///
/// ## Errors
///
/// Invalid observation counts or nonfinite inputs return an error.
///
/// ## Numerical behavior
///
/// Native scaled moments and binary64 formulas take O(n) work and memory.
/// Normal tails reuse the existing private statrs backend. Calibration is an
/// approximation; `asinh` avoids cancellation for negative skewness.
///
/// ## Examples
///
/// ```
/// use sci_stats::{skewtest, Alternative};
/// assert!(skewtest(&[1.,2.,3.,4.,5.,6.,7.,8000.], Alternative::Greater)?.pvalue < 0.001);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy skewtest and D’Agostino et al. (1990)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewtest.html).
pub fn skewtest(values: &[f64], alternative: Alternative) -> Result<TestResult> {
    normal_result(skew_statistic(values)?, alternative)
}

/// Anscombe–Glynn kurtosis test:
/// `Z=(1−2/(9A)−cbrt((1−2/A)/(1+X sqrt(2/(A−4)))))/sqrt(2/(9A))`.
///
/// ## Parameters and conventions
///
/// For biased Pearson kurtosis b₂, `X=(b₂−E)/sqrt(V)`,
/// `E=3(n−1)/(n+1)`, `V=24n(n−2)(n−3)/((n+1)²(n+3)(n+5))`,
/// `β=6(n²−5n+2)/((n+7)(n+9)) sqrt(6(n+3)(n+5)/(n(n−2)(n−3)))`,
/// `A=6+8/β(2/β+sqrt(1+4/β²))`. The alternative selects a normal
/// tail for lower, higher, or either kurtosis.
///
/// ## Domain and special values
///
/// Requires 5..=1,048,576 finite observations. Constant data or a zero
/// transformed denominator returns NaN fields. No masks or NaN omission.
///
/// ## Returns
///
/// Dimensionless Z and its approximate normal-null p-value.
///
/// ## Errors
///
/// Invalid counts and nonfinite observations return an error.
///
/// ## Numerical behavior
///
/// O(n) native scaled moments and binary64 formulas; normal tails use statrs
/// privately. The null approximation is unreliable for n≤20. Such samples
/// are accepted, without SciPy's warning mechanism. Signed cube roots support
/// negative denominators.
///
/// ## Examples
///
/// ```
/// use sci_stats::{kurtosistest, Alternative};
/// let values: Vec<_> = (0..30).map(|i| i as f64).collect();
/// assert!(kurtosistest(&values, Alternative::TwoSided)?.pvalue < 0.1);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy kurtosistest and Anscombe–Glynn (1983)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosistest.html).
pub fn kurtosistest(values: &[f64], alternative: Alternative) -> Result<TestResult> {
    normal_result(kurtosis_statistic(values)?, alternative)
}

/// D’Agostino–Pearson omnibus normality test, `K²=Z_skew²+Z_kurtosis²`.
///
/// ## Parameters and conventions
///
/// A one-dimensional sample. Uses the transformations of [`skewtest`] and
/// [`kurtosistest`], including Sci's continuous zero-skewness limit.
///
/// ## Domain and special values
///
/// Requires 8..=1,048,576 finite observations; constant data gives NaN fields.
/// No axes, masks, missing-data policies, or resampling.
///
/// ## Returns
///
/// Dimensionless K² and the chi-square-two upper tail `exp(−K²/2)`.
///
/// ## Errors
///
/// Invalid counts or nonfinite observations return an error.
///
/// ## Numerical behavior
///
/// Native O(n) work and memory; uses only the standard library and Sci's
/// native moments. Null calibration is approximate, especially for n≤20.
/// Extreme tails can underflow to zero.
///
/// ## Examples
///
/// ```
/// let values: Vec<_> = (0..30).map(|i| i as f64).collect();
/// assert!(sci_stats::normaltest(&values)?.statistic >= 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy normaltest and D’Agostino–Pearson (1973)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html).
pub fn normaltest(values: &[f64]) -> Result<TestResult> {
    let s = skew_statistic(values)?;
    let k = kurtosis_statistic(values)?;
    let statistic = s * s + k * k;
    Ok(TestResult {
        statistic,
        pvalue: (-0.5 * statistic).exp(),
    })
}

/// O’Brien variance transform:
/// `r_i=((n−1.5)n(x_i−μ)²−0.5 Σ(x_j−μ)²)/((n−1)(n−2))`.
///
/// ## Parameters and conventions
///
/// Each slice is one group. The arithmetic mean μ is computed within each
/// group. Transformed groups can be passed to [`f_oneway`]; their means equal
/// their original sample variances (`ddof=1`).
///
/// ## Domain and special values
///
/// At least one group, each with at least three finite observations; the total
/// count is at most 1,048,576. Constant groups yield zeros. Groups may differ
/// in length. NaN omission and multidimensional inputs are unsupported.
///
/// ## Returns
///
/// One vector per group, in original group and observation order, in squared
/// observation units. Individual transformed observations may be negative.
///
/// ## Errors
///
/// Invalid group sizes, nonfinite inputs, or unrepresentable results return errors.
///
/// ## Numerical behavior
///
/// O(n) work and memory with native scaled centering. Rescaling occurs after
/// the defining formula; binary64 rounding and underflow remain. SciPy's
/// convergence warning/check is replaced by explicit domain and overflow errors.
/// No external backend is used.
///
/// ## Examples
///
/// ```
/// let transformed = sci_stats::obrientransform(&[&[1.,2.,3.]])?;
/// assert!((sci_stats::mean(&transformed[0])? - 1.).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy obrientransform and Maxwell–Delaney (1990)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.obrientransform.html).
pub fn obrientransform(groups: &[&[f64]]) -> Result<Vec<Vec<f64>>> {
    inference::groups_check(groups, 1)?;
    if groups.iter().any(|x| x.len() < 3) {
        return Err(Error::InvalidInput(
            "O'Brien transform requires at least three observations per group",
        ));
    }
    groups
        .iter()
        .map(|x| {
            let n = x.len() as f64;
            let (scale, deviations) = centered_scaled(x)?;
            let ss = deviations.iter().map(|v| v * v).sum::<f64>();
            deviations
                .iter()
                .map(|v| {
                    let r = ((n - 1.5) * n * v * v - 0.5 * ss) / ((n - 1.) * (n - 2.));
                    finite((r * scale) * scale)
                })
                .collect()
        })
        .collect()
}

/// Alexander–Govern unequal-variance means test, `A=Σz_i²`.
///
/// ## Parameters and conventions
///
/// Each slice is an independent group, assumed normally distributed. Let
/// `s_i=std(x_i,1)/sqrt(n_i)`, `w_i=(1/s_i²)/Σ(1/s_j²)`,
/// `μ_w=Σw_i mean(x_i)`, and `t_i=(mean(x_i)−μ_w)/s_i`.
/// For `v=n_i−1`, `a=v−1/2`, `b=48a²`, `c=sqrt(a ln(1+t_i²/v))`,
/// `z_i=c+(c³+3c)/b−(4c⁷+33c⁵+240c³+855c)/(10b²+8bc⁴+1000b)`.
/// The approximate null is chi-square with k−1 degrees of freedom.
///
/// ## Domain and special values
///
/// At least two groups, each containing at least two finite observations;
/// combined size≤1,048,576. A constant group returns NaN fields. NaN omission,
/// masks, and axes are unsupported. Sci checks actual zero variance rather
/// than SciPy's offset-dependent near-constant threshold.
///
/// ## Returns
///
/// Dimensionless A and its approximate upper-tail p-value.
///
/// ## Errors
///
/// Invalid group sizes and nonfinite observations return errors.
///
/// ## Numerical behavior
///
/// O(N) native work and memory. Common shifted scaling and relative precision
/// weights avoid unnecessary overflow; log ratios avoid squaring large t.
/// Extremely disparate group scales can underflow a normalized spread to zero,
/// yielding NaN. Statrs supplies the chi-square tail through Sci's existing API.
/// Normal-population assumptions and small-sample approximations limit calibration.
///
/// ## Examples
///
/// ```
/// let r = sci_stats::alexandergovern(&[&[1.,2.,3.], &[3.,5.,8.]])?;
/// assert!(r.statistic >= 0. && (0. ..=1.).contains(&r.pvalue));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy alexandergovern and Alexander–Govern (1994), equations 1–9](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.alexandergovern.html).
pub fn alexandergovern(groups: &[&[f64]]) -> Result<TestResult> {
    inference::groups_check(groups, 2)?;
    if groups.iter().any(|g| g.len() < 2) {
        return Err(Error::InvalidInput(
            "Alexander-Govern requires two observations per group",
        ));
    }
    let all: Vec<_> = groups.iter().flat_map(|g| g.iter().copied()).collect();
    let (_, normalized) = centered_scaled(&all)?;
    let mut means = Vec::with_capacity(groups.len());
    let mut errors = Vec::with_capacity(groups.len());
    let mut offset = 0;
    for g in groups {
        let slice = &normalized[offset..offset + g.len()];
        means.push(mean(slice)?);
        errors.push(std(slice, 1)? / (g.len() as f64).sqrt());
        offset += g.len();
    }
    let minimum = errors.iter().copied().fold(f64::INFINITY, f64::min);
    if minimum == 0. {
        return Ok(TestResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        });
    }
    let weights: Vec<_> = errors.iter().map(|s| (minimum / s).powi(2)).collect();
    let weight_sum = weights.iter().sum::<f64>();
    let center = means
        .iter()
        .zip(&weights)
        .map(|(m, w)| m * (w / weight_sum))
        .sum::<f64>();
    let mut statistic = 0.;
    for ((g, mean), se) in groups.iter().zip(&means).zip(errors) {
        let v = (g.len() - 1) as f64;
        let a = v - 0.5;
        let b = 48. * a * a;
        let delta = (mean - center).abs();
        let t = delta / se / v.sqrt();
        let log_term = if t <= 1. {
            (t * t).ln_1p()
        } else {
            2. * (delta.ln() - se.ln() - 0.5 * v.ln()) + (1. / t).powi(2).ln_1p()
        };
        let c = (a * log_term.max(0.)).sqrt();
        let z = c + (c.powi(3) + 3. * c) / b
            - (4. * c.powi(7) + 33. * c.powi(5) + 240. * c.powi(3) + 855. * c)
                / (10. * b * b + 8. * b * c.powi(4) + 1000. * b);
        statistic += z * z;
    }
    inference::chi_result(statistic, (groups.len() - 1) as f64)
}

/// Shapiro–Wilk normality test `W=(Σ_i a_i*x_(i))²/Σ_i (x_i−mean(x))²`.
///
/// ## Parameters and conventions
///
/// `values` is an unweighted sample; x_(i) are sorted observations and a_i
/// are the AS R94 approximations to normalized normal-order-statistic weights.
/// Royston's sample-size-dependent normalizing transform gives the lower-W
/// normality p-value. SciPy's axis/mask/NaN policies and censored variants are unsupported.
///
/// ## Domain and special values
///
/// 3..=5000 finite observations. Constant data gives NaN fields, matching
/// SciPy 1.18.1. NaN/infinity observations are errors.
///
/// ## Returns
///
/// Dimensionless W in `[0,1]` and approximate normality p-value.
///
/// ## Errors
///
/// Invalid inputs/counts, backend failures or nonfinite outputs return errors.
///
/// ## Numerical behavior
///
/// Native shifted/scaled centering followed by private BSD-licensed SciPy AS R94
/// evaluation; O(n log n) sorting and O(n) memory. Scaling avoids overflow in ordered differences.
/// Normal weights and tails reuse statrs, matching SciPy 1.18.1 normal quantiles.
/// Rounding near W=1 can affect p-values. Samples larger than 5000 are rejected because
/// this calibration is unreliable beyond that range.
///
/// ## Examples
///
/// ```
/// let r = sci_stats::shapiro(&[-1.,0.,1.])?;
/// assert!((r.statistic-1.).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy shapiro and Royston AS R94 (1995)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html).
pub fn shapiro(values: &[f64]) -> Result<TestResult> {
    sample(values, 3)?;
    if values.len() > 5000 {
        return Err(Error::InvalidInput("Shapiro sample limit is 5000"));
    }
    let (_, normalized) = centered_scaled(values)?;
    if normalized.iter().all(|v| *v == 0.) {
        return Ok(TestResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        });
    }
    let r = shapiro_backend::evaluate(&normalized)?;
    let statistic = finite(r.statistic)?;
    let pvalue = finite(r.pvalue)?;
    if !(0. ..=1.).contains(&statistic) || !(0. ..=1.).contains(&pvalue) {
        return Err(Error::NumericalFailure(
            "Shapiro-Wilk backend returned invalid result",
        ));
    }
    Ok(TestResult { statistic, pvalue })
}
