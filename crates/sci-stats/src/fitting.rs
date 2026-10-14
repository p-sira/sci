use super::*;

/// Distribution families with analytic maximum-likelihood fits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitFamily {
    /// Normal with free mean and standard deviation (MLE uses ddof=0).
    Normal,
    /// Exponential with fixed location zero and free positive scale.
    Exponential,
    /// Uniform with free lower and upper endpoints.
    Uniform,
}

/// Analytic maximum-likelihood distribution fit `argmax_θ Σ ln f_θ(x_i)`.
///
/// ## Parameters and conventions
///
/// `values` is an unweighted sample; `family` chooses the supported model.
/// Normal MLE is `(mean(x),sqrt(Σ(x−mean(x))²/n))`; zero-location exponential
/// scale is mean(x); uniform support is [min(x),max(x)]. This adapts SciPy
/// generic fit to three analytic models. Arbitrary families, parameter bounds,
/// fixed parameters other than exponential location, optimizer callbacks,
/// integer parameters and maximum-product-of-spacings fits are unsupported.
///
/// ## Domain and special values
///
/// 2..=1048576 finite observations. Normal/uniform require nonconstant samples.
/// Exponential requires nonnegative values and positive mean; zeros are accepted.
/// NaN/infinity inputs are errors; all fitted parameters must be representable.
///
/// ## Returns
///
/// A native [`Distribution`] reusable with [`pdf`], [`cdf`] and [`ppf`].
/// Locations/scales have input units. No optimizer/result object is exposed.
///
/// ## Errors
///
/// Invalid samples/domains, zero fitted scale or unrepresentable parameters return errors.
///
/// ## Numerical behavior
///
/// O(n) compensated/scaled summaries, with no numerical optimizer. MLE assumes
/// independent identically distributed observations; uniform endpoint estimates
/// are biased. Standard-deviation underflow to zero is an error.
///
/// ## Examples
///
/// ```
/// let d = sci_stats::fit(&[1.,2.,3.], sci_stats::FitFamily::Normal)?;
/// assert_eq!(sci_stats::cdf(d,2.)?, 0.5);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fit.html),
/// [NIST normal parameter estimation](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3661.htm).
pub fn fit(values: &[f64], family: FitFamily) -> Result<Distribution> {
    sample(values, 2)?;
    Ok(match family {
        FitFamily::Normal => {
            let mean = mean(values)?;
            let std_dev = std(values, 0)?;
            positive(std_dev)?;
            Distribution::Normal { mean, std_dev }
        }
        FitFamily::Exponential => {
            if values.iter().any(|v| *v < 0.) {
                return Err(Error::InvalidInput(
                    "exponential fit requires nonnegative observations",
                ));
            }
            let scale = mean(values)?;
            positive(scale)?;
            Distribution::Exponential { scale }
        }
        FitFamily::Uniform => {
            let low = values.iter().copied().fold(f64::INFINITY, f64::min);
            let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            positive(high - low)?;
            Distribution::Uniform { low, high }
        }
    })
}

/// Empirical-CDF statistic used with refitted Monte Carlo goodness-of-fit tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GoodnessStatistic {
    /// Two-sided Kolmogorov–Smirnov maximum CDF deviation.
    KolmogorovSmirnov,
    /// Cramér–von Mises integrated squared CDF deviation.
    CramerVonMises,
}
fn goodness_fit(values: &[f64], family: FitFamily) -> Result<Distribution> {
    let fitted = fit(values, family)?;
    // SciPy goodness_of_fit makes this deliberate normal-only correction.
    Ok(match fitted {
        Distribution::Normal { mean, .. } => Distribution::Normal {
            mean,
            std_dev: std(values, 1)?,
        },
        d => d,
    })
}
fn goodness_statistic(
    values: &[f64],
    family: FitFamily,
    statistic: GoodnessStatistic,
) -> Result<f64> {
    let fitted = goodness_fit(values, family)?;
    match statistic {
        GoodnessStatistic::KolmogorovSmirnov => {
            Ok(ks_1samp(values, fitted, Alternative::TwoSided)?.statistic)
        }
        GoodnessStatistic::CramerVonMises => goodness::cvm_statistic(values, fitted),
    }
}

/// Refitted Monte Carlo goodness-of-fit test `p=(1+# {T_refitted_null≥T_refitted_obs})/(B+1)`.
///
/// ## Parameters and conventions
///
/// `values` is the observed sample; `family` chooses a supported analytic fit;
/// `statistic` chooses KS or Cramér–von Mises discrepancy. `null_samples` must
/// be independent draws of the same length from the fitted observed distribution.
/// The model is refitted separately to every draw before measuring discrepancy.
/// As in SciPy, normal fits use sample standard deviation (ddof=1) here, while
/// [`fit`] uses MLE ddof=0. Exponential location stays fixed at zero. This
/// adapts SciPy RNG ownership to supplied draws; use [`fit`] (and ddof=1 for
/// normal) to construct the null sampler. Known/fixed parameter dictionaries,
/// other families/statistics and automatic sampling are unsupported.
///
/// ## Domain and special values
///
/// [`fit`]'s sample domain applies to every draw. Null rows must match original
/// length; n*B≤16777216, B≥1. Constant normal/uniform draws return errors.
/// NaN and infinite observations are errors.
///
/// ## Returns
///
/// Dimensionless observed discrepancy, plus-one p-value and refitted null
/// discrepancies in supplied-draw order. All tests use the greater tail.
///
/// ## Errors
///
/// Invalid sample domains/shapes/work limits or any failed draw fit returns an
/// error; no partial result is returned. Null-sampler correctness is the caller's responsibility.
///
/// ## Numerical behavior
///
/// O(B*n log n) sorting and O(B+n) additional memory, reusing native fits and
/// [`monte_carlo_test`]. No asymptotic p-values are used for calibration.
/// Results have Monte Carlo uncertainty; correlated draws or sampling from an
/// incorrect fitted model invalidates the test. Floating-point ties use SciPy tolerance.
///
/// ## Examples
///
/// ```
/// let draws: &[&[f64]] = &[&[-1.,0.,1.], &[-2.,1.,3.]];
/// let r = sci_stats::goodness_of_fit(&[1.,2.,4.], draws, sci_stats::FitFamily::Normal,
///     sci_stats::GoodnessStatistic::CramerVonMises)?;
/// assert!((0.0..=1.0).contains(&r.pvalue));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy goodness_of_fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.goodness_of_fit.html).
pub fn goodness_of_fit(
    values: &[f64],
    null_samples: &[&[f64]],
    family: FitFamily,
    statistic: GoodnessStatistic,
) -> Result<ResamplingResult> {
    monte_carlo_test(
        values,
        null_samples,
        |row| goodness_statistic(row, family, statistic),
        Alternative::Greater,
    )
}
