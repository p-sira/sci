use super::*;
use statrs::distribution::{ChiSquared, ContinuousCDF, StudentsT};

/// Tail removed by [`trim1`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tail {
    /// Remove the lowest observations.
    Lower,
    /// Remove the highest observations.
    Upper,
}
/// P-value combination convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CombineMethod {
    /// −2Σln(p), chi-square upper tail with 2n degrees of freedom.
    Fisher,
    /// 2Σln(1−p), chi-square lower tail with 2n degrees of freedom.
    Pearson,
    /// Minimum p, tested with a Beta(1,n) distribution.
    Tippett,
    /// Weighted standard-normal inverse-survival scores.
    Stouffer,
    /// Sum of log odds, using the Mudholkar–George t approximation.
    MudholkarGeorge,
}
/// Converged sigma clipping result.
#[derive(Clone, Debug, PartialEq)]
pub struct SigmaClip {
    /// Retained observations in original input order.
    pub values: Vec<f64>,
    /// Final lower clipping threshold in observation units.
    pub lower: f64,
    /// Final upper clipping threshold in observation units.
    pub upper: f64,
}
fn limited(
    values: &[f64],
    limits: Option<(f64, f64)>,
    inclusive: (bool, bool),
) -> Result<Vec<f64>> {
    sample(values, 1)?;
    if let Some((low, high)) = limits {
        if low.is_nan() || high.is_nan() || low > high {
            return Err(Error::InvalidInput("invalid trimming limits"));
        }
        let out = values
            .iter()
            .copied()
            .filter(|x| {
                (if inclusive.0 { *x >= low } else { *x > low })
                    && (if inclusive.1 { *x <= high } else { *x < high })
            })
            .collect::<Vec<_>>();
        sample(&out, 1)?;
        Ok(out)
    } else {
        Ok(values.to_vec())
    }
}
/// Trimmed mean of retained observations.
///
/// ## Parameters
///
/// Limits select observations by value, rather than removing a proportion. Inclusive tuple selects lower and upper endpoint inclusion.
/// None retains all observations; explicit ddof applies after filtering.
///
/// ## Domain
///
/// Finite nonempty sample; at least one retained observation; valid residual count for the requested statistic.
///
/// ## Returns
///
/// Scalar statistic in observation units (squared units for variance).
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Direct binary64 formulas with bounded observation counts; cancellation can affect extreme samples.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!(sci_stats::tmean(&[1., 2., 100.], Some((0., 3.)), (true,true))?.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `tmean`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tmean.html).
pub fn tmean(values: &[f64], limits: Option<(f64, f64)>, inclusive: (bool, bool)) -> Result<f64> {
    let x = limited(values, limits, inclusive)?;
    mean(&x)
}
/// Trimmed variance of retained observations.
///
/// ## Parameters
///
/// Limits select observations by value, rather than removing a proportion. Inclusive tuple selects lower and upper endpoint inclusion.
/// None retains all observations; explicit ddof applies after filtering.
///
/// ## Domain
///
/// Finite nonempty sample; at least one retained observation; valid residual count for the requested statistic.
///
/// ## Returns
///
/// Scalar statistic in observation units (squared units for variance).
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Direct binary64 formulas with bounded observation counts; cancellation can affect extreme samples.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!(sci_stats::tvar(&[1., 2., 100.], Some((0., 3.)), (true,true), 1)?.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `tvar`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tvar.html).
pub fn tvar(
    values: &[f64],
    limits: Option<(f64, f64)>,
    inclusive: (bool, bool),
    ddof: usize,
) -> Result<f64> {
    let x = limited(values, limits, inclusive)?;
    variance(&x, ddof)
}
/// Trimmed standard deviation of retained observations.
///
/// ## Parameters
///
/// Limits select observations by value, rather than removing a proportion. Inclusive tuple selects lower and upper endpoint inclusion.
/// None retains all observations; explicit ddof applies after filtering.
///
/// ## Domain
///
/// Finite nonempty sample; at least one retained observation; valid residual count for the requested statistic.
///
/// ## Returns
///
/// Scalar statistic in observation units (squared units for variance).
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Direct binary64 formulas with bounded observation counts; cancellation can affect extreme samples.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!(sci_stats::tstd(&[1., 2., 100.], Some((0., 3.)), (true,true), 1)?.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `tstd`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tstd.html).
pub fn tstd(
    values: &[f64],
    limits: Option<(f64, f64)>,
    inclusive: (bool, bool),
    ddof: usize,
) -> Result<f64> {
    let x = limited(values, limits, inclusive)?;
    std(&x, ddof)
}
/// Trimmed standard error of retained observations.
///
/// ## Parameters
///
/// Limits select observations by value, rather than removing a proportion. Inclusive tuple selects lower and upper endpoint inclusion.
/// None retains all observations; explicit ddof applies after filtering.
///
/// ## Domain
///
/// Finite nonempty sample; at least one retained observation; valid residual count for the requested statistic.
///
/// ## Returns
///
/// Scalar statistic in observation units (squared units for variance).
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Direct binary64 formulas with bounded observation counts; cancellation can affect extreme samples.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!(sci_stats::tsem(&[1., 2., 100.], Some((0., 3.)), (true,true), 1)?.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `tsem`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tsem.html).
pub fn tsem(
    values: &[f64],
    limits: Option<(f64, f64)>,
    inclusive: (bool, bool),
    ddof: usize,
) -> Result<f64> {
    let x = limited(values, limits, inclusive)?;
    sem(&x, ddof)
}
/// Trimmed minimum among values above a lower limit.
///
/// ## Parameters
///
/// Optional scalar cutoff; inclusive controls whether the limit itself is retained.
///
/// ## Domain
///
/// Finite nonempty observations; at least one must survive; infinite cutoffs allowed, NaN rejected.
///
/// ## Returns
///
/// Retained extreme observation.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Direct binary64 formulas with bounded observation counts; cancellation can affect extreme samples.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::tmin(&[1., 2., 3.], Some(2.), true)?, 2.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `tmin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tmin.html).
pub fn tmin(values: &[f64], limit: Option<f64>, inclusive: bool) -> Result<f64> {
    let limits = limit.map(|l| (l, f64::INFINITY));
    let x = limited(values, limits, (inclusive, true))?;
    Ok(x.iter().copied().fold(f64::INFINITY, f64::min))
}
/// Trimmed maximum among values below an upper limit.
///
/// ## Parameters
///
/// Optional scalar cutoff; inclusive controls whether the limit itself is retained.
///
/// ## Domain
///
/// Finite nonempty observations; at least one must survive; infinite cutoffs allowed, NaN rejected.
///
/// ## Returns
///
/// Retained extreme observation.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Direct binary64 formulas with bounded observation counts; cancellation can affect extreme samples.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::tmax(&[1., 2., 3.], Some(2.), true)?, 2.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `tmax`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tmax.html).
pub fn tmax(values: &[f64], limit: Option<f64>, inclusive: bool) -> Result<f64> {
    let limits = limit.map(|l| (f64::NEG_INFINITY, l));
    let x = limited(values, limits, (true, inclusive))?;
    Ok(x.iter().copied().fold(f64::NEG_INFINITY, f64::max))
}
/// Removes floor(n*proportion) sorted observations from one tail.
///
/// ## Parameters
///
/// Tail selects lower or upper; output is sorted (SciPy ordering is unspecified).
///
/// ## Domain
///
/// Finite data, proportion in [0,1); empty input returns empty output.
///
/// ## Returns
///
/// Retained sorted observations.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Sorting requires O(n log n) work and O(n) output storage.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// use sci_stats::{trim1, Tail};
/// assert_eq!(trim1(&[1., 2., 3., 4.], 0.25, Tail::Upper)?, vec![1., 2., 3.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `trim1`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.trim1.html).
pub fn trim1(values: &[f64], proportion: f64, tail: Tail) -> Result<Vec<f64>> {
    sample(values, 0)?;
    if !proportion.is_finite() || !(0. ..1.).contains(&proportion) {
        return Err(Error::InvalidInput("trim proportion must be in [0,1)"));
    }
    let mut x = values.to_vec();
    x.sort_by(f64::total_cmp);
    let k = (x.len() as f64 * proportion).floor() as usize;
    Ok(match tail {
        Tail::Lower => x[k..].to_vec(),
        Tail::Upper => x[..x.len() - k].to_vec(),
    })
}
/// Unbiased cumulant estimates k₁ through k₄.
///
/// ## Parameters
///
/// k₁=mean; k₂=n*m₂/(n−1); k₃=n²*m₃/((n−1)(n−2));
/// k₄=n²*((n+1)m₄−3(n−1)m₂²)/((n−1)(n−2)(n−3)); m_j are central moments divided by n.
///
/// ## Domain
///
/// Order in 1..=4 and at least order finite observations.
///
/// ## Returns
///
/// Unbiased cumulant estimator in observation units raised to order.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Direct binary64 formulas with bounded observation counts; cancellation can affect extreme samples.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::kstat(&[1., 2., 3.], 2)?, 1.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `kstat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstat.html).
pub fn kstat(values: &[f64], order: usize) -> Result<f64> {
    if !(1..=4).contains(&order) {
        return Err(Error::InvalidInput("k-statistic order must be in [1,4]"));
    }
    sample(values, order)?;
    let n = values.len() as f64;
    finite(match order {
        1 => mean(values)?,
        2 => variance(values, 1)?,
        3 => n * n * moment(values, 3)? / ((n - 1.) * (n - 2.)),
        _ => {
            n * n * ((n + 1.) * moment(values, 4)? - 3. * (n - 1.) * moment(values, 2)?.powi(2))
                / ((n - 1.) * (n - 2.) * (n - 3.))
        }
    })
}
/// Unbiased estimated variance of k₁ or k₂.
///
/// ## Parameters
///
/// Var(k₁)=k₂/n; Var(k₂)=(2*n*k₂²+(n−1)*k₄)/(n*(n+1)).
///
/// ## Domain
///
/// Order one needs n≥2; order two needs n≥4; finite observations.
///
/// ## Returns
///
/// Estimated estimator variance, units raised to twice order; estimates may be negative.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Direct binary64 formulas with bounded observation counts; cancellation can affect extreme samples.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!(sci_stats::kstatvar(&[1., 2., 3., 5.], 2)?.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `kstatvar`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstatvar.html).
pub fn kstatvar(values: &[f64], order: usize) -> Result<f64> {
    let n = values.len() as f64;
    let k2 = kstat(values, 2)?;
    finite(match order {
        1 => k2 / n,
        2 => (2. * n * k2 * k2 + (n - 1.) * kstat(values, 4)?) / (n * (n + 1.)),
        _ => {
            return Err(Error::InvalidInput(
                "k-statistic variance order must be one or two",
            ));
        }
    })
}
/// Combined significance test for independent uniformly distributed null p-values.
///
/// ## Parameters
///
/// Fisher: −2Σln(p), chi-square survival at df=2n; Pearson: 2Σln(1−p), chi-square CDF of minus statistic.
/// Tippett: min(p), Beta(1,n) CDF; Stouffer: Σw*Φ⁻¹(1−p)/sqrt(Σw²), normal survival.
/// Mudholkar–George: Σln((1−p)/p), t survival after multiplying sqrt(3/n)/π*sqrt((5n+4)/(5n+2)), df=5n+4.
/// Weights are allowed only for Stouffer, positive finite and length-matched.
///
/// ## Domain
///
/// Nonempty probabilities in \[0,1\]. Mixed zero/one extremes can produce NaN for Stouffer or Mudholkar–George.
///
/// ## Returns
///
/// Method-specific statistic and combined p-value.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// O(n) work with statrs tail probabilities. Null assumptions require independent tests of continuous distributions.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// use sci_stats::{combine_pvalues, CombineMethod};
/// assert!(combine_pvalues(&[0.1, 0.02], CombineMethod::Fisher, None)?.pvalue < 0.05);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `combine_pvalues`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.combine_pvalues.html).
pub fn combine_pvalues(
    pvalues: &[f64],
    method: CombineMethod,
    weights: Option<&[f64]>,
) -> Result<TestResult> {
    sample(pvalues, 1)?;
    for &p in pvalues {
        probability_check(p)?;
    }
    if weights.is_some() && !matches!(method, CombineMethod::Stouffer) {
        return Err(Error::InvalidInput(
            "weights are supported only for Stouffer",
        ));
    }
    let n = pvalues.len() as f64;
    let d = ChiSquared::new(2. * n)
        .map_err(|_| Error::NumericalFailure("invalid combination distribution"))?;
    let (statistic, pvalue) = match method {
        CombineMethod::Fisher => {
            let s = -2. * pvalues.iter().map(|p| p.ln()).sum::<f64>();
            (s, d.sf(s))
        }
        CombineMethod::Pearson => {
            let s = 2. * pvalues.iter().map(|p| (-p).ln_1p()).sum::<f64>();
            (s, d.cdf(-s))
        }
        CombineMethod::Tippett => {
            let s = pvalues.iter().copied().fold(1., f64::min);
            (s, -(n * (-s).ln_1p()).exp_m1())
        }
        CombineMethod::Stouffer => {
            let owned;
            let w = if let Some(w) = weights {
                if w.len() != pvalues.len() {
                    return Err(Error::DimensionMismatch);
                }
                sample(w, 1)?;
                for &v in w {
                    positive(v)?;
                }
                w
            } else {
                owned = vec![1.; pvalues.len()];
                &owned
            };
            let maximum = w.iter().copied().fold(0., f64::max);
            let norm = w.iter().map(|v| (v / maximum).powi(2)).sum::<f64>().sqrt();
            let mut s = 0.;
            for (&p, &w) in pvalues.iter().zip(w) {
                s += (w / maximum) * -normal_ppf(p, 0., 1.)?;
            }
            s /= norm;
            (
                s,
                if s.is_nan() {
                    f64::NAN
                } else {
                    normal(0., 1.)?.sf(s)
                },
            )
        }
        CombineMethod::MudholkarGeorge => {
            let s = pvalues.iter().map(|p| (-p).ln_1p() - p.ln()).sum::<f64>();
            let df = 5. * n + 4.;
            let t = s * (3. / n).sqrt() / std::f64::consts::PI * (df / (df - 2.)).sqrt();
            (
                s,
                if t.is_nan() {
                    f64::NAN
                } else {
                    StudentsT::new(0., 1., df)
                        .map_err(|_| Error::NumericalFailure("invalid combination distribution"))?
                        .sf(t)
                },
            )
        }
    };
    Ok(TestResult { statistic, pvalue })
}
/// Iterative sigma clipping to `[mean−low*std(x,0), mean+high*std(x,0)]`.
///
/// ## Parameters
///
/// Low and high are positive standard-deviation multipliers. Iteration recomputes mean and population standard deviation.
///
/// ## Domain
///
/// Finite nonempty observations. Removing all observations or exceeding 128 passes returns an error.
///
/// ## Returns
///
/// Retained observations and final clipping thresholds.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// At most 128 linear passes; constant samples are retained with equal thresholds.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::sigmaclip(&[1., 1., 1.], 3., 3.)?.values, vec![1.;3]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `sigmaclip`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sigmaclip.html).
pub fn sigmaclip(values: &[f64], low: f64, high: f64) -> Result<SigmaClip> {
    sample(values, 1)?;
    positive(low)?;
    positive(high)?;
    let mut current = values.to_vec();
    for _ in 0..128 {
        let m = mean(&current)?;
        let s = std(&current, 0)?;
        let lower = finite(m - low * s)?;
        let upper = finite(m + high * s)?;
        let next = current
            .iter()
            .copied()
            .filter(|x| *x >= lower && *x <= upper)
            .collect::<Vec<_>>();
        if next.len() == current.len() {
            return Ok(SigmaClip {
                values: current,
                lower,
                upper,
            });
        }
        sample(&next, 1)?;
        current = next;
    }
    Err(Error::NumericalFailure(
        "sigma clipping exceeded 128 passes",
    ))
}
