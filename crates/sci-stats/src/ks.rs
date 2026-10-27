use super::*;

/// Kolmogorov–Smirnov statistic, tail probability, and extremum metadata.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KsResult {
    /// Maximum selected distance between cumulative distribution functions.
    pub statistic: f64,
    /// Null tail probability, using the method documented by the function.
    pub pvalue: f64,
    /// Observation where the selected CDF difference is attained.
    pub location: f64,
    /// +1 when the first empirical CDF exceeds the reference, −1 otherwise.
    pub sign: i8,
}
/// Ordered observations and theoretical quantiles for a probability plot.
#[derive(Clone, Debug, PartialEq)]
pub struct ProbabilityPlot {
    /// Filliben theoretical quantiles, ordered ascending.
    pub theoretical: Vec<f64>,
    /// Observations ordered ascending, matching theoretical quantiles.
    pub ordered: Vec<f64>,
}
fn select(plus: (f64, f64), minus: (f64, f64), alternative: Alternative) -> (f64, f64, i8) {
    if matches!(alternative, Alternative::Greater)
        || matches!(alternative, Alternative::TwoSided) && plus.0 >= minus.0
    {
        (plus.0, plus.1, 1)
    } else {
        (minus.0, minus.1, -1)
    }
}
/// One-sample Kolmogorov–Smirnov distance `D+=max(i/n−F(x_i))`, `D−=max(F(x_i)−(i−1)/n)`.
///
/// ## Parameters
///
/// Null is a fully specified continuous distribution. Greater selects D+, Less D−, TwoSided their maximum. Distribution parameters must be supplied independently of the tested observations.
///
/// ## Domain
///
/// Nonempty finite sample; continuous reference distributions only. Ties violate the continuous-null assumption; no automatic tie correction.
///
/// ## Returns
///
/// CDF distance, p-value, location, and sign.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// O(n log n) sorting. Reuses Sci special functions: exact finite-n one-sided Smirnov tails, asymptotic two-sided Kolmogorov tails (SciPy method="asymp"). Finite-n two-sided exact distribution is unsupported.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// use sci_stats::{ks_1samp, Distribution, Alternative};
/// assert!(ks_1samp(&[-1., 0., 1.], Distribution::Normal {mean:0.,std_dev:1.}, Alternative::TwoSided)?.statistic > 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `ks_1samp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_1samp.html).
pub fn ks_1samp(
    values: &[f64],
    reference: Distribution,
    alternative: Alternative,
) -> Result<KsResult> {
    continuous_null(reference)?;
    let x = sorted(values)?;
    let n = x.len() as f64;
    let mut plus = (-1., x[0]);
    let mut minus = (-1., x[0]);
    for (i, &v) in x.iter().enumerate() {
        let f = cdf(reference, v)?;
        let p = (i + 1) as f64 / n - f;
        let m = f - i as f64 / n;
        if p > plus.0 {
            plus = (p, v);
        }
        if m > minus.0 {
            minus = (m, v);
        }
    }
    let choice = if matches!(alternative, Alternative::TwoSided) && plus.0 == minus.0 {
        Alternative::Less
    } else {
        alternative
    };
    let (statistic, location, sign) = select(plus, minus, choice);
    let pvalue = if matches!(alternative, Alternative::TwoSided) {
        crate::special::kolmogorov(n.sqrt() * statistic)
    } else {
        crate::special::smirnov(x.len() as i32, statistic)
    };
    Ok(KsResult {
        statistic,
        pvalue: match pvalue {
            value if value.is_finite() => Ok(value),
            _ => Err(crate::Error::NonFiniteEvaluation),
        }?,
        location,
        sign,
    })
}
/// Two-sample Kolmogorov–Smirnov distance between empirical CDFs.
///
/// ## Parameters
///
/// Greater tests F_first>F_second; Less the opposite. Effective count e=n*m/(n+m); two-sided p=Kolmogorov(sqrt(e)*D), one-sided p=exp(−2eD²).
///
/// ## Domain
///
/// Nonempty finite samples; null assumes continuous distributions. Ties are accepted but invalidate the ordinary continuous null.
///
/// ## Returns
///
/// Selected distance, asymptotic-limit p-value, observed location, and sign.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// O((n+m) log(n+m)) sorting and linear CDF merge. Uses the limiting null, without SciPy's finite-sample corrections or exact enumeration; small-sample p-values can differ materially.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// use sci_stats::{ks_2samp, Alternative};
/// assert_eq!(ks_2samp(&[1.,2.], &[1.,2.], Alternative::TwoSided)?.statistic, 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `ks_2samp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html).
pub fn ks_2samp(first: &[f64], second: &[f64], alternative: Alternative) -> Result<KsResult> {
    sample(first, 1)?;
    sample(second, 1)?;
    if first.len() + second.len() > MAX_SAMPLES {
        return Err(Error::InvalidInput("combined KS sample exceeds limit"));
    }
    let a = sorted(first)?;
    let b = sorted(second)?;
    let mut knots = a.iter().chain(&b).copied().collect::<Vec<_>>();
    knots.sort_by(f64::total_cmp);
    knots.dedup();
    let mut plus = (-1., knots[0]);
    let mut minus = (-1., knots[0]);
    let (mut i, mut j) = (0, 0);
    for v in knots {
        while i < a.len() && a[i] <= v {
            i += 1;
        }
        while j < b.len() && b[j] <= v {
            j += 1;
        }
        let delta = i as f64 / a.len() as f64 - j as f64 / b.len() as f64;
        if delta > plus.0 {
            plus = (delta, v);
        }
        if -delta > minus.0 {
            minus = (-delta, v);
        }
    }
    let (statistic, location, sign) = select(plus, minus, alternative);
    let n = a.len() as f64;
    let m = b.len() as f64;
    let effective = n * m / (n + m);
    let pvalue = if matches!(alternative, Alternative::TwoSided) {
        crate::special::kolmogorov(effective.sqrt() * statistic)
    } else {
        (-2. * effective * statistic * statistic).exp()
    };
    Ok(KsResult {
        statistic,
        pvalue: match pvalue {
            value if value.is_finite() => Ok(value),
            _ => Err(crate::Error::NonFiniteEvaluation),
        }?,
        location,
        sign,
    })
}
/// Probability-plot quantiles `F⁻¹(p_i)` paired with sorted observations, using Filliben plotting positions.
///
/// ## Parameters
///
/// For n>1, endpoint probabilities are 1−0.5^(1/n) and 0.5^(1/n); interior probabilities (i−0.3175)/(n+0.365), with one-based i.
/// For n=1, p=0.5. Returns numeric plot data only (SciPy fit=false); no plotting or fitted-line overload.
///
/// ## Domain
///
/// Nonempty finite observations and valid distribution parameters.
///
/// ## Returns
///
/// Theoretical quantiles and ordered observations, equal-length vectors.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Sorting plus n bounded backend quantile inversions. Discrete distributions are accepted but may create repeated theoretical quantiles; fitting is left to linregress.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// use sci_stats::{probplot, Distribution};
/// assert_eq!(probplot(&[2.,1.], Distribution::Normal {mean:0.,std_dev:1.})?.ordered, vec![1.,2.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `probplot`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html).
pub fn probplot(values: &[f64], distribution: Distribution) -> Result<ProbabilityPlot> {
    let ordered = sorted(values)?;
    let n = ordered.len();
    let mut theoretical = Vec::with_capacity(n);
    for i in 0..n {
        let p = if n == 1 {
            0.5
        } else if i == 0 {
            -(0.5_f64.ln() / n as f64).exp_m1()
        } else if i == n - 1 {
            (0.5_f64.ln() / n as f64).exp()
        } else {
            (i as f64 + 1. - 0.3175) / (n as f64 + 0.365)
        };
        theoretical.push(ppf(distribution, p)?);
    }
    Ok(ProbabilityPlot {
        theoretical,
        ordered,
    })
}
