use super::*;
use statrs::distribution::{Binomial, Discrete, DiscreteCDF};

/// Directional mean and concentration for vector observations.
#[derive(Clone, Debug, PartialEq)]
pub struct DirectionalStats {
    /// Normalized mean vector; all NaN when the mean vector is zero.
    pub mean_direction: Vec<f64>,
    /// Length of the mean of supplied or normalized observations.
    pub mean_resultant_length: f64,
}

/// Exact binomial test of `H₀:p=probability`; two-sided sums masses no larger than the observed mass.
///
/// ## Parameters
///
/// Returns observed success proportion as statistic; lower/upper alternatives concern success probability.
///
/// ## Domain
///
/// Trials in \[1,65536\], successes≤trials, probability in \[0,1\]. Null probabilities zero and one are allowed.
///
/// ## Returns
///
/// Observed proportion and exact tail p-value.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Delegates binomial masses and tails to statrs; at most 65,537 probability evaluations. Uses SciPy's 1e-7 relative probability tie tolerance in log space.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// use sci_stats::{inference::binomtest, inference::Alternative};
/// assert_eq!(binomtest(5, 10, 0.5, Alternative::TwoSided)?.pvalue, 1.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `binomtest`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).
pub fn binomtest(
    successes: usize,
    trials: usize,
    probability: f64,
    alternative: Alternative,
) -> Result<TestResult> {
    if trials == 0 || trials > 65536 || successes > trials {
        return Err(Error::InvalidInput(
            "invalid binomial counts (trial limit 65536)",
        ));
    }
    probability_check(probability)?;
    let d = Binomial::new(probability, trials as u64)
        .map_err(|_| Error::InvalidInput("invalid binomial parameters"))?;
    let k = successes as u64;
    let pvalue = match alternative {
        Alternative::Less => d.cdf(k),
        Alternative::Greater => {
            if k == 0 {
                1.
            } else {
                d.sf(k - 1)
            }
        }
        Alternative::TwoSided => {
            let threshold = d.ln_pmf(k) + (1e-7_f64).ln_1p();
            (0..=trials as u64)
                .filter(|&j| d.ln_pmf(j) <= threshold)
                .map(|j| d.pmf(j))
                .sum::<f64>()
                .min(1.)
        }
    };
    Ok(TestResult {
        statistic: successes as f64 / trials as f64,
        pvalue,
    })
}
/// Directional mean `m=mean(v_i)`, direction `m/||m||`, resultant length `||m||`.
///
/// ## Parameters
///
/// Rows are vectors with a common dimension. If normalize=true, each observation is first divided by its Euclidean norm. Otherwise unit vectors are assumed.
///
/// ## Domain
///
/// Nonempty finite rectangular observations, dimensions 1..=64, total≤1048576 coordinates; normalized observations must be nonzero. Zero mean yields NaN direction.
///
/// ## Returns
///
/// Mean direction vector and mean resultant length.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// O(nd) work; stable hypot avoids squared-norm overflow; a norm outside finite range is rejected.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::descriptive::directional_stats(&[vec![1., 0.], vec![1., 0.]], true)?.mean_resultant_length, 1.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `directional_stats`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.directional_stats.html).
pub fn directional_stats(observations: &[Vec<f64>], normalize: bool) -> Result<DirectionalStats> {
    if observations.is_empty() {
        return Err(Error::InvalidInput("directional sample must not be empty"));
    }
    let d = observations[0].len();
    if d == 0
        || d > 64
        || observations
            .len()
            .checked_mul(d)
            .is_none_or(|n| n > MAX_SAMPLES)
    {
        return Err(Error::InvalidInput(
            "directional sample exceeds shape limits",
        ));
    }
    let mut m = vec![0.; d];
    let n = observations.len() as f64;
    for row in observations {
        if row.len() != d {
            return Err(Error::DimensionMismatch);
        }
        sample(row, 1)?;
        let length = if normalize {
            let norm = row.iter().fold(0_f64, |s, x| s.hypot(*x));
            positive(norm)?;
            norm
        } else {
            1.
        };
        for j in 0..d {
            m[j] += (row[j] / length) / n;
        }
    }
    let length = match m.iter().fold(0_f64, |s, x| s.hypot(*x)) {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }?;
    let direction = m.iter().map(|x| x / length).collect();
    Ok(DirectionalStats {
        mean_direction: direction,
        mean_resultant_length: length,
    })
}
/// Vasicek differential-entropy estimate `mean(ln(n*(x_(i+m)−x_(i−m))/(2m)))`.
///
/// ## Parameters
///
/// Explicit positive window m; sorted endpoints are repeated outside the sample. Supports Vasicek only. Natural logarithms unless base is supplied; units are relative to the observation scale.
///
/// ## Domain
///
/// Finite observations with 1≤m<n/2; duplicate order-statistic spacings can give negative infinity. Base positive and unequal to one.
///
/// ## Returns
///
/// Entropy estimate in nats or the chosen logarithm units.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// O(n log n) sorting and O(n) spacings; estimator bias depends on the window and underlying density. No automatic method selection.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!(sci_stats::descriptive::differential_entropy(&[1., 2., 3., 4., 5.], 1, None)?.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `differential_entropy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.differential_entropy.html).
pub fn differential_entropy(values: &[f64], window: usize, base: Option<f64>) -> Result<f64> {
    let x = sorted(values)?;
    let n = x.len();
    if window == 0 || window >= n.div_ceil(2) {
        return Err(Error::InvalidInput(
            "entropy window must satisfy 1<=2*window<n",
        ));
    }
    let denominator = if let Some(b) = base {
        positive(b)?;
        if b == 1. {
            return Err(Error::InvalidInput("entropy base must not be one"));
        }
        b.ln()
    } else {
        1.
    };
    let mut sum = 0.;
    for i in 0..n {
        let distance = match x[(i + window).min(n - 1)] - x[i.saturating_sub(window)] {
            value if value.is_finite() => Ok(value),
            _ => Err(crate::Error::NonFiniteEvaluation),
        }?;
        sum += distance.ln() + (n as f64 / (2. * window as f64)).ln();
    }
    Ok(sum / n as f64 / denominator)
}
/// Sample L-moment from probability-weighted moments `b_r=Σ x_(i)*C(i,r)/C(n−1,r)/n`.
///
/// ## Parameters
///
/// Orders 1..=4 only: L₁=b₀, L₂=2b₁−b₀, L₃=6b₂−6b₁+b₀, L₄=20b₃−30b₂+12b₁−b₀.
/// If standardize=true, orders three and four are divided by L₂, as in SciPy; scalar order adaptation.
///
/// ## Domain
///
/// Finite nonempty observations, order in 1..=4, n≥order. Standardized high orders of constant samples return NaN.
///
/// ## Returns
///
/// Scalar L-moment (observation units), or dimensionless L-skewness/L-kurtosis.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// O(n log n) sorting plus O(n*order²) bounded small-order weights. Scaling avoids overflow; constant data is treated exactly.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!((sci_stats::descriptive::lmoment(&[1., 2., 3., 4.], 1, false)? - 2.5).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `lmoment`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lmoment.html).
pub fn lmoment(values: &[f64], order: usize, standardize: bool) -> Result<f64> {
    if !(1..=4).contains(&order) {
        return Err(Error::InvalidInput("L-moment order must be in [1,4]"));
    }
    sample(values, order)?;
    let x = sorted(values)?;
    if x[0] == x[x.len() - 1] {
        return Ok(if order == 1 {
            x[0]
        } else if order == 2 || !standardize {
            0.
        } else {
            f64::NAN
        });
    }
    let n = x.len();
    let mut b = [0.; 4];
    let scale = x.iter().fold(0_f64, |a, v| a.max(v.abs()));
    if scale == 0. {
        return Ok(if standardize && order > 2 {
            f64::NAN
        } else {
            0.
        });
    }
    for (r, br) in b.iter_mut().enumerate().take(order) {
        for (i, &v) in x.iter().enumerate().skip(r) {
            let mut weight = 1.;
            for j in 0..r {
                weight *= (i - j) as f64 / (n - 1 - j) as f64;
            }
            *br += (v / scale) * weight / n as f64;
        }
    }
    let l2 = 2. * b[1] - b[0];
    let l = match order {
        1 => b[0],
        2 => l2,
        3 => 6. * b[2] - 6. * b[1] + b[0],
        _ => 20. * b[3] - 30. * b[2] + 12. * b[1] - b[0],
    };
    if standardize && order > 2 {
        Ok(l / l2)
    } else {
        match l * scale {
            value if value.is_finite() => Ok(value),
            _ => Err(crate::Error::NonFiniteEvaluation),
        }
    }
}
