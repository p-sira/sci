use super::*;

fn small_table(table: &[f64; 4]) -> Result<[usize; 4]> {
    let mut counts = [0; 4];
    for (i, &v) in table.iter().enumerate() {
        if !v.is_finite() || !(0. ..=64.).contains(&v) || v.fract() != 0. {
            return Err(Error::InvalidInput(
                "exact table requires nonnegative integer counts totaling at most 64",
            ));
        }
        counts[i] = v as usize;
    }
    if counts.iter().sum::<usize>() > 64 {
        return Err(Error::InvalidInput("unconditional exact count limit is 64"));
    }
    Ok(counts)
}
fn binomial_coefficients(n: usize) -> Vec<u128> {
    let mut result = vec![1; n + 1];
    for k in 1..=n {
        result[k] = result[k - 1] * (n - k + 1) as u128 / k as u128;
    }
    result
}
fn bernstein_split(coefficients: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = coefficients.len() - 1;
    let mut work = coefficients.to_vec();
    let mut left = vec![0.; n + 1];
    let mut right = vec![0.; n + 1];
    left[0] = work[0];
    right[n] = work[n];
    for step in 1..=n {
        for i in 0..=n - step {
            work[i] = work[i] / 2. + work[i + 1] / 2.;
        }
        left[step] = work[0];
        right[n - step] = work[n - step];
    }
    (left, right)
}
// A Bernstein polynomial stays within its coefficient hull. Subdivision gives
// a global upper bound, rather than relying on one local nuisance-parameter maximum.
fn bernstein_maximum(coefficients: Vec<f64>) -> Result<f64> {
    let mut best = coefficients[0].max(coefficients[coefficients.len() - 1]);
    let mut stack = vec![(coefficients, 0)];
    let mut visits = 0;
    while let Some((coefficients, depth)) = stack.pop() {
        visits += 1;
        if visits > 65536 {
            return Err(Error::NumericalFailure(
                "exact-test nuisance subdivision limit exceeded",
            ));
        }
        let upper = coefficients.iter().copied().fold(0_f64, f64::max);
        if upper <= best + 1e-12 {
            continue;
        }
        if depth >= 64 {
            return Err(Error::NumericalFailure(
                "exact-test nuisance depth limit exceeded",
            ));
        }
        let (left, right) = bernstein_split(&coefficients);
        best = best.max(left[left.len() - 1]);
        stack.push((right, depth + 1));
        stack.push((left, depth + 1));
    }
    Ok(best.clamp(0., 1.))
}
fn unconditional_probability(
    n1: usize,
    n2: usize,
    extreme: impl Fn(usize, usize) -> Result<bool>,
) -> Result<f64> {
    let c1 = binomial_coefficients(n1);
    let c2 = binomial_coefficients(n2);
    let denominator = binomial_coefficients(n1 + n2);
    let mut numerators = vec![0_u128; n1 + n2 + 1];
    for (x, &cx) in c1.iter().enumerate() {
        for (y, &cy) in c2.iter().enumerate() {
            if extreme(x, y)? {
                numerators[x + y] += cx * cy;
            }
        }
    }
    bernstein_maximum(
        numerators
            .iter()
            .zip(denominator)
            .map(|(&n, d)| n as f64 / d as f64)
            .collect(),
    )
}
fn barnard_statistic(x: usize, y: usize, n1: usize, n2: usize, pooled: bool) -> f64 {
    let p1 = x as f64 / n1 as f64;
    let p2 = y as f64 / n2 as f64;
    if p1 == p2 {
        return 0.;
    }
    let variance = if pooled {
        let p = (x + y) as f64 / (n1 + n2) as f64;
        p * (1. - p) * (1. / n1 as f64 + 1. / n2 as f64)
    } else {
        p1 * (1. - p1) / n1 as f64 + p2 * (1. - p2) / n2 as f64
    };
    (p1 - p2) / variance.sqrt()
}

/// Barnard unconditional exact test `p=sup_(θ∈[0,1]) P_θ(W at least as extreme as W_obs)`.
///
/// ## Parameters and conventions
///
/// `table` is row-major `[success_1,success_2,failure_1,failure_2]`; columns
/// are independent binomial samples. `pooled` selects the Wald variance
/// `p*(1−p)*(1/n1+1/n2)` with p=(x1+x2)/(n1+n2), otherwise
/// `p1*(1−p1)/n1+p2*(1−p2)/n2`. W=(p1−p2)/sqrt(variance); equal proportions
/// give W=0. `alternative` chooses |W|, lower W or upper W tails. The nuisance
/// binomial success probability is maximized globally. SciPy's SHGO sampling
/// parameter is replaced by deterministic Bernstein subdivision.
///
/// ## Domain and special values
///
/// Nonnegative integer f64 cell counts, total≤64. A zero column gives NaN
/// statistic and p=1. Unpooled perfect separation can give infinite W.
/// NaN/infinity/fractional counts are errors.
///
/// ## Returns
///
/// Dimensionless signed Wald statistic and unconditional p-value in `[0,1]`.
///
/// ## Errors
///
/// Invalid tables/counts or exhausted nuisance-search work limits return errors.
///
/// ## Numerical behavior
///
/// Binomial coefficients and rejection-region sums are exact u128 integers.
/// Probability is a degree≤64 Bernstein polynomial. Its coefficient-hull upper
/// bound is subdivided until within 1e-12 of the best evaluated value, with at
/// most 65536 nodes and depth 64; rounding adds binary64 error. Tail comparison
/// uses the computed Wald statistic, matching SciPy. No asymptotic probability
/// approximation or local-optimizer assumption is used.
///
/// ## Examples
///
/// ```
/// let r = sci_stats::barnard_exact(&[7.,12.,8.,3.],sci_stats::Alternative::Less,true)?;
/// assert!(r.pvalue < 0.05);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy barnard_exact and Barnard (1947)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.barnard_exact.html).
pub fn barnard_exact(
    table: &[f64; 4],
    alternative: Alternative,
    pooled: bool,
) -> Result<TestResult> {
    let [x, y, f1, f2] = small_table(table)?;
    let (n1, n2) = (x + f1, y + f2);
    if n1 == 0 || n2 == 0 {
        return Ok(TestResult {
            statistic: f64::NAN,
            pvalue: 1.,
        });
    }
    let statistic = barnard_statistic(x, y, n1, n2, pooled);
    let pvalue = unconditional_probability(n1, n2, |x, y| {
        let value = barnard_statistic(x, y, n1, n2, pooled);
        Ok(match alternative {
            Alternative::TwoSided => value.abs() >= statistic.abs(),
            Alternative::Less => value <= statistic,
            Alternative::Greater => value >= statistic,
        })
    })?;
    Ok(TestResult { statistic, pvalue })
}
fn fisher_tail(x: usize, y: usize, n1: usize, n2: usize, alternative: Alternative) -> Result<f64> {
    let (draws, observed) = if alternative == Alternative::Less {
        (n1, x)
    } else {
        (n2, y)
    };
    let successes = x + y;
    let failures = n1 + n2 - successes;
    let maximum = draws.min(successes);
    if observed >= maximum {
        return Ok(1.);
    }
    let success = binomial_coefficients(successes);
    let failure = binomial_coefficients(failures);
    let denominator = binomial_coefficients(n1 + n2)[draws];
    let numerator = (draws.saturating_sub(failures)..=observed)
        .map(|j| success[j] * failure[draws - j])
        .sum::<u128>();
    Ok(numerator as f64 / denominator as f64)
}
fn boschloo_one_sided(
    x: usize,
    y: usize,
    n1: usize,
    n2: usize,
    alternative: Alternative,
) -> Result<TestResult> {
    let statistic = fisher_tail(x, y, n1, n2, alternative)?;
    let pvalue = unconditional_probability(n1, n2, |x, y| {
        Ok(fisher_tail(x, y, n1, n2, alternative)? <= statistic * (1. + 1e-13))
    })?;
    Ok(TestResult { statistic, pvalue })
}

/// Boschloo unconditional exact test with Fisher's one-sided probability as statistic.
///
/// ## Parameters and conventions
///
/// `table` is row-major `[success_1,success_2,failure_1,failure_2]` with independent
/// binomial columns. For each table, F is its Fisher exact one-sided tail; the
/// rejection region is `F≤F_obs*(1+1e-13)`. Maximize its binomial probability
/// over common success probability θ∈`[0,1]`. TwoSided doubles the smaller of
/// the two one-sided unconditional p-values, clipped at one, and returns that
/// side's Fisher statistic, matching SciPy. SHGO's sample count is adapted to
/// deterministic globally bounded Bernstein subdivision.
///
/// ## Domain and special values
///
/// Nonnegative integer f64 counts with total≤64. A zero column gives NaN
/// fields, matching SciPy. Nonfinite and fractional counts are errors.
///
/// ## Returns
///
/// Fisher probability statistic and unconditional p-value, both dimensionless.
///
/// ## Errors
///
/// Invalid counts/domains or exhausted subdivision limits return errors.
///
/// ## Numerical behavior
///
/// Exact u128 hypergeometric/binomial coefficient sums, avoiding rounded PMF
/// accumulation above one in the general hypergeometric backend.
/// Nuisance maximization uses [`barnard_exact`]'s 1e-12 coefficient-hull bound,
/// 65536-node and depth-64 limits, up to twice for two-sided tests. Fisher tail
/// rounding can affect rejection-region ties; the explicit relative guard
/// matches SciPy. This differs from a two-sided Fisher probability.
///
/// ## Examples
///
/// ```
/// let r = sci_stats::boschloo_exact(&[7.,12.,8.,3.],sci_stats::Alternative::Less)?;
/// assert!(r.pvalue < 0.05);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy boschloo_exact and Boschloo (1970)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boschloo_exact.html).
pub fn boschloo_exact(table: &[f64; 4], alternative: Alternative) -> Result<TestResult> {
    let [x, y, f1, f2] = small_table(table)?;
    let (n1, n2) = (x + f1, y + f2);
    if n1 == 0 || n2 == 0 {
        return Ok(TestResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        });
    }
    if alternative != Alternative::TwoSided {
        return boschloo_one_sided(x, y, n1, n2, alternative);
    }
    let lower = boschloo_one_sided(x, y, n1, n2, Alternative::Less)?;
    let upper = boschloo_one_sided(x, y, n1, n2, Alternative::Greater)?;
    let mut result = if lower.pvalue < upper.pvalue {
        lower
    } else {
        upper
    };
    result.pvalue = (2. * result.pvalue).min(1.);
    Ok(result)
}

/// Poisson means E-test `T=(k1/n1−k2/n2−diff)/sqrt(k1/n1²+k2/n2²)`.
///
/// ## Parameters and conventions
///
/// `k1`,`k2` are nonnegative integer event counts, and `n1`,`n2` positive
/// exposure lengths in common time/space/sample units. The null rate difference
/// is `diff≥0` in events per exposure unit. Null rates are
/// `λ2=(k1+k2−n1*diff)/(n1+n2)` and λ1=λ2+diff. The E-test sums independent
/// Poisson(n1*λ1)/Poisson(n2*λ2) masses over pivot statistics at least as extreme
/// as observed, using |T| for TwoSided. Negative differences require exchanging
/// the two samples and reversing the alternative, as in SciPy.
///
/// ## Domain and special values
///
/// Finite nonnegative integer f64 counts totaling≤2048; finite positive exposures
/// and finite diff≥0. λ2≤0 returns T=0,p=1, matching SciPy. A zero-count null
/// pivot is NaN when diff=0 and is excluded from tails, matching SciPy's formula.
/// NaN/infinity inputs are errors; intermediate rates/variances must be representable.
///
/// ## Returns
///
/// Dimensionless signed pivot and approximate unconditional E-test p-value.
///
/// ## Errors
///
/// Invalid domains, unrepresentable arithmetic or more than 16777216 joint
/// Poisson support points return errors.
///
/// ## Numerical behavior
///
/// Private shared Poisson masses/quantiles; sums omit lower tails below 1e-10
/// and upper tails above 1−1e-16, matching SciPy's truncation. Total omitted
/// probability is at most about 2e-10, plus floating-point error. Counts and
/// joint-grid work are bounded before allocation. Very small p-values can
/// underflow; this fitted-null E-test is not a conditional binomial test.
///
/// ## Examples
///
/// ```
/// let r = sci_stats::poisson_means_test(0.,100.,3.,100.,0.,sci_stats::Alternative::TwoSided)?;
/// assert!((r.pvalue-0.08837900945483518).abs() < 1e-10);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy poisson_means_test and Krishnamoorthy–Thomson (2004)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson_means_test.html).
pub fn poisson_means_test(
    k1: f64,
    n1: f64,
    k2: f64,
    n2: f64,
    diff: f64,
    alternative: Alternative,
) -> Result<TestResult> {
    if !k1.is_finite()
        || !k2.is_finite()
        || k1 < 0.
        || k2 < 0.
        || k1.fract() != 0.
        || k2.fract() != 0.
        || k1 + k2 > 2048.
    {
        return Err(Error::InvalidInput(
            "Poisson counts must be integers totaling <=2048",
        ));
    }
    positive(n1)?;
    positive(n2)?;
    if !diff.is_finite() || diff < 0. {
        return Err(Error::InvalidInput(
            "Poisson rate difference must be finite and nonnegative",
        ));
    }
    let total = match n1 + n2 {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }?;
    let lambda2 = match (k1 + k2) / total - diff * (n1 / total) {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }?;
    if lambda2 <= 0. {
        return Ok(TestResult {
            statistic: 0.,
            pvalue: 1.,
        });
    }
    let pivot = |x: f64, y: f64| -> Result<f64> {
        let rate1 = match x / n1 {
            value if value.is_finite() => Ok(value),
            _ => Err(crate::Error::NonFiniteEvaluation),
        }?;
        let rate2 = match y / n2 {
            value if value.is_finite() => Ok(value),
            _ => Err(crate::Error::NonFiniteEvaluation),
        }?;
        let variance = match rate1 / n1 + rate2 / n2 {
            value if value.is_finite() => Ok(value),
            _ => Err(crate::Error::NonFiniteEvaluation),
        }?;
        let numerator = match rate1 - rate2 - diff {
            value if value.is_finite() => Ok(value),
            _ => Err(crate::Error::NonFiniteEvaluation),
        }?;
        let value = numerator / variance.sqrt();
        if variance == 0. {
            Ok(value)
        } else {
            match value {
                value if value.is_finite() => Ok(value),
                _ => Err(crate::Error::NonFiniteEvaluation),
            }
        }
    };
    let statistic = pivot(k1, k2)?;
    let d1 = Distribution::Poisson {
        rate: match n1 * (lambda2 + diff) {
            value if value.is_finite() => Ok(value),
            _ => Err(crate::Error::NonFiniteEvaluation),
        }?,
    };
    let d2 = Distribution::Poisson {
        rate: match n2 * lambda2 {
            value if value.is_finite() => Ok(value),
            _ => Err(crate::Error::NonFiniteEvaluation),
        }?,
    };
    let lower1 = ppf(d1, 1e-10)? as usize;
    let lower2 = ppf(d2, 1e-10)? as usize;
    let upper1 = ppf(d1, 1. - 1e-16)? as usize;
    let upper2 = ppf(d2, 1. - 1e-16)? as usize;
    if (upper1 - lower1 + 1)
        .checked_mul(upper2 - lower2 + 1)
        .is_none_or(|w| w > 16_777_216)
    {
        return Err(Error::InvalidInput(
            "Poisson joint support work limit exceeded",
        ));
    }
    let masses1 = (lower1..=upper1)
        .map(|x| pdf(d1, x as f64))
        .collect::<Result<Vec<_>>>()?;
    let masses2 = (lower2..=upper2)
        .map(|y| pdf(d2, y as f64))
        .collect::<Result<Vec<_>>>()?;
    let mut terms = Vec::with_capacity(masses2.len());
    for (j, &p2) in masses2.iter().enumerate() {
        let mut probability = 0.;
        for (i, &p1) in masses1.iter().enumerate() {
            let value = pivot((lower1 + i) as f64, (lower2 + j) as f64)?;
            let extreme = match alternative {
                Alternative::TwoSided => value.abs() >= statistic.abs(),
                Alternative::Less => value <= statistic,
                Alternative::Greater => value >= statistic,
            };
            if extreme {
                probability += p1 * p2;
            }
        }
        terms.push(probability);
    }
    let probability = terms.iter().sum::<f64>().clamp(0., 1.);
    Ok(TestResult {
        statistic,
        pvalue: probability,
    })
}
