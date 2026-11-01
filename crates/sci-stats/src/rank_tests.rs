use super::*;

/// Normalization convention for Kendall rank correlation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KendallVariant {
    /// Tau-b adjusts for tied pairs independently in each variable.
    B,
    /// Stuart tau-c adjusts for the minimum number of distinct categories.
    C,
}

/// Kendall rank correlation: `τ_b=S/sqrt((M−T_x)(M−T_y))` or
/// `τ_c=2S/(n²(c−1)/c)`.
///
/// ## Parameters and conventions
///
/// `x` and `y` are explicit paired ordinal values; only comparisons are used.
/// S is concordant minus discordant pairs, M=n(n−1)/2, T_x and T_y count
/// tied pairs, and c is the smaller number of distinct categories.
/// Both variants use the normal-null variance of S:
/// `V=[n(n−1)(2n+5)−A_x−A_y]/18+2T_xT_y/(n(n−1))+B_xB_y/(9n(n−1)(n−2))`,
/// where `A=Σt(t−1)(2t+5)` and `B=Σt(t−1)(t−2)` over tie blocks.
/// The alternative concerns negative, positive or either association. Only
/// SciPy method="asymptotic" is supported, with no axes or mask/NaN policies.
///
/// ## Domain and special values
///
/// Requires 3..=4096 finite paired observations. A constant variable gives
/// NaN fields. Two observations are rejected because the asymptotic variance
/// has an n−2 singularity; SciPy 1.18.1 raises a division error in that case.
/// Signed zeros form one category.
///
/// ## Returns
///
/// Dimensionless tau in [−1,1] and approximate p-value. No exact-tail dispatch.
///
/// ## Errors
///
/// Invalid counts, unequal lengths or nonfinite inputs return errors.
///
/// ## Numerical behavior
///
/// Native direct pair counting uses O(n²) work and O(n) memory, capped at
/// 8,386,560 pairs. Counts are exact within that limit. Statrs supplies normal
/// tails. This is slower than SciPy's Fenwick-tree implementation; small
/// samples or extensive ties can weaken asymptotic calibration.
///
/// ## Examples
///
/// ```
/// use sci_stats::{association::kendalltau, association::KendallVariant, inference::Alternative};
/// let r = kendalltau(&[1.,2.,3.], &[3.,2.,1.], KendallVariant::B, Alternative::TwoSided)?;
/// assert_eq!(r.statistic, -1.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy kendalltau and Kendall (1938, 1945)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html).
pub fn kendalltau(
    x: &[f64],
    y: &[f64],
    variant: KendallVariant,
    alternative: Alternative,
) -> Result<TestResult> {
    if x.len() != y.len() {
        return Err(Error::DimensionMismatch);
    }
    if x.len() > 4096 {
        return Err(Error::InvalidInput(
            "Kendall pair-count sample limit is 4096",
        ));
    }
    sample(x, 3)?;
    sample(y, 3)?;
    let tx = ranking::tie_counts(x);
    let ty = ranking::tie_counts(y);
    let n = x.len() as f64;
    let total = n * (n - 1.) / 2.;
    if tx.pairs == total || ty.pairs == total {
        return Ok(TestResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        });
    }
    let mut difference = 0_i64;
    for i in 0..x.len() {
        for j in i + 1..x.len() {
            if x[i] == x[j] || y[i] == y[j] {
                continue;
            }
            difference += if (x[i] < x[j]) == (y[i] < y[j]) {
                1
            } else {
                -1
            };
        }
    }
    let difference = difference as f64;
    let statistic = match variant {
        KendallVariant::B => difference / (total - tx.pairs).sqrt() / (total - ty.pairs).sqrt(),
        KendallVariant::C => {
            let c = tx.unique.min(ty.unique) as f64;
            2. * difference / (n * n * (c - 1.) / c)
        }
    }
    .clamp(-1., 1.);
    let m = n * (n - 1.);
    let variance = (m * (2. * n + 5.) - tx.variance_term - ty.variance_term) / 18.
        + 2. * tx.pairs * ty.pairs / m
        + tx.triples * ty.triples / (9. * m * (n - 2.));
    Ok(TestResult {
        statistic,
        pvalue: normality::normal_result(difference / variance.sqrt(), alternative)?.pvalue,
    })
}

/// Zero-difference convention for the asymptotic Wilcoxon signed-rank test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WilcoxonZero {
    /// Remove zero differences before ranking, SciPy's default.
    Wilcox,
    /// Rank zeros, then exclude their rank contributions and variance.
    Pratt,
    /// Rank zeros and split their rank contribution equally between signs.
    Split,
}

/// Wilcoxon signed-rank test on supplied differences, `W+=Σ_{d>0}rank(|d|)`.
///
/// ## Parameters and conventions
///
/// `differences` is an explicit one-dimensional sample of paired differences.
/// `zero` selects zero handling; ties in |d| use average ranks. Let R be the
/// included ranks: the null mean of W+ is `ΣR/2`, and variance `ΣR²/4`.
/// Pratt excludes zero ranks from both; Split includes them and adds half
/// their sum to W+ and W−. `correction` shifts W+−mean by 0.5 toward zero
/// for two-sided tests, +0.5 for Less, and −0.5 for Greater. Alternatives
/// concern the location of the differences relative to zero.
/// Only SciPy's asymptotic method is provided; no paired subtraction, axes,
/// missing-data policies, exact tails, or permutation dispatch.
///
/// ## Domain and special values
///
/// Requires 1..=1,048,576 finite differences. All-zero data returns W=0 and
/// NaN p for Wilcox/Pratt. Split returns half the total rank sum, with p=1
/// for a two-sided test or p=0.5 for an uncorrected one-sided test.
/// Positive and negative zero are treated identically.
///
/// ## Returns
///
/// Two-sided tests return min(W+,W−); one-sided tests return W+. The p-value
/// is approximate and dimensionless. Statistic units are rank sums.
///
/// ## Errors
///
/// Invalid observation counts or nonfinite differences return errors.
///
/// ## Numerical behavior
///
/// Native O(n log n) ranking and O(n) memory; normal tails reuse statrs.
/// Small samples, many ties or zeros can weaken the approximation. Computing
/// paired differences externally lets callers control subtraction rounding.
///
/// ## Examples
///
/// ```
/// use sci_stats::{inference::wilcoxon, inference::WilcoxonZero, inference::Alternative};
/// assert_eq!(wilcoxon(&[1.,2.,3.], WilcoxonZero::Wilcox, Alternative::TwoSided, false)?.statistic, 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy wilcoxon, Wilcoxon (1945), Pratt (1959), and Cureton (1967)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html).
pub fn wilcoxon(
    differences: &[f64],
    zero: WilcoxonZero,
    alternative: Alternative,
    correction: bool,
) -> Result<TestResult> {
    sample(differences, 1)?;
    let d: Vec<_> = differences
        .iter()
        .copied()
        .filter(|v| zero != WilcoxonZero::Wilcox || *v != 0.)
        .collect();
    if d.is_empty() {
        return Ok(TestResult {
            statistic: 0.,
            pvalue: f64::NAN,
        });
    }
    let magnitudes: Vec<_> = d.iter().map(|v| v.abs()).collect();
    let ranks = rankdata(&magnitudes, RankMethod::Average)?;
    let (mut plus, mut minus, mut expected, mut variance) = (0., 0., 0., 0.);
    for (d, r) in d.iter().zip(ranks) {
        if *d > 0. {
            plus += r;
        }
        if *d < 0. {
            minus += r;
        }
        if *d == 0. && zero == WilcoxonZero::Split {
            plus += r / 2.;
            minus += r / 2.;
        }
        if *d != 0. || zero == WilcoxonZero::Split {
            expected += r / 2.;
            variance += r * r / 4.;
        }
    }
    let delta: f64 = plus - expected;
    let sign = match alternative {
        Alternative::Less => -1.,
        Alternative::Greater => 1.,
        Alternative::TwoSided => {
            if delta == 0. {
                0.
            } else {
                delta.signum()
            }
        }
    };
    let z = if variance == 0. {
        f64::NAN
    } else {
        (delta - if correction { 0.5 * sign } else { 0. }) / variance.sqrt()
    };
    Ok(TestResult {
        statistic: if alternative == Alternative::TwoSided {
            plus.min(minus)
        } else {
            plus
        },
        pvalue: normality::normal_result(z, alternative)?.pvalue,
    })
}

/// Ansari–Bradley scale test, `AB=Σ_{x} min(R,N+1−R)`.
///
/// ## Parameters and conventions
///
/// Independent samples x and y with sizes n and m, N=n+m. Pooled ranks R
/// average ties. `alternative` concerns x's scale relative to y's scale;
/// smaller AB indicates larger x dispersion. For odd N, null mean is
/// `n(N+1)²/(4N)` and untied variance `nm(N+1)(N²+3)/(48N²)`; for even N,
/// these are `n(N+2)/4` and `nm(N+2)(N−2)/(48(N−1))`.
/// With ties, let F=Σmin(R,N+1−R)²: variances become
/// `nm(16NF−(N+1)⁴)/(16N²(N−1))` (odd) and
/// `nm(16F−N(N+2)²)/(16N(N−1))` (even).
/// Only SciPy's asymptotic method is supported; exact and permutation tails,
/// axes, and missing-data policies are absent.
///
/// ## Domain and special values
///
/// Nonempty finite samples with total count≤1,048,576. The tie-corrected
/// variance can be zero or negative; the division/square-root limit produces
/// NaN or signed-infinite standardized statistics, matching SciPy's formulas.
///
/// ## Returns
///
/// Symmetric rank sum AB and approximate normal p-value. AB is dimensionless.
///
/// ## Errors
///
/// Invalid counts or nonfinite observations return errors.
///
/// ## Numerical behavior
///
/// Native O(N log N) ranks and O(N) memory; normal tails use existing statrs.
/// Small samples or heavy ties can make the asymptotic approximation unreliable.
/// This scale test assumes comparable population locations.
///
/// ## Examples
///
/// ```
/// use sci_stats::{inference::ansari, inference::Alternative};
/// assert!(ansari(&[-1.,0.,1.], &[-8.,0.,8.], Alternative::TwoSided)?.statistic > 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy ansari and Ansari–Bradley (1960)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ansari.html).
pub fn ansari(x: &[f64], y: &[f64], alternative: Alternative) -> Result<TestResult> {
    let count = inference::groups_check(&[x, y], 2)?;
    let all: Vec<_> = x.iter().chain(y).copied().collect();
    let ranks = rankdata(&all, RankMethod::Average)?;
    let n = x.len() as f64;
    let m = y.len() as f64;
    let total = count as f64;
    let scores: Vec<_> = ranks.iter().map(|r| r.min(total + 1. - r)).collect();
    let statistic = scores[..x.len()].iter().sum::<f64>();
    let expected = if count % 2 == 0 {
        n * (total + 2.) / 4.
    } else {
        n * (total + 1.).powi(2) / (4. * total)
    };
    let ties = tiecorrect(&ranks)? < 1.;
    let variance = if ties {
        let f = scores.iter().map(|s| s * s).sum::<f64>();
        if count % 2 == 0 {
            m * n * (16. * f - total * (total + 2.).powi(2)) / (16. * total * (total - 1.))
        } else {
            m * n * (16. * total * f - (total + 1.).powi(4)) / (16. * total * total * (total - 1.))
        }
    } else if count % 2 == 0 {
        m * n * (total + 2.) * (total - 2.) / (48. * (total - 1.))
    } else {
        n * m * (total + 1.) * (total * total + 3.) / (48. * total * total)
    };
    Ok(TestResult {
        statistic,
        pvalue: normality::normal_result((expected - statistic) / variance.sqrt(), alternative)?
            .pvalue,
    })
}

/// Selected count for an exact quantile test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantileCount {
    /// T₁ counts observations ≤ the hypothesized quantile (SciPy statistic_type=1).
    AtOrBelow,
    /// T₂ counts observations < the hypothesized quantile (SciPy statistic_type=2).
    Below,
}

/// Exact one-sample quantile test result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantileTestResult {
    /// Selected observation count T₁ or T₂.
    pub statistic: usize,
    /// Which count was selected, including two-sided tie breaking.
    pub count: QuantileCount,
    /// Exact binomial-null tail probability.
    pub pvalue: f64,
}

/// Exact quantile test of `H₀: Q(p)=q` using `Y~Binomial(n,p)`.
///
/// ## Parameters and conventions
///
/// `values` is an independent identically distributed sample; `q` is the
/// finite hypothesized quantile and `probability` its order p in (0,1).
/// With `T₁=#{x≤q}` and `T₂=#{x<q}`, Greater uses P(Y≤T₁), Less uses
/// P(Y≥T₂), and TwoSided uses min(1,2min(P(Y≤T₁),P(Y≥T₂))). The latter
/// returns the count for the smaller tail, choosing T₁ on a tie. No confidence
/// interval object, axes, masks, or missing-data policies are provided.
///
/// ## Domain and special values
///
/// Requires 1..=1,048,576 finite observations, finite q and 0<p<1.
/// Observations equal to q are retained in T₁ and excluded from T₂.
/// q may lie outside the observed range.
///
/// ## Returns
///
/// Count, its explicit convention, and an exact binomial p-value; no units.
///
/// ## Errors
///
/// Invalid domains/counts or nonfinite observations return errors.
///
/// ## Numerical behavior
///
/// Native O(n) counting and O(1) extra memory. Existing statrs supplies
/// binomial tails through Sci's distribution API. Extreme tails can underflow.
/// "Exact" describes the null distribution, subject to binary64 CDF accuracy.
///
/// ## Examples
///
/// ```
/// use sci_stats::{inference::quantile_test, inference::Alternative, inference::QuantileCount};
/// let r = quantile_test(&[1.,2.,3.], 2., 0.5, Alternative::Greater)?;
/// assert_eq!((r.statistic,r.count),(2,QuantileCount::AtOrBelow));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy quantile_test and Conover (1999), section 3.2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.quantile_test.html).
pub fn quantile_test(
    values: &[f64],
    q: f64,
    probability: f64,
    alternative: Alternative,
) -> Result<QuantileTestResult> {
    sample(values, 1)?;
    if !q.is_finite() || !probability.is_finite() || probability <= 0. || probability >= 1. {
        return Err(Error::InvalidInput(
            "quantile must be finite and probability in (0,1)",
        ));
    }
    let t1 = values.iter().filter(|v| **v <= q).count();
    let t2 = values.iter().filter(|v| **v < q).count();
    let distribution = Distribution::Binomial {
        trials: values.len() as f64,
        probability,
    };
    let upper = cdf(distribution, t1 as f64)?;
    let lower = sf(distribution, t2 as f64 - 1.)?;
    let (statistic, count, pvalue) = match alternative {
        Alternative::Greater => (t1, QuantileCount::AtOrBelow, upper),
        Alternative::Less => (t2, QuantileCount::Below, lower),
        Alternative::TwoSided => {
            if upper <= lower {
                (t1, QuantileCount::AtOrBelow, (2. * upper).min(1.))
            } else {
                (t2, QuantileCount::Below, (2. * lower).min(1.))
            }
        }
    };
    Ok(QuantileTestResult {
        statistic,
        count,
        pvalue,
    })
}

/// Page ordered-alternative test, `L=Σ_j p_j Σ_i R_ij`.
///
/// ## Parameters and conventions
///
/// Rows are subjects/blocks, columns are treatments. Raw observations are
/// ranked within each row using average ties. `predicted_ranks` is an explicit
/// permutation of 1..=k, expressing increasing predicted treatment means.
/// Null mean `E=mk(k+1)²/4` and variance `V=mk²(k+1)(k²−1)/144` give
/// upper-tail normal p at `(L−E)/sqrt(V)`. Only SciPy method="asymptotic"
/// with ranked=false is supported; exact tails, automatic selection, and
/// pre-ranked input are absent. Rearrange columns or reverse predicted ranks
/// to express other monotone alternatives.
///
/// ## Domain and special values
///
/// At least two rectangular rows and three columns, finite observations,
/// total≤1,048,576. Ties are accepted without adjusting null variance,
/// matching SciPy's asymptotic formula. All-tied rows contribute their mean rank.
///
/// ## Returns
///
/// Rank-product statistic L and approximate upper-tail p-value; no units.
///
/// ## Errors
///
/// Invalid shapes, nonfinite observations, invalid permutations or exceeded
/// work limits return errors.
///
/// ## Numerical behavior
///
/// Native O(mk log k) ranking, O(k) extra memory. Normal tails use statrs.
/// Small block/treatment counts and extensive ties weaken the approximation.
///
/// ## Examples
///
/// ```
/// let r = sci_stats::inference::page_trend_test(&[&[1.,2.,3.], &[2.,3.,4.]], &[1,2,3])?;
/// assert!(r.pvalue < 0.1);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy page_trend_test and Page (1963)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.page_trend_test.html).
pub fn page_trend_test(rows: &[&[f64]], predicted_ranks: &[usize]) -> Result<TestResult> {
    inference::groups_check(rows, 2)?;
    let columns = rows[0].len();
    if columns < 3 || rows.iter().any(|r| r.len() != columns) || predicted_ranks.len() != columns {
        return Err(Error::InvalidInput(
            "Page's test requires rectangular rows with at least three columns",
        ));
    }
    let mut seen = vec![false; columns];
    for rank in predicted_ranks {
        if *rank == 0 || *rank > columns || seen[rank - 1] {
            return Err(Error::InvalidInput(
                "predicted ranks must permute 1..=columns",
            ));
        }
        seen[rank - 1] = true;
    }
    let mut statistic = 0.;
    for row in rows {
        let ranks = rankdata(row, RankMethod::Average)?;
        statistic += ranks
            .iter()
            .zip(predicted_ranks)
            .map(|(r, p)| r * *p as f64)
            .sum::<f64>();
    }
    let m = rows.len() as f64;
    let k = columns as f64;
    let expected = m * k * (k + 1.).powi(2) / 4.;
    let variance = m * k * k * (k + 1.) * (k * k - 1.) / 144.;
    Ok(TestResult {
        statistic,
        pvalue: normality::normal_result(
            (statistic - expected) / variance.sqrt(),
            Alternative::Greater,
        )?
        .pvalue,
    })
}

/// Mood squared-rank scale test, `Z=(T−n(N²−1)/12)/sqrt(V)`.
///
/// ## Parameters and conventions
///
/// `x` and `y` are independent samples with n and m observations, N=n+m.
/// For each tied block occupying ranks a..b, assign every observation
/// `φ=mean_{r=a..b}(r−(N+1)/2)²`. T sums these scores for x. The exact
/// random-label variance is `V=nm/(N(N−1)) Σ(φ−(N²−1)/12)²`.
/// The alternative selects lower, higher, or either scale for x relative to y.
///
/// ## Domain and special values
///
/// Nonempty finite samples; 3..=1,048,576 combined observations. Ties use
/// Mielke's averaged squared-rank scores. Zero score variance returns NaN
/// fields. No missing-data policies, masks, axes, or exact/permutation tails.
///
/// ## Returns
///
/// Dimensionless Z and an approximate standard-normal p-value.
///
/// ## Errors
///
/// Invalid sample counts or nonfinite observations return an error.
///
/// ## Numerical behavior
///
/// Native O(N log N) sorting and O(N) memory. Block moments avoid explicitly
/// enumerating tied ranks. Normal tails use the existing private statrs backend.
/// Asymptotic calibration is unreliable for very small samples.
///
/// ## Examples
///
/// ```
/// use sci_stats::{inference::mood, inference::Alternative};
/// assert!(mood(&[-1.,0.,1.], &[-8.,0.,8.], Alternative::Less)?.statistic < 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy mood and Mielke (1967)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mood.html).
pub fn mood(x: &[f64], y: &[f64], alternative: Alternative) -> Result<TestResult> {
    let count = inference::groups_check(&[x, y], 2)?;
    if count < 3 {
        return Err(Error::InvalidInput(
            "Mood's test requires at least three observations",
        ));
    }
    let mut pooled: Vec<_> = x
        .iter()
        .map(|v| (*v, true))
        .chain(y.iter().map(|v| (*v, false)))
        .collect();
    pooled.sort_by(|a, b| a.0.total_cmp(&b.0));
    let n = x.len() as f64;
    let m = y.len() as f64;
    let total = count as f64;
    let expected = (total * total - 1.) / 12.;
    let mut sum_x = 0.;
    let mut sum_squares = 0.;
    let mut start = 0;
    while start < count {
        let mut end = start + 1;
        while end < count && pooled[end].0 == pooled[start].0 {
            end += 1;
        }
        let t = (end - start) as f64;
        let centered_rank = (start + end + 1) as f64 / 2. - (total + 1.) / 2.;
        let score = centered_rank.powi(2) + (t * t - 1.) / 12.;
        sum_x += pooled[start..end].iter().filter(|v| v.1).count() as f64 * score;
        sum_squares += t * (score - expected).powi(2);
        start = end;
    }
    let variance = n * m / (total * (total - 1.)) * sum_squares;
    normality::normal_result(
        if variance == 0. {
            f64::NAN
        } else {
            (sum_x - n * expected) / variance.sqrt()
        },
        alternative,
    )
}

/// Mann–Whitney test, `U_x=ΣR_x−n(n+1)/2`, using pooled average ranks.
///
/// ## Parameters and conventions
///
/// `x` and `y` are independent samples of sizes n and m. `alternative`
/// concerns x being stochastically less than or greater than y. The returned
/// statistic always belongs to x, including for two-sided tests. Null mean
/// is nm/2 and variance is `nm/12 (N+1−Σ(t³−t)/(N(N−1)))` for tie sizes t.
/// `continuity=true` shifts U−nm/2 by 0.5 toward zero for two-sided tests,
/// by +0.5 for a lower tail, and by −0.5 for an upper tail.
/// Only SciPy method="asymptotic" is supported; exact, auto, and permutation
/// methods are not provided.
///
/// ## Domain and special values
///
/// Nonempty finite samples, combined size≤1,048,576. Average ranks support
/// ties and signed-zero equality. An all-equal pool returns NaN for two-sided
/// tests; one-sided tests return p=1 with continuity correction and NaN without.
/// No axes, masks, or NaN omission.
///
/// ## Returns
///
/// U for x and its approximate p-value; U counts pairwise comparisons with
/// half weight for ties and lies in `[0,nm]`.
///
/// ## Errors
///
/// Invalid counts or nonfinite observations return errors.
///
/// ## Numerical behavior
///
/// Native O(N log N) ranks and O(N) memory. Normal tails use statrs privately.
/// Small samples and extensive ties can weaken asymptotic calibration.
///
/// ## Examples
///
/// ```
/// use sci_stats::{inference::mannwhitneyu, inference::Alternative};
/// assert_eq!(mannwhitneyu(&[1.,2.,3.], &[4.,5.,6.], Alternative::Less, true)?.statistic, 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy mannwhitneyu and Mann–Whitney (1947)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html).
pub fn mannwhitneyu(
    x: &[f64],
    y: &[f64],
    alternative: Alternative,
    continuity: bool,
) -> Result<TestResult> {
    let count = inference::groups_check(&[x, y], 2)?;
    let all: Vec<_> = x.iter().chain(y).copied().collect();
    let ranks = rankdata(&all, RankMethod::Average)?;
    let n = x.len() as f64;
    let m = y.len() as f64;
    let statistic = ranks[..x.len()].iter().sum::<f64>() - n * (n + 1.) / 2.;
    let variance = n * m * (count + 1) as f64 * tiecorrect(&ranks)? / 12.;
    let delta = statistic - n * m / 2.;
    let sign = match alternative {
        Alternative::TwoSided => {
            if delta == 0. {
                0.
            } else {
                delta.signum()
            }
        }
        Alternative::Less => -1.,
        Alternative::Greater => 1.,
    };
    let z = (delta - if continuity { 0.5 * sign } else { 0. }) / variance.sqrt();
    let pvalue = normality::normal_result(z, alternative)?.pvalue;
    Ok(TestResult { statistic, pvalue })
}

/// Null-tail approximation for the Brunner–Munzel test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrunnerDistribution {
    /// Student t with estimated degrees of freedom, SciPy's default.
    StudentT,
    /// Standard normal limiting distribution.
    Normal,
}

/// Handling of observations equal to the pooled median.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MedianTies {
    /// Include equal observations in the below-median row, SciPy's default.
    Below,
    /// Include equal observations in the above-median row.
    Above,
    /// Exclude equal observations from the contingency counts.
    Ignore,
}

/// Mood median test result and its 2-by-k contingency table.
#[derive(Clone, Debug, PartialEq)]
pub struct MedianTestResult {
    /// Pearson chi-square statistic, optionally Yates corrected.
    pub statistic: f64,
    /// Chi-square upper-tail probability.
    pub pvalue: f64,
    /// Pooled sample median, in observation units.
    pub median: f64,
    /// Rows are above and below the median; columns retain input group order.
    pub table: Vec<Vec<f64>>,
}

/// Brunner–Munzel rank test:
/// `W=nm(mean(R_y)−mean(R_x))/((n+m)sqrt(nS_x+mS_y))`.
///
/// ## Parameters and conventions
///
/// `x` and `y` are independent one-dimensional samples. R denotes pooled
/// average ranks. For within-sample ranks r, `S_x=var(R_x−r_x,ddof=1)`
/// and similarly for y. `distribution` selects Student-t or normal tails.
/// Student-t degrees of freedom are `(nS_x+mS_y)²/((nS_x)²/(n−1)+(mS_y)²/(m−1))`.
/// SciPy's sign convention is preserved: W is positive when y tends to be larger;
/// the alternative's probability is evaluated at −W.
///
/// ## Domain and special values
///
/// Each sample has at least two finite observations; combined size≤1,048,576.
/// Ties use average ranks. Zero rank variance can yield signed infinity or
/// NaN W; Student-t then returns NaN p, while normal tails retain their limit.
/// No masks, axes, NaN omission, or permutation overloads.
///
/// ## Returns
///
/// Dimensionless W and the selected approximate p-value.
///
/// ## Errors
///
/// Invalid sample sizes and nonfinite observations return errors.
///
/// ## Numerical behavior
///
/// Native O(N log N) ranking and O(N) memory. Existing statrs supplies only
/// normal or t tails. Calibration can be inaccurate for small samples;
/// permutation calibration, recommended below about ten observations, is absent.
///
/// ## Examples
///
/// ```
/// use sci_stats::{inference::brunnermunzel, inference::Alternative, inference::BrunnerDistribution};
/// let r = brunnermunzel(&[1.,2.,3.,4.], &[2.,3.,4.,5.], Alternative::TwoSided, BrunnerDistribution::StudentT)?;
/// assert!(r.statistic > 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy brunnermunzel and Brunner–Munzel (2000)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.brunnermunzel.html).
pub fn brunnermunzel(
    x: &[f64],
    y: &[f64],
    alternative: Alternative,
    distribution: BrunnerDistribution,
) -> Result<TestResult> {
    inference::groups_check(&[x, y], 2)?;
    sample(x, 2)?;
    sample(y, 2)?;
    let pooled: Vec<_> = x.iter().chain(y).copied().collect();
    let ranks = rankdata(&pooled, RankMethod::Average)?;
    let (rx, ry) = ranks.split_at(x.len());
    let within_x = rankdata(x, RankMethod::Average)?;
    let within_y = rankdata(y, RankMethod::Average)?;
    let dx: Vec<_> = rx.iter().zip(within_x).map(|(a, b)| a - b).collect();
    let dy: Vec<_> = ry.iter().zip(within_y).map(|(a, b)| a - b).collect();
    let n = x.len() as f64;
    let m = y.len() as f64;
    let sx = n * variance(&dx, 1)?;
    let sy = m * variance(&dy, 1)?;
    let statistic = n * m * (mean(ry)? - mean(rx)?) / ((n + m) * (sx + sy).sqrt());
    let pvalue = match distribution {
        BrunnerDistribution::Normal => normality::normal_result(-statistic, alternative)?.pvalue,
        BrunnerDistribution::StudentT => {
            if sx + sy == 0. {
                f64::NAN
            } else {
                let df = (sx + sy).powi(2) / (sx.powi(2) / (n - 1.) + sy.powi(2) / (m - 1.));
                inference::t_probability(-statistic, df, alternative)?
            }
        }
    };
    Ok(TestResult { statistic, pvalue })
}

/// Mood median test: Pearson chi-square on counts above/below the pooled median.
///
/// ## Parameters and conventions
///
/// Each slice is an independent group. `ties` determines the placement of
/// observations equal to the pooled median. `correction` applies Yates's
/// correction only for two groups. Statistic `χ²=Σ(O−E)²/E`; expectations
/// come from row and column marginal products divided by the retained total.
/// Unlike SciPy, only Pearson's lambda=1 divergence is supported.
///
/// ## Domain and special values
///
/// At least two nonempty finite groups, total count≤1,048,576 and
/// group count≤524,288. Each retained column and both rows must be nonzero.
/// Groups may differ in length. No masks or NaN omission.
///
/// ## Returns
///
/// Statistic, approximate p-value, pooled median, and a 2-by-k table whose
/// first row is above-median counts and second is below-median counts.
///
/// ## Errors
///
/// Invalid counts, nonfinite data, or a zero expected contingency cell is an error.
///
/// ## Numerical behavior
///
/// Native O(N log N) median and counting, O(N+k) memory. Counts are exact
/// binary64 integers within the work limit. Chi-square tails reuse statrs
/// through the existing contingency API; small expected counts weaken calibration.
///
/// ## Examples
///
/// ```
/// use sci_stats::{inference::median_test, inference::MedianTies};
/// let r = median_test(&[&[1.,2.,3.], &[4.,5.,6.]], MedianTies::Below, false)?;
/// assert_eq!(r.table, vec![vec![0.,3.], vec![3.,0.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy median_test and Mood (1950)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_test.html).
pub fn median_test(
    groups: &[&[f64]],
    ties: MedianTies,
    correction: bool,
) -> Result<MedianTestResult> {
    if groups.len() > MAX_SAMPLES / 2 {
        return Err(Error::InvalidInput("median-test group limit is 524288"));
    }
    inference::groups_check(groups, 2)?;
    let all: Vec<_> = groups.iter().flat_map(|g| g.iter().copied()).collect();
    let center = median(&all)?;
    let mut table = vec![vec![0.; groups.len()]; 2];
    for (i, group) in groups.iter().enumerate() {
        for x in *group {
            if *x > center || (*x == center && ties == MedianTies::Above) {
                table[0][i] += 1.;
            }
            if *x < center || (*x == center && ties == MedianTies::Below) {
                table[1][i] += 1.;
            }
        }
    }
    let flat: Vec<_> = table.iter().flatten().copied().collect();
    let result = chi2_contingency(&flat, 2, groups.len(), correction)?;
    Ok(MedianTestResult {
        statistic: result.statistic,
        pvalue: result.pvalue,
        median: center,
        table,
    })
}

/// Fligner–Killeen scale test, `X²=Σn_i(ā_i−ā)²/var(a,ddof=1)`.
///
/// ## Parameters and conventions
///
/// Each slice is a group. `center` is mean or median, matching [`levene`].
/// For pooled average ranks R of within-group absolute deviations,
/// `a=Φ⁻¹(0.5+R/(2(N+1)))`. The asymptotic null is chi-square with k−1
/// degrees of freedom. SciPy's trimmed-center, axes and mask options are absent.
///
/// ## Domain and special values
///
/// At least two nonempty finite groups, combined size≤1,048,576. Ties use
/// average ranks. Identical ranked deviations yield NaN fields.
///
/// ## Returns
///
/// Dimensionless statistic and approximate upper-tail p-value.
///
/// ## Errors
///
/// Invalid counts, nonfinite inputs or overflowing absolute deviations are errors.
///
/// ## Numerical behavior
///
/// O(N log N) native ranks and O(N) memory. Existing statrs provides normal
/// quantiles and chi-square tails. Small samples weaken asymptotic calibration;
/// rounding of the centers can affect nearly equal observations.
///
/// ## Examples
///
/// ```
/// use sci_stats::{inference::fligner, inference::Center};
/// assert!(fligner(&[&[1.,2.,3.], &[1.,3.,8.]], Center::Median)?.statistic >= 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy fligner and Fligner–Killeen (1976)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fligner.html).
pub fn fligner(groups: &[&[f64]], center: Center) -> Result<TestResult> {
    let n = inference::groups_check(groups, 2)?;
    let mut deviations = Vec::with_capacity(n);
    for g in groups {
        let c = match center {
            Center::Mean => mean(g)?,
            Center::Median => median(g)?,
        };
        for x in *g {
            deviations.push(match (x - c).abs() {
                value if value.is_finite() => Ok(value),
                _ => Err(crate::Error::NonFiniteEvaluation),
            }?);
        }
    }
    let ranks = rankdata(&deviations, RankMethod::Average)?;
    let scores: Vec<_> = ranks
        .iter()
        .map(|r| normal_ppf(0.5 + r / (2. * (n as f64 + 1.)), 0., 1.))
        .collect::<Result<_>>()?;
    let average = mean(&scores)?;
    let spread = variance(&scores, 1)?;
    if spread == 0. {
        return Ok(TestResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        });
    }
    let mut numerator = 0.;
    let mut offset = 0;
    for g in groups {
        let a = mean(&scores[offset..offset + g.len()])?;
        numerator += g.len() as f64 * (a - average).powi(2);
        offset += g.len();
    }
    inference::chi_result(numerator / spread, (groups.len() - 1) as f64)
}
