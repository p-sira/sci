use super::*;

/// Spearman correlation `ρ=corr(rank_average(x),rank_average(y))`.
///
/// ## Parameters and conventions
///
/// `x` and `y` are paired ordinal observations; `alternative` selects the
/// direction of association. Reuses average ranks and [`pearsonr`], matching
/// SciPy `spearmanrho(method=None)`. This differs from legacy [`spearmanr`]
/// in the two-observation case. Axes, broadcasting and resampling are unsupported.
///
/// ## Domain and special values
///
/// 2..=1048576 finite paired observations. Constants return NaN fields;
/// two nonconstant observations return ±1 with two-sided p=1.
///
/// ## Returns
///
/// Dimensionless correlation and approximate p-value.
///
/// ## Errors
///
/// Invalid counts, nonfinite values and unequal lengths return errors.
///
/// ## Numerical behavior
///
/// O(n log n) ranking followed by scaled correlation and private statrs tails.
/// Small-sample asymptotic p-values can be poorly calibrated.
///
/// ## Examples
///
/// ```
/// let r = sci_stats::association::spearmanrho(&[1.,2.,3.], &[3.,2.,1.], sci_stats::inference::Alternative::TwoSided)?;
/// assert!((r.statistic+1.).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy spearmanrho](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanrho.html).
pub fn spearmanrho(x: &[f64], y: &[f64], alternative: Alternative) -> Result<TestResult> {
    if x.len() != y.len() {
        return Err(Error::DimensionMismatch);
    }
    sample(x, 2)?;
    sample(y, 2)?;
    pearsonr(
        &rankdata(x, RankMethod::Average)?,
        &rankdata(y, RankMethod::Average)?,
        alternative,
    )
}

/// Importance ranking for weighted Kendall correlation; rank zero has greatest weight.
#[derive(Clone, Copy, Debug)]
pub enum WeightedRank<'a> {
    /// Average descending lexicographic rankings by (x,y) and (y,x), SciPy rank=True.
    Symmetric,
    /// Descending lexicographic ranking by (x,y), SciPy rank=None.
    Lexicographic,
    /// Original observation indices, SciPy rank=False.
    Observation,
    /// Explicit zero-based nonnegative importance ranks; ties are allowed.
    Explicit(&'a [usize]),
}
fn lexicographic_ranks(x: &[f64], y: &[f64]) -> Vec<usize> {
    let mut indices = (0..x.len()).collect::<Vec<_>>();
    indices.sort_by(|&i, &j| {
        // Numeric comparisons make signed zeros a single category.
        x[j].partial_cmp(&x[i])
            .unwrap()
            .then(y[j].partial_cmp(&y[i]).unwrap())
    });
    let mut rank = vec![0; x.len()];
    for (r, i) in indices.into_iter().enumerate() {
        rank[i] = r;
    }
    rank
}
fn weighted_correlation(x: &[f64], y: &[f64], ranks: &[usize], additive: bool) -> f64 {
    let weights = ranks
        .iter()
        .map(|&r| 1. / (r as f64 + 1.))
        .collect::<Vec<_>>();
    let (mut score, mut untied_x, mut untied_y) = (0., 0., 0.);
    for i in 0..x.len() {
        for j in i + 1..x.len() {
            let weight = if additive {
                weights[i] + weights[j]
            } else {
                weights[i] * weights[j]
            };
            if x[i] != x[j] {
                untied_x += weight;
            }
            if y[i] != y[j] {
                untied_y += weight;
            }
            if x[i] != x[j] && y[i] != y[j] {
                score += if (x[i] < x[j]) == (y[i] < y[j]) {
                    weight
                } else {
                    -weight
                };
            }
        }
    }
    if untied_x == 0. || untied_y == 0. {
        f64::NAN
    } else {
        (score / untied_x.sqrt() / untied_y.sqrt()).clamp(-1., 1.)
    }
}

/// Hyperbolically weighted Kendall correlation `τ=Σ_(i<j) w_ij sign(Δx*Δy)/sqrt(U_x U_y)`.
///
/// ## Parameters and conventions
///
/// Paired ordinal `x`,`y`; `rank` chooses importance ranks r_i. Individual
/// weights are `1/(r_i+1)`; `additive` chooses their sum, otherwise product.
/// U_x and U_y sum pair weights over pairs untied in x and y, respectively.
/// Symmetric ranking averages the two lexicographic correlations. Custom
/// SciPy weigher callbacks and NaN-as-lowest ranking are unsupported.
///
/// ## Domain and special values
///
/// 2..=4096 finite paired observations, matching explicit ranks bounded by
/// 1048576. Constant variables return NaN statistic. Signed zeros are tied.
///
/// ## Returns
///
/// Dimensionless weighted tau in [−1,1]; p-value is always NaN because its
/// null distribution is not supplied, matching SciPy.
///
/// ## Errors
///
/// Nonfinite inputs, incompatible lengths or exceeded work/rank limits return errors.
///
/// ## Numerical behavior
///
/// Native O(n²) pair counting, at most two passes of 8386560 pairs; O(n) memory.
/// Binary64 weight sums introduce rounding. Lexicographic ties use stable input
/// order; their identical scores make their relative order immaterial.
///
/// ## Examples
///
/// ```
/// use sci_stats::{association::weightedtau, association::WeightedRank};
/// assert_eq!(weightedtau(&[1.,2.,3.], &[3.,2.,1.], WeightedRank::Symmetric, true)?.statistic, -1.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy weightedtau and Vigna (2015)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weightedtau.html).
pub fn weightedtau(
    x: &[f64],
    y: &[f64],
    rank: WeightedRank<'_>,
    additive: bool,
) -> Result<TestResult> {
    if x.len() != y.len() {
        return Err(Error::DimensionMismatch);
    }
    if x.len() > 4096 {
        return Err(Error::InvalidInput("weighted tau sample limit is 4096"));
    }
    sample(x, 2)?;
    sample(y, 2)?;
    let statistic = match rank {
        WeightedRank::Symmetric => {
            (weighted_correlation(x, y, &lexicographic_ranks(x, y), additive)
                + weighted_correlation(y, x, &lexicographic_ranks(y, x), additive))
                / 2.
        }
        WeightedRank::Lexicographic => {
            weighted_correlation(x, y, &lexicographic_ranks(x, y), additive)
        }
        WeightedRank::Observation => {
            weighted_correlation(x, y, &(0..x.len()).collect::<Vec<_>>(), additive)
        }
        WeightedRank::Explicit(ranks) => {
            if ranks.len() != x.len() {
                return Err(Error::DimensionMismatch);
            }
            if ranks.iter().any(|&r| r > MAX_SAMPLES) {
                return Err(Error::InvalidInput("importance rank exceeds 1048576"));
            }
            weighted_correlation(x, y, ranks, additive)
        }
    };
    Ok(TestResult {
        statistic,
        pvalue: f64::NAN,
    })
}

// Cumulative counts of dense y ranks. Updates and prefixes are O(log n).
// The caller inserts all observations in an x-tie block only after querying
// that block, so equal x values never contribute to concordance scores.
fn rank_prefix(tree: &[i64], mut end: usize) -> i64 {
    let mut count = 0;
    while end > 0 {
        count += tree[end];
        end &= end - 1;
    }
    count
}

fn rank_insert(tree: &mut [i64], mut index: usize) {
    while index < tree.len() {
        tree[index] += 1;
        index += index & index.wrapping_neg();
    }
}

/// Exact per-observation concordance scores, shared by Kendall's tau and
/// Somers' D. Both inputs have already passed finite paired-sample validation.
/// Sorting treats signed zeros as ties; all pair counts remain integers.
pub(super) fn concordance_scores(x: &[f64], y: &[f64]) -> (Vec<i64>, usize) {
    let n = x.len();
    let mut by_y = (0..n).collect::<Vec<_>>();
    by_y.sort_unstable_by(|&i, &j| y[i].partial_cmp(&y[j]).unwrap());
    let mut ranks = vec![0_usize; n];
    let mut unique = 0;
    for (position, &index) in by_y.iter().enumerate() {
        if position > 0 && y[index] != y[by_y[position - 1]] {
            unique += 1;
        }
        ranks[index] = unique;
    }

    let mut by_x = (0..n).collect::<Vec<_>>();
    by_x.sort_unstable_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());
    let mut tree = vec![0_i64; unique + 2];
    let mut scores = vec![0_i64; n];
    let mut processed = 0_i64;
    let mut untied_x = 0;
    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n && x[by_x[end]] == x[by_x[start]] {
            end += 1;
        }
        untied_x += processed as usize * (end - start);
        for &index in &by_x[start..end] {
            let less = rank_prefix(&tree, ranks[index]);
            let less_or_equal = rank_prefix(&tree, ranks[index] + 1);
            scores[index] += less - (processed - less_or_equal);
        }
        for &index in &by_x[start..end] {
            rank_insert(&mut tree, ranks[index] + 1);
        }
        processed += (end - start) as i64;
        start = end;
    }

    // For each point, include contributions from greater x values as well.
    tree.fill(0);
    processed = 0;
    let mut end = n;
    while end > 0 {
        let mut start = end - 1;
        while start > 0 && x[by_x[start - 1]] == x[by_x[end - 1]] {
            start -= 1;
        }
        for &index in &by_x[start..end] {
            let less = rank_prefix(&tree, ranks[index]);
            let less_or_equal = rank_prefix(&tree, ranks[index] + 1);
            scores[index] += (processed - less_or_equal) - less;
        }
        for &index in &by_x[start..end] {
            rank_insert(&mut tree, ranks[index] + 1);
        }
        processed += (end - start) as i64;
        end = start;
    }
    (scores, untied_x)
}

/// Somers' directional association `D(Y|X)=(concordant−discordant)/(pairs untied in X)`.
///
/// ## Parameters and conventions
///
/// Paired ordinal `x`,`y`; `alternative` selects the association tail. Define
/// q_i=Σ_j sign((x_i−x_j)(y_i−y_j)), S=Σ_i q_i, and
/// `V=Σ_i q_i²−S²/n`. The Brown–Benedetti normal statistic is `Z=S/(2sqrt(V))`.
/// This supports SciPy's paired-observation overload; table input and the
/// contingency-table result field are unsupported. The first variable is
/// the conditioning variable, so exchanging x and y can change D.
///
/// ## Domain and special values
///
/// 2..=4096 finite paired observations. Constant variables give NaN fields.
/// A nonzero S with zero variance gives an infinite Z and limiting p-value.
///
/// ## Returns
///
/// Dimensionless D in [−1,1] and an asymptotic normal p-value.
///
/// ## Errors
///
/// Nonfinite inputs, unequal lengths or exceeded pair-count limits return errors.
///
/// ## Numerical behavior
///
/// O(n log n) integer rank counting and O(n) memory. Variance is computed as the
/// sum of centered squared q_i, avoiding cancellation; private statrs tails.
/// Asymptotic probabilities can be unreliable for small or sparse samples.
///
/// ## Examples
///
/// ```
/// assert_eq!(sci_stats::association::somersd(&[1.,2.,3.], &[1.,2.,3.], sci_stats::inference::Alternative::TwoSided)?.statistic, 1.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy somersd and Brown–Benedetti (1977)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.somersd.html).
pub fn somersd(x: &[f64], y: &[f64], alternative: Alternative) -> Result<TestResult> {
    if x.len() != y.len() {
        return Err(Error::DimensionMismatch);
    }
    if x.len() > 4096 {
        return Err(Error::InvalidInput(
            "Somers pair-count sample limit is 4096",
        ));
    }
    sample(x, 2)?;
    sample(y, 2)?;
    if x.iter().all(|v| *v == x[0]) || y.iter().all(|v| *v == y[0]) {
        return Ok(TestResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        });
    }
    let (scores, untied) = concordance_scores(x, y);
    let scores = scores.into_iter().map(|v| v as f64).collect::<Vec<_>>();
    let center = mean(&scores)?;
    let sum = scores.iter().sum::<f64>();
    let variance = scores.iter().map(|v| (v - center).powi(2)).sum::<f64>();
    Ok(TestResult {
        statistic: sum / (2. * untied as f64),
        pvalue: normality::normal_result(sum / (2. * variance.sqrt()), alternative)?.pvalue,
    })
}

/// Chatterjee dependence `ξ=1−n*Σ|r_(i+1)−r_i|/(2Σ l_i(n−l_i))`.
///
/// ## Parameters and conventions
///
/// `x`,`y` are paired data sorted by x; r_i counts y_j≤y_i and l_i counts
/// y_j≥y_i. `y_continuous=true` uses `ξ=1−3Σ|Δr|/(n²−1)` and null standard
/// deviation sqrt(2/(5n)). False uses the tie-aware statistic and variance in
/// Chatterjee (2021), Eq. 2.2. P-values concern positive dependence only.
/// Unlike SciPy's arbitrary tie ordering, tied x values are rejected; no
/// random tie-breaking or permutation method is supplied.
///
/// ## Domain and special values
///
/// 2..=1048576 finite paired observations with distinct x values. Constant y
/// yields NaN fields in the tie-aware mode. Continuous mode applies its formula
/// even to ties (including ξ=1 for constant y), as SciPy does; this violates
/// its calibration assumptions. The statistic is asymmetric and can be negative.
///
/// ## Returns
///
/// Dimensionless ξ and greater-tail asymptotic p-value.
///
/// ## Errors
///
/// Invalid counts, unequal lengths, nonfinite values or tied x return errors.
///
/// ## Numerical behavior
///
/// O(n log n) sorting/ranking and O(n) memory. Normalized rank sums avoid
/// large integer powers; private statrs supplies normal tails. Null-variance
/// cancellation and small samples can reduce accuracy and calibration.
///
/// ## Examples
///
/// ```
/// let r = sci_stats::association::chatterjeexi(&[1.,2.,3.,4.], &[1.,2.,3.,4.], true)?;
/// assert!((r.statistic - 0.4).abs() < 1e-15);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy chatterjeexi and Chatterjee (2021)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chatterjeexi.html).
pub fn chatterjeexi(x: &[f64], y: &[f64], y_continuous: bool) -> Result<TestResult> {
    if x.len() != y.len() {
        return Err(Error::DimensionMismatch);
    }
    sample(x, 2)?;
    sample(y, 2)?;
    let mut indices = (0..x.len()).collect::<Vec<_>>();
    indices.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());
    if indices.windows(2).any(|p| x[p[0]] == x[p[1]]) {
        return Err(Error::InvalidInput("Chatterjee xi requires distinct x"));
    }
    if !y_continuous && y.iter().all(|v| *v == y[0]) {
        return Ok(TestResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        });
    }
    let ordered = indices.into_iter().map(|i| y[i]).collect::<Vec<_>>();
    let ranks = rankdata(&ordered, RankMethod::Max)?;
    let n = x.len() as f64;
    let delta = ranks.windows(2).map(|p| (p[1] - p[0]).abs()).sum::<f64>();
    let (statistic, standard_error) = if y_continuous {
        (1. - 3. * delta / (n * n - 1.), (0.4 / n).sqrt())
    } else {
        let lower = rankdata(
            &ordered.iter().map(|v| -v).collect::<Vec<_>>(),
            RankMethod::Max,
        )?;
        let d = lower
            .iter()
            .map(|v| (v / n) * (1. - v / n) / n)
            .sum::<f64>();
        let statistic = 1. - delta / (2. * n * n * d);
        let mut sorted = ranks;
        sorted.sort_by(f64::total_cmp);
        let (mut a, mut b, mut c, mut prefix) = (0., 0., 0., 0.);
        for (i, &rank) in sorted.iter().enumerate() {
            let u = rank / n;
            let position = (i + 1) as f64 / n;
            let weight = 2. - 2. * position + 1. / n;
            prefix += u / n;
            a += weight * u * u / n;
            b += (prefix + (1. - position) * u).powi(2) / n;
            c += weight * u / n;
        }
        let tau2 = (a - 2. * b + c * c).max(0.) / (d * d);
        (statistic, (tau2 / n).sqrt())
    };
    Ok(TestResult {
        statistic,
        pvalue: normality::normal_result(statistic / standard_error, Alternative::Greater)?.pvalue,
    })
}

#[cfg(test)]
mod concordance_tests {
    use super::*;

    fn pairwise_scores(x: &[f64], y: &[f64]) -> (Vec<i64>, usize) {
        let mut scores = vec![0_i64; x.len()];
        let mut untied = 0;
        for i in 0..x.len() {
            for j in i + 1..x.len() {
                if x[i] == x[j] {
                    continue;
                }
                untied += 1;
                if y[i] != y[j] {
                    let sign = if (x[i] < x[j]) == (y[i] < y[j]) {
                        1
                    } else {
                        -1
                    };
                    scores[i] += sign;
                    scores[j] += sign;
                }
            }
        }
        (scores, untied)
    }

    #[test]
    fn concordance_sweep_matches_pairwise_counts_and_somers_pvalues() -> Result<()> {
        let cases: Vec<(Vec<f64>, Vec<f64>)> = vec![
            (vec![1., 1.], vec![2., 2.]),
            (vec![0., -0., 1., 1., 2.], vec![0., 1., -0., 2., 2.]),
            (vec![1., 2., 3., 4., 5.], vec![5., 3., 4., 2., 1.]),
            (vec![1.; 30], (0..30).map(|i| i as f64).collect()),
            (
                (0..70).map(|i| (i % 7) as f64).collect(),
                (0..70).map(|i| (i * 11 % 9) as f64).collect(),
            ),
            (
                (0..127).map(|i| (i * 31 % 127) as f64).collect(),
                (0..127).map(|i| (i * 29 % 127) as f64).collect(),
            ),
            (
                (0..257).map(|i| (i * 37 % 17) as f64).collect(),
                (0..257).map(|i| (i * 13 % 29) as f64).collect(),
            ),
        ];
        for (x, y) in cases {
            let (expected, untied) = pairwise_scores(&x, &y);
            let (scores, got_untied) = concordance_scores(&x, &y);
            assert_eq!(scores, expected);
            assert_eq!(got_untied, untied);
            if x.iter().all(|v| *v == x[0]) || y.iter().all(|v| *v == y[0]) {
                continue;
            }
            let scores = expected.into_iter().map(|s| s as f64).collect::<Vec<_>>();
            let center = mean(&scores)?;
            let sum = scores.iter().sum::<f64>();
            let variance = scores.iter().map(|v| (v - center).powi(2)).sum::<f64>();
            for alternative in [
                Alternative::TwoSided,
                Alternative::Less,
                Alternative::Greater,
            ] {
                let actual = somersd(&x, &y, alternative)?;
                let expected_p =
                    normality::normal_result(sum / (2. * variance.sqrt()), alternative)?.pvalue;
                assert_eq!(
                    actual.statistic.to_bits(),
                    (sum / (2. * untied as f64)).to_bits()
                );
                assert_eq!(actual.pvalue.to_bits(), expected_p.to_bits());
            }
        }
        Ok(())
    }
}
