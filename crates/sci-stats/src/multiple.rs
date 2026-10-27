use super::*;
use sci_core::Matrix;

/// Pairwise test results, one row and column per input group.
#[derive(Clone, Debug, PartialEq)]
pub struct PairwiseTest {
    /// Mean differences: entry (i,j) is mean(group_i)−mean(group_j), in input units.
    pub statistic: Matrix,
    /// Familywise-adjusted two-sided p-values, with the same shape and ordering.
    pub pvalue: Matrix,
}

/// Tukey HSD/Tukey–Kramer pairwise comparisons `q_ij=|mean_i−mean_j|/sqrt(MSE*(1/n_i+1/n_j)/2)`.
///
/// ## Parameters and conventions
///
/// `groups` contains independent samples. `MSE=Σ_i Σ_j (x_ij−mean_i)²/(N−k)`;
/// p-values use the studentized-range survival function at q_ij, with k groups
/// and N−k degrees of freedom. Unequal sample sizes use Tukey–Kramer adjustment.
/// Equal variances and independent normally distributed errors are assumed.
/// SciPy equal_var=false (Games–Howell) and confidence-interval objects are unsupported.
///
/// ## Domain and special values
///
/// 2..=16 groups, each with at least two finite observations, combined count≤65536.
/// NaN/infinity data are errors. Zero within-group variance gives p=0 for
/// unequal group means and NaN for equal means, matching SciPy. Ordinarily
/// diagonal statistics are zero and diagonal p-values are one.
///
/// ## Returns
///
/// Two k×k row-major Sci matrices. Statistics are signed mean differences,
/// rather than studentized q, matching SciPy's result ordering and convention.
/// P-values are symmetric simultaneous two-sided probabilities.
///
/// ## Errors
///
/// Invalid group counts/domains, exceeded work limits, unrepresentable differences
/// or failed studentized-range evaluation return errors.
///
/// ## Numerical behavior
///
/// O(N+k²) native scaled summaries and at most k(k−1)/2 private SciPy-based
/// studentized-range mixture integrals. Composite Gaussian quadrature uses
/// orders 32 and 64, with at most 358400 integrand evaluations per pair.
/// Their absolute difference must be at most 2e-10; this is an error estimate,
/// not a rigorous bound. Normal integration is truncated to [-10,10] and the
/// standardized log-chi-square variable to [-32,12]; omitted tails are below
/// 1e-16 over the supported domain. Direct survival integration avoids CDF
/// subtraction, but extremely small p-values still have limited relative accuracy.
/// Two groups use the exact Student-t identity through the shared statrs backend.
/// Backend objects and result types are kept private. Heteroscedasticity or
/// nonnormal small samples can invalidate familywise calibration.
///
/// ## Examples
///
/// ```
/// let r = sci_stats::tukey_hsd(&[&[1.,2.,3.], &[4.,5.,6.]])?;
/// assert_eq!(r.statistic[0][1], -3.);
/// assert_eq!(r.pvalue[0][1], r.pvalue[1][0]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy tukey_hsd](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tukey_hsd.html),
/// [NIST Tukey's method](https://www.itl.nist.gov/div898/handbook/prc/section4/prc471.htm).
pub fn tukey_hsd(groups: &[&[f64]]) -> Result<PairwiseTest> {
    if !(2..=16).contains(&groups.len()) {
        return Err(Error::InvalidInput("Tukey HSD needs 2..=16 groups"));
    }
    let count = inference::groups_check(groups, 2)?;
    if count > 65536 {
        return Err(Error::InvalidInput("Tukey combined sample limit is 65536"));
    }
    for group in groups {
        sample(group, 2)?;
    }
    let pooled = groups
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    let centered = centered_sample(&pooled)?;
    let mut offset = 0;
    let mut means = Vec::with_capacity(groups.len());
    let mut squares = 0.;
    for group in groups {
        let deviations = &centered.deviations[offset..offset + group.len()];
        let center = mean(deviations)?;
        squares += deviations.iter().map(|v| (v - center).powi(2)).sum::<f64>();
        means.push(center);
        offset += group.len();
    }
    let k = groups.len();
    let df = (count - k) as f64;
    let mse = squares / df;
    let range = studentized_range::StudentizedRange::new(k, df);
    let mut statistic = vec![vec![0.; k]; k];
    let mut pvalue = vec![vec![if mse == 0. { f64::NAN } else { 1. }; k]; k];
    for i in 0..k {
        for j in i + 1..k {
            let delta = means[i] - means[j];
            let difference = match delta * centered.scale {
                value if value.is_finite() => Ok(value),
                _ => Err(crate::Error::NonFiniteEvaluation),
            }?;
            let se =
                (mse / 2. * (1. / groups[i].len() as f64 + 1. / groups[j].len() as f64)).sqrt();
            let q = delta.abs() / se;
            let p = if q.is_nan() {
                f64::NAN
            } else if k == 2 {
                inference::t_probability(q / std::f64::consts::SQRT_2, df, Alternative::TwoSided)?
            } else {
                range.sf(q)?
            };
            statistic[i][j] = difference;
            statistic[j][i] = -difference;
            pvalue[i][j] = p;
            pvalue[j][i] = p;
        }
    }
    Ok(PairwiseTest { statistic, pvalue })
}
