use super::*;
use std::sync::{Arc, OnceLock};

/// Multiple treatment-versus-control comparisons.
#[derive(Clone, Debug, PartialEq)]
pub struct DunnettResult {
    /// Signed pooled t statistics: treatment minus control, in treatment order.
    pub statistic: Vec<f64>,
    /// Familywise-adjusted p-values in the same order.
    pub pvalue: Vec<f64>,
}
struct DunnettQuadrature {
    normal: Arc<Vec<(f64, f64)>>,
    scales: Vec<(f64, f64)>,
}
impl DunnettQuadrature {
    fn new(order: usize, df: f64) -> Result<Self> {
        type Rule = Result<Arc<Vec<(f64, f64)>>>;
        static NORMAL_128: OnceLock<Rule> = OnceLock::new();
        static NORMAL_256: OnceLock<Rule> = OnceLock::new();
        let cache = if order == 128 {
            &NORMAL_128
        } else {
            &NORMAL_256
        };
        let normal = cache
            .get_or_init(|| {
                let (nodes, weights) = crate::special::orthogonal::roots_hermitenorm(order)?;
                let normalizer = (2. * std::f64::consts::PI).sqrt();
                Ok(Arc::new(
                    nodes
                        .into_iter()
                        .zip(weights)
                        .map(|(z, w)| (z, w / normalizer))
                        .collect(),
                ))
            })
            .clone()?;
        let (nodes, weights) = crate::special::orthogonal::roots_genlaguerre(order, df / 2. - 1.)?;
        let log_gamma = crate::special::gamma::gammaln(df / 2.);
        let scales = nodes
            .into_iter()
            .zip(weights)
            .filter(|(_, w)| *w > 0.)
            .map(|(v, w)| ((2. * v / df).sqrt(), (w.ln() - log_gamma).exp()))
            .collect();
        Ok(Self { normal, scales })
    }
}
fn dunnett_rules(df: usize) -> Result<&'static (DunnettQuadrature, DunnettQuadrature)> {
    type Rules = Result<(DunnettQuadrature, DunnettQuadrature)>;
    static CACHE: OnceLock<[OnceLock<Rules>; 255]> = OnceLock::new();
    let cache = CACHE.get_or_init(|| std::array::from_fn(|_| OnceLock::new()));
    cache[df - 2]
        .get_or_init(|| {
            Ok((
                DunnettQuadrature::new(128, df as f64)?,
                DunnettQuadrature::new(256, df as f64)?,
            ))
        })
        .as_ref()
        .map_err(Clone::clone)
}
fn dunnett_probability(
    statistic: f64,
    correlations: &[(f64, f64)],
    alternative: Alternative,
    quadrature: &DunnettQuadrature,
) -> Result<f64> {
    use statrs::distribution::ContinuousCDF;
    let normal = normal(0., 1.)?;
    let threshold = match alternative {
        Alternative::TwoSided => statistic.abs(),
        Alternative::Less => -statistic,
        Alternative::Greater => statistic,
    };
    let mut probability = 0.;
    for &(node, weight) in quadrature.normal.iter() {
        let mut conditional = 0.;
        for &(scale, w) in &quadrature.scales {
            let mut product = 1.;
            for &(r, sd) in correlations {
                let upper = (threshold * scale - r * node) / sd;
                let p = if alternative == Alternative::TwoSided {
                    let lower = (-threshold * scale - r * node) / sd;
                    if lower > 0. {
                        normal.sf(lower) - normal.sf(upper)
                    } else {
                        normal.cdf(upper) - normal.cdf(lower)
                    }
                } else {
                    normal.cdf(upper)
                };
                product *= p.clamp(0., 1.);
            }
            conditional += w * product;
        }
        probability += weight * conditional;
    }
    match (1. - probability).clamp(0., 1.) {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }
}

/// Dunnett multiple treatment-versus-control t tests with a shared control.
///
/// ## Parameters and conventions
///
/// `control` and `treatments` are independent normal samples with common variance.
/// `T_i=(mean_i−mean_control)/sqrt(MSE*(1/n_i+1/n_control))`, with pooled MSE and
/// ν=N−k−1 degrees of freedom for k treatments. Null t correlations are
/// `ρ_ij=r_i*r_j`, `r_i=sqrt((1/n_control)/(1/n_i+1/n_control))`. P-values use
/// the joint multivariate-t max statistic (absolute max for TwoSided). Less
/// tests treatments below control, Greater tests above. SciPy RNG-based joint
/// probabilities are adapted to deterministic normal/gamma Gaussian quadrature;
/// confidence intervals and unequal-variance comparisons are unsupported.
///
/// ## Domain and special values
///
/// 1..=8 treatments; each sample/control has at least two finite observations.
/// Residual degrees of freedom must be 2..=256. Nonzero within-group spread and
/// representable normalized summaries are required; zero spread is an error.
/// NaN/infinity data are errors.
///
/// ## Returns
///
/// Matching dimensionless t-statistic and familywise p-value vectors of length k.
///
/// ## Errors
///
/// Invalid domains/shapes/work limits, unrepresentable arithmetic or failure
/// of 128/256-order quadrature agreement within 5e-5 return errors.
///
/// ## Numerical behavior
///
/// Native scaled summaries; private shared Hermite/Laguerre quadrature and
/// normal-CDF backends evaluate the common-normal/chi-square mixture. At most
/// 2*k²*(128²+256²) normal-tail calls, ≤10485760 for k=8. Agreement estimates
/// discretization error but is not a rigorous accuracy guarantee; correlations
/// near one and small ν can fail convergence. Very small p-values lose relative
/// accuracy by CDF subtraction. A single treatment reuses the exact Student-t identity.
/// Thread-safe quadrature rules are cached for the 255 supported integer degrees
/// of freedom, using at most 2 MiB across calls; no caller data is cached.
///
/// ## Examples
///
/// ```
/// let r = sci_stats::inference::dunnett(&[1.,2.,3.],&[&[4.,5.,6.]],sci_stats::inference::Alternative::TwoSided)?;
/// assert!(r.pvalue[0] < 0.05);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy dunnett and Dunnett (1955)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dunnett.html).
pub fn dunnett(
    control: &[f64],
    treatments: &[&[f64]],
    alternative: Alternative,
) -> Result<DunnettResult> {
    sample(control, 2)?;
    if !(1..=8).contains(&treatments.len()) {
        return Err(Error::InvalidInput("Dunnett requires 1..=8 treatments"));
    }
    let mut groups = Vec::with_capacity(treatments.len() + 1);
    groups.push(control);
    groups.extend_from_slice(treatments);
    let count = inference::groups_check(&groups, 2)?;
    for group in &groups {
        sample(group, 2)?;
    }
    let df = (count - groups.len()) as f64;
    if !(2. ..=256.).contains(&df) {
        return Err(Error::InvalidInput(
            "Dunnett residual degrees of freedom must be 2..=256",
        ));
    }
    let pooled = groups
        .iter()
        .flat_map(|r| r.iter().copied())
        .collect::<Vec<_>>();
    let centered = centered_sample(&pooled)?;
    let (mut squares, mut offset) = (0., 0);
    let mut means = Vec::with_capacity(groups.len());
    for group in &groups {
        let slice = &centered.deviations[offset..offset + group.len()];
        let center = mean(slice)?;
        means.push(center);
        squares += slice.iter().map(|v| (v - center).powi(2)).sum::<f64>();
        offset += group.len();
    }
    let mse = squares / df;
    positive(mse)?;
    let nc = control.len() as f64;
    let mut statistic = Vec::with_capacity(treatments.len());
    let mut correlations = Vec::with_capacity(treatments.len());
    for (i, treatment) in treatments.iter().enumerate() {
        let factor = 1. / nc + 1. / treatment.len() as f64;
        statistic.push(match (means[i + 1] - means[0]) / (mse * factor).sqrt() {
            value if value.is_finite() => Ok(value),
            _ => Err(crate::Error::NonFiniteEvaluation),
        }?);
        let r = ((1. / nc) / factor).sqrt();
        correlations.push((r, (1. - r * r).sqrt()));
    }
    let quadratures = if treatments.len() > 1 {
        Some(dunnett_rules(df as usize)?)
    } else {
        None
    };
    let mut pvalue = Vec::with_capacity(treatments.len());
    for &t in &statistic {
        if treatments.len() == 1 {
            pvalue.push(inference::t_probability(t, df, alternative)?);
        } else {
            let (coarse, fine) = quadratures
                .as_ref()
                .ok_or(Error::NumericalFailure("missing Dunnett quadrature"))?;
            let lower = dunnett_probability(t, &correlations, alternative, coarse)?;
            let upper = dunnett_probability(t, &correlations, alternative, fine)?;
            if (lower - upper).abs() > 5e-5 {
                return Err(Error::NumericalFailure(
                    "Dunnett quadrature did not converge",
                ));
            }
            pvalue.push(upper);
        }
    }
    Ok(DunnettResult { statistic, pvalue })
}
