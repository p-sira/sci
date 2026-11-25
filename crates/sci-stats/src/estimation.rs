use super::*;
use statrs::function::gamma::ln_gamma;

/// Posterior mean and equal-tail credible interval for one parameter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CredibleEstimate {
    /// Posterior expectation; positive infinity when it does not exist.
    pub statistic: f64,
    /// Equal-tail lower endpoint, in the parameter's units.
    pub lower: f64,
    /// Equal-tail upper endpoint, in the parameter's units.
    pub upper: f64,
}

/// Bayesian summaries of normal-population mean, variance and standard deviation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BayesianMvs {
    /// Mean summary, in observation units.
    pub mean: CredibleEstimate,
    /// Variance summary, in squared observation units.
    pub variance: CredibleEstimate,
    /// Standard-deviation summary, in observation units.
    pub std_dev: CredibleEstimate,
}

struct Posterior {
    n: f64,
    center: f64,
    variance: f64,
}
impl Posterior {
    fn new(values: &[f64]) -> Result<Self> {
        sample(values, 2)?;
        let variance = variance(values, 0)?;
        if variance == 0. {
            return Err(Error::InvalidInput(
                "posterior requires nonzero sample variance",
            ));
        }
        Ok(Self {
            n: values.len() as f64,
            center: mean(values)?,
            variance,
        })
    }
    fn quantiles(&self, probability: f64) -> Result<[f64; 3]> {
        // Match SciPy's Gaussian approximation for samples larger than 1000.
        if self.n > 1000. {
            return Ok([
                normal_ppf(probability, self.center, (self.variance / self.n).sqrt())?,
                normal_ppf(
                    probability,
                    self.variance,
                    (2. / self.n).sqrt() * self.variance,
                )?,
                normal_ppf(
                    probability,
                    self.variance.sqrt(),
                    (self.variance / (2. * self.n)).sqrt(),
                )?,
            ]);
        }
        let t = ppf(Distribution::StudentT { df: self.n - 1. }, probability)?;
        let v = ppf(
            Distribution::InverseGamma {
                shape_value: (self.n - 1.) / 2.,
                scale: match (self.n / 2.) * self.variance {
                    value if value.is_finite() => Ok(value),
                    _ => Err(crate::Error::NonFiniteEvaluation),
                }?,
            },
            probability,
        )?;
        let mu = self.center + (self.variance / (self.n - 1.)).sqrt() * t;
        if probability > 0. && probability < 1. && !mu.is_finite() {
            return Err(crate::Error::NonFiniteEvaluation);
        }
        Ok([mu, v, v.sqrt()])
    }
    fn means(&self) -> Result<[f64; 3]> {
        if self.n > 1000. {
            return Ok([self.center, self.variance, self.variance.sqrt()]);
        }
        let a = (self.n - 1.) / 2.;
        Ok([
            if self.n == 2. {
                f64::INFINITY
            } else {
                self.center
            },
            if self.n <= 3. {
                f64::INFINITY
            } else {
                match self.variance * (self.n / (self.n - 3.)) {
                    value if value.is_finite() => Ok(value),
                    _ => Err(crate::Error::NonFiniteEvaluation),
                }?
            },
            if self.n == 2. {
                f64::INFINITY
            } else {
                match self.variance.sqrt()
                    * ((self.n / 2.).ln() / 2. + ln_gamma(a - 0.5) - ln_gamma(a)).exp()
                {
                    value if value.is_finite() => Ok(value),
                    _ => Err(crate::Error::NonFiniteEvaluation),
                }?
            },
        ])
    }
}

/// Posterior quantiles for mean, variance and standard deviation of a normal population.
///
/// ## Parameters and conventions
///
/// `values` is an unweighted sample and `probability` is the quantile probability.
/// With n observations, sample mean m and population variance C, the posteriors
/// are `m+sqrt(C/(n−1))*T_(n−1)`, `V~InvGamma((n−1)/2,n*C/2)`, and `sqrt(V)`.
/// This functional adaptation of SciPy returns evaluations of its three frozen
/// distributions; it does not expose distribution objects. For n>1000, SciPy's
/// normal approximations use means `(m,C,sqrt(C))` and standard deviations
/// `(sqrt(C/n),C*sqrt(2/n),sqrt(C/(2n)))`.
///
/// ## Domain and special values
///
/// 2..=1048576 finite observations with representable positive variance; p∈`[0,1]`.
/// Endpoint quantiles may be infinite. Large-sample normal approximations can
/// assign negative variance or standard deviation quantiles in extreme tails.
///
/// ## Returns
///
/// `[mean_quantile, variance_quantile, std_dev_quantile]`, in observation,
/// squared observation and observation units, respectively.
///
/// ## Errors
///
/// Invalid samples/probabilities and constant data return errors. Unrepresentable
/// intermediate parameters or interior quantiles return evaluation errors.
///
/// ## Numerical behavior
///
/// O(n) summaries reuse compensated/scaled Sci statistics. Private statrs tails
/// and bounded Sci quantile inversion are used. Extreme tails lose accuracy.
///
/// ## Examples
///
/// ```
/// assert_eq!(sci_stats::estimation::mvsdist(&[1.,2.,3.], 0.5)?[0], 2.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy mvsdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mvsdist.html).
pub fn mvsdist(values: &[f64], probability: f64) -> Result<[f64; 3]> {
    probability_check(probability)?;
    Posterior::new(values)?.quantiles(probability)
}

/// Posterior expectations and equal-tail credible intervals for normal mean and spread.
///
/// ## Parameters and conventions
///
/// `values` is an unweighted normal-population sample; `alpha` is interval
/// probability, not significance. [`mvsdist`] defines the posterior distributions.
/// Intervals use quantiles `(1−alpha)/2` and `(1+alpha)/2`. For n≤1000, variance
/// expectation is `n*C/(n−3)` (n>3) and standard-deviation expectation is
/// `sqrt(n*C/2)*Γ((n−2)/2)/Γ((n−1)/2)` (n>2).
///
/// ## Domain and special values
///
/// 2..=1048576 finite nonconstant observations and 0<alpha<1. Undefined
/// posterior expectations use positive infinity, matching SciPy, including
/// its Student-t mean convention at n=2. Constant samples are errors.
///
/// ## Returns
///
/// Named mean, variance and standard-deviation summaries, each with posterior
/// expectation and equal-tail endpoints. Units follow [`mvsdist`].
///
/// ## Errors
///
/// Invalid domains, nonfinite data and unrepresentable arithmetic return errors.
///
/// ## Numerical behavior
///
/// O(n) summaries and bounded private-backend quantiles. The Gaussian approximation
/// switches at n>1000, matching SciPy; very small samples have heavy posterior tails.
///
/// ## Examples
///
/// ```
/// let r = sci_stats::estimation::bayes_mvs(&[1.,2.,3.,4.], 0.9)?;
/// assert_eq!(r.mean.statistic, 2.5);
/// assert!(r.mean.lower < 2.5 && r.mean.upper > 2.5);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy bayes_mvs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bayes_mvs.html).
pub fn bayes_mvs(values: &[f64], alpha: f64) -> Result<BayesianMvs> {
    if !alpha.is_finite() || alpha <= 0. || alpha >= 1. {
        return Err(Error::InvalidInput("credible probability must be in (0,1)"));
    }
    let posterior = Posterior::new(values)?;
    let lower = posterior.quantiles((1. - alpha) / 2.)?;
    let upper = posterior.quantiles(0.5 + alpha / 2.)?;
    let means = posterior.means()?;
    let estimate = |i: usize| CredibleEstimate {
        statistic: means[i],
        lower: lower[i],
        upper: upper[i],
    };
    Ok(BayesianMvs {
        mean: estimate(0),
        variance: estimate(1),
        std_dev: estimate(2),
    })
}
