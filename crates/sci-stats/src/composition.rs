use super::*;
use distributions::Operation;

/// Native distribution description accepted by the shared density/CDF/quantile APIs.
#[derive(Clone, Copy, Debug)]
pub enum DistributionSpec {
    /// An ordinary parameterized Sci distribution.
    Base(Distribution),
    /// A continuous distribution conditioned on a finite interval.
    Truncated(TruncatedDistribution),
    /// A continuous order-statistic distribution.
    OrderStatistic(OrderStatisticDistribution),
    /// A user-supplied continuous scalar distribution.
    Custom(CustomDistribution),
}
impl From<Distribution> for DistributionSpec {
    fn from(value: Distribution) -> Self {
        Self::Base(value)
    }
}
/// Validated finite-interval conditional distribution, constructed by [`truncate`].
#[derive(Clone, Copy, Debug)]
pub struct TruncatedDistribution {
    base: Distribution,
    lower: f64,
    upper: f64,
    mass: f64,
    upper_tail: bool,
}
/// Validated continuous order-statistic distribution, constructed by [`order_statistic`].
#[derive(Clone, Copy, Debug)]
pub struct OrderStatisticDistribution {
    base: Distribution,
    rank: usize,
    count: usize,
}
/// Continuous distribution with checked scalar function-pointer callbacks.
#[derive(Clone, Copy, Debug)]
pub struct CustomDistribution {
    lower: f64,
    upper: f64,
    density: fn(f64) -> Result<f64>,
    cumulative: fn(f64) -> Result<f64>,
    survival: fn(f64) -> Result<f64>,
    quantile: fn(f64) -> Result<f64>,
}

fn probability_result(value: f64) -> Result<f64> {
    if (0. ..=1.).contains(&value) {
        Ok(value)
    } else {
        Err(Error::NumericalFailure(
            "distribution callback returned invalid probability",
        ))
    }
}
fn density_result(value: f64, logarithmic: bool) -> Result<f64> {
    if value.is_nan() {
        return Err(Error::NonFiniteEvaluation);
    }
    if logarithmic {
        return Ok(value);
    }
    let density = value.exp();
    if value.is_finite() && density.is_infinite() {
        return Err(Error::NonFiniteEvaluation);
    }
    Ok(density)
}
fn custom_evaluate(d: CustomDistribution, x: f64, op: Operation) -> Result<f64> {
    match op {
        Operation::Ppf => {
            let value = (d.quantile)(x)?;
            if value.is_nan()
                || value < d.lower
                || value > d.upper
                || (x > 0. && x < 1. && !value.is_finite())
            {
                return Err(Error::NumericalFailure(
                    "distribution callback returned invalid quantile",
                ));
            }
            Ok(value)
        }
        Operation::Cdf => {
            if x < d.lower {
                Ok(0.)
            } else if x >= d.upper {
                Ok(1.)
            } else {
                probability_result((d.cumulative)(x)?)
            }
        }
        Operation::Sf => {
            if x < d.lower {
                Ok(1.)
            } else if x >= d.upper {
                Ok(0.)
            } else {
                probability_result((d.survival)(x)?)
            }
        }
        Operation::Density | Operation::LogDensity => {
            let value = if x < d.lower || x > d.upper || x.is_infinite() {
                0.
            } else {
                (d.density)(x)?
            };
            if value.is_nan() || value < 0. {
                return Err(Error::NumericalFailure(
                    "distribution callback returned invalid density",
                ));
            }
            Ok(if matches!(op, Operation::LogDensity) {
                value.ln()
            } else {
                value
            })
        }
    }
}
fn truncated_probability(d: TruncatedDistribution, x: f64, survival: bool) -> Result<f64> {
    if x <= d.lower {
        return Ok(if survival { 1. } else { 0. });
    }
    if x >= d.upper {
        return Ok(if survival { 0. } else { 1. });
    }
    let mass = if d.upper_tail {
        if survival {
            sf(d.base, x)? - sf(d.base, d.upper)?
        } else {
            sf(d.base, d.lower)? - sf(d.base, x)?
        }
    } else if survival {
        cdf(d.base, d.upper)? - cdf(d.base, x)?
    } else {
        cdf(d.base, x)? - cdf(d.base, d.lower)?
    };
    Ok((mass / d.mass).clamp(0., 1.))
}
fn truncated_evaluate(d: TruncatedDistribution, x: f64, op: Operation) -> Result<f64> {
    match op {
        Operation::Cdf => truncated_probability(d, x, false),
        Operation::Sf => truncated_probability(d, x, true),
        Operation::Density | Operation::LogDensity => {
            let log_density = if x < d.lower || x > d.upper {
                f64::NEG_INFINITY
            } else {
                logpdf(d.base, x)? - d.mass.ln()
            };
            density_result(log_density, matches!(op, Operation::LogDensity))
        }
        Operation::Ppf => {
            if x == 0. {
                return Ok(d.lower);
            }
            if x == 1. {
                return Ok(d.upper);
            }
            let (mut lower, mut upper) = (d.lower, d.upper);
            for _ in 0..2048 {
                let middle = lower / 2. + upper / 2.;
                if middle == lower || middle == upper {
                    return Ok(upper);
                }
                // Use the smaller tail for both comparisons.
                let below = if x <= 0.5 {
                    truncated_probability(d, middle, false)? < x
                } else {
                    truncated_probability(d, middle, true)? > 1. - x
                };
                if below {
                    lower = middle;
                } else {
                    upper = middle;
                }
            }
            Err(Error::NumericalFailure(
                "truncated quantile bisection limit exceeded",
            ))
        }
    }
}
fn order_evaluate(d: OrderStatisticDistribution, x: f64, op: Operation) -> Result<f64> {
    let a = d.rank as f64;
    let b = (d.count - d.rank + 1) as f64;
    match op {
        Operation::Ppf => {
            let probability = ppf(Distribution::Beta { a, b }, x)?;
            let value = ppf(d.base, probability)?;
            if x > 0. && x < 1. {
                match value {
                    value if value.is_finite() => Ok(value),
                    _ => Err(crate::Error::NonFiniteEvaluation),
                }
            } else {
                Ok(value)
            }
        }
        Operation::Cdf | Operation::Sf => {
            let f = cdf(d.base, x)?;
            if matches!(op, Operation::Cdf) {
                if f <= 0.5 {
                    cdf(Distribution::Beta { a, b }, f)
                } else {
                    sf(Distribution::Beta { a: b, b: a }, sf(d.base, x)?)
                }
            } else if f <= 0.5 {
                sf(Distribution::Beta { a, b }, f)
            } else {
                cdf(Distribution::Beta { a: b, b: a }, sf(d.base, x)?)
            }
        }
        Operation::Density | Operation::LogDensity => {
            let log_density = logpdf(d.base, x)?;
            let log_f = if d.rank == 1 {
                0.
            } else {
                (a - 1.) * cdf(d.base, x)?.ln()
            };
            let log_s = if d.rank == d.count {
                0.
            } else {
                (b - 1.) * sf(d.base, x)?.ln()
            };
            // B(m,q)=(m−1)!/[q...(q+m−1)] for integer m. A short
            // recurrence avoids subtracting large log-gamma values when
            // one order-statistic shape is small, including B(1,n)=1/n.
            let small = a.min(b) as usize;
            let large = a.max(b);
            let constant = if small <= 32 {
                let log_large = large.ln();
                (0..small)
                    .map(|i| {
                        log_large + (i as f64 / large).ln_1p()
                            - if i == 0 { 0. } else { (i as f64).ln() }
                    })
                    .sum()
            } else {
                -crate::special::beta::betaln(a, b)
            };
            density_result(
                log_density + log_f + log_s + constant,
                matches!(op, Operation::LogDensity),
            )
        }
    }
}
pub(super) fn evaluate(distribution: DistributionSpec, x: f64, op: Operation) -> Result<f64> {
    if x.is_nan() {
        return Err(Error::InvalidInput("distribution argument must not be NaN"));
    }
    if matches!(op, Operation::Ppf) {
        probability_check(x)?;
    }
    match distribution {
        DistributionSpec::Base(d) => distributions::evaluate(d, x, op),
        DistributionSpec::Truncated(d) => truncated_evaluate(d, x, op),
        DistributionSpec::OrderStatistic(d) => order_evaluate(d, x, op),
        DistributionSpec::Custom(d) => custom_evaluate(d, x, op),
    }
}

/// Conditional continuous distribution `f_T(x)=f(x)/P(lower≤X≤upper)` on the interval.
///
/// ## Parameters and conventions
///
/// `distribution` is a native continuous base distribution, and `lower`,`upper`
/// are finite conditioning bounds in observation units. Effective bounds are
/// intersected with base support. Density is zero outside; CDF/SF renormalize
/// interval probabilities. This adapts SciPy's distribution object to a native
/// description accepted by [`pdf`], [`logpdf`], [`cdf`], [`sf`] and [`ppf`].
/// Nested transformations, discrete bases, sampling, moments and fitting are unsupported.
///
/// ## Domain and special values
///
/// Finite lower<upper, valid continuous base parameters and representable positive
/// interval probability. NaN/infinite bounds are errors. A support intersection
/// with zero probability is an error. PPF endpoints are effective support bounds.
///
/// ## Returns
///
/// Native conditional description with density in inverse observation units,
/// dimensionless probabilities and quantiles in observation units.
///
/// ## Errors
///
/// Invalid bases/bounds, empty intersections or underflowed interval mass return errors.
/// Subsequent evaluation propagates base errors and reports unrepresentable densities.
///
/// ## Numerical behavior
///
/// Uses the smaller base tail to normalize, preserving truncation beyond CDF
/// saturation. PPF uses at most 2048 bounded bisections (4096 base-tail calls),
/// stopping at adjacent representable points. Narrow interval subtraction loses
/// relative accuracy; truncations with unrepresentable mass are rejected.
///
/// ## Examples
///
/// ```
/// let d = sci_stats::truncate(sci_stats::Distribution::Normal { mean:0.,std_dev:1. },-1.,1.)?;
/// assert!((sci_stats::cdf(d,0.)?-0.5).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy truncate](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncate.html).
pub fn truncate(distribution: Distribution, lower: f64, upper: f64) -> Result<DistributionSpec> {
    continuous_null(distribution)?;
    if !lower.is_finite() {
        return Err(crate::Error::NonFiniteEvaluation);
    }
    if !upper.is_finite() {
        return Err(crate::Error::NonFiniteEvaluation);
    }
    if lower >= upper {
        return Err(Error::InvalidInput("truncation bounds must increase"));
    }
    let lower = lower.max(ppf(distribution, 0.)?);
    let upper = upper.min(ppf(distribution, 1.)?);
    if lower >= upper {
        return Err(Error::InvalidInput("empty truncated support"));
    }
    let upper_tail = cdf(distribution, lower)? > 0.5;
    let mass = if upper_tail {
        sf(distribution, lower)? - sf(distribution, upper)?
    } else {
        cdf(distribution, upper)? - cdf(distribution, lower)?
    };
    positive(mass)?;
    Ok(DistributionSpec::Truncated(TruncatedDistribution {
        base: distribution,
        lower,
        upper,
        mass,
        upper_tail,
    }))
}

/// Continuous r-th order statistic `f_r(x)=n!/(r−1)!/(n−r)! *f(x)*F(x)^(r−1)*S(x)^(n−r)`.
///
/// ## Parameters and conventions
///
/// `distribution` is a native continuous base; `rank` is one-based r, and `count`
/// is n independent identically distributed draws. Its CDF is
/// `I_(F(x))(r,n−r+1)` and quantile is `F⁻¹(BetaQuantile(p;r,n−r+1))`.
/// Adapts SciPy objects to a native description reusable by the five shared
/// density/CDF/quantile APIs. Discrete/nested bases and object methods are unsupported.
///
/// ## Domain and special values
///
/// Valid continuous parameters, 1≤rank≤count≤1048576. Infinite evaluation points
/// are accepted; NaN is an error. PPF endpoints retain the base support.
/// Singular finite support endpoints can have indeterminate density products,
/// which return an evaluation error instead of guessing a limit.
///
/// ## Returns
///
/// A native order-statistic description; quantiles retain base units and densities
/// have inverse base units. CDF and SF are dimensionless.
///
/// ## Errors
///
/// Invalid counts/bases return errors. Evaluation errors include indeterminate
/// density limits and nonrepresentable interior quantiles/densities.
///
/// ## Numerical behavior
///
/// Private shared beta tails/inversion and log gamma coefficients. Density is
/// computed in log space; smaller-tail CDF identities avoid upper-tail subtraction.
/// Tail quantiles can lose accuracy if an intermediate beta probability rounds
/// to an endpoint. Work limits inherit the bounded beta/base quantile inversions.
///
/// ## Examples
///
/// ```
/// let d = sci_stats::order_statistic(sci_stats::Distribution::Uniform { low:0.,high:1. },2,3)?;
/// assert!((sci_stats::cdf(d,0.5)?-0.5).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy order_statistic](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.order_statistic.html).
pub fn order_statistic(
    distribution: Distribution,
    rank: usize,
    count: usize,
) -> Result<DistributionSpec> {
    continuous_null(distribution)?;
    if rank == 0 || rank > count || count > MAX_SAMPLES {
        return Err(Error::InvalidInput(
            "order statistic requires 1<=rank<=count<=1048576",
        ));
    }
    Ok(DistributionSpec::OrderStatistic(
        OrderStatisticDistribution {
            base: distribution,
            rank,
            count,
        },
    ))
}

/// Construct a custom continuous scalar distribution from density, CDF, SF and quantile callbacks.
///
/// ## Parameters and conventions
///
/// `lower`,`upper` are support bounds, possibly infinite; `density`,`cumulative`,
/// `survival`,`quantile` are fallible scalar function pointers. Density integrates
/// to one; cumulative is its integral; survival is 1−CDF, preferably evaluated
/// directly for tail accuracy; quantile is `inf{x:F(x)≥p}`. Callback consistency,
/// normalization and monotonicity are caller obligations. This functional SciPy
/// adaptation accepts explicit scalar callbacks rather than defining Python classes.
/// Capturing closures, parameter dictionaries, discrete distributions, automatic
/// inversion, sampling and moments are unsupported.
///
/// ## Domain and special values
///
/// Non-NaN lower<upper. Construction checks CDF/SF at both support endpoints,
/// endpoint quantiles and median consistency (absolute tolerance 1e-10).
/// Every evaluation checks nonnegative density, probabilities in `[0,1]`, and
/// quantiles within support; interior quantiles must be finite. Infinite density
/// is permitted at singularities; log density of zero is negative infinity.
///
/// ## Returns
///
/// A native custom description reusable by [`pdf`], [`logpdf`], [`cdf`], [`sf`]
/// and [`ppf`]. Density/quantile units are supplied by the callbacks.
///
/// ## Errors
///
/// Invalid support or failed consistency checks return errors. Callback errors
/// propagate unchanged during construction/evaluation. Validation does not prove
/// that supplied callbacks define a normalized probability distribution.
///
/// ## Numerical behavior
///
/// Seven callback evaluations at construction, plus median CDF/SF and density
/// checks (ten total); each subsequent evaluation calls at most one callback.
/// Callback work/accuracy are controlled by the caller. Custom log density is
/// ln(density), so density underflow loses log-tail information.
///
/// ## Examples
///
/// ```
/// let d = sci_stats::make_distribution(0.,1.,|x| Ok(2.*x),|x| Ok(x*x),
///     |x| Ok(1.-x*x),|p| Ok(p.sqrt()))?;
/// assert_eq!(sci_stats::cdf(d,0.5)?,0.25);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy make_distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.make_distribution.html).
pub fn make_distribution(
    lower: f64,
    upper: f64,
    density: fn(f64) -> Result<f64>,
    cumulative: fn(f64) -> Result<f64>,
    survival: fn(f64) -> Result<f64>,
    quantile: fn(f64) -> Result<f64>,
) -> Result<DistributionSpec> {
    if lower.is_nan() || upper.is_nan() || lower >= upper {
        return Err(Error::InvalidInput("custom support must increase"));
    }
    for (point, expected) in [(lower, 0.), (upper, 1.)] {
        if (probability_result(cumulative(point)?)? - expected).abs() > 1e-10
            || (probability_result(survival(point)?)? - (1. - expected)).abs() > 1e-10
            || quantile(expected)? != point
        {
            return Err(Error::InvalidInput("inconsistent distribution endpoints"));
        }
    }
    let d = CustomDistribution {
        lower,
        upper,
        density,
        cumulative,
        survival,
        quantile,
    };
    let median = custom_evaluate(d, 0.5, Operation::Ppf)?;
    if (probability_result(cumulative(median)?)? - 0.5).abs() > 1e-10
        || (probability_result(survival(median)?)? - 0.5).abs() > 1e-10
    {
        return Err(Error::InvalidInput("inconsistent distribution median"));
    }
    custom_evaluate(d, median, Operation::Density)?;
    Ok(DistributionSpec::Custom(d))
}
