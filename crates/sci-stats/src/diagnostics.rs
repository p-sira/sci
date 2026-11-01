use super::*;

/// Coordinates for a power-transform normality plot.
#[derive(Clone, Debug, PartialEq)]
pub struct TransformPlot {
    /// Ascending, inclusive, equally spaced transform parameters λ.
    pub lambdas: Vec<f64>,
    /// Pearson correlations between normal theoretical quantiles and sorted
    /// transformed observations, matching `lambdas` in order.
    pub correlations: Vec<f64>,
}

fn transform_plot(
    values: &[f64],
    lower: f64,
    upper: f64,
    count: usize,
    transform: fn(&[f64], f64) -> Result<Vec<f64>>,
) -> Result<TransformPlot> {
    if !lower.is_finite() || !upper.is_finite() || lower >= upper || !(2..=65536).contains(&count) {
        return Err(Error::InvalidInput(
            "plot requires finite increasing bounds and 2..=65536 points",
        ));
    }
    sample(values, 2)?;
    if count
        .checked_mul(values.len())
        .is_none_or(|work| work > 16_777_216)
    {
        return Err(Error::InvalidInput(
            "transform-plot work limit is 16777216 observations",
        ));
    }
    if values.iter().all(|v| *v == values[0]) {
        return Err(Error::InvalidInput(
            "transform plot requires nonconstant observations",
        ));
    }
    let plot = probplot(
        values,
        Distribution::Normal {
            mean: 0.,
            std_dev: 1.,
        },
    )?;
    let mut lambdas = Vec::with_capacity(count);
    let mut correlations = Vec::with_capacity(count);
    for i in 0..count {
        let fraction = i as f64 / (count - 1) as f64;
        let lambda = if i == 0 {
            lower
        } else if i == count - 1 {
            upper
        } else {
            (1. - fraction) * lower + fraction * upper
        };
        // Both power transforms are monotone increasing. Sorting once suffices.
        let transformed = transform(&plot.ordered, lambda)?;
        lambdas.push(lambda);
        correlations.push(inference::correlation(&plot.theoretical, &transformed)?);
    }
    Ok(TransformPlot {
        lambdas,
        correlations,
    })
}

/// Box–Cox normality plot data: `r(λ)=corr(Φ⁻¹(p_i), sorted(BC_λ(x)))`.
///
/// ## Parameters and conventions
///
/// `values` is a one-dimensional sample; `lower` and `upper` bound the inclusive
/// equally spaced λ grid with explicit `count`. `BC_λ(x)=(x^λ−1)/λ`, with
/// BC₀(x)=ln(x). The plotting positions p_i follow [`probplot`]'s Filliben
/// convention. This returns data only; SciPy plotting axes are unsupported.
///
/// ## Domain and special values
///
/// At least two finite strictly positive, nonconstant observations. Finite
/// lower<upper, 2..=65536 grid points, and count*n≤16,777,216. A transform
/// that rounds to constant can yield NaN correlation.
///
/// ## Returns
///
/// Matching λ and dimensionless correlation vectors, both of length count.
///
/// ## Errors
///
/// Invalid domains/counts/bounds, nonfinite inputs, or overflowing transforms
/// return errors. The work limit is validated before quantile evaluation.
///
/// ## Numerical behavior
///
/// Native transforms, sorting and correlations; O(n log n+count*n) work and
/// O(n+count) memory. Normal quantiles reuse statrs through [`probplot`].
/// Binary64 rounding can affect nearly constant transformed samples. This
/// diagnostic does not optimize λ or test a hypothesis.
///
/// ## Examples
///
/// ```
/// let plot = sci_stats::diagnostics::boxcox_normplot(&[1.,2.,4.,8.], -1., 1., 3)?;
/// assert_eq!(plot.lambdas, vec![-1., 0., 1.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy boxcox_normplot](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox_normplot.html).
pub fn boxcox_normplot(
    values: &[f64],
    lower: f64,
    upper: f64,
    count: usize,
) -> Result<TransformPlot> {
    if values.iter().any(|x| *x <= 0.) {
        return Err(Error::InvalidInput(
            "Box-Cox requires positive observations",
        ));
    }
    transform_plot(values, lower, upper, count, boxcox)
}

/// Yeo–Johnson normality plot data: `r(λ)=corr(Φ⁻¹(p_i), sorted(YJ_λ(x)))`.
///
/// ## Parameters and conventions
///
/// `values` is a one-dimensional sample. An explicit count of λ values is
/// equally spaced on `[lower,upper]`. [`yeojohnson`] defines the piecewise
/// transform, including its logarithmic λ=0 and λ=2 limits. The p_i use
/// [`probplot`]'s Filliben positions. SciPy's plotting axes are unsupported.
///
/// ## Domain and special values
///
/// At least two finite nonconstant observations; negative and zero values
/// are accepted. Finite lower<upper, 2..=65536 grid points, and
/// count*n≤16,777,216. A transform rounded to constant yields NaN correlation.
///
/// ## Returns
///
/// Matching λ and dimensionless Pearson correlation vectors in ascending λ order.
///
/// ## Errors
///
/// Invalid counts/bounds, nonfinite inputs, or overflowing transforms return
/// errors. Work is bounded before quantile evaluation.
///
/// ## Numerical behavior
///
/// Native O(n log n+count*n) transforms/sorting/correlations and O(n+count)
/// memory. Normal quantiles reuse the private statrs backend. Rounding can
/// affect nearly constant transformed data; no optimization or p-value is returned.
///
/// ## Examples
///
/// ```
/// let plot = sci_stats::diagnostics::yeojohnson_normplot(&[-2.,0.,1.,4.], 0., 2., 3)?;
/// assert_eq!(plot.correlations.len(), 3);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy yeojohnson_normplot](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.yeojohnson_normplot.html).
pub fn yeojohnson_normplot(
    values: &[f64],
    lower: f64,
    upper: f64,
    count: usize,
) -> Result<TransformPlot> {
    transform_plot(values, lower, upper, count, yeojohnson)
}

/// Objective for choosing a Box–Cox parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoxCoxMethod {
    /// Maximize Pearson correlation with Filliben normal quantiles.
    Pearson,
    /// Maximize the profile log likelihood defined by [`boxcox_llf`].
    MaximumLikelihood,
}

struct BoundedObjective<F> {
    function: F,
    lower: f64,
    upper: f64,
}
impl<F: Fn(f64) -> Result<f64>> basin::CostFunction for BoundedObjective<F> {
    type Param = f64;
    type Output = f64;
    type Error = Error;
    fn cost(&self, x: &f64) -> Result<f64> {
        match (self.function)(*x)? {
            value if value.is_finite() => Ok(value),
            _ => Err(Error::NonFiniteEvaluation),
        }
    }
}
impl<F: Fn(f64) -> Result<f64>> basin::BoxConstraints for BoundedObjective<F> {
    fn lower(&self) -> &f64 {
        &self.lower
    }
    fn upper(&self) -> &f64 {
        &self.upper
    }
}
fn search_bounds(lower: f64, upper: f64) -> Result<()> {
    if !lower.is_finite() || !upper.is_finite() || lower >= upper || lower < -64. || upper > 64. {
        return Err(Error::InvalidInput("search requires -64<=lower<upper<=64"));
    }
    Ok(())
}
fn search_sample(values: &[f64]) -> Result<()> {
    sample(values, 3)?;
    if values.len() > 65536 || values.iter().all(|x| *x == values[0]) {
        return Err(Error::InvalidInput(
            "search requires 3..=65536 nonconstant observations",
        ));
    }
    Ok(())
}
fn bounded_minimum(lower: f64, upper: f64, function: impl Fn(f64) -> Result<f64>) -> Result<f64> {
    search_bounds(lower, upper)?;
    // Evaluate endpoints explicitly; Brent generally returns an interior point.
    let left = match function(lower)? {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }?;
    let right = match function(upper)? {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }?;
    let result = basin::Executor::new(
        BoundedObjective {
            function: &function,
            lower,
            upper,
        },
        basin::Brent::new()
            .with_absolute_position_tolerance(1e-10)
            .with_relative_position_tolerance(1e-10),
        basin::ScalarState::new(lower / 2. + upper / 2.),
    )
    .max_iter(128)
    .run()?;
    if result.reason != basin::TerminationReason::SolverConverged {
        return Err(Error::NumericalFailure(
            "bounded parameter search did not converge",
        ));
    }
    let mut parameter = *result.best_param();
    let mut cost = result.best_cost();
    if left < cost {
        parameter = lower;
        cost = left;
    }
    if right < cost {
        parameter = upper;
    }
    Ok(parameter)
}

/// Box–Cox parameter `argmax_(λ∈[lower,upper]) L(λ)` or normal-quantile correlation.
///
/// ## Parameters and conventions
///
/// `values` is a positive sample; `method` chooses profile likelihood or
/// Filliben normal-quantile Pearson correlation. [`boxcox_llf`] and
/// [`boxcox_normplot`] define the objectives. Bounds are hard constraints,
/// unlike SciPy's expandable starting bracket. No `method="all"`, custom
/// optimizer, or transformed-value `ymax` constraint is provided.
///
/// ## Domain and special values
///
/// 3..=65536 finite positive nonconstant observations and −64≤lower<upper≤64.
/// NaN/infinity inputs are errors. Endpoints can be returned when optimal.
///
/// ## Returns
///
/// Dimensionless λ for a local maximum in the supplied closed interval.
///
/// ## Errors
///
/// Invalid domains, overflowing/constant transforms, nonfinite objective
/// evaluations or failure to converge within 128 iterations return errors.
///
/// ## Numerical behavior
///
/// Private Basin Brent search, with absolute and relative position tolerances
/// 1e-10; at most 132 objective evaluations. Binary64 rounding and flat objectives
/// commonly limit parameter accuracy to about 1e-6; global optimality is not
/// guaranteed. Correlation sorts once; likelihood costs O(n) per evaluation.
///
/// ## Examples
///
/// ```
/// let l = sci_stats::diagnostics::boxcox_normmax(&[1.,2.,4.,8.], -2., 2., sci_stats::diagnostics::BoxCoxMethod::MaximumLikelihood)?;
/// assert!(l.abs() < 1e-5);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy boxcox_normmax](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox_normmax.html).
pub fn boxcox_normmax(values: &[f64], lower: f64, upper: f64, method: BoxCoxMethod) -> Result<f64> {
    search_bounds(lower, upper)?;
    search_sample(values)?;
    if values.iter().any(|x| *x <= 0.) {
        return Err(Error::InvalidInput(
            "Box-Cox requires positive observations",
        ));
    }
    match method {
        BoxCoxMethod::MaximumLikelihood => {
            bounded_minimum(lower, upper, |l| Ok(-boxcox_llf(l, values)?))
        }
        BoxCoxMethod::Pearson => {
            let plot = probplot(
                values,
                Distribution::Normal {
                    mean: 0.,
                    std_dev: 1.,
                },
            )?;
            bounded_minimum(lower, upper, |l| {
                Ok(-inference::correlation(
                    &plot.theoretical,
                    &boxcox(&plot.ordered, l)?,
                )?)
            })
        }
    }
}

/// Yeo–Johnson maximum-likelihood parameter `argmax_(λ∈[lower,upper]) L(λ)`.
///
/// ## Parameters and conventions
///
/// `values` is an unweighted sample; [`yeojohnson_llf`] defines L. `lower`
/// and `upper` are hard constraints, unlike SciPy's expandable bracket.
///
/// ## Domain and special values
///
/// 3..=65536 finite nonconstant observations; negatives and zeros are accepted.
/// Bounds satisfy −64≤lower<upper≤64. Endpoints may be returned.
///
/// ## Returns
///
/// Dimensionless local maximum-likelihood transform parameter λ.
///
/// ## Errors
///
/// Invalid samples/bounds, nonfinite transforms or objectives, and exhausted
/// iterations return errors. A transform that rounds to constant is an error.
///
/// ## Numerical behavior
///
/// Uses the same private Basin Brent search and 132-evaluation limit as
/// [`boxcox_normmax`], with O(n) work per objective evaluation. Flat objectives
/// limit parameter accuracy to about 1e-6; no global optimum is guaranteed.
///
/// ## Examples
///
/// ```
/// assert!((sci_stats::diagnostics::yeojohnson_normmax(&[-2.,-1.,0.,1.,2.], -1., 3.)? - 1.).abs() < 1e-5);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy yeojohnson_normmax](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.yeojohnson_normmax.html).
pub fn yeojohnson_normmax(values: &[f64], lower: f64, upper: f64) -> Result<f64> {
    search_bounds(lower, upper)?;
    search_sample(values)?;
    bounded_minimum(lower, upper, |l| Ok(-yeojohnson_llf(l, values)?))
}

/// One-shape distribution families supported by PPCC shape diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PpccFamily {
    /// Standard Student-t; shape is degrees of freedom.
    StudentT,
    /// Chi-square; shape is degrees of freedom.
    ChiSquared,
    /// Gamma with unit scale; shape is the gamma shape parameter.
    Gamma,
    /// Weibull minimum with unit scale; shape is the Weibull shape parameter.
    Weibull,
    /// Inverse gamma with unit scale; shape is the inverse-gamma shape parameter.
    InverseGamma,
}
impl PpccFamily {
    fn distribution(self, shape_value: f64) -> Distribution {
        match self {
            Self::StudentT => Distribution::StudentT { df: shape_value },
            Self::ChiSquared => Distribution::ChiSquared { df: shape_value },
            Self::Gamma => Distribution::Gamma {
                shape_value,
                scale: 1.,
            },
            Self::Weibull => Distribution::Weibull {
                shape_value,
                scale: 1.,
            },
            Self::InverseGamma => Distribution::InverseGamma {
                shape_value,
                scale: 1.,
            },
        }
    }
}
/// Shape parameters and matching probability-plot correlation coefficients.
#[derive(Clone, Debug, PartialEq)]
pub struct PpccPlot {
    /// Ascending equally spaced distribution shape parameters.
    pub shapes: Vec<f64>,
    /// Dimensionless Pearson correlations, in matching order.
    pub correlations: Vec<f64>,
}
fn ppcc_correlation(values: &[f64], family: PpccFamily, shape: f64) -> Result<f64> {
    let plot = probplot(values, family.distribution(shape))?;
    inference::correlation(&plot.theoretical, &plot.ordered)
}
fn ppcc_bounds(lower: f64, upper: f64) -> Result<()> {
    search_bounds(lower, upper)?;
    positive(lower)
}

/// Shape PPCC plot `r(s)=corr(F_s⁻¹(p_i), sorted(x))`.
///
/// ## Parameters and conventions
///
/// `values` is an unweighted sample; `family` chooses a supported one-shape
/// distribution. The p_i are Filliben positions from [`probplot`]. The inclusive
/// shape grid is `[lower,upper]` with `count` equally spaced values. This functional
/// adaptation returns data without SciPy plotting objects. The SciPy default
/// Tukey-lambda family and arbitrary distribution callbacks are unsupported.
///
/// ## Domain and special values
///
/// 3..=65536 finite nonconstant observations, 0<lower<upper≤64, 2..=1024 grid
/// points, and count*n≤65536. Nonfinite values are errors.
///
/// ## Returns
///
/// Matching shape and dimensionless correlation vectors of length count.
///
/// ## Errors
///
/// Invalid inputs/work limits or unrepresentable distribution quantiles return errors.
///
/// ## Numerical behavior
///
/// Private statrs-backed bounded quantiles; O(count*n) quantile inversions and
/// O(count*n log n) sorting. Quantile accuracy deteriorates in extreme tails.
///
/// ## Examples
///
/// ```
/// let p = sci_stats::diagnostics::ppcc_plot(&[1.,2.,3.,5.], 1., 3., 3, sci_stats::diagnostics::PpccFamily::Gamma)?;
/// assert_eq!(p.shapes, vec![1.,2.,3.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy ppcc_plot](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ppcc_plot.html).
pub fn ppcc_plot(
    values: &[f64],
    lower: f64,
    upper: f64,
    count: usize,
    family: PpccFamily,
) -> Result<PpccPlot> {
    ppcc_bounds(lower, upper)?;
    search_sample(values)?;
    if !(2..=1024).contains(&count) || count.checked_mul(values.len()).is_none_or(|w| w > 65536) {
        return Err(Error::InvalidInput(
            "PPCC grid work exceeds 65536 quantiles",
        ));
    }
    let mut shapes = Vec::with_capacity(count);
    let mut correlations = Vec::with_capacity(count);
    for i in 0..count {
        let f = i as f64 / (count - 1) as f64;
        let s = (1. - f) * lower + f * upper;
        shapes.push(s);
        correlations.push(ppcc_correlation(values, family, s)?);
    }
    Ok(PpccPlot {
        shapes,
        correlations,
    })
}

/// PPCC optimal shape `argmax_(s∈[lower,upper]) corr(F_s⁻¹(p_i), sorted(x))`.
///
/// ## Parameters and conventions
///
/// `values`, `family` and plotting positions follow [`ppcc_plot`]. Bounds are
/// hard constraints; SciPy's starting bracket can expand. Only the listed
/// [`PpccFamily`] distributions are supported; there is no default Tukey-lambda.
///
/// ## Domain and special values
///
/// 3..=256 finite nonconstant observations and 0<lower<upper≤64.
/// Endpoints may be optimal. NaN and infinite inputs return errors.
///
/// ## Returns
///
/// Dimensionless distribution shape for a local correlation maximum.
///
/// ## Errors
///
/// Invalid domains/work limits, nonfinite quantiles/correlations, or failed
/// convergence return errors.
///
/// ## Numerical behavior
///
/// Private Basin Brent with at most 132 evaluations and position tolerances
/// 1e-10; each evaluation uses bounded statrs-backed Sci quantile inversions.
/// Flat correlations generally limit shape accuracy to about 1e-5. A global
/// maximum is not guaranteed.
///
/// ## Examples
///
/// ```
/// let shape = sci_stats::diagnostics::ppcc_max(&[1.,2.,3.,5.], 0.5, 8., sci_stats::diagnostics::PpccFamily::Gamma)?;
/// assert!((0.5..=8.).contains(&shape));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy ppcc_max](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ppcc_max.html).
pub fn ppcc_max(values: &[f64], lower: f64, upper: f64, family: PpccFamily) -> Result<f64> {
    ppcc_bounds(lower, upper)?;
    search_sample(values)?;
    if values.len() > 256 {
        return Err(Error::InvalidInput("PPCC search sample limit is 256"));
    }
    bounded_minimum(lower, upper, |s| Ok(-ppcc_correlation(values, family, s)?))
}
