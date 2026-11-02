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
    let eval_objective = |x| -> Result<f64> {
        let value = function(x)?;
        if value.is_finite() {
            Ok(value)
        } else {
            Err(Error::NonFiniteEvaluation)
        }
    };
    // The closed interval matters: the minimum may lie exactly at a bound.
    let left_cost = eval_objective(lower)?;
    let right_cost = eval_objective(upper)?;
    let mut best = if left_cost <= right_cost {
        (lower, left_cost)
    } else {
        (upper, right_cost)
    };

    // Bounded Brent: safeguarded inverse-parabolic interpolation, otherwise a
    // golden-section step. Keep the three best interior abscissae (x, w, v).
    const GOLDEN: f64 = 0.381_966_011_250_105_1;
    const ABS_TOL: f64 = 1e-10;
    const REL_TOL: f64 = 1e-10;
    let (mut a, mut b) = (lower, upper);
    let mut x = a + GOLDEN * (b - a);
    let mut fx = eval_objective(x)?;
    if fx < best.1 {
        best = (x, fx);
    }
    let (mut w, mut v) = (x, x);
    let (mut fw, mut fv) = (fx, fx);
    let (mut step, mut previous_step): (f64, f64) = (0., 0.);

    for _ in 0..128 {
        let midpoint = a / 2. + b / 2.;
        let tol = ABS_TOL + REL_TOL * x.abs();
        if (x - midpoint).abs() <= 2. * tol - (b - a) / 2. {
            return Ok(best.0);
        }

        let old_step = step;
        let mut parabolic = false;
        if previous_step.abs() > tol {
            let r = (x - w) * (fx - fv);
            let q0 = (x - v) * (fx - fw);
            let mut p = (x - v) * q0 - (x - w) * r;
            let mut q = 2. * (q0 - r);
            if q > 0. {
                p = -p;
            } else {
                q = -q;
            }
            if q > 0.
                && p.abs() < (0.5 * q * previous_step).abs()
                && p > q * (a - x)
                && p < q * (b - x)
            {
                step = p / q;
                parabolic = true;
            }
        }
        if !parabolic {
            previous_step = if x < midpoint { b - x } else { a - x };
            step = GOLDEN * previous_step;
        } else {
            previous_step = old_step;
        }
        let displacement = if step.abs() >= tol {
            step
        } else if step >= 0. {
            tol
        } else {
            -tol
        };
        let u = x + displacement;
        let fu = eval_objective(u)?;
        if fu < best.1 {
            best = (u, fu);
        }
        if fu <= fx {
            if u < x {
                b = x;
            } else {
                a = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }
    Err(Error::NumericalFailure(
        "bounded parameter search did not converge",
    ))
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
/// Private bounded Brent search, with absolute and relative position tolerances
/// 1e-10; at most 131 objective evaluations. Binary64 rounding and flat objectives
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
/// Uses the same private bounded Brent search and 131-evaluation limit as
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
/// Private bounded Brent with at most 131 evaluations and position tolerances
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

#[cfg(test)]
mod minimizer_tests {
    use super::*;

    #[test]
    fn bounded_brent_interior_and_closed_endpoints() {
        let interior = bounded_minimum(-2., 5., |x| Ok((x - 0.375).powi(2))).unwrap();
        assert!((interior - 0.375).abs() < 1e-7);
        assert_eq!(bounded_minimum(-2., 5., Ok).unwrap(), -2.);
        assert_eq!(bounded_minimum(-2., 5., |x| Ok(-x)).unwrap(), 5.);
        assert_eq!(bounded_minimum(0.2, 0.5, |_| Ok(1.)).unwrap(), 0.2);
    }

    #[test]
    fn bounded_brent_checks_nonfinite_objectives() {
        assert!(matches!(
            bounded_minimum(-1., 1., |x| Ok(if x > -1. && x < 1. {
                f64::NAN
            } else {
                x * x
            })),
            Err(Error::NonFiniteEvaluation)
        ));
        assert!(matches!(
            bounded_minimum(-1., 1., |_| Ok(f64::INFINITY)),
            Err(Error::NonFiniteEvaluation)
        ));
        assert!(bounded_minimum(1., 1., |_| Ok(0.)).is_err());
    }
}
