use super::*;

const MAX_REGRESSION_SAMPLES: usize = 2048;

/// Intercept convention for Theil–Sen regression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TheilMethod {
    /// `median(y) − slope * median(x)`, SciPy's default.
    Separate,
    /// `median(y − slope*x)`.
    Joint,
}

/// Intercept convention for Siegel regression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SiegelMethod {
    /// `median(y − slope*x)`, SciPy's default.
    Hierarchical,
    /// Repeated median of pairwise line intercepts, independent of the final slope.
    Separate,
}

/// Robust line estimate `y = intercept + slope*x`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RobustRegression {
    /// Slope in units of y/x.
    pub slope: f64,
    /// Intercept in units of y.
    pub intercept: f64,
}

/// Theil–Sen estimate and approximate slope confidence interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TheilRegression {
    /// Median pairwise slope, in units of y/x.
    pub slope: f64,
    /// Intercept in units of y, using the specified convention.
    pub intercept: f64,
    /// Lower slope confidence bound; NaN if the tie-corrected variance is negative.
    pub low_slope: f64,
    /// Upper slope confidence bound; NaN if the tie-corrected variance is negative.
    pub high_slope: f64,
}

fn validate(x: &[f64], y: &[f64]) -> Result<()> {
    if x.len() != y.len() {
        return Err(Error::DimensionMismatch);
    }
    if x.len() > MAX_REGRESSION_SAMPLES {
        return Err(Error::InvalidInput(
            "robust regression sample limit is 2048",
        ));
    }
    sample(x, 2)?;
    sample(y, 2)?;
    if x.iter().all(|v| *v == x[0]) {
        return Err(Error::InvalidInput(
            "robust regression requires distinct x values",
        ));
    }
    Ok(())
}

fn pair_slope(xi: f64, xj: f64, yi: f64, yj: f64) -> Result<f64> {
    let (dx, dy) = (xj - xi, yj - yi);
    match if dx.is_finite() && dy.is_finite() {
        dy / dx
    } else {
        (0.5 * yj - 0.5 * yi) / (0.5 * xj - 0.5 * xi)
    } {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }
}

fn median_in_place(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    interpolate(values, 0.5)
}

fn line_intercept(x: &[f64], y: &[f64], slope: f64) -> Result<f64> {
    let mut residuals = x
        .iter()
        .zip(y)
        .map(|(a, b)| match (-slope).mul_add(*a, *b) {
            value if value.is_finite() => Ok(value),
            _ => Err(crate::Error::NonFiniteEvaluation),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(median_in_place(&mut residuals))
}

/// Theil–Sen slope `median((y_j−y_i)/(x_j−x_i))`, omitting equal-x pairs.
///
/// ## Parameters and conventions
///
/// `x` and `y` are explicit paired coordinates, in Sci's x-then-y order.
/// `method` selects the intercept definition. `confidence` lies in `(0,1)`;
/// SciPy's symmetric convention treats c and 1−c as the same confidence level.
/// Unlike SciPy's optional x, index coordinates must be supplied explicitly.
/// Slope limits use Sen's rank approximation: for N usable slopes,
/// `V=[n(n−1)(2n+5)−Σt_x(t_x−1)(2t_x+5)−Σt_y(t_y−1)(2t_y+5)]/18`,
/// `z=Φ⁻¹(min(c,1−c)/2)`, with zero-based sorted-slope indices
/// `L=max(round_even((N+z sqrt(V))/2)−1,0)` and
/// `U=min(round_even((N−z sqrt(V))/2),N−1)`.
///
/// ## Domain and special values
///
/// Requires 2..=2048 finite pairs and at least two distinct x coordinates.
/// Repeated x values are omitted only from pair slopes; all observations
/// contribute to the intercept and tie correction. No masks or NaN omission.
/// A negative V gives NaN confidence bounds, as in SciPy; estimates remain finite.
///
/// ## Returns
///
/// Slope, intercept, lower slope bound, and upper slope bound in named fields.
/// There is no intercept confidence interval.
///
/// ## Errors
///
/// Invalid confidence/counts, unequal lengths, nonfinite observations,
/// constant x, or unrepresentable slopes/intercepts return errors. Any unusable
/// arithmetic pair is rejected rather than silently excluded.
///
/// ## Numerical behavior
///
/// Native pair enumeration and sorting take O(n² log n) work and O(n²) memory,
/// bounded to 2,096,128 slopes. Half-scaled differences avoid opposite-extreme
/// subtraction overflow; fused multiply-add reduces intercept cancellation.
/// Confidence limits reuse statrs through Sci's normal quantile API. Calibration
/// is approximate and can be unreliable with heavy ties or very small samples.
///
/// ## Examples
///
/// ```
/// use sci_stats::{regression::theilslopes, regression::TheilMethod};
/// let fit = theilslopes(&[0.,1.,2.], &[1.,3.,5.], 0.95, TheilMethod::Separate)?;
/// assert_eq!((fit.slope, fit.intercept), (2., 1.));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy theilslopes and Sen (1968), equation 2.6](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.theilslopes.html).
pub fn theilslopes(
    x: &[f64],
    y: &[f64],
    confidence: f64,
    method: TheilMethod,
) -> Result<TheilRegression> {
    if !confidence.is_finite() || confidence <= 0. || confidence >= 1. {
        return Err(Error::InvalidInput("confidence must be in (0,1)"));
    }
    validate(x, y)?;
    let mut slopes = Vec::with_capacity(x.len() * (x.len() - 1) / 2);
    for i in 0..x.len() {
        for j in i + 1..x.len() {
            if x[i] != x[j] {
                slopes.push(pair_slope(x[i], x[j], y[i], y[j])?);
            }
        }
    }
    let slope = median_in_place(&mut slopes);
    let intercept = match method {
        TheilMethod::Joint => line_intercept(x, y, slope)?,
        TheilMethod::Separate => match (-slope).mul_add(median(x)?, median(y)?) {
            value if value.is_finite() => Ok(value),
            _ => Err(crate::Error::NonFiniteEvaluation),
        }?,
    };
    let n = x.len() as f64;
    let variance = (n * (n - 1.) * (2. * n + 5.)
        - ranking::tie_counts(x).variance_term
        - ranking::tie_counts(y).variance_term)
        / 18.;
    let (low_slope, high_slope) = if variance < 0. {
        (f64::NAN, f64::NAN)
    } else {
        let z = normal_ppf(confidence.min(1. - confidence) / 2., 0., 1.)?;
        let nslopes = slopes.len() as f64;
        let lower = (((nslopes + z * variance.sqrt()) / 2.).round_ties_even() - 1.)
            .clamp(0., nslopes - 1.) as usize;
        let upper = ((nslopes - z * variance.sqrt()) / 2.)
            .round_ties_even()
            .clamp(0., nslopes - 1.) as usize;
        (slopes[lower], slopes[upper])
    };
    Ok(TheilRegression {
        slope,
        intercept,
        low_slope,
        high_slope,
    })
}

/// Siegel slope `median_i(median_{j:x_j≠x_i}((y_j−y_i)/(x_j−x_i)))`.
///
/// ## Parameters and conventions
///
/// Explicit paired x-then-y coordinates. `Hierarchical` uses
/// `median(y−slope*x)`. `Separate` uses the repeated median of pairwise
/// intercepts `y_i−s_ij*x_i`. Index coordinates must be supplied explicitly;
/// SciPy axis, masked-array, and missing-data options are unsupported.
///
/// ## Domain and special values
///
/// Requires 2..=2048 finite pairs with at least two distinct x coordinates.
/// Equal-x pairs are omitted from each point's medians. Constant y is allowed.
///
/// ## Returns
///
/// Named slope in units of y/x and intercept in units of y; no confidence limits.
///
/// ## Errors
///
/// Invalid counts, mismatched lengths, nonfinite inputs, constant x, or
/// unrepresentable pair slopes/intercepts return errors.
///
/// ## Numerical behavior
///
/// Native repeated medians take O(n² log n) work and O(n) memory, with at most
/// 4,192,256 ordered pair evaluations. Difference scaling and fused multiply-add
/// reduce overflow and cancellation; binary64 rounding remains. No external
/// backend is used. The estimator has asymptotic breakdown point 50%.
///
/// ## Examples
///
/// ```
/// use sci_stats::{regression::siegelslopes, regression::SiegelMethod};
/// let fit = siegelslopes(&[0.,1.,2.], &[1.,3.,5.], SiegelMethod::Hierarchical)?;
/// assert_eq!((fit.slope, fit.intercept), (2., 1.));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy siegelslopes and Siegel (1982)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.siegelslopes.html).
pub fn siegelslopes(x: &[f64], y: &[f64], method: SiegelMethod) -> Result<RobustRegression> {
    validate(x, y)?;
    let mut slopes = Vec::with_capacity(x.len());
    let mut intercepts = Vec::with_capacity(x.len());
    let mut pairs = Vec::with_capacity(x.len() - 1);
    let mut pair_intercepts = Vec::with_capacity(x.len() - 1);
    for i in 0..x.len() {
        pairs.clear();
        pair_intercepts.clear();
        for j in 0..x.len() {
            if x[i] == x[j] {
                continue;
            }
            let slope = pair_slope(x[i], x[j], y[i], y[j])?;
            pairs.push(slope);
            if method == SiegelMethod::Separate {
                pair_intercepts.push(match (-slope).mul_add(x[i], y[i]) {
                    value if value.is_finite() => Ok(value),
                    _ => Err(crate::Error::NonFiniteEvaluation),
                }?);
            }
        }
        slopes.push(median_in_place(&mut pairs));
        if method == SiegelMethod::Separate {
            intercepts.push(median_in_place(&mut pair_intercepts));
        }
    }
    let slope = median_in_place(&mut slopes);
    let intercept = match method {
        SiegelMethod::Hierarchical => line_intercept(x, y, slope)?,
        SiegelMethod::Separate => median_in_place(&mut intercepts),
    };
    Ok(RobustRegression { slope, intercept })
}
