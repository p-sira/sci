use super::*;

/// Logrank survival test `Z=Σ(d_x−n_x*d/n)/sqrt(Σ V)`.
///
/// ## Parameters and conventions
///
/// `x`,`y` contain nonnegative times in common units. `x_event`,`y_event`
/// mark observed failures (true) or right-censoring (false), replacing SciPy's
/// CensoredData objects. At each unique failure time, risk sets include subjects
/// censored at that time, and `V=n_x*n_y*d*(n−d)/(n²*(n−1))` for n>1, else zero.
/// `alternative=Less` favors longer survival of X; Greater favors Y. Left and
/// interval censoring, delayed entry and weights are unsupported.
///
/// ## Domain and special values
///
/// Nonempty finite nonnegative time samples, combined count≤1048576, matching
/// event lengths. No failures or zero numerator/variance yields NaN fields.
/// A nonzero numerator with zero variance yields signed infinity and a limiting tail.
///
/// ## Returns
///
/// Dimensionless signed normal statistic and corresponding asymptotic p-value;
/// its square is the usual chi-square logrank statistic.
///
/// ## Errors
///
/// Invalid counts, negative/nonfinite times or incompatible event shapes return errors.
///
/// ## Numerical behavior
///
/// O((n+m) log(n+m)) sorting and O(n+m) memory. Risk/event counts are exact
/// within limits; normalized binary64 formulas avoid products of large counts.
/// Private statrs normal tails. Small risk sets weaken asymptotic calibration.
///
/// ## Examples
///
/// ```
/// let r = sci_stats::logrank(&[1.,2.,3.], &[true;3], &[1.,2.,3.], &[true;3], sci_stats::Alternative::TwoSided)?;
/// assert_eq!(r.statistic, 0.);
/// assert_eq!(r.pvalue, 1.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy logrank and Mantel (1966)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.logrank.html).
pub fn logrank(
    x: &[f64],
    x_event: &[bool],
    y: &[f64],
    y_event: &[bool],
    alternative: Alternative,
) -> Result<TestResult> {
    if x.len() != x_event.len() || y.len() != y_event.len() {
        return Err(Error::DimensionMismatch);
    }
    sample(x, 1)?;
    sample(y, 1)?;
    if x.len() + y.len() > MAX_SAMPLES || x.iter().chain(y).any(|v| *v < 0.) {
        return Err(Error::InvalidInput(
            "survival times must be nonnegative; combined count <=1048576",
        ));
    }
    let mut times = x
        .iter()
        .zip(x_event)
        .map(|(&t, &e)| (t, e, true))
        .chain(y.iter().zip(y_event).map(|(&t, &e)| (t, e, false)))
        .collect::<Vec<_>>();
    times.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let (mut risk_x, mut risk_y) = (x.len(), y.len());
    let (mut numerator, mut variance) = (0., 0.);
    let mut i = 0;
    while i < times.len() {
        let mut j = i;
        let (mut events_x, mut events_y, mut removed_x, mut removed_y) = (0, 0, 0, 0);
        while j < times.len() && times[j].0 == times[i].0 {
            let (_, event, first) = times[j];
            if first {
                removed_x += 1;
                events_x += usize::from(event);
            } else {
                removed_y += 1;
                events_y += usize::from(event);
            }
            j += 1;
        }
        let n = (risk_x + risk_y) as f64;
        let deaths = (events_x + events_y) as f64;
        let fraction = risk_x as f64 / n;
        numerator += events_x as f64 - fraction * deaths;
        if n > 1. {
            variance += fraction * (1. - fraction) * deaths * ((n - deaths) / (n - 1.));
        }
        risk_x -= removed_x;
        risk_y -= removed_y;
        i = j;
    }
    normality::normal_result(numerator / variance.sqrt(), alternative)
}
