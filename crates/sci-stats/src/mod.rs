use crate::{
    Error, Result,
    error::{finite, finite_slice},
};
use statrs::distribution::{Continuous, ContinuousCDF, Normal};

/// Arithmetic mean `μ=Σx_i/n`.
///
/// ## Parameters
///
/// A finite one-dimensional observation slice.
///
/// ## Domain
///
/// Nonempty finite observations; NaN and infinities are errors.
///
/// ## Returns
///
/// Mean in the same units as observations.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Sign-balanced Neumaier summation preserves small cancellation residuals.
/// A scaled fallback avoids overflow for large uncancelled sums. Binary64
/// rounding and subnormal underflow remain. At most 1048576 observations.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::mean(&[1., 2., 3.])?, 2.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [NumPy `mean`](https://numpy.org/doc/stable/reference/generated/numpy.mean.html).
pub fn mean(values: &[f64]) -> Result<f64> {
    sample(values, 1)?;
    // Consume an opposite-sign term whenever possible, keeping large terms
    // from overflowing before they cancel. Divide only after cancellation.
    let mut positive = values.iter().copied().filter(|x| *x >= 0.0);
    let mut negative = values.iter().copied().filter(|x| *x < 0.0);
    let (mut sum, mut correction) = (0_f64, 0_f64);
    loop {
        let term = if sum >= 0.0 {
            negative.next().or_else(|| positive.next())
        } else {
            positive.next().or_else(|| negative.next())
        };
        let Some(term) = term else {
            let total = sum + correction;
            return finite(if total.is_finite() {
                total / values.len() as f64
            } else {
                sum / values.len() as f64 + correction / values.len() as f64
            });
        };
        let next = sum + term;
        if !next.is_finite() {
            break;
        }
        correction += if sum.abs() >= term.abs() {
            (sum - next) + term
        } else {
            (term - next) + sum
        };
        sum = next;
    }
    // Overflow here requires a large uncancelled sum; normalization cannot
    // discard a cancellation residual that determines the final mean.
    let scale = values.iter().fold(0_f64, |m, x| m.max(x.abs()));
    if scale == 0. {
        return Ok(0.);
    }
    let n = values.len() as f64;
    // Neumaier compensation preserves small terms during cancellation.
    let (mut sum, mut correction) = (0_f64, 0_f64);
    for x in values {
        let term = (x / scale) / n;
        let next = sum + term;
        correction += if sum.abs() >= term.abs() {
            (sum - next) + term
        } else {
            (term - next) + sum
        };
        sum = next;
    }
    finite((sum + correction).clamp(-1., 1.) * scale)
}

/// Variance `Σ(x_i−μ)²/(n−ddof)`.
///
/// ## Parameters
///
/// Explicit ddof: zero gives population variance, one gives unbiased sample variance.
///
/// ## Domain
///
/// Nonempty finite observations and ddof<n; constant data returns zero.
///
/// ## Returns
///
/// Variance in squared observation units.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Observations are shifted before scaling to preserve small differences at
/// large offsets. Scaling first is used only when subtraction overflows.
/// A variance above the finite f64 range is an evaluation error; underflow
/// can return zero. At most 1048576 observations.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::variance(&[1., 2., 3.], 1)?, 1.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [NumPy `variance`](https://numpy.org/doc/stable/reference/generated/numpy.var.html).
pub fn variance(values: &[f64], ddof: usize) -> Result<f64> {
    if ddof >= values.len() {
        return Err(Error::InvalidInput(
            "ddof must be smaller than the observation count",
        ));
    }
    let (scale, normalized) = scaled_variance(values, ddof)?;
    finite((normalized * scale) * scale)
}

fn scaled_variance(values: &[f64], ddof: usize) -> Result<(f64, f64)> {
    sample(values, 1)?;
    if ddof >= values.len() {
        return Err(Error::InvalidInput(
            "ddof must be smaller than the observation count",
        ));
    }
    let (scale, deviations) = centered_scaled(values)?;
    Ok((scale, normalized_variance(&deviations, ddof)))
}

fn normalized_variance(deviations: &[f64], ddof: usize) -> f64 {
    deviations
        .iter()
        .map(|x| x.powi(2) / (deviations.len() - ddof) as f64)
        .sum()
}

struct CenteredSample {
    origin: f64,
    scale: f64,
    center: f64,
    deviations: Vec<f64>,
}

// Returns centered deviations divided by a finite scale. Shift first when
// possible: dividing large, nearly equal observations loses their differences.
fn centered_scaled(values: &[f64]) -> Result<(f64, Vec<f64>)> {
    let sample = centered_sample(values)?;
    Ok((sample.scale, sample.deviations))
}

fn centered_sample(values: &[f64]) -> Result<CenteredSample> {
    sample(values, 1)?;
    let mut origin = values[0];
    let mut shifted: Vec<_> = values.iter().map(|x| x - origin).collect();
    let mut scale = shifted.iter().fold(0_f64, |s, x| s.max(x.abs()));
    if !scale.is_finite() {
        origin = 0.;
        scale = values.iter().fold(0_f64, |s, x| s.max(x.abs()));
        for (y, x) in shifted.iter_mut().zip(values) {
            *y = x / scale;
        }
    } else if scale == 0.0 {
        return Ok(CenteredSample {
            origin,
            scale,
            center: 0.,
            deviations: shifted,
        });
    } else {
        for y in &mut shifted {
            *y /= scale;
        }
    }
    let center = mean(&shifted)?;
    for y in &mut shifted {
        *y -= center;
    }
    Ok(CenteredSample {
        origin,
        scale,
        center,
        deviations: shifted,
    })
}

fn normal(mean: f64, std_dev: f64) -> Result<Normal> {
    if !mean.is_finite() || !std_dev.is_finite() || std_dev <= 0.0 {
        return Err(Error::InvalidInput(
            "normal mean must be finite and standard deviation finite and positive",
        ));
    }
    Normal::new(mean, std_dev)
        .map_err(|_| Error::InvalidInput("invalid normal distribution parameters"))
}

/// Normal density `exp(−(x−μ)²/(2σ²))/(σ√(2π))`.
///
/// ## Parameters
///
/// Scalar x, finite mean μ, positive finite standard deviation σ (not variance).
///
/// ## Domain
///
/// Infinite x returns zero; NaN x and invalid parameters are errors.
///
/// ## Returns
///
/// Nonnegative density in inverse observation units.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Standardizes before private statrs evaluation, then divides by the scale.
/// Log-space fallback preserves representable densities when a standardized
/// tail underflows. True density overflow is an evaluation error; true underflow
/// returns zero. Rounded subtraction and backend tail accuracy remain limitations.
///
/// # Examples
///
/// ```
/// assert!((sci_stats::normal_pdf(0., 0., 1.)? - 0.3989422804014327).abs() < 1e-15);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [NIST normal distribution](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3661.htm).
pub fn normal_pdf(x: f64, mean: f64, std_dev: f64) -> Result<f64> {
    distributions::evaluate(
        Distribution::Normal { mean, std_dev },
        x,
        distributions::Operation::Density,
    )
}

/// Normal CDF `P(X≤x)=erfc((μ−x)/(σ√2))/2`.
///
/// ## Parameters
///
/// Scalar x, finite mean μ, positive finite standard deviation σ.
///
/// ## Domain
///
/// Infinite x returns zero or one; NaN x and invalid parameters are errors.
///
/// ## Returns
///
/// Dimensionless probability between zero and one.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Standardizes before the shared [`sci_special::ndtr`] evaluation; an
/// overflowing finite coordinate difference is divided before subtraction.
/// Infinite observations retain their limiting probabilities at every valid scale.
/// Extreme lower tails can underflow to zero; near-one probabilities can round to one.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::normal_cdf(0., 0., 1.)?, 0.5);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [NIST normal distribution](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3661.htm).
pub fn normal_cdf(x: f64, mean: f64, std_dev: f64) -> Result<f64> {
    distributions::evaluate(
        Distribution::Normal { mean, std_dev },
        x,
        distributions::Operation::Cdf,
    )
}

/// Normal quantile `q=μ+σ*Φ⁻¹(p)`, satisfying `P(X≤q)=p`.
///
/// ## Parameters
///
/// Probability p in \[0,1\], finite mean μ, positive finite standard deviation σ.
///
/// ## Domain
///
/// p=0 and p=1 return negative and positive infinity; NaN p and invalid parameters are errors.
///
/// ## Returns
///
/// Quantile in observation units.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Uses the shared [`sci_special::ndtri`] standard-normal inverse, followed by fused scale and
/// location arithmetic. This avoids overflowing intermediate scale products,
/// including when the product and location cancel. Interior quantiles outside
/// the finite range return an evaluation error; extreme probabilities can lose accuracy.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::normal_ppf(0.5, 2., 3.)?, 2.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [NIST normal distribution](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3661.htm).
pub fn normal_ppf(probability: f64, mean: f64, std_dev: f64) -> Result<f64> {
    distributions::evaluate(
        Distribution::Normal { mean, std_dev },
        probability,
        distributions::Operation::Ppf,
    )
}

mod descriptive;
mod distributions;
mod inference;
mod ranking;
mod transformations;
pub use descriptive::*;
pub use distributions::*;
pub use inference::*;
pub use ranking::*;
pub use transformations::*;

const MAX_SAMPLES: usize = 1_048_576;
fn sample(values: &[f64], minimum: usize) -> Result<()> {
    if values.len() < minimum || values.len() > MAX_SAMPLES {
        return Err(Error::InvalidInput(
            "invalid observation count (maximum 1048576)",
        ));
    }
    finite_slice(values)
}
fn positive(x: f64) -> Result<()> {
    if x.is_finite() && x > 0. {
        Ok(())
    } else {
        Err(Error::InvalidInput("parameter must be finite and positive"))
    }
}
fn probability_check(x: f64) -> Result<()> {
    if (0. ..=1.).contains(&x) {
        Ok(())
    } else {
        Err(Error::InvalidInput("probability must be in [0,1]"))
    }
}
fn sorted(x: &[f64]) -> Result<Vec<f64>> {
    sample(x, 1)?;
    let mut out = x.to_vec();
    out.sort_by(f64::total_cmp);
    Ok(out)
}
fn interpolate(x: &[f64], p: f64) -> f64 {
    let h = (x.len() - 1) as f64 * p;
    let i = h.floor() as usize;
    let j = h.ceil() as usize;
    if i == j {
        x[i]
    } else {
        let f = h - i as f64;
        (1. - f) * x[i] + f * x[j]
    }
}

/// Two-dimensional categorical-data statistics.
pub mod contingency;
pub use contingency::chi2_contingency;

/// Quasi-Monte Carlo discrepancy and coordinate scaling.
pub mod qmc;

mod extras;
pub use extras::*;

mod multivariate;
pub use multivariate::*;

mod utilities;
pub use utilities::*;

mod binning;
pub use binning::*;

mod ks;
pub use ks::*;

mod normality;
pub use normality::*;
mod robust;
pub use robust::*;
mod rank_tests;
pub use rank_tests::*;
mod diagnostics;
pub use diagnostics::*;
mod estimation;
pub use estimation::*;
mod association;
pub use association::*;
mod survival;
pub use survival::*;
mod characteristic;
pub use characteristic::*;
mod goodness;
pub use goodness::*;
mod resampling;
pub use resampling::*;
mod fitting;
pub use fitting::*;
mod sensitivity;
pub use sensitivity::*;
mod multiple;
pub use multiple::*;
mod composition;
pub use composition::*;
mod unconditional;
pub use unconditional::*;
mod transport;
pub use transport::*;
mod dunnett;
pub use dunnett::*;
mod multiscale;
pub use multiscale::*;

fn continuous_null(reference: Distribution) -> Result<()> {
    if matches!(
        reference,
        Distribution::DiscreteUniform { .. }
            | Distribution::Bernoulli { .. }
            | Distribution::Binomial { .. }
            | Distribution::Poisson { .. }
            | Distribution::Geometric { .. }
            | Distribution::NegativeBinomial { .. }
            | Distribution::Hypergeometric { .. }
    ) {
        return Err(Error::InvalidInput("null distribution must be continuous"));
    }
    cdf(reference, 0.)?;
    Ok(())
}
