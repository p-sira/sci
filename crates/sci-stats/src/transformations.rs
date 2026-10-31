use super::*;
use std::f64::consts::TAU;

fn circular(values: &[f64], low: f64, high: f64) -> Result<(f64, f64)> {
    sample(values, 1)?;
    if !low.is_finite() {
        return Err(crate::Error::NonFiniteEvaluation);
    }
    if !high.is_finite() {
        return Err(crate::Error::NonFiniteEvaluation);
    }
    positive(high - low)?;
    let (mut c, mut s) = (0., 0.);
    let n = values.len() as f64;
    for &x in values {
        let phase = match ((x - low) / (high - low)) * TAU {
            value if value.is_finite() => Ok(value),
            _ => Err(crate::Error::NonFiniteEvaluation),
        }?;
        c += phase.cos() / n;
        s += phase.sin() / n;
    }
    Ok((c, s))
}
fn transform(values: &[f64], f: impl Fn(f64) -> f64) -> Result<Vec<f64>> {
    let out = values.iter().map(|&x| f(x)).collect::<Vec<_>>();
    if out.iter().any(|x| !x.is_finite()) {
        Err(Error::NonFiniteEvaluation)
    } else {
        Ok(out)
    }
}
/// Circular mean `low+(high−low)*arg(mean(exp(i*θ)))/(2π)`, θ=2π(x−low)/(high−low).
///
/// ## Parameters
///
/// Angles use the units of [low,high); high−low is one full rotation.
///
/// ## Domain
///
/// Nonempty finite observations and finite low<high. Zero resultant has an implementation-dependent mean, as in SciPy.
///
/// ## Returns
///
/// Principal angle in [low,high), in the input angle units.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Direct binary64 formula; overflow is reported explicitly.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!(sci_stats::circmean(&[0., 0.], 0., std::f64::consts::TAU)?.abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `circmean`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.circmean.html).
pub fn circmean(values: &[f64], low: f64, high: f64) -> Result<f64> {
    let (c, s) = circular(values, low, high)?;
    match low + (high - low) * (s.atan2(c).rem_euclid(TAU) / TAU) {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }
}
/// Circular variance `1−|mean(exp(i*θ))|`.
///
/// ## Parameters
///
/// θ=2π(x−low)/(high−low); dimensionless, independent of angle units.
///
/// ## Domain
///
/// Nonempty finite angles and finite low<high.
///
/// ## Returns
///
/// Dimensionless circular variance in \[0,1\].
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Direct binary64 formula; overflow is reported explicitly.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::circvar(&[1., 1.], 0., std::f64::consts::TAU)?, 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `circvar`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.circvar.html).
pub fn circvar(values: &[f64], low: f64, high: f64) -> Result<f64> {
    let (c, s) = circular(values, low, high)?;
    Ok(1. - c.hypot(s).min(1.))
}
/// Circular standard deviation `sqrt(−2 ln(R))`, R=|mean(exp(i*θ))|.
///
/// ## Parameters
///
/// If normalize=false, multiplies by (high−low)/(2π) to return input units; true returns radians.
///
/// ## Domain
///
/// Nonempty finite angles; low<high. Exactly zero resultant returns positive infinity.
///
/// ## Returns
///
/// Circular standard deviation.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Direct binary64 formula; overflow is reported explicitly.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::circstd(&[0., 0.], 0., std::f64::consts::TAU, false)?, 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `circstd`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.circstd.html).
pub fn circstd(values: &[f64], low: f64, high: f64, normalize: bool) -> Result<f64> {
    let (c, s) = circular(values, low, high)?;
    let sd = (-2. * c.hypot(s).min(1.).ln()).sqrt();
    Ok(if normalize {
        sd
    } else {
        sd * ((high - low) / TAU)
    })
}
/// Shannon entropy `−Σp ln(p)` or relative entropy `Σp ln(p/q)`.
///
/// ## Parameters
///
/// Normalizes each nonnegative weight vector to sum one. None base uses natural logs (nats); base=2 gives bits.
///
/// ## Domain
///
/// Nonempty finite nonnegative weights with positive sums. q must have matching length. Base must be positive and unequal to one. p>0,q=0 gives infinity; zero p contributes zero.
///
/// ## Returns
///
/// Dimensionless entropy or Kullback–Leibler divergence in the selected logarithm units.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Weights are scaled before normalization, avoiding overflow in their sum; log ratios avoid intermediate overflow.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::entropy(&[1., 1.], None, Some(2.))?, 1.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `entropy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html).
pub fn entropy(pk: &[f64], qk: Option<&[f64]>, base: Option<f64>) -> Result<f64> {
    let p = probabilities(pk)?;
    let q = match qk {
        Some(q) => {
            if q.len() != p.len() {
                return Err(Error::DimensionMismatch);
            }
            Some(probabilities(q)?)
        }
        None => None,
    };
    let denominator = if let Some(b) = base {
        positive(b)?;
        if b == 1. {
            return Err(Error::InvalidInput("logarithm base must not equal one"));
        }
        b.ln()
    } else {
        1.
    };
    let mut out = 0.;
    for (i, &p) in p.iter().enumerate() {
        if p > 0. {
            out += if let Some(q) = &q {
                p * (p.ln() - q[i].ln())
            } else {
                -p * p.ln()
            };
        }
    }
    Ok(out / denominator)
}

pub(super) fn probabilities(values: &[f64]) -> Result<Vec<f64>> {
    sample(values, 1)?;
    if values.iter().any(|&x| x < 0.) {
        return Err(Error::InvalidInput("weights must be nonnegative"));
    }
    let maximum = values.iter().copied().fold(0., f64::max);
    positive(maximum)?;
    let sum = values.iter().map(|x| x / maximum).sum::<f64>();
    Ok(values.iter().map(|x| (x / maximum) / sum).collect())
}
/// Box–Cox power transform `y=(x^λ−1)/λ`, or `ln(x)` when λ=0.
///
/// ## Parameters
///
/// Lambda is explicitly supplied. Maximum-likelihood lambda selection and confidence intervals are unsupported.
///
/// ## Domain
///
/// Strictly positive finite inputs and finite lambda; empty input is allowed.
///
/// ## Returns
///
/// Transformed observations in original order.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Reuses the shared Sci special-function Box–Cox kernel, with a logarithmic
/// limit for tiny lambda and a scaled exponential for large powers. This avoids
/// premature underflow and intermediate overflow when the result is finite.
/// True result overflow is reported explicitly; binary64 rounding remains.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::boxcox(&[1., 2.], 1.)?, vec![0., 1.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `boxcox`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html).
pub fn boxcox(values: &[f64], lambda: f64) -> Result<Vec<f64>> {
    sample(values, 0)?;
    if !lambda.is_finite() {
        return Err(crate::Error::NonFiniteEvaluation);
    }
    if values.iter().any(|&x| x <= 0.) {
        return Err(Error::InvalidInput(
            "Box-Cox requires positive observations",
        ));
    }
    transform(values, |x| sci_special::boxcox(x, lambda))
}
/// Yeo–Johnson transform: Box–Cox of x+1 for x≥0; minus Box–Cox of 1−x with exponent 2−λ for x<0.
///
/// ## Parameters
///
/// Lambda is supplied explicitly; transform uses log1p and expm1. No optimizer overload.
///
/// ## Domain
///
/// Finite observations and lambda; empty input allowed.
///
/// ## Returns
///
/// Transformed observations in original order.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Reuses the shared Sci special-function Box–Cox kernels, including their
/// logarithmic limits and scaled exponentials. Tiny parameters and subnormal
/// observations avoid premature intermediate underflow. True result overflow
/// is reported explicitly; binary64 rounding and underflow remain.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::yeojohnson(&[-1., 0., 1.], 1.)?, vec![-1., 0., 1.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `yeojohnson`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.yeojohnson.html).
pub fn yeojohnson(values: &[f64], lambda: f64) -> Result<Vec<f64>> {
    sample(values, 0)?;
    if !lambda.is_finite() {
        return Err(crate::Error::NonFiniteEvaluation);
    }
    if lambda == 0.5 {
        // For this common half-power, eliminate log1p/expm1 per observation.
        // Rationalize the differences of square roots (and of the 3/2 power)
        // so both branches still preserve tiny values near zero.
        return transform(values, |x| {
            if x >= 0.0 {
                2.0 * (x / ((1.0 + x).sqrt() + 1.0))
            } else {
                let u = -x;
                let root = (1.0 + u).sqrt();
                (2.0 / 3.0) * x * ((u + root + 2.0) / (root + 1.0))
            }
        });
    }
    transform(values, |x| {
        if x >= 0. {
            sci_special::boxcox1p(x, lambda)
        } else {
            -sci_special::boxcox1p(-x, 2. - lambda)
        }
    })
}
/// Box–Cox profile log likelihood `(λ−1)Σln(x)−n/2*ln(var(y,0))`.
///
/// ## Parameters
///
/// Omits additive constants; y is boxcox(x,lambda).
///
/// ## Domain
///
/// Nonempty strictly positive finite observations and finite lambda. Constant transformed data returns positive infinity.
///
/// ## Returns
///
/// Scalar profile log likelihood.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Direct transformed variance; extreme exponents can overflow rather than using SciPy's extended log-variance machinery.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!(sci_stats::boxcox_llf(0., &[1., 2., 3.])?.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `boxcox_llf`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox_llf.html).
pub fn boxcox_llf(lambda: f64, values: &[f64]) -> Result<f64> {
    sample(values, 1)?;
    let y = boxcox(values, lambda)?;
    let logsum = values.iter().map(|x| x.ln()).sum::<f64>();
    let v = variance(&y, 0)?;
    let result = (lambda - 1.) * logsum - values.len() as f64 / 2. * v.ln();
    if result.is_nan() {
        Err(Error::NonFiniteEvaluation)
    } else {
        Ok(result)
    }
}
/// Yeo–Johnson profile likelihood `−n/2 ln(var(y,0))+(λ−1)Σsign(x)ln(1+|x|)`.
///
/// ## Parameters
///
/// Omits additive constants; y is yeojohnson(x,lambda).
///
/// ## Domain
///
/// Nonempty finite observations and finite lambda. Constant transformed data returns positive infinity.
///
/// ## Returns
///
/// Scalar profile log likelihood.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Direct binary64 formula; overflow is reported explicitly.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!(sci_stats::yeojohnson_llf(1., &[-1., 0., 1.])?.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `yeojohnson_llf`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.yeojohnson_llf.html).
pub fn yeojohnson_llf(lambda: f64, values: &[f64]) -> Result<f64> {
    sample(values, 1)?;
    let y = yeojohnson(values, lambda)?;
    let jacobian = values
        .iter()
        .map(|x| x.signum() * x.abs().ln_1p())
        .sum::<f64>();
    let result = -(values.len() as f64) / 2. * variance(&y, 0)?.ln() + (lambda - 1.) * jacobian;
    if result.is_nan() {
        Err(Error::NonFiniteEvaluation)
    } else {
        Ok(result)
    }
}
/// Power mean `(mean(x^p))^(1/p)`; p=0 gives geometric mean.
///
/// ## Parameters
///
/// Unweighted positive observations; finite p is limited to absolute value≤64.
///
/// ## Domain
///
/// Nonempty strictly positive finite data. Zero power uses gmean.
///
/// ## Returns
///
/// Generalized mean in observation units.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Log-sum-exp avoids power overflow and reuses a single scratch vector.
/// For |p| < 1e-4, centered logarithms with expm1/log1p preserve the correction
/// to the geometric mean. For |p| < 1e-150, that correction is below binary64
/// precision over the supported positive finite domain, so the geometric limit
/// is used. Constant data return their stored value exactly. Linear work and
/// storage; binary64 rounding and subnormal underflow remain.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!((sci_stats::pmean(&[1., 3.], 1.)? - 2.).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `pmean`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pmean.html).
pub fn pmean(values: &[f64], power: f64) -> Result<f64> {
    sample(values, 1)?;
    if !power.is_finite() {
        return Err(crate::Error::NonFiniteEvaluation);
    }
    if power.abs() > 64. || values.iter().any(|&x| x <= 0.) {
        return Err(Error::InvalidInput(
            "power mean needs positive values and |power|<=64",
        ));
    }
    if values.iter().all(|&x| x == values[0]) {
        return Ok(values[0]);
    }
    if power == 0. {
        return gmean(values);
    }
    if power.abs() < 1e-4 {
        let mut logs: Vec<_> = values.iter().map(|x| x.ln()).collect();
        let center = mean(&logs)?;
        // The largest binary64 log spread is below 1455. At this power the
        // correction to the geometric mean is far below one output ulp, and
        // forming power * log(x) can underflow.
        if power.abs() < 1e-150 {
            return match center.exp() {
                value if value.is_finite() => Ok(value),
                _ => Err(crate::Error::NonFiniteEvaluation),
            };
        }
        // Centering keeps the exponent below 0.15 in magnitude. expm1/log1p
        // retain the small correction that exp/log would round away near p=0.
        for log in &mut logs {
            *log = (power * (*log - center)).exp_m1();
        }
        return match (center + mean(&logs)?.ln_1p() / power).exp() {
            value if value.is_finite() => Ok(value),
            _ => Err(crate::Error::NonFiniteEvaluation),
        };
    }
    let mut logs: Vec<_> = values.iter().map(|x| power * x.ln()).collect();
    let m = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for log in &mut logs {
        *log = (*log - m).exp();
    }
    match ((m + mean(&logs)?.ln()) / power).exp() {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }
}
/// Expectile minimizes `Σ |alpha−I(x_i≤t)|*(x_i−t)²`.
///
/// ## Parameters
///
/// Unweighted sample; alpha=0.5 is the mean; endpoints return minimum and maximum.
///
/// ## Domain
///
/// Nonempty finite data and alpha in \[0,1\].
///
/// ## Returns
///
/// Scalar asymmetric least-squares location in observation units.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Sorting and binary search locate the interval containing the root in O(n log n)
/// work and O(n) memory, with at most 21 O(n) balance evaluations. Shifted/scaled
/// differences avoid overflow; a weighted mean of the two groups gives the root
/// without an iterative accuracy cutoff. At alpha=0.5 the shared compensated mean
/// is used directly. Tiny weights, group-mean rounding and underflow limit accuracy.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!((sci_stats::expectile(&[1., 2., 6.], 0.5)? - 3.).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `expectile`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expectile.html).
pub fn expectile(values: &[f64], alpha: f64) -> Result<f64> {
    sample(values, 1)?;
    probability_check(alpha)?;
    if alpha == 0.5 {
        return mean(values);
    }
    let ordered = sorted(values)?;
    let (first, last) = (ordered[0], ordered[ordered.len() - 1]);
    if alpha == 0. || first == last {
        return Ok(first);
    }
    if alpha == 1. {
        return Ok(last);
    }
    let width = last - first;
    let scale = if width.is_finite() {
        width
    } else {
        first.abs().max(last.abs())
    };
    if scale == 0. {
        return Ok(first);
    }
    let y = ordered
        .iter()
        .map(|x| {
            if width.is_finite() {
                (x - first) / scale
            } else {
                x / scale
            }
        })
        .collect::<Vec<_>>();
    let (mut lo, mut hi) = (0, y.len());
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let left = y[..mid].iter().map(|x| y[mid] - x).sum::<f64>();
        let right = y[mid..].iter().map(|x| x - y[mid]).sum::<f64>();
        if alpha * right > (1. - alpha) * left {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    // At a support endpoint, normalized gaps can round away. Keep the result
    // in the observed support rather than passing an empty slice to mean.
    if lo == 0 {
        return Ok(first);
    }
    if lo == y.len() {
        return Ok(last);
    }
    let left = mean(&ordered[..lo])?;
    let right = mean(&ordered[lo..])?;
    let wl = (1. - alpha) * lo as f64;
    let wr = alpha * (y.len() - lo) as f64;
    let (base, other, small) = if wr <= wl {
        (left, right, wr)
    } else {
        (right, left, wl)
    };
    let total = wl + wr;
    let fraction = small / total;
    let delta = other - base;
    match if !fraction.is_normal() {
        // Apply tiny masses before dividing by the group count; the fraction
        // itself may underflow while the physical displacement is representable.
        base + if delta.is_finite() {
            (small * delta) / total
        } else {
            (small * other - small * base) / total
        }
    } else if delta.is_finite() {
        delta.mul_add(fraction, base)
    } else {
        (1. - fraction) * base + fraction * other
    } {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }
}
/// One-dimensional Wasserstein-1 distance `∫|F_u(t)−F_v(t)|dt`.
///
/// ## Parameters
///
/// Unweighted empirical distributions; supports different sample sizes.
///
/// ## Domain
///
/// Nonempty finite samples; combined size≤1048576.
///
/// ## Returns
///
/// Nonnegative distance in observation units.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Sorts and merges CDF breakpoints in O((n+m) log(n+m)) work; no quadratic pairwise distances.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::wasserstein_distance(&[0., 1.], &[1., 2.])?, 1.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `wasserstein_distance`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html).
pub fn wasserstein_distance(first: &[f64], second: &[f64]) -> Result<f64> {
    empirical_distance(first, second, false)
}
/// One-dimensional energy distance `sqrt(2∫(F_u(t)−F_v(t))²dt)`.
///
/// ## Parameters
///
/// Unweighted empirical distributions; equivalent to sqrt(2E|U−V|−E|U−U′|−E|V−V′|).
///
/// ## Domain
///
/// Nonempty finite samples; combined size≤1048576.
///
/// ## Returns
///
/// Nonnegative distance in the square root of observation units.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Sorts CDF breakpoints; computes the integral before taking its square root.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::energy_distance(&[0.], &[2.])?, 2.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `energy_distance`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.energy_distance.html).
pub fn energy_distance(first: &[f64], second: &[f64]) -> Result<f64> {
    empirical_distance(first, second, true)
}

fn empirical_distance(first: &[f64], second: &[f64], energy: bool) -> Result<f64> {
    sample(first, 1)?;
    sample(second, 1)?;
    if first.len() + second.len() > MAX_SAMPLES {
        return Err(Error::InvalidInput("combined sample count exceeds limit"));
    }
    let a = sorted(first)?;
    let b = sorted(second)?;
    let mut knots = a.iter().chain(&b).copied().collect::<Vec<_>>();
    knots.sort_by(f64::total_cmp);
    knots.dedup();
    let (mut i, mut j) = (0, 0);
    let mut integral = 0.;
    for window in knots.windows(2) {
        while i < a.len() && a[i] <= window[0] {
            i += 1;
        }
        while j < b.len() && b[j] <= window[0] {
            j += 1;
        }
        let d = (i as f64 / a.len() as f64 - j as f64 / b.len() as f64).abs();
        integral += match window[1] - window[0] {
            value if value.is_finite() => Ok(value),
            _ => Err(crate::Error::NonFiniteEvaluation),
        }? * if energy { d * d } else { d };
    }
    match if energy {
        (2. * integral).sqrt()
    } else {
        integral
    } {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }
}
