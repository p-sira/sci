use super::*;

/// Saltelli first-order and total-effect indices for a scalar model output.
#[derive(Clone, Debug, PartialEq)]
pub struct SobolIndices {
    /// First-order indices, in input-variable order.
    pub first_order: Vec<f64>,
    /// Total-effect indices, in input-variable order.
    pub total_order: Vec<f64>,
}

/// Sobol sensitivity estimators `S_i=mean(B*(AB_i−A))/V`, `ST_i=mean((A−AB_i)²)/(2V)`.
///
/// ## Parameters and conventions
///
/// `f_a`,`f_b` are n scalar outputs on independent A/B sample matrices.
/// Each row of `f_ab` contains n outputs after replacing A's i-th input by
/// B's i-th input. Outputs are centered by the pooled A/B mean; V is the
/// population variance of pooled A/B outputs. This supports SciPy's supplied
/// evaluation dictionary with one scalar output and method="saltelli_2010".
/// Input generation, model callbacks, multi-output arrays and bootstrap result
/// methods are unsupported. Independent input coordinates are required.
///
/// ## Domain and special values
///
/// Finite, equal-length A/B outputs with power-of-two n≥2 and n≤1048576;
/// 1..=64 matching hybrid rows with (d+2)*n≤16777216. Constant pooled A/B
/// output yields zero indices, matching SciPy's nonfinite-index replacement.
/// NaN and infinity evaluations are errors. Estimates may lie outside `[0,1]`.
///
/// ## Returns
///
/// Dimensionless first-order and total-effect vectors of length d in input order.
///
/// ## Errors
///
/// Invalid shapes/domains/work limits or unrepresentable normalized hybrid
/// outputs and indices return errors.
///
/// ## Numerical behavior
///
/// O(d*n) work and O(n+d) additional memory. Shared shifted/scaled centering
/// avoids avoidable variance overflow and large-offset cancellation. Finite
/// sampling can produce negative first-order or greater-than-one indices;
/// numerical precision does not establish convergence of the estimator.
///
/// ## Examples
///
/// ```
/// let a = [0.,1.,2.,3.]; let b = [3.,2.,1.,0.];
/// let r = sci_stats::sobol_indices(&a,&b,&[&b])?;
/// assert_eq!(r.first_order.len(), 1);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy sobol_indices and Saltelli et al. (2010)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sobol_indices.html).
pub fn sobol_indices(f_a: &[f64], f_b: &[f64], f_ab: &[&[f64]]) -> Result<SobolIndices> {
    if f_a.len() != f_b.len() {
        return Err(Error::DimensionMismatch);
    }
    sample(f_a, 2)?;
    sample(f_b, 2)?;
    if !f_a.len().is_power_of_two()
        || !(1..=64).contains(&f_ab.len())
        || f_a
            .len()
            .checked_mul(f_ab.len() + 2)
            .is_none_or(|w| w > 16_777_216)
    {
        return Err(Error::InvalidInput(
            "Sobol needs power-of-two n, 1..=64 inputs and bounded evaluation count",
        ));
    }
    for row in f_ab {
        if row.len() != f_a.len() {
            return Err(Error::DimensionMismatch);
        }
        finite_slice(row)?;
    }
    let pooled = f_a.iter().chain(f_b).copied().collect::<Vec<_>>();
    let centered = centered_sample(&pooled)?;
    let mut first_order = Vec::with_capacity(f_ab.len());
    let mut total_order = Vec::with_capacity(f_ab.len());
    if centered.scale == 0. {
        return Ok(SobolIndices {
            first_order: vec![0.; f_ab.len()],
            total_order: vec![0.; f_ab.len()],
        });
    }
    let variance = normalized_variance(&centered.deviations, 0);
    let (a, b) = centered.deviations.split_at(f_a.len());
    for row in f_ab {
        let mut first = Vec::with_capacity(row.len());
        let mut total = Vec::with_capacity(row.len());
        for ((&hybrid, &a), &b) in row.iter().zip(a).zip(b) {
            let hybrid = finite((hybrid - centered.origin) / centered.scale - centered.center)?;
            let difference = finite(hybrid - a)?;
            first.push(finite(b * difference)?);
            total.push(finite(difference * difference)?);
        }
        first_order.push(finite(mean(&first)? / variance)?);
        total_order.push(finite(mean(&total)? / (2. * variance))?);
    }
    Ok(SobolIndices {
        first_order,
        total_order,
    })
}

/// Estimated rejection probability and test p-values under a supplied alternative.
#[derive(Clone, Debug, PartialEq)]
pub struct PowerResult {
    /// Rejection fraction, dimensionless in `[0,1]`.
    pub power: f64,
    /// Validated test p-values, in supplied sample order.
    pub pvalues: Vec<f64>,
}

/// Simulated test power `power=(1/B)Σ I(pvalue(test(sample_b))<significance)`.
///
/// ## Parameters and conventions
///
/// `samples` supplies independent draws under the alternative, each with the
/// same observation count. `test` is a deterministic fallible callback returning
/// a scalar p-value. `significance` is the rejection threshold; equality does
/// not reject, matching SciPy. RNG ownership is adapted to explicit draws.
/// Multi-sample/vectorized tests, sample-size grids and test kwargs are unsupported.
///
/// ## Domain and special values
///
/// Finite nonempty equal-length samples, 1..=1048576 draws, and n*B≤16777216.
/// Significance and all returned p-values must be in `[0,1]`; NaN/infinity are errors.
/// Significance zero never rejects; significance one retains p-values equal to one.
///
/// ## Returns
///
/// Dimensionless rejection fraction and matching p-value vector.
///
/// ## Errors
///
/// Invalid domains/shapes/work limits or invalid p-values return errors;
/// callback errors propagate unchanged.
///
/// ## Numerical behavior
///
/// O(B*n) input validation and O(B) result memory, plus caller-controlled test
/// work. Binomial Monte Carlo standard error is approximately
/// sqrt(power*(1−power)/B); the caller must supply independent alternative draws.
///
/// ## Examples
///
/// ```
/// let samples: &[&[f64]] = &[&[1.,2.,3.], &[2.,3.,4.]];
/// let r = sci_stats::power(samples, |x| Ok(sci_stats::ttest_1samp(x,0.,sci_stats::Alternative::Greater)?.pvalue), 0.05)?;
/// assert_eq!(r.pvalues.len(), 2);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy power](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.power.html).
pub fn power<F>(samples: &[&[f64]], test: F, significance: f64) -> Result<PowerResult>
where
    F: Fn(&[f64]) -> Result<f64>,
{
    probability_check(significance)?;
    let Some(first) = samples.first() else {
        return Err(Error::InvalidInput("power needs samples"));
    };
    sample(first, 1)?;
    let repetitions = samples.len();
    if repetitions > MAX_SAMPLES
        || first
            .len()
            .checked_mul(repetitions)
            .is_none_or(|w| w > 16_777_216)
    {
        return Err(Error::InvalidInput("power work limit exceeded"));
    }
    for row in samples {
        if row.len() != first.len() {
            return Err(Error::DimensionMismatch);
        }
        finite_slice(row)?;
    }
    let mut pvalues = Vec::with_capacity(samples.len());
    for row in samples {
        let pvalue = test(row)?;
        probability_check(pvalue)?;
        pvalues.push(pvalue);
    }
    let power = pvalues.iter().filter(|&&p| p < significance).count() as f64 / pvalues.len() as f64;
    Ok(PowerResult { power, pvalues })
}
