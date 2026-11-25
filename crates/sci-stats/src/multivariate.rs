use super::*;
use nalgebra::{DMatrix, DVector};
use statrs::distribution::{
    Categorical, Continuous, Dirichlet, Discrete, DiscreteCDF, MultivariateNormal,
    MultivariateStudent,
};

fn matrix(values: &[Vec<f64>], n: usize) -> Result<DMatrix<f64>> {
    if n == 0 || n > 64 || values.len() != n || values.iter().any(|r| r.len() != n) {
        return Err(Error::DimensionMismatch);
    }
    let mut flat = Vec::with_capacity(n * n);
    for r in values {
        sample(r, n)?;
        flat.extend_from_slice(r);
    }
    for (i, row) in values.iter().enumerate() {
        for (j, other) in values.iter().enumerate().take(i) {
            if row[j] != other[i] {
                return Err(Error::InvalidInput("covariance must be symmetric"));
            }
        }
    }
    Ok(DMatrix::from_row_slice(n, n, &flat))
}
fn coordinates(x: &[f64], mean: &[f64]) -> Result<()> {
    sample(x, 1)?;
    sample(mean, 1)?;
    if x.len() != mean.len() {
        return Err(Error::DimensionMismatch);
    }
    Ok(())
}
/// Density `exp(−(x−μ)ᵀΣ⁻¹(x−μ)/2)/sqrt((2π)^d det(Σ))`.
///
/// ## Parameters
///
/// Location and x are vectors; covariance is a matrix with one vector per row, matching the shared algebra API.
/// Covariance is symmetric positive definite; no singular-covariance overload.
///
/// ## Domain
///
/// Equal dimensions from 1 through 64; finite coordinates and symmetric positive-definite matrix.
///
/// ## Returns
///
/// Scalar joint density in inverse product observation units.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Statrs Cholesky and matrix inversion use O(d³) work; dimensions are bounded by 64. Near-singular matrices can be inaccurate.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!(sci_stats::multivariate::multivariate_normal_pdf(&[0., 0.], &[0., 0.], &[vec![1., 0.], vec![0., 1.]])?.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `multivariate_normal`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html).
pub fn multivariate_normal_pdf(
    x: &[f64],
    location: &[f64],
    covariance: &[Vec<f64>],
) -> Result<f64> {
    coordinates(x, location)?;
    let cov = matrix(covariance, x.len())?;
    let d = MultivariateNormal::new_from_nalgebra(DVector::from_column_slice(location), cov)
        .map_err(|_| Error::InvalidInput("covariance must be positive definite"))?;
    match d.pdf(&DVector::from_column_slice(x)) {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }
}
/// Natural logarithm of `exp(−(x−μ)ᵀΣ⁻¹(x−μ)/2)/sqrt((2π)^d det(Σ))`.
///
/// ## Parameters
///
/// Location and x are vectors; covariance is a matrix with one vector per row, matching the shared algebra API.
/// Covariance is symmetric positive definite; no singular-covariance overload.
///
/// ## Domain
///
/// Equal dimensions from 1 through 64; finite coordinates and symmetric positive-definite matrix.
///
/// ## Returns
///
/// Scalar log density.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Statrs Cholesky and matrix inversion use O(d³) work; dimensions are bounded by 64. Near-singular matrices can be inaccurate.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!(sci_stats::multivariate::multivariate_normal_logpdf(&[0., 0.], &[0., 0.], &[vec![1., 0.], vec![0., 1.]])?.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `multivariate_normal`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html).
pub fn multivariate_normal_logpdf(
    x: &[f64],
    location: &[f64],
    covariance: &[Vec<f64>],
) -> Result<f64> {
    coordinates(x, location)?;
    let cov = matrix(covariance, x.len())?;
    let d = MultivariateNormal::new_from_nalgebra(DVector::from_column_slice(location), cov)
        .map_err(|_| Error::InvalidInput("covariance must be positive definite"))?;
    match d.ln_pdf(&DVector::from_column_slice(x)) {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }
}
/// Density `Γ((ν+d)/2)/(Γ(ν/2)*(νπ)^(d/2)*sqrt(det(S)))*(1+δ/ν)^(-(ν+d)/2), δ=(x−μ)ᵀS⁻¹(x−μ)`.
///
/// ## Parameters
///
/// Location and x are vectors; covariance is a matrix with one vector per row, matching the shared algebra API.
/// The matrix is the t shape matrix, not covariance; covariance=df/(df−2)*shape when df>2.
///
/// ## Domain
///
/// Equal dimensions from 1 through 64; finite coordinates and symmetric positive-definite matrix; finite df>0 up to 1048576.
///
/// ## Returns
///
/// Scalar joint density in inverse product observation units.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Statrs Cholesky and matrix inversion use O(d³) work; dimensions are bounded by 64. Near-singular matrices can be inaccurate.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!(sci_stats::multivariate::multivariate_t_pdf(&[0., 0.], &[0., 0.], &[vec![1., 0.], vec![0., 1.]], 4.)?.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `multivariate_t`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_t.html).
pub fn multivariate_t_pdf(
    x: &[f64],
    location: &[f64],
    covariance: &[Vec<f64>],
    df: f64,
) -> Result<f64> {
    coordinates(x, location)?;
    let cov = matrix(covariance, x.len())?;
    positive(df)?;
    if df > MAX_SAMPLES as f64 {
        return Err(Error::InvalidInput("degrees of freedom exceed limit"));
    }
    let d = MultivariateStudent::new_from_nalgebra(DVector::from_column_slice(location), cov, df)
        .map_err(|_| Error::InvalidInput("covariance must be positive definite"))?;
    match d.pdf(&DVector::from_column_slice(x)) {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }
}
/// Natural logarithm of `Γ((ν+d)/2)/(Γ(ν/2)*(νπ)^(d/2)*sqrt(det(S)))*(1+δ/ν)^(-(ν+d)/2), δ=(x−μ)ᵀS⁻¹(x−μ)`.
///
/// ## Parameters
///
/// Location and x are vectors; covariance is a matrix with one vector per row, matching the shared algebra API.
/// The matrix is the t shape matrix, not covariance; covariance=df/(df−2)*shape when df>2.
///
/// ## Domain
///
/// Equal dimensions from 1 through 64; finite coordinates and symmetric positive-definite matrix; finite df>0 up to 1048576.
///
/// ## Returns
///
/// Scalar log density.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Statrs Cholesky and matrix inversion use O(d³) work; dimensions are bounded by 64. Near-singular matrices can be inaccurate.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!(sci_stats::multivariate::multivariate_t_logpdf(&[0., 0.], &[0., 0.], &[vec![1., 0.], vec![0., 1.]], 4.)?.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `multivariate_t`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_t.html).
pub fn multivariate_t_logpdf(
    x: &[f64],
    location: &[f64],
    covariance: &[Vec<f64>],
    df: f64,
) -> Result<f64> {
    coordinates(x, location)?;
    let cov = matrix(covariance, x.len())?;
    positive(df)?;
    if df > MAX_SAMPLES as f64 {
        return Err(Error::InvalidInput("degrees of freedom exceed limit"));
    }
    let d = MultivariateStudent::new_from_nalgebra(DVector::from_column_slice(location), cov, df)
        .map_err(|_| Error::InvalidInput("covariance must be positive definite"))?;
    match d.ln_pdf(&DVector::from_column_slice(x)) {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }
}
/// Dirichlet density `Γ(Σα)/ΠΓ(α_i)*Πx_i^(α_i−1)`.
///
/// ## Parameters
///
/// Uses all simplex coordinates, including the final one; output density is with respect to the first d−1 coordinates.
///
/// ## Domain
///
/// Equal dimensions d≥2 through 64; positive finite alpha up to 1048576; x_i strictly in (0,1), sum x within 1e-12 of one. Boundary densities unsupported.
///
/// ## Returns
///
/// Scalar simplex density.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Uses statrs privately, with native log-factorial evaluation for multinomial
/// masses to avoid overflowing ordinary multinomial coefficients. Binary64
/// normalization can underflow or lose accuracy for ill-conditioned parameters.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!(sci_stats::multivariate::dirichlet_pdf(&[0.3, 0.7], &[2., 3.])?.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `dirichlet`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dirichlet.html).
pub fn dirichlet_pdf(x: &[f64], alpha: &[f64]) -> Result<f64> {
    coordinates(x, alpha)?;
    if x.len() < 2
        || x.len() > 64
        || x.iter().any(|&v| v <= 0. || v >= 1.)
        || (x.iter().sum::<f64>() - 1.).abs() > 1e-12
    {
        return Err(Error::InvalidInput("invalid interior simplex coordinates"));
    }
    for &a in alpha {
        positive(a)?;
        if a > MAX_SAMPLES as f64 {
            return Err(Error::InvalidInput("concentration exceeds limit"));
        }
    }
    let d = Dirichlet::new(alpha.to_vec())
        .map_err(|_| Error::InvalidInput("invalid concentrations"))?;
    match d.pdf(&DVector::from_column_slice(x)) {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }
}
/// Logarithm of Dirichlet density `Γ(Σα)/ΠΓ(α_i)*Πx_i^(α_i−1)`.
///
/// ## Parameters
///
/// Uses all simplex coordinates, including the final one; output density is with respect to the first d−1 coordinates.
///
/// ## Domain
///
/// Equal dimensions d≥2 through 64; positive finite alpha up to 1048576; x_i strictly in (0,1), sum x within 1e-12 of one. Boundary densities unsupported.
///
/// ## Returns
///
/// Scalar log density.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Uses statrs privately, with native log-factorial evaluation for multinomial
/// masses to avoid overflowing ordinary multinomial coefficients. Binary64
/// normalization can underflow or lose accuracy for ill-conditioned parameters.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!(sci_stats::multivariate::dirichlet_logpdf(&[0.3, 0.7], &[2., 3.])?.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `dirichlet`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dirichlet.html).
pub fn dirichlet_logpdf(x: &[f64], alpha: &[f64]) -> Result<f64> {
    coordinates(x, alpha)?;
    if x.len() < 2
        || x.len() > 64
        || x.iter().any(|&v| v <= 0. || v >= 1.)
        || (x.iter().sum::<f64>() - 1.).abs() > 1e-12
    {
        return Err(Error::InvalidInput("invalid interior simplex coordinates"));
    }
    for &a in alpha {
        positive(a)?;
        if a > MAX_SAMPLES as f64 {
            return Err(Error::InvalidInput("concentration exceeds limit"));
        }
    }
    let d = Dirichlet::new(alpha.to_vec())
        .map_err(|_| Error::InvalidInput("invalid concentrations"))?;
    match d.ln_pdf(&DVector::from_column_slice(x)) {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }
}
/// multinomial probability `n!/Πk_i!*Πp_i^k_i`, n=Σk_i.
///
/// ## Parameters
///
/// Counts and probabilities have matching category order; n is inferred from counts. Probabilities must sum to one within 1e-12.
///
/// ## Domain
///
/// Between two and 64 categories, bounded nonnegative integer counts with total≤1048576; finite nonnegative probabilities.
///
/// ## Returns
///
/// Scalar probability mass.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Uses statrs privately, with native log-factorial evaluation for multinomial
/// masses to avoid overflowing ordinary multinomial coefficients. Binary64
/// normalization can underflow or lose accuracy for ill-conditioned parameters.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!(sci_stats::multivariate::multinomial_pmf(&[1., 2.], &[0.5, 0.5])?.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `multinomial`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multinomial.html).
pub fn multinomial_pmf(counts: &[f64], probabilities: &[f64]) -> Result<f64> {
    Ok(multinomial_logpmf(counts, probabilities)?.exp())
}
/// Logarithm of multinomial probability `n!/Πk_i!*Πp_i^k_i`, n=Σk_i.
///
/// ## Parameters
///
/// Counts and probabilities have matching category order; n is inferred from counts. Probabilities must sum to one within 1e-12.
///
/// ## Domain
///
/// Between two and 64 categories, bounded nonnegative integer counts with total≤1048576; finite nonnegative probabilities.
///
/// ## Returns
///
/// Scalar log probability.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Uses statrs privately, with native log-factorial evaluation for multinomial
/// masses to avoid overflowing ordinary multinomial coefficients. Binary64
/// normalization can underflow or lose accuracy for ill-conditioned parameters.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert!(sci_stats::multivariate::multinomial_logpmf(&[1., 2.], &[0.5, 0.5])?.is_finite());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `multinomial`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multinomial.html).
pub fn multinomial_logpmf(counts: &[f64], probabilities: &[f64]) -> Result<f64> {
    coordinates(counts, probabilities)?;
    if counts.len() < 2
        || counts.len() > 64
        || probabilities.iter().any(|&v| v < 0.)
        || (probabilities.iter().sum::<f64>() - 1.).abs() > 1e-12
    {
        return Err(Error::InvalidInput("invalid multinomial probabilities"));
    }
    let mut k = Vec::with_capacity(counts.len());
    let mut n = 0_u64;
    for &v in counts {
        if !(0. ..=MAX_SAMPLES as f64).contains(&v) || v.fract() != 0. {
            return Err(Error::InvalidInput(
                "counts must be nonnegative bounded integers",
            ));
        }
        k.push(v as u64);
        n += v as u64;
    }
    if n > MAX_SAMPLES as u64 {
        return Err(Error::InvalidInput("trial total exceeds limit"));
    }
    // statrs ln_pmf takes ln of an ordinary multinomial coefficient, which
    // overflows for moderate counts, and evaluates 0*ln(0) at zero weights.
    // The defining log-factorial formula reuses statrs without those hazards.
    let mut value = statrs::function::factorial::ln_factorial(n);
    for (&count, &p) in k.iter().zip(probabilities) {
        value -= statrs::function::factorial::ln_factorial(count);
        if count > 0 {
            if p == 0. {
                return Ok(f64::NEG_INFINITY);
            }
            value += count as f64 * p.ln();
        }
    }
    match value {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }
}
/// Categorical probability `P(X=k)=weight_k/Σweight`.
///
/// ## Parameters
///
/// Categories are zero-based indices; nonnegative weights are normalized automatically.
///
/// ## Domain
///
/// Nonempty finite nonnegative weights with positive total. Infinite arguments are allowed; NaN rejected.
///
/// ## Returns
///
/// Scalar category probability.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Uses statrs privately, with native log-factorial evaluation for multinomial
/// masses to avoid overflowing ordinary multinomial coefficients. Binary64
/// normalization can underflow or lose accuracy for ill-conditioned parameters.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::multivariate::categorical_pmf(&[1., 1.], 0.)?, 0.5);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `categorical`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.categorical.html).
pub fn categorical_pmf(weights: &[f64], category: f64) -> Result<f64> {
    let weights = super::transformations::probabilities(weights)?;
    if category.is_nan() {
        return Err(Error::InvalidInput("category must not be NaN"));
    }
    if category < 0. {
        return Ok(0.);
    }
    if category >= weights.len() as f64 {
        return Ok(0.);
    }
    if category.fract() != 0. {
        return Ok(0.);
    }
    let d = Categorical::new(&weights)
        .map_err(|_| Error::InvalidInput("invalid categorical weights"))?;
    Ok(d.pmf(category.floor() as u64))
}
/// Categorical CDF `P(X≤k)=Σ_{i≤floor(k)}weight_i/Σweight`.
///
/// ## Parameters
///
/// Categories are zero-based indices; nonnegative weights are normalized automatically.
///
/// ## Domain
///
/// Nonempty finite nonnegative weights with positive total. Infinite arguments are allowed; NaN rejected.
///
/// ## Returns
///
/// Scalar cumulative probability.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Uses statrs privately, with native log-factorial evaluation for multinomial
/// masses to avoid overflowing ordinary multinomial coefficients. Binary64
/// normalization can underflow or lose accuracy for ill-conditioned parameters.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::multivariate::categorical_cdf(&[1., 1.], 0.)?, 0.5);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `categorical_cdf`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.html).
pub fn categorical_cdf(weights: &[f64], category: f64) -> Result<f64> {
    let weights = super::transformations::probabilities(weights)?;
    if category.is_nan() {
        return Err(Error::InvalidInput("category must not be NaN"));
    }
    if category < 0. {
        return Ok(0.);
    }
    if category >= weights.len() as f64 {
        return Ok(1.);
    }

    let d = Categorical::new(&weights)
        .map_err(|_| Error::InvalidInput("invalid categorical weights"))?;
    Ok(d.cdf(category.floor() as u64))
}
