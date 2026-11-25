use super::*;
use nalgebra::{DMatrix, DVector};

/// Epps–Singleton two-sample characteristic-function test `W=(n+m)*Δgᵀ Σ⁺ Δg`.
///
/// ## Parameters and conventions
///
/// `x`,`y` are unweighted samples, possibly discrete. Explicit positive
/// frequencies `t` are divided by the pooled semi-IQR. g concatenates cosine
/// features followed by sine features. Σ combines population feature covariance
/// matrices with factors `(n+m)/n` and `(n+m)/m`. SciPy's small-sample factor
/// `1/(1+(n+m)^(-0.45)+10.1*(n^(-1.7)+m^(-1.7)))` applies when max(n,m)<25.
/// Chi-square degrees of freedom equal the retained covariance rank.
///
/// ## Domain and special values
///
/// At least five finite observations per sample, combined count≤1048576,
/// 1..=8 finite positive frequencies, and `(n+m)*(2*t.len())²≤16777216`.
/// Pooled semi-IQR must be positive. Duplicate frequencies are allowed and can
/// reduce rank, matching SciPy. No NaN omission, broadcasting or masks.
///
/// ## Returns
///
/// Nonnegative dimensionless W and an approximate chi-square p-value.
///
/// ## Errors
///
/// Invalid domains/work limits, zero semi-IQR, unrepresentable phases, zero
/// covariance rank or failed matrix decomposition return errors.
///
/// ## Numerical behavior
///
/// O((n+m)*k²+k³) work and O((n+m)*k+k²) memory for k=2*t.len(). Private nalgebra
/// SVD (at most 10,000 iterations) computes the covariance pseudoinverse, retaining singular values greater
/// than 1e-15 times the largest, matching NumPy pinv's default cutoff. Severe
/// ill-conditioning may change effective rank; small-sample calibration is approximate.
/// Private statrs supplies chi-square tails.
///
/// ## Examples
///
/// ```
/// let x = [1.,2.,3.,4.,5.];
/// let r = sci_stats::inference::epps_singleton_2samp(&x, &x, &[0.4,0.8])?;
/// assert_eq!(r.statistic, 0.);
/// assert_eq!(r.pvalue, 1.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy Epps–Singleton and Epps–Singleton (1986)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.epps_singleton_2samp.html).
pub fn epps_singleton_2samp(x: &[f64], y: &[f64], t: &[f64]) -> Result<TestResult> {
    sample(x, 5)?;
    sample(y, 5)?;
    sample(t, 1)?;
    let n = x.len() + y.len();
    let k = 2 * t.len();
    if n > MAX_SAMPLES || t.len() > 8 || n.checked_mul(k * k).is_none_or(|w| w > 16_777_216) {
        return Err(Error::InvalidInput("Epps-Singleton work limit exceeded"));
    }
    for &frequency in t {
        positive(frequency)?;
    }
    let pooled = x.iter().chain(y).copied().collect::<Vec<_>>();
    let scale = iqr(&pooled)? / 2.;
    positive(scale)?;
    let features = |values: &[f64]| -> Result<(DVector<f64>, DMatrix<f64>)> {
        let mut data = DMatrix::zeros(k, values.len());
        for (j, &v) in values.iter().enumerate() {
            for (i, &frequency) in t.iter().enumerate() {
                let phase = match frequency * (v / scale) {
                    value if value.is_finite() => Ok(value),
                    _ => Err(crate::Error::NonFiniteEvaluation),
                }?;
                data[(i, j)] = phase.cos();
                data[(i + t.len(), j)] = phase.sin();
            }
        }
        let mut means = DVector::zeros(k);
        for i in 0..k {
            means[i] = mean(&data.row(i).iter().copied().collect::<Vec<_>>())?;
        }
        for j in 0..values.len() {
            for i in 0..k {
                data[(i, j)] -= means[i];
            }
        }
        let covariance = &data * data.transpose() / values.len() as f64;
        Ok((means, covariance))
    };
    let (mx, cx) = features(x)?;
    let (my, cy) = features(y)?;
    let covariance = cx * (n as f64 / x.len() as f64) + cy * (n as f64 / y.len() as f64);
    let svd = covariance
        .try_svd(true, true, f64::EPSILON, 10_000)
        .ok_or(Error::NumericalFailure("covariance SVD did not converge"))?;
    let largest = svd.singular_values.iter().copied().fold(0_f64, f64::max);
    let cutoff = largest * 1e-15;
    let rank = svd.singular_values.iter().filter(|s| **s > cutoff).count();
    if rank == 0 {
        return Err(Error::NumericalFailure("zero covariance rank"));
    }
    let inverse = svd
        .pseudo_inverse(cutoff)
        .map_err(|_| Error::NumericalFailure("covariance pseudoinverse failed"))?;
    let delta = mx - my;
    let mut statistic = match n as f64 * delta.dot(&(&inverse * &delta)) {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }?
    .max(0.);
    if x.len().max(y.len()) < 25 {
        statistic /= 1.
            + (n as f64).powf(-0.45)
            + 10.1 * ((x.len() as f64).powf(-1.7) + (y.len() as f64).powf(-1.7));
    }
    inference::chi_result(statistic, rank as f64)
}
