use super::*;
use crate::special;
use std::f64::consts::PI;

/// Fitted null families for Anderson–Darling diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AndersonFamily {
    /// Normal, fitted by sample mean and standard deviation with ddof=1.
    Normal,
    /// Exponential with location fixed at zero and scale fitted by sample mean.
    Exponential,
}
/// Anderson–Darling statistic, tabulated thresholds and fitted parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct AndersonResult {
    /// Dimensionless Anderson–Darling A² statistic.
    pub statistic: f64,
    /// Critical A² values, corresponding to `significance_levels`.
    pub critical_values: [f64; 5],
    /// Significance percentages, ordered [15,10,5,2.5,1].
    pub significance_levels: [f64; 5],
    /// Fitted location, in observation units.
    pub location: f64,
    /// Fitted positive scale, in observation units.
    pub scale: f64,
}

/// Anderson–Darling statistic `A²=−n−Σ_(i=1)^n (2i−1)/n [ln F(x_i)+ln(1−F(x_(n+1−i)))]`.
///
/// ## Parameters and conventions
///
/// `values` is an unweighted sample; `family` chooses a fitted normal or
/// zero-location exponential null. Returns SciPy's legacy critical-value
/// result. Logistic, Gumbel and Weibull fitting, interpolated p-values, and
/// Monte Carlo calibration are unsupported. Critical values are Stephens'
/// tabulated values divided by `1+0.75/n+2.25/n²` (normal) or `1+0.6/n`
/// (exponential), rounded to three decimals as in SciPy.
///
/// ## Domain and special values
///
/// 2..=1048576 finite nonconstant observations. Exponential observations must
/// be nonnegative and have positive mean; zeros give infinite A². NaN and
/// infinite observations are errors. Normal location/scale must be representable.
///
/// ## Returns
///
/// A², critical values, significance percentages and fitted location/scale.
/// A² exceeding a threshold rejects at that threshold's significance level.
///
/// ## Errors
///
/// Invalid samples/domains or unrepresentable fitted parameters return errors.
///
/// ## Numerical behavior
///
/// O(n log n) sorting, O(n) memory. Normal log tails use the private shared
/// special-function backend to avoid ordinary-CDF underflow. Exponential log
/// tails use expm1. Critical values are approximate, especially at small n.
///
/// ## Examples
///
/// ```
/// let r = sci_stats::anderson(&[1.,2.,3.,4.,5.], sci_stats::AndersonFamily::Normal)?;
/// assert_eq!(r.location, 3.);
/// assert_eq!(r.significance_levels, [15.,10.,5.,2.5,1.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy anderson and Stephens (1974–1977)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html).
pub fn anderson(values: &[f64], family: AndersonFamily) -> Result<AndersonResult> {
    sample(values, 2)?;
    if values.iter().all(|x| *x == values[0]) {
        return Err(Error::InvalidInput(
            "Anderson test requires nonconstant observations",
        ));
    }
    let ordered = sorted(values)?;
    let n = values.len() as f64;
    let (location, scale, critical, correction, logcdf, logsf) = match family {
        AndersonFamily::Normal => {
            let location = mean(values)?;
            let scale = std(values, 1)?;
            positive(scale)?;
            let scores = zscore(&ordered, 1)?;
            let lower = scores
                .iter()
                .map(|&z| special::probability::log_ndtr(z))
                .collect::<Vec<_>>();
            let upper = scores
                .iter()
                .map(|&z| special::probability::log_ndtr(-z))
                .collect::<Vec<_>>();
            (
                location,
                scale,
                [0.561, 0.631, 0.752, 0.873, 1.035],
                1. + 0.75 / n + 2.25 / (n * n),
                lower,
                upper,
            )
        }
        AndersonFamily::Exponential => {
            if ordered[0] < 0. {
                return Err(Error::InvalidInput(
                    "exponential observations must be nonnegative",
                ));
            }
            let scale = mean(values)?;
            positive(scale)?;
            let scores = ordered.iter().map(|v| v / scale).collect::<Vec<_>>();
            let lower = scores
                .iter()
                .map(|&z| (-(-z).exp_m1()).ln())
                .collect::<Vec<_>>();
            let upper = scores.iter().map(|&z| -z).collect::<Vec<_>>();
            (
                0.,
                scale,
                [0.916, 1.062, 1.321, 1.591, 1.959],
                1. + 0.6 / n,
                lower,
                upper,
            )
        }
    };
    let statistic = -n
        - logcdf
            .iter()
            .zip(logsf.iter().rev())
            .enumerate()
            .map(|(i, (a, b))| ((2 * i + 1) as f64 / n) * (a + b))
            .sum::<f64>();
    if statistic.is_nan() {
        return Err(Error::NonFiniteEvaluation);
    }
    Ok(AndersonResult {
        statistic,
        critical_values: critical.map(|v| (v / correction * 1000.).round_ties_even() / 1000.),
        significance_levels: [15., 10., 5., 2.5, 1.],
        location,
        scale,
    })
}

fn cvm_infinite(x: f64) -> Result<f64> {
    let mut sum = 0.;
    for k in 0..256 {
        let k = k as f64;
        let y = 4. * k + 1.;
        let q = y * y / (16. * x);
        let term = match (special::gamma::gammaln(k + 0.5) - special::gamma::gammaln(k + 1.)).exp()
            / (PI.powf(1.5) * x.sqrt())
            * y.sqrt()
            * (-q).exp()
            * special::bessel::kv(0.25, q)
        {
            value if value.is_finite() => Ok(value),
            _ => Err(crate::Error::NonFiniteEvaluation),
        }?;
        sum += term;
        if term.abs() < 1e-7 {
            return Ok(sum);
        }
    }
    Err(Error::NumericalFailure(
        "Cramer-von Mises series limit exceeded",
    ))
}
fn cvm_correction(x: f64) -> Result<f64> {
    let ed2 = |y: f64| {
        let z = y * y / 4.;
        (-z).exp()
            * (y / 2.).powf(1.5)
            * (special::bessel::kv(0.25, z) + special::bessel::kv(0.75, z))
            / PI.sqrt()
    };
    let ed3 = |y: f64| {
        let z = y * y / 4.;
        (-z).exp()
            * (y / 2.).powf(2.5)
            * (2. * special::bessel::kv(0.25, z) + 3. * special::bessel::kv(0.75, z)
                - special::bessel::kv(1.25, z))
            / PI.sqrt()
    };
    let mut sum = 0.;
    for k in 0..256 {
        let k = k as f64;
        let m = 2. * k + 1.;
        let sx = 2. * x.sqrt();
        let g = (special::gamma::gammaln(k + 0.5) - special::gamma::gammaln(k + 1.)).exp();
        let term = match -g / PI
            * (m * ed2((4. * k + 3.) / sx) / (9. * x.powf(0.75))
                + ed3((4. * k + 1.) / sx) / (72. * x.powf(1.25))
                + 2. * (m + 2.) * (k + 0.5) * ed3((4. * k + 5.) / sx) / (12. * x.powf(1.25))
                + 7. * m * (ed2((4. * k + 1.) / sx) + ed2((4. * k + 5.) / sx))
                    / (144. * x.powf(0.75)))
        {
            value if value.is_finite() => Ok(value),
            _ => Err(crate::Error::NonFiniteEvaluation),
        }?;
        sum += term;
        if term.abs() < 1e-7 {
            return Ok(sum);
        }
    }
    Err(Error::NumericalFailure(
        "Cramer-von Mises correction series limit exceeded",
    ))
}

/// One-sample Cramér–von Mises test `T=1/(12n)+Σ_(i=1)^n (F(x_i)−(2i−1)/(2n))²`.
///
/// ## Parameters and conventions
///
/// `values` is sorted internally; `null` is a fully specified continuous Sci
/// distribution. Uses the finite-sample CDF approximation of Csörgő–Faraway
/// (1996), `P(T≤t)≈V(t)*(1+1/(12n))+ψ₁(t)/n`, evaluated with Bessel-K series.
/// Null parameters must be independently specified; fitted/composite nulls
/// invalidate this calibration. SciPy CDF callbacks, axes and masks are unsupported.
///
/// ## Domain and special values
///
/// 2..=1048576 finite observations, valid continuous null parameters. Discrete
/// nulls are rejected. NaN/infinite observations are errors. Samples outside
/// the null support can produce the largest statistic and zero approximate p-value.
///
/// ## Returns
///
/// Dimensionless T and finite-sample approximate greater-tail probability.
///
/// ## Errors
///
/// Invalid inputs/nulls, nonfinite evaluations or unconverged series return errors.
///
/// ## Numerical behavior
///
/// O(n log n) sorting and O(n) memory. Private Sci special-function backends
/// supply Bessel K and log gamma; each series is capped at 256 terms and stops
/// at term magnitude<1e-7, matching SciPy's approximation. Accuracy is poor
/// near finite-sample support endpoints and for T>2; very small upper-tail
/// probabilities can round to zero. The approximate CDF is clipped to `[0,1]`.
///
/// ## Examples
///
/// ```
/// let r = sci_stats::cramervonmises(&[-1.,0.,1.], sci_stats::Distribution::Normal { mean:0., std_dev:1. })?;
/// assert!((0.0..=1.0).contains(&r.pvalue));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy cramervonmises and Csörgő–Faraway (1996)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.cramervonmises.html).
pub fn cramervonmises(values: &[f64], null: Distribution) -> Result<TestResult> {
    continuous_null(null)?;
    sample(values, 2)?;
    let n = values.len() as f64;
    let statistic = cvm_statistic(values, null)?;
    let probability = if statistic <= 1. / (12. * n) {
        0.
    } else if statistic >= n / 3. {
        1.
    } else {
        cvm_infinite(statistic)? * (1. + 1. / (12. * n)) + cvm_correction(statistic)? / n
    };
    Ok(TestResult {
        statistic,
        pvalue: (1. - probability).clamp(0., 1.),
    })
}

pub(super) fn cvm_statistic(values: &[f64], null: Distribution) -> Result<f64> {
    let n = values.len() as f64;
    let ordered = sorted(values)?;
    let mut statistic = 1. / (12. * n);
    for (i, &value) in ordered.iter().enumerate() {
        statistic += (cdf(null, value)? - (2 * i + 1) as f64 / (2. * n)).powi(2);
    }
    match statistic {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }
}

/// Null calibration for the two-sample Cramér–von Mises statistic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CvmMethod {
    /// Enumerate all untied rank partitions; combined sample size is limited to 20.
    Exact,
    /// Finite-moment correction followed by the limiting CDF.
    Asymptotic,
}
fn cvm_rank_u(first_ranks: &[f64], second_ranks: &[f64]) -> f64 {
    let group = |ranks: &[f64]| {
        ranks.len() as f64
            * ranks
                .iter()
                .enumerate()
                .map(|(i, r)| (r - (i + 1) as f64).powi(2))
                .sum::<f64>()
    };
    group(first_ranks) + group(second_ranks)
}

/// Two-sample Cramér–von Mises test `T=U/(nm(n+m))−(4nm−1)/(6(n+m))`.
///
/// ## Parameters and conventions
///
/// `x`,`y` are unweighted samples. Pooled average ranks R_i,S_j define
/// `U=n*Σ(R_i−i)²+m*Σ(S_j−j)²` after sorting each sample.
/// Exact mode enumerates all partitions of untied ranks into sizes n and m,
/// matching SciPy's continuous-null calibration (even when observed data tie).
/// Asymptotic mode uses the Anderson finite-moment adjustment followed by the
/// limiting one-sample CDF. There is no automatic/permutation/masked dispatch.
///
/// ## Domain and special values
///
/// At least two finite observations per sample; combined count≤1048576.
/// Exact mode additionally requires n+m≤20. Ties are accepted but make exact
/// continuous-null p-values conservative. NaN/infinite data are errors.
///
/// ## Returns
///
/// Nonnegative dimensionless T and greater-tail probability.
///
/// ## Errors
///
/// Invalid domains/work limits, nonfinite evaluations or exhausted limiting
/// CDF series return errors.
///
/// ## Numerical behavior
///
/// O((n+m) log(n+m)) ranking and O(n+m) memory. Exact work is at most
/// 184756 partitions of 20 ranks, accumulated as integer counts. Asymptotic
/// mode evaluates at most 256 Bessel-K series terms with 1e-7 term cutoff;
/// probabilities close to zero lose relative accuracy.
///
/// ## Examples
///
/// ```
/// let r = sci_stats::cramervonmises_2samp(&[1.,2.,3.], &[4.,5.,6.], sci_stats::CvmMethod::Exact)?;
/// assert!((0.0..=1.0).contains(&r.pvalue));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy cramervonmises_2samp and Anderson (1962)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.cramervonmises_2samp.html).
pub fn cramervonmises_2samp(x: &[f64], y: &[f64], method: CvmMethod) -> Result<TestResult> {
    sample(x, 2)?;
    sample(y, 2)?;
    let count = x.len() + y.len();
    if count > MAX_SAMPLES || (method == CvmMethod::Exact && count > 20) {
        return Err(Error::InvalidInput(
            "Cramer-von Mises sample limit exceeded",
        ));
    }
    let a = sorted(x)?;
    let b = sorted(y)?;
    let pooled = a.iter().chain(&b).copied().collect::<Vec<_>>();
    let ranks = rankdata(&pooled, RankMethod::Average)?;
    let u = cvm_rank_u(&ranks[..a.len()], &ranks[a.len()..]);
    let (n, m, total) = (x.len() as f64, y.len() as f64, count as f64);
    let product = n * m;
    let statistic = (u / (product * total) - (4. * product - 1.) / (6. * total)).max(0.);
    let pvalue = match method {
        CvmMethod::Asymptotic => {
            let expectation = (1. + 1. / total) / 6.;
            let variance = (total + 1.)
                * (4. * product * total - 3. * (n * n + m * m) - 2. * product)
                / (180. * total * total * product);
            let normalized = 1. / 6. + (statistic - expectation) / (45. * variance).sqrt();
            if normalized < 0.003 {
                1.
            } else {
                (1. - cvm_infinite(normalized)?).clamp(0., 1.)
            }
        }
        CvmMethod::Exact => {
            let mut partition = (0..x.len()).collect::<Vec<_>>();
            let (mut extreme, mut partitions) = (0_u64, 0_u64);
            loop {
                let mut first = Vec::with_capacity(x.len());
                let mut second = Vec::with_capacity(y.len());
                let mut i = 0;
                for r in 0..count {
                    if i < partition.len() && partition[i] == r {
                        first.push((r + 1) as f64);
                        i += 1;
                    } else {
                        second.push((r + 1) as f64);
                    }
                }
                extreme += u64::from(cvm_rank_u(&first, &second) >= u);
                partitions += 1;
                let Some(i) = (0..partition.len())
                    .rev()
                    .find(|&i| partition[i] < count - partition.len() + i)
                else {
                    break;
                };
                partition[i] += 1;
                for j in i + 1..partition.len() {
                    partition[j] = partition[j - 1] + 1;
                }
            }
            extreme as f64 / partitions as f64
        }
    };
    Ok(TestResult { statistic, pvalue })
}

/// Empirical-CDF convention for multi-sample Anderson–Darling tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AndersonVariant {
    /// Midranks, accommodating ties in both continuous and discrete samples.
    Midrank,
    /// Right-continuous empirical CDFs for discrete samples.
    Right,
}

/// Multi-sample Anderson–Darling equality-of-distributions test.
///
/// ## Parameters and conventions
///
/// `groups` contains k independent samples; `variant` selects midrank or
/// right-continuous empirical CDFs. For pooled distinct values z_j, multiplicities
/// l_j, cumulative pooled counts B_j, and sample cumulative counts M_ij,
/// right-continuous raw statistic is
/// `A=Σ_i (1/n_i) Σ_(j<last) (l_j/N)*(N*M_ij−n_i*B_j)²/[B_j*(N−B_j)]`.
/// Midrank subtracts half the corresponding tie counts from M_ij and B_j,
/// uses denominator `B_j*(N−B_j)−N*l_j/4`, includes all j, and multiplies by
/// (N−1)/N. Returned statistic is `(A−(k−1))/sqrt(Var(A))`, using the
/// Scholz–Stephens finite-sample variance (1987, Eq. 4). P-values fit a quadratic
/// to logarithms of tabulated probabilities, matching SciPy method=None.
/// New SciPy continuous variant, permutation calibration and critical-value
/// result objects are unsupported.
///
/// ## Domain and special values
///
/// 2..=16 nonempty finite groups, combined size 4..=65536, with at least two
/// distinct pooled observations. A nonpositive variance is an evaluation error.
/// The normalized statistic can be negative. NaN/infinity observations are errors.
///
/// ## Returns
///
/// Dimensionless normalized statistic and approximate p-value capped at 0.25
/// and floored at 0.001 outside the tabulated range, as in SciPy. Inside the
/// range the interpolated estimate need not exactly equal a tabulated probability.
///
/// ## Errors
///
/// Invalid group counts/domains/work limits, zero variance, or failed least-squares
/// calibration return errors.
///
/// ## Numerical behavior
///
/// O(N log N+k*N log N) work and O(N) memory; group size/work bounds are checked
/// before allocation. Private nalgebra SVD fits the seven-point calibration,
/// with at most 10,000 iterations; nonconvergence returns a numerical error.
/// P-values are approximate and can be poorly calibrated for small or heavily
/// tied samples; clipping is part of SciPy compatibility, not exact calibration.
///
/// ## Examples
///
/// ```
/// let groups: &[&[f64]] = &[&[1.,2.,3.], &[4.,5.,6.]];
/// let r = sci_stats::anderson_ksamp(groups, sci_stats::AndersonVariant::Midrank)?;
/// assert!((0.001..=0.25).contains(&r.pvalue));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy anderson_ksamp and Scholz–Stephens (1987)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson_ksamp.html).
pub fn anderson_ksamp(groups: &[&[f64]], variant: AndersonVariant) -> Result<TestResult> {
    if !(2..=16).contains(&groups.len()) {
        return Err(Error::InvalidInput("Anderson test needs 2..=16 groups"));
    }
    let count = inference::groups_check(groups, 2)?;
    if !(4..=65536).contains(&count) {
        return Err(Error::InvalidInput(
            "Anderson pooled count must be 4..=65536",
        ));
    }
    let pooled = sorted(
        &groups
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<_>>(),
    )?;
    let mut unique = pooled.clone();
    unique.dedup();
    if unique.len() < 2 {
        return Err(Error::InvalidInput(
            "Anderson test needs distinct pooled observations",
        ));
    }
    let n = count as f64;
    let k = groups.len() as f64;
    let mut raw = 0.;
    for group in groups {
        let ordered = sorted(group)?;
        let ni = group.len() as f64;
        for (j, &z) in unique.iter().enumerate() {
            if variant == AndersonVariant::Right && j + 1 == unique.len() {
                break;
            }
            let before = pooled.partition_point(|v| *v < z) as f64;
            let after = pooled.partition_point(|v| *v <= z) as f64;
            let lj = after - before;
            let mij_after = ordered.partition_point(|v| *v <= z) as f64;
            let (bj, mij, denominator) = match variant {
                AndersonVariant::Right => (after, mij_after, after * (n - after)),
                AndersonVariant::Midrank => {
                    let bj = before + lj / 2.;
                    let mij_before = ordered.partition_point(|v| *v < z) as f64;
                    (
                        bj,
                        (mij_before + mij_after) / 2.,
                        bj * (n - bj) - n * lj / 4.,
                    )
                }
            };
            raw += lj / n * (n * mij - bj * ni).powi(2) / (denominator * ni);
        }
    }
    if variant == AndersonVariant::Midrank {
        raw *= (n - 1.) / n;
    }
    let h_groups = groups.iter().map(|row| 1. / row.len() as f64).sum::<f64>();
    let (mut harmonic, mut g) = (0., 0.);
    for (i, r) in (2..count).rev().enumerate() {
        harmonic += 1. / r as f64;
        g += harmonic / (i + 2) as f64;
    }
    let h = harmonic + 1.;
    let a = (4. * g - 6.) * (k - 1.) + (10. - 6. * g) * h_groups;
    let b = (2. * g - 4.) * k * k + 8. * h * k + (2. * g - 14. * h - 4.) * h_groups - 8. * h
        + 4. * g
        - 6.;
    let c = (6. * h + 2. * g - 2.) * k * k
        + (4. * h - 4. * g + 6.) * k
        + (2. * h - 6.) * h_groups
        + 4. * h;
    let d = (2. * h + 6.) * k * k - 4. * h * k;
    let variance = (a * n * n * n + b * n * n + c * n + d) / ((n - 1.) * (n - 2.) * (n - 3.));
    positive(variance)?;
    let statistic = match (raw - (k - 1.)) / variance.sqrt() {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }?;
    let m = k - 1.;
    let b0 = [0.675, 1.281, 1.645, 1.96, 2.326, 2.573, 3.085];
    let b1 = [-0.245, 0.25, 0.678, 1.149, 1.822, 2.364, 3.615];
    let b2 = [-0.105, -0.305, -0.362, -0.391, -0.396, -0.345, -0.154];
    let critical: Vec<f64> = (0..7)
        .map(|i| b0[i] + b1[i] / m.sqrt() + b2[i] / m)
        .collect();
    let pvalue = if statistic < critical[0] {
        0.25
    } else if statistic > critical[6] {
        0.001
    } else {
        // Use SVD least squares: nalgebra's dynamic QR solve requires a square system.
        let design = nalgebra::DMatrix::from_fn(7, 3, |i, j| match j {
            0 => critical[i] * critical[i],
            1 => critical[i],
            _ => 1.,
        });
        let response = nalgebra::DVector::from_vec(vec![
            0.25_f64.ln(),
            0.1_f64.ln(),
            0.05_f64.ln(),
            0.025_f64.ln(),
            0.01_f64.ln(),
            0.005_f64.ln(),
            0.001_f64.ln(),
        ]);
        let coefficients = design
            .try_svd(true, true, f64::EPSILON, 10_000)
            .ok_or(Error::NumericalFailure(
                "Anderson calibration SVD did not converge",
            ))?
            .solve(&response, 1e-14)
            .map_err(|_| Error::NumericalFailure("Anderson calibration fit failed"))?;
        match (coefficients[0] * statistic * statistic
            + coefficients[1] * statistic
            + coefficients[2])
            .exp()
        {
            value if value.is_finite() => Ok(value),
            _ => Err(crate::Error::NonFiniteEvaluation),
        }?
    };
    Ok(TestResult { statistic, pvalue })
}
