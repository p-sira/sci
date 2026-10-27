//! Two-dimensional contingency tables in row-major order.
use super::*;

/// Chi-square independence result and fitted frequencies.
#[derive(Clone, Debug, PartialEq)]
pub struct ContingencyResult {
    /// Pearson chi-square statistic, optionally corrected by Yates.
    pub statistic: f64,
    /// Upper-tail chi-square probability.
    pub pvalue: f64,
    /// Degrees of freedom (rows−1)(columns−1).
    pub df: usize,
    /// Expected cell counts in input row-major order.
    pub expected: Vec<f64>,
}
/// Association coefficient convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Association {
    /// Cramer's V: sqrt(chi²/(N*min(rows−1,columns−1))).
    Cramer,
    /// Tschuprow's T: sqrt(chi²/(N*sqrt((rows−1)*(columns−1)))).
    Tschuprow,
    /// Pearson's contingency coefficient: sqrt(chi²/(N+chi²)).
    Pearson,
}
/// Two-dimensional cross-tabulation of numeric categories.
#[derive(Clone, Debug, PartialEq)]
pub struct Crosstab {
    /// Sorted distinct categories of the first sample (rows).
    pub rows: Vec<f64>,
    /// Sorted distinct categories of the second sample (columns).
    pub columns: Vec<f64>,
    /// Row-major counts, rows.len()*columns.len() entries.
    pub counts: Vec<f64>,
}
fn table_check(table: &[f64], rows: usize, columns: usize) -> Result<()> {
    if rows == 0 || columns == 0 || rows.checked_mul(columns) != Some(table.len()) {
        return Err(Error::DimensionMismatch);
    }
    sample(table, 1)?;
    if table.iter().any(|x| *x < 0.) {
        return Err(Error::InvalidInput("cell counts must be nonnegative"));
    }
    Ok(())
}
/// Marginal sums `r_i=Σ_j t_ij`, `c_j=Σ_i t_ij`.
///
/// ## Parameters
///
/// Table is flat row-major, with explicit rows and columns; no higher-dimensional overload.
///
/// ## Domain
///
/// Nonempty finite nonnegative cell counts with matching shape.
///
/// ## Returns
///
/// Tuple (row sums, column sums), with lengths rows and columns.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// O(rows*columns) binary64 computation; arithmetic overflow returns an error.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::contingency::margins(&[1., 2., 3., 4.], 2, 2)?.0, vec![3., 7.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `contingency.margins`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.margins.html).
pub fn margins(table: &[f64], rows: usize, columns: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    table_check(table, rows, columns)?;
    let mut r = vec![0.; rows];
    let mut c = vec![0.; columns];
    for i in 0..rows {
        for j in 0..columns {
            r[i] += table[i * columns + j];
            c[j] += table[i * columns + j];
        }
    }
    is_finite_slice(&r).map_err(|_| Error::NonFiniteEvaluation)?;
    is_finite_slice(&c).map_err(|_| Error::NonFiniteEvaluation)?;
    Ok((r, c))
}
/// Independent expected counts `E_ij=row_sum_i*column_sum_j/total`.
///
/// ## Parameters
///
/// Flat row-major table and explicit dimensions.
///
/// ## Domain
///
/// Nonnegative finite cells with positive total; zero margins give zero expected cells.
///
/// ## Returns
///
/// Expected counts with the input shape and row-major ordering.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// O(rows*columns) binary64 computation; arithmetic overflow returns an error.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::contingency::expected_freq(&[2., 2., 2., 2.], 2, 2)?, vec![2.;4]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `contingency.expected_freq`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.expected_freq.html).
pub fn expected_freq(table: &[f64], rows: usize, columns: usize) -> Result<Vec<f64>> {
    let (r, c) = margins(table, rows, columns)?;
    let total = match r.iter().sum::<f64>() {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }?;
    positive(total)?;
    let mut out = Vec::with_capacity(table.len());
    for a in r {
        for &b in &c {
            out.push(match (a / total) * b {
                value if value.is_finite() => Ok(value),
                _ => Err(crate::Error::NonFiniteEvaluation),
            }?);
        }
    }
    Ok(out)
}
/// Chi-square independence test `χ²=Σ(O−E)²/E`, df=(rows−1)(columns−1).
///
/// ## Parameters
///
/// Applies Yates correction only when df=1: subtract min(0.5,|O−E|) from each absolute deviation. Pearson statistic only.
///
/// ## Domain
///
/// Finite nonnegative cells, positive expected cells; singleton dimensions return statistic=0,p=1,df=0.
///
/// ## Returns
///
/// Statistic, asymptotic p-value, df, and expected frequencies.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// O(rows*columns) binary64 computation; arithmetic overflow returns an error.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::chi2_contingency(&[2., 2., 2., 2.], 2, 2, true)?.pvalue, 1.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `contingency.chi2_contingency`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.chi2_contingency.html).
pub fn chi2_contingency(
    table: &[f64],
    rows: usize,
    columns: usize,
    correction: bool,
) -> Result<ContingencyResult> {
    let expected = expected_freq(table, rows, columns)?;
    if expected.contains(&0.) {
        return Err(Error::InvalidInput("expected cells must be positive"));
    }
    let df = (rows - 1) * (columns - 1);
    if df == 0 {
        return Ok(ContingencyResult {
            statistic: 0.,
            pvalue: 1.,
            df,
            expected,
        });
    }
    let statistic = match table
        .iter()
        .zip(&expected)
        .map(|(o, e)| {
            let mut d = (o - e).abs();
            if correction && df == 1 {
                d = (d - 0.5).max(0.);
            }
            d * d / e
        })
        .sum::<f64>()
    {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }?;
    let pvalue = super::sf(Distribution::ChiSquared { df: df as f64 }, statistic)?;
    Ok(ContingencyResult {
        statistic,
        pvalue,
        df,
        expected,
    })
}
/// Nominal association coefficient derived from the Pearson chi-square statistic.
///
/// ## Parameters
///
/// Method selects Cramer V, Tschuprow T, or Pearson contingency coefficient; row-major table.
///
/// ## Domain
///
/// At least two rows and columns; nonnegative finite cells with positive expectations.
///
/// ## Returns
///
/// Nonnegative dimensionless coefficient.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// O(rows*columns) binary64 computation; arithmetic overflow returns an error.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// use sci_stats::contingency::{association, Association};
/// assert_eq!(association(&[2., 2., 2., 2.], 2, 2, Association::Cramer, false)?, 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `contingency.association`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.association.html).
pub fn association(
    table: &[f64],
    rows: usize,
    columns: usize,
    method: Association,
    correction: bool,
) -> Result<f64> {
    if rows < 2 || columns < 2 {
        return Err(Error::InvalidInput(
            "association requires at least a 2x2 table",
        ));
    }
    let chi = chi2_contingency(table, rows, columns, correction)?.statistic;
    let n = match table.iter().sum::<f64>() {
        value if value.is_finite() => Ok(value),
        _ => Err(crate::Error::NonFiniteEvaluation),
    }?;
    Ok(match method {
        Association::Cramer => (chi / n / (rows - 1).min(columns - 1) as f64).sqrt(),
        Association::Tschuprow => (chi / n / (((rows - 1) * (columns - 1)) as f64).sqrt()).sqrt(),
        Association::Pearson => (chi / (n + chi)).sqrt(),
    })
}
/// Sample odds ratio `a*d/(b*c)` for a 2×2 table.
///
/// ## Parameters
///
/// Row-major order `[a,b,c,d]`. Implements SciPy kind="sample"; conditional MLE and confidence intervals are unsupported.
///
/// ## Domain
///
/// Nonnegative integer finite cells, each≤1048576. Degenerate margins return NaN; a zero denominator with positive numerator returns infinity.
///
/// ## Returns
///
/// Scalar sample odds ratio.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// O(rows*columns) binary64 computation; arithmetic overflow returns an error.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::contingency::odds_ratio(&[2., 1., 1., 2.])?, 4.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `contingency.odds_ratio`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.odds_ratio.html).
pub fn odds_ratio(table: &[f64; 4]) -> Result<f64> {
    for &x in table {
        if !x.is_finite() || !(0. ..=MAX_SAMPLES as f64).contains(&x) || x.fract() != 0. {
            return Err(Error::InvalidInput(
                "odds ratio needs bounded nonnegative integer counts",
            ));
        }
    }
    let [a, b, c, d] = *table;
    if a + b == 0. || c + d == 0. || a + c == 0. || b + d == 0. {
        return Ok(f64::NAN);
    }
    Ok(a * d / (b * c))
}
/// Relative risk `(exposed_cases/exposed_total)/(control_cases/control_total)`.
///
/// ## Parameters
///
/// Counts are explicit; no table orientation ambiguity; confidence intervals unsupported.
///
/// ## Domain
///
/// Integer counts in \[0,1048576\], positive totals, cases≤total. Zero control risk yields infinity or NaN.
///
/// ## Returns
///
/// Risk ratio, dimensionless.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// O(rows*columns) binary64 computation; arithmetic overflow returns an error.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::contingency::relative_risk(2., 4., 1., 4.)?, 2.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `contingency.relative_risk`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.relative_risk.html).
pub fn relative_risk(
    exposed_cases: f64,
    exposed_total: f64,
    control_cases: f64,
    control_total: f64,
) -> Result<f64> {
    for x in [exposed_cases, exposed_total, control_cases, control_total] {
        if !x.is_finite() || !(0. ..=MAX_SAMPLES as f64).contains(&x) || x.fract() != 0. {
            return Err(Error::InvalidInput(
                "relative risk needs bounded nonnegative integer counts",
            ));
        }
    }
    positive(exposed_total)?;
    positive(control_total)?;
    if exposed_cases > exposed_total || control_cases > control_total {
        return Err(Error::InvalidInput("cases exceed total"));
    }
    Ok((exposed_cases / exposed_total) / (control_cases / control_total))
}
/// Cross-tabulation `T_ij=count(first=row_i and second=column_j)`.
///
/// ## Parameters
///
/// Two finite numeric category vectors; categories sorted ascending; no sparse or explicit-level overload.
///
/// ## Domain
///
/// Equal-length nonempty samples; output at most 1048576 cells. Signed zeros are one category.
///
/// ## Returns
///
/// Sorted category lists and row-major counts.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// O(rows*columns) binary64 computation; arithmetic overflow returns an error.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::contingency::crosstab(&[1., 2.], &[3., 3.])?.counts, vec![1., 1.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `contingency.crosstab`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.crosstab.html).
pub fn crosstab(first: &[f64], second: &[f64]) -> Result<Crosstab> {
    if first.len() != second.len() {
        return Err(Error::DimensionMismatch);
    }
    let mut rows = sorted(first)?;
    rows.dedup();
    let mut columns = sorted(second)?;
    columns.dedup();
    let size = rows
        .len()
        .checked_mul(columns.len())
        .ok_or(Error::DimensionMismatch)?;
    if size > MAX_SAMPLES {
        return Err(Error::InvalidInput("crosstab output exceeds limit"));
    }
    let mut counts = vec![0.; size];
    for (&a, &b) in first.iter().zip(second) {
        let i = rows
            .binary_search_by(|v| v.partial_cmp(&a).unwrap())
            .unwrap();
        let j = columns
            .binary_search_by(|v| v.partial_cmp(&b).unwrap())
            .unwrap();
        counts[i * columns.len() + j] += 1.;
    }
    Ok(Crosstab {
        rows,
        columns,
        counts,
    })
}
