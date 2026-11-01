use super::*;

/// Reduction performed within bins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinStatistic {
    /// Number of observations; ignores their values after validation.
    Count,
    /// Sum; empty bins are zero.
    Sum,
    /// Arithmetic mean; empty bins are NaN.
    Mean,
    /// Population standard deviation (ddof=0); empty bins are NaN.
    Std,
    /// Median; empty bins are NaN.
    Median,
    /// Minimum; empty bins are NaN.
    Min,
    /// Maximum; empty bins are NaN.
    Max,
}
/// Cartesian binned reduction result.
#[derive(Clone, Debug, PartialEq)]
pub struct BinnedStatistic {
    /// Flat row-major reductions, last dimension varies fastest.
    pub statistic: Vec<f64>,
    /// Number of interior bins per dimension.
    pub shape: Vec<usize>,
    /// Supplied bin edges, in coordinate order.
    pub edges: Vec<Vec<f64>>,
    /// SciPy flat bin numbers for each observation, including extra outlier bins.
    /// Each coordinate bin is zero below, 1..=shape within, shape+1 above.
    pub bin_numbers: Vec<usize>,
}
/// Uniform-bin frequency histogram.
#[derive(Clone, Debug, PartialEq)]
pub struct Frequency {
    /// Cumulative counts or relative frequencies, in increasing bin order.
    pub frequencies: Vec<f64>,
    /// Lower edge of the first bin.
    pub lower_limit: f64,
    /// Width of each uniform bin.
    pub bin_size: f64,
    /// Number of observations below the lower or above the upper edge.
    pub extra_points: usize,
}
fn bin(x: f64, edges: &[f64]) -> usize {
    if x < edges[0] {
        0
    } else if x > edges[edges.len() - 1] {
        edges.len()
    } else if x == edges[edges.len() - 1] {
        edges.len() - 1
    } else {
        edges.partition_point(|v| *v <= x)
    }
}
/// Cartesian bin reduction: aggregate values whose coordinates fall in each product of edge intervals.
///
/// ## Parameters
///
/// Rows are points; edges are one increasing edge vector per coordinate. Bins are [left,right), with the final bin closed on the right.
/// Count/sum of empty bins is zero; other reductions are NaN. Bin numbers include an extra outlier bin on both sides of every dimension, as in SciPy.
///
/// ## Domain
///
/// Finite rectangular observations, 1..=16 dimensions; finite matching values; each edge vector strictly increasing with ≥2 edges. Sample coordinates and output each≤1048576 entries.
///
/// ## Returns
///
/// Flat row-major statistics, per-coordinate shape and edges, and flat SciPy bin numbers in observation order.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// O(n*d*log(max bins)) assignments; median additionally sorts each occupied bin. Exact edge comparisons; SciPy's rounded-right-edge tolerance is not applied.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// use sci_stats::{binning::binned_statistic_dd, binning::BinStatistic};
/// assert_eq!(binned_statistic_dd(&[vec![0.5, 0.5]], &[3.], &[vec![0.,1.],vec![0.,1.]], BinStatistic::Sum)?.statistic, vec![3.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `binned_statistic_dd`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic_dd.html).
pub fn binned_statistic_dd(
    points: &[Vec<f64>],
    values: &[f64],
    edges: &[Vec<f64>],
    statistic: BinStatistic,
) -> Result<BinnedStatistic> {
    let d = edges.len();
    if d == 0 || d > 16 || points.len() != values.len() {
        return Err(Error::DimensionMismatch);
    }
    sample(values, 0)?;
    if points.len().checked_mul(d).is_none_or(|n| n > MAX_SAMPLES) {
        return Err(Error::InvalidInput("sample coordinates exceed limit"));
    }
    let mut shape = Vec::with_capacity(d);
    let mut cells = 1_usize;
    let mut full_cells = 1_usize;
    for edge in edges {
        sample(edge, 2)?;
        if edge.windows(2).any(|w| w[0] >= w[1]) {
            return Err(Error::InvalidInput("bin edges must strictly increase"));
        }
        let n = edge.len() - 1;
        shape.push(n);
        cells = cells.checked_mul(n).ok_or(Error::DimensionMismatch)?;
        full_cells = full_cells
            .checked_mul(n + 2)
            .ok_or(Error::DimensionMismatch)?;
        if cells > MAX_SAMPLES || full_cells > MAX_SAMPLES {
            return Err(Error::InvalidInput(
                "bin grid exceeds 1048576 cells including outliers",
            ));
        }
    }
    let mut buckets = vec![Vec::new(); cells];
    let mut bin_numbers = Vec::with_capacity(points.len());
    for (row, &value) in points.iter().zip(values) {
        if row.len() != d {
            return Err(Error::DimensionMismatch);
        }
        sample(row, d)?;
        let mut flat = 0;
        let mut inside = true;
        let mut number = 0;
        for j in 0..d {
            let k = bin(row[j], &edges[j]);
            number = number * (shape[j] + 2) + k;
            if k == 0 || k == shape[j] + 1 {
                inside = false;
            }
            if k > 0 {
                flat = flat * shape[j] + k - 1;
            }
        }
        bin_numbers.push(number);
        if inside {
            buckets[flat].push(value);
        }
    }
    let mut out = Vec::with_capacity(cells);
    for b in buckets {
        out.push(if b.is_empty() {
            if matches!(statistic, BinStatistic::Count | BinStatistic::Sum) {
                0.
            } else {
                f64::NAN
            }
        } else {
            match statistic {
                BinStatistic::Count => b.len() as f64,
                BinStatistic::Sum => match b.iter().sum::<f64>() {
                    value if value.is_finite() => Ok(value),
                    _ => Err(crate::Error::NonFiniteEvaluation),
                }?,
                BinStatistic::Mean => mean(&b)?,
                BinStatistic::Std => std(&b, 0)?,
                BinStatistic::Median => median(&b)?,
                BinStatistic::Min => b.iter().copied().fold(f64::INFINITY, f64::min),
                BinStatistic::Max => b.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            }
        });
    }
    Ok(BinnedStatistic {
        statistic: out,
        shape,
        edges: edges.to_vec(),
        bin_numbers,
    })
}
/// One-dimensional binned reduction on intervals of explicit edges.
///
/// ## Parameters
///
/// Coordinates and values pair by index; intervals [left,right), final interval closed; explicit bin edges only.
///
/// ## Domain
///
/// Finite matching sample lengths and increasing finite edges; grid including outliers≤1048576.
///
/// ## Returns
///
/// Statistic vector, one-dimensional shape/edges, and one bin number per observation.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Assignment uses binary search in increasing edges; reductions use the shared descriptive APIs.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// use sci_stats::{binning::binned_statistic, binning::BinStatistic};
/// assert_eq!(binned_statistic(&[0.5, 1.5], &[2., 4.], &[0., 1., 2.], BinStatistic::Mean)?.statistic, vec![2., 4.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `binned_statistic`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html).
pub fn binned_statistic(
    coordinates: &[f64],
    values: &[f64],
    edges: &[f64],
    statistic: BinStatistic,
) -> Result<BinnedStatistic> {
    sample(coordinates, 0)?;
    let points = coordinates.iter().map(|&x| vec![x]).collect::<Vec<_>>();
    binned_statistic_dd(&points, values, &[edges.to_vec()], statistic)
}
/// Two-dimensional Cartesian binned reduction.
///
/// ## Parameters
///
/// x/y and values pair by index; x is the first dimension, y the second; y varies fastest in flat outputs.
///
/// ## Domain
///
/// Finite matching samples and increasing finite edges; total coordinates/grid including outliers≤1048576.
///
/// ## Returns
///
/// Statistics with shape [x bins,y bins], edge vectors in x/y order, flat bin numbers in observation order.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Assignment uses binary search in increasing edges; reductions use the shared descriptive APIs.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// use sci_stats::{binning::binned_statistic_2d, binning::BinStatistic};
/// assert_eq!(binned_statistic_2d(&[0.5], &[0.5], &[3.], &[0.,1.], &[0.,1.], BinStatistic::Sum)?.shape, vec![1,1]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `binned_statistic_2d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic_2d.html).
pub fn binned_statistic_2d(
    x: &[f64],
    y: &[f64],
    values: &[f64],
    x_edges: &[f64],
    y_edges: &[f64],
    statistic: BinStatistic,
) -> Result<BinnedStatistic> {
    if x.len() != y.len() {
        return Err(Error::DimensionMismatch);
    }
    sample(x, 0)?;
    sample(y, 0)?;
    let points = x
        .iter()
        .zip(y)
        .map(|(&a, &b)| vec![a, b])
        .collect::<Vec<_>>();
    binned_statistic_dd(
        &points,
        values,
        &[x_edges.to_vec(), y_edges.to_vec()],
        statistic,
    )
}
/// Cumulative uniform-bin frequencies `C_j=Σ_{k≤j}count(bin_k)`.
///
/// ## Parameters
///
/// Explicit bin count and finite lower/upper limits; final bin includes upper. Out-of-range values are counted in extra_points. No automatic range.
///
/// ## Domain
///
/// Finite nonempty sample, bins in \[1,1048574\], finite lower<upper with positive finite representable bin width.
///
/// ## Returns
///
/// Ordered frequencies, lower edge, bin width, and out-of-range count.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Assignment uses binary search in increasing edges; reductions use the shared descriptive APIs.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::binning::cumfreq(&[0., 1., 2.], 2, 0., 2.)?.extra_points, 0);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `cumfreq`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.cumfreq.html).
pub fn cumfreq(values: &[f64], bins: usize, lower: f64, upper: f64) -> Result<Frequency> {
    sample(values, 1)?;
    if bins == 0 || bins > MAX_SAMPLES - 2 {
        return Err(Error::InvalidInput("invalid bin count"));
    }
    if !lower.is_finite() {
        return Err(crate::Error::NonFiniteEvaluation);
    }
    if !upper.is_finite() {
        return Err(crate::Error::NonFiniteEvaluation);
    }
    positive(upper - lower)?;
    let width = (upper - lower) / bins as f64;
    positive(width)?;
    let edges = (0..=bins)
        .map(|i| {
            if i == bins {
                upper
            } else {
                lower + i as f64 * width
            }
        })
        .collect::<Vec<_>>();
    let result = binned_statistic(values, values, &edges, BinStatistic::Count)?;
    let extra_points = result
        .bin_numbers
        .iter()
        .filter(|&&k| k == 0 || k == bins + 1)
        .count();
    let mut frequencies = result.statistic;
    let mut sum = 0.;
    for v in &mut frequencies {
        sum += *v;
        *v = sum;
    }
    Ok(Frequency {
        frequencies,
        lower_limit: lower,
        bin_size: width,
        extra_points,
    })
}
/// Relative uniform-bin frequencies `f_j=count(bin_j)/n`.
///
/// ## Parameters
///
/// Explicit bin count and finite lower/upper limits; final bin includes upper. Out-of-range values are counted in extra_points. No automatic range.
///
/// ## Domain
///
/// Finite nonempty sample, bins in \[1,1048574\], finite lower<upper with positive finite representable bin width.
///
/// ## Returns
///
/// Ordered frequencies, lower edge, bin width, and out-of-range count.
///
/// ## Errors
///
/// Returns an error for invalid domains, nonfinite inputs, incompatible shapes,
/// or exceeded work limits, where applicable. Arithmetic overflow returns an
/// evaluation error; documented NaN and infinity results are permitted.
///
/// ## Numerical behavior
///
/// Assignment uses binary search in increasing edges; reductions use the shared descriptive APIs.
/// Inputs are limited to 1,048,576 observations. No Python runtime is used.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_stats::binning::relfreq(&[0., 1., 2.], 2, 0., 2.)?.extra_points, 0);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `relfreq`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.relfreq.html).
pub fn relfreq(values: &[f64], bins: usize, lower: f64, upper: f64) -> Result<Frequency> {
    sample(values, 1)?;
    if bins == 0 || bins > MAX_SAMPLES - 2 {
        return Err(Error::InvalidInput("invalid bin count"));
    }
    if !lower.is_finite() {
        return Err(crate::Error::NonFiniteEvaluation);
    }
    if !upper.is_finite() {
        return Err(crate::Error::NonFiniteEvaluation);
    }
    positive(upper - lower)?;
    let width = (upper - lower) / bins as f64;
    positive(width)?;
    let edges = (0..=bins)
        .map(|i| {
            if i == bins {
                upper
            } else {
                lower + i as f64 * width
            }
        })
        .collect::<Vec<_>>();
    let result = binned_statistic(values, values, &edges, BinStatistic::Count)?;
    let extra_points = result
        .bin_numbers
        .iter()
        .filter(|&&k| k == 0 || k == bins + 1)
        .count();
    let mut frequencies = result.statistic;
    for v in &mut frequencies {
        *v /= values.len() as f64;
    }
    Ok(Frequency {
        frequencies,
        lower_limit: lower,
        bin_size: width,
        extra_points,
    })
}
