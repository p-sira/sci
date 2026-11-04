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
    let edge_refs = edges.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let (statistic, shape, bin_numbers) = binned_core(
        points.len(),
        |i, j| points[i][j],
        values,
        &edge_refs,
        statistic,
        || {
            for row in points {
                if row.len() != d {
                    return Err(Error::DimensionMismatch);
                }
                sample(row, d)?;
            }
            Ok(())
        },
    )?;
    Ok(BinnedStatistic {
        statistic,
        shape,
        edges: edges.to_vec(),
        bin_numbers,
    })
}
/// Statistic, shape and bin numbers produced by [`binned_core`].
type BinnedParts = (Vec<f64>, Vec<usize>, Vec<usize>);
/// Shared Cartesian binning kernel.
///
/// Coordinates are read through `coord(observation, dimension)` so callers do
/// not need to materialize one row vector per observation. Validation order
/// matches the public `binned_statistic_dd` contract: values, coordinate
/// limit, edges, then rows via `validate_rows`. Values are grouped per bin with
/// a stable counting sort, so every reduction sees exactly the same values in
/// the same observation order as a per-bin vector would.
fn binned_core(
    n: usize,
    coord: impl Fn(usize, usize) -> f64,
    values: &[f64],
    edges: &[&[f64]],
    statistic: BinStatistic,
    validate_rows: impl FnOnce() -> Result<()>,
) -> Result<BinnedParts> {
    let d = edges.len();
    sample(values, 0)?;
    if n.checked_mul(d).is_none_or(|n| n > MAX_SAMPLES) {
        return Err(Error::InvalidInput("sample coordinates exceed limit"));
    }
    let mut shape = Vec::with_capacity(d);
    let mut cells = 1_usize;
    let mut full_cells = 1_usize;
    for &edge in edges {
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
    validate_rows()?;
    const OUTSIDE: usize = usize::MAX;
    let mut counts = vec![0_usize; cells];
    let mut flats = Vec::with_capacity(n);
    let mut bin_numbers = Vec::with_capacity(n);
    for i in 0..n {
        let mut flat = 0;
        let mut inside = true;
        let mut number = 0;
        for j in 0..d {
            let k = bin(coord(i, j), edges[j]);
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
            counts[flat] += 1;
            flats.push(flat);
        } else {
            flats.push(OUTSIDE);
        }
    }
    let empty = if matches!(statistic, BinStatistic::Count | BinStatistic::Sum) {
        0.
    } else {
        f64::NAN
    };
    if statistic == BinStatistic::Count {
        let out = counts
            .iter()
            .map(|&c| if c == 0 { empty } else { c as f64 })
            .collect();
        return Ok((out, shape, bin_numbers));
    }
    // Stable counting sort of in-range values into contiguous per-bin runs.
    let mut starts = Vec::with_capacity(cells + 1);
    let mut total = 0;
    starts.push(0);
    for &c in &counts {
        total += c;
        starts.push(total);
    }
    let mut cursor = starts[..cells].to_vec();
    let mut grouped = vec![0.; total];
    for (&flat, &value) in flats.iter().zip(values) {
        if flat != OUTSIDE {
            grouped[cursor[flat]] = value;
            cursor[flat] += 1;
        }
    }
    let mut out = Vec::with_capacity(cells);
    for cell in 0..cells {
        let b = &grouped[starts[cell]..starts[cell + 1]];
        out.push(if b.is_empty() {
            empty
        } else {
            match statistic {
                BinStatistic::Count => b.len() as f64,
                BinStatistic::Sum => match b.iter().sum::<f64>() {
                    value if value.is_finite() => Ok(value),
                    _ => Err(crate::Error::NonFiniteEvaluation),
                }?,
                BinStatistic::Mean => mean(b)?,
                BinStatistic::Std => std(b, 0)?,
                BinStatistic::Median => median(b)?,
                BinStatistic::Min => b.iter().copied().fold(f64::INFINITY, f64::min),
                BinStatistic::Max => b.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            }
        });
    }
    Ok((out, shape, bin_numbers))
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
    if coordinates.len() != values.len() {
        return Err(Error::DimensionMismatch);
    }
    // Coordinates were validated above, so the per-row checks are no-ops.
    let (statistic, shape, bin_numbers) = binned_core(
        coordinates.len(),
        |i, _| coordinates[i],
        values,
        &[edges],
        statistic,
        || Ok(()),
    )?;
    Ok(BinnedStatistic {
        statistic,
        shape,
        edges: vec![edges.to_vec()],
        bin_numbers,
    })
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
    if x.len() != values.len() {
        return Err(Error::DimensionMismatch);
    }
    // Coordinates were validated above, so the per-row checks are no-ops.
    let (statistic, shape, bin_numbers) = binned_core(
        x.len(),
        |i, j| if j == 0 { x[i] } else { y[i] },
        values,
        &[x_edges, y_edges],
        statistic,
        || Ok(()),
    )?;
    Ok(BinnedStatistic {
        statistic,
        shape,
        edges: vec![x_edges.to_vec(), y_edges.to_vec()],
        bin_numbers,
    })
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
#[cfg(test)]
mod tests {
    use super::*;

    /// Previous per-observation/per-bin vector implementation, kept as an oracle.
    fn reference(
        points: &[Vec<f64>],
        values: &[f64],
        edges: &[Vec<f64>],
        statistic: BinStatistic,
    ) -> (Vec<f64>, Vec<usize>) {
        let shape = edges.iter().map(|e| e.len() - 1).collect::<Vec<_>>();
        let cells = shape.iter().product();
        let mut buckets = vec![Vec::new(); cells];
        let mut numbers = Vec::new();
        for (row, &value) in points.iter().zip(values) {
            let (mut flat, mut inside, mut number) = (0, true, 0);
            for j in 0..edges.len() {
                let k = bin(row[j], &edges[j]);
                number = number * (shape[j] + 2) + k;
                inside &= k != 0 && k != shape[j] + 1;
                if k > 0 {
                    flat = flat * shape[j] + k - 1;
                }
            }
            numbers.push(number);
            if inside {
                buckets[flat].push(value);
            }
        }
        let out = buckets
            .iter()
            .map(|b| match (b.is_empty(), statistic) {
                (true, BinStatistic::Count | BinStatistic::Sum) => 0.,
                (true, _) => f64::NAN,
                (false, BinStatistic::Count) => b.len() as f64,
                (false, BinStatistic::Sum) => b.iter().sum(),
                (false, BinStatistic::Mean) => mean(b).unwrap(),
                (false, BinStatistic::Std) => std(b, 0).unwrap(),
                (false, BinStatistic::Median) => median(b).unwrap(),
                (false, BinStatistic::Min) => b.iter().copied().fold(f64::INFINITY, f64::min),
                (false, BinStatistic::Max) => b.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            })
            .collect();
        (out, numbers)
    }

    fn same_bits(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    #[test]
    fn grouped_kernel_matches_per_bin_vectors_bit_for_bit() {
        let n = 257;
        let x = (0..n)
            .map(|i| ((i * 37 % 101) as f64 - 50.) / 23.)
            .collect::<Vec<_>>();
        let y = (0..n)
            .map(|i| ((i * 53 % 89) as f64 - 44.) / 17.)
            .collect::<Vec<_>>();
        let v = (0..n)
            .map(|i| ((i * 29 % 61) as f64).sin() * 1e3 + 0.1 * i as f64)
            .collect::<Vec<_>>();
        let ex = vec![-1.5, -0.7, 0., 0.3, 1.0, 1.5];
        let ey = vec![-2., -1., 0., 1., 2.];
        let stats = [
            BinStatistic::Count,
            BinStatistic::Sum,
            BinStatistic::Mean,
            BinStatistic::Std,
            BinStatistic::Median,
            BinStatistic::Min,
            BinStatistic::Max,
        ];
        let p1 = x.iter().map(|&a| vec![a]).collect::<Vec<_>>();
        let p2 = x
            .iter()
            .zip(&y)
            .map(|(&a, &b)| vec![a, b])
            .collect::<Vec<_>>();
        let p3 = (0..n)
            .map(|i| vec![x[i], y[i], v[i] / 1e3])
            .collect::<Vec<_>>();
        let e3 = vec![ex.clone(), ey.clone(), vec![-1., -0.25, 0.5, 1.]];
        for s in stats {
            let (want, nums) = reference(&p1, &v, std::slice::from_ref(&ex), s);
            let got = binned_statistic(&x, &v, &ex, s).unwrap();
            assert!(same_bits(&got.statistic, &want), "{s:?} 1d");
            assert_eq!(got.bin_numbers, nums);
            assert_eq!(got.edges, vec![ex.clone()]);
            let dd = binned_statistic_dd(&p1, &v, std::slice::from_ref(&ex), s).unwrap();
            assert_eq!(dd.bin_numbers, got.bin_numbers);
            assert!(same_bits(&dd.statistic, &got.statistic));

            let edges2 = vec![ex.clone(), ey.clone()];
            let (want, nums) = reference(&p2, &v, &edges2, s);
            let got = binned_statistic_2d(&x, &y, &v, &ex, &ey, s).unwrap();
            assert!(same_bits(&got.statistic, &want), "{s:?} 2d");
            assert_eq!(got.bin_numbers, nums);
            assert_eq!(got.shape, vec![5, 4]);
            assert_eq!(got.edges, edges2);

            let (want, nums) = reference(&p3, &v, &e3, s);
            let got = binned_statistic_dd(&p3, &v, &e3, s).unwrap();
            assert!(same_bits(&got.statistic, &want), "{s:?} 3d");
            assert_eq!(got.bin_numbers, nums);
        }
    }

    #[test]
    fn wrappers_keep_validation_errors() {
        let e = [0., 1.];
        assert!(matches!(
            binned_statistic(&[0.5], &[1., 2.], &e, BinStatistic::Sum),
            Err(Error::DimensionMismatch)
        ));
        assert!(binned_statistic(&[f64::NAN], &[1.], &e, BinStatistic::Sum).is_err());
        assert!(binned_statistic(&[0.5], &[1.], &[1., 0.], BinStatistic::Sum).is_err());
        assert!(matches!(
            binned_statistic_2d(&[0.5], &[0.5], &[1., 2.], &e, &e, BinStatistic::Sum),
            Err(Error::DimensionMismatch)
        ));
        assert!(matches!(
            binned_statistic_dd(
                &[vec![0.5]],
                &[1.],
                &[e.to_vec(), e.to_vec()],
                BinStatistic::Sum
            ),
            Err(Error::DimensionMismatch)
        ));
        assert!(
            binned_statistic(&[0.5, 0.5], &[f64::MAX, f64::MAX], &e, BinStatistic::Sum).is_err()
        );
    }
}
