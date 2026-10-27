use super::{nonempty, samples, size, work};
use crate::{Error, Result};

/// Boundary convention for relative extrema.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeMode {
    /// Repeat the nearest endpoint sample.
    Clip,
    /// Wrap periodically to the other endpoint.
    Wrap,
}
/// Returns indices satisfying `compare(x[i], x[i±j])` for every j=1..order.
///
/// ## Parameters
///
/// `pub fn argrelextrema(x:&[f64],compare:impl Fn(f64,f64)->bool,order:usize,mode:EdgeMode)->Result<Vec<usize>>`
///
/// Scalar comparison closure replaces SciPy vector comparator. EdgeMode selects clipped or periodic boundaries. Returns indices in ascending order; no multidimensional axis dispatch.
///
/// ## Domain
///
/// Fallible functions reject NaN and infinity in inputs. Input slices may be empty only where the definition is meaningful.
///
/// ## Returns
///
/// Output ordering, units and shapes follow the conventions above.
///
/// ## Errors
///
/// Returns the shared Error for invalid domains or shapes, singular evaluation, or nonfinite results. Sample vectors are limited to 1048576 entries and direct numerical work to 16777216 operations; smaller algorithm limits above take precedence.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic. Cancellation and poorly conditioned filters or polynomial fits can reduce accuracy; no uniform relative-error guarantee is made. FFT routines use RustFFT, polynomial fits and DPSS use bounded private nalgebra solvers.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result = argrelextrema(&[1.,3.,1.],|a,b|a>b,1,EdgeMode::Clip)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy argrelextrema definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.argrelextrema.html).
pub fn argrelextrema(
    x: &[f64],
    compare: impl Fn(f64, f64) -> bool,
    order: usize,
    mode: EdgeMode,
) -> Result<Vec<usize>> {
    samples(x)?;
    size(order)?;
    work(x.len(), order)?;
    if order == 0 {
        return Err(Error::InvalidInput("extrema order must be positive"));
    }
    let n = x.len();
    Ok((0..n)
        .filter(|&i| {
            (1..=order).all(|j| {
                let (left, right) = match mode {
                    EdgeMode::Clip => (i.saturating_sub(j), (i + j).min(n - 1)),
                    EdgeMode::Wrap => ((i + n - j % n) % n, (i + j % n) % n),
                };
                compare(x[i], x[left]) && compare(x[i], x[right])
            })
        })
        .collect())
}
/// Returns strict local maxima greater than every neighbor within order samples.
///
/// ## Parameters
///
/// `pub fn argrelmax(x:&[f64],order:usize,mode:EdgeMode)->Result<Vec<usize>>`
///
/// EdgeMode selects clipped or wrapped boundaries; endpoints cannot be strict extrema under Clip. Returns ascending indices.
///
/// ## Domain
///
/// Fallible functions reject NaN and infinity in inputs. Input slices may be empty only where the definition is meaningful.
///
/// ## Returns
///
/// Output ordering, units and shapes follow the conventions above.
///
/// ## Errors
///
/// Returns the shared Error for invalid domains or shapes, singular evaluation, or nonfinite results. Sample vectors are limited to 1048576 entries and direct numerical work to 16777216 operations; smaller algorithm limits above take precedence.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic. Cancellation and poorly conditioned filters or polynomial fits can reduce accuracy; no uniform relative-error guarantee is made. FFT routines use RustFFT, polynomial fits and DPSS use bounded private nalgebra solvers.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result = argrelmax(&[1.,3.,1.],1,EdgeMode::Clip)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy argrelmax definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.argrelmax.html).
pub fn argrelmax(x: &[f64], order: usize, mode: EdgeMode) -> Result<Vec<usize>> {
    argrelextrema(x, |a, b| a > b, order, mode)
}
/// Returns strict local minima smaller than every neighbor within order samples.
///
/// ## Parameters
///
/// `pub fn argrelmin(x:&[f64],order:usize,mode:EdgeMode)->Result<Vec<usize>>`
///
/// EdgeMode selects clipped or wrapped boundaries; endpoints cannot be strict extrema under Clip. Returns ascending indices.
///
/// ## Domain
///
/// Fallible functions reject NaN and infinity in inputs. Input slices may be empty only where the definition is meaningful.
///
/// ## Returns
///
/// Output ordering, units and shapes follow the conventions above.
///
/// ## Errors
///
/// Returns the shared Error for invalid domains or shapes, singular evaluation, or nonfinite results. Sample vectors are limited to 1048576 entries and direct numerical work to 16777216 operations; smaller algorithm limits above take precedence.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic. Cancellation and poorly conditioned filters or polynomial fits can reduce accuracy; no uniform relative-error guarantee is made. FFT routines use RustFFT, polynomial fits and DPSS use bounded private nalgebra solvers.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result = argrelmin(&[3.,1.,3.],1,EdgeMode::Clip)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy argrelmin definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.argrelmin.html).
pub fn argrelmin(x: &[f64], order: usize, mode: EdgeMode) -> Result<Vec<usize>> {
    argrelextrema(x, |a, b| a < b, order, mode)
}

/// Returns local maxima, with each flat maximum represented by its midpoint rounded down.
///
/// ## Parameters
///
/// `pub fn find_peaks(x:&[f64])->Result<Vec<usize>>`
///
/// Endpoints are excluded. Returns ascending sample indices. Height, threshold, distance, prominence, width, and plateau-size selection options are unsupported; prominence and width are separate functions.
///
/// ## Domain
///
/// Fallible functions reject NaN and infinity in inputs. Input slices may be empty only where the definition is meaningful.
///
/// ## Returns
///
/// Output ordering, units and shapes follow the conventions above.
///
/// ## Errors
///
/// Returns the shared Error for invalid domains or shapes, singular evaluation, or nonfinite results. Sample vectors are limited to 1048576 entries and direct numerical work to 16777216 operations; smaller algorithm limits above take precedence.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic. Cancellation and poorly conditioned filters or polynomial fits can reduce accuracy; no uniform relative-error guarantee is made. FFT routines use RustFFT, polynomial fits and DPSS use bounded private nalgebra solvers.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result = find_peaks(&[1.,3.,1.])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy find_peaks definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html).
pub fn find_peaks(x: &[f64]) -> Result<Vec<usize>> {
    samples(x)?;
    let mut result = Vec::new();
    let mut i = 1;
    while i + 1 < x.len() {
        if x[i] > x[i - 1] {
            let start = i;
            while i + 1 < x.len() && x[i + 1] == x[start] {
                i += 1;
            }
            if i + 1 < x.len() && x[i] > x[i + 1] {
                result.push((start + i) / 2);
            }
        }
        i += 1;
    }
    Ok(result)
}

/// Peak prominences and left/right base indices, in input peak order.
#[derive(Clone, Debug, PartialEq)]
pub struct Prominences {
    /// Vertical distance to the higher base.
    pub values: Vec<f64>,
    /// Left minimum indices.
    pub left_bases: Vec<usize>,
    /// Right minimum indices.
    pub right_bases: Vec<usize>,
}
/// Computes `x[peak] - max(x[left_base], x[right_base])` for each peak.
///
/// ## Parameters
///
/// `pub fn peak_prominences(x:&[f64],peaks:&[usize],window:Option<usize>)->Result<Prominences>`
///
/// Searches until a strictly higher sample, the endpoint, or half the optional window. Even window lengths round up to odd. Equal-height peaks do not stop the search. Returns Prominences in caller peak order; invalid local maxima have zero prominence.
///
/// ## Domain
///
/// Fallible functions reject NaN and infinity in inputs. Input slices may be empty only where the definition is meaningful.
///
/// ## Returns
///
/// Output ordering, units and shapes follow the conventions above.
///
/// ## Errors
///
/// Returns the shared Error for invalid domains or shapes, singular evaluation, or nonfinite results. Sample vectors are limited to 1048576 entries and direct numerical work to 16777216 operations; smaller algorithm limits above take precedence.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic. Cancellation and poorly conditioned filters or polynomial fits can reduce accuracy; no uniform relative-error guarantee is made. FFT routines use RustFFT, polynomial fits and DPSS use bounded private nalgebra solvers.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result = peak_prominences(&[1.,3.,1.],&[1],None)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy peak_prominences definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html).
pub fn peak_prominences(x: &[f64], peaks: &[usize], window: Option<usize>) -> Result<Prominences> {
    nonempty(x)?;
    size(peaks.len())?;
    work(x.len(), peaks.len())?;
    if peaks.iter().any(|&p| p >= x.len())
        || window.is_some_and(|n| !(2..=super::MAX_SAMPLES).contains(&n))
    {
        return Err(Error::InvalidInput(
            "peak index or prominence window out of range",
        ));
    }
    let mut result = Prominences {
        values: vec![],
        left_bases: vec![],
        right_bases: vec![],
    };
    for &p in peaks {
        let half = window.map(|w| w / 2).unwrap_or(x.len());
        let mut left = p;
        let mut right = p;
        let mut i = p;
        while i > p.saturating_sub(half) {
            i -= 1;
            if x[i] > x[p] {
                break;
            }
            if x[i] < x[left] {
                left = i;
            }
        }
        i = p;
        while i + 1 < x.len() && i - p < half {
            i += 1;
            if x[i] > x[p] {
                break;
            }
            if x[i] < x[right] {
                right = i;
            }
        }
        result.values.push(match x[p] - x[left].max(x[right]) {
            value if value.is_finite() => Ok(value),
            _ => Err(Error::NonFiniteEvaluation),
        }?);
        result.left_bases.push(left);
        result.right_bases.push(right);
    }
    Ok(result)
}
/// Peak widths, evaluation heights, and interpolated left/right sample coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct PeakWidths {
    /// Width in samples, including fractional samples.
    pub widths: Vec<f64>,
    /// Height of the horizontal evaluation line.
    pub heights: Vec<f64>,
    /// Left intersection coordinate.
    pub left_ips: Vec<f64>,
    /// Right intersection coordinate.
    pub right_ips: Vec<f64>,
}
/// Measures width at `height=x[peak]-relative_height*prominence` using linear interpolation.
///
/// ## Parameters
///
/// `pub fn peak_widths(x:&[f64],peaks:&[usize],relative_height:f64,window:Option<usize>)->Result<PeakWidths>`
///
/// relative_height is finite and nonnegative, typically 0.5. The optional window limits prominence search. Returns PeakWidths in caller order; coordinates and widths are in samples. Explicit precomputed prominence data is unsupported.
///
/// ## Domain
///
/// Fallible functions reject NaN and infinity in inputs. Input slices may be empty only where the definition is meaningful.
///
/// ## Returns
///
/// Output ordering, units and shapes follow the conventions above.
///
/// ## Errors
///
/// Returns the shared Error for invalid domains or shapes, singular evaluation, or nonfinite results. Sample vectors are limited to 1048576 entries and direct numerical work to 16777216 operations; smaller algorithm limits above take precedence.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic. Cancellation and poorly conditioned filters or polynomial fits can reduce accuracy; no uniform relative-error guarantee is made. FFT routines use RustFFT, polynomial fits and DPSS use bounded private nalgebra solvers.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result = peak_widths(&[1.,3.,1.],&[1],0.5,None)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy peak_widths definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_widths.html).
pub fn peak_widths(
    x: &[f64],
    peaks: &[usize],
    relative_height: f64,
    window: Option<usize>,
) -> Result<PeakWidths> {
    if !relative_height.is_finite() || relative_height < 0. {
        return Err(Error::InvalidInput(
            "relative height must be finite and nonnegative",
        ));
    }
    let prominence = peak_prominences(x, peaks, window)?;
    let mut out = PeakWidths {
        widths: vec![],
        heights: vec![],
        left_ips: vec![],
        right_ips: vec![],
    };
    for (j, &p) in peaks.iter().enumerate() {
        let height = match x[p] - prominence.values[j] * relative_height {
            value if value.is_finite() => Ok(value),
            _ => Err(Error::NonFiniteEvaluation),
        }?;
        let mut l = p;
        let mut r = p;
        while l > prominence.left_bases[j] && x[l] > height {
            l -= 1;
        }
        while r < prominence.right_bases[j] && x[r] > height {
            r += 1;
        }
        let left = if x[l] < height && l < p {
            l as f64 + (height - x[l]) / (x[l + 1] - x[l])
        } else {
            l as f64
        };
        let right = if x[r] < height && r > p {
            r as f64 - (height - x[r]) / (x[r - 1] - x[r])
        } else {
            r as f64
        };
        out.widths.push(right - left);
        out.heights.push(height);
        out.left_ips.push(left);
        out.right_ips.push(right);
    }
    Ok(out)
}
