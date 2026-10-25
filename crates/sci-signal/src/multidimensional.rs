use super::{Mode, nonempty, samples, size, work};
use crate::{
    Error, Result,
    complex::{Complex, native, owned},
};

/// Extension outside a two-dimensional input matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Boundary {
    /// Constant finite fill value.
    Fill(f64),
    /// Periodic repetition along each axis.
    Wrap,
    /// Reflection including the endpoint (half-sample symmetry).
    Symmetric,
}
fn shape(x: &[Vec<f64>]) -> Result<(usize, usize)> {
    size(x.len())?;
    let columns = x.first().map_or(0, Vec::len);
    size(columns)?;
    if x.is_empty() || columns == 0 || x.iter().any(|r| r.len() != columns) {
        return Err(Error::DimensionMismatch);
    }
    size(x.len() * columns)?;
    for row in x {
        samples(row)?;
    }
    Ok((x.len(), columns))
}
fn index(i: isize, n: usize, boundary: Boundary) -> Option<usize> {
    let n = n as isize;
    match boundary {
        Boundary::Fill(_) => {
            if i >= 0 && i < n {
                Some(i as usize)
            } else {
                None
            }
        }
        Boundary::Wrap => Some(i.rem_euclid(n) as usize),
        Boundary::Symmetric => {
            let i = i.rem_euclid(2 * n);
            Some(if i < n { i } else { 2 * n - 1 - i } as usize)
        }
    }
}
fn crop(n: usize, m: usize, mode: Mode, correlation: bool) -> (usize, usize) {
    match mode {
        Mode::Full => (0, n + m - 1),
        Mode::Same => (if correlation { m / 2 } else { (m - 1) / 2 }, n),
        Mode::Valid => (n.min(m) - 1, n.abs_diff(m) + 1),
    }
}
fn convolution(
    x: &[Vec<f64>],
    h: &[Vec<f64>],
    mode: Mode,
    boundary: Boundary,
    correlation: bool,
) -> Result<Vec<Vec<f64>>> {
    let (nr, nc) = shape(x)?;
    let (mr, mc) = shape(h)?;
    if let Boundary::Fill(v) = boundary
        && !v.is_finite()
    {
        return Err(Error::InvalidInput("fill value must be finite"));
    }
    if mode == Mode::Valid && !((nr >= mr && nc >= mc) || (nr <= mr && nc <= mc)) {
        return Err(Error::DimensionMismatch);
    }
    let (sr, rows) = crop(nr, mr, mode, correlation);
    let (sc, columns) = crop(nc, mc, mode, correlation);
    size(rows * columns)?;
    work(rows * columns, mr * mc)?;
    let mut out = vec![vec![0.; columns]; rows];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            for (a, kernel_row) in h.iter().enumerate() {
                for (b, &tap) in kernel_row.iter().enumerate() {
                    let r = (i + sr) as isize
                        - if correlation {
                            (mr - 1 - a) as isize
                        } else {
                            a as isize
                        };
                    let c = (j + sc) as isize
                        - if correlation {
                            (mc - 1 - b) as isize
                        } else {
                            b as isize
                        };
                    let sample = match (index(r, nr, boundary), index(c, nc, boundary)) {
                        (Some(r), Some(c)) => x[r][c],
                        _ => match boundary {
                            Boundary::Fill(v) => v,
                            _ => unreachable!(),
                        },
                    };
                    *v += sample * tap;
                }
            }
        }
        super::check_real(row)?;
    }
    Ok(out)
}

/// Computes `y[r,c]=sum_{a,b} h[a,b] x[r-a,c-b]` with boundary extension.
///
/// ## Parameters
///
/// Full returns (nr+mr-1) by (nc+mc-1); Same returns input shape using offsets floor((mr-1)/2), floor((mc-1)/2); Valid returns complete overlaps and requires one input to dominate both dimensions. Matrices use the same row-major vectors as Sci algebra.
///
/// ## Domain
///
/// Inputs must be finite, nonempty and rectangular. Boundary fill values must be finite.
///
/// ## Returns
///
/// Output shape and ordering follow the conventions above.
///
/// ## Errors
///
/// Returns shared errors for invalid shapes or parameters, arithmetic overflow, matrices above 1048576 entries, or direct work above 16777216 operations.
///
/// ## Numerical behavior
///
/// Direct f64 summation, or private RustFFT for analytic signals. Cancellation may reduce accuracy and no uniform relative-error guarantee is made.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result=convolve2d(&[vec![1.,2.]],&[vec![1.,1.]],Mode::Full,Boundary::Fill(0.))?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy convolve2d](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html).
pub fn convolve2d(
    x: &[Vec<f64>],
    h: &[Vec<f64>],
    mode: Mode,
    boundary: Boundary,
) -> Result<Vec<Vec<f64>>> {
    convolution(x, h, mode, boundary, false)
}
/// Computes real two-dimensional cross-correlation, reversing both kernel axes for convolution.
///
/// ## Parameters
///
/// Full and Valid follow convolve2d. Same returns input shape, using offsets mr/2,mc/2, which differ by one from convolve2d for even kernels, matching SciPy. Matrices use the same row-major vectors as Sci algebra.
///
/// ## Domain
///
/// Inputs must be finite, nonempty and rectangular. Boundary fill values must be finite.
///
/// ## Returns
///
/// Output shape and ordering follow the conventions above.
///
/// ## Errors
///
/// Returns shared errors for invalid shapes or parameters, arithmetic overflow, matrices above 1048576 entries, or direct work above 16777216 operations.
///
/// ## Numerical behavior
///
/// Direct f64 summation, or private RustFFT for analytic signals. Cancellation may reduce accuracy and no uniform relative-error guarantee is made.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result=correlate2d(&[vec![1.,2.]],&[vec![1.,1.]],Mode::Same,Boundary::Fill(0.))?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy correlate2d](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate2d.html).
pub fn correlate2d(
    x: &[Vec<f64>],
    h: &[Vec<f64>],
    mode: Mode,
    boundary: Boundary,
) -> Result<Vec<Vec<f64>>> {
    convolution(x, h, mode, boundary, true)
}
/// Computes separable convolution `y[r,c]=sum_{a,b} column_filter[a] row_filter[b] x[r-a,c-b]` with centered kernels.
///
/// ## Parameters
///
/// Row filter acts across columns; column filter acts across rows. Both kernels must be nonempty odd-length. Uses half-sample symmetric reflection. Returns input shape. Matrices use the same row-major vectors as Sci algebra.
///
/// ## Domain
///
/// Inputs must be finite, nonempty and rectangular. Boundary fill values must be finite.
///
/// ## Returns
///
/// Output shape and ordering follow the conventions above.
///
/// ## Errors
///
/// Returns shared errors for invalid shapes or parameters, arithmetic overflow, matrices above 1048576 entries, or direct work above 16777216 operations.
///
/// ## Numerical behavior
///
/// Direct f64 summation, or private RustFFT for analytic signals. Cancellation may reduce accuracy and no uniform relative-error guarantee is made.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result=sepfir2d(&[vec![1.,2.]],&[0.25,0.5,0.25],&[1.])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy sepfir2d](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sepfir2d.html).
pub fn sepfir2d(
    x: &[Vec<f64>],
    row_filter: &[f64],
    column_filter: &[f64],
) -> Result<Vec<Vec<f64>>> {
    let (rows, columns) = shape(x)?;
    nonempty(row_filter)?;
    nonempty(column_filter)?;
    if row_filter.len().is_multiple_of(2) || column_filter.len().is_multiple_of(2) {
        return Err(Error::InvalidInput("separable FIR kernels must be odd"));
    }
    work(rows * columns, row_filter.len() + column_filter.len())?;
    let filtered = convolve2d(x, &[row_filter.to_vec()], Mode::Same, Boundary::Symmetric)?;
    convolve2d(
        &filtered,
        &column_filter.iter().map(|&v| vec![v]).collect::<Vec<_>>(),
        Mode::Same,
        Boundary::Symmetric,
    )
}
/// Returns the median in each centered rectangular neighborhood with zero padding.
///
/// ## Parameters
///
/// kernel gives positive odd (row count,column count). Returns the input shape. A kernel may exceed the input shape. Matrices use the same row-major vectors as Sci algebra.
///
/// ## Domain
///
/// Inputs must be finite, nonempty and rectangular. Boundary fill values must be finite.
///
/// ## Returns
///
/// Output shape and ordering follow the conventions above.
///
/// ## Errors
///
/// Returns shared errors for invalid shapes or parameters, arithmetic overflow, matrices above 1048576 entries, or direct work above 16777216 operations.
///
/// ## Numerical behavior
///
/// Direct f64 summation, or private RustFFT for analytic signals. Cancellation may reduce accuracy and no uniform relative-error guarantee is made.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result=medfilt2d(&[vec![1.,2.,3.]], [1,3])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy medfilt2d](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt2d.html).
pub fn medfilt2d(x: &[Vec<f64>], kernel: [usize; 2]) -> Result<Vec<Vec<f64>>> {
    let (rows, columns) = shape(x)?;
    size(kernel[0])?;
    size(kernel[1])?;
    if kernel.iter().any(|&n| n == 0 || n.is_multiple_of(2)) {
        return Err(Error::InvalidInput("median kernels must be positive odd"));
    }
    work(rows * columns, kernel[0] * kernel[1])?;
    let mut out = vec![vec![0.; columns]; rows];
    let mut neighborhood = Vec::with_capacity(kernel[0] * kernel[1]);
    for (i, row) in out.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            neighborhood.clear();
            for a in 0..kernel[0] {
                for b in 0..kernel[1] {
                    let r = (i + a) as isize - (kernel[0] / 2) as isize;
                    let c = (j + b) as isize - (kernel[1] / 2) as isize;
                    neighborhood.push(
                        match (
                            index(r, rows, Boundary::Fill(0.)),
                            index(c, columns, Boundary::Fill(0.)),
                        ) {
                            (Some(r), Some(c)) => x[r][c],
                            _ => 0.,
                        },
                    );
                }
            }
            let rank = neighborhood.len() / 2;
            neighborhood.select_nth_unstable_by(rank, f64::total_cmp);
            *v = neighborhood[rank];
        }
    }
    Ok(out)
}
/// Returns `IFFT2(FFT2(x) h_rows h_columns)`, retaining DC, removing even-length Nyquist, doubling positive bins, and zeroing negative bins on each axis.
///
/// ## Parameters
///
/// Input and output are row-major matrices. Returns shared Complex samples in the input shape; no FFT-length dispatch. This separable two-dimensional analytic signal does not generally preserve the real part of arbitrary input. Matrices use the same row-major vectors as Sci algebra.
///
/// ## Domain
///
/// Inputs must be finite, nonempty and rectangular. Boundary fill values must be finite.
///
/// ## Returns
///
/// Output shape and ordering follow the conventions above.
///
/// ## Errors
///
/// Returns shared errors for invalid shapes or parameters, arithmetic overflow, matrices above 1048576 entries, or direct work above 16777216 operations.
///
/// ## Numerical behavior
///
/// Direct f64 summation, or private RustFFT for analytic signals. Cancellation may reduce accuracy and no uniform relative-error guarantee is made.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _result=hilbert2(&[vec![1.,2.],vec![3.,4.]])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy hilbert2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert2.html).
pub fn hilbert2(x: &[Vec<f64>]) -> Result<Vec<Vec<Complex>>> {
    let (rows, columns) = shape(x)?;
    let mut spectrum: Vec<_> = x
        .iter()
        .map(|row| {
            super::fft(
                &row.iter()
                    .map(|&re| Complex { re, im: 0. })
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let weight = |k: usize, n: usize| {
        if k == 0 {
            1.
        } else if k < n.div_ceil(2) {
            2.
        } else {
            0.
        }
    };
    for j in 0..columns {
        let column: Vec<_> = spectrum.iter().map(|row| row[j]).collect();
        let values = super::fft(&column);
        for (i, row) in spectrum.iter_mut().enumerate() {
            row[j] = owned(native(values[i]) * weight(i, rows) * weight(j, columns));
        }
    }
    for j in 0..columns {
        let column: Vec<_> = spectrum.iter().map(|row| row[j]).collect();
        let values = super::ifft(&column);
        for (i, row) in spectrum.iter_mut().enumerate() {
            row[j] = values[i];
        }
    }
    let output: Vec<_> = spectrum.iter().map(|row| super::ifft(row)).collect();
    for row in &output {
        super::check_complex(row)?;
    }
    Ok(output)
}
