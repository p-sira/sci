use crate::error::{finite, finite_slice};
mod fourier;
mod hankel;
mod periodic;
use crate::{Error, Result, complex::Complex};
pub use fourier::{fft, ifft, rfft};
pub use hankel::{fht, fhtoffset, ifht};
pub use periodic::{
    cc_diff, cs_diff, diff, hilbert, ihilbert, itilbert, sc_diff, shift, ss_diff, tilbert,
};
const LIMIT: usize = 1_048_576;
fn length(n: usize) -> Result<()> {
    if n == 0 || n > LIMIT {
        Err(Error::InvalidInput("FFT length must be 1..=1048576"))
    } else {
        Ok(())
    }
}
fn values(x: &[Complex]) -> Result<()> {
    if x.len() > LIMIT || x.iter().any(|v| !v.re.is_finite() || !v.im.is_finite()) {
        return Err(Error::InvalidInput("invalid FFT input"));
    }
    Ok(())
}
fn inferred(x: &[Complex], n: Option<usize>) -> Result<usize> {
    values(x)?;
    let n = n.unwrap_or_else(|| x.len().saturating_sub(1) * 2);
    length(n)?;
    Ok(n)
}
fn tensor(shape: &[usize], axes: &[usize], len: usize) -> Result<usize> {
    if shape.is_empty() || shape.len() > 16 {
        return Err(Error::InvalidInput("FFT rank must be 1..=16"));
    }
    let mut size = 1usize;
    for &n in shape {
        length(n)?;
        size = size
            .checked_mul(n)
            .ok_or(Error::InvalidInput("FFT shape overflow"))?;
        if size > LIMIT {
            return Err(Error::InvalidInput("FFT entry limit"));
        }
    }
    if size != len {
        return Err(Error::DimensionMismatch);
    }
    let mut seen = vec![false; shape.len()];
    for &axis in axes {
        if axis >= shape.len() || seen[axis] {
            return Err(Error::InvalidInput(
                "FFT axes must be distinct and in range",
            ));
        }
        seen[axis] = true;
    }
    Ok(size)
}
fn nd(x: &[Complex], shape: &[usize], axes: &[usize], inverse: bool) -> Result<Vec<Complex>> {
    tensor(shape, axes, x.len())?;
    values(x)?;
    let mut out = x.to_vec();
    let mut planner = rustfft::FftPlanner::<f64>::new();
    for &axis in axes {
        let n = shape[axis];
        let stride = shape[axis + 1..].iter().product::<usize>();
        let block = n * stride;
        let plan = if inverse {
            planner.plan_fft_inverse(n)
        } else {
            planner.plan_fft_forward(n)
        };
        let mut line = vec![num_complex::Complex64::new(0.0, 0.0); n];
        let mut scratch =
            vec![num_complex::Complex64::new(0.0, 0.0); plan.get_inplace_scratch_len()];
        for base in (0..out.len()).step_by(block) {
            for offset in 0..stride {
                for (k, v) in line.iter_mut().enumerate() {
                    let z = out[base + offset + k * stride];
                    *v = num_complex::Complex64::new(z.re, z.im);
                }
                plan.process_with_scratch(&mut line, &mut scratch);
                let scale = if inverse { n as f64 } else { 1.0 };
                for (k, v) in line.iter().enumerate() {
                    out[base + offset + k * stride] = Complex {
                        re: finite(v.re / scale)?,
                        im: finite(v.im / scale)?,
                    };
                }
            }
        }
    }
    Ok(out)
}
fn reduced(shape: &[usize], axes: &[usize]) -> Result<(Vec<usize>, usize)> {
    let axis = *axes.last().ok_or(Error::InvalidInput(
        "real transforms need at least one axis",
    ))?;
    if axis >= shape.len() {
        return Err(Error::InvalidInput("axis out of range"));
    }
    let mut out = shape.to_vec();
    out[axis] = shape[axis] / 2 + 1;
    Ok((out, axis))
}
fn rows<T: Clone>(x: &[Vec<T>]) -> Result<(Vec<T>, [usize; 2])> {
    let n = x.len();
    let m = x.first().map_or(0, Vec::len);
    length(n)?;
    length(m)?;
    if n.checked_mul(m).is_none_or(|v| v > LIMIT) {
        return Err(Error::InvalidInput("FFT matrix size"));
    }
    if x.iter().any(|r| r.len() != m) {
        return Err(Error::DimensionMismatch);
    }
    Ok((x.iter().flatten().cloned().collect(), [n, m]))
}
fn reshape<T: Clone>(x: &[T], columns: usize) -> Vec<Vec<T>> {
    x.chunks(columns).map(<[T]>::to_vec).collect()
}

/// Inverts the half-spectrum of real data: `x[j]=(1/n) Σₖ X[k]exp(2πijk/n)` after Hermitian completion.
///
/// ## Parameters
///
/// `input` contains DC then nonnegative frequencies. `n=None` infers even length `2*(m-1)`; specify odd lengths. Bins are cropped or zero-padded to `n/2+1`; imaginary DC and even Nyquist components are ignored. Only backward normalization is supported.
/// ## Domain
///
/// Finite complex bins, at most 1,048,576 input entries; n is 1..=1,048,576. Empty bins may be zero-padded with explicit n; inferred zero lengths are rejected.
/// ## Returns
///
/// A real vector of n samples in original sample units.
/// ## Errors
///
/// Invalid sizes, shapes, axes, nonfinite inputs, work limits, or arithmetic overflow return shared errors.
/// ## Numerical behavior
///
/// Uses the private RustFFT backend, with O(N log N) transform work per selected axis. Floating-point roundoff grows with transform size and cancellation. No window, unit conversion, or sample-spacing factor is applied.
/// # Examples
///
/// ```
/// let x=sci_fft::irfft(&sci_fft::rfft(&[1.0,2.0,3.0]),Some(3))?;
/// assert!((x[1]-2.0).abs()<1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy irfft](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.irfft.html).
pub fn irfft(input: &[Complex], n: Option<usize>) -> Result<Vec<f64>> {
    let n = inferred(input, n)?;
    let mut full = vec![Complex { re: 0.0, im: 0.0 }; n];
    for (k, &v) in input.iter().take(n / 2 + 1).enumerate() {
        full[k] = v;
        if k == 0 || n.is_multiple_of(2) && k == n / 2 {
            full[k].im = 0.0;
        } else {
            full[n - k] = Complex {
                re: v.re,
                im: -v.im,
            };
        }
    }
    ifft(&full).iter().map(|v| finite(v.re)).collect()
}

/// Transforms Hermitian time data to a real spectrum: `H(x)=n*irfft(conj(x))`.
///
/// ## Parameters
///
/// `input` supplies the first half of Hermitian time samples; `n` and padding/cropping follow [`irfft`]. Forward normalization is unscaled.
/// ## Domain
///
/// The finite-value, inferred-length, and explicit-length domain of [`irfft`].
/// ## Returns
///
/// A real spectrum of n bins in standard DFT order.
/// ## Errors
///
/// Invalid sizes, shapes, axes, nonfinite inputs, work limits, or arithmetic overflow return shared errors.
/// ## Numerical behavior
///
/// Uses the private RustFFT backend, with O(N log N) transform work per selected axis. Floating-point roundoff grows with transform size and cancellation. No window, unit conversion, or sample-spacing factor is applied.
/// # Examples
///
/// ```
/// let x=sci_fft::hfft(&[sci_complex::Complex{re:2.0,im:0.0}],Some(1))?;
/// assert_eq!(x,vec![2.0]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy hfft](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.hfft.html).
pub fn hfft(input: &[Complex], n: Option<usize>) -> Result<Vec<f64>> {
    let n = inferred(input, n)?;
    let conjugate = input
        .iter()
        .map(|v| Complex {
            re: v.re,
            im: -v.im,
        })
        .collect::<Vec<_>>();
    irfft(&conjugate, Some(n))?
        .into_iter()
        .map(|v| finite(v * n as f64))
        .collect()
}

/// Inverts a real spectrum into Hermitian time samples: `IH(y)=conj(rfft(y))/n`.
///
/// ## Parameters
///
/// `input` is real DFT-order spectrum. `n=None` uses input length; explicit n crops or pads real samples before transforming. Backward inverse normalization divides by n.
/// ## Domain
///
/// Finite real input of at most 1,048,576 entries and n in 1..=1,048,576; empty input needs explicit positive n.
/// ## Returns
///
/// The first `floor(n/2)+1` complex time samples, including DC and even Nyquist.
/// ## Errors
///
/// Invalid sizes, shapes, axes, nonfinite inputs, work limits, or arithmetic overflow return shared errors.
/// ## Numerical behavior
///
/// Uses the private RustFFT backend, with O(N log N) transform work per selected axis. Floating-point roundoff grows with transform size and cancellation. No window, unit conversion, or sample-spacing factor is applied.
/// # Examples
///
/// ```
/// let x=sci_fft::ihfft(&[2.0],None)?;
/// assert_eq!(x[0].re,2.0);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy ihfft](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.ihfft.html).
pub fn ihfft(input: &[f64], n: Option<usize>) -> Result<Vec<Complex>> {
    if input.len() > LIMIT {
        return Err(Error::InvalidInput("FFT input size"));
    }
    finite_slice(input)?;
    let n = n.unwrap_or(input.len());
    length(n)?;
    let mut padded = vec![0.0; n];
    let m = input.len().min(n);
    padded[..m].copy_from_slice(&input[..m]);
    rfft(&padded)
        .into_iter()
        .map(|v| {
            Ok(Complex {
                re: finite(v.re / n as f64)?,
                im: finite(-v.im / n as f64)?,
            })
        })
        .collect()
}

/// Computes a separable multidimensional DFT `X[k]=Σⱼ x[j]exp(−2πi Σₐ jₐkₐ/nₐ)`.
///
/// ## Parameters
///
/// `input` is flat row-major storage with `shape`. `axes` lists transform axes in order; empty axes copy the input. SciPy resizing, norm, workers, overwrite, and plan options are omitted; the shape is fixed.
/// ## Domain
///
/// Finite input components, nonzero axes, at most 1,048,576 entries and 16 dimensions. Shapes must match storage; axes are distinct zero-based indices. NaN/infinity are rejected.
/// ## Returns
///
/// Complex flat row-major data of the same shape and standard DFT bin order along each transformed axis.
/// ## Errors
///
/// Invalid sizes, shapes, axes, nonfinite inputs, work limits, or arithmetic overflow return shared errors.
/// ## Numerical behavior
///
/// Uses the private RustFFT backend, with O(N log N) transform work per selected axis. Floating-point roundoff grows with transform size and cancellation. No window, unit conversion, or sample-spacing factor is applied.
/// # Examples
///
/// ```
/// let x=[sci_complex::Complex{re:1.0,im:0.0}];
/// assert_eq!(sci_fft::fftn(&x,&[1,1],&[0,1])?,x);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy fftn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fftn.html).
pub fn fftn(input: &[Complex], shape: &[usize], axes: &[usize]) -> Result<Vec<Complex>> {
    nd(input, shape, axes, false)
}

/// Computes the inverse multidimensional DFT with positive exponent and division by `Πₐ nₐ`.
///
/// ## Parameters
///
/// `input` is flat row-major storage with `shape`. `axes` lists transform axes in order; empty axes copy the input. SciPy resizing, norm, workers, overwrite, and plan options are omitted; the shape is fixed.
/// ## Domain
///
/// Finite input components, nonzero axes, at most 1,048,576 entries and 16 dimensions. Shapes must match storage; axes are distinct zero-based indices. NaN/infinity are rejected.
/// ## Returns
///
/// Complex flat row-major data of the same shape and standard DFT bin order along each transformed axis.
/// ## Errors
///
/// Invalid sizes, shapes, axes, nonfinite inputs, work limits, or arithmetic overflow return shared errors.
/// ## Numerical behavior
///
/// Uses the private RustFFT backend, with O(N log N) transform work per selected axis. Floating-point roundoff grows with transform size and cancellation. No window, unit conversion, or sample-spacing factor is applied.
/// # Examples
///
/// ```
/// let x=[sci_complex::Complex{re:1.0,im:0.0}];
/// assert_eq!(sci_fft::ifftn(&x,&[1,1],&[0,1])?,x);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy ifftn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.ifftn.html).
pub fn ifftn(input: &[Complex], shape: &[usize], axes: &[usize]) -> Result<Vec<Complex>> {
    nd(input, shape, axes, true)
}

/// Computes the multidimensional DFT of real data and retains the last transformed axis’s nonnegative bins.
///
/// ## Parameters
///
/// `input` is flat row-major real data; `shape` and nonempty `axes` specify dimensions. Only the last listed axis is shortened, to floor(n/2)+1. Normalization is unscaled forward; resizing and alternate normalization are omitted.
/// ## Domain
///
/// Finite input components, nonzero axes, at most 1,048,576 entries and 16 dimensions. Shapes must match storage; axes are distinct zero-based indices. NaN/infinity are rejected.
/// ## Returns
///
/// `(values, reduced_shape)`, with flat complex row-major values.
/// ## Errors
///
/// Invalid sizes, shapes, axes, nonfinite inputs, work limits, or arithmetic overflow return shared errors.
/// ## Numerical behavior
///
/// Uses the private RustFFT backend, with O(N log N) transform work per selected axis. Floating-point roundoff grows with transform size and cancellation. No window, unit conversion, or sample-spacing factor is applied.
/// # Examples
///
/// ```
/// let (x,shape)=sci_fft::rfftn(&[1.0,2.0],&[1,2],&[0,1])?;
/// assert_eq!(shape,vec![1,2]);assert_eq!(x[0].re,3.0);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy rfftn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.rfftn.html).
pub fn rfftn(input: &[f64], shape: &[usize], axes: &[usize]) -> Result<(Vec<Complex>, Vec<usize>)> {
    tensor(shape, axes, input.len())?;
    finite_slice(input)?;
    let (small, axis) = reduced(shape, axes)?;
    let complex = input
        .iter()
        .map(|v| Complex { re: *v, im: 0.0 })
        .collect::<Vec<_>>();
    let full = nd(&complex, shape, axes, false)?;
    let stride = shape[axis + 1..].iter().product::<usize>();
    let block = shape[axis] * stride;
    let out = full
        .chunks(block)
        .flat_map(|b| b[..small[axis] * stride].iter().copied())
        .collect();
    Ok((out, small))
}

/// Inverts a reduced real-data multidimensional spectrum to real samples.
///
/// ## Parameters
///
/// `shape` is the desired full output shape; `input` must have exactly that shape with the last listed axis replaced by floor(n/2)+1. Input resizing is omitted. `axes` is nonempty and follows [`rfftn`]. Imaginary endpoints are discarded after the other inverse axes; only backward normalization is supported.
/// ## Domain
///
/// Finite input components, nonzero axes, at most 1,048,576 entries and 16 dimensions. Shapes must match storage; axes are distinct zero-based indices. NaN/infinity are rejected.
/// ## Returns
///
/// Flat real row-major samples of the requested shape.
/// ## Errors
///
/// Invalid sizes, shapes, axes, nonfinite inputs, work limits, or arithmetic overflow return shared errors.
/// ## Numerical behavior
///
/// Uses the private RustFFT backend, with O(N log N) transform work per selected axis. Floating-point roundoff grows with transform size and cancellation. No window, unit conversion, or sample-spacing factor is applied.
/// # Examples
///
/// ```
/// let (s,_)=sci_fft::rfftn(&[1.0,2.0,3.0],&[1,3],&[0,1])?;
/// let x=sci_fft::irfftn(&s,&[1,3],&[0,1])?;
/// assert!((x[1]-2.0).abs()<1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy irfftn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.irfftn.html).
pub fn irfftn(input: &[Complex], shape: &[usize], axes: &[usize]) -> Result<Vec<f64>> {
    let full_len = shape
        .iter()
        .try_fold(1usize, |p, n| p.checked_mul(*n))
        .ok_or(Error::InvalidInput("FFT shape overflow"))?;
    tensor(shape, axes, full_len)?;
    let (small, axis) = reduced(shape, axes)?;
    tensor(&small, axes, input.len())?;
    values(input)?;
    let intermediate = nd(input, &small, &axes[..axes.len() - 1], true)?;
    let n = shape[axis];
    let stride = shape[axis + 1..].iter().product::<usize>();
    let block = n * stride;
    let small_block = small[axis] * stride;
    let mut out = Vec::with_capacity(full_len);
    let mut line = vec![Complex { re: 0.0, im: 0.0 }; small[axis]];
    let mut full = vec![Complex { re: 0.0, im: 0.0 }; full_len];
    for (outer, b) in intermediate.chunks(small_block).enumerate() {
        for offset in 0..stride {
            for (k, v) in line.iter_mut().enumerate() {
                *v = b[offset + k * stride];
            }
            for (k, &v) in line.iter().enumerate() {
                let index = outer * block + offset + k * stride;
                full[index] = v;
                if k == 0 || n.is_multiple_of(2) && k == n / 2 {
                    full[index].im = 0.0;
                } else {
                    full[outer * block + offset + (n - k) * stride] = Complex {
                        re: v.re,
                        im: -v.im,
                    };
                }
            }
        }
    }
    out.extend(nd(&full, shape, &[axis], true)?.into_iter().map(|v| v.re));
    Ok(out)
}

/// Transforms Hermitian multidimensional time samples: `H(x)=Πₐ nₐ*irfftn(conj(x))`.
///
/// ## Parameters
///
/// Shapes, axes, and backward normalization follow [`rfftn`] and [`irfftn`]. For hfftn shape is full output shape; for ihfftn it is full input shape. Padding and alternate normalization are omitted.
/// ## Domain
///
/// Finite input components, nonzero axes, at most 1,048,576 entries and 16 dimensions. Shapes must match storage; axes are distinct zero-based indices. NaN/infinity are rejected.
/// ## Returns
///
/// Flat real row-major spectrum of the requested full shape.
/// ## Errors
///
/// Invalid sizes, shapes, axes, nonfinite inputs, work limits, or arithmetic overflow return shared errors.
/// ## Numerical behavior
///
/// Uses the private RustFFT backend, with O(N log N) transform work per selected axis. Floating-point roundoff grows with transform size and cancellation. No window, unit conversion, or sample-spacing factor is applied.
/// # Examples
///
/// ```
/// let (s,_)=sci_fft::ihfftn(&[2.0],&[1,1],&[0,1])?;
/// assert_eq!(sci_fft::hfftn(&s,&[1,1],&[0,1])?,vec![2.0]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy hfftn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.hfftn.html).
pub fn hfftn(input: &[Complex], shape: &[usize], axes: &[usize]) -> Result<Vec<f64>> {
    values(input)?;
    let conjugate = input
        .iter()
        .map(|v| Complex {
            re: v.re,
            im: -v.im,
        })
        .collect::<Vec<_>>();
    let out = irfftn(&conjugate, shape, axes)?;
    let scale = axes.iter().map(|a| shape[*a] as f64).product::<f64>();
    out.into_iter().map(|v| finite(v * scale)).collect()
}

/// Inverts a real multidimensional spectrum: `IH(y)=conj(rfftn(y))/Πₐ nₐ`.
///
/// ## Parameters
///
/// Shapes, axes, and backward normalization follow [`rfftn`] and [`irfftn`]. For hfftn shape is full output shape; for ihfftn it is full input shape. Padding and alternate normalization are omitted.
/// ## Domain
///
/// Finite input components, nonzero axes, at most 1,048,576 entries and 16 dimensions. Shapes must match storage; axes are distinct zero-based indices. NaN/infinity are rejected.
/// ## Returns
///
/// `(values, reduced_shape)` containing the first Hermitian time half along the last transform axis.
/// ## Errors
///
/// Invalid sizes, shapes, axes, nonfinite inputs, work limits, or arithmetic overflow return shared errors.
/// ## Numerical behavior
///
/// Uses the private RustFFT backend, with O(N log N) transform work per selected axis. Floating-point roundoff grows with transform size and cancellation. No window, unit conversion, or sample-spacing factor is applied.
/// # Examples
///
/// ```
/// let (s,_)=sci_fft::ihfftn(&[2.0],&[1,1],&[0,1])?;
/// assert_eq!(sci_fft::hfftn(&s,&[1,1],&[0,1])?,vec![2.0]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy ihfftn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.ihfftn.html).
pub fn ihfftn(
    input: &[f64],
    shape: &[usize],
    axes: &[usize],
) -> Result<(Vec<Complex>, Vec<usize>)> {
    let (mut out, small) = rfftn(input, shape, axes)?;
    let scale = axes.iter().map(|a| shape[*a] as f64).product::<f64>();
    for v in &mut out {
        v.re = finite(v.re / scale)?;
        v.im = finite(-v.im / scale)?;
    }
    Ok((out, small))
}

/// Two-dimensional specialization of [`fftn`], transforming both axes.
///
/// ## Parameters
///
/// `input` uses rectangular shared rows. All conventions and normalization follow [`fftn`]; resizing is omitted.
/// ## Domain
///
/// Finite input components, nonzero axes, at most 1,048,576 entries and 16 dimensions. Shapes must match storage; axes are distinct zero-based indices. NaN/infinity are rejected.
/// ## Returns
///
/// Shared output rows; real-input/Hermitian-inverse transforms reduce the column count, while real-output transforms expand it to n.
/// ## Errors
///
/// Invalid sizes, shapes, axes, nonfinite inputs, work limits, or arithmetic overflow return shared errors.
/// ## Numerical behavior
///
/// Reuses [`fftn`] with exactly its work bounds and numerical behavior.
/// # Examples
///
/// ```
/// let x=sci_fft::fft2(&[vec![sci_complex::Complex{re:2.0,im:0.0}]])?;
/// assert_eq!(x[0][0].re,2.0);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy fft2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft2.html).
pub fn fft2(input: &[Vec<Complex>]) -> Result<Vec<Vec<Complex>>> {
    let (flat, shape) = rows(input)?;
    let out = fftn(&flat, &shape, &[0, 1])?;
    Ok(reshape(&out, shape[1]))
}

/// Two-dimensional specialization of [`ifftn`], transforming both axes.
///
/// ## Parameters
///
/// `input` uses rectangular shared rows. All conventions and normalization follow [`ifftn`]; resizing is omitted.
/// ## Domain
///
/// Finite input components, nonzero axes, at most 1,048,576 entries and 16 dimensions. Shapes must match storage; axes are distinct zero-based indices. NaN/infinity are rejected.
/// ## Returns
///
/// Shared output rows; real-input/Hermitian-inverse transforms reduce the column count, while real-output transforms expand it to n.
/// ## Errors
///
/// Invalid sizes, shapes, axes, nonfinite inputs, work limits, or arithmetic overflow return shared errors.
/// ## Numerical behavior
///
/// Reuses [`ifftn`] with exactly its work bounds and numerical behavior.
/// # Examples
///
/// ```
/// let x=sci_fft::ifft2(&[vec![sci_complex::Complex{re:2.0,im:0.0}]])?;
/// assert_eq!(x[0][0].re,2.0);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy ifft2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.ifft2.html).
pub fn ifft2(input: &[Vec<Complex>]) -> Result<Vec<Vec<Complex>>> {
    let (flat, shape) = rows(input)?;
    let out = ifftn(&flat, &shape, &[0, 1])?;
    Ok(reshape(&out, shape[1]))
}

/// Two-dimensional specialization of [`rfftn`], transforming both axes.
///
/// ## Parameters
///
/// `input` uses rectangular shared rows. All conventions and normalization follow [`rfftn`]; resizing is omitted.
/// ## Domain
///
/// Finite input components, nonzero axes, at most 1,048,576 entries and 16 dimensions. Shapes must match storage; axes are distinct zero-based indices. NaN/infinity are rejected.
/// ## Returns
///
/// Shared output rows; real-input/Hermitian-inverse transforms reduce the column count, while real-output transforms expand it to n.
/// ## Errors
///
/// Invalid sizes, shapes, axes, nonfinite inputs, work limits, or arithmetic overflow return shared errors.
/// ## Numerical behavior
///
/// Reuses [`rfftn`] with exactly its work bounds and numerical behavior.
/// # Examples
///
/// ```
/// let x=sci_fft::rfft2(&[vec![2.0]])?;
/// assert_eq!(x[0][0].re,2.0);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy rfft2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.rfft2.html).
pub fn rfft2(input: &[Vec<f64>]) -> Result<Vec<Vec<Complex>>> {
    let (flat, shape) = rows(input)?;
    let (out, small) = rfftn(&flat, &shape, &[0, 1])?;
    Ok(reshape(&out, small[1]))
}

/// Two-dimensional specialization of [`irfftn`], transforming both axes.
///
/// ## Parameters
///
/// `input` uses rectangular shared rows. `n` is the full output column count; input columns must be floor(n/2)+1. All conventions and normalization follow [`irfftn`]; resizing is omitted.
/// ## Domain
///
/// Finite input components, nonzero axes, at most 1,048,576 entries and 16 dimensions. Shapes must match storage; axes are distinct zero-based indices. NaN/infinity are rejected.
/// ## Returns
///
/// Shared output rows; real-input/Hermitian-inverse transforms reduce the column count, while real-output transforms expand it to n.
/// ## Errors
///
/// Invalid sizes, shapes, axes, nonfinite inputs, work limits, or arithmetic overflow return shared errors.
/// ## Numerical behavior
///
/// Reuses [`irfftn`] with exactly its work bounds and numerical behavior.
/// # Examples
///
/// ```
/// let x=sci_fft::irfft2(&[vec![sci_complex::Complex{re:2.0,im:0.0}]],1)?;
/// assert_eq!(x,vec![vec![2.0]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy irfft2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.irfft2.html).
pub fn irfft2(input: &[Vec<Complex>], n: usize) -> Result<Vec<Vec<f64>>> {
    let (flat, mut shape) = rows(input)?;
    shape[1] = n;
    let out = irfftn(&flat, &shape, &[0, 1])?;
    Ok(reshape(&out, n))
}

/// Two-dimensional specialization of [`hfftn`], transforming both axes.
///
/// ## Parameters
///
/// `input` uses rectangular shared rows. `n` is the full output column count; input columns must be floor(n/2)+1. All conventions and normalization follow [`hfftn`]; resizing is omitted.
/// ## Domain
///
/// Finite input components, nonzero axes, at most 1,048,576 entries and 16 dimensions. Shapes must match storage; axes are distinct zero-based indices. NaN/infinity are rejected.
/// ## Returns
///
/// Shared output rows; real-input/Hermitian-inverse transforms reduce the column count, while real-output transforms expand it to n.
/// ## Errors
///
/// Invalid sizes, shapes, axes, nonfinite inputs, work limits, or arithmetic overflow return shared errors.
/// ## Numerical behavior
///
/// Reuses [`hfftn`] with exactly its work bounds and numerical behavior.
/// # Examples
///
/// ```
/// let x=sci_fft::hfft2(&[vec![sci_complex::Complex{re:2.0,im:0.0}]],1)?;
/// assert_eq!(x,vec![vec![2.0]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy hfft2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.hfft2.html).
pub fn hfft2(input: &[Vec<Complex>], n: usize) -> Result<Vec<Vec<f64>>> {
    let (flat, mut shape) = rows(input)?;
    shape[1] = n;
    let out = hfftn(&flat, &shape, &[0, 1])?;
    Ok(reshape(&out, n))
}

/// Two-dimensional specialization of [`ihfftn`], transforming both axes.
///
/// ## Parameters
///
/// `input` uses rectangular shared rows. All conventions and normalization follow [`ihfftn`]; resizing is omitted.
/// ## Domain
///
/// Finite input components, nonzero axes, at most 1,048,576 entries and 16 dimensions. Shapes must match storage; axes are distinct zero-based indices. NaN/infinity are rejected.
/// ## Returns
///
/// Shared output rows; real-input/Hermitian-inverse transforms reduce the column count, while real-output transforms expand it to n.
/// ## Errors
///
/// Invalid sizes, shapes, axes, nonfinite inputs, work limits, or arithmetic overflow return shared errors.
/// ## Numerical behavior
///
/// Reuses [`ihfftn`] with exactly its work bounds and numerical behavior.
/// # Examples
///
/// ```
/// let x=sci_fft::ihfft2(&[vec![2.0]])?;
/// assert_eq!(x[0][0].re,2.0);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy ihfft2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.ihfft2.html).
pub fn ihfft2(input: &[Vec<f64>]) -> Result<Vec<Vec<Complex>>> {
    let (flat, shape) = rows(input)?;
    let (out, small) = ihfftn(&flat, &shape, &[0, 1])?;
    Ok(reshape(&out, small[1]))
}

/// Computes DFT frequencies `f[k]=k/(nd)` for k before the negative-frequency half, otherwise `(k−n)/(nd)`.
///
/// ## Parameters
///
/// `n` is transform length and `d` is signed sample spacing, in time or distance units. Frequencies are cycles per spacing unit, not angular frequencies. Even Nyquist is negative.
/// ## Domain
///
/// n in 1..=1,048,576 and finite nonzero d. Negative spacing reverses signs; NaN/infinity are rejected.
/// ## Returns
///
/// A real frequency vector of length n in DFT bin order.
/// ## Errors
///
/// Invalid sizes, shapes, axes, nonfinite inputs, work limits, or arithmetic overflow return shared errors.
/// ## Numerical behavior
///
/// O(n) arithmetic; sequential divisions avoid overflow in n*d. Extreme spacing may underflow frequencies or overflow and return an error.
/// # Examples
///
/// ```
/// assert_eq!(sci_fft::fftfreq(2,1.0)?,vec![0.0,-0.5]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy fftfreq](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fftfreq.html).
pub fn fftfreq(n: usize, d: f64) -> Result<Vec<f64>> {
    length(n)?;
    if !d.is_finite() || d == 0.0 {
        return Err(Error::InvalidInput(
            "sample spacing must be finite and nonzero",
        ));
    }
    let scale = finite((1.0 / n as f64) / d)?;
    (0..n)
        .map(|k| {
            finite(
                if k < n.div_ceil(2) {
                    k as f64
                } else {
                    k as f64 - n as f64
                } * scale,
            )
        })
        .collect()
}

/// Computes nonnegative real-DFT frequencies `f[k]=k/(nd)`.
///
/// ## Parameters
///
/// `n` is transform length and `d` is signed sample spacing, in time or distance units. Frequencies are cycles per spacing unit, not angular frequencies. Even Nyquist is positive.
/// ## Domain
///
/// n in 1..=1,048,576 and finite nonzero d. Negative spacing reverses signs; NaN/infinity are rejected.
/// ## Returns
///
/// A real frequency vector of length floor(n/2)+1.
/// ## Errors
///
/// Invalid sizes, shapes, axes, nonfinite inputs, work limits, or arithmetic overflow return shared errors.
/// ## Numerical behavior
///
/// O(n) arithmetic; sequential divisions avoid overflow in n*d. Extreme spacing may underflow frequencies or overflow and return an error.
/// # Examples
///
/// ```
/// assert_eq!(sci_fft::rfftfreq(2,1.0)?,vec![0.0,0.5]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy rfftfreq](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.rfftfreq.html).
pub fn rfftfreq(n: usize, d: f64) -> Result<Vec<f64>> {
    length(n)?;
    if !d.is_finite() || d == 0.0 {
        return Err(Error::InvalidInput(
            "sample spacing must be finite and nonzero",
        ));
    }
    let scale = finite((1.0 / n as f64) / d)?;
    (0..n / 2 + 1).map(|k| finite(k as f64 * scale)).collect()
}

/// Centers the zero-frequency bin by rotating DFT data left by ceil(n/2).
///
/// ## Parameters
///
/// `input` is a complex one-dimensional vector in DFT order for fftshift, centered order for ifftshift. Real data can be represented with zero imaginary components. Multidimensional axes are omitted.
/// ## Domain
///
/// Finite complex values and at most 1,048,576 entries. Empty input is returned unchanged; NaN/infinity are rejected.
/// ## Returns
///
/// A permuted complex vector of identical length.
/// ## Errors
///
/// Invalid sizes, shapes, axes, nonfinite inputs, work limits, or arithmetic overflow return shared errors.
/// ## Numerical behavior
///
/// O(n) exact permutation, with no arithmetic on sample values.
/// # Examples
///
/// ```
/// let x=[sci_complex::Complex{re:2.0,im:0.0}];
/// assert_eq!(sci_fft::fftshift(&x)?,x);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy fftshift](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fftshift.html).
pub fn fftshift(input: &[Complex]) -> Result<Vec<Complex>> {
    values(input)?;
    let mut out = input.to_vec();
    let mid = out.len().div_ceil(2);
    out.rotate_left(mid);
    Ok(out)
}

/// Undoes centering by rotating data left by floor(n/2).
///
/// ## Parameters
///
/// `input` is a complex one-dimensional vector in DFT order for fftshift, centered order for ifftshift. Real data can be represented with zero imaginary components. Multidimensional axes are omitted.
/// ## Domain
///
/// Finite complex values and at most 1,048,576 entries. Empty input is returned unchanged; NaN/infinity are rejected.
/// ## Returns
///
/// A permuted complex vector of identical length.
/// ## Errors
///
/// Invalid sizes, shapes, axes, nonfinite inputs, work limits, or arithmetic overflow return shared errors.
/// ## Numerical behavior
///
/// O(n) exact permutation, with no arithmetic on sample values.
/// # Examples
///
/// ```
/// let x=[sci_complex::Complex{re:2.0,im:0.0}];
/// assert_eq!(sci_fft::ifftshift(&x)?,x);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy ifftshift](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.ifftshift.html).
pub fn ifftshift(input: &[Complex]) -> Result<Vec<Complex>> {
    values(input)?;
    let mut out = input.to_vec();
    let mid = out.len() / 2;
    out.rotate_left(mid);
    Ok(out)
}

fn smooth(mut n: usize, real: bool) -> bool {
    for p in if real {
        &[2, 3, 5][..]
    } else {
        &[2, 3, 5, 7, 11][..]
    } {
        while n > 1 && n.is_multiple_of(*p) {
            n /= p;
        }
    }
    n == 1
}

/// Finds the smallest smooth length ≥ target.
///
/// ## Parameters
///
/// `real` selects SciPy 1.18.1 factors {2,3,5}; complex lengths use {2,3,5,7,11}. These are SciPy-compatible size recommendations; RustFFT performance depends on its own kernels.
/// ## Domain
///
/// Integer target in 0..=1,048,576.
/// ## Returns
///
/// The nearest supported smooth size in the selected direction. Zero returns zero.
/// ## Errors
///
/// Invalid sizes, shapes, axes, nonfinite inputs, work limits, or arithmetic overflow return shared errors.
/// ## Numerical behavior
///
/// Bounded trial division over at most 1,048,576 candidates; no transform planning. Factor sets are version-specific and do not promise optimal RustFFT speed.
/// # Examples
///
/// ```
/// assert_eq!(sci_fft::next_fast_len(17,true)?,18);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy next_fast_len](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.next_fast_len.html).
pub fn next_fast_len(target: usize, real: bool) -> Result<usize> {
    if target > LIMIT {
        return Err(Error::InvalidInput("fast length search limit"));
    }
    if target == 0 {
        return Ok(0);
    }
    (target..=LIMIT)
        .find(|n| smooth(*n, real))
        .ok_or(Error::NumericalFailure("no bounded fast FFT length"))
}

/// Finds the largest smooth length ≤ target.
///
/// ## Parameters
///
/// `real` selects SciPy 1.18.1 factors {2,3,5}; complex lengths use {2,3,5,7,11}. These are SciPy-compatible size recommendations; RustFFT performance depends on its own kernels.
/// ## Domain
///
/// Integer target in 0..=1,048,576.
/// ## Returns
///
/// The nearest supported smooth size in the selected direction. Zero returns zero.
/// ## Errors
///
/// Invalid sizes, shapes, axes, nonfinite inputs, work limits, or arithmetic overflow return shared errors.
/// ## Numerical behavior
///
/// Bounded trial division over at most 1,048,576 candidates; no transform planning. Factor sets are version-specific and do not promise optimal RustFFT speed.
/// # Examples
///
/// ```
/// assert_eq!(sci_fft::prev_fast_len(17,true)?,16);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy prev_fast_len](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.prev_fast_len.html).
pub fn prev_fast_len(target: usize, real: bool) -> Result<usize> {
    if target > LIMIT {
        return Err(Error::InvalidInput("fast length search limit"));
    }
    if target == 0 {
        return Ok(0);
    }
    (1..=target)
        .rev()
        .find(|n| smooth(*n, real))
        .ok_or(Error::NumericalFailure("no bounded fast FFT length"))
}

/// Real-transform kernel convention, matching SciPy types 1 through 4.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RealTransformType {
    /// Integer-grid cosine/sine transform with endpoint conventions.
    I,
    /// Half-shifted input grid; inverse kernel is type III.
    II,
    /// Half-shifted output grid; inverse kernel is type II.
    III,
    /// Half-shifted input and output grids; self-inverse up to scaling.
    IV,
}
type RealPlan = Box<dyn Fn(&mut [f64], &mut [f64])>;
fn real_plan(
    planner: &mut rustdct::DctPlanner<f64>,
    n: usize,
    kind: RealTransformType,
    sine: bool,
) -> (usize, RealPlan) {
    macro_rules! plan {
        ($build:ident,$run:ident) => {{
            let plan = planner.$build(n);
            let scratch = plan.get_scratch_len();
            (
                scratch,
                Box::new(move |x: &mut [f64], s: &mut [f64]| plan.$run(x, s)) as RealPlan,
            )
        }};
    }
    match (sine, kind) {
        (false, RealTransformType::I) => plan!(plan_dct1, process_dct1_with_scratch),
        (false, RealTransformType::II) => plan!(plan_dct2, process_dct2_with_scratch),
        (false, RealTransformType::III) => plan!(plan_dct3, process_dct3_with_scratch),
        (false, RealTransformType::IV) => plan!(plan_dct4, process_dct4_with_scratch),
        (true, RealTransformType::I) => plan!(plan_dst1, process_dst1_with_scratch),
        (true, RealTransformType::II) => plan!(plan_dst2, process_dst2_with_scratch),
        (true, RealTransformType::III) => plan!(plan_dst3, process_dst3_with_scratch),
        (true, RealTransformType::IV) => plan!(plan_dst4, process_dst4_with_scratch),
    }
}
fn real_nd(
    input: &[f64],
    shape: &[usize],
    axes: &[usize],
    kind: RealTransformType,
    sine: bool,
    inverse: bool,
) -> Result<Vec<f64>> {
    tensor(shape, axes, input.len())?;
    finite_slice(input)?;
    if !sine && kind == RealTransformType::I && axes.iter().any(|a| shape[*a] == 1) {
        return Err(Error::InvalidInput("DCT I needs at least two samples"));
    }
    let kernel = if inverse {
        match kind {
            RealTransformType::II => RealTransformType::III,
            RealTransformType::III => RealTransformType::II,
            v => v,
        }
    } else {
        kind
    };
    let mut planner = rustdct::DctPlanner::new();
    let mut out = input.to_vec();
    for &axis in axes {
        let n = shape[axis];
        let stride = shape[axis + 1..].iter().product::<usize>();
        let block = n * stride;
        let (scratch_len, plan) = real_plan(&mut planner, n, kernel, sine);
        let mut scratch = vec![0.0; scratch_len];
        let mut line = vec![0.0; n];
        let scale = if !inverse {
            2.0
        } else if kind == RealTransformType::I {
            1.0 / if sine { (n + 1) as f64 } else { (n - 1) as f64 }
        } else {
            1.0 / n as f64
        };
        for base in (0..out.len()).step_by(block) {
            for offset in 0..stride {
                for (k, v) in line.iter_mut().enumerate() {
                    *v = out[base + offset + k * stride];
                }
                plan(&mut line, &mut scratch);
                for (k, v) in line.iter().enumerate() {
                    out[base + offset + k * stride] = finite(v * scale)?;
                }
            }
        }
    }
    Ok(out)
}

/// Computes the separable discrete cosine transform of type I, II, III, or IV. Type II is `Y[k]=2 Σⱼ x[j]cos(πk(2j+1)/(2n))`; the other types use the endpoint and half-grid conventions below.
///
/// ## Parameters
///
/// `kind` selects SciPy’s integer type. `input` is a one-dimensional real slice. Only SciPy backward normalization with orthogonalize=false is supported. Type I cosine halves endpoint weights inside the doubled sum; type III cosine halves its first input coefficient. Type III sine halves its last coefficient with alternating sign. Type I sine uses sin(π(j+1)(k+1)/(n+1)); type IV uses half-grid shifts on both sides. Type I cosine uses cos(πjk/(n−1)); type III uses cos(πj(2k+1)/(2n)). Inverse type II uses the type III kernel and vice versa; type I and IV use their own kernels, divided by the forward/inverse composition factor.
/// ## Domain
///
/// Finite real input, nonzero dimensions, at most 1,048,576 entries and 16 axes. Type I cosine requires length ≥2 on every transformed axis. NaN/infinity are rejected.
/// ## Returns
///
/// Real coefficients or samples in flat input ordering, of the same shape.
/// ## Errors
///
/// Invalid sizes, shapes, axes, nonfinite inputs, work limits, or arithmetic overflow return shared errors.
/// ## Numerical behavior
///
/// Private RustDCT backend reuses a plan and scratch per axis. Backward forward scaling is twice the backend kernel; inverses use reciprocal n, n−1 for DCT I, or n+1 for DST I. No resizing, orthonormal normalization, workers, or backend controls. f64 roundoff and cancellation apply.
/// # Examples
///
/// ```
/// let x=sci_fft::dct(&[1.0,2.0],sci_fft::RealTransformType::II)?;
/// assert_eq!(x.len(),2);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy dct](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html).
pub fn dct(input: &[f64], kind: RealTransformType) -> Result<Vec<f64>> {
    real_nd(input, &[input.len()], &[0], kind, false, false)
}

/// Inverts the separable discrete cosine transform of type I, II, III, or IV. Type II is `Y[k]=2 Σⱼ x[j]cos(πk(2j+1)/(2n))`; the other types use the endpoint and half-grid conventions below.
///
/// ## Parameters
///
/// `kind` selects SciPy’s integer type. `input` is a one-dimensional real slice. Only SciPy backward normalization with orthogonalize=false is supported. Type I cosine halves endpoint weights inside the doubled sum; type III cosine halves its first input coefficient. Type III sine halves its last coefficient with alternating sign. Type I sine uses sin(π(j+1)(k+1)/(n+1)); type IV uses half-grid shifts on both sides. Type I cosine uses cos(πjk/(n−1)); type III uses cos(πj(2k+1)/(2n)). Inverse type II uses the type III kernel and vice versa; type I and IV use their own kernels, divided by the forward/inverse composition factor.
/// ## Domain
///
/// Finite real input, nonzero dimensions, at most 1,048,576 entries and 16 axes. Type I cosine requires length ≥2 on every transformed axis. NaN/infinity are rejected.
/// ## Returns
///
/// Real coefficients or samples in flat input ordering, of the same shape.
/// ## Errors
///
/// Invalid sizes, shapes, axes, nonfinite inputs, work limits, or arithmetic overflow return shared errors.
/// ## Numerical behavior
///
/// Private RustDCT backend reuses a plan and scratch per axis. Backward forward scaling is twice the backend kernel; inverses use reciprocal n, n−1 for DCT I, or n+1 for DST I. No resizing, orthonormal normalization, workers, or backend controls. f64 roundoff and cancellation apply.
/// # Examples
///
/// ```
/// let x=sci_fft::idct(&[1.0,2.0],sci_fft::RealTransformType::II)?;
/// assert_eq!(x.len(),2);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy idct](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.idct.html).
pub fn idct(input: &[f64], kind: RealTransformType) -> Result<Vec<f64>> {
    real_nd(input, &[input.len()], &[0], kind, false, true)
}

/// Computes the separable discrete sine transform of type I, II, III, or IV. Type II is `Y[k]=2 Σⱼ x[j]sin(π(k+1)(2j+1)/(2n))`; the other types use the endpoint and half-grid conventions below.
///
/// ## Parameters
///
/// `kind` selects SciPy’s integer type. `input` is a one-dimensional real slice. Only SciPy backward normalization with orthogonalize=false is supported. Type I cosine halves endpoint weights inside the doubled sum; type III cosine halves its first input coefficient. Type III sine halves its last coefficient with alternating sign. Type I sine uses sin(π(j+1)(k+1)/(n+1)); type IV uses half-grid shifts on both sides. Type III sine uses sin(π(2k+1)(j+1)/(2n)). Inverse type II uses the type III kernel and vice versa; type I and IV use their own kernels, divided by the forward/inverse composition factor.
/// ## Domain
///
/// Finite real input, nonzero dimensions, at most 1,048,576 entries and 16 axes. Type I cosine requires length ≥2 on every transformed axis. NaN/infinity are rejected.
/// ## Returns
///
/// Real coefficients or samples in flat input ordering, of the same shape.
/// ## Errors
///
/// Invalid sizes, shapes, axes, nonfinite inputs, work limits, or arithmetic overflow return shared errors.
/// ## Numerical behavior
///
/// Private RustDCT backend reuses a plan and scratch per axis. Backward forward scaling is twice the backend kernel; inverses use reciprocal n, n−1 for DCT I, or n+1 for DST I. No resizing, orthonormal normalization, workers, or backend controls. f64 roundoff and cancellation apply.
/// # Examples
///
/// ```
/// let x=sci_fft::dst(&[1.0,2.0],sci_fft::RealTransformType::II)?;
/// assert_eq!(x.len(),2);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy dst](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dst.html).
pub fn dst(input: &[f64], kind: RealTransformType) -> Result<Vec<f64>> {
    real_nd(input, &[input.len()], &[0], kind, true, false)
}

/// Inverts the separable discrete sine transform of type I, II, III, or IV. Type II is `Y[k]=2 Σⱼ x[j]sin(π(k+1)(2j+1)/(2n))`; the other types use the endpoint and half-grid conventions below.
///
/// ## Parameters
///
/// `kind` selects SciPy’s integer type. `input` is a one-dimensional real slice. Only SciPy backward normalization with orthogonalize=false is supported. Type I cosine halves endpoint weights inside the doubled sum; type III cosine halves its first input coefficient. Type III sine halves its last coefficient with alternating sign. Type I sine uses sin(π(j+1)(k+1)/(n+1)); type IV uses half-grid shifts on both sides. Type III sine uses sin(π(2k+1)(j+1)/(2n)). Inverse type II uses the type III kernel and vice versa; type I and IV use their own kernels, divided by the forward/inverse composition factor.
/// ## Domain
///
/// Finite real input, nonzero dimensions, at most 1,048,576 entries and 16 axes. Type I cosine requires length ≥2 on every transformed axis. NaN/infinity are rejected.
/// ## Returns
///
/// Real coefficients or samples in flat input ordering, of the same shape.
/// ## Errors
///
/// Invalid sizes, shapes, axes, nonfinite inputs, work limits, or arithmetic overflow return shared errors.
/// ## Numerical behavior
///
/// Private RustDCT backend reuses a plan and scratch per axis. Backward forward scaling is twice the backend kernel; inverses use reciprocal n, n−1 for DCT I, or n+1 for DST I. No resizing, orthonormal normalization, workers, or backend controls. f64 roundoff and cancellation apply.
/// # Examples
///
/// ```
/// let x=sci_fft::idst(&[1.0,2.0],sci_fft::RealTransformType::II)?;
/// assert_eq!(x.len(),2);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy idst](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.idst.html).
pub fn idst(input: &[f64], kind: RealTransformType) -> Result<Vec<f64>> {
    real_nd(input, &[input.len()], &[0], kind, true, true)
}

/// Computes the separable discrete cosine transform of type I, II, III, or IV. Type II is `Y[k]=2 Σⱼ x[j]cos(πk(2j+1)/(2n))`; the other types use the endpoint and half-grid conventions below.
///
/// ## Parameters
///
/// `kind` selects SciPy’s integer type. `shape` is flat row-major storage shape; `axes` are distinct transform axes. Empty axes copy input. Only SciPy backward normalization with orthogonalize=false is supported. Type I cosine halves endpoint weights inside the doubled sum; type III cosine halves its first input coefficient. Type III sine halves its last coefficient with alternating sign. Type I sine uses sin(π(j+1)(k+1)/(n+1)); type IV uses half-grid shifts on both sides. Type I cosine uses cos(πjk/(n−1)); type III uses cos(πj(2k+1)/(2n)). Inverse type II uses the type III kernel and vice versa; type I and IV use their own kernels, divided by the forward/inverse composition factor.
/// ## Domain
///
/// Finite real input, nonzero dimensions, at most 1,048,576 entries and 16 axes. Type I cosine requires length ≥2 on every transformed axis. NaN/infinity are rejected.
/// ## Returns
///
/// Real coefficients or samples in flat input ordering, of the same shape.
/// ## Errors
///
/// Invalid sizes, shapes, axes, nonfinite inputs, work limits, or arithmetic overflow return shared errors.
/// ## Numerical behavior
///
/// Private RustDCT backend reuses a plan and scratch per axis. Backward forward scaling is twice the backend kernel; inverses use reciprocal n, n−1 for DCT I, or n+1 for DST I. No resizing, orthonormal normalization, workers, or backend controls. f64 roundoff and cancellation apply.
/// # Examples
///
/// ```
/// let x=sci_fft::dctn(&[1.0,2.0],&[2],&[0],sci_fft::RealTransformType::II)?;
/// assert_eq!(x.len(),2);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy dctn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dctn.html).
pub fn dctn(
    input: &[f64],
    shape: &[usize],
    axes: &[usize],
    kind: RealTransformType,
) -> Result<Vec<f64>> {
    real_nd(input, shape, axes, kind, false, false)
}

/// Inverts the separable discrete cosine transform of type I, II, III, or IV. Type II is `Y[k]=2 Σⱼ x[j]cos(πk(2j+1)/(2n))`; the other types use the endpoint and half-grid conventions below.
///
/// ## Parameters
///
/// `kind` selects SciPy’s integer type. `shape` is flat row-major storage shape; `axes` are distinct transform axes. Empty axes copy input. Only SciPy backward normalization with orthogonalize=false is supported. Type I cosine halves endpoint weights inside the doubled sum; type III cosine halves its first input coefficient. Type III sine halves its last coefficient with alternating sign. Type I sine uses sin(π(j+1)(k+1)/(n+1)); type IV uses half-grid shifts on both sides. Type I cosine uses cos(πjk/(n−1)); type III uses cos(πj(2k+1)/(2n)). Inverse type II uses the type III kernel and vice versa; type I and IV use their own kernels, divided by the forward/inverse composition factor.
/// ## Domain
///
/// Finite real input, nonzero dimensions, at most 1,048,576 entries and 16 axes. Type I cosine requires length ≥2 on every transformed axis. NaN/infinity are rejected.
/// ## Returns
///
/// Real coefficients or samples in flat input ordering, of the same shape.
/// ## Errors
///
/// Invalid sizes, shapes, axes, nonfinite inputs, work limits, or arithmetic overflow return shared errors.
/// ## Numerical behavior
///
/// Private RustDCT backend reuses a plan and scratch per axis. Backward forward scaling is twice the backend kernel; inverses use reciprocal n, n−1 for DCT I, or n+1 for DST I. No resizing, orthonormal normalization, workers, or backend controls. f64 roundoff and cancellation apply.
/// # Examples
///
/// ```
/// let x=sci_fft::idctn(&[1.0,2.0],&[2],&[0],sci_fft::RealTransformType::II)?;
/// assert_eq!(x.len(),2);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy idctn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.idctn.html).
pub fn idctn(
    input: &[f64],
    shape: &[usize],
    axes: &[usize],
    kind: RealTransformType,
) -> Result<Vec<f64>> {
    real_nd(input, shape, axes, kind, false, true)
}

/// Computes the separable discrete sine transform of type I, II, III, or IV. Type II is `Y[k]=2 Σⱼ x[j]sin(π(k+1)(2j+1)/(2n))`; the other types use the endpoint and half-grid conventions below.
///
/// ## Parameters
///
/// `kind` selects SciPy’s integer type. `shape` is flat row-major storage shape; `axes` are distinct transform axes. Empty axes copy input. Only SciPy backward normalization with orthogonalize=false is supported. Type I cosine halves endpoint weights inside the doubled sum; type III cosine halves its first input coefficient. Type III sine halves its last coefficient with alternating sign. Type I sine uses sin(π(j+1)(k+1)/(n+1)); type IV uses half-grid shifts on both sides. Type III sine uses sin(π(2k+1)(j+1)/(2n)). Inverse type II uses the type III kernel and vice versa; type I and IV use their own kernels, divided by the forward/inverse composition factor.
/// ## Domain
///
/// Finite real input, nonzero dimensions, at most 1,048,576 entries and 16 axes. Type I cosine requires length ≥2 on every transformed axis. NaN/infinity are rejected.
/// ## Returns
///
/// Real coefficients or samples in flat input ordering, of the same shape.
/// ## Errors
///
/// Invalid sizes, shapes, axes, nonfinite inputs, work limits, or arithmetic overflow return shared errors.
/// ## Numerical behavior
///
/// Private RustDCT backend reuses a plan and scratch per axis. Backward forward scaling is twice the backend kernel; inverses use reciprocal n, n−1 for DCT I, or n+1 for DST I. No resizing, orthonormal normalization, workers, or backend controls. f64 roundoff and cancellation apply.
/// # Examples
///
/// ```
/// let x=sci_fft::dstn(&[1.0,2.0],&[2],&[0],sci_fft::RealTransformType::II)?;
/// assert_eq!(x.len(),2);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy dstn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dstn.html).
pub fn dstn(
    input: &[f64],
    shape: &[usize],
    axes: &[usize],
    kind: RealTransformType,
) -> Result<Vec<f64>> {
    real_nd(input, shape, axes, kind, true, false)
}

/// Inverts the separable discrete sine transform of type I, II, III, or IV. Type II is `Y[k]=2 Σⱼ x[j]sin(π(k+1)(2j+1)/(2n))`; the other types use the endpoint and half-grid conventions below.
///
/// ## Parameters
///
/// `kind` selects SciPy’s integer type. `shape` is flat row-major storage shape; `axes` are distinct transform axes. Empty axes copy input. Only SciPy backward normalization with orthogonalize=false is supported. Type I cosine halves endpoint weights inside the doubled sum; type III cosine halves its first input coefficient. Type III sine halves its last coefficient with alternating sign. Type I sine uses sin(π(j+1)(k+1)/(n+1)); type IV uses half-grid shifts on both sides. Type III sine uses sin(π(2k+1)(j+1)/(2n)). Inverse type II uses the type III kernel and vice versa; type I and IV use their own kernels, divided by the forward/inverse composition factor.
/// ## Domain
///
/// Finite real input, nonzero dimensions, at most 1,048,576 entries and 16 axes. Type I cosine requires length ≥2 on every transformed axis. NaN/infinity are rejected.
/// ## Returns
///
/// Real coefficients or samples in flat input ordering, of the same shape.
/// ## Errors
///
/// Invalid sizes, shapes, axes, nonfinite inputs, work limits, or arithmetic overflow return shared errors.
/// ## Numerical behavior
///
/// Private RustDCT backend reuses a plan and scratch per axis. Backward forward scaling is twice the backend kernel; inverses use reciprocal n, n−1 for DCT I, or n+1 for DST I. No resizing, orthonormal normalization, workers, or backend controls. f64 roundoff and cancellation apply.
/// # Examples
///
/// ```
/// let x=sci_fft::idstn(&[1.0,2.0],&[2],&[0],sci_fft::RealTransformType::II)?;
/// assert_eq!(x.len(),2);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy idstn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.idstn.html).
pub fn idstn(
    input: &[f64],
    shape: &[usize],
    axes: &[usize],
    kind: RealTransformType,
) -> Result<Vec<f64>> {
    real_nd(input, shape, axes, kind, true, true)
}
