//! # Fourier transforms
//!
//! Complex discrete Fourier transforms using RustFFT internally. Inputs and
//! outputs use [`sci_complex::Complex`]; real-input transforms accept the
//! same `f32` or `f64` slices used by descriptive statistics.
//!
//! ## Conventions
//!
//! The forward transform is unnormalized; the inverse divides by the length.
//! Frequencies appear in standard DFT order, starting with the zero-frequency
//! bin. No windowing or sample-spacing scaling is applied.
//!
//! ## Examples
//!
//! ```
//! use sci_fft::real as fft; use sci_complex as complex;
//! let spectrum = fft::rfft(&[1.0, 1.0, 1.0, 1.0]);
//! assert_eq!(complex::abs(spectrum[0]), 4.0);
//! ```
//!
//! ## References
//!
//! - [SciPy FFT conventions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html).

use crate::complex::{Complex, native, owned};

use super::FftFloat;
use num_complex::Complex as Native;
use std::{cell::RefCell, sync::Arc};

// Keep only the most recent plan of each kind. Larger transforms use temporary
// workspaces, avoiding unbounded retained memory when lengths vary.
const CACHE_LENGTH: usize = 65_536;

#[derive(Default)]
struct Cache<T: FftFloat> {
    forward: Option<ComplexWorkspace<T>>,
    inverse: Option<ComplexWorkspace<T>>,
    real: Option<RealWorkspace<T>>,
    inverse_real: Option<InverseRealWorkspace<T>>,
}
thread_local! {
    static CACHE_F32: RefCell<Cache<f32>> = RefCell::new(Cache::default());
    static CACHE_F64: RefCell<Cache<f64>> = RefCell::new(Cache::default());
}

// Dispatch to independent per-precision caches without exposing backend types.
pub(super) mod private {
    use super::*;
    pub trait Backend: rustfft::FftNum {
        fn transform(input: &[Complex<Self>], inverse: bool) -> Vec<Complex<Self>>;
        fn real(input: &[Self]) -> Vec<Complex<Self>>;
        fn real_into(input: &[Self], output: &mut [Complex<Self>]);
        fn real_strided(input: &[Self], output: &mut [Complex<Self>], n: usize, stride: usize);
        fn inverse_real(input: &[Complex<Self>], n: usize) -> crate::Result<Vec<Self>>;
        fn inverse_real_into(input: &[Complex<Self>], output: &mut [Self]) -> crate::Result<()>;
    }
    macro_rules! backend {
        ($t:ty, $cache:ident) => {
            impl Backend for $t {
                fn transform(input: &[Complex<Self>], inverse: bool) -> Vec<Complex<Self>> {
                    $cache.with_borrow_mut(|cache| transform(input, inverse, cache))
                }
                fn real(input: &[Self]) -> Vec<Complex<Self>> {
                    let mut output = vec![
                        Complex { re: 0.0, im: 0.0 };
                        if input.is_empty() {
                            0
                        } else {
                            input.len() / 2 + 1
                        }
                    ];
                    Self::real_into(input, &mut output);
                    output
                }
                fn real_into(input: &[Self], output: &mut [Complex<Self>]) {
                    $cache.with_borrow_mut(|cache| real_transform_into(input, output, cache));
                }
                fn real_strided(
                    input: &[Self],
                    output: &mut [Complex<Self>],
                    n: usize,
                    stride: usize,
                ) {
                    $cache.with_borrow_mut(|cache| {
                        real_transform_strided(input, output, n, stride, cache)
                    });
                }
                fn inverse_real(input: &[Complex<Self>], n: usize) -> crate::Result<Vec<Self>> {
                    let mut output = vec![0.0; n];
                    Self::inverse_real_into(input, &mut output)?;
                    Ok(output)
                }
                fn inverse_real_into(
                    input: &[Complex<Self>],
                    output: &mut [Self],
                ) -> crate::Result<()> {
                    $cache
                        .with_borrow_mut(|cache| inverse_real_transform_into(input, output, cache))
                }
            }
        };
    }
    backend!(f32, CACHE_F32);
    backend!(f64, CACHE_F64);
}

struct ComplexWorkspace<T: FftFloat> {
    plan: Arc<dyn rustfft::Fft<T>>,
    buffer: Vec<Native<T>>,
    scratch: Vec<Native<T>>,
}
impl<T: FftFloat> ComplexWorkspace<T> {
    fn new(n: usize, inverse: bool) -> Self {
        let mut planner = rustfft::FftPlanner::<T>::new();
        let plan = if inverse {
            planner.plan_fft_inverse(n)
        } else {
            planner.plan_fft_forward(n)
        };
        Self {
            scratch: vec![Native::new(T::zero(), T::zero()); plan.get_inplace_scratch_len()],
            buffer: vec![Native::new(T::zero(), T::zero()); n],
            plan,
        }
    }
    fn run(&mut self, input: &[Complex<T>], inverse: bool) -> Vec<Complex<T>> {
        for (buffer, &value) in self.buffer.iter_mut().zip(input) {
            *buffer = native(value);
        }
        self.plan
            .process_with_scratch(&mut self.buffer, &mut self.scratch);
        let scale = if inverse {
            T::from(input.len()).unwrap()
        } else {
            T::one()
        };
        self.buffer
            .iter()
            .map(|&value| owned(value / scale))
            .collect()
    }
}

struct RealWorkspace<T: FftFloat> {
    plan: Arc<dyn realfft::RealToComplex<T>>,
    buffer: Vec<T>,
    output: Vec<Native<T>>,
    scratch: Vec<Native<T>>,
}
impl<T: FftFloat> RealWorkspace<T> {
    fn new(n: usize) -> Self {
        let plan = realfft::RealFftPlanner::<T>::new().plan_fft_forward(n);
        Self {
            buffer: plan.make_input_vec(),
            output: plan.make_output_vec(),
            scratch: plan.make_scratch_vec(),
            plan,
        }
    }
    fn run_into(&mut self, input: &[T], output: &mut [Complex<T>]) {
        self.buffer.copy_from_slice(input);
        if self
            .plan
            .process_with_scratch(&mut self.buffer, &mut self.output, &mut self.scratch)
            .is_err()
        {
            // Preserve the infallible API if the backend rejects a workspace.
            // These buffers are always sized by the same plan.
            let values: Vec<_> = input
                .iter()
                .map(|&re| Complex { re, im: T::zero() })
                .collect();
            let mut result = ComplexWorkspace::new(input.len(), false).run(&values, false);
            result.truncate(input.len() / 2 + 1);
            output.copy_from_slice(&result);
            return;
        }
        for (dst, &value) in output.iter_mut().zip(&self.output) {
            *dst = owned(value);
        }
    }

    // Process all lines with one cached plan, one workspace borrow and direct
    // gather/scatter. Short strided axes otherwise pay a cache dispatch and
    // two extra buffer copies for every transform.
    fn run_strided(&mut self, input: &[T], output: &mut [Complex<T>], stride: usize) {
        let n = self.buffer.len();
        let block = n * stride;
        let small_block = self.output.len() * stride;
        for (outer, chunk) in input.chunks_exact(block).enumerate() {
            for offset in 0..stride {
                for (k, value) in self.buffer.iter_mut().enumerate() {
                    *value = chunk[offset + k * stride];
                }
                let base = outer * small_block + offset;
                if self
                    .plan
                    .process_with_scratch(&mut self.buffer, &mut self.output, &mut self.scratch)
                    .is_err()
                {
                    // The plan and buffers have matching dimensions, so this
                    // should never occur; preserve the public fallback path.
                    let values: Vec<_> = (0..n)
                        .map(|k| Complex {
                            re: chunk[offset + k * stride],
                            im: T::zero(),
                        })
                        .collect();
                    let result = ComplexWorkspace::new(n, false).run(&values, false);
                    for (k, value) in result.into_iter().take(self.output.len()).enumerate() {
                        output[base + k * stride] = value;
                    }
                } else {
                    for (k, &value) in self.output.iter().enumerate() {
                        output[base + k * stride] = owned(value);
                    }
                }
            }
        }
    }
}

struct InverseRealWorkspace<T: FftFloat> {
    plan: Arc<dyn realfft::ComplexToReal<T>>,
    input: Vec<Native<T>>,
    output: Vec<T>,
    scratch: Vec<Native<T>>,
}
impl<T: FftFloat> InverseRealWorkspace<T> {
    fn new(n: usize) -> Self {
        let plan = realfft::RealFftPlanner::<T>::new().plan_fft_inverse(n);
        Self {
            input: plan.make_input_vec(),
            output: plan.make_output_vec(),
            scratch: plan.make_scratch_vec(),
            plan,
        }
    }

    fn run_into(&mut self, bins: &[Complex<T>], output: &mut [T]) -> crate::Result<()> {
        self.input.fill(Native::new(T::zero(), T::zero()));
        for (dst, &src) in self.input.iter_mut().zip(bins) {
            *dst = native(src);
        }
        // The imaginary parts of DC and even Nyquist do not contribute.
        self.input[0].im = T::zero();
        if self.output.len().is_multiple_of(2) {
            self.input[self.output.len() / 2].im = T::zero();
        }
        self.plan
            .process_with_scratch(&mut self.input, &mut self.output, &mut self.scratch)
            .map_err(|_| crate::Error::NumericalFailure("inverse real FFT backend failed"))?;
        let scale = T::from(self.output.len()).unwrap();
        for (dst, &value) in output.iter_mut().zip(&self.output) {
            *dst = match value / scale {
                result if result.is_finite() => Ok(result),
                _ => Err(crate::Error::NonFiniteEvaluation),
            }?;
        }
        Ok(())
    }
}

fn transform<T: FftFloat>(
    input: &[Complex<T>],
    inverse: bool,
    cache: &mut Cache<T>,
) -> Vec<Complex<T>> {
    if input.is_empty() {
        return Vec::new();
    }
    if input.len() > CACHE_LENGTH {
        return ComplexWorkspace::new(input.len(), inverse).run(input, inverse);
    }
    let slot = if inverse {
        &mut cache.inverse
    } else {
        &mut cache.forward
    };
    let workspace = slot.get_or_insert_with(|| ComplexWorkspace::new(input.len(), inverse));
    if workspace.buffer.len() != input.len() {
        *workspace = ComplexWorkspace::new(input.len(), inverse);
    }
    workspace.run(input, inverse)
}

/// Computes `X[k] = Σ x[j] exp(−2πi j k / n)`.
///
/// ## Parameters
///
/// `input` contains complex samples in time order.
///
/// ## Domain
///
/// Any length, including zero. Nonfinite components propagate through arithmetic.
///
/// ## Returns
///
/// All `n` complex frequency bins in standard DFT order; empty input gives empty output.
///
/// ## Numerical behavior
///
/// O(n log n) private backend computation. Each thread retains the most recent
/// forward, inverse, real forward, and real inverse FFT plans and work buffers for lengths up to
/// 65,536. Larger lengths use temporary workspaces. Plan storage does not grow
/// with the number of different lengths. Each precision has its own cache.
/// Roundoff depends on the selected type, length, and data.
///
/// # Examples
///
/// ```
/// use sci_fft::fourier as fft; use sci_complex::Complex;
/// assert_eq!(fft::fft(&[Complex { re: 2.0, im: 1.0 }]),
///            vec![Complex { re: 2.0, im: 1.0 }]);
/// ```
pub fn fft<T: FftFloat>(input: &[Complex<T>]) -> Vec<Complex<T>> {
    T::transform(input, false)
}

/// Computes `x[j] = (1/n) Σ X[k] exp(2πi j k / n)`.
///
/// ## Parameters
///
/// `input` contains all `n` complex bins in standard DFT order. The inverse
/// normalization makes `ifft(fft(x))` approximately equal to `x`.
///
/// ## Domain
///
/// Any length, including zero. Nonfinite components propagate through arithmetic.
///
/// ## Returns
///
/// `n` complex samples in time order; empty input gives empty output.
///
/// ## Numerical behavior
///
/// O(n log n) private backend computation. Each thread retains the most recent
/// forward, inverse, real forward, and real inverse FFT plans and work buffers for lengths up to
/// 65,536. Larger lengths use temporary workspaces. Plan storage does not grow
/// with the number of different lengths. Each precision has its own cache.
/// Roundoff depends on the selected type, length, and data.
///
/// # Examples
///
/// ```
/// use sci_fft::fourier as fft; use sci_complex::Complex;
/// let input = [Complex { re: 3.0, im: -2.0 }];
/// assert_eq!(fft::ifft(&fft::fft(&input)), input);
/// ```
pub fn ifft<T: FftFloat>(input: &[Complex<T>]) -> Vec<Complex<T>> {
    T::transform(input, true)
}

/// Computes the nonnegative-frequency half of a real-input DFT.
///
/// ## Parameters
///
/// `input` contains real samples in time order. The transform is unnormalized,
/// matching [`fft`]. The private RealFFT backend uses a half-length complex
/// transform for even lengths and a full-length transform for odd lengths.
///
/// ## Domain
///
/// Any length, including zero. Nonfinite samples propagate through arithmetic.
///
/// ## Returns
///
/// For `n > 0`, `floor(n/2) + 1` complex bins, including DC and, for even lengths,
/// Nyquist. Empty input gives empty output.
///
/// ## Numerical behavior
///
/// O(n log n) private backend computation. Each thread retains the most recent
/// forward, inverse, real forward, and real inverse FFT plans and work buffers for lengths up to
/// 65,536. Larger lengths use temporary workspaces. Plan storage does not grow
/// with the number of different lengths. Each precision has its own cache.
/// Roundoff depends on the selected type, length, and data.
///
/// # Examples
///
/// ```
/// let result = sci_fft::real::rfft(&[1.0, 1.0, 1.0, 1.0]);
/// assert_eq!(result.len(), 3);
/// assert_eq!(result[0].re, 4.0);
/// ```
pub fn rfft<T: FftFloat>(input: &[T]) -> Vec<Complex<T>> {
    T::real(input)
}

pub(super) fn real_strided<T: FftFloat>(
    input: &[T],
    output: &mut [Complex<T>],
    n: usize,
    stride: usize,
) {
    T::real_strided(input, output, n, stride)
}

fn real_transform_into<T: FftFloat>(input: &[T], output: &mut [Complex<T>], cache: &mut Cache<T>) {
    if input.is_empty() {
        return;
    }
    if input.len() > CACHE_LENGTH {
        return RealWorkspace::new(input.len()).run_into(input, output);
    }
    let workspace = cache
        .real
        .get_or_insert_with(|| RealWorkspace::new(input.len()));
    if workspace.buffer.len() != input.len() {
        *workspace = RealWorkspace::new(input.len());
    }
    workspace.run_into(input, output)
}

fn real_transform_strided<T: FftFloat>(
    input: &[T],
    output: &mut [Complex<T>],
    n: usize,
    stride: usize,
    cache: &mut Cache<T>,
) {
    if n > CACHE_LENGTH {
        return RealWorkspace::new(n).run_strided(input, output, stride);
    }
    let workspace = cache.real.get_or_insert_with(|| RealWorkspace::new(n));
    if workspace.buffer.len() != n {
        *workspace = RealWorkspace::new(n);
    }
    workspace.run_strided(input, output, stride);
}

pub(super) fn inverse_real<T: FftFloat>(input: &[Complex<T>], n: usize) -> crate::Result<Vec<T>> {
    T::inverse_real(input, n)
}

pub(super) fn inverse_real_into<T: FftFloat>(
    input: &[Complex<T>],
    output: &mut [T],
) -> crate::Result<()> {
    T::inverse_real_into(input, output)
}

fn inverse_real_transform_into<T: FftFloat>(
    input: &[Complex<T>],
    output: &mut [T],
    cache: &mut Cache<T>,
) -> crate::Result<()> {
    let n = output.len();
    if n > CACHE_LENGTH {
        return InverseRealWorkspace::new(n).run_into(input, output);
    }
    let workspace = cache
        .inverse_real
        .get_or_insert_with(|| InverseRealWorkspace::new(n));
    if workspace.output.len() != n {
        *workspace = InverseRealWorkspace::new(n);
    }
    workspace.run_into(input, output)
}
