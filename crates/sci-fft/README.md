# Sci-FFT

The default `hankel` feature enables `hankel::{fht, ifht, fhtoffset}` with the private Rust gamma backend. All transforms build without XSF, C++ compilation, or libclang. Use `default-features = false` to omit FFTLog. The Sci facade selects this capability with `fft-hankel`.

Complex transforms are under `fourier`, real Fourier and cosine/sine transforms
under `real`, FFTLog under `hankel`, and frequency/periodic helpers under
`utilities`. `FftFloat` and `RealTransformType` remain at the crate root.

Fourier, cosine/sine, and periodic transforms support native `f32` and `f64`; FFTLog Hankel transforms use `f64`.

Fourier transforms on shared real and complex data.

Part of [Sci](https://github.com/p-sira/sci), a unified scientific computing library for Rust.

Forward transforms are unnormalized; inverses divide by transformed lengths.
The multidimensional and extended transform functions validate finite data and bound work to 1,048,576 entries and 16 axes. Multidimensional input uses flat row-major storage plus explicit
shapes and axes. Two-dimensional wrappers accept shared row matrices.
Periodic differential, Hilbert, Tilbert, hyperbolic, and shift operators
accept finite real samples and reuse the shared Fourier kernels. Their sign,
endpoint, and Nyquist conventions are documented per function.
Signal processing reuses `fourier::{fft, ifft}` and `real::rfft` from this crate.

For a length-`n` sequence, the forward transform is
`X[k] = Σ x[j] exp(-2π i j k/n)` and the inverse is
`x[j] = (1/n) Σ X[k] exp(2π i j k/n)`. Real transforms use the same
forward sign and return the nonnegative-frequency half-spectrum.

## Acknowledgments

We thank `num-complex`, `rustfft`, `realfft`, `rustdct`,
and `special` for their valuable numerical building blocks.
