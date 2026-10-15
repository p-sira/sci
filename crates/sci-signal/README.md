# Sci-Signal

Signal processing on one-dimensional real samples and shared complex values.

Part of [Sci](https://github.com/p-sira/sci), a unified scientific computing library for Rust.

Functions use explicit parameters. Digital angular frequencies use
radians/sample unless a sample rate is supplied; analog frequencies use
radians/time.
Finite inputs are required by fallible functions. Arithmetic overflow is
reported as an evaluation error. Existing Fourier helpers retain their IEEE
propagation and empty-input conventions.

For finite sequences, full convolution is
`y[k] = Σ x[j] h[k-j]`, with samples outside either input treated as zero.
Frequency responses and digital filter frequencies use radians/sample;
analog frequencies use radians per unit time. Coefficients are returned in
the ordering required by the corresponding filtering function.

# Example

```rust
fn main() -> Result<(), sci_core::Error> {
    use sci_signal::{convolve, Mode};
    assert_eq!(convolve(&[1., 2.], &[1., -1.], Mode::Full)?, vec![1., 1., -2.]);
    Ok(())
}
```

## Acknowledgments

We thank `nalgebra`, `num-complex`, `pm-remez`, and
`rustfft` for their signal-processing and numerical building blocks.
