# Sci-Stats

Statistical analysis, hypothesis tests, and probability distributions.

Part of [Sci](https://github.com/p-sira/sci), a unified scientific computing library for Rust.

Descriptive statistics use generic floating-point values; hypothesis tests and
native distributions use `f64`.
Distribution evaluation uses statrs and shared special functions privately; callers supply parameters
directly rather than constructing a backend distribution.

Finite observations are required, with at most 1,048,576 samples per slice.
Mean is `Σx/n`; population variance uses denominator `n`, while sample
variance uses `n - 1` (`ddof = 1`). Probability functions use the
parameterization stated in each function and return the documented IEEE
endpoint values; invalid parameters are reported as `Error`.

In Sci 0.3.0, statistical functions and their result types are grouped by
family. For example, use `sci_stats::descriptive::mean`,
`sci_stats::distributions::normal_cdf`, `sci_stats::inference::ttest_ind`,
and `sci_stats::association::pearsonr`. The shared `Error` and `Result`
types remain at the crate root.

## Examples

```rust
fn main() -> Result<(), sci_core::Error> {
    use sci_complex as complex; use sci_fft as fft; use sci_stats as stats;
    let amplitudes: Vec<f64> = fft::rfft(&[1.0, 2.0, 3.0, 4.0])
        .into_iter().map(complex::abs).collect();
    assert!(stats::descriptive::mean(&amplitudes)? > 0.0);
    Ok(())
}
```

## Acknowledgments

We thank `nalgebra`, `statrs`, `basin`,
`lapjv`, `ndarray`, and SciPy (BSD-3-Clause) for their statistical and
numerical building blocks.
