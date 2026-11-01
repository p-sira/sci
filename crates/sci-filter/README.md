# Sci-Filter

Functional Bayesian state estimation on shared row-major `Matrix` values.

Kalman functions live under `sci_filter::kalman` and particle functions under
`sci_filter::particle`. Shared `Gaussian`, `Update`, `SigmaPoints`, and
`ParticleSet` types remain available from the crate root.

Part of [Sci](https://github.com/p-sira/sci), a unified scientific computing library for Rust.

Linear and extended Kalman steps and the scaled unscented filter delegate to
private Smelly backends, including particle propagation and moments;
weight normalization, effective sample size, and systematic resampling are native.
Sample filtering and FIR/IIR design remain in `sci-signal`.

Every call validates finite values, shapes, covariance matrices, and work
limits. Functions return owned results; failures leave their inputs unchanged.
Models use additive, independent process and observation noise. Coordinates
are Euclidean: angle wrapping and manifold residuals are not supported.
Covariances are symmetric positive-semidefinite on input; sigma-point
filtering additionally requires positive definiteness. Singular innovation
matrices and nonfinite evaluations are reported through `Error` rather
than converted to a plausible estimate.

```rust
fn main() -> Result<(), sci_filter::Error> {
    use sci_filter::{Gaussian, kalman::{kalman_predict, kalman_update}};
    let prior = Gaussian { mean: vec![0.], covariance: vec![vec![1.]] };
    let predicted = kalman_predict(&prior, &[vec![1.]], &[vec![0.1]])?;
    let updated = kalman_update(&predicted, &[2.], &[vec![1.]], &[vec![0.5]])?;
    assert!(updated.estimate.mean[0] > 1.);
    Ok(())
}
```

## Acknowledgments

We thank `smelly` and `nalgebra` for their state-estimation algorithms.
