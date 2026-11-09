# Sci-Optimize

Numerical optimization and minimization.

Minimizers live under `sci_optimize::minimize` and numerical derivatives under
`sci_optimize::derivatives`. Nonlinear least-squares solvers live under
`sci_optimize::least_squares`. `Options`, `Status`, and `Minimum` remain at
the crate root.

Part of [Sci](https://github.com/p-sira/sci), a unified scientific computing library for Rust.

Unconstrained minimization using Basin internally. Objective and gradient
callbacks take the same `&[f64]` used by differentiation and linear algebra.
Callers never implement backend traits or construct solver state.

## Algorithms

`minimize` uses Nelder–Mead, a derivative-free simplex method.
`minimize_with_gradient` uses fixed-step gradient descent. Its learning
rate must suit the objective; convergence is not guaranteed for arbitrary
functions. Limits return a result with a status, while invalid evaluations
return the shared `Error`.
`least_squares::dogbox` minimizes the half-sum of squared residuals subject
to per-coordinate bounds. It uses a rectangular trust region, SVD for the
Gauss–Newton step, and a bound-aware numerical Jacobian by default. Use
`dogbox_with_jacobian` for an analytical Jacobian or their `try_` variants
for fallible callbacks. Infinite lower/upper bounds are supported.
Minimizers require finite objective and gradient evaluations. Dogbox requires
finite initial residuals and Jacobian samples; nonfinite trial residuals shrink
the trust region. A successful status
means the method met its configured stopping rule, not that a global
minimum was proved; inspect the returned point, value, and status together.

## Examples

```rust
fn main() -> Result<(), sci_core::Error> {
    let result = sci_optimize::minimize::minimize(
        |x| (x[0] - 2.0).powi(2),
        &[0.0], sci_optimize::Options::default(),
    )?;
    assert!((result.point[0] - 2.0).abs() < 1e-6);

    let fit = sci_optimize::least_squares::dogbox(
        |x| vec![x[0] - 3.0, x[0] - 3.0],
        &[0.0], &[f64::NEG_INFINITY], &[2.0],
        sci_optimize::least_squares::DogboxOptions::default(),
    )?;
    assert!((fit.point[0] - 2.0).abs() < 1e-6);
    Ok(())
}
```

## References

- Nelder and Mead (1965), A simplex method for function minimization,
  [doi:10.1093/comjnl/7.4.308](https://doi.org/10.1093/comjnl/7.4.308).
- [SciPy `least_squares(method="dogbox")`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html), based on Voglis and Lagaris (2004).

## Acknowledgments

We thank `basin` for its private optimization backends.
