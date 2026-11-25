# Sci-Differentiate

Numerical differentiation using finite differences.

Part of [Sci](https://github.com/p-sira/sci), a unified scientific computing library for Rust.

Second-order central finite differences on `T: sci_core::Float` values (`f32` or `f64`). Gradient vectors
and Jacobian/Hessian matrices share the slice and row-major conventions of
`sci_optimize` and `sci_linalg`.

## Accuracy

Symmetric central differences have truncation error `O(h²)` for smooth functions.
Denominators use actual rounded coordinate spans; diagonal Hessians use
unequal three-point divided differences. Rounded asymmetry adds midpoint
displacement error for first/mixed derivatives and an error proportional to
the difference between axial increments for diagonal second derivatives.
Roundoff grows as `h` shrinks. Choose an absolute step suited to the
coordinate scale; a very small step may be unrepresentable. Callbacks should
be smooth near the requested point. Each input coordinate is perturbed
independently; no adaptive step selection is performed. Hessian roundoff
grows approximately as `1/h²`, so second derivatives generally need a larger
step than first derivatives (roughly `EPSILON.powf(0.25)` at unit scale).

## Evaluation and work limits

`derivative`, `gradient`, `jacobian`, and `hessian` evaluate one point
with a caller-selected absolute step and return values through `Result`.
Adaptive orders, one-sided steps, batching, and broadcasting are unsupported.
Capture extra arguments in deterministic closures. Each call permits at most
1,000,000 sampled points and output entries; callback execution time and
allocation remain the caller's responsibility.

## Definitions

For a scalar function `f`, `derivative(f, x, h)` uses
`(f(x + h) - f(x - h)) / (x + h - (x - h))`. Gradients, Jacobians, and
Hessians apply the same centered stencil coordinate by coordinate, using
the actual rounded coordinates in each denominator. A Jacobian has one row
per output component and one column per input coordinate.

## Examples

```rust
fn main() -> Result<(), sci_core::Error> {
    let slope = sci_differentiate::derivative(f64::sin, 0.0, 1e-5)?;
    assert!((slope - 1.0).abs() < 1e-9);
    Ok(())
}
```
