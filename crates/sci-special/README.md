# Sci-Special

Special mathematical functions.

Elliptic integrals and logsumexp/softmax reductions accept `T: sci_core::Float`; other special functions use `f64`.

Part of [Sci](https://github.com/p-sira/sci), a unified scientific computing library for Rust.

Gamma, Bessel, elliptic, probability, polynomial, and combinatorial functions
with floating-point scalars and shared Sci types. Specialized algorithms use private
special, xsf, and ellip backends; elementary formulas use the standard library.
Use `sci_core::finite` to reject nonfinite scalar results explicitly.
There is no broadcasting, general complex-input overloading, exact integer
mode, or derivative-axis output for array functions. Polynomial constructors
return descending coefficients. Read each function's documented real domain,
output ordering, and bounded order/size limits.
Scalar functions return documented IEEE values; fallible integrals and
sequences return the shared `sci_special::Result`.

## Examples

```rust
fn main() -> Result<(), sci_core::Error> {
    use sci_special as special;
    assert_eq!(special::gamma(5.0), 24.0);
    let k = special::elliptic::ellipk(0.5_f64)?;
    assert!((k - 1.8540746773013719).abs() < 1e-14);
    Ok(())
}
```

## References

- [NIST DLMF, gamma functions](https://dlmf.nist.gov/5).
- [NIST DLMF, elliptic integrals](https://dlmf.nist.gov/19).

## Acknowledgments

We thank `ellip`, `nalgebra`, `num-complex`, `special`,
and `xsf` for their specialized numerical algorithms.
