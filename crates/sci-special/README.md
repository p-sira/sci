# Sci-Special

Special mathematical functions.

Elliptic integrals, 104 scalar and polynomial APIs, and stable reductions accept
`T: sci_core::Float`. Other sequence and wave APIs currently use `f64`.
Specialized kernels generally evaluate internally in binary64 and convert to
the requested type.

Part of [Sci](https://github.com/p-sira/sci), a unified scientific computing library for Rust.

Gamma, Bessel, elliptic, probability, polynomial, and combinatorial functions
are organized into public mathematical-family modules. For example,
`gamma::gamma`, `bessel::jv`, `orthogonal::eval_legendre`, and
`elliptic::ellipf` live in their respective modules. There are no root-level
numerical functions. Specialized algorithms use private pure Rust kernels and
ellip backends; elementary formulas use the standard library.
Use `value.is_finite()` to check scalar results for NaN and infinity.
Complex gamma (including log-gamma, reciprocal gamma and digamma),
error/Bessel/Hankel/Airy/Fresnel APIs are available under
`complex` using shared generic complex values and errors.
There is no broadcasting, implicit complex-input overloading, exact integer
mode, or derivative-axis output for array functions. Polynomial constructors
return descending coefficients. Read each function's documented real domain,
output ordering, and bounded order/size limits.
Scalar functions return documented IEEE values; fallible integrals and
sequences return the shared `sci_special::Result`.

Elliptic integrals use Ellip's notation: `ellipf(phi, m)` computes `F(phi, m)`,
with radians and elliptic parameter `m = k²`; incomplete third-kind arguments
are `(phi, n, m)`. Supported principal values are documented per function.
SciPy is an independent reference; complete SciPy parity is not required.

## Examples

```rust
fn main() -> Result<(), sci_core::Error> {
    use sci_special as special;
    assert_eq!(special::gamma::gamma(5.0), 24.0);
    let k = special::elliptic::ellipk(0.5_f64)?;
    assert!((k - 1.8540746773013719).abs() < 1e-14);
    Ok(())
}
```

## References

- [NIST DLMF, gamma functions](https://dlmf.nist.gov/5).
- [NIST DLMF, elliptic integrals](https://dlmf.nist.gov/19).

## Acknowledgments

We thank `ellip`, `nalgebra`, `num-complex`, `complex-bessel`, and `lambert_w`
for their specialized numerical algorithms. Third-party license notices are
retained in `LICENSES`.
