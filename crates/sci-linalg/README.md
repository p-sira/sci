# Sci-Linalg

Linear algebra for real and complex vectors and matrices.

Part of [Sci](https://github.com/p-sira/sci), a unified scientific computing library for Rust.

Operations on `f64` vectors and matrices through `VectorInput` and
`MatrixInput`. Inputs may use shared slices/rows, ndarray arrays, nalgebra
matrices, or faer matrices, including read-only and mutable views. Different
libraries may be mixed within a call. Entry ordering follows logical indices,
including transposed, column-major, and strided storage; inputs are never mutated.
Results use shared row vectors, matching `sci_differentiate::jacobian`,
and can be copied back through the input traits' conversion methods.

Faer is the private numerical backend for products, decompositions, and
solves. Matrix functions use bounded native assembly with faer kernels.
Input types select storage only.

## Features

The `faer` feature (enabled by default) exposes faer input adapters and
conversions. `nalgebra` and `ndarray` enable their respective adapters and
optional dependencies. All three can be enabled together. With default
features disabled, shared slices/rows still use faer for computations.
The umbrella `sci` crate forwards these as `linalg-faer`, `linalg-nalgebra`,
and `linalg-ndarray`; its defaults enable all three.
The original dot/product helpers propagate IEEE values; added numerical
algorithms validate finite inputs and report shared errors.

## API layout

Shared `Matrix`, `MatrixInput`, `VectorInput`, `Error`, and `Result` types are
available at the crate root. Numerical functions are organized by family:

| Namespace | Contents |
| --- | --- |
| `basic` | Products, transpose, inverses, determinants, norms, matrix properties |
| `decompositions` | Cholesky, LU, QR, SVD, Schur, and related factorizations |
| `solvers` | Linear systems, least squares, Lyapunov, Sylvester, and Riccati equations |
| `eigen` | Eigenvalues, eigenvectors, generalized eigenvalue problems |
| `matrix_functions` | Matrix exponential, logarithm, roots, powers, and trigonometric functions |
| `structured` | Toeplitz, Hankel, circulant, and other matrix constructors |
| `subspaces` | Orthonormal bases, null spaces, Procrustes alignment, and sketching |
| `interpolative` | Interpolative decomposition and rank estimation |

For example, use `sci_linalg::decompositions::svd` for SVD and
`sci_linalg::solvers::solve` for a linear system. This organization replaces
flat numerical function exports in Sci 0.3.

## Input conventions

Numerical calls adapt storage directly to faer without intermediate shared
rows. Faer matrices/views, fixed arrays, and contiguous C/F-order ndarray
storage can be borrowed; other layouts copy directly into backend storage.
Scaling and destructive algorithms retain owned working copies. Required real
vector arguments also accept library vectors; optional vectors retain slice
signatures. Typed BLAS/LAPACK function pointers retain shared slice/row
signatures. Complex inputs to `cdf2rdf` retain shared complex vectors/rows.
Basic products, transpose, and matrix collections support empty axes; dense
numerical algorithms require nonempty axes.
Shared output rows cannot retain a column count for zero-row matrices. Other conversions
preserve empty axes. Untyped empty matrix literals/collections may need a
type annotation; generic function items used as pointers need a concrete
signature. See `MatrixInput` and `VectorInput` for runnable examples.

Dense algorithm input axes are bounded at 256, constructors at 1,000,000
output entries. Input conversions and basic helpers bound each axis and the
entry count at 1,000,000. Faer uses bounded
algorithm-specific sweep limits. Tighter limits and signature adaptations
are documented per function.

Matrix products use `C[i,j] = Σ A[i,k] B[k,j]`. Solvers interpret vectors
in the same logical row and column order as their matrices; no implicit
transposition or conjugation is performed. Singular systems, invalid
decompositions, and nonfinite algorithm outputs are returned as `Error`.

## Examples

```rust
fn main() -> Result<(), sci_core::Error> {
    use sci_linalg as linalg;
    let jacobian = sci_differentiate::jacobian(
        |x| vec![x[0] * x[1], x[0] + x[1]], &[2.0, 3.0], 1e-5,
    )?;
    let direction = linalg::basic::matvec(&jacobian, &[1.0, 0.0])?;
    assert!((direction[0] - 3.0).abs() < 1e-8);
    Ok(())
}
```

## Acknowledgments

We thank `dyn-stack`, `faer`, `num-complex`, and the
optional `nalgebra` and `ndarray` interoperability libraries.
