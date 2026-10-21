# Data and numerical conventions

Real scalars use `f64`; vectors accept `&[f64]` and return `Vec<f64>`.
Matrices use `Vec<Vec<f64>>` with one vector per row. Jacobians use output
components as rows and input coordinates as columns. Complex values use
`complex::Complex { re: f64, im: f64 }`. Fallible functions share
`sci::Error` and `sci::Result`.

These conventions let differentiation feed optimization and matrix operations,
and let Fourier spectra feed complex arithmetic and descriptive statistics.

Linear algebra accepts real matrices and vectors from **faer, ndarray, and
nalgebra** through `linalg::MatrixInput` and `linalg::VectorInput` when the
corresponding Cargo features are enabled. Owned storage, fixed-size nalgebra
matrices, and strided or transposed views can be mixed within a call.

Most matrix and vector results use Sci's shared `Vec<Vec<f64>>` and `Vec<f64>`
types. Adapter methods can copy compatible inputs and results to supported
faer, ndarray, and nalgebra types. Numerical algorithms copy inputs into their
private backend storage; shared-row and contiguous-vector inputs may be borrowed
only during adapter conversion:

```rust
use sci::linalg::{self, MatrixInput};

let a = ndarray::arr2(&[[2.0, 1.0], [0.0, 3.0]]);
let b = nalgebra::DMatrix::identity(2, 2);
let product = linalg::matmul(&a.view(), &b)?;
let matrix = product.to_faer()?;
assert_eq!(linalg::solve(&matrix, &[5.0, 6.0])?, vec![1.5, 2.0]);
# Ok::<(), sci::Error>(())
```

These adapters select input storage; algorithms still choose their internal
backends. Optional vector parameters and typed BLAS/LAPACK function handles
retain slice/row signatures. Complex inputs retain existing shared types.
Conversions preserve IEEE values and library empty axes, with a limit of
1,000,000 entries and 1,000,000 per axis. Shared rows cannot represent a `0 × n`
shape with `n > 0`; conversion to shared rows rejects that shape. Numerical
algorithms retain their documented tighter limits.

Sparse matrices use `sci::SparseMatrix`, an owned CSR type shared with
`sci-core` and `sci-sparse`. Row pointers and column indices are zero-based;
columns are sorted and unique per row. Explicit zeros are retained. Sparse
operations validate public storage before use and reject nonfinite inputs and
results. `sparse::from_coo`, `from_csr`, and `from_csc` canonicalize storage;
`from_dense` and `to_dense` connect sparse data with shared row matrices and
existing linear algebra adapters. `sci::SparseArray` supplies rank-1..32 COO
coordinates for sparse axis operations. Graphs distinguish unstored non-edges
from explicitly stored zero-weight edges; unreachable graph distances and
caller-selected null fills may be nonfinite. See [sparse coverage](../function_coverages/SPARSE_COVERAGE.md)
for supported operations and limits.

## Numerical conventions

The forward FFT is unnormalized; its inverse divides by the sample count.
Elliptic integrals use the parameter `m = k²` and amplitudes in radians.
Distribution parameters and units are specified per function. Variance uses
`ddof = 0` for population variance and `ddof = 1` for unbiased sample variance.

[Back to README](../README.md).
