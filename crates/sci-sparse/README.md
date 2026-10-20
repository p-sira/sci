# Sci-Sparse

Sparse matrix construction, arithmetic, and direct linear solves with `f64`
values and the shared `sci_core::SparseMatrix` CSR representation.

```rust
use sci_sparse::{from_coo, matvec, linalg::spsolve};

let a = from_coo(2, 2, &[(0, 0, 2.0), (1, 1, 4.0)])?;
assert_eq!(matvec(&a, &[3.0, 2.0])?, vec![6.0, 8.0]);
assert_eq!(spsolve(&a, &[6.0, 8.0])?, vec![3.0, 2.0]);
# Ok::<(), sci_sparse::Error>(())
```

Use `sci::sparse` through the `sparse` Cargo feature, enabled by default, or
use this crate directly. Only `sci-core` and the private faer backend are required.

COO, CSR, CSC, and dense constructors produce canonical CSR. Duplicate
coordinates are summed, columns are sorted, and explicit zeros are retained.
NaN, infinity, malformed storage, and nonfinite arithmetic results return errors.
Axes and stored entries are limited to 1,000,000; dense conversion to 1,000,000
cells; sparse multiplication to 20,000,000 scalar products. LU order is limited
to 1,024; inverse order to 1,000. Ill-conditioned solves may lose accuracy;
no condition estimation or iterative refinement is performed.

v0.1.0 covers matrix construction, block assembly, transposition, triangular
extraction, Kronecker operations, sparse arithmetic, LU and triangular solves,
inversion, matrix powers, norms, and bandwidth inspection. Graph algorithms,
iterative/eigenvalue solvers, random sampling, NPZ serialization, arbitrary-rank
arrays, and additional mutable storage formats are outside this release.
See the repository's `function_coverages/SPARSE_COVERAGE.md` for the full inventory audit.

Part of [Sci](https://github.com/p-sira/sci). Sparse LU uses
[faer](https://docs.rs/faer/0.24.4/).
