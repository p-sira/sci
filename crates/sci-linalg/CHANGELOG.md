# Changelog

## 0.3.0

Use this release with `sci-core 0.3` and matching Sci crate releases when exchanging shared types.

### Breaking changes

- Move numerical functions into family modules, for example `solve` to `solvers::solve` and `svd` to `decompositions::svd`. See the [migration guide](../../MIGRATING.md).
- Replace `ishermitian` with `basic::issymmetric(&a, atol, rtol)` for real matrices, and `solve_lyapunov` with `solvers::solve_continuous_lyapunov`.
- Remove `find_best_blas_type`, `get_blas_funcs`, `get_lapack_funcs`, and their metadata/handle types. Call numerical functions directly; see the migration guide for differences in finite-value checking.

### Improvements

- Speed up symmetric tridiagonal eigenpairs and symmetric or banded eigenvalue-only calculations, with less temporary storage. Results retain ascending order and can differ from eigenvector-producing routines by roundoff.
- Reduce copying for compatible faer, ndarray, fixed-array, and strided inputs, including reversed views. Improve triangular solves, LU reuse, pseudoinverses, decompositions, and transpose operations.
- Reduce dependencies: dense linear algebra no longer pulls in `sci-special`.
- Include the bundled SciPy-derived license notice in the published package.

## 0.2.0

- Upgrade to shared `sci-core 0.2` types. Update cooperating Sci crates together; `sci-core 0.1` types are incompatible.
- Improve dense `solve` accuracy through iterative refinement while retaining singular-system errors and finite-result checks.

## 0.1.0

- Initial release of dense linear algebra, factorizations, matrix functions, and input adapters.
