# Changelog

## 0.2.1

- Adapt numerical matrix inputs directly to faer, removing intermediate shared-row allocations while preserving interoperability and shared result types.
- Borrow faer views, fixed arrays, and contiguous ndarray storage for compatible operations; retain owned workspaces for scaling and destructive algorithms. Preserve negative strides in solve refinement.
- Delegate triangular solves to faer in-place kernels, reusing borrowed factor storage across Cholesky solves.
- Assemble packed bands in backend storage and borrow internal matrices between algorithms. Remove temporary Cholesky factor transposes, output submatrix copies, and singular-value vector clones.
- Add native-storage benchmarks and run all SciPy reference cases through reversed borrowed views that reject intermediate row materialization.

- Compute symmetric eigenvalues without eigenvectors in `eigvalsh`, `eigvalsh_tridiagonal`, and `eigvals_banded`. Preserve scaling, ascending order, finite-value validation, and existing work limits; results may differ from eigenvector-producing routines by roundoff.
- Scale owned backend matrices in place, removing a temporary matrix allocation from decompositions and related algorithms while retaining division for very small scales.
- Transpose directly from logical input entries into shared output rows, removing a temporary backend matrix while preserving strided inputs, empty axes, and IEEE values.
- Add size-sweep performance benchmarks and independent SciPy regression cases for extreme scales and repeated eigenvalues.

## 0.2.0

Changes since the repository tag `v0.1.0`.

- Require `sci-core 0.2` and update Sci dependency requirements to matching releases. Shared error and numerical types are incompatible with those from `sci-core 0.1`.
- Refine dense `solve` results with compensated product residuals and up to four updates that reuse the LU factors. Preserve explicit singular-system errors and finite-result checks.

## 0.1.0

- Initial release of dense linear algebra, factorizations, matrix functions, and input adapters.
