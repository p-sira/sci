# Changelog

## 0.3.0

- Remove the hidden `finite` helper; use `Float::is_finite()` for scalar checks. Rename `finite_slice` to `is_finite_slice` for shared slice validation.

- Re-export the shared `Float` bound; make finite checks and matrix, spline, and sparse storage generic with `f64` defaults.

- Update shared Sci dependency requirements to the new minor versions.

## 0.2.0

Changes since the repository tag `v0.1.0`.

- Add shared `SparseMatrix` storage in canonical CSR format and `SparseArray` storage in COO format for ranks 1 through 32.
- Release shared types and errors as `sci-core 0.2`; downstream crates must use the same core version to exchange these types.

## 0.1.0

- Initial release of shared errors, matrix storage, convolution modes, and spline representations.
