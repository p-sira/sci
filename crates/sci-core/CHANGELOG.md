# Changelog

## 0.3.0

### Breaking changes

- Remove `finite`; use `value.is_finite()` for scalar checks. Rename `finite_slice` to `is_finite_slice`; it continues to return `Result<()>`.

### Added

- Re-export the shared `Float` trait bound.
- Support generic matrix, spline, and sparse storage with `f64` defaults, allowing supported numerical APIs to share `f32` data.

### Improvements

- Speed up COO-to-CSR construction while preserving duplicate-entry and signed-zero conventions.

## 0.2.0

- Add shared `SparseMatrix` storage in canonical CSR format and `SparseArray` storage in COO format for ranks 1 through 32.
- Release shared types and errors as `sci-core 0.2`; downstream crates must use the same core version to exchange these types.

## 0.1.0

- Initial release of shared errors, matrix storage, convolution modes, and spline representations.
