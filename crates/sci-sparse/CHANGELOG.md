# Changelog

## 0.2.0

### Breaking changes

- Move sparse functions into family modules, for example `identity` to `construct::identity` and `matvec` to `operations::matvec`. See the [migration guide](../../MIGRATING.md).
- Move graph algorithms to `sci-graph` or `sci::graph`. Remove `csgraph` and the `graph` feature from `sci-sparse`; neither default features nor `full` now provides graph algorithms. Both crates accept the shared `SparseMatrix` type.
- Remove `issparse`, `get_index_dtype`, `IndexDtype`, and `safely_cast_index_arrays`. Use concrete sparse types and checked `array::csr_indices_i32` or `array::csr_indices_i64` conversions, which return `CsrIndices<T>`.
- Replace `aslinearoperator(&matrix)` with `LinearOperator::try_from(&matrix)` for an owned operator, or pass a matrix reference directly to solvers that accept `LinearMap`.
- Replace aliases `factorized`, `bmat`, `diags_array`, `eye_array`, and `matrix_transpose` with `splu`, `block_array`, `diags`, `eye`, and `transpose` in their family modules.
- Use shared types from `sci-core 0.3`; they are incompatible with `sci-core 0.2` types.

### Features

- With `default-features = false`, use CSR/COO construction, arithmetic, and array operations with only `sci-core` as a dependency. Enable `solver` for sparse linear algebra, `random` for seeded generation, and `npz` for `io::load_npz` and `io::save_npz`.
- Defaults enable `full` (solvers and random generation) and `npz`. The facade enables solvers and sampling through `sparse`, and NPZ interchange through `sparse-npz`.

### Improvements

- Reduce allocations in sparse construction, structural filtering, narrow matrix products, and iterative solvers.
- Disable unused OS entropy and thread-local random-number-generator dependencies; seeded sparse generation remains available.

## 0.1.0

- Initial release.
- Add sparse construction, arithmetic, axis operations, index utilities, and bounded NPZ interchange using shared `sci-core 0.2` CSR and COO types.
- Add graph algorithms, direct and iterative linear solvers, spectral decompositions, and matrix functions.
- Require `sci-complex 0.2` and `sci-linalg 0.2` so registry builds share the same core error type.
