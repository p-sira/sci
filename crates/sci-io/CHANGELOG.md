# Changelog

## 0.2.0

### Breaking changes

- Move Matrix Market functions into `matrix_market` and MATLAB functions into `matlab`, for example `matrix_market::mmread` and `matlab::loadmat`. See the [migration guide](../../MIGRATING.md).
- Use shared types from `sci-core 0.3`; they are incompatible with `sci-core 0.2` types.

### Improvements

- Read dense numeric MATLAB data with fewer copies while preserving scalar types, endianness, compression, and shape ordering.
- Reduce allocations in Matrix Market writing, MATLAB encoding, and single-channel WAV normalization.
- Use Matrix Market I/O without pulling in sparse solvers or `sci-sparse`.
- Select facade codecs independently with `io-matrix-market`, `io-wav`, and `io-matlab`. With defaults disabled, the facade's `io` feature provides shared I/O types without codecs.

## 0.1.0

- Initial release.
- Add bounded Matrix Market reading, writing, and metadata inspection for real dense and sparse matrices.
- Add WAV reading and writing and a dense numeric MATLAB subset with explicit format and size limits.
- Provide independent `matrix-market`, `wav`, and `matlab` features, enabled by default; use shared `sci-core 0.2` matrix types.
