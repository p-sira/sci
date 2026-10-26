# Changelog

## 0.2.0

- Update shared Sci dependency requirements to the new minor versions.
- Replace matrw with native numeric decoding through the existing bounded MATLAB
  parser. Preserve scalar types, special values, both endian orders, compression,
  shape ordering, and validation limits while removing the normalized-file copy.

- The Sci facade exposes each format through `io-matrix-market`, `io-wav`, and `io-matlab`. With facade defaults disabled, `io` alone provides shared I/O types without codecs. Matrix Market avoids sparse NPZ dependencies. MATLAB and NPZ share the Rust/miniz compression backend.

## 0.1.0

- Initial release, added after the repository tag `v0.1.0`.
- Add bounded Matrix Market reading, writing, and metadata inspection for real dense and sparse matrices.
- Add WAV reading and writing and a dense numeric MATLAB subset with explicit format and size limits.
- Provide independent `matrix-market`, `wav`, and `matlab` features, enabled by default; use shared `sci-core 0.2` matrix types.
