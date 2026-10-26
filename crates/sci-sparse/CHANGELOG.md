# Changelog

## 0.2.0

- Remove runtime type/dtype helpers and redundant aliases. Add csr_indices_i32 and csr_indices_i64 returning CsrIndices<T>; replace aslinearoperator with LinearOperator::try_from(&matrix).
- Update workspace dependencies for the new minor releases.
- Disable OS entropy and thread-local RNG features; retain seeded sparse generation.
- Reuse iterative solver workspaces and swap TFQMR buffers instead of cloning them; borrow accepted Yen paths while generating candidates.
- The default `npz` feature enables `load_npz` and `save_npz`. Use `default-features = false` to retain sparse numerical operations without ZIP and compression dependencies. The Sci facade selects NPZ with `sparse-npz`.

## 0.1.0

- Initial release, added after the repository tag `v0.1.0`.
- Add sparse construction, arithmetic, axis operations, index utilities, and bounded NPZ interchange using shared `sci-core 0.2` CSR and COO types.
- Add graph algorithms, direct and iterative linear solvers, spectral decompositions, and matrix-function operations through documented Rust adaptations of 83 modern SciPy function names.
- Require `sci-complex 0.2` and `sci-linalg 0.2` so registry builds share the same core error type.
