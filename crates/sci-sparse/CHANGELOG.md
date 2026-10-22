# Changelog

## 0.1.0

- Initial release, added after the repository tag `v0.1.0`.
- Add sparse construction, arithmetic, axis operations, index utilities, and bounded NPZ interchange using shared `sci-core 0.2` CSR and COO types.
- Add graph algorithms, direct and iterative linear solvers, spectral decompositions, and matrix-function operations through documented Rust adaptations of 83 modern SciPy function names.
- Require `sci-complex 0.2` and `sci-linalg 0.2` so registry builds share the same core error type.
