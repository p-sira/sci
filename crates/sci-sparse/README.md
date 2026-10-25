# Sci-Sparse

The default `npz` feature enables `load_npz` and `save_npz`. Use `default-features = false` to retain sparse numerical operations without ZIP and compression dependencies. The Sci facade selects NPZ with `sparse-npz`.

Real sparse arrays, graph algorithms, and linear algebra using shared Sci records
and private numerical backends. v0.1.0 covers all 83 modern SciPy sparse function
names through documented Rust adaptations, excluding deprecated compatibility
namespaces, legacy matrix predicates, `rand`, and `use_solver`.

```rust
use sci_sparse::{from_coo, matvec, linalg::spsolve};
let a = from_coo(2, 2, &[(0, 0, 2.0), (1, 1, 4.0)])?;
assert_eq!(matvec(&a, &[3.0, 2.0])?, [6.0, 8.0]);
assert_eq!(spsolve(&a, &[6.0, 8.0])?, [3.0, 2.0]);
# Ok::<(), sci_sparse::Error>(())
```

Use `sci::sparse` through the default-enabled `sparse` feature, or depend on this
crate directly. `SparseMatrix` holds canonical CSR; `SparseArray` holds rank-1..32
COO. Construction canonicalizes duplicates and retains explicit zeros. Graphs
interpret stored zeros as edges. APIs validate public storage and bound work.

The API includes construction/arithmetic, axis operations, seeded random sampling,
NPZ persistence, all 25 graph functions, operators, direct and iterative solves,
incomplete LU, partial spectra, SVD, matrix functions, norms and bandwidths.
Spectral routines use dense Sci/faer fallbacks with axes <=256; graph and iterative
axes are also bounded at 256. Primary function coverage does not imply Python
signature, unrestricted-domain or large-scale algorithm performance parity.
See [coverage and conventions](../../function_coverages/SPARSE_COVERAGE.md) for
supported options, omitted Python controls, accuracy limitations and work bounds.

Private backends include faer, petgraph, lapjv, rand and ZIP compression. QMR and
TFQMR adaptations retain SciPy's BSD notice in `SCIPY_LICENSE.txt`.
