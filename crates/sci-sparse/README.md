# Sci-Sparse

By default, `full` enables sparse solvers and seeded random sampling; `npz`
enables `load_npz` and `save_npz`. To use only canonical sparse storage,
construction, arithmetic, and array transforms, disable defaults:

```toml
sci-sparse = { version = "0.2", default-features = false }
```

Enable independent backends with `features = ["solver"]`,
`features = ["random"]`, or `features = ["npz"]`. `full` selects the first two.
Graph algorithms are exclusively provided by `sci-graph` or `sci::graph`, and
have no dependency on `sci-sparse`. The Sci facade selects `full` with `sparse`
and NPZ with `sparse-npz`.

Real sparse arrays and linear algebra using shared Sci records
and private numerical backends. Numerical operations use Rust signatures and
explicit inventory dispositions. Runtime type/dtype utilities, redundant aliases,
deprecated compatibility namespaces, legacy matrix predicates, `rand`, and
`use_solver` are omitted or replaced.

```rust
use sci_sparse::{construct::from_coo, operations::matvec, linalg::spsolve};
let a = from_coo(2, 2, &[(0, 0, 2.0), (1, 1, 4.0)])?;
assert_eq!(matvec(&a, &[3.0, 2.0])?, [6.0, 8.0]);
assert_eq!(spsolve(&a, &[6.0, 8.0])?, [3.0, 2.0]);
# Ok::<(), sci_sparse::Error>(())
```

Use `sci::sparse` through the default-enabled `sparse` feature, or depend on this
crate directly. `SparseMatrix` holds canonical CSR; `SparseArray` holds rank-1..32
COO. Construction canonicalizes duplicates and retains explicit zeros.
APIs validate public storage and bound work.

The API includes construction/arithmetic, axis operations, seeded random sampling,
NPZ persistence, operators, direct and iterative solves, incomplete LU, partial
spectra, SVD, matrix functions, norms and bandwidths.
Spectral routines use dense Sci/faer fallbacks with axes <=256; iterative
axes are also bounded at 256. Primary function coverage does not imply Python
signature, unrestricted-domain or large-scale algorithm performance parity.
See [coverage and conventions](../../function_coverages/SPARSE_COVERAGE.md) for
supported options, omitted Python controls, accuracy limitations and work bounds.

Private backends include faer, rand and ZIP compression. QMR and TFQMR
adaptations retain SciPy's BSD notice in
`SCIPY_LICENSE.txt`.

Public numerical functions are grouped under `construct`, `operations`,
`array`, `linalg`, and `io` (when their corresponding backends are
enabled). `SparseMatrix`, `SparseArray`,
`Matrix`, `Error`, and `Result` remain available at the crate root.
