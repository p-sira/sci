# Sparse v0.1.0 coverage and numerical conventions

`sci-sparse` is exposed as `sci::sparse` through the default-enabled `sparse`
feature. It uses shared `sci_core::SparseMatrix`, `Matrix`, `Error`, and `Result`;
faer 0.24.4 provides private sparse LU. No backend types appear in the API.

The [inventory](scipy_function_list/scipy.sparse.md) contains 93 primary
functions across construction (36), graph algorithms (25), and linear algebra
(32), plus 55 compatibility-qualified names. There are no Cython-only sparse
functions. v0.1.0 supports 16 construction names and 9 linear algebra names
through documented Rust adaptations. It also supplies 13 functional helpers
for shared storage construction, validation, conversion, and arithmetic.
Compatibility names reuse primary implementations; they are not extra algorithms.
The table below gives every inventory function a disposition.

## Scope and adaptations

- One real, two-dimensional CSR type replaces SciPy's matrix/array variants.
  `eye_array`, `diags_array`, `matrix_transpose`, `bmat`, and `factorized` are
  aliases of shared implementations. No dynamic format or dtype arguments.
- `from_coo` sums duplicate coordinates in original input order after stable
  sorting. `from_csr` and `from_csc` accept unsorted duplicates and canonicalize.
  Explicit zeros, including negative zero, are retained; `from_dense` omits
  zeros. `find` returns row-major triples and excludes zeros. Arithmetic may
  retain zeros; matrix multiplication drops exact cancellation zeros.
- `diags` requires explicit shape, distinct offsets, and exact unpadded diagonal
  lengths. There is no scalar broadcasting or truncation. An empty diagonal
  list produces a shaped zero matrix, extending SciPy's currently supported domain.
- `block_array`/`bmat` accept a rectangular grid of references. Absent blocks
  require explicitly shaped zero matrices. Stacks and block diagonals require
  a nonempty input list. `kronsum(A,B) = I_n ⊗ A + B ⊗ I_m`.
- `spsolve` and `SparseLu::solve` accept vector right sides. `factorized` returns
  a `SparseLu` record with a `solve` method instead of a callable. Pivoting and
  fill-reducing ordering come from faer rather than SuperLU. No configurable
  permutations, incomplete factors, condition estimate, or iterative refinement.
- `spsolve_triangular` checks numerical triangularity, allowing off-triangle
  explicit zeros. Unit-diagonal mode ignores stored diagonal values.
- Norms support Frobenius, 1, −1, infinity, and negative infinity over whole
  matrices. Spectral norm, axis options, and `keepdims` are unsupported.
  Frobenius uses stable `hypot` accumulation; SciPy's sparse squaring formula
  overflows or underflows for extreme inputs, so those references use SciPy's
  dense BLAS-backed vector norm. Bandwidth and triangularity ignore stored zeros.
- All inputs and computed results must be finite. Singular systems return
  errors. Sparse storage can represent empty axes; shared dense row vectors
  cannot represent `(0,n)` for positive `n`, so `to_dense` rejects that shape.
- Every axis and input/output stored count is capped at 1,000,000. Dense
  conversion is capped at 1,000,000 cells. Each sparse product is capped at
  20,000,000 scalar products, checked before arithmetic. Kronecker products
  cap stored pairs at 1,000,000. Block assembly caps total scanned input rows
  at 20,000,000, including repeated references. Matrix powers permit 0..=1024 with at most
  20 sparse products. LU permits orders 1..=1024 to bound worst-case fill-in;
  inverse permits 1..=1000 to bound potentially dense output.

Graph algorithms (`csgraph`), iterative/eigenvalue/SVD solvers, matrix functions
such as sparse `expm`, operator interfaces, random sampling, NPZ persistence,
arbitrary-rank arrays, format predicates and index-dtype utilities, padded
`spdiags`, and additional mutable formats are outside v0.1.0. Existing dense
matrix-function and eigenvalue APIs remain in `sci::linalg`. These omissions
are not counted as implemented. Warning/exception classes and mutable Python
format class methods are not reproduced; shared Rust records and errors replace
the supported construction/operation paths.

## Reproduction and verification

`tests/scipy/sparse_cases.json` contains input cases only. Run
`uv run tests/scipy/generate_sparse.py` with SciPy 1.18.1 and NumPy 2.5.4 to
regenerate separate committed input and expected CSVs. `tests/sparse.rs` tests
every numerical export through `sci::sparse`, including aliases, output shapes,
ordering, signed zero, invalid storage/domains, empty axes, overflow, and work
limits. Reference comparisons use `3e-12` absolute plus relative tolerance for
small well-conditioned problems, with relative-only checks for tiny nonzero
values. Index and structural results are exact. Additional properties check
transpose involution, inverse/solve residuals, power recurrence, and shared dense
linear algebra interoperability. Python is not needed by ordinary Rust tests.

`python tests/scipy/audit_sparse.py` checks all inventory dispositions against
Rust exports and reference fixtures. Run the workspace formatting, tests, clippy,
rustdoc, and `tests/check_features.py` checks described in `docs/development.md`.

<!-- inventory -->

| Namespace | Inventory name | v0.1.0 disposition | Sci API |
| --- | --- | --- | --- |
| `scipy.sparse` | `block_array` | Supported adaptation | `sparse::block_array` |
| `scipy.sparse` | `block_diag` | Supported adaptation | `sparse::block_diag` |
| `scipy.sparse` | `bmat` | Supported adaptation | `sparse::bmat` |
| `scipy.sparse` | `diags` | Supported adaptation | `sparse::diags` |
| `scipy.sparse` | `diags_array` | Supported adaptation | `sparse::diags_array` |
| `scipy.sparse` | `expand_dims` | Outside v0.1.0 | — |
| `scipy.sparse` | `eye` | Supported adaptation | `sparse::eye` |
| `scipy.sparse` | `eye_array` | Supported adaptation | `sparse::eye_array` |
| `scipy.sparse` | `find` | Supported adaptation | `sparse::find` |
| `scipy.sparse` | `get_index_dtype` | Outside v0.1.0 | — |
| `scipy.sparse` | `hstack` | Supported adaptation | `sparse::hstack` |
| `scipy.sparse` | `identity` | Supported adaptation | `sparse::identity` |
| `scipy.sparse` | `issparse` | Outside v0.1.0 | — |
| `scipy.sparse` | `isspmatrix` | Outside v0.1.0 | — |
| `scipy.sparse` | `isspmatrix_bsr` | Outside v0.1.0 | — |
| `scipy.sparse` | `isspmatrix_coo` | Outside v0.1.0 | — |
| `scipy.sparse` | `isspmatrix_csc` | Outside v0.1.0 | — |
| `scipy.sparse` | `isspmatrix_csr` | Outside v0.1.0 | — |
| `scipy.sparse` | `isspmatrix_dia` | Outside v0.1.0 | — |
| `scipy.sparse` | `isspmatrix_dok` | Outside v0.1.0 | — |
| `scipy.sparse` | `isspmatrix_lil` | Outside v0.1.0 | — |
| `scipy.sparse` | `kron` | Supported adaptation | `sparse::kron` |
| `scipy.sparse` | `kronsum` | Supported adaptation | `sparse::kronsum` |
| `scipy.sparse` | `load_npz` | Outside v0.1.0 | — |
| `scipy.sparse` | `matrix_transpose` | Supported adaptation | `sparse::matrix_transpose` |
| `scipy.sparse` | `permute_dims` | Outside v0.1.0 | — |
| `scipy.sparse` | `rand` | Outside v0.1.0 | — |
| `scipy.sparse` | `random` | Outside v0.1.0 | — |
| `scipy.sparse` | `random_array` | Outside v0.1.0 | — |
| `scipy.sparse` | `safely_cast_index_arrays` | Outside v0.1.0 | — |
| `scipy.sparse` | `save_npz` | Outside v0.1.0 | — |
| `scipy.sparse` | `spdiags` | Outside v0.1.0 | — |
| `scipy.sparse` | `swapaxes` | Outside v0.1.0 | — |
| `scipy.sparse` | `tril` | Supported adaptation | `sparse::tril` |
| `scipy.sparse` | `triu` | Supported adaptation | `sparse::triu` |
| `scipy.sparse` | `vstack` | Supported adaptation | `sparse::vstack` |
| `scipy.sparse.csgraph` | `bellman_ford` | Outside v0.1.0 | — |
| `scipy.sparse.csgraph` | `breadth_first_order` | Outside v0.1.0 | — |
| `scipy.sparse.csgraph` | `breadth_first_tree` | Outside v0.1.0 | — |
| `scipy.sparse.csgraph` | `connected_components` | Outside v0.1.0 | — |
| `scipy.sparse.csgraph` | `construct_dist_matrix` | Outside v0.1.0 | — |
| `scipy.sparse.csgraph` | `csgraph_from_dense` | Outside v0.1.0 | — |
| `scipy.sparse.csgraph` | `csgraph_from_masked` | Outside v0.1.0 | — |
| `scipy.sparse.csgraph` | `csgraph_masked_from_dense` | Outside v0.1.0 | — |
| `scipy.sparse.csgraph` | `csgraph_to_dense` | Outside v0.1.0 | — |
| `scipy.sparse.csgraph` | `csgraph_to_masked` | Outside v0.1.0 | — |
| `scipy.sparse.csgraph` | `depth_first_order` | Outside v0.1.0 | — |
| `scipy.sparse.csgraph` | `depth_first_tree` | Outside v0.1.0 | — |
| `scipy.sparse.csgraph` | `dijkstra` | Outside v0.1.0 | — |
| `scipy.sparse.csgraph` | `floyd_warshall` | Outside v0.1.0 | — |
| `scipy.sparse.csgraph` | `johnson` | Outside v0.1.0 | — |
| `scipy.sparse.csgraph` | `laplacian` | Outside v0.1.0 | — |
| `scipy.sparse.csgraph` | `maximum_bipartite_matching` | Outside v0.1.0 | — |
| `scipy.sparse.csgraph` | `maximum_flow` | Outside v0.1.0 | — |
| `scipy.sparse.csgraph` | `min_weight_full_bipartite_matching` | Outside v0.1.0 | — |
| `scipy.sparse.csgraph` | `minimum_spanning_tree` | Outside v0.1.0 | — |
| `scipy.sparse.csgraph` | `reconstruct_path` | Outside v0.1.0 | — |
| `scipy.sparse.csgraph` | `reverse_cuthill_mckee` | Outside v0.1.0 | — |
| `scipy.sparse.csgraph` | `shortest_path` | Outside v0.1.0 | — |
| `scipy.sparse.csgraph` | `structural_rank` | Outside v0.1.0 | — |
| `scipy.sparse.csgraph` | `yen` | Outside v0.1.0 | — |
| `scipy.sparse.linalg` | `aslinearoperator` | Outside v0.1.0 | — |
| `scipy.sparse.linalg` | `bicg` | Outside v0.1.0 | — |
| `scipy.sparse.linalg` | `bicgstab` | Outside v0.1.0 | — |
| `scipy.sparse.linalg` | `cg` | Outside v0.1.0 | — |
| `scipy.sparse.linalg` | `cgs` | Outside v0.1.0 | — |
| `scipy.sparse.linalg` | `eigs` | Outside v0.1.0 | — |
| `scipy.sparse.linalg` | `eigsh` | Outside v0.1.0 | — |
| `scipy.sparse.linalg` | `expm` | Outside v0.1.0 | — |
| `scipy.sparse.linalg` | `expm_multiply` | Outside v0.1.0 | — |
| `scipy.sparse.linalg` | `factorized` | Supported adaptation | `sparse::linalg::factorized` |
| `scipy.sparse.linalg` | `funm_multiply_krylov` | Outside v0.1.0 | — |
| `scipy.sparse.linalg` | `gcrotmk` | Outside v0.1.0 | — |
| `scipy.sparse.linalg` | `gmres` | Outside v0.1.0 | — |
| `scipy.sparse.linalg` | `inv` | Supported adaptation | `sparse::linalg::inv` |
| `scipy.sparse.linalg` | `is_sptriangular` | Supported adaptation | `sparse::linalg::is_sptriangular` |
| `scipy.sparse.linalg` | `lgmres` | Outside v0.1.0 | — |
| `scipy.sparse.linalg` | `lobpcg` | Outside v0.1.0 | — |
| `scipy.sparse.linalg` | `lsmr` | Outside v0.1.0 | — |
| `scipy.sparse.linalg` | `lsqr` | Outside v0.1.0 | — |
| `scipy.sparse.linalg` | `matrix_power` | Supported adaptation | `sparse::linalg::matrix_power` |
| `scipy.sparse.linalg` | `minres` | Outside v0.1.0 | — |
| `scipy.sparse.linalg` | `norm` | Supported adaptation | `sparse::linalg::norm` |
| `scipy.sparse.linalg` | `onenormest` | Outside v0.1.0 | — |
| `scipy.sparse.linalg` | `qmr` | Outside v0.1.0 | — |
| `scipy.sparse.linalg` | `spbandwidth` | Supported adaptation | `sparse::linalg::spbandwidth` |
| `scipy.sparse.linalg` | `spilu` | Outside v0.1.0 | — |
| `scipy.sparse.linalg` | `splu` | Supported adaptation | `sparse::linalg::splu` |
| `scipy.sparse.linalg` | `spsolve` | Supported adaptation | `sparse::linalg::spsolve` |
| `scipy.sparse.linalg` | `spsolve_triangular` | Supported adaptation | `sparse::linalg::spsolve_triangular` |
| `scipy.sparse.linalg` | `svds` | Outside v0.1.0 | — |
| `scipy.sparse.linalg` | `tfqmr` | Outside v0.1.0 | — |
| `scipy.sparse.linalg` | `use_solver` | Outside v0.1.0 | — |
| `scipy.sparse.base` | `issparse` | Outside v0.1.0 | — |
| `scipy.sparse.base` | `isspmatrix` | Outside v0.1.0 | — |
| `scipy.sparse.bsr` | `isspmatrix_bsr` | Outside v0.1.0 | — |
| `scipy.sparse.construct` | `block_diag` | Compatibility alias; shared implementation | `sparse::block_diag` |
| `scipy.sparse.construct` | `bmat` | Compatibility alias; shared implementation | `sparse::bmat` |
| `scipy.sparse.construct` | `diags` | Compatibility alias; shared implementation | `sparse::diags` |
| `scipy.sparse.construct` | `eye` | Compatibility alias; shared implementation | `sparse::eye` |
| `scipy.sparse.construct` | `get_index_dtype` | Outside v0.1.0 | — |
| `scipy.sparse.construct` | `hstack` | Compatibility alias; shared implementation | `sparse::hstack` |
| `scipy.sparse.construct` | `identity` | Compatibility alias; shared implementation | `sparse::identity` |
| `scipy.sparse.construct` | `issparse` | Outside v0.1.0 | — |
| `scipy.sparse.construct` | `kron` | Compatibility alias; shared implementation | `sparse::kron` |
| `scipy.sparse.construct` | `kronsum` | Compatibility alias; shared implementation | `sparse::kronsum` |
| `scipy.sparse.construct` | `rand` | Outside v0.1.0 | — |
| `scipy.sparse.construct` | `random` | Outside v0.1.0 | — |
| `scipy.sparse.construct` | `spdiags` | Outside v0.1.0 | — |
| `scipy.sparse.construct` | `vstack` | Compatibility alias; shared implementation | `sparse::vstack` |
| `scipy.sparse.coo` | `isspmatrix_coo` | Outside v0.1.0 | — |
| `scipy.sparse.csc` | `isspmatrix_csc` | Outside v0.1.0 | — |
| `scipy.sparse.csr` | `isspmatrix_csr` | Outside v0.1.0 | — |
| `scipy.sparse.dia` | `isspmatrix_dia` | Outside v0.1.0 | — |
| `scipy.sparse.dok` | `isspmatrix_dok` | Outside v0.1.0 | — |
| `scipy.sparse.extract` | `find` | Compatibility alias; shared implementation | `sparse::find` |
| `scipy.sparse.extract` | `tril` | Compatibility alias; shared implementation | `sparse::tril` |
| `scipy.sparse.extract` | `triu` | Compatibility alias; shared implementation | `sparse::triu` |
| `scipy.sparse.lil` | `isspmatrix_lil` | Outside v0.1.0 | — |
| `scipy.sparse.linalg.dsolve` | `factorized` | Compatibility alias; shared implementation | `sparse::linalg::factorized` |
| `scipy.sparse.linalg.dsolve` | `spilu` | Outside v0.1.0 | — |
| `scipy.sparse.linalg.dsolve` | `splu` | Compatibility alias; shared implementation | `sparse::linalg::splu` |
| `scipy.sparse.linalg.dsolve` | `spsolve` | Compatibility alias; shared implementation | `sparse::linalg::spsolve` |
| `scipy.sparse.linalg.dsolve` | `spsolve_triangular` | Compatibility alias; shared implementation | `sparse::linalg::spsolve_triangular` |
| `scipy.sparse.linalg.dsolve` | `test` | Outside v0.1.0 | — |
| `scipy.sparse.linalg.dsolve` | `use_solver` | Outside v0.1.0 | — |
| `scipy.sparse.linalg.eigen` | `eigs` | Outside v0.1.0 | — |
| `scipy.sparse.linalg.eigen` | `eigsh` | Outside v0.1.0 | — |
| `scipy.sparse.linalg.eigen` | `lobpcg` | Outside v0.1.0 | — |
| `scipy.sparse.linalg.eigen` | `svds` | Outside v0.1.0 | — |
| `scipy.sparse.linalg.eigen` | `test` | Outside v0.1.0 | — |
| `scipy.sparse.linalg.interface` | `aslinearoperator` | Outside v0.1.0 | — |
| `scipy.sparse.linalg.isolve` | `bicg` | Outside v0.1.0 | — |
| `scipy.sparse.linalg.isolve` | `bicgstab` | Outside v0.1.0 | — |
| `scipy.sparse.linalg.isolve` | `cg` | Outside v0.1.0 | — |
| `scipy.sparse.linalg.isolve` | `cgs` | Outside v0.1.0 | — |
| `scipy.sparse.linalg.isolve` | `gcrotmk` | Outside v0.1.0 | — |
| `scipy.sparse.linalg.isolve` | `gmres` | Outside v0.1.0 | — |
| `scipy.sparse.linalg.isolve` | `lgmres` | Outside v0.1.0 | — |
| `scipy.sparse.linalg.isolve` | `lsmr` | Outside v0.1.0 | — |
| `scipy.sparse.linalg.isolve` | `lsqr` | Outside v0.1.0 | — |
| `scipy.sparse.linalg.isolve` | `minres` | Outside v0.1.0 | — |
| `scipy.sparse.linalg.isolve` | `qmr` | Outside v0.1.0 | — |
| `scipy.sparse.linalg.isolve` | `test` | Outside v0.1.0 | — |
| `scipy.sparse.linalg.isolve` | `tfqmr` | Outside v0.1.0 | — |
| `scipy.sparse.linalg.matfuncs` | `expm` | Outside v0.1.0 | — |
| `scipy.sparse.linalg.matfuncs` | `inv` | Compatibility alias; shared implementation | `sparse::linalg::inv` |
| `scipy.sparse.linalg.matfuncs` | `spsolve` | Compatibility alias; shared implementation | `sparse::linalg::spsolve` |
