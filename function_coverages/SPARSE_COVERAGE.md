# Sparse coverage and numerical conventions

The Rust API omits runtime type/dtype utilities and redundant aliases; see the disposition table below. With facade defaults disabled, NPZ requires `sparse-npz`; direct `sci-sparse` users select the default-enabled `npz` feature. Other sparse operations compile without ZIP or compression. NPZ uses the same flate2 Rust/miniz backend as MATLAB; numerical behavior and reference fixtures are unchanged.

`sci-sparse` is exposed as `sci::sparse` through the default-enabled `sparse`
feature. The original modern inventory has **83 names**. Retained function-name coverage
is **74/83**: construction/utilities **20/27**, graph algorithms **25/25**, and
linear algebra **29/31**. Removed names and redesigned operations are recorded below.
This is function-name coverage through the Rust adaptations below; it is not
Python signature, performance, storage-class, or unrestricted-domain parity.
All supported names have independent SciPy references and public Sci tests.
There are no placeholders or untested names counted as supported.

The [inventory](scipy_function_list/scipy.sparse.md) has 93 primary functions
and 55 compatibility-qualified functions. Excluded legacy APIs are `rand`
(use `random_array` or `random`), `use_solver` (private backend selection), and
the eight `isspmatrix*` predicates for old Python matrix classes. Deprecated
compatibility namespaces, Python warning/exception classes, mutable storage
classes and their methods are not reproduced. Primary aliases (`bmat`,
`factorized`, `eye_array`, `diags_array`, `matrix_transpose`) are removed; their
canonical numerical operations remain available.
All 148 inventory entries have explicit dispositions in the table below.

## Shared storage and construction

- Shared `sci_core::SparseMatrix` is finite real, two-dimensional canonical CSR.
  Shared `SparseArray` is arbitrary-rank canonical COO with rank 1..32. Public
  fields are validated before numerical use. Backend types remain private.
- `from_coo`, `from_csr` and `from_csc` canonicalize duplicates using stable
  summation in original duplicate order. Explicit zeros, including signed zero,
  are retained except by `from_dense`, `eliminate_zeros`, and cancellation in
  sparse matrix multiplication. `find` omits zero values.
- Matrix/array variants share these records; there are no dynamic dtype/format
  arguments. `as_array` converts CSR to COO. `expand_dims`, `permute_dims`, and
  `swapaxes` accept COO, including negative axes. Repeated expansion composes
  multiple new axes. Matrix arithmetic and `transpose` retain CSR inputs.
- `diags` requires explicit shape, distinct offsets and exact unpadded lengths;
  `spdiags` instead aligns padded rows by output column. Blocks require explicit
  shaped zero matrices in place of Python None, and nonempty input grids/lists.
- `random_array` and `random` sample distinct coordinates and uniform `[0,1)`
  data using private seeded StdRng. Counts use ties-to-even density rounding;
  streams differ from NumPy. No sampler callback; callers can replace sampled
  values with validated data. Runtime `issparse` checks are omitted; use concrete Rust types.
- `csr_indices_i32` and `csr_indices_i64` return concrete checked integer vectors
  in named `CsrIndices<T>` fields. `get_index_dtype`, `IndexDtype`, and
  `safely_cast_index_arrays` are removed. Rust usize remains the shared record's
  storage type. `LinearOperator::try_from(&matrix)` replaces `aslinearoperator`;
  solvers also accept borrowed CSR matrices directly through `LinearMap`.
- NPZ uses private ZIP compression and bounded pickle-free NPY parsing. All five
  SciPy matrix formats (CSR, CSC, COO, DIA, BSR), real f32/f64, integer and bool
  data can be read as CSR. Writing emits f64 CSR arrays. Signed integer data above
  2^53 may round on conversion. Rank other than two, complex/object dtypes,
  Fortran-order data and old Unicode format tags are unsupported. Duplicate ZIP
  member names are rejected before array decoding. Paths and file
  extensions are managed explicitly by callers through seekable reader/writers.

## Graph conventions

- A stored entry, including zero, represents an edge. Undirected paths use the
  smaller available opposite weight. Dense/masked conversion makes absent-edge
  rules explicit; nonfinite masked data are allowed, but unmasked numerical
  edges must be finite. Dense null fill values may be NaN or infinity.
- Paths return source-major distance and predecessor rows. `Option<usize>`
  replaces SciPy's negative predecessor/matching sentinels; unreachable distances
  are positive infinity. Source predecessors are None. Ties may choose different
  valid paths, matchings or forests; invariants and objectives are tested.
- Dijkstra supports source lists, inclusive cutoff, unweighted and minimum-only
  searches. Automatic paths select Dijkstra for nonnegative weights and
  Bellman–Ford otherwise; cutoff and minimum-only controls are rejected when
  automatic selection chooses Bellman–Ford. Floyd–Warshall/Johnson detect any negative cycle;
  Bellman–Ford detects cycles reachable from requested sources. Dijkstra and Yen
  reject negative weights. Matrix-sized source lists are always explicit output
  rows, even for one source; predecessor computation is always enabled.
- Component labels are normalized by the smallest vertex. Traversals visit
  neighbors in ascending order. Reverse Cuthill–McKee tie ordering may differ
  from SciPy. `symmetric_mode=true` trusts the supplied structural symmetry.
- Laplacians ignore self edges, use incoming degrees by default, optionally
  use outgoing degrees and/or A+Aᵀ symmetrization, and return degree factors.
  Normalized Laplacians require nonnegative degrees; isolated factors are one.
- MST and weighted full matching follow SciPy's zero-is-absent convention.
  Structural matching/rank include stored zeros. Full matching returns sorted
  row indices and errors if no full matching exists. Maximum flow accepts
  nonnegative integral f64 capacities and returns antisymmetric net flows.
- Private petgraph kernels supply paths, SCC, MST, matching, flow and Yen spur
  searches. Private lapjv supplies weighted assignment. Native bounded assembly
  handles conversion, traversal, Laplacians, ordering and path reconstruction.

## Linear algebra conventions

- Faer supplies sparse LU with private factors; `factorized` returns a reusable
  `SparseLu` record. Vector right sides are supported. No SuperLU permutations,
  panel tuning, equilibration, condition estimates or iterative refinement.
  Singular matrices return errors. Triangular solves support an ignored stored
  diagonal in unit mode and ignore explicitly stored off-triangle zeros.
- ILUT is a threshold incomplete factorization without pivoting, using relative
  U dropping, multiplier dropping and bounded row fill. It can fail on a zero
  pivot even for nonsingular A. Its inverse implements `LinearMap`; factors
  and approximations can differ from SciPy's SuperLU incomplete factorization.
- `LinearMap` supports CSR and callback-backed `LinearOperator`, with forward and
  transpose products. Callback shapes/finiteness are checked. User callbacks
  must bound their own work and supply truly linear/adjoint actions.
- CG, BiCGSTAB and LSMR use private faer iterations. Other named iterations use
  their respective bounded recurrences or residual-minimizing Krylov spaces;
  real QMR/TFQMR recurrences are adapted from SciPy under its included BSD license.
  GMRES uses restarting; LGMRES retains corrections; GCROT retains orthogonal
  correction/image pairs and truncates oldest pairs. Recycling is local to a call.
  Projected solves update a Givens QR factorization, using Sci/faer SVD for
  deficient systems. Arnoldi breakdown is relative to the preconditioned operator
  action. Projected residuals trigger true-residual checks, also performed at
  each cycle end; uniform operator/right-side scaling preserves the relative
  stopping criterion within floating-point arithmetic limits.
- `SolverOptions` controls tolerances, initial guess, iterations, restart,
  recycling, inverse preconditioning and MINRES shift. QMR, TFQMR and the
  full-reorthogonalized MINRES adaptation require identity preconditioning.
  MINRES checks exact symmetry, solves A−shift I and retains a full symmetric
  Arnoldi/Lanczos space, restarting only at the bounded basis limit. Callbacks,
  external recycling state, mutable flexible preconditioners and logging are omitted.
- Iterative results preserve exhausted approximations and include convergence,
  iteration count and true residual. Breakdown returns an error. Convergence
  uses `||b-Ax|| <= max(atol,rtol*||b||)`. LSQR/LSMR use zero initial guesses,
  optional Tikhonov damping and report primal/regularized normal residuals.
  Their convergence uses either true residual or true normal residual at the
  same threshold; SciPy's stop codes, conlim and variance estimates are omitted.
- CG/BiCGSTAB materialize operator/preconditioner columns for bounded faer
  workspace. LSQR/LSMR materialize operator columns and append damping rows.
  The remaining iterations operate through forward/transpose actions.
- `eigs`, `eigsh`, `svds`, `lobpcg`, `expm`, and `expm_multiply` reuse bounded
  Sci/faer dense decompositions and matrix functions. These are dense fallbacks,
  not large-scale ARPACK/LOBPCG performance equivalents. `lobpcg` takes the full
  symmetric-decomposition fallback; the initial block determines k only.
  Generalized/shift-invert spectra, constraints, preconditioned eigenproblems,
  histories and backend iteration controls are omitted. Spectral selection and
  eigenvector column ordering are explicit; partial singular values are ascending.
- `expm_multiply` accepts explicit times and vector right sides. Arnoldi
  `funm_multiply_krylov` instead accepts a scalar complex function, evaluates
  projected functions via existing Sci/faer Schur kernels, uses two-pass
  orthogonalization, and has a basis limit rather than restart controls.
  Small successive-action changes and small Arnoldi coupling are required
  together; these are heuristic, not rigorous error bounds for
  arbitrary functions. Coupled repeated projected eigenvalues require derivatives
  and return an error. Complex vectors use the shared Sci complex record.
- Norms include Frobenius, 1, −1, ±infinity and spectral 2-norm. `norm_axis`
  handles vector p-norm reductions, including zero-count and ±infinity orders.
  Finite negative p follows SciPy sparse-power semantics: unstored zeros do not
  contribute, stored zeros yield zero, and an empty stored reduction yields +∞.
  `onenormest` computes the exact sparse 1-norm, satisfying the estimator's
  lower-bound contract without stochastic tuning or witness outputs.
- All numerical input data must be finite. Nonfinite arithmetic returns errors,
  except documented unreachable/null graph output and negative-p norm infinity.
  Empty CSR axes are supported; shared dense rows cannot retain shape `(0,n)`
  for positive n, so `to_dense` rejects that conversion.

## Work limits and accuracy

CSR axes/stored entries are capped at 1,000,000. COO rank is 1..32, with the same
axis/nnz cap. Dense conversion is capped at 1,000,000 cells; sparse multiplication
at 20,000,000 scalar products; block assembly at 20,000,000 scanned input rows;
Kronecker products at 1,000,000 stored pairs. Matrix powers allow 0..1024 with at
most 20 products. Sparse LU order is 1..1024; inverse order is 1..1000.

Graph/matching axes are at most 256. Bellman–Ford bounds sources×vertices×edges
at 20,000,000; Yen bounds k×vertices×edges likewise, with k<=64. Path weights
satisfy `|w| <= f64::MAX/(4n+1)` to bound backend arithmetic. Maximum-flow capacity
sum is at most 2^53−1. NPZ allows at most 16 members and 64 MiB total uncompressed
payload. Random sampling volume is at most 20,000,000. Spectral/ILUT and iterative
axes are at most 256, iterations at most 1024, restart bases at most 255 and
recycled directions at most 32. Exponential actions accept at most 64 times.
Arnoldi actions accept a basis cap of at most 256 and inherit projected matrix
function work limits. All these restrictions are checked explicitly.

F64 roundoff, cancellation, underflow, ill-conditioning and nearly defective
spectra limit accuracy. Residuals can be small despite large forward error.
Eigenvector signs, repeated-eigenspace bases and tied graph optima are nonunique.
Frobenius uses hypot accumulation to avoid unnecessary overflow/underflow.

## Reproduction and verification

Input-only cases live in `tests/scipy/sparse_cases.json` and
`sparse_parity_cases.json`. Additional NPZ encoding and rejection cases live in
`sparse_npz_validation_cases.json`; regenerate their deterministic archives and
SciPy references with `generate_sparse_npz_validation.py`.
Regenerate the separate checked-in input/expected
CSVs and NPZ fixtures with `generate_sparse.py` and `generate_sparse_parity.py`,
using pinned SciPy 1.18.1 and NumPy 2.5.4. Ordinary Rust tests read these fixtures
without Python. `tests/sparse.rs` and `tests/sparse_parity.rs` exercise every
numerical function through the public Sci facade, including aliases, shapes,
ordering, signed zeros, invalid inputs, special values, bounded work, graph
invariants, operator actions and eigen/solve residuals.

Small well-conditioned scalar/matrix references use 3e-12 absolute+relative
tolerance; spectral/matrix functions use 3e-11; converged iterative solutions
use 3e-9. Structural results are exact. Tiny norms use relative-only checks.
The extreme Frobenius references use SciPy's BLAS-backed dense norm because
sparse squaring over/underflows. For the two-dimensional Krylov action case,
SciPy 1.18.1 fails with an empty projected slice even with the correct matrix
function callback; the independent reference uses `scipy.linalg.funm` instead.

Run `tests/scipy/check_sparse_npz.py` after the Rust parity tests to verify a
Rust-written archive with SciPy. `tests/scipy/audit_sparse.py` checks every modern
export and fixture and accounts for all inventory entries. Run workspace fmt,
tests, clippy, rustdoc, and `tests/check_features.py` as in `docs/development.md`.

<!-- inventory -->

| Namespace | Inventory name | Current disposition | Sci API |
| --- | --- | --- | --- |
| `scipy.sparse` | `block_array` | Supported adaptation | `sparse::block_array` |
| `scipy.sparse` | `block_diag` | Supported adaptation | `sparse::block_diag` |
| `scipy.sparse` | `bmat` | Removed redundant alias | `block_array` |
| `scipy.sparse` | `diags` | Supported adaptation | `sparse::diags` |
| `scipy.sparse` | `diags_array` | Removed redundant alias | `diags` |
| `scipy.sparse` | `expand_dims` | Supported adaptation | `sparse::expand_dims` |
| `scipy.sparse` | `eye` | Supported adaptation | `sparse::eye` |
| `scipy.sparse` | `eye_array` | Removed redundant alias | `eye` |
| `scipy.sparse` | `find` | Supported adaptation | `sparse::find` |
| `scipy.sparse` | `get_index_dtype` | Removed or redesigned for Rust | — |
| `scipy.sparse` | `hstack` | Supported adaptation | `sparse::hstack` |
| `scipy.sparse` | `identity` | Supported adaptation | `sparse::identity` |
| `scipy.sparse` | `issparse` | Removed or redesigned for Rust | — |
| `scipy.sparse` | `isspmatrix` | Excluded legacy API | — |
| `scipy.sparse` | `isspmatrix_bsr` | Excluded legacy API | — |
| `scipy.sparse` | `isspmatrix_coo` | Excluded legacy API | — |
| `scipy.sparse` | `isspmatrix_csc` | Excluded legacy API | — |
| `scipy.sparse` | `isspmatrix_csr` | Excluded legacy API | — |
| `scipy.sparse` | `isspmatrix_dia` | Excluded legacy API | — |
| `scipy.sparse` | `isspmatrix_dok` | Excluded legacy API | — |
| `scipy.sparse` | `isspmatrix_lil` | Excluded legacy API | — |
| `scipy.sparse` | `kron` | Supported adaptation | `sparse::kron` |
| `scipy.sparse` | `kronsum` | Supported adaptation | `sparse::kronsum` |
| `scipy.sparse` | `load_npz` | Supported adaptation | `sparse::load_npz` |
| `scipy.sparse` | `matrix_transpose` | Removed redundant alias | `transpose` |
| `scipy.sparse` | `permute_dims` | Supported adaptation | `sparse::permute_dims` |
| `scipy.sparse` | `rand` | Excluded legacy API | — |
| `scipy.sparse` | `random` | Supported adaptation | `sparse::random` |
| `scipy.sparse` | `random_array` | Supported adaptation | `sparse::random_array` |
| `scipy.sparse` | `safely_cast_index_arrays` | Removed or redesigned for Rust | `sparse::csr_indices_i32`, `sparse::csr_indices_i64` |
| `scipy.sparse` | `save_npz` | Supported adaptation | `sparse::save_npz` |
| `scipy.sparse` | `spdiags` | Supported adaptation | `sparse::spdiags` |
| `scipy.sparse` | `swapaxes` | Supported adaptation | `sparse::swapaxes` |
| `scipy.sparse` | `tril` | Supported adaptation | `sparse::tril` |
| `scipy.sparse` | `triu` | Supported adaptation | `sparse::triu` |
| `scipy.sparse` | `vstack` | Supported adaptation | `sparse::vstack` |
| `scipy.sparse.csgraph` | `bellman_ford` | Supported adaptation | `sparse::csgraph::bellman_ford` |
| `scipy.sparse.csgraph` | `breadth_first_order` | Supported adaptation | `sparse::csgraph::breadth_first_order` |
| `scipy.sparse.csgraph` | `breadth_first_tree` | Supported adaptation | `sparse::csgraph::breadth_first_tree` |
| `scipy.sparse.csgraph` | `connected_components` | Supported adaptation | `sparse::csgraph::connected_components` |
| `scipy.sparse.csgraph` | `construct_dist_matrix` | Supported adaptation | `sparse::csgraph::construct_dist_matrix` |
| `scipy.sparse.csgraph` | `csgraph_from_dense` | Supported adaptation | `sparse::csgraph::csgraph_from_dense` |
| `scipy.sparse.csgraph` | `csgraph_from_masked` | Supported adaptation | `sparse::csgraph::csgraph_from_masked` |
| `scipy.sparse.csgraph` | `csgraph_masked_from_dense` | Supported adaptation | `sparse::csgraph::csgraph_masked_from_dense` |
| `scipy.sparse.csgraph` | `csgraph_to_dense` | Supported adaptation | `sparse::csgraph::csgraph_to_dense` |
| `scipy.sparse.csgraph` | `csgraph_to_masked` | Supported adaptation | `sparse::csgraph::csgraph_to_masked` |
| `scipy.sparse.csgraph` | `depth_first_order` | Supported adaptation | `sparse::csgraph::depth_first_order` |
| `scipy.sparse.csgraph` | `depth_first_tree` | Supported adaptation | `sparse::csgraph::depth_first_tree` |
| `scipy.sparse.csgraph` | `dijkstra` | Supported adaptation | `sparse::csgraph::dijkstra` |
| `scipy.sparse.csgraph` | `floyd_warshall` | Supported adaptation | `sparse::csgraph::floyd_warshall` |
| `scipy.sparse.csgraph` | `johnson` | Supported adaptation | `sparse::csgraph::johnson` |
| `scipy.sparse.csgraph` | `laplacian` | Supported adaptation | `sparse::csgraph::laplacian` |
| `scipy.sparse.csgraph` | `maximum_bipartite_matching` | Supported adaptation | `sparse::csgraph::maximum_bipartite_matching` |
| `scipy.sparse.csgraph` | `maximum_flow` | Supported adaptation | `sparse::csgraph::maximum_flow` |
| `scipy.sparse.csgraph` | `min_weight_full_bipartite_matching` | Supported adaptation | `sparse::csgraph::min_weight_full_bipartite_matching` |
| `scipy.sparse.csgraph` | `minimum_spanning_tree` | Supported adaptation | `sparse::csgraph::minimum_spanning_tree` |
| `scipy.sparse.csgraph` | `reconstruct_path` | Supported adaptation | `sparse::csgraph::reconstruct_path` |
| `scipy.sparse.csgraph` | `reverse_cuthill_mckee` | Supported adaptation | `sparse::csgraph::reverse_cuthill_mckee` |
| `scipy.sparse.csgraph` | `shortest_path` | Supported adaptation | `sparse::csgraph::shortest_path` |
| `scipy.sparse.csgraph` | `structural_rank` | Supported adaptation | `sparse::csgraph::structural_rank` |
| `scipy.sparse.csgraph` | `yen` | Supported adaptation | `sparse::csgraph::yen` |
| `scipy.sparse.linalg` | `aslinearoperator` | Removed or redesigned for Rust | `LinearOperator::try_from` |
| `scipy.sparse.linalg` | `bicg` | Supported adaptation | `sparse::linalg::bicg` |
| `scipy.sparse.linalg` | `bicgstab` | Supported adaptation | `sparse::linalg::bicgstab` |
| `scipy.sparse.linalg` | `cg` | Supported adaptation | `sparse::linalg::cg` |
| `scipy.sparse.linalg` | `cgs` | Supported adaptation | `sparse::linalg::cgs` |
| `scipy.sparse.linalg` | `eigs` | Supported adaptation | `sparse::linalg::eigs` |
| `scipy.sparse.linalg` | `eigsh` | Supported adaptation | `sparse::linalg::eigsh` |
| `scipy.sparse.linalg` | `expm` | Supported adaptation | `sparse::linalg::expm` |
| `scipy.sparse.linalg` | `expm_multiply` | Supported adaptation | `sparse::linalg::expm_multiply` |
| `scipy.sparse.linalg` | `factorized` | Removed redundant alias | `splu` |
| `scipy.sparse.linalg` | `funm_multiply_krylov` | Supported adaptation | `sparse::linalg::funm_multiply_krylov` |
| `scipy.sparse.linalg` | `gcrotmk` | Supported adaptation | `sparse::linalg::gcrotmk` |
| `scipy.sparse.linalg` | `gmres` | Supported adaptation | `sparse::linalg::gmres` |
| `scipy.sparse.linalg` | `inv` | Supported adaptation | `sparse::linalg::inv` |
| `scipy.sparse.linalg` | `is_sptriangular` | Supported adaptation | `sparse::linalg::is_sptriangular` |
| `scipy.sparse.linalg` | `lgmres` | Supported adaptation | `sparse::linalg::lgmres` |
| `scipy.sparse.linalg` | `lobpcg` | Supported adaptation | `sparse::linalg::lobpcg` |
| `scipy.sparse.linalg` | `lsmr` | Supported adaptation | `sparse::linalg::lsmr` |
| `scipy.sparse.linalg` | `lsqr` | Supported adaptation | `sparse::linalg::lsqr` |
| `scipy.sparse.linalg` | `matrix_power` | Supported adaptation | `sparse::linalg::matrix_power` |
| `scipy.sparse.linalg` | `minres` | Supported adaptation | `sparse::linalg::minres` |
| `scipy.sparse.linalg` | `norm` | Supported adaptation | `sparse::linalg::norm` |
| `scipy.sparse.linalg` | `onenormest` | Supported adaptation | `sparse::linalg::onenormest` |
| `scipy.sparse.linalg` | `qmr` | Supported adaptation | `sparse::linalg::qmr` |
| `scipy.sparse.linalg` | `spbandwidth` | Supported adaptation | `sparse::linalg::spbandwidth` |
| `scipy.sparse.linalg` | `spilu` | Supported adaptation | `sparse::linalg::spilu` |
| `scipy.sparse.linalg` | `splu` | Supported adaptation | `sparse::linalg::splu` |
| `scipy.sparse.linalg` | `spsolve` | Supported adaptation | `sparse::linalg::spsolve` |
| `scipy.sparse.linalg` | `spsolve_triangular` | Supported adaptation | `sparse::linalg::spsolve_triangular` |
| `scipy.sparse.linalg` | `svds` | Supported adaptation | `sparse::linalg::svds` |
| `scipy.sparse.linalg` | `tfqmr` | Supported adaptation | `sparse::linalg::tfqmr` |
| `scipy.sparse.linalg` | `use_solver` | Excluded legacy API | — |
| `scipy.sparse.base` | `issparse` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.base` | `isspmatrix` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.bsr` | `isspmatrix_bsr` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.construct` | `block_diag` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.construct` | `bmat` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.construct` | `diags` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.construct` | `eye` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.construct` | `get_index_dtype` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.construct` | `hstack` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.construct` | `identity` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.construct` | `issparse` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.construct` | `kron` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.construct` | `kronsum` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.construct` | `rand` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.construct` | `random` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.construct` | `spdiags` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.construct` | `vstack` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.coo` | `isspmatrix_coo` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.csc` | `isspmatrix_csc` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.csr` | `isspmatrix_csr` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.dia` | `isspmatrix_dia` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.dok` | `isspmatrix_dok` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.extract` | `find` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.extract` | `tril` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.extract` | `triu` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.lil` | `isspmatrix_lil` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.linalg.dsolve` | `factorized` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.linalg.dsolve` | `spilu` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.linalg.dsolve` | `splu` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.linalg.dsolve` | `spsolve` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.linalg.dsolve` | `spsolve_triangular` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.linalg.dsolve` | `test` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.linalg.dsolve` | `use_solver` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.linalg.eigen` | `eigs` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.linalg.eigen` | `eigsh` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.linalg.eigen` | `lobpcg` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.linalg.eigen` | `svds` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.linalg.eigen` | `test` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.linalg.interface` | `aslinearoperator` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.linalg.isolve` | `bicg` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.linalg.isolve` | `bicgstab` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.linalg.isolve` | `cg` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.linalg.isolve` | `cgs` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.linalg.isolve` | `gcrotmk` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.linalg.isolve` | `gmres` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.linalg.isolve` | `lgmres` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.linalg.isolve` | `lsmr` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.linalg.isolve` | `lsqr` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.linalg.isolve` | `minres` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.linalg.isolve` | `qmr` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.linalg.isolve` | `test` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.linalg.isolve` | `tfqmr` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.linalg.matfuncs` | `expm` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.linalg.matfuncs` | `inv` | Excluded legacy compatibility namespace | — |
| `scipy.sparse.linalg.matfuncs` | `spsolve` | Excluded legacy compatibility namespace | — |
