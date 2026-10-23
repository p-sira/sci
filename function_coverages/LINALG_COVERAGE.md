# `linalg` coverage and SciPy adaptations

The `linalg` namespace exports **96 of 96** top-level
`scipy.linalg` inventory function names. The nine
`scipy.linalg.interpolative` names are also exported in their own namespace.
Every name has at least two independent reference cases, for **587 cases**
total across 105 qualified APIs. These APIs have documented functional
signatures and limitations; name coverage does not establish full SciPy
behavioral parity. There are no placeholder implementations.

The source of truth is [SCIPY_FUNCTIONS.md](SCIPY_FUNCTIONS.md) and its
[linear algebra inventory](scipy-func-list/scipy.linalg.md). Deprecated
`basic`, `decomp*`, `matfuncs`, `misc`, and `special_matrices` paths repeat the
same top-level names; they do not require duplicate implementations.
The BLAS/LAPACK/Cython binding namespaces remain absent. Top-level dispatchers
return typed function handles for a documented subset of real operations;
backend types remain private. Python classes and methods are not counted
as free functions.

## Backend and adapter features

Faer is the required private numerical backend. The `sci-linalg` features
`faer` (default), `nalgebra`, and `ndarray` independently expose the existing
input traits and owned conversions for each library. Nalgebra and ndarray
are optional dependencies used only by adapters. Shared inputs and every
numerical API work with `--no-default-features`.
The umbrella `sci` crate forwards `linalg-faer`, `linalg-nalgebra`, and
`linalg-ndarray`; its defaults retain interoperability with all three libraries.
Sci-special and other workspace crates can still depend transitively on
nalgebra; this change consolidates sci-linalg's computational backend.

Faer 0.24's unblocked LU divides by exact zero pivots. Singular factorization
retries through faer column factorizations and trailing products, skipping
exact zero columns; this preserves finite singular LU factors. Solves reject
zero or nonfinite pivots. No fallback uses another linear algebra library.

## Matrix and vector interoperability

Real matrix parameters accept the shared `MatrixInput` trait, implemented for
row vectors and fixed arrays, ndarray arrays/views, nalgebra dynamic/fixed
matrices/views, and faer matrices/views. Required real vector parameters use
`VectorInput`, supporting slices, ndarray vectors/views, nalgebra column
vectors/views, and faer rows/columns/views. Matrix collections also accept these
types, including mixed libraries through `&dyn MatrixInput`. Existing aliases
reuse the same adapters. Faer is the private computational backend independently of
input storage; inputs are read-only and results retain shared Sci types.

The traits provide owned conversions to shared rows/vectors and each library.
Logical indexing preserves row/column order for strided and transposed views,
IEEE values, and signed zero. Library-to-library conversions retain empty axes.
Shared rows cannot represent `0 × n` for positive n: conversion to shared rows
returns a dimension error. Basic products, transpose, and collections inspect
logical shapes directly. Shared output rows retain this representation limit.
Input conversions and basic helpers bound each axis and entry count at
1,000,000; numerical algorithms retain their existing tighter bounds.

All **587 cases** run through shared, ndarray, nalgebra, and faer storage.
Additional independently generated SciPy BLAS fixtures test basic products,
transpose, and dot products. Rust tests exercise mixed libraries, strided and
mutable views, fixed storage, empty axes, IEEE conversion, cross-module results,
validation, and allocation bounds. Optional vector parameters, typed dispatch
function pointers, and complex conversion inputs retain existing slice/row
signatures. Untyped empty literals and stored generic function items may require
explicit type annotations. This adds storage interoperability without expanding
SciPy numerical coverage or introducing another algorithm backend.

The supplemental `linalg_coverage` target runs with only the `linalg` feature.
It checks strict rank-cutoff boundaries, complementary range/null projectors,
singular rectangular LU reconstruction and solve rejection, the inclusive
256-axis bound, nonfinite outputs and thresholds, matrix-function callback and
branch validation, isolated-column balancing, and QZ selection/reordering.
The reference suite includes singular zero-pivot LU, zero-matrix decompositions,
rank cutoffs, and triangular principal matrix functions with negative and
repeated eigenvalues. References are generated with NumPy 2.5.4 and SciPy 1.18.1.

## Implemented public names

| Group | Names |
| --- | --- |
| General and structured systems | `solve`, `inv`, `det`, `solve_triangular`, `solve_banded`, `solve_toeplitz`, `matmul_toeplitz`, `solve_circulant` |
| Cholesky | `cholesky`, `cho_factor`, `cho_solve`, `cholesky_banded`, `cho_solve_banded`, `solveh_banded` |
| Pivoted LU | `lu`, `lu_factor`, `lu_solve` |
| QR and RQ | `qr`, `rq`, `qr_multiply`, `qr_update`, `qr_insert`, `qr_delete` |
| Eigenproblems and reductions | `eigh`, `eigvalsh`, `eigh_tridiagonal`, `eigvalsh_tridiagonal`, `eig_banded`, `eigvals_banded`, `eigvals`, `eig`, `hessenberg`, `schur`, `rsf2csf`, `cdf2rdf`, `ldl`, `qz`, `ordqz`, `cossin`, `matrix_balance` |
| SVD and rank | `svd`, `svdvals`, `diagsvd`, `pinv`, `pinvh`, `lstsq`, `orth`, `null_space` |
| Subspaces and alignment | `polar`, `orthogonal_procrustes`, `subspace_angles` |
| Matrix functions | `expm`, `sinm`, `cosm`, `tanm`, `sinhm`, `coshm`, `tanhm`, `expm_frechet`, `expm_cond`, `sqrtm`, `logm`, `signm`, `funm`, `fractional_matrix_power` |
| Matrix equations | `solve_sylvester`, `solve_continuous_lyapunov`, `solve_lyapunov`, `solve_discrete_lyapunov`, `solve_continuous_are`, `solve_discrete_are` |
| Structured matrices | `toeplitz`, `hankel`, `circulant`, `hilbert`, `invhilbert`, `pascal`, `invpascal`, `hadamard`, `helmert`, `fiedler`, `companion`, `fiedler_companion`, `leslie`, `block_diag`, `convolution_matrix`, `dft`, `khatri_rao` |
| Sketching | `clarkson_woodruff_transform` |
| Typed dispatch | `find_best_blas_type`, `get_blas_funcs`, `get_lapack_funcs` |
| Norms and structure | `norm`, `bandwidth`, `issymmetric`, `ishermitian` |

The existing `dot`, `matvec`, `matmul`, and `transpose` helpers remain available
and are exercised by interoperability and reconstruction tests. They are not
names in the top-level SciPy inventory and do not contribute to the 96 count.

## Data and signature conventions

Real matrices use row-major `Vec<Vec<f64>>` (`linalg::Matrix`) and real vectors
use `f64` slices. `dft` and `eigvals` return the shared `complex::Complex`.
Complex matrix outputs use `ComplexMatrix`; complex inputs are accepted only
by `cdf2rdf`. General complex matrix inputs, NumPy broadcasting, batches,
dtype selection, mutation,
LAPACK driver selection, and worker controls are unsupported.

System solvers and least squares accept one vector RHS. QR, RQ, and SVD return
economy factors. Named owned result types document all factor shapes and output
ordering. LU uses `A=P L U`, matching SciPy. `lu_factor` requires a square
matrix and returns a packed factor with a **final row permutation**, replacing
SciPy's stepwise pivot array; `lu_solve` accepts that representation and supports
ordinary and transposed systems. `cho_factor` zeroes the unused triangle;
SciPy leaves it unspecified. Both factor types are backend-independent and
are revalidated when used, including permutation bijectivity.

Symmetric eigenproblems and Cholesky use the requested lower or upper triangle.
Unused finite triangle/padding entries are ignored. Generalized eigenproblems,
subsets, sorting of the original `schur` function, and its complex-output
option are unsupported. `rsf2csf` separately converts real Schur factors,
and `qz`/`ordqz` return complex generalized Schur factors. Eigenvalues
from symmetric problems are ascending; SVD values are descending; general
eigenvalues are in unspecified Schur order. Eigenvectors are columns.

`eigvalsh`, `eigvalsh_tridiagonal`, and `eigvals_banded` use the scaled
eigenvalues-only backend path. They do not form eigenvectors; results may
differ from vector-producing routines by roundoff. Banded and tridiagonal
inputs still expand to dense storage and retain the 256-axis work bound.

Pseudoinverses always return `(matrix, rank)` and retain values strictly above
`atol + rtol*largest_magnitude`. Optional pseudoinverse/range/null-space relative
cutoffs default to `max(m,n)*epsilon`; least squares defaults to `epsilon`,
matching SciPy's gelsd convention. Least-squares residuals have one entry only
when m>n and full column rank. `polar` uses a boolean side selector. Principal
angles are descending and measured in radians, with empty output for zero-rank
spaces. Orthogonal Procrustes permits reflections and does no centering/scaling.

`norm` supports eight explicit matrix orders via `MatrixNorm`. Vector p norms,
axis selection, and keepdims are absent; a one-row matrix with Frobenius order
provides a vector Euclidean norm. `PascalKind`, `DftScale`, and shared
`signal::Mode` replace SciPy strings. DFT uses the negative exponent, with
optional division by n or sqrt(n). Integer exact modes are omitted.

Constructors permit zero-sized matrices except Hadamard, Helmert, DFT, companion,
Leslie, and convolution matrices. An empty matrix cannot retain a column count.
An empty list of blocks returns 0×0, differing from SciPy's empty-list promotion.
`diagsvd` requires exactly min(m,n) diagonal entries. `convolution_matrix` same
mode has max(kernel length, input length) rows, following SciPy, while the
existing signal convolution helper uses the first input's length in same mode.

QR updates are rank one. Insertions accept matrices of consecutive rows/columns;
deletions must leave the selected axis nonempty. Updates, insertions, and deletions
reconstruct and refactor through the shared QR algorithm; they do not implement
SciPy's specialized incremental complexity, or its insertion rcond option.
Supplied Q is not tested for orthonormality: these functions factor the given
product and its requested modification.

## Validation and numerical bounds

Except for type metadata and the predicates described below, new numerical
APIs reject nonfinite inputs and outputs through the shared
`Error`/`Result` types. Singular solves and failed factorizations return errors;
singular LU can be stored but cannot be used to solve. No condition-number
warnings are emitted. Rank decisions and solutions near singularity can lose
accuracy. Existing basic dot/product helpers retain their IEEE propagation.

Dense input axes are limited to **256**. Faer stages use bounded
algorithm-specific iteration/sweep limits. Ordinary real and complex Schur
reductions use the public generalized stages on `(A,I)` and verify their
Hessenberg/quasi-triangular/triangular output structure. SVD, symmetric eigenproblems, QR, RQ, Hessenberg,
and Schur routines scale inputs before backend evaluation. Floating-point
rank, eigenvectors, and general eigenvalues remain sensitive to roundoff.
Wide null-space inputs are padded with zero rows to obtain the complete right
singular basis. Principal angles use residual sine values near zero and cosine
values near π/2 to avoid inverse-cosine loss of accuracy.

General/symmetric bands are expanded to dense matrices and reuse the shared
solvers and eigensystems. Bandwidth must be less than n; stored axes must also
fit the dense bound. General and circulant/Toeplitz solves use O(n³) LU;
Toeplitz multiplication is dense rather than FFT-based. These bounded adapters
provide the documented mathematical results without SciPy's structured work costs.

Sylvester equations use a private LU backend on a Kronecker system, requiring
**m*n<=256**. Continuous Lyapunov reuses that solver; discrete Lyapunov uses the
direct Kronecker equation with **n<=16**. Schur-based Bartels–Stewart and bilinear Lyapunov alternatives remain unsupported.
Riccati equations use separate ordered generalized Schur adapters, described below. Nearly resonant
matrix equations can be ill-conditioned. Continuous Lyapunov follows SciPy's
`A X+X Aᵀ=Q` sign; discrete Lyapunov solves `X-A X Aᵀ=Q`.

Matrix exponentials use a private [13/13] Padé evaluation through faer products
and LU solves, explicit input pre-scaling to one-norm <= 1/2, and at most 34
squarings. The formula follows [Higham (2005)](https://doi.org/10.1137/04061101X).
Input one-norm is at most **2^32**; pre-scaling bounds the Padé intermediates. Circular
matrix functions use exp(i A). Hyperbolic sine/cosine use independent even/odd
Taylor series at one-norm <=1/2 (12 updates, omitted term below 2e-34), then
at most 34 double-angle steps through private faer products. This preserves
small sine entries without subtracting exponentials. Hyperbolic tangent uses
`T(2A)=2T(A)(I+T(A)^2)⁻¹` with backend LU, avoiding exponential overflow for
large real eigenvalues (including scalar ±2^32). True or intermediate overflow
returns an error, and underflow can produce zero. Large or nonnormal arguments
have no uniform accuracy guarantee. Tangent functions require invertible
denominators and can lose accuracy near poles. Independent regression fixtures
from `generate_hyperbolic.py` cover 63 cases with tiny values, saturation, nearly overflowing
finite outputs, triangular and coupled matrices. Scalar and quadratic spectral
identities provide references where SciPy exponential differences lose accuracy;
SciPy `funm` supplies general tangent references when its direct exponential
solve is ill-conditioned.
Fréchet derivatives use the exact augmented block exponential and require
**n<=128**. Condition numbers assemble all n² Fréchet directions and require
**n<=8**, then use the derivative operator's spectral norm with Frobenius
input/output norms. An underflowed zero exponential norm returns an error.

Constructors allocate at most **1,000,000 entries**, with each axis also bounded
at 1,000,000. Pascal/inverse Pascal orders are at most **256** and inverse Hilbert
orders at most **128**. Large integer entries round above 2^53, and these matrices
can be severely ill-conditioned. Fiedler and coefficient-ratio overflow is an error.

Structure predicates are deliberate IEEE exceptions: exact symmetry ignores the
diagonal, equal infinities compare true, and off-diagonal NaN compares false.
Providing either tolerance activates allclose semantics including the diagonal.
`ishermitian` is the real-valued alias of `issymmetric`; complex Hermitian inputs
are unsupported. `bandwidth` tests exact nonzero finite entries.

SciPy-derived companion layouts and principal-angle identities retain their
BSD-3-Clause notice in [LICENSES/scipy-linalg.txt](LICENSES/scipy-linalg.txt).

## Verification

Input-only cases live in `tests/scipy/linalg_cases.json`. The generator pins
**SciPy 1.18.1 / NumPy 2.5.4**, uses their public APIs, and writes separate
checked-in input/expected CSVs. Normal Rust tests require no Python.

The **570** cases exercise all 105 qualified APIs, shapes, output ordering, triangle/band
layouts, normalization, rectangular and rank-deficient systems, explicit cutoffs,
zero-rank spaces, and near-zero subspace angles. Well-conditioned small systems
use **3e-11 absolute plus 3e-11 relative** tolerances for different dense
algorithms/reduction orders. No nonfinite numerical reference results are accepted.
SciPy's matrix Frobenius path can overflow on 1e200 values, so the generator uses
its stable vector BLAS norm on the flattened data for that order.

Nonunique factors are compared using reconstruction, projectors, and Gram
matrices. Rank-deficient polar orthogonal extensions are allowed to differ;
the unique positive factor, reconstruction, and orthonormality are verified.
Separate tests check exact labels/ranks and shapes, invalid domains, bounded
work, singularities, IEEE predicates, signed-zero copying, overflow, tiny/large
scaling, Moore–Penrose identities, matrix equation residuals, exponential and
trigonometric identities, and Fréchet directional derivatives. A central
difference derivative check uses an explicit 1e-8 bound for truncation and
cancellation. Null-space residuals allow the selected singular-value cutoff.
Cross-module tests use differentiation Jacobians, signal convolution, FFT output,
and the existing matrix product helpers.

```console
uv run tests/scipy/generate_linalg.py
python tests/scipy/audit_linalg.py
cargo fmt --all -- --check
cargo test
cargo clippy --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --no-deps
```

## Additional functional conventions and limitations

`eig` returns right eigenvectors and eigenvalues as shared complex values, for
real ordinary eigenproblems. Vectors are unit norm and stored in columns.
Generalized/left eigenvectors and homogeneous eig options remain absent.
`cdf2rdf` requires adjacent conjugate eigenvalues and conjugate vector columns;
it checks these with 1e-12 absolute plus 1e-10 relative tolerance. This validation
is stricter than SciPy's conversion. `rsf2csf` verifies real quasi-triangular
structure and orthogonal vectors, then delegates another complex Schur reduction.
Its eigenvalue order/phases can differ from SciPy's two-by-two conversion.

`ldl` uses private faer Bunch–Kaufman pivoting, permits indefinite and singular
matrices, and returns `(outer, D, permutation)` as an owned result. D has one-
or two-by-two blocks; `outer[permutation,:]` is unit triangular, and
`A=outer D outerᵀ`. Upper factors use reversed-axis lower factorization. Pivot
choices and factors can differ from SciPy. Input scaling protects large values.

`matrix_balance` returns `(B,T)` satisfying `B=T⁻¹ A T`. It supports isolation
permutations and radix-two scaling with at most 256 sweeps and 512 powers per
adjustment; results can differ from LAPACK. Diagonal scaling requires a 5%
off-diagonal one-norm improvement. Zero transformations and overflow are errors.

`qz` uses complex triangular output for real A/B, including singular B. It
returns S,T,Q,Z and homogeneous alpha/beta, with `A=Q S Zᴴ`, `B=Q T Zᴴ`.
Zero beta represents infinity; both zero is indeterminate. Faer QR feeds real
generalized Hessenberg/QZ, followed by complex triangularization. This
separates real blocks before the complex zero-beta chase, preserving accumulated
rotations on exactly singular real pencils. The transformed forms are recovered from the
original pencil and the returned vectors so backend zero-pivot sign conventions
cannot silently break reconstruction. Both forms must reach triangular structure.
Backend QZ sweep limits depend on order. `ordqz` accepts a scalar predicate on
(alpha,beta), moves selected eigenvalues first with local unitary swaps, and
reports their count. Inseparable or ill-conditioned reordering returns an error.
Real-output QZ and SciPy string selectors are replaced by complex output and
the predicate.

`cossin` accepts a real orthogonal matrix and two explicit partitions. It uses
faer SVD of X11, shared SVD null spaces, and SVD of the remaining X22 block.
It returns block-orthogonal U/VH, a central CS factor, and ascending angles.
Coupled directions precede uncoupled directions in the lower/right factors;
the central layout is permutation-equivalent to SciPy's layout. Full factors
are always returned. Cross norms below 1e-12 are treated as zero; near endpoints
basis directions can lose accuracy. Input orthogonality is checked at 1e-10
Frobenius tolerance.

`sqrtm` uses complex backend Schur and a native scalar triangular recurrence,
including repeated eigenvalues. Singular roots are accepted when that recurrence
exists; unsupported nilpotent roots return an error. `logm` requires nonsingular
input, takes at most 64 triangular square roots to reduce the Frobenius distance
from identity below 0.25, then evaluates the resolvent integral by 16-point
Gauss–Legendre quadrature using shared special-function nodes and backend LU. Both functions return
complex principal-branch matrices; negative real eigenvalues take the upper-bank
branch. Branch-cut sensitivity and nonnormality can harm accuracy.

`fractional_matrix_power` uses shared logarithm and Padé exponential evaluated with faer;
noninteger/negative powers require nonsingular input. Integer powers use bounded binary
exponentiation, permitting nonnegative powers of singular matrices. The exponent
magnitude and exponential one-norm are bounded at 2^32. `signm` excludes imaginary-
axis eigenvalues and uses `A sqrt(A²)⁻¹`. `funm` accepts a shared complex scalar
callback and uses backend Schur followed by scalar Parlett divided differences.
It rejects coupled repeated eigenvalues because the callback supplies no
required derivatives. Nearly repeated eigenvalues can lose accuracy. Callback
panics propagate. These adaptations do not claim SciPy's advanced clustered
Schur–Padé accuracy or error-estimation conventions.

Both Riccati solvers assemble a Hamiltonian/symplectic pencil, obtain the stable
subspace through ordered faer QZ, and recover X with LU. State order is at most
128; control dimension at most 256. R must be positive definite (stricter than
SciPy continuous ARE); Q/R must be symmetric. Results must be real, symmetric,
satisfy a scaled 1e-8 residual, and stabilize the closed loop. Singular or
inseparable subspaces return errors. Descriptor E, cross term S, and balancing
options remain absent; badly conditioned unbalanced problems may fail.

`clarkson_woodruff_transform` takes explicit bucket indices and ±1 signs for
reproducible CountSketch accumulation. SciPy RNG object and sparse-input overloads
are omitted. Independent uniform draws are required for the usual probability
bound; supplied arbitrary maps have no norm-preservation guarantee. Reference
inputs contain NumPy RNG draws and are evaluated by SciPy's public transform.

`find_best_blas_type` reports the fixed double-real prefix and row-major storage
of the shared matrices, including an empty input list. It inspects shapes only
and accepts IEEE entries. `get_blas_funcs` returns typed callable handles for
`dot`, `nrm2`, `gemv`, `gemm`; `get_lapack_funcs` supports `gesv`, `getrf`, `getrs`,
`potrf`, `potrs`, `geev`, `syev`, `syevd`, `gesvd`, `gesdd`, and `gges`. Optional d
prefixes are accepted. Driver aliases reuse the same Rust backend. These handles
use the owned functional signatures rather than raw work buffers, leading
dimensions, mutable results, or Python tuples. Unsupported names return errors.
This is limited dispatcher coverage; raw low-level BLAS/LAPACK parity is absent.

## Interpolative namespace

All nine names are exported under `linalg::interpolative`:
`estimate_rank`, `estimate_spectral_norm`, `estimate_spectral_norm_diff`,
`id_to_svd`, `interp_decomp`, `reconstruct_interp_matrix`,
`reconstruct_matrix_from_id`, `reconstruct_skel_matrix`, and `svd`.

The APIs use finite real dense matrices with axes at most 256. `Rank::Fixed(k)`
or `Rank::Relative(epsilon)` replaces SciPy's mixed eps_or_k argument.
Numerical rank counts singular values strictly above epsilon times the largest;
a zero matrix has rank zero. Pivoted faer QR chooses skeleton columns and shared
SVD pseudoinverse computes the projection. Pivot choices may differ from SciPy;
a singular-value cutoff does not guarantee an ID residual at that tolerance.
Reconstruction validates full permutations and exact coefficient shapes,
including zero/full-rank representations. Direct truncated SVD gives the shared
`SvdResult` with VH, rather than SciPy's V. Norm/rank estimates compute the full
SVD deterministically, replacing randomized operator/iteration options.
`id_to_svd` reconstructs the bounded dense matrix before SVD.

Fixtures use SciPy interpolative reconstruction/decomposition where appropriate.
For deterministic SVD/norm/rank adaptations, SciPy dense SVD and spectral norms
provide independent references. The zero-matrix fixed-rank ID reference is the
zero identity, avoiding SciPy's undefined projection from division by zero.
Nonunique IDs are checked by residuals; eigen/Schur/CS factors by reconstruction,
orthogonality and spectra, with separate structure checks. NaN/infinity,
singularity, invalid permutations, callback outputs, work bounds, branch values,
very small/large scales, rank-zero shapes, and dispatch invocation are tested.

## Remaining behavioral gaps

No top-level or interpolative inventory names remain absent. Full SciPy
behavioral parity remains unfinished: low-level BLAS/LAPACK/Cython namespaces,
complex matrix inputs beyond the conversion helper, batches, generalized eig
options, driver/storage/mutation controls, advanced matrix-function accuracy,
descriptor/cross-term Riccati options, randomized/operator interpolative paths,
and the documented signature/domain adaptations above remain gaps.

Matrix multiplication also bounds rows*inner_dimension*columns at 16,777,216
scalar products. This work check precedes reading or copying external storage;
empty products remain supported. Storage bounds continue to apply separately.
