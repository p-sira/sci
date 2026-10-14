# Interpolation coverage

All **25 public function names** in `scipy.interpolate` have exported, tested
Sci signatures. This measures numerical function-name coverage, with the
compatibility namespaces in [the inventory](scipy-func-list/scipy.interpolate.md)
collapsed. Python classes, axis/batch overloads and every SciPy option are not
claimed. There are no placeholder functions. Legacy `splrep` and `splprep`
reuse the modern fitting implementations.

## Representations and conventions

`Spline`, `ParametricSpline` and `SurfaceSpline` are shared `sci-core` types,
re-exported by the facade and interpolation module. All functions validate
public representation fields before numerical use. Scalar splines store the
complete nondecreasing knot vector, exact-length coefficients, and polynomial
degree (order minus one). The base interval is `[t[k], t[n]]`, where n is the
coefficient count. SciPy's unused padded coefficients are omitted.

Curves store one spline per coordinate and the increasing sample parameters;
input points are sample-major. Surface coefficients are flat, with the y basis
index varying fastest. Surface evaluations retain x rows and y columns.
Regular grids use flat row-major values with the last axis varying fastest.
No backend type is exposed. Input/sample units are retained; derivatives divide
by coordinate units and integrals multiply by coordinate units.

## Supported functions

| Functions | Supported behavior |
| --- | --- |
| `barycentric_interpolate`, `krogh_interpolate`, `lagrange`, `pchip_interpolate`, `pade` | Existing scalar polynomial, Hermite, shape-preserving and rational APIs; see [numerical coverage](NUMERICS_COVERAGE.md#interpolation) |
| `approximate_taylor_polynomial` | Scalar callback; degree ≤ fitting order ≤31; positive finite sampling scale; descending coefficients in displacement from the expansion point |
| `make_interp_spline` | Not-a-knot degrees 1..=5, including midpoint sites for even degrees; private rank-checked SVD collocation |
| `make_lsq_spline` | Supplied complete knots, degrees 0..=5, positive weights; minimizes squared weighted residuals using private SVD |
| `make_splrep`, `splrep` | Explicit residual target and weights; interpolation degrees 1,3,5; positive smoothing degrees 3,5 through private pure-Rust Dierckx fitting |
| `make_splprep`, `splprep` | Curves with 1..=3 coordinates; explicit parameters or normalized chord length; shared-knot interpolation degrees 1,3,5 and smoothing degrees 3,5 |
| `generate_knots` | Owned refinement snapshots; weighted LSQ target; at most 64 samples; one residual-guided median-site insertion per step; zero target gives not-a-knot knots |
| `make_smoothing_spline` | Natural cubic minimization of `Σ w_i(y_i-S(x_i))² + λ∫(S''(x))²dx`, with explicit λ≥0; exact piecewise-polynomial roughness quadrature and private constrained LU |
| `splev` | Ordered scalar queries and derivatives through degree; extrapolate, zero, reject or clamp outside the base interval |
| `splder`, `splantider` | Coefficient recurrences; derivative order through degree; antiderivative order at most 16, with SciPy integration constants |
| `spalde` | Query-major matrices, columns in ascending derivative order; queries within the closed base interval |
| `splint` | Analytic definite integral, with zero extension outside the base interval and signed reversed bounds |
| `insert` | Nonperiodic interior Boehm insertion; multiplicity through degree+1; unchanged function up to roundoff |
| `sproot` | Cubic splines with simple interior knots; bounded private companion-matrix Schur solves and Newton refinement; sorted roots in the closed base interval |
| `bisplrep` | Fixed-knot weighted tensor-product LSQ fitting (SciPy task=-1), degrees 1..=5; private rank-checked SVD |
| `bisplev` | Cartesian products of nondecreasing x/y queries, mixed partial derivatives through the degrees, and FITPACK-style coordinate clamping |
| `interpn` | Scalar multilinear/nearest interpolation on 1..=8 ascending or descending axes, singleton axes, errors/fills/extrapolation |
| `griddata` | Nearest in 1..=8 dimensions; linear in 1D and on private 2D Delaunay triangulations; not-a-knot cubic in 1D; optional coordinate rescaling |

## Limits, errors and numerical behavior

Spline fitting accepts at most 128 samples. General scalar representations have
at most 512 knots and degree at most 32; antiderivatives must remain within those
limits. Public queries and scalar data are finite. IEEE special fill values are
allowed explicitly by the multidimensional functions. Invalid shapes, domains,
nondifferentiable knot multiplicities, nonfinite arithmetic, rank deficiency
and backend nonconvergence return shared Sci errors.

Dense solves have explicit dimension and work checks; SVD is limited to 1,000
iterations. Scalar evaluation bounds
query count times `(degree+1)^2` by 10,000,000; all-derivative evaluation uses the
cube. Grid corner work, nearest comparisons and triangle scanning each have
10,000,000-operation bounds. Scattered sites are limited to 10,000; 1D linear
and cubic construction uses the 128-sample fitting limit. General grid storage
and query counts are limited to 1,000,000.

Dierckx smoothing is limited to 20 backend iterations, and its residual target
allows 0.1% relative tolerance. Returned fits undergo independent residual
checks through Sci's evaluation kernel. Small targets or extreme scales can
fail. Finite ill-conditioned results have no accuracy guarantee.

Root solving permits 256 Schur iterations per span and four Newton refinements.
Leading polynomial coefficients below 64ε relative to the largest are discarded.
Roots within 1e-12 of normalized span endpoints are clamped; duplicates within
1e-12 of the base interval width are merged. Near-multiple roots can be missed
or inaccurate. Identically zero spans do not enumerate infinitely many roots.
Triangle membership uses 64ε barycentric tolerance; cocircular triangulation
choices may differ from SciPy.

## Remaining SciPy behavior gaps

Function-name coverage does not include vectorized axes/batching, callable
Python polynomial/spline objects, custom boundary constraints, periodic
fitting/insertion, custom fitting bounds, persistent task caches, diagnostic
return tuples or user-selectable solve methods. Degree-zero not-a-knot
interpolation and positive smoothing with linear/even degrees remain unsupported.
`make_smoothing_spline` requires explicit λ; automatic GCV selection is absent.
`bisplrep` requires fixed knots; adaptive SURFIT knot selection and smoothing
are absent. `interpn` omits higher-order tensor splines, PCHIP and splinef2d.
`griddata` omits 2D Clough–Tocher cubics and higher-dimensional linear simplices.
`generate_knots` omits custom bounds/nest and FITPACK's batched insertion
trajectory; its positive-target sequence may differ while satisfying the same
weighted LSQ criterion.

## Verification

[interpolation_cases.json](tests/scipy/interpolation_cases.json) contains inputs
only. The [pinned generator](tests/scipy/generate_interpolation.py) uses SciPy
1.18.1 and NumPy 2.5.4 to write separate input/expected CSVs. Ordinary Rust tests
read these committed fixtures without Python. The new 124 reference cases cover
all 20 added names; together with the existing 43 interpolation cases this gives
**167 reference cases across 25 names**.

Fixed-knot surface fitting references use public SciPy `LSQBivariateSpline`
with explicit bounds, mathematically equivalent to task=-1 `bisplrep`.
For repeated knots at unclamped base boundaries, evaluation/extrapolation uses
the first/last nonempty polynomial span and clamps derivative queries when
requested. These cases use public SciPy PPoly references built from BSpline,
excluding zero-width spans; legacy splev may return zero outside such a boundary
or extrapolate derivatives despite ext=3. Scalar references check values, derivative ordering, degree, knots and coefficients;
curve/surface references check coordinate order and output shapes. Comparisons
use 2e-10 absolute plus 2e-10 relative tolerance for bounded low-order dense solves
and coefficient transformations. Root fixtures use well-separated simple roots;
no near-multiple-root accuracy claim is made.

Independent properties verify polynomial reproduction, derivative/antiderivative
round trips, knot-insertion invariance, signed integration reversal, root
residuals, natural boundary conditions and residual-guided refinement. Invalid
sizes, NaN/infinity, malformed shared types, unsupported modes and work limits
have separate tests. Taylor controls and indistinguishable generated nodes are
validated before callbacks.

```console
uv run tests/scipy/generate_interpolation.py
python tests/scipy/audit_numerics.py
cargo test --no-default-features --features interpolate --test interpolation
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps
```
