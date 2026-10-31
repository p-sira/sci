# Added SciPy numerical modules

These modules provide 41 public inventory names in four namespaces.
It does **not** establish complete public SciPy parity. Scope comes from
[SCIPY_FUNCTIONS.md](SCIPY_FUNCTIONS.md), using the public namespaces, with
compatibility paths and Python/Cython duplicates collapsed.

| SciPy namespace | Sci namespace | Tested names / inventory |
| --- | --- | ---: |
| `scipy.constants` | `constants` | 7 / 7 |
| `scipy.integrate` | `integrate` | 7 / 20 |
| `scipy.interpolate` | `interpolate` | 23 / 25 |
| `scipy.cluster.vq` | `cluster::vq` | 4 / 5 |
| `scipy.cluster.hierarchy` | `cluster::hierarchy` | 27 / 30 |

These counts omit redundant `py_vq`, `splrep`, and `splprep` aliases.
Use `vq`, `make_splrep`, and `make_splprep` respectively.
Classes, their methods, and NumPy vectorized overloads are not counted.
There are no placeholder implementations.

## Constants

Implemented: `convert_temperature`, `find`, `lambda2nu`, `nu2lambda`, `precision`,
`unit`, and `value`. All SciPy numeric named constants and conversion factors
are exported with their original spelling; Rust's `dyn` keyword is `r#dyn`.
`PHYSICAL_CONSTANTS` replaces SciPy's dictionary with a sorted immutable slice
of 445 owned-type entries, including historical keys. Values and uncertainties
are the SciPy 1.18.1 CODATA 2022 snapshot, in SI units. `find` searches only
SciPy's current CODATA keys, as SciPy does. Unknown exact lookup keys return
an error. Historical lookup warnings and `find(disp=True)` printing are omitted.

Temperature scales use a `TemperatureScale` enum instead of strings. Negative
absolute temperatures and IEEE special values follow SciPy arithmetic;
identical-scale conversions return the input exactly. Frequency/wavelength
conversion accepts negative values, signed zeros, infinities, and NaN.

## Integration

Implemented: `trapezoid`, `cumulative_trapezoid`, `simpson`, `cumulative_simpson`,
`romb`, `fixed_quad`, and `newton_cotes`.

Sampled APIs use slices, optional explicit coordinates, and scalar `dx`;
axis/broadcasting options are absent. At most 1,000,001 samples are accepted.
Finite data and coordinates are required, and overflowing results are errors.
Trapezoid and Simpson intervals normalize samples by powers of two and balance
multiplication of samples, widths and weights. This avoids common intermediate
overflow and premature subnormal rounding. Simpson ratios normalize interval
widths before addition, allowing representable integrals over very large widths.
Scalar and cumulative trapezoid/Simpson sums use Neumaier compensation to retain small residuals
when intervals cancel. Each cumulative prefix is rounded separately; the
compensation remains available for later prefixes. Extreme spacing ratios,
ill-conditioned weighted cancellation, subnormal rounding, and overflowing
running sums remain limitations.
`dx` is ignored when explicit coordinates are provided.

Trapezoids accept repeated and decreasing coordinates. Simpson requires
nondegenerate one- and two-interval widths for three or more samples;
degenerate SciPy zero-denominator conventions are not reproduced.
Cumulative Simpson requires strictly increasing explicit coordinates and
positive uniform spacing for three or more samples. This is stricter than
SciPy's uniform negative/zero `dx` arithmetic. It uses neighboring samples,
including future samples, rather than integrating prefixes independently.
Cumulative trapezoid accepts only a zero initial value; cumulative Simpson
prepends and adds an arbitrary finite initial value. Output lengths match SciPy.

`romb` accepts `2^k+1` samples, including two samples, and omits printed tables.
Its native trapezoid refinements apply spacing before summing samples, use
compensation, and avoid explicitly forming an overflowing total interval width.
Richardson corrections divide first if subtraction would overflow.
`fixed_quad` reuses shared Legendre quadrature, accepts orders 1..=256 and finite
bounds, and returns a scalar without SciPy's unused tuple entry. Its callback
is scalar and may capture extra parameters. Zero-width intervals skip evaluation.
Finite bounds map through clamped convex combinations, including when their
difference overflows. At most 256 finite callback values are buffered and
normalized before compensated weighting; balanced products restore the interval
and value scales afterward. This avoids common intermediate overflow and
preserves representable subnormal constant integrals. Callback rounding,
weighted cancellation, underflow and true or tableau overflow remain limits.
`newton_cotes` accepts 2..=13 explicit nodes, normalized to start at zero and
end at `N`. It returns weights in input order and the leading error coefficient;
SciPy's scalar order shorthand and `equal` flag are omitted. A centered
Vandermonde system uses the existing private nalgebra backend.

Remaining: `cubature`, `dblquad`, `lebedev_rule`, `nquad`, `nsum`, `odeint`,
`qmc_quad`, `quad`, `quad_vec`, `solve_bvp`, `solve_ivp`, `tanhsinh`, `tplquad`.
Adaptive quadrature, infinite intervals, multidimensional integration,
summation, Lebedev rules, and ODE/BVP integration still need implementations.

## Interpolation

Implemented: `barycentric_interpolate`, `krogh_interpolate`, `lagrange`,
`pchip_interpolate`, and `pade`.

Inputs are finite scalar nodes and values; output axes, batching, and derivative
output options are absent. Polynomial methods support extrapolation and
arbitrary node order. Barycentric interpolation accepts 1..=256 distinct nodes,
using log-scaled weights. Krogh interpolation accepts up to 32 entries and
supports contiguous repeated nodes, whose values specify successive derivatives
without factorial scaling. Lagrange interpolation accepts up to 32 distinct
nodes and returns descending coefficient vectors, including leading zeros.
The polynomial methods permit at most 1,000,000 node/query products.
Barycentric evaluation scales distances and values to avoid intermediate
reciprocal/numerator overflow, and reuses one scratch vector across queries.
Subnormal terms may still underflow; extrapolation and denominator cancellation
retain their existing accuracy limitations.

PCHIP accepts 2..=1,000,000 strictly increasing nodes and at most 1,000,000
queries. It extrapolates endpoint cubics and preserves shape only inside the
sampled range. Endpoint derivatives use scaled slopes, and node matches return
the stored value exactly even when the query and node have different zero signs.
Sample differences and secant/endpoint derivatives must remain representable;
nonfinite arithmetic remains an error. Padé accepts up to 65 ascending Taylor
coefficients and returns descending numerator/denominator vectors with `Q(0)=1`,
using private nalgebra LU. Optional numerator degree can limit the input coefficient prefix. Singular
systems and nonfinite output fail; ill-conditioned finite results have no
accuracy guarantee. High-degree monomial interpolation has the same limitation.

All 25 main-namespace interpolation function names now have tested native
signatures, including spline fitting/evaluation/calculus, curves, tensor
surfaces, regular grids and scattered data. See
[INTERPOLATE_COVERAGE.md](INTERPOLATE_COVERAGE.md) for supported methods,
shared spline types, work limits and remaining SciPy behavior gaps.

## Vector quantization

Implemented: `vq`, `py_vq` (alias), `whiten`, `kmeans`, and `kmeans2`.

Data uses nonempty finite row-major matrices. Labels are zero-based, earliest
code wins ties, and distances are Euclidean rather than squared. Inputs are
never automatically whitened. `whiten` divides by the population standard
deviation without centering; zero-variance features are retained without a warning.
It reuses shared statistics. Extreme variance overflow returns an error.

Both k-means APIs require explicit initial centroids. Random initialization,
stochastic restarts, array-library options, and warning/missing-cluster modes
are omitted. Empty clusters retain their centroids in `kmeans2` and are dropped
in `kmeans`. `kmeans2` performs a requested positive number of assignment/mean
steps and returns the final assignment labels before the final centroid update,
matching SciPy. `kmeans` takes an absolute mean-distance change threshold plus
an explicit iteration bound; nonconvergence returns an error. Its returned
mean distance is always measured against the returned centroids. Each matrix
has at most 1,000,000 entries; total feature comparisons are bounded by
10,000,000, including a final reassignment for `kmeans`.

27 of 30 hierarchy functions now have supported native signatures and tests.
See [CLUSTER_COVERAGE.md](CLUSTER_COVERAGE.md) for limits and the three remaining names.

## Verification

`tests/scipy/constants_cases.jsonl` and `numerics_cases.jsonl` contain inputs only.
Pinned generators use SciPy 1.18.1 and NumPy 2.5.4; the Rust tests ingest separate
checked-in input and expected CSVs. There are 715 numerical cases, 19 constant
cases, and comparisons for all 445 CODATA table entries. The constants generator
also uses `rustfmt` to reproduce the checked-in Rust data snapshot.

Reference comparisons check vector lengths, matrix shapes, row ordering, and
label values. Finite numerical comparisons use `3e-12` absolute plus `3e-12`
relative tolerance for low-order quadrature, polynomial coefficient conversion,
and LU roundoff. CODATA table values/units/uncertainties are compared exactly.
NaN, infinity, signed zero, invalid domains, work bounds, and input shapes have
separate tests. Independent properties check quadrature moments, polynomial
and Hermite identities, Padé Taylor matching, PCHIP monotonicity, temperature
and wavelength round trips, and statistics/spatial/algebra interoperability.

The generator uses fixed random seeds for barycentric weight permutation and
SciPy k-means. SciPy 1.18.1 misclassifies single-centroid matrix guesses in
`kmeans`; single-cluster references use its public `k=1` mode instead. A single
cluster's converged mean is independent of initialization. Deprecated public
`lagrange`, `pade`, and `py_vq` remain covered because the inventory includes them.

Ten additional interpolation references exercise subnormal query distances,
large positive/negative values, and large PCHIP endpoint slopes. Near-node
linear references use SciPy Krogh interpolation because SciPy's barycentric
implementation overflows its reciprocals. Large-value references normalize
sample values before SciPy evaluation and restore units afterward, using the
interpolants' homogeneity under constant value scaling. Rust property tests check
scale invariance, signed-zero node reproduction, and errors for overflowing results.
The additional spline/grid fixture suite brings interpolation coverage to
23 exported/tested names and 167 reference cases, with removed compatibility aliases
collapsed. See the dedicated coverage report for signature adaptations.

```console
uv run tests/scipy/generate_constants.py
uv run tests/scipy/generate_numerics.py
python tests/scipy/audit_numerics.py
cargo fmt --all -- --check
cargo test
cargo clippy --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --no-deps
```

## Wider remaining scope

Full public SciPy coverage remains incomplete. Entire namespaces still absent
include `sparse` (including graph and sparse linear algebra), `ndimage`, `odr`,
`io` and `datasets`. The ten distinct FFTPACK periodic operators live in `fft`;
four lower-level kernel/cache names remain absent (see
[FFT_COVERAGE.md](FFT_COVERAGE.md)). `optimize` has seven tested main-namespace names (see
[OPTIMIZE_COVERAGE.md](OPTIMIZE_COVERAGE.md)). The dedicated `fft` module now
provides all 35 numerical names; six Python runtime configuration names remain
absent (see [FFT_COVERAGE.md](FFT_COVERAGE.md)). The expanded dense linear algebra APIs and remaining gaps are
listed in [LINALG_COVERAGE.md](LINALG_COVERAGE.md). Additional
statistics gaps and signature adaptations are listed in
[STATISTICS_COVERAGE.md](STATISTICS_COVERAGE.md). Existing special, signal,
geometry, and differentiation coverage retains the limitations documented in
its module references and coverage files. SciPy runtime configuration, testing
utilities, plotting, Python classes, low-level BLAS/LAPACK, and Cython binding
surfaces require separate scope decisions and are not counted as numerical
function parity.

214 additional integration references cover two- through five-point scalar and
cumulative rules with subnormal samples or widths, large samples with tiny
widths, tiny samples with large widths, irregular grids, sign changes, and cancellation. SciPy
evaluates normalized samples and widths in NumPy longdouble before units are
restored; constants use the equivalent trapezoid identity because SciPy's
irregular Simpson weights lose constant integrals at extreme spacing ratios; cancellation references evaluate SciPy trapezoids directly
in longdouble. They never use Sci or its backends. Cancellation comparisons
allow absolute roundoff proportional to the integrated data magnitude;
subnormal references require nonzero outputs and allow one minimum subnormal
unit. Independent tests verify exact constant integrals, reversal, scalar versus
cumulative consistency, unit cancellation residuals, and true-overflow errors.

240 additional quadrature references cover Romberg endpoint/refinement and
Richardson scaling, Gaussian orders 1, 2, 5, 16, 64 and 256, subnormal intervals and
callback values, large values on tiny intervals, opposite-sign extreme bounds,
and normalized polynomial moments. The pinned SciPy generator evaluates
Romberg after sample/spacing normalization and fixed quadrature on normalized
intervals and functions, restoring units with NumPy longdouble. These cases use
relative comparisons plus one minimum-subnormal unit, so zero cannot replace a
nonzero reference. Independent tests check callback counts and bound inclusion,
reversal, polynomial moments, tiny constants, and true-overflow errors.

Differentiation's fixed stencils now divide by actual rounded coordinate spans.
Diagonal Hessians use unequal three-point divided differences and mixed entries
use rectangular secants, retaining the existing evaluation counts. This removes
false slopes and curvature from asymmetric floating-point increments. There are
45 independent differentiation cases, including 22 rounded-coordinate cases.
The pinned SciPy generator checks adaptive references; exact linear/quadratic
Hessian identities replace SciPy's spurious ~1.7e-13 curvature residuals.
Independent tests cover identity slopes across subnormal/overflowing spans,
linear and shifted-polynomial Hessians, and representable subnormal curvature.
Symmetric stencils retain second-order truncation; rounded asymmetry, callback
rounding, conditioning, and the existing work bounds remain documented limits.
