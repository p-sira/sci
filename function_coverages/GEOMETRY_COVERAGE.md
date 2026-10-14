# `spatial` coverage and SciPy adaptations

`spatial` implements the 33 numerical function
names in the supplied [spatial inventory](scipy-func-list/scipy.spatial.md).
`spatial::distance` holds all 27 distance functions; they are also exported
from `spatial`. The other six numerical functions are exported from `spatial`.
The four deprecated kdtree/qhull function entries duplicate these functions;
the deprecated module paths are not reproduced. No Cython-only entries exist.

## Numerical function coverage

| Group | Implemented names | Reference tests |
| --- | --- | --- |
| Numeric vector distances | `braycurtis`, `canberra`, `chebyshev`, `cityblock`, `correlation`, `cosine`, `euclidean`, `hamming`, `jensenshannon`, `mahalanobis`, `minkowski`, `seuclidean`, `sqeuclidean` | Multiple unweighted/weighted inputs, normalization, p including infinity, zero norms, empty vectors where SciPy defines a result |
| Boolean distances | `dice`, `jaccard`, `rogerstanimoto`, `russellrao`, `sokalsneath`, `yule` | Weighted/unweighted inputs, matching and mismatching coordinates, all-false special values |
| Pairwise distances | `cdist`, `pdist` | All 19 Metric variants, rectangular shapes and condensed ordering |
| Distance storage | `is_valid_dm`, `is_valid_y`, `num_obs_dm`, `num_obs_y`, `squareform` | Several matrix/vector sizes, invalid symmetry/lengths, tolerance, both conversion directions |
| Point-set distances | `distance_matrix`, `minkowski_distance`, `minkowski_distance_p`, `directed_hausdorff` | Several norms/dimensions and directed witness validation |
| Alignment/interpolation | `procrustes`, `geometric_slerp` | Reflections, dilation, nonzero disparity, rank deficiency, unit spheres, extrapolation, near-coincident endpoints |
| Simplex lookup | `tsearch` | SciPy Delaunay lookup in 2D and 3D; independent multiple-cell/boundary tests |

There are 158 pinned SciPy 1.18.1 / NumPy 2.5.4 reference cases, stored separately
from results in `tests/fixtures/geometry_inputs.csv` and `geometry_expected.csv`.
`tests/scipy/generate_geometry.py` regenerates results; `audit_geometry.py`
compares the inventory with exports and verifies at least two cases per name.
Ordinary Rust tests use only checked-in fixtures. Comparisons use 3e-12 absolute
and relative tolerance for floating-point reductions, SVD, and cancellation near
zero. NaN/infinity comparisons are explicit. Hausdorff witnesses are checked
against returned distances because SciPy shuffles and can choose different tied
indices. Independent tests cover symmetry, triangle inequalities, storage round
trips, sphere norms, similarity invariance, boundary ties, signed zero, extreme
scales, invalid domains, work limits, and algebra interoperability.

## Signatures and supported domains

- Scalar coordinates and weights are `f64`; vectors are slices and matrices
  contain one owned vector per row. All coordinate inputs must be finite.
  Distances are in coordinate units, squared distances in squared units;
  ratios, probability distances, and standardized distances are dimensionless.
- Individual weighted metrics accept `Option<&[f64]>`. Weights must be finite,
  nonnegative and equal in length to the vectors; an explicit all-zero weight
  vector is invalid. Omitted weights mean ones. Chebyshev ignores weight
  magnitude and excludes zero-weight coordinates. Correlation takes an explicit
  `centered` flag; cosine reuses its uncentered kernel.
- Boolean metrics require 0/1 `f64` inputs. SciPy's numeric arithmetic extensions
  of Dice, Rogers–Tanimoto, Russell–Rao, Sokal–Sneath, and Yule are unsupported.
  Jaccard converts nonzero inputs to true, following modern SciPy. Dice with
  all-false vectors returns NaN; Sokal–Sneath rejects them; Jaccard returns zero
  for empty unions. Empty Chebyshev returns zero; empty Rogers–Tanimoto and
  Russell–Rao return NaN rather than SciPy's exceptions.
- `Metric` replaces metric strings in cdist/pdist. These calls use unit weights,
  centered correlation, and explicit p, variance, inverse covariance, or log
  base parameters. Automatic sample covariance/variance inference, callbacks,
  Python output buffers, batching, and broadcasting are unsupported.
- Lp functions accept positive p, including p<1 quasinorms and positive infinity.
  `minkowski_distance_p` returns the sum of pth powers, except at infinity where
  it returns the maximum difference. Its signature handles one vector pair.
- Standardized Euclidean requires strictly positive finite variances.
  Mahalanobis requires an exactly symmetric positive definite inverse
  covariance matrix and uses private nalgebra Cholesky. SciPy's singular or
  indefinite parameter behavior is excluded.
- Jensen–Shannon normalizes each finite nonnegative vector independently and
  requires positive total mass. The optional finite log base must exceed one;
  omitted base uses natural logarithms. Axis/keepdims options are omitted.
- `DistanceForm` and `DistanceOutput` select squareform conversion direction.
  Square input always receives exact symmetry and zero-diagonal checks; finite
  values are required in either form. Force/checks flags are omitted. Empty
  condensed input maps to [[0]]; empty square input maps to []. Negative values
  are allowed. No triangle-inequality or nonnegativity check is performed.
- `is_valid_dm` uses a finite nonnegative absolute tolerance and requires finite
  entries; negative diagonal entries must also be within that tolerance. SciPy
  uses a one-sided diagonal tolerance. `num_obs_dm` validates square shape and
  finiteness but does not require symmetry or zero diagonal. `is_valid_y` and
  `num_obs_y` inspect lengths only; NaN/infinity values are allowed there.
  Empty y is valid but its observation count is ambiguous and rejected.
  Warning/throw/name options are replaced by bool predicates or shared errors.
- Directed Hausdorff uses exhaustive search and returns a named distance/index
  result. RNG options are omitted; ties choose the first original indices.
- Procrustes returns named reference/transformed/disparity fields in SciPy's
  order. It centers and normalizes both inputs, then fits the second with an
  optimal rotation/reflection and nonnegative dilation using private nalgebra
  SVD. Duplicate points and rank-deficient point sets are supported; constant
  point sets are invalid.
- Geometric slerp returns one vector per interpolation parameter. Inputs must
  lie on the unit sphere within norm tolerance 1e-9. Parameter order is preserved
  and extrapolation is allowed, following SciPy 1.18. Antipodes within an explicit
  chord tolerance are rejected because their interpolation path is ambiguous;
  SciPy warns and chooses a path. Tolerance must lie in [0,1). Equal unit
  endpoints are copied, and nonunit coincident endpoints are rejected.
- `tsearch` accepts vertices, simplex index rows, queries and barycentric
  tolerance in place of a Delaunay object. It returns `Option<usize>` (None
  outside) instead of -1. The first containing simplex wins boundary ties.
  Private nalgebra LU solves supplied cells; nondegenerate arbitrary simplices
  are accepted. It does not construct or verify a Delaunay triangulation.

## Numerical limits and remaining gaps

Inputs and allocated outputs are limited to 1,000,000 entries (zero-dimensional
collections also count rows). Pairwise distances and directed Hausdorff allow
10,000,000 coordinate operations. Mahalanobis dimension is at most 64 and its
pairwise work limit counts one cubic factorization plus quadratic work per pair. Procrustes dimension
is at most 64, `nd²+d³` is at most 10,000,000, and SVD is capped at 10,000
iterations. Simplex lookup dimension is 1..=64 and `s d³+q s d²` is at most
10,000,000. Errors use the shared Sci Result. Overflow returns an error;
undefined metric ratios return NaN as documented. Underflow, centering
cancellation, ill-conditioned factorizations, and large-angle trigonometric
phase loss limit accuracy; there is no uniform relative error guarantee.

Weighted L2 norms accumulate weighted components with hypot; generalized Lp
norms use log-domain components to avoid intermediate weight-sum overflow.
Zero-weight coordinates are excluded from arithmetic and normalization scales
for cosine, correlation and Bray–Curtis. Extreme weights, subnormal component
rounding and logarithm rounding remain accuracy limits. Audit fixtures use
independent Decimal weighted norms and scale-invariant SciPy ratio references.

The three plotting functions (`convex_hull_plot_2d`, `delaunay_plot_2d`,
`voronoi_plot_2d`) are excluded from this numerical API. Spatial classes and
class methods remain unimplemented: convex hull/Delaunay/Voronoi construction,
halfspace intersections, spherical Voronoi, KDTree/cKDTree and rectangle
operations, and the Rotation/RigidTransform/Slerp/RotationSpline interfaces.
The deprecated compatibility namespaces and Python warning/lifecycle behavior
are excluded. These gaps are not counted as implemented coverage.

Mahalanobis batches validate and factor the supplied inverse covariance once.
Their work bound accounts for d³ + pairs*d² instead of repeating cubic
factorization per pair. Empty outputs still validate the covariance. The
standalone vector metric uses the same factorization and distance kernel.
