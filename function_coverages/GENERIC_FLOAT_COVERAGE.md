# Generic floating-point coverage

This audit covers the unreleased Sci 0.3.0 family and the existing 0.2.0 versions
of sci-sparse and sci-io. Package versions remain unchanged.

The convention follows ellip: `T: num_traits::Float`, re-exported as
`sci_core::Float` and `sci::Float`. Literal conversion uses the existing
`numeric_literals` dependency as in ellip. Generic numerical work retains its
scalar precision; no input arrays are silently promoted to binary64. Validation,
work limits, normalization, units, and error conventions remain per-function.

| Area | Generic public numerical functions | Precision and constraints |
| --- | ---: | --- |
| Complex | 34 | Every scalar operation; `Complex<T = f64>`; precision-dependent binary scaling and asymptotic thresholds |
| Constants | 3 | Temperature, wavelength, and frequency conversion; database and named constants remain `f64` |
| Differentiation | 4 | Derivative, gradient, Jacobian, Hessian, including callback inputs/outputs |
| FFT | 40 | Fourier, multidimensional, real/Hermitian, cosine/sine, frequency/shift helpers, and periodic operators; native `f32`/`f64` |
| Integration | 6 | Trapezoid/Simpson and cumulative forms, Romberg, fixed Gaussian quadrature |
| Interpolation | 4 | Barycentric, Krogh, Lagrange, PCHIP |
| Special | 28 | All 25 elliptic functions and logsumexp/softmax/log_softmax |
| Statistics | 23 | Mean/variance and the complete descriptive module |
| **Total** | **142** | Each function has independent reference coverage in both precisions |

`is_finite_slice` is also generic; scalar checks use `Float::is_finite`. Shared `Matrix`, `Spline`,
`ParametricSpline`, `SurfaceSpline`, `SparseMatrix`, `SparseArray`, `Complex`,
and `stats::Description` accept a scalar type and default to `f64`.

FFTs seal `FftFloat` to supported native types and keep private backend traits
and cache access out of the named public API. Each thread retains at most one
forward, inverse and real plan per precision for lengths up to 65,536; larger
transforms use temporary buffers. The specialized incomplete Bulirsch function
uses `BulirschFloat` dispatch to select native backend precision without
publishing ellip's precision-control trait.

Gaussian quadrature generates nodes/weights in binary64 through the existing
special-function backend, then rounds them to `T`. The constant conversions
round their binary64 constants to `T`. These routines cannot supply precision
beyond the source data. Float conversion assumptions match ellip's supported
`f32`/`f64` types; arbitrary custom Float implementations must represent the
small constants and integer sizes used by the documented algorithms.

## Inventory disposition

The SciPy inventory namespaces consulted are `scipy.constants`,
`scipy.differentiate`, `scipy.fft`, the distinct periodic operations from
`scipy.fftpack`, `scipy.integrate`, `scipy.interpolate`, `scipy.special`, and
`scipy.stats`. This is a scalar-precision adaptation of existing Rust exports;
no inventory names are added, and no aliases or Cython duplicates are counted
as new algorithms. Existing signatures, equations and domain adaptations remain
in the function documentation and module coverage reports. NumPy supplies the
independent complex and basic statistics definitions where SciPy has no direct
equivalent; quadrature supplies independent non-SciPy elliptic references.

## Validation

`tests/scipy/generate_generic_float.py` pins NumPy 2.5.4 and SciPy 1.18.1.
434 input-only cases are stored separately in `generic_float_inputs.csv`;
`generic_float_expected.csv` contains 868 expected rows. Inputs are rounded to
the target scalar precision before independent binary64 reference evaluation.
The Rust test exercises every listed function through the public Sci facade,
compares flattened shapes/order as well as values, and runs without Python.

Binary32 references use 5e-6 absolute/relative tolerances on these short,
moderately conditioned examples. Finite differences permit 3e-4 for cancellation
and the cubic derivative's known `h²` truncation error. Binary64 comparisons
use 5e-12 plus that explicit truncation term. NaN and infinity are classified
explicitly; separate boundary/property tests exercise signed zero, subnormals,
overflow scaling, invalid domains, empty inputs, precision-specific perturbation
collapse, transform round trips, cache reuse and cross-module interoperability.
Existing binary64 fixtures and tests continue to exercise the broader domains.
Sci returns sorted trimmed observations; SciPy's unspecified partition order is
sorted in the reference generator before comparison.

## Remaining binary64 APIs

Generic storage does not yet make all consumers generic. These groups retain
explicit `f64` signatures and are not counted as completed precision support:

- Dense linear algebra and its faer/ndarray/nalgebra input adapters; sparse
  algorithms, graph operations, linear operators, and sparse file conversions.
- Optimization, clustering, spatial geometry/distances, signal processing,
  state estimation/filtering, and numerical I/O conversion helpers.
- Distribution, inference, ranking, resampling, robust and other statistics
  modules outside the descriptive group.
- Non-elliptic special functions and polynomial/quadrature constructors tied to
  the existing XSF/statrs/nalgebra or binary64 private algorithms.
- FFTLog Hankel transforms and their complex log-gamma backend.
- Newton–Cotes weights and spline, multidimensional, rational and fitting
  interpolation APIs.

Several of these groups have backends capable of binary32 arithmetic, but their
current Sci adapters, result types and shared private algorithms require a
further coordinated migration. This audit records them as remaining work;
accepting `f32` storage alone is not claimed as numerical support. Existing
binary64 consumers now specify reduction types explicitly where generic finite
validation would otherwise leave inference ambiguous.
