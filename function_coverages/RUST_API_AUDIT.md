# Second Rust API audit

This pass reviews the workspace's exported function declarations, short forwarding
implementations, compatibility comments, backend exports, and runtime controls.
The detailed changes concern special functions, statistics, and signal processing;
SciPy inventories supply numerical definitions rather than export requirements.

## Changes

| API | Decision and reason |
| --- | --- |
| `stats::kstest` | Remove the two-sample-only forwarding alias. `ks_1samp` and `ks_2samp` make the supported input explicit without Python overload dispatch. |
| `special::psi` | Remove the synonym of `gamma::digamma`; signature and implementation were identical. |
| `special::assoc_laguerre` | Remove the argument-order wrapper; use `orthogonal::eval_genlaguerre(n, k, x)`. |
| `special::ellipf` | Prefer Ellip’s `elliptic::ellipf(phi, m)` notation for `F(phi, m)`; retain `elliptic::ellipkinc` as a qualified compatibility alias. Parameter convention remains `m=k²`. No root-level numerical alias is retained. |
| `signal::choose_conv_method`, `ConvolutionMethod` | Remove the SciPy-calibrated timing model. Sci uses different kernels and work limits; numerical parity cannot validate a performance recommendation. Explicit direct, FFT, and overlap-add functions remain available. |
| `signal::closest_STFT_dual_window` | Rename to `closest_stft_dual_window`; retain its numerical definition, signature, and independent fixtures. |

See [MIGRATING.md](../MIGRATING.md) for replacements. Historical SciPy fixture
labels for numerical aliases remain independent reference cases; tests invoke
the retained public APIs. Cost-selector cases are removed and signal fixtures
regenerated with the pinned Python environment.

## Retained APIs

- Integer-order Bessel functions `jn`, `yn`, and `kn` express a useful typed
  domain. Their `i32` order distinguishes them from real-order functions.
- `ishermitian` names a mathematical property even though its current real
  scalar domain makes it equivalent to `issymmetric`.
- `identity`, `median`, `std`, specialized clustering methods, and scalar
  distribution helpers express useful mathematical operations with narrower
  inputs or conventions. A short implementation alone does not justify removal.
- Complex scalar functions remain functional APIs over Sci's shared `Complex`.
  Rust's `f64` methods do not replace complex branch behavior.
- Matrix/graph validity predicates examine runtime structure and numerical
  invariants. Their purpose differs from Python dynamic-type predicates.
- Matrix Market, WAV, MATLAB, and NPZ interoperability helpers remain useful
  Rust I/O, with concrete storage types and explicit validation.
- FFT transforms, spectral operators, spline evaluation/derivatives/integration,
  and factorization/solve pairs perform distinct operations. Backend types and
  Python runtime controls remain private or excluded.

## Further design candidates

These limitations remain explicit rather than being described as completed
redesigns:

- `stats::make_distribution` stores function pointers. It accepts noncapturing
  callbacks only. Supporting captured parameters requires deciding whether to
  introduce a borrowed/generic distribution interface or owned callbacks; changing
  the current `Copy` distribution description has consequences across evaluation
  functions. Avoid global mutable state as a workaround.
- Several APIs use booleans for mathematical choices, including triangular
  orientation in dense linear algebra, connection type in sparse graphs, and
  bias/normalization in statistics. Named enums or options would make call sites
  clearer. Design these in coherent module groups, with explicit defaults and
  migration examples, rather than replacing every boolean mechanically.
- Several multi-output functions return tuples. Existing named result types
  provide a model for improving ambiguous output ordering; replacements need to
  preserve shape and mathematical conventions across modules.

These are ergonomics candidates, not additional compatibility functions to
remove. Existing numerical gaps and accuracy limits remain in the module
coverage reports and [numerical audit](NUMERICAL_ACCURACY_AUDIT.md).
