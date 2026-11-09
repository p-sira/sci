# Changelog

## 0.3.0

- Cache per-segment Hermite inverse-interpolation coefficients and activate
  coarse probability bins for smaller tables without changing seeded draws
  or interpolation bits. A matched single-thread 1024-draw release run
  measured 17.7 µs versus SciPy 1.18.1's 41.4 µs. Add contiguous Sobol
  comparison cases and document remaining nested-row allocation overhead
  and the large centered Latin-hypercube benchmark gap; see
  `function_coverages/SAMPLING_BENCHMARK_COMPARISON.md`.
- Stream Sobol coordinates through contiguous row-major buffers in the QMC
  multivariate-normal and multinomial transforms, avoiding intermediate
  per-point allocations while preserving the existing seeded sequences.
  Add reusable Sobol `random_flat_into` output and optimize its state updates;
  accelerate plain Halton with exact reciprocal digit division, specialize
  Latin-hypercube initialization and floating-point stratum boundary checks,
  and reduce temporary Poisson-disk candidate allocations.
  Optimize inverse-CDF sampling with the native normal quantile fast path,
  indexed interpolation tables and early termination of construction probes;
  cache rejection sampler envelope transforms and invariant geometry, and
  validate only new tangent relationships during adaptive TDR insertion.
  Add paired SciPy 1.18.1 native performance harnesses for all 17 sampling/QMC
  engine variants, with separate construction/draw results and explicit
  comparability notes for methods with different sampling algorithms.
- Add Halton, Latin hypercube, Poisson-disk, multivariate-normal QMC and
  multinomial QMC generators alongside Sobol, with a shared `QmcEngine` trait
  and bounded integer-interval sampling. Introduce `stats::sampling` with
  discrete alias/guide-table, native and interpolated inverse-CDF, and bounded
  ratio-of-uniforms/log-tangent rejection samplers. Seeded generators use Rust
  `StdRng`; document SciPy adaptations and work limits.
- Add a stateful Sobol quasi-Monte Carlo generator in `qmc` with the full
  21,201-dimensional Joe–Kuo D(6) direction table, up to 64-bit sequences,
  optional reproducible LMS scrambling, base-two batches, fast-forward and reset.
  Validate generated points against independent SciPy references.
- Improve the new QMC engines' numerical and sequence correctness: handle all
  `u128` index digits in scrambled Halton, map binary64 samples exactly to
  integer intervals (including the full `i64` range), preserve Latin-hypercube
  strata at floating-point boundaries, and retire Poisson-disk active points
  only after consecutive failed proposals.
- Improve sampling accuracy and robustness for upper-tail conditional inverse
  CDFs, large-offset and subnormal Hermite interpolation, and rejection-envelope
  checks. Reduce work through cached Halton scramble tails and rejection weights,
  fewer inverse-CDF evaluations, precomputed Poisson-disk neighbor offsets,
  streamed multinomial QMC draws, and squared-distance comparisons.
- Add CodSpeed benchmarks for QMC point generation and integer mapping,
  discrete samplers, inverse quantiles, and rejection sampling, including
  separate setup and draw workloads.
- Use selection instead of a full sort for descriptive medians, median
  absolute deviation, and repeated medians in `siegelslopes`, retaining the
  original total ordering, signed-zero results, and even-sample interpolation.
  Theil–Sen still sorts all slopes for confidence-interval rank bounds.
  Native benchmarks improved roughly 5–8× for medians and median absolute
  deviation, and 2.6× for `siegelslopes`.
- Share a stable, contiguous bin-grouping kernel across 1D, 2D, and ND binned
  statistics, avoiding per-observation coordinate vectors and per-bin value
  vectors while retaining observation order and statistical reductions.
- Replace private `basin` bounded scalar minimization in transform and PPCC
  parameter searches with a safeguarded Brent implementation. Preserve closed
  bounds, finite objective validation, and bounded iteration; remove the direct
  `basin` dependency.

- Make descriptive statistics generic over `Float`, including summaries, quantiles, moments, trimming, and standardized scores.

- Remove the two-sample-only kstest alias; select ks_1samp or ks_2samp explicitly.

- Remove abs, exp and log scalar wrappers. Use f64::abs, f64::exp and f64::ln.
- Disable unused statrs sampling features while retaining multivariate distributions.
- Reuse sorted sample storage in one-sided and two-sided trimming.

- Eliminate temporary partition allocations in exact two-sample
  Cramér–von Mises and reuse permutation-invariant covariance and beta
  thresholds in multiscale graph correlation. Preserve exact-test outcomes
  and bounds; native benchmarks improved approximately 1.9× and 2.2×,
  respectively.

## 0.2.0

Changes since the repository tag `v0.1.0`.

- Require `sci-core 0.2` and update Sci dependency requirements to matching releases. Shared error and numerical types are incompatible with those from `sci-core 0.1`.
- Upgrade statrs from 0.18 to 0.19.1 and nalgebra from 0.33 to 0.35.
- Add the Lévy distribution and use checked specialized quantiles for chi and inverse-gamma distributions, with bounded inversion as a fallback.
- Correct geometric-distribution quantile endpoints when the success probability is one.
- Replace anofox-statistics with private BSD-licensed SciPy adaptations for Shapiro–Wilk and studentized-range calculations. Improve three-point Shapiro p-values and integrate Tukey HSD survival probabilities directly.
- Reuse stable shared Student-t and Box–Cox kernels; improve Yeo–Johnson transforms and power means near zero power.

## 0.1.0

- Initial release of distributions, statistical tests, summaries, and transformations.
