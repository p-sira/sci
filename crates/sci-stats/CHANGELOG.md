# Changelog

## 0.3.0

- Add Sobol (21,201 dimensions, 64-bit sequences, LMS scrambling), Halton,
  Latin hypercube, Poisson-disk and multivariate/multinomial QMC engines,
  with shared `QmcEngine` and bounded integer sampling APIs.
- Add `stats::sampling` with discrete alias/guide-table, inverse-CDF and
  ratio-of-uniforms/transformed-density rejection samplers.
- Improve sampling accuracy and speed with contiguous Sobol output, optimized
  QMC generation, cached Hermite interpolation and rejection envelopes while
  preserving seeded sequences. Add CodSpeed and SciPy 1.18.1 comparisons;
  see `function_coverages/SAMPLING_BENCHMARK_COMPARISON.md`.
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
