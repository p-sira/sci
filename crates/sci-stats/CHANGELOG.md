# Changelog

## 0.3.0

Use this release with `sci-core 0.3` and matching Sci crate releases when exchanging shared types.

### Breaking changes

- Move statistical functions into family modules, for example `mean` to `descriptive::mean` and `normal_cdf` to `distributions::normal_cdf`. See the [migration guide](../../MIGRATING.md).
- Remove the two-sample `kstest` alias; select `goodness::ks_1samp` or `goodness::ks_2samp` explicitly.
- Remove scalar `abs`, `exp`, and `log` wrappers; use Rust's `abs`, `exp`, and `ln` methods.

### Added

- Support `f32` and `f64` descriptive statistics, including summaries, quantiles, moments, trimming, and standardized scores.
- Add Sobol sequences with up to 21,201 dimensions, 64-bit sequence indices, and LMS scrambling; add Halton, Latin hypercube, Poisson-disk, multivariate normal, and multinomial QMC engines with a shared `QmcEngine` interface and bounded integer sampling.
- Add `sampling` with discrete alias/guide-table, inverse-CDF, ratio-of-uniforms, and transformed-density rejection samplers.

### Fixed

- Improve Lévy probabilities and rank-sum p-values; preserve subnormal two-sided normal p-values.

### Improvements

- Speed up QMC generation and repeated sampling while preserving seeded sequences.
- Speed up medians, median absolute deviation, repeated-median regression, binned statistics, exact two-sample Cramér–von Mises tests, and multiscale graph correlation with less sorting and temporary storage.
- Reduce repeated work in hypothesis-test tails, trimming, and transform parameter searches.
- Reduce dependencies by removing direct `basin` use and unused statrs sampling features.

See the [sampling benchmarks](../../function_coverages/SAMPLING_BENCHMARK_COMPARISON.md) and [non-sampling performance comparisons](../../function_coverages/STATS_PERFORMANCE_COMPARISON.md) for benchmark conditions and results.

## 0.2.0

- Upgrade to shared `sci-core 0.2` types. Update cooperating Sci crates together; `sci-core 0.1` types are incompatible.
- Upgrade statrs from 0.18 to 0.19.1 and nalgebra from 0.33 to 0.35.
- Add the Lévy distribution and use checked specialized quantiles for chi and inverse-gamma distributions, with bounded inversion as a fallback.
- Correct geometric-distribution quantile endpoints when the success probability is one.
- Replace anofox-statistics with private BSD-licensed SciPy adaptations for Shapiro–Wilk and studentized-range calculations. Improve three-point Shapiro p-values and integrate Tukey HSD survival probabilities directly.
- Improve Student-t, Box–Cox, Yeo–Johnson, and power-mean accuracy, including behavior near zero power.

## 0.1.0

- Initial release of distributions, statistical tests, summaries, and transformations.
