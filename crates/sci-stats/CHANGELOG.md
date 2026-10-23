# Changelog

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
