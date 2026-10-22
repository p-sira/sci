# Changelog

## 0.2.0

Changes since the repository tag `v0.1.0`.

- Require `sci-core 0.2` and matching `0.2` releases of the existing Sci crates.
  Shared error and numerical types are incompatible with `sci-core 0.1` types.
- Add default-enabled `sparse` and `io` modules through the initial `0.1.0`
  releases of `sci-sparse` and `sci-io`, and re-export the shared sparse types.
- Include numerical accuracy and performance improvements in integration,
  dense solves, optimization, convolution, special functions, and statistics.
  Each crate records its changes in `crates/<crate>/CHANGELOG.md`.
- Update statrs to 0.19.1, unify nalgebra on 0.35, and replace
  anofox-statistics with private SciPy adaptations in statistics.
- Anchor package include patterns to the crate root so local environments'
  license and changelog files are excluded from release archives.

## 0.1.0

- First release.
