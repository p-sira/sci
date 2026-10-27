# Changelog

## 0.3.0

- Add generic complex log-gamma, reciprocal gamma and digamma with shared errors,
  signed branch-cut banks, overflow-resistant evaluation and independent fixtures.

- Add eight generic scalar special functions and 19 shared-complex functions,
  with bounded numerical work, independent precision fixtures and an explicit
  native XSF capability audit. See function_coverages/SPECIAL_NATIVE_COVERAGE.md.

- Add ellip-style generic floats to 142 numerical APIs, with native `f32`/`f64`
  arithmetic, generic shared containers defaulting to `f64`, independent reference
  fixtures, and documented remaining binary64 APIs. See MIGRATING.md and
  function_coverages/GENERIC_FLOAT_COVERAGE.md.

- Complete a second API audit: remove remaining compatibility aliases and the SciPy-calibrated convolution cost selector; use snake case for the STFT dual-window helper.

- Remove SciPy runtime compatibility APIs and redundant aliases; add concrete sparse index conversions and standard owned operator conversion. See MIGRATING.md for replacements.

- Consolidate particle filtering on Smelly 0.2, decode dense numeric MATLAB files
  with the shared native parser, and disable unused statrs and rand features.
- Reduce internal copies in signal validation and construction, SVD pseudoinverses,
  distance weights, surface interpolation, statistical trimming, and sparse solvers.
- Retain numerical algorithms and validation apart from the documented API redesigns.

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
