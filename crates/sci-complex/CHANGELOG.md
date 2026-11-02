# Changelog

## 0.3.0

- Avoid trigonometric rounding residues in the imaginary component of real-axis
  integer `powf`/`powc` results; preserve signed-zero branch limits and tiny
  nonzero imaginary components near the negative-real axis.

- Support `Complex<T = f64>` and native generic arithmetic; derive binary scaling and overflow thresholds from the selected precision.

- Update shared Sci dependency requirements to the new minor versions.

## 0.2.0

Changes since the repository tag `v0.1.0`.

- Require `sci-core 0.2` and update Sci dependency requirements to matching releases. Shared error and numerical types are incompatible with those from `sci-core 0.1`.

## 0.1.0

- Initial release of complex-number utilities and elementary functions.
