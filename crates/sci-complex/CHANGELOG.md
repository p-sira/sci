# Changelog

## 0.3.0

Use this release with `sci-core 0.3` and matching Sci crate releases when exchanging shared types.

### Added

- Support `Complex<f32>` and `Complex<f64>` with arithmetic and overflow thresholds appropriate to each precision. `Complex` continues to default to `f64`.

### Fixed

- Return exact zero imaginary components for real-axis integer powers in `powf` and `powc`. Preserve signed-zero branch limits and tiny imaginary components near the negative real axis.

## 0.2.0

- Upgrade to shared `sci-core 0.2` types. Update cooperating Sci crates together; `sci-core 0.1` types are incompatible.

## 0.1.0

- Initial release of complex-number utilities and elementary functions.
