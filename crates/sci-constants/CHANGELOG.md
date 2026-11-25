# Changelog

## 0.3.0

Use this release with `sci-core 0.3` and matching Sci crate releases when exchanging shared types.

### Added

- Support `f32` and `f64` temperature, wavelength, and frequency conversions. The CODATA database remains `f64`.

### Improvements

- Speed up case-insensitive CODATA name searches with fewer allocations.

## 0.2.0

- Upgrade to shared `sci-core 0.2` types. Update cooperating Sci crates together; `sci-core 0.1` types are incompatible.

## 0.1.0

- Initial release of physical constants and unit conversions.
