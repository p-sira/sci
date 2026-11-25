# Changelog

## 0.3.0

Use this release with `sci-core 0.3` and matching Sci crate releases when exchanging shared types.

### Breaking changes

- Move transforms into family modules, for example `rfft` to `real::rfft`. See the [migration guide](../../MIGRATING.md) for import changes.
- Remove the `real` argument from `next_fast_len` and `prev_fast_len`. Both now choose lengths with prime factors at most 11; this is a bounded heuristic rather than CPU-specific tuning.

### Added

- Support native `f32` and `f64` Fourier, real, cosine/sine, multidimensional, and periodic transforms. FFTLog remains `f64`.
- Provide `fht`, `ifht`, and `fhtoffset` through the default-enabled `hankel` feature without a C++ compiler or libclang. Disable default features to omit FFTLog; facade users select it with `fft-hankel`.

### Improvements

- Speed up real inverse and multidimensional real transforms, with less temporary storage and reusable transform plans.

## 0.2.0

- Upgrade to shared `sci-core 0.2` types. Update cooperating Sci crates together; `sci-core 0.1` types are incompatible.

## 0.1.0

- Initial release of Fourier transforms, real transforms, and periodic operators.
