# Changelog

## 0.3.0

Use this release with `sci-core 0.3` and matching Sci crate releases when exchanging shared types.

### Added

- Support `f32` and `f64` throughout finite-difference derivatives, gradients, Jacobians, and Hessians. Callbacks evaluate in the selected precision.

### Compatibility

- Explicit generic arguments now include the scalar type, for example `derivative::<f32, _>(...)`. Choose steps and tolerances appropriate to the precision.

## 0.2.0

- Upgrade to shared `sci-core 0.2` types. Update cooperating Sci crates together; `sci-core 0.1` types are incompatible.

## 0.1.0

- Initial release of finite-difference derivatives, Jacobians, and Hessians.
