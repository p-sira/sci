# Changelog

## 0.3.0

Use this release with `sci-core 0.3` and matching Sci crate releases when exchanging shared types.

### Breaking changes

- Move optimization functions into family modules, for example `minimize` to `minimize::minimize`. See the [migration guide](../../MIGRATING.md).

### Added

- Add `least_squares::dogbox`, `dogbox_with_jacobian`, and their fallible `try_` variants for dense bound-constrained nonlinear least squares. Use rectangular trust-region dogleg steps, SVD Gauss–Newton solves, and analytical or bound-aware finite-difference Jacobians. Document callback errors, stopping rules, and work limits.

### Fixed

- Prevent false stationarity in dogbox from symmetric finite-difference samples and preserve exact intersections with bounds.

### Improvements

- Reuse Dogbox's Gauss–Newton step and Cauchy coefficients across rejected trust-region trials.
- Reduce temporary storage for Rosenbrock Hessian-vector products.

## 0.2.0

- Upgrade to shared `sci-core 0.2` types. Update cooperating Sci crates together; `sci-core 0.1` types are incompatible.
- Improve `rosen` and `rosen_der` accuracy near the Rosenbrock valley by preserving the rounding error in squared terms.

## 0.1.0

- Initial release of optimization, root finding, and derivative helpers.
