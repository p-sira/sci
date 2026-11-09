# Changelog

## 0.3.0

- Update shared Sci dependency requirements to the new minor versions.
- Add dense bound-constrained nonlinear least squares through `least_squares::dogbox`, with rectangular trust-region dogleg steps, SVD Gauss–Newton solves, analytical or bound-aware finite-difference Jacobians, and SciPy reference tests.
- Prevent false dogbox stationarity from symmetric numerical-difference samples and preserve exact bound intersections; document callback errors, stopping rules and numerical limits.

## 0.2.0

Changes since the repository tag `v0.1.0`.

- Require `sci-core 0.2` and update Sci dependency requirements to matching releases. Shared error and numerical types are incompatible with those from `sci-core 0.1`.
- Improve `rosen` and `rosen_der` accuracy near the Rosenbrock valley by preserving the rounding error in squared terms.

## 0.1.0

- Initial release of optimization, root finding, and derivative helpers.
