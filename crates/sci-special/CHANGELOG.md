# Changelog

## 0.3.0

- Replace the private `xsf` and `special` dependencies with pure Rust kernels, reusing `complex-bessel`, `ellip`, `lambert_w`, and `nalgebra`. Retain all 353 public numerical functions and remove the C++/libclang build requirement.
- Extend 104 scalar and polynomial APIs to `sci_core::Float`, with independent f32/f64 fixtures. Document overload gaps and numerical work limits in `function_coverages/SPECIAL_COVERAGE.md`.
- Correct Airy/Struve integral transition accuracy, negative modified-Fresnel reflection, and spherical derivative/scaled-Bessel signed-zero conventions using independent references.

- Make all elliptic wrappers and logsumexp/softmax reductions generic, with private backend dispatch for Bulirsch precision.

- Consolidate psi, assoc_laguerre and ellipf on digamma, eval_genlaguerre and ellipkinc; see MIGRATING.md for argument order.

- Remove geterr, seterr, ErrorPolicy, abbreviated quadrature-root aliases, and scalar wrappers cbrt, exp2, exp10, expm1, log1p and round. Use standard f64 methods; round maps to round_ties_even.
- Avoid principal Lambert W cancellation near zero with a bounded Taylor expansion, preserving signed zero and the relative accuracy of tiny inputs.

## 0.2.0

Changes since the repository tag `v0.1.0`.

- Require `sci-core 0.2` and update Sci dependency requirements to matching releases. Shared error and numerical types are incompatible with those from `sci-core 0.1`.
- Improve Student-t `stdtr` and `stdtrit` accuracy near the median and use stable elementary formulas for one and two degrees of freedom.
- Use the private XSF Box–Cox kernels to preserve finite results for tiny parameters and large powers, while explicitly retaining endpoint conventions.

## 0.1.0

- Initial release of special mathematical functions.
