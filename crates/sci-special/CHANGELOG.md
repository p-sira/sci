# Changelog

## Real modified Bessel performance follow-up

- Port Cephes' real `iv` kernel with internally scaled `ive`, replacing complex AMOS dispatch for real arguments. Bound Temme iterations and large-order asymptotics; preserve finite overflow-adjacent and subnormal values with exponential and recurrence scaling.
- Add pinned independent SciPy fixtures for both float APIs, transition boundaries, reflection, and exceptional values, plus recurrence and scaling identities. Retain the Cephes/Boost license notices.

## Numerical coverage follow-up

- Extend independent public-API references across Legendre continuation, large-order Bessel functions, error-function approximation intervals, complex Fresnel axes, integral asymptotics, and beta/gamma inverse tails.
- Preserve tiny beta probabilities during inverse reflection and retain small quantile coordinates. Repair Tricomi U's zero-second-parameter dispatch and near-integer cancellation with shared transformations and recurrences.
- Combine Bessel J recurrence scales before exponentiation, retain subnormal results, and fall back to the shorter recurrence when normalization to J₀/J₁ overflows.
- Evaluate the K₀ integral tail with a shifted, scaled Gaussian rule instead of cancellation-prone series and a short asymptotic expansion.
- Remove constant-false validation blocks from parameter-free polynomial constructors. Document measured coverage and remaining numerical accuracy limits.

## Ellip notation and documentation follow-up

- Prefer `ellipf(phi, m)` for `F(phi, m)` and retain `ellipkinc` as an alias.
- Admit Ellip-supported finite real incomplete cases and principal values; document Sci's finite-input/result restrictions separately.
- Document elliptic definitions, wave equations, output ordering, derivative units, work bounds, and numerical limitations. SciPy parity is not a design requirement.
- Update `lambert_w` to 2.0.8 and remove the superseded near-zero Taylor workaround; retain the branch-point Puiseux correction, verified against new independent 90-digit references.

## 0.3.0

- Add complex Lambert W, E₁/Ei, Spence's integral, complex logarithmic products,
  and bounded complex Riemann/Hurwitz zeta with signed-cut conventions, strict
  convergence/error handling and independent numerical references.
- Avoid `ellipj(u,1)` overflow for large finite arguments by evaluating the
  exact hyperbolic identities; use AGM outside the near-unit perturbation range.

- Expand independent reference coverage and domain/work-limit regressions. Correct terminating hypergeometric series, small-argument U, oblate radial indexing/derivatives, Legendre singular endpoints, and modified-Struve series selection.
- Replace the dense Mathieu eigenproblem with bounded tridiagonal iteration. Cache quadrature rules and use double-double Airy/Struve integral series on common domains. Remove private copied algorithms unused by the public API.

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
