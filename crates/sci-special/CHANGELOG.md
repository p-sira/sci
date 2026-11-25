# Changelog

## 0.3.0

Use this release with `sci-core 0.3` and matching Sci crate releases when exchanging shared types.

### Breaking changes

- Move numerical functions into mathematical-family modules, for example `gamma::gamma`, `bessel::jv`, and `orthogonal::eval_legendre`. Numerical functions are no longer exported at the crate root. See the [migration guide](../../MIGRATING.md) for paths and replacements.
- Remove `ellipf`; use `elliptic::ellipkinc(phi, m)` for the incomplete first-kind elliptic integral, with amplitude in radians and parameter `m = k²`.
- Replace `psi` with `gamma::digamma` and `assoc_laguerre(x, n, k)` with `orthogonal::eval_genlaguerre(n, k, x)`, noting the reordered arguments. Replace abbreviated quadrature-root aliases with the corresponding `orthogonal::roots_*` functions listed in the migration guide.
- Remove `geterr`, `seterr`, and `ErrorPolicy`; inspect returned values or shared errors as documented per function.
- Remove scalar `cbrt`, `exp2`, `exp10`, `expm1`, `log1p`, and `round` wrappers. Use Rust's `cbrt`, `exp2`, `10.0_f64.powf(x)`, `exp_m1`, `ln_1p`, and `round_ties_even` respectively.

### Added

- Build entirely in Rust without XSF, C++ compilation, or libclang.
- Support `f32` and `f64` across scalar and polynomial APIs, elliptic integrals, and logsumexp/softmax reductions. Some specialized kernels still evaluate internally in `f64`; sequence and wave APIs retain documented precision limits. See the [precision coverage guide](../../function_coverages/GENERIC_FLOAT_COVERAGE.md).
- Expand `complex` with generic gamma, log-gamma, reciprocal gamma, digamma, scaled/complementary error functions, Dawson's integral, Lambert W branches, exponential integrals, Spence's integral, logarithmic products, and bounded Riemann/Hurwitz zeta evaluation. Branch cuts and convergence limits are documented per function.
- Accept supported real incomplete elliptic cases and Cauchy principal values, including applicable `m >= 1` inputs. Elliptic wrappers still require finite inputs and results; incomplete third-kind arguments are `(phi, n, m)`.

### Fixed

- Preserve representable subnormal normal-CDF and complementary-error-function tails.
- Improve small-tail CDF accuracy for noncentral chi-square, F, and Student-t distributions.
- Prevent `digammainv` failures caused by Newton iteration cycling at the rounding limit.
- Improve `gamma_ratio(a, b)` accuracy for nearby positive arguments at least 8.
- Preserve the lower branch of scaled complex Airy functions for negative-real inputs with imaginary `-0.0`.
- Avoid `ellipj(u, 1)` overflow for large finite `u` and correct near-unit evaluation outside the perturbation range.
- Preserve tiny inverse-beta probabilities and quantiles, subnormal Bessel results, and finite modified-Bessel values near overflow.
- Improve hypergeometric evaluation for terminating series, negative-argument cancellation, small arguments, and near-integer Tricomi U parameters.
- Correct Airy/Struve integral transitions, K₀ integral tails, negative modified-Fresnel reflection, oblate radial indexing and derivatives, Legendre singular endpoints, and Bessel signed-zero conventions.

### Improvements

- Use published numerical backends `bessel` 0.1.3, `hypergeometric` 0.1.4, `orthogonal` 0.2.2, and `ellip` 1.2.4. Improve Lambert W near zero through the updated `lambert_w` backend, retaining branch-point safeguards.
- Speed up real modified Bessel functions, Bessel derivatives, hypergeometric functions, polynomial evaluation, elliptic integrals, and Gaussian quadrature.
- Speed up Mathieu and Lamé calculations, noncentral-t inversion, exponential integrals, normal quantiles, and softmax/log-softmax reductions with less temporary storage.
- Avoid dispatched fused multiply-add calls in moderate positive beta/log-beta, precompute positive exponential-integral series ratios, and reduce redundant Sturm recurrence work and workspace allocations in Mathieu and spheroidal calculations.
- Replace dense Gaussian quadrature and Lamé eigenproblems with bounded symmetric tridiagonal solvers, removing the direct `nalgebra` dependency.
- Improve near-endpoint Chebyshev accuracy. Numerical accuracy and supported-domain limits remain documented in the [special-function coverage guide](../../function_coverages/SPECIAL_COVERAGE.md).

See the [performance comparisons](../../function_coverages/SPECIAL_PERFORMANCE_COMPARISON.md) for benchmark conditions and results.

## 0.2.0

- Upgrade to shared `sci-core 0.2` types. Update cooperating Sci crates together; `sci-core 0.1` types are incompatible.
- Improve Student-t `stdtr` and `stdtrit` accuracy near the median and use stable elementary formulas for one and two degrees of freedom.
- Use the private XSF Box–Cox kernels to preserve finite results for tiny parameters and large powers, retaining endpoint conventions.

## 0.1.0

- Initial release of special mathematical functions.
