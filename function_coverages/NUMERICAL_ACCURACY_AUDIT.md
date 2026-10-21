# Numerical accuracy audit, October 2026

Reviewed the sibling `../sci-testing/docs/accuracy/` reports and
`../sci-testing/results/measurements/` for Sci commit
`b52c598535920279dc0aadb14a9085b8545ba4c1`.
The non-sparse numerical implementations in that snapshot match the `dev` base
of this patch. Sparse APIs and their adapters are excluded throughout.

The inventory contains **1,097 non-sparse exported names, representing 1,016
implementations across 14 modules**. The measurements contain 2,064,437 cases
with no Sci evaluation failures. Aliases, compatibility namespaces and Cython
duplicates are dispositions of existing implementations, rather than new work.
The inventory and linked module lists in `SCIPY_FUNCTIONS.md` were compared with
the exported APIs and public fixture dispatchers. This patch adds no exported
names or new compatibility claims.

Both absolute differences and differences normalized by the reference output's
maximum absolute component were inspected. Epsilon-scaled componentwise relative
differences alone substantially overstate discrepancies around zeros and in
nonunique factors. The checked-in independent references remain the accuracy
oracle; Sci's measured outputs never generate expected results.

## Changes and rechecks

| API | Cause and correction | Maximum normalized error before → after |
| --- | --- | --- |
| `linalg::solve` | LU rounding on ill-conditioned Hilbert matrices; at most four refinement updates, with compensated product residuals and reused faer factors | `5.44e-8` → `1.10e-16` |
| `optimize::rosen` | Rounded squares destroy the small valley residual; evaluate that residual with fused multiply-add | `8.40e-9` → `4.94e-16` |
| `optimize::rosen_der` | Reuse the same fused residual in the analytical gradient | `2.00e-6` → `7.55e-16` |
| `special::stdtr`, `special::stdtrit`, Student-t `stats::{cdf,sf,ppf}` | Incomplete-beta complements lose central deviations; switch beta branches near the median, reuse the shared functions, and use stable elementary formulas for df=1 and df=2 | Student-t quantiles: `1.02e-9` → `5.48e-13` |
| `stats::shapiro` | Computing `acos(sqrt(W))` amplifies rounded W near one for n=3; use the exact gap ratio and `atan` | Independent near-symmetric cases agree within `2e-15` absolute + `2e-14` relative |
| `integrate::newton_cotes` | Uniform-rule Vandermonde solutions and moment subtraction introduce avoidable error; use exact rational tabulations for all supported uniform orders | All 12 uniform orders agree with SciPy's tabulations within binary64 rounding |

The normalized error above is `max_i |actual_i-reference_i| / max_i |reference_i|`
for each output vector, then maximized across cases. Rechecks used 14,391 original
harness cases for the ten directly affected APIs, with no evaluation failures.
Only Student-t rows are included for the generic distribution APIs. Uniform
Newton–Cotes rules and extreme Student-t cases have additional focused fixtures.

The original Shapiro recheck still differs from SciPy by up to `2.85e-8` in
p-value. For `shapiro/linspace-0004`, SciPy reports `0.9999999715408785`, while
the stable formula gives `0.9999999999999999`. High-precision evaluation of the
exact three-point distribution resolves this reference rounding artifact.

## Module disposition

| Modules | Findings |
| --- | --- |
| cluster, constants, filter | Measured differences are consistent with binary64 reductions and existing backend accuracy. No implementation correction identified. |
| complex | Large reported relative differences include cancellation residuals preserved by Sci and inaccurate reference powers: `i^2147483647` is exactly `-i`; reciprocal square roots of finite huge complex inputs need not vanish. Preserve the existing scaled arithmetic. |
| differentiate | Fixed second-order stencils have documented truncation and roundoff error; the `h=0.01` sine derivative follows its stated secant definition. Adaptive SciPy derivatives have different work and accuracy contracts. |
| fft, signal | Large componentwise relative differences occur around transform zeros, removed trends and cancellation residuals. Output-vector errors are small; no shape, normalization or evaluation failure was found. |
| integrate | Uniform Newton–Cotes corrections above; nonuniform rules retain the documented conditioning and subtracted-moment limitations. |
| interpolate, spatial | Differences around zero spline integrals, exact self-distances and geometric reconstruction residuals are small in absolute value. Nonunique representations and fitted coefficients need the existing structural tests. |
| linalg | Solve correction above. Nonunique factorization signs, projectors and reconstruction zeros require structural rather than individual-factor comparisons. |
| optimize | Rosenbrock corrections above. Reported minimizer locations and finite-difference helpers satisfy their documented tolerances and fixed-step conventions. |
| special | Student-t corrections above. Exact polynomial recurrences preserve parity zeros that SciPy's coefficient construction perturbs. Noncentral-F parameter inverses agree with independent `scipy.stats.ncf` inversion; the `scipy.special` CDFLIB inverses are less accurate on these samples. |
| stats | Student-t and three-point Shapiro corrections above. The zero-skewness convention is deliberately continuous, unlike SciPy's Y=1 substitution. MGC optimal-scale ties can move under last-bit rounding. Transform optimizers have flat objectives; Dunnett uses bounded quadrature with its documented error criterion. |

Remaining limitations include ill-conditioned dense systems, clustered
nonuniform quadrature nodes, fixed-step differentiation, backend incomplete-beta
accuracy at large degrees of freedom, and relative accuracy of very small
noncentral-t probabilities obtained by subtraction. These samples establish no
whole-domain error bound and do not expand any module's supported API surface.

## Reproduction and regression coverage

`tests/scipy/numerical_accuracy_cases.json` stores inputs separately from
expected values. Regenerate the 188 references with:

```sh
uv run tests/scipy/generate_numerical_accuracy.py
CARGO_BUILD_JOBS=1 cargo test --test numerical_accuracy
```

The generator pins NumPy 2.5.4, SciPy 1.18.1 and mpmath 1.3.0. References use
SciPy Newton–Cotes rules, 100-digit evaluations of exact binary64 Rosenbrock
inputs and linear systems, density quadrature/inversion and high-precision beta tails for Student-t, and the
exact three-point Shapiro distribution. Extreme df=1 and df=2 tails use independent
high-precision closed forms. Rust tests exercise every changed function through
the public Sci facade, including shared-module agreement, inverse round trips,
shapes, quadrature symmetry, NaN/infinity, invalid domains and exact medians.
Comparisons are componentwise; tiny nonzero references have no absolute
tolerance that would allow zero to pass. Large-shape incomplete-beta accuracy
is explicitly allowed `2e-12` relative tolerance against the density integral.

The sibling testing checkout contains uncommitted work. Rechecks used a
temporary copy of its Rust adapters with only the sparse adapters removed;
its sources, revision pin, results and reports were left unchanged. Builds and
the final formatting, test, Clippy and rustdoc checks run sequentially with
`CARGO_BUILD_JOBS=1`.
