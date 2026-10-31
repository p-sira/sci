# Dependency functions beyond the SciPy inventory

This audit compares the installed dependency versions with `Cargo.toml` with Sci's public
functional APIs. `SCIPY_FUNCTIONS.md` and its linked special-function inventory
contain no harmonic-number or logarithmic combinatorial utility names. There
is no SciPy complex arithmetic namespace; NumPy supplies complex references.
SciPy aliases and Cython duplicates remain covered by existing module audits.

## Implemented additions

| Dependency capability | Sci API | Adaptation and independent reference |
| --- | --- | --- |
| Statrs `harmonic` | `special::combinatorics::harmonic(n)` | Reuses Sci digamma; H_0 = 0 corrects the backend's H_0 = 1 convention; SciPy digamma reference |
| Statrs `gen_harmonic` | `special::combinatorics::gen_harmonic(n, m)` | Finite exponents, H_0 = 0, compensated sum, at most 1,000,000 terms; NumPy extended-precision defining sum |
| Statrs `ln_factorial` | `special::combinatorics::ln_factorial(n)` | Reuses Sci log-gamma; SciPy `gammaln` reference |
| Statrs `ln_binomial` | `special::combinatorics::ln_binomial(n, k)` | Invalid selections return an error; compensated log-product avoids log-factorial cancellation; exact Python `math.comb` reference |
| Statrs `multinomial` / `checked_multinomial` | `special::combinatorics::multinomial(n, counts)` | Checked total without integer overflow; explicit work and numerical overflow errors; exact Python integer reference |
| Logarithm of the preceding coefficient | `special::combinatorics::ln_multinomial(n, counts)` | Shared sequential log-binomial factors avoid ordinary-coefficient overflow; exact Python integer logarithm reference |
| num-complex `powu` | `complex::powu(z, exponent)` | Full `u32` exponent range; reuses Sci's robust multiplication; NumPy powers and exact roots-of-unity reference |
| num-complex `cbrt` | `complex::cbrt(z)` | Principal complex root; signed-zero branch side; scaled finite off-axis magnitude; NumPy extended-precision principal power reference |

Every addition is exported and exercised through `sci` in
`tests/dependency_extras.rs`. Cases are kept separately in
`tests/scipy/dependency_extras_cases.jsonl`; the version-pinned generator writes
`tests/fixtures/dependency_extras.csv`. Tests include invalid domains, aggregate
work limits, empty sums/products, large integer indices, overflow, tiny and
huge complex components, signed zero, NaN/infinity, recurrence, symmetry,
permutation, and composition with shared complex arithmetic.

No new dependency is required. Trivial sums/products use native arithmetic;
complex algorithms reuse existing private backends and Sci functions.
Binary64 coefficients are approximate rather than exact integer arithmetic.

## Existing coverage and remaining candidates

Ellip 1.2.2's **24 stable numerical functions** are exported from
`special::elliptic`, with public-API CSV comparisons in
`tests/special_extended.rs`. This includes `ellipke`, all seven Bulirsch
functions (`cel`, `cel1`, `cel2`, `cel3`, `el1`, `el2`, `el3`), the additional
Legendre integrals, `ellippiinc_bulirsch`, Jacobi zeta, and Heuman lambda.
`ellipf` also has SciPy's `ellipkinc` alias; `ellipkm1` uses the Carlson identity.
There are no missing stable Ellip numerical names. Unstable precision-control
and unchecked backend APIs, test assertions and backend types stay private.
Finite-input/result restrictions and singularities are documented separately;
supported real principal-value branches are available through the Ellip-style
facade.

The other dependencies provide existing SciPy operations or specialized backend
interfaces: special/XSF special functions, RustFFT/RealFFT/RustDCT transforms,
Statrs/Anofox statistical operations, Kodama linkage, LapJV assignment,
Delaunator geometry, splinefit fitting, pm-remez filter design, Basin solvers,
Smelly estimation, and faer/nalgebra/ndarray linear algebra.
Their algorithm types and storage/adapter APIs stay private. Existing coverage
reports describe supported numerical signatures; this audit does not assert
complete parity with every dependency method or optional feature.

Remaining candidates include num-complex arbitrary-base logarithms and
exponentials and its convenience base-2/base-10 functions, Basin's additional
solver strategies, and backend-specific fitting/statistical operations.
Those need separate API and reference-test work. Optional backend settings and
synonyms for existing Sci operations are not counted as new numerical coverage.
