# Special-function coverage and numerical limits

Audited against [the SciPy 1.18.1 inventory](scipy_function_list/scipy.special.md).
Run `python tests/scipy/audit_special.py` to reproduce the namespace/name audit.
All **353 existing public numerical functions** are retained. The native API
follow-up adds **eight scalar and 22 qualified complex APIs**; see
[SPECIAL_NATIVE_COVERAGE.md](SPECIAL_NATIVE_COVERAGE.md) for the XSF source
audit, Rust adaptations, deliberate exclusions and remaining numerical gaps.
No numerical
function family was removed in the pure Rust backend migration.

## Inventory disposition

| Namespace | Inventory names | Direct functions with public CSV tests | Aliases with tested replacements | Native Rust replacements |
| --- | ---: | ---: | ---: | ---: |
| `scipy.special` | 355 | 330 | 17 | 8 |
| `scipy.special.cython_special` | 234 | 227 | 1 | 6 |
| `scipy.special.basic` | 65 | 63 | 2 | 0 |
| `scipy.special.orthogonal` | 46 | 31 | 15 | 0 |
| `scipy.special.specfun` | 2 | 2 | 0 | 0 |
| `scipy.special.spfun_stats` | 1 | 1 | 0 | 0 |

`add_newdocs` and `sf_error` contain no numerical function entries. These
703 qualified inventory names repeat algorithms across namespaces; they do not
represent 703 distinct implementations. Cython-only declarations and deprecated
compatibility namespaces introduce no additional numerical name gaps.

The 25 top-level names without identically named Sci exports are intentional:

- `psi` uses `digamma`; `assoc_laguerre(x,n,k)` uses
  `eval_genlaguerre(n,k,x)`.
- The 15 legacy aliases `c_roots`, `cg_roots`, `h_roots`, `he_roots`, `j_roots`,
  `js_roots`, `l_roots`, `la_roots`, `p_roots`, `ps_roots`, `s_roots`, `t_roots`,
  `ts_roots`, `u_roots`, and `us_roots` use the corresponding `roots_*` APIs.
- `cbrt`, `exp2`, `expm1`, and `log1p` use float methods `cbrt`, `exp2`,
  `exp_m1`, and `ln_1p`. `exp10(x)` uses `10.0_f64.powf(x)`.
  SciPy's ties-to-even `round` uses `round_ties_even` on concrete Rust floats;
  Rust's `round` has a different tie convention.
- Python's process-wide `geterr` and `seterr` have no Rust equivalents.
  Sci exposes documented IEEE scalar values and shared `Result` errors.

These are API substitutions, not missing numerical algorithms. The six Sci
combinatorial helpers outside the SciPy inventory retain their independent
references and property checks in the existing numerical suites.

## Remaining signature and precision gaps

SciPy supplies an independent reference inventory; 100% API, signature, domain,
or special-value parity is not required. Name coverage does **not** imply all
SciPy overloads or domains are implemented.
The remaining gaps include NumPy broadcasting/output buffers, additional complex
input families beyond the 19 APIs in `special::complex`, exact-integer modes, noninteger polynomial degrees, and
Legendre/harmonic derivative-axis outputs. Sequences retain their documented
Rust shapes and ordering. Python polynomial classes and error-state context
managers are not reproduced.

The migration makes **104 scalar and polynomial APIs** generic over
`sci_core::Float`, in addition to the previously generic elliptic integrals and
stable reductions. The newly generic APIs have independent `f32` and `f64`
fixtures. Most specialized kernels promote to binary64 internally and convert
back; they do not implement arbitrary precision. Real wave functions, complex
results outside the new generic namespace, sequence adapters, and some elementary/combinatorial APIs still use
`f64`. Elliptic integrals continue to compute in the selected float type.

## Work limits and accuracy

Existing order/count limits continue to apply. Additional explicit bounds prevent
unbounded recurrences in the translated families:

| Family | Supported work and remaining limits |
| --- | --- |
| Mathieu | Order at most 128 and `abs(q) <= 100`; Fourier recurrences at most 251 terms; Sturm eigenvalue bisection at most 128 steps and four tridiagonal inverse iterations. Angular inputs are degrees and returned derivatives are with respect to radians, following SciPy. |
| Spheroidal | `0 <= m <= n <= 128`, finite `abs(c) <= 64`; fixed private recurrence storage; eigenvalue bisection at most 128 iterations per value. Parameter sign is reduced by the defining `c²` convention. |
| Parabolic cylinder | D/V require finite inputs and `abs(v) <= 4096`; W requires `abs(a), abs(x) <= 5`. Values and first derivatives are returned in that order. |
| Hypergeometric | 1F1 requires `abs(a) <= 100000`; 2F1 bounds `abs(a), abs(b), abs(c)` and U bounds `abs(a), abs(b)` by 100,000. All inputs must be finite. Series/asymptotic and continuation accuracy is parameter dependent; severe cancellation can return NaN or lose relative accuracy. |
| Noninteger Legendre P | Finite `abs(v) <= 100000`, `abs(m) <= 4096`, and `x` in `[-1,1]`. |
| Wright Bessel | Nonnegative finite parameters; positive defining series in logarithmic form, at most 100,000 terms. Failure to converge returns NaN. Very large arguments can exceed this budget. |
| Kelvin zeros | Shared scaled Bessel identities, at most 32 bracketing steps and 64 safeguarded Newton iterations per root; count at most 4096. Failure returns a shared numerical error. |
| Fresnel zeros | At most 64 Newton iterations per root, count at most 4096. Failed/nonfinite refinement returns an error. |
| Airy/Struve integrals | Bounded Gaussian quadrature near the series/asymptotic transitions; translated asymptotic kernels beyond those ranges. Large oscillatory arguments remain limited by argument reduction and asymptotic truncation. |

The parameter work bounds above are not uniform accuracy guarantees. In
particular, the general 1F1 fallback still has severe cancellation for large
positive `a` at negative `x` (for example, `a=65.1, b=-3.2, x=-20`). These
parameters require a stronger continuation algorithm before they can be counted
as accurately supported. The checked reference grid for the improved direct
series covers `abs(a), abs(b) <= 64` and `abs(x) <= 100`, with representative
large-argument asymptotic cases outside that region.

Nonfinite conventions also differ from some SciPy overloads: incomplete beta
and lower gamma require finite shape parameters; general Bessel/Struve orders
and 0F1 parameters must be finite. Invalid infinite parameters return NaN.
Polynomial evaluators can return their leading-term infinite limits where
SciPy's arithmetic returns NaN. I0/I1 at infinity return their mathematical
infinite limits. `comb(NaN,k)` propagates NaN and the real lower Lambert W branch
rejects infinities. These contracts have separate public tests; SciPy CSV
comparisons cover shared domains and matching special-value conventions.

The infallible scalar facade maps backend domain/convergence errors to NaN.
Ordinary real overflow and poles can return signed infinities. Complex AMOS
errors can produce NaN components. Accuracy near zeros, singularities, and large
orders is not certified by the representative fixture suite.

## Independent validation and corrected references

`tests/scipy/generate_special.py` and `generate_special_float.py` pin
**SciPy 1.18.1 / NumPy 2.5.4**. Inputs are separate JSON files; ordinary Rust tests
read committed CSVs. Float references quantize inputs before calling SciPy and
round expected outputs to the selected precision. No expected values come from
Sci or its Rust dependencies.

The reference generator documents SciPy limitations rather than weakening
comparison tolerances:

- Airy-integral legacy series termination loses about `4e-10` at `x=0.5`;
  independent quadrature supplies the reference.
- The large-argument Struve integral in SciPy 1.18.1 disagrees with quadrature
  (at `x=32`, approximately `3.4414` versus `2.8779`). The defining integral is
  the reference.
- Negative modified Fresnel inputs overwrite an intermediate component in
  SciPy's legacy branch. Reflection of the positive-input integral supplies
  the reference and preserves complex conjugacy of the two sign conventions.

The existing independent scaled-Bessel reference for `keip_zeros` remains.
Expanded fixtures additionally verify Kelvin values through complex Bessel identities,
modified Struve values through defining integrals, Hankel derivatives through
independent J/Y derivatives, and oblate radial functions through ODE continuation.
The latter caught two indexing errors inherited from the C++ translation; the
original Fortran indexing and the radial Wronskian independently verify the fixes.
The generator also replaces erroneous scaled-Y overflow signs and checks
modified-Bessel integrals and the hypergeometric U failure case with quadrature.

Additional checks cover parabolic-cylinder recurrences, modified-Fresnel
reflection, zero-parameter spheroidal/Legendre agreement, inverse distributions,
zero ordering, shared-type interoperability, and invalid inputs/work limits.

## Backend and build

Private Rust kernels supply the Cephes, Faddeeva, hypergeometric, integral, and
wave algorithms. `complex-bessel` supplies complex Bessel/Airy/Hankel kernels;
`ellip`, `lambert_w`, and the existing `nalgebra` polynomial solver are reused. Third-party
notices are packaged under `crates/sci-special/LICENSES`.

The final two signal-filter calls to scaled complex K also use `complex-bessel`.
The workspace no longer depends on `xsf`, `bindgen`, or `clang-sys`, and no C++
compiler or libclang is required for these special-function kernels.

## Executable line coverage

A clean `cargo llvm-cov --workspace --all-features --no-report` run, followed
by LCOV reporting with `--ignore-filename-regex '(^|/)tests/'`, measures the
following at commit `39ffe30`, before the documentation/dependency follow-up.
All tests passed; the filter excludes test files
and does not exclude numerical backends.

| Scope | Covered lines | Executable lines | Coverage |
| --- | ---: | ---: | ---: |
| Workspace | 40,611 | 43,516 | 93.32% |
| `sci-special` | 16,724 | 18,370 | 91.04% |

The 99% aim is not reached. Reaching it with this source layout requires at
least 2,470 additional workspace lines to be exercised, or 1,463 lines for
`sci-special` itself to reach 99%. These counts precede the documentation
follow-up. The largest special-function gaps are remaining continuation
and exceptional branches in `specfun.rs` (468 lines), Faddeeva approximation
and exceptional paths (193 lines), and general-order Bessel J (71 lines).
Validation/error and rare precision-failure paths also remain. These are
executable-line gaps, distinct from the numerical-function inventory above.

Codecov's patch metric has a different denominator: on `f4c7df4` it reports
87.41%, improved from 68.91%, against the unchanged 94.90% gate. The later
transformation fixtures require a fresh CI upload; neither this gate nor
CodSpeed's performance gate is claimed to be resolved.

## Elliptic conventions and dependency follow-up

Elliptic integrals follow Ellip's notation and argument order: `F(phi, m)` is
`ellipf`, with `ellipkinc` retained as an alias; `Π(phi, n, m)` takes amplitude,
characteristic, then parameter. `m = k²`, `kc² = 1−m`, and Bulirsch's `p = 1−n`
are distinct conventions. Amplitudes use radians; Bulirsch's `x` is a tangent.
The wrappers admit supported finite real `m >= 1` incomplete cases and Cauchy
principal values (negative `y` for `R_C`, negative `p` for `R_J` and supported
Bulirsch cases, and supported third-kind characteristics). Finite inputs and
finite results remain Sci restrictions; backend domain failures are shared errors.
SciPy fixtures cover shared domains; independent defining integrals and identities
cover extensions. Full SciPy parity is not a completion criterion.

`lambert_w` is updated from 2.0.5 to 2.0.8. Its upstream near-zero expansion
replaces Sci's Taylor workaround. The branch-point Puiseux correction remains:
at the next binary64 value above the rounded `−1/e`, upstream's principal
branch differs from a 90-digit defining-equation reference by about `2e-9`.
Standard-library Decimal bisection independently supplies new checked-in
references for both branches in this region; it uses no Sci/backend coefficients.
Signed-zero/domain checks and inverse identities remain. Both public branches
still evaluate in binary64 before conversion to `T`.
