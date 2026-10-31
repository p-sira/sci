# Special-function coverage and numerical limits

Audited against [the SciPy 1.18.1 inventory](scipy_function_list/scipy.special.md).
Run `python tests/scipy/audit_special.py` to reproduce the namespace/name audit.
All **353 existing public numerical functions** are retained. The native API
follow-up adds **eight scalar and 25 qualified complex APIs**; see
[SPECIAL_NATIVE_COVERAGE.md](SPECIAL_NATIVE_COVERAGE.md) for the XSF source
audit, Rust adaptations, deliberate exclusions and remaining numerical gaps.
No numerical
function family was removed in the pure Rust backend migration.
In the unreleased Sci 0.3.0, all 362 formerly root-exported numerical functions
are assigned to mathematical-family modules, including `elliptic::ellipj`.
The inventory counts above refer to algorithms, not root-level Rust exports. See
[the migration guide](../MIGRATING.md#special-function-family-modules-breaking)
for paths. `sci::special` has no root-level numerical functions or aliases;
only shared `Error` and `Result` remain as root re-exports.

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

The 25 SciPy inventory names without identically named Sci functions are intentional:

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
input families beyond the 25 APIs in `special::complex`, exact-integer modes, noninteger polynomial degrees, and
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
| Real `iv` / `ive` | Cephes Temme series and continued fractions for `abs(v) <= 50` (999 iterations each), eleven-term uniform asymptotics for larger orders, and a 100-term large-argument expansion. Internal exponential scaling and recurrence rescaling preserve finite overflow-adjacent and subnormal results; negative noninteger orders use reflection. Complex APIs retain AMOS. |
| Mathieu | Order at most 128 and `abs(q) <= 100`; Fourier recurrences at most 251 terms, with adaptive coefficient tails. Isolated bands use a bounded scalar Schur-complement solve; other bands retain at most 128 Sturm bisections. Fourier coefficients use two tridiagonal inverse iterations. Angular inputs are degrees and returned derivatives are with respect to radians, following SciPy. |
| Spheroidal | `0 <= m <= n <= 128`, finite `abs(c) <= 64`; fixed private recurrence storage; eigenvalue bisection at most 128 iterations per value. Parameter sign is reduced by the defining `c²` convention. |
| Parabolic cylinder | D/V require finite inputs and `abs(v) <= 4096`; W requires `abs(a), abs(x) <= 5`. Values and first derivatives are returned in that order. |
| Hypergeometric | 1F1 requires `abs(a) <= 100000`; 2F1 bounds `abs(a), abs(b), abs(c)` and U bounds `abs(a), abs(b)` by 100,000. All inputs must be finite. Series/asymptotic and continuation accuracy is parameter dependent; severe cancellation can return NaN or lose relative accuracy. |
| Noninteger Legendre P | Finite `abs(v) <= 100000`, `abs(m) <= 4096`, and `x` in `[-1,1]`. |
| Wright Bessel | Nonnegative finite parameters; positive defining series in logarithmic form, at most 100,000 terms. Failure to converge returns NaN. Very large arguments can exceed this budget. |
| Kelvin zeros | Shared scaled Bessel identities, at most 32 bracketing steps and 64 safeguarded Newton iterations per root; count at most 4096. Failure returns a shared numerical error. |
| Fresnel zeros | At most 64 Newton iterations per root, count at most 4096. Failed/nonfinite refinement returns an error. |
| Airy/Struve integrals | Bounded Gaussian quadrature near the series/asymptotic transitions; translated asymptotic kernels beyond those ranges. Large oscillatory arguments remain limited by argument reduction and asymptotic truncation. |

The parameter work bounds above are not uniform accuracy guarantees. Real 1F1
now uses double-double outward Taylor continuation for `a > 0`, `b < a`,
`-8 <= b <= 64` and `-1000 <= x < 0`. This fixes the former catastrophic
cancellation at `a=65.1, b=-3.2, x=-20`. An integer `a-b` uses a scaled,
normalized Laguerre recurrence to avoid propagating a recessive ODE solution.
Continuation permits 256 origin terms and 8192 steps of 128 terms; nonconvergence,
nonfinite continuation intermediates and budget exhaustion return NaN. The
integer-difference recurrence takes at most 100008 steps. More negative `b`,
other parameter regions and near-zero/pole conditioning remain accuracy gaps.
See [the investigation](HYP1F1_CANCELLATION.md) for the failure mechanism,
independent references and tested limits. Existing direct-series fixtures and
representative large-argument asymptotic cases are retained.

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

## Coverage-driven numerical regressions

The coverage follow-up adds **557 scalar reference rows** and **542 f32/f64
native reference rows**, plus public recurrence, tail round-trip and derivative
tests. The scalar fixture now has 5165 rows; the native fixture has 1976.
These exercise noninteger Legendre continuation, large-order Bessel transitions,
all reachable error-function Chebyshev intervals, complex Fresnel axes and nearby
points, integral asymptotics, and extreme beta/gamma inverse probabilities.
Inputs remain separate from generated results, with NumPy 2.5.4 / SciPy 1.18.1
pinned. Tests do not call Python or private backends at runtime.

The stricter comparisons exposed and corrected these failures:

- Beta inverse reflection discarded probabilities when `1-p` rounded to one.
  `betaincinv(1000,0.1,1e-20)` formerly returned about `1.11e-16` instead of
  `0.960342172976`. Upper-tail refinement also retains small output coordinates;
  analytic `I_x(a,1)=x^a` tests prevent reflection from erasing tiny quantiles.
- Tricomi U's `b=0` dispatcher could return an initialized zero or a badly
  cancelled finite answer. It now reuses `U(a,0,x)=x U(a+1,2,x)` when the
  intermediate values are finite, with a bounded contiguous recurrence near
  `a=1,b=2,0<x<1`. Defining Laplace quadrature replaces incorrect SciPy
  references in that near-integer region.
- Bessel J's high-order seed could underflow before a recurrence multiplier
  restored it. Multipliers now enter before exponentiation, and a shorter
  recurrence replaces overflowing normalization to J₀/J₁. For example,
  `J_499.5(120)` is about `2.00954e-248`, and `J_499(88)` is a representable
  subnormal near `1.01693e-313`. A 100-digit Decimal defining series with exact
  integer factorials supplies independent references where AMOS flushes results.
- The K₀ tail from `it2i0k0(19.9)` had about `2e-6` relative error, previously
  hidden by an absolute tolerance. For `4<=x<6`, a cached Gaussian rule integrates
  the shifted scaled kernel over `[0,48]`; the omitted tail is below `1e-20`
  relatively. Shorter Gauss-Laguerre rules evaluate the same tail for `x>=6`.
  Boundary and derivative tests check the transitions and defining integrand.

New cases use relative comparisons with a `1e-300` absolute floor, reduced to
`1e-323` for explicit subnormal Bessel regressions. Small beta shapes amplify
CDF normalization errors by roughly `1/a`: at `a=.001,b=1000,p=.6`, a forward
CDF discrepancy around `1.3e-13` gives about `2.2e-10` relative quantile error.
Those cases allow `3e-12/a`, rather than hiding nonzero outputs with a large
absolute tolerance. U cases at `x=10` retain a `2e-9` relative allowance because
the legacy integral dispatch supplies about nine digits. These are remaining
accuracy limits, not uniform machine-precision claims.

The follow-up also removes 22 constant-false validation blocks (`if !(true)`)
from polynomial constructors without parameter-domain checks. Degree and
numerical validation remain in the shared helpers; no coverage filters were
added for numerical kernels.

References: [Bessel defining series](https://dlmf.nist.gov/10.2.E2),
[U transformation](https://dlmf.nist.gov/13.2.E40),
[U recurrence](https://dlmf.nist.gov/13.3.E7), and
[U defining integral](https://dlmf.nist.gov/13.4.E4).

## Executable line coverage

The current measurement starts with `cargo llvm-cov clean --workspace`, then
runs:

```sh
cargo llvm-cov --workspace --all-features --locked --lcov \
  --output-path /tmp/sci-coverage.info --ignore-filename-regex '(^|/)tests/'
```

All instrumented tests pass. The file filter matches CI and excludes test files;
numerical backends remain included. These are local LLVM executable-line totals,
not a claim about a completed remote Codecov upload or its patch metric.

| Run and scope | Covered lines | Executable lines | Coverage |
| --- | ---: | ---: | ---: |
| Initial local run at `d21d103`: workspace | 40,092 | 42,757 | 93.77% |
| Initial local run at `d21d103`: `sci-special` | 17,222 | 18,807 | 91.57% |
| Coverage follow-up including CodSpeed `a620025`: workspace | 40,575 | 43,063 | 94.22% |
| Coverage follow-up including CodSpeed `a620025`: `sci-special` | 17,705 | 19,113 | 92.63% |

The denominator changed because this follow-up incorporates CodSpeed's new
performance implementations, numerical corrections, and removal of unreachable
constant-false checks. The change in percentages is not attributable solely to
added tests.

The **99% aim remains unmet**. There are **2,488 uncovered workspace lines** and
**1,408 uncovered `sci-special` lines**. With the current executable-line counts,
99% requires exercising at least **2,058 additional workspace lines**, or
**1,217 additional special-function lines** if targeting that crate alone.

| Largest remaining special-function source gaps | Uncovered lines |
| --- | ---: |
| `backend/specfun.rs`: continuation, recurrence and exceptional paths | 426 |
| Faddeeva complex error functions and auxiliary approximation paths | 155 |
| `backend/legacy.rs`: dispatch, validation and integral paths | 67 |
| Cephes general-order Bessel J | 64 |
| Struve algorithms | 48 |
| Orthogonal polynomial kernels | 38 |
| Beta kernels | 33 |
| Spherical Bessel kernels | 33 |
| Inverse incomplete beta | 31 |

Other gaps include exceptional evaluation/convergence paths, argument/domain
boundaries, and recently added performance branches. Some internal precision
options and defensive branches may not be reachable through current public
APIs; they need a reachability audit before being classified as removable.
These executable-line gaps are distinct from missing numerical API/domain
coverage. Neither coverage thresholds nor benchmark baselines were changed;
remote Codecov and CodSpeed gate results require fresh CI analysis.

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

Real `iv`/`ive` regression fixtures (`bessel_iv.csv`) cover both public float
APIs, the Temme/uniform and series/continued-fraction boundaries, negative
orders and arguments, overflow, and subnormals. The pinned generator uses
SciPy, with the independent defining series and positive-order reflection
where SciPy overflows intermediate K values or AMOS loses reflection accuracy.
The binary64 tolerance is 5e-12 relative plus two subnormal ULPs; binary32
uses 3e-6 relative plus two subnormal ULPs. Recurrence, parity, symmetry,
scaling, and large-argument limits are checked separately. No new inventory
names or aliases are introduced; Python/Cython `iv`/`ive` share these real APIs.

## Modified Mathieu performance validation

All four modified Mathieu APIs share the Bessel-product kernel. It now computes
only J and J′ at the first argument, J and J′ at the second argument for kind
one, and J, Y, Y′ at the second argument for kind two. Unused outputs require
no buffers or recurrences. The Miller recurrence skips its preliminary
underflow-order search only when the existing envelope at one extra order
lies below the `1e-200` boundary; other inputs retain that search. Recurrence
normalization, coefficient truncation, and summation tolerances are unchanged.

The pinned `generate_special.py` generator adds 190 independent SciPy cases
for `mathieu_modcem1`, `mathieu_modcem2`, `mathieu_modsem1`, and
`mathieu_modsem2`, including both order parities, the origin, `q = 100`, and
large positive/negative coordinates that exercise the Bessel asymptotic
branch. Values and coordinate derivatives retain the existing `2e-11`
absolute/relative fixture tolerances. Public API property tests additionally
check the `2/pi` Wronskian (DLMF 28.22) and first-kind parity, including the
sign of the derivative. The Wronskian allows `2e-10` absolute error because
at order six and small `q`, cancellation before division by the first Fourier
coefficient produces residuals near `7e-11`; SciPy has the same limitation
(residuals near `4e-10` for the even pair). Existing tests cover invalid
domains and work limits.
The Python/Cython inventory names continue to share these four Rust APIs;
this optimization introduces no new names or domain restrictions.

## Remaining-regression recurrence and precision work

The integer parabolic-cylinder sequence now reuses the scalar kernel's full
sequence in one call. Moderate associated Legendre tables use the established
unnormalized degree recurrence and factorial-product normalization; the
logarithmically scaled path remains for high orders and near the poles.
Kolmogorov survival probabilities evaluate their tail directly, retaining the
logarithmic machinery for inverses. Splitting the exponential in subnormal
tails avoids premature underflow and cancellation from adding `log(2)` to a
large exponent. Tail fixtures allow two smallest-subnormal ULPs, with relative
tolerance `2e-13` elsewhere.

Bounded double-double series share Kelvin's exact small-integer product and
quotient helpers and its pre-split fixed multiplier. Airy and Struve integrals
retain double-double cancellation control and their existing work limits.
Positive-parameter, positive-argument 0F1 series use compensated binary64 sums
only when a recurrence-roundoff and geometric-tail bound is below 16 binary64
epsilons; other cases retain the double-double or Bessel paths. Lambda
sequences keep suitable series anchors in double-double, preserving phase
through the backward recurrence before rounding the output vectors.

The pinned reference generator adds 220 cases: 30 integer-cylinder sequences,
48 Legendre tables, 16 Kolmogorov probabilities, 70 Lambda sequences, and 56
positive 0F1 cases. It checks output shapes and ordering as well as values.
Independent Hermite identities and elementary half-order Lambda identities
exercise the public APIs. SciPy's Lambda sequence can lose about `4e-9` at
`v=256,x=16`, and its 0F1 gamma/Bessel normalization loses about `3e-14` at
`v=100,z=0.5` or overflows above `v=172`. The generator therefore uses a bounded
100-digit Decimal defining series for Lambda and for those large-parameter
0F1 references. Those checks retain tight tolerances instead of accepting the
SciPy normalization errors. No numerical inventory names, domains, or aliases
are added or deferred by these optimizations.
