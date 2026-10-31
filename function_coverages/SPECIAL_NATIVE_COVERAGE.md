# Native special-function API coverage

The reference is the local C++ XSF repository at
`8e321105803530610770bcb6c6db2017322c976b`. This audit reviews all **66 direct
public-layer headers** under `include/xsf` and `include/xsf/cpu`. Descendant
Cephes, AMOS, specfun, Carlson and third-party directories contain implementation
algorithms, not additional Sci namespaces. Header-family review does not imply
coverage of every overload, template instantiation or mathematical domain.

Reproduce the audit with:

```sh
python tests/scipy/audit_special.py
python tests/scipy/audit_special_native.py --xsf /home/psira/Code/cpp/xsf
```

The optional source path checks for newly added or unreviewed headers. The
checked-in [disposition manifest](../tests/scipy/special_native_inventory.json)
records each reviewed header and its treatment. SciPy remains an independent
reference, not a requirement for complete API or domain parity.

## Implemented additions

**Eight generic scalar APIs** use `T: sci_core::Float`:

| API | Mathematical contract and Rust adaptation |
| --- | --- |
| `log1pmx(x)` | `ln(1+x)−x`; cancellation-resistant series in `T`, standard `ln_1p` elsewhere |
| `digammainv(y)` | Unique positive inverse of digamma; existing Sci digamma/trigamma with bounded positivity-preserving Newton refinement |
| `gamma_ratio(a,b)` | Positive finite arguments; reuse bounded Pochhammer products where safe and log-beta/log-gamma elsewhere, avoiding overflowing individual gamma values |
| `scaled_exp1(x)` | `x exp(x) E₁(x)`; continuous endpoints `F(0)=0`, `F(+∞)=1`, bounded continued fraction without exponential overflow |
| `iv_ratio(v,x)` | `I_v(x)/I_(v−1)(x)` for `v>=1/2`, `x>=0`; shared scaled Perron fraction |
| `iv_ratio_c(v,x)` | Direct complementary ratio, preserving values lost by subtracting a rounded ratio from one |
| `iv_ratioinv(v,r)` | Nonnegative inverse ratio; bounded bisection and complementary residual near `r=1` |
| `normalized_gen_harmonic(j,k,n,a)` | Inclusive truncated power-law probability; integer indices, finite exponent, at most one million terms, shared `Result`, maximum-scaled compensated sums |

**33 generic complex APIs** live in `special::complex`, avoiding C++ overload
replication and ambiguity with real scalar calls:

- `gamma`, `loggamma`, `rgamma`, `digamma`, `erf`, `erfc`, `erfcx`, `erfi`, `dawsn`, `wofz`.
- `jv`, `jve`, `yv`, `yve`, `iv`, `ive`, `kv`, `kve`.
- `hankel1`, `hankel1e`, `hankel2`, `hankel2e`.
- `airy`, `airye`, `fresnel`.
- `lambertw`, `exp1`, `expi`, `spence`, `xlogy`, `xlog1py`, `zeta`, `hurwitz_zeta`.

Arguments/results use `sci::complex::Complex<T>`; errors use `sci::Result`.
Airy returns `[Ai,Ai′,Bi,Bi′]`; Fresnel returns `(S,C)`. Existing real APIs
remain in their respective family modules; the former root-level `wofz` and
Hankel functions now reside under `special::elementary::wofz` and
`special::bessel::{hankel1, hankel1e, hankel2, hankel2e}`.
The complex namespace requires finite inputs and finite results, returning
errors on overflow, conversion overflow and backend failures. The kernels
compute in binary64 before conversion to `T`. No arbitrary-precision claim is
made. Negative-real branch cuts retain signed imaginary zero; a shared Schwarz
reflection adapter corrects AMOS's treatment of both zero signs as the upper
bank, interchanging Hankel kinds when necessary.

Complex arithmetic remains in `sci::complex`; Bessel/Airy algorithms reuse the
existing private Rust AMOS backend, and gamma/error/Fresnel reuse private kernels.
Perron's fraction reuses the existing private double-double arithmetic; it is
bounded by 1000 terms. The inverse is bounded by 128 fraction evaluations.
No C++ dependency or new Rust dependency is introduced.

The new complex additions extend the original 25-function native migration:

| API | Algorithm, mathematical domain and Rust treatment |
| --- | --- |
| `lambertw(z,k)` | Integer branch of `w exp(w)=z`, bounded Halley iteration with rational and Puiseux starts, residual verification and signed branch-cut banks; nonprincipal zero is singular |
| `exp1(z)`, `expi(z)` | Complex exponential integrals using bounded series, continued fraction, asymptotic evaluation and near-axis correction; preserve signed cut banks; finite magnitude limited to 1e6 |
| `spence(z)` | `Li₂(1-z)` on the principal branch, evaluated with reflection/Landen/inversion and bounded power/Bernoulli series; retains the two banks of the negative-real cut |
| `xlogy(x,y)`, `xlog1py(x,y)` | Stable complex logarithmic products, with exact zero prefactor and branch handling; `log1p`-style evaluation preserves sub-ulp increments |
| `zeta(s)`, `hurwitz_zeta(s,q)` | Riemann and shifted Hurwitz zeta with bounded compensated Euler–Maclaurin summation, twelve Bernoulli corrections and cancellation-based rejection; `0≤Re(s)≤30`, `|Im(s)|≤32`, and for Hurwitz `0<Re(q)≤64`, `|Im(q)|≤32` |

These APIs use `Complex<T>`/`Result` and binary64 numerical kernels. Domain
limits protect numerical accuracy and bounded work; they do not claim complete
complex analytic continuation. In particular, near-pole SciPy complex zeta and
near-cut SciPy E₁ values were independently checked because the reference can
lose precision there. The Jacobi `ellipj(u,1)` hyperbolic limit also avoids
overflow for large finite `u`; near-unit modulus cases fall back to AGM when
the perturbation becomes nonuniform.

Complex `loggamma`, `rgamma`, and `digamma` complete this gamma-family
increment. Log-gamma is the analytic branch real on the positive axis, with
signed-zero banks of the negative-real cut and an unwrapped imaginary part;
it is not `log(gamma(z))`. Reciprocal gamma is entire, returns exact zero at
gamma poles, and exponentiates negative log-gamma to retain representable values
when gamma overflows. Digamma reuses real/root approximations and differentiates
the shared Stirling expansion, with at most eight shifts and scaled cotangent
reflection. These APIs retain the shared finite-input/result error contract.
Independent fixtures include near-pole, root, Taylor/recurrence/asymptotic,
conjugate, cut and overflow regimes; property tests check recurrences,
reciprocity, tiny pole distances and large complex arguments.

Complex `erfcx`, `erfi`, and Dawson's integral (`dawsn`) reuse the private
Faddeeva family. Scaled erfc is evaluated as `w(iz)` rather than forming the
potentially overflowing product `exp(z²)*erfc(z)`. Dawson uses its dedicated
near-axis expansions instead of multiplying an overflowing erfi by a tiny
exponential. All three are entire, share the finite-input/result error
contract, and use generic Sci complex values. Faddeeva series have explicit
52-term table bounds or 128-term computed-sum limits; exhaustion returns a
numerical error. These bounds also protect the existing error-function APIs.
There is no new dependency or public approximation/tolerance API.

The added 258 reference rows cover both precisions, quadrants, axes, tiny
arguments, approximation-region boundaries and large arguments where the
scaled functions remain finite. Public property tests check rotation, parity,
conjugation, scaling identities, real-facade interoperability, invalid inputs,
binary64 overflow and conversion overflow. Close-to-zero components retain
relative-only fixture comparisons. The real erfcx interoperability comparison
allows 2e−14 because its separate rational approximation differs by about
3e−15 at `x=1e30`; this does not relax the independent complex references.

## Deliberate exclusions and replacements

These XSF interfaces are not sensible additions to the public Sci special API:

| XSF surface | Sci treatment |
| --- | --- |
| `float`/`double` overloads and integer-width variants | Generic `Float`, typed integer degrees/indices and shared containers |
| `cyl_bessel_*`, legacy `itsh0`/`itth0`/`itsl0`, `_l` polynomial suffixes and internal algorithm aliases | Reuse existing mathematical function names; do not multiply synonymous APIs |
| `exp2`, `exp10`, `expm1`, `log1p`, `cbrt`, numerical constants | Standard float methods/constants and owning Sci arithmetic modules |
| `evalpoly`, `zlog1`, raw storage/output-pointer conventions | Existing Sci polynomial/complex operations; slices and owned results |
| NumPy/CuPy adapters, `mdspan`, layouts, CPU/device configuration | Rust types and existing module interoperability |
| Global `set_error`, numeric error codes and backend exceptions | Shared `Result` and documented scalar IEEE contracts |
| Root-solver engines, recurrence callbacks, dual-number storage and normalization-policy template plumbing | Private algorithms and explicit mathematical outputs; no public implementation machinery |
| Approximation branches, Faddeeva `w_im`, continued-fraction generators, `specfun`/AMOS internals and test error metrics | Private implementation/testing details, not additional numerical API coverage |
| Distribution-table and multivariate-normal kernels | Belong in `sci::stats`; missing statistical capabilities are deferred there, not counted as covered by this special-function work |

This does not exclude useful mathematical derivatives or distributions merely
because XSF implements them through templates. Their mathematical APIs need a
coherent design in the owning module; exposing the C++ machinery is unnecessary.

## Remaining numerical and signature gaps

The following remain useful candidates, rather than intentional exclusions:

- Complex normal probabilities and their logarithms, sine/cosine integrals,
  hypergeometric and Wright functions; complex exponential integrals outside
  the new bounded numerical domains.
- Hurwitz/Riemann zeta continuation outside the documented Euler–Maclaurin
  domain, including severe cancellation near zeros and poles.
- Complex Carlson integrals, Legendre and spherical-Bessel arguments;
  Legendre/harmonic derivatives and higher derivative tensors.
- Noninteger polynomial-degree continuations, with domains and branch contracts
  distinct from integer polynomial evaluation.
- Statistical cosine/von-Mises/Tukey-lambda kernels, Poisson-binomial and
  Wilcoxon distribution tables, Cramér–von Mises, generalized hyperbolic/inverse
  Gaussian densities and multivariate normal CDFs, where the owning stats module
  does not already provide the requested operation.
- Existing real-kernel accuracy/work limitations, including 1F1 outside its
  improved negative-argument continuation range and bounded wave-function
  orders, remain as recorded in
  [SPECIAL_COVERAGE.md](SPECIAL_COVERAGE.md).

These gaps are not counted as implemented. Full XSF/SciPy overload parity and
100% public-name replication are not completion criteria. This API coverage
audit is separate from Codecov executable-line coverage; no new 99% claim is made.

## Independent validation

`tests/scipy/generate_special_native.py` pins **NumPy 2.5.4 / SciPy 1.18.1**.
Separate JSON inputs generate **1976 checked-in f32/f64 reference rows** for
the original 33 additions. Ordinary Rust tests exercise only public `sci` APIs and read
the committed CSV, without Python or a backend reference at runtime.

- Bessel ratios/complements use 100-digit standard-library Decimal defining
  0F1 sums and large-argument asymptotics, not the implemented Perron fraction.
  Inverse references use SciPy Brent solving against these independent sums.
- Digamma inverses use SciPy digamma and a root solve in logarithmic coordinates.
- Gamma ratios use exact integer factorial ratios or independent SciPy log-gamma
  differences; tiny/large arguments and individually overflowing gamma values
  are included.
- Scaled E₁ uses SciPy E₁ on the safe small interval and its defining Laplace
  integral by quadrature for large arguments.
- Logarithmic remainders and normalized harmonic probabilities use independent
  100-digit Decimal logarithms and defining powers.
- Complex functions use SciPy's complex overloads, with ordered component checks.

The eight later complex APIs have separately checked public tests in
`tests/special_complex_{lambert,expint,polylog,zeta}.rs`; their independent
fixtures and reference generators live under `tests/scipy/` and
`tests/fixtures/`. The Lambert W suite uses over four thousand reference rows
across seven integer branches, both complex precisions and cuts. Exponential
integral fixtures use SciPy as the main reference and 100-digit mpmath only
where its near-axis evaluation loses the constant branch jump. Zeta tests
use SciPy's complex Riemann values, selected high-precision Hurwitz values,
and recurrence, conjugation and half-argument identities. Complex Spence and
stable logarithmic products are checked against SciPy on cut banks,
cancellation-sensitive scales and large finite arguments.

Additional tests cover invalid inputs, work limits, infinity/NaN/signed-zero
contracts, gamma recurrence, conjugacy, Bessel inverse round trips, complements,
normalized interval additivity, Hankel identities, complex K scaling, branch-cut
continuation identities and shared complex arithmetic interoperability.
Binary64 comparisons use relative tolerances reflecting inverse-reference and
backend rounding; tiny scalar remainders/complements have no large absolute
floor that would hide a returned zero.
