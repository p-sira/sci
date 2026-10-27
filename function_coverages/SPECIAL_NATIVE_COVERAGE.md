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

**19 generic complex APIs** live in `special::complex`, avoiding C++ overload
replication and ambiguity with real scalar calls:

- `gamma`, `erf`, `erfc`, `wofz`.
- `jv`, `jve`, `yv`, `yve`, `iv`, `ive`, `kv`, `kve`.
- `hankel1`, `hankel1e`, `hankel2`, `hankel2e`.
- `airy`, `airye`, `fresnel`.

Arguments/results use `sci::complex::Complex<T>`; errors use `sci::Result`.
Airy returns `[Ai,Ai′,Bi,Bi′]`; Fresnel returns `(S,C)`. Existing real APIs
and the original root-level `wofz` and Hankel functions are preserved.
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

- Additional complex digamma, log-gamma, reciprocal gamma, error variants,
  exponential/sine/cosine integrals, Spence, hypergeometric, zeta and Wright
  functions; arbitrary complex Lambert W branches.
- Complex Carlson integrals, Legendre and spherical-Bessel arguments;
  Legendre/harmonic derivatives and higher derivative tensors.
- Noninteger polynomial-degree continuations, with domains and branch contracts
  distinct from integer polynomial evaluation.
- Statistical cosine/von-Mises/Tukey-lambda kernels, Poisson-binomial and
  Wilcoxon distribution tables, Cramér–von Mises, generalized hyperbolic/inverse
  Gaussian densities and multivariate normal CDFs, where the owning stats module
  does not already provide the requested operation.
- Existing real-kernel accuracy/work limitations, especially the general 1F1
  cancellation case and bounded wave-function orders, remain as recorded in
  [SPECIAL_COVERAGE.md](SPECIAL_COVERAGE.md).

These gaps are not counted as implemented. Full XSF/SciPy overload parity and
100% public-name replication are not completion criteria. This API coverage
audit is separate from Codecov executable-line coverage; no new 99% claim is made.

## Independent validation

`tests/scipy/generate_special_native.py` pins **NumPy 2.5.4 / SciPy 1.18.1**.
Separate JSON inputs generate **1010 checked-in f32/f64 reference rows** for
all 27 additions. Ordinary Rust tests exercise only public `sci` APIs and read
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

Additional tests cover invalid inputs, work limits, infinity/NaN/signed-zero
contracts, gamma recurrence, conjugacy, Bessel inverse round trips, complements,
normalized interval additivity, Hankel identities, complex K scaling, branch-cut
continuation identities and shared complex arithmetic interoperability.
Binary64 comparisons use relative tolerances reflecting inverse-reference and
backend rounding; tiny scalar remainders/complements have no large absolute
floor that would hide a returned zero.
