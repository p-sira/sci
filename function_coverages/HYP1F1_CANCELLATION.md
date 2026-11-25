# Real 1F1 cancellation investigation

The former backend returned approximately **12,796,153.6931455** for
`hyp1f1(65.1, -3.2, -20)`. The independently precision-checked result is
**−9.615930282785953**. The public API and denominator-pole conventions are
preserved; this investigation changes the private algorithm selection.

## Cause

The defining series is

`M(a,b,x) = Σ (a)_n x^n / ((b)_n n!)`.

For that input, the sum of absolute terms divided by the absolute result is
approximately **3.05×10^40**. Kummer's transformation,
`M(a,b,x) = exp(x) M(b-a,b,-x)`, reduces this condition number to approximately
**1.30×10^23**, still beyond binary64 precision. Compensated summation cannot
recover digits lost while constructing and cancelling those terms.

The earlier `abs(a), abs(b) <= 64` dispatch restriction also hid a sharp
algorithm boundary. The moderate-parameter compensated series and the general
Zhang–Jin fallback both encounter this cancellation; merely raising the
threshold or adding guard digits to a single full-argument sum is insufficient.
The new fixtures include the adjacent binary64 inputs on both sides of `a=64`.

## Implemented continuation

For finite `0 < a <= 100000`, `b < a`, `-8 <= b <= 64` and
`-1000 <= x < 0`, excluding existing denominator poles, use the defining series
near the origin and continue along the negative real ray using
[Kummer's differential equation](https://dlmf.nist.gov/13.2#E1):

`z M'' + (b-z) M' - a M = 0`.

For a local Taylor series `Σ c_n t^n` about `z`, its coefficients satisfy

`z(n+2)(n+1)c_(n+2) = (a+n)c_n + (n+1)(z-b-n)c_(n+1)`.

The private implementation reuses Sci's double-double arithmetic. It propagates
both value and derivative, scales coefficients by powers of the step, and keeps
coordinates in double-double precision. Steps stay within half the distance to
the origin, at most 0.5, with an additional phase bound for large `a`. Two
consecutive small terms are required to finish a local expansion.

The starting point is a binary fraction chosen so `abs(a*z) <= 16` and
`abs(z) <= 0.25`. Origin initialization permits 256 terms. Continuation permits
8192 steps with at most 128 terms each. Nonconvergence, nonfinite continuation
intermediates and exhausted work return NaN. No user tolerance or new dependency
is introduced. This follows the outward-integration approach in
[DLMF §13.29(ii)](https://dlmf.nist.gov/13.29#ii).

## Recessive polynomial exception

If `a-b` is exactly a nonnegative integer `n`, the solution is

`M(b+n,b,x) = exp(x) M(-n,b,-x)`.

Outward integration can amplify roundoff into the other solution while this
solution decays exponentially. Instead, evaluate the normalized Laguerre
polynomial through its three-term recurrence, reusing double-double arithmetic,
then multiply by `exp(x/2)` twice. Splitting the exponential retains representable
results even when `exp(x)` itself underflows. Work is at most 100008 recurrence
steps under the input bounds. Exact integer differences are checked with their
subtraction residual, so a nearby noninteger is not silently rounded into this
exception.

## Independent verification

Run:

```sh
uv run tests/scipy/generate_hyp1f1_cancellation.py
cargo test --test special_domains
cargo test --test special_extended
```

The generator pins NumPy 2.5.4 and SciPy 1.18.1. Separate JSON inputs generate
**239 checked-in binary64 reference cases**. Expected values use Python Decimal
and the original defining series with a conservative precision budget, then
repeat with 80 extra digits and require identical binary64 rounding. They never
use Sci's continuation, polynomial recurrence or implementation backend.
The CSV also records SciPy values and the logarithm of the series condition
number for comparison; SciPy is not the sole oracle in this difficult region.

Cases include the original failure, the `a=64` boundary, tiny outputs, both
sides of denominator poles and integer differences, large positive `a` up to
100000, the continuation domain boundaries, and exponential underflow regimes.
Rust tests use relative tolerance `3e-12` without an absolute floor. Additional
public API tests check a contiguous identity, an exact elementary exponential
polynomial, the continuation budget, poles and equal-parameter reduction.
All prior special-function fixtures remain in place.

## Remaining limits

The dispatch rectangle is a work/domain restriction, not a uniform accuracy
certificate. Very negative `b` can amplify errors in the origin initialization:
exploratory continuation at `(a,b,x)=(100000,-63.2,-20)` was unstable, so this
algorithm is deliberately not selected there. The older fallback remains
uncertified for `b < -8`, other excluded parameter regions and arguments below
−1000. Negative `a` and positive-argument cancellation are separate remaining
candidates. Near zeros or parameter poles, relative accuracy can deteriorate.
Large oscillatory cases can exhaust the explicit continuation step budget and
return NaN; `(100000,0.2,-1000)` has a regression test for that behavior.
