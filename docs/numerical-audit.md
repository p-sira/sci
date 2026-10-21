# Power transforms and convolution audit

This targeted audit fixes numerical errors in `sci-special` and `sci-stats`
and reduces work in `sci-signal`. It does not claim exhaustive numerical
verification of every crate or full SciPy overload compatibility.

## Findings and changes

- `boxcox(0.5, 5e-324)` returned `-1` instead of approximately
  `-0.6931471805599453`: multiplying the logarithm by a subnormal lambda rounded
  away essential precision. The shared special-function kernels now reuse the
  installed xsf backend's logarithmic limits and scaled exponential. For
  example, `boxcox(100, 155)` now returns a finite value near `6.45e307`, despite
  `100^155` overflowing. Explicit zero endpoints preserve `-1/lambda` for
  positive lambda, including tiny parameters where a logarithmic limit is
  inappropriate.
- Statistics Box–Cox and Yeo–Johnson reuse those kernels. Yeo–Johnson evaluates
  its defining Box–Cox identity without forming `1+x`, preserving subnormal
  observations. The fallible statistics APIs retain their finite-input
  validation and report true result overflow.
- Power means lost their geometric limit near zero power. Centered logarithms
  with `expm1` and `log1p` retain small corrections. An extremely small power
  uses the geometric limit once the correction is below binary64 precision.
  Constant data return the stored value exactly. The ordinary log-sum-exp path
  reuses its allocation instead of allocating a second transformed vector.
- Direct convolution now restricts each inner loop to products contributing
  to the requested output. This removes the inner cropping branch and skips
  discarded products, while preserving each output's accumulation order.
  Correlation inherits the optimization through convolution. The existing
  conservative input-product work bound remains in effect.

## Independent verification

`tests/scipy/generate_power_stability.py` pins SciPy 1.18.1 and NumPy 2.5.4.
It generates 19 separate input/reference cases. Transform references use SciPy
special functions; Yeo–Johnson uses its documented Box–Cox identity because
SciPy's direct statistics formula can underflow for subnormal observations.
Power means use their defining identity in 400-digit Decimal arithmetic,
avoiding binary64 reference overflow and loss near zero power. Rust tests
also check mean bounds, monotonicity, scaling, constant inputs and domains.

`tests/scipy/generate_cropped_convolution.py`, with the same pins, generates
384 direct SciPy cases for convolution and correlation in full/same/valid
modes, covering both argument length orderings and odd/even lengths 1–8.
The integer-valued fixtures require exact output equality, including shape
and ordering. Inputs and expected values are committed separately; ordinary
Rust tests need no Python runtime.

The inventories in `SCIPY_FUNCTIONS.md` identify statistics compatibility
namespaces and special-function Python/Cython duplicates as shared algorithms.
Signal compatibility names likewise reuse canonical functions. This change
adds no exports or new namespace coverage. The existing statistics and signal
coverage audits pass; masked statistics, other distribution families and
unsupported overloads remain outside this change.

## Local performance measurements

Release builds with rustc 1.98.1, 100 Divan samples, deterministic existing
benchmark inputs. Median times are illustrative local measurements and vary
with scheduling and hardware; they are not performance guarantees.

| Kernel | Before | After |
| --- | ---: | ---: |
| Full convolution, 1024 × 32 | 22.06 µs | 9.90 µs |
| Valid convolution, 1024 × 1024 | 689.5 µs | 3.66 µs |
| Same correlation, 1024 × 32 | 35.50 µs | 9.88 µs |
| Power mean, n=1000, p=1.5 | 12.89 µs | 12.56 µs |
| Power mean, n=1000, p=1e-18 | 12.59 µs | 14.07 µs |

The near-zero power-mean path trades some runtime for correct limiting behavior;
the small ordinary-power timing change is within normal benchmark variability.
Its scratch allocation count decreases from two to one. Valid convolution with
equal lengths evaluates n retained products instead of iterating over n² pairs.

Reproduce with:

```sh
cargo bench --no-default-features --features stats,signal,complex \
  --bench signal --bench stats -- 'convol|pmean'
```

## Remaining numerical limitations

Binary64 rounding, true overflow/underflow, ill-conditioned cancellation and
backend approximation error remain. Direct convolution retains ordinary
summation, so finite exact sums can still overflow in intermediate products
or partial sums. Box–Cox special values follow the documented scalar contracts;
statistics rejects nonfinite inputs. No API signatures, work bounds, dependencies
or supported SciPy overloads are added or expanded.
