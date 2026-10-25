# `fft` coverage and SciPy adaptations

All 35 numerical names remain available by default. With facade defaults disabled, `fft` omits the three FFTLog functions; enable `fft-hankel` for them. Direct `sci-fft` users select the default-enabled `hankel` feature. All FFT transforms compile without XSF. FFTLog gamma coefficients now use private Rust implementations validated with expanded SciPy fixtures.

Inventory: SciPy 1.18.1, [scipy.fft](scipy_function_list/scipy.fft.md).
All 35 numerical function names have tested native signatures. Ten distinct
periodic operators from SciPy FFTPACK are also exported from `fft`. Six Python
backend/worker management names remain unsupported. Python/Cython compatibility
exports are not additional numerical algorithms.

## Supported names

- Complex transforms: `fft`, `ifft`, `fftn`, `ifftn`, `fft2`, `ifft2`.
- Real transforms: `rfft`, `irfft`, `rfftn`, `irfftn`, `rfft2`, `irfft2`.
- Hermitian transforms: `hfft`, `ihfft`, `hfftn`, `ihfftn`, `hfft2`, `ihfft2`.
- Cosine/sine transforms: `dct`, `idct`, `dst`, `idst`, `dctn`, `idctn`,
  `dstn`, `idstn`, with types I–IV.
- FFTLog: `fht`, `ifht`, `fhtoffset`.
- Utilities: `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift`, `next_fast_len`,
  `prev_fast_len`.
- Periodic operators: `diff`, `hilbert`, `ihilbert`, `tilbert`, `itilbert`,
  `shift`, `cc_diff`, `cs_diff`, `sc_diff`, `ss_diff`.

## Conventions and adaptations

`fft` is a dedicated workspace crate; signal-only exports live under `signal`.
The shared `fft`, `ifft`, and
`rfft` signatures are unchanged: they return vectors, support empty inputs,
and propagate IEEE values. New functions return shared Results, require finite
values, and bound inputs/outputs to 1,048,576 entries and ranks to 16.

RustFFT and RealFFT remain private. Real input uses RealFFT’s half-length
complex kernel for even lengths; odd lengths use its full-length kernel.
Each thread caches only the most recent forward, inverse, and real plans with
their work buffers, for lengths up to 65,536. Larger calls use temporary
workspaces, so changing lengths cannot grow the cache without bound. Forward complex transforms are unnormalized, inverses
divide by the product of transformed lengths. Only SciPy backward normalization
is supported. No automatic windowing, angular-frequency conversion, or sample
spacing scaling is applied. Alternate normalization, workers, overwrite flags,
precomputed plans, and custom backend selection are absent.

N-dimensional functions use flat row-major slices, explicit shapes, and
ordered distinct zero-based axes. Empty complex-transform axes copy input;
real/Hermitian transforms require a nonempty axis list. Two-dimensional wrappers
use rectangular shared rows and transform both axes. No multidimensional
resizing is supported. `rfftn`/`ihfftn` return values plus the reduced shape;
only the last selected axis is reduced to floor(n/2)+1. `irfftn`/`hfftn` take the
full desired output shape and require exactly the matching reduced input size.
Two-dimensional real-output wrappers take the full column count explicitly.
Real inverses discard imaginary DC/even-Nyquist components after inverting the
other selected axes, preserving the multidimensional Hermitian convention.

One-dimensional `irfft` and `hfft` take optional full output length, defaulting
to 2*(input_length−1). Odd output lengths must be explicit. Bins are cropped or
zero-padded, and imaginary DC/even-Nyquist components are ignored. `ihfft` can
crop or pad real input to its optional transform length. Explicit positive
lengths permit zero-padding empty inputs; inferred empty/zero lengths error.

RustDCT supplies private types I–IV kernels and scratch plans. `RealTransformType`
replaces SciPy integer types. DCT/DST forward scaling and endpoint conventions
match backward normalization with orthogonalize=false. Inverse types II/III
swap kernels; type I/IV use their own kernels with composition-factor scaling.
DCT I needs at least two samples per transformed axis. No orthonormal modes,
orthogonalization, padding, or batching beyond explicit N-dimensional axes.

Frequency helpers use finite nonzero signed spacing d, and return cycles per
spacing unit. Even Nyquist is negative for `fftfreq`, positive for `rfftfreq`.
Sequential division avoids n*d overflow; extreme values can underflow to zero
or produce a checked overflow. Shift helpers accept finite one-dimensional
shared complex vectors; empty inputs are unchanged. Real values may be carried
with zero imaginary components. Multidimensional shifts are absent.

Smooth-length helpers match SciPy 1.18.1 factor sets: {2,3,5} for real and
{2,3,5,7,11} for complex. Targets are 0..=1,048,576; zero returns zero. Searches
are bounded trial division. These sizes are version-specific SciPy-compatible
recommendations, not guarantees of optimal RustFFT performance.

## FFTLog

`fht` approximates the SciPy k dr Hankel-integral convention on a logarithmic
periodic grid. Its center relation is offset=ln(k_c r_c); output samples satisfy
k[j]=exp(offset)/r[n−1−j]. `bias` is the power-law exponent. `ifht` reverses the
discrete transform, and `fhtoffset` finds a nearest low-ringing offset using
ties-to-even rounding. Captured single-sequence slices replace SciPy batching.
Real orders may be negative, and signed nonzero spacing permits descending
grids. Numerical coefficients use private Rust translations of SciPy/XSF's
Hare complex log-gamma and Cephes rising-factorial algorithms, with the existing
`special` crate for real log-gamma and RustFFT for Fourier work. Complex gamma
uses fixed Taylor/Stirling expansions, at most eight recurrence steps, and one
reflection. Real ratios take at most 64 recurrence steps before a log-gamma
fallback. SciPy licenses are retained and packaged in `LICENSES/`.

Singular DC gamma ratios, underflowed coefficients or bias scaling, nonfinite
intermediates, and zero inverse Nyquist coefficients return shared errors.
SciPy warnings and singular coefficient substitutions are not reproduced.
Small coefficients, large bias factors, periodic endpoint mismatch, and floating
point cancellation limit accuracy; no continuous-integral error estimate is
provided. Low-ringing offsets improve Nyquist phase alignment but cannot ensure
continuous-integral accuracy. Extreme offset phases can lose fractional bits.

The ordinary fixture tolerance remains 3e-12 absolute plus relative. New cases
at order 20,001 and a DC-only case at order 1,000,000 use 3e-10 for cancellation
in nearby log-gamma values. The former discrepancy is also present in the old
XSF backend (about 4e-11 in the first forward sample). No existing tolerance
was relaxed. Additional fixtures cover Taylor centers, reflection, negative
spacing, asymptotic regimes, near-pole phases, and ratios beyond 64 recurrence
steps. Public DC identities and inverse round trips supplement SciPy references.

Supplemental validation evaluated all 6,018 existing FFTLog cases from
`sci-testing` commit `539d39538b2b91544776b0b48d0eb6ddca7743c7` through this
branch's public Sci APIs (Rust 1.99.1, dev profile). References were the saved
SciPy 1.18.1 binary64 values; the recorded XSF baseline used Sci commit
`b52c598535920279dc0aadb14a9085b8545ba4c1`. No cases failed the ordinary
3e-12 absolute-plus-relative comparison.

| Function | Cases | Maximum absolute difference from reference | Maximum / mean difference in epsilon |
| --- | ---: | ---: | ---: |
| `fht` | 2,006 | 4.93e-14 | 659.60 / 0.53341 |
| `ifht` | 2,006 | 4.00e-15 | 466.76 / 0.27887 |
| `fhtoffset` | 2,006 | 0 | 0 / 0 |

The epsilon metric divides by the smaller absolute value, with the harness's
zero/subnormal convention; means include interleaved imaginary zeros. The
recorded XSF maxima/means were 619.05/0.53128 and 612.77/0.27612 for forward
and inverse transforms. Only three forward and three inverse cases changed,
by at most 3.56e-15 in absolute value. These sweeps mostly vary input values
over a few parameter profiles; the 64 added fixtures exercise more coefficient
regimes. Neither suite establishes a whole-domain accuracy bound.

The harness records Sci and SciPy for FFTLog. GSL's
[discrete Hankel transform](https://www.gnu.org/software/gsl/doc/html/dht.html)
uses Bessel-zero sampling on a finite interval and has no adapter in these
measurements. Comparing it with logarithmic periodic FFTLog would require
continuous test functions with matched integral conventions and separate grids.

The problematic cases were also checked against an independent mpmath dense
DFT evaluation of Hamilton's discrete multiplier, converged at 100 and 160
decimal digits. Sixteen such references are committed in `fftlog_accuracy.csv`
and regenerated by `generate_fftlog_accuracy.py` from `fft_cases.json`.

| Case | Rust maximum relative error | Recorded XSF maximum relative error | SciPy maximum relative error |
| --- | ---: | ---: | ---: |
| Worst extended forward case, order 2 | 2.41e-13 | 2.32e-13 | 1.66e-13 |
| Worst extended inverse case, order 1.5 | 1.50e-14 | 1.74e-14 | 1.19e-13 |
| New forward case, order 20,001 | 2.36e-11 | — | 1.97e-11 |
| New DC-only case, order 1,000,000 | 1.02e-16 | — | 1.05e-11 |

These independent references expose error in SciPy as well as Sci. The large
positive-order DC case benefits from the real gamma-ratio asymptotic formula;
the full transform still loses precision in nearby complex log-gamma values.
The large-order high-precision comparisons allow 3e-12 absolute plus 3e-11
relative error; the other cases retain 3e-12 absolute plus relative tolerance.

## Periodic operators

`diff`, `hilbert`, `ihilbert`, `tilbert`, `itilbert`, `shift`, `cc_diff`,
`cs_diff`, `sc_diff`, and `ss_diff` multiply shared real FFT bins by their
mathematical Fourier symbols, then use the shared inverse. Inputs are finite
nonempty real sequences up to 1,048,576 samples. Explicit periods must be finite
and positive; operator parameters must be finite. Displacements, hyperbolic
parameters, and Tilbert heights use the independent-variable units of period.
Python kernel caches and axis batching are absent.

`diff` bounds signed derivative/integral orders at −64..=64. Negative orders
use zero integration constants. Nonzero orders remove DC; odd orders remove
even Nyquist. Order zero returns an exact copy after validation. `hilbert` has
the FFTPACK positive-i sign convention, opposite to the imaginary component
of `signal::hilbert`; its inverse negates that symbol. Both remove DC/Nyquist.
Tilbert symbols are i*coth(ωh) and −i*tanh(ωh), also removing DC/Nyquist.
Forward Tilbert requires nonzero h; inverse h=0 is supported as a zero operator.

Hyperbolic cosh/cosh and sinh/sinh operators preserve their real Nyquist
multipliers and have DC symbols 1 and a/b. Cosh/sinh and sinh/cosh are imaginary
symbols with signs −i and +i, removing DC/Nyquist. Hyperbolic denominator b must
be nonzero for cosh/sinh and sinh/sinh. Log-domain evaluation avoids overflow
when separate sinh/cosh values overflow but their ratio is representable.
Tiny ratios may underflow to zero; nonfinite results error. `shift` preserves
the mean and uses the SciPy packed-kernel even-Nyquist multiplier cos(phase)+sin(phase).
That exceptional multiplier is retained explicitly and tested; it differs from
a continuous cosine-mode shift’s cosine projection. Extreme finite
phases can lose argument-reduction accuracy. Periodic endpoint mismatch can
cause ringing, and no continuous derivative/integral error estimate is provided.

## FFTPACK inventory disposition

The [FFTPACK inventory](scipy_function_list/scipy.fftpack.md) contains 31
main-namespace names: 13 duplicated exports use the existing `fft` APIs,
10 periodic operators are migrated here, and eight compatibility variants
(`rfft`, `irfft`, `rfftfreq`, `next_fast_len`, `idct`, `idst`, `idctn`, `idstn`)
are removed in favor of modern complex-bin, length-helper, and normalized
inverse conventions. The `sci-fftpack` crate, facade module, and feature are
removed. Deprecated subnamespace aliases add no algorithms.

The four `scipy.fftpack.convolve` kernel/cache functions (`convolve`,
`convolve_z`, `destroy_convolve_cache`, `init_convolution_kernel`) were never
implemented and remain outside the API. No compatibility placeholders remain.

## Remaining names

`get_workers`, `register_backend`, `set_backend`, `set_global_backend`,
`set_workers`, and `skip_backend` configure SciPy Python runtime state. There
are no placeholder equivalents or exported backend types.

## Verification

Pinned SciPy 1.18.1 / NumPy 2.5.4 generation reads input-only
`tests/scipy/fft_cases.json` and writes separate checked-in input/expected CSVs.
Rust tests exercise every supported numerical name with multiple cases,
including odd/even lengths, axis permutations/subsets, reduced axes, type I–IV
normalization, negative spacing, gamma orders/biases, and shape/order checks.
Comparisons use 3e-12 absolute plus relative tolerance for short FFT/DCT/FFTLog
factorizations. Independent properties cover inverse round trips, Hermitian
endpoint projection, exact shift inversion, smooth-length bounds/idempotence,
invalid values/shapes/axes, work bounds, overflow, and cross-module statistics.
Original signal FFT and Fourier pipeline tests continue to pass.

Periodic references read `tests/scipy/fft_periodic_cases.json` and write
`fft_periodic_inputs.csv` and `fft_periodic_expected.csv` using the same pinned
versions and 3e-12 absolute plus relative tolerance. Every migrated operator
has multiple public reference cases for ordering, odd/even Nyquist, and signed
parameters. Independent tests cover sine derivatives and shifts, derivative/
integral and Hilbert/Tilbert inverses, signal Hilbert sign interoperability,
stable hyperbolic ratios, invalid domains, work bounds, and overflow.

```console
uv run tests/scipy/generate_fft.py
uv run tests/scipy/generate_fft_periodic.py
python tests/scipy/audit_fft.py
cargo fmt --all -- --check
cargo test
cargo clippy --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --no-deps
```
