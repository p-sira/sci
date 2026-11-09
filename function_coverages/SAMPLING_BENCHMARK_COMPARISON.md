# Sampling engine performance comparison (Sci 0.3.0 / SciPy 1.18.1)

The paired benchmarks are `benches/sampling_scipy_comparison.rs` and
`tests/scipy/benchmark_sampling_engines.py` (SciPy 1.18.1, NumPy 2.5.4).
They compare individual generation calls with the same requested `n`, dimension
`d`, distribution parameters and seed where applicable. The seed fixes each
implementation's RNG independently; it does not generally reproduce the same
random sequence in Rust and NumPy.

## Timing contract

- `construct_only` measures creation of the generator, with no samples drawn.
  Input weight vectors, probability laws and covariance matrices are prepared
  outside this window. These rows have `n=0`.
- `draw_random_only` constructs a **fresh** generator before starting the timer.
  It measures one `random(n)` / `rvs(size=n)` call, including the allocation of
  its output. Resetting state and dropping the generator or output are excluded.
- The implementations run in one process each, with native thread-pool limits
  of one; Python sets those limits before importing NumPy/SciPy. The harnesses
  report median and minimum elapsed nanoseconds from nine timed repetitions
  after two untimed warmups by default. Increase both with `--repeat` and
  `--warmup` when comparing small changes. Run Rust and Python consecutively on
  an idle host, preferably several times, to gauge frequency and thermal noise.
- `--quick` uses the first sample size in each case. For the full matrix,
  `n=128,1024,8192` where supported, with smaller sizes for expensive or
  bounded engines. The CSV carries the exact `n`, `d`, phase, method parameters,
  comparison category and output layout for each measurement.

The Rust runner is invoked by the coordinating agent (which owns Cargo):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  cargo bench --bench sampling_scipy_comparison \
  --no-default-features --features stats -- \
  --repeat 21 --warmup 5 --out target/sampling_rust.csv
```

From `/sci`, with the pinned Python environment:

```sh
env -u LD_LIBRARY_PATH \
  PYTHONHOME=/home/psira/.local/share/uv/python/cpython-3.13-linux-x86_64-gnu \
  PYTHONPATH=.venv/lib/python3.13/site-packages \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  RAYON_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python tests/scipy/benchmark_sampling_engines.py \
  --repeat 21 --warmup 5 --out target/sampling_scipy.csv
```

## Output format and algorithm matching

| Cases | Interpretation |
| --- | --- |
| `sobol_plain`, `sobol_scrambled` | Sci returns nested `Vec<Vec<f64>>` with a row allocation for every point; SciPy returns a contiguous `float64` NumPy array. The timings include those different output costs. |
| `sobol_plain_flat`, `sobol_scrambled_flat` | Sci calls `random_flat(n)` to allocate a contiguous row-major `Vec<f64>`; SciPy's ordinary `Sobol.random(n)` already returns a C-contiguous `(n,d)` array. The Python harness verifies that layout outside the timed window. These are the comparable output-format cases: `n=128,1024,8192`, `d=4`, `bits=30`. |
| Plain vs. scrambled Sobol | Unscrambled generators use the same Joe–Kuo digital sequence. The scrambled cases use `seed=42` in both programs but different scrambling RNGs, so they are *adapted* comparisons; neither requires identical generated values. |
| Halton and Latin hypercube | The main mathematical parameters match. Sci still allocates nested rows; SciPy uses NumPy arrays, and seeded permutations differ. Scrambled Halton is marked *adapted*. |
| Multivariate normal QMC | Both use an unscrambled, 30-bit Sobol sequence, the same inverse-normal clipping expression and the same covariance. The inverse-normal backend, matrix arithmetic and output containers differ; this is a matched mathematical workload. |
| Multinomial QMC and Poisson disk | The target distributions and primary parameters match, but categorical tie conventions and Poisson candidate schedules differ, respectively. Both remain *adapted*. |
| Discrete alias and guide | Matched probability weights and sample count; RNG streams and returned containers differ. |
| Fast generator inversion, numerical inverse Hermite/polynomial | Matched target distributions but different table construction/quantile algorithms. Sci's polynomial sampler uses linear interpolation, whereas SciPy uses order 3; comparisons are *adapted*. |
| Ratio-of-uniforms and transformed rejection | Target densities and their domains match, but Rust invokes native scalar closures and SciPy uses Python/NumPy callbacks or different UNU.RAN rejection envelopes. `SimpleRatioUniforms` uses a certified rectangular envelope in Sci and an automatically chosen envelope in SciPy. These results are *adapted*. |

The `comparability` CSV column describes comparable mathematical *tasks*, not
guaranteed bitwise output parity. Even the flat Sobol pair retains differences
between a Rust vector and a NumPy array's object metadata, allocation strategy
and function-call overhead. Comparing `sobol_plain_flat` across languages is
the appropriate evidence for the contiguous Sobol generator; comparing
`sobol_plain` across languages additionally measures row-container overhead.

## Validation and interpretation

The Python harness rejects incorrect full batch sizes and verifies the shape,
type and C-contiguous `float64` layout of the new Sobol flat cases. The full
matrix completed a one-repeat smoke run in the pinned Python environment, and
the quick matrix completed with three timed repetitions and one warmup.
These smoke runs establish harness execution, **not** stable cross-language
speed ratios. Rust build, full benchmarks and parity regression tests must be
run serially by the coordinating agent.

## Validated release results (2026-11-09)

The coordinating agent subsequently completed both full-size, single-thread
release matrices with **21 timed repeats and 5 warmups** per case, serially,
with native BLAS/OpenMP/Rayon thread limits set to one. The measured CSVs are
`target/sampling_rust_final.csv` and `target/sampling_scipy_final.csv` (68
matching case/phase/size records each, excluding headers). They are local
benchmark artifacts and are not committed. Values below are median draw times
in microseconds; interpret them as measurements on this host, not universal
speed ratios.

| Engine and batch | Sci Rust | SciPy | Rust / SciPy |
| --- | ---: | ---: | ---: |
| Sobol plain nested, 8192×4 | 221.89 | 55.73 | 3.98× |
| Sobol scrambled nested, 8192×4 | 225.66 | 58.60 | 3.85× |
| Sobol plain contiguous, 8192×4 | 38.48 | 55.25 | 0.70× |
| Sobol scrambled contiguous, 8192×4 | 38.48 | 58.95 | 0.65× |
| Halton plain, 8192×4 | 556.37 | 1032.06 | 0.54× |
| Halton scrambled, 8192×4 | 1266.87 | 3959.23 | 0.32× |
| Latin hypercube scrambled, 8192×4 | 528.01 | 533.11 | 0.99× |
| Latin hypercube centered, 8192×4 | 407.60 | 371.63 | 1.10× |
| Numerical inverse Hermite, 1024 | 17.67 | 41.35 | 0.43× |
| Transformed-density rejection, 1024 | 68.10 | 94.92 | 0.72× |

The only cases with a larger Rust median among all 49 draw rows were nested
Sobol plain and scrambled at 1024 and 8192 points, and centered Latin
hypercube at 8192 points. Nested Sobol remains limited primarily by thousands
of heap-allocated row vectors; standalone profiling of alternative safe row
construction approaches found no consistent gain. `random_flat` retains
bitwise-equivalent contiguous row-major coordinates without changing the
existing `random` API. The centered LHS midpoint-shuffle candidate and the
TDR stack-boundaries candidate were reverted because repeat measurements
showed no reliable speedup. TDR algorithms and Python callback paths remain
different; this benchmark does **not** establish identical rejection work.

Cached Hermite interpolation segments and coarse probability bins are now
validated against uncached reference interpolation and seeded draws, including
endpoint/upper-tail cases. They reduced the measured 1024-point draw time to
17.67 µs; the earlier ~41 µs Rust measurement was taken before this change.
The improvement must not be interpreted as evidence of identical knot
selection with SciPy's separate inversion implementation.

Serial validation passed all **78** `sci-stats` library unit tests and **15**
Sobol integration tests, including bitwise stream and error-state regressions.
The full `sobol_comparison` release run also passed **29** independent SciPy
fixture checks and completed all **213** measurement cases, recorded in
`target/sobol_comparison_final.csv`. At most eight Cargo build/test jobs were
used, and no Cargo builds ran concurrently. `git diff --check` passed.

Measured limitations remain for nested Sobol and large centered LHS. Their
losses are openly retained rather than making an across-the-board performance
claim. For all engines, smaller numerical gaps and cross-language differences
may fluctuate with CPU frequency, memory allocator, Python dispatch and
rejection paths; rerun paired matrices serially before attributing changes.

Construction costs are deliberately excluded from draw times. Where algorithms
differ (particularly inversion and rejection), performance gaps alone cannot
identify an optimization target until generator work, callback cost, accuracy
and output representation have been considered separately. For small batches,
repeated full-process runs are useful because Python dispatch and timer overhead
can be a substantial share of the observed duration.
