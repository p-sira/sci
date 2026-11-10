# Sci special vs SciPy: paired performance protocol

The two standalone benchmarks compare Sci 0.3.0 public `sci::special` APIs with SciPy 1.18.1 / NumPy 2.5.4. They cover 20 real-domain functions from gamma, incomplete gamma/beta, errors, Bessel, hypergeometric, exponential integral, elliptic integrals, orthogonal polynomials, and a reduction. Grid sizes are 64 and 1,024 float64 values.

With the prime agent owning sequential Cargo operations (maximum six build jobs/test threads), register this new target in `Cargo.toml`:

```toml
[[bench]]
name = "special_performance_comparison"
harness = false
required-features = ["special"]
```

Run from `/sci`, with no concurrent load:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  CARGO_BUILD_JOBS=6 cargo bench --bench special_performance_comparison \
  --no-default-features --features special -- \
  --repeat 13 --warmup 3 --batch 4 --out target/special_rust.csv

env -u LD_LIBRARY_PATH \
  PYTHONHOME=/home/psira/.local/share/uv/python/cpython-3.13-linux-x86_64-gnu \
  PYTHONPATH=.venv/lib/python3.13/site-packages \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  RAYON_NUM_THREADS=1 .venv/bin/python \
  tests/scipy/benchmark_special_functions.py \
  --repeat 13 --warmup 3 --batch 4 --out target/special_scipy.csv
```

Each case uses the same formula `low + (high - low) * (i / (n - 1))` and, for Carlson integrals, the same auxiliary arrays `y = x + 0.4`, `z = x + 1.2`, and `p = x + 2.2`. Grids are prepared separately from computation. `prepare_only` measures construction of all four inputs. `compute_only` includes output allocation and destruction, but no input creation. Each reported time is the median of 13 timed blocks, divided by four operations per block; first/fourth quartiles and minimum are supplied. Warmups and GC policy are fixed. The `first_value`, `last_value`, and `sum_value` fields provide numerical diagnostics across implementations; the complete output should be checked with existing numerical fixtures when asserting accuracy.

**Comparability:** Rust evaluates scalar kernels over a `Vec<f64>` and returns a `Vec<f64>`; SciPy evaluates vectorized ufuncs over a NumPy `float64` array. Both produce `n` results, but array dispatch and allocation models differ. These rows are labeled `adapted`; elapsed-time ratios describe public API throughput, not isolated numerical-kernel speed. The reduction `logsumexp` is labeled `matched` mathematically and returns a scalar in both implementations; floating-point reduction order can differ. Optional `--scalar-loop` on Python adds a diagnostic using scalar SciPy calls inside a Python loop, which includes Python dispatch and `np.fromiter` overhead; avoid treating it as a pure kernel comparison. Input preparation timings compare Rust vectors with NumPy arrays and are labeled `adapted`.

For a side-by-side comparison, join CSV rows by `case`, `n`, `input_profile`, `parameters`, and `phase`. For computation, compare Rust `execution_mode=scalar_loop` to SciPy `execution_mode=vectorized_ufunc` (or the optional scalar loop diagnostic). Record CPU, operating system, compiler/Rust flags, SciPy and NumPy versions, BLAS settings, and thermal/load conditions. Re-run on both sides when a difference is near noise or quartile ranges overlap; never mix a Rust run from one host with a SciPy run from another.

**Domain limitations:** All values are finite and real; cases avoid branch cuts, principal-value branches, singular endpoints, poles, and parameter-dependent ill-conditioning. Elliptic `m` is the parameter `k²`, and the incomplete integrals use `phi` in radians, following the same argument order in both implementations. The suite does not measure complex continuation, exceptional cases, exact accuracy tolerances, API error handling, or multi-thread scaling. Existing `benches/special.rs` and `benches/special_scipy.rs` continue to provide CodSpeed-target benchmarks independently.

## Sci 0.3.0 performance measurements

Local release runs, `ellip` 1.2.4, pinned SciPy 1.18.1 and NumPy 2.5.4, 13 timed repetitions, three warmups, four calls per timed block. Native libraries and runtime comparisons were single-threaded; Cargo operations were sequential with at most six build jobs. Raw local CSVs (not committed) are `target/special_rust_final.csv` and `target/special_scipy_final.csv`. All 80 paired rows (20 APIs × two input lengths × two phases) had matching keys. On the measured `compute_only` API workloads, Rust was faster in 34 of 40 rows; at 64 values Rust was faster in all 20 rows. Ratios near one are susceptible to scheduling and timer noise.

The following table reports medians at **1,024 values**, in microseconds. Each implementation executes its public API as described above; the scalar-loop/vectorized-output differences must be retained when interpreting the ratios.

| Function | Rust | SciPy | SciPy / Rust |
| --- | ---: | ---: | ---: |
| gamma | 8.35 | 8.59 | 1.03× |
| gammaln | 14.04 | 13.85 | 0.99× |
| digamma | 21.70 | 12.41 | 0.57× |
| betainc | 115.90 | 301.97 | 2.61× |
| gammainc | 108.89 | 101.06 | 0.93× |
| erf | 12.00 | 12.07 | 1.01× |
| erfinv | 16.55 | 13.93 | 0.84× |
| jv | 211.55 | 805.48 | 3.81× |
| kv | 14.77 | 61.10 | 4.14× |
| hyp1f1 | 48.61 | 107.28 | 2.21× |
| hyp2f1 | 125.58 | 117.79 | 0.94× |
| exp1 | 255.10 | 251.75 | 0.99× |
| ellipk | 13.15 | 15.28 | 1.16× |
| ellipe | 10.28 | 15.38 | 1.50× |
| ellipkinc | 92.18 | 117.09 | 1.27× |
| ellipeinc | 140.11 | 202.79 | 1.45× |
| elliprf | 67.50 | 138.36 | 2.05× |
| elliprj | 99.68 | 246.02 | 2.47× |
| eval_legendre | 9.53 | 24.83 | 2.60× |
| logsumexp | 5.71 | 103.02 | 18.04× |

Across the first, last and sum-of-output summaries, the largest absolute difference divided by `(1 + |SciPy summary|)` was less than `3e-14`. This supports numerical agreement for the sampled grids only; the independent accuracy and boundary tests remain authoritative.

For the dependency upgrade itself, native Divan medians of existing workloads before (`ellip` 1.2.3) and after (`ellip` 1.2.4), in microseconds, were `ellipkinc` 14.33 → 11.79, `ellipdinc` 13.92 → 11.38, `ellipeinc` 19.53 → 17.75 and `elliprj` 14.89 → 13.84. Other elliptic workloads varied in both directions; do not extrapolate these specific improvements to all elliptic functions.

An isolated optimized Rust microbenchmark of the previous and new elementary implementations at 1,024 observations measured `softmax` 6.63 → 6.42 µs and `log_softmax` 5.63 → 5.23 µs. The new `stirling2(n,n−1)` closed form avoids the old quadratic recurrence; at `n=1024`, the same standalone harness measured approximately 285 µs versus 1.6 ns per call. These are kernel-level observations, not CodSpeed comparisons; three new CodSpeed cases monitor the optimized Stirling and normalized-harmonic boundary paths going forward.

### Native-kernel follow-up on PR #51

Further improvements target redundant *numeric work*, regardless of whether Sci already beats SciPy. The same original and optimized Rust binaries were run alternately on CPU core 2, using the same inputs, 13 timed repetitions, five warmups and four calls per block, repeated in seven matched pairs. Times below are median-of-run medians at 1,024 outputs, except for the three named Divan cases. The `target/special_old_*_paired.csv` and `target/special_new_*_paired.csv` files are local-only.

| Operation | Previous Rust | Optimized Rust | Native speedup |
| --- | ---: | ---: | ---: |
| Real `exp1` on 1,024 values | 255.08 µs | 161.83 µs | **1.58×** |
| Real `erfinv` on 1,024 values | 16.43 µs | 14.81 µs | **1.11×** |
| `jnjnp_zeros(1)` | 162.9 ns | 53.4 ns | **3.05×** |

`exp1` retains the original continued-fraction depth while replacing two divisions per step with one. Inverse-normal polynomial evaluation uses fixed-degree Horner loops. The first merged J/J′ zero is a fixed mathematical origin entry and now bypasses bounded root-search setup. Exact unit-shape beta/gamma cases skip general series or Halley inversion, with explicit endpoint and tail tests; those specialized cases are not represented in the general 20-function matrix. Spheroidal characteristic and invalid-coordinate fast paths retain the original scalar arithmetic.

General gamma, digamma, incomplete-beta and most elliptic workloads did **not** show reproducible improvements on the common-input matrix after the native edits; isolated differences during a single run disappeared in repeated alternating measurements. A proposed *single shared* spheroidal spectrum achieved a local 3.6× speedup but **was reverted**: existing integration tests require exact bitwise agreement with individual scalar eigenvalue solves, and the shared truncation changes the last few bits. The incomplete-gamma complementary series retains its original independently rounded log-gamma evaluation after a proposed reassociation changed a reference tail by several ulps. Speculative replacements of power functions in Bessel/Struve recurrences were not retained without independent performance evidence. This follow-up does not establish that every kernel is optimal.

CodSpeed's earlier `jnjnp_zeros` 7.0→7.8 µs warning occurred before the exact-first-zero optimization, with unchanged wrapper/backend and a warning that base/head results used different hardware. Alternating identical pre-optimization source binaries did not reproduce a stable regression; the new shortcut has a separate same-core improvement above. Await a fresh CodSpeed run before interpreting the remote regression as resolved.
