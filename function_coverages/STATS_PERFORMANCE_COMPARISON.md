# Non-sampling stats performance comparison (Sci 0.3.0 / SciPy 1.18.1)

The paired runners are `benches/stats_scipy_comparison.rs` and
`tests/scipy/benchmark_stats_functions.py`. They exercise 16 public
non-sampling operations with two sample counts (`n=128`, `n=2048`) each.
SciPy **1.18.1** and NumPy **2.5.4** are required by the Python runner.
Neither runner uses stochastic inputs.

## Comparable operations

| Family | Operations | Matching conventions |
| --- | --- | --- |
| Descriptive | `mean`, `variance_ddof1`, `quantile_linear_p30`, `zscore_ddof1` | Arithmetic mean, sample variance (`ddof=1`), linearly interpolated 30th percentile, sample z-score (`ddof=1`) |
| Ranking | `rankdata_average_ties` | Average ranks; exactly 31 distinct integer levels |
| Association | `pearsonr_two_sided`, `spearmanr_two_sided` | Correlation coefficient and two-sided p-value |
| Hypothesis | `ttest_1samp_two_sided`, `ttest_ind_welch_two_sided`, `mannwhitneyu_asymptotic`, `jarque_bera` | One-sample null mean 0.1; Welch unequal variances; continuity-corrected asymptotic Mann–Whitney; asymptotic Jarque–Bera |
| Transform | `boxcox_lambda05`, `yeojohnson_lambda05` | Fixed lambda 0.5; parameter fitting excluded |
| Distance | `wasserstein_distance` | Unweighted, one-dimensional samples |
| Distribution | `normal_cdf_batch`, `normal_ppf_batch` | Normal location 0.5 and scale 1.5; `n` independent evaluation points; **adapted** because Sci evaluates scalars in a Rust loop and SciPy uses vectorized NumPy arrays |

Every Python and Rust case constructs `float64` inputs with the same arithmetic
formulas, integer modular patterns and ordering. The base samples are
`((37*i % 257)-128)/19 + (i/n)*0.03`; the second sample is
`(((59*i+13) % 263)-131)/23 - (i/n)*0.02`. Ranked data use
`(17*i+3) % 31`; positive data use `abs(x)+0.75`. Distribution inputs are
uniform grids over `[-4,4]` and `[0.01,0.99]`, both with inclusive endpoints.

`prepare_only` measures input creation and destruction, including the array
allocation. `compute_only` prepares the inputs outside the timed block, then
measures the public operation and the allocation/destruction of its result.
Each measured block makes `batch=4` calls by default to amortize clock overhead.
For each of 21 timed blocks after five warmups in the final paired run, the runner divides the block
time by its call count and reports the **median**, lower and upper quartile
observations (`p25_ns`, `p75_ns`) and minimum (`min_ns`). All values are
nanoseconds per invocation. Statistics are over blocks, not individual calls.

Both programs set native BLAS, OpenMP and Rayon thread limits to one. Run them
serially when the machine is otherwise idle. From the `/sci` workspace:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  RAYON_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  cargo bench --bench stats_scipy_comparison --no-default-features \
  --features stats -- --repeat 21 --warmup 5 --batch 4 \
  --out target/stats_rust_final.csv

env -u LD_LIBRARY_PATH \
  PYTHONHOME=/home/psira/.local/share/uv/python/cpython-3.13-linux-x86_64-gnu \
  PYTHONPATH=.venv/lib/python3.13/site-packages \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  RAYON_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python tests/scipy/benchmark_stats_functions.py \
  --repeat 21 --warmup 5 --batch 4 --out target/stats_scipy_final.csv
```

Both accept `--quick` for `n=128` only. CSV cases, parameters, input profiles,
phase names and measurement fields match, making an exact-key join possible.

## Interpretation and validation

`comparability=matched` indicates equivalent numerical operation and explicit
parameters, not equivalent implementation costs or bitwise identical results.
Native `Vec<f64>` and NumPy `ndarray` have different allocation, conversion
and dispatch overhead. The output layout and known backend differences are
reported in CSV metadata. Setup comparisons are always marked `adapted`:
Python constructs NumPy arrays with `np.fromiter`, while Rust collects vectors.
The distribution batch cases are likewise `adapted`; SciPy's C/NumPy vector
dispatch and Rust's per-element scalar calls measure different API forms.

## Measured results (2026-11-09)

Both full 21-repeat release runs completed, with **64 matching records each**
(16 cases × two sizes × two phases). The table reports `compute_only` medians
at 2,048 elements, in microseconds, on this host. These are public API call
times, including output creation, rather than comparisons of equivalent
inner kernels. Native library thread limits were set to one, and Rust/SciPy
benchmarks ran serially with a maximum of eight Cargo build jobs.

| Operation | Rust (µs) | SciPy (µs) | Rust / SciPy |
| --- | ---: | ---: | ---: |
| Mean | 8.07 | 177.54 | 0.05× |
| Sample variance | 12.90 | 428.52 | 0.03× |
| Linear 30th percentile | 6.22 | 19.52 | 0.32× |
| Average tied ranks | 36.81 | 140.17 | 0.26× |
| Z-score | 18.75 | 209.54 | 0.09× |
| Pearson correlation | 30.15 | 276.19 | 0.11× |
| Spearman correlation | 119.17 | 567.23 | 0.21× |
| One-sample t-test | 23.40 | 453.77 | 0.05× |
| Welch two-sample t-test | 45.00 | 727.08 | 0.06× |
| Asymptotic Mann–Whitney | 204.48 | 724.29 | 0.28× |
| Jarque–Bera | 32.09 | 534.23 | 0.06× |
| Fixed-λ Box–Cox | 44.10 | 56.31 | 0.78× |
| Fixed-λ Yeo–Johnson | 9.41 | 94.03 | 0.10× |
| 1D Wasserstein distance | 85.19 | 377.15 | 0.23× |
| Normal CDF batch (adapted) | 44.47 | 82.89 | 0.54× |
| Normal PPF batch (adapted) | 53.03 | 123.55 | 0.43× |

All 32 `compute_only` Rust cases (`n=128` and `n=2048`) had a smaller
median than their SciPy counterparts in the paired run; this does **not**
establish faster performance for the entire `sci-stats` API. The normal
CDF/PPF rows compare a Rust scalar loop against SciPy vectorized dispatch;
they are explicitly adapted. For `rankdata` at 2,048 points, repeated Rust
measurements ranged from 22.77 to 36.81 µs, illustrating run-to-run noise.
SciPy's Python dispatch overhead particularly affects short scalar calls.

The preceding Rust quantile implementation (full sort) measured 31.17 µs at
2,048 values; after exact-neighbor selection, repeated medians were 6.22 and
6.18 µs. In native single-thread `stats_tests` benchmarks, the older vs
updated implementation measured: Kendall tau 37.64→15.64 µs, Somers' D
30.52→10.00 µs, Anderson k-sample 23.46→6.56 µs, Poisson means test
38.55→23.39 µs, and BWS exact test 27.72→20.46 µs. These native comparisons
are not interchangeable with the paired cross-language matrix. Tukey HSD
with distinct comparisons remained ~1.09 µs; caching is useful only when
multiple contrasts share exactly the same studentized statistic.
Additional CodSpeed workloads cover quantile at 8,192 observations and
Kendall tau/Somers' D at 1,024 paired observations; single-thread release
smoke medians were 27.44, 118.2 and 80.66 µs, respectively.

Serial verification passed **95 `sci-stats` library tests** and **17 main
statistics integration tests**, including independent SciPy numerical
references. Full-workspace Clippy with `-D warnings`, Rust formatting and
`git diff --check` passed. The local `target/stats_rust_final.csv` and
`target/stats_scipy_final.csv` files retain individual timing records; they
are not tracked. These runs cover representative APIs rather than every
combination of functions, inputs and parameterizations in `sci-stats`.
