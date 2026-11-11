# Sci 0.3.0: other-subcrate performance audit

This pass investigates numerical hot paths outside `sci-stats` and
`sci-special`. The source baseline is `origin/dev` at `51fc9e6`; the candidate
is `perf/other-subcrates-0.3.0`. All existing crate versions are unchanged.

## Measurement protocol

- Build the exact same Divan benchmark definitions against both source trees,
  **in separate Cargo target directories**. Reusing a target directory across
  different source trees with identically named/versioned workspace packages
  can reuse the wrong dependency artifacts. In particular, an early shared-
  target baseline inadvertently contained a candidate's quadrature cache;
  all measurements below were repeated using a fully isolated baseline build.
- Use release (`cargo bench --no-run -j 4`), one Divan thread, a single pinned
  CPU core (`taskset -c 4`), with one Cargo build and at most four build jobs
  at a time. Run three alternating old/new passes with 25 samples each and a
  0.15-second maximum sampling interval. Report the median of the three
  per-pass medians, rounded to three significant digits.
- Interpret each number as *time per benchmark invocation*, not normalized
  time per scalar. Benchmarks are listed in the corresponding `benches/*.rs`;
  shape, parameter and input-generation choices are part of their definitions.
  Timings are hardware-specific, and shorter measurements carry more noise.

| Crate | Benchmark | Baseline | Candidate | Speedup |
| --- | --- | ---: | ---: | ---: |
| cluster | `cut_tree_one_count` | 5.15 ms | 30.8 µs | 167× |
| cluster | `large_tree_benches::leaders` | 1.74 ms | 32.0 µs | 54.6× |
| interpolate | `bisplev_dense_grid` | 397 µs | 32.2 µs | 12.3× |
| interpolate | `interpn_singleton_axes` | 292 µs | 24.1 µs | 12.1× |
| graph | `reconstruct_distances(64)` | 872 µs | 123 µs | 7.09× |
| sparse | `canonical_csr_conversion(512)` | 75.3 µs | 11.8 µs | 6.40× |
| graph | `reconstruct_distances(128)` | 3.52 ms | 620 µs | 5.67× |
| cluster | `cut_tree_all` | 6.48 ms | 1.44 ms | 4.50× |
| cluster | `fcluster_maxclust` | 102 µs | 26.3 µs | 3.88× |
| sparse | `csr_matmul(128)` | 188 µs | 49.3 µs | 3.81× |
| signal | `general_cosine_eight_harmonics` | 22.5 µs | 5.99 µs | 3.75× |
| optimize | `rosen_hess_prod_large` | 987 µs | 273 µs | 3.61× |
| signal | `sosfilt_large(16)` | 542 µs | 185 µs | 2.93× |
| signal | `coherence_many_segments(512)` | 371 µs | 136 µs | 2.73× |
| signal | `fftconvolve_large(32)` | 515 µs | 215 µs | 2.39× |
| signal | `welch_many_segments(128)` | 139 µs | 66.6 µs | 2.08× |
| linalg | `eigh_tridiagonal(256)` | 6.09 ms | 3.02 ms | 2.02× |
| filter | `kalman_predict_spd_dimension(32)` | 58.7 µs | 36.2 µs | 1.62× |
| fft | `irfftn_strided_axis` | 56.6 µs | 36.7 µs | 1.54× |
| fft | `rfftn_strided_axis` | 51.0 µs | 33.7 µs | 1.51× |
| spatial | `cdist_cosine_many_pairs` | 673 µs | 561 µs | 1.20× |
| constants | `find` | 12.2 µs | 10.7 µs | 1.14× |

## Changes deliberately not retained

- Uniform-spacing integration and cached fixed quadrature did not reliably
  improve benchmarks. Both source implementations were restored byte-for-byte
  to the baseline; the corresponding reference tests and regression benchmarks
  remain.
- An output-indexed `upfirdn` loop was slower for the benchmarked up/down
  ratios, so the original input/tap accumulation order was restored.
- Additional Toeplitz and QR materialization shortcuts were inconclusive and
  reverted, retaining only the measured tridiagonal eigenpair improvement.
- Cached 2-D `griddata` triangle transforms lacked a consistent benefit and
  were removed; the independent regression benchmark remains.
- Wide sparse products use the original ordered-map accumulator; the faster
  dense-marker workspace applies only to small result matrices. The original
  transpose cursor and its accumulation order were restored.

Remaining minor variations in unmodified paths (for example, large sparse
products, short Toeplitz calls, or Dijkstra on 128 vertices) should not be
interpreted as algorithmic regressions without repeated matched-hardware
profiling. The audit also retains high-precision, structural and boundary tests
for affected numerical kernels rather than relying solely on speed comparisons.
