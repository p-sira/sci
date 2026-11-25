# Filter coverage and numerical conventions

`sci-filter`, exposed as `sci::filter` with the `filter` feature, provides Bayesian
state estimation. Its ten numerical exports are implemented, documented, and
exercised through the public facade. Smelly 0.2.0 supplies the linear, extended,
and scaled unscented Kalman algorithms, particle propagation, and weighted
moments. Effective sample size uses the elementary sum-of-squares identity. Backend types remain
private; the API uses f64, shared `Matrix`, `Error`, and `Result`, and owned
`Gaussian`, `Update`, `ParticleSet`, and `SigmaPoints` records.

## Inventory audit and module boundary

Reviewed `SCIPY_FUNCTIONS.md` and its linked
[`scipy.signal` inventory](scipy_function_list/scipy.signal.md). There is no
`scipy.filter`, `scipy.filters`, or Bayesian Kalman/particle-filter namespace.
The signal inventory's sample-filtering, convolution, FIR/IIR design, window,
spectral, Savitzky–Golay and Wiener functions stay in `sci-signal`, which already
implements these public names. Signal compatibility namespaces repeat the same
algorithms, and there are no Cython-only Bayesian-filter entries.
This module is an independent extension rather than additional SciPy coverage.
No signal functions or compatibility aliases are duplicated in `filter`.

| Public function | Supported operation | Reference |
| --- | --- | --- |
| `kalman_predict` | x⁻ = F x; P⁻ = F P Fᵀ + Q | Independent SciPy matrix products |
| `kalman_update` | y=z−Hx; K=PHᵀ(HPHᵀ+R)⁻¹; x⁺=x+Ky | SciPy solve and independent Joseph covariance |
| `extended_kalman_predict` | x⁻=f(x); P⁻=J P Jᵀ+Q | Analytic model Jacobians and SciPy products |
| `extended_kalman_update` | Nonlinear h(x), analytic H at prior mean | SciPy solve and Joseph covariance |
| `unscented_kalman_step` | Merwe prediction, regenerated sigma points, observation update | Independent SciPy Cholesky/solve and weighted transforms |
| `particle_predict` | Caller-provided stochastic or deterministic process model | Independently evaluated model formulas |
| `particle_update` | w⁺ ∝ w exp(log likelihood) | SciPy logsumexp, with common offset removed first |
| `particle_estimate` | Weighted population mean/covariance | Independent NumPy weighted products |
| `effective_sample_size` | 1/Σ normalized(w)² | Independent sum-of-squares identity |
| `systematic_resample` | Explicit-offset inverse CDF sampling | Independent NumPy searchsorted |

## Supported conventions and backend adaptations

- Functions consume borrowed estimates and return owned results. A failure leaves
  input records unchanged; callback side effects, including RNG draws, persist.
- State/observation order and units are caller-defined. Noise is independent,
  additive covariance per step in squared coordinate units. Explicit time steps
  and controls belong in transition matrices or model callbacks.
- Linear and extended filters have separate prediction and update functions.
  EKF process and observation callbacks have supplied Jacobians, each evaluated
  once at the prior mean. Nonlinear process means are applied directly while
  Smelly propagates covariance through the Jacobian.
- UKF is a complete prediction/update step, using Merwe weights with
  c=α²(n+κ), λ=c−n, w₀=λ/c, wᵢ=1/(2c), and covariance correction
  1−α²+β. Smelly generates sigma points from the predicted covariance **after
  adding Q**, so observation and cross covariance include process noise. The
  independent generator follows this convention; FilterPy's usual cached-point
  update differs. Each model runs at most 2n+1 times, with ordinary Euclidean
  coordinates/residuals and pure callbacks. Defaults are α=.5, β=2, κ=0.
- Covariance inputs are symmetric positive semidefinite. UKF initial and predicted
  covariances must be positive definite for Cholesky. Innovations must be
  nonsingular; no pseudoinverse or jitter is applied. Small covariance asymmetry
  is averaged, with tolerance 128 ε n relative to the largest entry. Negative
  eigenvalues within that roundoff tolerance are accepted without projection;
  scaled bounded eigensystems check larger violations. Subtractive Kalman
  covariance updates can lose definiteness on ill-conditioned inputs; detected
  output violations return numerical errors.
- All numerical inputs and model outputs must be finite. Particle log likelihoods
  may also be negative infinity for impossible observations; NaN, positive
  infinity, and zero total posterior probability are errors. Both prior weight
  normalization and likelihood normalization remove large common scales first.
  Tiny relative weights may underflow to zero; enormous finite log offsets do
  not erase prior weight ratios. Zero-prior particles remain impossible even if
  their likelihood exceeds every positive-prior particle's likelihood.
- Particle states are one vector per row, with explicit nonnegative relative
  weights of positive mass. No initialization distribution is assumed. Generate
  initial particles and process samples with caller-controlled randomness;
  `particle_predict` accepts FnMut to support a seeded generator and validates
  each model output. Smelly takes contiguous particle columns privately; Sci
  preserves the public row ordering and normalized weights.
- Particle weighting, effective sample size, and systematic resampling are native
  bounded elementary operations. Native log normalization removes common
  likelihood offsets before combining them with retained prior logarithms,
  preserving prior ratios even near ±1e308 and recovering underflowed priors.
  Resampling retains the documented division and zero-mass boundary conventions.
  Smelly moments receive the normalized linear weights through its likelihood
  update; log-space fallback can introduce additional binary64 roundoff for tiny
  relative weights. No backend RNG is used.
- Resampling selects the first cumulative mass strictly greater than
  (i+offset)/N, skipping zero weights. It returns exactly N rows with weight 1/N.
  The offset lies in [0,1); a uniform random draw is the caller's responsibility.
  Resampling is explicit, with no automatic ESS threshold. Moments are population
  moments, with no unbiased sample correction.

## Work limits and accuracy

State and observation dimensions are 1..=32. There are 1..=65536 particles,
at most 1048576 particle coordinates, and particle count × dimension² must not
exceed 16777216. All particle functions enforce these bounds before allocation.
Covariance eigenvalue validation has at most 10000 iterations and works on a
scaled matrix to avoid validation overflow. User callback work is not bounded by
Sci. Algorithms use f64, no uniform accuracy guarantee applies to ill-conditioned
systems, strong nonlinearities, tiny sigma spreads, or weights near underflow.
UKF requires c ≥ 1e-6 n, finite α>0, finite β≥0 and finite κ; smaller spreads
are rejected to bound weight cancellation. Arithmetic overflow is an error.

## Verification

`tests/scipy/filter_cases.jsonl` stores inputs only. The generator
`tests/scipy/generate_filter.py` pins NumPy 2.5.4 and SciPy 1.18.1, and generates
101 checked-in cases into separate input and expected CSVs. SciPy has no direct
Bayesian-filter equivalent, so references implement documented identities using
SciPy solves, Cholesky factors, and logsumexp; no backend or Sci output generates
expected values. Rust compares all values, shapes, coordinate ordering and
innovations. The combined tolerance 2e-11 × (1+|reference|) allows roundoff from
small-spread UKF cancellation and independent matrix multiplication/solve orders;
straightforward scalar formulas agree much more closely. Exact zero posterior
weights and deterministic resampling row order are also checked independently.

Properties cover linear-model agreement between KF/EKF/UKF over repeated steps,
observation variance reduction, weight-scale invariance, translation of moments,
zero-weight exclusion, and ESS after resampling. Invalid-input/callback tests cover
nonfinite values, indefinite/asymmetric covariance, singular observations, wrong
shapes, scaling and work limits, and unchanged input records. Interoperability
checks use shared matrices with `sci::linalg` and moment definitions with
`sci::stats`. Ordinary Rust tests need no Python.

Run `python3 tests/scipy/audit_filter.py` to verify fixture coverage and inventory
boundary, and `uv run tests/scipy/generate_filter.py` to regenerate references.
The workspace minimum Rust version is 1.89, matching Smelly's nalgebra 0.35.

## Remaining scope

The requested linear, extended, unscented, and particle filter families are
implemented within the conventions above. RTS/state-estimation smoothing,
information/square-root/cubature/ensemble filters, nonadditive noise, manifold
residuals, missing observations, batch filtering, automatic Jacobians, implicit
particle distributions/RNGs, and resampling schemes other than systematic are
not currently exported. Signal-processing filters remain available in `signal`.

## References

- [Särkkä and Svensson, Bayesian Filtering and Smoothing, second edition](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf).
- [Smelly 0.2.0](https://docs.rs/smelly/0.2.0/smelly/).
- [Douc, Cappé and Moulines, Comparison of Resampling Schemes for Particle Filtering](https://arxiv.org/abs/cs/0507025).
