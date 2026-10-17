# `stats` coverage and SciPy adaptations

Native implementations use private statrs 0.19.1, Basin 1.15.1, nalgebra 0.35,
shared Sci special functions and lapjv 0.3.0 backends. Shapiro and Tukey use
private adaptations of BSD-licensed SciPy algorithms and mixture integrals;
the retained license is `crates/sci-stats/SCIPY_LICENSE.txt`. No Python runtime is used. **Main-namespace function
coverage is complete for the documented Rust signatures; full SciPy statistics
parity is not complete.** The audit against `SCIPY_FUNCTIONS.md` and
`scipy-func-list/scipy.stats.md` covers **137 of 137 scipy.stats function names**,
all **7 contingency names**, and all **4 QMC utility function names**.
Distribution coverage is **30 of 149 callable families** with evaluation methods;
this does not claim their entire object-oriented SciPy method surface.

The coverage audit fails for exported functions without committed reference
cases and checks main-namespace inventory completion. There are no placeholder
exports. Run `python tests/scipy/audit_statistics.py` to reproduce the counts.
SciPy compatibility namespaces (`stats`, `morestats`, `distributions`, etc.)
repeat canonical names; no separate legacy Rust modules are added. The inventory
lists no Cython-only statistics namespace. Masked statistics are not implemented:
overlapping mstats names can use the canonical finite, unmasked APIs, while
mask-specific functions and missing-data dispatch remain gaps. Right-censoring
is supported through explicit logrank event flags.

Statrs 0.19.1 additions used by Sci:

- `Distribution::Levy` evaluates SciPy `levy` density/log-density, CDF/SF and
  quantiles with native location/scale parameters; it also works with existing
  truncation, order-statistic and continuous-null APIs. Moments, sampling and
  fitting remain unsupported.
- Chi and inverse-gamma quantiles first use the new safeguarded Newton inverses.
  Sci validates the smaller-tail probability residual before accepting a result;
  its bounded bisection remains the fallback. Endpoints and domains are unchanged.
- The new generic inverse remains unsuitable as a blanket replacement for Sci's
  inversion, because it does not report convergence and has different work limits.
- New upstream hypothesis tests overlap Sci's existing APIs; their direct moment
  sums and CDF-subtracted tails do not preserve Sci's extreme-scale/tail behavior.
  KDE and online accumulators are not exposed by this change.

## Supported native groups

- Descriptive summaries, arithmetic/geometric/harmonic/power means, central
  moments, skewness/kurtosis with corrections, quantiles, L-moments through order
  four, k-statistics through order four, robust spread and trimming, z-scores.
- Rank methods, percentile ranks, tie correction, empirical CDF, false discovery
  control, circular/directional summaries, histogram and Cartesian bin reductions.
- Pearson, Spearman and point-biserial correlations; regression; one-sample,
  paired, pooled and Welch t-tests; ANOVA; Bartlett and Levene tests; Kruskal and
  Friedman tests; rank-sum tests; Jarque–Bera, exact binomial and Fisher tests;
  chi-square/power-divergence tests and all five p-value combination methods.
- Native Siegel repeated-median and Theil–Sen regression (with approximate slope
  confidence limits), skewness/kurtosis and omnibus normality tests, O’Brien
  transforms and Alexander–Govern means tests. Ansari–Bradley, Brunner–Munzel,
  Fligner–Killeen, Mood scale/median tests, Kendall tau-b/tau-c, Mann–Whitney,
  Wilcoxon signed-rank and Page trend tests; exact binomial quantile tests.
- Box–Cox and Yeo–Johnson normality plot data on explicit bounded parameter grids.
- Box–Cox/Yeo–Johnson transforms and profile likelihoods for supplied lambda;
  expectiles; Wasserstein-1 and energy distances; Vasicek differential entropy;
  bounded sigma clipping; probability-plot data; one/two-sample KS statistics.
- Two-dimensional contingency margins, expectations, Yates-corrected independence
  tests, three association coefficients, sample odds ratio, relative risk, crosstabs.
- All four QMC discrepancy measures, centered discrepancy updates, affine scaling,
  minimum Euclidean separation and mean Euclidean minimum-spanning-tree edge length.

## Backend use for the preceding native rank/normality group

These 19 function algorithms are native Rust. They add no dependencies and
never invoke Python. Their use of existing external libraries is explicit:

| Functions | Existing private backend use |
| --- | --- |
| `normaltest`, `obrientransform`, `siegelslopes` | None: standard library and native Sci helpers only |
| `skewtest`, `kurtosistest`, `ansari`, `mood`, `mannwhitneyu`, `wilcoxon`, `kendalltau`, `page_trend_test` | statrs 0.19.1 normal tails |
| `brunnermunzel` | statrs 0.19.1 Student-t or normal tails |
| `alexandergovern`, `median_test` | statrs 0.19.1 chi-square tails |
| `fligner` | statrs 0.19.1 normal quantiles and chi-square tails |
| `theilslopes`, `boxcox_normplot`, `yeojohnson_normplot` | Existing normal quantile APIs backed by shared XSF special functions |
| `quantile_test` | statrs 0.19.1 binomial tails |

No new algorithm calls sci-special or nalgebra. Those remain existing statistics
module dependencies for other functions. Public results expose native Sci types;
backend distribution objects remain private.

The coverage count means a tested, documented functional signature, not complete
SciPy overload parity. Exact, automatic and permutation rank-test dispatch remain
unsupported: Ansari, Kendall, Mann–Whitney, Wilcoxon and Page implement their
explicit asymptotic methods. Kendall uses bounded direct pair counting with at
most 4096 observations; Siegel and Theil–Sen accept at most 2048 pairs. Wilcoxon
accepts supplied differences and all three zero conventions. Page accepts raw
rectangular rows and an explicit predicted-rank permutation. Median tests support
Pearson divergence only; Fligner supports mean or median centers. Quantile tests
return the statistic's count convention, but no confidence-interval object.

Sci uses the continuous zero-skewness limit Z=0, instead of SciPy 1.18.1's
nonzero substitution. Symmetric-sample references use the exact zero third-moment
identity and independent SciPy normal/chi-square tails. Constant Fligner scores
and constant Mood scores give NaN rather than using a spurious roundoff variance.
Alexander–Govern tests actual zero spread, avoiding SciPy's offset-dependent
near-constant cutoff. Kurtosis and omnibus normality calibration is unreliable
for n≤20; the accepted small samples are documented rather than warning at runtime.

## Additional main-namespace coverage

The additional 36 supported signatures complete main-namespace coverage from
101 to 137 names. Every name has checked-in references and public API tests:

- `bayes_mvs` and `mvsdist`: normal-population posterior expectations, equal-tail
  intervals and quantile evaluations. Frozen objects are adapted to named Sci
  summaries and `[mean, variance, standard deviation]` quantiles. Constant
  samples are rejected. SciPy's Gaussian approximation switches at n>1000.
- `boxcox_normmax`, `yeojohnson_normmax`, `ppcc_max`, `ppcc_plot`: hard-bounded
  searches and explicit shape grids. Basin Brent stays private; fixed finite
  bounds replace SciPy's expandable bracket. PPCC supports Student-t, chi-square,
  gamma, Weibull and inverse gamma; the default Tukey-lambda family is absent.
  Search inputs/iterations are bounded, and nonfinite objectives return errors.
- `spearmanrho`, `weightedtau`, `somersd`, `chatterjeexi`: canonical univariate
  correlations. Spearman rho reuses ranks and Pearson probabilities. Weighted
  tau supports hyperbolic additive/product weights and all importance-ranking
  conventions, without arbitrary weighers. Somers supports paired observations
  without the table overload/result. Chatterjee rejects tied x instead of
  choosing an arbitrary/random ordering. Weighted tau has no p-value (NaN).
- `logrank`: nonnegative event times with explicit boolean right-censor flags,
  including failure/censor ties. Left/interval censoring and delayed entry are absent.
- `epps_singleton_2samp`: explicit positive frequencies and private nalgebra
  SVD pseudoinverse, including duplicate-frequency rank reduction.
- `anderson`, `anderson_ksamp`, `cramervonmises`, `cramervonmises_2samp`:
  fitted normal/zero-location exponential Anderson critical-value diagnostics;
  midrank/right multi-sample Anderson calibration; continuous-null one-sample
  Cramér–von Mises; exact/asymptotic two-sample Cramér–von Mises. Multi-sample
  Anderson p-values retain SciPy's [0.001,0.25] tabulated-range clipping. Cramér–von
  Mises series use shared private Bessel/log-gamma functions with 256-term caps
  and SciPy's 1e-7 term cutoff. Exact two-sample work requires combined n≤20.
- `permutation_test`, `monte_carlo_test`, `bootstrap`, `bws_test`: fallible scalar
  callbacks, exact independent partitions, supplied null draws, and supplied
  bootstrap index schedules. The caller owns RNG generation/independence. Exact
  permutation/BWS inputs total at most 20. Bootstrap supports percentile/basic
  intervals only; BCa, pairing, axes, vectorization and random permutation dispatch
  are absent. Plus-one Monte Carlo and exact permutation tail conventions match SciPy.
- `fit`, `goodness_of_fit`: analytic normal, zero-location exponential and uniform
  fits, with fitted native distributions reusable by density/CDF/quantile APIs.
  Goodness-of-fit refits every supplied null draw and supports KS/Cramér–von Mises;
  its normal standard deviation uses ddof=1, matching SciPy. Generic fit bounds,
  optimizers, other families and fixed-parameter dictionaries are unsupported.
- `estimated_cdf`: all nine Hyndman–Fan scalar estimated-CDF conventions, with
  explicit ties and singleton behavior. Linear singleton equality is NaN.
- `sobol_indices`, `power`: scalar Saltelli 2010 indices from supplied A/B/hybrid
  evaluations and rejection fractions from fallible tests on supplied alternative
  draws. Multi-output sensitivity, sampling engines, bootstrap result objects,
  vectorization and sample-size grids are absent.
- `shapiro`, `tukey_hsd`: private SciPy-based AS R94 and studentized-range algorithms.
  Shapiro accepts 3..=5000 samples after shared shifted/scaled normalization;
  constants return NaN, matching SciPy 1.18.1. Tukey returns Sci matrices of signed
  mean differences and adjusted p-values for 2..=16 groups, equal-variance only.
  Two-group probabilities reuse the exact Student-t identity; Games–Howell and
  confidence-interval objects are absent. Native summaries avoid backend overflow.

- `make_distribution`, `truncate`, `order_statistic`: native continuous
  distribution descriptions accepted by the existing `pdf`, `logpdf`, `cdf`,
  `sf`, `ppf` functions. Custom distributions use checked fallible scalar
  function pointers; normalization/monotonicity remain caller obligations.
  Truncation supports finite bounds with smaller-tail normalization and bounded
  quantile inversion. Order-statistic CDFs reuse beta probabilities; short
  integer-beta recurrences preserve endpoint normalization at n=1048576.
  Discrete/nested transforms, capturing callbacks and object methods are absent.
- `barnard_exact`, `boschloo_exact`, `poisson_means_test`: unconditional tests.
  Binomial tables total at most 64; u128 coefficient sums and globally bounded
  Bernstein subdivision maximize nuisance probability within 1e-12. This
  replaces SHGO sampling, with 65536-node/depth-64 caps. Boschloo uses exact
  integer hypergeometric tails to avoid general-backend PMF accumulation above
  one. Poisson counts total at most 2048, diff≥0; explicit joint support tails
  match SciPy's 1e-10/1−1e-16 cutoffs and work is capped at 16777216 points.
- `dunnett`: pooled normal treatment-versus-control comparisons, 1..=8 treatments
  and residual df 2..=256. Single comparisons reuse Student-t; deterministic
  common-normal/chi-square mixture quadrature replaces SciPy RNG calibration.
  128/256-order rules must agree within 5e-5 or evaluation fails. Rules are
  cached without caller data in a bounded thread-safe cache. Confidence
  intervals and unequal-variance multiple comparisons are absent.
- `multiscale_graphcorr`: paired Euclidean point rows with a full local
  correlation map, smoothed optimal scale and supplied permutation schedule.
  Native rank/prefix covariance and four-neighbor smoothing follow the MGC
  definition; beta thresholds reuse shared functions. Inputs have 5..=128 rows,
  dimensions≤64 and `(B+1)*n²≤16777216`. Supplied permutations use conservative
  floating-point tie comparisons. Two-sample/data-distance callback overloads,
  automatic RNG and parallel workers are absent.
- `wasserstein_distance_nd`: unweighted Euclidean transport with bounded
  replication to lcm(n,m)≤256, reduced exactly to private lapjv assignment.
  Native scaled coordinates avoid intermediate distance overflow. Arbitrary
  weights/metrics are absent; shape, coordinate and cubic-work limits are explicit.

The public APIs specify their equations, units, shapes, special values, numerical
limits, adaptations and examples. Callback errors propagate unchanged; invalid
resampling inputs/work limits are rejected before callbacks. New property tests
cover CDF/quantile inversion, posterior variance/standard-deviation relationships,
Spearman reuse, Shapiro scale invariance, Tukey/t-test identity, permutation
exchange symmetry, bootstrap reflection, and exact additive-model Sobol indices.

## Distribution evaluation

Univariate parameters are native `Distribution` variants, passed to free `pdf`,
`logpdf`, `cdf`, `sf`, and `ppf` functions. The density functions also evaluate
integer probability masses, explicitly documented. The backend stays private.
Supported SciPy families are:

`norm`, `beta`, `gamma`, `chi2`, `chi`, `t`, `f`, `expon`, `cauchy`, `laplace`, `lognorm`, `uniform`, `weibull_min`, `pareto`, `invgamma`, `gumbel_r`, `triang`, `erlang`, `randint`, `bernoulli`, `binom`, `poisson`, `geom`, `nbinom`, `hypergeom`.

Joint density/log-density functions cover `multivariate_normal`, `multivariate_t`,
`dirichlet` (interior simplex only), and `multinomial` (mass/log-mass). Categorical
mass/CDF is an additional arbitrary-weight helper. Shared normal functions retain
their original signatures. Automatic random sampling, generic distribution fitting, distribution expectations,
confidence-interval object methods, generic moments, frozen distributions, and
other distribution families are unsupported. Integer uniform uses exclusive high;
gamma/exponential/Erlang use scale, converted to statrs rate; lognormal uses
log-mean/log-standard-deviation; chi degrees of freedom are integer because of
the backend. Multivariate matrices use one Vec per row, interoperable with algebra.

## Signature and numerical adaptations

- Arrays are one-dimensional unless explicitly documented as matrices or Cartesian
  point sets. There are no axis, broadcasting, weights (except Stouffer), dtype,
  device, plotting, sparse or mask overloads. Censoring is explicit in logrank;
  supported permutation/resampling signatures are described above.
  NaN and infinity observations are rejected; scalar distribution observations
  may be infinite. Undefined constant-sample statistics preserve documented NaNs.
- Observation slices are bounded by 1,048,576; moment powers by 64; multivariate/QMC
  dimensions by 64; binning dimensions by 16; pairwise QMC work by 16,777,216.
  Distribution shapes/counts are capped at 1,048,576. Fisher/binomial exact tests
  and hypergeometric population are capped at 65,536. Sigma clipping has 128 passes,
  expectiles at most 21 balance evaluations after sorting, continuous quantiles at most 6144 bounded
  CDF/survival evaluations, and discrete quantiles at most 106 evaluations.
  Normal CDF/SF/PPF reuse shared `ndtr`/`ndtri`, with scale-safe standardization
  and fused quantile location/scale arithmetic. Normal densities have a log-space
  fallback for standardized tail underflow. Epps–Singleton and multi-sample
  Anderson SVDs each have a 10,000-iteration limit and explicit failure errors.
- Statistical alternatives, rank conventions, FDR, centers, bin reductions, and
  discrepancy types are native enums. Results have named fields; tuple orders,
  matrix shapes, bin linearization and theoretical/observed ordering are documented.
- Quantiles use linear interpolation only. Trimming by proportion returns sorted
  observations. Trimmed moments use explicit value limits with inclusive endpoints.
  Bin edges are explicit; rightmost edges use exact equality, without SciPy's
  decimal-rounded edge inclusion tolerance. Histogram ranges are explicit.
- Kurtosis has explicit Fisher/Pearson and bias flags; SEM and variance corrections
  are explicit. L-moments are scalar-order, orders 1–4; differential entropy is
  Vasicek only. Expectiles and empirical distances are unweighted.
- ANOVA is the equal-variance test. Levene supports mean/median centers. Spearman
  and rank-sum probabilities use their documented asymptotic approximations.
  Fisher delegates to statrs, whose two-sided mass tie threshold is 1e-4 rather
  than SciPy's tighter threshold; near-tie tables can differ.
- KS one-sided one-sample tails reuse the special-function Smirnov backend;
  two-sided one-sample tails are asymptotic (SciPy method="asymp"). Two-sample
  tails use the limiting null, without SciPy's finite-sample correction/exact test.
  `kstest` supports the two-sample overload. Continuous null distributions with
  independently supplied parameters are required; ties invalidate ordinary KS
  probability calibration. `probplot` returns Filliben plot data (fit=false).
- Chi-square frequencies need matching totals and positive expectations. Power
  divergence evaluates the finite zero-count limit for lambda > -1, while SciPy's
  literal formula can return NaN for lambda=-0.5. Negative-binomial p=1 is a point
  mass at zero; geometric p=1 is a point mass at one. SciPy 1.18.1 geom.ppf returns
  zero for interior probabilities at p=1; the native API uses the defining limit.
- Densities/masses at infinite observations return zero and log-density -infinity,
  including SciPy paths that return NaN from inf-inf. Discrete CDF arguments are
  floored, including hypergeometric, whose SciPy 1.18.1 noninteger path returns NaN.
  Discrete ppf(0) is support minimum minus one; unbounded ppf(1) is infinity,
  including degenerate Poisson/geometric/negative-binomial support conventions.
- A constant transformed sample has infinite Box–Cox/Yeo–Johnson likelihood.
  SciPy sometimes yields finite likelihoods from spurious roundoff variance;
  those fixtures use the zero-variance identity. SciPy's tiny-lambda Box–Cox llf
  log-variance path loses about 1e-8: references use the stable SciPy transform
  and independent long-double NumPy variance in the defining likelihood.
- Native scaled/compensated mean, variance, standard deviation and central moments
  reduce avoidable overflow. Multinomial log masses use statrs log factorials in
  the defining formula: the backend's ordinary multinomial coefficient overflows,
  and its log-mass path also evaluates 0*log(0) at zero-probability categories.
- QMC geometry supports Euclidean metric only; duplicate points are rejected for
  MST because SciPy's sparse graph omits zero-length edges. QMC engines and sampling
  classes are unsupported. Conditional odds-ratio MLE and confidence intervals are
  unsupported; odds_ratio implements kind="sample". Contingency tables are 2D only.

## Verification

`tests/scipy/statistics_cases.json` contains inputs separately from expected
results. `generate_statistics.py` pins SciPy 1.18.1 and NumPy 2.5.4 and emits
5101 cases into `stats_inputs.csv` and `stats_expected.csv`. Ordinary Rust tests
read the CSV files; they require no Python. The fixture generator explicitly
identifies independent mathematical references for the documented corner
adaptations and KS limiting probabilities. References never use Sci/statrs.

The ordinary combined tolerance is `2e-10 * (1 + abs(expected))`, allowing statrs
error-function/beta/gamma approximations and cancellation in log-factorial
multinomial probabilities at counts up to one million. All discrepancies were
investigated before accepting these tolerances. Bounded transform searches use
2e-6 and PPCC maxima 2e-5 combined tolerances because flat binary64 objectives
limit position agreement. Epps–Singleton uses 2e-9: a tested covariance condition
number of 1.4e8 produces about 3e-9 differences between NumPy SVD and direct solves.
Tukey adjusted probabilities use the ordinary 2e-10 absolute/relative tolerance.
The direct survival mixture integral uses composite Gaussian orders 32 and 64
with a 2e-10 absolute convergence check and at most 358400 integrand evaluations
per pair. Truncation tails are below 1e-16 over the supported domain; the order
comparison is an error estimate, not a rigorous accuracy bound. Additional
`range_inputs.csv` and `range_expected.csv` fixtures cover residual degrees of
freedom through 65533 and the 16-group and combined-count boundaries. Shapiro
passes the ordinary tolerance, including n=3,11,12,25,100,5000 and extreme units.
Dunnett probabilities use 5e-5 for fixed-order mixture quadrature; statistics
retain the ordinary tolerance. References use SciPy multivariate-t CDF with
one million integration points and a fixed RNG seed, reducing SciPy's default
Monte Carlo noise. MGC maps/scales and supplied null draws are independently
evaluated by SciPy; custom distribution callbacks use SciPy beta references.
NaN and signed infinities are
compared explicitly; signed zeros and ties have direct property checks. Public
APIs are tested for input boundaries, degeneracies, invalid shapes/domains,
output lengths and ordering, and work limits. Independent checks cover
normal quantile/CDF round trips, z-score moments, QMC incremental/recomputed
agreement and scaling inversion, multinomial/binomial identity, independent
normal product densities, contingency margins, and FFT/complex/stats composition.

## Remaining full-parity gaps

No main-namespace function names remain unimplemented. This means tested native
signatures with the explicit adaptations above, rather than all SciPy overloads
or identical algorithm dispatch. Contingency and QMC utility function names
are also complete. Legacy namespaces reuse these canonical algorithms.

The 31 mstats-only function names, all masked-input behavior and compatibility
aliases of mask-only functions remain unsupported. Of 73 mstats names, 42 share
unmasked canonical algorithms; this is not a claim of masked-array support.
The other 120 callable distribution families, generic distribution methods,
QMC/sampling engines and SciPy class APIs remain gaps. The coverage audit prints
these distinct totals, so function-name completion cannot imply full distribution
or namespace parity. Unsupported overloads and numerical work/accuracy limits
are documented with each public function and summarized above.

## Audit regression references

Mean uses sign-balanced compensated summation before division, with a scaled
fallback for large uncancelled sums. Central moments shift observations before
scaling whenever subtraction is finite; opposite extreme signs fall back to
scaled centering. Variance, standard deviation, higher moments, skewness,
kurtosis, correlation, regression, and standardized scores reuse this centering
convention. `zscore`, `zmap`, and `gzscore` divide in normalized coordinates,
preserving small differences at large offsets and avoiding intermediate spread
overflow or subnormal rounding. A constant reference retains Sci's NaN/equal
score and signed-infinity/unequal score behavior; unrepresentable scores from
a nonconstant reference are evaluation errors.
Regression rescales coefficients without first overflowing a scale ratio.

`tests/scipy/generate_audit_regressions.py` generates independent references from
input-only JSON using SciPy 1.18.1 / NumPy 2.5.4. Exact rational definitions on
binary64 inputs supplement SciPy where direct evaluation loses cancellation or
cannot represent intermediate results. Tiny outputs use relative comparisons
and only one minimum subnormal of absolute tolerance. These are numerical
regressions within existing APIs; inventory coverage is unchanged.
Standardized-score references use exact rational centering and high-precision
square roots, with eight binary64 epsilons of absolute tolerance for rounding
near zero and `5e-13` relative tolerance elsewhere. Geometric-score references
use NumPy logarithms before applying this definition.
