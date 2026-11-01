# Numerical quality standard: evidence behind the `sci-*` name

**Status:** proposed, repository-wide quality contract. Existing
[development guidance](development.md), [conventions](conventions.md), and
[`AGENTS.md`](../AGENTS.md) already require many of these practices. This
document makes the intended standard explicit; it does not certify all
current Sci functions or introduce a new passing CI gate by itself.

## Minimum evidence for a new or materially modified algorithm

| Area | Required evidence or disclosure |
| --- | --- |
| Mathematical contract | Definition, parameterization and units, shapes, precision, ordering, supported domains and branches; meaningful reference or derivation |
| Independent accuracy | Cases evaluated by an implementation-independent source, analytical identity, high-precision method, or a source with explicitly documented limitations |
| Numerical extremes | Boundaries, zeros and poles, subnormal inputs, large magnitudes, cancellation, conditioning and overflow/underflow where relevant |
| Failure and resource contracts | Invalid inputs, nonfinite values, nonconvergence, errors, maximum work or allocation, and guarantees about terminating bounded routines |
| Mathematical invariants | Recurrences, conservation laws, symmetry, monotonicity, reconstruction, inverse/round-trip and cross-module compatibility where applicable |
| Types and reproducibility | Supported `f32`/`f64` or other types, conversion behavior, random seeds, ordering, and relevant platform/architecture variation |
| Performance | Representative benchmarks, work/space complexity, regression review and documented accuracy-versus-speed tradeoffs |
| Provenance | Inputs, expected values, reference versions, Sci revision, reproducible generators and tolerance rationale |

**Reference independence matters.** Comparing Sci to its private algorithm
backend is useful for verifying a wrapper but insufficient to validate the
backend's mathematics. Comparing two algorithms that share the same numerical
implementation is not independent corroboration. SciPy is one valuable
external comparison—not a requirement for complete API parity and not an
infallible oracle, particularly in extreme domains.

Use absolute and relative tolerances justified by conditioning and scale.
Near zero or a sign change, relative error alone can be misleading; track
absolute error and appropriate invariant residuals. Record exact expectations
for integer, shape and error contracts. Never silently relax tolerances to
make a failing test pass.

## Two complementary evidence pipelines

**In-repository tests** are the fast, required regression layer. Sci has
checked-in independent fixtures, deterministic Rust integration/unit tests,
properties, documentation examples, feature-isolation checks, coverage and
CodSpeed benchmarks. Ordinary Rust tests must not require Python at runtime.
Track what is *exercised* separately from what is *numerically accurate*:
line coverage and function-inventory coverage alone prove neither accuracy
nor correctness on unseen inputs.

**[`sci-testing`](https://github.com/p-sira/sci-testing)** is the external,
version-pinned comparative and exploratory evidence layer. It maintains
independent reference values as well as Sci/competitor outputs, detailed case
metadata, exceptions and provenance. Its existing presentation uses Sci as
the baseline for *inter-library differences*; **agreement with Sci must not
be interpreted as accuracy**. Evaluate Sci against independent reference
values when making accuracy claims. See the
[sci-testing 0.3 modernization plan](sci-testing-roadmap.md).

The two pipelines should share **documented numerical definitions and case
provenance**, without blindly duplicating millions of test records inside
the Sci crate. Promote a compact set of high-value external findings into
checked-in Sci regression fixtures. Use the larger external suite to discover
failures, study distributions of error and track long-tail behavior.

## Review and reporting policy

1. State the algorithm's supported numerical and operational scope **before**
   claiming equivalence, accuracy or speed.
2. For meaningful modifications, require a focused independent reference
   regression plus boundary/property tests. Distinguish untested cases and
   intentional deviations from upstream behavior.
3. Record reproducible benchmark settings and investigate material slowdowns.
   Numerical correctness is not waived for a faster implementation.
4. When a discrepancy is found, preserve its input, reference provenance and
   observed failure, then add a minimal focused regression in the main repo.
   Correct the implementation or clearly document a justified limitation.
5. Do not assert repository-wide or domain-wide numerical certification from
   an incomplete inventory, a finite sample, a passing coverage percentage,
   or a comparator that uses Sci as its baseline.

**Adoption work remaining:** define an auditable per-function evidence
inventory and release-facing criteria; connect the external comparison
suite to current Sci 0.3 APIs; then decide which reports should gate PRs
versus scheduled audits. Do not mark existing functions as meeting a new
standard merely because this document is present.
