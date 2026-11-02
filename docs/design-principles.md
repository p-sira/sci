# Sci design principles: a trustworthy scientific computing ecosystem

This document records the architectural rationale for Sci 0.3 and the intended
meaning of the `sci-*` crate family. It is a design direction, **not** a claim
that every existing function has already passed the proposed quality gates.
The accompanying [numerical quality standard](numerical-quality-standard.md)
distinguishes current practice from proposed release requirements.

## Purpose

Sci aims to make numerical experimentation and scientific software development
in Rust **coherent, predictable, and independently verifiable**. The value is
not simply having more algorithms than another library: it is being able to
move between algorithms without relearning incidental differences in types,
errors, numerical conventions, or documentation quality.

For example, a robotics researcher should be able to compare filtering,
signal-processing, linear-algebra, and optimization approaches while relying
on well-documented conventions. Consistency removes *arbitrary* differences;
it must not hide *mathematically significant* differences between a Kalman
filter, a particle filter, or their assumptions and failure modes.

The `sci-*` name should indicate a **commitment to scientific rigor, backed by
evidence and clearly stated limits**. It is not a promise that every function
is universally accurate, optimal, or interchangeable.

## Three useful levels of abstraction

| Layer | Responsibility | Who should choose it? |
| --- | --- | --- |
| `sci` | Convenient unified namespace and cross-domain workflows | Applications, teaching, exploration, and work spanning numerical domains |
| `sci-*` | Consistent domain APIs, validation, shared types, and interoperability | Libraries and applications needing a particular domain, e.g. `sci-filter` or `sci-signal` |
| Focused upstream crates | Specialized algorithms and interfaces, with their own expertise and tradeoffs | Callers needing just that specialized functionality |

This layering is **not a mandate to route everything through Sci**. A caller
who needs elliptic integrals alone should consider `ellip` directly rather
than paying for the broader `sci-special` dependency graph. Conversely,
`sci-special` is useful when a caller wants special-function families under
Sci's common contracts; its elliptic wrappers delegate to `ellip` and add Sci
validation and error mapping. Likewise, Sci can reuse upstream numerical
implementations without replacing their upstream identities.

Do not split into ever-smaller crates merely to improve a package count.
Prefer boundaries that have a clear numerical or user-facing purpose. Consider
optional subfeatures inside a domain crate **only when the saved compilation
cost is meaningful and the resulting API and test matrix remain maintainable**.

## Consistency without artificial uniformity

Shared conventions should cover, where applicable:

- Names and argument order; units, parameterizations, normalization, and
  real/complex branch choices.
- Supported scalar precision, shapes, data ownership, error types, and
  interoperability. See [data conventions](conventions.md).
- Domain validation; NaN, infinity, signed zero, overflow, singularities,
  nonconvergence, and iteration/work limits.
- Reproducibility contracts for randomized algorithms, including seed control
  and clearly identified nondeterministic behavior.
- Mathematical definitions, accuracy limitations, references, and runnable
  examples—not merely signatures or API names.

Algorithms need not expose identical options where their mathematics differ.
In particular, a deterministic numerical routine is not automatically
real-time safe, and cross-platform roundoff need not be bitwise identical.
Claims must be scoped to the supported domains and evidence.

## Default features and the dependency contract

**Current 0.3 design direction:** preserve the full-featured `sci` umbrella
by default, while keeping individual domain crates independently usable and
allowing selected features of the umbrella. This serves two distinct workflows:

```toml
# A comprehensive application-facing scientific environment.
sci = "0.3"

# A focused numerical dependency suitable for another library.
sci-integrate = "0.3"

# A selected subset using the same Sci namespace.
sci = { version = "0.3", default-features = false, features = ["filter", "signal"] }
```

These are alternative dependency declarations, not lines to combine into one
manifest. Rust compiles selected dependencies from source in the usual Cargo
workflow, unlike the common prebuilt-wheel installation path for Python's
SciPy. A broad default therefore has a genuine **build-time and disk-space
cost**. The convenience tradeoff is intentional, not an argument that unused
modules are free.

For library authors, prefer the smallest suitable `sci-*` crate—or an
appropriate specialist crate—over pulling the umbrella transitively. Features
are additive across a Cargo graph: a dependency enabling `sci` defaults can
force those capabilities into a larger build. Optional backends and formats
should remain selectable where they materially affect that graph. Maintain
`tests/check_features.py` so isolated configurations exclude unintended
backends; changing the default-feature policy requires updating its explicit
invariants rather than bypassing them.

Keep **downstream consumer builds** separate from Sci's own development
footprint when assessing dependency cost. `target/` includes debug symbols,
incremental caches, repeated test executables, coverage builds and feature
combinations. Final stripped binary size, clean consumer build directory size,
and cumulative multi-worktree artifact size answer different questions.
Record workload, feature set, compiler, profile, target triple and linked code
when publishing size or performance comparisons.

## What deserves the Sci name

The intended value over direct upstream usage is:

1. **A coherent interface** across numerical domains and carefully maintained
   interoperability, not merely renamed re-exports.
2. **Independent numerical validation**, including reference values,
   mathematical properties, boundaries, and adversarial inputs.
3. **Explicit limits and honest failure reporting**, including where Sci's
   behavior intentionally differs from SciPy or from its backend.
4. **Performance transparency** based on reproducible evidence, including
   regressions and accuracy/performance tradeoffs.
5. **Reproducibility and provenance** so users can understand what was tested,
   with which reference, on which source revision and platform.

Scientific trust is earned per algorithm and supported regime. Neither
passing tests nor agreeing with SciPy, Sci, or a shared backend proves a
whole-domain accuracy guarantee. Publish the remaining limitations; do not
use an ecosystem-wide badge or score to imply stronger guarantees than the
underlying evidence supports.

## Architectural decision boundaries

- **Keep** a convenient full-featured umbrella and separately usable domain
  crates for Sci 0.3. A featureless umbrella default was considered, but it
  weakens the principal convenience use case without removing the user's
  option to depend directly on domain crates.
- **Prefer** well-maintained upstream backends for genuinely complex methods,
  with explicit Sci adaptation and independent tests. Implement simple
  mathematical formulas directly when this is clearer and safer.
- **Measure** dependency/build-footprint proposals on isolated consumer builds,
  not on accumulated maintainer caches; avoid sacrificing numerical quality
  solely to remove dependencies.
- **Revisit** internal feature gates or more focused crates when concrete
  downstream demand and measurable savings outweigh maintenance cost.

See [development and verification](development.md) and the
[numerical quality standard](numerical-quality-standard.md) for how to make
these principles actionable.
