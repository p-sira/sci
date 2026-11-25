# Reconnecting `sci-testing` to Sci 0.3

**Status: migration plan, not completed integration.** The separate
[`p-sira/sci-testing`](https://github.com/p-sira/sci-testing) checkout is
valuable evidence infrastructure. It should remain a separate project for
large comparisons and cross-language dependencies unless a concrete workflow
shows that merging it would be simpler. The ordinary Sci test suite must
remain independently runnable without Python, C++ comparison backends or
millions of generated cases.

## Verified local snapshot (2026-11-01)

- The `sci-testing` checkout is on `fix/accuracy-ci` with a clean tree. Its
  `sci/` submodule and `sci-revision.txt` both pin Sci commit
  `b52c598535920279dc0aadb14a9085b8545ba4c1` (2026-10-21).
  Current Sci 0.3 `dev` is `563010feb1ace5cf89d9285010ef4ce6d818cf68`
  (2026-11-01). These are **different API/code snapshots**; the comparison
  harness has not been validated against current 0.3.
- The checked-in compiler API inventory lists **1,195 exported names**, with
  **1,109 distinct public functions** according to its saved report. The
  report records **2,260,692 cases**. These are historical snapshot counts,
  not current Sci 0.3 coverage claims.
- `Cargo.toml` points to the pinned `sci/` submodule; Rust adapters import
  source from its `tests/common` and other fixture modules. Generated
  adapters and CSV recipes refer to the *old* public exports. Changing the
  submodule pointer alone is not a sufficient migration.
- The report stores independently generated reference values alongside Sci,
  SciPy, Boost.Math/GSL and other outputs. It uses Sci as the *displayed
  comparison baseline*, not as a mathematical accuracy oracle.
- `uv.lock` pins NumPy 2.5.4, SciPy 1.18.1, mpmath 1.3.0 and FilterPy 1.4.5.
  The tooling already records hashes, fixture inputs, source revisions and
  native comparison versions; avoid discarding that provenance.

## Recommended incremental migration

1. **Preserve the historical baseline.** Tag or otherwise retain the pinned
   results and generated reports before mutation. Work in a separate branch
   of `sci-testing`; keep the Sci submodule reference immutable during audit.
   Do not replace old reference outputs with new Sci outputs.
2. **Reconcile API inventories first.** Use the current Sci 0.3 compiler
   exports and compare against `api_inventory.csv`. Classify every old name
   as unchanged, moved to a family namespace, renamed, intentionally removed,
   or genuinely new. Regenerate generated adapters against the new public
   contract; preserve canonical algorithm identities where possible.
3. **Update source and fixture adapters.** The Rust runner includes Sci
   fixtures through the submodule and matches named API strings; update these
   dependencies and their integration-test conventions. Migrate old input
   recipes or fixture paths only with explicit case-ID and parameter mapping.
   Check argument order, `f32`/`f64`, error semantics, output shape, and
   changes in supported domains before treating any old case as equivalent.
4. **Start with a bounded pilot**, for example `special.elliptic` (already
   backed by `ellip`) or `integrate`; restore compilation, representative
   independent references and coverage checks for that namespace. Then add
   `filter` and `signal` for robotics workflows, followed by the remaining
   domain crates. Do not attempt a simultaneous regeneration of all 2.26
   million cases before establishing the migration mapping.
5. **Refresh results deliberately.** Use `--full` when inputs, recipes,
   comparison adapters or independent reference definitions have changed.
   Use the existing Sci-only/`--module` refresh only when cases and saved
   independent results still match their hashes and semantics. Preserve
   source/runner/benchmark provenance and keep historical results accessible.
6. **Separate CI responsibilities.** Keep Sci's fast Rust/fixture/property
   tests as mandatory PR checks. Add a small, deterministic external
   accuracy smoke suite for selected high-risk kernels; run the full external
   comparison on schedule or before releases. Avoid overlapping heavy Cargo
   builds, isolate the external `CARGO_TARGET_DIR`, and enforce reasonable
   disk/memory limits.
7. **Report scientific evidence honestly.** Track Sci-versus-independent-
   reference error separately from Sci-versus-other-library differences;
   include failures, skipped/unsupported cases, tolerance and reference
   quality. A case count or matching comparator alone is not a correctness
   certification. Publish supported domains and unresolved discrepancies.

## Proposed milestones and acceptance criteria

| Milestone | Evidence required before proceeding |
| --- | --- |
| Inventory reconciliation | Diff of old/current exported names; explicit dispositions; generator output checked in |
| Pilot namespace | Updated Rust adapter compiles on pinned Sci 0.3; independent values, domain checks and provenance validated |
| Broader domain coverage | All intentionally supported exports have recipes and meaningful successful evaluations; exclusions documented |
| CI smoke suite | Runs without re-evaluating every historical case; deterministic sources, stable tolerances and bounded resource use |
| New reference snapshot | Pinned source, Python/native reference versions, recorded source and input hashes, accurate docs and reports |

**Not attempted as part of this documentation:** moving the `sci/` submodule,
regenerating references, rewriting the old API adapters, or rerunning the full
suite. Those actions alter independent evidence and need their own focused,
reviewable migration work.
