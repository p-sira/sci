Tests and coverage select changed crates and their transitive dependents using
Cargo metadata. Root integration tests and Clippy benchmark targets are selected
from their `required-features`; their features are enabled explicitly. Changes
to test files, referenced fixtures, or referenced inventories also select their
consumers. Isolated feature checks follow the same selection. Formatting and
inventory audits always run.
Reference generators do not affect compiled tests until their fixtures change.

Pull requests compare against the merge base with the target branch. Pushes
compare against the previous head, including every commit in the push. Renames
check both paths. Documentation-only changes skip numerical checks.

Manual runs, unavailable history, removed crates, unknown fixture consumers,
and shared manifest, lockfile, facade, toolchain, CI, or test infrastructure
changes run the full workspace. CodSpeed retains its separate benchmark selector.

Coverage uses one Codecov carryforward flag per crate. A single instrumented run
collects results, which are split into disjoint reports containing only that
crate's `src/` files. Each report is uploaded separately with its crate name as
the flag. Crates without instrumented source records do not produce an upload.
Dependencies exercised incidentally are not uploaded under a refreshed flag.
Codecov retains their previous measurements.

When tests or fixtures change, coverage refreshes the crates named by their test
targets' required features. Before replacing a flag, it runs the crate's tests
and all affected dependent tests, including the public API tests. Ordinary CI
test selection remains unchanged.

The first run after these workflow/configuration changes runs the full workspace
and uploads every available crate flag to establish a baseline. Merge this run
into the target branch before relying on carried coverage there. To establish
or refresh a branch baseline manually, run the **Coverage** workflow through
GitHub Actions with `workflow_dispatch`. The configuration follows Codecov's
[carryforward flag guidance](https://docs.codecov.com/docs/carryforward-flags).

Run selector regression tests with:

```sh
python -m unittest discover -s .github/scripts -p 'test_*.py'
```

Each crate has a packaged `CHANGELOG.md`. The `0.2.0` entries describe changes
since the repository tag `v0.1.0`; the newly added `sci-sparse` and `sci-io`
start at `0.1.0`. Existing crates move together to `0.2.0` because their public
types now come from `sci-core 0.2`.

Publish dependencies before dependents. For this release, publish these groups
in order (crates within a group can be published in any order):

1. `sci-core 0.2.0` (already published).
2. `sci-complex`, `sci-constants`, `sci-differentiate`, `sci-filter`,
   `sci-interpolate`, `sci-optimize`, and `sci-spatial`, all at `0.2.0`.
3. `sci-fft` and `sci-special`, both at `0.2.0`.
4. `sci-integrate`, `sci-linalg`, `sci-signal`, and `sci-stats`, all at `0.2.0`.
5. `sci-cluster 0.2.0` and `sci-sparse 0.1.0`.
6. `sci-io 0.1.0`.
7. `sci 0.2.0`.

Run `cargo package --workspace --allow-dirty` to verify the release set before
committing, then `cargo publish --workspace --exclude sci-core` from the clean
commit. Cargo stages selected workspace packages and verifies their tarballs
together, then publishes in dependency order. Publishing `sci-sparse` alone
requires the new complex and linear algebra releases to be available first;
older releases use a different core error type and cannot satisfy its APIs.
