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
