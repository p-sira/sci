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

Run selector regression tests with:

```sh
python -m unittest discover -s .github/scripts -p 'test_*.py'
```
