# Implementation policy

- Implement trivial numerical functions directly; delegate complex algorithms to suitable private backends.
- Expose functional APIs using `f64` and shared Sci types. Keep backend APIs private and modules interoperable.
- Validate inputs, bound numerical work, and handle errors explicitly. Preserve unrelated user changes.

## Documentation

- Give every public numerical function a mathematical definition, including its equation where applicable.
- Use explicit sections for parameters and conventions, domain and special values, return values, errors, numerical behavior, runnable examples, and authoritative references. Omit sections only when genuinely inapplicable.
- Specify units, normalization, parameter versus modulus conventions, argument and output ordering, shapes, branches, singularities, NaN/infinity behavior, accuracy limitations, and work limits where relevant. Document adaptations from SciPy and actual supported behavior.

## Implementation workflow

1. Read `SCIPY_FUNCTIONS.md` and its linked inventory for the requested module. Identify the requested namespace, aliases, and compatibility or Cython duplicates before determining coverage.
2. Compare the inventory with existing public APIs and tests. Reuse implementations for aliases and shared algorithms; confirm SciPy signatures, definitions, domains, and return conventions before implementing missing functions.
3. Implement related functions in coherent groups using the shared functional API. Add documentation and tests with each group. Record unsupported behavior and signature adaptations explicitly; do not count placeholders or untested names as complete.
4. Audit requested inventory coverage against exported functions and tests before reporting completion. Report any remaining gaps.

## Testing workflow

- Write reproducible Python generators under `tests/scipy/` with pinned SciPy and NumPy versions. Store input cases separately from expected results; compute references with SciPy and write checked-in CSV fixtures for Rust to ingest.
- Exercise every implemented numerical function through its public Sci API. Include multiple representative inputs, boundaries, special values, and invalid domains where applicable; verify shapes and output ordering as well as values.
- For functions without a SciPy equivalent, generate independent references using documented identities or SciPy quadrature. Never generate expected values from Sci or its implementation backend.
- Use justified absolute and relative tolerances, with explicit comparisons for NaN, infinities, and signed zero where meaningful. Investigate discrepancies before changing tolerances; document any reference or algorithm accuracy limits.
- Add independent property, recurrence, inverse round-trip, and cross-module interoperability tests where relevant. Keep ordinary Rust tests independent of Python by reading the committed CSVs.
- Regenerate affected fixtures, then run `cargo fmt --all -- --check`, `cargo test`, `cargo clippy --all-targets -- -D warnings`, and `RUSTDOCFLAGS='-D warnings' cargo doc --no-deps`.
