# Development and verification

## Implementation approach

Prefer established Rust crates maintained by domain experts for complex
algorithms. Keep backend APIs private and adapt them to Sci's functional API,
shared types, validation, and errors. Reuse existing Sci modules; implement
simple formulas directly with the standard library. Document each function's
mathematical definition, supported domain, conventions, and numerical limits.

SciPy supplies independent reference values and compatibility comparisons.
The [inventory](../SCIPY_FUNCTIONS.md) and [coverage reports](../function_coverages/)
track that work; public documentation should describe Sci's supported behavior.

## Validation

Run the workspace checks, including every domain crate's documentation examples:

```sh
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps
python tests/check_features.py
```

Documentation-only changes do not require fixture regeneration.

## Benchmarks

Benchmarks for representative functions of each module live in
`benches/modules.rs` and use [divan](https://docs.rs/divan) through the
CodSpeed compatibility layer. Run them locally with:

```sh
cargo bench -p sci --bench modules
```

The `CodSpeed` workflow runs the same suite on every push to `main` and on pull
requests, and reports performance changes with
[CodSpeed](https://app.codspeed.io/p-sira/sci). To reproduce a CodSpeed
measurement locally, install `cargo-codspeed` and run:

```sh
cargo codspeed build -m simulation -p sci
cargo codspeed run -p sci
```

## Build footprint

All capabilities remain enabled by default. For smaller dependency graphs,
set `default-features = false` and select facade module features plus any
required capabilities: `fft-hankel`, `io-matrix-market`, `io-wav`, `io-matlab`,
and `sparse-npz`. The domain crates keep these capabilities in their defaults;
the workspace disables domain defaults on shared dependencies to prevent
signal processing or Matrix Market from enabling Hankel or NPZ transitively.
Cargo still combines features requested elsewhere in a consumer's graph.

`tests/check_features.py` compiles each facade feature and isolated sparse
configuration, checking selected architectural dependency boundaries without
freezing backend implementations. Documentation is tested separately by the
regular CI documentation job, not regenerated for every feature combination.
The FFT extended reference suite and benchmark require `fft-hankel`; sparse
parity requires `sparse-npz`;
the I/O integration suite requires all three format features. Numerical
APIs retain their existing conventions. FFTLog now uses private Rust translations
of SciPy/XSF's complex log-gamma and Cephes gamma-ratio algorithms, with the
existing `special` crate supplying real log-gamma. The gamma work is bounded
independently of order and bias. This removes XSF from every FFT build.
Expanded independent SciPy fixtures exercise reflection, Taylor and asymptotic
regions, negative spacing, gamma poles, and the bounded ratio fallback.
Large-order comparisons have explicit tolerances for log-gamma cancellation.
`uv run tests/scipy/generate_fftlog_accuracy.py` regenerates 16 independent
discrete FFTLog references using mpmath dense DFTs at 100 and 160 digits. Their
inputs and tolerances are stored in `fft_cases.jsonl`; ordinary Rust tests read
the committed CSV. These measure the discrete mapping's roundoff, rather than
the discretization error of the continuous Hankel integral.

MATLAB and NPZ share flate2's Rust/miniz backend. ZIP selects `deflate-flate2`,
and each format explicitly enables the workspace flate2 backend. This avoids compiling
zlib-rs alongside miniz. Archive bytes may differ while decoded values and
format compatibility remain the same.

Particle propagation and moments reuse Smelly 0.2 without a random generator.
MATLAB numeric decoding reuses Sci's native bounded format parser. Statrs keeps
its `std` and `nalgebra` features with sampling disabled; sparse generation uses
seeded `StdRng` without OS entropy or thread-local RNG features.

The workspace uses `special` 0.14 with default features disabled and `std`
enabled. Only `sci-special` enables its `lambert_w` feature; FFTLog needs only
real log-gamma. Elliptic integrals continue to call `ellip` directly, without
enabling `special/elliptic`.

TODO: audit the overlapping incomplete gamma/beta functions and negative-argument
digamma/trigamma before replacing their XSF calls. Replacing XSF for functions
unsupported by `special` (including Bessel, Airy, Struve, hypergeometric,
Faddeeva, Mathieu, and spheroidal functions) remains TODO. These functions
retain their current implementations; `sci-special` still requires XSF.

## Pre-commit formatting

Enable the repository's pre-commit hook once per clone:

```sh
git config core.hooksPath .githooks
```

## Reference fixtures

The integration suite ingests checked-in CSVs generated by pinned versions of
SciPy and NumPy. The general module suite uses SciPy 1.16.2 / NumPy 2.3.3;
the complete special-function suite uses SciPy 1.18.1 / NumPy 2.5.4.
The comparisons exercise the public functional
API, including Jacobian/matrix products and Fourier/complex/statistics pipelines.

Regenerate fixtures only when inputs or numerical behavior change. Each generator
pins its own Python dependencies in its script metadata; run it with `uv`. For example:

```sh
uv run tests/scipy/generate_audit_regressions.py
uv run tests/scipy/generate.py
uv run tests/scipy/generate_cluster.py
uv run tests/scipy/generate_sparse.py
uv run tests/scipy/generate_sparse_parity.py
uv run tests/scipy/generate_sparse_npz_validation.py
python tests/scipy/audit_sparse.py
python tests/scipy/audit_signal.py
```

Python and SciPy are required only for fixture generation. Ordinary Rust tests
read the checked-in CSVs. Comparisons use absolute and relative tolerances;
Statrs' approximate normal CDF is checked at a relative tolerance of `1e-10`.
Special-function input cases are defined in
`tests/scipy/special_cases.jsonl`, with one independent JSON object per
line. Other large input catalogs use the same JSON Lines format. Run
`python tests/scipy/case_catalog.py` to check their canonical formatting.
Expected values are kept in separately generated CSV fixtures for Rust
tests to use without Python. References come from SciPy, elliptic
identities, or independent SciPy quadrature. CSV rows carry
relative and absolute tolerances, normally `2e-11`. Noncentral F parameter
inverses use `1e-4` relative tolerance because SciPy's CDFLIB inversion is
approximate; noncentral t comparisons use `5e-6` relative / `1e-7` absolute.
Separate inverse-CDF round trips use much tighter tolerances. Coverage audits,
boundary checks, and quadrature moments supplement the reference comparisons.
Optimization comparisons check final points and costs rather than iteration
trajectories, because the algorithms differ.

The audit regression generator also covers extreme-scale normal probabilities,
weighted norms, expectiles and gradient convergence. Normal references apply
Decimal location/scale arithmetic to SciPy's standard distribution; exact
rational expectile balances and Decimal weighted norms avoid SciPy's range and
root-tolerance limitations. Native quadratic gradient-descent cases use the
independent update identity and Python's stable norm. Scale invariance,
zero-weight exclusion, inverse round trips and invalid-domain checks supplement
these fixtures.

## Execution coverage

The `core`, `coverage`, `linalg_coverage`, and `statistics_coverage` integration
targets supplement the module reference suites with validation, overflow,
support, callback, and independent property checks. `core` runs without optional features;
`linalg_coverage` requires only `linalg`; `statistics_coverage` requires only
`stats`; `coverage` requires all numerical module features except `stats`,
plus `fft-hankel`.

Install cargo-llvm-cov and LLVM tools, then run the same workspace coverage
command as CI. Integration tests and their helpers are excluded from the report
because they exercise the library rather than implement its APIs. Reports stay
under the Git-ignored `target/` directory:

```sh
rustup component add llvm-tools-preview --toolchain stable
cargo install cargo-llvm-cov --locked
cargo +stable llvm-cov --workspace --all-features --locked --lcov --output-path target/coverage/lcov.info --ignore-filename-regex '(^|/)tests/'
cargo +stable llvm-cov report --html --ignore-filename-regex '(^|/)tests/'
```

Check the reduced-feature test targets separately:

```sh
cargo test --no-default-features --test core
cargo test --no-default-features --features linalg --test linalg_coverage
cargo test --no-default-features --features stats --test statistics_coverage
```

Line coverage measures execution of the supported Rust APIs. SciPy inventory
coverage and remaining compatibility gaps are documented in the module coverage
reports and checked by the audit scripts above. Backend failure recovery and
unreachable validation branches can remain uncovered; a percentage does not
establish numerical accuracy or full SciPy behavioral parity.
The stable coverage run measures unit and integration tests. Documentation
examples are tested separately by `cargo test --workspace`.

The statistics boundary generator stores input-only profiles separately from
556 expected evaluations. It tests degenerate distributions, support endpoints,
and both sides of SciPy's 1,000-observation normal-posterior approximation switch.
It documents two SciPy degenerate-case defects and uses equivalent independently
evaluated Poisson point-mass identities for those references.

After `cargo test --test sparse_parity`, run
`uv run tests/scipy/check_sparse_npz.py` to verify the Rust-written archive with SciPy.
To limit build memory, run Cargo commands serially with `CARGO_BUILD_JOBS=1`.

[Back to README](../README.md).

Scientific I/O fixtures use SciPy 1.18.1 / NumPy 2.5.4. Regenerate with
`uv run tests/scipy/generate_io.py`, then run `cargo test --test io` and
`uv run tests/scipy/check_io_writes.py` to verify Rust-written files with SciPy.
Run `python tests/scipy/audit_io.py` to check inventory disposition and API tests.
The [I/O coverage report](../function_coverages/IO_COVERAGE.md) documents exact
comparison rules and two SciPy Matrix Market reference limitations.
