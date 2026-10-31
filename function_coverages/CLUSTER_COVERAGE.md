# `cluster` coverage and SciPy adaptations

Inventory: SciPy 1.18.1, `scipy-func-list/scipy.cluster.md`. Public names are
counted in their own namespace; aliases reuse their implementation.

`cluster::vq` retains four inventory names, omitting the `py_vq` alias, with the existing deterministic,
explicit-centroid adaptations documented in `NUMERICS_COVERAGE.md`.
`cluster::hierarchy` has 27 of 30 functions with tested native signatures.

## Supported hierarchy functions

- Linkage: `linkage`, `single`, `complete`, `average`, `weighted`, `centroid`,
  `median`, `ward`. Private Kodama 0.3 implements the numerical backend.
- Tree queries: `num_obs_linkage`, `is_valid_linkage`, `is_monotonic`,
  `leaves_list`, `to_tree`, `correspond`, `is_isomorphic`.
- Statistics: `cophenet`, `inconsistent`, `is_valid_im`, `maxdists`,
  `max_rstat` (the `maxRstat` alias is removed), `maxinconsts`.
- Conversion and partitions: `from_mlab_linkage`, `to_mlab_linkage`, `fcluster`,
  `fclusterdata`, `leaders`, `cut_tree`.

## Signature adaptations and limits

Linkage uses shared `Vec<Vec<f64>>` matrices with four columns: zero-based left
and right child IDs, merge height, and exact leaf count. Input validation
rejects nonfinite values, fractional IDs, reused children, incorrect counts,
and more than 512 leaves. This is stricter than SciPy’s malformed-input
validation. Linkage distances must be nonnegative; centroid, median, and Ward
require Euclidean geometry, which cannot be inferred from condensed values.
Squared-distance methods reject magnitudes above `sqrt(MAX/(8n))` before the
backend; all backend outputs undergo structural and finite validation.
Kodama tie choices can differ from SciPy while defining valid trees.

`linkage` and its seven wrappers accept condensed distances. Compute shared
spatial `pdist` first for observation inputs. Optimal ordering is unavailable.
No backend types are public. Centroid and median may take O(n³) worst-case
work; other methods take O(n²). Memory is O(n²).

`fcluster` uses `Criterion` for all five SciPy criteria. Inconsistency depth is
explicit; depth zero behaves as depth one. Supplied monotone statistics are
validated. Thresholds include equality. Maximum-count criteria require positive
integral counts and may produce fewer clusters at tied heights. Partition
labels are one-based; callers should compare partitions up to label bijections.
`fclusterdata` accepts finite Euclidean observations up to 512 × 512. Other
metrics can use `pdist`, `linkage`, and `fcluster` separately. Explicit supplied
inconsistency matrices are handled through `maxinconsts` and `Monocrit`.

`cut_tree` uses `Cut` to request all stages, exact counts, or strict height
cuts. All cuts require monotone linkages; this avoids SciPy’s height-sorted
inversion behavior. Output is leaf-by-request integer rows, with zero-based
labels ordered by first leaf. Output is limited to 1,000,000 entries. Empty
request slices return one empty row per leaf.

`to_tree` returns flat `ClusterNode` values ordered by ID, with indexed optional
children. The last element is the root. Recursive Python objects and `rd` are
adapted to this shared representation. `leaders` returns `(node_ids, cluster_ids)`
sorted by node ID, adapting SciPy’s traversal order. Cluster labels are `usize`.
`cophenet` returns `(optional_correlation, condensed_distances)`; constant inputs
produce NaN correlation. `inconsistent` uses scaled two-pass sample moments,
which may differ from SciPy’s raw moments near cancellation. All numerical
outputs are checked for overflow except the documented undefined correlation.
SciPy 1.18.1’s installed validator returns true for several malformed trees,
including negative heights and forward references. These cases are exercised
as native rejection properties rather than claimed as matching SciPy.
SciPy `cut_tree` also mishandles repeated singleton-stage requests; native cuts
preserve each request independently and property tests cover this adaptation.
Boolean validators omit Python warning/throw/name controls and return false
for unsupported sizes or invalid inputs. `is_valid_im` follows SciPy’s actual
nonnegative checks on the first three columns, adding finite-value checks.

## Remaining names

- `optimal_leaf_ordering`: requires a suitable bounded optimization backend.
- `dendrogram`: plotting and coordinate/truncation/color options are absent.
- `set_link_color_palette`: Python plotting state has no native equivalent yet.

No placeholders are exported or counted. Python classes, warnings, and their
methods are not counted as numerical function coverage.

## Verification

The pinned SciPy 1.18.1 / NumPy 2.5.4 generator reads input-only
`tests/scipy/cluster_cases.jsonl` and writes separate checked-in input and
expected CSVs. Rust tests exercise every supported public name, check output
shapes and ordering, and compare f64 values with `3e-12` absolute plus `3e-12`
relative tolerance. Integer outputs are exact; flat labels are checked up to
bijections. Independent tests cover conversion round trips, leaf permutations,
cluster-count progression, subtree counts, invalid domains, work bounds,
inversions, and interoperability with vector quantization and spatial distances.

```console
uv run tests/scipy/generate_cluster.py
python tests/scipy/audit_cluster.py
cargo fmt --all -- --check
cargo test
cargo clippy --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --no-deps
```
