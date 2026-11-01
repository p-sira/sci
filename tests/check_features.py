"""Check the facade with no modules and with each module enabled separately.

Run with Python 3.11 or newer from any directory. Numerical reference tests
remain in the Rust suite; this checks Cargo feature wiring and documentation.
Optional arguments select features to check; use `core` for the core-only facade,
`default` for all default capabilities, or `crate-sparse-*` to check isolated
`sci-sparse` backends directly.
"""

import os
import subprocess
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
manifest = tomllib.loads((ROOT / "Cargo.toml").read_text())
features = manifest["features"]
modules = sorted(
    name.removeprefix("sci-")
    for name in manifest["dependencies"]
    if name.startswith("sci-") and name != "sci-core"
)
adapters = sorted(set(features) - {"default", *modules})
assert sorted(features["default"]) == sorted([*modules, *adapters])
assert set(manifest["dependencies"]) == {"sci-core", *(f"sci-{m}" for m in modules)}
for module in modules:
    dependency = f"sci-{module}"
    # The sparse facade continues exposing solvers and graphs for consumers
    # selecting `sparse` alone. Only direct sci-sparse no-default builds are lean.
    expected = [f"dep:{dependency}"]
    if module == "sparse":
        expected.append("sci-sparse/full")
    assert features[module] == expected
    assert manifest["dependencies"][dependency]["optional"]

for adapter in adapters:
    module, name = adapter.split("-", 1)
    assert module in modules
    assert features[adapter] == [module, f"sci-{module}/{name}"]
    domain = tomllib.loads((ROOT / f"crates/sci-{module}/Cargo.toml").read_text())
    assert name in domain["features"]
    assert not manifest["workspace"]["dependencies"][f"sci-{module}"].get(
        "default-features", True
    )

# Check direct dependencies in the member manifests separately from the
# resolved tree below. The tree tests transitive closure, not direct edges.
# Matrix Market uses shared CSR helpers from sci-core, and clustering uses
# local distance/statistics helpers rather than sci-spatial or sci-stats.
lean_direct_dependencies = {
    "sci-io": {"sci-core", "hound", "flate2"},
    "sci-cluster": {"sci-core", "kodama"},
}
for crate, expected in lean_direct_dependencies.items():
    domain = tomllib.loads((ROOT / f"crates/{crate}/Cargo.toml").read_text())
    assert set(domain["dependencies"]) == expected, (
        crate,
        "direct dependencies",
        set(domain["dependencies"]) ^ expected,
    )
    if crate == "sci-io":
        assert domain["features"]["matrix-market"] == []
        assert domain["dependencies"]["hound"]["optional"]
        assert domain["dependencies"]["flate2"]["optional"]

sparse_domain = tomllib.loads((ROOT / "crates/sci-sparse/Cargo.toml").read_text())
assert sparse_domain["features"]["default"] == ["full", "npz"]
assert sparse_domain["features"]["full"] == ["solver", "graph", "random"]
assert sparse_domain["features"]["solver"] == [
    "dep:faer", "dep:sci-linalg", "dep:sci-complex", "dep:dyn-stack"
]
assert sparse_domain["features"]["graph"] == [
    "dep:petgraph", "dep:lapjv", "dep:ndarray"
]
assert sparse_domain["features"]["random"] == ["dep:rand"]
assert sparse_domain["features"]["npz"] == ["dep:zip", "dep:flate2"]
assert set(sparse_domain["dependencies"]) == {
    "sci-core", "faer", "sci-linalg", "sci-complex", "dyn-stack",
    "petgraph", "lapjv", "ndarray", "rand", "zip", "flate2",
}
for name in set(sparse_domain["dependencies"]) - {"sci-core"}:
    assert sparse_domain["dependencies"][name]["optional"], name

# Check the resolved build graph: compiling successfully alone does not prove
# that a reduced feature avoids its optional or transitive backends.
backend_rules = {
    "default": ({"zip", "flate2", "miniz_oxide", "smelly"}, {"zlib-rs", "matrw", "kalman_filters", "getrandom"}),
    "filter": ({"smelly", "nalgebra"}, {"kalman_filters", "rand", "rand_distr", "getrandom"}),
    "cluster": (
        {"sci-cluster", "sci-core", "kodama"},
        {
            "sci-stats", "sci-spatial", "sci-special", "sci-linalg",
            "sci-sparse", "nalgebra", "faer",
        },
    ),
    "fft": ({"rustfft"}, {"xsf", "bindgen", "clang-sys"}),
    "fft-hankel": ({"special"}, {"xsf", "bindgen", "clang-sys", "ellip", "lambert_w"}),
    "sparse": (
        {"sci-sparse", "faer", "sci-linalg", "petgraph", "lapjv", "rand"},
        {"zip", "flate2", "zlib-rs", "getrandom"},
    ),
    "sparse-npz": ({"zip", "flate2", "miniz_oxide", "faer", "petgraph"}, {"zlib-rs"}),
    "io": ({"sci-io"}, {"sci-sparse", "hound", "matrw", "zip", "flate2"}),
    "io-matrix-market": (
        {"sci-io", "sci-core"},
        {
            "sci-sparse", "sci-spatial", "sci-stats", "sci-linalg",
            "sci-special", "hound", "matrw", "zip", "flate2",
            "nalgebra", "faer",
        },
    ),
    "io-wav": ({"hound"}, {"sci-sparse", "matrw", "zip", "flate2", "xsf"}),
    "io-matlab": ({"flate2", "miniz_oxide"}, {"sci-sparse", "hound", "zip", "zlib-rs", "xsf", "matrw"}),
}

backend_deps = {
    "faer", "sci-linalg", "sci-complex", "dyn-stack",
    "petgraph", "lapjv", "ndarray", "rand", "zip", "flate2",
}
sparse_crate_rules = {
    "crate-sparse-core": ({"sci-sparse", "sci-core"}, backend_deps),
    "crate-sparse-solver": (
        {"sci-sparse", "sci-core", "faer", "sci-linalg", "sci-complex", "dyn-stack"},
        {"petgraph", "lapjv", "rand", "zip", "flate2"},
    ),
    "crate-sparse-graph": (
        {"sci-sparse", "sci-core", "petgraph", "lapjv", "ndarray"},
        {"faer", "sci-linalg", "sci-complex", "dyn-stack", "rand", "zip", "flate2"},
    ),
    "crate-sparse-random": (
        {"sci-sparse", "sci-core", "rand"},
        {"faer", "sci-linalg", "sci-complex", "dyn-stack", "petgraph", "lapjv", "ndarray", "zip", "flate2"},
    ),
    "crate-sparse-npz": (
        {"sci-sparse", "sci-core", "zip", "flate2", "miniz_oxide"},
        {"faer", "sci-linalg", "sci-complex", "dyn-stack", "petgraph", "lapjv", "rand"},
    ),
}
sparse_crate_features = {
    "crate-sparse-core": None,
    "crate-sparse-solver": "solver",
    "crate-sparse-graph": "graph",
    "crate-sparse-random": "random",
    "crate-sparse-npz": "npz",
}

selected = sys.argv[1:] or ["default", "core", *modules, *adapters, *sparse_crate_rules]
unknown = set(selected) - {"default", "core", *modules, *adapters, *sparse_crate_rules}
if unknown:
    raise SystemExit(f"Unknown features: {', '.join(sorted(unknown))}")

for feature in selected:
    if feature in sparse_crate_rules:
        args = ["-p", "sci-sparse", "--no-default-features", "--locked"]
        if crate_feature := sparse_crate_features[feature]:
            args += ["--features", crate_feature]
        print(f"Checking {feature}", flush=True)
        tree = subprocess.check_output(
            ["cargo", "tree", *args, "-e", "normal,build", "--prefix", "none", "--format", "{p}"],
            cwd=ROOT,
            text=True,
        )
        packages = {line.split()[0] for line in tree.splitlines()}
        required, forbidden = sparse_crate_rules[feature]
        assert required <= packages, (feature, "missing", required - packages)
        assert not forbidden & packages, (feature, "unexpected", forbidden & packages)
        subprocess.run(["cargo", "check", "--lib", *args], cwd=ROOT, check=True)
        subprocess.run(
            ["cargo", "doc", "--no-deps", *args],
            cwd=ROOT,
            env={**os.environ, "RUSTDOCFLAGS": "-D warnings"},
            check=True,
        )
        continue
    module = None if feature == "core" else feature
    args = ["-p", "sci", "--no-default-features", "--locked"]
    if module:
        args += ["--features", module]
    print(f"Checking {module or 'core only'}", flush=True)
    if feature in backend_rules:
        tree = subprocess.check_output(
            ["cargo", "tree", *args, "-e", "normal,build", "--prefix", "none", "--format", "{p}"],
            cwd=ROOT,
            text=True,
        )
        packages = {line.split()[0] for line in tree.splitlines()}
        required, forbidden = backend_rules[feature]
        assert required <= packages, (feature, "missing", required - packages)
        assert not forbidden & packages, (feature, "unexpected", forbidden & packages)
    subprocess.run(["cargo", "check", "--lib", *args], cwd=ROOT, check=True)
    subprocess.run(
        ["cargo", "doc", "--no-deps", *args],
        cwd=ROOT,
        env={**os.environ, "RUSTDOCFLAGS": "-D warnings"},
        check=True,
    )

print(f"All {len(selected)} selected feature checks passed.")
