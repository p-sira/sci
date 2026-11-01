"""Check that Sci's optional modules compile without unrelated dependencies.

Run with Python 3.11+ from anywhere. With no arguments, check the default
configuration, core-only facade, every module and adapter, and the isolated
``sci-sparse`` configurations. Pass names (e.g. ``core io-wav``) to select a
subset. General documentation and numerical tests run in separate CI jobs.
"""

import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def manifest(path):
    return tomllib.loads((ROOT / path).read_text())


def validate_features():
    """Validate the public feature mapping without freezing implementation deps."""
    root = manifest("Cargo.toml")
    features = root["features"]
    modules = sorted(
        name.removeprefix("sci-")
        for name in root["dependencies"]
        if name.startswith("sci-") and name != "sci-core"
    )
    adapters = sorted(set(features) - {"default", *modules})

    for module in modules:
        dep = f"sci-{module}"
        assert root["dependencies"][dep]["optional"], dep
        assert f"dep:{dep}" in features[module], module

    for adapter in adapters:
        module, _, name = adapter.partition("-")
        assert module in modules and name, adapter
        assert module in features[adapter], adapter
        assert f"sci-{module}/{name}" in features[adapter], adapter
        assert name in manifest(f"crates/sci-{module}/Cargo.toml")["features"], adapter

    # The project's full default is a policy choice, not an isolation rule.
    # Ensure only that every default name exists; do not hard-code its contents.
    assert set(features["default"]) <= set(features), "unknown default feature"
    return modules, adapters


def packages(args):
    """Resolved normal/build packages, including transitive dependencies."""
    output = subprocess.check_output(
        ["cargo", "tree", "--locked", "-e", "normal,build", "--prefix", "none", "--format", "{p}", *args],
        cwd=ROOT,
        text=True,
    )
    return {line.split()[0] for line in output.splitlines() if line.strip()}


def check_isolation(label, args, excluded):
    found = packages(args) & excluded
    assert not found, f"{label} unexpectedly brings in: {', '.join(sorted(found))}"


def main(selected):
    modules, adapters = validate_features()
    sparse_features = {"core": None, "solver": "solver", "graph": "graph", "random": "random", "npz": "npz"}
    sparse_labels = {f"crate-sparse-{feature}" for feature in sparse_features}
    available = {"default", "core", *modules, *adapters, *sparse_labels}
    unknown = set(selected) - available
    if unknown:
        raise SystemExit(f"Unknown features: {', '.join(sorted(unknown))}")
    if not selected:
        selected = ["default", "core", *modules, *adapters, *sorted(sparse_labels)]

    # Only protect meaningful isolation boundaries. Exact backend/package
    # inventories and positive implementation dependencies are not contracts.
    domains = {f"sci-{module}" for module in modules}
    excluded = {
        "core": domains,
        "cluster": domains - {"sci-cluster"},
        "io": domains - {"sci-io"} | {"hound", "flate2", "zip"},
        "io-matrix-market": domains - {"sci-io"} | {"hound", "flate2", "zip"},
        "io-wav": domains - {"sci-io"} | {"flate2", "zip"},
        "io-matlab": domains - {"sci-io"} | {"hound", "zip"},
        "fft": {"sci-special"},
        "fft-hankel": {"sci-special"},
        "sparse": {"zip", "flate2"},  # NPZ is a separate opt-in feature.
    }
    # One distinct backend per sparse family is enough to catch accidentally
    # enabling optional solver/graph/random/archive subsystems.
    sparse_excluded = {
        "core": {"faer", "petgraph", "rand", "zip"},
        "solver": {"petgraph", "rand", "zip"},
        "graph": {"faer", "rand", "zip"},
        "random": {"faer", "petgraph", "zip"},
        "npz": {"faer", "petgraph", "rand"},
    }

    for label in selected:
        if label in sparse_labels:
            feature = label.removeprefix("crate-sparse-")
            args = ["-p", "sci-sparse", "--no-default-features"]
            if sparse_features[feature]:
                args += ["--features", sparse_features[feature]]
            check_isolation(label, args, sparse_excluded[feature])
        else:
            args = ["-p", "sci", "--no-default-features"]
            if label != "core":
                args += ["--features", label]
            if label in excluded:
                check_isolation(label, args, excluded[label])

        print(f"Checking {label}", flush=True)
        subprocess.run(["cargo", "check", "--lib", *args, "--locked"], cwd=ROOT, check=True)

    print(f"All {len(selected)} selected feature configurations passed.")


if __name__ == "__main__":
    main(sys.argv[1:])
