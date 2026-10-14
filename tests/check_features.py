"""Check the facade with no modules and with each module enabled separately.

Run with Python 3.11 or newer from any directory. Numerical reference tests
remain in the Rust suite; this checks Cargo feature wiring and documentation.
"""

import os
import subprocess
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
manifest = tomllib.loads((ROOT / "Cargo.toml").read_text())
features = manifest["features"]
modules = sorted(
    name for name, values in features.items() if values == [f"dep:sci-{name}"]
)
adapters = sorted(set(features) - {"default", *modules})
assert sorted(features["default"]) == sorted([*modules, *adapters])
assert set(manifest["dependencies"]) == {"sci-core", *(f"sci-{m}" for m in modules)}
for module in modules:
    dependency = f"sci-{module}"
    assert features[module] == [f"dep:{dependency}"]
    assert manifest["dependencies"][dependency]["optional"]

for adapter in adapters:
    module, name = adapter.split("-", 1)
    assert module in modules
    assert features[adapter] == [module, f"sci-{module}/{name}"]
    domain = tomllib.loads((ROOT / f"crates/sci-{module}/Cargo.toml").read_text())
    assert name in domain["features"]

for module in [None, *modules, *adapters]:
    args = ["-p", "sci", "--no-default-features", "--locked"]
    if module:
        args += ["--features", module]
    print(f"Checking {module or 'core only'}", flush=True)
    subprocess.run(["cargo", "check", "--lib", *args], cwd=ROOT, check=True)
    subprocess.run(
        ["cargo", "doc", "--no-deps", *args],
        cwd=ROOT,
        env={**os.environ, "RUSTDOCFLAGS": "-D warnings"},
        check=True,
    )

print(
    f"All {len(modules)} modules, {len(adapters)} adapters and the core-only facade passed."
)
