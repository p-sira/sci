#!/usr/bin/env python3
"""Select the CodSpeed benchmark shards to run.

Every `[[bench]]` target of the root `sci` package is one shard, so that each
upload to CodSpeed stays well below the 1000-benchmark limit.

Usage:
    codspeed_shards.py --all
    codspeed_shards.py <base-rev> [<head-rev>]

With `--all`, every shard is selected. Otherwise, a shard is selected when its
benchmark file changed, or when one of the `sci-*` crates it benchmarks (taken
from the target's `required-features`), or any of their workspace
dependencies, changed. Changes to shared files (lockfile, root manifest,
facade crate, shared bench helpers, native toolchain setup) select every shard,
except root manifest edits limited to `[[test]]`/`[[example]]` entries.

Prints a GitHub Actions matrix as JSON: `{"include": [{"bench": ..., "features": ...}]}`.
"""

import json
import subprocess
import sys
import tomllib

# Changes under these paths affect every benchmark target.
SHARED_PREFIXES = (
    "Cargo.lock",
    "rust-toolchain",
    "src/",
    "benches/common/",
    ".cargo/",
    ".github/actions/",
)


def metadata():
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps", "--locked"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(out)


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def manifest_affects_benches(base, head):
    """Whether the root manifest changed outside its test and example targets."""

    def load(rev):
        try:
            manifest = tomllib.loads(git("show", f"{rev}:Cargo.toml"))
        except subprocess.CalledProcessError:
            return None
        manifest.pop("test", None)
        manifest.pop("example", None)
        return manifest

    return load(base) != load(head)


def feature_crate(feature):
    """Map a `sci` feature to the workspace crate it enables."""
    return "sci-" + feature.split("-", 1)[0]


def main(argv):
    meta = metadata()
    packages = {p["name"]: p for p in meta["packages"]}
    root = packages["sci"]
    workspace_root = meta["workspace_root"].rstrip("/") + "/"

    # Normal and build dependencies between workspace crates.
    deps = {
        name: {
            d["name"]
            for d in pkg["dependencies"]
            if d["name"] in packages and d.get("kind") != "dev"
        }
        for name, pkg in packages.items()
    }

    def closure(crates):
        seen, stack = set(), list(crates)
        while stack:
            crate = stack.pop()
            if crate in seen or crate not in packages:
                continue
            seen.add(crate)
            stack.extend(deps[crate])
        return seen

    shards = []
    for target in root["targets"]:
        if "bench" not in target["kind"]:
            continue
        features = target.get("required-features", [])
        shards.append(
            {
                "bench": target["name"],
                "features": ",".join(features),
                "path": target["src_path"].removeprefix(workspace_root),
                "crates": closure(feature_crate(f) for f in features) | {"sci-core"},
            }
        )
    shards.sort(key=lambda s: s["bench"])

    if argv == ["--all"]:
        selected = shards
    else:
        base, head = argv[0], argv[1] if len(argv) > 1 else "HEAD"
        files = git("diff", "--name-only", base, head).split()
        shared = any(f.startswith(SHARED_PREFIXES) for f in files) or (
            "Cargo.toml" in files and manifest_affects_benches(base, head)
        )
        if shared:
            selected = shards
        else:
            changed_crates = {
                f.split("/")[1] for f in files if f.startswith("crates/") and f.count("/") >= 2
            }
            changed_files = set(files)
            selected = [
                s
                for s in shards
                if s["path"] in changed_files or s["crates"] & changed_crates
            ]

    matrix = {"include": [{"bench": s["bench"], "features": s["features"]} for s in selected]}
    print(json.dumps(matrix, separators=(",", ":")))
    print(
        "Selected shards: " + (", ".join(s["bench"] for s in selected) or "none"),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
