"""Select affected workspace crates and public-API targets, or run their checks.

PRs use the merge base; pushes use the previous head. Manual runs, missing
history, and shared build/test infrastructure changes select the workspace.
Cargo metadata supplies dependencies and target feature requirements.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from coverage_reports import write_reports

ROOT = Path(__file__).resolve().parents[2]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def metadata():
    return json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--no-deps", "--format-version", "1", "--locked"],
            cwd=ROOT,
            text=True,
        )
    )


def select(meta, files):
    packages = {p["name"]: p for p in meta["packages"]}
    facade = packages["sci"]
    shared = (
        "src/",
        ".cargo/",
        ".github/",
        "tests/common/",
        "benches/common/",
        "examples/",
    )
    full = files is None or any(
        f
        in {
            "Cargo.toml",
            "Cargo.lock",
            "build.rs",
            "clippy.toml",
            ".clippy.toml",
            "codecov.yml",
            "tests/check_features.py",
        }
        or f.startswith((*shared, "rust-toolchain"))
        for f in files
    )
    if full:
        return {"full": True, "active": True}

    changed = {
        name
        for name, pkg in packages.items()
        if name != "sci"
        and any(
            (ROOT / f).is_relative_to(Path(pkg["manifest_path"]).parent) for f in files
        )
    }
    if any(
        f.startswith("crates/")
        and not any(
            (ROOT / f).is_relative_to(Path(p["manifest_path"]).parent)
            for n, p in packages.items()
            if n != "sci"
        )
        for f in files
    ):
        return {"full": True, "active": True}
    affected = set(changed)
    while True:
        dependents = {
            name
            for name, pkg in packages.items()
            if name != "sci" and any(d["name"] in affected for d in pkg["dependencies"])
        }
        if dependents <= affected:
            break
        affected |= dependents

    targets = {"test": [], "bench": []}
    coverage = set(affected)
    consumed_inputs = set()
    features = {n.removeprefix("sci-") for n in affected} - {"core"}
    features.update(
        f
        for f in facade["features"]
        if "-" in f and "sci-" + f.split("-", 1)[0] in affected
    )
    for target in facade["targets"]:
        kind = target["kind"][0]
        if kind not in targets:
            continue
        path = Path(target["src_path"])
        relative = path.relative_to(ROOT).as_posix()
        required = target.get("required-features", [])
        target_crates = {"sci-" + f.split("-", 1)[0] for f in required}
        # References can use include_str!, include_bytes!, or runtime file I/O.
        source = path.read_text()
        inputs = {
            f
            for f in files
            if f.startswith(("tests/fixtures/", "function_coverages/"))
            and Path(f).name in source
        }
        consumed_inputs.update(inputs)
        if kind == "test" and (relative in files or inputs):
            coverage.update(target_crates or {"sci-core"})
        if (
            "sci-core" in affected
            or target_crates & affected
            or relative in files
            or inputs
        ):
            targets[kind].append(target["name"])
            features.update(required)

    # Unknown fixture consumers must never silently lose test coverage.
    if any(f.startswith("tests/fixtures/") and f not in consumed_inputs for f in files):
        return {"full": True, "active": True}
    return {
        "full": False,
        "active": bool(affected or any(targets.values())),
        "packages": sorted(affected),
        "coverage": sorted(coverage),
        "features": sorted(features),
        **targets,
    }


def changed_files():
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    try:
        if os.environ["GITHUB_EVENT_NAME"] == "pull_request":
            base = event["pull_request"]["base"]["sha"]
            base = git("merge-base", base, "HEAD").strip()
        elif os.environ["GITHUB_EVENT_NAME"] == "push":
            base = event["before"]
        else:
            return None
        # Include both sides of renames so the old module is checked too.
        return (
            git("diff", "--name-only", "--no-renames", "-z", base, "HEAD")
            .strip("\0")
            .split("\0")
        )
    except subprocess.CalledProcessError:
        return None


def run(scope, mode):
    if not scope["active"]:
        return
    if mode == "coverage":
        meta = metadata()
        report_packages = (
            {p["name"] for p in meta["packages"]}
            if scope["full"]
            else set(scope["coverage"])
        )
        if not report_packages:
            output_flags([])
            return
        if not scope["full"]:
            # Refresh each flag with every test that can exercise its crate,
            # even when the change only touched one test or reference fixture.
            scope = select(
                meta,
                [
                    Path(p["manifest_path"]).relative_to(ROOT).as_posix()
                    for p in meta["packages"]
                    if p["name"] in report_packages
                ],
            )
    full = scope["full"]
    if mode == "features":
        modules = [] if full else ["core", *scope["features"]]
        subprocess.run(
            [sys.executable, "tests/check_features.py", *modules], cwd=ROOT, check=True
        )
        return

    commands = {
        "test": ["cargo", "test"],
        "clippy": ["cargo", "clippy"],
        "doc": ["cargo", "doc", "--no-deps"],
        "coverage": ["cargo", "+stable", "llvm-cov", "--no-report"],
    }
    command = commands[mode]
    if mode == "coverage":
        subprocess.run(
            ["cargo", "+stable", "llvm-cov", "clean", "--workspace"],
            cwd=ROOT,
            check=True,
        )
    suffix = ["--", "-D", "warnings"] if mode == "clippy" else []
    groups = []
    if full:
        groups.append(
            ["--workspace"] + (["--all-features"] if mode == "coverage" else [])
        )
    else:
        if scope["packages"]:
            groups.append(
                [arg for p in scope["packages"] for arg in ["-p", p]]
                + ["--all-features"]
            )
        facade = ["-p", "sci", "--no-default-features"]
        if scope["features"]:
            facade += ["--features", ",".join(scope["features"])]
        targets = ["--lib"]
        targets += [arg for t in scope["test"] for arg in ["--test", t]]
        if mode == "clippy":
            targets += [arg for t in scope["bench"] for arg in ["--bench", t]]
        groups.append(facade + ([] if mode == "doc" else targets))
        if mode == "test":
            groups.append(facade + ["--doc"])
    for args in groups:
        if mode == "clippy" and (full or "sci" not in args):
            args += ["--all-targets"]
        invocation = command + args + ["--locked"] + suffix
        print("Running:", " ".join(invocation), flush=True)
        subprocess.run(invocation, cwd=ROOT, check=True)
    if mode == "coverage":
        subprocess.run(
            [
                "cargo",
                "+stable",
                "llvm-cov",
                "report",
                # Reporting has its own package selection; it does not inherit
                # the packages selected by the preceding --no-report runs.
                *[arg for p in sorted(report_packages) for arg in ["--package", p]],
                "--lcov",
                "--output-path",
                "target/coverage/lcov.info",
                "--ignore-filename-regex",
                "(^|/)tests/",
            ],
            cwd=ROOT,
            check=True,
        )
        flags = write_reports(
            ROOT,
            meta,
            report_packages,
            ROOT / "target/coverage/lcov.info",
            ROOT / "target/coverage/flags",
        )
        output_flags(flags)


def output_flags(flags):
    value = json.dumps(flags, separators=(",", ":"))
    print("Coverage flags:", value, flush=True)
    if "GITHUB_OUTPUT" in os.environ:
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
            output.write(f"flags={value}\n")


def main():
    if sys.argv[1] == "select":
        scope = select(metadata(), changed_files())
        print(json.dumps(scope, separators=(",", ":")))
    else:
        run(json.loads(os.environ["CI_SCOPE"]), sys.argv[1])


if __name__ == "__main__":
    main()
