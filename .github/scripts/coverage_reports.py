"""Split LCOV into disjoint crate reports for Codecov carryforward flags."""

from pathlib import Path


def write_reports(root, meta, selected, source, destination):
    packages = {
        p["name"]: Path(p["manifest_path"]).parent / "src"
        for p in meta["packages"]
        if p["name"] in selected
    }
    if set(packages) != set(selected):
        raise ValueError("Coverage selection contains unknown workspace packages")
    records = {name: [] for name in packages}
    record = []
    for line in source.read_text().splitlines(keepends=True):
        record.append(line.rstrip("\r\n") + "\n")
        if line.strip() != "end_of_record":
            continue
        paths = [l[3:].strip() for l in record if l.startswith("SF:")]
        if len(paths) != 1:
            raise ValueError("Expected exactly one source file per LCOV record")
        path = Path(paths[0])
        path = path if path.is_absolute() else root / path
        for name, directory in packages.items():
            if path.is_relative_to(directory):
                # Relative paths survive downloading the artifact in another job.
                records[name].append(
                    "".join(
                        f"SF:{path.relative_to(root).as_posix()}\n"
                        if l.startswith("SF:")
                        else l
                        for l in record
                    )
                )
                break
        record = []
    if record:
        raise ValueError("Unterminated LCOV record")
    destination.mkdir(parents=True, exist_ok=True)
    # Avoid uploading a flag left behind by a previous local generation.
    for old in destination.glob("*.info"):
        old.unlink()
    flags = []
    for name, contents in sorted(records.items()):
        if contents:
            (destination / f"{name}.info").write_text("".join(contents))
            flags.append(name)
    if not flags:
        raise ValueError("No library coverage found for the selected packages")
    return flags
