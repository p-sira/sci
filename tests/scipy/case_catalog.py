"""Read line-oriented JSON numerical input catalogs (one complete case per line).

JSON Lines keeps heterogeneous numerical arguments and optional metadata typed,
without embedding JSON-encoded arrays inside CSV cells. Rust tests continue to
read the generated CSV reference fixtures, not these input catalogs.
"""

import json
from pathlib import Path


def read_cases(path: Path) -> list:
    cases = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{path}:{number}: empty JSON Lines record")
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: invalid JSON Lines record") from error
    return cases


def format_cases(cases: list) -> str:
    """Canonical, reviewable one-record-per-line layout (no data normalization)."""
    return "".join(
        json.dumps(case, ensure_ascii=False, separators=(", ", ": ")) + "\n"
        for case in cases
    )


if __name__ == "__main__":
    directory = Path(__file__).parent
    names = (
        "special_cases.jsonl",
        "special_native_cases.jsonl",
        "special_float_cases.jsonl",
        "hyp1f1_cancellation_cases.jsonl",
    )
    for name in names:
        path = directory / name
        if path.read_text(encoding="utf-8") != format_cases(read_cases(path)):
            raise SystemExit(f"Noncanonical JSON Lines formatting: {path}")
        print(f"{name}: {len(read_cases(path))} records, canonical formatting")
