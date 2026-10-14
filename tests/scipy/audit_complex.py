"""Audit every exported complex numerical function against reference inputs.

SciPy's inventory has no complex namespace or Python/Cython duplicates. This
module deliberately implements scalar elementary operations rather than the
complex overloads of unrelated SciPy namespaces. See COMPLEX_COVERAGE.md.
"""

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
exports = set(
    re.findall(
        r"^pub fn (\w+)\(",
        (ROOT / "crates/sci-complex/src/mod.rs").read_text(),
        re.MULTILINE,
    )
)
with (ROOT / "tests/fixtures/complex_inputs.csv").open() as source:
    cases = list(csv.DictReader(source))
covered = {case["api"] for case in cases}
with (ROOT / "tests/fixtures/dependency_extras.csv").open() as source:
    extras = [row for row in csv.DictReader(source) if row["function"] in exports]
covered.update(row["function"] for row in extras)
tests = (ROOT / "tests/dependency_extras.rs").read_text()
for name in {row["function"] for row in extras}:
    assert f"c::{name}(" in tests, (name, "missing public test")
if exports != covered:
    raise SystemExit(
        f"Missing cases: {exports - covered}; stale cases: {covered - exports}"
    )
with (ROOT / "tests/fixtures/complex_expected.csv").open() as source:
    expected = list(csv.DictReader(source))
if [c["id"] for c in cases] != [e["id"] for e in expected]:
    raise SystemExit("Reference IDs do not match input IDs")
print(
    f"Covered {len(exports)} numerical functions with {len(cases) + len(extras)} reference cases"
)
