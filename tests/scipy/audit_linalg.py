"""Audit public linalg inventory names against exports and reference tests.

Compatibility namespaces repeat their top-level APIs. Interpolative names
are audited in their own namespace. Runtime dispatch and raw
BLAS/LAPACK/Cython interfaces remain outside the functional surface.
Run using Python 3.11+, without SciPy.
"""
from api_disposition import ALIASES, EXCLUDED
import csv
import re
from collections import Counter
from pathlib import Path
from case_catalog import read_cases

ROOT=Path(__file__).resolve().parents[2]


def inventory(namespace):
    selected=functions=False
    names=set()
    for line in (ROOT/"function_coverages/scipy_function_list/scipy.linalg.md").read_text().splitlines():
        if line.startswith("### "):
            selected=line==f"### scipy.linalg{namespace}"
        if line.startswith("#### "):
            functions=line=="#### Functions"
        if selected and functions:
            match=re.match(r"- (?:\[)?`(\w+)`",line)
            if match:names.add(match[1])
    return names


source="\n".join(p.read_text() for p in (ROOT/"crates/sci-linalg/src").glob("*.rs") if p.stem!="interpolative")
exported=set(re.findall(r"pub fn (\w+)",source))
exported.update(re.findall(r"pub use \w+ as (\w+);",source))
original={"dot","matvec","matmul","transpose"}
public=inventory("")
added=exported-original
assert added==public-EXCLUDED["linalg"]-set(ALIASES["linalg"]),(added-public,public-added)
interpolative_source=(ROOT/"crates/sci-linalg/src/interpolative.rs").read_text()
interpolative=set(re.findall(r"pub fn (\w+)",interpolative_source))
assert interpolative==inventory(".interpolative")
qualified=added|{"interpolative."+name for name in interpolative}
cases=read_cases(ROOT/"tests/scipy/linalg_cases.jsonl")
counts=Counter(ALIASES["linalg"].get(c["api"], c["api"]) for c in cases)
assert set(counts)==qualified,(qualified-set(counts),set(counts)-qualified)
assert all(n>=2 for n in counts.values()),counts
with (ROOT/"tests/fixtures/linalg_inputs.csv").open() as f:inputs=list(csv.DictReader(f))
with (ROOT/"tests/fixtures/linalg_expected.csv").open() as f:expected=list(csv.DictReader(f))
assert len(cases)==len(inputs)==len(expected)
assert [c["api"] for c in cases]==[c["api"] for c in inputs]
assert [c["id"] for c in inputs]==[c["id"] for c in expected]
assert [c["id"] for c in inputs]==list(map(str,range(len(cases))))
tests=(ROOT/"tests/linalg.rs").read_text()
for name in added:
    assert re.search(rf"\bl::{name}\(",tests),(name,"missing public test dispatch")
for name in interpolative:
    assert re.search(rf"\bl::interpolative::{name}\(",tests),(name,"missing public test dispatch")
assert "pub use sci_linalg as linalg;" in (ROOT/"src/lib.rs").read_text()
assert "pub use dense::*;" in source and "pub use structured::*;" in source
module=(ROOT/"crates/sci-linalg/src/mod.rs").read_text()
for path in (ROOT/"crates/sci-linalg/src").glob("*.rs"):
    if path.stem not in {"mod","lib","backend","interpolative","input"}:
        assert f"pub use {path.stem}::*;" in module,(path,"missing module export")
assert "pub mod interpolative;" in module
assert "pub use input::{MatrixInput, VectorInput};" in module
for library in ("ndarray", "nalgebra", "faer"):
    assert f"fn {library}_inputs_match_scipy_references()" in tests
coverage=(ROOT/"function_coverages/LINALG_COVERAGE.md").read_text()
for name in public|inventory('.interpolative'):
    assert f"`{name}`" in coverage,(name,"missing coverage documentation")
assert f"**{len(cases)} cases**" in coverage
remaining=sorted(public-added)
print(f"scipy.linalg: {len(added)}/{len(public)} exported inventory names with >=2 reference cases; {len(cases)} cases")
print("Remaining top-level names: "+", ".join(remaining))
print(f"scipy.linalg.interpolative: {len(interpolative)}/{len(inventory('.interpolative'))}; all names have >=2 reference cases")
print("BLAS/LAPACK dispatch and solve_lyapunov are intentionally omitted; raw bindings, general complex inputs, and Python classes remain unsupported.")
print("Name coverage measures the documented functional subset, not full SciPy behavioral parity.")
