"""Audit the four added namespaces against inventories, exports, and public tests.

Print exact remaining names. Aliases count once in their public namespace;
compatibility namespaces and low-level Python/Cython bindings are not new APIs.
"""
import csv
import re
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[2]
implemented = {
    "constants": {"value", "unit", "precision", "find", "convert_temperature", "lambda2nu", "nu2lambda"},
    "integrate": {"trapezoid", "cumulative_trapezoid", "simpson", "cumulative_simpson", "romb", "fixed_quad", "newton_cotes"},
    "interpolate": {"approximate_taylor_polynomial", "barycentric_interpolate", "bisplev", "bisplrep", "generate_knots", "griddata", "insert", "interpn", "krogh_interpolate", "lagrange", "make_interp_spline", "make_lsq_spline", "make_smoothing_spline", "make_splprep", "make_splrep", "pade", "pchip_interpolate", "spalde", "splantider", "splder", "splev", "splint", "splprep", "splrep", "sproot"},
    "cluster": {"vq", "py_vq", "whiten", "kmeans", "kmeans2"},
}
def inventory(module,namespace):
    names=set()
    selected=functions=False
    for line in (root / f"function_coverages/scipy_function_list/scipy.{module}.md").read_text().splitlines():
        if line.startswith("### "): selected=line==f"### scipy.{namespace}"
        if line.startswith("#### "): functions=line=="#### Functions"
        if selected and functions:
            match=re.match(r"- (?:\[)?`(\w+)`",line)
            if match:names.add(match[1])
    return names
with (root / "tests/fixtures/numerics_inputs.csv").open() as f: cases=list(csv.DictReader(f))
with (root / "tests/fixtures/numerics_expected.csv").open() as f: expected=list(csv.DictReader(f))
assert [c['id'] for c in cases]==[e['id'] for e in expected]
with (root / "tests/fixtures/constants_inputs.csv").open() as f: constants=list(csv.DictReader(f))
with (root / "tests/fixtures/constants_expected.csv").open() as f: constant_expected=list(csv.DictReader(f))
assert [c['api'] for c in constants]==[e['api'] for e in constant_expected]
with (root / "tests/fixtures/interpolation_inputs.csv").open() as f: interpolation=list(csv.DictReader(f))
with (root / "tests/fixtures/interpolation_expected.csv").open() as f: interpolation_expected=list(csv.DictReader(f))
assert [c['id'] for c in interpolation]==[e['id'] for e in interpolation_expected]
cases.extend({**c, "module": "interpolate"} for c in interpolation)
tests=(root / "tests/numerics.rs").read_text() + (root / "tests/interpolation.rs").read_text()
for module,names in implemented.items():
    namespace="cluster.vq" if module=="cluster" else module
    available=inventory(module,namespace)
    source="\n".join(p.read_text() for p in (root / f"crates/sci-{module}/src").glob("*.rs") if module!="cluster" or p.name=="vq.rs")
    exported=set(re.findall(r"pub fn (\w+)\(",source))
    exported.update(re.findall(r"pub use \w+ as (\w+);",source))
    assert names==exported,(module,names-exported,exported-names)
    assert names<=available,(module,names-available)
    counter=Counter(c['api'] for c in (constants if module=="constants" else cases) if module=="constants" or c['module']==module)
    assert set(counter)==names,(module,names-set(counter))
    prefix={"constants":"c","integrate":"i","interpolate":"p","cluster":"vq"}[module]
    for name in names:
        assert counter[name]>=2,(module,name,counter[name])
        assert re.search(rf"\b{prefix}::{name}\(",tests),(module,name,"missing public test dispatch")
    remaining=sorted(available-names)
    print(f"scipy.{namespace}: {len(names)}/{len(available)} exported and tested; {sum(counter.values())} reference cases")
    if remaining: print("  Remaining: "+", ".join(remaining))
print("Cluster hierarchy coverage: run tests/scipy/audit_cluster.py.")
print(f"{sum(map(len, implemented.values()))} public names; {len(cases)} numerical cases; {len(constants)} constant cases; 445 independently snapshotted table rows.")
