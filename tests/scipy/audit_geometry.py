"""Audit numerical spatial inventory, exported APIs, and fixture coverage."""
import csv
import re
from pathlib import Path
root=Path(__file__).resolve().parents[2]
inventory=set()
functions=False
for line in (root/'function_coverages/scipy_function_list/scipy.spatial.md').read_text().splitlines():
    if line.startswith('#### '):functions=line=='#### Functions'
    if functions:
        match=re.match(r'- \[`(\w+)`\]',line)
        if match:inventory.add(match[1])
excluded={'convex_hull_plot_2d','delaunay_plot_2d','voronoi_plot_2d'}
expected=inventory-excluded
exported=set()
for source in (root/'crates/sci-spatial/src').glob('*.rs'):
    text=source.read_text()
    exported.update(re.findall(r'pub fn (\w+)\(',text))
    exported.update(re.findall(r'vector_distance!\(\s*(\w+)',text))
with (root/'tests/fixtures/geometry_inputs.csv').open() as f:
    cases=list(csv.DictReader(f))
with (root/'tests/fixtures/geometry_expected.csv').open() as f:
    results=list(csv.DictReader(f))
covered={c['api'] for c in cases}
assert exported==expected,(expected-exported,exported-expected)
assert covered==expected,(expected-covered,covered-expected)
assert [c['id'] for c in cases]==[c['id'] for c in results]
for name in expected:assert sum(c['api']==name for c in cases)>=2,name
print(f'{len(expected)} numerical functions exported and covered by {len(cases)} SciPy cases; 3 plotting functions excluded.')
