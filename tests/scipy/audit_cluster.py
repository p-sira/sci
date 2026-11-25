"""Audit public cluster inventories, exports, fixtures, and Rust dispatch."""
import csv
import re
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def inventory(namespace):
    names=set();selected=functions=False
    for line in (ROOT/'function_coverages/scipy_function_list/scipy.cluster.md').read_text().splitlines():
        if line.startswith('### '):selected=line==f'### scipy.cluster.{namespace}'
        if line.startswith('#### '):functions=line=='#### Functions'
        if selected and functions:
            match=re.match(r'- (?:\[)?`(\w+)`',line)
            if match:names.add(match[1])
    return names
source=(ROOT/'crates/sci-cluster/src/hierarchy.rs').read_text()
exported=set(re.findall(r'pub fn (\w+)\(',source))
exported.update(re.findall(r'pub use \w+ as (\w+);',source))
with (ROOT/'tests/fixtures/cluster_inputs.csv').open() as f:cases=list(csv.DictReader(f))
with (ROOT/'tests/fixtures/cluster_expected.csv').open() as f:expected=list(csv.DictReader(f))
assert [c['id'] for c in cases]==[c['id'] for c in expected]
counts=Counter('max_rstat' if c['api']=='maxRstat' else c['api'] for c in cases)
assert set(counts)==exported
assert all(counts[name]>=2 for name in exported)
tests=(ROOT/'tests/cluster.rs').read_text()
for name in exported:assert f'h::{name}(' in tests,(name,'missing public test')
available = (inventory('hierarchy') - {'maxRstat'}) | {'max_rstat'}
remaining=available-exported
assert exported<=available
report=(ROOT/'function_coverages/CLUSTER_COVERAGE.md').read_text()
assert all(f'`{name}`' in report for name in remaining)
print(f'scipy.cluster.hierarchy: {len(exported)}/30 exported and tested; {len(cases)} SciPy reference cases')
print('Remaining: '+', '.join(sorted(remaining)))
print('scipy.cluster.vq: 4 retained names; py_vq removed; covered by audit_numerics.py')
