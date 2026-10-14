"""Audit SciPy optimize names and native signatures; compatibility aliases repeat names."""
import csv
import re
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
names=set();selected=functions=False
for line in (ROOT/'function_coverages/scipy_function_list/scipy.optimize.md').read_text().splitlines():
    if line.startswith('### '):selected=line=='### scipy.optimize'
    if line.startswith('#### '):functions=line=='#### Functions'
    if selected and functions:
        match=re.match(r'- (?:\[)?`(\w+)`',line)
        if match:names.add(match[1])
source='\n'.join(p.read_text() for p in (ROOT/'crates/sci-optimize/src').glob('*.rs'))
exported=set(re.findall(r'pub fn (\w+)(?:<|\()',source))
covered=names&exported
with (ROOT/'tests/fixtures/optimize_inputs.csv').open() as f:cases=list(csv.DictReader(f))
with (ROOT/'tests/fixtures/optimize_expected.csv').open() as f:expected=list(csv.DictReader(f))
assert [c['id'] for c in cases]==[c['id'] for c in expected]
counts=Counter(c['api'] for c in cases)
assert set(counts)==covered-{'minimize'}
assert all(n>=4 for n in counts.values())
tests=(ROOT/'tests/optimize_helpers.rs').read_text()
for name in counts:assert f'o::{name}(' in tests
assert 'optimize::minimize(' in (ROOT/'tests/scipy_compatibility.rs').read_text()
remaining=names-covered
report=(ROOT/'function_coverages/OPTIMIZE_COVERAGE.md').read_text()
assert all(f'`{name}`' in report for name in remaining)
print(f'scipy.optimize: {len(covered)}/{len(names)} names with tested supported native signatures; {len(cases)} new reference cases')
print('Remaining: '+', '.join(sorted(remaining)))
print('elementwise and Cython root solvers remain absent; compatibility namespaces reuse public algorithms')
