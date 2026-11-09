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
# Algorithm-specific Rust adaptation of the dense dogbox subset.
dogbox_apis={'dogbox', 'dogbox_with_jacobian', 'try_dogbox', 'try_dogbox_with_jacobian'}
assert dogbox_apis <= exported
covered.add('least_squares')
with (ROOT/'tests/fixtures/optimize_inputs.csv').open() as f:cases=list(csv.DictReader(f))
with (ROOT/'tests/fixtures/optimize_expected.csv').open() as f:expected=list(csv.DictReader(f))
assert [c['id'] for c in cases]==[c['id'] for c in expected]
counts=Counter(c['api'] for c in cases)
assert set(counts)==covered-{'minimize', 'least_squares'}
assert all(n>=4 for n in counts.values())
tests=(ROOT/'tests/optimize_helpers.rs').read_text()
for name in counts:assert f'o::derivatives::{name}(' in tests
assert 'optimize::minimize::minimize(' in (ROOT/'tests/scipy_compatibility.rs').read_text()
with (ROOT/'tests/fixtures/dogbox_inputs.csv').open() as f:dogbox_cases=list(csv.DictReader(f))
with (ROOT/'tests/fixtures/dogbox_expected.csv').open() as f:dogbox_expected=list(csv.DictReader(f))
assert [c['id'] for c in dogbox_cases]==[c['id'] for c in dogbox_expected]
assert len(dogbox_cases)>=16
dogbox_tests=(ROOT/'tests/dogbox.rs').read_text()
for name in dogbox_apis:assert re.search(r'\b'+name+r'\(', dogbox_tests)
remaining=names-covered
report=(ROOT/'function_coverages/OPTIMIZE_COVERAGE.md').read_text()
assert all(f'`{name}`' in report for name in remaining)
print(f'scipy.optimize: {len(covered)}/{len(names)} names with tested supported native signatures; {len(cases)} new reference cases')
print(f'least_squares: dense dogbox subset, {len(dogbox_cases)} reference cases; other methods remain absent')
print('Remaining: '+', '.join(sorted(remaining)))
print('elementwise and Cython root solvers remain absent; compatibility namespaces reuse public algorithms')
