"""Audit every sparse inventory name, Rust exports, and reference fixture coverage.

Run: python tests/scipy/audit_sparse.py [--write]
No SciPy installation required. --write refreshes the inventory table only.
"""
import csv
import re
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
PRIMARY = {'scipy.sparse', 'scipy.sparse.linalg', 'scipy.sparse.csgraph'}
source = ROOT/'function_coverages/scipy_function_list/scipy.sparse.md'
namespace = None
in_functions = False
inventory = []
for line in source.read_text().splitlines():
    if line.startswith('### '):
        namespace = line[4:]
        in_functions = False
    elif line.startswith('#### '):
        in_functions = line == '#### Functions'
    elif in_functions and line.startswith('- '):
        inventory.append((namespace, re.search(r'`([^`]+)`', line).group(1)))
assert len(inventory) == 148
sources = {
    'scipy.sparse': (ROOT/'crates/sci-sparse/src/lib.rs').read_text(),
    'scipy.sparse.linalg': (ROOT/'crates/sci-sparse/src/linalg.rs').read_text(),
    'scipy.sparse.csgraph': '',
}
exports = {}
for ns, text in sources.items():
    exports[ns] = set(re.findall(r'pub fn (\w+)\(', text)) | set(re.findall(r'pub use \w+ as (\w+);', text))
fixtures = {r['api'] for r in csv.DictReader((ROOT/'tests/fixtures/sparse_inputs.csv').open())}
extra = exports['scipy.sparse'] - {name for ns,name in inventory if ns=='scipy.sparse'}
# Methods are tested separately through the public factorization record.
extra |= exports['scipy.sparse.linalg'] - {name for ns,name in inventory if ns=='scipy.sparse.linalg'} - {'solve'}
for name in extra:
    assert name in fixtures, f'uncovered public helper {name}'
rows = ['| Namespace | Inventory name | v0.1.0 disposition | Sci API |', '| --- | --- | --- | --- |']
counts = {}
for ns,name in inventory:
    primary = ns if ns in PRIMARY else 'scipy.sparse.linalg' if ns.startswith('scipy.sparse.linalg.') else 'scipy.sparse'
    if name in exports[primary]:
        assert name in fixtures, f'missing SciPy reference fixture {name}'
        path = 'sparse::' + ('linalg::' if primary.endswith('.linalg') else '') + name
        status = 'Supported adaptation' if ns in PRIMARY else 'Compatibility alias; shared implementation'
        target = f'`{path}`'
    else:
        status = 'Outside v0.1.0'
        target = '—'
    rows.append(f'| `{ns}` | `{name}` | {status} | {target} |')
    if ns in PRIMARY:
        done,total=counts.get(ns,(0,0));counts[ns]=(done+(name in exports[primary]),total+1)
report = ROOT/'function_coverages/SPARSE_COVERAGE.md'
text = report.read_text()
prefix = text.split('<!-- inventory -->')[0]
wanted = prefix+'<!-- inventory -->\n\n'+'\n'.join(rows)+'\n'
if '--write' in sys.argv:
    report.write_text(wanted)
else:
    assert text == wanted, 'sparse coverage inventory is stale; run with --write'
assert '"lu_solve"' in (ROOT/'tests/sparse.rs').read_text()
print(f'Sparse inventory: {counts}; {len(extra)} extra functional helpers; all 148 names audited.')
