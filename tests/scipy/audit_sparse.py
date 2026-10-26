"""Audit modern sparse inventory, exports, fixtures, and explicit legacy exclusions.

Run: python tests/scipy/audit_sparse.py [--write]. No SciPy needed.
Compatibility namespaces, runtime type utilities and redundant aliases are
excluded. Old alias fixture labels exercise canonical public numerical APIs.
"""
from api_disposition import ALIASES, EXCLUDED
import csv
import re
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
PRIMARY = {'scipy.sparse', 'scipy.sparse.linalg', 'scipy.sparse.csgraph'}
LEGACY = {'rand', 'use_solver', 'isspmatrix', *(f'isspmatrix_{f}' for f in ('bsr','coo','csc','csr','dia','dok','lil'))}
namespace = None
in_functions = False
inventory = []
for line in (ROOT/'function_coverages/scipy_function_list/scipy.sparse.md').read_text().splitlines():
    if line.startswith('### '):
        namespace = line[4:]
        in_functions = False
    elif line.startswith('#### '):
        in_functions = line == '#### Functions'
    elif in_functions and line.startswith('- '):
        inventory.append((namespace, re.search(r'`([^`]+)`', line).group(1)))
assert len(inventory) == 148
base=ROOT/'crates/sci-sparse/src'
sources={
 'scipy.sparse':'\n'.join((base/f'{name}.rs').read_text() for name in ('lib','array','io')),
 'scipy.sparse.csgraph':(base/'csgraph.rs').read_text(),
 'scipy.sparse.linalg':(base/'linalg.rs').read_text()+'\n'+'\n'.join(p.read_text() for p in sorted((base/'linalg').glob('*.rs'))),
}
exports={ns:set(re.findall(r'pub fn (\w+)(?:<[^\n]*>)?\(',text))|set(re.findall(r'pub use \w+ as (\w+);',text)) for ns,text in sources.items()}
# Record methods are exercised explicitly, not counted as namespace functions.
exports['scipy.sparse.linalg']-={'new','solve'}
fixtures=set()
for file in ('sparse_inputs.csv','sparse_parity_inputs.csv'):
    fixtures|={r['api'] for r in csv.DictReader((ROOT/'tests/fixtures'/file).open())}
fixtures |= {ALIASES['sparse'].get(name, name) for name in fixtures}
if 'safely_cast_index_arrays' in fixtures:
    fixtures |= {'csr_indices_i32', 'csr_indices_i64'}
rows=['| Namespace | Inventory name | Current disposition | Sci API |','| --- | --- | --- | --- |']
counts={}
for ns,name in inventory:
    if ns not in PRIMARY:
        status,target='Excluded legacy compatibility namespace','—'
    elif name in EXCLUDED['sparse']:
        assert name not in exports[ns], f'removed API exported: {name}'
        replacements = {'safely_cast_index_arrays': '`sparse::csr_indices_i32`, `sparse::csr_indices_i64`', 'aslinearoperator': '`LinearOperator::try_from`'}
        status, target = 'Removed or redesigned for Rust', replacements.get(name, '—')
    elif name in ALIASES['sparse']:
        canonical = ALIASES['sparse'][name]
        assert name not in exports[ns] and canonical in exports[ns]
        assert canonical in fixtures
        status, target = 'Removed redundant alias', f'`{canonical}`'
    elif name in LEGACY:
        assert name not in exports[ns],f'legacy function unexpectedly exported: {name}'
        status,target='Excluded legacy API','—'
    else:
        assert name in exports[ns],f'missing modern API: {ns}.{name}'
        assert name in fixtures,f'missing independent reference fixture: {ns}.{name}'
        status='Supported adaptation'
        path='sparse::'+('linalg::' if ns.endswith('.linalg') else 'csgraph::' if ns.endswith('.csgraph') else '')+name
        target=f'`{path}`'
    rows.append(f'| `{ns}` | `{name}` | {status} | {target} |')
    if ns in PRIMARY:
        done,total=counts.get(ns,(0,0));counts[ns]=(done+(name not in LEGACY|EXCLUDED['sparse']|set(ALIASES['sparse'])),total+(name not in LEGACY))
extra=set()
for ns,names in exports.items():
    extra|=names-{name for n,name in inventory if n==ns}
for name in extra:assert name in fixtures,f'untested functional helper: {name}'
report=ROOT/'function_coverages/SPARSE_COVERAGE.md'
text=report.read_text();wanted=text.split('<!-- inventory -->')[0]+'<!-- inventory -->\n\n'+'\n'.join(rows)+'\n'
if '--write' in sys.argv:report.write_text(wanted)
else:assert text==wanted,'sparse coverage inventory stale; run with --write'
assert '"lu_solve"' in (ROOT/'tests/sparse.rs').read_text()
print(f'Sparse modern coverage: {counts}; {len(extra)} functional helpers; all 148 inventory names audited.')
