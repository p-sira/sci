"""Audit distinct public SciPy I/O function names, exports and public API tests."""
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[2]
inventory=(ROOT/'function_coverages/scipy_function_list/scipy.io.md').read_text()
public=inventory.split('## Public API namespaces',1)[1].split('## Additional and compatibility namespaces',1)[0]
names=set()
in_functions=False
for line in public.splitlines():
    if line.startswith('#'):
        in_functions=line=='#### Functions'
    if in_functions and (match:=re.match(r'- \[`([^`]+)`\]',line)):
        names.add(match[1])
supported={'mminfo':'market','mmread':'market','mmwrite':'market','loadmat':'matlab',
           'savemat':'matlab','whosmat':'matlab','matfile_version':'matlab','read':'wavfile','write':'wavfile'}
deferred={'hb_read','hb_write','readsav','loadarff','varmats_from_mat'}
assert set(supported).isdisjoint(deferred)
assert names==set(supported)|deferred,(names,set(supported)|deferred)
tests=(ROOT/'tests/io.rs').read_text()
for name,module in supported.items():
    source=(ROOT/f'crates/sci-io/src/{module}.rs').read_text()
    assert re.search(rf'pub fn {name}\b',source),name
    assert re.search(rf'\b(?:io|mat|wav)::{name}\(',tests),name
report=(ROOT/'function_coverages/IO_COVERAGE.md').read_text()
for name in names:assert name in report,name
assert len(names)==14
print(f'I/O audit: {len(supported)} distinct supported names; {len(deferred)} explicitly deferred.')
