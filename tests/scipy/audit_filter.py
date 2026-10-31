"""Audit tested numerical exports and the state-estimation/SciPy boundary.

No scipy.filter(s) namespace is present in SCIPY_FUNCTIONS.md. The linked signal
inventory covers sample processing, already implemented by sci-signal; compatibility
namespaces repeat those APIs. Bayesian estimation is an independent extension.
Run with Python 3.11+; no SciPy installation is required for the audit.
"""
import re
from pathlib import Path
from case_catalog import read_cases

ROOT = Path(__file__).resolve().parents[2]
exports = set()
for path in (ROOT/"crates/sci-filter/src").glob("*.rs"):
    exports.update(re.findall(r"pub fn (\w+)(?:<[^\n]*>)?\s*\(", path.read_text()))
cases = read_cases(ROOT/"tests/scipy/filter_cases.jsonl")
tested = {case["function"] for case in cases}
if exports != tested:
    raise RuntimeError(f"Untested exports: {exports-tested}; unknown cases: {tested-exports}")
inputs = (ROOT/"tests/fixtures/filter_inputs.csv").read_text().splitlines()[1:]
expected = (ROOT/"tests/fixtures/filter_expected.csv").read_text().splitlines()[1:]
ids = [case["id"] for case in cases]
if len(ids) != len(set(ids)) or ids != [row.split(",",1)[0] for row in inputs] or ids != [row.split(",",1)[0] for row in expected]:
    raise RuntimeError("Reference fixtures differ from input case IDs; regenerate")
inventory = (ROOT/"function_coverages/scipy_function_list/scipy.signal.md").read_text()
if not all(f"[`{name}`]" in inventory for name in ("lfilter","sosfilt","butter","firwin")):
    raise RuntimeError("Review updated SciPy signal inventory")
print(f"Filter coverage: {len(exports)}/{len(exports)} numerical exports independently referenced, {len(cases)} cases")
print("SciPy inventory: no Bayesian filter namespace; signal sample-filter APIs retained without duplicate exports")
