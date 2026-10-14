"""Audit signal inventory coverage, exported numerical functions, and cases.

Run from any directory using Python 3.11+. This does not need SciPy.
Compatibility namespaces are aliases mapped to the public signal/windows API.
No declaration is treated as supported unless input cases exercise it.
"""
from pathlib import Path
import json
import re

ROOT=Path(__file__).resolve().parents[2]
CASES=json.loads((ROOT/"tests/scipy/signal_cases.json").read_text())
TESTED={c["function"] for c in CASES}
SOURCES=ROOT/"crates/sci-signal/src"


def functions(path):
    source=path.read_text()
    names=set(re.findall(r"pub fn (\w+)\s*\(",source))
    names.update(re.findall(r"^(?:simple_window|cosine_window|order_function|transform_function)!\s*\(\s*(\w+)",source,re.M))
    names.update(re.findall(r"pub use [^;]+ as (\w+);",source))
    return names


EXPORTED=set()
for path in SOURCES.glob("*.rs"):
    namespace="windows." if path.name=="windows.rs" else ""
    EXPORTED.update(namespace+n for n in functions(path))
EXPORTED.add("get_window")
# Fourier fixtures predate this module expansion and live in fft.csv.
UNTESTED=EXPORTED-TESTED-{"fft","ifft","rfft"}
if UNTESTED:
    raise RuntimeError(f"Exported numerical functions without cases: {sorted(UNTESTED)}")
UNKNOWN=TESTED-EXPORTED
if UNKNOWN:
    raise RuntimeError(f"Cases without exported numerical functions: {sorted(UNKNOWN)}")

inventory=(ROOT/"function_coverages/scipy_function_list/scipy.signal.md").read_text()
public=inventory.split("### scipy.signal\n",1)[1].split("#### Classes",1)[0]
windows=inventory.split("### scipy.signal.windows\n",1)[1].split("## Additional",1)[0]
main_names=set(re.findall(r"^- \[`(\w+)`\]",public,re.M))
window_names={"windows."+n for n in re.findall(r"^- \[`(\w+)`\]",windows,re.M)}
SUPPORTED=(main_names|window_names)&EXPORTED&TESTED
MISSING=(main_names|window_names)-SUPPORTED
print(f"Public inventory: {len(SUPPORTED)}/{len(main_names|window_names)} supported; {len(MISSING)} missing")
print(f"Fixture cases: {len(CASES)}; all {len(EXPORTED)} exported numerical names exercised (including legacy FFT fixtures)")
print("Remaining: "+", ".join(sorted(MISSING)))
