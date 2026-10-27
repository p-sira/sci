# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Generate scalar f32/f64 references after quantizing each floating input.

Run with ``uv run tests/scipy/generate_special_float.py``. Inputs and integer
parameter types are recorded separately in special_float_cases.json. Expected
values come exclusively from SciPy or the independent reference identities in
generate_special.py. Rust tests read the committed CSV without Python. Sci-specific finite-parameter domains
and limiting IEEE conventions that differ from SciPy are checked separately by
nonfinite_values_follow_sci_domains_and_limits in tests/special_float.rs.
"""
import csv
import json
from pathlib import Path

import numpy as np
import scipy
from generate_special import reference


def generate():
    assert np.__version__ == "2.5.4"
    assert scipy.__version__ == "1.18.1"
    root = Path(__file__).resolve().parent
    cases = json.loads((root / "special_float_cases.json").read_text())
    with (root.parent / "fixtures" / "special_float.csv").open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["precision", "function", "arguments", "expected"])
        with np.errstate(all="ignore"):
            for dtype in (np.float32, np.float64):
                for case in cases:
                    args = [float(dtype(x)) if kind == "T" else x
                            for x, kind in zip(case["args"], case["types"], strict=True)]
                    expected = reference(case["name"], args)
                    # Lambert W is exposed as its real principal branch.
                    if np.iscomplexobj(expected):
                        if expected.imag != 0:
                            expected = np.nan
                        else:
                            expected = expected.real
                    expected = float(dtype(np.asarray(expected).reshape(())))
                    writer.writerow([dtype.__name__.replace("float", "f"), case["name"],
                                     ";".join(str(x).lower() for x in args), expected])


if __name__ == "__main__":
    generate()
