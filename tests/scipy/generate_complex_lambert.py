# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Generate independent complex Lambert W test inputs and SciPy references.

Run `uv run --offline tests/scipy/generate_complex_lambert.py`.
The checked-in JSONL cases contain *inputs only*; expected values come from
SciPy's implementation, never from Sci or its Rust backend.
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import lambertw

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "complex_lambert_cases.jsonl"
OUTPUT = ROOT.parent / "fixtures" / "complex_lambert.csv"


def create_inputs():
    cases = set()
    reals = [-30, -10, -3, -1, -0.8, -0.5, -0.4, -0.367, -0.3, -0.2, -0.1,
             -1e-6, -1e-14, 0, 1e-14, 1e-6, 0.1, 0.5, 1, 3, 10, 30]
    imags = [-30, -7, -1, -0.3, -1e-5, -1e-12, 0, 1e-12, 1e-5, 0.3, 1, 7, 30]
    for k in [-5, -2, -1, 0, 1, 2, 5]:
        for r in reals:
            for i in imags:
                if r == i == 0 and k:
                    continue
                cases.add((k, float(r), float(i)))
        for r in [-1 / np.e, -1 / np.e + 1e-8, -1 / np.e - 1e-8]:
            for i in [0.0, -1e-8, 1e-8]:
                cases.add((k, float(r), i))
    INPUTS.write_text("".join(json.dumps({"k": k, "re": r, "im": i}) + "\n"
                              for k, r, i in sorted(cases)))


def main():
    if not INPUTS.exists():
        create_inputs()
    cases = [json.loads(line) for line in INPUTS.read_text().splitlines()]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["type", "k", "re", "im", "expected_re", "expected_im"])
        for precision in [np.float64, np.float32]:
            for case in cases:
                z = complex(float(precision(case["re"])), float(precision(case["im"])))
                w = complex(lambertw(z, case["k"]))
                if not (np.isfinite(w.real) and np.isfinite(w.imag)):
                    continue
                # Float32 arguments are rounded before independent evaluation.
                writer.writerow(["f32" if precision == np.float32 else "f64", case["k"],
                                 repr(z.real), repr(z.imag), repr(w.real), repr(w.imag)])
    print(f"Generated {len(cases)} inputs and reference rows in {OUTPUT}")


if __name__ == "__main__":
    main()
