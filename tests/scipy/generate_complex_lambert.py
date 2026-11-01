# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Generate independent complex Lambert W test inputs and SciPy references.

Run `uv run --offline tests/scipy/generate_complex_lambert.py`.
The checked-in fixture exercises every branch and precision on all real-axis
boundary inputs, both near-cut banks, a wide complex-plane grid, and the
coalescing -1/e root. The larger Cartesian product remains reproducible with
`--full --output /tmp/complex_lambert_full.csv` and can be supplied to the
Rust test with SCI_LAMBERTW_FIXTURE=/tmp/complex_lambert_full.csv.
Expected values always come from SciPy, not from the Rust implementation.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.special import lambertw

OUTPUT = Path(__file__).resolve().parents[1] / "fixtures" / "complex_lambert.csv"
BRANCHES = (-5, -2, -1, 0, 1, 2, 5)
REALS = (-30, -10, -3, -1, -0.8, -0.5, -0.4, -0.367, -0.3, -0.2, -0.1,
         -1e-6, -1e-14, 0, 1e-14, 1e-6, 0.1, 0.5, 1, 3, 10, 30)
IMAGS = (-30, -7, -1, -0.3, -1e-5, -1e-12, 0, 1e-12, 1e-5, 0.3, 1, 7, 30)


def create_inputs(*, full=False):
    cases = set()
    for k in BRANCHES:
        # Every real parameter is tested on the negative-cut banks and axis.
        for r in REALS:
            for i in IMAGS if full else (-1e-12, 0, 1e-12):
                if r == i == 0 and k:
                    continue
                cases.add((k, float(r), float(i)))
        if not full:
            # Large arguments, near-cut negative arguments, the origin and
            # positive arguments each exercise *every* imaginary magnitude.
            # Full Cartesian products add many interior points with the same
            # recurrence/branch path; retain them via --full when investigating.
            for r in (-30, -0.4, 0, 1, 30):
                for i in IMAGS:
                    if r == i == 0 and k:
                        continue
                    cases.add((k, float(r), float(i)))
        # All branches and both precisions around the ill-conditioned double
        # root; this region exposed the previous Windows/macOS regression.
        for r in [-1 / np.e, -1 / np.e + 1e-8, -1 / np.e - 1e-8]:
            for i in [0.0, -1e-8, 1e-8]:
                cases.add((k, float(r), i))
    return sorted(cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="generate the exhaustive grid")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="reference CSV destination")
    args = parser.parse_args()
    if args.full and args.output.resolve() == OUTPUT.resolve():
        parser.error("--full requires --output to avoid overwriting the compact fixture")
    cases = create_inputs(full=args.full)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with args.output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["type", "k", "re", "im", "expected_re", "expected_im"])
        for precision in [np.float64, np.float32]:
            for k, re, im in cases:
                z = complex(float(precision(re)), float(precision(im)))
                w = complex(lambertw(z, k))
                if not (np.isfinite(w.real) and np.isfinite(w.imag)):
                    # SciPy's evaluation at the exact double root returns
                    # NaN on these two sheets; the analytic value W=-1 is
                    # checked separately in the Rust regression test.
                    if (precision == np.float64 and k in (-1, 0)
                            and z.real == -1 / np.e and z.imag == 0.0):
                        continue
                    raise ValueError(f"Unexpected nonfinite SciPy reference: {precision}, {k}, {z}")
                # Float32 arguments are rounded before independent evaluation.
                writer.writerow(["f32" if precision == np.float32 else "f64", k,
                                 repr(z.real), repr(z.imag), repr(w.real), repr(w.imag)])
                rows += 1
    print(f"Generated {len(cases)} inputs and {rows} reference rows in {args.output}")


if __name__ == "__main__":
    main()
