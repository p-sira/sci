# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Generate basic interoperability references using SciPy's double-real BLAS.

Inputs live in linalg_interop_cases.jsonl; expected values come exclusively
from SciPy/NumPy, independently of Sci and its Rust backends.
"""
import csv
from pathlib import Path
from case_catalog import read_cases

import numpy as np
import scipy
from scipy.linalg import blas

assert (np.__version__, scipy.__version__) == ("2.5.4", "1.18.1")
ROOT = Path(__file__).resolve().parents[2]


def encode(a):
    if a.ndim == 1:
        return ";".join(map(str, a))
    return ":".join(encode(row) for row in a)


cases = read_cases(ROOT / "tests/scipy/linalg_interop_cases.jsonl")
with (ROOT / "tests/fixtures/linalg_interop_inputs.csv").open("w", newline="") as inputs, (
    ROOT / "tests/fixtures/linalg_interop_expected.csv"
).open("w", newline="") as expected:
    iw, ew = csv.writer(inputs), csv.writer(expected)
    iw.writerow(["id", "a", "b", "x"])
    ew.writerow(["id", "matmul", "matvec", "transpose", "dot"])
    for i, case in enumerate(cases):
        a, b, x = (np.asarray(case[key], dtype=np.float64) for key in ("a", "b", "x"))
        iw.writerow([i, encode(a), encode(b), encode(x)])
        ew.writerow([i, encode(blas.dgemm(1.0, a, b)), encode(blas.dgemv(1.0, a, x)),
                     encode(a.T), blas.ddot(x, x[::-1])])
