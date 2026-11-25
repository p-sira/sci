# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""References for stable matrix hyperbolic evaluation.

General cases use SciPy; 3x3 tanhm references use SciPy funm with NumPy tanh
because forming coshm can be ill-conditioned even when tanhm is well-conditioned.
Diagonal cases use NumPy scalar hyperbolic functions. Trace-zero 2x2 cases use
A^2=(a^2+bc)I with exact rational arithmetic for a^2+bc and NumPy scalar values.
These spectral identities preserve tiny entries and avoid exponential overflow.

Rust comparisons use 5e-12 relative tolerance, allowing backend reduction order
and accumulation across double-angle steps. General dense cases and exact zero
entries also allow 32 binary64 epsilons times the matrix maximum: cancellation
can leave roundoff at the matrix scale. Small nonzero spectral cases permit only
one minimum subnormal of absolute error; an erroneous zero cannot pass.

"""
import csv
from pathlib import Path
from case_catalog import read_cases
import numpy as np
import scipy
from scipy import linalg

assert (np.__version__, scipy.__version__) == ("2.5.4", "1.18.1")
ROOT = Path(__file__).resolve().parents[1]
cases = read_cases(Path(__file__).with_name("hyperbolic_cases.jsonl"))

def encode(a):
    return ":".join(";".join(str(float(x)) for x in row) for row in a)

with (ROOT / "fixtures/hyperbolic_inputs.csv").open("w") as source, (ROOT / "fixtures/hyperbolic_expected.csv").open("w") as target:
    inputs = csv.writer(source, lineterminator="\n")
    outputs = csv.writer(target, lineterminator="\n")
    inputs.writerow(["id", "api", "matrix"])
    outputs.writerow(["id", "matrix"])
    for i, case in enumerate(cases):
        a = np.asarray(case["matrix"], dtype=float)
        api = case["api"]
        inputs.writerow([i, api, encode(a)])
        if np.array_equal(a, np.diag(a.diagonal())):
            expected = np.diag(getattr(np, api[:-1])(a.diagonal()))
        elif a.shape == (2, 2) and a[0, 0] == -a[1, 1]:
            from fractions import Fraction
            q = float(Fraction(float(a[0, 0]))**2 + Fraction(float(a[0, 1])) * Fraction(float(a[1, 0])))
            r = np.sqrt(abs(q))
            if api == "coshm":
                value = np.cos(r) if q < 0 else np.cosh(r)
                expected = np.eye(2) * value
            else:
                function = (np.sin if api == "sinhm" else np.tan) if q < 0 else (np.sinh if api == "sinhm" else np.tanh)
                value = 1.0 if r == 0 else function(r) / r
                expected = value * a
        elif api == "tanhm" and a.shape == (3, 3):
            expected, error = linalg.funm(a, np.tanh, disp=False)
            if error > 1e-12 or np.iscomplexobj(expected):
                raise RuntimeError("Tanh reference needs a well-conditioned real Schur evaluation")
        else:
            expected = getattr(linalg, api)(a)
        outputs.writerow([i, encode(expected)])
