# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Power-transform regressions, with independent SciPy/Decimal references.

SciPy supplies transform references. Power means use their defining identity
with 400-digit Decimal arithmetic: direct binary64 powers can round to one
near p=0 or overflow at large p. For |p|<1e-100 the geometric limit differs
by less than binary64 precision for these data. Inputs are stored separately.
Relative tolerances cover binary64 logarithm/exponential rounding (including
restoring large exponents); one minimum subnormal covers final rounding.
"""
import csv
from decimal import Decimal, localcontext
from pathlib import Path
from case_catalog import read_cases

import numpy as np
import scipy
from scipy import special

if (np.__version__, scipy.__version__) != ("2.5.4", "1.18.1"):
    raise RuntimeError("Use NumPy 2.5.4 and SciPy 1.18.1")
root = Path(__file__).resolve().parents[1]
cases = read_cases(Path(__file__).with_name("power_stability_cases.jsonl"))

def reference(case):
    x, p = case["x"], case["p"]
    if case["api"] == "pmean":
        with localcontext() as ctx:
            ctx.prec = 400
            logs = [Decimal(v).ln() for v in x]
            if abs(p) < 1e-100:
                result = (sum(logs) / len(x)).exp()
            else:
                power = Decimal(p)
                result = ((sum((power * v).exp() for v in logs) / len(x)).ln() / power).exp()
            return [float(result)]
    if case["api"] == "yeojohnson":
        # The documented identity in terms of Box-Cox(1+x) avoids premature
        # underflow in SciPy stats.yeojohnson for subnormal observations.
        return [float(special.boxcox1p(v, p) if v >= 0 else -special.boxcox1p(-v, 2-p)) for v in x]
    return getattr(special, case["api"])(x, p).tolist()

with (root / "fixtures/power_stability_inputs.csv").open("w") as f, (root / "fixtures/power_stability_expected.csv").open("w") as g:
    inputs, outputs = csv.writer(f, lineterminator="\n"), csv.writer(g, lineterminator="\n")
    inputs.writerow(["id", "api", "x", "power"])
    outputs.writerow(["id", "values", "rtol", "atol"])
    for case in cases:
        inputs.writerow([case["id"], case["api"], ";".join(map(str, case["x"])), case["p"]])
        outputs.writerow([case["id"], ";".join(map(str, reference(case))), 3e-13, 5e-324])
