# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Generate real I/exp(-|x|)I references using independent SciPy kernels.

Run: uv run tests/scipy/generate_bessel_iv.py. Inputs are separate in
bessel_iv_cases.json. Both input and output are quantized for each precision.
Tolerance is 5e-12 relative for binary64 (uniform asymptotic exponent rounding
and AMOS/Temme differences), 3e-6 for binary32, plus two subnormal ULPs.
SciPy's real iv can overflow intermediate K values. Use the independent
DLMF 10.25.2 series with SciPy gamma factors for x < 0.001. Negative-order
ive references use DLMF 10.27.2 with positive-order SciPy ive/kve and a reduced
sine argument, avoiding AMOS reflection error near integer orders.
"""
import csv
import json
from pathlib import Path

import numpy as np
import scipy
from scipy import special


def generate():
    assert np.__version__ == "2.5.4"
    assert scipy.__version__ == "1.18.1"
    root = Path(__file__).resolve().parent
    cases = json.loads((root / "bessel_iv_cases.json").read_text())
    pairs = [(v, x) for v in cases["orders"] for x in cases["arguments"]] + cases["extra"]
    with (root.parent / "fixtures/bessel_iv.csv").open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["precision", "v", "x", "iv", "ive"])
        with np.errstate(all="ignore"):
            for dtype in (np.float32, np.float64):
                for v, x in pairs:
                    v, x = float(dtype(v)), float(dtype(x))
                    # Sci specifies +infinity at x=0 for negative fractional
                    # orders; SciPy ive returns NaN. Test this limit in Rust.
                    if x == 0 and v < 0 and not v.is_integer():
                        continue
                    values = [special.iv(v, x), special.ive(v, x)]
                    if v < 0 and not v.is_integer() and x > 0:
                        # Reduce sin(pi*a) before multiplication: negative-order
                        # AMOS loses relative accuracy near integer orders.
                        a = -v
                        sine = (-1)**round(a) * np.sin(np.pi*(a-round(a)))
                        values[1] = (special.ive(a, x) + 2/np.pi*sine*
                                     special.kve(a, x)*np.exp(-2*x))
                    if 0 < x < 0.001:
                        # DLMF 10.25.2. Logarithmic prefactors avoid SciPy's
                        # intermediate K overflow, including reflected orders.
                        a = abs(v) if v.is_integer() else v
                        term = total = 1.0
                        for k in range(1, 200):
                            term *= (x*x/4) / (k*(a+k))
                            total += term
                            if abs(term) <= abs(total)*1e-16:
                                break
                        else:
                            raise RuntimeError((v, x, "reference series"))
                        log_value = (a*(np.log(x)-np.log(2)) - special.gammaln(a+1)
                                     + np.log(abs(total)))
                        sign = special.gammasgn(a+1) * np.sign(total)
                        values = [sign*np.exp(log_value), sign*np.exp(log_value-x)]
                        if a == 1 and x < 1e-300:
                            values = [x/2, x/2]
                    writer.writerow([dtype.__name__.replace("float", "f"), v, x,
                                     *[float(dtype(value)) for value in values]])


if __name__ == "__main__":
    generate()
