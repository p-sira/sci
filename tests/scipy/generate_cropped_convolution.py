# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Independent direct convolution/correlation references for all cropping modes.

Inputs cover both length orderings, odd/even lengths, a scalar kernel and
cancellation. Integer-valued data give exact binary64 sums, so Rust requires
exact equality. Input vectors and expected results are separate CSV files.
"""
import csv
from pathlib import Path
import numpy as np
import scipy
from scipy import signal

if (np.__version__, scipy.__version__) != ("2.5.4", "1.18.1"):
    raise RuntimeError("Use NumPy 2.5.4 and SciPy 1.18.1")
root = Path(__file__).resolve().parents[1] / "fixtures"
with (root / "cropped_convolution_inputs.csv").open("w") as f, (root / "cropped_convolution_expected.csv").open("w") as g:
    inputs, outputs = csv.writer(f, lineterminator="\n"), csv.writer(g, lineterminator="\n")
    inputs.writerow(["id", "api", "x", "h", "mode"])
    outputs.writerow(["id", "values"])
    for n in range(1, 9):
        for m in range(1, 9):
            x = [float((3*i+1)%7-3) for i in range(n)]
            h = [float((5*i+2)%7-3) for i in range(m)]
            for api in ["convolve", "correlate"]:
                for mode in ["full", "same", "valid"]:
                    case = f"{api}_{n}_{m}_{mode}"
                    inputs.writerow([case, api, ";".join(map(str,x)), ";".join(map(str,h)), mode])
                    values = getattr(signal, api)(x, h, mode=mode, method="direct")
                    outputs.writerow([case, ";".join(map(str,values))])
