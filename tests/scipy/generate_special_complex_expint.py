# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1", "mpmath==1.3.0"]
# ///
"""Create independent complex E1/Ei references from pinned SciPy.

Run: uv run --offline tests/scipy/generate_special_complex_expint.py
The input CSV contains only arguments, including signed imaginary zeros.
Float32 inputs are quantized before evaluating with SciPy complex128; this
tests the generic public API's binary64 computation and float32 conversion.
"""

import csv
import math
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy
from scipy import special

if (np.__version__, scipy.__version__) != ("2.5.4", "1.18.1"):
    raise RuntimeError("Use NumPy 2.5.4 and SciPy 1.18.1")

INPUT = Path(__file__).with_name("special_complex_expint_inputs.csv")
OUTPUT = Path(__file__).parents[1] / "fixtures" / "special_complex_expint.csv"


def main():
    with INPUT.open(newline="") as inputs, OUTPUT.open("w", newline="") as target:
        rows = list(csv.DictReader(inputs))
        writer = csv.writer(target)
        writer.writerow(["precision", "re", "im", "e1_re", "e1_im", "ei_re", "ei_im"])
        for precision in [np.float64, np.float32]:
            for row in rows:
                # Keep the signed zero through the binary64 -> binary32 conversion.
                re = float(precision(float(row["re"])))
                im = float(precision(float(row["im"])))
                z = complex(re, im)
                e1 = complex(special.exp1(z))
                ei = complex(special.expi(z))
                if im != 0.0 and (
                    (re <= -36.0 and abs(im) < 1.0)
                    or (re >= 35.0 and abs(im) < 0.1)
                    or (re > 0.0 and abs(im) < 1e-7)
                ):
                    # SciPy's continuation sometimes retains an extra +/- pi
                    # in a tiny component near the real axis, where this is a
                    # large componentwise error. Use 100-digit independent Ei
                    # and E1 values only in these narrow cancellation regions.
                    with mp.workdps(100):
                        mp_z = mp.mpc(re, im)
                        e1 = complex(mp.e1(mp_z))
                        ei = complex(mp.ei(mp_z))
                if re < 0 and im == 0.0:
                    # The branch is specified by the sign of imaginary zero.
                    # Older SciPy releases lost the lower-bank sign for x<-30;
                    # evaluate the real Ei value and insert the exact jump.
                    e1 = complex(-float(special.expi(-re)),
                                 math.pi if math.copysign(1.0, im) < 0 else -math.pi)
                if not all(np.isfinite(v) for v in [e1.real, e1.imag, ei.real, ei.imag]):
                    continue
                if precision == np.float32 and any(
                    not np.isfinite(np.float32(v)) for v in [e1.real, e1.imag, ei.real, ei.imag]
                ):
                    continue
                expected = [e1.real, e1.imag, ei.real, ei.imag]
                if precision == np.float32:
                    expected = [float(np.float32(v)) for v in expected]
                writer.writerow([
                    "f64" if precision == np.float64 else "f32",
                    repr(re), repr(im), *(repr(v) for v in expected),
                ])


if __name__ == "__main__":
    main()
