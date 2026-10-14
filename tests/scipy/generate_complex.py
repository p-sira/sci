# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Generate independent elementary complex references.

Run: uv run tests/scipy/generate_complex.py (or the pinned project .venv).
SciPy has no complex arithmetic namespace. NumPy provides the elementary
complex references; polar conversion uses its documented magnitude/angle
identities. Cases are stored separately in complex_inputs.csv.
Rust uses 3e-14 absolute and relative tolerances on moderate values; nonzero
results below 1e-6 use relative comparisons so underflow cannot pass as zero.
IEEE classifications and selected signed-zero branch results are checked
explicitly. Inverse functions include subnormal components, extreme magnitudes,
and values adjacent to branch points. Logarithm includes near-unit inputs;
exponential and trigonometric cases check finite components despite overflowing
exponential magnitudes. Tangent cases cover pole neighbors and the subnormal
transition. Extreme real powers use extended precision and square-root identities
where complex128 logarithm rounding or phase cancellation harms the reference.
Finite multiplication and squared magnitude use exact rational arithmetic to
resolve cancellation below longdouble precision and combine tiny squares
before subnormal rounding.
"""
import csv
from fractions import Fraction
import math
from pathlib import Path
import numpy as np
import scipy

if (np.__version__, scipy.__version__) != ("2.5.4", "1.18.1"):
    raise RuntimeError("Use NumPy 2.5.4 and SciPy 1.18.1")
ROOT = Path(__file__).resolve().parents[1] / "fixtures"
ALIASES = dict(subtract="subtract", multiply="multiply", divide="divide",
               negate="negative", conjugate="conjugate", abs="absolute",
               arg="angle", ln="log", asin="arcsin", acos="arccos",
               atan="arctan", asinh="arcsinh", acosh="arccosh", atanh="arctanh")
def rounded(value):
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf

with (ROOT / "complex_inputs.csv").open() as source, (ROOT / "complex_expected.csv").open("w", newline="") as target:
    writer = csv.writer(target, lineterminator="\n")
    writer.writerow(["id", "first", "second"])
    with np.errstate(all="ignore"):
        for case in csv.DictReader(source):
            name = case["api"]
            z = np.complex128(complex(float(case["re"]), float(case["im"])))
            w = np.complex128(complex(float(case["other_re"]), float(case["other_im"])))
            if name in ("add", "subtract", "multiply", "divide"):
                if name == "multiply" and np.isfinite(z) and np.isfinite(w):
                    # Exact rational binary64 arithmetic resolves cancellation
                    # below longdouble precision and overflowing products.
                    a,b,c,d = map(Fraction, (float(z.real),float(z.imag),float(w.real),float(w.imag)))
                    writer.writerow([case["id"], rounded(a*c-b*d), rounded(a*d+b*c)])
                    continue
                elif name == "divide" and np.isfinite(z) and np.isfinite(w):
                    # Wider independent arithmetic avoids NumPy complex128's
                    # overflowing intermediate sum for (1.7e308+1.7e308j)/(1+1j).
                    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
                        raise RuntimeError("Division fixtures require extended-range longdouble")
                    result = np.divide(np.clongdouble(z), np.clongdouble(w))
                else:
                    result = getattr(np, ALIASES.get(name, name))(z, w)
            elif name == "powi" and abs(int(case["other_re"])) > 2**30:
                # Exact unit-root cycle: NumPy's general complex power uses
                # angle reduction and loses ~3e-7 for i**(2**31-1).
                if z == 1:
                    result = np.complex128(1)
                elif z == 1j:
                    result = np.complex128([1, 1j, -1, -1j][int(case["other_re"]) % 4])
                else:
                    raise ValueError("Large integer reference needs an exact unit root")
            elif name in ("powf", "powc") and w.imag == 0 and float(case["other_re"]) in (0.5, -0.5) and z != 0 and np.isfinite(z):
                # Principal square-root identities avoid the general power's
                # spurious cos(pi/2) component on the negative real axis.
                result = np.sqrt(np.clongdouble(z))
                if float(case["other_re"]) == -0.5:
                    result = np.reciprocal(result)
            elif name in ("powf", "powc") and case["id"].startswith("power_"):
                # complex128 power evaluates exp(p*log(z)); at extreme scales
                # logarithm rounding can exceed 5e-14 relative error. Extended
                # arithmetic supplies a more accurate independent reference.
                if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
                    raise RuntimeError("Extreme powers require extended-range longdouble")
                result = np.power(np.clongdouble(z), np.longdouble(case["other_re"]))
            elif name in ("powf", "powc", "powi"):
                result = np.power(z, w if name == "powc" else float(case["other_re"]) if name == "powf" else int(case["other_re"]))
            elif name == "reciprocal":
                result = np.reciprocal(z)
            elif name == "norm_sqr":
                if np.isfinite(z):
                    a,b = Fraction(float(z.real)),Fraction(float(z.imag))
                    result = rounded(a*a+b*b)
                else:
                    result = np.real(z)**2 + np.imag(z)**2
            elif name == "to_polar":
                writer.writerow([case["id"], float(np.abs(z)), float(np.angle(z))])
                continue
            elif name == "from_polar":
                result = float(case["re"]) * complex(np.cos(float(case["im"])), np.sin(float(case["im"])))
            else:
                result = getattr(np, ALIASES.get(name, name))(z)
            writer.writerow([case["id"], float(np.real(result)), float(np.imag(result))])
