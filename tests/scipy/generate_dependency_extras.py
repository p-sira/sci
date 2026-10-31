# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Independent references for dependency utilities absent from SciPy's inventory.

Run with uv run tests/scipy/generate_dependency_extras.py. Inputs are separate
JSON. Harmonic references use SciPy digamma or the defining finite sum in
NumPy longdouble (including negative real orders). Combinatorial references
use exact Python integer coefficients before taking logarithms. Complex roots
use NumPy clongdouble principal powers to retain extreme magnitude accuracy.
No Sci or Rust backend is used. Tolerances allow binary64 rounding and digamma
approximation; integer-valued small coefficients are also tested exactly.
"""
import csv
import math
from pathlib import Path
from case_catalog import read_cases

import numpy as np
import scipy
from scipy import special


def reference(name, args):
    if name == "harmonic":
        n, = args
        return [0.0 if n == 0 else special.digamma(n + 1.0) - special.digamma(1.0)]
    if name == "gen_harmonic":
        n, m = args
        return [np.sum(np.arange(1, n + 1, dtype=np.longdouble) ** -np.longdouble(m), dtype=np.longdouble)]
    if name == "ln_factorial":
        return [special.gammaln(args[0] + 1.0)]
    if name == "ln_binomial":
        return [math.log(math.comb(*args))]
    if name in {"ln_multinomial", "multinomial"}:
        n, counts = args
        value = 1
        for count in counts:
            value *= math.comb(n, count)
            n -= count
        assert n == 0
        return [math.log(value) if name == "ln_multinomial" else float(value)]
    if name == "powu" and args[:2] in ([0, 1], [-1, 0], [1, 0]):
        value = complex(*args[:2]) ** (args[2] % 4)
        return [value.real, value.imag]
    z = np.clongdouble(complex(*args[:2]))
    value = z ** (np.longdouble(1) / 3) if name == "cbrt" else z ** args[2]
    return [float(value.real), float(value.imag)]


def main():
    if (np.__version__, scipy.__version__) != ("2.5.4", "1.18.1"):
        raise RuntimeError("Use NumPy 2.5.4 and SciPy 1.18.1")
    directory = Path(__file__).resolve().parent
    cases = read_cases(directory / "dependency_extras_cases.jsonl")
    with (directory.parent / "fixtures" / "dependency_extras.csv").open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["function", "arguments", "expected"])
        for case in cases:
            args = case["args"]
            writer.writerow([case["name"], ";".join("/".join(map(str, a)) if isinstance(a, list) else str(a) for a in args), ";".join(str(float(v)) for v in reference(case["name"], args))])


if __name__ == "__main__":
    main()
