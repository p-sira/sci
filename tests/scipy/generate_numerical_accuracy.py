# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1", "mpmath==1.3.0"]
# ///
"""Regressions from the non-sparse sci-testing accuracy audit.

Run: uv run tests/scipy/generate_numerical_accuracy.py
Inputs are stored separately in numerical_accuracy_cases.jsonl. SciPy supplies
Newton-Cotes rules. Other references use independent mathematical definitions
at 100 decimal digits: exact binary64 Rosenbrock polynomials, Student-t density
integration/inversion with beta tails, linear systems, and the exact n=3
Shapiro distribution.
Student quantiles near the median require this precision because even SciPy
rounding can dominate their very small magnitude. Rust checks each component
relatively, so a zero cannot replace a small nonzero objective or quantile.
"""

import csv
from pathlib import Path
from case_catalog import read_cases

import mpmath as mp
import numpy as np
import scipy
from scipy import integrate, stats

assert (np.__version__, scipy.__version__, mp.__version__) == (
    "2.5.4", "1.18.1", "1.3.0"
)
mp.mp.dps = 100
ROOT = Path(__file__).resolve().parents[1]


def student_cdf(df, t):
    # Integrate the defining density, avoiding a beta-inverse implementation.
    if abs(t) > 1:
        # The equivalent high-precision tail integral stays well conditioned
        # for the enormous quantiles of very small degrees of freedom.
        tail = mp.betainc(df/2, mp.mpf("0.5"), 0, df/(df+t*t), regularized=True) / 2
        return tail if t < 0 else 1 - tail
    density = mp.gamma((df + 1) / 2) / (mp.sqrt(df * mp.pi) * mp.gamma(df / 2))
    return mp.mpf("0.5") + density * mp.quad(
        lambda z: (1 + z * z / df) ** (-(df + 1) / 2), [0, t]
    )


def reference(case):
    api = case["api"]
    x = list(map(mp.mpf, case.get("x", [])))
    p = list(map(mp.mpf, case.get("p", [])))
    if api in ("rosen", "rosen_der"):
        gradient = [mp.mpf(0)] * len(x)
        objective = mp.mpf(0)
        for i, (a, b) in enumerate(zip(x, x[1:])):
            residual = b - a * a
            objective += 100 * residual**2 + (1 - a)**2
            gradient[i] += -400 * a * residual + 2 * (a - 1)
            gradient[i + 1] += 200 * residual
        return [objective] if api == "rosen" else gradient
    if api == "student_special_cdf":
        df, t = p
        if df == 1:
            tail = mp.atan(1 / abs(t)) / mp.pi
        else:
            norm = mp.sqrt(t*t + 2)
            tail = 1 / (norm * (norm + abs(t)))
        return [tail if t < 0 else 1 - tail]
    if api == "student_special_ppf":
        df, probability = p
        if df == 1:
            return [-mp.cot(mp.pi * probability)]
        return [(probability - mp.mpf("0.5")) * mp.sqrt(2 / probability / (1-probability))]
    if api == "student_cdf":
        df, t = p
        return [student_cdf(df, t), student_cdf(df, -t)]
    if api == "student_ppf":
        df, probability = p
        if probability == mp.mpf("0.5"):
            return [mp.mpf(0)]
        density = mp.gamma((df + 1) / 2) / (mp.sqrt(df * mp.pi) * mp.gamma(df / 2))
        # SciPy supplies only an initial guess; the reference solves the
        # independent high-precision density integral to 90 decimal places.
        start = mp.mpf(stats.t.ppf(float(probability), float(df)))
        return [mp.findroot(
            lambda t: student_cdf(df, t) - probability, start,
            df=lambda t: density * (1 + t*t/df) ** (-(df + 1)/2),
            solver="newton", tol=mp.mpf("1e-90"), maxsteps=40,
        )]
    if api == "newton_cotes":
        weights, error = integrate.newton_cotes(np.asarray(case["x"], dtype=float))
        return [*weights, error]
    if api == "shapiro":
        a, b, c = sorted(x)
        # For n=3, the squared order-statistic correlation has this exact law.
        mean = (a + b + c) / 3
        w = (c - a)**2 / (2 * sum((v - mean)**2 for v in x))
        return [w, 1 - 6 / mp.pi * mp.acos(mp.sqrt(min(w, 1)))]
    if api == "solve":
        return list(mp.lu_solve(mp.matrix(case["a"]), mp.matrix(x)))
    raise ValueError(api)


def encode(values):
    return ";".join(repr(float(v)) for v in values)


with (
    (ROOT / "fixtures/numerical_accuracy_inputs.csv").open("w", newline="") as fi,
    (ROOT / "fixtures/numerical_accuracy_expected.csv").open("w", newline="") as fe,
):
    iw, ew = csv.writer(fi, lineterminator="\n"), csv.writer(fe, lineterminator="\n")
    iw.writerow(["id", "api", "x", "p", "a"])
    ew.writerow(["id", "values"])
    cases = read_cases(Path(__file__).with_name("numerical_accuracy_cases.jsonl"))
    for case in cases:
        iw.writerow([
            case["id"], case["api"], encode(case.get("x", [])),
            encode(case.get("p", [])), ":".join(map(encode, case.get("a", []))),
        ])
        ew.writerow([case["id"], encode(reference(case))])
print(f"Wrote {len(cases)} numerical accuracy references")
