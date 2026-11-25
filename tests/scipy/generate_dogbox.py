# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Generate independent dogbox references using scipy.optimize.least_squares.

Run: uv run tests/scipy/generate_dogbox.py

The JSON Lines catalog is the source of input cases. Two generated CSVs keep
inputs and expected SciPy outputs separate for Rust tests. Arrays in CSV cells
use semicolons; matrices are row-major. The residual models are:

* linear: A @ x - b; a contains the m-by-n matrix, b contains m observations.
* rosenbrock: [10*(x[1]-x[0]**2), 1-x[0]] (two parameters/residuals).
* exponential: x[0]*exp(x[1]*t)-b; a contains observation times t.
* curved: [x[0]**2+x[1]-b[0]] (two parameters, one residual).
* square: [(a[0]*x[0])**2-b[0]] (one parameter and residual).

All runs use method='dogbox', two-point finite-difference Jacobians, tolerances
1e-10 and max_nfev=600. nfev, njev and status are informative: other solvers can
converge with different iteration counts and termination criteria. Rank-deficient
and underdetermined solutions are nonunique; compare their residuals and costs.
"""

import csv
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import least_squares

from case_catalog import read_cases


if (np.__version__, scipy.__version__) != ("2.5.4", "1.18.1"):
    raise RuntimeError("Expected NumPy 2.5.4 and SciPy 1.18.1")

ROOT = Path(__file__).resolve().parents[1] / "fixtures"


def encode(values):
    """Flatten a vector/matrix into a round-trippable CSV cell."""
    return ";".join(format(float(v), ".17g") for v in np.asarray(values).ravel())


def residual(model, a, b):
    if model == "linear":
        return lambda x: a @ x - b
    if model == "rosenbrock":
        return lambda x: np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])
    if model == "exponential":
        return lambda x: x[0] * np.exp(x[1] * a) - b
    if model == "curved":
        return lambda x: np.array([x[0] ** 2 + x[1] - b[0]])
    if model == "square":
        return lambda x: np.array([(a[0] * x[0]) ** 2 - b[0]])
    raise ValueError(f"unknown residual model: {model}")


def dimensions(model, a, b, n):
    if model == "linear":
        if a.ndim != 2 or a.shape[1] != n or b.shape != (a.shape[0],):
            raise ValueError("linear case requires A: (m, n), b: (m,)")
        return a.shape[0]
    if model == "rosenbrock":
        if n != 2 or a.size or b.size:
            raise ValueError("rosenbrock requires two parameters and no data")
        return 2
    if model == "exponential":
        if n != 2 or a.ndim != 1 or b.ndim != 1 or not a.size or a.shape != b.shape:
            raise ValueError("exponential requires two parameters and matching t/y vectors")
        return len(b)
    if model == "curved":
        if n != 2 or a.size or b.shape != (1,):
            raise ValueError("curved requires two parameters and one target")
        return 1
    if model == "square":
        if n != 1 or a.shape != (1,) or b.shape != (1,):
            raise ValueError("square requires one parameter, scale, and target")
        return 1
    raise ValueError(f"unknown residual model: {model}")


def generate():
    cases = read_cases(Path(__file__).with_name("dogbox_cases.jsonl"))
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("duplicate dogbox case ids")

    with (ROOT / "dogbox_inputs.csv").open("w", newline="") as inputs, (
        ROOT / "dogbox_expected.csv"
    ).open("w", newline="") as expected:
        iw = csv.writer(inputs, lineterminator="\n")
        ew = csv.writer(expected, lineterminator="\n")
        iw.writerow(["id", "model", "n", "m", "x0", "lower", "upper", "a", "b"])
        ew.writerow(["id", "x", "fun", "cost", "optimality", "active_mask", "status", "success", "nfev", "njev"])

        for case in cases:
            model = case["model"]
            x0 = np.asarray(case["x0"], dtype=float)
            lower = np.asarray(case["lower"], dtype=float)
            upper = np.asarray(case["upper"], dtype=float)
            a = np.asarray(case.get("a", []), dtype=float)
            b = np.asarray(case.get("b", []), dtype=float)
            n = len(x0)
            m = dimensions(model, a, b, n)
            if x0.ndim != 1 or not n or lower.shape != x0.shape or upper.shape != x0.shape:
                raise ValueError(f"{case['id']}: invalid parameter/bounds dimensions")
            if not np.all(np.isfinite(x0)) or not np.all(lower < upper):
                raise ValueError(f"{case['id']}: invalid parameters or bounds")
            if not np.all((lower <= x0) & (x0 <= upper)):
                raise ValueError(f"{case['id']}: initial point violates bounds")
            if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
                raise ValueError(f"{case['id']}: nonfinite residual data")

            fn = residual(model, a, b)
            result = least_squares(
                fn,
                x0,
                bounds=(lower, upper),
                method="dogbox",
                jac="2-point",
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
                max_nfev=600,
            )
            if not result.success or not np.all(np.isfinite(result.x)):
                raise RuntimeError(f"{case['id']}: SciPy dogbox failed: {result.message}")
            if not np.isclose(result.cost, np.dot(result.fun, result.fun) / 2):
                raise RuntimeError(f"{case['id']}: inconsistent least squares cost")

            iw.writerow([case["id"], model, n, m, encode(x0), encode(lower), encode(upper), encode(a), encode(b)])
            ew.writerow([
                case["id"], encode(result.x), encode(result.fun),
                format(result.cost, ".17g"), format(result.optimality, ".17g"),
                encode(result.active_mask), result.status, result.success,
                result.nfev, result.njev,
            ])


if __name__ == "__main__":
    generate()
