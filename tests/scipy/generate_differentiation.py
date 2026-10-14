# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Generate differentiation references using the public SciPy API.

Run: uv run tests/scipy/generate_differentiation.py (or the pinned .venv).
Inputs are separate from expected results in differentiation_inputs.csv.
SciPy's adaptive order-8 estimates approximate the mathematical derivatives,
not Sci's fixed stencils. First-derivative tolerances account for O(h²)
truncation and floating-point subtraction; Hessian tolerances also account for
O(eps/h²) roundoff and the quartic/Rosenbrock stencil's 2*h²/200*h² error.
Zero derivatives use a nonzero absolute tolerance for SciPy convergence.
Rounded-coordinate cases use exact linear/quadratic Hessian identities after
checking SciPy's results; this removes SciPy's spurious roundoff curvature.
"""
import csv
from pathlib import Path

import numpy as np
import scipy
from scipy import differentiate, optimize

if (np.__version__, scipy.__version__) != ("2.5.4", "1.18.1"):
    raise RuntimeError("Use NumPy 2.5.4 and SciPy 1.18.1")

ROOT = Path(__file__).resolve().parents[1] / "fixtures"
SCALAR = {
    "identity": lambda x:x,
    "sin": np.sin,
    "exp": np.exp,
    "cubic": lambda x: x**3 - 2*x,
    "square": lambda x: x*x,
    "constant": lambda x: np.zeros_like(x) + 7,
}
MULTIVARIATE = {
    "linear": lambda x:x[0],
    "local_square": lambda x:(x[0]-1)**2,
    "local_bilinear": lambda x:(x[0]-1)*(x[1]-1),
    "identity_vector": lambda x:np.asarray([x[0],2*x[0]]),
    "quadratic": lambda x: x[0]**2 + 3*x[0]*x[1] + 2*x[1]**2,
    "exp_product": lambda x: np.exp(x[0]*x[1]),
    "trig": lambda x: np.sin(x[0])*np.cos(x[1]),
    "rosen": optimize.rosen,
    "quartic": lambda x: x[0]**4,
    "constant": lambda x: np.zeros_like(x[0]) + 7,
    "vector": lambda x: np.asarray([x[0]*x[1], x[0] + x[1]]),
    "rectangular": lambda x: np.asarray([np.sin(x[0]), x[0]*x[1], np.exp(x[1])]),
}

with (ROOT / "differentiation_inputs.csv").open() as source, (ROOT / "differentiation_expected.csv").open("w", newline="") as target:
    writer = csv.writer(target, lineterminator="\n")
    writer.writerow(["rows", "columns", "values"])
    for case in csv.DictReader(source):
        x = np.asarray([float(v) for v in case["point"].split(";")])
        api = case["api"]
        kwargs = {"tolerances": {"atol": 1e-10, "rtol": 1e-10}}
        if api == "derivative":
            result = differentiate.derivative(SCALAR[case["function"]], x[0], **kwargs)
            values = np.asarray([[result.df]])
        else:
            f = MULTIVARIATE[case["function"]]
            if api == "hessian":
                result = differentiate.hessian(f, x, **kwargs)
                values = result.ddf
            else:
                result = differentiate.jacobian(f, x, **kwargs)
                values = np.atleast_2d(result.df)
        if not np.all(result.success) or not np.all(np.isfinite(values)):
            raise RuntimeError((case, result))
        if api == "hessian" and case["function"] in ("linear", "local_square", "local_bilinear"):
            # Polynomial identities provide exact independent derivatives;
            # SciPy's nested adaptive differences leave ~1.7e-13 spurious
            # curvature even for a linear function. Check before replacing it.
            exact = {"linear":np.asarray([[0.]]), "local_square":np.asarray([[2.]]), "local_bilinear":np.asarray([[0.,1.],[1.,0.]])}[case["function"]]
            if not np.allclose(values, exact, atol=1e-10, rtol=1e-10):
                raise RuntimeError((case, values, exact))
            values = exact
        writer.writerow([*values.shape, ";".join(str(v) for v in values.ravel())])
