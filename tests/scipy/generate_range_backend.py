# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Public Tukey HSD references covering the full quadrature domain.

Input-only JSON specifies even group sizes and locations. Each group alternates
location-1 and location+1. CSV inputs retain this compact sample definition;
expected row-major statistics and p-values come exclusively from SciPy.
Absolute/relative tolerance is 2e-10, also used by the main statistics fixtures.
"""

import csv
from pathlib import Path
from case_catalog import read_cases

import numpy as np
import scipy
from scipy import stats

if (np.__version__, scipy.__version__) != ("2.5.4", "1.18.1"):
    raise RuntimeError("Use NumPy 2.5.4 and SciPy 1.18.1")

ROOT = Path(__file__).resolve().parent
cases = read_cases(ROOT / "range_backend_cases.jsonl")
with (
    (ROOT.parent / "fixtures/range_inputs.csv").open("w", newline="") as inputs,
    (ROOT.parent / "fixtures/range_expected.csv").open("w", newline="") as expected,
):
    iw, ew = csv.writer(inputs), csv.writer(expected)
    iw.writerow(["id", "sizes", "locations"])
    ew.writerow(["id", "statistics", "pvalues"])
    for i, case in enumerate(cases):
        sizes, locations = case["sizes"], case["locations"]
        assert len(sizes) == len(locations) and all(n % 2 == 0 for n in sizes)
        groups = [np.tile([-1.0, 1.0], n // 2) + loc for n, loc in zip(sizes, locations)]
        result = stats.tukey_hsd(*groups)
        iw.writerow([i, ";".join(map(str, sizes)), ";".join(map(str, locations))])
        ew.writerow([
            i,
            ";".join(repr(float(x)) for x in result.statistic.ravel()),
            ";".join(repr(float(x)) for x in result.pvalue.ravel()),
        ])
print(f"Wrote {len(cases)} studentized-range reference cases")
