"""Generate independent quantile references for Sci's sampling engines.

Requirements: scipy==1.18.1, numpy==2.5.4.
Run from repository root with:
    env -u LD_LIBRARY_PATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        .venv/bin/python tests/scipy/generate_sampling_engine_references.py

Inputs are kept separate from the SciPy-derived expected quantiles. These
cases test deterministic inverse CDF methods; seeded Rust RNG streams are not
expected to match NumPy's random streams.
"""

import csv
from pathlib import Path

import numpy as np
import scipy
from scipy import stats


CASES = [
    # id, distribution, lower, upper, probabilities
    ("uniform", stats.uniform(loc=-2.0, scale=4.0), -2.0, 2.0,
     (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)),
    ("exponential", stats.expon(scale=1.0), 0.0, 5.0,
     (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)),
    ("normal", stats.norm(), -3.0, 3.0,
     (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)),
    ("beta", stats.beta(2.0, 5.0), 0.0, 1.0,
     (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)),
]


def main() -> None:
    if scipy.__version__ != "1.18.1" or np.__version__ != "2.5.4":
        raise RuntimeError("Requires pinned SciPy 1.18.1 and NumPy 2.5.4")
    path = Path(__file__).resolve().parent
    with (path / "sampling_quantile_cases.csv").open("w", newline="") as inputs, (
        path / "sampling_quantile_expected.csv"
    ).open("w", newline="") as expected:
        case_writer = csv.writer(inputs, lineterminator="\n")
        expected_writer = csv.writer(expected, lineterminator="\n")
        case_writer.writerow(("id", "lower", "upper", "probabilities"))
        expected_writer.writerow(("id", "quantiles"))
        for name, distribution, lower, upper, probabilities in CASES:
            lower_mass = distribution.cdf(lower)
            upper_mass = distribution.cdf(upper)
            values = [
                float(distribution.ppf(lower_mass + p * (upper_mass - lower_mass)))
                for p in probabilities
            ]
            case_writer.writerow((
                name, format(lower, ".17g"), format(upper, ".17g"),
                ";".join(format(p, ".17g") for p in probabilities),
            ))
            expected_writer.writerow((name, ";".join(format(v, ".17g") for v in values)))


if __name__ == "__main__":
    main()
