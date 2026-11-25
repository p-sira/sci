"""Generate independent unsrambled Halton fixtures with SciPy 1.18.1.

    env -u LD_LIBRARY_PATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        .venv/bin/python -B tests/scipy/generate_halton.py

Inputs and expected outputs are separate CSV files. The expected coordinates
are in row-major order and are never computed by Sci's Rust implementation.
"""

import csv
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import qmc


if (np.__version__, scipy.__version__) != ("2.5.4", "1.18.1"):
    raise RuntimeError("Expected NumPy 2.5.4 and SciPy 1.18.1")


# (id, dimensions, number of skipped points, number of sampled points)
CASES = [
    ("d1_origin", 1, 0, 1),
    ("d1_n8", 1, 0, 8),
    ("d2_n16", 2, 0, 16),
    ("d3_n7", 3, 0, 7),
    ("d5_n13", 5, 0, 13),
    ("d16_n8", 16, 0, 8),
    ("d64_n4", 64, 0, 4),
    ("d65_n4", 65, 0, 4),
    ("d128_n3", 128, 0, 3),
    ("d256_n2", 256, 0, 2),
    ("d2_skip1", 2, 1, 8),
    ("d5_skip3", 5, 3, 13),
    ("d32_skip7", 32, 7, 6),
    ("d64_skip127", 64, 127, 5),
    ("d256_skip255", 256, 255, 3),
    ("d4_skip1024", 4, 1024, 4),
]


def generate():
    if len({case[0] for case in CASES}) != len(CASES):
        raise ValueError("duplicate Halton case ids")

    root = Path(__file__).resolve().parent
    with (root / "halton_cases.csv").open("w", newline="") as inputs, (
        root / "halton_expected.csv"
    ).open("w", newline="") as expected:
        iw = csv.writer(inputs, lineterminator="\n")
        ew = csv.writer(expected, lineterminator="\n")
        iw.writerow(["id", "d", "skip", "n"])
        ew.writerow(["id", "rows", "columns", "points_row_major"])

        for case_id, d, skip, n in CASES:
            if not 1 <= d <= 512 or skip < 0 or n < 1 or d * n > 1_048_576:
                raise ValueError(f"{case_id}: invalid sampling parameters")
            sampler = qmc.Halton(d=d, scramble=False)
            if skip:
                sampler.fast_forward(skip)
            points = sampler.random(n)
            if points.shape != (n, d) or not np.all((0 <= points) & (points < 1)):
                raise ValueError(f"{case_id}: invalid SciPy output")
            iw.writerow([case_id, d, skip, n])
            ew.writerow([case_id, n, d, ";".join(format(float(x), ".17g") for x in points.flat)])


if __name__ == "__main__":
    generate()
