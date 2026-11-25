"""Generate exact, unsrambled Sobol reference points from SciPy 1.18.1.

Run from the repository root with:

    .venv/bin/python tests/scipy/generate_sobol.py

When launched from the Chat On Steroids AppImage, remove its LD_LIBRARY_PATH:

    env -u LD_LIBRARY_PATH .venv/bin/python tests/scipy/generate_sobol.py

sobol_cases.csv contains only inputs. sobol_expected.csv contains independently
computed SciPy reference matrices in row-major order, with dimensions recorded
separately to detect shape regressions. Every case starts from a fresh engine.
"""

import csv
import warnings
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import qmc


if (np.__version__, scipy.__version__) != ("2.5.4", "1.18.1"):
    raise RuntimeError("Expected NumPy 2.5.4 and SciPy 1.18.1")


# (id, dimension, bits, operation, fast-forward count, n or m)
# For "random", the final argument is n; for "base2", it is the exponent m.
# Includes Sobol's zero point, dimensional direction numbers, Gray-code carries,
# non-power-of-two sample counts, fast-forward, and period boundaries.
CASES = [
    ("d1_origin", 1, 8, "random", 0, 1),
    ("d1_m0", 1, 8, "base2", 0, 0),
    ("d1_bits1", 1, 1, "base2", 0, 1),
    ("d1_bits1_last", 1, 1, "random", 1, 1),
    ("d2_bits2", 2, 2, "base2", 0, 2),
    ("d2_m4", 2, 8, "base2", 0, 4),
    ("d3_m0", 3, 8, "base2", 0, 0),
    ("d3_m3", 3, 8, "base2", 0, 3),
    ("d5_m3", 5, 8, "base2", 0, 3),
    ("d8_m4", 8, 10, "base2", 0, 4),
    ("d16_m3", 16, 10, "base2", 0, 3),
    ("d32_m2", 32, 10, "base2", 0, 2),
    ("d64_m2", 64, 10, "base2", 0, 2),
    ("d1_n3", 1, 8, "random", 0, 3),
    ("d3_n7", 3, 8, "random", 0, 7),
    ("d8_n9", 8, 10, "random", 0, 9),
    ("d2_skip1_n7", 2, 10, "random", 1, 7),
    ("d2_skip3_n5", 2, 10, "random", 3, 5),
    ("d4_skip7_n9", 4, 10, "random", 7, 9),
    ("d5_skip31_n8", 5, 10, "random", 31, 8),
    ("d8_skip127_n2", 8, 10, "random", 127, 2),
    ("d3_skip1023_n2", 3, 12, "random", 1023, 2),
    ("d2_bits8_last", 2, 8, "random", 255, 1),
    ("d1_skip1_m0", 1, 8, "base2", 1, 0),
    ("d2_skip4_m2", 2, 8, "base2", 4, 2),
    ("d3_skip8_m3", 3, 8, "base2", 8, 3),
    ("d16_skip16_m4", 16, 10, "base2", 16, 4),
    ("d4_bits52_skip", 4, 52, "random", 65535, 4),
    ("d2_bits64_skip", 2, 64, "random", 1023, 4),
]


def generate():
    if len({case[0] for case in CASES}) != len(CASES):
        raise ValueError("duplicate Sobol case ids")

    root = Path(__file__).resolve().parent
    with (root / "sobol_cases.csv").open("w", newline="") as inputs, (
        root / "sobol_expected.csv"
    ).open("w", newline="") as expected:
        input_writer = csv.writer(inputs, lineterminator="\n")
        expected_writer = csv.writer(expected, lineterminator="\n")
        input_writer.writerow(["id", "d", "bits", "method", "skip", "argument"])
        expected_writer.writerow(["id", "rows", "columns", "points_row_major"])

        for case_id, d, bits, method, skip, argument in CASES:
            if not 1 <= d <= 64 or not 1 <= bits <= 64 or skip < 0:
                raise ValueError(f"{case_id}: invalid dimensions, bits, or skip")
            n = argument if method == "random" else 1 << argument
            if method not in ("random", "base2") or n < 1 or skip + n > 1 << bits:
                raise ValueError(f"{case_id}: invalid generation request")

            engine = qmc.Sobol(d=d, scramble=False, bits=bits)
            with warnings.catch_warnings():
                # Arbitrary sample counts intentionally exercise random(n).
                warnings.filterwarnings("ignore", message="The balance properties of Sobol")
                if skip:
                    if bits > 32:
                        # SciPy 1.18.1 fast_forward incorrectly selects a uint32
                        # kernel for uint64 direction numbers (dtype mismatch).
                        # Consuming the prefix gives the same reference sequence.
                        engine.random(skip)
                    else:
                        engine.fast_forward(skip)
                points = engine.random(n) if method == "random" else engine.random_base2(argument)

            if points.shape != (n, d) or not np.all((points >= 0.0) & (points < 1.0)):
                raise ValueError(f"{case_id}: unexpected SciPy shape or range")

            input_writer.writerow([case_id, d, bits, method, skip, argument])
            expected_writer.writerow([
                case_id,
                n,
                d,
                ";".join(format(float(value), ".17g") for value in points.flat),
            ])


if __name__ == "__main__":
    generate()
