# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Independent Legendre-basis eigenvalues: uv run tests/scipy/generate_spheroidal_characteristics.py."""

import csv
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import eigvalsh_tridiagonal


def characteristic(m, n, c, sign, size):
    # In the orthonormal associated Legendre basis, multiplication by x
    # connects adjacent degrees with coefficient alpha_l. Applying x twice
    # gives a symmetric tridiagonal operator within each parity subspace.
    degrees = m + (n - m) % 2 + 2 * np.arange(size, dtype=np.float64)

    def alpha_squared(l):
        return ((l + 1) ** 2 - m**2) / ((2 * l + 1) * (2 * l + 3))

    diagonal = degrees * (degrees + 1) + sign * c**2 * (
        alpha_squared(degrees) + alpha_squared(degrees - 1)
    )
    off_diagonal = sign * c**2 * np.sqrt(
        alpha_squared(degrees[:-1]) * alpha_squared(degrees[:-1] + 1)
    )
    return eigvalsh_tridiagonal(diagonal, off_diagonal, lapack_driver="sterf")[(n - m) // 2]


def main():
    if (np.__version__, scipy.__version__) != ("2.5.4", "1.18.1"):
        raise RuntimeError("Use NumPy 2.5.4 and SciPy 1.18.1")
    directory = Path(__file__).resolve().parent
    with (directory / "spheroidal_characteristic_inputs.csv").open(newline="") as source:
        cases = list(csv.DictReader(source))
    output = directory.parent / "fixtures" / "spheroidal_characteristics.csv"
    with output.open("w", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(["m", "n", "c", "prolate", "oblate"])
        for case in cases:
            m, n, c = int(case["m"]), int(case["n"]), float(case["c"])
            values = [characteristic(m, n, c, sign, 192) for sign in [1, -1]]
            larger = [characteristic(m, n, c, sign, 224) for sign in [1, -1]]
            np.testing.assert_allclose(values, larger, atol=2e-11, rtol=2e-13)
            writer.writerow([m, n, c, *values])


if __name__ == "__main__":
    main()
