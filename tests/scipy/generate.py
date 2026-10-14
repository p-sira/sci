# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy==2.3.3",
#   "scipy==1.16.2",
# ]
# ///
"""Generate reference values with SciPy for Sci's integration tests."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import fft, linalg, optimize, special, stats
from scipy.optimize._numdiff import approx_derivative


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def write_csv(name: str, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    with (FIXTURES / name).open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def finite_difference(function, x: float, step: float) -> float:
    result = approx_derivative(
        lambda value: np.asarray([function(value[0])]),
        np.asarray([x]),
        method="3-point",
        abs_step=step,
    )
    return float(result[0])


def generate_differentiation() -> None:
    cases = [
        ("sin", np.sin, 0.3, 1e-5),
        ("exp", np.exp, -0.7, 1e-5),
        ("cubic", lambda x: x**3 - 2 * x, 1.25, 1e-5),
    ]
    write_csv(
        "differentiation.csv",
        ("function", "x", "step", "expected"),
        ((name, x, step, finite_difference(function, x, step)) for name, function, x, step in cases),
    )


def generate_special() -> None:
    rows = [
        ("gamma", 0.5, 0.0, special.gamma(0.5)),
        ("gamma", 5.25, 0.0, special.gamma(5.25)),
        ("digamma", 1.0, 0.0, special.digamma(1.0)),
        ("digamma", 8.5, 0.0, special.digamma(8.5)),
        ("erf", -0.75, 0.0, special.erf(-0.75)),
        ("erf", 1.25, 0.0, special.erf(1.25)),
        ("beta", 2.5, 3.5, special.beta(2.5, 3.5)),
        ("ellipk", 0.4, 0.0, special.ellipk(0.4)),
        ("ellipe", 0.4, 0.0, special.ellipe(0.4)),
        ("gammaln", 5.25, 0.0, special.gammaln(5.25)),
        ("trigamma", 2.5, 0.0, special.polygamma(1, 2.5)),
        ("erfc", 0.75, 0.0, special.erfc(0.75)),
        ("erfinv", 0.8, 0.0, special.erfinv(0.8)),
        ("betaln", 2.5, 3.5, special.betaln(2.5, 3.5)),
        ("lambertw", 2.0, 0.0, special.lambertw(2.0).real),
        ("lambertwm1", -0.1, 0.0, special.lambertw(-0.1, -1).real),
        ("gammainc", 2.5, 3.0, special.gammainc(2.5, 3.0)),
    ]
    write_csv("special.csv", ("function", "x", "y", "expected"), rows)
    write_csv(
        "incomplete_beta.csv",
        ("a", "b", "x", "expected", "expected_inverse"),
        (
            (a, b, x, special.betainc(a, b, x),
             special.betaincinv(a, b, special.betainc(a, b, x)))
            for a, b, x in ((2.5, 3.5, 0.25), (0.5, 0.5, 0.75), (5.0, 2.0, 0.6))
        ),
    )


def generate_fft() -> None:
    values = np.asarray([1 + 0.5j, -2 + 1j, 0.25 - 0.75j, 3 + 0j], dtype=np.complex128)
    transformed = fft.fft(values)
    write_csv(
        "fft.csv",
        ("index", "input_real", "input_imag", "expected_real", "expected_imag"),
        (
            (index, value.real, value.imag, expected.real, expected.imag)
            for index, (value, expected) in enumerate(zip(values, transformed, strict=True))
        ),
    )


def generate_statistics() -> None:
    normal = stats.norm(loc=1.5, scale=2.25)
    rows = []
    for x in (-2.0, 1.5, 4.25):
        rows.append(("normal_pdf", 1.5, 2.25, x, normal.pdf(x)))
        rows.append(("normal_cdf", 1.5, 2.25, x, normal.cdf(x)))
    for probability in (0.01, 0.5, 0.99):
        rows.append(("normal_ppf", 1.5, 2.25, probability, normal.ppf(probability)))
    write_csv(
        "statistics.csv",
        ("function", "parameter_1", "parameter_2", "x", "expected"),
        rows,
    )


def generate_interoperability() -> None:
    point = np.asarray([2.0, 3.0])
    step = 1e-5
    jacobian = approx_derivative(
        lambda x: np.asarray([x[0] * x[1], x[0] + x[1]]),
        point, method="3-point", abs_step=step,
    )
    direction = np.asarray([1.5, -0.5])
    product = linalg.blas.dgemv(1.0, jacobian, direction)
    write_csv(
        "jacobian.csv",
        ("row", "point_0", "point_1", "step", "expected_0", "expected_1", "direction_0", "direction_1", "expected_product"),
        ((row, *point, step, *jacobian[row], *direction, product[row]) for row in range(2)),
    )
    matrix = np.asarray([[1.0, 2.0], [-3.0, 4.0]])
    gram = linalg.blas.dgemm(1.0, matrix, matrix, trans_b=True)
    write_csv(
        "linalg.csv", ("row", "input_0", "input_1", "expected_0", "expected_1"),
        ((row, *matrix[row], *gram[row]) for row in range(2)),
    )
    amplitudes = np.abs(fft.rfft(np.asarray([1.0, 2.0, 3.0, 4.0])))
    summary = stats.describe(amplitudes)
    write_csv(
        "spectrum_statistics.csv", ("index", "amplitude", "mean", "variance"),
        ((index, amplitude, summary.mean, summary.variance) for index, amplitude in enumerate(amplitudes)),
    )


def generate_optimization() -> None:
    def sphere(value: np.ndarray) -> float:
        return float(np.dot(value, value))

    starts = ((2.0, -3.0), (-4.0, 1.5))
    rows = []
    for x0, x1 in starts:
        result = optimize.minimize(sphere, np.asarray([x0, x1]), method="BFGS", tol=1e-12)
        if not result.success and result.fun > 1e-12:
            raise RuntimeError(result.message)
        rows.append(("sphere", x0, x1, result.x[0], result.x[1], result.fun))
    write_csv(
        "optimization.csv",
        ("problem", "start_0", "start_1", "expected_0", "expected_1", "expected_cost"),
        rows,
    )


if __name__ == "__main__":
    generate_differentiation()
    generate_special()
    generate_fft()
    generate_statistics()
    generate_optimization()
    generate_interoperability()
