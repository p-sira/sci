"""Generate independent 120-digit 1F1 references for exact Divan benchmark inputs.

No Sci or SciPy kernel participates in the expected values. Decimal.from_float
preserves the exact binary64 benchmark arguments before summing the defining
hypergeometric series. This generator uses only Python's standard library.
"""

import csv
from decimal import Decimal, localcontext
from pathlib import Path


MASK = (1 << 64) - 1
A = 6364136223846793005
C = 1442695040888963407


def values(seed: int, lo: float, hi: float) -> list[float]:
    state = (seed * A + C) & MASK
    result = []
    for _ in range(256):
        state = (state * A + C) & MASK
        v = ((state >> 11) / float(1 << 53)) * 2.0 - 1.0
        u = 0.5 * max(1e-6, min(v + 1.0, 2.0 - 1e-6))
        result.append(lo + (hi - lo) * u)
    return result


def hyp1f1(a: float, b: float, x: float) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = 120
        aa, bb, xx = map(Decimal.from_float, (a, b, x))
        term = total = Decimal(1)
        for k in range(1, 512):
            term = term * (aa + k - 1) * xx / ((bb + k - 1) * k)
            total += term
            if k > 25 and abs(term) < Decimal("1e-105") * max(
                abs(total), Decimal("1e-80")
            ):
                return total
    raise ArithmeticError(f"1F1 series did not converge for {(a, b, x)}")


def main() -> None:
    destination = Path(__file__).parents[1] / "fixtures/hyp1f1_bench_exact.csv"
    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["a", "b", "x", "expected"])
        for a, b, x in zip(
            values(52, -3.0, 3.0),
            values(53, 0.5, 5.0),
            values(54, -5.0, 5.0),
        ):
            writer.writerow([repr(a), repr(b), repr(x), format(hyp1f1(a, b, x), ".55g")])


if __name__ == "__main__":
    main()
