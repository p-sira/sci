"""Export the Joe–Kuo Sobol direction-number data bundled with SciPy.

Requires scipy==1.18.1 and numpy==2.5.4. Run from the repository root:

    python tests/scipy/export_sobol_directions.py

The output contains 21,201 records of 19 little-endian u32 values: one
primitive polynomial followed by 18 initial direction-number coefficients.
Source: scipy/stats/_sobol_direction_numbers.npz (SciPy BSD-3-Clause).
The original Joe–Kuo 2008 direction-number data carry a separate BSD notice.
"""

from importlib.resources import files
from pathlib import Path

import numpy as np
import scipy


def main() -> None:
    if scipy.__version__ != "1.18.1" or np.__version__ != "2.5.4":
        raise RuntimeError("Requires scipy==1.18.1 and numpy==2.5.4")

    with np.load(files("scipy.stats") / "_sobol_direction_numbers.npz") as source:
        polynomial = source["poly"]
        initial = source["vinit"]

    if polynomial.shape != (21_201,) or initial.shape != (21_201, 18):
        raise ValueError("Unexpected Joe–Kuo direction-number table dimensions")
    if (polynomial < 0).any() or (initial < 0).any():
        raise ValueError("Direction-number data must be unsigned")

    table = np.column_stack((polynomial, initial)).astype("<u4")
    output = Path(__file__).resolve().parents[2] / "crates/sci-stats/src/qmc/sobol_directions.bin"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(table.tobytes(order="C"))
    print(f"Exported {table.shape[0]} dimensions to {output}")


if __name__ == "__main__":
    main()
