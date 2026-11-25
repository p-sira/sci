#!/usr/bin/env python3
"""SciPy 1.18.1 / NumPy 2.5.4 pair for special_performance_comparison.rs.

Run serially from /sci while no other benchmark uses the CPU:

  env -u LD_LIBRARY_PATH \
    PYTHONHOME=/home/psira/.local/share/uv/python/cpython-3.13-linux-x86_64-gnu \
    PYTHONPATH=.venv/lib/python3.13/site-packages \
    .venv/bin/python tests/scipy/benchmark_special_functions.py \
    --repeat 13 --warmup 3 --batch 4 --out target/special_scipy.csv

Both benchmarks use deterministic, mathematically identical float64 grids and
measure setup and computation separately. Computation includes output allocation
and release, never grid preparation. Sci uses public scalar Rust APIs over a Vec;
SciPy vectorizes ufuncs over float64 ndarray inputs. `--scalar-loop` additionally
measures SciPy scalar calls from Python, which includes substantial interpreter
and per-call dispatch overhead and should only be read as a diagnostic.

The cases use finite real inputs away from singularities and branch cuts. They
neither cover complex continuation, principal values, singular endpoints nor
error handling. Input/result shapes and finite values are checked, and output
summaries can be compared across runners (allowing float roundoff).
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Configure native libraries *before* NumPy/SciPy import.
for _name in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "RAYON_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
import scipy
from scipy import special as sp

EXPECTED_VERSIONS = ("1.18.1", "2.5.4")
SIZES = (64, 1024)
FIELDS = (
    "implementation", "case", "family", "phase", "execution_mode", "n",
    "parameters", "input_profile", "repetitions", "batch_calls",
    "median_ns", "p25_ns", "p75_ns", "min_ns", "comparability",
    "difference", "output_layout", "first_value", "last_value", "sum_value",
)


@dataclass(frozen=True)
class Spec:
    name: str
    family: str
    low: float
    high: float
    parameters: str


SPECS = (
    Spec("gamma", "gamma", 0.5, 7.5, "gamma(x)"),
    Spec("gammaln", "gamma", 0.5, 18.0, "gammaln(x)"),
    Spec("digamma", "gamma", 0.5, 18.0, "digamma(x)"),
    Spec("betainc", "beta", 0.05, 0.95, "a=2.5;b=3.5"),
    Spec("gammainc", "gamma", 0.2, 12.0, "a=2.5"),
    Spec("erf", "error", -3.0, 3.0, "erf(x)"),
    Spec("erfinv", "error", -0.9, 0.9, "erfinv(x)"),
    Spec("jv", "bessel", 0.2, 16.0, "v=1.5"),
    Spec("kv", "bessel", 0.2, 16.0, "v=1.5"),
    Spec("hyp1f1", "hypergeometric", -3.0, 3.0, "a=1.5;b=2.5"),
    Spec("hyp2f1", "hypergeometric", -0.6, 0.6, "a=0.8;b=1.3;c=2.8"),
    Spec("exp1", "integrals", 0.2, 12.0, "E1(x);x>0"),
    Spec("ellipk", "elliptic", -0.75, 0.85, "m=x;parameter_not_modulus"),
    Spec("ellipe", "elliptic", -0.75, 0.85, "m=x;parameter_not_modulus"),
    Spec("ellipkinc", "elliptic", 0.01, 1.1, "phi=x;m=0.35;radians"),
    Spec("ellipeinc", "elliptic", 0.01, 1.1, "phi=x;m=0.35;radians"),
    Spec("elliprf", "elliptic", 0.2, 3.0, "y=x+0.4;z=x+1.2"),
    Spec("elliprj", "elliptic", 0.2, 3.0, "y=x+0.4;z=x+1.2;p=x+2.2"),
    Spec("eval_legendre", "orthogonal", -0.8, 0.8, "degree=10"),
    Spec("logsumexp", "reduction", -5.0, 5.0, "axis=none;unweighted"),
)


@dataclass
class Inputs:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    p: np.ndarray


def prepare(spec: Spec, n: int) -> Inputs:
    # Keep the order of operations identical to the Rust preparation formula.
    x = np.fromiter(
        (spec.low + (spec.high - spec.low) * (k / (n - 1)) for k in range(n)),
        dtype=np.float64, count=n,
    )
    y = np.fromiter((v + 0.4 for v in x), dtype=np.float64, count=n)
    z = np.fromiter((v + 1.2 for v in x), dtype=np.float64, count=n)
    p = np.fromiter((v + 2.2 for v in x), dtype=np.float64, count=n)
    return Inputs(x, y, z, p)


OPERATIONS: dict[str, Callable[[Inputs], object]] = {
    "gamma": lambda i: sp.gamma(i.x),
    "gammaln": lambda i: sp.gammaln(i.x),
    "digamma": lambda i: sp.digamma(i.x),
    "betainc": lambda i: sp.betainc(2.5, 3.5, i.x),
    "gammainc": lambda i: sp.gammainc(2.5, i.x),
    "erf": lambda i: sp.erf(i.x),
    "erfinv": lambda i: sp.erfinv(i.x),
    "jv": lambda i: sp.jv(1.5, i.x),
    "kv": lambda i: sp.kv(1.5, i.x),
    "hyp1f1": lambda i: sp.hyp1f1(1.5, 2.5, i.x),
    "hyp2f1": lambda i: sp.hyp2f1(0.8, 1.3, 2.8, i.x),
    "exp1": lambda i: sp.exp1(i.x),
    "ellipk": lambda i: sp.ellipk(i.x),
    "ellipe": lambda i: sp.ellipe(i.x),
    "ellipkinc": lambda i: sp.ellipkinc(i.x, 0.35),
    "ellipeinc": lambda i: sp.ellipeinc(i.x, 0.35),
    "elliprf": lambda i: sp.elliprf(i.x, i.y, i.z),
    "elliprj": lambda i: sp.elliprj(i.x, i.y, i.z, i.p),
    "eval_legendre": lambda i: sp.eval_legendre(10, i.x),
    "logsumexp": lambda i: sp.logsumexp(i.x),
}

# Optional Python scalar-call diagnostic: outputs still use an array to match
# the Rust Vec length, but Python dispatch/iteration costs are inherently higher.
SCALAR: dict[str, Callable[[int, Inputs], float]] = {
    "gamma": lambda k, i: sp.gamma(i.x[k]),
    "gammaln": lambda k, i: sp.gammaln(i.x[k]),
    "digamma": lambda k, i: sp.digamma(i.x[k]),
    "betainc": lambda k, i: sp.betainc(2.5, 3.5, i.x[k]),
    "gammainc": lambda k, i: sp.gammainc(2.5, i.x[k]),
    "erf": lambda k, i: sp.erf(i.x[k]),
    "erfinv": lambda k, i: sp.erfinv(i.x[k]),
    "jv": lambda k, i: sp.jv(1.5, i.x[k]),
    "kv": lambda k, i: sp.kv(1.5, i.x[k]),
    "hyp1f1": lambda k, i: sp.hyp1f1(1.5, 2.5, i.x[k]),
    "hyp2f1": lambda k, i: sp.hyp2f1(0.8, 1.3, 2.8, i.x[k]),
    "exp1": lambda k, i: sp.exp1(i.x[k]),
    "ellipk": lambda k, i: sp.ellipk(i.x[k]),
    "ellipe": lambda k, i: sp.ellipe(i.x[k]),
    "ellipkinc": lambda k, i: sp.ellipkinc(i.x[k], 0.35),
    "ellipeinc": lambda k, i: sp.ellipeinc(i.x[k], 0.35),
    "elliprf": lambda k, i: sp.elliprf(i.x[k], i.y[k], i.z[k]),
    "elliprj": lambda k, i: sp.elliprj(i.x[k], i.y[k], i.z[k], i.p[k]),
    "eval_legendre": lambda k, i: sp.eval_legendre(10, i.x[k]),
}


def scalar_loop(spec: Spec, input_values: Inputs) -> np.ndarray:
    func = SCALAR[spec.name]
    return np.fromiter((func(k, input_values) for k in range(len(input_values.x))),
                       dtype=np.float64, count=len(input_values.x))


def summary(result: object, n: int, is_reduction: bool) -> tuple[float, float, float]:
    values = np.asarray(result)
    expected_shape = () if is_reduction else (n,)
    if values.shape != expected_shape or values.dtype != np.float64:
        raise ValueError(f"Unexpected dtype or shape: {values.dtype}, {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite benchmark output")
    flat = np.ravel(values)
    return (float(flat[0]), float(flat[-1]), sum(float(v) for v in flat))


def measure(spec: Spec, n: int, phase: str, operation: Callable[[Inputs], object] | None,
            repeat: int, warmup: int, batch: int) -> tuple[int, int, int, int]:
    times: list[int] = []
    for trial in range(repeat + warmup):
        if phase == "prepare_only":
            start = time.perf_counter_ns()
            for _ in range(batch):
                values = prepare(spec, n)
                del values
            elapsed = time.perf_counter_ns() - start
        else:
            values = prepare(spec, n)
            assert operation is not None
            start = time.perf_counter_ns()
            for _ in range(batch):
                result = operation(values)
                del result
            elapsed = time.perf_counter_ns() - start
        if trial >= warmup:
            times.append(elapsed // batch)
    times.sort()
    mid = len(times) // 2
    median = (times[mid - 1] + times[mid]) // 2 if len(times) % 2 == 0 else times[mid]
    return median, times[len(times) // 4], times[3 * len(times) // 4], times[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="CSV destination (stdout if omitted)")
    parser.add_argument("--quick", action="store_true", help="Only n=64")
    parser.add_argument("--scalar-loop", action="store_true", help="Add Python scalar-call diagnostic")
    parser.add_argument("--repeat", type=int, default=13)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.repeat <= 100 or not 0 <= args.warmup <= 100 or not 1 <= args.batch <= 64:
        parser.error("repeat must be 1..100; warmup 0..100; batch 1..64")
    if (scipy.__version__, np.__version__) != EXPECTED_VERSIONS:
        parser.error(f"Requires SciPy/NumPy {EXPECTED_VERSIONS}, found {(scipy.__version__, np.__version__)}")
    if set(OPERATIONS) != {s.name for s in SPECS} or set(SCALAR) != set(OPERATIONS) - {"logsumexp"}:
        parser.error("Case definitions and operations are inconsistent")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
    output = args.out.open("w", newline="", encoding="utf-8") if args.out else sys.stdout
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for spec in SPECS:
            for n in SIZES[:1] if args.quick else SIZES:
                inputs = prepare(spec, n)
                is_reduction = spec.name == "logsumexp"
                first, last, checksum = summary(OPERATIONS[spec.name](inputs), n, is_reduction)
                for phase in ("prepare_only", "compute_only"):
                    modes = ("prepare",) if phase == "prepare_only" else (
                        ("reduction",) if is_reduction else
                        ("vectorized_ufunc", "scalar_python_loop") if args.scalar_loop else ("vectorized_ufunc",)
                    )
                    for mode in modes:
                        operation = (lambda i, s=spec: scalar_loop(s, i)) if mode == "scalar_python_loop" else OPERATIONS[spec.name]
                        if mode == "scalar_python_loop":
                            # Every result uses the same mathematical arguments, but scalar
                            # and vectorized kernels may choose distinct internal pathways.
                            scalar_values = operation(inputs)
                            vector_values = OPERATIONS[spec.name](inputs)
                            if not np.allclose(scalar_values, vector_values, rtol=1e-12, atol=1e-13):
                                raise ValueError(f"Scalar/vector SciPy disagreement: {spec.name}")
                        med, p25, p75, minimum = measure(
                            spec, n, phase, operation, args.repeat, args.warmup, args.batch)
                        if phase == "prepare_only":
                            comparable, difference, layout = (
                                "adapted", "rust_vec_vs_numpy_ndarray_preparation", "prepared_four_vecs_vs_four_arrays")
                        elif is_reduction:
                            comparable, difference, layout = (
                                "matched", "different_reduction_dispatch_and_order", "scalar_f64_vs_numpy_scalar")
                        elif mode == "scalar_python_loop":
                            comparable, difference, layout = (
                                "adapted", "python_scalar_dispatch_and_fromiter_vs_rust_loop", "vec_f64_vs_numpy_ndarray")
                        else:
                            comparable, difference, layout = (
                                "adapted", "rust_scalar_loop_vs_scipy_vectorized_ufunc", "vec_f64_vs_numpy_ndarray")
                        writer.writerow({
                            "implementation": f"scipy_{scipy.__version__}",
                            "case": spec.name, "family": spec.family, "phase": phase,
                            "execution_mode": mode, "n": n, "parameters": spec.parameters,
                            "input_profile": "linear_grid_v1", "repetitions": args.repeat,
                            "batch_calls": args.batch, "median_ns": med, "p25_ns": p25,
                            "p75_ns": p75, "min_ns": minimum,
                            "comparability": comparable, "difference": difference,
                            "output_layout": layout, "first_value": f"{first:.17e}",
                            "last_value": f"{last:.17e}", "sum_value": f"{checksum:.17e}",
                        })
    finally:
        if gc_was_enabled:
            gc.enable()
        if args.out:
            output.close()


if __name__ == "__main__":
    main()
