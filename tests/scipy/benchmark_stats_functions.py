#!/usr/bin/env python3
"""SciPy 1.18.1 / NumPy 2.5.4 pair for benches/stats_scipy_comparison.rs.

From /sci, run serially when no other benchmark is using the CPU:

  env -u LD_LIBRARY_PATH \
    PYTHONHOME=/home/psira/.local/share/uv/python/cpython-3.13-linux-x86_64-gnu \
    PYTHONPATH=.venv/lib/python3.13/site-packages \
    .venv/bin/python tests/scipy/benchmark_stats_functions.py \
    --repeat 13 --warmup 3 --batch 4 --out target/stats_scipy.csv

All numerical inputs use identical Python and Rust arithmetic formulas and
float64. `prepare_only` measures input construction; `compute_only` prepares
inputs before measuring. Outputs are allocated and released inside the latter
phase. Timing blocks amortize clock overhead; CSV rows report per-operation
median, interquartile bounds and minimum. The Rust API uses slices and native
vectors while SciPy uses NumPy arrays; these differences are recorded.
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

# Configure native libraries before importing NumPy or SciPy.
for _env_name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_env_name] = "1"

import numpy as np
import scipy
from scipy import stats


EXPECTED_VERSIONS = ("1.18.1", "2.5.4")
SIZES = (128, 2048)
FIELDS = (
    "implementation", "case", "family", "phase", "n", "parameters",
    "input_profile", "repetitions", "batch_calls", "median_ns", "p25_ns",
    "p75_ns", "min_ns", "comparability", "difference", "output_layout",
)


@dataclass(frozen=True)
class Spec:
    key: str
    family: str
    profile: str
    parameters: str
    comparability: str
    difference: str
    output: str


SPECS = (
    Spec("mean", "descriptive", "sample_formula_v1", "arithmetic_mean", "matched", "slice_vs_ndarray;scipy_tmean_dispatch", "numpy_scalar"),
    Spec("variance_ddof1", "descriptive", "sample_formula_v1", "ddof=1", "matched", "slice_vs_ndarray;scipy_tvar_dispatch", "numpy_scalar"),
    Spec("quantile_linear_p30", "descriptive", "sample_formula_v1", "probability=0.3;linear_interpolation", "matched", "sort_vs_scipy_percentile_backend", "numpy_scalar"),
    Spec("rankdata_average_ties", "ranking", "integer_ties_31_v1", "method=average;31_distinct_values", "matched", "sort_backend_and_output_container", "numpy_ndarray"),
    Spec("zscore_ddof1", "descriptive", "sample_formula_v1", "ddof=1;axis=none", "matched", "native_vector_vs_numpy_dispatch", "numpy_ndarray"),
    Spec("pearsonr_two_sided", "association", "paired_formula_v1", "alternative=two-sided", "matched", "beta_tail_backend_and_result_container", "scipy_result"),
    Spec("spearmanr_two_sided", "association", "paired_formula_v1", "alternative=two-sided", "matched", "rank_and_probability_backend", "scipy_result"),
    Spec("ttest_1samp_two_sided", "hypothesis", "sample_formula_v1", "popmean=0.1;alternative=two-sided", "matched", "tail_probability_backend", "scipy_result"),
    Spec("ttest_ind_welch_two_sided", "hypothesis", "paired_formula_v1", "equal_var=false;alternative=two-sided", "matched", "tail_probability_backend", "scipy_result"),
    Spec("mannwhitneyu_asymptotic", "hypothesis", "paired_formula_v1", "method=asymptotic;use_continuity=true;alternative=two-sided", "matched", "rank_and_normal_tail_backend", "scipy_result"),
    Spec("jarque_bera", "hypothesis", "sample_formula_v1", "asymptotic_chi2_2", "matched", "moment_and_tail_backend", "scipy_result"),
    Spec("boxcox_lambda05", "transform", "positive_formula_v1", "lambda=0.5;fixed_no_fit", "matched", "native_vec_vs_ndarray", "numpy_ndarray"),
    Spec("yeojohnson_lambda05", "transform", "sample_formula_v1", "lambda=0.5;fixed_no_fit", "matched", "native_vec_vs_ndarray", "numpy_ndarray"),
    Spec("wasserstein_distance", "distance", "paired_formula_v1", "uniform_weights;one_dimensional", "matched", "sort_backend_and_memory_layout", "numpy_scalar"),
    Spec("normal_cdf_batch", "distribution", "normal_grid_-4_to_4_v1", "normal_loc=0.5;scale=1.5;cdf", "adapted", "rust_scalar_loop_vs_scipy_vectorized_ndarray", "numpy_ndarray"),
    Spec("normal_ppf_batch", "distribution", "probability_grid_0.01_to_0.99_v1", "normal_loc=0.5;scale=1.5;ppf", "adapted", "rust_scalar_loop_vs_scipy_vectorized_ndarray", "numpy_ndarray"),
)


@dataclass
class Inputs:
    x: np.ndarray
    y: np.ndarray


def sample_x(i: int, n: int) -> float:
    return ((i * 37 % 257) - 128) / 19.0 + (i / n) * 0.03


def sample_y(i: int, n: int) -> float:
    return (((i * 59 + 13) % 263) - 131) / 23.0 - (i / n) * 0.02


def prepare(profile: str, n: int) -> Inputs:
    if profile in ("sample_formula_v1", "paired_formula_v1"):
        values = (sample_x(i, n) for i in range(n))
    elif profile == "integer_ties_31_v1":
        values = (((i * 17 + 3) % 31) for i in range(n))
    elif profile == "positive_formula_v1":
        values = (abs(sample_x(i, n)) + 0.75 for i in range(n))
    elif profile == "normal_grid_-4_to_4_v1":
        values = (-4.0 + 8.0 * (i / (n - 1)) for i in range(n))
    elif profile == "probability_grid_0.01_to_0.99_v1":
        values = (0.01 + 0.98 * (i / (n - 1)) for i in range(n))
    else:
        raise ValueError(f"Unrecognized input profile: {profile}")
    x = np.fromiter(values, dtype=np.float64, count=n)
    y = (
        np.fromiter((sample_y(i, n) for i in range(n)), dtype=np.float64, count=n)
        if profile == "paired_formula_v1"
        else np.empty(0, dtype=np.float64)
    )
    return Inputs(x, y)


# The Python callable is selected once before each measurement block, so the
# measured operation does not include a 16-way dispatch on its case label.
OPERATIONS: dict[str, Callable[[Inputs], object]] = {
    "mean": lambda i: stats.tmean(i.x),
    "variance_ddof1": lambda i: stats.tvar(i.x, ddof=1),
    "quantile_linear_p30": lambda i: stats.scoreatpercentile(i.x, 30.0, interpolation_method="fraction"),
    "rankdata_average_ties": lambda i: stats.rankdata(i.x, method="average"),
    "zscore_ddof1": lambda i: stats.zscore(i.x, ddof=1),
    "pearsonr_two_sided": lambda i: stats.pearsonr(i.x, i.y, alternative="two-sided"),
    "spearmanr_two_sided": lambda i: stats.spearmanr(i.x, i.y, alternative="two-sided"),
    "ttest_1samp_two_sided": lambda i: stats.ttest_1samp(i.x, 0.1, alternative="two-sided"),
    "ttest_ind_welch_two_sided": lambda i: stats.ttest_ind(i.x, i.y, equal_var=False, alternative="two-sided"),
    "mannwhitneyu_asymptotic": lambda i: stats.mannwhitneyu(i.x, i.y, method="asymptotic", use_continuity=True, alternative="two-sided"),
    "jarque_bera": lambda i: stats.jarque_bera(i.x),
    "boxcox_lambda05": lambda i: stats.boxcox(i.x, lmbda=0.5),
    "yeojohnson_lambda05": lambda i: stats.yeojohnson(i.x, lmbda=0.5),
    "wasserstein_distance": lambda i: stats.wasserstein_distance(i.x, i.y),
    "normal_cdf_batch": lambda i: stats.norm.cdf(i.x, loc=0.5, scale=1.5),
    "normal_ppf_batch": lambda i: stats.norm.ppf(i.x, loc=0.5, scale=1.5),
}


def verify(spec: Spec, n: int) -> None:
    inputs = prepare(spec.profile, n)
    if inputs.x.shape != (n,) or inputs.x.dtype != np.float64 or not np.all(np.isfinite(inputs.x)):
        raise RuntimeError(f"{spec.key} has invalid x inputs")
    if spec.profile == "paired_formula_v1" and inputs.y.shape != (n,):
        raise RuntimeError(f"{spec.key} has invalid y inputs")
    result = OPERATIONS[spec.key](inputs)
    if result is None:
        raise RuntimeError(f"{spec.key} returned None")
    if spec.output == "numpy_ndarray" and (
        not isinstance(result, np.ndarray) or result.shape != (n,)
        or result.dtype != np.float64
    ):
        raise RuntimeError(f"{spec.key} produced an unexpected output array")


def measure(
    spec: Spec,
    phase: str,
    n: int,
    repetitions: int,
    warmups: int,
    batch: int,
) -> tuple[int, int, int, int]:
    times: list[int] = []
    operation = OPERATIONS[spec.key]
    for iteration in range(warmups + repetitions):
        if phase == "prepare_only":
            start = time.perf_counter_ns()
            for _ in range(batch):
                inputs = prepare(spec.profile, n)
                del inputs
            elapsed = time.perf_counter_ns() - start
        else:
            inputs = prepare(spec.profile, n)
            start = time.perf_counter_ns()
            for _ in range(batch):
                result = operation(inputs)
                del result
            elapsed = time.perf_counter_ns() - start
        if iteration >= warmups:
            times.append(elapsed // batch)
    times.sort()
    midpoint = len(times) // 2
    median = (
        (times[midpoint - 1] + times[midpoint]) // 2
        if len(times) % 2 == 0
        else times[midpoint]
    )
    return median, times[len(times) // 4], times[3 * len(times) // 4], times[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="CSV output file (stdout by default)")
    parser.add_argument("--quick", action="store_true", help="Run only n=128")
    parser.add_argument("--repeat", type=int, default=13, help="Timed blocks per case (1..100)")
    parser.add_argument("--warmup", type=int, default=3, help="Untimed blocks per case (0..100)")
    parser.add_argument("--batch", type=int, default=4, help="Operations per block (1..64)")
    args = parser.parse_args()
    if not 1 <= args.repeat <= 100 or not 0 <= args.warmup <= 100 or not 1 <= args.batch <= 64:
        parser.error("repeat must be 1..=100; warmup 0..=100; batch 1..=64")
    versions = (scipy.__version__, np.__version__)
    if versions != EXPECTED_VERSIONS:
        parser.error(f"Expected SciPy/NumPy {EXPECTED_VERSIONS}, found {versions}")
    if len(OPERATIONS) != len(SPECS) or set(OPERATIONS) != {spec.key for spec in SPECS}:
        parser.error("Benchmark specs and operations are inconsistent")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
    stream = args.out.open("w", newline="", encoding="utf-8") if args.out else sys.stdout
    try:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        gc_enabled = gc.isenabled()
        gc.disable()
        try:
            for spec in SPECS:
                sizes = SIZES[:1] if args.quick else SIZES
                for n in sizes:
                    verify(spec, n)
                    for phase in ("prepare_only", "compute_only"):
                        median, p25, p75, minimum = measure(
                            spec, phase, n, args.repeat, args.warmup, args.batch,
                        )
                        is_prepare = phase == "prepare_only"
                        writer.writerow({
                            "implementation": f"scipy_{scipy.__version__}",
                            "case": spec.key,
                            "family": spec.family,
                            "phase": phase,
                            "n": n,
                            "parameters": spec.parameters,
                            "input_profile": spec.profile,
                            "repetitions": args.repeat,
                            "batch_calls": args.batch,
                            "median_ns": median,
                            "p25_ns": p25,
                            "p75_ns": p75,
                            "min_ns": minimum,
                            "comparability": "adapted" if is_prepare else spec.comparability,
                            "difference": (
                                "rust_vec_preparation_vs_numpy_ndarray_preparation"
                                if is_prepare else spec.difference
                            ),
                            "output_layout": "numpy_ndarray_inputs" if is_prepare else spec.output,
                        })
        finally:
            if gc_enabled:
                gc.enable()
    finally:
        if args.out:
            stream.close()


if __name__ == "__main__":
    main()
