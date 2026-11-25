#!/usr/bin/env python3
"""SciPy 1.18.1 / NumPy 2.5.4 baseline for sampling_scipy_comparison.rs.

Run from /sci (with its local Python 3.13 SciPy venv):

  env -u LD_LIBRARY_PATH \
    PYTHONHOME=/home/psira/.local/share/uv/python/cpython-3.13-linux-x86_64-gnu \
    PYTHONPATH=.venv/lib/python3.13/site-packages \
    .venv/bin/python tests/scipy/benchmark_sampling_engines.py \
    --out target/sampling_scipy.csv

Both programs use the same cases, batch sizes, dimensions, probability laws
and seed. Each construction is measured without drawing. Each draw starts
with a freshly constructed generator OUTSIDE the timed window; its complete
output allocation is timed, with destruction outside the window. Microbench
times are medians in nanoseconds, one process and one thread. Different
container layouts, RNGs and numerical methods are recorded as CSV metadata.
The Sobol flat cases use the same contiguous row-major SciPy ndarray as their
ordinary cases, paired against Rust's contiguous random_flat() output.
Matching scramble seeds reproduce each implementation's own sequence only.

Comparability:
  matched: same documented mathematical generator family and main parameters.
  adapted: mathematical target or family overlaps but implementations diverge.
Do not interpret the adapted cases as apples-to-apples kernel speed ratios.
In particular, Sci's inverse-polynomial sampler is linear, whereas SciPy's
minimum supported order is 3; and SciPy UNU.RAN can build better envelopes
than Sci's certified SimpleRatioUniforms rectangle.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Set before importing NumPy/SciPy; the Rust comparison runs on one thread too.
for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
import scipy
from scipy import stats
from scipy.stats import qmc, sampling


SEED = 42
TRIALS = 256
STD_SIZES = (128, 1024, 8192)
SHORT_SIZES = (128, 1024)
POISSON_SIZES = (64, 128)
MULTINOMIAL_SIZES = (8, 32)
EXPECTED_VERSIONS = ("1.18.1", "2.5.4")
FIELDS = (
    "implementation",
    "case",
    "family",
    "phase",
    "n",
    "d",
    "parameters",
    "repetitions",
    "median_ns",
    "min_ns",
    "comparability",
    "difference",
    "output_layout",
)


@dataclass(frozen=True)
class Spec:
    key: str
    family: str
    d: int
    sizes: tuple[int, ...]
    parameters: str
    comparability: str
    difference: str


SPECS = (
    Spec("sobol_plain", "qmc", 4, STD_SIZES, "bits=30;scramble=false", "matched", "different_output_container"),
    Spec("sobol_scrambled", "qmc", 4, STD_SIZES, "bits=30;scramble=true;seed=42", "adapted", "scramble_rng_differs"),
    Spec("sobol_plain_flat", "qmc", 4, STD_SIZES, "bits=30;scramble=false", "matched", "contiguous_row_major;vec_vs_ndarray"),
    Spec("sobol_scrambled_flat", "qmc", 4, STD_SIZES, "bits=30;scramble=true;seed=42", "adapted", "contiguous_row_major;scramble_rng_differs"),
    Spec("halton_plain", "qmc", 4, STD_SIZES, "scramble=false", "matched", "different_output_container"),
    Spec("halton_scrambled", "qmc", 4, STD_SIZES, "scramble=true;seed=42", "adapted", "scramble_rng_differs"),
    Spec("latin_hypercube", "qmc", 4, STD_SIZES, "strength=1;scramble=true;seed=42", "matched", "rng_and_output_layout_differ"),
    Spec("latin_hypercube_centered", "qmc", 4, STD_SIZES, "strength=1;scramble=false;seed=42", "matched", "rng_and_output_layout_differ"),
    Spec("poisson_disk", "qmc", 2, POISSON_SIZES, "radius=0.06;ncandidates=30;hypersphere=volume;seed=42", "adapted", "independent_candidate_schedules"),
    Spec("multivariate_normal_qmc", "qmc", 4, SHORT_SIZES, "bits=30;unscrambled_sobol;cov_offdiag=0.2", "matched", "inverse_normal_backend_and_output_layout_differ"),
    Spec("multinomial_qmc", "qmc", 3, MULTINOMIAL_SIZES, "p=0.2:0.3:0.5;trials=256;unscrambled_sobol", "adapted", "tie_conventions_differ"),
    Spec("discrete_alias_urn", "sampling", 1, STD_SIZES, "weights=1..256;seed=42", "matched", "different_rng_and_memory_layout"),
    Spec("discrete_guide_table", "sampling", 1, STD_SIZES, "weights=1..256;seed=42", "matched", "different_rng_and_memory_layout"),
    Spec("fast_generator_inversion", "sampling", 1, STD_SIZES, "normal=0:1;seed=42", "adapted", "native_ppf_vs_precomputed_unuran"),
    Spec("numerical_inverse_hermite", "sampling", 1, SHORT_SIZES, "expon_domain=0:6;u_resolution=1e-5;seed=42", "adapted", "different_knot_selection"),
    Spec("numerical_inverse_polynomial", "sampling", 1, SHORT_SIZES, "expon_domain=0:6;u_resolution=1e-5;seed=42", "adapted", "sci_linear_vs_scipy_order3"),
    Spec("ratio_uniforms", "sampling", 1, SHORT_SIZES, "unnormalized_gaussian;umax=1;vmax=sqrt(2/e);seed=42", "adapted", "python_pdf_callback_and_rng"),
    Spec("simple_ratio_uniforms", "sampling", 1, SHORT_SIZES, "density=2x;support=0:1;seed=42", "adapted", "certified_rectangle_vs_auto_envelope"),
    Spec("transformed_density_rejection", "sampling", 1, SHORT_SIZES, "truncated_gaussian=-4:4;transform_c=0;seed=42", "adapted", "different_adaptive_envelopes_and_callbacks"),
)


class TruncatedGaussian:
    """Unnormalized truncated normal with analytic derivative for UNU.RAN."""

    @staticmethod
    def pdf(x: float) -> float:
        return math.exp(-0.5 * x * x) if -4.0 <= x <= 4.0 else 0.0

    @staticmethod
    def dpdf(x: float) -> float:
        return -x * TruncatedGaussian.pdf(x)


# These describe each source law and are intentionally constructed once.
# Neither implementation includes preparation of raw weights or covariance
# input matrices in its measured engine-construction phase.
_WEIGHTS = np.arange(1, 257, dtype=np.float64)
_DISCRETE = stats.rv_discrete(
    name="sampling_comparison_weights",
    values=(np.arange(256), _WEIGHTS / _WEIGHTS.sum()),
)
_STANDARD_NORMAL = stats.norm()
_EXPONENTIAL = stats.expon()
_BETA21 = stats.beta(2, 1)
_TRUNCATED_NORMAL = TruncatedGaussian()
_COVARIANCE = np.array(
    [[1.0, 0.2, 0.0, 0.0], [0.2, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.2], [0.0, 0.0, 0.2, 1.0]],
    dtype=np.float64,
)
_MEAN = np.zeros(4, dtype=np.float64)
_PVALS = np.array([0.2, 0.3, 0.5], dtype=np.float64)


def create(spec: Spec) -> object:
    key = spec.key
    if key in ("sobol_plain", "sobol_plain_flat"):
        return qmc.Sobol(d=spec.d, scramble=False, bits=30)
    if key in ("sobol_scrambled", "sobol_scrambled_flat"):
        return qmc.Sobol(d=spec.d, scramble=True, bits=30, rng=SEED)
    if key == "halton_plain":
        return qmc.Halton(d=spec.d, scramble=False)
    if key == "halton_scrambled":
        return qmc.Halton(d=spec.d, scramble=True, rng=SEED)
    if key == "latin_hypercube":
        return qmc.LatinHypercube(d=spec.d, scramble=True, strength=1, rng=SEED)
    if key == "latin_hypercube_centered":
        return qmc.LatinHypercube(d=spec.d, scramble=False, strength=1, rng=SEED)
    if key == "poisson_disk":
        return qmc.PoissonDisk(d=2, radius=0.06, hypersphere="volume", ncandidates=30, rng=SEED)
    if key == "multivariate_normal_qmc":
        return qmc.MultivariateNormalQMC(
            mean=_MEAN, cov=_COVARIANCE,
            inv_transform=True, engine=qmc.Sobol(d=4, bits=30, scramble=False),
        )
    if key == "multinomial_qmc":
        return qmc.MultinomialQMC(
            pvals=_PVALS, n_trials=TRIALS,
            engine=qmc.Sobol(d=1, bits=30, scramble=False),
        )
    if key == "discrete_alias_urn":
        return sampling.DiscreteAliasUrn(_DISCRETE, random_state=SEED)
    if key == "discrete_guide_table":
        return sampling.DiscreteGuideTable(_DISCRETE, random_state=SEED)
    if key == "fast_generator_inversion":
        return sampling.FastGeneratorInversion(_STANDARD_NORMAL, random_state=SEED)
    if key == "numerical_inverse_hermite":
        return sampling.NumericalInverseHermite(
            _EXPONENTIAL, domain=(0.0, 6.0), order=3,
            u_resolution=1e-5, random_state=SEED,
        )
    if key == "numerical_inverse_polynomial":
        # SciPy supports orders 3..17; Sci uses an order-1 linear table.
        return sampling.NumericalInversePolynomial(
            _EXPONENTIAL, domain=(0.0, 6.0), order=3,
            u_resolution=1e-5, random_state=SEED,
        )
    if key == "ratio_uniforms":
        vmax = math.sqrt(2.0 / math.e)
        return sampling.RatioUniforms(
            lambda x: np.exp(-0.5 * x * x),
            umax=1.0, vmin=-vmax, vmax=vmax, c=0.0, random_state=SEED,
        )
    if key == "simple_ratio_uniforms":
        return sampling.SimpleRatioUniforms(
            _BETA21, mode=1.0, pdf_area=1.0,
            domain=(0.0, 1.0), random_state=SEED,
        )
    if key == "transformed_density_rejection":
        return sampling.TransformedDensityRejection(
            _TRUNCATED_NORMAL, c=0.0, domain=(-4.0, 4.0),
            random_state=SEED,
        )
    raise ValueError(f"Unknown benchmark case: {key}")


def draw(spec: Spec, engine: object, n: int) -> object:
    if spec.family == "qmc":
        return engine.random(n)
    return engine.rvs(size=n)


def measure(
    spec: Spec, phase: str, n: int, repetitions: int, warmups: int,
) -> tuple[int, int]:
    timings: list[int] = []
    for index in range(warmups + repetitions):
        if phase == "construct_only":
            start = time.perf_counter_ns()
            result = create(spec)
            elapsed = time.perf_counter_ns() - start
        else:
            prepared = create(spec)
            start = time.perf_counter_ns()
            result = draw(spec, prepared, n)
            elapsed = time.perf_counter_ns() - start

        # Keep result alive through the measurement; release it afterward.
        if result is None:
            raise RuntimeError(f"{spec.key} unexpectedly produced None")
        if phase == "draw_random_only" and len(result) != n:
            raise RuntimeError(
                f"{spec.key} returned {len(result)} observations; requested {n}"
            )
        if phase == "draw_random_only" and spec.key.endswith("_flat"):
            if (not isinstance(result, np.ndarray) or result.shape != (n, spec.d)
                    or result.dtype != np.float64 or not result.flags.c_contiguous):
                raise RuntimeError(f"{spec.key} did not produce a C-contiguous f64 (n,d) array")
        if index >= warmups:
            timings.append(elapsed)
    timings.sort()
    middle = len(timings) // 2
    median = (
        (timings[middle - 1] + timings[middle]) // 2
        if len(timings) % 2 == 0
        else timings[middle]
    )
    return median, timings[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Output CSV (stdout by default)")
    parser.add_argument("--quick", action="store_true", help="Smallest batch for each case")
    parser.add_argument("--repeat", type=int, default=9, help="Timed repetitions (default: 9)")
    parser.add_argument("--warmup", type=int, default=2, help="Untimed warmup runs (default: 2)")
    args = parser.parse_args()
    if not 1 <= args.repeat <= 100 or not 0 <= args.warmup <= 100:
        parser.error("repeat must be 1..100 and warmup 0..100")
    versions = scipy.__version__, np.__version__
    if versions != EXPECTED_VERSIONS:
        parser.error(
            f"Expected scipy/numpy {EXPECTED_VERSIONS} for the pinned comparison, got {versions}"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
    stream = args.out.open("w", newline="", encoding="utf-8") if args.out else sys.stdout
    try:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            for spec in SPECS:
                for phase in ("construct_only", "draw_random_only"):
                    sizes = (0,) if phase == "construct_only" else spec.sizes[:1] if args.quick else spec.sizes
                    for n in sizes:
                        median, minimum = measure(spec, phase, n, args.repeat, args.warmup)
                        writer.writerow({
                            "implementation": f"scipy_{scipy.__version__}",
                            "case": spec.key,
                            "family": spec.family,
                            "phase": phase,
                            "n": n,
                            "d": spec.d,
                            "parameters": spec.parameters,
                            "repetitions": args.repeat,
                            "median_ns": median,
                            "min_ns": minimum,
                            "comparability": spec.comparability,
                            "difference": spec.difference,
                            "output_layout": (
                                "numpy_ndarray_c_row_major"
                                if spec.key.endswith("_flat") else "numpy_ndarray"
                            ),
                        })
        finally:
            if was_enabled:
                gc.enable()
    finally:
        if args.out:
            stream.close()


if __name__ == "__main__":
    main()
