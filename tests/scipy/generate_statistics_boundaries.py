# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Generate distribution support and normal-posterior boundary references.

Run with uv or the pinned .venv. Input-only JSON specifies profiles and the
deterministic sample pattern. CSV input and expected files are separate.
Rust comparisons use 2e-10 absolute plus relative tolerance for different
CDF/inverse implementations, and explicitly compare infinities and NaNs.
The n=1000/1001 cases straddle SciPy's Gaussian-posterior approximation switch.
SciPy nbinom(r, p=1) returns NaN for densities at positive infinity. For those
two cases, use its mathematically equivalent Poisson(0) point mass at zero.
SciPy geom(p=1) returns zero for interior quantiles, below its support. Use
the identity Geometric(1) = 1 + Poisson(0), retaining the endpoint convention.
"""
import csv
import json
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

if (np.__version__, scipy.__version__) != ("2.5.4", "1.18.1"):
    raise RuntimeError("Use NumPy 2.5.4 and SciPy 1.18.1")

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads(Path(__file__).with_name("statistics_boundary_cases.json").read_text())


def distribution(name, p):
    return {
        "Poisson": lambda: stats.poisson(p[0]),
        "NegativeBinomial": lambda: stats.nbinom(*p),
        "Bernoulli": lambda: stats.bernoulli(p[0]),
        "Binomial": lambda: stats.binom(*p),
        "Geometric": lambda: stats.geom(p[0]),
        "DiscreteUniform": lambda: stats.randint(*map(int, p)),
        "Beta": lambda: stats.beta(*p),
        "Gamma": lambda: stats.gamma(p[0], scale=p[1]),
        "InverseGamma": lambda: stats.invgamma(p[0], scale=p[1]),
    }[name]()


def encode(values):
    return ";".join(str(float(v)) for v in values)


with (ROOT / "fixtures/statistics_boundary_inputs.csv").open("w", newline="") as fi, (
    ROOT / "fixtures/statistics_boundary_expected.csv"
).open("w", newline="") as fe:
    inputs = csv.writer(fi, lineterminator="\n")
    expected = csv.writer(fe, lineterminator="\n")
    inputs.writerow(["id", "api", "family", "parameters", "argument"])
    expected.writerow(["id", "values"])
    index = 0

    def write(api, family, parameters, argument, value):
        global index
        inputs.writerow([index, api, family, encode(parameters), argument])
        expected.writerow([index, encode(np.asarray(value).ravel())])
        index += 1

    with np.errstate(all="ignore"):
        for profile in CASES["distributions"]:
            name, p = profile["name"], profile["parameters"]
            d = distribution(name, p)
            discrete = name not in {"Beta", "Gamma", "InverseGamma"}
            for api in ["pdf", "logpdf", "cdf", "sf", "ppf"]:
                method = {"pdf": "pmf", "logpdf": "logpmf"}.get(api, api) if discrete else api
                arguments = CASES["probabilities"] if api == "ppf" else CASES["observations"]
                for argument in map(float, arguments):
                    oracle = d
                    if name == "NegativeBinomial" and p[1] == 1 and argument == np.inf and api in {"pdf", "logpdf"}:
                        oracle = stats.poisson(0)
                    if name == "Geometric" and p[0] == 1 and api == "ppf":
                        value = stats.poisson(0).ppf(argument) + 1
                    else:
                        value = getattr(oracle, method)(argument)
                    write(api, name, p, argument, [value])
        posterior = CASES["posterior"]
        for n in posterior["lengths"]:
            values = np.resize(np.asarray(posterior["pattern"]), n)
            for probability in posterior["probabilities"]:
                result = [d.ppf(probability) for d in stats.mvsdist(values)]
                write("mvsdist", "", [probability], encode(values), result)
            summaries = stats.bayes_mvs(values, alpha=posterior["confidence"])
            result = [v for summary in summaries for v in (summary.statistic, *summary.minmax)]
            write("bayes_mvs", "", [posterior["confidence"]], encode(values), result)
