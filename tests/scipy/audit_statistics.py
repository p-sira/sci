"""Audit statistics exports, public fixtures, and the complete SciPy inventory.

Run: python tests/scipy/audit_statistics.py. Missing canonical function names
or exported functions without reference fixtures fail the audit.
Compatibility namespaces are checked as aliases, not independent algorithms.
Masked-array-only behavior is not counted as implemented.
"""

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NATIVE_DISTRIBUTIONS = {
    "norm": "Normal",
    "levy": "Levy",
    "beta": "Beta",
    "gamma": "Gamma",
    "chi2": "ChiSquared",
    "chi": "Chi",
    "t": "StudentT",
    "f": "F",
    "expon": "Exponential",
    "cauchy": "Cauchy",
    "laplace": "Laplace",
    "lognorm": "LogNormal",
    "uniform": "Uniform",
    "weibull_min": "Weibull",
    "pareto": "Pareto",
    "invgamma": "InverseGamma",
    "gumbel_r": "Gumbel",
    "triang": "Triangular",
    "erlang": "Erlang",
    "randint": "DiscreteUniform",
    "bernoulli": "Bernoulli",
    "binom": "Binomial",
    "poisson": "Poisson",
    "geom": "Geometric",
    "nbinom": "NegativeBinomial",
    "hypergeom": "Hypergeometric",
}
MULTIVARIATE = ["multivariate_normal", "multivariate_t", "dirichlet", "multinomial"]


def inventory():
    namespaces = {}
    namespace = section = None
    for line in (ROOT / "function_coverages/scipy_function_list/scipy.stats.md").read_text().splitlines():
        if line.startswith("### scipy.stats"):
            namespace = line[4:]
            namespaces[namespace] = {"functions": set(), "objects": set()}
        elif line.startswith("#### "):
            section = line[5:]
        elif namespace and line.startswith("- "):
            name = re.search(r"`([^`]+)`", line)
            if name and section == "Functions":
                namespaces[namespace]["functions"].add(name[1])
            if name and section == "Callable objects and distribution APIs":
                namespaces[namespace]["objects"].add(name[1])
    return namespaces


def audit():
    exported = {}
    for path in sorted((ROOT / "crates/sci-stats/src").glob("*.rs")):
        namespace = (
            "scipy.stats." + path.stem
            if path.stem in ["contingency", "qmc"]
            else "scipy.stats"
        )
        for name in re.findall(r"^pub fn (\w+)(?:<[^\n]*>)?\(", path.read_text(), re.MULTILINE):
            exported.setdefault(namespace, set()).add(name)
    exported["scipy.stats"].add("chi2_contingency")  # shared public alias
    cases = json.loads((ROOT / "tests/scipy/statistics_cases.json").read_text())
    tested = {c["api"] for c in cases}
    with (ROOT / "tests/fixtures/statistics.csv").open() as f:
        tested.update(row["function"] for row in csv.DictReader(f))
    missing = set.union(*exported.values()) - tested
    if missing:
        raise SystemExit(
            "Exports without reference fixtures: " + ", ".join(sorted(missing))
        )
    for family in NATIVE_DISTRIBUTIONS.values():
        operations = {c["api"] for c in cases if c["d"] == family}
        if operations != {"pdf", "logpdf", "cdf", "sf", "ppf"}:
            raise SystemExit("Incomplete distribution fixtures: " + family)
    for family in MULTIVARIATE:
        if (
            not {
                family + "_" + suffix
                for suffix in (
                    ["pmf", "logpmf"] if family == "multinomial" else ["pdf", "logpdf"]
                )
            }
            <= tested
        ):
            raise SystemExit("Missing multivariate fixtures: " + family)
    with (ROOT / "tests/fixtures/stats_inputs.csv").open() as f:
        inputs = list(csv.DictReader(f))
    with (ROOT / "tests/fixtures/stats_expected.csv").open() as f:
        expected = list(csv.DictReader(f))
    if len(inputs) != len(cases) or len(expected) != len(cases):
        raise SystemExit("Fixtures are stale; regenerate statistics references")
    for i, (c, row, e) in enumerate(zip(cases, inputs, expected, strict=True)):
        encode = lambda x: ";".join(str(float(v)) for v in x)
        wanted = {
            "id": str(i),
            "api": c["api"],
            "params": encode(c["p"]),
            "x": encode(c["x"]),
            "y": encode(c["y"]),
            "groups": "/".join(encode(r) for r in c["g"]),
            "distribution": c["d"],
        }
        if row != wanted or e["id"] != str(i):
            raise SystemExit("Fixture input drift at case " + str(i))
    namespaces = inventory()
    report = (ROOT / "function_coverages/STATISTICS_COVERAGE.md").read_text()
    normalized_report = " ".join(report.split())
    main_count = len(namespaces["scipy.stats"]["functions"])
    if f"{main_count} of {main_count} scipy.stats function names" not in normalized_report:
        raise SystemExit("Coverage report has stale main-namespace counts")
    if f"{len(cases)} cases" not in normalized_report:
        raise SystemExit("Coverage report has a stale fixture count")
    for ns in ["scipy.stats", "scipy.stats.contingency", "scipy.stats.qmc"]:
        names = namespaces[ns]["functions"]
        covered = names & exported[ns]
        gaps = names - covered
        print(
            f"{ns}: {len(covered)}/{len(names)} function names with supported native signatures"
        )
        if gaps:
            raise SystemExit("Missing canonical functions: " + ", ".join(sorted(gaps)))
    print(
        f"{len(exported['scipy.stats'] | exported['scipy.stats.contingency'] | exported['scipy.stats.qmc'])} unique native numerical functions exercised"
    )
    names = namespaces["scipy.stats"]["objects"]
    print(
        f"{len(NATIVE_DISTRIBUTIONS) + len(MULTIVARIATE)}/{len(names)} callable distribution families with tested evaluation APIs"
    )
    masked = namespaces["scipy.stats.mstats"]["functions"]
    print(
        f"mstats: {len(masked & exported['scipy.stats'])}/{len(masked)} names share unmasked algorithms; masking is unsupported"
    )
    print(
        f"{len(cases)} checked-in reference cases; compatibility namespaces introduce no claimed additional algorithms"
    )
    return namespaces, exported


if __name__ == "__main__":
    audit()
