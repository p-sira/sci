"""Audit special-function inventory names against public APIs and CSV references.

Run with ``python tests/scipy/audit_special.py``; this static audit needs only
the Python standard library. Aliases and native Rust replacements are recorded
explicitly rather than counted as separate algorithms.
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALIASES = {
    "assoc_laguerre": "eval_genlaguerre", "psi": "digamma",
    **dict(zip(("c", "cg", "h", "he", "j", "js", "l", "la", "p", "ps", "s", "t", "ts", "u", "us"),
               ("chebyc", "gegenbauer", "hermite", "hermitenorm", "jacobi", "sh_jacobi", "laguerre", "genlaguerre", "legendre", "sh_legendre", "chebys", "chebyt", "sh_chebyt", "chebyu", "sh_chebyu"), strict=True)),
}
ALIASES = {(f"{key}_roots" if len(key) <= 2 and key not in {"psi"} else key):
           (f"roots_{value}" if key not in {"assoc_laguerre", "psi"} else value)
           for key, value in ALIASES.items()}
NATIVE = {"cbrt", "exp10", "exp2", "expm1", "log1p", "round", "geterr", "seterr"}


def audit():
    source = "\n".join(path.read_text() for path in (ROOT / "crates/sci-special/src").glob("*.rs"))
    public = set(re.findall(r"^pub fn (\w+)", source, re.MULTILINE))
    with (ROOT / "tests/fixtures/special_extended.csv").open() as stream:
        references = {row["function"] for row in csv.DictReader(stream)}
    text = (ROOT / "function_coverages/scipy_function_list/scipy.special.md").read_text()
    for section in re.split(r"^### ", text, flags=re.MULTILINE)[1:]:
        namespace = section.splitlines()[0]
        # Class methods are separate inventory entries, not numerical functions.
        functions = section.split("#### Classes")[0]
        names = set(re.findall(r"^- \[?`(\w+)`", functions, re.MULTILINE))
        direct = names & public
        remaining = names - public - ALIASES.keys() - NATIVE
        assert not remaining, (namespace, sorted(remaining))
        assert direct <= references, (namespace, sorted(direct - references))
        for alias in names & ALIASES.keys():
            assert ALIASES[alias] in public & references, alias
        print(f"{namespace}: {len(names)} names, {len(direct)} direct, "
              f"{len(names & ALIASES.keys())} aliases, {len(names & NATIVE)} native replacements")
    assert len(public) == 353, len(public)
    print(f"Public numerical functions: {len(public)}; all previous names retained.")


if __name__ == "__main__":
    audit()
