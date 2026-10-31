"""Audit special-function inventory names against public APIs and CSV references.

Run with ``python tests/scipy/audit_special.py``; this static audit needs only
the Python standard library. Aliases and native Rust replacements are recorded
explicitly rather than counted as separate algorithms.
"""
import csv
import re
from collections import Counter
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


def public_families():
    """Read the public numerical exports, including the elliptic glob.

    Implementation functions are private and cannot establish public coverage;
    only family re-exports in lib.rs do. Verify that each implementation name
    appears in exactly one public family.
    """
    source = ROOT / "crates/sci-special/src"
    facade = (source / "lib.rs").read_text()
    assert re.search(r"^pub mod complex;\s*$", facade, re.MULTILINE)
    assert not re.search(r"^pub fn \w+", facade, re.MULTILINE)
    # The only root-level `pub use` is for the shared Error/Result types.
    root_uses = re.findall(r"^pub use ([^;]+);", facade, re.MULTILINE)
    assert root_uses == ["sci_core::{Error, Result}"], root_uses

    implementation = set()
    for path in source.glob("*.rs"):
        if path.name in {"complex.rs", "lib.rs"}:
            continue
        contents = path.read_text()
        implementation.update(re.findall(r"^pub fn (\w+)", contents, re.MULTILINE))
        implementation.update(re.findall(r"^pub use \w+ as (\w+);", contents, re.MULTILINE))

    elliptic = (source / "elliptic.rs").read_text()
    elliptic_names = set(re.findall(r"^pub fn (\w+)", elliptic, re.MULTILINE))
    elliptic_names.update(re.findall(r"^pub use \w+ as (\w+);", elliptic, re.MULTILINE))

    families = {}
    exports = Counter()
    modules = re.findall(r"^pub mod (\w+) \{(.*?)^\}", facade, re.MULTILINE | re.DOTALL)
    expected_modules = {
        "gamma", "beta", "bessel", "orthogonal", "hypergeometric", "wave",
        "integrals", "zeta", "probability", "elementary", "combinatorics", "elliptic",
    }
    assert {name for name, _ in modules} == expected_modules
    for family, body in modules:
        declarations = re.findall(r"pub use crate::implementation::([^;]+);", body)
        assert declarations, family
        for declaration in declarations:
            declaration = declaration.strip()
            if declaration == "elliptic::*":
                names = elliptic_names
            elif declaration.startswith("{") and declaration.endswith("}"):
                names = [name.strip() for name in declaration[1:-1].split(",") if name.strip()]
            else:
                names = [declaration]
            for name in names:
                if name == "ValueDerivativeMatrices":
                    assert family == "orthogonal"
                    continue  # public type, not a numerical function
                assert name in implementation, (family, name)
                exports[name] += 1
                families[name] = family

    duplicates = {name: count for name, count in exports.items() if count != 1}
    assert not duplicates, duplicates
    assert set(exports) == implementation, (
        "missing", sorted(implementation - set(exports)),
        "extra", sorted(set(exports) - implementation),
    )
    return families


def audit():
    families = public_families()
    public = set(families)
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
    assert len(public) == 362, len(public)
    print(f"Public family numerical exports (including ellipkinc alias): {len(public)} "
          f"across {len(set(families.values()))} families; none at the root.")


if __name__ == "__main__":
    audit()
