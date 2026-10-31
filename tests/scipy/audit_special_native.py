"""Audit native Sci APIs and the explicit XSF header-family disposition.

Run with Python's standard library; optional --xsf PATH additionally checks the
local reference header set. This is a source-family review, not a denominator
claim for every C++ overload, template instantiation, or mathematical domain.
"""
import argparse
import csv
import json
import re
from pathlib import Path
from audit_special import public_families
from case_catalog import read_cases

ROOT = Path(__file__).resolve().parents[2]
SCALAR = {'digammainv','gamma_ratio','iv_ratio','iv_ratio_c','iv_ratioinv',
          'log1pmx','scaled_exp1','normalized_gen_harmonic'}
COMPLEX_BASE = {'erfcx','erfi','dawsn','loggamma','rgamma','digamma','gamma','erf','erfc','wofz','jv','jve','yv','yve','iv','ive','kv','kve',
           'hankel1','hankel1e','hankel2','hankel2e','airy','airye','fresnel'}
COMPLEX_ADDITIONS = {
    'lambertw': 'special_complex_lambert.rs',
    'spence': 'special_complex_polylog.rs',
    'xlogy': 'special_complex_polylog.rs',
    'xlog1py': 'special_complex_polylog.rs',
    'exp1': 'special_complex_expint.rs',
    'expi': 'special_complex_expint.rs',
    'zeta': 'special_complex_zeta.rs',
    'hurwitz_zeta': 'special_complex_zeta.rs',
}
COMPLEX = COMPLEX_BASE | set(COMPLEX_ADDITIONS)
SCALAR_FAMILIES = {
    'digammainv': 'gamma',
    'gamma_ratio': 'gamma',
    'iv_ratio': 'bessel',
    'iv_ratio_c': 'bessel',
    'iv_ratioinv': 'bessel',
    'log1pmx': 'elementary',
    'scaled_exp1': 'integrals',
    'normalized_gen_harmonic': 'combinatorics',
}


def audit(xsf=None):
    source = ROOT/'crates/sci-special/src'
    families = public_families()
    assert set(SCALAR_FAMILIES) == SCALAR
    assert {name: families.get(name) for name in SCALAR} == SCALAR_FAMILIES
    for filename,expected in [('native.rs',SCALAR),('complex.rs',COMPLEX)]:
        exported = set(re.findall(r'^pub fn (\w+)<T: Float>',(source/filename).read_text(),re.M))
        assert exported == expected,(filename,exported ^ expected)
    names = SCALAR | {'complex.'+name for name in COMPLEX_BASE}
    cases = read_cases(ROOT / 'tests/scipy/special_native_cases.jsonl')
    assert {case['name'] for case in cases} == names
    with (ROOT/'tests/fixtures/special_native.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    for precision in ('f32','f64'):
        selected = [row for row in rows if row['precision']==precision]
        assert len(selected)==len(cases)
        assert {row['function'] for row in selected}==names
    # The 0.3.0 follow-up validates new overloads in separate independent
    # reference suites. Keep the original 1976-row native baseline intact.
    for name, filename in COMPLEX_ADDITIONS.items():
        suite = ROOT / 'tests' / filename
        assert suite.exists(), (name, filename)
        assert re.search(r'\b' + re.escape(name) + r'\b', suite.read_text()), (name, filename)
    manifest = json.loads((ROOT/'tests/scipy/special_native_inventory.json').read_text())
    headers = [header for group in manifest['groups'] for header in group['headers']]
    assert len(headers)==len(set(headers))==66
    if xsf:
        include = Path(xsf)/'include/xsf'
        actual = {str(path.relative_to(include)) for directory in (include,include/'cpu')
                  for path in directory.glob('*.h')}
        assert actual==set(headers),(actual-set(headers),set(headers)-actual)
    print(f'{len(headers)} XSF public-layer headers reviewed; {len(SCALAR)} scalar family APIs and '
          f'{len(COMPLEX)} complex-module APIs; {len(rows)} original f32/f64 public reference rows '
          f'and {len(COMPLEX_ADDITIONS)} independently tested complex follow-up APIs.')
    print('Numerical overload/domain gaps and intentional exclusions: function_coverages/SPECIAL_NATIVE_COVERAGE.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xsf',type=Path)
    audit(parser.parse_args().xsf)
