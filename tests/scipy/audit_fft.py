"""Audit FFT public numerical inventory against exports, fixtures, and tests."""
import csv
import re
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
names=set(re.findall(r'^- \[`(\w+)`\]',(ROOT/'function_coverages/scipy_function_list/scipy.fft.md').read_text(),re.M))
source='\n'.join(p.read_text() for p in (ROOT/'crates/sci-fft/src').glob('*.rs'))
exported=set(re.findall(r'pub fn (\w+)\(',source))|{'fft','ifft','rfft'}
assert 'pub use fourier::{fft, ifft, rfft};' in source
periodic_source=(ROOT/'crates/sci-fft/src/periodic.rs').read_text()
periodic=set(re.findall(r'pub fn (\w+)\(',periodic_source))
assert periodic=={'diff','hilbert','ihilbert','tilbert','itilbert','shift','cc_diff','cs_diff','sc_diff','ss_diff'}
assert periodic.isdisjoint(names)
exported-=periodic
assert exported<=names
with (ROOT/'tests/fixtures/fft_inputs.csv').open() as f:cases=list(csv.DictReader(f))
with (ROOT/'tests/fixtures/fft_expected.csv').open() as f:expected=list(csv.DictReader(f))
assert [c['id'] for c in cases]==[c['id'] for c in expected]
counts=Counter(c['api'] for c in cases)
assert set(counts)==exported
assert all(counts[name]>=4 for name in exported)
tests=(ROOT/'tests/fft_extended.rs').read_text()
for name in exported:assert f'f::{name}(' in tests,(name,'missing public test')
remaining=names-exported
report=(ROOT/'function_coverages/FFT_COVERAGE.md').read_text()
assert all(f'`{name}`' in report for name in remaining)
print(f'scipy.fft: {len(exported)}/{len(names)} names; all numerical functions exported and tested; {len(cases)} reference cases')
print('Remaining Python runtime configuration: '+', '.join(sorted(remaining)))

with (ROOT/'tests/fixtures/fft_periodic_inputs.csv').open() as f:cases=list(csv.DictReader(f))
with (ROOT/'tests/fixtures/fft_periodic_expected.csv').open() as f:expected=list(csv.DictReader(f))
assert [c['id'] for c in cases]==[c['id'] for c in expected]
counts=Counter(c['api'] for c in cases)
assert set(counts)==periodic
assert all(counts[name]>=4 for name in periodic)
tests=(ROOT/'tests/fft_periodic.rs').read_text()
for name in periodic:assert f'f::{name}(' in tests,(name,'missing public test')
assert all(f'`{name}`' in report for name in periodic)
text=(ROOT/'function_coverages/scipy_function_list/scipy.fftpack.md').read_text()
main=text.split('### scipy.fftpack\n',1)[1].split('### scipy.fftpack.convolve',1)[0]
legacy=set(re.findall(r'^- \[`(\w+)`\]',main,re.M))
removed={'rfft','irfft','rfftfreq','next_fast_len','idct','idst','idctn','idstn'}
shared=legacy-periodic-removed
assert len(shared)==13 and shared<=exported
assert legacy==shared|periodic|removed
assert all(f'`{name}`' in report for name in removed)
assert not (ROOT/'crates/sci-fftpack').exists()
assert 'sci_fftpack' not in (ROOT/'src/lib.rs').read_text()
assert 'fftpack' not in (ROOT/'Cargo.toml').read_text()
print(f'Periodic operators: {len(periodic)}/10 exported and tested; {len(cases)} reference cases')
print('FFTPACK inventory: 13 shared APIs, 10 migrated operators, 8 removed compatibility variants; 4 kernel/cache names excluded')
