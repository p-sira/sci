# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Generate FFT fixtures using input-only JSON, SciPy 1.18.1 / NumPy 2.5.4.

Run: uv run tests/scipy/generate_fft.py
Reference comparisons use 3e-12 absolute + relative tolerance for the short
transforms here, reflecting roundoff from different FFT/DCT factorizations.
Smooth-length and permutation/frequency ordering checks are exact where useful.
Order 20001 and DC-only order 1000000 cases use 3e-10: both the original XSF backend and this Rust
translation differ from SciPy by roughly 4e-11 in the first forward sample.
Subtracting log-gamma values around 8e4 loses absolute precision in the gamma
ratio; a few ulps there propagate through the coefficients and transform.
"""
import csv
import json
from pathlib import Path
import numpy as np
import scipy
from scipy import fft
assert (np.__version__,scipy.__version__)==('2.5.4','1.18.1')
ROOT=Path(__file__).resolve().parents[1]
def encode(a):return ';'.join(str(v) for v in np.asarray(a).ravel())
with (ROOT/'fixtures/fft_inputs.csv').open('w',newline='') as fi,(ROOT/'fixtures/fft_expected.csv').open('w',newline='') as fe:
    iw=csv.writer(fi,lineterminator='\n');ew=csv.writer(fe,lineterminator='\n')
    iw.writerow(['id','api','re','im','shape','axes','n','d','target','type','dln','mu','offset','bias','tolerance']);ew.writerow(['id','shape','re','im'])
    for c in json.loads(Path(__file__).with_name('fft_cases.json').read_text()):
        api=c['api'];re=np.array(c.get('re',[]),dtype=float);im=np.array(c.get('im',np.zeros_like(re)),dtype=float)
        iw.writerow([c['id'],api,encode(re),encode(im),encode(c.get('shape',[])),encode(c.get('axes',[])),c.get('n') if c.get('n') is not None else '',c.get('d',''),c.get('target',''),c.get('type',''),c.get('dln',''),c.get('mu',''),c.get('offset',''),c.get('bias',''),c.get('tolerance',3e-12)])
        if api in ('fht','ifht'):v=getattr(fft,api)(re,c['dln'],c['mu'],offset=c['offset'],bias=c['bias'])
        elif api=='fhtoffset':v=fft.fhtoffset(c['dln'],c['mu'],initial=c['offset'],bias=c['bias'])
        elif api in ('fftfreq','rfftfreq'):v=getattr(fft,api)(c['n'],c['d'])
        elif api in ('next_fast_len','prev_fast_len'):
            # Independent number-theory reference: enumerate 11-smooth products,
            # rather than reproducing Sci's trial-division search or using its backend.
            lengths = {1}
            for prime in (2, 3, 5, 7, 11):
                expanded = set()
                for base in lengths:
                    while base <= 1_048_576:
                        expanded.add(base)
                        base *= prime
                lengths = expanded
            target = c['target']
            v = 0 if target == 0 else (min(n for n in lengths if n >= target) if api == 'next_fast_len' else max(n for n in lengths if n <= target))
        elif api in ('dct','idct','dst','idst','dctn','idctn','dstn','idstn'):
            x=re.reshape(c['shape']);kwargs=dict(type=c['type'],norm='backward',orthogonalize=False)
            if api.endswith('n'):kwargs['axes']=c['axes']
            v=getattr(fft,api)(x,**kwargs)
        elif api in ('fftn','ifftn','rfftn','irfftn','hfftn','ihfftn','fft2','ifft2','rfft2','irfft2','hfft2','ihfft2'):
            real_input=api in ('rfftn','ihfftn','rfft2','ihfft2')
            x=re if real_input else re+1j*im
            input_shape=c['shape'].copy();axes=c.get('axes',[0,1]);kwargs=dict(axes=axes)
            if api in ('irfftn','hfftn','irfft2','hfft2'):
                input_shape[axes[-1]]=input_shape[axes[-1]]//2+1
                kwargs['s']=[c['shape'][a] for a in axes]
            v=getattr(fft,api)(x.reshape(input_shape),**kwargs)
        else:
            x=re if api in ('rfft','ihfft') else re+1j*im
            kwargs=dict(n=c.get('n')) if api in ('irfft','hfft','ihfft') else {}
            v=getattr(fft,api)(x,**kwargs)
        a=np.atleast_1d(np.asarray(v));ew.writerow([c['id'],encode(a.shape),encode(a.real),encode(a.imag)])
