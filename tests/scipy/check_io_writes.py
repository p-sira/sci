# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Verify Rust-written files with SciPy after cargo test --test io.

Each reference comes from the committed SciPy-generated CSV, never Rust.
Empty dense Matrix Market uses the same independent empty-shape reference
as generate_io.py because SciPy's C++ reader crashes on that valid boundary.
"""
import csv
import io
from pathlib import Path
import numpy as np
import scipy
from scipy import io as sio, sparse
assert (np.__version__,scipy.__version__)==('2.5.4','1.18.1')
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'target/io-rust'
FIX=ROOT/'tests/fixtures'


def rows(name):
    with (FIX/name).open() as f:
        return list(csv.DictReader(f))


def values(s,dtype=float):
    return np.array(s.split(';') if s else [],dtype=dtype)


def equal(a,b):
    np.testing.assert_array_equal(a,b,strict=False)
    mask=np.isfinite(b) & (b==0)
    np.testing.assert_array_equal(np.signbit(a[mask]),np.signbit(b[mask]))


checked=0
for r in rows('io_market_expected.csv'):
    path=OUT/(r['id']+'.mtx')
    shape=(int(r['rows']),int(r['cols']))
    if r['id']=='empty_dense':
        assert sio.mminfo(path)[:3]==(*shape,0)
        # Structural check complements the reference for the empty payload.
        assert path.read_text().splitlines()==['%%MatrixMarket matrix array real general','0 3']
        a=sparse.coo_array(shape).toarray()
    else:
        a=sio.mmread(path,spmatrix=False)
        if sparse.issparse(a):a=a.toarray()
    assert a.shape==shape
    # SciPy canonicalizes textual negative zero; its independently decoded
    # values are compared here, while Rust checks preservation against the input.
    equal(a.ravel(),values(r['values']));checked+=1
for r in rows('io_wav_expected.csv'):
    rate,a=sio.wavfile.read(OUT/(r['id']+'.wav'))
    assert rate==int(r['rate'])
    assert (1 if a.ndim==1 else a.shape[1])==int(r['channels'])
    assert a.dtype.name==r['dtype']
    equal(a.ravel(),values(r['values'],r['dtype']));checked+=1
for r in rows('io_matlab_expected.csv'):
    for compression in (0,1):
        path=OUT/f"{r['id']}_{compression}.mat"
        a=sio.loadmat(path)[r['name']]
        assert a.shape==tuple(values(r['shape'],int))
        assert a.real.dtype.name==r['dtype']
        assert np.iscomplexobj(a)==bool(int(r['complex']))
        assert sio.whosmat(path)==[(r['name'],a.shape,r['class'])]
        equal(a.real.ravel(order='F'),values(r['real'],r['dtype']))
        if int(r['complex']):equal(a.imag.ravel(order='F'),values(r['imag'],r['dtype']))
        checked+=1
print(f'SciPy verified {checked} Rust-written Matrix Market, WAV and MATLAB files.')
