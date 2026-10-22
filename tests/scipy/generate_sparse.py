# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Generate sparse references: uv run tests/scipy/generate_sparse.py.

Input-only cases are in sparse_cases.json. No Sci or Rust backend is used.
Comparisons use 3e-12 absolute plus relative tolerance for small well-conditioned
LU solves, products, and norms. Structural/index outputs are compared exactly.
"""
import csv
import json
from pathlib import Path
import numpy as np
import scipy
from scipy import sparse as s
from scipy.sparse import linalg as l
assert (np.__version__, scipy.__version__) == ('2.5.4', '1.18.1')
ROOT = Path(__file__).resolve().parents[1]

def encode(x):
    return ';'.join(str(v) for v in np.asarray(x).ravel())

def triples(x):
    return ':'.join(encode(row) for row in x)

def matrix(c, key='a'):
    rows, cols = (c['rows'], c['cols']) if key == 'a' else (c['brows'], c['bcols'])
    t = c.get(key, [])
    return s.coo_array((np.array([v[2] for v in t], dtype=float),
                        (np.array([v[0] for v in t], dtype=int), np.array([v[1] for v in t], dtype=int))),
                       shape=(rows, cols)).tocsr()

with (ROOT/'fixtures/sparse_inputs.csv').open('w', newline='') as fi, (ROOT/'fixtures/sparse_expected.csv').open('w', newline='') as fe:
    iw, ew = csv.writer(fi, lineterminator='\n'), csv.writer(fe, lineterminator='\n')
    iw.writerow(['id','api','rows','cols','a','brows','bcols','b','x','k'])
    ew.writerow(['id','rows','cols','values'])
    for c in json.loads(Path(__file__).with_name('sparse_cases.json').read_text()):
        api=c['api']; a=matrix(c); b=matrix(c,'b') if 'b' in c else None
        x=c.get('x',[]); k=c.get('k',0)
        iw.writerow([c['id'],api,c['rows'],c['cols'],triples(c.get('a',[])),c.get('brows',0),c.get('bcols',0),triples(c.get('b',[])),triples(x) if api in ('diags','diags_array') else encode(x),encode(k)])
        if api in ('from_coo','from_csr','from_csc','from_dense','to_dense','validate'): result=a.toarray()
        elif api in ('eye','eye_array'): result=s.eye_array(c['rows'],c['cols'],k=k)
        elif api=='identity': result=s.identity(c['rows'])
        elif api in ('transpose','matrix_transpose'): result=a.T
        elif api=='find': result=np.column_stack(s.find(a))
        elif api=='eliminate_zeros': result=a.copy();result.eliminate_zeros()
        elif api in ('tril','triu'): result=getattr(s,api)(a,k)
        elif api=='scale': result=a*k
        elif api=='add': result=a+b
        elif api=='multiply': result=a.multiply(b)
        elif api=='matvec': result=a@np.array(x)
        elif api=='matmul': result=a@b
        elif api in ('hstack','vstack','block_diag'): result=getattr(s,api)([a,b])
        elif api in ('block_array','bmat'): result=s.block_array([[a,b],[b,a]])
        elif api in ('kron','kronsum'): result=getattr(s,api)(a,b)
        elif api in ('diags','diags_array'): result=s.diags_array(x,offsets=k,shape=a.shape,dtype=float)
        elif api in ('spsolve','splu','factorized','lu_solve'): result=l.spsolve(a,np.array(x))
        elif api=='spsolve_triangular': result=l.spsolve_triangular(a,np.array(x),lower=k in (0,2),unit_diagonal=k in (2,3))
        elif api=='norm':
            # Sparse Frobenius squares entries; use SciPy BLAS nrm2 for extremes.
            result=scipy.linalg.norm(a.toarray().ravel()) if c['id'] in ('norm_large','norm_tiny') else l.norm(a,ord={0:'fro',1:1,2:np.inf,3:-1,4:-np.inf}[k])
        elif api in ('inv','matrix_power','spbandwidth','is_sptriangular'): result=getattr(l,api)(a,k) if api=='matrix_power' else getattr(l,api)(a)
        else: raise ValueError(api)
        if s.issparse(result): result=result.toarray()
        result=np.atleast_2d(np.asarray(result,dtype=float))
        ew.writerow([c['id'],*result.shape,encode(result)])
