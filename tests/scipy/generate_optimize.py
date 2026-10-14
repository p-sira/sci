# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Generate optimization-helper references from input-only JSON.

Run: uv run tests/scipy/generate_optimize.py
Analytical polynomials use 3e-12 absolute + relative tolerance. Forward differences
use 3e-9 absolute + relative: subtracting independently rounded libm outputs and
objective reductions amplifies their last-bit differences by 1/epsilon. These
cases use steps >= 2^-20; this tolerance is not a derivative accuracy guarantee.
"""
import csv
import json
from pathlib import Path
import numpy as np
import scipy
from scipy import optimize as o
assert (np.__version__,scipy.__version__)==('2.5.4','1.18.1')
ROOT=Path(__file__).resolve().parents[1]
def encode(a):return ';'.join(str(v) for v in np.asarray(a).ravel())
def objective(kind):
    if kind=='quadratic':return lambda x: x[0]**2+3*x[1]**2+x[0]*x[1]
    if kind=='vector':return lambda x: np.array([x[0]**2+3*x[1],x[0]*x[1],np.sin(x[0])+np.cos(x[1])])
    if kind=='rosen':return o.rosen
    raise ValueError(kind)
with (ROOT/'fixtures/optimize_inputs.csv').open('w',newline='') as fi,(ROOT/'fixtures/optimize_expected.csv').open('w',newline='') as fe:
    iw=csv.writer(fi,lineterminator='\n');ew=csv.writer(fe,lineterminator='\n')
    iw.writerow(['id','api','x','p','epsilon','function','bias']);ew.writerow(['id','rows','columns','values'])
    for c in json.loads(Path(__file__).with_name('optimize_cases.json').read_text()):
        x=np.array(c['x']);name=c['api'];eps=np.array(c.get('epsilon',[]));bias=c.get('bias',0.0)
        iw.writerow([c['id'],name,encode(x),encode(c.get('p',[])),encode(eps),c.get('function',''),bias])
        if name=='rosen_hess_prod':v=o.rosen_hess_prod(x,c['p'])
        elif name=='approx_fprime':v=o.approx_fprime(x,objective(c['function']),eps)
        elif name=='check_grad':
            gradient=(lambda x: np.array([2*x[0]+x[1],6*x[1]+x[0]])+bias) if c['function']=='quadratic' else (lambda x: o.rosen_der(x)+bias)
            v=o.check_grad(objective(c['function']),gradient,x,epsilon=eps)
        else:v=getattr(o,name)(x)
        a=np.atleast_2d(np.asarray(v));ew.writerow([c['id'],*a.shape,encode(a)])
