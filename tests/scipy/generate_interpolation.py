# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Generate references for all twenty added interpolation names.

Inputs live in interpolation_cases.jsonl. Encodings: semicolon vectors, colon
matrix rows, pipe-separated spline fields. Fit results are tested by evaluation,
knots and degree; calculus representations are also compared coefficient-wise.
Knot-refinement snapshots for positive targets may differ from SciPy; exact
reference cases cover interpolation and a target satisfied by the first fit.
Run: uv run tests/scipy/generate_interpolation.py.
"""
import csv
from pathlib import Path
from case_catalog import read_cases
import numpy as np
import scipy
from scipy import interpolate as p
assert (np.__version__, scipy.__version__) == ("2.5.4", "1.18.1")
root=Path(__file__).resolve().parents[2]
def encode(v):
    if v is None:return ""
    if isinstance(v,dict):
        if 'knots' in v:return '|'.join(encode(v[k]) for k in ['knots','coefficients','degree'])
        return '|'.join(encode(v[k]) for k in ['x_knots','y_knots','coefficients','x_degree','y_degree'])
    if isinstance(v,bool):return str(v).lower()
    if isinstance(v,list) and v and isinstance(v[0],list):return ':'.join(encode(row) for row in v)
    a=np.asarray(v)
    if a.ndim==0:return str(a.item())
    if a.ndim==1:return ';'.join(str(x) for x in a)
    return ':'.join(encode(row) for row in a)
def tck(s):
    return (np.asarray(s['knots']),np.pad(s['coefficients'],(0,len(s['knots'])-len(s['coefficients']))),s['degree'])
def surface(s):return (np.asarray(s['x_knots']),np.asarray(s['y_knots']),np.asarray(s['coefficients']),s['x_degree'],s['y_degree'])
def fit_result(s,q):return s(q),s.t,s.k
def spline_result(s,q):
    if not isinstance(s,p.BSpline):s=p.BSpline(*s)
    return s(q),s.t,s.c[:len(s.t)-s.k-1]
cases=read_cases(root/'tests/scipy/interpolation_cases.jsonl')
with (root/'tests/fixtures/interpolation_inputs.csv').open('w',newline='') as fi,(root/'tests/fixtures/interpolation_expected.csv').open('w',newline='') as fe:
    iw=csv.writer(fi,lineterminator='\n');ew=csv.writer(fe,lineterminator='\n')
    iw.writerow(['id','api',*list('abcdefgh')]);ew.writerow(['id','first','second','third'])
    for index,case in enumerate(cases):
        api=case['api'];a=case['args'];iw.writerow([index,api,*map(encode,a),*['']*(8-len(a))])
        if api=='make_interp_spline':r=fit_result(p.make_interp_spline(a[0],a[1],k=a[2]),a[3])
        elif api=='make_lsq_spline':r=fit_result(p.make_lsq_spline(a[0],a[1],a[2],k=a[3],w=a[4]),a[5])
        elif api in ['make_splrep','splrep']:
            if api=='make_splrep':s=p.make_splrep(a[0],a[1],k=a[2],s=a[3],w=a[4])
            else:s=p.BSpline(*p.splrep(a[0],a[1],k=a[2],s=a[3],w=a[4]))
            r=fit_result(s,a[5])
        elif api=='make_smoothing_spline':r=fit_result(p.make_smoothing_spline(a[0],a[1],w=a[2],lam=a[3]),a[4])
        elif api=='approximate_taylor_polynomial':
            f={'square':lambda x:x*x,'exp':np.exp,'sin':np.sin,'constant':lambda x:np.ones_like(x)}[a[0]]
            c=p.approximate_taylor_polynomial(f,a[1],a[2],a[3],a[4]).c
            r=(np.pad(c,(a[2]+1-len(c),0)),None,None)
        elif api=='splev':
            if case.get('reference')=='nondegenerate_ppoly':
                # Sci's endpoint extension uses the first/last nonempty base
                # polynomial span, including for unclamped repeated-end knots.
                pp=p.PPoly.from_spline(tck(a[1]))
                k=a[1]['degree'];t=np.asarray(a[1]['knots']);lo=t[k];hi=t[-k-1]
                spans=np.flatnonzero((np.diff(pp.x)>0)&(pp.x[:-1]>=lo)&(pp.x[1:]<=hi))
                pp=p.PPoly(pp.c[:,spans],np.r_[pp.x[spans],hi])
                q=np.asarray(a[0]);q=np.clip(q,lo,hi) if a[3]==3 else q
                r=(pp(q,nu=a[2]),None,None)
            else:r=(p.splev(a[0],tck(a[1]),der=a[2],ext=a[3]),None,None)
        elif api=='splder':r=spline_result(p.splder(tck(a[0]),a[1]),a[2])
        elif api=='splantider':r=spline_result(p.splantider(tck(a[0]),a[1]),a[2])
        elif api=='spalde':r=(p.spalde(a[0],tck(a[1])),None,None)
        elif api=='splint':r=(p.splint(a[0],a[1],tck(a[2])),None,None)
        elif api=='insert':r=spline_result(p.insert(a[0],tck(a[1]),m=a[2]),a[3])
        elif api=='sproot':r=(p.sproot(tck(a[0]),mest=512),None,None)
        elif api in ['make_splprep','splprep']:
            if api=='make_splprep':s,u=p.make_splprep(np.asarray(a[0]).T,u=a[1],k=a[2],s=a[3],w=a[4]);v=np.asarray(s(a[5])).T;t=s.t
            else:
                s,u=p.splprep(np.asarray(a[0]).T,u=a[1],k=a[2],s=a[3],w=a[4]);v=np.asarray(p.splev(a[5],s)).T;t=s[0]
            r=(v,u,t)
        elif api=='generate_knots':r=(list(p.generate_knots(a[0],a[1],k=a[2],s=a[3],w=a[4])),None,None)
        elif api=='interpn':r=(p.interpn(a[0],np.asarray(a[1]).reshape(tuple(len(x) for x in a[0])),a[2],method=a[3],bounds_error=a[4],fill_value=a[5]),None,None)
        elif api=='griddata':r=(p.griddata(np.asarray(a[0]),np.asarray(a[1]),np.asarray(a[2]),method=a[3],fill_value=a[4],rescale=a[5]).ravel(),None,None)
        elif api=='bisplrep':
            s=p.LSQBivariateSpline(a[0],a[1],a[2],a[3][a[5]+1:-a[5]-1],a[4][a[6]+1:-a[6]-1],kx=a[5],ky=a[6],w=a[7],bbox=[a[3][a[5]],a[3][-a[5]-1],a[4][a[6]],a[4][-a[6]-1]])
            r=(*s.get_knots(),s.get_coeffs())
        elif api=='bisplev':r=(np.asarray(p.bisplev(a[0],a[1],surface(a[2]),dx=a[3],dy=a[4])).reshape(len(a[0]),len(a[1])),None,None)
        else:raise ValueError(api)
        ew.writerow([index,*map(encode,r)])
