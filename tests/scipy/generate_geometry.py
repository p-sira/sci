# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Generate geometry references from separate geometry_inputs.csv cases.

Run: uv run tests/scipy/generate_geometry.py (or the pinned .venv).
References use SciPy alone. Rust comparisons use 3e-12 absolute and relative
error for f64 reductions/SVD and near-zero cancellation; no approximate search
is involved. Hausdorff witnesses are checked geometrically because ties and
random traversal can choose different indices. tsearch fixtures use a single
simplex, so SciPy's Delaunay cell numbering is unambiguous.
"""
import csv
import warnings
from pathlib import Path
import numpy as np
import scipy
from scipy import spatial
from scipy.spatial import distance
if (np.__version__, scipy.__version__) != ("2.5.4", "1.18.1"):
    raise RuntimeError("Use NumPy 2.5.4 and SciPy 1.18.1")
ROOT = Path(__file__).resolve().parents[1] / "fixtures"
def vector(s):
    return np.array([float(x) for x in s.split(';')]) if s else np.array([])
def matrix(s):
    return np.array([vector(row) for row in s.split(':')]) if s else np.empty((0,0))
with (ROOT / 'geometry_inputs.csv').open() as source, (ROOT / 'geometry_expected.csv').open('w',newline='') as target:
    writer=csv.writer(target,lineterminator='\n')
    writer.writerow(['id','rows','columns','values'])
    for c in csv.DictReader(source):
        name=c['api']; u=vector(c['u']) if ':' not in c['u'] else matrix(c['u'])
        v=vector(c['v']) if ':' not in c['v'] else matrix(c['v'])
        kw={}
        if c['w']:kw['w']=vector(c['w'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if name in ('pdist','cdist'):
                metric=c['param']
                if metric=='minkowski':kw['p']=float(c['extra'])
                if metric=='seuclidean':kw['V']=vector(c['extra'])
                if metric=='mahalanobis':kw['VI']=matrix(c['extra'])
                result=distance.pdist(matrix(c['u']),metric,**kw) if name=='pdist' else distance.cdist(matrix(c['u']),matrix(c['v']),metric,**kw)
            elif name=='distance_matrix':result=spatial.distance_matrix(matrix(c['u']),matrix(c['v']),float(c['param']))
            elif name.startswith('minkowski_distance'):result=getattr(spatial,name)(u,v,float(c['param']))
            elif name=='minkowski':result=distance.minkowski(u,v,float(c['param']),**kw)
            elif name=='correlation':result=distance.correlation(u,v,centered=c['extra']=='true',**kw)
            elif name=='jensenshannon':result=distance.jensenshannon(u,v,base=float(c['param']) if c['param'] else None)
            elif name=='seuclidean':result=distance.seuclidean(u,v,vector(c['extra']))
            elif name=='mahalanobis':result=distance.mahalanobis(u,v,matrix(c['extra']))
            elif name in ('is_valid_y','num_obs_y'):result=getattr(distance,name)(u)
            elif name=='is_valid_dm':result=distance.is_valid_dm(matrix(c['u']),tol=float(c['param']))
            elif name=='num_obs_dm':result=distance.num_obs_dm(matrix(c['u']))
            elif name=='squareform':result=distance.squareform(vector(c['u']) if c['param']=='condensed' else matrix(c['u']))
            elif name=='directed_hausdorff':result=distance.directed_hausdorff(matrix(c['u']),matrix(c['v']),rng=0)
            elif name=='geometric_slerp':result=spatial.geometric_slerp(u,v,vector(c['w']),tol=float(c['param']))
            elif name=='procrustes':
                a,b,d=spatial.procrustes(matrix(c['u']),matrix(c['v']))
                result=np.r_[a.ravel(),b.ravel(),d]
            elif name=='tsearch':result=spatial.Delaunay(matrix(c['u'])).find_simplex(matrix(c['v']),tol=float(c['param']))
            else:result=getattr(distance,name)(u,v,**kw)
        result=np.atleast_2d(np.asarray(result,dtype=float))
        writer.writerow([c['id'],*result.shape,';'.join(str(x) for x in result.ravel())])
