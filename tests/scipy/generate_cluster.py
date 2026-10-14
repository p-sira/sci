# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Generate cluster fixtures from input-only cases with pinned SciPy.

Run: uv run tests/scipy/generate_cluster.py
Numerical comparisons use 3e-12 absolute + relative tolerance for f64 linkage
recurrences and local moments. Tie trees are tested by independent properties;
label partitions for maximum-count criteria permit arbitrary label bijections.
"""
import csv
import json
from pathlib import Path
import warnings
import numpy as np
import scipy
from scipy.cluster import hierarchy as h
assert (np.__version__,scipy.__version__)==('2.5.4','1.18.1')
ROOT=Path(__file__).resolve().parents[1]
def encode(x):
    if x is None:return ''
    a=np.asarray(x)
    if a.ndim==2:return ':'.join(';'.join(str(v) for v in row) for row in a)
    return ';'.join(str(v) for v in a.ravel())
cases=json.loads(Path(__file__).with_name('cluster_cases.json').read_text())
with (ROOT/'fixtures/cluster_inputs.csv').open('w',newline='') as fi,(ROOT/'fixtures/cluster_expected.csv').open('w',newline='') as fe:
    iw=csv.writer(fi,lineterminator='\n');ew=csv.writer(fe,lineterminator='\n')
    iw.writerow(['id','api','z','y','depth','column','threshold','criterion','a','b','counts','heights','x','method'])
    ew.writerow(['id','rows','columns','values'])
    for c in cases:
        name=c['api'];z=np.array(c.get('z',[]),dtype=float).reshape(-1,len(c.get('z',[[0,0,0,0]])[0]) if c.get('z') else 4)
        r=np.array(c['r'],dtype=float).reshape(-1,4) if 'r' in c else h.inconsistent(z,c.get('depth',2)) if z.size and name in ('is_valid_im','maxRstat','maxinconsts') else None
        source=r if name=='is_valid_im' else z
        iw.writerow([c['id'],name,encode(source),encode(c.get('y')),c.get('depth',2),c.get('column',3),c.get('threshold',''),c.get('criterion',''),encode(c.get('labels',c.get('a'))),encode(c.get('b')),encode(c.get('counts')),encode(c.get('heights')),encode(c.get('x')),c.get('method','')])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if name in ('single','complete','average','weighted','centroid','median','ward'):v=getattr(h,name)(c['y'])
            elif name=='linkage':v=h.linkage(c['y'],method=c['method'])
            elif name in ('num_obs_linkage','is_valid_linkage','is_monotonic','leaves_list','maxdists','to_mlab_linkage','from_mlab_linkage'):v=getattr(h,name)(z)
            elif name=='to_tree':
                _,nodes=h.to_tree(z,rd=True);v=[[n.id,n.left.id if n.left else -1,n.right.id if n.right else -1,n.dist,n.count] for n in nodes]
            elif name=='cophenet':corr,d=h.cophenet(z,c['y']);v=np.r_[corr,d]
            elif name=='correspond':v=h.correspond(z,c['y'])
            elif name=='inconsistent':v=h.inconsistent(z,c['depth'])
            elif name=='is_valid_im':v=h.is_valid_im(r)
            elif name=='maxRstat':v=h.maxRstat(z,h.inconsistent(z,c['depth']),c['column'])
            elif name=='maxinconsts':v=h.maxinconsts(z,r)
            elif name=='is_isomorphic':v=h.is_isomorphic(np.array(c['a'],dtype=int),np.array(c['b'],dtype=int))
            elif name=='fcluster':v=h.fcluster(z,c['threshold'],criterion=c['criterion'],depth=c['depth'],monocrit=h.maxdists(z))
            elif name=='fclusterdata':v=h.fclusterdata(c['x'],c['threshold'],criterion=c['criterion'],method=c['method'])
            elif name=='leaders':
                nodes,ids=h.leaders(z,np.array(c['labels'],dtype=np.int32));order=np.argsort(nodes);v=np.c_[nodes[order],ids[order]]
            elif name=='cut_tree':v=h.cut_tree(z,n_clusters=c.get('counts'),height=c.get('heights'))
            else:raise ValueError(name)
        a=np.atleast_2d(np.asarray(v,dtype=float));ew.writerow([c['id'],*a.shape,encode(a.ravel())])
