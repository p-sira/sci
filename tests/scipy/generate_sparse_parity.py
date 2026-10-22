# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Independent references for modern sparse API additions.

Run: uv run tests/scipy/generate_sparse_parity.py
Input-only sparse_parity_cases.json is maintained separately. Eigenvectors,
matching ties and RCM permutations are validated by invariant tests rather than
backend-dependent signs/ties. Iterative solution references use SciPy's named
methods; tightly converged solutions use 3e-9 tolerances. Spectral/matrix function
references use 3e-11; exact shapes, graph distances and structural outputs use
3e-12 or exact integer comparisons. RNG streams differ; only density, domain,
shape and statistical properties are compared, never seed-specific values.
"""
import csv
import json
from pathlib import Path
import numpy as np
import scipy
from scipy import sparse as s
from scipy.sparse import csgraph as g, linalg as l
assert (np.__version__,scipy.__version__)==('2.5.4','1.18.1')
ROOT=Path(__file__).resolve().parents[1]
F=ROOT/'fixtures'
def enc(x):return ';'.join(str(v) for v in np.asarray(x).ravel())
def triples(a):
 c=a.tocoo();return ':'.join(f'{r};{col};{v}' for r,col,v in zip(c.row,c.col,c.data))
def label_order(labels):
 groups=sorted(set(labels),key=lambda k:np.flatnonzero(labels==k)[0]);return np.array([groups.index(v) for v in labels])
with (F/'sparse_parity_inputs.csv').open('w',newline='') as fi,(F/'sparse_parity_expected.csv').open('w',newline='') as fe:
 iw=csv.writer(fi,lineterminator='\n');ew=csv.writer(fe,lineterminator='\n')
 iw.writerow(['id','api','rows','cols','a','p','x','zero_edge']);ew.writerow(['id','key','rows','cols','values'])
 def emit(c,key,x):
  if s.issparse(x):x=x.toarray()
  x=np.atleast_2d(np.asarray(x,dtype=float));ew.writerow([c['id'],key,*x.shape,enc(x)])
 for c in json.loads(Path(__file__).with_name('sparse_parity_cases.json').read_text()):
  api=c['api'];a=s.csr_array(np.array(c['a'],dtype=float));p=c['p'];x=c['x'];n=a.shape[0]
  if c.get('zero_edge'):a=s.csr_array((np.append(a.data,0.),(np.append(a.tocoo().row,0),np.append(a.tocoo().col,1))),shape=a.shape)
  iw.writerow([c['id'],api,*a.shape,triples(a),enc(p),':'.join(enc(v) for v in x) if api=='spdiags' else enc(x),int(c.get('zero_edge',False))])
  if api in ('shortest_path','dijkstra','bellman_ford','floyd_warshall','johnson'):
   opts=dict(directed=bool(p[0]),unweighted=bool(p[1]),return_predecessors=True)
   if api=='floyd_warshall':d,pr=g.floyd_warshall(a,**opts);d=d[x] if x else d
   else:
    if x:opts['indices']=x
    if api=='dijkstra' and len(p)>2:opts.update(min_only=bool(p[2]),limit=p[3])
    out=getattr(g,api)(a,**opts);d=out[0]
   emit(c,'d',d)
  elif api in ('breadth_first_order','depth_first_order'):
   order,pred=getattr(g,api)(a,int(p[0]),directed=bool(p[1]));emit(c,'order',order);emit(c,'pred',pred)
  elif api in ('breadth_first_tree','depth_first_tree'):emit(c,'a',getattr(g,api)(a,int(p[0]),directed=bool(p[1])))
  elif api=='connected_components':
   count,labels=g.connected_components(a,directed=bool(p[0]),connection='strong' if p[1] else 'weak');emit(c,'count',count);emit(c,'labels',label_order(labels))
  elif api=='minimum_spanning_tree':emit(c,'weight',g.minimum_spanning_tree(a).sum())
  elif api=='reverse_cuthill_mckee':
   perm=g.reverse_cuthill_mckee(a,symmetric_mode=bool(p[0]));emit(c,'permutation',perm)
  elif api=='laplacian':
   v,diag=g.laplacian(a,normed=bool(p[0]),use_out_degree=bool(p[1]),symmetrized=bool(p[2]),return_diag=True);emit(c,'a',v);emit(c,'diag',diag)
  elif api=='maximum_bipartite_matching':emit(c,'count',np.sum(g.maximum_bipartite_matching(a,perm_type='row' if p[0] else 'column')!=-1))
  elif api=='structural_rank':emit(c,'count',g.structural_rank(a))
  elif api=='min_weight_full_bipartite_matching':
   rows,cols=g.min_weight_full_bipartite_matching(a,maximize=bool(p[0]));emit(c,'rows',rows);emit(c,'cols',cols)
  elif api=='maximum_flow':
   v=g.maximum_flow(a.astype(np.int64),int(p[0]),int(p[1]));emit(c,'value',v.flow_value);emit(c,'a',v.flow)
  elif api=='yen':emit(c,'d',g.yen(a,int(p[0]),int(p[1]),int(p[2]),directed=bool(p[3]),unweighted=bool(p[4])))
  elif api in ('csgraph_from_dense','csgraph_masked_from_dense'):
   v=getattr(g,api)(a.toarray(),null_value=p[0]);
   if isinstance(v,np.ma.MaskedArray):emit(c,'mask',np.ma.getmaskarray(v));emit(c,'a',v.filled(0.))
   else:emit(c,'a',v)
  elif api=='csgraph_from_masked':emit(c,'a',g.csgraph_from_masked(np.ma.array(a.toarray(),mask=a.toarray()==p[0])))
  elif api=='csgraph_to_dense':emit(c,'a',g.csgraph_to_dense(a,null_value=p[0]))
  elif api=='csgraph_to_masked':
   v=g.csgraph_to_masked(a);emit(c,'a',v.filled(0.));emit(c,'mask',np.ma.getmaskarray(v))
  elif api in ('reconstruct_path','construct_dist_matrix'):
   d,pr=g.shortest_path(a,directed=bool(p[0]),return_predecessors=True)
   if api=='reconstruct_path':emit(c,'pred',pr[0]);emit(c,'a',g.reconstruct_path(a,pr[0],directed=bool(p[0])))
   else:emit(c,'pred',pr);emit(c,'d',g.construct_dist_matrix(a,pr,directed=bool(p[0])))
  elif api in ('cg','bicg','bicgstab','cgs','qmr','tfqmr','gmres','lgmres','gcrotmk','minres'):
   opts=dict(rtol=1e-12,maxiter=512)
   if api!='minres':opts['atol']=1e-13
   if len(p)>1:
    opts.update(maxiter=int(p[2]),rtol=p[3])
    if api!='minres':opts['atol']=p[1]
   if p and api=='gmres':opts['restart']=int(p[0])
   if p and api=='lgmres':opts['inner_m']=int(p[0])
   if p and api=='gcrotmk':opts['m']=int(p[0])
   # SciPy MINRES can report false convergence under tiny uniform scaling.
   # Use its independent direct solve for the scale-invariance regressions.
   if api=='minres' and '_review_scale_' in c['id']:
    v=l.spsolve(a,np.array(x));info=0
   else:v,info=getattr(l,api)(a,np.array(x),**opts)
   assert info==0,(api,info);emit(c,'x',v)
  elif api in ('lsqr','lsmr'):
   opts=dict(atol=1e-13,btol=1e-13,damp=p[0] if p else 0.);emit(c,'x',getattr(l,api)(a,np.array(x),**opts)[0])
  elif api in ('eigs','eigsh','lobpcg'):
   # Full dense SciPy reference avoids ARPACK's small-dimension fallback restrictions.
   if api=='eigs':w=scipy.linalg.eigvals(a.toarray());idx=np.argsort(abs(w),kind='stable')[-int(p[0]):];w=w[idx];emit(c,'wre',w.real);emit(c,'wim',w.imag)
   else:
    w=scipy.linalg.eigvalsh(a.toarray());emit(c,'w',w[:int(p[0])] if len(p)>1 and not p[1] else w[-int(p[0]):])
  elif api=='svds':emit(c,'s',np.sort(scipy.linalg.svdvals(a.toarray())[:int(p[0])]))
  elif api=='expm':emit(c,'a',l.expm(a))
  elif api=='expm_multiply':emit(c,'a',np.array([l.expm_multiply(a*t,np.array(x)) for t in [0.,0.5,1.]]))
  elif api=='funm_multiply_krylov':
   # SciPy 1.18.1's restarted action fails for this n=2 case (empty projected slice).
   ref=scipy.linalg.funm(a.toarray(),np.exp)@np.array(x) if n==2 else l.funm_multiply_krylov(scipy.linalg.expm,a,np.array(x),rtol=1e-12)
   emit(c,'x',ref)
  elif api=='onenormest':emit(c,'value',scipy.linalg.norm(a.toarray(),1))
  elif api=='spilu':emit(c,'x',l.spilu(a.tocsc(),drop_tol=0.,fill_factor=20.).solve(np.array(x)))
  elif api=='aslinearoperator':emit(c,'x',l.aslinearoperator(a).matvec(np.array(x)));emit(c,'xt',l.aslinearoperator(a).rmatvec(np.array(x)))
  elif api=='norm_axis':
   if len(p)==2:emit(c,'x',l.norm(a,ord=float(p[0]),axis=int(p[1])))
   else:emit(c,'x',l.norm(a,ord=2,axis=0));emit(c,'xt',l.norm(a,ord=-1,axis=1))
  elif api=='spectral_norm':emit(c,'value',scipy.linalg.norm(a.toarray(),2))
  elif api in ('as_array','expand_dims','permute_dims','swapaxes'):
   v=s.coo_array(a)
   if api=='expand_dims':v=s.expand_dims(v,axis=int(p[0]))
   if api=='permute_dims':v=s.permute_dims(v,axes=tuple(map(int,p)))
   if api=='swapaxes':v=s.swapaxes(v,*map(int,p))
   emit(c,'shape',v.shape);emit(c,'data',v.toarray().ravel())
  elif api=='issparse':emit(c,'value',s.issparse(a if p[0] else a.toarray()))
  elif api=='get_index_dtype':emit(c,'value',np.dtype(s.get_index_dtype([np.array(x,dtype=np.int64)],check_contents=True)).itemsize*8)
  elif api=='safely_cast_index_arrays':
   idx,ptr=s.safely_cast_index_arrays(a,idx_dtype=np.int32 if p[0]==32 else np.int64);emit(c,'indices',idx);emit(c,'indptr',ptr)
  elif api in ('random','random_array'):
   v=s.random_array(a.shape,density=p[0],rng=np.random.default_rng(int(p[1])));emit(c,'nnz',v.nnz);emit(c,'shape',v.shape)
  elif api=='spdiags':emit(c,'a',s.spdiags(x,list(map(int,p)),*a.shape))
  elif api in ('load_npz','save_npz'):
   path=F/(c['id']+'.npz');s.save_npz(path,a,compressed=bool(p[0]));emit(c,'a',s.load_npz(path))
  else:raise ValueError(api)
# Reader compatibility fixtures exercise all five formats and numeric dtypes.
for format in ['csr','csc','coo','dia','bsr']:
 for dtype in [np.float64,np.int32]:
  a=getattr(s,format+'_array')(np.array([[1,0,2],[0,-3,0],[4,0,5]],dtype=dtype));s.save_npz(F/f'sparse_{format}_{np.dtype(dtype).name}.npz',a)

# Invalid archives test rejection before indexing, object loading, or decompression.
import zipfile
for case,arrays in {
 'bad_pointer':dict(format=np.array(b'csr'),shape=np.array([2,2]),data=np.array([1.]),indices=np.array([0]),indptr=np.array([0,2,1])),
 'negative_shape':dict(format=np.array(b'csr'),shape=np.array([-1,2]),data=np.array([]),indices=np.array([],dtype=int),indptr=np.array([0])),
 'object':dict(format=np.array(b'csr'),shape=np.array([1,1]),data=np.array([object()],dtype=object),indices=np.array([0]),indptr=np.array([0,1])),
 'complex':dict(format=np.array(b'csr'),shape=np.array([1,1]),data=np.array([1j]),indices=np.array([0]),indptr=np.array([0,1])),
 'nonfinite':dict(format=np.array(b'csr'),shape=np.array([1,1]),data=np.array([np.inf]),indices=np.array([0]),indptr=np.array([0,1])),
}.items():np.savez_compressed(F/f'sparse_invalid_{case}.npz',**arrays)
with zipfile.ZipFile(F/'sparse_invalid_oversize.npz','w',compression=zipfile.ZIP_DEFLATED) as z:
 # The central-directory size is validated before any payload is allocated/read.
 with z.open('payload.npy','w') as dest:
  for _ in range(65):dest.write(bytes(1024*1024))

with (F/'sparse_npz_expected.csv').open('w',newline='') as out:
 writer=csv.writer(out,lineterminator='\n');writer.writerow(['name','rows','cols','values'])
 for name,dtype in [('float32','f4'),('bool','?'),('int64','i8'),('big_float','>f8'),('big_int','>i4'),('unsigned','u8')]:
  values=np.array([[1,0,2],[0,3,0],[4,0,5]],dtype=np.dtype(dtype).newbyteorder('='))
  if name=='unsigned':values[0,0]=2**63+1
  a=s.csr_array(values);ref=a.toarray().astype(float)
  a.data=a.data.astype(dtype)
  filename='sparse_dtype_'+name;s.save_npz(F/(filename+'.npz'),a)
  writer.writerow([filename,*ref.shape,enc(ref)])
 data=np.array([[[1.,2.],[3.,4.]],[[4.,3.],[2.,1.]],[[1.,2.],[3.,4.]]])
 a=s.bsr_array((data,np.array([0,0,1]),np.array([0,2,3])),shape=(4,4))
 s.save_npz(F/'sparse_bsr_duplicate_blocks.npz',a)
 writer.writerow(['sparse_bsr_duplicate_blocks',4,4,enc(a.toarray())])

print('Generated',len(json.loads(Path(__file__).with_name('sparse_parity_cases.json').read_text())),'sparse parity cases.')
