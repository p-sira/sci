# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Generate independent public linalg references from input-only cases.

Matrix factors with nonunique bases are compared through projectors or
reconstruction, plus independent Rust structural/property tests. No Sci
or nalgebra output is used. Run with uv or the project's pinned .venv.
"""
import csv
from pathlib import Path
from case_catalog import read_cases

import numpy as np
import scipy
from scipy import linalg as la

assert (np.__version__, scipy.__version__) == ("2.5.4", "1.18.1")
ROOT = Path(__file__).resolve().parents[2]


def encode(v):
    if v is None:
        return "none"
    if isinstance(v, str):
        return v
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v)).lower()
    if isinstance(v, list) and v and isinstance(v[0], list) and v[0] and isinstance(v[0][0], list):
        return "|".join(encode(x) for x in v)
    a = np.asarray(v)
    if a.ndim == 0:
        return str(a.item())
    if a.ndim == 1:
        return ";".join(str(x) for x in a)
    return ":".join(encode(row) for row in a)


def spectral(result):
    w, v = result
    return [w, v*v]


def evaluate(api, args):
    if api.startswith("interpolative."):
        from scipy.linalg import interpolative as ip
        name=api.split(".")[1]
        if name=="estimate_rank":
            # SciPy randomized estimate is an overestimate. SVD gives the
            # documented deterministic numerical-rank adaptation.
            w=la.svdvals(args[0]);return [int(np.sum(w>args[1]*w[0]))]
        if name=="estimate_spectral_norm":return [la.norm(args[0],2)]
        if name=="estimate_spectral_norm_diff":return [la.norm(np.asarray(args[0])-args[1],2)]
        if name=="interp_decomp":
            a=np.asarray(args[0]);k=args[1]
            if k==0 or not np.any(a):return [np.zeros_like(a),k]
            idx,proj=ip.interp_decomp(a,k,rand=False)
            return [ip.reconstruct_matrix_from_id(ip.reconstruct_skel_matrix(a,k,idx),idx,proj),k]
        if name=="svd":
            a=np.asarray(args[0]);k=args[1]
            if k==0:return [np.zeros_like(a),[],np.zeros((len(a),len(a))),np.zeros((a.shape[1],a.shape[1]))]
            u,w,vh=la.svd(a,full_matrices=False);u,w,vh=u[:,:k],w[:k],vh[:k,:];return [(u*w)@vh,w,u@u.T,vh.T@vh]
        if name=="id_to_svd":
            b,idx,proj=map(np.asarray,args);u,w,v=ip.id_to_svd(b,idx,proj);return [(u*w)@v.T,w]
        if name=="reconstruct_skel_matrix":return [ip.reconstruct_skel_matrix(np.asarray(args[0]),args[1],np.asarray(args[2]))]
        if name=="reconstruct_interp_matrix":
            idx,proj=args
            if not proj:return [np.empty((0,len(idx)))]
            return [ip.reconstruct_interp_matrix(np.asarray(idx),np.asarray(proj))]
        if name=="reconstruct_matrix_from_id":
            b,idx,proj=args
            if not proj:return [np.zeros((len(b),len(idx)))]
            return [ip.reconstruct_matrix_from_id(np.asarray(b),np.asarray(idx),np.asarray(proj))]
    if api=="find_best_blas_type":
        prefix,dtype,fortran=la.find_best_blas_type(tuple(np.asarray(a,dtype=float) for a in args[0]));return [prefix,str(dtype),fortran]
    if api=="get_blas_funcs":
        name=args[0].removeprefix('d') if args[0]!='dot' else 'dot'
        arrays=[np.asarray(a,dtype=float) for a in args[1:]]
        f=la.get_blas_funcs(name,arrays=arrays,dtype=np.float64)
        if name in {'gemv','gemm'}:return [f(1.,*arrays)]
        return [f(*arrays)]
    if api=="get_lapack_funcs":
        name=args[0].removeprefix('d');a=np.asarray(args[1],dtype=float)
        f=la.get_lapack_funcs(name,(a,))
        if name=='gesv':return [f(a,np.asarray(args[2],dtype=float))[2]]
        lu,piv,info=f(a);return [lu]
    if api=="eig":
        w,v=la.eig(args[0]);order=sorted(range(len(w)),key=lambda i:(w[i].real,w[i].imag));return [w[order].real,w[order].imag,np.abs(v[:,order])**2]
    if api=="ldl":
        a=np.asarray(args[0]);outer,d,perm=la.ldl(a,lower=args[1]);return [outer@d@outer.T]
    if api in {"sqrtm","logm","signm","fractional_matrix_power","funm"}:
        a=np.asarray(args[0])
        if api=='funm':v=la.funm(a,np.exp if args[1]=='exp' else np.sin)
        else:v=getattr(la,api)(a,*args[1:])
        return [v.real,v.imag]
    if api=="rsf2csf":
        t,z=la.rsf2csf(*args);v=z@t@z.conj().T;return [v.real,v.imag]
    if api=="cdf2rdf":
        w=np.asarray(args[0]);v=np.asarray(args[1]);return list(la.cdf2rdf(w[0]+1j*w[1],v[0]+1j*v[1]))
    if api in {"qz","ordqz"}:
        a,b=map(np.asarray,args[:2])
        if api=='ordqz':s,t,alpha,beta,q,z=la.ordqz(a,b,sort=args[2],output='complex')
        else:s,t,q,z=la.qz(a,b,output='complex');alpha,beta=np.diag(s),np.diag(t)
        mask=np.abs(beta)>0;w=sorted(alpha[mask]/beta[mask],key=lambda z:(z.real,z.imag))
        # Infinite eigenvalues are checked through homogeneous values in Rust.
        return [(q@s@z.conj().T).real,(q@t@z.conj().T).real,np.real(w),np.imag(w)]
    if api=='matrix_balance':
        a=np.asarray(args[0]);b,t=la.matrix_balance(a,permute=args[1],scale=args[2]);w=sorted(la.eigvals(b),key=lambda z:(z.real,z.imag));return [t@b@la.inv(t),np.real(w),np.imag(w)]
    if api=='clarkson_woodruff_transform':return [la.clarkson_woodruff_transform(np.asarray(args[0]),args[1],rng=args[4])]
    if api=='cossin':
        a,p,q=args;u,theta,vh=la.cossin(a,p=p,q=q,separate=True);return [theta]
    x = args[0]
    if api in {"lu", "lu_factor", "lu_solve"}:
        a = np.array(x, dtype=float)
        if api == "lu":
            return list(la.lu(a))
        packed, piv = la.lu_factor(a)
        if api == "lu_factor":
            perm = np.arange(len(a))
            for i, j in enumerate(piv):
                perm[i], perm[j] = perm[j], perm[i]
            return [packed, perm]
        return [la.lu_solve((packed, piv), args[1], trans=int(args[2]))]
    if api == "rq":
        r,q=la.rq(x,mode="economic")
        return [r@r.T,q.T@q,r@q]
    if api == "qr_multiply":
        product,r=la.qr_multiply(x,args[1],mode="left" if args[2] else "right")
        return [product@product.T,r.T@r]
    if api in {"qr_update","qr_insert","qr_delete"}:
        q,r=np.asarray(x,dtype=float),np.asarray(args[1],dtype=float)
        if api=="qr_update":q,r=la.qr_update(q,r,np.asarray(args[2],dtype=float),np.asarray(args[3],dtype=float))
        elif api=="qr_insert":q,r=la.qr_insert(q,r,np.asarray(args[2],dtype=float),args[3],which="row" if args[4] else "col")
        else:q,r=la.qr_delete(q,r,args[2],p=args[3],which="row" if args[4] else "col")
        return [q@r]
    if api == "qr":
        q, r = la.qr(x, mode="economic")
        return [q@q.T, r.T@r, q@r]
    if api == "svd":
        u, s, vh = la.svd(x, full_matrices=False)
        return [u@u.T, s, vh.T@vh, (u*s)@vh]
    if api in {"orth", "null_space"}:
        q = getattr(la, api)(x, rcond=args[1])
        return [q@q.T, q.shape[1]]
    if api in {"hessenberg", "schur"}:
        t, z = la.hessenberg(x, calc_q=True) if api == "hessenberg" else la.schur(x, output="real")
        return [z@t@z.T, z.T@z]
    if api == "eigvals":
        w = sorted(la.eigvals(x), key=lambda z: (z.real,z.imag))
        return [np.real(w),np.imag(w)]
    if api in {"cholesky", "cho_factor", "cho_solve"}:
        lower = args[-1]
        if api != "cho_solve":
            # cho_factor deliberately leaves the unused triangle unspecified.
            return [la.cholesky(x,lower=lower)]
        return [la.cho_solve(la.cho_factor(np.asarray(x,dtype=float),lower=lower),args[1])]
    if api == "solve_triangular":
        return [la.solve_triangular(x,args[1],lower=args[2],unit_diagonal=args[3],trans=int(args[4]))]
    if api in {"solve_toeplitz", "matmul_toeplitz"}:
        cr = x if args[1] is None else (x,args[1])
        return [getattr(la,api)(cr,args[2])]
    if api == "pinv":
        return list(la.pinv(x,atol=args[1],rtol=args[2],return_rank=True))
    if api == "pinvh":
        return list(la.pinvh(x,atol=args[1],rtol=args[2],lower=args[3],return_rank=True))
    if api == "lstsq":
        solution, residuals, rank, s = la.lstsq(np.asarray(x,dtype=float),np.asarray(args[1],dtype=float),cond=args[2])
        return [solution,np.atleast_1d(residuals),rank,s]
    if api == "polar":
        u,p=la.polar(x,side="right" if args[1] else "left")
        # Rank-deficient U has a nonunique orthogonal extension. Compare its
        # orthonormal Gram matrix, the unique P, and A=UP/PU in Rust.
        return [u.T@u if u.shape[0]>=u.shape[1] else u@u.T,p]
    if api in {"eigh", "eigvalsh"}:
        r = getattr(la,api)(x,lower=args[1])
        return spectral(r) if api == "eigh" else [r]
    if api == "eigh_tridiagonal":
        return spectral(la.eigh_tridiagonal(*args))
    if api == "solve_banded":
        return [la.solve_banded(tuple(args[:2]),args[2],args[3])]
    if api in {"solveh_banded", "cho_solve_banded"}:
        ab,b,lower=args
        if api == "cho_solve_banded":
            return [la.cho_solve_banded((la.cholesky_banded(ab,lower=lower),lower),b)]
        return [la.solveh_banded(ab,b,lower=lower)]
    if api in {"cholesky_banded", "eig_banded", "eigvals_banded"}:
        ab,lower=args
        r=getattr(la,api)(ab,lower=lower)
        if api == "eig_banded":
            return spectral(r)
        if api == "cholesky_banded":
            r=r.copy()
            # Match deterministic zero padding in the owned Rust factor.
            k,n=r.shape[0]-1,r.shape[1]
            for i in range(k+1):
                for j in range(n):
                    if (lower and i+j>=n) or (not lower and i+j<k):r[i,j]=0
        return [r]
    if api == "block_diag":
        return [la.block_diag(*x) if x else np.empty((0,0))]
    if api in {"issymmetric", "ishermitian", "bandwidth"}:
        x=np.asarray(x,dtype=float).reshape((0,0)) if not x else np.asarray(x,dtype=float)
        r=getattr(la,api)(x,*args[1:])
        return list(r) if isinstance(r,tuple) else [r]
    if api == "dft":
        r=la.dft(x,scale=None if args[1]=="none" else args[1])
        return [r.real,r.imag]
    if api == "norm":
        order={"fro":"fro","1":1,"-1":-1,"inf":np.inf,"-inf":-np.inf,"2":2,"-2":-2,"nuc":"nuc"}[args[1]]
        # SciPy's Frobenius array path can overflow for 1e200; independently
        # evaluate the same norm with its stable vector BLAS path.
        if order=="fro":return [la.norm(np.asarray(x,dtype=float).ravel())]
        return [la.norm(x,ord=order)]
    r=getattr(la,api)(*args)
    return list(r) if isinstance(r,tuple) else [r]


cases=read_cases(ROOT/"tests/scipy/linalg_cases.jsonl")
with (ROOT/"tests/fixtures/linalg_inputs.csv").open("w",newline="") as inf, (ROOT/"tests/fixtures/linalg_expected.csv").open("w",newline="") as outf:
    iw,ew=csv.writer(inf,lineterminator="\n"),csv.writer(outf,lineterminator="\n")
    iw.writerow(["id","api","a","b","c","d","e"])
    ew.writerow(["id","first","second","third","fourth"])
    for i,c in enumerate(cases):
        args=c["args"]
        iw.writerow([i,c["api"],*[encode(a) for a in args],*[""]*(5-len(args))])
        r=evaluate(c["api"],args)
        for v in r:
            values=np.asarray(v)
            if values.dtype.kind in "fci":
                assert np.all(np.isfinite(values)),(i,c["api"],v)
        ew.writerow([i,*[encode(v) for v in r],*[""]*(4-len(r))])
print(f"Generated {len(cases)} reference cases across {len({c['api'] for c in cases})} APIs.")
