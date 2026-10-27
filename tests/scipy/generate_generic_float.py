# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""References for generic public APIs, with separately stored inputs.

Run with the pinned .venv or uv run. Round inputs to each target precision,
then compute independent NumPy/SciPy binary64 references. In particular, no Sci
or private Rust backend is used. Non-SciPy elliptic definitions use quadrature
(DLMF 19); finite differences use polynomial derivative identities. Binary64
references avoid conflating SciPy's float32 implementation error with Sci's.
The Rust comparison permits 5e-6 relative/absolute error for binary32, 3e-4
for binary32 finite-difference cancellation, 5e-12 for binary64, and O(h^2)
stencil error.
Extreme IEEE range and branch behavior are exercised separately in Rust.
"""
import csv
import re
from pathlib import Path
import numpy as np
import scipy
from scipy import constants, fft, fftpack, integrate, interpolate, special, stats

if (np.__version__, scipy.__version__) != ("2.5.4", "1.18.1"):
    raise RuntimeError("Use NumPy 2.5.4 and SciPy 1.18.1")
ROOT = Path(__file__).resolve().parents[1] / "fixtures"


def elliptic(name, x):
    direct = {"ellipk", "ellipe", "ellipkinc", "ellipeinc", "elliprf", "elliprd", "elliprg", "elliprj", "elliprc", "ellipkm1"}
    if name == "ellipkinc":
        return special.ellipkinc(*x)
    if name in direct:
        return getattr(special, name)(*x)
    if name == "ellipke":
        return [special.ellipk(x[0]), special.ellipe(x[0])]
    if name in ("ellippi", "ellippiinc", "ellippiinc_bulirsch", "ellipd", "ellipdinc"):
        if name == "ellippi":
            phi, n, m = np.pi / 2, *x
        elif name in ("ellippiinc", "ellippiinc_bulirsch"):
            phi, n, m = x
        elif name == "ellipd":
            phi, m = np.pi / 2, x[0]
        else:
            phi, m = x
        if name.startswith("ellipd"):
            f = lambda t: np.sin(t)**2 / np.sqrt(1 - m*np.sin(t)**2)
        else:
            f = lambda t: 1 / ((1 - n*np.sin(t)**2) * np.sqrt(1 - m*np.sin(t)**2))
        return integrate.quad(f, 0, phi, epsabs=2e-13, epsrel=2e-13)[0]
    if name in ("jacobi_zeta", "heuman_lambda"):
        phi, m = x
        def zeta(phi, m):
            return special.ellipeinc(phi, m) - special.ellipe(m)*special.ellipkinc(phi, m)/special.ellipk(m)
        if name == "jacobi_zeta":
            return zeta(phi, m)
        return special.ellipkinc(phi, 1-m)/special.ellipk(1-m) + 2*special.ellipk(m)*zeta(phi, 1-m)/np.pi
    # Bulirsch complete/incomplete definitions, DLMF 19.2/19.16.
    if name.startswith("cel"):
        kc = x[0]
        p, a, b = {"cel": lambda: x[1:], "cel1": lambda: (1.,1.,1.),
                   "cel2": lambda: (1.,*x[1:]), "cel3": lambda: (x[1],1.,1.)}[name]()
        upper = np.pi/2
    else:
        upper, kc = np.arctan(x[0]), x[1]
        p, a, b = {"el1": lambda: (1.,1.,1.), "el2": lambda: (1.,*x[2:]),
                   "el3": lambda: (x[2],1.,1.)}[name]()
    return integrate.quad(lambda t: (a*np.cos(t)**2+b*np.sin(t)**2) / ((np.cos(t)**2+p*np.sin(t)**2)*np.sqrt(np.cos(t)**2+kc*kc*np.sin(t)**2)), 0, upper, epsabs=2e-13, epsrel=2e-13)[0]


def reference(group, name, x, p):
    if group == "complex":
        z, b = complex(*x[:2]), complex(*x[2:])
        binary = dict(add=np.add, subtract=np.subtract, multiply=np.multiply, divide=np.divide, powc=np.power)
        unary = dict(negate=np.negative, conjugate=np.conjugate, reciprocal=np.reciprocal,
                     real=np.real, imag=np.imag, abs=np.abs, norm_sqr=lambda z:z.real*z.real+z.imag*z.imag,
                     arg=np.angle, to_polar=lambda z:[np.abs(z),np.angle(z)],
                     from_polar=lambda z:z.real*np.exp(1j*z.imag), ln=np.log,
                     asin=np.arcsin, acos=np.arccos, atan=np.arctan, asinh=np.arcsinh,
                     acosh=np.arccosh, atanh=np.arctanh, cbrt=lambda z:np.abs(z)**(1/3)*np.exp(1j*np.angle(z)/3),
                     powf=lambda z:z**b.real, powi=lambda z:z**3, powu=lambda z:z**3)
        if name in binary:
            return binary[name](z,b)
        return unary.get(name, getattr(np,name,None))(z)
    if group == "special":
        if name in ("logsumexp","softmax","log_softmax"):
            return getattr(special,name)(x)
        return elliptic(name,x)
    if group == "differentiate":
        a,b=x
        return dict(derivative=lambda:3*a*a, gradient=lambda:[2*a+b,a+2*b],
                    jacobian=lambda:[b,a,1.,1.], hessian=lambda:[2.,1.,1.,2.])[name]()
    if group == "integrate":
        if name == "fixed_quad":
            return integrate.fixed_quad(lambda t:t*t,0.,1.,n=4)[0]
        kwargs=dict(dx=p)
        if name.startswith("cumulative_"):
            kwargs["initial"]=0.
        return getattr(integrate,name)(x,**kwargs)
    if group == "interpolate":
        nodes=np.arange(4.)
        if name == "lagrange":
            coefficients = interpolate.lagrange(nodes,x).c
            return np.pad(coefficients, (4-len(coefficients),0))
        return getattr(interpolate,name)(nodes,x,[-0.5,0.,0.5,1.5,3.,3.5])
    if group == "stats":
        ddof=1
        if name == "mean":return np.mean(x)
        if name == "variance":return np.var(x,ddof=ddof)
        if name == "std":return np.std(x,ddof=ddof)
        if name == "quantile":return np.quantile(x,0.3)
        if name == "scoreatpercentile":return stats.scoreatpercentile(x,30.)
        if name == "median":return np.median(x)
        if name == "median_abs_deviation":return stats.median_abs_deviation(x,scale=2.)
        if name == "moment":return stats.moment(x,3)
        if name in ("sem","variation","gstd","zscore"):return getattr(stats,name)(x,ddof=ddof)
        if name == "zmap":return stats.zmap(x+1.,x,ddof=ddof)
        if name == "gzscore":return stats.gzscore(x,ddof=ddof)
        if name in ("skew","kurtosis"):return getattr(stats,name)(x,bias=False)
        if name == "trimboth":return np.sort(stats.trimboth(x,0.125))
        if name == "trim_mean":return stats.trim_mean(x,0.125)
        if name == "mode":
            m=stats.mode(x);return [m.mode,m.count]
        if name == "describe":
            s=stats.describe(x,ddof=ddof,bias=False)
            return [s.nobs,*s.minmax,s.mean,s.variance,s.skewness,s.kurtosis]
        return getattr(stats,name)(x)
    if group == "constants":
        if name == "convert_temperature":return constants.convert_temperature(x[0],"Celsius","Fahrenheit")
        return getattr(constants,name)(x[0])
    if group == "fft":
        z=x.astype(complex)+0.25j*x[::-1]
        if name in ("fft","ifft","fftshift","ifftshift"):return getattr(fft,name)(z)
        if name in ("rfft","ihfft"):return getattr(fft,name)(x)
        if name in ("irfft","hfft"):return getattr(fft,name)(z[:3],n=4)
        if name in ("fftn","ifftn","fft2","ifft2"):return getattr(fft,name)(z.reshape(2,2))
        if name in ("rfftn","ihfftn","rfft2","ihfft2"):return getattr(fft,name)(x.reshape(2,2))
        if name in ("irfftn","hfftn","irfft2","hfft2"):return getattr(fft,name)(z.reshape(2,2),s=(2,2))
        if name in ("fftfreq","rfftfreq"):return getattr(fft,name)(4,d=0.5)
        if name.startswith(("dct","idct","dst","idst")):
            data=x.reshape(2,2) if name.endswith("n") else x
            return getattr(fft,name)(data,type=int(p))
        if name in ("hilbert","ihilbert"):return getattr(fftpack,name)(x)
        if name == "diff":return fftpack.diff(x,order=1,period=2*np.pi)
        if name == "shift":return fftpack.shift(x,a=0.25,period=2*np.pi)
        if name in ("tilbert","itilbert"):return getattr(fftpack,name)(x,h=0.5,period=2*np.pi)
        return getattr(fftpack,name)(x,a=0.25,b=0.5,period=2*np.pi)
    raise ValueError((group,name))


def flatten(value):
    a=np.asarray(value).ravel()
    if np.iscomplexobj(a):
        a=np.column_stack((a.real,a.imag)).ravel()
    return ";".join(format(float(x),".17g") for x in a)


with (ROOT/"generic_float_inputs.csv").open() as source, (ROOT/"generic_float_expected.csv").open("w",newline="") as target:
    w=csv.writer(target,lineterminator="\n");w.writerow(["case","precision","values"])
    cases = list(csv.DictReader(source))
    # Count only tested generic Rust exports; aliases/Cython duplicates add no APIs.
    repository = ROOT.parents[1]
    expected_apis = set()
    for group in {row["group"] for row in cases}:
        for module in (repository / "crates" / f"sci-{group}" / "src").glob("*.rs"):
            expected_apis.update((group, name) for name in re.findall(r"pub fn (\w+)<T:", module.read_text()))
    actual_apis = {(row["group"], row["function"]) for row in cases}
    if actual_apis != expected_apis:
        raise RuntimeError(f"Precision coverage mismatch: {actual_apis ^ expected_apis}")
    for i,row in enumerate(cases):
        for precision,dtype in [(32,np.float32),(64,np.float64)]:
            x=np.asarray(row["values"].split(";"),dtype=dtype).astype(np.float64)
            p=float(dtype(row["parameter"])) if row["parameter"] else 0.
            with np.errstate(all="ignore"):
                result=reference(row["group"],row["function"],x,p)
            w.writerow([i,precision,flatten(result)])
