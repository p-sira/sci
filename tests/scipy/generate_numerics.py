# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Generate integration, interpolation, and vector-quantization references.

Inputs are in numerics_cases.jsonl. CSV fields use semicolon-separated vectors
and colon-separated matrix rows. Expected results come exclusively from SciPy.
Extreme integration references normalize samples and spacing before SciPy
evaluation, then restore units in extended precision. Cancellation references
use SciPy trapezoids in longdouble. Romberg and fixed quadrature references
normalize sample/callback values and interval widths, then restore units in
longdouble to avoid overflowing intermediate reference arithmetic. Extreme interpolation cases may select an
independent SciPy interpolator with
the same mathematical result, or scale values before evaluation and restore
their units afterward, to avoid overflow in SciPy's reference implementation.
Run: uv run tests/scipy/generate_numerics.py, or use the pinned .venv.
"""
import csv
import warnings
from pathlib import Path
from case_catalog import read_cases
import numpy as np
import scipy
from scipy import integrate, interpolate
from scipy.cluster import vq

assert (np.__version__, scipy.__version__) == ("2.5.4", "1.18.1")
root = Path(__file__).resolve().parents[2]
functions = {"sin": np.sin,"exp": np.exp,"square": lambda x: x*x,"cubic": lambda x:x**3,"constant":lambda x:np.ones_like(x), "huge_constant":lambda x:np.full_like(x,1e308), "tiny_constant":lambda x:np.full_like(x,1e-308), "subnormal_constant":lambda x:np.full_like(x,5e-324), "scaled_square":lambda x:(x/1e308)**2*1e-308}
def encode(v):
    a = np.asarray(v)
    if a.ndim == 0: return str(a.item())
    if a.ndim == 1: return ";".join(str(x) for x in a)
    return ":".join(encode(row) for row in a)
cases = read_cases(root / "tests/scipy/numerics_cases.jsonl")
with (root / "tests/fixtures/numerics_inputs.csv").open("w",newline="") as inputs, (root / "tests/fixtures/numerics_expected.csv").open("w",newline="") as expected:
    iw = csv.writer(inputs,lineterminator="\n")
    ew = csv.writer(expected,lineterminator="\n")
    iw.writerow(["id","module","api","a","b","c","d","e"])
    ew.writerow(["id","first","second"])
    for index,case in enumerate(cases):
        module,api = case["module"],case["api"]
        args = case["args"]
        iw.writerow([case.get("id",index),module,api,*[encode(v) if v is not None else "" for v in args],*[""]*(5-len(args))])
        if module == "integrate":
            if case.get("reference") == "scaled_romb":
                values = np.asarray(args[0], dtype=np.longdouble)
                scale = np.max(np.abs(values))
                result = integrate.romb(values / scale, dx=1.)
                result = float(np.longdouble(result) * scale * np.longdouble(args[1]))
            elif case.get("reference") == "scaled_fixed_quad":
                name, a, b, order = args
                if name == "scaled_square":
                    result = integrate.fixed_quad(lambda x:x*x, np.longdouble(a)/np.longdouble(1e308), np.longdouble(b)/np.longdouble(1e308), n=order)[0]
                    result = float(np.longdouble(result)*np.longdouble(1e308)*np.longdouble(1e-308))
                else:
                    amplitude = {"constant":1., "huge_constant":1e308, "tiny_constant":1e-308, "subnormal_constant":5e-324}[name]
                    # Unit-interval SciPy quadrature plus constant homogeneity
                    # avoids overflow in SciPy's bounds and weighted sum.
                    result = integrate.fixed_quad(functions["constant"], 0., 1., n=order)[0]
                    result = float(np.longdouble(result)*np.longdouble(amplitude)*(np.longdouble(b)-np.longdouble(a)))
            elif case.get("reference") == "longdouble_trapezoid":
                # Extended precision retains the unit residual between 1e16
                # intervals. The cumulative outputs are rounded only afterward.
                if np.finfo(np.longdouble).nmant < 63:
                    raise RuntimeError("Cancellation references require extended precision")
                kwargs = dict(x=np.asarray(args[1],dtype=np.longdouble))
                if api.startswith("cumulative_"): kwargs["initial"] = args[3]
                result = np.asarray(getattr(integrate,api)(np.asarray(args[0],dtype=np.longdouble),**kwargs),dtype=float)
            elif case.get("reference") == "scaled_integral":
                # Homogeneity in samples and spacing gives an independent SciPy
                # reference without overflowing weighted sums or rounding tiny
                # samples/widths before the final integral is formed.
                values = np.asarray(args[0], dtype=np.longdouble)
                scale = np.max(np.abs(values))
                kwargs = dict(dx=1.)
                width_scale = np.longdouble(args[2])
                if args[1] is not None:
                    coordinates = np.asarray(args[1], dtype=np.longdouble)
                    width_scale = np.max(np.abs(np.diff(coordinates)))
                    kwargs["x"] = coordinates / width_scale
                if api.startswith("cumulative_"): kwargs["initial"] = 0.
                reference_api = api
                if np.all(values == values[0]):
                    # Both rules integrate constants exactly. On extreme
                    # irregular grids, SciPy Simpson's huge opposing weights
                    # lose even a constant's integral in extended precision.
                    reference_api = "cumulative_trapezoid" if api.startswith("cumulative_") else "trapezoid"
                result = getattr(integrate, reference_api)(values / scale, **kwargs)
                result = np.asarray(result, dtype=np.longdouble) * np.longdouble(scale) * width_scale
                if api.startswith("cumulative_") and args[3] is not None:
                    result += np.longdouble(args[3])
                result = np.asarray(result, dtype=float)
            elif api == "fixed_quad":
                result = integrate.fixed_quad(functions[args[0]],args[1],args[2],n=args[3])[0]
            elif api == "newton_cotes":
                result = integrate.newton_cotes(np.array(args[0],dtype=float))
            elif api == "romb":
                result = integrate.romb(args[0],dx=args[1])
            else:
                kwargs = dict(x=args[1],dx=args[2])
                if api.startswith("cumulative_"): kwargs["initial"]=args[3]
                result = getattr(integrate,api)(args[0],**kwargs)
        elif module == "interpolate":
            if api == "pade":
                p,q = interpolate.pade(*args)
                result = (np.pad(p.c,(args[2] + 1-len(p.c),0)) if args[2] is not None else p.c,q.c)
            elif api == "lagrange":
                p=interpolate.lagrange(*args)
                result=np.pad(p.c,(len(args[0])-len(p.c),0))
            else:
                reference = case.get("reference", api)
                reference_args = list(args)
                if "value_scale" in case:
                    reference_args[1] = np.asarray(args[1], dtype=float) / case["value_scale"]
                result=getattr(interpolate,reference)(*reference_args, **({"rng": np.random.default_rng(42)} if reference == "barycentric_interpolate" else {}))
                if "value_scale" in case:
                    result = np.asarray(result) * case["value_scale"]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",RuntimeWarning)
                if api == "kmeans2":
                    result=vq.kmeans2(np.array(args[0],dtype=float),np.array(args[1],dtype=float),iter=args[2],minit="matrix")
                elif api == "kmeans":
                    # SciPy 1.18.1 misclassifies a one-row matrix as scalar k.
                    # For one cluster, the final centroid is the data mean regardless of seed.
                    guess = 1 if len(args[1]) == 1 else np.array(args[1],dtype=float)
                    result=vq.kmeans(np.array(args[0],dtype=float),guess,thresh=args[2],rng=np.random.default_rng(42))
                elif api == "whiten": result=vq.whiten(np.array(args[0],dtype=float))
                else: result=getattr(vq,api)(np.array(args[0],dtype=float),np.array(args[1],dtype=float))
        if isinstance(result,tuple): ew.writerow([case.get("id",index),encode(result[0]),encode(result[1])])
        else: ew.writerow([case.get("id",index),encode(result),""])
