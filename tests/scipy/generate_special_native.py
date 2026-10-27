# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Native special APIs: independent defining sums/integrals and SciPy references.

Run with uv run tests/scipy/generate_special_native.py. Inputs are kept separately
in special_native_cases.json. Bessel ratios use 100-digit Decimal defining 0F1
series (or large-x asymptotic series), never XSF's Perron fraction. Inverses use
SciPy brentq with those references. Digamma inverses solve SciPy digamma in log x.
Scaled E1 uses its defining Laplace integral, avoiding an overflowing exp(x).
Complex functions use the corresponding SciPy complex overloads. Both precisions
quantize inputs before reference evaluation. No Sci or Rust backend is used.
"""
import csv
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import scipy
from scipy import integrate, optimize, special


def ratio(v, x):
    with localcontext() as ctx:
        ctx.prec = 100
        v, x = Decimal.from_float(v), Decimal.from_float(x)
        if x == 0:
            return Decimal(0)
        if v == Decimal('0.5'):
            e = (-2*x).exp()
            return (1-e)/(1+e)
        def series(order):
            total = term = Decimal(1)
            for k in range(1, 30000):
                if x <= 10000:
                    term *= x*x / (4*k*(order+k))
                else:
                    term *= -(4*order*order-(2*k-1)**2)/(8*k*x)
                total += term
                if abs(term) < abs(total)*Decimal('1e-95'):
                    return total
            raise RuntimeError(('series failed', v, x))
        if x <= 10000:
            return x/(2*v)*series(v)/series(v-1)
        return series(v)/series(v-1)


def reference(name, args):
    if name.startswith('complex.'):
        name = name.removeprefix('complex.')
        if name in {'jv','jve','yv','yve','iv','ive','kv','kve','hankel1','hankel1e','hankel2','hankel2e'}:
            result = getattr(special, name)(args[0], complex(*args[1:]))
        else:
            result = getattr(special, name)(complex(*args))
        return [part for z in np.asarray(result).reshape(-1) for part in (float(np.real(z)), float(np.imag(z)))]
    if name == 'log1pmx':
        with localcontext() as ctx:
            ctx.prec = 100
            x = Decimal.from_float(args[0])
            return [float((1+x).ln()-x) if abs(x) > Decimal('1e-40') else float(-x*x/2)]
    if name == 'scaled_exp1':
        x, = args
        if x == 0: return [0.0]
        # t=xu resolves the boundary layer at very small x.
        if x < 1:
            return [x*math.exp(x)*special.exp1(x)]
        return [integrate.quad(lambda t: math.exp(-t)/(1+t/x), 0, np.inf,
                               epsabs=2e-14, epsrel=2e-14)[0]]
    if name == 'gamma_ratio':
        a,b = args
        if a == int(a) and b == int(b):
            return [float(Decimal(math.factorial(int(a)-1))/Decimal(math.factorial(int(b)-1)))]
        return [math.exp(special.gammaln(a)-special.gammaln(b))]
    if name == 'digammainv':
        y, = args
        root = optimize.brentq(lambda t: special.digamma(math.exp(t))-y,
                               -744, min(709, max(1,y+1)), xtol=5e-14, rtol=1e-15)
        return [math.exp(root)]
    if name in {'iv_ratio', 'iv_ratio_c'}:
        with localcontext() as ctx:
            ctx.prec = 100
            r = ratio(*args)
            return [float(r if name == 'iv_ratio' else 1-r)]
    if name == 'iv_ratioinv':
        v, r = args
        if r == 0: return [0.0]
        if r == 1: return [math.inf]
        if v == .5: return [math.atanh(r)]
        with localcontext() as ctx:
            ctx.prec = 100
            target = Decimal.from_float(r)
            return [optimize.brentq(lambda x: float(ratio(v,x)-target), 0,
                                    2*v*r/((1-r)*(1+r)), xtol=1e-300, rtol=1e-14)]
    if name == 'normalized_gen_harmonic':
        j,k,n,a = args
        with localcontext() as ctx:
            ctx.prec = 100
            a = Decimal.from_float(a)
            # Independent defining powers in Decimal; no maximum-shift formula.
            terms = [(-a*Decimal(i).ln()).exp() for i in range(1,n+1)]
            return [float(sum(terms[j-1:k])/sum(terms))]
    raise ValueError(name)


def generate():
    assert np.__version__ == '2.5.4'
    assert scipy.__version__ == '1.18.1'
    root = Path(__file__).resolve().parent
    cases = json.loads((root/'special_native_cases.json').read_text())
    with (root.parent/'fixtures/special_native.csv').open('w', newline='') as output:
        writer = csv.writer(output,lineterminator='\n')
        writer.writerow(['precision','function','arguments','expected'])
        for dtype in (np.float32,np.float64):
            for case in cases:
                args = [int(x) if case['name']=='normalized_gen_harmonic' and i<3 else float(dtype(x))
                        for i,x in enumerate(case['args'])]
                with np.errstate(over="ignore"):
                    values = [float(dtype(x)) for x in reference(case['name'],args)]
                writer.writerow([dtype.__name__.replace('float','f'),case['name'],
                                 ';'.join(map(str,args)), ';'.join(map(str,values))])


if __name__ == '__main__':
    generate()
