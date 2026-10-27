# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Independent defining-series references for real 1F1 cancellation.

Inputs live in hyp1f1_cancellation_cases.json. Decimal precision exceeds a
conservative exponential-term bound by 100 digits; regenerate at 80 additional
digits and require identical binary64 rounding. This uses the original defining
series, never Taylor ODE continuation or Sci's backend. SciPy is recorded as an
additional comparison, since it too can lose digits at large parameters.
"""
import csv
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import scipy
from scipy import special


def defining_series(args, precision):
    with localcontext() as context:
        context.prec = precision
        a, b, x = map(Decimal.from_float, args)
        term = total = absolute = Decimal(1)
        threshold = Decimal(10)**(-precision+20)
        for n in range(20000):
            term *= (a+n)*x / ((b+n)*(n+1))
            total += term
            absolute += abs(term)
            if term == 0 or (n > abs(x)+abs(b)+2 and abs(term) < threshold*abs(total)):
                return float(total), float(absolute.log10()-abs(total).log10())
        raise RuntimeError(('reference work limit',args))


def generate():
    assert np.__version__ == '2.5.4'
    assert scipy.__version__ == '1.18.1'
    root = Path(__file__).resolve().parent
    cases = json.loads((root/'hyp1f1_cancellation_cases.json').read_text())
    with (root.parent/'fixtures/hyp1f1_cancellation.csv').open('w',newline='') as stream:
        writer = csv.writer(stream,lineterminator='\n')
        writer.writerow(['a','b','x','expected','scipy','log10_series_condition'])
        for args in cases:
            args = list(map(float,args))
            a,b,x = args
            precision = max(100, math.ceil((2*abs(x)+2*math.sqrt(abs(a*x)))/math.log(10))+100)
            expected, condition = defining_series(args,precision)
            checked,_ = defining_series(args,precision+80)
            assert expected == checked,(args,expected,checked)
            writer.writerow([*args,expected,special.hyp1f1(*args),condition])
    print(f'{len(cases)} independently precision-checked 1F1 cases')


if __name__ == '__main__':
    generate()
