# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1", "mpmath==1.3.0"]
# ///
"""Independent finite FFTLog references using dense DFTs at 100/160 digits.

Inputs and per-case tolerances live in fft_cases.json. This evaluates Hamilton's
FFTLog multiplier U_mu(q+i*eta) = 2^(q+i*eta) Gamma((mu+1+q+i*eta)/2)
/ Gamma((mu+1-q-i*eta)/2), including the specified grid phases, bias factors,
and real Nyquist projection. It calls neither Sci nor SciPy's FFTLog backend.
These references measure the discrete mapping, not continuous-integral error.

Reference: https://jila-amo.colorado.edu/~ajsh/FFTLog/index.html
Run: uv run tests/scipy/generate_fftlog_accuracy.py
"""
import csv
import json
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy

assert (np.__version__, scipy.__version__, mp.__version__) == ('2.5.4', '1.18.1', '1.3.0')
ROOT = Path(__file__).resolve().parents[1]


def reference(case, precision):
    with mp.workdps(precision):
        # Exact binary64 inputs, matching the public Rust API's parameters.
        x = list(map(mp.mpf, case['re']))
        n = len(x)
        dln, mu, offset, q = (mp.mpf(float(case[k])) for k in ('dln', 'mu', 'offset', 'bias'))
        center = mp.mpf(n - 1) / 2
        inverse = case['api'] == 'ifht'
        biased = [v * mp.exp(q * ((j - center) * dln + offset) if inverse
                            else -q * (j - center) * dln) for j, v in enumerate(x)]
        coefficients = []
        for k in range(n // 2 + 1):
            eta = 2 * mp.pi * k / (n * dln)
            z = q + mp.j * eta
            if k == 0:
                # Rising factorial gives the analytic continuation when both
                # real gamma factors have poles at an integer bias.
                u = mp.power(2, q) * mp.rf((mu + 1 - q) / 2, q)
            else:
                u = (mp.power(2, z) * mp.gamma((mu + 1 + z) / 2)
                     / mp.gamma((mu + 1 - z) / 2) * mp.exp(-mp.j * eta * offset))
            if n % 2 == 0 and k == n // 2:
                u = mp.re(u)
            coefficients.append(u)
        coefficients += [mp.conj(coefficients[n - k]) for k in range(len(coefficients), n)]
        spectrum = [mp.fsum(v * mp.exp(-2 * mp.pi * mp.j * j * k / n)
                            for j, v in enumerate(biased)) for k in range(n)]
        spectrum = [v / mp.conj(u) if inverse else v * u for v, u in zip(spectrum, coefficients)]
        transformed = [mp.re(mp.fsum(v * mp.exp(2 * mp.pi * mp.j * j * k / n)
                                    for k, v in enumerate(spectrum)) / n) for j in range(n)][::-1]
        return [v * mp.exp(q * (j - center) * dln if inverse
                           else -q * ((j - center) * dln + offset))
                for j, v in enumerate(transformed)]


with (ROOT / 'fixtures/fftlog_accuracy.csv').open('w', newline='') as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(['id', 're', 'atol', 'rtol'])
    cases = json.loads(Path(__file__).with_name('fft_cases.json').read_text())
    for case in cases:
        if 'accuracy_tolerance' not in case:
            continue
        low, high = reference(case, 100), reference(case, 160)
        with mp.workdps(170):
            assert max(abs(a - b) for a, b in zip(low, high)) < mp.mpf('1e-80') * max(1, *map(abs, high))
        atol, rtol = case['accuracy_tolerance']
        writer.writerow([case['id'], ';'.join(repr(float(v)) for v in high), atol, rtol])
