# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Generate independent SciPy oracles for the complete special-function facade.

Run with ``uv run tests/scipy/generate_special.py``. Each CSV record stores
the function, semicolon-separated arguments, and flattened real results.
The JSON contains input cases only; expected results always come from SciPy.
Real Lambert W comparisons use 8e-15 relative and zero absolute tolerance so
tiny arguments are checked proportionally; inverse identities separately
check the defining equation and real branch domains.
"""

import csv
import json
from pathlib import Path

import numpy as np
import scipy
from scipy import special, integrate, optimize


def stable_keip_zeros(count):
    """Independent bracketed roots of Im[-a K1(a*x)] via SciPy AMOS.

    Identity: https://dlmf.nist.gov/10.61.E2 (differentiate K0).
    SciPy keip_zeros uses a cancellation-sensitive series around x=9.4;
    its second zero differs by ~4e-10 from the complex K identity.
    Removing exp(-x/sqrt(2)) keeps the reference finite for long sequences.
    """
    a = np.exp(1j * np.pi / 4)
    def residual(x):
        return (-a * np.exp(-1j * x * a.imag) * special.kve(1, a*x)).imag
    return np.array([
        optimize.brentq(residual, 4.93 + i*np.pi*np.sqrt(2) - 0.5,
                        4.93 + i*np.pi*np.sqrt(2) + 0.5, xtol=5e-14)
        for i in range(count)
    ])


def reference(name, args):
    """SciPy evaluations or independent quadrature for ellip-only functions."""
    if name in {"ber", "bei", "ker", "kei", "berp", "beip", "kerp", "keip", "kelvin"} and args[0] >= 0.5:
        # DLMF 10.61: AMOS Bessel identities avoid cancellation in SciPy's
        # Kelvin power-series transition (notably bei' near x=12).
        x, = args
        rotation = np.exp(1j*np.pi/4)
        z = rotation*x
        values = (special.iv(0,z), special.kv(0,z),
                  rotation*special.iv(1,z), -rotation*special.kv(1,z))
        if name == "kelvin":
            return values
        index = {"ber":0,"bei":0,"ker":1,"kei":1,
                 "berp":2,"beip":2,"kerp":3,"keip":3}[name]
        return values[index].imag if name in {"bei","kei","beip","keip"} else values[index].real
    if name == "modstruve" and -args[0]-0.5 >= 0 and (-args[0]-0.5).is_integer():
        # DLMF 11.4.3: L_{-n-1/2}(x)=I_{n+1/2}(x). SciPy's direct
        # Struve dispatch loses accuracy around x=12 for this exact case.
        return special.iv(-args[0], args[1])
    if name == "modstruve" and args[0] > -0.5 and args[1] > 0:
        # DLMF 11.5.6: independent integral for L_v. The direct SciPy
        # Struve transition also loses accuracy at positive half orders.
        v, x = args
        factor = np.exp((v+1)*np.log(x/2)-special.gammaln(v+0.5))*4/np.sqrt(np.pi)
        value = integrate.quad(
            lambda t: np.sinh(x*t)/x*(1+t)**(v-0.5), 0, 1,
            weight="alg", wvar=(0, v-0.5), epsabs=2e-13, epsrel=2e-13,
        )[0]
        return factor*value
    if name in {"h1vp", "h2vp"}:
        # DLMF 10.4.3: H^(1,2)=J±iY. Evaluating the real derivatives
        # independently avoids AMOS's cross-component cancellation when
        # |Y| is many orders of magnitude larger than |J|.
        sign = 1 if name == "h1vp" else -1
        return special.jvp(*args) + sign*1j*special.yvp(*args)
    if name == "yve" and args[1] > 0:
        # The real-axis exponential factor is one. SciPy's scaled AMOS
        # overflow branch can return +inf where Y_v(x) tends to -inf.
        return special.yv(*args)
    if name == "it2i0k0" and args[0] > 0:
        x, = args
        # Independent defining integrals avoid SciPy 1.18's legacy large-x
        # transcription errors (e.g. K's tail at x=16 and I's at x=64).
        first = integrate.quad(lambda t: (special.i0(t)-1)/t, 0, x,
                               epsabs=2e-13, epsrel=2e-13)[0]
        second = integrate.quad(lambda t: special.k0(t)/t, x, np.inf,
                                epsabs=1e-300, epsrel=2e-13)[0]
        return first, second
    if name == "obl_rad2" and args[3] < 1.5:
        m, n, c, x = args
        # Continue the accurately evaluated large-argument solution using
        # the defining radial ODE (DLMF 30.2.1, z=i*x). This avoids the
        # incorrectly indexed qstar convolution in SciPy's small-x path.
        cv = special.obl_cv(m, n, c)
        initial = special.obl_rad2(m, n, c, 2.0)
        def ode(t, state):
            value, derivative = state
            denominator = 1+t*t
            return derivative, (-2*t*derivative
                                -(c*c*t*t-cv+m*m/denominator)*value)/denominator
        solution = integrate.solve_ivp(ode, (2.0, x), initial, method="DOP853",
                                       rtol=2e-13, atol=2e-14)
        if not solution.success:
            raise RuntimeError(solution.message)
        return tuple(solution.y[:, -1])
    if name == "hyperu":
        value = special.hyperu(*args)
        a, b, x = args
        if not np.isfinite(value) and a > 0 and x > 0:
            # SciPy's legacy dispatch can fail for valid inputs (e.g. 12,10,.01).
            # DLMF 13.4.4 with s=x*t supplies an independent convergent integral.
            return integrate.quad(
                lambda s: np.exp(-s + (a-1)*np.log(s)
                                 + (b-a-1)*np.log1p(s/x)
                                 - special.gammaln(a) - a*np.log(x)),
                0, np.inf, epsabs=2e-13, epsrel=2e-13,
            )[0]
        return value
    if name in {"modfresnelp", "modfresnelm"} and args[0] < 0:
        # SciPy's negative-x branch overwrites Re(F) before computing Im(F).
        # Reflect the positive-x reference using the defining Fresnel integral.
        x, = args
        sign = 1 if name == "modfresnelp" else -1
        positive, _ = getattr(special, name)(-x)
        f = np.sqrt(np.pi/2)*(1 + sign*1j) - positive
        k = np.exp(-sign*1j*(x*x + np.pi/4))*f/np.sqrt(np.pi)
        return f, k
    if name == "itstruve0":
        # SciPy 1.18.1's large-x formula disagrees with its defining integral.
        return integrate.quad(lambda t: special.struve(0, t), 0, abs(args[0]),
                              epsabs=2e-12, epsrel=2e-12)[0]
    if name == "itairy":
        # SciPy's legacy series stops at 1e-5 and loses ~4e-10 already at x=0.5.
        # Independent quadrature verifies the improved 1e-15 series termination.
        x, = args
        return tuple(integrate.quad(lambda t: special.airy(sign*t)[index], 0, x,
                                    epsabs=2e-13, epsrel=2e-13)[0]
                     for sign in (1, -1) for index in (0, 2))
    if name == "keip_zeros":
        return stable_keip_zeros(*args)
    if name == "kelvin_zeros":
        return (*special.kelvin_zeros(*args)[:7], stable_keip_zeros(*args))
    if name == "trigamma":
        return special.polygamma(1, *args)
    if name == "lambertwm1":
        return special.lambertw(*args, -1).real
    if name == "ellipke":
        return special.ellipk(*args), special.ellipe(*args)
    if name == "ellipf":
        return special.ellipkinc(*args)
    if name.startswith("cel"):
        kc = args[0]
        if name == "cel":
            p, a, b = args[1:]
        elif name == "cel1":
            p, a, b = 1, 1, 1
        elif name == "cel2":
            p, a, b = 1, *args[1:]
        else:
            p, a, b = args[1], 1, 1
        limit = np.pi / 2
        m, n = 1 - kc * kc, 1 - p
    elif name.startswith("el") and name in {"el1", "el2", "el3"}:
        limit, m = np.arctan(args[0]), 1 - args[1] ** 2
        n, a, b = 0, 1, 1
        if name == "el2":
            a, b = args[2:]
        elif name == "el3":
            n = 1 - args[2]
    elif name in {"ellippi", "ellippiinc", "ellippiinc_bulirsch"}:
        limit, n, m = (np.pi / 2, *args) if name == "ellippi" else args
        a, b = 1, 1
    elif name in {"ellipd", "ellipdinc"}:
        limit, m = (np.pi / 2, args[0]) if name == "ellipd" else args
        n, a, b = 0, 0, 1
    elif name == "jacobi_zeta":
        phi, m = args
        return special.ellipeinc(phi, m) - special.ellipe(m) / special.ellipk(m) * special.ellipkinc(phi, m)
    elif name == "heuman_lambda":
        phi, m = args
        f, e = special.ellipkinc(phi, 1 - m), special.ellipeinc(phi, 1 - m)
        return 2 / np.pi * (special.ellipe(m) * f + special.ellipk(m) * (e - f))
    else:
        return getattr(special, name)(*args)
    return integrate.quad(
        lambda t: (a * np.cos(t)**2 + b * np.sin(t)**2)
        / ((1 - n * np.sin(t)**2) * np.sqrt(1 - m * np.sin(t)**2)),
        0, limit, epsabs=2e-13, epsrel=2e-13,
    )[0]


def flatten(value):
    if isinstance(value, tuple):
        return np.concatenate([flatten(component) for component in value])
    return np.asarray(value).ravel()


def generate():
    directory = Path(__file__).resolve().parent
    cases = json.loads((directory / "special_cases.json").read_text())
    if (np.__version__, scipy.__version__) != ("2.5.4", "1.18.1"):
        raise RuntimeError("Use NumPy 2.5.4 and SciPy 1.18.1")
    path = directory.parent / "fixtures" / "special_extended.csv"
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["function", "arguments", "expected", "rtol", "atol"])
        for case in cases:
            arguments = case["args"].copy()
            if case["name"] == "wofz":
                arguments[0] = complex(*arguments[0])
            value = reference(case["name"], arguments)
            if case.get("polynomial"):
                value = value.coeffs
            result = flatten(value)
            if case["name"] in {"y0_zeros", "y1_zeros", "y1p_zeros", "pbdn_seq", "lambertw"}:
                if np.any(np.imag(result) != 0):
                    raise ValueError("Real-domain reference unexpectedly has imaginary components")
                result = np.real(result)
            if np.iscomplexobj(result):
                result = np.column_stack([result.real, result.imag]).ravel()
            writer.writerow([
                case["name"],
                ";".join(
                    "/".join(map(str, arg)) if isinstance(arg, list)
                    else str(arg).lower() if isinstance(arg, bool) else str(arg)
                    for arg in case["args"]
                ),
                ";".join(map(str, result)),
                1e-4 if case["name"] in {"ncfdtridfd", "ncfdtridfn", "ncfdtrinc"}
                else 5e-6 if case["name"].startswith("nctdtr")
                else 8e-15 if case["name"] in {"lambertw", "lambertwm1"} else 2e-11,
                1e-7 if case["name"].startswith("nctdtr")
                else 0.0 if case["name"] in {"lambertw", "lambertwm1"} else 2e-11,
            ])


if __name__ == "__main__":
    generate()
