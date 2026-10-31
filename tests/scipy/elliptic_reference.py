"""Independent references for real elliptic domains beyond the old Sci wrappers.

Uses SciPy's Carlson integrals (DLMF 19.25.5, 19.25.14) for principal values,
and defining quadrature for regular m >= 1 paths. No Sci or Ellip code is used.
Versions are checked by the calling generators (NumPy 2.5.4/SciPy 1.18.1).
"""
import numpy as np
from scipy import integrate, special


def third_kind(phi, n, m):
    """Pi(phi,n,m), on the principal amplitude interval, including PV poles."""
    s = np.sin(phi)
    c2 = 0.0 if abs(phi) == np.pi / 2 else np.cos(phi)**2
    y = 1 - m*s*s
    return s*special.elliprf(c2, y, 1) + n*s**3*special.elliprj(c2, y, 1, 1-n*s*s)/3


def extended_reference(name, x):
    """Return (handled, result); ordinary cases stay in their existing generator."""
    if name in {"ellipf", "ellipkinc", "ellipeinc", "ellipdinc"} and x[1] >= 1:
        phi, m = x
        numerator = (lambda t: np.sin(t)**2) if name == "ellipdinc" else (lambda t: 1.)
        f = ((lambda t: np.sqrt(1-m*np.sin(t)**2)) if name == "ellipeinc"
             else (lambda t: numerator(t)/np.sqrt(1-m*np.sin(t)**2)))
        return True, integrate.quad(f, 0, phi, epsabs=2e-13, epsrel=2e-13)[0]
    if name == "ellippi" and x[0] > 1:
        return True, third_kind(np.pi/2, *x)
    if name in {"ellippiinc", "ellippiinc_bulirsch"} and (x[1] >= 1 or x[2] >= 1):
        return True, third_kind(*x)
    if name in {"cel", "cel3"} and x[1] < 0:
        kc, p = x[:2]
        a, b = x[2:] if name == "cel" else (1., 1.)
        m, n = 1-kc*kc, 1-p
        return True, (a-b)/n*special.ellipk(m) + (b-a*p)/n*third_kind(np.pi/2, n, m)
    if name == "el3" and x[2] < 0:
        tangent, kc, p = x
        return True, third_kind(np.arctan(tangent), 1-p, 1-kc*kc)
    if name == "jacobi_zeta" and x[1] == 1:
        return True, np.sin(x[0])*np.sign(np.cos(x[0]))
    return False, None
