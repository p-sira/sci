# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Independent Bayesian filter references from documented matrix identities.

SciPy has no Bayesian filter namespace. These references use scipy.linalg solves
and Cholesky factors plus the equations in FILTER_COVERAGE.md. No Rust library or
implementation backend generates expectations. UKF includes Q before generating
observation sigma points, unlike the cached-point FilterPy default.
Run: uv run tests/scipy/generate_filter.py.
"""
import csv
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import cholesky, solve
from scipy.special import logsumexp

if (np.__version__, scipy.__version__) != ("2.5.4", "1.18.1"):
    raise RuntimeError("Use NumPy 2.5.4 and SciPy 1.18.1")
ROOT = Path(__file__).resolve().parents[1]


def encode(v):
    if isinstance(v, list):
        return (":" if v and isinstance(v[0], list) else ";").join(map(encode, v))
    return str(v)


def model(x, name):
    if name == "linear":
        return x.copy() if len(x) == 1 else np.array([x[0] + .2*x[1], x[1]])
    if name == "square":
        return x*x
    if name == "coupled":
        return np.array([x[0] + .1*x[1]**2, x[1]])
    if name == "range":
        return np.array([np.linalg.norm(x)])
    raise ValueError(name)


def jacobian(x, name):
    if name == "square":
        return np.diag(2*x)
    if name == "coupled":
        return np.array([[1., .2*x[1]], [0., 1.]])
    if name == "range":
        return (x/np.linalg.norm(x)).reshape(1, -1)
    raise ValueError(name)


def update(x, p, z, h, r, predicted=None):
    y = z - (h @ x if predicted is None else predicted)
    s = h @ p @ h.T + r
    k = solve(s, h @ p, assume_a="pos").T
    # Joseph covariance provides an independent, numerically stable witness.
    a = np.eye(len(x)) - k @ h
    return x + k @ y, a @ p @ a.T + k @ r @ k.T, y


def sigmas(x, p, c):
    columns = cholesky(c*p, lower=True).T
    return np.concatenate((x[None, :], x + columns, x - columns))


def transform(points, wm, wc, noise):
    mean = wm @ points
    delta = points - mean
    return mean, (delta.T*wc) @ delta + noise


def reference(name, args):
    a = [np.array(v, dtype=float) if isinstance(v, list) else v for v in args]
    if name.startswith("kalman") or name.startswith("extended"):
        x, p = a[:2]
        if name == "kalman_predict":
            f, q = a[2:]
            return f @ x, f @ p @ f.T + q
        if name == "kalman_update":
            z, h, r = a[2:]
            return update(x, p, z, h, r)
        if name == "extended_kalman_predict":
            q, m = a[2:]
            j = jacobian(x, m)
            return model(x, m), j @ p @ j.T + q
        z, r, m = a[2:]
        return update(x, p, z, jacobian(x, m), r, model(x, m))
    if name == "unscented_kalman_step":
        x,p,q,z,r,scaling,f,h = a
        alpha,beta,kappa = scaling
        n = len(x)
        c = alpha**2*(n+kappa)
        wm = np.full(2*n+1, .5/c)
        wm[0] = (c-n)/c
        wc = wm.copy()
        wc[0] += 1-alpha**2+beta
        transformed = np.array([model(v, f) for v in sigmas(x,p,c)])
        xp,pp = transform(transformed,wm,wc,q)
        # Resample after adding Q, then include Q in the cross covariance.
        points = sigmas(xp,pp,c)
        observed = np.array([v.copy() if h == "linear" else model(v,h) for v in points])
        zp,s = transform(observed,wm,wc,r)
        cross = ((points-xp).T*wc) @ (observed-zp)
        k = solve(s,cross.T,assume_a="pos").T
        y = z-zp
        return xp+k@y,pp-k@s@k.T,y
    states,weights = a[:2]
    weights = weights / weights.max()
    weights /= weights.sum()
    if name == "particle_predict":
        return np.array([model(v,a[2]) for v in states]),weights
    if name == "particle_update":
        with np.errstate(divide="ignore"):
            likelihoods = np.where(weights > 0., a[2], -np.inf)
            logs = np.log(weights)+(likelihoods-np.max(likelihoods[weights > 0.]))
        weights = np.exp(logs-logsumexp(logs))
        return states,weights
    if name == "particle_estimate":
        mean = weights @ states
        delta = states-mean
        return mean,(delta.T*weights)@delta
    if name == "effective_sample_size":
        return (np.array([1./(weights@weights)]),)
    if name == "systematic_resample":
        cdf = np.cumsum(weights)
        cdf[-1] = 1.
        positions = (np.arange(len(states))+a[2])/len(states)
        indices = np.searchsorted(cdf,positions,side="right")
        return states[indices],np.full(len(states),1./len(states))
    raise ValueError(name)


def main():
    cases = json.loads((ROOT/"scipy/filter_cases.json").read_text())
    with (ROOT/"fixtures/filter_inputs.csv").open("w", newline="") as i, (ROOT/"fixtures/filter_expected.csv").open("w", newline="") as e:
        wi,we = csv.writer(i, lineterminator="\n"),csv.writer(e, lineterminator="\n")
        wi.writerow(["id","function","arguments"])
        we.writerow(["id","values"])
        for case in cases:
            wi.writerow([case["id"],case["function"],*[encode(v) for v in case["args"]]])
            result = reference(case["function"],case["args"])
            we.writerow([case["id"],";".join(format(float(v),".17g") for part in result for v in np.asarray(part).reshape(-1))])
    print(f"Generated {len(cases)} independent filter cases")


if __name__ == "__main__":
    main()
