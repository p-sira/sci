# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Independent references for audit fixes; inputs contain no expected values.

Mean, variance and central moments use their exact rational definitions on
binary64 observations: direct SciPy/NumPy evaluation can overflow, underflow,
or lose the cancellation residuals these cases exercise. Standard deviation
is the square root of this variance. Regression slopes/intercepts use exact
least-squares identities; the regression fixtures are exact lines, so errors
are zero. Skew/kurtosis use SciPy on translated data (translation invariance).
FFT, Savitzky-Golay and distance outputs come directly from SciPy.
Z-scores use exact rational centering and Decimal square roots because direct
SciPy evaluation can round the mean or overflow/underflow the spread. Geometric
z-scores apply the same definition to NumPy's binary64 logarithms.

Prelaunch cases use 128-digit Decimal definitions for KL/relative entropy,
pseudo-Huber loss and posterior weights: SciPy's direct binary64 formulas lose
accuracy or overflow for these inputs. T-tests use Decimal means, variances,
standard errors and Welch degrees of freedom with SciPy Student-t tails.
Coherence uses SciPy on separately amplitude-normalized signals, exploiting
its dimensionless invariance to avoid SciPy's squared-spectrum range errors.
Integer matrix powers use SciPy directly. Nonfinite references are compared
explicitly in Rust; tiny divergences allow one minimum subnormal of absolute
rounding error and must remain nonzero when their reference is nonzero.

Normal references use SciPy's standard distribution and Decimal location/scale
arithmetic; log densities restore tiny scales after standardized tail underflow.
Expectiles use exact rational solutions of the documented weighted residual
balance, because SciPy's root optimizer loses extreme-alpha solutions. Weighted
Lp distances use Decimal powers; ratio metrics remove zero-weight coordinates
before a scale-invariant SciPy evaluation. Gradient descent is a native extra:
references evaluate its quadratic update identity and use Python's stable hypot
for the documented stopping criterion, independently of Sci or Basin.

Comparisons use 5e-13 relative tolerance for exact statistics and 2e-11 for
backend-dependent transforms/fits. Tiny nonzero results have only one minimum
subnormal of absolute tolerance, so a zero result cannot pass accidentally.
Standardized scores allow eight binary64 epsilons of absolute error: subtracting
the normalized mean can lose a few ulps for scores near zero. Their relative
tolerance remains 5e-13.
"""

import csv
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy
from scipy import fft, linalg, signal, special, stats
from scipy.spatial import distance

assert (np.__version__, scipy.__version__) == ("2.5.4", "1.18.1")
ROOT = Path(__file__).resolve().parents[1]


def encode(value):
    if value and isinstance(value[0], list):
        return ":".join(encode(row) for row in value)
    return ";".join(str(float(v)) for v in value)


def reference(case):
    api, x, y, p = (case[key] for key in ("api", "x", "y", "p"))
    if api == "gradient_descent":
        curvature, rate, tolerance, limit = p
        point = np.array(x)
        for iteration in range(int(limit) + 1):
            if math.hypot(*(curvature * point)) <= tolerance:
                status = 0
                break
            if iteration == limit:
                status = 1
                break
            point = point - rate * (curvature * point)
        cost = sum(0.5 * curvature * v * v for v in point)
        return [*point.tolist(), cost, float(iteration), float(status)]
    if api == "expectile":
        values = sorted(map(Fraction, x))
        alpha = Fraction(p[0])
        if alpha == 0 or values[0] == values[-1]:
            return [float(values[0])]
        if alpha == 1:
            return [float(values[-1])]
        for split in range(1, len(values)):
            wl, wr = 1 - alpha, alpha
            solution = (wl * sum(values[:split]) + wr * sum(values[split:])) / (
                wl * split + wr * (len(values) - split)
            )
            if values[split - 1] <= solution <= values[split]:
                return [float(solution)]
        raise AssertionError("expectile balance has no solution")
    if api.startswith("normal_"):
        from decimal import Decimal, localcontext

        operation = api.removeprefix("normal_").removeprefix("generic_")
        mean, scale = p
        with localcontext() as context:
            context.prec = 128
            if operation == "ppf":
                z = float(special.ndtri(x[0]))
                return [float(Decimal(scale) * Decimal(z) + Decimal(mean))]
            z = float((Decimal(x[0]) - Decimal(mean)) / Decimal(scale))
        if operation in ("pdf", "logpdf"):
            log_density = float(stats.norm.logpdf(z)) - math.log(scale)
            return [math.exp(log_density) if operation == "pdf" else log_density]
        return [float(getattr(stats.norm, operation)(z))]
    if api in ("euclidean", "sqeuclidean", "cityblock", "minkowski", "chebyshev"):
        from decimal import Decimal, localcontext

        power, weights = (float(p[0]), p[1:]) if api == "minkowski" else (
            {"euclidean": 2, "sqeuclidean": 2, "cityblock": 1, "chebyshev": math.inf}[api], p
        )
        with localcontext() as context:
            context.prec = 128
            differences = [(abs(Decimal(a) - Decimal(b)), Decimal(w)) for a, b, w in zip(x, y, weights) if w > 0]
            if math.isinf(power):
                return [float(max((d for d, _ in differences), default=0))]
            total = sum(w * d ** Decimal(power) for d, w in differences)
            value = total if api == "sqeuclidean" else total ** (1 / Decimal(power))
            return [float(value)]
    if api in ("cosine", "correlation", "braycurtis"):
        active = [(a, b, w) for a, b, w in zip(x, y, p) if w > 0]
        a, b, weights = map(np.array, zip(*active))
        scale = max(np.max(np.abs(a)), np.max(np.abs(b)))
        if api == "braycurtis":
            a, b = a / scale, b / scale
        else:
            a, b = a / np.max(np.abs(a)), b / np.max(np.abs(b))
        return [float(getattr(distance, api)(a, b, w=weights / np.max(weights)))]
    if api in (
        "ttest_ind",
        "ttest_ind_from_stats",
        "rel_entr",
        "kl_div",
        "pseudo_huber",
        "particle_update",
    ):
        from decimal import Decimal, localcontext

        with localcontext() as context:
            context.prec = 128
            d = lambda value: Decimal(float(value))
            if api in ("rel_entr", "kl_div", "pseudo_huber"):
                a, b = map(d, x)
                if api == "pseudo_huber":
                    # Sci retains its documented NaN for negative delta and
                    # the quadratic infinite-delta limit; SciPy 1.18.1 differs.
                    if a.is_nan() or b.is_nan() or a < 0:
                        return [math.nan]
                    if a == 0:
                        return [0.0]
                    if a.is_infinite():
                        return [float(b * b / 2)]
                    if b.is_infinite():
                        return [math.inf]
                if (
                    not (a.is_finite() and b.is_finite())
                    or a <= 0
                    or (api != "pseudo_huber" and b <= 0)
                ):
                    return [float(getattr(special, api)(*map(float, x)))]
                if api == "pseudo_huber":
                    value = a * b * b / ((a * a + b * b).sqrt() + a)
                else:
                    value = a * (a / b).ln()
                    if api == "kl_div":
                        value += b - a
                return [float(value)]
            if api == "particle_update":
                weights, likelihoods = list(map(d, x)), list(map(d, y))
                shift = max(ll for w, ll in zip(weights, likelihoods) if w > 0)
                posterior = [
                    w * (ll - shift).exp() if w > 0 and ll.is_finite() else Decimal(0)
                    for w, ll in zip(weights, likelihoods)
                ]
                total = sum(posterior)
                return [float(w / total) for w in posterior]
            if api == "ttest_ind":

                def summary(values):
                    values = list(map(d, values))
                    n = len(values)
                    mean = sum(values) / n
                    std = (sum((v - mean) ** 2 for v in values) / (n - 1)).sqrt()
                    return mean, std, n

                m1, s1, n1 = summary(x)
                m2, s2, n2 = summary(y)
            else:
                m1, s1, n1 = d(x[0]), d(x[1]), int(x[2])
                m2, s2, n2 = d(y[0]), d(y[1]), int(y[2])
            if p[0]:
                df = Decimal(n1 + n2 - 2)
                se = (
                    ((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2)
                    / df
                    * (Decimal(1) / n1 + Decimal(1) / n2)
                ).sqrt()
            else:
                a, b = s1 * s1 / n1, s2 * s2 / n2
                se = (a + b).sqrt()
                df = (
                    (a + b) ** 2 / (a * a / (n1 - 1) + b * b / (n2 - 1))
                    if a + b
                    else Decimal(1)
                )
            t = (
                float((m1 - m2) / se)
                if se
                else (math.nan if m1 == m2 else math.copysign(math.inf, float(m1 - m2)))
            )
            df = float(df)
            probability = [
                lambda: 2 * stats.t.sf(abs(t), df),
                lambda: stats.t.cdf(t, df),
                lambda: stats.t.sf(t, df),
            ][int(p[1])]()
            return [t, float(probability), df]
    if api == "coherence":
        a, b = np.asarray(x), np.asarray(y)
        a = np.ldexp(a, -math.frexp(np.max(np.abs(a)))[1])
        b = np.ldexp(b, -math.frexp(np.max(np.abs(b)))[1])
        frequencies, values = signal.coherence(
            a,
            b,
            window=case["matrix"][0],
            noverlap=int(p[0]),
            nfft=int(p[1]),
            nperseg=len(case["matrix"][0]),
            fs=p[2],
        )
        return [*frequencies.tolist(), *values.tolist()]
    if api == "fractional_matrix_power":
        result = linalg.fractional_matrix_power(case["matrix"], p[0]).astype(complex)
        return (
            np.column_stack((result.real.ravel(), result.imag.ravel())).ravel().tolist()
        )
    if api in ("zscore", "zmap", "gzscore"):
        from decimal import Decimal, localcontext

        if api == "gzscore":
            x = np.log(x).tolist()
        values = list(map(Fraction, y if api == "zmap" else x))
        center = sum(values) / len(values)
        variance = sum((v - center) ** 2 for v in values) / (len(values) - int(p[0]))
        with localcontext() as context:
            context.prec = 100
            spread = (
                Decimal(variance.numerator) / Decimal(variance.denominator)
            ).sqrt()
            result = []
            for v in map(Fraction, x):
                delta = v - center
                result.append(
                    float(
                        (Decimal(delta.numerator) / Decimal(delta.denominator)) / spread
                    )
                )
            return result
    if api in ("mean", "variance", "std", "moment"):
        values = list(map(Fraction, x))
        center = sum(values) / len(values)
        if api == "mean":
            return [float(center)]
        order = int(p[0]) if api == "moment" else 2
        divisor = len(values) if api == "moment" else len(values) - int(p[0])
        moment = sum((v - center) ** order for v in values) / divisor
        if api == "std":
            # sqrt of the fraction without overflowing its f64 conversion.
            from decimal import Decimal, localcontext

            with localcontext() as context:
                context.prec = 100
                return [
                    float(
                        (Decimal(moment.numerator) / Decimal(moment.denominator)).sqrt()
                    )
                ]
        return [float(moment)]
    if api in ("skew", "kurtosis"):
        translated = np.asarray(x) - x[0]
        return [float(getattr(stats, api)(translated))]
    if api == "linregress":
        a, b = list(map(Fraction, x)), list(map(Fraction, y))
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        aa = sum((v - ma) ** 2 for v in a)
        slope = sum((u - ma) * (v - mb) for u, v in zip(a, b)) / aa
        intercept = mb - slope * ma
        assert all(v == intercept + slope * u for u, v in zip(a, b))
        return [float(slope), float(intercept), 0.0, 0.0]
    if api in ("fft", "ifft", "rfft"):
        value = np.asarray(x) if api == "rfft" else np.asarray(x) + 1j * np.asarray(y)
        result = getattr(fft, api)(value)
        return np.column_stack((result.real, result.imag)).ravel().tolist()
    if api == "savgol_coeffs":
        return signal.savgol_coeffs(
            int(p[0]), int(p[1]), int(p[2]), delta=p[3]
        ).tolist()
    if api == "savgol_filter":
        return signal.savgol_filter(
            x, int(p[0]), int(p[1]), int(p[2]), delta=p[3], mode="interp"
        ).tolist()
    if api == "cdist":
        return distance.cdist(x, y, "mahalanobis", VI=case["matrix"]).ravel().tolist()
    if api == "pdist":
        return distance.pdist(x, "mahalanobis", VI=case["matrix"]).tolist()
    raise ValueError(api)


cases = json.loads(Path(__file__).with_name("audit_regressions_cases.json").read_text())
with (
    (ROOT / "fixtures/audit_regressions_inputs.csv").open("w", newline="") as inputs,
    (ROOT / "fixtures/audit_regressions_expected.csv").open(
        "w", newline=""
    ) as expected,
):
    iw, ew = (
        csv.writer(inputs, lineterminator="\n"),
        csv.writer(expected, lineterminator="\n"),
    )
    iw.writerow(["id", "api", "x", "y", "p", "matrix"])
    ew.writerow(["id", "values", "rtol", "atol"])
    for case in cases:
        iw.writerow(
            [
                case["id"],
                case["api"],
                *(encode(case[key]) for key in ("x", "y", "p", "matrix")),
            ]
        )
        result = reference(case)
        exact = case["api"] in (
            "expectile", "gradient_descent", "euclidean", "sqeuclidean", "cityblock", "minkowski", "chebyshev",
            "mean",
            "variance",
            "std",
            "moment",
            "linregress",
            "zscore",
            "zmap",
            "gzscore",
            "ttest_ind",
            "ttest_ind_from_stats",
            "rel_entr",
            "kl_div",
            "pseudo_huber",
            "particle_update",
            "fractional_matrix_power",
        )
        exact = exact or case["api"].startswith("normal_")
        tiny = any(0 < abs(v) < 1e-200 for v in result)
        atol = math.ulp(0.0) if tiny else (0.0 if exact else 2e-11)
        if case["api"] in ("zscore", "zmap", "gzscore"):
            atol = 8 * np.finfo(float).eps
        ew.writerow([case["id"], encode(result), 5e-13 if exact else 2e-11, atol])
