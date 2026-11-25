# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Generate independent SciPy references from statistics_cases.jsonl.

Run with the pinned .venv or uv run. Inputs and expected outputs are separate.
Tests use a combined 2e-10 absolute/relative tolerance for statrs approximations;
quantiles additionally allow backend tail-CDF errors amplified by inverse density.
NaN and infinities are explicit. Rust tests require no Python runtime.
"""

import csv
from fractions import Fraction
import warnings
from pathlib import Path
from case_catalog import read_cases

import numpy as np
import scipy
from scipy import stats
from scipy.stats import qmc

if (np.__version__, scipy.__version__) != ("2.5.4", "1.18.1"):
    raise RuntimeError("Use NumPy 2.5.4 and SciPy 1.18.1")
ROOT = Path(__file__).resolve().parents[1] / "fixtures"
ALTS = ["two-sided", "less", "greater"]


def distribution(name, p):
    constructors = {
        "Levy": lambda: stats.levy(loc=p[0], scale=p[1]),
        "Normal": lambda: stats.norm(loc=p[0], scale=p[1]),
        "Beta": lambda: stats.beta(*p),
        "Gamma": lambda: stats.gamma(p[0], scale=p[1]),
        "Erlang": lambda: stats.erlang(p[0], scale=p[1]),
        "DiscreteUniform": lambda: stats.randint(*[int(v) for v in p]),
        "ChiSquared": lambda: stats.chi2(*p),
        "Chi": lambda: stats.chi(*p),
        "StudentT": lambda: stats.t(*p),
        "F": lambda: stats.f(*p),
        "Exponential": lambda: stats.expon(scale=p[0]),
        "Cauchy": lambda: stats.cauchy(loc=p[0], scale=p[1]),
        "Laplace": lambda: stats.laplace(loc=p[0], scale=p[1]),
        "LogNormal": lambda: stats.lognorm(p[1], scale=np.exp(p[0])),
        "Uniform": lambda: stats.uniform(loc=p[0], scale=p[1] - p[0]),
        "Weibull": lambda: stats.weibull_min(p[0], scale=p[1]),
        "Pareto": lambda: stats.pareto(p[1], scale=p[0]),
        "InverseGamma": lambda: stats.invgamma(p[0], scale=p[1]),
        "Gumbel": lambda: stats.gumbel_r(loc=p[0], scale=p[1]),
        "Triangular": lambda: stats.triang(
            (p[2] - p[0]) / (p[1] - p[0]), loc=p[0], scale=p[1] - p[0]
        ),
        "Bernoulli": lambda: stats.bernoulli(*p),
        "Binomial": lambda: stats.binom(*p),
        "Poisson": lambda: stats.poisson(*p),
        "Geometric": lambda: stats.geom(*p),
        "NegativeBinomial": lambda: stats.nbinom(*p),
        "Hypergeometric": lambda: stats.hypergeom(*p),
    }
    return constructors[name]()


def reference(c):
    api, p, x, y, g, d = (c[k] for k in ("api", "p", "x", "y", "g", "d"))
    g = [np.asarray(row, dtype=float) for row in g]
    p = [float(v) for v in p]
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if api == "wasserstein_distance_nd":
        n = int(p[0])
        return [stats.wasserstein_distance_nd(np.asarray(g[:n]),np.asarray(g[n:]))]
    if api == "dunnett":
        alt = ALTS[int(p[0])]
        # SciPy's public test supplies statistics. Independently increase its
        # multivariate-t CDF budget for reproducible, less noisy probabilities.
        r = stats.dunnett(*g[1:],control=g[0],alternative=alt,rng=np.random.default_rng(8421))
        counts = np.array([len(row) for row in g],dtype=float)
        df = counts.sum()-len(g)
        k = len(g)-1
        correlations = np.sqrt((1/counts[0])/(1/counts[0]+1/counts[1:]))
        covariance = np.outer(correlations,correlations)
        np.fill_diagonal(covariance,1.)
        if k == 1:
            t = stats.ttest_ind(g[1],g[0],alternative=alt)
            probabilities = [t.pvalue]
        else:
            joint = stats.multivariate_t(shape=covariance,df=df)
            probabilities = []
            for t in r.statistic:
                threshold = abs(t) if alt == "two-sided" else -t if alt == "less" else t
                lower = np.full(k,-threshold if alt == "two-sided" else -np.inf)
                upper = np.full(k,threshold)
                value = joint.cdf(upper,lower_limit=lower,maxpts=1000000,random_state=np.random.default_rng(8421))
                probabilities.append(1-value)
        return np.r_[r.statistic,probabilities]
    if api == "multiscale_graphcorr":
        n = int(p[0])
        first,second = np.asarray(g[:n]),np.asarray(g[n:2*n])
        permutations = np.asarray(g[2*n:],dtype=int)
        r = stats.multiscale_graphcorr(first,second,reps=0,random_state=0)
        null = np.array([stats.multiscale_graphcorr(first,second[row],reps=0,random_state=0).statistic for row in permutations])
        # Conservative floating-point equality matches Sci's explicit adaptation.
        probability = (1+np.sum(null >= r.statistic-100*np.finfo(float).eps*abs(r.statistic)))/(len(null)+1)
        scale = r.mgc_dict["opt_scale"]
        if np.ndim(scale) == 0:
            scale = r.mgc_dict["mgc_map"].shape
        return np.r_[r.statistic,probability,r.mgc_dict["mgc_map"].shape,r.mgc_dict["mgc_map"].ravel(),scale,null]
    if api in ("barnard_exact","boschloo_exact"):
        r = getattr(stats,api)(x.reshape(2,2),alternative=ALTS[int(p[0])],n=128,
                             **({"pooled":bool(p[1])} if api == "barnard_exact" else {}))
        return [r.statistic,r.pvalue]
    if api == "poisson_means_test":
        r = stats.poisson_means_test(*p[:4],diff=p[4],alternative=ALTS[int(p[5])])
        return [r.statistic,r.pvalue]
    if api == "make_distribution":
        # Independently use SciPy's established beta distribution to verify
        # the scalar callback example's density/CDF/SF/quantile definitions.
        dist = stats.beta(2,1)
        return [dist.pdf(x[0]),dist.logpdf(x[0]),dist.cdf(x[0]),dist.sf(x[0]),dist.ppf(p[0])]
    if api == "truncate":
        dist = stats.truncnorm(p[0],p[1])
        return [dist.pdf(x[0]),dist.logpdf(x[0]),dist.cdf(x[0]),dist.sf(x[0]),dist.ppf(p[2])]
    if api == "order_statistic":
        base = [stats.Normal(),stats.Uniform(a=0,b=1)][int(p[2])]
        dist = stats.order_statistic(base,r=int(p[0]),n=int(p[1]))
        return [dist.pdf(x[0]),dist.logpdf(x[0]),dist.cdf(x[0]),dist.ccdf(x[0]),dist.icdf(p[3])]
    if api == "shapiro":
        # Normalize before SciPy as well: Shapiro W is location/scale invariant,
        # and SciPy's absolute range cutoff is inappropriate for subnormal units.
        shifted = x.astype(np.longdouble)-np.longdouble(x[0])
        scale = np.max(np.abs(shifted))
        normalized = np.asarray(shifted/scale,dtype=float) if scale else x
        r = stats.shapiro(normalized)
        return [r.statistic,r.pvalue]
    if api == "tukey_hsd":
        r = stats.tukey_hsd(*g)
        return np.r_[r.statistic.ravel(),r.pvalue.ravel()]
    if api == "estimated_cdf":
        methods = ["inverted_cdf","averaged_inverted_cdf","closest_observation","interpolated_inverted_cdf",
                   "hazen","weibull","linear","median_unbiased","normal_unbiased"]
        return [stats.estimated_cdf(x,p[0],method=methods[int(p[1])])]
    if api == "sobol_indices":
        outputs = dict(f_A=x[None,:], f_B=y[None,:], f_AB=np.asarray(g)[:,None,:])
        r = stats.sobol_indices(func=outputs,n=len(x))
        return np.r_[r.first_order.ravel(),r.total_order.ravel()]
    if api == "power":
        test = (lambda a: stats.ttest_1samp(a,0,alternative="greater").pvalue) if p[1] == 0 else (lambda a: stats.norm.sf(np.mean(a)*np.sqrt(len(a))))
        def rvs(size):
            if size != (len(g),len(g[0])):
                raise ValueError(size)
            return np.asarray(g)
        r = stats.power(test,rvs,len(g[0]),significance=p[0],vectorized=False,n_resamples=len(g))
        return np.r_[r.power,r.pvalues]
    if api == "anderson_ksamp":
        r = stats.anderson_ksamp(g, variant=["midrank", "right"][int(p[0])])
        return [r.statistic,r.pvalue]
    if api == "fit":
        family = [stats.norm,stats.expon,stats.uniform][int(p[0])]
        fitted = family.fit(x, **({"floc":0} if p[0] == 1 else {}))
        # Uniform Rust parameters are low/high; SciPy uses location/scale.
        params = (fitted[0],fitted[0]+fitted[1]) if p[0] == 2 else fitted
        return np.r_[params, family(*fitted).cdf(y[0]), family(*fitted).ppf(p[1])]
    if api == "goodness_of_fit":
        family = [stats.norm,stats.expon,stats.uniform][int(p[0])]
        fitted = family.fit(x, **({"floc":0} if p[0] == 1 else {}))
        location, scale = fitted
        if p[0] == 0:
            scale = np.std(x,ddof=1)
        schedule = (np.asarray(g)-location)/scale
        class ScheduledNullGenerator(np.random.Generator):
            def draw(self,size):
                if size != schedule.shape:
                    raise ValueError(size)
                return schedule.copy()
            def standard_normal(self,size=None,**kwargs):
                return self.draw(size)
            def standard_exponential(self,size=None,**kwargs):
                return self.draw(size)
            def uniform(self,low=0,high=1,size=None):
                if low != 0 or high != 1:
                    raise ValueError((low,high))
                return self.draw(size)
        rng = ScheduledNullGenerator(np.random.PCG64(0))
        r = stats.goodness_of_fit(family,x,known_params={"loc":0} if p[0] == 1 else None,
                                 statistic=["ks","cvm"][int(p[1])], n_mc_samples=len(g), rng=rng)
        return np.r_[r.statistic,r.pvalue,r.null_distribution]
    if api in ("permutation_test", "monte_carlo_test", "bootstrap"):
        statistic = [np.mean, np.median, lambda a: np.mean(a**2)][int(p[-1])]
        if api == "permutation_test":
            r = stats.permutation_test((x,y), lambda a,b: statistic(a)-statistic(b), vectorized=False,
                                       n_resamples=np.inf, alternative=ALTS[int(p[0])])
            return np.r_[r.statistic, r.pvalue, r.null_distribution]
        if api == "monte_carlo_test":
            def rvs(size):
                if size != (len(g),len(x)):
                    raise ValueError(size)
                return np.asarray(g)
            r = stats.monte_carlo_test(x, rvs, statistic, vectorized=False,
                                       n_resamples=len(g), alternative=ALTS[int(p[0])])
            return np.r_[r.statistic, r.pvalue, r.null_distribution]
        class ScheduledGenerator(np.random.Generator):
            def integers(self, low, high=None, size=None, dtype=np.int64, endpoint=False):
                if size != (len(g),len(x)) or low != 0 or high != len(x):
                    raise ValueError((low,high,size))
                return np.asarray(g,dtype=dtype)
        rng = ScheduledGenerator(np.random.PCG64(0))
        r = stats.bootstrap((x,), statistic, vectorized=False, n_resamples=len(g),
                            confidence_level=p[0], method=["percentile","basic"][int(p[1])], rng=rng)
        return np.r_[statistic(x), r.confidence_interval.low, r.confidence_interval.high,
                     r.standard_error, r.bootstrap_distribution]
    if api == "bws_test":
        r = stats.bws_test(x,y,alternative=ALTS[int(p[0])], method=stats.PermutationMethod(n_resamples=np.inf))
        return np.r_[r.statistic, r.pvalue, r.null_distribution]
    if api == "logrank":
        first = stats.CensoredData.right_censored(x, censored=np.asarray(g[0]) == 0)
        second = stats.CensoredData.right_censored(y, censored=np.asarray(g[1]) == 0)
        r = stats.logrank(first, second, alternative=ALTS[int(p[0])])
        return [r.statistic, r.pvalue]
    if api == "epps_singleton_2samp":
        r = stats.epps_singleton_2samp(x, y, t=p)
        return [r.statistic, r.pvalue]
    if api == "anderson":
        r = stats.anderson(x, dist=["norm", "expon"][int(p[0])])
        return np.r_[r.statistic, r.critical_values, r.significance_level, r.fit_result.params]
    if api == "cramervonmises":
        null = stats.norm(loc=y[0], scale=y[1])
        r = stats.cramervonmises(x, null.cdf)
        return [r.statistic, r.pvalue]
    if api == "cramervonmises_2samp":
        r = stats.cramervonmises_2samp(x, y, method=["exact", "asymptotic"][int(p[0])])
        return [r.statistic, r.pvalue]
    if api == "bayes_mvs":
        return np.r_[[v for r in stats.bayes_mvs(x, alpha=p[0]) for v in (r.statistic, *r.minmax)]]
    if api == "mvsdist":
        return [d.ppf(p[0]) for d in stats.mvsdist(x)]
    if api in ("boxcox_normmax", "yeojohnson_normmax", "ppcc_max"):
        from scipy.optimize import minimize_scalar
        if api == "boxcox_normmax":
            if p[2] == 0:
                def objective(l):
                    return -stats.pearsonr(stats.probplot(x, fit=False)[0], np.sort(stats.boxcox(x, lmbda=l))).statistic
            else:
                def objective(l):
                    return -stats.boxcox_llf(l, x)
        elif api == "yeojohnson_normmax":
            def objective(l):
                return -stats.yeojohnson_llf(l, x)
        else:
            family = [stats.t, stats.chi2, stats.gamma, stats.weibull_min, stats.invgamma][int(p[2])]
            def objective(l):
                return -stats.pearsonr(*stats.probplot(x, sparams=(l,), dist=family, fit=False)).statistic
        # Independent SciPy bounded optimization reproduces the hard-bound
        # adaptation. Include endpoints, which minimize_scalar does not return.
        r = minimize_scalar(objective, bounds=p[:2], method="bounded", options={"xatol": 1e-10, "maxiter": 128})
        if not r.success:
            raise RuntimeError(r.message)
        return [min((p[0], r.x, p[1]), key=objective)]
    if api == "ppcc_plot":
        family = [stats.t, stats.chi2, stats.gamma, stats.weibull_min, stats.invgamma][int(p[3])]
        shapes, correlations = stats.ppcc_plot(x, p[0], p[1], N=int(p[2]), dist=family)
        return np.r_[shapes, correlations]
    if api == "weightedtau":
        rank = [True, None, False, y.astype(int)][int(p[0])]
        r = stats.weightedtau(x, g[0], rank=rank, additive=bool(p[1]))
        return [r.statistic, r.pvalue]
    if api in ("somersd", "spearmanrho"):
        r = getattr(stats, api)(x, y, alternative=ALTS[int(p[0])])
        return [r.statistic, r.pvalue]
    if api == "chatterjeexi":
        r = stats.chatterjeexi(x, y, y_continuous=bool(p[0]))
        return [r.statistic, r.pvalue]
    def zero_skew():
        # Exact identity on binary64 inputs, independent of Sci's moments.
        values = list(map(Fraction, map(float, x)))
        mu = sum(values) / len(values)
        return len(set(values)) > 1 and sum((v - mu)**3 for v in values) == 0
    if api in ("skewtest", "kurtosistest"):
        r = getattr(stats, api)(x, alternative=ALTS[int(p[0])])
        if api == "skewtest" and zero_skew():
            # SciPy's Y=1 substitution is discontinuous at zero skewness.
            return [0., 1. if int(p[0]) == 0 else 0.5]
        return [r.statistic, r.pvalue]
    if api == "normaltest":
        r = stats.normaltest(x)
        if zero_skew():
            statistic = stats.kurtosistest(x).statistic**2
            return [statistic, stats.chi2.sf(statistic, 2)]
        return [r.statistic, r.pvalue]
    if api == "alexandergovern":
        r = stats.alexandergovern(*g)
        return [r.statistic, r.pvalue]
    if api == "obrientransform":
        return np.concatenate(stats.obrientransform(*g))
    if api == "siegelslopes":
        r = stats.siegelslopes(y, x, method=["hierarchical", "separate"][int(p[0])])
        return [r.slope, r.intercept]
    if api == "theilslopes":
        r = stats.theilslopes(y, x, alpha=p[0], method=["separate", "joint"][int(p[1])])
        return [r.slope, r.intercept, r.low_slope, r.high_slope]
    if api == "brunnermunzel":
        r = stats.brunnermunzel(x, y, alternative=ALTS[int(p[0])], distribution=["t", "normal"][int(p[1])])
        return [r.statistic, r.pvalue]
    if api == "kendalltau":
        r = stats.kendalltau(x, y, variant=["b", "c"][int(p[0])], alternative=ALTS[int(p[1])], method="asymptotic")
        return [r.statistic, r.pvalue]
    if api == "wilcoxon":
        r = stats.wilcoxon(x, zero_method=["wilcox", "pratt", "zsplit"][int(p[0])], alternative=ALTS[int(p[1])], correction=bool(p[2]), method="asymptotic")
        return [r.statistic, r.pvalue]
    if api == "ansari":
        r = stats.ansari(x, y, alternative=ALTS[int(p[0])], method="asymptotic")
        return [r.statistic, r.pvalue]
    if api == "quantile_test":
        r = stats.quantile_test(x, q=p[0], p=p[1], alternative=ALTS[int(p[2])])
        return [r.statistic, r.statistic_type, r.pvalue]
    if api == "page_trend_test":
        r = stats.page_trend_test(g, predicted_ranks=y.astype(int), method="asymptotic")
        return [r.statistic, r.pvalue]
    if api == "mood":
        if np.unique(np.r_[x, y]).size == 1:
            return [np.nan, np.nan]
        r = stats.mood(x, y, alternative=ALTS[int(p[0])])
        return [r.statistic, r.pvalue]
    if api == "mannwhitneyu":
        r = stats.mannwhitneyu(x, y, alternative=ALTS[int(p[0])], use_continuity=bool(p[1]), method="asymptotic")
        return [r.statistic, r.pvalue]
    if api == "fligner":
        center = ["mean", "median"][int(p[0])]
        deviations = np.concatenate([np.abs(row - getattr(np, center)(row)) for row in g])
        if np.unique(deviations).size == 1:
            return [np.nan, np.nan]  # zero score variance, not roundoff variance
        r = stats.fligner(*g, center=center)
        return [r.statistic, r.pvalue]
    if api == "median_test":
        r = stats.median_test(*g, ties=["below", "above", "ignore"][int(p[0])], correction=bool(p[1]))
        return np.r_[r.statistic, r.pvalue, r.median, r.table.ravel()]
    if api in ("boxcox_normplot", "yeojohnson_normplot"):
        lambdas, correlations = getattr(stats, api)(x, p[0], p[1], N=int(p[2]))
        return np.r_[lambdas, correlations]
    if api in ["ks_1samp", "ks_2samp", "kstest"]:
        alt = ALTS[int(p[0])]
        if api == "ks_1samp":
            r = stats.ks_1samp(
                x, stats.norm(loc=y[0], scale=y[1]).cdf, alternative=alt, method="asymp"
            )
        else:
            r = stats.ks_2samp(x, y, alternative=alt, method="asymp")
            # Sci supports the limiting two-sample distribution. SciPy's
            # ks_2samp asymp additionally applies finite-sample corrections.
            # Use SciPy's independent limiting distribution as the p reference.
            effective = len(x) * len(y) / (len(x) + len(y))
            pvalue = (
                stats.kstwobign.sf(np.sqrt(effective) * r.statistic)
                if alt == "two-sided"
                else np.exp(-2 * effective * r.statistic**2)
            )
            return [r.statistic, pvalue, r.statistic_location, r.statistic_sign]
        return [r.statistic, r.pvalue, r.statistic_location, r.statistic_sign]
    if api == "probplot":
        r = stats.probplot(x, dist=stats.norm(loc=y[0], scale=y[1]), fit=False)
        return np.r_[r[0], r[1]]
    if api in ["abs", "exp", "log"]:
        return [getattr(np, api)(x[0])]
    if api in ["tmean", "tvar", "tstd", "tsem"]:
        return [
            getattr(stats, api)(
                x,
                limits=p[:2],
                inclusive=tuple(bool(v) for v in p[2:4]),
                **({"ddof": int(p[4])} if api != "tmean" else {}),
            )
        ]
    if api in ["tmin", "tmax"]:
        return [getattr(stats, api)(x, p[0], inclusive=bool(p[1]))]
    if api == "trim1":
        return np.sort(stats.trim1(x, p[0], tail=["left", "right"][int(p[1])]))
    if api in ["kstat", "kstatvar"]:
        return [getattr(stats, api)(x, int(p[0]))]
    if api == "lmoment":
        return np.atleast_1d(stats.lmoment(x, order=int(p[0]), standardize=bool(p[1])))
    if api == "sigmaclip":
        r = stats.sigmaclip(x, p[0], p[1])
        return np.r_[r.clipped, r.lower, r.upper]
    if api == "differential_entropy":
        return [
            stats.differential_entropy(
                x, window_length=int(p[0]), base=p[1] or None, method="vasicek"
            )
        ]
    if api == "binomtest":
        r = stats.binomtest(int(p[0]), int(p[1]), p[2], alternative=ALTS[int(p[3])])
        return [r.statistic, r.pvalue]
    if api == "combine_pvalues":
        r = stats.combine_pvalues(
            x,
            method=["fisher", "pearson", "tippett", "stouffer", "mudholkar_george"][
                int(p[0])
            ],
            weights=y if len(y) else None,
        )
        return [r.statistic, r.pvalue]
    if api == "directional_stats":
        r = stats.directional_stats(np.asarray(g), normalize=bool(p[0]))
        return np.r_[r.mean_direction, r.mean_resultant_length]
    if api == "discrepancy":
        return [
            qmc.discrepancy(
                np.asarray(g),
                method=["CD", "WD", "MD", "L2-star"][int(p[0])],
                iterative=bool(p[1]),
            )
        ]
    if api == "geometric_discrepancy":
        return [
            qmc.geometric_discrepancy(
                np.asarray(g), method=["mindist", "mst"][int(p[0])]
            )
        ]
    if api == "update_discrepancy":
        return [
            qmc.update_discrepancy(
                x, np.asarray(g), qmc.discrepancy(np.asarray(g), iterative=True)
            )
        ]
    if api == "scale":
        return qmc.scale(np.asarray(g), x, y, reverse=bool(p[0])).ravel()
    if api in ["multivariate_normal_pdf", "multivariate_normal_logpdf"]:
        return [
            getattr(stats.multivariate_normal(mean=y, cov=g), api.split("_")[-1])(x)
        ]
    if api in ["multivariate_t_pdf", "multivariate_t_logpdf"]:
        return [
            getattr(stats.multivariate_t(loc=y, shape=g, df=p[0]), api.split("_")[-1])(
                x
            )
        ]
    if api in ["dirichlet_pdf", "dirichlet_logpdf"]:
        return [getattr(stats.dirichlet(y), api.split("_")[-1])(x)]
    if api in ["multinomial_pmf", "multinomial_logpmf"]:
        return [getattr(stats.multinomial(n=int(x.sum()), p=y), api.split("_")[-1])(x)]
    if api in ["categorical_pmf", "categorical_cdf"]:
        # Independent categorical identities; SciPy rv_discrete serves as the
        # reference distribution for arbitrary normalized category weights.
        probs = x / x.sum()
        dist = stats.rv_discrete(values=(np.arange(len(x)), probs))
        return [getattr(dist, api.split("_")[-1])(p[0])]
    if api in ["binned_statistic", "binned_statistic_2d", "binned_statistic_dd"]:
        statistic = ["count", "sum", "mean", "std", "median", "min", "max"][int(p[-1])]
        if api == "binned_statistic":
            r = stats.binned_statistic(x, y, statistic=statistic, bins=g[0])
            edges = [r.bin_edges]
        elif api == "binned_statistic_2d":
            r = stats.binned_statistic_2d(x, y, g[2], statistic=statistic, bins=g[:2])
            edges = [r.x_edge, r.y_edge]
        else:
            r = stats.binned_statistic_dd(
                np.asarray(g),
                x,
                statistic=statistic,
                bins=y.reshape(int(p[0]), int(p[1])),
            )
            edges = r.bin_edges
        return np.r_[
            r.statistic.shape, r.statistic.ravel(), *[e for e in edges], r.binnumber
        ]
    if api in ["cumfreq", "relfreq"]:
        r = getattr(stats, api)(x, numbins=int(p[0]), defaultreallimits=p[1:3])
        return np.r_[r[0], r.lowerlimit, r.binsize, r.extrapoints]
    if d:
        op = api
        if d in [
            "Bernoulli",
            "Binomial",
            "Poisson",
            "Geometric",
            "NegativeBinomial",
            "Hypergeometric",
            "DiscreteUniform",
        ]:
            op = {"pdf": "pmf", "logpdf": "logpmf"}.get(op, op)
        # Infinite observations have zero density/mass by the defining limits.
        # Several SciPy density implementations instead evaluate inf-inf to NaN.
        if np.isinf(x[0]) and op in ["pdf", "pmf"]:
            return [0.0]
        if np.isinf(x[0]) and op in ["logpdf", "logpmf"]:
            return [-np.inf]
        # Floor discrete CDF arguments explicitly; SciPy hypergeom alone returns
        # NaN for noninteger x in this runtime, unlike the CDF definition.
        arg = np.floor(x[0]) if d == "Hypergeometric" and op in ["cdf", "sf"] else x[0]
        # Geom(p=1) is a point mass at one; SciPy 1.18.1's ppf returns
        # zero inside (0,1). Verify the defining degenerate limit independently.
        if d == "Geometric" and p[0] == 1 and op == "ppf" and 0 < arg < 1:
            return [1.0]
        return [getattr(distribution(d, p), op)(arg)]
    if api in ["mean", "variance", "std", "median"]:
        return [
            getattr(np, {"variance": "var"}.get(api, api))(
                x, **({"ddof": int(p[0])} if p else {})
            )
        ]
    if api in ["sem", "variation", "gstd", "zscore", "gzscore"]:
        return np.atleast_1d(getattr(stats, api)(x, ddof=int(p[0])))
    if api == "zmap":
        return stats.zmap(x, y, ddof=int(p[0]))
    if api in ["gmean", "hmean", "iqr", "jarque_bera", "ecdf", "mode", "tiecorrect"]:
        r = getattr(stats, api)(x)
        if api == "jarque_bera":
            return [r.statistic, r.pvalue]
        if api == "ecdf":
            return np.r_[r.cdf.quantiles, r.cdf.probabilities, r.sf.probabilities]
        if api == "mode":
            return [r.mode, r.count]
        return [r]
    if api == "quantile":
        return [stats.quantile(x, p[0])]
    if api == "scoreatpercentile":
        return [stats.scoreatpercentile(x, p[0])]
    if api == "median_abs_deviation":
        return [stats.median_abs_deviation(x, scale=p[0])]
    if api in ["trim_mean", "trimboth"]:
        r = getattr(stats, api)(x, p[0])
        return np.sort(r) if api == "trimboth" else [r]
    if api == "moment":
        return [stats.moment(x, moment=int(p[0]))]
    if api == "skew":
        return [stats.skew(x, bias=bool(p[0]))]
    if api == "kurtosis":
        return [stats.kurtosis(x, fisher=bool(p[0]), bias=bool(p[1]))]
    if api == "describe":
        r = stats.describe(x, ddof=int(p[0]), bias=bool(p[1]))
        return [r.nobs, *r.minmax, r.mean, r.variance, r.skewness, r.kurtosis]
    if api == "rankdata":
        return stats.rankdata(
            x, method=["average", "min", "max", "dense", "ordinal"][int(p[0])]
        )
    if api == "percentileofscore":
        return [
            stats.percentileofscore(
                x, p[0], kind=["rank", "strict", "weak", "mean"][int(p[1])]
            )
        ]
    if api == "false_discovery_control":
        return stats.false_discovery_control(x, method=["bh", "by"][int(p[0])])
    if api in ["boxcox", "yeojohnson"]:
        return getattr(stats, api)(x, lmbda=p[0])
    if api in ["boxcox_llf", "yeojohnson_llf"]:
        # A constant transformed sample has zero variance mathematically. The
        # SciPy llf functions sometimes produce finite likelihoods from roundoff.
        if np.all(x == x[0]):
            return [np.inf]
        if api == "boxcox_llf" and abs(p[0]) < 1e-6:
            # SciPy's log-variance llf path loses about 1e-8 at tiny lambda.
            # Independently evaluate the documented likelihood using SciPy's
            # stable transform, with long-double NumPy variance and log sums.
            z = stats.boxcox(x, lmbda=p[0]).astype(np.longdouble)
            return [
                (p[0] - 1) * np.log(x.astype(np.longdouble)).sum()
                - len(x) / 2 * np.log(np.var(z))
            ]
        return [getattr(stats, api)(p[0], x)]
    if api == "pmean":
        return [stats.pmean(x, p[0])]
    if api == "expectile":
        return [stats.expectile(x, alpha=p[0])]
    if api in ["circmean", "circvar", "circstd"]:
        return [
            getattr(stats, api)(
                x,
                low=p[0],
                high=p[1],
                **({"normalize": bool(p[2])} if api == "circstd" else {}),
            )
        ]
    if api == "entropy":
        return [stats.entropy(x, y if len(y) else None, base=p[0] or None)]
    if api in ["wasserstein_distance", "energy_distance"]:
        return [getattr(stats, api)(x, y)]
    if api == "ttest_ind_from_stats":
        r = stats.ttest_ind_from_stats(
            *p[:6], equal_var=bool(p[6]), alternative=ALTS[int(p[7])]
        )
        # SciPy does not expose df here: independent Welch/pooled definition.
        _, s1, n1, _, s2, n2 = p[:6]
        a = s1 * s1 / n1
        b = s2 * s2 / n2
        df = (
            n1 + n2 - 2
            if p[6]
            else (a + b) ** 2 / (a * a / (n1 - 1) + b * b / (n2 - 1))
        )
        return [r.statistic, r.pvalue, df]
    if api in ["ttest_1samp", "ttest_rel", "ttest_ind"]:
        if api == "ttest_1samp":
            r = stats.ttest_1samp(x, p[0], alternative=ALTS[int(p[1])])
        elif api == "ttest_ind":
            r = stats.ttest_ind(x, y, equal_var=bool(p[0]), alternative=ALTS[int(p[1])])
        else:
            r = stats.ttest_rel(x, y, alternative=ALTS[int(p[0])])
        return [r.statistic, r.pvalue, r.df]
    if api in ["pearsonr", "spearmanr", "ranksums", "linregress", "pointbiserialr"]:
        r = getattr(stats, api)(
            x,
            y,
            **({} if api == "pointbiserialr" else {"alternative": ALTS[int(p[0])]}),
        )
        if api == "linregress":
            return [
                r.slope,
                r.intercept,
                r.rvalue,
                r.pvalue,
                r.stderr,
                r.intercept_stderr,
            ]
        return [r.statistic, r.pvalue]
    if api in ["f_oneway", "bartlett", "kruskal", "friedmanchisquare", "levene"]:
        r = getattr(stats, api)(
            *g, **({"center": ["mean", "median"][int(p[0])]} if api == "levene" else {})
        )
        return [r.statistic, r.pvalue]
    if api == "fisher_exact":
        r = stats.fisher_exact(x.reshape(2, 2), alternative=ALTS[int(p[0])])
        return [r.statistic, r.pvalue]
    if api in ["chisquare", "power_divergence"]:
        if api == "power_divergence" and p[1] == -0.5 and np.any(x == 0):
            # Freeman-Tukey identity gives the finite zero-frequency limit;
            # SciPy's literal O*(O/E)^lambda produces 0*inf -> NaN.
            statistic = 4 * np.sum((np.sqrt(x) - np.sqrt(y)) ** 2)
            return [statistic, stats.chi2.sf(statistic, len(x) - 1 - int(p[0]))]
        r = getattr(stats, api)(
            x,
            y,
            ddof=int(p[0]),
            **({"lambda_": p[1]} if api == "power_divergence" else {}),
        )
        return [r.statistic, r.pvalue]
    if api in ["margins", "expected_freq", "chi2_contingency", "association"]:
        a = x.reshape(int(p[0]), int(p[1]))
        if api == "association":
            return [
                stats.contingency.association(
                    a.astype(int),
                    method=["cramer", "tschuprow", "pearson"][int(p[2])],
                    correction=bool(p[3]),
                )
            ]
        r = getattr(stats.contingency, api)(
            a, **({"correction": bool(p[2])} if api == "chi2_contingency" else {})
        )
        if api == "margins":
            return np.r_[r[0].ravel(), r[1].ravel()]
        if api == "chi2_contingency":
            return np.r_[r.statistic, r.pvalue, r.dof, r.expected_freq.ravel()]
        return r.ravel()
    if api == "relative_risk":
        return [stats.contingency.relative_risk(*[int(v) for v in p]).relative_risk]
    if api == "odds_ratio":
        return [
            stats.contingency.odds_ratio(
                x.reshape(2, 2).astype(int), kind="sample"
            ).statistic
        ]
    if api == "crosstab":
        r = stats.contingency.crosstab(x, y)
        return np.r_[*r.elements, r.count.ravel()]
    raise ValueError(api)


def encode(values):
    return ";".join(str(float(v)) for v in values)


cases = read_cases(Path(__file__).with_name("statistics_cases.jsonl"))
# Validate all references before opening output files, preserving existing
# fixtures if SciPy rejects an input or a reference adaptation fails.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    results = [reference(c) for c in cases]

with (
    (ROOT / "stats_inputs.csv").open("w", newline="") as inputs,
    (ROOT / "stats_expected.csv").open("w", newline="") as outputs,
):
    iw, ow = (
        csv.writer(inputs, lineterminator="\n"),
        csv.writer(outputs, lineterminator="\n"),
    )
    iw.writerow(["id", "api", "params", "x", "y", "groups", "distribution"])
    ow.writerow(["id", "values"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, c in enumerate(cases):
            r = results[i]
            iw.writerow(
                [
                    i,
                    c["api"],
                    encode(c["p"]),
                    encode(c["x"]),
                    encode(c["y"]),
                    "/".join(encode(g) for g in c["g"]),
                    c["d"],
                ]
            )
            ow.writerow([i, encode(r)])
    print(f"Wrote {len(cases)} statistics reference cases")
