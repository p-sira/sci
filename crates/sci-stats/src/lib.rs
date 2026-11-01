//! # Sci-Stats
//!
//! Statistical analysis, hypothesis tests, and probability distributions.
//!
//! Descriptive statistics (mean, variance, quantiles, moments, trimming, and
//! standardization) accept `T: sci_core::Float`. Distribution and inference
//! algorithms currently accept `f64`.
//!
//! Part of [Sci](https://github.com/p-sira/sci), a unified scientific computing library for Rust.
//!
//! Descriptive statistics use generic floating-point values; hypothesis tests and
//! native distributions use `f64`.
//! Distribution evaluation uses statrs and shared special functions privately; callers supply parameters
//! directly rather than constructing a backend distribution.
//!
//! Finite observations are required, with at most 1,048,576 samples per slice.
//! Mean is `Σx/n`; population variance uses denominator `n`, while sample
//! variance uses `n - 1` (`ddof = 1`). Probability functions use the
//! parameterization stated in each function and return the documented IEEE
//! endpoint values; invalid parameters are reported as [`Error`].
//!
//! Public functions are organized by statistical family: [`descriptive`],
//! [`distributions`], [`inference`], [`association`], [`ranking`],
//! [`regression`], [`resampling`], and other specialized modules.
//! Shared [`Error`] and [`Result`] types remain at the crate root.
//!
//! ## Examples
//!
//! ```
//! use sci_complex as complex; use sci_fft as fft; use sci_stats as stats;
//! let amplitudes: Vec<f64> = fft::real::rfft(&[1.0, 2.0, 3.0, 4.0])
//!     .into_iter().map(complex::abs).collect();
//! assert!(stats::descriptive::mean(&amplitudes)? > 0.0);
//! # Ok::<(), sci_core::Error>(())
//! ```
//!
//! ## Acknowledgments
//!
//! We thank `nalgebra`, `statrs`, `basin`,
//! `lapjv`, `ndarray`, and SciPy (BSD-3-Clause) for their statistical and
//! numerical building blocks.

#![forbid(unsafe_code)]
#![deny(missing_docs, rustdoc::broken_intra_doc_links)]

use sci_core as error;
pub use sci_core::{Error, Result};
use sci_special as special;
#[path = "mod.rs"]
mod implementation;
/// Correlations and measures of statistical association.
pub mod association {
    pub use crate::implementation::{
        KendallVariant, MgcResult, WeightedRank, chatterjeexi, kendalltau, multiscale_graphcorr,
        pearsonr, pointbiserialr, somersd, spearmanr, spearmanrho, weightedtau,
    };
}

/// Binned statistics and empirical frequency summaries.
pub mod binning {
    pub use crate::implementation::{
        BinStatistic, BinnedStatistic, Frequency, binned_statistic, binned_statistic_2d,
        binned_statistic_dd, cumfreq, relfreq,
    };
}

/// Contingency tables and exact tests for categorical data.
pub mod contingency {
    pub use crate::implementation::contingency::{
        Association, ContingencyResult, Crosstab, association, chi2_contingency, crosstab,
        expected_freq, margins, odds_ratio, relative_risk,
    };
    pub use crate::implementation::{barnard_exact, boschloo_exact, fisher_exact};
}

/// Location, dispersion, shape, trimming, and sample summaries.
pub mod descriptive {
    pub use crate::implementation::{
        Description, DirectionalStats, SigmaClip, Tail, circmean, circstd, circvar, describe,
        differential_entropy, directional_stats, entropy, expectile, gmean, gstd, gzscore, hmean,
        iqr, kstat, kstatvar, kurtosis, lmoment, mean, median, median_abs_deviation, mode, moment,
        pmean, quantile, scoreatpercentile, sem, sigmaclip, skew, std, tmax, tmean, tmin,
        trim_mean, trim1, trimboth, tsem, tstd, tvar, variance, variation, zmap, zscore,
    };
}

/// Distribution diagnostics and transformation parameter plots.
pub mod diagnostics {
    pub use crate::implementation::{
        BoxCoxMethod, PpccFamily, PpccPlot, ProbabilityPlot, TransformPlot, boxcox_normmax,
        boxcox_normplot, ppcc_max, ppcc_plot, probplot, yeojohnson_normmax, yeojohnson_normplot,
    };
}

/// Statistical distances between probability samples.
pub mod distances {
    pub use crate::implementation::{
        energy_distance, wasserstein_distance, wasserstein_distance_nd,
    };
}

/// Probability distributions, composition, and parameter fitting.
pub mod distributions {
    pub use crate::implementation::{
        CustomDistribution, Distribution, DistributionSpec, FitFamily, OrderStatisticDistribution,
        TruncatedDistribution, cdf, fit, logpdf, make_distribution, normal_cdf, normal_pdf,
        normal_ppf, order_statistic, pdf, ppf, sf, truncate,
    };
}

/// Bayesian estimators and credible intervals.
pub mod estimation {
    pub use crate::implementation::{BayesianMvs, CredibleEstimate, bayes_mvs, mvsdist};
}

/// Goodness-of-fit and distribution comparison tests.
pub mod goodness {
    pub use crate::implementation::{
        AndersonFamily, AndersonResult, AndersonVariant, CvmMethod, GoodnessStatistic, KsResult,
        anderson, anderson_ksamp, chisquare, cramervonmises, cramervonmises_2samp, goodness_of_fit,
        ks_1samp, ks_2samp, power_divergence,
    };
}

/// Hypothesis tests, multiple comparisons, and statistical power.
pub mod inference {
    pub use crate::implementation::{
        Alternative, BrunnerDistribution, Center, CombineMethod, DunnettResult, FdrMethod,
        MedianTestResult, MedianTies, PairwiseTest, PowerResult, QuantileCount, QuantileTestResult,
        TTestResult, TestResult, WilcoxonZero, ansari, bartlett, binomtest, brunnermunzel,
        combine_pvalues, dunnett, epps_singleton_2samp, f_oneway, false_discovery_control, fligner,
        friedmanchisquare, kruskal, levene, mannwhitneyu, median_test, mood, page_trend_test,
        poisson_means_test, power, quantile_test, ranksums, ttest_1samp, ttest_ind,
        ttest_ind_from_stats, ttest_rel, tukey_hsd, wilcoxon,
    };
}

/// Multivariate probability densities and masses.
pub mod multivariate {
    pub use crate::implementation::{
        categorical_cdf, categorical_pmf, dirichlet_logpdf, dirichlet_pdf, multinomial_logpmf,
        multinomial_pmf, multivariate_normal_logpdf, multivariate_normal_pdf,
        multivariate_t_logpdf, multivariate_t_pdf,
    };
}

/// Normality and homoscedasticity diagnostics.
pub mod normality {
    pub use crate::implementation::{
        alexandergovern, jarque_bera, kurtosistest, normaltest, obrientransform, shapiro, skewtest,
    };
}

/// Ranking, percentiles, and empirical distribution functions.
pub mod ranking {
    pub use crate::implementation::{
        EmpiricalCdf, EstimatedCdfMethod, PercentileKind, RankMethod, ecdf, estimated_cdf,
        percentileofscore, rankdata, tiecorrect,
    };
}

/// Quasi-Monte Carlo discrepancy and coordinate scaling.
pub use implementation::qmc;

/// Linear and robust regression estimators.
pub mod regression {
    pub use crate::implementation::{
        Regression, RobustRegression, SiegelMethod, TheilMethod, TheilRegression, linregress,
        siegelslopes, theilslopes,
    };
}

/// Bootstrap, Monte Carlo, and permutation procedures.
pub mod resampling {
    pub use crate::implementation::{
        BootstrapMethod, BootstrapResult, ResamplingResult, bootstrap, bws_test, monte_carlo_test,
        permutation_test,
    };
}

/// Variance-based sensitivity measures.
pub mod sensitivity {
    pub use crate::implementation::{SobolIndices, sobol_indices};
}

/// Survival analysis and group comparisons.
pub mod survival {
    pub use crate::implementation::logrank;
}

/// Distribution-transforming and power-transform functions.
pub mod transformations {
    pub use crate::implementation::{boxcox, boxcox_llf, yeojohnson, yeojohnson_llf};
}
