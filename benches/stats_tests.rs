//! Benchmarks covering the hypothesis tests, resampling, association and
//! diagnostic functions of `sci::stats`.
//!
//! Inputs are deterministic so that results are reproducible across runs.

mod common;

use common::{points, positive_values, values};
use divan::{Bencher, black_box};
use sci::stats::{self, Alternative, Distribution, contingency};

fn main() {
    divan::main();
}

/// Sample of `n` values shifted by `shift`.
fn shifted(n: usize, seed: u64, shift: f64) -> Vec<f64> {
    values(n, seed).into_iter().map(|v| v + shift).collect()
}

/// Three groups of `n` values with slightly different locations.
fn groups(n: usize) -> Vec<Vec<f64>> {
    (0..3)
        .map(|i| shifted(n, 20 + i as u64, 0.1 * i as f64))
        .collect()
}

fn group_refs(groups: &[Vec<f64>]) -> Vec<&[f64]> {
    groups.iter().map(Vec::as_slice).collect()
}

fn point_refs(points: &[Vec<f64>]) -> Vec<&[f64]> {
    points.iter().map(Vec::as_slice).collect()
}

/// Deterministic permutation of `0..n` derived from a seed.
fn permutation(n: usize, seed: u64) -> Vec<usize> {
    let keys = values(n, seed);
    let mut index: Vec<usize> = (0..n).collect();
    index.sort_by(|&a, &b| keys[a].total_cmp(&keys[b]));
    index
}

/// Deterministic bootstrap resample indices in `0..n`.
fn resample(n: usize, seed: u64) -> Vec<usize> {
    values(n, seed)
        .into_iter()
        .map(|v| (((v + 1.0) * 0.5 * n as f64) as usize).min(n - 1))
        .collect()
}

mod ttest_benches {
    use super::*;

    #[divan::bench]
    fn ttest_1samp(bencher: Bencher) {
        let x = values(200, 1);
        bencher.bench(|| {
            stats::ttest_1samp(black_box(&x), black_box(0.1), Alternative::TwoSided).unwrap()
        });
    }

    #[divan::bench]
    fn ttest_rel(bencher: Bencher) {
        let (x, y) = (values(200, 1), shifted(200, 2, 0.1));
        bencher.bench(|| {
            stats::ttest_rel(black_box(&x), black_box(&y), Alternative::TwoSided).unwrap()
        });
    }

    #[divan::bench]
    fn ttest_ind(bencher: Bencher) {
        let (x, y) = (values(200, 1), shifted(200, 2, 0.1));
        bencher.bench(|| {
            stats::ttest_ind(black_box(&x), black_box(&y), false, Alternative::TwoSided).unwrap()
        });
    }

    #[divan::bench]
    fn ttest_ind_from_stats(bencher: Bencher) {
        bencher.bench(|| {
            stats::ttest_ind_from_stats(
                black_box(0.2),
                black_box(1.1),
                black_box(40),
                black_box(0.5),
                black_box(0.9),
                black_box(60),
                false,
                Alternative::TwoSided,
            )
            .unwrap()
        });
    }
}

mod correlation_benches {
    use super::*;

    fn pair() -> (Vec<f64>, Vec<f64>) {
        let x = values(200, 1);
        let noise = values(200, 2);
        let y = x.iter().zip(&noise).map(|(a, b)| a + 0.5 * b).collect();
        (x, y)
    }

    #[divan::bench]
    fn pearsonr(bencher: Bencher) {
        let (x, y) = pair();
        bencher.bench(|| {
            stats::pearsonr(black_box(&x), black_box(&y), Alternative::TwoSided).unwrap()
        });
    }

    #[divan::bench]
    fn spearmanr(bencher: Bencher) {
        let (x, y) = pair();
        bencher.bench(|| {
            stats::spearmanr(black_box(&x), black_box(&y), Alternative::TwoSided).unwrap()
        });
    }

    #[divan::bench]
    fn spearmanrho(bencher: Bencher) {
        let (x, y) = pair();
        bencher.bench(|| {
            stats::spearmanrho(black_box(&x), black_box(&y), Alternative::TwoSided).unwrap()
        });
    }

    #[divan::bench]
    fn pointbiserialr(bencher: Bencher) {
        let x = values(200, 1);
        let binary: Vec<f64> = values(200, 2)
            .into_iter()
            .map(|v| if v > 0.0 { 1.0 } else { 0.0 })
            .collect();
        bencher.bench(|| stats::pointbiserialr(black_box(&binary), black_box(&x)).unwrap());
    }

    #[divan::bench]
    fn linregress(bencher: Bencher) {
        let (x, y) = pair();
        bencher.bench(|| {
            stats::linregress(black_box(&x), black_box(&y), Alternative::TwoSided).unwrap()
        });
    }

    #[divan::bench]
    fn kendalltau(bencher: Bencher) {
        let (x, y) = pair();
        bencher.bench(|| {
            stats::kendalltau(
                black_box(&x),
                black_box(&y),
                stats::KendallVariant::B,
                Alternative::TwoSided,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn weightedtau(bencher: Bencher) {
        let (x, y) = pair();
        bencher.bench(|| {
            stats::weightedtau(
                black_box(&x),
                black_box(&y),
                stats::WeightedRank::Symmetric,
                true,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn somersd(bencher: Bencher) {
        let (x, y) = pair();
        bencher
            .bench(|| stats::somersd(black_box(&x), black_box(&y), Alternative::TwoSided).unwrap());
    }

    #[divan::bench]
    fn chatterjeexi(bencher: Bencher) {
        let (x, y) = pair();
        bencher.bench(|| stats::chatterjeexi(black_box(&x), black_box(&y), true).unwrap());
    }

    #[divan::bench]
    fn multiscale_graphcorr(bencher: Bencher) {
        let x = points(20, 2, 1);
        let y: Vec<Vec<f64>> = x
            .iter()
            .zip(values(20, 3))
            .map(|(p, e)| vec![p[0] * p[0] + 0.1 * e])
            .collect();
        let perms: Vec<Vec<usize>> = (0..5).map(|s| permutation(20, 100 + s)).collect();
        let (xr, yr) = (point_refs(&x), point_refs(&y));
        let pr: Vec<&[usize]> = perms.iter().map(Vec::as_slice).collect();
        bencher.bench(|| {
            stats::multiscale_graphcorr(black_box(&xr), black_box(&yr), black_box(&pr)).unwrap()
        });
    }
}

mod variance_benches {
    use super::*;

    #[divan::bench]
    fn f_oneway(bencher: Bencher) {
        let g = groups(100);
        let refs = group_refs(&g);
        bencher.bench(|| stats::f_oneway(black_box(&refs)).unwrap());
    }

    #[divan::bench]
    fn bartlett(bencher: Bencher) {
        let g = groups(100);
        let refs = group_refs(&g);
        bencher.bench(|| stats::bartlett(black_box(&refs)).unwrap());
    }

    #[divan::bench]
    fn levene(bencher: Bencher) {
        let g = groups(100);
        let refs = group_refs(&g);
        bencher.bench(|| stats::levene(black_box(&refs), stats::Center::Median).unwrap());
    }

    #[divan::bench]
    fn fligner(bencher: Bencher) {
        let g = groups(100);
        let refs = group_refs(&g);
        bencher.bench(|| stats::fligner(black_box(&refs), stats::Center::Median).unwrap());
    }

    #[divan::bench]
    fn alexandergovern(bencher: Bencher) {
        let g = groups(100);
        let refs = group_refs(&g);
        bencher.bench(|| stats::alexandergovern(black_box(&refs)).unwrap());
    }

    #[divan::bench]
    fn obrientransform(bencher: Bencher) {
        let g = groups(100);
        let refs = group_refs(&g);
        bencher.bench(|| stats::obrientransform(black_box(&refs)).unwrap());
    }

    #[divan::bench]
    fn tukey_hsd(bencher: Bencher) {
        // Two groups: three or more groups use a studentized-range quadrature
        // costing tens of milliseconds per pair, beyond the benchmark budget.
        let g = groups(30);
        let refs = group_refs(&g[..2]);
        bencher.bench(|| stats::tukey_hsd(black_box(&refs)).unwrap());
    }

    #[divan::bench]
    fn dunnett(bencher: Bencher) {
        // A single treatment: several treatments use a multivariate quadrature
        // costing tens of milliseconds, beyond the benchmark budget.
        let control = values(30, 1);
        let g = groups(30);
        let refs = group_refs(&g[..1]);
        bencher.bench(|| {
            stats::dunnett(black_box(&control), black_box(&refs), Alternative::TwoSided).unwrap()
        });
    }
}

mod rank_benches {
    use super::*;

    #[divan::bench]
    fn kruskal(bencher: Bencher) {
        let g = groups(100);
        let refs = group_refs(&g);
        bencher.bench(|| stats::kruskal(black_box(&refs)).unwrap());
    }

    #[divan::bench]
    fn friedmanchisquare(bencher: Bencher) {
        let g = groups(100);
        let refs = group_refs(&g);
        bencher.bench(|| stats::friedmanchisquare(black_box(&refs)).unwrap());
    }

    #[divan::bench]
    fn ranksums(bencher: Bencher) {
        let (x, y) = (values(200, 1), shifted(200, 2, 0.1));
        bencher.bench(|| {
            stats::ranksums(black_box(&x), black_box(&y), Alternative::TwoSided).unwrap()
        });
    }

    #[divan::bench]
    fn mannwhitneyu(bencher: Bencher) {
        let (x, y) = (values(200, 1), shifted(200, 2, 0.1));
        bencher.bench(|| {
            stats::mannwhitneyu(black_box(&x), black_box(&y), Alternative::TwoSided, true).unwrap()
        });
    }

    #[divan::bench]
    fn wilcoxon(bencher: Bencher) {
        let d = shifted(200, 1, 0.1);
        bencher.bench(|| {
            stats::wilcoxon(
                black_box(&d),
                stats::WilcoxonZero::Wilcox,
                Alternative::TwoSided,
                false,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn ansari(bencher: Bencher) {
        let (x, y) = (values(200, 1), shifted(200, 2, 0.1));
        bencher
            .bench(|| stats::ansari(black_box(&x), black_box(&y), Alternative::TwoSided).unwrap());
    }

    #[divan::bench]
    fn mood(bencher: Bencher) {
        let (x, y) = (values(200, 1), shifted(200, 2, 0.1));
        bencher.bench(|| stats::mood(black_box(&x), black_box(&y), Alternative::TwoSided).unwrap());
    }

    #[divan::bench]
    fn brunnermunzel(bencher: Bencher) {
        let (x, y) = (values(200, 1), shifted(200, 2, 0.1));
        bencher.bench(|| {
            stats::brunnermunzel(
                black_box(&x),
                black_box(&y),
                Alternative::TwoSided,
                stats::BrunnerDistribution::StudentT,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn median_test(bencher: Bencher) {
        let g = groups(100);
        let refs = group_refs(&g);
        bencher.bench(|| {
            stats::median_test(black_box(&refs), stats::MedianTies::Below, true).unwrap()
        });
    }

    #[divan::bench]
    fn quantile_test(bencher: Bencher) {
        let x = values(200, 1);
        bencher.bench(|| {
            stats::quantile_test(
                black_box(&x),
                black_box(0.1),
                black_box(0.5),
                Alternative::TwoSided,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn page_trend_test(bencher: Bencher) {
        let rows: Vec<Vec<f64>> = (0..50)
            .map(|i| {
                values(4, 30 + i)
                    .into_iter()
                    .enumerate()
                    .map(|(j, v)| v + 0.3 * j as f64)
                    .collect()
            })
            .collect();
        let refs = group_refs(&rows);
        let ranks = [1, 2, 3, 4];
        bencher.bench(|| stats::page_trend_test(black_box(&refs), black_box(&ranks)).unwrap());
    }
}

mod normality_benches {
    use super::*;

    #[divan::bench]
    fn jarque_bera(bencher: Bencher) {
        let x = values(200, 1);
        bencher.bench(|| stats::jarque_bera(black_box(&x)).unwrap());
    }

    #[divan::bench]
    fn skewtest(bencher: Bencher) {
        let x = values(200, 1);
        bencher.bench(|| stats::skewtest(black_box(&x), Alternative::TwoSided).unwrap());
    }

    #[divan::bench]
    fn kurtosistest(bencher: Bencher) {
        let x = values(200, 1);
        bencher.bench(|| stats::kurtosistest(black_box(&x), Alternative::TwoSided).unwrap());
    }

    #[divan::bench]
    fn normaltest(bencher: Bencher) {
        let x = values(200, 1);
        bencher.bench(|| stats::normaltest(black_box(&x)).unwrap());
    }

    #[divan::bench]
    fn shapiro(bencher: Bencher) {
        let x = values(200, 1);
        bencher.bench(|| stats::shapiro(black_box(&x)).unwrap());
    }

    #[divan::bench]
    fn anderson(bencher: Bencher) {
        let x = values(200, 1);
        bencher.bench(|| stats::anderson(black_box(&x), stats::AndersonFamily::Normal).unwrap());
    }

    #[divan::bench]
    fn anderson_ksamp(bencher: Bencher) {
        let g = groups(50);
        let refs = group_refs(&g);
        bencher.bench(|| {
            stats::anderson_ksamp(black_box(&refs), stats::AndersonVariant::Midrank).unwrap()
        });
    }

    #[divan::bench]
    fn cramervonmises(bencher: Bencher) {
        let x = values(200, 1);
        let null = Distribution::Normal {
            mean: 0.0,
            std_dev: 0.6,
        };
        bencher.bench(|| stats::cramervonmises(black_box(&x), black_box(null)).unwrap());
    }

    #[divan::bench]
    fn cramervonmises_2samp_asymptotic(bencher: Bencher) {
        let (x, y) = (values(200, 1), shifted(200, 2, 0.1));
        bencher.bench(|| {
            stats::cramervonmises_2samp(black_box(&x), black_box(&y), stats::CvmMethod::Asymptotic)
                .unwrap()
        });
    }

    #[divan::bench]
    fn cramervonmises_2samp_exact(bencher: Bencher) {
        let (x, y) = (values(7, 1), shifted(7, 2, 0.1));
        bencher.bench(|| {
            stats::cramervonmises_2samp(black_box(&x), black_box(&y), stats::CvmMethod::Exact)
                .unwrap()
        });
    }

    #[divan::bench]
    fn ks_1samp(bencher: Bencher) {
        let x = values(200, 1);
        let reference = Distribution::Normal {
            mean: 0.0,
            std_dev: 0.6,
        };
        bencher.bench(|| {
            stats::ks_1samp(black_box(&x), black_box(reference), Alternative::TwoSided).unwrap()
        });
    }

    #[divan::bench]
    fn ks_2samp(bencher: Bencher) {
        let (x, y) = (values(200, 1), shifted(200, 2, 0.1));
        bencher.bench(|| {
            stats::ks_2samp(black_box(&x), black_box(&y), Alternative::TwoSided).unwrap()
        });
    }

    #[divan::bench]
    fn kstest(bencher: Bencher) {
        let (x, y) = (values(200, 1), shifted(200, 2, 0.1));
        bencher
            .bench(|| stats::kstest(black_box(&x), black_box(&y), Alternative::TwoSided).unwrap());
    }

    #[divan::bench]
    fn probplot(bencher: Bencher) {
        let x = values(200, 1);
        let dist = Distribution::Normal {
            mean: 0.0,
            std_dev: 1.0,
        };
        bencher.bench(|| stats::probplot(black_box(&x), black_box(dist)).unwrap());
    }
}

mod diagnostics_benches {
    use super::*;

    #[divan::bench]
    fn boxcox_normplot(bencher: Bencher) {
        let x = positive_values(100, 1);
        bencher.bench(|| stats::boxcox_normplot(black_box(&x), -2.0, 2.0, 20).unwrap());
    }

    #[divan::bench]
    fn yeojohnson_normplot(bencher: Bencher) {
        let x = values(100, 1);
        bencher.bench(|| stats::yeojohnson_normplot(black_box(&x), -2.0, 2.0, 20).unwrap());
    }

    #[divan::bench]
    fn boxcox_normmax_pearson(bencher: Bencher) {
        let x = positive_values(100, 1);
        bencher.bench(|| {
            stats::boxcox_normmax(black_box(&x), -2.0, 2.0, stats::BoxCoxMethod::Pearson).unwrap()
        });
    }

    #[divan::bench]
    fn boxcox_normmax_mle(bencher: Bencher) {
        let x = positive_values(100, 1);
        bencher.bench(|| {
            stats::boxcox_normmax(
                black_box(&x),
                -2.0,
                2.0,
                stats::BoxCoxMethod::MaximumLikelihood,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn yeojohnson_normmax(bencher: Bencher) {
        let x = values(100, 1);
        bencher.bench(|| stats::yeojohnson_normmax(black_box(&x), -2.0, 2.0).unwrap());
    }

    #[divan::bench]
    fn ppcc_plot(bencher: Bencher) {
        let x = positive_values(20, 1);
        bencher.bench(|| {
            stats::ppcc_plot(black_box(&x), 0.5, 5.0, 5, stats::PpccFamily::Weibull).unwrap()
        });
    }

    #[divan::bench]
    fn ppcc_max(bencher: Bencher) {
        let x = positive_values(8, 1);
        bencher.bench(|| {
            stats::ppcc_max(black_box(&x), 0.5, 5.0, stats::PpccFamily::InverseGamma).unwrap()
        });
    }
}

mod frequency_benches {
    use super::*;

    fn observed() -> (Vec<f64>, Vec<f64>) {
        let observed: Vec<f64> = (0..50).map(|i| 20.0 + (i % 7) as f64).collect();
        let total: f64 = observed.iter().sum();
        let expected = vec![total / 50.0; 50];
        (observed, expected)
    }

    /// 4x5 contingency table of counts in row-major order.
    fn table() -> Vec<f64> {
        (0..20).map(|i| 5.0 + ((i * 7) % 11) as f64).collect()
    }

    #[divan::bench]
    fn chisquare(bencher: Bencher) {
        let (obs, exp) = observed();
        bencher.bench(|| stats::chisquare(black_box(&obs), black_box(&exp), 0).unwrap());
    }

    #[divan::bench]
    fn power_divergence(bencher: Bencher) {
        let (obs, exp) = observed();
        bencher.bench(|| {
            stats::power_divergence(black_box(&obs), black_box(&exp), 0, black_box(2.0 / 3.0))
                .unwrap()
        });
    }

    #[divan::bench]
    fn fisher_exact(bencher: Bencher) {
        let t = [12.0, 5.0, 7.0, 15.0];
        bencher.bench(|| stats::fisher_exact(black_box(&t), Alternative::TwoSided).unwrap());
    }

    #[divan::bench]
    fn barnard_exact(bencher: Bencher) {
        let t = [7.0, 12.0, 8.0, 3.0];
        bencher.bench(|| stats::barnard_exact(black_box(&t), Alternative::Less, true).unwrap());
    }

    #[divan::bench]
    fn boschloo_exact(bencher: Bencher) {
        let t = [7.0, 12.0, 8.0, 3.0];
        bencher.bench(|| stats::boschloo_exact(black_box(&t), Alternative::Less).unwrap());
    }

    #[divan::bench]
    fn poisson_means_test(bencher: Bencher) {
        bencher.bench(|| {
            stats::poisson_means_test(
                black_box(10.0),
                black_box(100.0),
                black_box(18.0),
                black_box(100.0),
                0.0,
                Alternative::TwoSided,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn chi2_contingency(bencher: Bencher) {
        let t = table();
        bencher.bench(|| stats::chi2_contingency(black_box(&t), 4, 5, true).unwrap());
    }

    #[divan::bench]
    fn margins(bencher: Bencher) {
        let t = table();
        bencher.bench(|| contingency::margins(black_box(&t), 4, 5).unwrap());
    }

    #[divan::bench]
    fn expected_freq(bencher: Bencher) {
        let t = table();
        bencher.bench(|| contingency::expected_freq(black_box(&t), 4, 5).unwrap());
    }

    #[divan::bench]
    fn association(bencher: Bencher) {
        let t = table();
        bencher.bench(|| {
            contingency::association(black_box(&t), 4, 5, contingency::Association::Cramer, false)
                .unwrap()
        });
    }

    #[divan::bench]
    fn odds_ratio(bencher: Bencher) {
        let t = [12.0, 5.0, 7.0, 15.0];
        bencher.bench(|| contingency::odds_ratio(black_box(&t)).unwrap());
    }

    #[divan::bench]
    fn relative_risk(bencher: Bencher) {
        bencher.bench(|| {
            contingency::relative_risk(
                black_box(12.0),
                black_box(40.0),
                black_box(7.0),
                black_box(45.0),
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn crosstab(bencher: Bencher) {
        let a: Vec<f64> = (0..200).map(|i| (i % 4) as f64).collect();
        let b: Vec<f64> = (0..200).map(|i| ((i * 7) % 5) as f64).collect();
        bencher.bench(|| contingency::crosstab(black_box(&a), black_box(&b)).unwrap());
    }
}

mod resampling_benches {
    use super::*;

    fn mean_difference(x: &[f64], y: &[f64]) -> sci::stats::Result<f64> {
        Ok(stats::mean(x)? - stats::mean(y)?)
    }

    #[divan::bench]
    fn permutation_test(bencher: Bencher) {
        let (x, y) = (values(5, 1), shifted(5, 2, 0.3));
        bencher.bench(|| {
            stats::permutation_test(
                black_box(&x),
                black_box(&y),
                mean_difference,
                Alternative::TwoSided,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn monte_carlo_test(bencher: Bencher) {
        let x = shifted(30, 1, 0.1);
        let draws: Vec<Vec<f64>> = (0..99).map(|s| values(30, 100 + s)).collect();
        let refs = group_refs(&draws);
        bencher.bench(|| {
            stats::monte_carlo_test(
                black_box(&x),
                black_box(&refs),
                stats::mean,
                Alternative::Greater,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn bootstrap(bencher: Bencher) {
        let x = values(50, 1);
        let indices: Vec<Vec<usize>> = (0..99).map(|s| resample(50, 100 + s)).collect();
        let refs: Vec<&[usize]> = indices.iter().map(Vec::as_slice).collect();
        bencher.bench(|| {
            stats::bootstrap(
                black_box(&x),
                black_box(&refs),
                stats::mean,
                0.95,
                stats::BootstrapMethod::Percentile,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn bws_test(bencher: Bencher) {
        let (x, y) = (values(5, 1), shifted(5, 2, 0.3));
        bencher.bench(|| {
            stats::bws_test(black_box(&x), black_box(&y), Alternative::TwoSided).unwrap()
        });
    }

    #[divan::bench]
    fn power(bencher: Bencher) {
        let samples: Vec<Vec<f64>> = (0..20).map(|s| shifted(30, 100 + s, 0.2)).collect();
        let refs = group_refs(&samples);
        bencher.bench(|| {
            stats::power(
                black_box(&refs),
                |x| Ok(stats::ttest_1samp(x, 0.0, Alternative::Greater)?.pvalue),
                0.05,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn sobol_indices(bencher: Bencher) {
        let f_a = values(256, 1);
        let f_b = values(256, 2);
        let f_ab: Vec<Vec<f64>> = (0..3).map(|s| values(256, 3 + s)).collect();
        let refs = group_refs(&f_ab);
        bencher.bench(|| {
            stats::sobol_indices(black_box(&f_a), black_box(&f_b), black_box(&refs)).unwrap()
        });
    }
}

mod misc_benches {
    use super::*;

    #[divan::bench]
    fn logrank(bencher: Bencher) {
        let x = positive_values(100, 1);
        let y = positive_values(100, 2);
        let x_event: Vec<bool> = values(100, 3).iter().map(|v| *v > -0.6).collect();
        let y_event: Vec<bool> = values(100, 4).iter().map(|v| *v > -0.6).collect();
        bencher.bench(|| {
            stats::logrank(
                black_box(&x),
                black_box(&x_event),
                black_box(&y),
                black_box(&y_event),
                Alternative::TwoSided,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn wasserstein_distance_nd(bencher: Bencher) {
        let a = points(30, 2, 1);
        let b = points(30, 2, 2);
        let (ar, br) = (point_refs(&a), point_refs(&b));
        bencher.bench(|| stats::wasserstein_distance_nd(black_box(&ar), black_box(&br)).unwrap());
    }
}
