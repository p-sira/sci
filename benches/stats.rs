//! Benchmarks covering the public API of `sci::stats` (descriptive statistics,
//! distributions, transformations, binning, ranking, robust regression,
//! estimation, fitting, QMC and multivariate densities).
//!
//! Hypothesis tests are covered by a separate benchmark target.

mod common;

use common::{linspace, points, positive_values, spd_matrix, unit_values, values};
use divan::{Bencher, black_box};
use sci::stats;

fn main() {
    divan::main();
}

/// Sample size used for most one-dimensional statistics.
const N: usize = 1_000;
/// Number of scalar evaluation points for distribution functions.
const GRID: usize = 128;
/// Number of probabilities for (possibly iterative) quantile functions.
const PPF_GRID: usize = 16;

mod descriptive_benches {
    use super::*;

    #[divan::bench]
    fn mean(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::descriptive::mean(black_box(&x)).unwrap());
    }

    #[divan::bench]
    fn variance(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::descriptive::variance(black_box(&x), 1).unwrap());
    }

    #[divan::bench]
    fn std(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::descriptive::std(black_box(&x), 1).unwrap());
    }

    #[divan::bench]
    fn sem(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::descriptive::sem(black_box(&x), 1).unwrap());
    }

    #[divan::bench]
    fn variation(bencher: Bencher) {
        let x = positive_values(N, 1);
        bencher.bench(|| stats::descriptive::variation(black_box(&x), 1).unwrap());
    }

    #[divan::bench]
    fn quantile(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::descriptive::quantile(black_box(&x), 0.3).unwrap());
    }

    #[divan::bench]
    fn scoreatpercentile(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::descriptive::scoreatpercentile(black_box(&x), 75.0).unwrap());
    }

    #[divan::bench]
    fn median(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::descriptive::median(black_box(&x)).unwrap());
    }

    #[divan::bench]
    fn iqr(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::descriptive::iqr(black_box(&x)).unwrap());
    }

    #[divan::bench]
    fn median_abs_deviation(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::descriptive::median_abs_deviation(black_box(&x), 1.0).unwrap());
    }

    #[divan::bench]
    fn gmean(bencher: Bencher) {
        let x = positive_values(N, 1);
        bencher.bench(|| stats::descriptive::gmean(black_box(&x)).unwrap());
    }

    #[divan::bench]
    fn hmean(bencher: Bencher) {
        let x = positive_values(N, 1);
        bencher.bench(|| stats::descriptive::hmean(black_box(&x)).unwrap());
    }

    #[divan::bench]
    fn gstd(bencher: Bencher) {
        let x = positive_values(N, 1);
        bencher.bench(|| stats::descriptive::gstd(black_box(&x), 1).unwrap());
    }

    #[divan::bench]
    fn moment(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::descriptive::moment(black_box(&x), 3).unwrap());
    }

    #[divan::bench]
    fn skew(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::descriptive::skew(black_box(&x), false).unwrap());
    }

    #[divan::bench]
    fn kurtosis(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::descriptive::kurtosis(black_box(&x), true, false).unwrap());
    }

    #[divan::bench]
    fn zscore(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::descriptive::zscore(black_box(&x), 1).unwrap());
    }

    #[divan::bench]
    fn zmap(bencher: Bencher) {
        let scores = values(N, 1);
        let compare = values(N, 2);
        bencher.bench(|| {
            stats::descriptive::zmap(black_box(&scores), black_box(&compare), 1).unwrap()
        });
    }

    #[divan::bench]
    fn gzscore(bencher: Bencher) {
        let x = positive_values(N, 1);
        bencher.bench(|| stats::descriptive::gzscore(black_box(&x), 1).unwrap());
    }

    #[divan::bench]
    fn trimboth(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::descriptive::trimboth(black_box(&x), 0.1).unwrap());
    }

    #[divan::bench]
    fn trim_mean(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::descriptive::trim_mean(black_box(&x), 0.1).unwrap());
    }

    #[divan::bench]
    fn mode(bencher: Bencher) {
        // Discretized data so that the mode is meaningful.
        let x: Vec<f64> = values(N, 1).iter().map(|v| (v * 20.0).round()).collect();
        bencher.bench(|| stats::descriptive::mode(black_box(&x)).unwrap());
    }

    #[divan::bench]
    fn describe(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::descriptive::describe(black_box(&x), 1, false).unwrap());
    }
}

mod extras_benches {
    use super::*;

    const LIMITS: Option<(f64, f64)> = Some((-0.8, 0.8));

    #[divan::bench]
    fn tmean(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::descriptive::tmean(black_box(&x), LIMITS, (true, true)).unwrap());
    }

    #[divan::bench]
    fn tvar(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::descriptive::tvar(black_box(&x), LIMITS, (true, true), 1).unwrap());
    }

    #[divan::bench]
    fn tstd(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::descriptive::tstd(black_box(&x), LIMITS, (true, true), 1).unwrap());
    }

    #[divan::bench]
    fn tsem(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::descriptive::tsem(black_box(&x), LIMITS, (true, true), 1).unwrap());
    }

    #[divan::bench]
    fn tmin(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::descriptive::tmin(black_box(&x), Some(-0.5), true).unwrap());
    }

    #[divan::bench]
    fn tmax(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::descriptive::tmax(black_box(&x), Some(0.5), true).unwrap());
    }

    #[divan::bench]
    fn trim1(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| {
            stats::descriptive::trim1(black_box(&x), 0.1, stats::descriptive::Tail::Upper).unwrap()
        });
    }

    #[divan::bench]
    fn kstat(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::descriptive::kstat(black_box(&x), 4).unwrap());
    }

    #[divan::bench]
    fn kstatvar(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::descriptive::kstatvar(black_box(&x), 2).unwrap());
    }

    const COMBINE_METHODS: [stats::inference::CombineMethod; 5] = [
        stats::inference::CombineMethod::Fisher,
        stats::inference::CombineMethod::Pearson,
        stats::inference::CombineMethod::Tippett,
        stats::inference::CombineMethod::Stouffer,
        stats::inference::CombineMethod::MudholkarGeorge,
    ];

    #[divan::bench(args = COMBINE_METHODS)]
    fn combine_pvalues(bencher: Bencher, method: stats::inference::CombineMethod) {
        let p = unit_values(256, 3);
        bencher.bench(|| stats::inference::combine_pvalues(black_box(&p), method, None).unwrap());
    }

    #[divan::bench]
    fn sigmaclip(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::descriptive::sigmaclip(black_box(&x), 1.5, 1.5).unwrap());
    }
}

mod distribution_benches {
    use super::*;
    use stats::distributions::Distribution;

    const FAMILIES: [&str; 5] = ["normal", "gamma", "beta", "student_t", "levy"];

    fn family(name: &str) -> Distribution {
        match name {
            "normal" => Distribution::Normal {
                mean: 0.5,
                std_dev: 1.5,
            },
            "gamma" => Distribution::Gamma {
                shape_value: 2.5,
                scale: 1.5,
            },
            "beta" => Distribution::Beta { a: 2.0, b: 3.5 },
            "student_t" => Distribution::StudentT { df: 4.5 },
            "levy" => Distribution::Levy {
                location: 0.0,
                scale: 1.0,
            },
            _ => unreachable!(),
        }
    }

    /// Evaluation points inside the support of each family.
    fn support_grid(name: &str) -> Vec<f64> {
        match name {
            "normal" | "student_t" => linspace(-4.0, 4.0, GRID),
            "beta" => linspace(0.005, 0.995, GRID),
            _ => linspace(0.05, 10.0, GRID),
        }
    }

    fn eval(
        bencher: Bencher,
        distribution: impl Into<stats::distributions::DistributionSpec> + Copy,
        xs: Vec<f64>,
        f: fn(stats::distributions::DistributionSpec, f64) -> sci::stats::Result<f64>,
    ) {
        let d: stats::distributions::DistributionSpec = distribution.into();
        bencher.bench(|| {
            black_box(&xs)
                .iter()
                .map(|&x| f(black_box(d), x).unwrap())
                .sum::<f64>()
        });
    }

    #[divan::bench(args = FAMILIES)]
    fn pdf(bencher: Bencher, name: &str) {
        eval(
            bencher,
            family(name),
            support_grid(name),
            stats::distributions::pdf,
        );
    }

    #[divan::bench(args = FAMILIES)]
    fn logpdf(bencher: Bencher, name: &str) {
        eval(
            bencher,
            family(name),
            support_grid(name),
            stats::distributions::logpdf,
        );
    }

    #[divan::bench(args = FAMILIES)]
    fn cdf(bencher: Bencher, name: &str) {
        eval(
            bencher,
            family(name),
            support_grid(name),
            stats::distributions::cdf,
        );
    }

    #[divan::bench(args = FAMILIES)]
    fn sf(bencher: Bencher, name: &str) {
        eval(
            bencher,
            family(name),
            support_grid(name),
            stats::distributions::sf,
        );
    }

    #[divan::bench(args = FAMILIES)]
    fn ppf(bencher: Bencher, name: &str) {
        // Iterative inversion is expensive for some families: use a small grid.
        eval(
            bencher,
            family(name),
            linspace(0.005, 0.995, PPF_GRID),
            stats::distributions::ppf,
        );
    }

    #[divan::bench]
    fn normal_pdf(bencher: Bencher) {
        let xs = linspace(-4.0, 4.0, GRID);
        bencher.bench(|| {
            black_box(&xs)
                .iter()
                .map(|&x| stats::distributions::normal_pdf(x, 0.5, 1.5).unwrap())
                .sum::<f64>()
        });
    }

    #[divan::bench]
    fn normal_cdf(bencher: Bencher) {
        let xs = linspace(-4.0, 4.0, GRID);
        bencher.bench(|| {
            black_box(&xs)
                .iter()
                .map(|&x| stats::distributions::normal_cdf(x, 0.5, 1.5).unwrap())
                .sum::<f64>()
        });
    }

    #[divan::bench]
    fn normal_ppf(bencher: Bencher) {
        let ps = linspace(0.005, 0.995, GRID);
        bencher.bench(|| {
            black_box(&ps)
                .iter()
                .map(|&p| stats::distributions::normal_ppf(p, 0.5, 1.5).unwrap())
                .sum::<f64>()
        });
    }

    #[divan::bench]
    fn normal_cdf_ppf(bencher: Bencher) {
        let xs = linspace(-4.0, 4.0, GRID);
        bencher.bench(|| {
            black_box(&xs)
                .iter()
                .map(|&x| {
                    let p = stats::distributions::normal_cdf(x, 0.0, 1.0).unwrap();
                    stats::distributions::normal_ppf(p, 0.0, 1.0).unwrap()
                })
                .sum::<f64>()
        });
    }
}

mod composition_benches {
    use super::*;
    use stats::distributions::Distribution;

    const NORMAL: Distribution = Distribution::Normal {
        mean: 0.0,
        std_dev: 1.0,
    };

    #[divan::bench]
    fn truncate(bencher: Bencher) {
        bencher.bench(|| {
            stats::distributions::truncate(black_box(NORMAL), black_box(-1.0), 2.0).unwrap()
        });
    }

    #[divan::bench]
    fn truncated_cdf(bencher: Bencher) {
        let d = stats::distributions::truncate(NORMAL, -1.0, 2.0).unwrap();
        let xs = linspace(-0.99, 1.99, GRID);
        bencher.bench(|| {
            black_box(&xs)
                .iter()
                .map(|&x| stats::distributions::cdf(black_box(d), x).unwrap())
                .sum::<f64>()
        });
    }

    #[divan::bench]
    fn order_statistic(bencher: Bencher) {
        bencher.bench(|| {
            stats::distributions::order_statistic(black_box(NORMAL), black_box(3), 7).unwrap()
        });
    }

    #[divan::bench]
    fn order_statistic_pdf(bencher: Bencher) {
        let d = stats::distributions::order_statistic(NORMAL, 3, 7).unwrap();
        let xs = linspace(-3.0, 3.0, GRID);
        bencher.bench(|| {
            black_box(&xs)
                .iter()
                .map(|&x| stats::distributions::pdf(black_box(d), x).unwrap())
                .sum::<f64>()
        });
    }

    fn custom() -> sci::stats::Result<stats::distributions::DistributionSpec> {
        stats::distributions::make_distribution(
            0.0,
            1.0,
            |x| Ok(2.0 * x),
            |x| Ok(x * x),
            |x| Ok(1.0 - x * x),
            |p| Ok(p.sqrt()),
        )
    }

    #[divan::bench]
    fn make_distribution(bencher: Bencher) {
        bencher.bench(|| custom().unwrap());
    }

    #[divan::bench]
    fn custom_ppf(bencher: Bencher) {
        let d = custom().unwrap();
        let ps = linspace(0.005, 0.995, GRID);
        bencher.bench(|| {
            black_box(&ps)
                .iter()
                .map(|&p| stats::distributions::ppf(black_box(d), p).unwrap())
                .sum::<f64>()
        });
    }
}

mod transformation_benches {
    use super::*;
    use std::f64::consts::PI;

    fn angles() -> Vec<f64> {
        values(N, 4).iter().map(|v| 0.5 + v * PI).collect()
    }

    #[divan::bench]
    fn circmean(bencher: Bencher) {
        let x = angles();
        bencher.bench(|| stats::descriptive::circmean(black_box(&x), -PI, PI).unwrap());
    }

    #[divan::bench]
    fn circvar(bencher: Bencher) {
        let x = angles();
        bencher.bench(|| stats::descriptive::circvar(black_box(&x), -PI, PI).unwrap());
    }

    #[divan::bench]
    fn circstd(bencher: Bencher) {
        let x = angles();
        bencher.bench(|| stats::descriptive::circstd(black_box(&x), -PI, PI, false).unwrap());
    }

    #[divan::bench]
    fn entropy(bencher: Bencher) {
        let pk = unit_values(N, 1);
        let qk = unit_values(N, 2);
        bencher.bench(|| {
            stats::descriptive::entropy(black_box(&pk), Some(black_box(&qk)), Some(2.0)).unwrap()
        });
    }

    #[divan::bench]
    fn boxcox(bencher: Bencher) {
        let x = positive_values(N, 1);
        bencher.bench(|| stats::transformations::boxcox(black_box(&x), 0.5).unwrap());
    }

    #[divan::bench]
    fn yeojohnson(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::transformations::yeojohnson(black_box(&x), 0.5).unwrap());
    }

    #[divan::bench]
    fn boxcox_llf(bencher: Bencher) {
        let x = positive_values(N, 1);
        bencher.bench(|| stats::transformations::boxcox_llf(0.5, black_box(&x)).unwrap());
    }

    #[divan::bench]
    fn yeojohnson_llf(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::transformations::yeojohnson_llf(0.5, black_box(&x)).unwrap());
    }

    #[divan::bench(args = [1.5, 1e-18])]
    fn pmean(bencher: Bencher, power: f64) {
        let x = positive_values(N, 1);
        bencher.bench(|| stats::descriptive::pmean(black_box(&x), black_box(power)).unwrap());
    }

    #[divan::bench]
    fn expectile(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::descriptive::expectile(black_box(&x), 0.3).unwrap());
    }

    #[divan::bench]
    fn wasserstein_distance(bencher: Bencher) {
        let a = values(N, 1);
        let b = values(N, 2);
        bencher.bench(|| {
            stats::distances::wasserstein_distance(black_box(&a), black_box(&b)).unwrap()
        });
    }

    #[divan::bench]
    fn energy_distance(bencher: Bencher) {
        let a = values(N, 1);
        let b = values(N, 2);
        bencher.bench(|| stats::distances::energy_distance(black_box(&a), black_box(&b)).unwrap());
    }
}

mod binning_benches {
    use super::*;

    #[divan::bench]
    fn binned_statistic(bencher: Bencher) {
        let x = values(N, 1);
        let v = values(N, 2);
        let edges = linspace(-1.0, 1.0, 21);
        bencher.bench(|| {
            stats::binning::binned_statistic(
                black_box(&x),
                black_box(&v),
                black_box(&edges),
                stats::binning::BinStatistic::Mean,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn binned_statistic_median(bencher: Bencher) {
        let x = values(N, 1);
        let v = values(N, 2);
        let edges = linspace(-1.0, 1.0, 21);
        bencher.bench(|| {
            stats::binning::binned_statistic(
                black_box(&x),
                black_box(&v),
                black_box(&edges),
                stats::binning::BinStatistic::Median,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn binned_statistic_2d(bencher: Bencher) {
        let x = values(N, 1);
        let y = values(N, 2);
        let v = values(N, 3);
        let edges = linspace(-1.0, 1.0, 11);
        bencher.bench(|| {
            stats::binning::binned_statistic_2d(
                black_box(&x),
                black_box(&y),
                black_box(&v),
                black_box(&edges),
                black_box(&edges),
                stats::binning::BinStatistic::Sum,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn binned_statistic_dd(bencher: Bencher) {
        let p = points(N, 3, 1);
        let v = values(N, 2);
        let edges = vec![linspace(-1.0, 1.0, 6); 3];
        bencher.bench(|| {
            stats::binning::binned_statistic_dd(
                black_box(&p),
                black_box(&v),
                black_box(&edges),
                stats::binning::BinStatistic::Std,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn cumfreq(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::binning::cumfreq(black_box(&x), 20, -1.0, 1.0).unwrap());
    }

    #[divan::bench]
    fn relfreq(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::binning::relfreq(black_box(&x), 20, -1.0, 1.0).unwrap());
    }
}

mod ranking_benches {
    use super::*;

    /// Data with ties so that tie handling is exercised.
    fn tied() -> Vec<f64> {
        values(N, 1).iter().map(|v| (v * 100.0).round()).collect()
    }

    #[divan::bench]
    fn rankdata(bencher: Bencher) {
        let x = tied();
        bencher.bench(|| {
            stats::ranking::rankdata(black_box(&x), stats::ranking::RankMethod::Average).unwrap()
        });
    }

    #[divan::bench]
    fn tiecorrect(bencher: Bencher) {
        let ranks = stats::ranking::rankdata(&tied(), stats::ranking::RankMethod::Average).unwrap();
        bencher.bench(|| stats::ranking::tiecorrect(black_box(&ranks)).unwrap());
    }

    #[divan::bench]
    fn percentileofscore(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| {
            stats::ranking::percentileofscore(
                black_box(&x),
                0.25,
                stats::ranking::PercentileKind::Rank,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn false_discovery_control(bencher: Bencher) {
        let p = unit_values(N, 1);
        bencher.bench(|| {
            stats::inference::false_discovery_control(
                black_box(&p),
                stats::inference::FdrMethod::BenjaminiHochberg,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn ecdf(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::ranking::ecdf(black_box(&x)).unwrap());
    }

    #[divan::bench]
    fn estimated_cdf(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| {
            stats::ranking::estimated_cdf(
                black_box(&x),
                0.25,
                stats::ranking::EstimatedCdfMethod::Linear,
            )
            .unwrap()
        });
    }
}

mod robust_benches {
    use super::*;

    /// Noisy linear relationship with `n` pairs.
    fn line(n: usize) -> (Vec<f64>, Vec<f64>) {
        let x = linspace(0.0, 10.0, n);
        let noise = values(n, 5);
        let y = x
            .iter()
            .zip(&noise)
            .map(|(x, e)| 2.0 * x + 1.0 + e)
            .collect();
        (x, y)
    }

    #[divan::bench]
    fn theilslopes(bencher: Bencher) {
        let (x, y) = line(128);
        bencher.bench(|| {
            stats::regression::theilslopes(
                black_box(&x),
                black_box(&y),
                0.95,
                stats::regression::TheilMethod::Separate,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn siegelslopes(bencher: Bencher) {
        let (x, y) = line(128);
        bencher.bench(|| {
            stats::regression::siegelslopes(
                black_box(&x),
                black_box(&y),
                stats::regression::SiegelMethod::Hierarchical,
            )
            .unwrap()
        });
    }
}

mod utility_benches {
    use super::*;

    #[divan::bench]
    fn binomtest(bencher: Bencher) {
        bencher.bench(|| {
            stats::inference::binomtest(
                black_box(430),
                black_box(1_000),
                0.45,
                stats::inference::Alternative::TwoSided,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn directional_stats(bencher: Bencher) {
        let obs: Vec<Vec<f64>> = points(500, 3, 1)
            .into_iter()
            .map(|mut p| {
                p[2] += 2.0;
                p
            })
            .collect();
        bencher.bench(|| stats::descriptive::directional_stats(black_box(&obs), true).unwrap());
    }

    #[divan::bench]
    fn differential_entropy(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::descriptive::differential_entropy(black_box(&x), 8, None).unwrap());
    }

    #[divan::bench]
    fn lmoment(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::descriptive::lmoment(black_box(&x), 4, true).unwrap());
    }
}

mod estimation_benches {
    use super::*;

    #[divan::bench]
    fn mvsdist(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::estimation::mvsdist(black_box(&x), 0.9).unwrap());
    }

    #[divan::bench]
    fn bayes_mvs(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::estimation::bayes_mvs(black_box(&x), 0.9).unwrap());
    }
}

mod fitting_benches {
    use super::*;

    const FAMILIES: [stats::distributions::FitFamily; 3] = [
        stats::distributions::FitFamily::Normal,
        stats::distributions::FitFamily::Exponential,
        stats::distributions::FitFamily::Uniform,
    ];

    #[divan::bench(args = FAMILIES)]
    fn fit(bencher: Bencher, family: stats::distributions::FitFamily) {
        let x = positive_values(N, 1);
        bencher.bench(|| stats::distributions::fit(black_box(&x), family).unwrap());
    }

    #[divan::bench]
    fn goodness_of_fit(bencher: Bencher) {
        let x = values(64, 1);
        let draws: Vec<Vec<f64>> = (0..32).map(|s| values(64, 100 + s)).collect();
        let null: Vec<&[f64]> = draws.iter().map(Vec::as_slice).collect();
        bencher.bench(|| {
            stats::goodness::goodness_of_fit(
                black_box(&x),
                black_box(&null),
                stats::distributions::FitFamily::Normal,
                stats::goodness::GoodnessStatistic::KolmogorovSmirnov,
            )
            .unwrap()
        });
    }
}

mod characteristic_benches {
    use super::*;

    #[divan::bench]
    fn epps_singleton_2samp(bencher: Bencher) {
        let x = values(100, 1);
        let y: Vec<f64> = values(120, 2).iter().map(|v| 0.2 + 1.2 * v).collect();
        let t = [0.4, 0.8];
        bencher.bench(|| {
            stats::inference::epps_singleton_2samp(black_box(&x), black_box(&y), black_box(&t))
                .unwrap()
        });
    }
}

mod qmc_benches {
    use super::*;
    use stats::qmc;

    fn unit_points(n: usize, dim: usize) -> Vec<Vec<f64>> {
        unit_values(n * dim, 9)
            .chunks(dim)
            .map(<[f64]>::to_vec)
            .collect()
    }

    const METHODS: [qmc::Discrepancy; 4] = [
        qmc::Discrepancy::Centered,
        qmc::Discrepancy::WrapAround,
        qmc::Discrepancy::Mixture,
        qmc::Discrepancy::L2Star,
    ];

    #[divan::bench(args = METHODS)]
    fn discrepancy(bencher: Bencher, method: qmc::Discrepancy) {
        let p = unit_points(128, 4);
        bencher.bench(|| qmc::discrepancy(black_box(&p), method, false).unwrap());
    }

    #[divan::bench]
    fn update_discrepancy(bencher: Bencher) {
        let p = unit_points(128, 4);
        let initial = qmc::discrepancy(&p, qmc::Discrepancy::Centered, false).unwrap();
        let new_point = [0.3, 0.6, 0.1, 0.9];
        bencher.bench(|| {
            qmc::update_discrepancy(black_box(&new_point), black_box(&p), initial).unwrap()
        });
    }

    #[divan::bench]
    fn scale(bencher: Bencher) {
        let p = unit_points(256, 4);
        let lower = [-1.0, 0.0, 2.0, -5.0];
        let upper = [1.0, 10.0, 3.0, 5.0];
        bencher.bench(|| qmc::scale(black_box(&p), &lower, &upper, false).unwrap());
    }

    const GEOMETRIC: [qmc::GeometricDiscrepancy; 2] = [
        qmc::GeometricDiscrepancy::MinimumDistance,
        qmc::GeometricDiscrepancy::MinimumSpanningTree,
    ];

    #[divan::bench(args = GEOMETRIC)]
    fn geometric_discrepancy(bencher: Bencher, method: qmc::GeometricDiscrepancy) {
        let p = unit_points(128, 4);
        bencher.bench(|| qmc::geometric_discrepancy(black_box(&p), method).unwrap());
    }
}

mod multivariate_benches {
    use super::*;

    const DIM: usize = 8;

    fn mvn_inputs() -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
        (values(DIM, 1), values(DIM, 2), spd_matrix(DIM))
    }

    #[divan::bench]
    fn multivariate_normal_pdf(bencher: Bencher) {
        let (x, loc, cov) = mvn_inputs();
        bencher.bench(|| {
            stats::multivariate::multivariate_normal_pdf(
                black_box(&x),
                black_box(&loc),
                black_box(&cov),
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn multivariate_normal_logpdf(bencher: Bencher) {
        let (x, loc, cov) = mvn_inputs();
        bencher.bench(|| {
            stats::multivariate::multivariate_normal_logpdf(
                black_box(&x),
                black_box(&loc),
                black_box(&cov),
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn multivariate_t_pdf(bencher: Bencher) {
        let (x, loc, cov) = mvn_inputs();
        bencher.bench(|| {
            stats::multivariate::multivariate_t_pdf(
                black_box(&x),
                black_box(&loc),
                black_box(&cov),
                5.0,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn multivariate_t_logpdf(bencher: Bencher) {
        let (x, loc, cov) = mvn_inputs();
        bencher.bench(|| {
            stats::multivariate::multivariate_t_logpdf(
                black_box(&x),
                black_box(&loc),
                black_box(&cov),
                5.0,
            )
            .unwrap()
        });
    }

    fn simplex_point() -> Vec<f64> {
        let w = unit_values(DIM, 3);
        let total: f64 = w.iter().sum();
        let mut x: Vec<f64> = w.iter().map(|v| v / total).collect();
        // Make the sum exactly one up to rounding.
        let rest: f64 = x[1..].iter().sum();
        x[0] = 1.0 - rest;
        x
    }

    #[divan::bench]
    fn dirichlet_pdf(bencher: Bencher) {
        let x = simplex_point();
        let alpha = positive_values(DIM, 4);
        bencher.bench(|| {
            stats::multivariate::dirichlet_pdf(black_box(&x), black_box(&alpha)).unwrap()
        });
    }

    #[divan::bench]
    fn dirichlet_logpdf(bencher: Bencher) {
        let x = simplex_point();
        let alpha = positive_values(DIM, 4);
        bencher.bench(|| {
            stats::multivariate::dirichlet_logpdf(black_box(&x), black_box(&alpha)).unwrap()
        });
    }

    fn multinomial_inputs() -> (Vec<f64>, Vec<f64>) {
        let counts: Vec<f64> = (0..DIM).map(|i| (3 + 2 * i) as f64).collect();
        (counts, simplex_point())
    }

    #[divan::bench]
    fn multinomial_pmf(bencher: Bencher) {
        let (counts, p) = multinomial_inputs();
        bencher.bench(|| {
            stats::multivariate::multinomial_pmf(black_box(&counts), black_box(&p)).unwrap()
        });
    }

    #[divan::bench]
    fn multinomial_logpmf(bencher: Bencher) {
        let (counts, p) = multinomial_inputs();
        bencher.bench(|| {
            stats::multivariate::multinomial_logpmf(black_box(&counts), black_box(&p)).unwrap()
        });
    }

    #[divan::bench]
    fn categorical_pmf(bencher: Bencher) {
        let w = positive_values(64, 5);
        bencher.bench(|| {
            (0..64)
                .map(|k| stats::multivariate::categorical_pmf(black_box(&w), k as f64).unwrap())
                .sum::<f64>()
        });
    }

    #[divan::bench]
    fn categorical_cdf(bencher: Bencher) {
        let w = positive_values(64, 5);
        bencher.bench(|| {
            (0..64)
                .map(|k| stats::multivariate::categorical_cdf(black_box(&w), k as f64).unwrap())
                .sum::<f64>()
        });
    }
}
