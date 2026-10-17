mod common;
use common::concat_pair;
use common::vector as numbers;

use sci::{Error, Result, stats::*};

fn alternative(p: f64) -> Alternative {
    match p as usize {
        0 => Alternative::TwoSided,
        1 => Alternative::Less,
        _ => Alternative::Greater,
    }
}
fn test(r: TestResult) -> Vec<f64> {
    vec![r.statistic, r.pvalue]
}
fn ttest(r: TTestResult) -> Vec<f64> {
    vec![r.statistic, r.pvalue, r.df]
}
fn distribution(name: &str, p: &[f64]) -> Distribution {
    match name {
        "Erlang" => Distribution::Erlang {
            shape_value: p[0],
            scale: p[1],
        },
        "DiscreteUniform" => Distribution::DiscreteUniform {
            low: p[0],
            high: p[1],
        },
        "Levy" => Distribution::Levy {
            location: p[0],
            scale: p[1],
        },
        "Normal" => Distribution::Normal {
            mean: p[0],
            std_dev: p[1],
        },
        "Beta" => Distribution::Beta { a: p[0], b: p[1] },
        "Gamma" => Distribution::Gamma {
            shape_value: p[0],
            scale: p[1],
        },
        "ChiSquared" => Distribution::ChiSquared { df: p[0] },
        "Chi" => Distribution::Chi { df: p[0] },
        "StudentT" => Distribution::StudentT { df: p[0] },
        "F" => Distribution::F {
            numerator_df: p[0],
            denominator_df: p[1],
        },
        "Exponential" => Distribution::Exponential { scale: p[0] },
        "Cauchy" => Distribution::Cauchy {
            location: p[0],
            scale: p[1],
        },
        "Laplace" => Distribution::Laplace {
            location: p[0],
            scale: p[1],
        },
        "LogNormal" => Distribution::LogNormal {
            log_mean: p[0],
            log_std: p[1],
        },
        "Uniform" => Distribution::Uniform {
            low: p[0],
            high: p[1],
        },
        "Weibull" => Distribution::Weibull {
            shape_value: p[0],
            scale: p[1],
        },
        "Pareto" => Distribution::Pareto {
            minimum: p[0],
            shape_value: p[1],
        },
        "InverseGamma" => Distribution::InverseGamma {
            shape_value: p[0],
            scale: p[1],
        },
        "Gumbel" => Distribution::Gumbel {
            location: p[0],
            scale: p[1],
        },
        "Triangular" => Distribution::Triangular {
            low: p[0],
            high: p[1],
            mode: p[2],
        },
        "Bernoulli" => Distribution::Bernoulli { probability: p[0] },
        "Binomial" => Distribution::Binomial {
            trials: p[0],
            probability: p[1],
        },
        "Poisson" => Distribution::Poisson { rate: p[0] },
        "Geometric" => Distribution::Geometric { probability: p[0] },
        "NegativeBinomial" => Distribution::NegativeBinomial {
            successes: p[0],
            probability: p[1],
        },
        "Hypergeometric" => Distribution::Hypergeometric {
            population: p[0],
            successes: p[1],
            draws: p[2],
        },
        _ => panic!("unknown distribution {name}"),
    }
}
fn evaluate(fields: &[&str]) -> Result<Vec<f64>> {
    let (api, p, x, y) = (
        fields[1],
        numbers(fields[2]),
        numbers(fields[3]),
        numbers(fields[4]),
    );
    if !fields[6].is_empty() {
        let d = distribution(fields[6], &p);
        return Ok(vec![match api {
            "pdf" => pdf(d, x[0])?,
            "logpdf" => logpdf(d, x[0])?,
            "cdf" => cdf(d, x[0])?,
            "sf" => sf(d, x[0])?,
            "ppf" => ppf(d, x[0])?,
            _ => panic!("distribution operation"),
        }]);
    }
    let groups = if fields[5].is_empty() {
        vec![]
    } else {
        fields[5].split('/').map(numbers).collect::<Vec<_>>()
    };
    let g = groups.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let flatten_bins = |b: BinnedStatistic| {
        let mut out = b.shape.into_iter().map(|v| v as f64).collect::<Vec<_>>();
        out.extend(b.statistic);
        out.extend(b.edges.into_iter().flatten());
        out.extend(b.bin_numbers.into_iter().map(|v| v as f64));
        out
    };
    let stat = |p: f64| {
        [
            BinStatistic::Count,
            BinStatistic::Sum,
            BinStatistic::Mean,
            BinStatistic::Std,
            BinStatistic::Median,
            BinStatistic::Min,
            BinStatistic::Max,
        ][p as usize]
    };
    Ok(match api {
        "wasserstein_distance_nd" => vec![wasserstein_distance_nd(
            &g[..p[0] as usize],
            &g[p[0] as usize..],
        )?],
        "dunnett" => {
            let r = dunnett(g[0], &g[1..], alternative(p[0]))?;
            assert_eq!(r.statistic.len(), g.len() - 1);
            assert_eq!(r.pvalue.len(), r.statistic.len());
            r.statistic.into_iter().chain(r.pvalue).collect()
        }
        "multiscale_graphcorr" => {
            let n = p[0] as usize;
            let indices = g[2 * n..]
                .iter()
                .map(|row| row.iter().map(|v| *v as usize).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            let permutations = indices.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let r = multiscale_graphcorr(&g[..n], &g[n..2 * n], &permutations)?;
            assert_eq!(r.null_distribution.len(), permutations.len());
            let shape = [r.mgc_map.len(), r.mgc_map[0].len()];
            assert!(r.mgc_map.iter().all(|row| row.len() == shape[1]));
            [
                vec![r.statistic, r.pvalue, shape[0] as f64, shape[1] as f64],
                r.mgc_map.into_iter().flatten().collect(),
                vec![r.optimal_scale.0 as f64, r.optimal_scale.1 as f64],
                r.null_distribution,
            ]
            .concat()
        }
        "barnard_exact" => test(barnard_exact(
            x.as_slice().try_into().unwrap(),
            alternative(p[0]),
            p[1] != 0.,
        )?),
        "boschloo_exact" => test(boschloo_exact(
            x.as_slice().try_into().unwrap(),
            alternative(p[0]),
        )?),
        "poisson_means_test" => test(poisson_means_test(
            p[0],
            p[1],
            p[2],
            p[3],
            p[4],
            alternative(p[5]),
        )?),
        "make_distribution" | "truncate" | "order_statistic" => {
            let (d, probability) = match api {
                "make_distribution" => (
                    make_distribution(
                        0.,
                        1.,
                        |x| Ok(2. * x),
                        |x| Ok(x * x),
                        |x| Ok(1. - x * x),
                        |p| Ok(p.sqrt()),
                    )?,
                    p[0],
                ),
                "truncate" => (
                    truncate(
                        Distribution::Normal {
                            mean: 0.,
                            std_dev: 1.,
                        },
                        p[0],
                        p[1],
                    )?,
                    p[2],
                ),
                _ => (
                    order_statistic(
                        if p[2] == 0. {
                            Distribution::Normal {
                                mean: 0.,
                                std_dev: 1.,
                            }
                        } else {
                            Distribution::Uniform { low: 0., high: 1. }
                        },
                        p[0] as usize,
                        p[1] as usize,
                    )?,
                    p[3],
                ),
            };
            vec![
                pdf(d, x[0])?,
                logpdf(d, x[0])?,
                cdf(d, x[0])?,
                sf(d, x[0])?,
                ppf(d, probability)?,
            ]
        }
        "shapiro" => test(shapiro(&x)?),
        "tukey_hsd" => {
            let r = tukey_hsd(&g)?;
            assert_eq!(r.statistic.len(), g.len());
            assert_eq!(r.pvalue.len(), g.len());
            assert!(
                r.statistic
                    .iter()
                    .chain(&r.pvalue)
                    .all(|row| row.len() == g.len())
            );
            r.statistic
                .into_iter()
                .flatten()
                .chain(r.pvalue.into_iter().flatten())
                .collect()
        }
        "estimated_cdf" => vec![estimated_cdf(
            &x,
            p[0],
            [
                EstimatedCdfMethod::InvertedCdf,
                EstimatedCdfMethod::AveragedInvertedCdf,
                EstimatedCdfMethod::ClosestObservation,
                EstimatedCdfMethod::InterpolatedInvertedCdf,
                EstimatedCdfMethod::Hazen,
                EstimatedCdfMethod::Weibull,
                EstimatedCdfMethod::Linear,
                EstimatedCdfMethod::MedianUnbiased,
                EstimatedCdfMethod::NormalUnbiased,
            ][p[1] as usize],
        )?],
        "sobol_indices" => {
            let r = sobol_indices(&x, &y, &g)?;
            assert_eq!(r.first_order.len(), g.len());
            assert_eq!(r.total_order.len(), g.len());
            r.first_order.into_iter().chain(r.total_order).collect()
        }
        "power" => {
            let r = power(
                &g,
                |a| {
                    if p[1] == 0. {
                        Ok(ttest_1samp(a, 0., Alternative::Greater)?.pvalue)
                    } else {
                        sf(
                            Distribution::Normal {
                                mean: 0.,
                                std_dev: 1.,
                            },
                            mean(a)? * (a.len() as f64).sqrt(),
                        )
                    }
                },
                p[0],
            )?;
            [vec![r.power], r.pvalues].concat()
        }
        "anderson_ksamp" => test(anderson_ksamp(
            &g,
            if p[0] == 0. {
                AndersonVariant::Midrank
            } else {
                AndersonVariant::Right
            },
        )?),
        "fit" | "goodness_of_fit" => {
            let family = [
                FitFamily::Normal,
                FitFamily::Exponential,
                FitFamily::Uniform,
            ][p[0] as usize];
            if api == "goodness_of_fit" {
                let r = goodness_of_fit(
                    &x,
                    &g,
                    family,
                    if p[1] == 0. {
                        GoodnessStatistic::KolmogorovSmirnov
                    } else {
                        GoodnessStatistic::CramerVonMises
                    },
                )?;
                [vec![r.statistic, r.pvalue], r.null_distribution].concat()
            } else {
                let d = fit(&x, family)?;
                let params = match d {
                    Distribution::Normal { mean, std_dev } => vec![mean, std_dev],
                    Distribution::Exponential { scale } => vec![0., scale],
                    Distribution::Uniform { low, high } => vec![low, high],
                    _ => panic!("unexpected fitted family"),
                };
                [params, vec![cdf(d, y[0])?, ppf(d, p[1])?]].concat()
            }
        }
        "permutation_test" | "monte_carlo_test" | "bootstrap" => {
            let statistic = |a: &[f64]| match p[p.len() - 1] as usize {
                0 => mean(a),
                1 => median(a),
                _ => mean(&a.iter().map(|v| v * v).collect::<Vec<_>>()),
            };
            if api == "bootstrap" {
                let indices = g
                    .iter()
                    .map(|row| row.iter().map(|v| *v as usize).collect::<Vec<_>>())
                    .collect::<Vec<_>>();
                let rows = indices.iter().map(Vec::as_slice).collect::<Vec<_>>();
                let r = bootstrap(
                    &x,
                    &rows,
                    statistic,
                    p[0],
                    if p[1] == 0. {
                        BootstrapMethod::Percentile
                    } else {
                        BootstrapMethod::Basic
                    },
                )?;
                assert_eq!(r.bootstrap_distribution.len(), g.len());
                [
                    vec![r.statistic, r.lower, r.upper, r.standard_error],
                    r.bootstrap_distribution,
                ]
                .concat()
            } else {
                let r = if api == "permutation_test" {
                    permutation_test(
                        &x,
                        &y,
                        |a, b| Ok(statistic(a)? - statistic(b)?),
                        alternative(p[0]),
                    )?
                } else {
                    monte_carlo_test(&x, &g, statistic, alternative(p[0]))?
                };
                [vec![r.statistic, r.pvalue], r.null_distribution].concat()
            }
        }
        "bws_test" => {
            let r = bws_test(&x, &y, alternative(p[0]))?;
            [vec![r.statistic, r.pvalue], r.null_distribution].concat()
        }
        "logrank" => {
            let ex = g[0].iter().map(|v| *v != 0.).collect::<Vec<_>>();
            let ey = g[1].iter().map(|v| *v != 0.).collect::<Vec<_>>();
            test(logrank(&x, &ex, &y, &ey, alternative(p[0]))?)
        }
        "epps_singleton_2samp" => test(epps_singleton_2samp(&x, &y, &p)?),
        "anderson" => {
            let r = anderson(
                &x,
                if p[0] == 0. {
                    AndersonFamily::Normal
                } else {
                    AndersonFamily::Exponential
                },
            )?;
            [
                vec![r.statistic],
                r.critical_values.to_vec(),
                r.significance_levels.to_vec(),
                vec![r.location, r.scale],
            ]
            .concat()
        }
        "cramervonmises" => test(cramervonmises(
            &x,
            Distribution::Normal {
                mean: y[0],
                std_dev: y[1],
            },
        )?),
        "cramervonmises_2samp" => test(cramervonmises_2samp(
            &x,
            &y,
            if p[0] == 0. {
                CvmMethod::Exact
            } else {
                CvmMethod::Asymptotic
            },
        )?),
        "bayes_mvs" => {
            let r = bayes_mvs(&x, p[0])?;
            [r.mean, r.variance, r.std_dev]
                .into_iter()
                .flat_map(|r| [r.statistic, r.lower, r.upper])
                .collect()
        }
        "mvsdist" => mvsdist(&x, p[0])?.to_vec(),
        "boxcox_normmax" => vec![boxcox_normmax(
            &x,
            p[0],
            p[1],
            if p[2] == 0. {
                BoxCoxMethod::Pearson
            } else {
                BoxCoxMethod::MaximumLikelihood
            },
        )?],
        "yeojohnson_normmax" => vec![yeojohnson_normmax(&x, p[0], p[1])?],
        "ppcc_plot" | "ppcc_max" => {
            let families = [
                PpccFamily::StudentT,
                PpccFamily::ChiSquared,
                PpccFamily::Gamma,
                PpccFamily::Weibull,
                PpccFamily::InverseGamma,
            ];
            if api == "ppcc_max" {
                vec![ppcc_max(&x, p[0], p[1], families[p[2] as usize])?]
            } else {
                let r = ppcc_plot(&x, p[0], p[1], p[2] as usize, families[p[3] as usize])?;
                assert_eq!(r.shapes.len(), p[2] as usize);
                assert_eq!(r.correlations.len(), r.shapes.len());
                r.shapes.into_iter().chain(r.correlations).collect()
            }
        }
        "spearmanrho" => test(spearmanrho(&x, &y, alternative(p[0]))?),
        "somersd" => test(somersd(&x, &y, alternative(p[0]))?),
        "chatterjeexi" => test(chatterjeexi(&x, &y, p[0] != 0.)?),
        "weightedtau" => {
            let ranks = y.iter().map(|v| *v as usize).collect::<Vec<_>>();
            let rank = [
                WeightedRank::Symmetric,
                WeightedRank::Lexicographic,
                WeightedRank::Observation,
                WeightedRank::Explicit(&ranks),
            ][p[0] as usize];
            test(weightedtau(&x, g[0], rank, p[1] != 0.)?)
        }
        "mood" | "mannwhitneyu" | "brunnermunzel" | "fligner" | "alexandergovern" | "ansari"
        | "wilcoxon" | "kendalltau" | "page_trend_test" => {
            let r = match api {
                "mood" => mood(&x, &y, alternative(p[0]))?,
                "mannwhitneyu" => mannwhitneyu(&x, &y, alternative(p[0]), p[1] != 0.)?,
                "brunnermunzel" => brunnermunzel(
                    &x,
                    &y,
                    alternative(p[0]),
                    if p[1] == 0. {
                        BrunnerDistribution::StudentT
                    } else {
                        BrunnerDistribution::Normal
                    },
                )?,
                "alexandergovern" => alexandergovern(&g)?,
                "ansari" => ansari(&x, &y, alternative(p[0]))?,
                "wilcoxon" => wilcoxon(
                    &x,
                    [
                        WilcoxonZero::Wilcox,
                        WilcoxonZero::Pratt,
                        WilcoxonZero::Split,
                    ][p[0] as usize],
                    alternative(p[1]),
                    p[2] != 0.,
                )?,
                "kendalltau" => kendalltau(
                    &x,
                    &y,
                    if p[0] == 0. {
                        KendallVariant::B
                    } else {
                        KendallVariant::C
                    },
                    alternative(p[1]),
                )?,
                "page_trend_test" => {
                    page_trend_test(&g, &y.iter().map(|v| *v as usize).collect::<Vec<_>>())?
                }
                _ => fligner(
                    &g,
                    if p[0] == 0. {
                        Center::Mean
                    } else {
                        Center::Median
                    },
                )?,
            };
            vec![r.statistic, r.pvalue]
        }
        "quantile_test" => {
            let r = quantile_test(&x, p[0], p[1], alternative(p[2]))?;
            vec![
                r.statistic as f64,
                if r.count == QuantileCount::AtOrBelow {
                    1.
                } else {
                    2.
                },
                r.pvalue,
            ]
        }
        "median_test" => {
            let ties = [MedianTies::Below, MedianTies::Above, MedianTies::Ignore][p[0] as usize];
            let r = median_test(&g, ties, p[1] != 0.)?;
            assert_eq!(r.table.len(), 2);
            assert!(r.table.iter().all(|row| row.len() == g.len()));
            let mut out = vec![r.statistic, r.pvalue, r.median];
            out.extend(r.table.into_iter().flatten());
            out
        }
        "boxcox_normplot" | "yeojohnson_normplot" => {
            let r = if api == "boxcox_normplot" {
                boxcox_normplot(&x, p[0], p[1], p[2] as usize)?
            } else {
                yeojohnson_normplot(&x, p[0], p[1], p[2] as usize)?
            };
            assert_eq!(r.lambdas.len(), p[2] as usize);
            assert_eq!(r.correlations.len(), r.lambdas.len());
            r.lambdas.into_iter().chain(r.correlations).collect()
        }
        "skewtest" | "kurtosistest" | "normaltest" => {
            let r = match api {
                "skewtest" => skewtest(&x, alternative(p[0]))?,
                "kurtosistest" => kurtosistest(&x, alternative(p[0]))?,
                _ => normaltest(&x)?,
            };
            vec![r.statistic, r.pvalue]
        }
        "obrientransform" => {
            let transformed = obrientransform(&g)?;
            assert_eq!(transformed.len(), groups.len());
            for (actual, group) in transformed.iter().zip(&groups) {
                assert_eq!(actual.len(), group.len());
            }
            transformed.into_iter().flatten().collect()
        }
        "siegelslopes" => {
            let r = siegelslopes(
                &x,
                &y,
                if p[0] == 0. {
                    SiegelMethod::Hierarchical
                } else {
                    SiegelMethod::Separate
                },
            )?;
            vec![r.slope, r.intercept]
        }
        "theilslopes" => {
            let r = theilslopes(
                &x,
                &y,
                p[0],
                if p[1] == 0. {
                    TheilMethod::Separate
                } else {
                    TheilMethod::Joint
                },
            )?;
            vec![r.slope, r.intercept, r.low_slope, r.high_slope]
        }
        "ks_1samp" => {
            let r = ks_1samp(
                &x,
                Distribution::Normal {
                    mean: y[0],
                    std_dev: y[1],
                },
                alternative(p[0]),
            )?;
            vec![r.statistic, r.pvalue, r.location, r.sign as f64]
        }
        "ks_2samp" | "kstest" => {
            let r = if api == "kstest" {
                kstest(&x, &y, alternative(p[0]))?
            } else {
                ks_2samp(&x, &y, alternative(p[0]))?
            };
            vec![r.statistic, r.pvalue, r.location, r.sign as f64]
        }
        "probplot" => {
            let r = probplot(
                &x,
                Distribution::Normal {
                    mean: y[0],
                    std_dev: y[1],
                },
            )?;
            [r.theoretical, r.ordered].concat()
        }
        "abs" => vec![abs(x[0])],
        "exp" => vec![exp(x[0])],
        "log" => vec![log(x[0])],
        "tmean" => vec![tmean(&x, Some((p[0], p[1])), (p[2] != 0., p[3] != 0.))?],
        "tvar" => vec![tvar(
            &x,
            Some((p[0], p[1])),
            (p[2] != 0., p[3] != 0.),
            p[4] as usize,
        )?],
        "tstd" => vec![tstd(
            &x,
            Some((p[0], p[1])),
            (p[2] != 0., p[3] != 0.),
            p[4] as usize,
        )?],
        "tsem" => vec![tsem(
            &x,
            Some((p[0], p[1])),
            (p[2] != 0., p[3] != 0.),
            p[4] as usize,
        )?],
        "tmin" => vec![tmin(&x, Some(p[0]), p[1] != 0.)?],
        "tmax" => vec![tmax(&x, Some(p[0]), p[1] != 0.)?],
        "trim1" => trim1(&x, p[0], [Tail::Lower, Tail::Upper][p[1] as usize])?,
        "kstat" => vec![kstat(&x, p[0] as usize)?],
        "kstatvar" => vec![kstatvar(&x, p[0] as usize)?],
        "lmoment" => vec![lmoment(&x, p[0] as usize, p[1] != 0.)?],
        "sigmaclip" => {
            let r = sigmaclip(&x, p[0], p[1])?;
            [r.values, vec![r.lower, r.upper]].concat()
        }
        "differential_entropy" => vec![differential_entropy(
            &x,
            p[0] as usize,
            if p[1] == 0. { None } else { Some(p[1]) },
        )?],
        "binomtest" => test(binomtest(
            p[0] as usize,
            p[1] as usize,
            p[2],
            alternative(p[3]),
        )?),
        "combine_pvalues" => test(combine_pvalues(
            &x,
            [
                CombineMethod::Fisher,
                CombineMethod::Pearson,
                CombineMethod::Tippett,
                CombineMethod::Stouffer,
                CombineMethod::MudholkarGeorge,
            ][p[0] as usize],
            if y.is_empty() { None } else { Some(&y) },
        )?),
        "directional_stats" => {
            let r = directional_stats(&groups, p[0] != 0.)?;
            [r.mean_direction, vec![r.mean_resultant_length]].concat()
        }
        "discrepancy" => vec![qmc::discrepancy(
            &groups,
            [
                qmc::Discrepancy::Centered,
                qmc::Discrepancy::WrapAround,
                qmc::Discrepancy::Mixture,
                qmc::Discrepancy::L2Star,
            ][p[0] as usize],
            p[1] != 0.,
        )?],
        "geometric_discrepancy" => vec![qmc::geometric_discrepancy(
            &groups,
            [
                qmc::GeometricDiscrepancy::MinimumDistance,
                qmc::GeometricDiscrepancy::MinimumSpanningTree,
            ][p[0] as usize],
        )?],
        "update_discrepancy" => vec![qmc::update_discrepancy(
            &x,
            &groups,
            qmc::discrepancy(&groups, qmc::Discrepancy::Centered, true)?,
        )?],
        "scale" => qmc::scale(&groups, &x, &y, p[0] != 0.)?
            .into_iter()
            .flatten()
            .collect(),
        "multivariate_normal_pdf" => vec![multivariate_normal_pdf(&x, &y, &groups)?],
        "multivariate_normal_logpdf" => vec![multivariate_normal_logpdf(&x, &y, &groups)?],
        "multivariate_t_pdf" => vec![multivariate_t_pdf(&x, &y, &groups, p[0])?],
        "multivariate_t_logpdf" => vec![multivariate_t_logpdf(&x, &y, &groups, p[0])?],
        "dirichlet_pdf" => vec![dirichlet_pdf(&x, &y)?],
        "dirichlet_logpdf" => vec![dirichlet_logpdf(&x, &y)?],
        "multinomial_pmf" => vec![multinomial_pmf(&x, &y)?],
        "multinomial_logpmf" => vec![multinomial_logpmf(&x, &y)?],
        "categorical_pmf" => vec![categorical_pmf(&x, p[0])?],
        "categorical_cdf" => vec![categorical_cdf(&x, p[0])?],
        "binned_statistic" => flatten_bins(binned_statistic(&x, &y, &groups[0], stat(p[0]))?),
        "binned_statistic_2d" => flatten_bins(binned_statistic_2d(
            &x,
            &y,
            &groups[2],
            &groups[0],
            &groups[1],
            stat(p[0]),
        )?),
        "binned_statistic_dd" => {
            let edges = y
                .chunks(p[1] as usize)
                .map(|x| x.to_vec())
                .collect::<Vec<_>>();
            flatten_bins(binned_statistic_dd(&groups, &x, &edges, stat(p[2]))?)
        }
        "cumfreq" => {
            let r = cumfreq(&x, p[0] as usize, p[1], p[2])?;
            [
                r.frequencies,
                vec![r.lower_limit, r.bin_size, r.extra_points as f64],
            ]
            .concat()
        }
        "relfreq" => {
            let r = relfreq(&x, p[0] as usize, p[1], p[2])?;
            [
                r.frequencies,
                vec![r.lower_limit, r.bin_size, r.extra_points as f64],
            ]
            .concat()
        }
        "mean" => vec![mean(&x)?],
        "variance" => vec![variance(&x, p[0] as usize)?],
        "std" => vec![std(&x, p[0] as usize)?],
        "sem" => vec![sem(&x, p[0] as usize)?],
        "variation" => vec![variation(&x, p[0] as usize)?],
        "median" => vec![median(&x)?],
        "iqr" => vec![iqr(&x)?],
        "quantile" => vec![quantile(&x, p[0])?],
        "scoreatpercentile" => vec![scoreatpercentile(&x, p[0])?],
        "median_abs_deviation" => vec![median_abs_deviation(&x, p[0])?],
        "gmean" => vec![gmean(&x)?],
        "hmean" => vec![hmean(&x)?],
        "gstd" => vec![gstd(&x, p[0] as usize)?],
        "pmean" => vec![pmean(&x, p[0])?],
        "moment" => vec![moment(&x, p[0] as usize)?],
        "skew" => vec![skew(&x, p[0] != 0.)?],
        "kurtosis" => vec![kurtosis(&x, p[0] != 0., p[1] != 0.)?],
        "zscore" => zscore(&x, p[0] as usize)?,
        "zmap" => zmap(&x, &y, p[0] as usize)?,
        "gzscore" => gzscore(&x, p[0] as usize)?,
        "trim_mean" => vec![trim_mean(&x, p[0])?],
        "trimboth" => trimboth(&x, p[0])?,
        "mode" => {
            let (m, n) = mode(&x)?;
            vec![m, n as f64]
        }
        "describe" => {
            let d = describe(&x, p[0] as usize, p[1] != 0.)?;
            vec![
                d.count as f64,
                d.min,
                d.max,
                d.mean,
                d.variance,
                d.skewness,
                d.kurtosis,
            ]
        }
        "rankdata" => rankdata(
            &x,
            [
                RankMethod::Average,
                RankMethod::Min,
                RankMethod::Max,
                RankMethod::Dense,
                RankMethod::Ordinal,
            ][p[0] as usize],
        )?,
        "tiecorrect" => vec![tiecorrect(&x)?],
        "percentileofscore" => vec![percentileofscore(
            &x,
            p[0],
            [
                PercentileKind::Rank,
                PercentileKind::Strict,
                PercentileKind::Weak,
                PercentileKind::Mean,
            ][p[1] as usize],
        )?],
        "false_discovery_control" => false_discovery_control(
            &x,
            [FdrMethod::BenjaminiHochberg, FdrMethod::BenjaminiYekutieli][p[0] as usize],
        )?,
        "ecdf" => {
            let e = ecdf(&x)?;
            [e.quantiles, e.probabilities, e.survival].concat()
        }
        "boxcox" => boxcox(&x, p[0])?,
        "boxcox_llf" => vec![boxcox_llf(p[0], &x)?],
        "yeojohnson" => yeojohnson(&x, p[0])?,
        "yeojohnson_llf" => vec![yeojohnson_llf(p[0], &x)?],
        "expectile" => vec![expectile(&x, p[0])?],
        "circmean" => vec![circmean(&x, p[0], p[1])?],
        "circvar" => vec![circvar(&x, p[0], p[1])?],
        "circstd" => vec![circstd(&x, p[0], p[1], p[2] != 0.)?],
        "entropy" => vec![entropy(
            &x,
            if y.is_empty() { None } else { Some(&y) },
            if p[0] == 0. { None } else { Some(p[0]) },
        )?],
        "wasserstein_distance" => vec![wasserstein_distance(&x, &y)?],
        "energy_distance" => vec![energy_distance(&x, &y)?],
        "ttest_1samp" => ttest(ttest_1samp(&x, p[0], alternative(p[1]))?),
        "ttest_rel" => ttest(ttest_rel(&x, &y, alternative(p[0]))?),
        "ttest_ind" => ttest(ttest_ind(&x, &y, p[0] != 0., alternative(p[1]))?),
        "ttest_ind_from_stats" => ttest(ttest_ind_from_stats(
            p[0],
            p[1],
            p[2] as usize,
            p[3],
            p[4],
            p[5] as usize,
            p[6] != 0.,
            alternative(p[7]),
        )?),
        "pearsonr" => test(pearsonr(&x, &y, alternative(p[0]))?),
        "spearmanr" => test(spearmanr(&x, &y, alternative(p[0]))?),
        "pointbiserialr" => test(pointbiserialr(&x, &y)?),
        "ranksums" => test(ranksums(&x, &y, alternative(p[0]))?),
        "linregress" => {
            let r = linregress(&x, &y, alternative(p[0]))?;
            vec![
                r.slope,
                r.intercept,
                r.rvalue,
                r.pvalue,
                r.stderr,
                r.intercept_stderr,
            ]
        }
        "jarque_bera" => test(jarque_bera(&x)?),
        "f_oneway" => test(f_oneway(&g)?),
        "bartlett" => test(bartlett(&g)?),
        "kruskal" => test(kruskal(&g)?),
        "friedmanchisquare" => test(friedmanchisquare(&g)?),
        "levene" => test(levene(&g, [Center::Mean, Center::Median][p[0] as usize])?),
        "fisher_exact" => test(fisher_exact(
            x.as_slice().try_into().unwrap(),
            alternative(p[0]),
        )?),
        "chisquare" => test(chisquare(&x, &y, p[0] as usize)?),
        "power_divergence" => test(power_divergence(&x, &y, p[0] as usize, p[1])?),
        "margins" => concat_pair!(contingency::margins(&x, p[0] as usize, p[1] as usize)?),
        "expected_freq" => contingency::expected_freq(&x, p[0] as usize, p[1] as usize)?,
        "chi2_contingency" => {
            let r = chi2_contingency(&x, p[0] as usize, p[1] as usize, p[2] != 0.)?;
            [vec![r.statistic, r.pvalue, r.df as f64], r.expected].concat()
        }
        "association" => vec![contingency::association(
            &x,
            p[0] as usize,
            p[1] as usize,
            [
                contingency::Association::Cramer,
                contingency::Association::Tschuprow,
                contingency::Association::Pearson,
            ][p[2] as usize],
            p[3] != 0.,
        )?],
        "odds_ratio" => vec![contingency::odds_ratio(x.as_slice().try_into().unwrap())?],
        "relative_risk" => vec![contingency::relative_risk(p[0], p[1], p[2], p[3])?],
        "crosstab" => {
            let c = contingency::crosstab(&x, &y)?;
            [c.rows, c.columns, c.counts].concat()
        }
        _ => panic!("unknown API {api}"),
    })
}

#[test]
fn scipy_references() {
    let mut failures = Vec::new();
    let inputs = common::lines(include_str!("fixtures/stats_inputs.csv"));
    let expected = common::lines(include_str!("fixtures/stats_expected.csv"));
    assert_eq!(
        inputs.clone().count(),
        expected.clone().count(),
        "fixture row counts"
    );
    for (input, expected) in inputs.zip(expected) {
        let fields = common::fields(input);
        let expected = expected.split_once(',').unwrap();
        assert_eq!(fields[0], expected.0);
        let reference = numbers(expected.1);
        let actual = match evaluate(&fields) {
            Ok(v) => v,
            Err(e) => {
                failures.push(format!("case {input}: {e}"));
                continue;
            }
        };
        assert_eq!(actual.len(), reference.len(), "case {input}: shape");
        for (i, (&a, &b)) in actual.iter().zip(&reference).enumerate() {
            let matches = if b.is_nan() {
                a.is_nan()
            } else if b.is_infinite() {
                a == b
            } else {
                // Parameter searches have flat objectives: binary64 objective
                // noise limits position agreement even with tight stopping bounds.
                let tolerance = match fields[1] {
                    "boxcox_normmax" | "yeojohnson_normmax" => 2e-6,
                    "ppcc_max" => 2e-5,
                    // Three-frequency covariance here has condition number
                    // 1.4e8: NumPy SVD and direct solves differ by ~3e-9.
                    "epps_singleton_2samp" => 2e-9,
                    "dunnett" if i >= fields[5].split('/').count() - 1 => 5e-5,
                    _ => 2e-10,
                };
                a.is_finite() && approx::abs_diff_eq!(a, b, epsilon = tolerance * (1. + b.abs()))
            };
            if !matches {
                failures.push(format!(
                    "case {input}, output {i}: actual {a:e}, expected {b:e}"
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} discrepancies:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn studentized_range_scipy_references() -> Result<()> {
    let inputs = common::lines(include_str!("fixtures/range_inputs.csv"));
    let expected = common::lines(include_str!("fixtures/range_expected.csv"));
    assert_eq!(inputs.clone().count(), expected.clone().count());
    for (input, expected) in inputs.zip(expected) {
        let fields = common::fields(input);
        let reference = common::fields(expected);
        assert_eq!(fields[0], reference[0]);
        let sizes = numbers(fields[1]);
        let locations = numbers(fields[2]);
        let groups: Vec<Vec<f64>> = sizes
            .iter()
            .zip(locations)
            .map(|(n, loc)| {
                (0..*n as usize)
                    .map(|i| loc + if i % 2 == 0 { -1. } else { 1. })
                    .collect()
            })
            .collect();
        let slices: Vec<_> = groups.iter().map(Vec::as_slice).collect();
        let result = tukey_hsd(&slices)?;
        for (actual, expected) in [
            (&result.statistic, reference[1]),
            (&result.pvalue, reference[2]),
        ] {
            assert_eq!(actual.len(), groups.len());
            assert!(actual.iter().all(|row| row.len() == groups.len()));
            for (a, b) in actual.iter().flatten().zip(numbers(expected)) {
                assert!(
                    (a - b).abs() <= 2e-10 * (1. + b.abs()),
                    "case {input}: {a:e} vs {b:e}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn invalid_domains_and_shapes() {
    let bad = [f64::NAN];
    assert!(mean(&[]).is_err());
    assert!(mean(&bad).is_err());
    assert!(variance(&[1.], 1).is_err());
    assert!(sem(&[1.], 0).is_err());
    assert!(quantile(&[1.], -0.1).is_err());
    assert!(moment(&[1.], 65).is_err());
    assert!(gmean(&[-1.]).is_err());
    assert!(hmean(&[-1.]).is_err());
    assert!(gstd(&[0., 1.], 1).is_err());
    assert!(gzscore(&[0.], 0).is_err());
    assert!(median_abs_deviation(&[1.], 0.).is_err());
    assert!(trimboth(&[1.], 0.5).is_err());
    assert!(rankdata(&bad, RankMethod::Average).is_err());
    assert!(tiecorrect(&bad).is_err());
    assert!(percentileofscore(&[1.], f64::NAN, PercentileKind::Rank).is_err());
    assert!(false_discovery_control(&[1.1], FdrMethod::BenjaminiHochberg).is_err());
    assert!(ecdf(&[]).is_err());
    assert!(boxcox(&[0.], 1.).is_err());
    assert!(yeojohnson(&bad, 1.).is_err());
    assert!(boxcox_llf(1., &[]).is_err());
    assert!(yeojohnson_llf(1., &bad).is_err());
    assert!(circmean(&[1.], 1., 1.).is_err());
    assert!(circvar(&bad, 0., 1.).is_err());
    assert!(circstd(&[], 0., 1., false).is_err());
    assert!(entropy(&[0., 0.], None, None).is_err());
    assert!(entropy(&[1.], Some(&[1., 2.]), None).is_err());
    assert!(entropy(&[1.], None, Some(1.)).is_err());
    assert!(pmean(&[0.], 1.).is_err());
    assert!(expectile(&[1.], 2.).is_err());
    assert!(wasserstein_distance(&[], &[1.]).is_err());
    assert!(energy_distance(&bad, &[1.]).is_err());
    assert_eq!(
        pearsonr(&[1., 2.], &[1.], Alternative::TwoSided),
        Err(Error::DimensionMismatch)
    );
    assert!(spearmanr(&[1.], &[1.], Alternative::TwoSided).is_err());
    assert!(pointbiserialr(&[2., 3.], &[1., 2.]).is_err());
    assert!(linregress(&[1., 1.], &[1., 2.], Alternative::TwoSided).is_err());
    assert!(ttest_1samp(&[1.], 0., Alternative::TwoSided).is_err());
    assert!(ttest_rel(&[1., 2.], &[1.], Alternative::TwoSided).is_err());
    assert!(ttest_ind(&bad, &[1., 2.], true, Alternative::TwoSided).is_err());
    assert!(ttest_ind_from_stats(0., -1., 4, 0., 1., 4, true, Alternative::TwoSided).is_err());
    assert!(f_oneway(&[&[1.], &[2.]]).is_err());
    assert!(bartlett(&[&[1.], &[2.]]).is_err());
    assert!(kruskal(&[&[1.], &[1.]]).is_err());
    assert!(friedmanchisquare(&[&[1.], &[1., 2.], &[2.]]).is_err());
    assert!(levene(&[&bad, &[1.]], Center::Median).is_err());
    assert!(ranksums(&[], &[1.], Alternative::Less).is_err());
    assert!(fisher_exact(&[1.5, 2., 3., 4.], Alternative::TwoSided).is_err());
    assert!(fisher_exact(&[65536.; 4], Alternative::Less).is_err());
    assert!(chisquare(&[1., 2.], &[2., 2.], 0).is_err());
    assert!(power_divergence(&[1., 2.], &[0., 3.], 0, 0.).is_err());
    assert!(jarque_bera(&bad).is_err());
    assert!(contingency::margins(&[1.], 2, 2).is_err());
    assert!(contingency::expected_freq(&[0.; 4], 2, 2).is_err());
    assert!(chi2_contingency(&[0., 0., 1., 1.], 2, 2, false).is_err());
    assert!(
        contingency::association(&[1.], 1, 1, contingency::Association::Cramer, false).is_err()
    );
    assert!(contingency::odds_ratio(&[-1., 2., 3., 4.]).is_err());
    assert!(contingency::relative_risk(2., 1., 1., 2.).is_err());
    assert!(contingency::crosstab(&[1.], &[1., 2.]).is_err());
}

#[test]
fn remaining_domains_and_work_limits() {
    let x = [1., 2., 3., 4.];
    let nan = [f64::NAN];
    assert!(std(&nan, 0).is_err());
    assert!(median(&[]).is_err());
    assert!(iqr(&nan).is_err());
    assert!(mode(&nan).is_err());
    assert!(describe(&x, 4, true).is_err());
    assert!(scoreatpercentile(&x, 101.).is_err());
    assert!(variation(&x, 4).is_err());
    assert!(skew(&nan, true).is_err());
    assert!(kurtosis(&nan, true, true).is_err());
    assert!(zscore(&x, 4).is_err());
    assert!(zmap(&nan, &x, 1).is_err());
    assert!(tmean(&x, Some((5., 6.)), (true, true)).is_err());
    assert!(tvar(&x, Some((2., 2.)), (true, true), 1).is_err());
    assert!(tstd(&x, None, (true, true), 4).is_err());
    assert!(tsem(&[1.], None, (true, true), 0).is_err());
    assert!(tmin(&x, Some(f64::NAN), true).is_err());
    assert!(tmax(&x, Some(0.), true).is_err());
    assert!(trim1(&x, 1., Tail::Upper).is_err());
    assert!(kstat(&x, 5).is_err());
    assert!(kstatvar(&x, 3).is_err());
    assert!(lmoment(&[1.], 2, true).is_err());
    assert!(sigmaclip(&x, 0., 2.).is_err());
    assert!(differential_entropy(&x, 2, None).is_err());
    assert!(binomtest(3, 2, 0.5, Alternative::TwoSided).is_err());
    assert!(combine_pvalues(&[1.1], CombineMethod::Fisher, None).is_err());
    assert!(combine_pvalues(&[0.2], CombineMethod::Stouffer, Some(&[0.])).is_err());
    assert!(combine_pvalues(&[0.2], CombineMethod::Fisher, Some(&[1.])).is_err());
    assert!(directional_stats(&[vec![0., 0.]], true).is_err());
    assert!(directional_stats(&[vec![1.], vec![1., 2.]], true).is_err());
    assert!(
        ks_1samp(
            &x,
            Distribution::Poisson { rate: 1. },
            Alternative::TwoSided
        )
        .is_err()
    );
    assert!(ks_2samp(&[], &x, Alternative::TwoSided).is_err());
    assert!(kstest(&nan, &x, Alternative::TwoSided).is_err());
    assert!(
        probplot(
            &[],
            Distribution::Normal {
                mean: 0.,
                std_dev: 1.
            }
        )
        .is_err()
    );
    assert!(qmc::discrepancy(&[vec![1.1]], qmc::Discrepancy::Centered, false).is_err());
    assert!(qmc::discrepancy(&vec![vec![0.5]; 4097], qmc::Discrepancy::Centered, false).is_err());
    assert!(
        qmc::geometric_discrepancy(&[vec![0.]], qmc::GeometricDiscrepancy::MinimumDistance)
            .is_err()
    );
    assert!(
        qmc::geometric_discrepancy(
            &[vec![0.], vec![0.]],
            qmc::GeometricDiscrepancy::MinimumSpanningTree
        )
        .is_err()
    );
    assert!(qmc::update_discrepancy(&[1., 2.], &[vec![0.5]], 1.).is_err());
    assert!(qmc::scale(&[vec![0.5]], &[2.], &[1.], false).is_err());
    assert!(qmc::scale(&[vec![2.]], &[0.], &[1.], true).is_err());
    let matrix = [vec![1., 0.], vec![0., 1.]];
    assert!(multivariate_normal_pdf(&[1.], &[0., 0.], &matrix).is_err());
    assert!(
        multivariate_normal_logpdf(&[0., 0.], &[0., 0.], &[vec![1., 2.], vec![2., 1.]]).is_err()
    );
    assert!(multivariate_t_pdf(&[0., 0.], &[0., 0.], &matrix, 0.).is_err());
    assert!(
        multivariate_t_logpdf(&[0., 0.], &[0., 0.], &[vec![1., 2.], vec![1., 1.]], 2.).is_err()
    );
    assert!(dirichlet_pdf(&[0., 1.], &[2., 3.]).is_err());
    assert!(dirichlet_logpdf(&[0.2, 0.3], &[2., 3.]).is_err());
    assert!(multinomial_pmf(&[1.5, 2.], &[0.5, 0.5]).is_err());
    assert!(multinomial_logpmf(&[1., 2.], &[0.3, 0.3]).is_err());
    assert!(categorical_pmf(&[-1., 1.], 0.).is_err());
    assert!(categorical_cdf(&[0., 0.], 0.).is_err());
    assert!(binned_statistic(&x, &x, &[0., 1., 1.], BinStatistic::Mean).is_err());
    assert!(binned_statistic_2d(&x, &[1.], &x, &[0., 1.], &[0., 1.], BinStatistic::Mean).is_err());
    assert!(
        binned_statistic_dd(&[vec![1., 2.]], &[1.], &[vec![0., 1.]], BinStatistic::Mean).is_err()
    );
    assert!(cumfreq(&x, 0, 0., 1.).is_err());
    assert!(relfreq(&x, 2, 1., 0.).is_err());
    assert!(mean(&vec![1.; 1_048_577]).is_err());
    assert!(variance(&[-f64::MAX, f64::MAX], 0).is_err());
}

#[test]
fn distribution_validation() {
    let invalid = [
        Distribution::Normal {
            mean: 0.,
            std_dev: 0.,
        },
        Distribution::Beta { a: 0., b: 1. },
        Distribution::Gamma {
            shape_value: 1.,
            scale: 0.,
        },
        Distribution::Erlang {
            shape_value: 1.5,
            scale: 1.,
        },
        Distribution::DiscreteUniform { low: 1., high: 1. },
        Distribution::ChiSquared { df: 0. },
        Distribution::Chi { df: 1.5 },
        Distribution::StudentT { df: 0. },
        Distribution::F {
            numerator_df: 1.,
            denominator_df: 0.,
        },
        Distribution::Exponential { scale: 0. },
        Distribution::Cauchy {
            location: 0.,
            scale: 0.,
        },
        Distribution::Laplace {
            location: f64::NAN,
            scale: 1.,
        },
        Distribution::LogNormal {
            log_mean: 0.,
            log_std: 0.,
        },
        Distribution::Uniform { low: 1., high: 0. },
        Distribution::Weibull {
            shape_value: 0.,
            scale: 1.,
        },
        Distribution::Pareto {
            minimum: 0.,
            shape_value: 1.,
        },
        Distribution::InverseGamma {
            shape_value: 1.,
            scale: 0.,
        },
        Distribution::Levy {
            location: f64::NAN,
            scale: 1.,
        },
        Distribution::Levy {
            location: f64::INFINITY,
            scale: 1.,
        },
        Distribution::Levy {
            location: 0.,
            scale: 0.,
        },
        Distribution::Levy {
            location: 0.,
            scale: -1.,
        },
        Distribution::Levy {
            location: 0.,
            scale: f64::INFINITY,
        },
        Distribution::Gumbel {
            location: 0.,
            scale: 0.,
        },
        Distribution::Triangular {
            low: 0.,
            high: 1.,
            mode: 2.,
        },
        Distribution::Bernoulli { probability: 1.1 },
        Distribution::Binomial {
            trials: 1.5,
            probability: 0.5,
        },
        Distribution::Poisson { rate: -1. },
        Distribution::Geometric { probability: 0. },
        Distribution::NegativeBinomial {
            successes: 0.,
            probability: 0.5,
        },
        Distribution::Hypergeometric {
            population: 10.,
            successes: 11.,
            draws: 5.,
        },
        Distribution::Hypergeometric {
            population: 65537.,
            successes: 10.,
            draws: 5.,
        },
    ];
    for d in invalid {
        assert!(pdf(d, 0.).is_err(), "{d:?}");
        assert!(logpdf(d, 0.).is_err());
        assert!(cdf(d, f64::INFINITY).is_err());
        assert!(sf(d, f64::NEG_INFINITY).is_err());
        assert!(ppf(d, 0.).is_err());
    }
    let d = Distribution::Normal {
        mean: 0.,
        std_dev: 1.,
    };
    assert!(pdf(d, f64::NAN).is_err());
    assert!(cdf(d, f64::NAN).is_err());
    assert!(sf(d, f64::NAN).is_err());
    assert!(logpdf(d, f64::NAN).is_err());
    assert!(ppf(d, -0.1).is_err());
}

#[test]
fn levy_identities_and_distribution_interoperability() -> Result<()> {
    for scale in [1e-200, 1., 1e200] {
        let levy = Distribution::Levy {
            location: 0.,
            scale,
        };
        // Lévy(0,c) equals inverse-gamma(shape=1/2, scale=c/2).
        let inverse_gamma = Distribution::InverseGamma {
            shape_value: 0.5,
            scale: scale / 2.,
        };
        let x = 2. * scale;
        let a = sf(levy, x)?;
        let b = sf(inverse_gamma, x)?;
        assert!(
            // statrs erf and incomplete-gamma use different approximations;
            // their ~5e-11 discrepancy is within the ordinary fixture tolerance.
            a > 0. && (a - b).abs() <= 2e-10 * b,
            "scale={scale:e}, sf={a:e}, inverse_gamma_sf={b:e}"
        );
        assert!((logpdf(levy, x)? - logpdf(inverse_gamma, x)?).abs() < 2e-12);
    }
    // For c/x << 1, SF = sqrt(2*c/(π*x)) * (1 + O(c/x)).
    // This independently checks a subnormal scale without the rounding loss
    // from forming c/(2*x) in the inverse-gamma backend.
    let tiny = Distribution::Levy {
        location: 0.,
        scale: 1e-320,
    };
    let reference = 1e-320_f64.sqrt() * (2. / (10. * std::f64::consts::PI)).sqrt();
    assert!((sf(tiny, 10.)? / reference - 1.).abs() < 2e-12);
    let levy = Distribution::Levy {
        location: -3.,
        scale: 2.,
    };
    for p in [0.001, 0.2, 0.5, 0.9, 0.999] {
        let x = ppf(levy, p)?;
        assert!((cdf(levy, x)? - p).abs() < 2e-10);
        assert!((cdf(levy, x)? + sf(levy, x)? - 1.).abs() < 2e-12);
    }
    let conditional = truncate(levy, -2., 10.)?;
    let ordered = order_statistic(levy, 2, 5)?;
    for d in [conditional, ordered] {
        assert!((cdf(d, ppf(d, 0.6)?)? - 0.6).abs() < 2e-10);
    }
    assert!(
        ppf(
            Distribution::Levy {
                location: 0.,
                scale: f64::MAX
            },
            0.99
        )
        .is_err()
    );
    assert!(pdf(levy, f64::NAN).is_err());
    assert!(ppf(levy, 1.1).is_err());
    Ok(())
}

#[test]
fn specialized_quantiles_preserve_smaller_tail_probabilities() -> Result<()> {
    for d in [
        Distribution::Chi { df: 1. },
        Distribution::Chi { df: 1000. },
        Distribution::InverseGamma {
            shape_value: 0.2,
            scale: 1.,
        },
        Distribution::InverseGamma {
            shape_value: 1000.,
            scale: 0.01,
        },
    ] {
        for p in [1e-100, 1e-12, 0.01, 0.5, 0.99, 1. - 1e-12] {
            let x = ppf(d, p)?;
            let (actual, expected) = if p <= 0.5 {
                (cdf(d, x)?, p)
            } else {
                (sf(d, x)?, 1. - p)
            };
            assert!(x.is_finite() && x > 0.);
            assert!(
                (actual / expected - 1.).abs() < 2e-10,
                "{d:?}, p={p:e}: {actual:e} vs {expected:e}"
            );
        }
        assert_eq!(ppf(d, 0.)?, 0.);
        assert_eq!(ppf(d, 1.)?, f64::INFINITY);
    }
    Ok(())
}

#[test]
fn additional_rank_tests_validate_domains_and_symmetries() -> Result<()> {
    let two = Alternative::TwoSided;
    let x = [1., 2., 3., 4.];
    let y = [2., 4., 1., 3.];
    for variant in [KendallVariant::B, KendallVariant::C] {
        let r = kendalltau(&x, &y, variant, two)?;
        let reverse = kendalltau(&y, &x, variant, two)?;
        assert_eq!(r, reverse);
        let reversed_y = y.map(|v| -v);
        let negative = kendalltau(&x, &reversed_y, variant, two)?;
        assert_eq!(r.statistic, -negative.statistic);
        assert_eq!(r.pvalue, negative.pvalue);
        assert!(kendalltau(&x, &y[..3], variant, two).is_err());
        assert!(kendalltau(&[1., 2.], &[1., 2.], variant, two).is_err());
        assert!(kendalltau(&[1.; 4], &y, variant, two)?.pvalue.is_nan());
        assert!(kendalltau(&vec![0.; 4097], &vec![0.; 4097], variant, two).is_err());
        assert_eq!(
            kendalltau(
                &[-f64::MAX, 0., f64::MAX],
                &[-f64::MAX, 0., f64::MAX],
                variant,
                two
            )?
            .statistic,
            1.
        );
    }
    for zero in [
        WilcoxonZero::Wilcox,
        WilcoxonZero::Pratt,
        WilcoxonZero::Split,
    ] {
        for correction in [true, false] {
            let d = [0., 1., -2., 3., 3.];
            let reverse = d.map(|v| -v);
            let first = wilcoxon(&d, zero, two, correction)?;
            let second = wilcoxon(&reverse, zero, two, correction)?;
            assert_eq!(first, second);
            assert!(
                (wilcoxon(&d, zero, Alternative::Less, correction)?.pvalue
                    - wilcoxon(&reverse, zero, Alternative::Greater, correction)?.pvalue)
                    .abs()
                    < 1e-14
            );
        }
        assert!(wilcoxon(&[], zero, two, false).is_err());
        let r = wilcoxon(&[0.; 3], zero, two, false)?;
        if zero == WilcoxonZero::Split {
            assert_eq!(r.pvalue, 1.);
        } else {
            assert!(r.pvalue.is_nan());
            assert_eq!(r.statistic, 0.);
        }
    }
    assert!(ansari(&[], &x, two).is_err());
    assert!(
        (ansari(&x, &y, Alternative::Less)?.pvalue + ansari(&x, &y, Alternative::Greater)?.pvalue
            - 1.)
            .abs()
            < 1e-14
    );
    let q = quantile_test(&[1., 2., 2., 3.], 2., 0.5, Alternative::Greater)?;
    assert_eq!((q.statistic, q.count), (3, QuantileCount::AtOrBelow));
    let q = quantile_test(&[1., 2., 2., 3.], 2., 0.5, Alternative::Less)?;
    assert_eq!((q.statistic, q.count), (1, QuantileCount::Below));
    assert!(quantile_test(&[], 0., 0.5, two).is_err());
    for p in [f64::NAN, 0., 1., -0.1, 1.1] {
        assert!(quantile_test(&x, 0., p, two).is_err());
    }
    for q in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(quantile_test(&x, q, 0.5, two).is_err());
    }
    let rows: [&[f64]; 2] = [&[1., 2., 3.], &[2., 3., 4.]];
    let page = page_trend_test(&rows, &[1, 2, 3])?;
    let reversed: [&[f64]; 2] = [&[3., 2., 1.], &[4., 3., 2.]];
    assert_eq!(page, page_trend_test(&reversed, &[3, 2, 1])?);
    assert!(page_trend_test(&rows[..1], &[1, 2, 3]).is_err());
    assert!(page_trend_test(&[&[1., 2.], &[2., 3.]], &[1, 2]).is_err());
    assert!(page_trend_test(&[&[1., 2., 3.], &[2., 3.]], &[1, 2, 3]).is_err());
    for permutation in [&[1, 1, 3][..], &[0, 2, 3], &[1, 2, 4], &[1, 2]] {
        assert!(page_trend_test(&rows, permutation).is_err());
    }
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let nonfinite = [1., 2., 3., bad];
        assert!(kendalltau(&x, &nonfinite, KendallVariant::B, two).is_err());
        assert!(wilcoxon(&nonfinite, WilcoxonZero::Pratt, two, false).is_err());
        assert!(ansari(&x, &nonfinite, two).is_err());
        assert!(quantile_test(&nonfinite, 2., 0.5, two).is_err());
        assert!(page_trend_test(&[&x, &nonfinite], &[1, 2, 3, 4]).is_err());
    }
    Ok(())
}

#[test]
fn native_gap_groups_validate_domains_and_preserve_identities() -> Result<()> {
    let alt = Alternative::TwoSided;
    let values = [1., 2., 3., 4., 5., 6., 7., 8000.];
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut nonfinite = values;
        nonfinite[0] = bad;
        assert!(skewtest(&nonfinite, alt).is_err());
        assert!(kurtosistest(&nonfinite, alt).is_err());
        assert!(normaltest(&nonfinite).is_err());
        assert!(obrientransform(&[&nonfinite]).is_err());
        assert!(alexandergovern(&[&values, &nonfinite]).is_err());
        assert!(siegelslopes(&values, &nonfinite, SiegelMethod::Separate).is_err());
        assert!(theilslopes(&values, &nonfinite, 0.95, TheilMethod::Joint).is_err());
        assert!(brunnermunzel(&values, &nonfinite, alt, BrunnerDistribution::Normal).is_err());
        assert!(mood(&values, &nonfinite, alt).is_err());
        assert!(mannwhitneyu(&values, &nonfinite, alt, true).is_err());
        assert!(median_test(&[&values, &nonfinite], MedianTies::Below, false).is_err());
        assert!(fligner(&[&values, &nonfinite], Center::Mean).is_err());
        assert!(boxcox_normplot(&nonfinite, -1., 1., 3).is_err());
        assert!(yeojohnson_normplot(&nonfinite, -1., 1., 3).is_err());
    }
    assert!(skewtest(&values[..7], alt).is_err());
    assert!(kurtosistest(&values[..4], alt).is_err());
    assert!(normaltest(&values[..7]).is_err());
    assert!(obrientransform(&[]).is_err());
    assert!(obrientransform(&[&values[..2]]).is_err());
    assert!(obrientransform(&[&[-f64::MAX, 0., f64::MAX]]).is_err());
    assert!(alexandergovern(&[&values]).is_err());
    assert!(alexandergovern(&[&values, &[1.]]).is_err());
    let n = normaltest(&values)?;
    let s = skewtest(&values, alt)?;
    let k = kurtosistest(&values, alt)?;
    assert!((n.statistic - s.statistic.powi(2) - k.statistic.powi(2)).abs() < 1e-14);
    for test in [skewtest, kurtosistest] {
        assert!(
            (test(&values, Alternative::Less)?.pvalue
                + test(&values, Alternative::Greater)?.pvalue
                - 1.)
                .abs()
                < 1e-14
        );
        assert!(test(&[2.; 8], alt)?.statistic.is_nan());
    }
    assert!(normaltest(&[2.; 8])?.pvalue.is_nan());
    let symmetric = [-4., -3., -2., -1., 1., 2., 3., 4.];
    assert!(skewtest(&symmetric, alt)?.statistic.abs() < 1e-14);
    assert!((skewtest(&symmetric, alt)?.pvalue - 1.).abs() < 1e-14);
    let base = [0., 2., 4., 8.];
    let shifted = base.map(|x| x + 1e16);
    let scaled = base.map(|x| 3. * x);
    let transformed = obrientransform(&[&base, &shifted, &scaled])?;
    for (i, (input, row)) in [&base, &shifted, &scaled]
        .iter()
        .zip(&transformed)
        .enumerate()
    {
        assert!((mean(row)? - variance(*input, 1)?).abs() < 1e-12);
        for (actual, original) in row.iter().zip(&transformed[0]) {
            let multiplier = if i == 2 { 9. } else { 1. };
            assert!((actual - multiplier * original).abs() < 1e-12);
        }
    }
    // Transformed groups interoperate with the existing ANOVA API.
    assert!((0. ..=1.).contains(&f_oneway(&[&transformed[0], &transformed[2]])?.pvalue));
    let a = [1., 2., 3., 4.];
    let b = [2., 4., 7., 11.];
    let changed_a = a.map(|v| -3. * v + 10.);
    let changed_b = b.map(|v| -3. * v + 10.);
    assert!(
        (alexandergovern(&[&a, &b])?.statistic
            - alexandergovern(&[&changed_a, &changed_b])?.statistic)
            .abs()
            < 1e-12
    );
    let x = [0., 1., 2., 3., 4.];
    let y = [1., 3., 5., 7., 999.];
    for method in [SiegelMethod::Separate, SiegelMethod::Hierarchical] {
        let fit = siegelslopes(&x, &y, method)?;
        assert_eq!((fit.slope, fit.intercept), (2., 1.));
        let reverse_x: Vec<_> = x.iter().rev().copied().collect();
        let reverse_y: Vec<_> = y.iter().rev().copied().collect();
        assert_eq!(fit, siegelslopes(&reverse_x, &reverse_y, method)?);
        assert!(siegelslopes(&[1., 1.], &[1., 2.], method).is_err());
        assert!(siegelslopes(&[0., 1.], &[0.], method).is_err());
        assert!(siegelslopes(&[0., 1e-308], &[0., f64::MAX], method).is_err());
        let extremes = siegelslopes(&[-f64::MAX, f64::MAX], &[-f64::MAX, f64::MAX], method)?;
        assert_eq!((extremes.slope, extremes.intercept), (1., 0.));
    }
    for method in [TheilMethod::Separate, TheilMethod::Joint] {
        let fit = theilslopes(&x, &y, 0.9, method)?;
        assert_eq!(fit, theilslopes(&x, &y, 0.1, method)?);
        assert!(fit.low_slope <= fit.slope && fit.slope <= fit.high_slope);
        for confidence in [f64::NAN, -1., 0., 1., 2.] {
            assert!(theilslopes(&x, &y, confidence, method).is_err());
        }
        assert!(theilslopes(&[1., 1.], &[1., 2.], 0.9, method).is_err());
        assert!(theilslopes(&x, &y[..4], 0.9, method).is_err());
        assert!(theilslopes(&[0., 1e-308], &[0., f64::MAX], 0.9, method).is_err());
        let extremes = theilslopes(&[-f64::MAX, f64::MAX], &[-f64::MAX, f64::MAX], 0.95, method)?;
        assert_eq!((extremes.slope, extremes.intercept), (1., 0.));
    }
    let too_long = vec![0.; 2049];
    assert!(siegelslopes(&too_long, &too_long, SiegelMethod::Separate).is_err());
    assert!(theilslopes(&too_long, &too_long, 0.95, TheilMethod::Separate).is_err());
    for distribution in [BrunnerDistribution::StudentT, BrunnerDistribution::Normal] {
        let first = brunnermunzel(&a, &b, alt, distribution)?;
        let reverse = brunnermunzel(&b, &a, alt, distribution)?;
        assert!((first.statistic + reverse.statistic).abs() < 1e-14);
        assert!((first.pvalue - reverse.pvalue).abs() < 1e-14);
        assert!(brunnermunzel(&[1.], &b, alt, distribution).is_err());
        assert!(
            brunnermunzel(&[1., 1.], &[1., 1.], alt, distribution)?
                .statistic
                .is_nan()
        );
    }
    let first = mannwhitneyu(&a, &b, alt, true)?;
    let second = mannwhitneyu(&b, &a, alt, true)?;
    assert_eq!(
        first.statistic + second.statistic,
        (a.len() * b.len()) as f64
    );
    assert_eq!(first.pvalue, second.pvalue);
    assert!(mannwhitneyu(&[], &a, alt, true).is_err());
    let first = mood(&a, &b, alt)?;
    let second = mood(&b, &a, alt)?;
    assert!((first.statistic + second.statistic).abs() < 1e-14);
    assert!((first.pvalue - second.pvalue).abs() < 1e-14);
    assert!(mood(&[1.], &[2.], alt).is_err());
    assert!(mood(&[], &a, alt).is_err());
    assert!(mood(&[1., 1.], &[1., 1.], alt)?.pvalue.is_nan());
    assert!(median_test(&[&[1., 1.], &[1., 1.]], MedianTies::Below, false).is_err());
    assert!(median_test(&[&[2., 2.], &[1., 2., 4.]], MedianTies::Ignore, false).is_err());
    assert!(median_test(&[&a], MedianTies::Above, false).is_err());
    assert!(median_test(&[&a, &[]], MedianTies::Above, false).is_err());
    let test = median_test(&[&a, &b], MedianTies::Below, true)?;
    let flat: Vec<_> = test.table.iter().flatten().copied().collect();
    let contingency = chi2_contingency(&flat, 2, 2, true)?;
    assert_eq!(
        (test.statistic, test.pvalue),
        (contingency.statistic, contingency.pvalue)
    );
    assert!(fligner(&[&a], Center::Mean).is_err());
    assert!(fligner(&[&a, &[]], Center::Mean).is_err());
    assert!(
        fligner(&[&[0.; 3], &[1.; 3]], Center::Median)?
            .pvalue
            .is_nan()
    );
    for plot in [boxcox_normplot, yeojohnson_normplot] {
        assert!(plot(&a, -1., 1., 1).is_err());
        assert!(plot(&a, 1., -1., 3).is_err());
        assert!(plot(&a, f64::NEG_INFINITY, 1., 3).is_err());
        assert!(plot(&[1.; 4], -1., 1., 3).is_err());
        assert!(plot(&[1.], -1., 1., 3).is_err());
        assert!(plot(&a, -1., 1., 65537).is_err());
        assert!(plot(&vec![1.; 257], -1., 1., 65536).is_err());
        assert!(plot(&[1e100, 2e100], 100., 101., 2).is_err());
    }
    assert!(boxcox_normplot(&[-1., 1.], -1., 1., 3).is_err());
    let plot = boxcox_normplot(&[1., 2., 4., 8.], -1., 1., 3)?;
    let quantiles = probplot(
        &[1., 2., 4., 8.],
        Distribution::Normal {
            mean: 0.,
            std_dev: 1.,
        },
    )?;
    for (lambda, r) in plot.lambdas.iter().zip(plot.correlations) {
        let transformed = boxcox(&quantiles.ordered, *lambda)?;
        assert!((r - pearsonr(&quantiles.theoretical, &transformed, alt)?.statistic).abs() < 1e-14);
    }
    Ok(())
}

#[test]
fn standardized_scores_preserve_scale_and_special_values() -> Result<()> {
    let base = [0., 1., 4.];
    for ddof in [0, 1, 2] {
        let expected = zscore(&base, ddof)?;
        for transformed in [
            [1e16, 1e16 + 2., 1e16 + 8.],
            [0., 1e-308, 4e-308],
            [0., -1e307, -4e307],
        ] {
            let actual = zscore(&transformed, ddof)?;
            assert!(mean(&actual)?.abs() < 2e-15);
            assert!((variance(&actual, ddof)? - 1.).abs() < 2e-15);
            let sign = if transformed[1] < 0. { -1. } else { 1. };
            for (a, b) in actual.iter().zip(&expected) {
                assert!((a - sign * b).abs() < 2e-15);
            }
            assert_eq!(actual, zmap(&transformed, &transformed, ddof)?);
        }
    }
    for constant in [-f64::MAX, -0., f64::from_bits(1), f64::MAX] {
        assert!(zscore(&[constant; 3], 1)?.iter().all(|x| x.is_nan()));
        if constant > 0. {
            assert!(gzscore(&[constant; 3], 0)?.iter().all(|x| x.is_nan()));
        } else {
            assert!(gzscore(&[constant; 3], 0).is_err());
        }
    }
    assert!(gzscore(&[2.; 3], 0)?.iter().all(|x| x.is_nan()));
    let special = zmap(&[1., 2., 3.], &[2., 2.], 0)?;
    assert_eq!(special[0], f64::NEG_INFINITY);
    assert!(special[1].is_nan());
    assert_eq!(special[2], f64::INFINITY);
    assert!(zmap(&[1.], &[0., 1e-308], 0).is_err());
    assert!(zmap(&[], &[1., 2.], 2).is_err());
    assert!(zmap(&[], &[f64::INFINITY], 0).is_err());
    assert!(zmap(&[], &[], 0).is_err());
    assert_eq!(zmap(&[], &[1., 2.], 0)?, Vec::<f64>::new());
    Ok(())
}

#[test]
fn scaling_and_independent_properties() -> Result<()> {
    assert_eq!(mean(&[f64::MAX, f64::MAX])?, f64::MAX);
    assert_eq!(mean(&[-f64::MAX, f64::MAX])?, 0.);
    assert_eq!(variance(&[f64::MAX, f64::MAX], 0)?, 0.);
    assert_eq!(std(&[-f64::MAX, f64::MAX], 0)?, f64::MAX);
    assert!((mean(&[1e16, 1., -1e16])? - 1. / 3.).abs() < 1e-15);
    assert_eq!(
        hmean(&[f64::MIN_POSITIVE / 16.; 2])?,
        f64::MIN_POSITIVE / 16.
    );
    assert_eq!(
        rankdata(&[-0., 0., 1.], RankMethod::Average)?,
        vec![1.5, 1.5, 3.]
    );
    assert_eq!(wasserstein_distance(&[1., 2., 3.], &[1., 2., 3.])?, 0.);
    assert_eq!(energy_distance(&[1., 2., 3.], &[1., 2., 3.])?, 0.);
    let sample = vec![vec![0.1, 0.3], vec![0.4, 0.6], vec![0.9, 0.2]];
    let initial = qmc::discrepancy(&sample, qmc::Discrepancy::Centered, true)?;
    let mut updated = sample.clone();
    updated.push(vec![0.7, 0.8]);
    assert!(
        (qmc::update_discrepancy(&[0.7, 0.8], &sample, initial)?
            - qmc::discrepancy(&updated, qmc::Discrepancy::Centered, false)?)
        .abs()
            < 1e-14
    );
    let scaled = qmc::scale(&sample, &[-2., 3.], &[1., 10.], false)?;
    let restored = qmc::scale(&scaled, &[-2., 3.], &[1., 10.], true)?;
    for (a, b) in sample.iter().flatten().zip(restored.iter().flatten()) {
        assert!((a - b).abs() < 1e-14);
    }
    let (r, c) = contingency::margins(&[10., 20., 30., 40.], 2, 2)?;
    let expected = contingency::expected_freq(&[10., 20., 30., 40.], 2, 2)?;
    assert_eq!(contingency::margins(&expected, 2, 2)?, (r, c));
    assert!(
        friedmanchisquare(&[&[1., 1.], &[1., 1.], &[1., 1.]])?
            .statistic
            .is_nan()
    );
    assert!(
        (multinomial_pmf(&[2., 3.], &[0.3, 0.7])?
            - pdf(
                Distribution::Binomial {
                    trials: 5.,
                    probability: 0.3
                },
                2.
            )?)
        .abs()
            < 1e-14
    );
    let q = multivariate_normal_pdf(&[1., 2.], &[0., 0.], &[vec![1., 0.], vec![0., 1.]])?;
    assert!((q - normal_pdf(1., 0., 1.)? * normal_pdf(2., 0., 1.)?).abs() < 1e-14);
    let z = zscore(&[2., 3., 7., 11.], 0)?;
    assert!(mean(&z)?.abs() < 1e-14);
    assert!((variance(&z, 0)? - 1.).abs() < 1e-14);
    let d = Distribution::Normal {
        mean: 2.,
        std_dev: 3.,
    };
    for p in [0.001, 0.1, 0.5, 0.9, 0.999] {
        assert!((cdf(d, ppf(d, p)?)? - p).abs() < 1e-12);
    }
    let f = sci::fft::rfft(&[1., 2., 3., 4.]);
    let amplitudes = f.into_iter().map(sci::complex::abs).collect::<Vec<_>>();
    assert_eq!(describe(&amplitudes, 1, true)?.count, amplitudes.len());
    Ok(())
}

#[test]
fn added_statistics_domains_and_failures() -> Result<()> {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let x = [1., 2., invalid];
        assert!(bayes_mvs(&x, 0.9).is_err());
        assert!(mvsdist(&x, 0.5).is_err());
        assert!(shapiro(&x).is_err());
        assert!(fit(&x, FitFamily::Normal).is_err());
        assert!(anderson(&x, AndersonFamily::Normal).is_err());
        assert!(estimated_cdf(&x, 0., EstimatedCdfMethod::Linear).is_err());
        assert!(chatterjeexi(&[0., 1., 2.], &x, false).is_err());
        assert!(somersd(&x, &[0., 1., 2.], Alternative::TwoSided).is_err());
        assert!(spearmanrho(&x, &[0., 1., 2.], Alternative::TwoSided).is_err());
        assert!(weightedtau(&x, &[0., 1., 2.], WeightedRank::Observation, true).is_err());
        assert!(logrank(&x, &[true; 3], &[1., 2.], &[true; 2], Alternative::TwoSided).is_err());
        assert!(sobol_indices(&[0., 1.], &[1., 0.], &[&[0., invalid]]).is_err());
        assert!(power(&[&x], |_| Ok(0.5), 0.05).is_err());
    }
    for probability in [-1., 0., 1., 2., f64::NAN] {
        assert!(bayes_mvs(&[1., 2., 3.], probability).is_err());
        assert!(
            bootstrap(
                &[1., 2.],
                &[&[0, 1], &[1, 1]],
                mean,
                probability,
                BootstrapMethod::Basic
            )
            .is_err()
        );
    }
    for probability in [-1., 2., f64::NAN] {
        assert!(mvsdist(&[1., 2., 3.], probability).is_err());
    }
    assert!(bayes_mvs(&[1., 1.], 0.9).is_err());
    assert!(mvsdist(&[], 0.5).is_err());
    assert!(shapiro(&[1., 2.]).is_err());
    assert!(shapiro(&vec![1.; 5001]).is_err());
    assert!(fit(&[1., 1.], FitFamily::Normal).is_err());
    assert!(fit(&[1., 1.], FitFamily::Uniform).is_err());
    assert!(fit(&[-1., 2.], FitFamily::Exponential).is_err());
    assert!(fit(&[0., 0.], FitFamily::Exponential).is_err());
    assert!(chatterjeexi(&[-0., 0., 1.], &[0., 1., 2.], true).is_err());
    assert!(spearmanrho(&[1., 2.], &[1.], Alternative::Less).is_err());
    assert!(somersd(&[1., 2.], &[1.], Alternative::Less).is_err());
    assert!(weightedtau(&[1., 2.], &[2., 1.], WeightedRank::Explicit(&[0]), true).is_err());
    assert!(
        weightedtau(
            &[1., 2.],
            &[2., 1.],
            WeightedRank::Explicit(&[0, 1_048_577]),
            true
        )
        .is_err()
    );
    assert!(estimated_cdf(&[1., 2.], f64::NAN, EstimatedCdfMethod::Linear).is_err());
    assert!(estimated_cdf(&[], 0., EstimatedCdfMethod::Linear).is_err());
    for bounds in [(1., 1.), (2., 1.), (f64::NAN, 1.), (-65., 1.), (-1., 65.)] {
        assert!(boxcox_normmax(&[1., 2., 3.], bounds.0, bounds.1, BoxCoxMethod::Pearson).is_err());
        assert!(yeojohnson_normmax(&[1., 2., 3.], bounds.0, bounds.1).is_err());
    }
    assert!(boxcox_normmax(&[0., 1., 2.], -1., 1., BoxCoxMethod::MaximumLikelihood).is_err());
    assert!(boxcox_normmax(&[1., 1., 1.], -1., 1., BoxCoxMethod::Pearson).is_err());
    assert!(yeojohnson_normmax(&vec![1.; 65537], -1., 1.).is_err());
    assert!(ppcc_max(&vec![1.; 257], 0.1, 1., PpccFamily::Gamma).is_err());
    assert!(ppcc_plot(&[1., 2., 3.], 0., 1., 3, PpccFamily::Gamma).is_err());
    assert!(ppcc_plot(&[1., 2., 3.], 0.1, 1., 1, PpccFamily::Gamma).is_err());
    assert!(anderson(&[1., 1.], AndersonFamily::Normal).is_err());
    assert!(anderson(&[-1., 2.], AndersonFamily::Exponential).is_err());
    assert!(anderson_ksamp(&[&[1., 2.]], AndersonVariant::Midrank).is_err());
    assert!(anderson_ksamp(&[&[1., 1.], &[1., 1.]], AndersonVariant::Right).is_err());
    assert!(tukey_hsd(&[&[1., 2.]]).is_err());
    assert!(tukey_hsd(&[&[1.], &[1., 2.]]).is_err());
    assert!(cramervonmises(&[1., 2.], Distribution::Poisson { rate: 1. }).is_err());
    assert!(
        cramervonmises(
            &[1.],
            Distribution::Normal {
                mean: 0.,
                std_dev: 1.
            }
        )
        .is_err()
    );
    assert!(cramervonmises_2samp(&[1.; 10], &[2.; 11], CvmMethod::Exact).is_err());
    assert!(logrank(&[1., 2.], &[true], &[1.], &[true], Alternative::Less).is_err());
    assert!(logrank(&[-1.], &[true], &[1.], &[true], Alternative::Less).is_err());
    assert!(epps_singleton_2samp(&[1.; 5], &[1.; 5], &[0.4, 0.8]).is_err());
    assert!(epps_singleton_2samp(&[1.; 4], &[1.; 5], &[0.4]).is_err());
    assert!(epps_singleton_2samp(&[1., 2., 3., 4., 5.], &[2., 3., 4., 5., 6.], &[0.]).is_err());
    assert!(sobol_indices(&[1., 2., 3.], &[3., 2., 1.], &[&[1., 2., 3.]]).is_err());
    assert!(sobol_indices(&[1., 2.], &[3., 2.], &[&[1.]]).is_err());
    assert!(sobol_indices(&[1., 2.], &[3., 2.], &[]).is_err());
    assert!(power(&[], |_| Ok(0.5), 0.05).is_err());
    assert!(power(&[&[1., 2.], &[1.]], |_| Ok(0.5), 0.05).is_err());
    assert!(power(&[&[1., 2.]], |_| Ok(f64::NAN), 0.05).is_err());
    assert!(power(&[&[1., 2.]], |_| Ok(2.), 0.05).is_err());
    assert!(power(&[&[1., 2.]], |_| Ok(0.5), f64::NAN).is_err());
    assert!(monte_carlo_test(&[1., 2.], &[], mean, Alternative::Greater).is_err());
    assert!(monte_carlo_test(&[1., 2.], &[&[1.]], mean, Alternative::Greater).is_err());
    assert!(
        bootstrap(
            &[1., 2.],
            &[&[0, 2], &[0, 1]],
            mean,
            0.9,
            BootstrapMethod::Basic
        )
        .is_err()
    );
    assert!(bootstrap(&[1., 2.], &[&[0, 1]], mean, 0.9, BootstrapMethod::Basic).is_err());
    assert!(
        bootstrap(
            &[1., 2.],
            &[&[0], &[0, 1]],
            mean,
            0.9,
            BootstrapMethod::Basic
        )
        .is_err()
    );
    assert!(bws_test(&[1.], &[1., 2.], Alternative::Less).is_err());
    assert!(
        permutation_test(
            &[1.; 10],
            &[2.; 11],
            |_, _| panic!("limit must precede callback"),
            Alternative::Less
        )
        .is_err()
    );
    let error = Error::NumericalFailure("deliberate callback failure");
    assert_eq!(
        monte_carlo_test(
            &[1., 2.],
            &[&[1., 2.]],
            |_| Err(error.clone()),
            Alternative::Greater
        ),
        Err(error.clone())
    );
    assert_eq!(
        permutation_test(
            &[1., 2.],
            &[3., 4.],
            |_, _| Err(error.clone()),
            Alternative::Greater
        ),
        Err(error.clone())
    );
    assert_eq!(
        bootstrap(
            &[1., 2.],
            &[&[0, 1], &[1, 1]],
            |_| Err(error.clone()),
            0.9,
            BootstrapMethod::Basic
        ),
        Err(error.clone())
    );
    assert_eq!(
        power(&[&[1., 2.]], |_| Err(error.clone()), 0.05),
        Err(error)
    );
    let large = vec![0.; 65536];
    let draws = vec![large.as_slice(); 257];
    assert!(
        monte_carlo_test(
            &large,
            &draws,
            |_| panic!("work limit before callback"),
            Alternative::Less
        )
        .is_err()
    );
    assert!(power(&draws, |_| panic!("work limit before callback"), 0.05).is_err());
    assert!(
        goodness_of_fit(
            &[1., 2.],
            &[&[1., 1.]],
            FitFamily::Normal,
            GoodnessStatistic::CramerVonMises
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn added_statistics_identities_and_interoperability() -> Result<()> {
    let x = [1., 2., 3., 4., 5.];
    let posterior = bayes_mvs(&x, 0.9)?;
    let q = mvsdist(&x, 0.05)?;
    assert!((q[1].sqrt() - q[2]).abs() < 1e-14);
    assert!((posterior.variance.lower - q[1]).abs() < 1e-12);
    assert!((posterior.std_dev.lower.powi(2) - posterior.variance.lower).abs() < 1e-12);
    for p in [0.01, 0.1, 0.5, 0.9, 0.99] {
        let value = quantile(&x, p)?;
        assert!((estimated_cdf(&x, value, EstimatedCdfMethod::Linear)? - p).abs() < 1e-14);
    }
    assert_eq!(
        estimated_cdf(&[-f64::MAX, f64::MAX], 0., EstimatedCdfMethod::Linear)?,
        0.5
    );
    let fitted = fit(&x, FitFamily::Normal)?;
    let quantile = ppf(fitted, 0.7)?;
    assert!((cdf(fitted, quantile)? - 0.7).abs() < 1e-12);
    let a = [1., 2., 3.];
    let b = [3., 2., 1.];
    assert_eq!(
        spearmanrho(&a, &b, Alternative::Less)?,
        pearsonr(
            &rankdata(&a, RankMethod::Average)?,
            &rankdata(&b, RankMethod::Average)?,
            Alternative::Less
        )?
    );
    let base = shapiro(&x)?;
    let shifted = shapiro(&x.map(|v| v * 1e-300))?;
    assert!((base.statistic - shifted.statistic).abs() < 1e-12);
    assert!((base.pvalue - shifted.pvalue).abs() < 1e-12);
    let comparison = tukey_hsd(&[&a, &x])?;
    let t = ttest_ind(&a, &x, true, Alternative::TwoSided)?;
    assert!((comparison.pvalue[0][1] - t.pvalue).abs() < 2e-7);
    assert_eq!(comparison.statistic[0][1], -comparison.statistic[1][0]);
    let null = Distribution::Uniform { low: 0., high: 1. };
    let cvm = cramervonmises(&[1. / 6., 0.5, 5. / 6.], null)?;
    assert!((cvm.statistic - 1. / 36.).abs() < 1e-15);
    assert_eq!(cvm.pvalue, 1.);
    let perm = permutation_test(
        &a,
        &x,
        |a, b| Ok(mean(a)? - mean(b)?),
        Alternative::TwoSided,
    )?;
    let reversed = permutation_test(
        &x,
        &a,
        |a, b| Ok(mean(a)? - mean(b)?),
        Alternative::TwoSided,
    )?;
    assert_eq!(perm.null_distribution.len(), 56);
    assert_eq!(perm.pvalue, reversed.pvalue);
    assert_eq!(perm.statistic, -reversed.statistic);
    let indices: &[&[usize]] = &[&[0, 0, 1], &[0, 1, 2], &[1, 2, 2]];
    let percentile = bootstrap(&a, indices, mean, 0.9, BootstrapMethod::Percentile)?;
    let basic = bootstrap(&a, indices, mean, 0.9, BootstrapMethod::Basic)?;
    assert_eq!(basic.lower, 2. * mean(&a)? - percentile.upper);
    assert_eq!(basic.upper, 2. * mean(&a)? - percentile.lower);
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut ab_x = Vec::new();
    let mut ab_y = Vec::new();
    for i in 0..16 {
        let coordinate = |bit| if i & (1 << bit) == 0 { -1. } else { 1. };
        let (ax, ay, bx, by) = (coordinate(0), coordinate(1), coordinate(2), coordinate(3));
        a.push(ax + 2. * ay);
        b.push(bx + 2. * by);
        ab_x.push(bx + 2. * ay);
        ab_y.push(ax + 2. * by);
    }
    let r = sobol_indices(&a, &b, &[&ab_x, &ab_y])?;
    for (actual, expected) in r
        .first_order
        .iter()
        .chain(&r.total_order)
        .zip([0.2, 0.8, 0.2, 0.8])
    {
        assert!((actual - expected).abs() < 1e-14);
    }
    let groups = [&[1., 2., 3., 4., 5.][..], &[2., 3., 4., 5., 6.][..]];
    let statistic = |a: &[f64]| {
        ks_1samp(a, fit(a, FitFamily::Uniform)?, Alternative::TwoSided).map(|r| r.statistic)
    };
    assert_eq!(
        goodness_of_fit(
            &x,
            &groups,
            FitFamily::Uniform,
            GoodnessStatistic::KolmogorovSmirnov
        )?,
        monte_carlo_test(&x, &groups, statistic, Alternative::Greater)?
    );
    Ok(())
}

#[test]
fn final_inventory_groups_validate_domains_and_identities() -> Result<()> {
    let normal = Distribution::Normal {
        mean: 0.,
        std_dev: 1.,
    };
    let discrete = Distribution::Poisson { rate: 1. };
    assert!(truncate(discrete, 0., 2.).is_err());
    assert!(order_statistic(discrete, 1, 2).is_err());
    for bounds in [
        (1., 1.),
        (2., 1.),
        (f64::NAN, 1.),
        (0., f64::INFINITY),
        (100., 101.),
    ] {
        assert!(truncate(normal, bounds.0, bounds.1).is_err());
    }
    assert!(truncate(Distribution::Uniform { low: 0., high: 1. }, 2., 3.).is_err());
    for (rank, count) in [(0, 1), (1, 0), (3, 2), (1, 1_048_577)] {
        assert!(order_statistic(normal, rank, count).is_err());
    }
    let custom = make_distribution(
        0.,
        1.,
        |x| Ok(2. * x),
        |x| Ok(x * x),
        |x| Ok(1. - x * x),
        |p| Ok(p.sqrt()),
    )?;
    assert!(cdf(custom, f64::NAN).is_err());
    assert!(ppf(custom, 1.1).is_err());
    assert!(make_distribution(1., 0., |_| Ok(1.), Ok, |x| Ok(1. - x), Ok).is_err());
    assert!(make_distribution(0., 1., |_| Ok(-1.), Ok, |x| Ok(1. - x), Ok).is_err());
    assert!(make_distribution(0., 1., |_| Ok(1.), |_| Ok(0.5), |_| Ok(0.5), Ok).is_err());
    assert!(make_distribution(0., 1., |_| Ok(1.), Ok, |x| Ok(1. - x), |_| Ok(2.)).is_err());
    let failure = Error::NumericalFailure("custom callback failure");
    assert!(
        matches!(make_distribution(0.,1.,|_| Ok(1.),|_| Err(Error::NumericalFailure("custom callback failure")),|x| Ok(1.-x),Ok),Err(e) if e == failure)
    );
    for p in [0.01, 0.5, 0.99] {
        let q = ppf(custom, p)?;
        assert!((cdf(custom, q)? - p).abs() < 1e-14);
    }
    let conditional = truncate(normal, 9., 10.)?;
    for p in [0.01, 0.5, 0.99] {
        let q = ppf(conditional, p)?;
        assert!((9. ..=10.).contains(&q));
        assert!((cdf(conditional, q)? - p).abs() < 1e-12);
        assert!((cdf(conditional, q)? + sf(conditional, q)? - 1.).abs() < 1e-12);
    }
    let uniform = Distribution::Uniform { low: 0., high: 1. };
    let order = order_statistic(uniform, 2, 3)?;
    for x in [0.1, 0.5, 0.9] {
        assert!((pdf(order, x)? - 6. * x * (1. - x)).abs() < 1e-13);
        assert!((cdf(order, x)? - (3. * x * x - 2. * x * x * x)).abs() < 1e-13);
    }
    let maximum = order_statistic(uniform, 1_048_576, 1_048_576)?;
    assert!((pdf(maximum, 1.)? / 1_048_576. - 1.).abs() < 1e-12);
    for invalid in [-1., 0.5, 65., f64::NAN, f64::INFINITY] {
        let table = [invalid, 1., 1., 1.];
        assert!(barnard_exact(&table, Alternative::Less, true).is_err());
        assert!(boschloo_exact(&table, Alternative::Less).is_err());
    }
    assert!(barnard_exact(&[20.; 4], Alternative::Less, true).is_err());
    let table = [7., 12., 8., 3.];
    let reversed = [12., 7., 3., 8.];
    let first = barnard_exact(&table, Alternative::Less, true)?;
    let second = barnard_exact(&reversed, Alternative::Greater, true)?;
    assert_eq!(first.statistic, -second.statistic);
    assert!((first.pvalue - second.pvalue).abs() < 1e-12);
    let first = boschloo_exact(&table, Alternative::Less)?;
    let second = boschloo_exact(&reversed, Alternative::Greater)?;
    assert!((first.statistic - second.statistic).abs() < 1e-15);
    assert!((first.pvalue - second.pvalue).abs() < 1e-12);
    assert!(first.pvalue <= fisher_exact(&table, Alternative::Less)?.pvalue);
    for p in [
        (-1., 1., 1., 1., 0.),
        (0.5, 1., 1., 1., 0.),
        (1., 0., 1., 1., 0.),
        (1., 1., 1., 1., -1.),
        (2049., 1., 0., 1., 0.),
    ] {
        assert!(poisson_means_test(p.0, p.1, p.2, p.3, p.4, Alternative::TwoSided).is_err());
    }
    let first = poisson_means_test(0., 100., 3., 100., 0., Alternative::Less)?;
    let second = poisson_means_test(3., 100., 0., 100., 0., Alternative::Greater)?;
    assert_eq!(first.statistic, -second.statistic);
    assert!((first.pvalue - second.pvalue).abs() < 1e-14);
    assert!(wasserstein_distance_nd(&[], &[&[1.]]).is_err());
    assert!(wasserstein_distance_nd(&[&[1.]], &[&[1., 2.]]).is_err());
    assert!(wasserstein_distance_nd(&[&[1., 2.], &[1.]], &[&[1., 2.]]).is_err());
    assert!(wasserstein_distance_nd(&[&[f64::NAN]], &[&[1.]]).is_err());
    let large1 = vec![&[1.][..]; 17];
    let large2 = vec![&[1.][..]; 19];
    assert!(wasserstein_distance_nd(&large1, &large2).is_err());
    let first: &[&[f64]] = &[&[0.], &[1.]];
    let second: &[&[f64]] = &[&[1.], &[2.], &[3.]];
    assert!(
        (wasserstein_distance_nd(first, second)? - wasserstein_distance(&[0., 1.], &[1., 2., 3.])?)
            .abs()
            < 1e-14
    );
    assert!(
        (wasserstein_distance_nd(first, second)? - wasserstein_distance_nd(second, first)?).abs()
            < 1e-14
    );
    let n = [1., 2., 3.];
    let d = dunnett(&n, &[&[4., 5., 6.]], Alternative::TwoSided)?;
    let t = ttest_ind(&[4., 5., 6.], &n, true, Alternative::TwoSided)?;
    assert!((d.statistic[0] - t.statistic).abs() < 1e-14);
    assert!((d.pvalue[0] - t.pvalue).abs() < 1e-14);
    assert!(dunnett(&n, &[], Alternative::Less).is_err());
    assert!(dunnett(&[1.], &[&n], Alternative::Less).is_err());
    assert!(dunnett(&[1., 1.], &[&[2., 2.]], Alternative::Less).is_err());
    assert!(dunnett(&n, &[&[1., f64::NAN]], Alternative::Less).is_err());
    assert!(dunnett(&[1.; 257], &[&n], Alternative::Less).is_err());
    let points: &[&[f64]] = &[&[0.], &[1.], &[2.], &[3.], &[4.]];
    let permutations: &[&[usize]] = &[&[0, 1, 2, 3, 4], &[4, 3, 2, 1, 0], &[1, 3, 0, 4, 2]];
    let mgc = multiscale_graphcorr(points, points, permutations)?;
    assert!((mgc.statistic - 1.).abs() < 1e-12);
    assert_eq!(mgc.mgc_map.len(), 5);
    assert!(multiscale_graphcorr(points, &points[..4], permutations).is_err());
    assert!(multiscale_graphcorr(points, points, &[]).is_err());
    assert!(multiscale_graphcorr(points, points, &[&[0, 1, 2, 3, 3]]).is_err());
    assert!(multiscale_graphcorr(points, points, &[&[0, 1, 2, 3, 5]]).is_err());
    assert!(multiscale_graphcorr(points, points, &[&[0, 1, 2]]).is_err());
    assert!(multiscale_graphcorr(&[&[f64::NAN][..]; 5], points, permutations).is_err());
    Ok(())
}

#[test]
fn global_exact_limits_and_distribution_callbacks() -> Result<()> {
    // Perfect one-sided separation has only one extreme table. Its null mass
    // θ^5(1−θ)^5 is maximized at θ=1/2, an independent global-maximum identity.
    let table = [5., 0., 0., 5.];
    let barnard = barnard_exact(&table, Alternative::Greater, false)?;
    assert_eq!(barnard.statistic, f64::INFINITY);
    assert!((barnard.pvalue - 2_f64.powi(-10)).abs() < 1e-13);
    assert!((boschloo_exact(&table, Alternative::Greater)?.pvalue - 2_f64.powi(-10)).abs() < 1e-13);
    let d = make_distribution(
        0.,
        1.,
        |x| {
            if x == 0.25 {
                Err(Error::NumericalFailure("density failure"))
            } else {
                Ok(2. * x)
            }
        },
        |x| Ok(x * x),
        |x| Ok(1. - x * x),
        |p| Ok(p.sqrt()),
    )?;
    assert_eq!(
        pdf(d, 0.25),
        Err(Error::NumericalFailure("density failure"))
    );
    let d = make_distribution(
        0.,
        1.,
        |x| Ok(2. * x),
        |x| if x == 0.25 { Ok(2.) } else { Ok(x * x) },
        |x| Ok(1. - x * x),
        |p| Ok(p.sqrt()),
    )?;
    assert!(cdf(d, 0.25).is_err());
    let points = vec![&[0.][..]; 128];
    let indices = (0..128).collect::<Vec<_>>();
    let permutations = vec![indices.as_slice(); 1024];
    assert!(multiscale_graphcorr(&points, &points, &permutations).is_err());
    let points = vec![&[0.][..]; 129];
    assert!(multiscale_graphcorr(&points, &points, &[indices.as_slice()]).is_err());
    assert!(
        multiscale_graphcorr(&[&[0.; 65][..]; 5], &[&[0.][..]; 5], &[&[0, 1, 2, 3, 4]]).is_err()
    );
    let points: &[&[f64]] = &[&[0., 0.], &[0., 1.], &[1., 0.], &[1., 1.], &[2., 0.]];
    let permutations: &[&[usize]] = &[&[0, 1, 2, 3, 4], &[1, 3, 0, 4, 2]];
    let result = multiscale_graphcorr(points, points, permutations)?;
    let scaled = points
        .iter()
        .map(|row| row.iter().map(|x| x * 1e150).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let rows = scaled.iter().map(Vec::as_slice).collect::<Vec<_>>();
    assert!(
        (result.statistic - multiscale_graphcorr(&rows, &rows, permutations)?.statistic).abs()
            < 1e-12
    );
    Ok(())
}
