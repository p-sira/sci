mod common;
use common::{matrix, vector as numbers};

use sci::{
    Error, Result, complex::Complex, fft, filter, linalg, optimize, signal, spatial, special, stats,
};

fn evaluate(row: &[&str]) -> Result<Vec<f64>> {
    let (x, y) = if matches!(row[1], "cdist" | "pdist") {
        (vec![], vec![])
    } else {
        (numbers(row[2]), numbers(row[3]))
    };
    let p = numbers(row[4]);
    let alternative = |value| match value as usize {
        0 => stats::Alternative::TwoSided,
        1 => stats::Alternative::Less,
        2 => stats::Alternative::Greater,
        _ => panic!("invalid alternative"),
    };
    Ok(match row[1] {
        "expectile" => vec![stats::expectile(&x, p[0])?],
        "normal_pdf" => vec![stats::normal_pdf(x[0], p[0], p[1])?],
        "normal_cdf" => vec![stats::normal_cdf(x[0], p[0], p[1])?],
        "normal_ppf" => vec![stats::normal_ppf(x[0], p[0], p[1])?],
        "normal_generic_pdf" | "normal_logpdf" | "normal_generic_cdf" | "normal_sf"
        | "normal_generic_ppf" => {
            let d = stats::Distribution::Normal {
                mean: p[0],
                std_dev: p[1],
            };
            vec![match row[1] {
                "normal_generic_pdf" => stats::pdf(d, x[0])?,
                "normal_logpdf" => stats::logpdf(d, x[0])?,
                "normal_generic_cdf" => stats::cdf(d, x[0])?,
                "normal_sf" => stats::sf(d, x[0])?,
                _ => stats::ppf(d, x[0])?,
            }]
        }
        "gradient_descent" => {
            let r = optimize::minimize_with_gradient(
                |x| x.iter().map(|v| 0.5 * p[0] * v * v).sum(),
                |x| x.iter().map(|v| p[0] * v).collect(),
                &x,
                optimize::Options {
                    learning_rate: p[1],
                    tolerance: p[2],
                    max_iterations: p[3] as u64,
                },
            )?;
            r.point
                .into_iter()
                .chain([
                    r.cost,
                    r.iterations as f64,
                    match r.status {
                        optimize::Status::Converged => 0.,
                        optimize::Status::IterationLimit => 1.,
                        optimize::Status::NoProgress => 2.,
                    },
                ])
                .collect()
        }
        "cosine" => vec![spatial::cosine(&x, &y, Some(&p))?],
        "correlation" => vec![spatial::correlation(&x, &y, Some(&p), true)?],
        "braycurtis" => vec![spatial::braycurtis(&x, &y, Some(&p))?],
        "euclidean" => vec![spatial::euclidean(&x, &y, Some(&p))?],
        "sqeuclidean" => vec![spatial::sqeuclidean(&x, &y, Some(&p))?],
        "cityblock" => vec![spatial::cityblock(&x, &y, Some(&p))?],
        "chebyshev" => vec![spatial::chebyshev(&x, &y, Some(&p))?],
        "minkowski" => vec![spatial::minkowski(&x, &y, p[0], Some(&p[1..]))?],
        "ttest_ind_from_stats" | "ttest_ind" => {
            let t = if row[1] == "ttest_ind" {
                stats::ttest_ind(&x, &y, p[0] != 0., alternative(p[1]))?
            } else {
                stats::ttest_ind_from_stats(
                    x[0],
                    x[1],
                    x[2] as usize,
                    y[0],
                    y[1],
                    y[2] as usize,
                    p[0] != 0.,
                    alternative(p[1]),
                )?
            };
            vec![t.statistic, t.pvalue, t.df]
        }
        "rel_entr" => vec![special::elementary::rel_entr(x[0], x[1])],
        "kl_div" => vec![special::elementary::kl_div(x[0], x[1])],
        "pseudo_huber" => vec![special::elementary::pseudo_huber(x[0], x[1])],
        "particle_update" => {
            let prior = filter::ParticleSet {
                states: (0..x.len()).map(|i| vec![i as f64]).collect(),
                weights: x,
            };
            let posterior = filter::particle_update(&prior, |state| Ok(y[state[0] as usize]))?;
            assert_eq!(posterior.states, prior.states);
            assert_eq!(posterior.weights.len(), prior.weights.len());
            posterior.weights
        }
        "coherence" => {
            let (frequencies, values) = signal::coherence(
                &x,
                &y,
                &matrix(row[5])[0],
                p[0] as usize,
                p[1] as usize,
                p[2],
            )?;
            assert_eq!(frequencies.len(), p[1] as usize / 2 + 1);
            assert_eq!(frequencies.len(), values.len());
            frequencies.into_iter().chain(values).collect()
        }
        "fractional_matrix_power" => {
            let a = matrix(row[5]);
            let power = linalg::fractional_matrix_power(&a, p[0])?;
            assert_eq!(power.len(), a.len());
            assert!(power.iter().all(|row| row.len() == a.len()));
            power
                .into_iter()
                .flatten()
                .flat_map(|z| [z.re, z.im])
                .collect()
        }
        "mean" => vec![stats::mean(&x)?],
        "variance" => vec![stats::variance(&x, p[0] as usize)?],
        "std" => vec![stats::std(&x, p[0] as usize)?],
        "zscore" => stats::zscore(&x, p[0] as usize)?,
        "zmap" => stats::zmap(&x, &y, p[0] as usize)?,
        "gzscore" => stats::gzscore(&x, p[0] as usize)?,
        "moment" => vec![stats::moment(&x, p[0] as usize)?],
        "skew" => vec![stats::skew(&x, true)?],
        "kurtosis" => vec![stats::kurtosis(&x, true, true)?],
        "linregress" => {
            let r = stats::linregress(&x, &y, stats::Alternative::TwoSided)?;
            vec![r.slope, r.intercept, r.stderr, r.intercept_stderr]
        }
        "fft" | "ifft" | "rfft" => {
            let values: Vec<_> = x
                .iter()
                .zip(&y)
                .map(|(&re, &im)| Complex { re, im })
                .collect();
            let output = match row[1] {
                "fft" => fft::fft(&values),
                "ifft" => fft::ifft(&values),
                _ => fft::rfft(&x),
            };
            assert_eq!(
                output.len(),
                if row[1] == "rfft" {
                    x.len() / 2 + 1
                } else {
                    x.len()
                }
            );
            output.iter().flat_map(|z| [z.re, z.im]).collect()
        }
        "savgol_coeffs" => {
            signal::savgol_coeffs(p[0] as usize, p[1] as usize, p[2] as usize, p[3])?
        }
        "savgol_filter" => {
            signal::savgol_filter(&x, p[0] as usize, p[1] as usize, p[2] as usize, p[3])?
        }
        "cdist" => {
            let (x, y, vi) = (matrix(row[2]), matrix(row[3]), matrix(row[5]));
            let out = spatial::cdist(&x, &y, spatial::Metric::Mahalanobis(&vi))?;
            assert_eq!(out.len(), x.len());
            assert!(out.iter().all(|row| row.len() == y.len()));
            out.into_iter().flatten().collect()
        }
        "pdist" => {
            let (x, vi) = (matrix(row[2]), matrix(row[5]));
            let out = spatial::pdist(&x, spatial::Metric::Mahalanobis(&vi))?;
            assert_eq!(out.len(), x.len() * (x.len() - 1) / 2);
            out
        }
        api => panic!("unknown regression API {api}"),
    })
}

#[test]
fn audit_fixes_match_independent_references() {
    let inputs = common::lines(include_str!("fixtures/audit_regressions_inputs.csv"));
    let expected: Vec<_> =
        common::lines(include_str!("fixtures/audit_regressions_expected.csv")).collect();
    assert_eq!(inputs.clone().count(), expected.len());
    for (input, expected) in inputs.zip(expected) {
        let row: Vec<_> = common::fields(input);
        let reference: Vec<_> = common::fields(expected);
        assert_eq!(row[0], reference[0]);
        let values = numbers(reference[1]);
        let rtol: f64 = reference[2].parse().unwrap();
        let atol: f64 = reference[3].parse().unwrap();
        // Repeated calls and successive cases exercise workspace reuse across
        // lengths and input values, including forward/inverse interleaving.
        for _ in 0..3 {
            let actual = evaluate(&row).unwrap_or_else(|error| panic!("case {input}: {error}"));
            assert_eq!(actual.len(), values.len(), "shape: {input}");
            for (a, b) in actual.into_iter().zip(&values) {
                if b.is_nan() {
                    assert!(a.is_nan(), "case {input}: expected NaN, got {a}");
                    continue;
                }
                if b.is_infinite() {
                    assert_eq!(a, *b, "case {input}: infinity sign");
                    continue;
                }
                if matches!(row[1], "rel_entr" | "kl_div" | "pseudo_huber")
                    && *b != 0.
                    && b.abs() < 1e-200
                {
                    assert_ne!(a, 0., "case {input}: lost representable nonzero result");
                }
                assert!(
                    a.is_finite() && approx::abs_diff_eq!(a, b, epsilon = atol + rtol * b.abs()),
                    "case {input}: {a} != {b}"
                );
            }
        }
    }
}

#[test]
fn numerical_losses_and_probabilities_preserve_independent_identities() {
    for x in [1e-300_f64, 1e-100, 1., 1e100, 1e308] {
        assert_eq!(special::elementary::kl_div(x, x), 0.);
        for y in [x.next_down(), x.next_up()] {
            let divergence = special::elementary::kl_div(x, y);
            assert!(divergence >= 0. && divergence.is_finite());
            // Neighbor differences at these scales give strictly positive KL
            // except when its squared relative difference truly underflows.
            if x >= 1e-100 {
                assert!(divergence > 0.);
            }
        }
    }
    for delta in [1e-308, 1., 1e308] {
        for r in [1e-155, 1., 1e150, 1e308] {
            assert_eq!(
                special::elementary::pseudo_huber(delta, r),
                special::elementary::pseudo_huber(delta, -r)
            );
        }
    }
    let prior = filter::ParticleSet {
        states: vec![vec![0.], vec![1.], vec![2.]],
        weights: vec![1., 1., f64::from_bits(1)],
    };
    let posterior = filter::particle_update(&prior, |state| {
        Ok(if state[0] == 2. {
            0.
        } else {
            f64::NEG_INFINITY
        })
    })
    .unwrap();
    assert_eq!(posterior.states, prior.states);
    assert_eq!(posterior.weights, vec![0., 0., 1.]);
    let impossible = filter::particle_update(&prior, |_| Ok(f64::NEG_INFINITY));
    assert!(matches!(impossible, Err(Error::NumericalFailure(_))));
    for exponent in [-4294967296., -4294967297., 4294967297., f64::NAN] {
        assert!(linalg::fractional_matrix_power(&[[0.]], exponent).is_err());
    }
}

#[test]
fn precision_fixes_preserve_scale_exclusion_and_inverse_identities() -> Result<()> {
    for scale in [1e-308, 1., 1e308] {
        let d = stats::Distribution::Normal {
            mean: 0.,
            std_dev: scale,
        };
        for p in [0.1, 0.5, 0.9] {
            let x = stats::normal_ppf(p, 0., scale)?;
            if x.is_finite() {
                assert_eq!(stats::ppf(d, p)?, x);
                assert_eq!(stats::cdf(d, x)?, stats::normal_cdf(x, 0., scale)?);
                assert!((stats::cdf(d, x)? - p).abs() < 2e-14);
                assert!((stats::cdf(d, x)? + stats::sf(d, x)? - 1.).abs() < 2e-15);
            }
        }
        assert_eq!(stats::normal_cdf(f64::INFINITY, 0., scale)?, 1.);
        assert_eq!(stats::normal_cdf(f64::NEG_INFINITY, 0., scale)?, 0.);
    }
    let u = [1e-300, 2e-300];
    let v = [2e-300, 1e-300];
    let with_excluded_u = [u[0], u[1], 1e308];
    let with_excluded_v = [v[0], v[1], -1e308];
    for metric in [spatial::cosine, spatial::braycurtis] {
        assert_eq!(
            metric(&u, &v, None)?,
            metric(&with_excluded_u, &with_excluded_v, Some(&[1., 1., 0.]))?
        );
    }
    assert_eq!(
        spatial::correlation(&u, &v, None, true)?,
        spatial::correlation(
            &with_excluded_u,
            &with_excluded_v,
            Some(&[1., 1., 0.]),
            true
        )?
    );
    for factor in [1e-150, 1., 1e150] {
        let x = [factor, 2. * factor];
        let norm = spatial::euclidean(&x, &[0.; 2], Some(&[1e308; 2]));
        if factor <= 1. {
            let norm = norm?;
            let scaled = norm / 1e154 / factor;
            assert!((scaled - 5_f64.sqrt()).abs() < 2e-15);
        } else {
            assert!(norm?.is_finite());
        }
    }
    for alpha in [0.1, 0.25, 0.5, 0.75, 0.9] {
        let x = [-3., 1., 4., 10.];
        let result = stats::expectile(&x, alpha)?;
        let negated: Vec<_> = x.iter().map(|v| -v).collect();
        assert!((stats::expectile(&negated, 1. - alpha)? + result).abs() < 2e-14);
        for scale in [1e-100, 1e100] {
            let scaled: Vec<_> = x.iter().map(|v| v * scale).collect();
            assert!((stats::expectile(&scaled, alpha)? / scale - result).abs() < 2e-14);
        }
    }
    Ok(())
}

#[test]
fn precision_fixes_keep_invalid_domains_and_true_overflow_explicit() {
    for (mean, scale) in [(f64::NAN, 1.), (0., 0.), (0., -1.), (0., f64::INFINITY)] {
        let d = stats::Distribution::Normal {
            mean,
            std_dev: scale,
        };
        assert!(stats::normal_cdf(f64::INFINITY, mean, scale).is_err());
        assert!(stats::normal_ppf(0., mean, scale).is_err());
        assert!(stats::sf(d, f64::INFINITY).is_err());
        assert!(stats::ppf(d, 1.).is_err());
    }
    assert!(stats::normal_pdf(f64::NAN, 0., 1.).is_err());
    assert!(stats::normal_ppf(f64::NAN, 0., 1.).is_err());
    assert!(stats::normal_ppf(1.1, 0., 1.).is_err());
    assert!(stats::normal_ppf(0.99, 0., 1e308).is_err());
    assert!(stats::normal_pdf(0., 0., f64::from_bits(1)).is_err());
    assert!(stats::expectile(&[], 0.5).is_err());
    assert!(stats::expectile(&[1.], f64::NAN).is_err());
    assert!(stats::expectile(&[f64::INFINITY], 0.5).is_err());
    assert!(spatial::euclidean(&[1e308; 2], &[-1e308; 2], None).is_err());
    assert!(spatial::euclidean(&[1.], &[0.], Some(&[0.])).is_err());
    assert!(spatial::euclidean(&[1.], &[0.], Some(&[-1.])).is_err());
    assert!(spatial::minkowski(&[1.], &[0.], 0., None).is_err());
}

struct BudgetMatrix(usize, usize);
impl linalg::MatrixInput for BudgetMatrix {
    fn matrix_shape(&self) -> Result<(usize, usize)> {
        Ok((self.0, self.1))
    }
    fn matrix_entry(&self, _: usize, _: usize) -> f64 {
        panic!("work must be bounded before input reads")
    }
}

#[test]
fn work_limits_precede_callbacks_and_input_reads() {
    let options = optimize::Options {
        max_iterations: 0,
        ..Default::default()
    };
    let objective = |_: &[f64]| -> f64 { panic!("invalid work must precede callbacks") };
    assert!(optimize::minimize(objective, &vec![0.0; 1000], options).is_err());
    assert!(
        optimize::minimize(
            objective,
            &[0.0],
            optimize::Options {
                max_iterations: u64::MAX,
                ..options
            }
        )
        .is_err()
    );
    assert!(optimize::minimize(objective, &[0.0; 129], Default::default()).is_err());
    assert!(
        optimize::minimize_with_gradient(
            objective,
            |_| panic!("invalid work"),
            &[0.0],
            optimize::Options {
                max_iterations: 16_777_216,
                ..options
            }
        )
        .is_err()
    );
    // Largest n whose n*(n+1) simplex fits the storage bound.
    assert_eq!(
        optimize::minimize(|_| 1.0, &vec![0.0; 999], options)
            .unwrap()
            .point
            .len(),
        999
    );
    assert_eq!(
        linalg::matmul(&BudgetMatrix(257, 256), &BudgetMatrix(256, 256)),
        Err(Error::InvalidInput(
            "matrix multiplication work exceeds 16777216"
        ))
    );
    let a = vec![vec![0.0; 256]; 256];
    let b = vec![vec![1.0; 256]; 256];
    assert_eq!(linalg::matmul(&a, &b).unwrap(), a);
}

#[test]
fn batch_mahalanobis_reuses_a_validated_metric_with_new_work_bound() {
    let vi: Vec<_> = (0..16)
        .map(|i| (0..16).map(|j| f64::from(i == j)).collect::<Vec<_>>())
        .collect();
    let x: Vec<_> = (0..64).map(|i| vec![i as f64; 16]).collect();
    // The old per-pair cubic work count rejected this batch.
    let distances = spatial::cdist(&x, &x, spatial::Metric::Mahalanobis(&vi)).unwrap();
    for (i, row) in distances.iter().enumerate() {
        for (j, &d) in row.iter().enumerate() {
            let expected = 4.0 * i.abs_diff(j) as f64;
            // Sixteen hypot accumulations introduce a few ulps of roundoff.
            assert!((d - expected).abs() <= 8.0 * f64::EPSILON * expected);
        }
    }
    let condensed = spatial::pdist(&x, spatial::Metric::Mahalanobis(&vi)).unwrap();
    assert_eq!(
        condensed,
        distances
            .iter()
            .enumerate()
            .flat_map(|(i, row)| row[i + 1..].iter().copied())
            .collect::<Vec<_>>()
    );
    let invalid = vec![vec![0.0; 16]; 16];
    assert!(spatial::cdist(&[], &x, spatial::Metric::Mahalanobis(&invalid)).is_err());
    assert!(spatial::pdist(&x[..1], spatial::Metric::Mahalanobis(&invalid)).is_err());
    assert!(spatial::cdist(&x, &x, spatial::Metric::Mahalanobis(&vi[..15])).is_err());
}

#[test]
fn fft_workspaces_preserve_round_trips_and_thread_isolation() {
    let threads: Vec<_> = (0..4)
        .map(|thread| {
            std::thread::spawn(move || {
                for n in [1, 2, 7, 32, 65_536, 65_537, 2] {
                    let x: Vec<_> = (0..n)
                        .map(|i| (i as f64 * 0.01 + thread as f64).sin())
                        .collect();
                    let restored = fft::irfft(&fft::rfft(&x), Some(n)).unwrap();
                    assert_eq!(restored.len(), n);
                    for (a, b) in restored.iter().zip(&x) {
                        assert!((a - b).abs() < 2e-11);
                    }
                    let z: Vec<_> = x.iter().map(|&re| Complex { re, im: -0.5 * re }).collect();
                    for (a, b) in fft::ifft(&fft::fft(&z)).iter().zip(z) {
                        assert!((a.re - b.re).abs() < 2e-11 && (a.im - b.im).abs() < 2e-11);
                    }
                }
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }
    assert!(fft::rfft::<f64>(&[]).is_empty());
    assert!(fft::rfft(&[f64::NAN, 1.0]).iter().any(|z| z.re.is_nan()));
}
