mod common;
use common::assert_close;
use common::vector as numbers;

use sci::{Error, stats as s};

fn distribution(name: &str, p: &[f64]) -> s::distributions::Distribution {
    use s::distributions::Distribution::*;
    match name {
        "Poisson" => Poisson { rate: p[0] },
        "NegativeBinomial" => NegativeBinomial {
            successes: p[0],
            probability: p[1],
        },
        "Bernoulli" => Bernoulli { probability: p[0] },
        "Binomial" => Binomial {
            trials: p[0],
            probability: p[1],
        },
        "Geometric" => Geometric { probability: p[0] },
        "DiscreteUniform" => DiscreteUniform {
            low: p[0],
            high: p[1],
        },
        "Beta" => Beta { a: p[0], b: p[1] },
        "Gamma" => Gamma {
            shape_value: p[0],
            scale: p[1],
        },
        "InverseGamma" => InverseGamma {
            shape_value: p[0],
            scale: p[1],
        },
        _ => panic!("unknown distribution {name}"),
    }
}

#[test]
fn support_endpoints_degenerate_distributions_and_posteriors_match_scipy() {
    let inputs: Vec<_> =
        common::lines(include_str!("fixtures/statistics_boundary_inputs.csv")).collect();
    let expected: Vec<_> =
        common::lines(include_str!("fixtures/statistics_boundary_expected.csv")).collect();
    assert_eq!(inputs.len(), expected.len());
    assert_eq!(inputs.len(), 556);
    for (input, expected) in inputs.iter().zip(expected) {
        let fields: Vec<_> = common::fields(input);
        let reference: Vec<_> = common::fields(expected);
        assert_eq!(fields[0], reference[0]);
        let p = numbers(fields[3]);
        let got = match fields[1] {
            "mvsdist" => s::estimation::mvsdist(&numbers(fields[4]), p[0])
                .unwrap()
                .to_vec(),
            "bayes_mvs" => {
                let r = s::estimation::bayes_mvs(&numbers(fields[4]), p[0]).unwrap();
                [r.mean, r.variance, r.std_dev]
                    .into_iter()
                    .flat_map(|v| [v.statistic, v.lower, v.upper])
                    .collect()
            }
            api => {
                let d = distribution(fields[2], &p);
                let x = fields[4].parse().unwrap();
                let result = match api {
                    "pdf" => s::distributions::pdf(d, x),
                    "logpdf" => s::distributions::logpdf(d, x),
                    "cdf" => s::distributions::cdf(d, x),
                    "sf" => s::distributions::sf(d, x),
                    "ppf" => s::distributions::ppf(d, x),
                    _ => panic!("unknown API {api}"),
                };
                vec![result.unwrap_or_else(|e| panic!("{input}: {e}"))]
            }
        };
        let expected = numbers(reference[1]);
        assert_eq!(got.len(), expected.len(), "{input}");
        for (actual, expected) in got.into_iter().zip(expected) {
            assert_close(actual, expected, 2e-10, 2e-10, input);
        }
    }
}

#[test]
fn distribution_validation_checks_every_parameter_before_endpoint_shortcuts() {
    use s::distributions::Distribution::*;
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let invalid = [
            Normal {
                mean: bad,
                std_dev: 1.0,
            },
            Normal {
                mean: 0.0,
                std_dev: bad,
            },
            Beta { a: 1.0, b: bad },
            Gamma {
                shape_value: 1.0,
                scale: bad,
            },
            Cauchy {
                location: bad,
                scale: 1.0,
            },
            Laplace {
                location: 0.0,
                scale: bad,
            },
            LogNormal {
                log_mean: bad,
                log_std: 1.0,
            },
            Uniform {
                low: bad,
                high: 1.0,
            },
            Weibull {
                shape_value: 1.0,
                scale: bad,
            },
            Pareto {
                minimum: 1.0,
                shape_value: bad,
            },
            InverseGamma {
                shape_value: bad,
                scale: 1.0,
            },
            Gumbel {
                location: bad,
                scale: 1.0,
            },
            Triangular {
                low: 0.0,
                high: 1.0,
                mode: bad,
            },
            Binomial {
                trials: bad,
                probability: 0.5,
            },
            NegativeBinomial {
                successes: bad,
                probability: 1.0,
            },
            Hypergeometric {
                population: 10.0,
                successes: 5.0,
                draws: bad,
            },
        ];
        for d in invalid {
            assert!(s::distributions::pdf(d, 0.0).is_err(), "{d:?}");
            assert!(s::distributions::logpdf(d, f64::INFINITY).is_err(), "{d:?}");
            assert!(s::distributions::cdf(d, f64::INFINITY).is_err(), "{d:?}");
            assert!(s::distributions::sf(d, f64::NEG_INFINITY).is_err(), "{d:?}");
            assert!(s::distributions::ppf(d, 0.0).is_err(), "{d:?}");
        }
    }
    for d in [
        Beta {
            a: 1_048_577.0,
            b: 1.0,
        },
        Erlang {
            shape_value: 0.0,
            scale: 1.0,
        },
        Gamma {
            shape_value: 1.0,
            scale: f64::from_bits(1),
        },
        DiscreteUniform {
            low: 0.5,
            high: 2.0,
        },
        DiscreteUniform {
            low: 0.0,
            high: 1_048_577.0,
        },
        Uniform {
            low: -f64::MAX,
            high: f64::MAX,
        },
    ] {
        assert!(s::distributions::cdf(d, f64::INFINITY).is_err(), "{d:?}");
    }
    let d = Poisson { rate: 1.0 };
    for x in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
        assert!(s::distributions::ppf(d, x).is_err());
    }
    let operations: [fn(s::distributions::Distribution, f64) -> sci::Result<f64>; 4] = [
        s::distributions::pdf,
        s::distributions::logpdf,
        s::distributions::cdf,
        s::distributions::sf,
    ];
    for f in operations {
        assert!(f(d, 9_007_199_254_740_992.0).is_err());
    }
}

#[test]
fn resampling_preserves_late_callback_errors_and_validates_all_draws_first() {
    use std::cell::Cell;
    let error = Error::NumericalFailure("statistic rejected draw");
    let calls = Cell::new(0);
    let result = s::resampling::permutation_test(
        &[1.0, 2.0],
        &[3.0, 4.0],
        |_, _| {
            calls.set(calls.get() + 1);
            if calls.get() == 3 {
                Err(error.clone())
            } else {
                Ok(0.0)
            }
        },
        s::inference::Alternative::TwoSided,
    );
    assert_eq!(result, Err(error));
    assert_eq!(calls.get(), 3);
    let never = |_: &[f64]| -> sci::Result<f64> { panic!("invalid draw reached callback") };
    let draws: &[&[f64]] = &[&[0.0, 1.0], &[f64::NAN, 1.0]];
    assert!(
        s::resampling::monte_carlo_test(
            &[0.0, 1.0],
            draws,
            never,
            s::inference::Alternative::Greater
        )
        .is_err()
    );
    assert!(s::inference::power(draws, never, 0.05).is_err());
    assert_eq!(
        s::inference::power(&[&[0.0]], |_| Ok(0.05), 0.05)
            .unwrap()
            .power,
        0.0
    );
    for p in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
        assert!(s::inference::power(&[&[0.0]], |_| Ok(p), 0.05).is_err());
    }
}

#[test]
fn custom_distribution_callbacks_enforce_density_probability_and_support_contracts() {
    let bad_density = s::distributions::make_distribution(
        0.0,
        1.0,
        |x| if x == 0.25 { Ok(-1.0) } else { Ok(1.0) },
        Ok,
        |x| Ok(1.0 - x),
        Ok,
    )
    .unwrap();
    assert!(s::distributions::pdf(bad_density, 0.25).is_err());
    assert!(s::distributions::logpdf(bad_density, 0.25).is_err());
    assert_eq!(s::distributions::pdf(bad_density, -1.0), Ok(0.0));
    assert_eq!(
        s::distributions::logpdf(bad_density, 2.0),
        Ok(f64::NEG_INFINITY)
    );
    let invalid_callbacks: [fn(f64) -> sci::Result<f64>; 4] = [
        |_| Ok(-0.1),
        |_| Ok(1.1),
        |_| Ok(f64::NAN),
        |_| Ok(f64::INFINITY),
    ];
    for callback in invalid_callbacks {
        assert!(
            s::distributions::make_distribution(
                0.0,
                1.0,
                |_| Ok(1.0),
                callback,
                |x| Ok(1.0 - x),
                Ok
            )
            .is_err()
        );
        assert!(
            s::distributions::make_distribution(0.0, 1.0, |_| Ok(1.0), Ok, callback, Ok).is_err()
        );
    }
    let d = s::distributions::make_distribution(
        0.0,
        1.0,
        |_| Ok(1.0),
        |x| if x == 0.25 { Ok(f64::NAN) } else { Ok(x) },
        |x| if x == 0.25 { Ok(1.1) } else { Ok(1.0 - x) },
        |p| if p == 0.25 { Ok(2.0) } else { Ok(p) },
    )
    .unwrap();
    assert!(s::distributions::cdf(d, 0.25).is_err());
    assert!(s::distributions::sf(d, 0.25).is_err());
    assert!(s::distributions::ppf(d, 0.25).is_err());
    let error = Error::NumericalFailure("custom probability failed");
    let d = s::distributions::make_distribution(
        0.0,
        1.0,
        |_| Ok(1.0),
        |x| {
            if x == 0.25 {
                Err(Error::NumericalFailure("custom probability failed"))
            } else {
                Ok(x)
            }
        },
        |x| Ok(1.0 - x),
        Ok,
    )
    .unwrap();
    assert_eq!(s::distributions::cdf(d, 0.25), Err(error));
}

#[test]
fn discrete_quantiles_are_left_continuous_inverse_cdfs() {
    use s::distributions::Distribution::*;
    for d in [
        Binomial {
            trials: 20.0,
            probability: 0.3,
        },
        Poisson { rate: 5.0 },
        Geometric { probability: 0.2 },
        NegativeBinomial {
            successes: 2.5,
            probability: 0.4,
        },
        Hypergeometric {
            population: 40.0,
            successes: 15.0,
            draws: 10.0,
        },
    ] {
        let mut previous = f64::NEG_INFINITY;
        for p in [0.01, 0.1, 0.5, 0.9, 0.99] {
            let q = s::distributions::ppf(d, p).unwrap();
            assert!(q >= previous && q.fract() == 0.0);
            assert!(s::distributions::cdf(d, q).unwrap() >= p - 2e-14);
            assert!(s::distributions::cdf(d, q - 1.0).unwrap() < p + 2e-14);
            assert_close(
                s::distributions::cdf(d, q).unwrap() + s::distributions::sf(d, q).unwrap(),
                1.0,
                2e-10,
                2e-10,
                "discrete tails",
            );
            previous = q;
        }
    }
}
