mod common;

use common::{finite_close, matrix, number, rows, vector};
use sci::{integrate, linalg, optimize, special, stats};

#[test]
fn independent_accuracy_references() {
    let inputs: Vec<_> = rows(include_str!("fixtures/numerical_accuracy_inputs.csv")).collect();
    let expected: Vec<_> = rows(include_str!("fixtures/numerical_accuracy_expected.csv")).collect();
    assert_eq!(inputs.len(), expected.len());
    for (input, expected) in inputs.into_iter().zip(expected) {
        assert_eq!(input[0], expected[0]);
        let x = vector(input[2]);
        let p = vector(input[3]);
        let reference = vector(expected[1]);
        let actual = match input[1] {
            "rosen" => vec![optimize::derivatives::rosen(&x).unwrap()],
            "rosen_der" => optimize::derivatives::rosen_der(&x).unwrap(),
            "student_ppf" => {
                let distribution = stats::distributions::Distribution::StudentT { df: p[0] };
                let result = special::probability::stdtrit(p[0], p[1]);
                let quantile = stats::distributions::ppf(distribution, p[1]).unwrap();
                if p[1] > 0.25 && p[1] < 0.75 {
                    assert_eq!(quantile, result);
                } else {
                    finite_close(quantile, result, 0., 2e-12);
                }
                // Compare the smaller tail as well as the quantile itself.
                let recovered = if p[1] <= 0.5 {
                    stats::distributions::cdf(distribution, result).unwrap()
                } else {
                    stats::distributions::sf(distribution, result).unwrap()
                };
                finite_close(recovered, p[1].min(1. - p[1]), 0., 2e-12);
                vec![result]
            }
            "student_cdf" => {
                let distribution = stats::distributions::Distribution::StudentT { df: p[0] };
                let cdf = stats::distributions::cdf(distribution, p[1]).unwrap();
                let sf = stats::distributions::sf(distribution, p[1]).unwrap();
                assert_eq!(cdf, special::probability::stdtr(p[0], p[1]));
                assert_eq!(sf, special::probability::stdtr(p[0], -p[1]));
                vec![cdf, sf]
            }
            "student_special_cdf" => {
                let result = special::probability::stdtr(p[0], p[1]);
                assert_eq!(
                    stats::distributions::cdf(
                        stats::distributions::Distribution::StudentT { df: p[0] },
                        p[1]
                    )
                    .unwrap(),
                    result
                );
                assert_eq!(
                    stats::distributions::sf(
                        stats::distributions::Distribution::StudentT { df: p[0] },
                        -p[1]
                    )
                    .unwrap(),
                    result
                );
                vec![result]
            }
            "student_special_ppf" => {
                let result = special::probability::stdtrit(p[0], p[1]);
                assert_eq!(
                    stats::distributions::ppf(
                        stats::distributions::Distribution::StudentT { df: p[0] },
                        p[1]
                    )
                    .unwrap(),
                    result
                );
                vec![result]
            }
            "newton_cotes" => {
                let (mut weights, error) = integrate::quadrature::newton_cotes(&x).unwrap();
                assert_eq!(weights.len(), x.len());
                for (a, b) in weights.iter().zip(weights.iter().rev()) {
                    assert_eq!(a, b);
                }
                // Every closed rule integrates constants and linear terms.
                let n = (x.len() - 1) as f64;
                finite_close(weights.iter().sum(), n, 1e-13, 0.);
                finite_close(
                    weights.iter().zip(&x).map(|(w, x)| w * x).sum(),
                    n * n / 2.,
                    1e-12,
                    0.,
                );
                weights.push(error);
                weights
            }
            "shapiro" => {
                let result = stats::normality::shapiro(&x).unwrap();
                vec![result.statistic, result.pvalue]
            }
            "solve" => linalg::solvers::solve(&matrix(input[4]), &x).unwrap(),
            api => panic!("unregistered accuracy API {api}"),
        };
        assert_eq!(actual.len(), reference.len(), "{}", input[0]);
        let (atol, rtol) = match input[1] {
            // AS R94 normalization can round the gaps by a few ulps.
            "shapiro" => (2e-15, 2e-14),
            "solve" => (0., 5e-13),
            // Incomplete-beta normalization at df=1000 loses about 6e-13
            // relatively against the 100-digit density integral.
            "student_cdf" | "student_ppf" | "student_special_cdf" | "student_special_ppf" => {
                (0., 2e-12)
            }
            _ => (0., 2e-14),
        };
        for (index, (&actual, &expected)) in actual.iter().zip(&reference).enumerate() {
            assert!(actual.is_finite(), "{} component {index}", input[0]);
            assert!(
                (actual - expected).abs() <= atol + rtol * expected.abs(),
                "{} component {index}: {actual} versus {expected}",
                input[0]
            );
        }
    }
}

#[test]
fn domains_endpoints_and_exact_medians() {
    for df in [0.25, 1., 2., 4.5, 1000.] {
        let distribution = stats::distributions::Distribution::StudentT { df };
        assert_eq!(
            special::probability::stdtrit(df, 0.5).to_bits(),
            0_f64.to_bits()
        );
        assert_eq!(special::probability::stdtrit(df, 0.), f64::NEG_INFINITY);
        assert_eq!(special::probability::stdtrit(df, 1.), f64::INFINITY);
        assert_eq!(
            stats::distributions::ppf(distribution, 0.).unwrap(),
            f64::NEG_INFINITY
        );
        assert_eq!(
            stats::distributions::ppf(distribution, 1.).unwrap(),
            f64::INFINITY
        );
        assert_eq!(special::probability::stdtr(df, f64::NEG_INFINITY), 0.);
        assert_eq!(special::probability::stdtr(df, f64::INFINITY), 1.);
    }
    for df in [-1., 0., f64::NAN] {
        assert!(special::probability::stdtrit(df, 0.4).is_nan());
        assert!(special::probability::stdtr(df, 1e-10).is_nan());
        assert!(
            stats::distributions::ppf(stats::distributions::Distribution::StudentT { df }, 0.4)
                .is_err()
        );
    }
    for p in [-0.1, 1.1, f64::NAN] {
        assert!(special::probability::stdtrit(4.5, p).is_nan());
    }
    assert!(special::probability::stdtr(f64::NAN, 0.).is_nan());
    assert!(special::probability::stdtr(3., f64::NAN).is_nan());
    assert!(stats::normality::shapiro(&[0., 1.]).is_err());
    assert!(stats::normality::shapiro(&[0., 1., f64::INFINITY]).is_err());
    assert!(
        stats::normality::shapiro(&[1., 1., 1.])
            .unwrap()
            .statistic
            .is_nan()
    );
    assert!(linalg::solvers::solve(&[vec![1., 2.], vec![2., 4.]], &[1., 2.]).is_err());
    assert!(integrate::quadrature::newton_cotes(&[0., 0., 2.]).is_err());
    assert!(integrate::quadrature::newton_cotes(&[0., f64::NAN, 2.]).is_err());
    assert!(optimize::derivatives::rosen(&[number("NaN"), 1.]).is_err());
}
