mod common;
use common::vector;

use sci::{differentiate, linalg, optimize as o};
fn objective(kind: &str, x: &[f64]) -> Vec<f64> {
    match kind {
        "quadratic" => vec![x[0] * x[0] + 3.0 * x[1] * x[1] + x[0] * x[1]],
        "vector" => vec![
            x[0] * x[0] + 3.0 * x[1],
            x[0] * x[1],
            x[0].sin() + x[1].cos(),
        ],
        "rosen" => vec![o::derivatives::rosen(x).unwrap()],
        _ => panic!("unknown objective"),
    }
}
#[test]
fn public_helpers_match_scipy() {
    let inputs: Vec<_> = common::lines(include_str!("fixtures/optimize_inputs.csv")).collect();
    let expected: Vec<_> = common::lines(include_str!("fixtures/optimize_expected.csv")).collect();
    assert_eq!(inputs.len(), expected.len());
    let mut covered = std::collections::BTreeSet::new();
    for (input, expected) in inputs.iter().zip(expected) {
        let c: Vec<_> = common::fields(input);
        let e: Vec<_> = common::fields(expected);
        assert_eq!(c[0], e[0]);
        covered.insert(c[1]);
        let x = vector(c[2]);
        let p = vector(c[3]);
        let epsilon = vector(c[4]);
        let bias: f64 = c[6].parse().unwrap();
        let got = match c[1] {
            "rosen" => vec![vec![o::derivatives::rosen(&x).unwrap()]],
            "rosen_der" => vec![o::derivatives::rosen_der(&x).unwrap()],
            "rosen_hess" => o::derivatives::rosen_hess(&x).unwrap(),
            "rosen_hess_prod" => vec![o::derivatives::rosen_hess_prod(&x, &p).unwrap()],
            "approx_fprime" => {
                o::derivatives::approx_fprime(|x| objective(c[5], x), &x, &epsilon).unwrap()
            }
            "check_grad" => vec![vec![
                o::derivatives::check_grad(
                    |x| objective(c[5], x)[0],
                    |x| {
                        let g = if c[5] == "quadratic" {
                            vec![2.0 * x[0] + x[1], 6.0 * x[1] + x[0]]
                        } else {
                            o::derivatives::rosen_der(x).unwrap()
                        };
                        g.into_iter().map(|v| v + bias).collect()
                    },
                    &x,
                    &epsilon,
                )
                .unwrap(),
            ]],
            _ => panic!("unknown API"),
        };
        assert_eq!(got.len(), e[1].parse::<usize>().unwrap(), "{} rows", c[0]);
        assert!(
            got.iter()
                .all(|r| r.len() == e[2].parse::<usize>().unwrap()),
            "{} columns",
            c[0]
        );
        let tolerance = if matches!(c[1], "approx_fprime" | "check_grad") {
            3e-9
        } else {
            3e-12
        };
        for (a, b) in got.into_iter().flatten().zip(vector(e[3])) {
            assert!(
                a.is_finite() && approx::abs_diff_eq!(a, b, epsilon = tolerance * (1.0 + b.abs())),
                "{}: {a} != {b}",
                c[0]
            );
        }
    }
    assert_eq!(covered.len(), 6);
}
#[test]
fn analytical_identities_and_interoperability() {
    for x in [vec![-1.2, 1.0], vec![0.1, 0.2, 0.3, 0.4], vec![1.0; 8]] {
        let p: Vec<_> = (0..x.len()).map(|i| i as f64 - 2.0).collect();
        let h = o::derivatives::rosen_hess(&x).unwrap();
        let hp = o::derivatives::rosen_hess_prod(&x, &p).unwrap();
        let dense = linalg::basic::matvec(&h, &p).unwrap();
        for (a, b) in hp.iter().zip(&dense) {
            assert!(approx::abs_diff_eq!(
                a,
                b,
                epsilon = 1e-12 * (1.0 + b.abs())
            ));
        }
        let g = o::derivatives::rosen_der(&x).unwrap();
        let numeric =
            differentiate::gradient(|x| o::derivatives::rosen(x).unwrap(), &x, 1e-5).unwrap();
        for (a, b) in g.iter().zip(&numeric) {
            assert!(approx::abs_diff_eq!(a, b, epsilon = 2e-6 * (1.0 + b.abs())));
        }
        let nh =
            differentiate::jacobian(|x| o::derivatives::rosen_der(x).unwrap(), &x, 1e-5).unwrap();
        for (i, row) in h.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                assert_eq!(*v, h[j][i]);
                assert!((v - nh[i][j]).abs() <= 2e-6 * (1.0 + v.abs()));
            }
        }
    }
    let result = o::minimize::minimize_with_gradient(
        |x| o::derivatives::rosen(x).unwrap(),
        |x| o::derivatives::rosen_der(x).unwrap(),
        &[0.9, 0.8],
        o::minimize::Options {
            max_iterations: 20000,
            learning_rate: 0.001,
            tolerance: 1e-6,
        },
    )
    .unwrap();
    assert!(result.cost < 1e-6);
    let mut calls = 0;
    let jac = o::derivatives::approx_fprime(
        |x| {
            calls += 1;
            vec![x.iter().sum()]
        },
        &[1.0, 2.0, 3.0],
        &[0.125],
    )
    .unwrap();
    assert_eq!(calls, 4);
    assert_eq!(jac, vec![vec![1.0; 3]]);
    let mut bad_calls = 0;
    assert!(
        o::derivatives::approx_fprime(
            |_| {
                bad_calls += 1;
                vec![0.0]
            },
            &[1e20],
            &[1e-20]
        )
        .is_err()
    );
    assert_eq!(bad_calls, 0);
}
#[test]
fn invalid_domains_and_overflow() {
    for x in [
        vec![f64::NAN, 0.0],
        vec![0.0, f64::INFINITY],
        vec![f64::MAX, 1.0],
        vec![1.0; 1_000_001],
    ] {
        assert!(o::derivatives::rosen(&x).is_err());
        assert!(o::derivatives::rosen_der(&x).is_err());
        assert!(o::derivatives::rosen_hess(&x).is_err());
        assert!(o::derivatives::rosen_hess_prod(&x, &x).is_err());
    }
    for x in [vec![], vec![1.0]] {
        assert!(o::derivatives::rosen_der(&x).is_err());
        assert!(o::derivatives::rosen_hess(&x).is_err());
        assert!(o::derivatives::rosen_hess_prod(&x, &x).is_err());
    }
    assert!(o::derivatives::rosen_hess(&[1.0; 257]).is_err());
    assert!(o::derivatives::rosen_hess_prod(&[1.0, 1.0], &[1.0]).is_err());
    assert!(o::derivatives::rosen_hess_prod(&[1.0, 1.0], &[f64::NAN, 0.0]).is_err());
    for steps in [
        vec![],
        vec![0.0],
        vec![f64::NAN],
        vec![f64::INFINITY],
        vec![1.0; 3],
    ] {
        assert!(o::derivatives::approx_fprime(|_| vec![1.0], &[1.0, 1.0], &steps).is_err());
    }
    for x in [vec![], vec![f64::NAN], vec![1.0; 513]] {
        assert!(o::derivatives::approx_fprime(|_| vec![1.0], &x, &[1.0]).is_err());
    }
    assert!(o::derivatives::approx_fprime(|_| vec![], &[1.0], &[0.1]).is_err());
    assert!(o::derivatives::approx_fprime(|_| vec![f64::NAN], &[1.0], &[0.1]).is_err());
    assert!(
        o::derivatives::approx_fprime(
            |x| if x[0] == 1.0 {
                vec![1.0]
            } else {
                vec![1.0, 2.0]
            },
            &[1.0],
            &[0.1]
        )
        .is_err()
    );
    assert!(
        o::derivatives::approx_fprime(
            |x| vec![if x[0] == 1.0 { 0.0 } else { f64::INFINITY }],
            &[1.0],
            &[0.1]
        )
        .is_err()
    );
    assert!(o::derivatives::approx_fprime(|_| vec![0.0], &[f64::MAX], &[f64::MAX]).is_err());
    assert!(o::derivatives::check_grad(|x| x[0], |_| vec![], &[1.0], &[0.1]).is_err());
    assert!(o::derivatives::check_grad(|x| x[0], |_| vec![f64::NAN], &[1.0], &[0.1]).is_err());
    assert!(o::derivatives::check_grad(|_| f64::INFINITY, |_| vec![1.0], &[1.0], &[0.1]).is_err());
}
