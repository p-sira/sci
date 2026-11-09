mod common;

use common::{fields, lines, vector};
use sci::optimize::least_squares::{
    DogboxOptions, DogboxStatus, dogbox, dogbox_with_jacobian, try_dogbox, try_dogbox_with_jacobian,
};

fn residuals(model: &str, x: &[f64], a: &[f64], b: &[f64]) -> Vec<f64> {
    match model {
        "linear" => b
            .iter()
            .enumerate()
            .map(|(i, &target)| {
                a[i * x.len()..(i + 1) * x.len()]
                    .iter()
                    .zip(x)
                    .map(|(a, x)| a * x)
                    .sum::<f64>()
                    - target
            })
            .collect(),
        "rosenbrock" => vec![10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]],
        "exponential" => a
            .iter()
            .zip(b)
            .map(|(&t, &target)| x[0] * (x[1] * t).exp() - target)
            .collect(),
        "curved" => vec![x[0] * x[0] + x[1] - b[0]],
        "square" => vec![(a[0] * x[0]).powi(2) - b[0]],
        other => panic!("unexpected dogbox model: {other}"),
    }
}

fn analytical_jacobian(model: &str, x: &[f64], a: &[f64], b: &[f64]) -> Vec<Vec<f64>> {
    match model {
        "linear" => a.chunks_exact(x.len()).map(|row| row.to_vec()).collect(),
        "rosenbrock" => vec![vec![-20.0 * x[0], 10.0], vec![-1.0, 0.0]],
        "exponential" => a
            .iter()
            .map(|t| vec![(x[1] * t).exp(), x[0] * t * (x[1] * t).exp()])
            .collect(),
        "curved" => {
            assert_eq!(b.len(), 1);
            vec![vec![2.0 * x[0], 1.0]]
        }
        "square" => vec![vec![2.0 * a[0] * a[0] * x[0]]],
        other => panic!("unexpected dogbox model: {other}"),
    }
}

#[test]
fn dogbox_matches_scipy_reference() {
    let inputs: Vec<_> = lines(include_str!("fixtures/dogbox_inputs.csv")).collect();
    let expected: Vec<_> = lines(include_str!("fixtures/dogbox_expected.csv")).collect();
    assert_eq!(inputs.len(), 16);
    assert_eq!(inputs.len(), expected.len());

    for (input, expected) in inputs.iter().zip(&expected) {
        let c: Vec<_> = fields(input);
        let e: Vec<_> = fields(expected);
        assert_eq!(c[0], e[0]);
        let name = c[0];
        let model = c[1];
        let n: usize = c[2].parse().unwrap();
        let m: usize = c[3].parse().unwrap();
        let x0 = vector(c[4]);
        let lower = vector(c[5]);
        let upper = vector(c[6]);
        let a = vector(c[7]);
        let b = vector(c[8]);
        assert_eq!(x0.len(), n);
        let expected_x = vector(e[1]);
        let expected_f = vector(e[2]);
        assert_eq!(expected_f.len(), m);
        let expected_cost: f64 = e[3].parse().unwrap();
        let expected_optimality: f64 = e[4].parse().unwrap();
        let expected_mask = vector(e[5]);
        let options = DogboxOptions {
            gradient_tolerance: 1e-10,
            cost_tolerance: 1e-10,
            step_tolerance: 1e-10,
            ..DogboxOptions::default()
        };

        for exact in [false, true] {
            let result = if exact {
                dogbox_with_jacobian(
                    |x| residuals(model, x, &a, &b),
                    |x| analytical_jacobian(model, x, &a, &b),
                    &x0,
                    &lower,
                    &upper,
                    options,
                )
            } else {
                dogbox(
                    |x| residuals(model, x, &a, &b),
                    &x0,
                    &lower,
                    &upper,
                    options,
                )
            }
            .unwrap_or_else(|err| panic!("{name} (analytical={exact}): {err}"));
            assert!(
                !matches!(
                    result.status,
                    DogboxStatus::IterationLimit | DogboxStatus::NoProgress
                ),
                "{name} (analytical={exact}): {:?} at {:?}, cost={}",
                result.status,
                result.point,
                result.cost,
            );
            assert_eq!(result.point.len(), n);
            assert_eq!(result.residuals.len(), m);
            assert!(
                result
                    .point
                    .iter()
                    .enumerate()
                    .all(|(i, v)| *v >= lower[i] && *v <= upper[i])
            );
            let allowed = 3e-6 * (1.0 + expected_cost.abs());
            assert!(
                (result.cost - expected_cost).abs() <= allowed,
                "{name} (analytical={exact}): cost {} != {expected_cost}, point={:?}, optimum={:?}",
                result.cost,
                result.point,
                expected_x,
            );
            if !name.contains("rank_deficient")
                && !name.contains("underdetermined")
                && !name.contains("ill_conditioned")
            {
                for (got, expected) in result.point.iter().zip(&expected_x) {
                    assert!(
                        (got - expected).abs() <= 3e-4 * (1.0 + expected.abs()),
                        "{name} (analytical={exact}): {got} != {expected}"
                    );
                }
            }
            for (got, expected) in result.residuals.iter().zip(&expected_f) {
                assert!(
                    (got - expected).abs() <= 4e-4 * (1.0 + expected.abs()),
                    "{name} (analytical={exact}): residual {got} != {expected}"
                );
            }
            if name == "linear_both_bounds"
                || name == "linear_one_bound"
                || name == "rosenbrock_bounded"
                || name.starts_with("linear_rounded_")
            {
                assert_eq!(
                    result
                        .active_mask
                        .iter()
                        .map(|v| *v as f64)
                        .collect::<Vec<_>>(),
                    expected_mask,
                    "{name} (analytical={exact}): wrong active bounds"
                );
            }
            assert!(
                // For the scaled square, the default difference interval is
                // larger than the root. Match SciPy's attainable optimality;
                // its step termination does not imply a small raw gradient.
                result.optimality <= 1e-5_f64.max(1.1 * expected_optimality),
                "{name} (analytical={exact}): optimality {}",
                result.optimality
            );
            if model == "square" {
                assert!(result.iterations > 0, "{name}: false initial stationarity");
            }
            if name == "square_scaled_negative_start" {
                assert!(result.cost < 1e-14, "{name}: {result:?}");
                let root = -b[0].sqrt() / a[0];
                assert!((result.point[0] - root).abs() <= 1e-7 * root.abs());
            }
            if name.starts_with("linear_rounded_") {
                let bound = if name.ends_with("upper") {
                    upper[0]
                } else {
                    lower[0]
                };
                assert_eq!(result.point[0], bound);
                assert_eq!(result.optimality, 0.0);
                assert_eq!(result.status, DogboxStatus::Gradient);
            }
            assert!(result.function_evaluations > 0 && result.jacobian_evaluations > 0);
        }
    }
}

#[test]
fn dogbox_bound_stationarity_and_validation() {
    let o = DogboxOptions::default();
    let lower = [0.0];
    let upper = [2.0];
    let outward = dogbox(|x| vec![x[0] + 1.0], &[0.0], &lower, &upper, o).unwrap();
    assert_eq!(outward.point, vec![0.0]);
    assert_eq!(outward.active_mask, vec![-1]);
    assert_eq!(outward.status, DogboxStatus::Gradient);
    assert_eq!(outward.optimality, 0.0);
    let inward = dogbox(|x| vec![x[0] - 1.0], &[0.0], &lower, &upper, o).unwrap();
    assert!((inward.point[0] - 1.0).abs() < 1e-7);
    assert_eq!(inward.active_mask, vec![0]);

    // The initial dogleg reaches x=0 where ln(x) is infinite. SciPy's
    // dogbox shrinks the trust region and retries without losing the iterate.
    let recovered = dogbox(|x| vec![x[0].ln() + 2.0], &[1.0], &[0.0], &[10.0], o).unwrap();
    assert!((recovered.point[0] - (-2.0_f64).exp()).abs() < 1e-6);
    assert!(recovered.cost < 1e-12);

    // Small parameter scales must not trigger an absolute radius cutoff:
    // the first trial is uphill but the next smaller rectangle improves.
    let tiny = dogbox_with_jacobian(
        |x| vec![1.0 - 1e20 * x[0] + 2e40 * x[0] * x[0]],
        |x| vec![vec![-1e20 + 4e40 * x[0]]],
        &[0.0],
        &[0.0],
        &[1e-20],
        o,
    )
    .unwrap();
    assert!(tiny.cost < 0.4, "tiny-scale solve: {tiny:?}");

    assert!(dogbox(|_| vec![0.0], &[], &[], &[], o).is_err());
    assert!(dogbox(|_| vec![0.0], &[0.0], &[1.0], &[2.0], o).is_err());
    assert!(dogbox(|_| vec![0.0], &[0.0], &[1.0], &[0.0], o).is_err());
    assert!(dogbox(|_| vec![f64::NAN], &[0.0], &lower, &upper, o).is_err());
    assert!(dogbox(|_| vec![], &[0.0], &lower, &upper, o).is_err());
    assert!(
        dogbox_with_jacobian(|_| vec![1.0], |_| vec![vec![]], &[0.0], &lower, &upper, o).is_err()
    );
    assert!(
        try_dogbox(
            |_| Err(sci::Error::NonFiniteEvaluation),
            &[0.0],
            &lower,
            &upper,
            o
        )
        .is_err()
    );
    assert!(
        try_dogbox_with_jacobian(
            |_| Ok(vec![1.0]),
            |_| Err(sci::Error::NonFiniteEvaluation),
            &[0.0],
            &lower,
            &upper,
            o,
        )
        .is_err()
    );
}

#[test]
fn dogbox_fallible_callbacks_and_shared_matrix_interoperate() {
    use std::cell::Cell;

    let calls = Cell::new(0);
    let residual = |x: &[f64]| {
        calls.set(calls.get() + 1);
        Ok(vec![x[0] - 1.0, 2.0 * (x[0] - 1.0)])
    };
    let numerical =
        try_dogbox(residual, &[0.0], &[-2.0], &[2.0], DogboxOptions::default()).unwrap();
    assert_eq!(numerical.function_evaluations, calls.get());
    calls.set(0);
    let analytical = try_dogbox_with_jacobian(
        residual,
        |_| -> sci::Result<sci::Matrix> { Ok(vec![vec![1.0], vec![2.0]]) },
        &[0.0],
        &[-2.0],
        &[2.0],
        DogboxOptions::default(),
    )
    .unwrap();
    assert_eq!(analytical.function_evaluations, calls.get());
    for result in [numerical, analytical] {
        assert_eq!(result.status, DogboxStatus::Gradient);
        assert!((result.point[0] - 1.0).abs() < 1e-12);
        assert!(result.cost < 1e-24);
        assert_eq!(result.gradient.len(), 1);
    }
    let failure = sci::Error::NumericalFailure("residual failed at trial");
    assert_eq!(
        try_dogbox(
            |x| if x[0] == 0.0 {
                Ok(vec![-1.0])
            } else {
                Err(failure.clone())
            },
            &[0.0],
            &[-2.0],
            &[2.0],
            DogboxOptions::default(),
        ),
        Err(failure),
    );
}

#[test]
fn dogbox_finite_differences_respect_bounds_and_initialization_budget() {
    // Cover sign reversal at either bound and a clipped difference interval
    // whose floating-point reconstruction can otherwise exceed the upper bound.
    for (start, lo, hi) in [
        (0.25, 0.2, 0.25),
        (-0.25, -0.25, -0.2),
        (-1e-9, -1e-9, 5e-9),
    ] {
        let result = dogbox(
            |x| {
                assert!(x[0].is_finite() && lo <= x[0] && x[0] <= hi);
                vec![x[0]]
            },
            &[start],
            &[lo],
            &[hi],
            DogboxOptions {
                max_iterations: 0,
                ..DogboxOptions::default()
            },
        )
        .unwrap();
        assert_eq!(result.point, vec![start]);
        assert_eq!(result.iterations, 0);
        assert_eq!(result.function_evaluations, 2);
        assert_eq!(result.jacobian_evaluations, 1);
        assert!((result.gradient[0] - start).abs() <= 1e-14 * start.abs());
    }
    for max_iterations in [16_777_216, u64::MAX] {
        assert!(
            dogbox(
                |_| panic!("inadmissible budget must fail before callbacks"),
                &[0.0],
                &[-1.0],
                &[1.0],
                DogboxOptions {
                    max_iterations,
                    ..DogboxOptions::default()
                },
            )
            .is_err()
        );
    }
}

#[test]
fn dogbox_rejects_invalid_controls_and_callback_shapes() {
    let o = DogboxOptions::default();
    for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        for options in [
            DogboxOptions {
                gradient_tolerance: invalid,
                ..o
            },
            DogboxOptions {
                cost_tolerance: invalid,
                ..o
            },
            DogboxOptions {
                step_tolerance: invalid,
                ..o
            },
        ] {
            assert!(dogbox(|_| vec![1.0], &[0.0], &[-1.0], &[1.0], options).is_err());
        }
    }
    for (start, lo, hi) in [
        (f64::NAN, -1.0, 1.0),
        (f64::INFINITY, -1.0, f64::INFINITY),
        (0.0, f64::NAN, 1.0),
        (0.0, -1.0, f64::NAN),
        (0.0, 0.0, 0.0),
    ] {
        assert!(dogbox(|_| vec![1.0], &[start], &[lo], &[hi], o).is_err());
    }
    assert_eq!(
        dogbox(
            |x| if x[0] == 0.0 {
                vec![-1.0]
            } else {
                vec![1.0, 2.0]
            },
            &[0.0],
            &[-1.0],
            &[1.0],
            o
        ),
        Err(sci::Error::DimensionMismatch)
    );
    assert_eq!(
        dogbox_with_jacobian(
            |_| vec![1.0],
            |_| vec![vec![f64::NAN]],
            &[0.0],
            &[-1.0],
            &[1.0],
            o
        ),
        Err(sci::Error::NonFiniteEvaluation)
    );
}
