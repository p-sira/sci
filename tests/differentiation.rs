//! Public API coverage for the complete scipy.differentiate inventory and gradient.
mod common;
use common::finite_close;
use common::vector as numbers;

use std::cell::Cell;

use sci::{Error, differentiate as d, linalg};

fn scalar(name: &str, x: f64) -> f64 {
    match name {
        "identity" => x,
        "sin" => x.sin(),
        "exp" => x.exp(),
        "cubic" => x.powi(3) - 2.0 * x,
        "square" => x * x,
        "constant" => 7.0,
        _ => panic!("unknown scalar fixture {name}"),
    }
}

fn objective(name: &str, x: &[f64]) -> f64 {
    match name {
        "linear" => x[0],
        "local_square" => (x[0] - 1.0).powi(2),
        "local_bilinear" => (x[0] - 1.0) * (x[1] - 1.0),
        "quadratic" => x[0].powi(2) + 3.0 * x[0] * x[1] + 2.0 * x[1].powi(2),
        "exp_product" => (x[0] * x[1]).exp(),
        "trig" => x[0].sin() * x[1].cos(),
        "rosen" => x
            .windows(2)
            .map(|v| 100.0 * (v[1] - v[0].powi(2)).powi(2) + (1.0 - v[0]).powi(2))
            .sum(),
        "quartic" => x[0].powi(4),
        "constant" => 7.0,
        _ => panic!("unknown objective fixture {name}"),
    }
}

fn vector(name: &str, x: &[f64]) -> Vec<f64> {
    match name {
        "identity_vector" => vec![x[0], 2.0 * x[0]],
        "vector" => vec![x[0] * x[1], x[0] + x[1]],
        "rectangular" => vec![x[0].sin(), x[0] * x[1], x[1].exp()],
        _ => panic!("unknown vector fixture {name}"),
    }
}

#[test]
fn all_inventory_functions_and_gradient_match_scipy() {
    let inputs: Vec<_> =
        common::lines(include_str!("fixtures/differentiation_inputs.csv")).collect();
    let expected: Vec<_> =
        common::lines(include_str!("fixtures/differentiation_expected.csv")).collect();
    assert_eq!(inputs.len(), expected.len());
    for (input, expected) in inputs.iter().zip(expected) {
        let fields: Vec<_> = common::fields(input);
        let point = numbers(fields[2]);
        let step = fields[3].parse().unwrap();
        let actual = match fields[0] {
            "derivative" => vec![vec![
                d::derivative(|x| scalar(fields[1], x), point[0], step).unwrap(),
            ]],
            "gradient" => vec![d::gradient(|x| objective(fields[1], x), &point, step).unwrap()],
            "jacobian" => d::jacobian(|x| vector(fields[1], x), &point, step).unwrap(),
            "hessian" => d::hessian(|x| objective(fields[1], x), &point, step).unwrap(),
            _ => panic!("unknown API"),
        };
        let expected: Vec<_> = common::fields(expected);
        let rows: usize = expected[0].parse().unwrap();
        let columns: usize = expected[1].parse().unwrap();
        assert_eq!(actual.len(), rows, "{input}");
        assert!(actual.iter().all(|row| row.len() == columns), "{input}");
        let reference = numbers(expected[2]);
        assert_eq!(reference.len(), rows * columns);
        for (actual, expected) in actual.iter().flatten().zip(reference) {
            finite_close(
                *actual,
                expected,
                fields[4].parse().unwrap(),
                fields[5].parse().unwrap(),
            );
        }
    }
}

#[test]
fn invalid_inputs_are_rejected_before_callbacks() {
    let scalar = |_| panic!("invalid input reached callback");
    let objective = |_: &[f64]| panic!("invalid input reached callback");
    let vector = |_: &[f64]| -> Vec<f64> { panic!("invalid input reached callback") };
    for step in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(d::derivative(scalar, 0.0, step), Err(Error::InvalidStep));
        assert_eq!(
            d::gradient(objective, &[0.0], step),
            Err(Error::InvalidStep)
        );
        assert_eq!(d::jacobian(vector, &[0.0], step), Err(Error::InvalidStep));
        assert_eq!(d::hessian(objective, &[0.0], step), Err(Error::InvalidStep));
    }
    for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::MAX, 1e30] {
        assert!(d::derivative(scalar, x, 1e-5).is_err());
        assert!(d::gradient(objective, &[0.0, x], 1e-5).is_err());
        assert!(d::jacobian(vector, &[0.0, x], 1e-5).is_err());
        assert!(d::hessian(objective, &[0.0, x], 1e-5).is_err());
    }
    assert!(d::derivative(scalar, f64::MAX, f64::MAX).is_err());
    assert!(d::gradient(objective, &vec![0.0; 500_001], 1.0).is_err());
    assert!(d::jacobian(vector, &vec![0.0; 500_001], 1.0).is_err());
    assert!(d::hessian(objective, &vec![0.0; 708], 1.0).is_err());
}

#[test]
fn rejects_nonfinite_evaluations_and_arithmetic() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            d::derivative(|_| value, 0.0, 1.0),
            Err(Error::NonFiniteEvaluation)
        );
        assert_eq!(
            d::gradient(|_| value, &[0.0], 1.0),
            Err(Error::NonFiniteEvaluation)
        );
        assert_eq!(
            d::jacobian(|_| vec![value], &[0.0], 1.0),
            Err(Error::NonFiniteEvaluation)
        );
        assert_eq!(
            d::jacobian(|_| vec![value], &[], 1.0),
            Err(Error::NonFiniteEvaluation)
        );
        assert_eq!(
            d::hessian(|_| value, &[0.0], 1.0),
            Err(Error::NonFiniteEvaluation)
        );
        // Exercise the mixed stencil separately from center and axial values.
        assert_eq!(
            d::hessian(
                |x| if x.iter().all(|v| *v != 0.0) {
                    value
                } else {
                    0.0
                },
                &[0.0, 0.0],
                1.0
            ),
            Err(Error::NonFiniteEvaluation)
        );
    }
    assert_eq!(
        d::derivative(|x: f64| x.signum() * f64::MAX, 0.0, 0.5),
        Err(Error::NonFiniteEvaluation)
    );
    assert_eq!(
        d::hessian(|x| if x[0] == 0.0 { 0.0 } else { f64::MAX }, &[0.0], 0.5),
        Err(Error::NonFiniteEvaluation)
    );
}

#[test]
fn empty_shapes_and_changing_dimensions() {
    assert!(
        d::gradient(|_| panic!("empty gradient"), &[], 1.0)
            .unwrap()
            .is_empty()
    );
    assert!(
        d::hessian(|_| panic!("empty Hessian"), &[], 1.0)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        d::jacobian(|_| vec![1.0, 2.0], &[], 1.0).unwrap(),
        vec![vec![], vec![]]
    );
    assert!(
        d::jacobian(|_| vec![], &[1.0, 2.0], 1.0)
            .unwrap()
            .is_empty()
    );
    assert!(matches!(
        d::jacobian(|x| vec![0.0; if x[0] > 0.0 { 1 } else { 2 }], &[0.0], 1.0),
        Err(Error::InconsistentOutputDimension { .. })
    ));
    assert!(matches!(
        d::jacobian(
            |x| vec![0.0; if x[0] != 0.0 { 1 } else { 2 }],
            &[0.0, 0.0],
            1.0
        ),
        Err(Error::InconsistentOutputDimension { .. })
    ));
    assert!(d::jacobian(|_| vec![0.0; 1001], &vec![0.0; 1000], 1.0).is_err());
}

#[test]
fn evaluation_counts_and_symmetric_matrix_interoperate() {
    let calls = Cell::new(0);
    let f = |x: &[f64]| {
        calls.set(calls.get() + 1);
        x[0].powi(2) + 3.0 * x[0] * x[1] + 2.0 * x[1].powi(2)
    };
    let point = [0.0, 0.0];
    let hessian = d::hessian(f, &point, 0.125).unwrap();
    assert_eq!(calls.replace(0), 9);
    assert_eq!(hessian, vec![vec![2.0, 3.0], vec![3.0, 4.0]]);
    assert_eq!(
        linalg::basic::matvec(&hessian, &[2.0, -1.0]).unwrap(),
        vec![1.0, 2.0]
    );
    d::gradient(f, &point, 0.125).unwrap();
    assert_eq!(calls.replace(0), 4);
    let from_gradient = d::jacobian(|x| d::gradient(f, x, 0.125).unwrap(), &point, 0.125).unwrap();
    assert_eq!(from_gradient, hessian);
    assert_eq!(calls.replace(0), 16);
    d::jacobian(
        |x| {
            calls.set(calls.get() + 1);
            x.to_vec()
        },
        &point,
        0.125,
    )
    .unwrap();
    assert_eq!(calls.replace(0), 4);
    d::derivative(
        |x| {
            calls.set(calls.get() + 1);
            x * x
        },
        0.0,
        0.125,
    )
    .unwrap();
    assert_eq!(calls.get(), 2);
}

#[test]
fn second_order_convergence_and_extreme_scales() {
    let error = |h: f64| (d::hessian(|x| x[0].powi(4), &[1.0], h).unwrap()[0][0] - 12.0).abs();
    finite_close(error(0.125) / error(0.0625), 4.0, 1e-12, 0.0);
    let error = |h: f64| (d::derivative(|x| x.powi(3), 1.0, h).unwrap() - 3.0).abs();
    finite_close(error(0.125) / error(0.0625), 4.0, 1e-12, 0.0);
    assert_eq!(d::derivative(|x| x, 0.0, f64::MAX).unwrap(), 1.0);
    assert_eq!(
        d::derivative(|x: f64| x.signum() * f64::MAX, 0.0, 1.0).unwrap(),
        f64::MAX
    );
    assert_eq!(d::derivative(|x| x, 0.0, f64::from_bits(1)).unwrap(), 1.0);
    let step = 1e200;
    assert_eq!(
        d::hessian(|x| x[0] / step, &[0.0], step).unwrap(),
        vec![vec![0.0]]
    );
    assert_eq!(
        d::hessian(|_| f64::MAX, &[0.0], 1.0).unwrap(),
        vec![vec![0.0]]
    );
}

#[test]
fn rounded_coordinate_stencils_preserve_linear_and_quadratic_identities() {
    for (x, step) in [
        (1.0, 1.2e-16),
        (-1.0, 1.2e-16),
        (0.0, f64::from_bits(1)),
        (0.0, f64::MAX),
    ] {
        assert_eq!(d::derivative(|x| x, x, step).unwrap(), 1.0);
        assert_eq!(d::gradient(|x| x[0], &[x], step).unwrap(), vec![1.0]);
        assert_eq!(
            d::jacobian(|x| vec![x[0]], &[x], step).unwrap(),
            vec![vec![1.0]]
        );
        assert_eq!(d::hessian(|x| x[0], &[x], step).unwrap(), vec![vec![0.0]]);
    }
    assert_eq!(
        d::hessian(|x: &[f64]| (x[0] - 1.0).powi(2), &[1.0], 1.2e-16).unwrap(),
        vec![vec![2.0]]
    );
    assert_eq!(
        d::hessian(|x| (x[0] - 1.0) * (x[1] - 1.0), &[1.0, 1.0], 1.2e-16).unwrap(),
        vec![vec![0.0, 1.0], vec![1.0, 0.0]]
    );
    let tiny = f64::from_bits(1);
    assert_eq!(
        d::hessian(|x| (tiny * x[0]) * x[0] / 2.0, &[0.0], 2.0).unwrap(),
        vec![vec![tiny]]
    );
}
