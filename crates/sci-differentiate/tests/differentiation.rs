use sci_differentiate::{Error, derivative, gradient, hessian, jacobian};

#[test]
fn differentiates_scalar_function() {
    let value = derivative(|x: f64| x.powi(3), 2.0, 1e-5).unwrap();
    assert!((value - 12.0).abs() < 1e-8);
}

#[test]
fn calculates_gradient() {
    let value = gradient(|x: &[f64]| x[0].powi(2) + 3.0 * x[1], &[2.0, 4.0], 1e-5).unwrap();
    assert!((value[0] - 4.0).abs() < 1e-8);
    assert!((value[1] - 3.0).abs() < 1e-8);
}

#[test]
fn calculates_jacobian() {
    let value = jacobian(
        |x: &[f64]| vec![x[0] * x[1], x[0] + x[1]],
        &[2.0, 3.0],
        1e-5,
    )
    .unwrap();
    assert!((value[0][0] - 3.0).abs() < 1e-8);
    assert!((value[0][1] - 2.0).abs() < 1e-8);
    assert!((value[1][0] - 1.0).abs() < 1e-8);
    assert!((value[1][1] - 1.0).abs() < 1e-8);
}

#[test]
fn rejects_invalid_step() {
    assert_eq!(derivative(|x| x, 1.0, 0.0), Err(Error::InvalidStep));
    assert_eq!(gradient(|_| 0.0, &[], f64::NAN), Err(Error::InvalidStep));
}

#[test]
fn supports_zero_dimensional_jacobian() {
    assert_eq!(
        jacobian(|_| vec![1.0, 2.0], &[], 1e-5).unwrap(),
        vec![vec![], vec![]]
    );
}

#[test]
fn finite_difference_stencils_do_not_revisit_callback_points() {
    let calls = std::cell::Cell::new(0);
    let point = [1.0, 2.0, 3.0];
    let gradient_value = gradient(
        |x: &[f64]| {
            calls.set(calls.get() + 1);
            x.iter().map(|v| v * v).sum()
        },
        &point,
        0.25,
    )
    .unwrap();
    assert_eq!(calls.get(), 6);
    assert_eq!(gradient_value, vec![2.0, 4.0, 6.0]);

    calls.set(0);
    let jacobian_value = jacobian(
        |x: &[f64]| {
            calls.set(calls.get() + 1);
            vec![x[0] + x[1], x[2] * x[2]]
        },
        &point,
        0.25,
    )
    .unwrap();
    assert_eq!(calls.get(), 6);
    assert_eq!(
        jacobian_value,
        vec![vec![1.0, 1.0, 0.0], vec![0.0, 0.0, 6.0]]
    );

    calls.set(0);
    let hessian_value = hessian(
        |x: &[f64]| {
            calls.set(calls.get() + 1);
            x[0] * x[1] + x[2] * x[2]
        },
        &point,
        0.25,
    )
    .unwrap();
    assert_eq!(calls.get(), 19);
    assert_eq!(hessian_value[0][1], 1.0);
    assert_eq!(hessian_value[1][0], 1.0);
    assert_eq!(hessian_value[2][2], 2.0);
}
