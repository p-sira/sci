use sci_differentiate::{Error, derivative, gradient, jacobian};

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
