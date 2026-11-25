use sci::{
    Error,
    complex::{self, Complex},
    differentiate, fft, linalg, optimize, special, stats,
};

#[test]
fn shapes_and_controls_are_validated() {
    assert_eq!(
        linalg::basic::dot(&[1.0], &[]),
        Err(Error::DimensionMismatch)
    );
    assert_eq!(
        linalg::basic::matvec(&[vec![1.0], vec![]], &[1.0]),
        Err(Error::DimensionMismatch)
    );
    assert_eq!(
        linalg::basic::matmul(&[vec![1.0, 2.0]], &[vec![1.0]]),
        Err(Error::DimensionMismatch)
    );
    assert!(stats::descriptive::mean::<f64>(&[]).is_err());
    assert!(stats::descriptive::variance(&[1.0], 1).is_err());
    assert!(stats::distributions::normal_pdf(0.0, 0.0, 0.0).is_err());
    assert!(stats::distributions::normal_cdf(f64::NAN, 0.0, 1.0).is_err());
    assert!(stats::distributions::normal_ppf(-0.1, 0.0, 1.0).is_err());
    assert!(optimize::minimize::minimize(|_| 0.0, &[], Default::default()).is_err());
    assert!(
        optimize::minimize::minimize(
            |_| 0.0,
            &[1.0],
            optimize::minimize::Options {
                tolerance: f64::NAN,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert_eq!(
        optimize::minimize::minimize_with_gradient(|_| 1.0, |_| vec![], &[1.0], Default::default(),),
        Err(Error::DimensionMismatch)
    );
}

#[test]
fn invalid_evaluations_and_unrepresentable_steps_are_rejected() {
    assert_eq!(
        differentiate::derivative(|_| f64::NAN, 0.0, 1e-5),
        Err(Error::NonFiniteEvaluation)
    );
    assert!(differentiate::derivative(|x| x, 1e30, 1e-5).is_err());
    assert!(
        differentiate::jacobian(
            |x| if x[0] > 0.0 {
                vec![1.0]
            } else {
                vec![1.0, 2.0]
            },
            &[0.0],
            1e-5,
        )
        .is_err()
    );
    assert_eq!(
        optimize::minimize::minimize(|_| f64::NAN, &[0.0], Default::default()),
        Err(Error::NonFiniteEvaluation)
    );
    let error = Error::InvalidInput("callback failure");
    assert_eq!(
        optimize::minimize::try_minimize(|_| Err(error.clone()), &[0.0], Default::default()),
        Err(error)
    );
}

#[test]
fn limits_are_distinct_from_convergence() {
    let result = optimize::minimize::minimize(
        |x| x[0].powi(2),
        &[2.0],
        optimize::minimize::Options {
            max_iterations: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.status, optimize::minimize::Status::IterationLimit);
    assert_eq!(result.iterations, 0);
}

#[test]
fn finite_difference_gradients_feed_optimization() {
    let objective = |x: &[f64]| linalg::basic::dot(x, x).unwrap();
    let result = optimize::minimize::try_minimize_with_gradient(
        |x| linalg::basic::dot(x, x),
        |x| differentiate::gradient(objective, x, 1e-5),
        &[3.0, -4.0],
        Default::default(),
    )
    .unwrap();
    assert_eq!(result.status, optimize::minimize::Status::Converged);
    assert!(result.cost < 1e-12);
}

#[test]
fn fft_round_trips_odd_and_even_lengths() {
    for length in [0, 1, 3, 4, 7, 8] {
        let input = (0..length)
            .map(|i| Complex {
                re: i as f64,
                im: (i as f64).sin(),
            })
            .collect::<Vec<_>>();
        let recovered = fft::fourier::ifft(&fft::fourier::fft(&input));
        assert_eq!(recovered.len(), input.len());
        for (actual, expected) in recovered.iter().zip(&input) {
            assert!((actual.re - expected.re).abs() < 1e-12);
            assert!((actual.im - expected.im).abs() < 1e-12);
        }
        let real = (0..length).map(|i| i as f64).collect::<Vec<_>>();
        assert_eq!(
            fft::real::rfft(&real).len(),
            if length == 0 { 0 } else { length / 2 + 1 }
        );
    }
    assert_eq!(
        complex::multiply(Complex { re: 0.0, im: 1.0 }, Complex { re: 0.0, im: 1.0 }).re,
        -1.0
    );
}

#[test]
fn special_function_domains_and_boundaries_are_predictable() {
    assert!(special::gamma::digamma(-1.0_f64).is_nan());
    assert!(special::gamma::digamma(f64::NEG_INFINITY).is_nan());
    assert!(special::beta::betainc(-1.0_f64, 2.0, 0.5).is_nan());
    assert_eq!(special::elementary::erfinv(1.0), f64::INFINITY);
    assert_eq!(special::elementary::erfinv(-1.0), f64::NEG_INFINITY);
    assert!(special::elliptic::ellipk(1.0).is_err());
    assert!(special::elliptic::elliprf(0.0, 0.0, 1.0).is_err());
    assert!(special::elliptic::ellipk(f64::NAN).is_err());
    assert_eq!(
        stats::distributions::normal_ppf(0.0, 0.0, 1.0).unwrap(),
        f64::NEG_INFINITY
    );
    assert_eq!(
        stats::distributions::normal_ppf(1.0, 0.0, 1.0).unwrap(),
        f64::INFINITY
    );
}
