use super::whittaker_henderson;
use crate::Error;
use nalgebra::{DMatrix, DVector};

// Independent dense normal-equation reference for modest, well-conditioned
// problems. This deliberately constructs the full D^T D, unlike the solver.
fn dense_reference(y: &[f64], lambda: f64, order: usize, weights: &[f64]) -> Vec<f64> {
    let n = y.len();
    let mut a = DMatrix::from_diagonal(&DVector::from_column_slice(weights));
    let mut differences = vec![1.; order + 1];
    for k in 1..=order {
        differences[k] = differences[k - 1] * (order - k + 1) as f64 / k as f64;
    }
    for (k, coefficient) in differences.iter_mut().enumerate() {
        if (order - k) % 2 == 1 {
            *coefficient = -*coefficient;
        }
    }
    for row in 0..n - order {
        for i in 0..=order {
            for j in 0..=order {
                a[(row + i, row + j)] += lambda * differences[i] * differences[j];
            }
        }
    }
    let b = DVector::from_iterator(n, y.iter().zip(weights).map(|(&x, &weight)| x * weight));
    a.cholesky().unwrap().solve(&b).as_slice().to_vec()
}

fn close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            got.is_finite() && (got - want).abs() <= tolerance * want.abs().max(1.),
            "sample {i}: expected {want}, got {got}, tolerance {tolerance}"
        );
    }
}

#[test]
fn banded_solution_matches_independent_dense_reference() {
    for n in [4, 11, 24, 48] {
        let y: Vec<_> = (0..n)
            .map(|i| (0.27 * i as f64).sin() + 0.3 * (0.71 * i as f64).cos())
            .collect();
        for order in [1, 2, 3, 5, 8].into_iter().filter(|&p| p < n) {
            let masks = [
                vec![1.; n],
                (0..n).map(|i| 0.5 + (i % 7) as f64 / 3.).collect(),
                (0..n).map(|i| if i % 4 == 0 { 0. } else { 1. }).collect(),
            ];
            for weights in &masks {
                if weights.iter().filter(|&&weight| weight > 0.).count() < order {
                    continue;
                }
                for lambda in [0.02, 1., 10.] {
                    let got = whittaker_henderson(&y, lambda, order, weights).unwrap();
                    let want = dense_reference(&y, lambda, order, weights);
                    close(&got, &want, if order >= 5 { 2e-7 } else { 2e-10 });
                }
            }
        }
    }
}

#[test]
fn large_banded_systems_satisfy_weighted_normal_equations() {
    for (n, order, lambda) in [(128, 2, 10.), (256, 8, 0.5)] {
        let y: Vec<_> = (0..n)
            .map(|i| (0.17 * i as f64).sin() + (0.39 * i as f64).cos())
            .collect();
        let weights: Vec<_> = (0..n)
            .map(|i| if i % 5 == 0 { 0. } else { 1. + (i % 3) as f64 })
            .collect();
        let out = whittaker_henderson(&y, lambda, order, &weights).unwrap();
        let mut residual: Vec<_> = (0..n).map(|i| weights[i] * (out[i] - y[i])).collect();
        let mut differences = vec![1.; order + 1];
        for i in 1..=order {
            differences[i] = differences[i - 1] * (order - i + 1) as f64 / i as f64;
        }
        for (i, coefficient) in differences.iter_mut().enumerate() {
            if (order - i) % 2 == 1 {
                *coefficient = -*coefficient;
            }
        }
        for row in 0..n - order {
            let difference: f64 = (0..=order).map(|k| differences[k] * out[row + k]).sum();
            for k in 0..=order {
                residual[row + k] += lambda * differences[k] * difference;
            }
        }
        for (i, &value) in residual.iter().enumerate() {
            assert!(
                value.is_finite() && value.abs() < 1e-7,
                "order {order}, sample {i}: normal-equation residual {value}"
            );
        }
    }
}

#[test]
fn smoothing_preserves_unpenalized_polynomials_with_zero_weights() {
    let y: Vec<_> = (0..32).map(|i| -2. + 0.07 * i as f64).collect();
    let weights: Vec<_> = (0..32).map(|i| if i % 3 == 0 { 0. } else { 1. }).collect();
    let out = whittaker_henderson(&y, 20., 2, &weights).unwrap();
    close(&out, &y, 2e-9);

    let constant = vec![1.75; 48];
    let weights: Vec<_> = (0..48).map(|i| if i % 6 == 0 { 0. } else { 1. }).collect();
    let out = whittaker_henderson(&constant, 1., 8, &weights).unwrap();
    close(&out, &constant, 1e-7);
}

#[test]
fn scaled_weights_and_penalties_have_identical_solutions() {
    let y: Vec<_> = (0..20).map(|i| (0.13 * i as f64).sin()).collect();
    let weights: Vec<f64> = (0..20)
        .map(|i| if i % 4 == 0 { 0. } else { 0.5 + (i % 3) as f64 })
        .collect();
    let reference = whittaker_henderson(&y, 2.5, 2, &weights).unwrap();
    for scale in [1e-250, 1e250] {
        let scaled: Vec<_> = weights.iter().map(|w| w * scale).collect();
        let actual = whittaker_henderson(&y, 2.5 * scale, 2, &scaled).unwrap();
        close(&actual, &reference, 1e-10);
    }

    // Both the old W*y and lambda*D^T*D could overflow for these valid inputs.
    let huge = vec![1e307; 12];
    let scaled = whittaker_henderson(&huge, 1e300, 2, &[1e300; 12]).unwrap();
    for value in scaled {
        assert!((value / 1e307 - 1.).abs() < 1e-11);
    }

    let expected = whittaker_henderson(&y, 1., 2, &[1.; 20]).unwrap();
    let large = whittaker_henderson(&y, f64::MAX, 2, &[f64::MAX; 20]).unwrap();
    close(&large, &expected, 1e-10);
    let tiny = f64::from_bits(1);
    let small = whittaker_henderson(&y, tiny, 2, &[tiny; 20]).unwrap();
    close(&small, &expected, 1e-10);
}

#[test]
fn smoothing_preserves_validation_and_singular_semantics() {
    let y = [1., -1., 2., 4.];
    assert_eq!(whittaker_henderson(&y, 0., 2, &[0.; 4]).unwrap(), y);
    assert!(whittaker_henderson(&[], 1., 1, &[]).is_err());
    assert!(whittaker_henderson(&y, 1., 0, &[1.; 4]).is_err());
    assert!(whittaker_henderson(&y, 1., 4, &[1.; 4]).is_err());
    assert!(whittaker_henderson(&y, 1., 9, &[1.; 4]).is_err());
    assert!(whittaker_henderson(&y, 1., 2, &[1.; 3]).is_err());
    assert!(whittaker_henderson(&vec![1.; 257], 1., 2, &vec![1.; 257]).is_err());
    for lambda in [-1., f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(whittaker_henderson(&y, lambda, 2, &[1.; 4]).is_err());
    }
    for weight in [-1., f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(whittaker_henderson(&y, 1., 2, &[weight, 1., 1., 1.]).is_err());
    }
    for sample in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(whittaker_henderson(&[sample, 1., 2.], 1., 1, &[1.; 3]).is_err());
    }
    assert!(matches!(
        whittaker_henderson(&y, 1., 2, &[0., 1., 0., 0.]),
        Err(Error::InvalidInput(
            "positive weights must anchor every unpenalized polynomial degree"
        ))
    ));
}
