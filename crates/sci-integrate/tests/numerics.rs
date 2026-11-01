use sci_integrate::quadrature::{fixed_quad, newton_cotes};

#[test]
fn gauss_legendre_moments_through_order_256() {
    for n in [1, 2, 3, 8, 32, 64, 128, 256] {
        for degree in [0, 1, 2, 3, 6, 10, 20, 40] {
            if degree as usize >= 2 * n {
                continue;
            }
            let actual = fixed_quad(|x: f64| x.powi(degree), -1.0, 1.0, n).unwrap();
            let expected = if degree % 2 == 0 {
                2.0 / (degree + 1) as f64
            } else {
                0.0
            };
            assert!(
                (actual - expected).abs() < 2e-12,
                "n={n}, degree={degree}: {actual} vs {expected}"
            );
        }
    }
}

#[test]
fn gauss_legendre_f32_and_reversed_intervals() {
    for n in [1, 2, 7, 64, 256] {
        let constant = fixed_quad(|_| 1.0_f32, -2.0, 3.0, n).unwrap();
        assert!((constant - 5.0).abs() < 2e-5, "n={n}: {constant}");
        let quadratic = fixed_quad(|x: f32| x * x, 1.0, 0.0, n).unwrap();
        if n >= 2 {
            assert!((quadratic + 1.0 / 3.0).abs() < 2e-6, "n={n}: {quadratic}");
        }
    }
}

#[test]
fn nonuniform_newton_cotes_matches_polynomial_moments() {
    for nodes in [
        vec![0.0, 0.3, 2.0],
        vec![0.0, 0.4, 1.3, 3.0],
        vec![0.0, 0.4, 0.9, 1.8, 3.1, 5.0],
    ] {
        let n = nodes.len() - 1;
        let (weights, error) = newton_cotes(&nodes).unwrap();
        assert!(error.is_finite());
        for power in 0..=n {
            let actual: f64 = nodes
                .iter()
                .zip(&weights)
                .map(|(&x, &w)| w * x.powi(power as i32))
                .sum();
            let exact = (n as f64).powi((power + 1) as i32) / (power + 1) as f64;
            assert!(
                (actual - exact).abs() < 1e-8 * exact.max(1.0),
                "n={n}, power={power}"
            );
        }
    }
}
