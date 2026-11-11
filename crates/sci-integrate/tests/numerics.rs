use sci_integrate::quadrature::{fixed_quad, newton_cotes};
use sci_integrate::sampled::{cumulative_simpson, cumulative_trapezoid, simpson, trapezoid};

#[test]
fn uniform_spacing_integrals_and_cumulative_prefixes() {
    let samples: Vec<f64> = (0..=128).map(|i| (i as f64).powi(2)).collect();
    let exact_simpson = 128.0_f64.powi(3) / 3.0;
    let exact_trapezoid = exact_simpson + 128.0 / 6.0;
    let whole_trapezoid = trapezoid(&samples, None, 1.0).unwrap();
    let whole_simpson = simpson(&samples, None, 1.0).unwrap();
    assert!((whole_trapezoid - exact_trapezoid).abs() < 1e-9);
    assert!((whole_simpson - exact_simpson).abs() < 1e-9);
    assert_eq!(
        cumulative_trapezoid(&samples, None, 1.0, None)
            .unwrap()
            .last(),
        Some(&whole_trapezoid)
    );
    assert_eq!(
        cumulative_simpson(&samples, None, 1.0, None)
            .unwrap()
            .last(),
        Some(&whole_simpson)
    );
    assert_eq!(simpson(&samples[..2], None, -1.0).unwrap(), -0.5);
    assert_eq!(simpson(&samples[..1], None, 1.0).unwrap(), 0.0);
}

#[test]
fn gauss_legendre_rules_are_thread_safe_and_preserve_evaluations() {
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                for order in [17, 47, 127] {
                    let calls = std::cell::Cell::new(0);
                    let integral = fixed_quad(
                        |x: f64| {
                            calls.set(calls.get() + 1);
                            x.exp()
                        },
                        0.0,
                        1.0,
                        order,
                    );
                    let integral = integral.unwrap();
                    assert!((integral - (std::f64::consts::E - 1.0)).abs() < 3e-13);
                    assert_eq!(calls.get(), order);
                }
            });
        }
    });
}

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
