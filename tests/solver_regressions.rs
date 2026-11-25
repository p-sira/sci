//! Independent algebraic regression checks for the lightweight numerical solvers.
//! These tests intentionally do not use the implementation's previous backends.

use sci::{integrate::quadrature, special::orthogonal};

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= tolerance,
        "actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.2e}",
    );
}

#[test]
fn legendre_rules_preserve_gaussian_moments_through_order_256() {
    for order in [1, 2, 3, 8, 16, 32, 64, 128, 256] {
        let (nodes, weights) = orthogonal::roots_legendre(order).unwrap();
        assert_eq!(nodes.len(), order);
        assert_eq!(weights.len(), order);
        for i in 0..order {
            assert!((-1.0..=1.0).contains(&nodes[i]), "order={order}, i={i}");
            assert!(
                weights[i] > 0.0 && weights[i].is_finite(),
                "order={order}, i={i}"
            );
            if i > 0 {
                assert!(nodes[i] > nodes[i - 1], "order={order}, i={i}");
            }
            close(nodes[i] + nodes[order - i - 1], 0.0, 3e-12);
            close(weights[i], weights[order - i - 1], 3e-12);
        }
        for degree in 0..=usize::min(2 * order - 1, 16) {
            let moment: f64 = nodes
                .iter()
                .zip(&weights)
                .map(|(&node, &weight)| weight * node.powi(degree as i32))
                .sum();
            let expected = if degree % 2 == 0 {
                2.0 / (degree + 1) as f64
            } else {
                0.0
            };
            close(moment, expected, 2e-11);
        }
    }
}

#[test]
fn gaussian_family_rules_have_correct_first_moments() {
    for order in [1, 2, 8, 32, 64] {
        let (x, w) = orthogonal::roots_laguerre(order).unwrap();
        close(w.iter().sum(), 1.0, 2e-11);
        if order >= 2 {
            close(x.iter().zip(&w).map(|(x, w)| x * w).sum(), 1.0, 2e-11);
        }

        let (x, w) = orthogonal::roots_hermite(order).unwrap();
        let mass = std::f64::consts::PI.sqrt();
        close(w.iter().sum(), mass, 2e-11);
        close(x.iter().zip(&w).map(|(x, w)| x * w).sum(), 0.0, 2e-11);
    }
}

#[test]
fn fixed_quad_high_orders_match_exact_polynomial_integrals() {
    for order in [1, 2, 8, 32, 64, 128, 256] {
        let value = quadrature::fixed_quad(|x: f64| x * x, 0.0, 1.0, order).unwrap();
        // A degree-two polynomial requires at least two Gaussian nodes.
        if order >= 2 {
            close(value, 1.0 / 3.0, 2e-11);
        } else {
            close(value, 0.25, 2e-11);
        }
    }
    // A tiny relative weight error must not turn a representable integral
    // into an overflow at the largest binary64 value.
    assert_eq!(
        quadrature::fixed_quad(|_| f64::MAX, 0.0, 1.0, 2).unwrap(),
        f64::MAX
    );
}
