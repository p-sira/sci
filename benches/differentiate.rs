//! Benchmarks covering every public function of `sci::differentiate`.
//!
//! Inputs are deterministic so that results are reproducible across runs.

mod common;

use common::values;
use divan::{Bencher, black_box};

fn main() {
    divan::main();
}

/// Extended Rosenbrock function.
fn rosenbrock(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
        .sum()
}

/// Coupled nonlinear vector field with one output per input coordinate.
fn field(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n)
        .map(|i| x[i].sin() * x[(i + 1) % n] + x[i] * x[i] - x[(i + n - 1) % n].exp())
        .collect()
}

mod finite_differences {
    use super::*;

    /// Scalar derivative evaluated at 256 points.
    #[divan::bench]
    fn derivative(bencher: Bencher) {
        let xs = values(256, 1);
        bencher.bench(|| {
            black_box(&xs)
                .iter()
                .map(|&x| sci::differentiate::derivative(|t| t.sin() * t.exp(), x, 1e-5).unwrap())
                .sum::<f64>()
        });
    }

    #[divan::bench]
    fn gradient(bencher: Bencher) {
        let point = values(64, 2);
        bencher
            .bench(|| sci::differentiate::gradient(rosenbrock, black_box(&point), 1e-6).unwrap());
    }

    #[divan::bench]
    fn jacobian(bencher: Bencher) {
        let point = values(32, 3);
        bencher.bench(|| sci::differentiate::jacobian(field, black_box(&point), 1e-6).unwrap());
    }

    #[divan::bench]
    fn hessian(bencher: Bencher) {
        let point = values(16, 4);
        bencher.bench(|| sci::differentiate::hessian(rosenbrock, black_box(&point), 1e-4).unwrap());
    }
}
