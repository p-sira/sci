//! Benchmarks covering every public function of `sci::optimize`.
//!
//! Inputs are deterministic so that results are reproducible across runs.

mod common;

use common::values;
use divan::{Bencher, black_box};
use sci::optimize;

fn main() {
    divan::main();
}

/// Ill-conditioned convex quadratic `Σ (i + 1) (x_i - 1)²` and its gradient.
fn quadratic(x: &[f64]) -> f64 {
    x.iter()
        .enumerate()
        .map(|(i, v)| (i + 1) as f64 * (v - 1.0).powi(2))
        .sum()
}

fn quadratic_gradient(x: &[f64]) -> Vec<f64> {
    x.iter()
        .enumerate()
        .map(|(i, v)| 2.0 * (i + 1) as f64 * (v - 1.0))
        .collect()
}

/// Capped iteration budget keeping each minimization short.
fn options(max_iterations: u64, learning_rate: f64) -> optimize::minimize::Options {
    optimize::minimize::Options {
        max_iterations,
        tolerance: 1e-8,
        learning_rate,
    }
}

mod minimize_benches {
    use super::*;

    #[divan::bench]
    fn minimize(bencher: Bencher) {
        let start = [-1.2, 1.0];
        bencher.bench(|| {
            optimize::minimize::minimize(
                |x| optimize::derivatives::rosen(x).unwrap(),
                black_box(&start),
                options(200, 0.1),
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn try_minimize(bencher: Bencher) {
        let start = [0.0, 0.0, 0.0, 0.0];
        bencher.bench(|| {
            optimize::minimize::try_minimize(
                |x| Ok(quadratic(x)),
                black_box(&start),
                options(200, 0.1),
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn minimize_with_gradient(bencher: Bencher) {
        let start = [0.0, 0.0, 0.0, 0.0];
        bencher.bench(|| {
            optimize::minimize::minimize_with_gradient(
                quadratic,
                quadratic_gradient,
                black_box(&start),
                options(200, 0.1),
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn try_minimize_with_gradient(bencher: Bencher) {
        let start = [-1.2, 1.0];
        bencher.bench(|| {
            optimize::minimize::try_minimize_with_gradient(
                optimize::derivatives::rosen,
                optimize::derivatives::rosen_der,
                black_box(&start),
                options(200, 1e-3),
            )
            .unwrap()
        });
    }
}

mod derivative_benches {
    use super::*;

    #[divan::bench]
    fn rosen(bencher: Bencher) {
        let x = values(1000, 1);
        bencher.bench(|| optimize::derivatives::rosen(black_box(&x)).unwrap());
    }

    #[divan::bench]
    fn rosen_der(bencher: Bencher) {
        let x = values(1000, 1);
        bencher.bench(|| optimize::derivatives::rosen_der(black_box(&x)).unwrap());
    }

    #[divan::bench]
    fn rosen_hess(bencher: Bencher) {
        let x = values(100, 1);
        bencher.bench(|| optimize::derivatives::rosen_hess(black_box(&x)).unwrap());
    }

    #[divan::bench]
    fn rosen_hess_prod(bencher: Bencher) {
        let x = values(1000, 1);
        let p = values(1000, 2);
        bencher.bench(|| {
            optimize::derivatives::rosen_hess_prod(black_box(&x), black_box(&p)).unwrap()
        });
    }

    #[divan::bench]
    fn approx_fprime(bencher: Bencher) {
        let x = values(32, 1);
        let epsilon = [1e-7];
        bencher.bench(|| {
            optimize::derivatives::approx_fprime(
                |x| optimize::derivatives::rosen_der(x).unwrap(),
                black_box(&x),
                black_box(&epsilon),
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn check_grad(bencher: Bencher) {
        let x = values(64, 1);
        let epsilon = [1e-7];
        bencher.bench(|| {
            optimize::derivatives::check_grad(
                |x| optimize::derivatives::rosen(x).unwrap(),
                |x| optimize::derivatives::rosen_der(x).unwrap(),
                black_box(&x),
                black_box(&epsilon),
            )
            .unwrap()
        });
    }
}

mod dogbox_benches {
    use super::*;
    use optimize::least_squares::{DogboxOptions, dogbox_with_jacobian};

    // Replicated independent residuals keep callback work small relative to the
    // dense SVD. The logarithm forces a nonfinite trial before recovery.
    fn solve(n: usize, copies: usize, logarithmic: bool, rank_deficient: bool) {
        let start = vec![1.0; n];
        let lower = vec![0.0; n];
        let upper = vec![10.0; n];
        let result = dogbox_with_jacobian(
            |x| {
                (0..n * copies)
                    .map(|i| {
                        let j = if rank_deficient { i % (n / 2) } else { i % n };
                        if logarithmic {
                            x[j].ln() + 2.0
                        } else {
                            x[j] - 3.0
                        }
                    })
                    .collect()
            },
            |x| {
                (0..n * copies)
                    .map(|i| {
                        let j = if rank_deficient { i % (n / 2) } else { i % n };
                        let mut row = vec![0.0; n];
                        row[j] = if logarithmic { x[j].recip() } else { 1.0 };
                        row
                    })
                    .collect()
            },
            &start,
            &lower,
            &upper,
            DogboxOptions::default(),
        )
        .unwrap();
        black_box(result);
    }

    /// Typical bounded curve fit: `y = a·exp(-b·t) + c·exp(-d·t)` on 64 noisy
    /// samples, starting away from the optimum with well-conditioned columns.
    #[divan::bench]
    fn normal(bencher: Bencher) {
        let t: Vec<f64> = (0..64).map(|i| i as f64 * 0.125).collect();
        let y: Vec<f64> = t
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                3.0 * (-1.5 * t).exp() + 1.0 * (-0.2 * t).exp() + 1e-3 * (i as f64).sin()
            })
            .collect();
        let start = [1.0, 1.0, 2.0, 0.5];
        let lower = [0.0; 4];
        let upper = [10.0; 4];
        bencher.bench(|| {
            let result = dogbox_with_jacobian(
                |p| {
                    t.iter()
                        .zip(&y)
                        .map(|(&t, &y)| p[0] * (-p[1] * t).exp() + p[2] * (-p[3] * t).exp() - y)
                        .collect()
                },
                |p| {
                    t.iter()
                        .map(|&t| {
                            let e1 = (-p[1] * t).exp();
                            let e2 = (-p[3] * t).exp();
                            vec![e1, -p[0] * t * e1, e2, -p[2] * t * e2]
                        })
                        .collect()
                },
                black_box(&start),
                &lower,
                &upper,
                DogboxOptions::default(),
            )
            .unwrap();
            black_box(result)
        });
    }

    #[divan::bench]
    fn rejected_trials(bencher: Bencher) {
        bencher.bench(|| solve(16, 4, true, false));
    }

    #[divan::bench]
    fn tall(bencher: Bencher) {
        bencher.bench(|| solve(16, 32, false, false));
    }

    #[divan::bench]
    fn rank_deficient(bencher: Bencher) {
        bencher.bench(|| solve(16, 4, false, true));
    }
}
