//! Benchmarks covering every public function of `sci::filter`.
//!
//! All filter steps are pure functions of their inputs, so every iteration
//! starts from the same deterministic state.

mod common;

use common::{points, unit_values, values};
use divan::{Bencher, black_box};
use sci::filter::{self, Gaussian, ParticleSet, SigmaPoints};

fn main() {
    divan::main();
}

/// Time step of the constant-velocity model.
const DT: f64 = 0.1;

/// Number of particles used by the particle-filter benchmarks.
const PARTICLES: usize = 500;

fn identity(n: usize, scale: f64) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| (0..n).map(|j| if i == j { scale } else { 0.0 }).collect())
        .collect()
}

/// 3-D constant-velocity state `[position; velocity]` with correlated covariance.
fn prior() -> Gaussian {
    let mut covariance = identity(6, 1.0);
    for i in 0..3 {
        covariance[i][i + 3] = 0.2;
        covariance[i + 3][i] = 0.2;
    }
    Gaussian {
        mean: vec![1.0, -0.5, 2.0, 0.3, 0.1, -0.2],
        covariance,
    }
}

/// Constant-velocity transition matrix.
fn transition() -> Vec<Vec<f64>> {
    let mut f = identity(6, 1.0);
    for (i, row) in f.iter_mut().take(3).enumerate() {
        row[i + 3] = DT;
    }
    f
}

/// Position-only measurement matrix.
fn measurement() -> Vec<Vec<f64>> {
    (0..3)
        .map(|i| (0..6).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect()
}

/// Nonlinear dynamics: constant velocity with quadratic drag.
fn process(x: &[f64]) -> sci::Result<Vec<f64>> {
    let mut y = x.to_vec();
    for i in 0..3 {
        y[i] += DT * x[i + 3];
        y[i + 3] -= DT * 0.1 * x[i + 3] * x[i + 3].abs();
    }
    Ok(y)
}

fn process_jacobian(x: &[f64]) -> sci::Result<Vec<Vec<f64>>> {
    let mut f = transition();
    for i in 0..3 {
        f[i + 3][i + 3] = 1.0 - DT * 0.2 * x[i + 3].abs();
    }
    Ok(f)
}

/// Nonlinear observation: range to the origin and two position coordinates.
fn observe(x: &[f64]) -> sci::Result<Vec<f64>> {
    let range = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
    Ok(vec![range, x[1], x[2]])
}

fn observe_jacobian(x: &[f64]) -> sci::Result<Vec<Vec<f64>>> {
    let range = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
    let mut h = vec![vec![0.0; 6]; 3];
    for j in 0..3 {
        h[0][j] = x[j] / range;
    }
    h[1][1] = 1.0;
    h[2][2] = 1.0;
    Ok(h)
}

mod kalman_benches {
    use super::*;

    #[divan::bench]
    fn kalman_predict(bencher: Bencher) {
        let x = prior();
        let f = transition();
        let q = identity(6, 0.01);
        bencher
            .bench(|| filter::kalman_predict(black_box(&x), black_box(&f), black_box(&q)).unwrap());
    }

    #[divan::bench]
    fn kalman_update(bencher: Bencher) {
        let x = prior();
        let h = measurement();
        let r = identity(3, 0.5);
        let z = [1.1, -0.4, 2.1];
        bencher.bench(|| {
            filter::kalman_update(black_box(&x), black_box(&z), black_box(&h), black_box(&r))
                .unwrap()
        });
    }

    #[divan::bench]
    fn extended_kalman_predict(bencher: Bencher) {
        let x = prior();
        let q = identity(6, 0.01);
        bencher.bench(|| {
            filter::extended_kalman_predict(black_box(&x), black_box(&q), process, process_jacobian)
                .unwrap()
        });
    }

    #[divan::bench]
    fn extended_kalman_update(bencher: Bencher) {
        let x = prior();
        let r = identity(3, 0.5);
        let z = [2.4, -0.4, 2.1];
        bencher.bench(|| {
            filter::extended_kalman_update(
                black_box(&x),
                black_box(&z),
                black_box(&r),
                observe,
                observe_jacobian,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn unscented_kalman_step(bencher: Bencher) {
        let x = prior();
        let q = identity(6, 0.01);
        let r = identity(3, 0.5);
        let z = [2.4, -0.4, 2.1];
        bencher.bench(|| {
            filter::unscented_kalman_step(
                black_box(&x),
                black_box(&q),
                black_box(&z),
                black_box(&r),
                SigmaPoints::default(),
                process,
                observe,
            )
            .unwrap()
        });
    }
}

mod particle_benches {
    use super::*;

    /// Deterministic weighted 2-D particle cloud.
    fn particles() -> ParticleSet {
        ParticleSet {
            states: points(PARTICLES, 2, 51),
            weights: unit_values(PARTICLES, 52),
        }
    }

    #[divan::bench]
    fn particle_predict(bencher: Bencher) {
        let p = particles();
        // Fixed-seed process noise, replayed identically on every iteration.
        let noise = values(2 * PARTICLES, 53);
        bencher.bench(|| {
            let mut i = 0;
            filter::particle_predict(black_box(&p), |x| {
                let y = vec![
                    x[0] + DT * x[1] + 0.01 * noise[i],
                    0.99 * x[1] - DT * x[0].sin() + 0.01 * noise[i + 1],
                ];
                i += 2;
                Ok(y)
            })
            .unwrap()
        });
    }

    #[divan::bench]
    fn particle_update(bencher: Bencher) {
        let p = particles();
        let z = [0.2, -0.1];
        bencher.bench(|| {
            filter::particle_update(black_box(&p), |x| {
                let dx = x[0] - z[0];
                let dy = x[1] - z[1];
                Ok(-0.5 * (dx * dx + dy * dy) / 0.25)
            })
            .unwrap()
        });
    }

    #[divan::bench]
    fn particle_estimate(bencher: Bencher) {
        let p = particles();
        bencher.bench(|| filter::particle_estimate(black_box(&p)).unwrap());
    }

    #[divan::bench]
    fn effective_sample_size(bencher: Bencher) {
        let p = particles();
        bencher.bench(|| filter::effective_sample_size(black_box(&p)).unwrap());
    }

    #[divan::bench]
    fn systematic_resample(bencher: Bencher) {
        let p = particles();
        bencher.bench(|| filter::systematic_resample(black_box(&p), black_box(0.37)).unwrap());
    }
}
