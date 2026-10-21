//! Deterministic input generators shared by the benchmark targets.
//!
//! Each benchmark target includes this module, so not every helper is used by
//! every target.
#![allow(dead_code)]

use divan::{Bencher, black_box};

/// Deterministic pseudo-random values in `[-1, 1)` from a linear congruential generator.
pub fn values(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

/// Deterministic pseudo-random values in `(0, 1)`.
pub fn unit_values(n: usize, seed: u64) -> Vec<f64> {
    values(n, seed)
        .into_iter()
        .map(|v| 0.5 * (v + 1.0).clamp(1e-6, 2.0 - 1e-6))
        .collect()
}

/// Deterministic pseudo-random values in `(0.1, 10.1)`.
pub fn positive_values(n: usize, seed: u64) -> Vec<f64> {
    values(n, seed).into_iter().map(|v| 5.1 + 5.0 * v).collect()
}

/// `n` points of dimension `dim`.
pub fn points(n: usize, dim: usize, seed: u64) -> Vec<Vec<f64>> {
    values(n * dim, seed)
        .chunks(dim)
        .map(<[f64]>::to_vec)
        .collect()
}

/// Evenly spaced grid of `n` values on `[start, stop]`.
pub fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![start];
    }
    (0..n)
        .map(|i| start + (stop - start) * i as f64 / (n - 1) as f64)
        .collect()
}

/// Diagonally dominant matrix, well conditioned for factorizations.
pub fn square_matrix(n: usize) -> Vec<Vec<f64>> {
    let mut a = points(n, n, 7);
    for (i, row) in a.iter_mut().enumerate() {
        row[i] += n as f64;
    }
    a
}

/// Symmetric, diagonally dominant matrix.
pub fn symmetric_matrix(n: usize) -> Vec<Vec<f64>> {
    let a = square_matrix(n);
    (0..n)
        .map(|i| (0..n).map(|j| 0.5 * (a[i][j] + a[j][i])).collect())
        .collect()
}

/// Symmetric positive-definite matrix.
pub fn spd_matrix(n: usize) -> Vec<Vec<f64>> {
    let mut a = symmetric_matrix(n);
    for (i, row) in a.iter_mut().enumerate() {
        row[i] += n as f64;
    }
    a
}

/// Noisy two-tone signal sampled at 1 kHz.
pub fn sine_signal(n: usize) -> Vec<f64> {
    let noise = values(n, 11);
    (0..n)
        .map(|i| {
            let t = i as f64 / 1000.0;
            (2.0 * std::f64::consts::PI * 50.0 * t).sin()
                + 0.5 * (2.0 * std::f64::consts::PI * 120.0 * t).sin()
                + 0.1 * noise[i]
        })
        .collect()
}

/// Runs `f` once before measuring so that one-time initialization (such as
/// CPU cache-size detection in matrix kernels or building a cached FFT plan)
/// is excluded.
pub fn warm<T>(bencher: Bencher, mut f: impl FnMut() -> T) {
    black_box(f());
    bencher.bench_local(f);
}
