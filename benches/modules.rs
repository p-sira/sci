//! Benchmarks covering representative functions of each Sci module.
//!
//! Inputs are deterministic so that results are reproducible across runs.

use divan::{Bencher, black_box};
use sci::{
    cluster, complex::Complex, differentiate, fft, integrate, interpolate, linalg, optimize,
    signal, spatial, special, stats,
};

fn main() {
    divan::main();
}

/// Deterministic pseudo-random values in `[-1, 1)` from a linear congruential generator.
fn values(n: usize, seed: u64) -> Vec<f64> {
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

fn points(n: usize, dim: usize, seed: u64) -> Vec<Vec<f64>> {
    values(n * dim, seed)
        .chunks(dim)
        .map(<[f64]>::to_vec)
        .collect()
}

/// Diagonally dominant matrix, well conditioned for factorizations.
fn square_matrix(n: usize) -> Vec<Vec<f64>> {
    let mut a = points(n, n, 7);
    for (i, row) in a.iter_mut().enumerate() {
        row[i] += n as f64;
    }
    a
}

fn symmetric_matrix(n: usize) -> Vec<Vec<f64>> {
    let a = square_matrix(n);
    (0..n)
        .map(|i| (0..n).map(|j| 0.5 * (a[i][j] + a[j][i])).collect())
        .collect()
}

fn sine_signal(n: usize) -> Vec<f64> {
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
/// CPU cache-size detection in the matrix kernels or building a cached FFT
/// plan) is excluded.
fn warm<T>(bencher: Bencher, mut f: impl FnMut() -> T) {
    black_box(f());
    bencher.bench_local(f);
}

mod linalg_benches {
    use super::*;

    const SIZES: [usize; 2] = [16, 64];

    #[divan::bench(args = SIZES)]
    fn matmul(bencher: Bencher, n: usize) {
        let a = square_matrix(n);
        let b = symmetric_matrix(n);
        warm(bencher, || {
            linalg::matmul(black_box(&a), black_box(&b)).unwrap()
        });
    }

    #[divan::bench(args = SIZES)]
    fn solve(bencher: Bencher, n: usize) {
        let a = square_matrix(n);
        let b = values(n, 3);
        warm(bencher, || {
            linalg::solve(black_box(&a), black_box(&b)).unwrap()
        });
    }

    #[divan::bench(args = SIZES)]
    fn det(bencher: Bencher, n: usize) {
        let a = square_matrix(n);
        warm(bencher, || linalg::det(black_box(&a)).unwrap());
    }

    #[divan::bench(args = SIZES)]
    fn inv(bencher: Bencher, n: usize) {
        let a = square_matrix(n);
        warm(bencher, || linalg::inv(black_box(&a)).unwrap());
    }

    #[divan::bench(args = SIZES)]
    fn svd(bencher: Bencher, n: usize) {
        let a = square_matrix(n);
        warm(bencher, || linalg::svd(black_box(&a)).unwrap());
    }

    #[divan::bench(args = SIZES)]
    fn eigh(bencher: Bencher, n: usize) {
        let a = symmetric_matrix(n);
        warm(bencher, || linalg::eigh(black_box(&a), true).unwrap());
    }
}

mod fft_benches {
    use super::*;

    const SIZES: [usize; 3] = [256, 1000, 4096];

    #[divan::bench(args = SIZES)]
    fn fft(bencher: Bencher, n: usize) {
        let input: Vec<Complex> = values(n, 5)
            .chunks(1)
            .zip(values(n, 6))
            .map(|(re, im)| Complex { re: re[0], im })
            .collect();
        warm(bencher, || fft::fft(black_box(&input)));
    }

    #[divan::bench(args = SIZES)]
    fn rfft(bencher: Bencher, n: usize) {
        let input = values(n, 5);
        warm(bencher, || fft::rfft(black_box(&input)));
    }

    #[divan::bench(args = SIZES)]
    fn dct_ii(bencher: Bencher, n: usize) {
        let input = values(n, 5);
        warm(bencher, || {
            fft::dct(black_box(&input), fft::RealTransformType::II).unwrap()
        });
    }
}

mod stats_benches {
    use super::*;

    const N: usize = 10_000;

    #[divan::bench]
    fn mean(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::mean(black_box(&x)).unwrap());
    }

    #[divan::bench]
    fn variance(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::variance(black_box(&x), 1).unwrap());
    }

    #[divan::bench]
    fn median(bencher: Bencher) {
        let x = values(N, 1);
        bencher.bench(|| stats::median(black_box(&x)).unwrap());
    }

    #[divan::bench]
    fn normal_cdf_ppf(bencher: Bencher) {
        let x = values(1_000, 2);
        bencher.bench(|| {
            x.iter()
                .map(|&v| {
                    let p = stats::normal_cdf(black_box(v), 0.0, 1.0).unwrap();
                    stats::normal_ppf(p, 0.0, 1.0).unwrap()
                })
                .sum::<f64>()
        });
    }
}

mod special_benches {
    use super::*;

    fn grid() -> Vec<f64> {
        (1..=1_000).map(|i| i as f64 / 100.0).collect()
    }

    #[divan::bench]
    fn gamma(bencher: Bencher) {
        let x = grid();
        bencher.bench(|| x.iter().map(|&v| special::gamma(black_box(v))).sum::<f64>());
    }

    #[divan::bench]
    fn erf(bencher: Bencher) {
        let x = values(1_000, 4);
        bencher.bench(|| x.iter().map(|&v| special::erf(black_box(v))).sum::<f64>());
    }

    #[divan::bench]
    fn bessel_jv(bencher: Bencher) {
        let x = grid();
        bencher.bench(|| {
            x.iter()
                .map(|&v| special::jv(2.5, black_box(v)))
                .sum::<f64>()
        });
    }

    #[divan::bench]
    fn hyp2f1(bencher: Bencher) {
        let x: Vec<f64> = values(1_000, 9).iter().map(|v| 0.9 * v).collect();
        bencher.bench(|| {
            x.iter()
                .map(|&v| special::hyp2f1(0.5, 1.5, 2.5, black_box(v)))
                .sum::<f64>()
        });
    }

    #[divan::bench]
    fn ellipk(bencher: Bencher) {
        let m: Vec<f64> = values(1_000, 10).iter().map(|v| 0.99 * v).collect();
        bencher.bench(|| {
            m.iter()
                .map(|&v| special::elliptic::ellipk(black_box(v)).unwrap())
                .sum::<f64>()
        });
    }
}

mod integrate_benches {
    use super::*;

    const N: usize = 10_001;

    fn samples() -> Vec<f64> {
        (0..N).map(|i| (i as f64 / 1000.0).sin()).collect()
    }

    #[divan::bench]
    fn trapezoid(bencher: Bencher) {
        let y = samples();
        bencher.bench(|| integrate::trapezoid(black_box(&y), None, 0.001).unwrap());
    }

    #[divan::bench]
    fn simpson(bencher: Bencher) {
        let y = samples();
        bencher.bench(|| integrate::simpson(black_box(&y), None, 0.001).unwrap());
    }
}

mod interpolate_benches {
    use super::*;

    fn nodes() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..100).map(|i| i as f64 * 0.2).collect();
        let y: Vec<f64> = x.iter().map(|v| v.sin()).collect();
        let q: Vec<f64> = (0..2_000).map(|i| i as f64 * 0.00995).collect();
        (x, y, q)
    }

    #[divan::bench]
    fn make_interp_spline_cubic(bencher: Bencher) {
        let (x, y, _) = nodes();
        bencher.bench(|| interpolate::make_interp_spline(black_box(&x), black_box(&y), 3).unwrap());
    }

    #[divan::bench]
    fn splev_cubic(bencher: Bencher) {
        let (x, y, q) = nodes();
        let spline = interpolate::make_interp_spline(&x, &y, 3).unwrap();
        bencher.bench(|| {
            interpolate::splev(
                black_box(&q),
                &spline,
                0,
                interpolate::SplineExtension::Extrapolate,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn pchip_interpolate(bencher: Bencher) {
        let (x, y, q) = nodes();
        bencher.bench(|| {
            interpolate::pchip_interpolate(black_box(&x), black_box(&y), black_box(&q)).unwrap()
        });
    }
}

mod signal_benches {
    use super::*;

    const N: usize = 4_096;

    #[divan::bench(args = [16, 128])]
    fn convolve(bencher: Bencher, taps: usize) {
        let x = sine_signal(N);
        let h = values(taps, 12);
        bencher.bench(|| signal::convolve(black_box(&x), black_box(&h), sci::Mode::Full).unwrap());
    }

    #[divan::bench]
    fn butter_design(bencher: Bencher) {
        bencher.bench(|| {
            signal::butter(
                black_box(6),
                &[100.0],
                signal::BandType::Lowpass,
                Some(1000.0),
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn lfilter_butter(bencher: Bencher) {
        let x = sine_signal(N);
        let (b, a) = signal::butter(6, &[100.0], signal::BandType::Lowpass, Some(1000.0)).unwrap();
        bencher.bench(|| signal::lfilter(black_box(&b), black_box(&a), black_box(&x)).unwrap());
    }

    #[divan::bench]
    fn welch(bencher: Bencher) {
        let x = sine_signal(N);
        let window: Vec<f64> = (0..256)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / 256.0).cos())
            .collect();
        bencher.bench(|| {
            signal::welch(
                black_box(&x),
                &window,
                128,
                256,
                1000.0,
                true,
                signal::Scaling::Density,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn find_peaks(bencher: Bencher) {
        let x = sine_signal(N);
        bencher.bench(|| signal::find_peaks(black_box(&x)).unwrap());
    }
}

mod spatial_benches {
    use super::*;

    #[divan::bench]
    fn pdist_euclidean(bencher: Bencher) {
        let x = points(200, 3, 13);
        bencher.bench(|| spatial::pdist(black_box(&x), spatial::Metric::Euclidean).unwrap());
    }

    #[divan::bench]
    fn cdist_euclidean(bencher: Bencher) {
        let x = points(200, 3, 13);
        let y = points(100, 3, 14);
        bencher.bench(|| {
            spatial::cdist(black_box(&x), black_box(&y), spatial::Metric::Euclidean).unwrap()
        });
    }
}

mod cluster_benches {
    use super::*;

    #[divan::bench]
    fn kmeans2(bencher: Bencher) {
        let obs = points(500, 2, 15);
        let initial = obs[..4].to_vec();
        bencher.bench(|| cluster::vq::kmeans2(black_box(&obs), black_box(&initial), 10).unwrap());
    }

    #[divan::bench]
    fn linkage_average(bencher: Bencher) {
        let x = points(100, 3, 16);
        let y = spatial::pdist(&x, spatial::Metric::Euclidean).unwrap();
        bencher.bench(|| {
            cluster::hierarchy::linkage(black_box(&y), cluster::hierarchy::Method::Average).unwrap()
        });
    }
}

mod optimize_benches {
    use super::*;

    #[divan::bench]
    fn minimize_rosenbrock(bencher: Bencher) {
        bencher.bench(|| {
            optimize::minimize(
                |x| optimize::rosen(x).unwrap(),
                black_box(&[-1.2, 1.0]),
                optimize::Options::default(),
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn minimize_with_gradient_rosenbrock(bencher: Bencher) {
        bencher.bench(|| {
            optimize::minimize_with_gradient(
                |x| optimize::rosen(x).unwrap(),
                |x| optimize::rosen_der(x).unwrap(),
                black_box(&[0.9, 0.8]),
                optimize::Options {
                    max_iterations: 2_000,
                    learning_rate: 0.001,
                    tolerance: 1e-6,
                },
            )
            .unwrap()
        });
    }
}

mod differentiate_benches {
    use super::*;

    #[divan::bench]
    fn gradient(bencher: Bencher) {
        let x = values(32, 17);
        bencher.bench(|| {
            differentiate::gradient(
                |p| p.iter().map(|v| v.sin() * v.exp()).sum(),
                black_box(&x),
                1e-5,
            )
            .unwrap()
        });
    }
}
