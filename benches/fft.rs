//! Benchmarks covering every public function of `sci::fft`.
//!
//! Inputs are deterministic so that results are reproducible across runs.

mod common;

use common::{sine_signal, values, warm};
use divan::{Bencher, black_box};
use sci::complex::Complex;
use sci::fft::RealTransformType;

fn main() {
    divan::main();
}

/// One-dimensional transform length.
const N: usize = 1024;
/// Side length of two-dimensional inputs.
const SIDE: usize = 32;
/// Row-major shape of two-dimensional inputs.
const SHAPE: [usize; 2] = [SIDE, SIDE];
/// Axes transformed by the multidimensional benchmarks.
const AXES: [usize; 2] = [0, 1];
/// Period used by the periodic pseudo-differential operators.
const PERIOD: f64 = std::f64::consts::TAU;

fn complex_values(n: usize, seed: u64) -> Vec<Complex> {
    values(2 * n, seed)
        .chunks(2)
        .map(|c| Complex { re: c[0], im: c[1] })
        .collect()
}

fn rows<T: Clone>(flat: &[T], cols: usize) -> Vec<Vec<T>> {
    flat.chunks(cols).map(<[T]>::to_vec).collect()
}

/// Half-spectrum of a real signal of length `N`.
fn half_spectrum() -> Vec<Complex> {
    sci::fft::rfft(&sine_signal(N))
}

/// Reduced 2-D spectrum of a real `SIDE x SIDE` grid.
fn half_spectrum_2d() -> Vec<Complex> {
    sci::fft::rfftn(&values(SIDE * SIDE, 5), &SHAPE, &AXES)
        .unwrap()
        .0
}

mod complex_transforms {
    use super::*;

    #[divan::bench]
    fn fft(bencher: Bencher) {
        let input = complex_values(N, 1);
        warm(bencher, || sci::fft::fft(black_box(&input)));
    }

    #[divan::bench]
    fn ifft(bencher: Bencher) {
        let input = complex_values(N, 2);
        warm(bencher, || sci::fft::ifft(black_box(&input)));
    }

    #[divan::bench]
    fn fftn(bencher: Bencher) {
        let input = complex_values(SIDE * SIDE, 3);
        warm(bencher, || {
            sci::fft::fftn(black_box(&input), &SHAPE, &AXES).unwrap()
        });
    }

    #[divan::bench]
    fn ifftn(bencher: Bencher) {
        let input = complex_values(SIDE * SIDE, 4);
        warm(bencher, || {
            sci::fft::ifftn(black_box(&input), &SHAPE, &AXES).unwrap()
        });
    }

    #[divan::bench]
    fn fft2(bencher: Bencher) {
        let input = rows(&complex_values(SIDE * SIDE, 5), SIDE);
        warm(bencher, || sci::fft::fft2(black_box(&input)).unwrap());
    }

    #[divan::bench]
    fn ifft2(bencher: Bencher) {
        let input = rows(&complex_values(SIDE * SIDE, 6), SIDE);
        warm(bencher, || sci::fft::ifft2(black_box(&input)).unwrap());
    }
}

mod real_transforms {
    use super::*;

    #[divan::bench]
    fn rfft(bencher: Bencher) {
        let input = sine_signal(N);
        warm(bencher, || sci::fft::rfft(black_box(&input)));
    }

    #[divan::bench]
    fn irfft(bencher: Bencher) {
        let input = half_spectrum();
        warm(bencher, || {
            sci::fft::irfft(black_box(&input), Some(N)).unwrap()
        });
    }

    #[divan::bench]
    fn hfft(bencher: Bencher) {
        let input = half_spectrum();
        warm(bencher, || {
            sci::fft::hfft(black_box(&input), Some(N)).unwrap()
        });
    }

    #[divan::bench]
    fn ihfft(bencher: Bencher) {
        let input = sine_signal(N);
        warm(bencher, || {
            sci::fft::ihfft(black_box(&input), None).unwrap()
        });
    }

    #[divan::bench]
    fn rfftn(bencher: Bencher) {
        let input = values(SIDE * SIDE, 7);
        warm(bencher, || {
            sci::fft::rfftn(black_box(&input), &SHAPE, &AXES).unwrap()
        });
    }

    #[divan::bench]
    fn irfftn(bencher: Bencher) {
        let input = half_spectrum_2d();
        warm(bencher, || {
            sci::fft::irfftn(black_box(&input), &SHAPE, &AXES).unwrap()
        });
    }

    #[divan::bench]
    fn hfftn(bencher: Bencher) {
        let input = half_spectrum_2d();
        warm(bencher, || {
            sci::fft::hfftn(black_box(&input), &SHAPE, &AXES).unwrap()
        });
    }

    #[divan::bench]
    fn ihfftn(bencher: Bencher) {
        let input = values(SIDE * SIDE, 8);
        warm(bencher, || {
            sci::fft::ihfftn(black_box(&input), &SHAPE, &AXES).unwrap()
        });
    }

    #[divan::bench]
    fn rfft2(bencher: Bencher) {
        let input = rows(&values(SIDE * SIDE, 9), SIDE);
        warm(bencher, || sci::fft::rfft2(black_box(&input)).unwrap());
    }

    #[divan::bench]
    fn irfft2(bencher: Bencher) {
        let input = rows(&half_spectrum_2d(), SIDE / 2 + 1);
        warm(bencher, || {
            sci::fft::irfft2(black_box(&input), SIDE).unwrap()
        });
    }

    #[divan::bench]
    fn hfft2(bencher: Bencher) {
        let input = rows(&half_spectrum_2d(), SIDE / 2 + 1);
        warm(bencher, || {
            sci::fft::hfft2(black_box(&input), SIDE).unwrap()
        });
    }

    #[divan::bench]
    fn ihfft2(bencher: Bencher) {
        let input = rows(&values(SIDE * SIDE, 10), SIDE);
        warm(bencher, || sci::fft::ihfft2(black_box(&input)).unwrap());
    }
}

mod helpers {
    use super::*;

    #[divan::bench]
    fn fftfreq(bencher: Bencher) {
        bencher.bench(|| sci::fft::fftfreq(black_box(N), black_box(0.001)).unwrap());
    }

    #[divan::bench]
    fn rfftfreq(bencher: Bencher) {
        bencher.bench(|| sci::fft::rfftfreq(black_box(N), black_box(0.001)).unwrap());
    }

    #[divan::bench]
    fn fftshift(bencher: Bencher) {
        let input = complex_values(N, 11);
        bencher.bench(|| sci::fft::fftshift(black_box(&input)).unwrap());
    }

    #[divan::bench]
    fn ifftshift(bencher: Bencher) {
        let input = complex_values(N, 12);
        bencher.bench(|| sci::fft::ifftshift(black_box(&input)).unwrap());
    }

    /// Searches upward from 128 awkward targets for both factor sets.
    #[divan::bench]
    fn next_fast_len(bencher: Bencher) {
        let targets: Vec<usize> = (0..128).map(|i| 1000 + 37 * i).collect();
        bencher.bench(|| {
            black_box(&targets)
                .iter()
                .map(|&t| {
                    sci::fft::next_fast_len(t, true).unwrap()
                        + sci::fft::next_fast_len(t, false).unwrap()
                })
                .sum::<usize>()
        });
    }

    /// Searches downward from 128 awkward targets for both factor sets.
    #[divan::bench]
    fn prev_fast_len(bencher: Bencher) {
        let targets: Vec<usize> = (0..128).map(|i| 1000 + 37 * i).collect();
        bencher.bench(|| {
            black_box(&targets)
                .iter()
                .map(|&t| {
                    sci::fft::prev_fast_len(t, true).unwrap()
                        + sci::fft::prev_fast_len(t, false).unwrap()
                })
                .sum::<usize>()
        });
    }
}

mod trigonometric_transforms {
    use super::*;

    const KIND: RealTransformType = RealTransformType::II;

    #[divan::bench]
    fn dct(bencher: Bencher) {
        let input = sine_signal(N);
        warm(bencher, || sci::fft::dct(black_box(&input), KIND).unwrap());
    }

    #[divan::bench]
    fn idct(bencher: Bencher) {
        let input = sine_signal(N);
        warm(bencher, || sci::fft::idct(black_box(&input), KIND).unwrap());
    }

    #[divan::bench]
    fn dst(bencher: Bencher) {
        let input = sine_signal(N);
        warm(bencher, || sci::fft::dst(black_box(&input), KIND).unwrap());
    }

    #[divan::bench]
    fn idst(bencher: Bencher) {
        let input = sine_signal(N);
        warm(bencher, || sci::fft::idst(black_box(&input), KIND).unwrap());
    }

    #[divan::bench]
    fn dctn(bencher: Bencher) {
        let input = values(SIDE * SIDE, 13);
        warm(bencher, || {
            sci::fft::dctn(black_box(&input), &SHAPE, &AXES, KIND).unwrap()
        });
    }

    #[divan::bench]
    fn idctn(bencher: Bencher) {
        let input = values(SIDE * SIDE, 14);
        warm(bencher, || {
            sci::fft::idctn(black_box(&input), &SHAPE, &AXES, KIND).unwrap()
        });
    }

    #[divan::bench]
    fn dstn(bencher: Bencher) {
        let input = values(SIDE * SIDE, 15);
        warm(bencher, || {
            sci::fft::dstn(black_box(&input), &SHAPE, &AXES, KIND).unwrap()
        });
    }

    #[divan::bench]
    fn idstn(bencher: Bencher) {
        let input = values(SIDE * SIDE, 16);
        warm(bencher, || {
            sci::fft::idstn(black_box(&input), &SHAPE, &AXES, KIND).unwrap()
        });
    }
}

mod hankel {
    use super::*;

    const HANKEL_N: usize = 256;
    const DLN: f64 = 0.05;
    const MU: f64 = 0.5;

    /// Samples of `r^(mu+1) exp(-r^2/2)` on a logarithmic grid.
    fn input() -> Vec<f64> {
        (0..HANKEL_N)
            .map(|i| {
                let r = (DLN * (i as f64 - HANKEL_N as f64 / 2.0)).exp();
                r.powf(MU + 1.0) * (-r * r / 2.0).exp()
            })
            .collect()
    }

    #[divan::bench]
    fn fht(bencher: Bencher) {
        let input = input();
        let offset = sci::fft::fhtoffset(DLN, MU, 0.0, 0.0).unwrap();
        warm(bencher, || {
            sci::fft::fht(black_box(&input), DLN, MU, offset, 0.0).unwrap()
        });
    }

    #[divan::bench]
    fn ifht(bencher: Bencher) {
        let input = input();
        let offset = sci::fft::fhtoffset(DLN, MU, 0.0, 0.0).unwrap();
        warm(bencher, || {
            sci::fft::ifht(black_box(&input), DLN, MU, offset, 0.0).unwrap()
        });
    }

    /// Evaluates the low-ringing offset for 64 orders.
    #[divan::bench]
    fn fhtoffset(bencher: Bencher) {
        let orders: Vec<f64> = (0..64).map(|i| i as f64 * 0.25).collect();
        bencher.bench(|| {
            black_box(&orders)
                .iter()
                .map(|&mu| sci::fft::fhtoffset(DLN, mu, 0.5, 0.0).unwrap())
                .sum::<f64>()
        });
    }
}

mod periodic {
    use super::*;

    #[divan::bench]
    fn diff(bencher: Bencher) {
        let input = sine_signal(N);
        warm(bencher, || {
            sci::fft::diff(black_box(&input), 1, PERIOD).unwrap()
        });
    }

    #[divan::bench]
    fn hilbert(bencher: Bencher) {
        let input = sine_signal(N);
        warm(bencher, || sci::fft::hilbert(black_box(&input)).unwrap());
    }

    #[divan::bench]
    fn ihilbert(bencher: Bencher) {
        let input = sine_signal(N);
        warm(bencher, || sci::fft::ihilbert(black_box(&input)).unwrap());
    }

    #[divan::bench]
    fn tilbert(bencher: Bencher) {
        let input = sine_signal(N);
        warm(bencher, || {
            sci::fft::tilbert(black_box(&input), 0.5, PERIOD).unwrap()
        });
    }

    #[divan::bench]
    fn itilbert(bencher: Bencher) {
        let input = sine_signal(N);
        warm(bencher, || {
            sci::fft::itilbert(black_box(&input), 0.5, PERIOD).unwrap()
        });
    }

    #[divan::bench]
    fn shift(bencher: Bencher) {
        let input = sine_signal(N);
        warm(bencher, || {
            sci::fft::shift(black_box(&input), 0.1, PERIOD).unwrap()
        });
    }

    #[divan::bench]
    fn cc_diff(bencher: Bencher) {
        let input = sine_signal(N);
        warm(bencher, || {
            sci::fft::cc_diff(black_box(&input), 0.1, 0.2, PERIOD).unwrap()
        });
    }

    #[divan::bench]
    fn cs_diff(bencher: Bencher) {
        let input = sine_signal(N);
        warm(bencher, || {
            sci::fft::cs_diff(black_box(&input), 0.1, 0.2, PERIOD).unwrap()
        });
    }

    #[divan::bench]
    fn sc_diff(bencher: Bencher) {
        let input = sine_signal(N);
        warm(bencher, || {
            sci::fft::sc_diff(black_box(&input), 0.1, 0.2, PERIOD).unwrap()
        });
    }

    #[divan::bench]
    fn ss_diff(bencher: Bencher) {
        let input = sine_signal(N);
        warm(bencher, || {
            sci::fft::ss_diff(black_box(&input), 0.1, 0.2, PERIOD).unwrap()
        });
    }
}
