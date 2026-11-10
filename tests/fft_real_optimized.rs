//! Independent DFT references for the real FFT paths and unusual reduced axes.
use sci::{complex::Complex, fft};
use std::f64::consts::TAU;

fn close(a: f64, b: f64, case: &str) {
    assert!(
        (a - b).abs() <= 3e-11 * (1.0 + b.abs()),
        "{case}: expected {b}, got {a}"
    );
}

fn frequency(index: usize, shape: &[usize], axis: usize) -> usize {
    let stride: usize = shape[axis + 1..].iter().product();
    index / stride % shape[axis]
}

fn direct_dft(
    input: &[f64],
    shape: &[usize],
    axes: &[usize],
    output: usize,
    small: &[usize],
) -> Complex {
    let mut result = Complex { re: 0.0, im: 0.0 };
    for (j, &value) in input.iter().enumerate() {
        // Axes excluded from the DFT retain their original coordinate.
        // Summing over them would incorrectly collapse independent slices.
        if (0..shape.len())
            .any(|a| !axes.contains(&a) && frequency(j, shape, a) != frequency(output, small, a))
        {
            continue;
        }
        let phase: f64 = axes
            .iter()
            .map(|&a| {
                frequency(j, shape, a) as f64 * frequency(output, small, a) as f64 / shape[a] as f64
            })
            .sum();
        let (s, c) = (-TAU * phase).sin_cos();
        result.re += value * c;
        result.im += value * s;
    }
    result
}

#[test]
fn reduced_multidimensional_fft_matches_direct_dft_for_axis_orders() {
    for shape in [&[3, 4, 5][..], &[1, 3, 2][..], &[2, 5, 1][..]] {
        let len = shape.iter().product::<usize>();
        let input: Vec<f64> = (0..len)
            .map(|j| (j as f64 * 0.217).cos() + (j as f64 * 0.41).sin())
            .collect();
        for axes in [
            &[0, 1, 2][..],
            &[2, 0][..],
            &[0, 2][..],
            &[1][..],
            &[2, 1, 0][..],
            &[1, 0][..],
        ] {
            let (actual, reduced) = fft::real::rfftn(&input, shape, axes).unwrap();
            let last = *axes.last().unwrap();
            assert_eq!(reduced[last], shape[last] / 2 + 1);
            assert_eq!(actual.len(), reduced.iter().product());
            for (i, got) in actual.iter().enumerate() {
                let reference = direct_dft(&input, shape, axes, i, &reduced);
                let case = format!("shape={shape:?}, axes={axes:?}, reduced={reduced:?}, bin={i}");
                close(got.re, reference.re, &format!("{case}, real"));
                close(got.im, reference.im, &format!("{case}, imaginary"));
            }
            let restored = fft::real::irfftn(&actual, shape, axes).unwrap();
            for (i, (&got, &want)) in restored.iter().zip(&input).enumerate() {
                close(
                    got,
                    want,
                    &format!("shape={shape:?}, axes={axes:?}, sample={i}, inverse"),
                );
            }
        }
    }
}

#[test]
fn strided_forward_real_fft_matches_selected_direct_dft_bins() {
    // Same layout as the strided performance benchmark: 256 short real
    // transforms on axis 0, followed by a complex FFT on axis 2.
    let shape = [16, 32, 8];
    let axes = [2, 0];
    let input: Vec<f64> = (0..shape.iter().product::<usize>())
        .map(|j| (j as f64 * 0.117).sin() + (j as f64 * 0.43).cos())
        .collect();
    let (actual, reduced) = fft::real::rfftn(&input, &shape, &axes).unwrap();
    assert_eq!(reduced, [9, 32, 8]);
    for (k0, k1, k2) in [(0, 0, 0), (2, 3, 4), (8, 31, 7), (6, 21, 1)] {
        let bin = (k0 * shape[1] + k1) * shape[2] + k2;
        let expected = direct_dft(&input, &shape, &axes, bin, &reduced);
        close(
            actual[bin].re,
            expected.re,
            &format!("strided FFT bin={bin}, real"),
        );
        close(
            actual[bin].im,
            expected.im,
            &format!("strided FFT bin={bin}, imaginary"),
        );
    }
    for (i, (&actual, &expected)) in fft::real::irfftn(&actual, &shape, &axes)
        .unwrap()
        .iter()
        .zip(&input)
        .enumerate()
    {
        close(actual, expected, &format!("strided FFT inverse sample={i}"));
    }
}

#[test]
fn inverse_real_fft_matches_direct_inverse_with_padding_cropping_and_endpoints() {
    for n in [1, 2, 3, 4, 7, 8, 31, 32] {
        for m in [0, 1, n / 2 + 1, n / 2 + 3] {
            let bins: Vec<_> = (0..m)
                .map(|k| Complex {
                    re: (k as f64 * 0.23 + 1.0).sin(),
                    im: (k as f64 * 0.52 + 2.0).cos(),
                })
                .collect();
            let mut full = vec![Complex { re: 0.0, im: 0.0 }; n];
            for (k, &v) in bins.iter().take(n / 2 + 1).enumerate() {
                full[k] = v;
                if k == 0 || n.is_multiple_of(2) && k == n / 2 {
                    full[k].im = 0.0;
                } else {
                    full[n - k] = Complex {
                        re: v.re,
                        im: -v.im,
                    };
                }
            }
            let restored = fft::real::irfft(&bins, Some(n)).unwrap();
            for (j, &got) in restored.iter().enumerate() {
                let expected: f64 = full
                    .iter()
                    .enumerate()
                    .map(|(k, v)| {
                        let (s, c) = (TAU * k as f64 * j as f64 / n as f64).sin_cos();
                        v.re * c - v.im * s
                    })
                    .sum::<f64>()
                    / n as f64;
                close(
                    got,
                    expected,
                    &format!("n={n}, bins={m}, sample={j}, inverse"),
                );
            }
        }
    }
}

#[test]
fn inverse_real_fft_native_f32_round_trip_and_validation() {
    for n in [1, 2, 5, 16, 127, 128, 257] {
        let samples: Vec<f32> = (0..n)
            .map(|j| (j as f32 * 0.17).sin() + (j as f32 * 0.13).cos())
            .collect();
        let spectrum = fft::real::rfft(&samples);
        let restored = fft::real::irfft(&spectrum, Some(n)).unwrap();
        for (a, b) in restored.iter().zip(samples) {
            assert!((a - b).abs() < 2e-5 * (1.0 + b.abs()));
        }
    }
    assert!(
        fft::real::irfft(
            &[Complex {
                re: f32::NAN,
                im: 0.0
            }],
            Some(1)
        )
        .is_err()
    );
    assert!(fft::real::rfftn(&[f64::MAX; 2], &[2], &[0]).is_err());
}
