use sci_core::Mode;
use sci_signal::{
    convolution::{convolve, fftconvolve, upfirdn},
    filtering::{lfilter, sosfilt},
    spectral::{Scaling, coherence, csd, welch},
    windows::general_cosine,
};
use std::f64::consts::TAU;

fn close(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() <= tol * (1. + expected.abs()),
        "actual {actual:?}, expected {expected:?}, tolerance {tol}"
    );
}

#[test]
fn upfirdn_agrees_with_explicit_upsampling_and_fir() {
    for (nx, nh, up, down) in [
        (1, 1, 1, 1),
        (2, 5, 3, 2),
        (8, 11, 1, 7),
        (9, 4, 7, 9),
        (9, 19, 9, 1),
        (12, 8, 2, 5),
        (15, 3, 1, 1),
    ] {
        let x: Vec<_> = (0..nx).map(|i| (0.4 * i as f64).sin()).collect();
        let h: Vec<_> = (0..nh)
            .map(|i| ((i + 1) as f64).cos() / (i + 1) as f64)
            .collect();
        let mut full = vec![0.; (nx - 1) * up + nh];
        for (i, &sample) in x.iter().enumerate() {
            for (j, &tap) in h.iter().enumerate() {
                full[i * up + j] += sample * tap;
            }
        }
        let expected: Vec<_> = full.into_iter().step_by(down).collect();
        let actual = upfirdn(&h, &x, up, down).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.into_iter().zip(expected) {
            close(a, e, 1e-14);
        }
    }
}

#[test]
fn real_fft_convolution_agrees_with_direct_for_all_modes() {
    for (nx, nh) in [(1, 1), (3, 2), (2, 7), (6, 6), (15, 8), (128, 33)] {
        let x: Vec<_> = (0..nx).map(|i| (i as f64 * 0.37).cos() - 0.2).collect();
        let h: Vec<_> = (0..nh).map(|i| (i as f64 * 0.23).sin() + 0.7).collect();
        for mode in [Mode::Full, Mode::Same, Mode::Valid] {
            let expected = convolve(&x, &h, mode).unwrap();
            let actual = fftconvolve(&x, &h, mode).unwrap();
            assert_eq!(actual.len(), expected.len());
            for (a, e) in actual.into_iter().zip(expected) {
                close(a, e, 3e-13);
            }
        }
    }
}

// Small independent DFT: frequency-domain tests do not use Sci's FFT backend.
fn direct_dft(samples: &[f64], nfft: usize) -> Vec<(f64, f64)> {
    (0..=nfft / 2)
        .map(|k| {
            samples
                .iter()
                .enumerate()
                .fold((0., 0.), |(re, im), (j, &x)| {
                    let angle = -TAU * (k * j) as f64 / nfft as f64;
                    (re + x * angle.cos(), im + x * angle.sin())
                })
        })
        .collect()
}

fn spectrum_segment(x: &[f64], window: &[f64], nfft: usize, detrend: bool) -> Vec<(f64, f64)> {
    let mean = if detrend {
        x.iter().sum::<f64>() / x.len() as f64
    } else {
        0.
    };
    let windowed: Vec<_> = x
        .iter()
        .zip(window)
        .map(|(&x, &w)| (x - mean) * w)
        .collect();
    direct_dft(&windowed, nfft)
}

#[test]
fn welch_matches_independent_segment_dfts() {
    let x = [0.5, 2., -1., 0.25, 0.75, -0.5, 3., 1., -1.5];
    let window = [0.5, 1., 0.7, 0.25];
    for nfft in [4, 5, 8] {
        for detrend in [false, true] {
            for scaling in [Scaling::Density, Scaling::Spectrum] {
                let (frequencies, actual) =
                    welch(&x, &window, 2, nfft, 12., detrend, scaling).unwrap();
                let count = (x.len() - window.len()) / 2 + 1;
                let denominator = match scaling {
                    Scaling::Density => 12. * window.iter().map(|w| w * w).sum::<f64>(),
                    Scaling::Spectrum => window.iter().sum::<f64>().powi(2),
                };
                let mut expected = vec![0.; nfft / 2 + 1];
                for start in (0..=x.len() - window.len()).step_by(2) {
                    for (k, (re, im)) in
                        spectrum_segment(&x[start..start + window.len()], &window, nfft, detrend)
                            .into_iter()
                            .enumerate()
                    {
                        let doubling = if k == 0 || (nfft % 2 == 0 && k == nfft / 2) {
                            1.
                        } else {
                            2.
                        };
                        expected[k] += (re * re + im * im) * doubling / denominator / count as f64;
                    }
                }
                assert_eq!(expected.len(), actual.len());
                for (k, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
                    close(a, e, 1e-12);
                    close(frequencies[k], k as f64 * 12. / nfft as f64, 1e-14);
                }
            }
        }
    }
}

#[test]
fn coherence_reuses_dfts_without_changing_cross_or_auto_power() {
    let x = [0.5, 2., -1., 0.25, 0.75, -0.5, 3., 1., -1.5];
    let y = [1.5, -0.5, 0.3, 2., -1., 1.25, 0.2, 0.7, 2.5];
    let window = [0.5, 1., 0.7, 0.25];
    for nfft in [4, 5, 8] {
        let (_, actual) = coherence(&x, &y, &window, 2, nfft, 12.).unwrap();
        let mut cross = vec![(0., 0.); nfft / 2 + 1];
        let mut power_x = vec![0.; nfft / 2 + 1];
        let mut power_y = vec![0.; nfft / 2 + 1];
        for start in (0..=x.len() - window.len()).step_by(2) {
            let a = spectrum_segment(&x[start..start + window.len()], &window, nfft, true);
            let b = spectrum_segment(&y[start..start + window.len()], &window, nfft, true);
            for (k, (&(ar, ai), &(br, bi))) in a.iter().zip(&b).enumerate() {
                cross[k].0 += ar * br + ai * bi;
                cross[k].1 += ar * bi - ai * br;
                power_x[k] += ar * ar + ai * ai;
                power_y[k] += br * br + bi * bi;
            }
        }
        for (k, &a) in actual.iter().enumerate() {
            let (re, im) = cross[k];
            let expected = (re.hypot(im) / power_x[k].sqrt() / power_y[k].sqrt()).powi(2);
            close(a, expected, 1e-12);
        }
        // The single-signal path remains consistent with cross power.
        let (_, auto) = welch(&x, &window, 2, nfft, 12., true, Scaling::Density).unwrap();
        let (_, cross_auto) = csd(&x, &x, &window, 2, nfft, 12., true, Scaling::Density).unwrap();
        for (a, z) in auto.into_iter().zip(cross_auto) {
            close(a, z.re, 1e-12);
            close(z.im, 0., 1e-12);
        }
    }
    let (_, zero) = coherence(&[0.; 4], &[0.; 4], &[1.; 2], 1, 2, 1.).unwrap();
    assert!(zero.into_iter().all(f64::is_nan));
    let (_, silent_channel) = coherence(&x, &[0.; 9], &window, 2, 4, 12.).unwrap();
    assert!(silent_channel.into_iter().all(f64::is_nan));
    assert!(coherence(&[], &[], &[1.], 0, 1, 1.).is_err());
    assert!(coherence(&[1.], &[], &[1.], 0, 1, 1.).is_err());
    assert!(coherence(&[1.], &[f64::NAN], &[1.], 0, 1, 1.).is_err());
    assert!(coherence(&[f64::NAN], &[1.], &[1.], 0, 1, 1.).is_err());
}

#[test]
fn generalized_cosine_recurrence_preserves_harmonics_and_symmetry() {
    for m in [0, 1, 2, 5, 16, 129] {
        for symmetric in [true, false] {
            for harmonics in [1, 2, 3, 5, 8, 9, 16] {
                let coefficients: Vec<_> = (0..harmonics)
                    .map(|k| (k as f64 * 0.91).cos() / (k + 1) as f64)
                    .collect();
                let actual = general_cosine(m, &coefficients, symmetric).unwrap();
                assert_eq!(actual.len(), m);
                if m <= 1 {
                    assert_eq!(actual, vec![1.; m]);
                    continue;
                }
                let n = m + usize::from(!symmetric);
                for (i, &a) in actual.iter().enumerate() {
                    let angle = -std::f64::consts::PI + TAU * i as f64 / (n - 1) as f64;
                    let expected = coefficients
                        .iter()
                        .enumerate()
                        .map(|(k, &w)| w * (k as f64 * angle).cos())
                        .sum();
                    close(a, expected, 3e-13);
                }
            }
        }
    }
}

#[test]
fn sos_cascade_matches_individual_direct_form_sections() {
    let sections = [
        [0.3, 0.2, 0.1, 1., -0.5, 0.2],
        [0.8, -0.1, 0.05, 1., -0.2, 0.1],
        [1.0, 0.0, -0.1, 1., 0.3, 0.1],
    ];
    for n in [0, 1, 2, 3, 63, 1024] {
        let x: Vec<_> = (0..n)
            .map(|i| (0.11 * i as f64).sin() + (0.07 * i as f64).cos())
            .collect();
        let expected = sections.iter().fold(x.clone(), |signal, section| {
            lfilter(&section[..3], &section[3..], &signal).unwrap()
        });
        let actual = sosfilt(&sections, &x).unwrap();
        assert_eq!(expected.len(), actual.len());
        for (a, e) in actual.into_iter().zip(expected) {
            close(a, e, 1e-14);
        }
    }
    assert!(sosfilt(&[[1., 0., 0., 0., 0., 0.]], &[1.]).is_err());
    assert!(sosfilt(&sections, &[f64::INFINITY]).is_err());
}
