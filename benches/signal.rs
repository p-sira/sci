//! Benchmarks covering the public API of `sci::signal`.
//!
//! Inputs are deterministic and kept small so that each iteration stays cheap
//! under CodSpeed simulation.

mod common;

use common::*;
use divan::{Bencher, black_box};
use sci::complex::Complex;
use sci::signal::{
    self, BandType, BesselNorm, Boundary, ChirpMethod, Discretization, EdgeMode, IirFamily, Mode,
    RootRepresentative, Scaling, StateSpace, Zpk, windows, windows::Window,
};

fn main() {
    divan::main();
}

const N: usize = 1024;
const FS: f64 = 1000.;

fn complex_values(n: usize, seed: u64) -> Vec<Complex> {
    let re = values(n, seed);
    let im = values(n, seed + 1);
    re.into_iter()
        .zip(im)
        .map(|(re, im)| Complex { re, im })
        .collect()
}

fn image(n: usize, seed: u64) -> Vec<Vec<f64>> {
    points(n, n, seed)
}

/// Digital order-`n` Butterworth lowpass (normalized frequency, Nyquist = 1).
fn digital_butter(n: usize) -> (Vec<f64>, Vec<f64>) {
    signal::butter(n, &[0.2], BandType::Lowpass, Some(2.)).unwrap()
}

/// Analog order-`n` Butterworth lowpass with unit cutoff.
fn analog_butter(n: usize) -> (Vec<f64>, Vec<f64>) {
    signal::zpk2tf(&signal::buttap(n).unwrap()).unwrap()
}

fn digital_zpk(n: usize) -> Zpk {
    signal::bilinear_zpk(
        &signal::lp2lp_zpk(&signal::buttap(n).unwrap(), 2.).unwrap(),
        10.,
    )
    .unwrap()
}

fn digital_sos(n: usize) -> Vec<[f64; 6]> {
    signal::zpk2sos(&digital_zpk(n)).unwrap()
}

/// Grid of digital frequencies in `[0, pi]`.
fn digital_frequencies(n: usize) -> Vec<f64> {
    linspace(0., std::f64::consts::PI, n)
}

/// Logarithmic grid of analog frequencies in `[0.01, 100]`.
fn analog_frequencies(n: usize) -> Vec<f64> {
    linspace(-2., 2., n)
        .into_iter()
        .map(|e| 10f64.powf(e))
        .collect()
}

mod convolution_benches {
    use super::*;

    #[divan::bench]
    fn convolve(bencher: Bencher) {
        let x = sine_signal(N);
        let h = values(32, 3);
        bencher.bench(|| signal::convolve(black_box(&x), black_box(&h), Mode::Full).unwrap());
    }

    #[divan::bench(args = [32, 1024])]
    fn convolve_valid(bencher: Bencher, kernel_len: usize) {
        let x = sine_signal(N);
        let h = values(kernel_len, 3);
        bencher.bench(|| signal::convolve(black_box(&x), black_box(&h), Mode::Valid).unwrap());
    }

    #[divan::bench]
    fn correlate(bencher: Bencher) {
        let x = sine_signal(N);
        let y = values(32, 3);
        bencher.bench(|| signal::correlate(black_box(&x), black_box(&y), Mode::Same).unwrap());
    }

    #[divan::bench]
    fn correlation_lags(bencher: Bencher) {
        bencher
            .bench(|| signal::correlation_lags(black_box(N), black_box(256), Mode::Full).unwrap());
    }

    #[divan::bench]
    fn fftconvolve(bencher: Bencher) {
        let x = sine_signal(N);
        let h = values(128, 3);
        bencher.bench(|| signal::fftconvolve(black_box(&x), black_box(&h), Mode::Full).unwrap());
    }

    #[divan::bench]
    fn oaconvolve(bencher: Bencher) {
        let x = sine_signal(N);
        let h = values(64, 3);
        bencher.bench(|| signal::oaconvolve(black_box(&x), black_box(&h), Mode::Full).unwrap());
    }

    #[divan::bench]
    fn deconvolve(bencher: Bencher) {
        let h: Vec<f64> = (0..8).map(|i| 0.5f64.powi(i)).collect();
        let x = signal::convolve(&values(256, 5), &h, Mode::Full).unwrap();
        bencher.bench(|| signal::deconvolve(black_box(&x), black_box(&h)).unwrap());
    }

    #[divan::bench]
    fn upfirdn(bencher: Bencher) {
        let h = signal::firwin(31, &[0.25], true, &Window::Hamming, true, 2.).unwrap();
        let x = sine_signal(N);
        bencher.bench(|| signal::upfirdn(black_box(&h), black_box(&x), 3, 2).unwrap());
    }

    #[divan::bench]
    fn resample_poly(bencher: Bencher) {
        let x = sine_signal(N);
        bencher.bench(|| signal::resample_poly(black_box(&x), 3, 2, None).unwrap());
    }

    #[divan::bench]
    fn decimate(bencher: Bencher) {
        let x = sine_signal(N);
        bencher.bench(|| signal::decimate(black_box(&x), 4, 8, true).unwrap());
    }
}

mod prototype_benches {
    use super::*;

    #[divan::bench]
    fn buttap(bencher: Bencher) {
        bencher.bench(|| signal::buttap(black_box(8)).unwrap());
    }

    #[divan::bench]
    fn cheb1ap(bencher: Bencher) {
        bencher.bench(|| signal::cheb1ap(black_box(8), black_box(1.)).unwrap());
    }

    #[divan::bench]
    fn cheb2ap(bencher: Bencher) {
        bencher.bench(|| signal::cheb2ap(black_box(8), black_box(40.)).unwrap());
    }

    #[divan::bench]
    fn besselap(bencher: Bencher) {
        bencher.bench(|| signal::besselap(black_box(8), BesselNorm::Phase).unwrap());
    }

    #[divan::bench]
    fn ellipap(bencher: Bencher) {
        bencher.bench(|| signal::ellipap(black_box(8), black_box(1.), black_box(40.)).unwrap());
    }

    #[divan::bench]
    fn lp2lp_zpk(bencher: Bencher) {
        let zpk = signal::buttap(8).unwrap();
        bencher.bench(|| signal::lp2lp_zpk(black_box(&zpk), black_box(2.)).unwrap());
    }

    #[divan::bench]
    fn lp2hp_zpk(bencher: Bencher) {
        let zpk = signal::buttap(8).unwrap();
        bencher.bench(|| signal::lp2hp_zpk(black_box(&zpk), black_box(2.)).unwrap());
    }

    #[divan::bench]
    fn lp2bp_zpk(bencher: Bencher) {
        let zpk = signal::buttap(8).unwrap();
        bencher.bench(|| signal::lp2bp_zpk(black_box(&zpk), black_box(2.), black_box(1.)).unwrap());
    }

    #[divan::bench]
    fn lp2bs_zpk(bencher: Bencher) {
        let zpk = signal::buttap(8).unwrap();
        bencher.bench(|| signal::lp2bs_zpk(black_box(&zpk), black_box(2.), black_box(1.)).unwrap());
    }

    #[divan::bench]
    fn bilinear_zpk(bencher: Bencher) {
        let zpk = signal::buttap(8).unwrap();
        bencher.bench(|| signal::bilinear_zpk(black_box(&zpk), black_box(10.)).unwrap());
    }

    #[divan::bench]
    fn lp2lp(bencher: Bencher) {
        let (b, a) = analog_butter(4);
        warm(bencher, || {
            signal::lp2lp(black_box(&b), black_box(&a), black_box(2.)).unwrap()
        });
    }

    #[divan::bench]
    fn lp2hp(bencher: Bencher) {
        let (b, a) = analog_butter(4);
        warm(bencher, || {
            signal::lp2hp(black_box(&b), black_box(&a), black_box(2.)).unwrap()
        });
    }

    #[divan::bench]
    fn lp2bp(bencher: Bencher) {
        let (b, a) = analog_butter(4);
        warm(bencher, || {
            signal::lp2bp(black_box(&b), black_box(&a), black_box(2.), black_box(1.)).unwrap()
        });
    }

    #[divan::bench]
    fn lp2bs(bencher: Bencher) {
        let (b, a) = analog_butter(4);
        warm(bencher, || {
            signal::lp2bs(black_box(&b), black_box(&a), black_box(2.), black_box(1.)).unwrap()
        });
    }

    #[divan::bench]
    fn bilinear(bencher: Bencher) {
        let (b, a) = analog_butter(4);
        warm(bencher, || {
            signal::bilinear(black_box(&b), black_box(&a), black_box(10.)).unwrap()
        });
    }
}

mod iir_design_benches {
    use super::*;

    #[divan::bench]
    fn butter(bencher: Bencher) {
        bencher.bench(|| {
            signal::butter(black_box(8), black_box(&[0.2]), BandType::Lowpass, Some(2.)).unwrap()
        });
    }

    #[divan::bench]
    fn butter_bandpass(bencher: Bencher) {
        bencher.bench(|| {
            signal::butter(
                black_box(4),
                black_box(&[0.2, 0.4]),
                BandType::Bandpass,
                Some(2.),
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn cheby1(bencher: Bencher) {
        bencher.bench(|| {
            signal::cheby1(
                black_box(8),
                black_box(1.),
                black_box(&[0.2]),
                BandType::Lowpass,
                Some(2.),
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn cheby2(bencher: Bencher) {
        bencher.bench(|| {
            signal::cheby2(
                black_box(8),
                black_box(40.),
                black_box(&[0.2]),
                BandType::Lowpass,
                Some(2.),
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn bessel(bencher: Bencher) {
        bencher.bench(|| {
            signal::bessel(
                black_box(8),
                black_box(&[0.2]),
                BandType::Lowpass,
                Some(2.),
                BesselNorm::Phase,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn ellip(bencher: Bencher) {
        bencher.bench(|| {
            signal::ellip(
                black_box(6),
                black_box(1.),
                black_box(40.),
                black_box(&[0.2]),
                BandType::Lowpass,
                Some(2.),
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn iirfilter(bencher: Bencher) {
        bencher.bench(|| {
            signal::iirfilter(
                black_box(6),
                black_box(&[0.2, 0.4]),
                BandType::Bandstop,
                Some(2.),
                IirFamily::ChebyshevI(1.),
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn iirdesign(bencher: Bencher) {
        bencher.bench(|| {
            signal::iirdesign(
                black_box(0.2),
                black_box(0.3),
                black_box(1.),
                black_box(40.),
                Some(2.),
                IirFamily::Elliptic(1., 40.),
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn buttord(bencher: Bencher) {
        bencher
            .bench(|| signal::buttord(black_box(0.2), black_box(0.3), 1., 40., Some(2.)).unwrap());
    }

    #[divan::bench]
    fn cheb1ord(bencher: Bencher) {
        bencher
            .bench(|| signal::cheb1ord(black_box(0.2), black_box(0.3), 1., 40., Some(2.)).unwrap());
    }

    #[divan::bench]
    fn cheb2ord(bencher: Bencher) {
        bencher
            .bench(|| signal::cheb2ord(black_box(0.2), black_box(0.3), 1., 40., Some(2.)).unwrap());
    }

    #[divan::bench]
    fn ellipord(bencher: Bencher) {
        bencher
            .bench(|| signal::ellipord(black_box(0.2), black_box(0.3), 1., 40., Some(2.)).unwrap());
    }

    #[divan::bench]
    fn band_stop_obj(bencher: Bencher) {
        bencher.bench(|| {
            signal::band_stop_obj(
                black_box(1.),
                0,
                black_box([0.8, 4.]),
                black_box([2., 3.]),
                1.,
                40.,
                IirFamily::Butterworth,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn iirnotch(bencher: Bencher) {
        bencher.bench(|| signal::iirnotch(black_box(50.), black_box(30.), FS).unwrap());
    }

    #[divan::bench]
    fn iirpeak(bencher: Bencher) {
        bencher.bench(|| signal::iirpeak(black_box(50.), black_box(30.), FS).unwrap());
    }

    #[divan::bench]
    fn iircomb(bencher: Bencher) {
        bencher
            .bench(|| signal::iircomb(black_box(20.), black_box(30.), false, 200., false).unwrap());
    }

    #[divan::bench]
    fn gammatone_fir(bencher: Bencher) {
        bencher.bench(|| signal::gammatone(black_box(440.), true, 4, 128, 16000.).unwrap());
    }

    #[divan::bench]
    fn gammatone_iir(bencher: Bencher) {
        bencher.bench(|| signal::gammatone(black_box(440.), false, 4, 0, 16000.).unwrap());
    }

    #[divan::bench]
    fn findfreqs(bencher: Bencher) {
        let (b, a) = analog_butter(4);
        warm(bencher, || {
            signal::findfreqs(black_box(&b), black_box(&a), 256).unwrap()
        });
    }
}

mod representation_benches {
    use super::*;

    #[divan::bench]
    fn zpk2tf(bencher: Bencher) {
        let zpk = digital_zpk(8);
        bencher.bench(|| signal::zpk2tf(black_box(&zpk)).unwrap());
    }

    #[divan::bench]
    fn tf2zpk(bencher: Bencher) {
        let (b, a) = digital_butter(8);
        warm(bencher, || {
            signal::tf2zpk(black_box(&b), black_box(&a)).unwrap()
        });
    }

    #[divan::bench]
    fn normalize(bencher: Bencher) {
        let (b, a) = digital_butter(8);
        let b: Vec<f64> = std::iter::once(0.)
            .chain(b.iter().map(|v| 3. * v))
            .collect();
        let a: Vec<f64> = a.iter().map(|v| 3. * v).collect();
        bencher.bench(|| signal::normalize(black_box(&b), black_box(&a)).unwrap());
    }

    #[divan::bench]
    fn zpk2sos(bencher: Bencher) {
        let zpk = digital_zpk(8);
        bencher.bench(|| signal::zpk2sos(black_box(&zpk)).unwrap());
    }

    #[divan::bench]
    fn tf2sos(bencher: Bencher) {
        let (b, a) = digital_butter(8);
        warm(bencher, || {
            signal::tf2sos(black_box(&b), black_box(&a)).unwrap()
        });
    }

    #[divan::bench]
    fn sos2zpk(bencher: Bencher) {
        let sos = digital_sos(8);
        bencher.bench(|| signal::sos2zpk(black_box(&sos)).unwrap());
    }

    #[divan::bench]
    fn sos2tf(bencher: Bencher) {
        let sos = digital_sos(8);
        bencher.bench(|| signal::sos2tf(black_box(&sos)).unwrap());
    }

    #[divan::bench]
    fn residue(bencher: Bencher) {
        let (b, a) = analog_butter(4);
        warm(bencher, || {
            signal::residue(black_box(&b), black_box(&a), 1e-3).unwrap()
        });
    }

    #[divan::bench]
    fn residuez(bencher: Bencher) {
        let (b, a) = digital_butter(4);
        warm(bencher, || {
            signal::residuez(black_box(&b), black_box(&a), 1e-3).unwrap()
        });
    }

    #[divan::bench]
    fn invres(bencher: Bencher) {
        let (b, a) = analog_butter(4);
        let p = signal::residue(&b, &a, 1e-3).unwrap();
        warm(bencher, || {
            signal::invres(
                black_box(&p.residues),
                black_box(&p.poles),
                black_box(&p.direct),
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn invresz(bencher: Bencher) {
        let (b, a) = digital_butter(4);
        let p = signal::residuez(&b, &a, 1e-3).unwrap();
        warm(bencher, || {
            signal::invresz(
                black_box(&p.residues),
                black_box(&p.poles),
                black_box(&p.direct),
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn unique_roots(bencher: Bencher) {
        let mut roots = signal::buttap(8).unwrap().poles;
        roots.extend(roots.clone());
        bencher.bench(|| {
            signal::unique_roots(black_box(&roots), 1e-3, RootRepresentative::Mean).unwrap()
        });
    }
}

mod response_benches {
    use super::*;

    #[divan::bench]
    fn freqz(bencher: Bencher) {
        let (b, a) = digital_butter(8);
        let w = digital_frequencies(512);
        bencher.bench(|| signal::freqz(black_box(&b), black_box(&a), black_box(&w)).unwrap());
    }

    #[divan::bench]
    fn freqs(bencher: Bencher) {
        let (b, a) = analog_butter(8);
        let w = analog_frequencies(512);
        bencher.bench(|| signal::freqs(black_box(&b), black_box(&a), black_box(&w)).unwrap());
    }

    #[divan::bench]
    fn freqz_sos(bencher: Bencher) {
        let sos = digital_sos(8);
        let w = digital_frequencies(512);
        bencher.bench(|| signal::freqz_sos(black_box(&sos), black_box(&w)).unwrap());
    }

    #[divan::bench]
    fn freqs_zpk(bencher: Bencher) {
        let zpk = signal::buttap(8).unwrap();
        let w = analog_frequencies(512);
        bencher.bench(|| signal::freqs_zpk(black_box(&zpk), black_box(&w)).unwrap());
    }

    #[divan::bench]
    fn freqz_zpk(bencher: Bencher) {
        let zpk = digital_zpk(8);
        let w = digital_frequencies(512);
        bencher.bench(|| signal::freqz_zpk(black_box(&zpk), black_box(&w)).unwrap());
    }

    #[divan::bench]
    fn group_delay(bencher: Bencher) {
        let (b, a) = digital_butter(4);
        // Stay away from the zeros at z = -1, where the group delay is undefined.
        let w = linspace(0., 3., 512);
        bencher.bench(|| signal::group_delay(black_box(&b), black_box(&a), black_box(&w)).unwrap());
    }

    #[divan::bench]
    fn bode(bencher: Bencher) {
        let (b, a) = analog_butter(8);
        let w = analog_frequencies(512);
        bencher.bench(|| signal::bode(black_box(&b), black_box(&a), black_box(&w)).unwrap());
    }

    #[divan::bench]
    fn dbode(bencher: Bencher) {
        let (b, a) = digital_butter(8);
        let w = digital_frequencies(512);
        bencher.bench(|| signal::dbode(black_box(&b), black_box(&a), black_box(&w)).unwrap());
    }
}

mod filtering_benches {
    use super::*;

    #[divan::bench]
    fn lfilter(bencher: Bencher) {
        let (b, a) = digital_butter(8);
        let x = sine_signal(N);
        bencher.bench(|| signal::lfilter(black_box(&b), black_box(&a), black_box(&x)).unwrap());
    }

    #[divan::bench]
    fn lfilter_with_state(bencher: Bencher) {
        let (b, a) = digital_butter(8);
        let zi = signal::lfilter_zi(&b, &a).unwrap();
        let x = sine_signal(N);
        bencher.bench(|| {
            signal::lfilter_with_state(black_box(&b), black_box(&a), black_box(&x), black_box(&zi))
                .unwrap()
        });
    }

    #[divan::bench]
    fn lfilter_zi(bencher: Bencher) {
        let (b, a) = digital_butter(8);
        warm(bencher, || {
            signal::lfilter_zi(black_box(&b), black_box(&a)).unwrap()
        });
    }

    #[divan::bench]
    fn lfiltic(bencher: Bencher) {
        let (b, a) = digital_butter(8);
        let past_y = values(8, 3);
        let past_x = values(8, 4);
        bencher.bench(|| {
            signal::lfiltic(
                black_box(&b),
                black_box(&a),
                black_box(&past_y),
                black_box(&past_x),
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn filtfilt(bencher: Bencher) {
        let (b, a) = digital_butter(4);
        let x = sine_signal(N);
        warm(bencher, || {
            signal::filtfilt(black_box(&b), black_box(&a), black_box(&x), 15).unwrap()
        });
    }

    #[divan::bench]
    fn sosfilt(bencher: Bencher) {
        let sos = digital_sos(8);
        let x = sine_signal(N);
        bencher.bench(|| signal::sosfilt(black_box(&sos), black_box(&x)).unwrap());
    }

    #[divan::bench]
    fn sosfilt_zi(bencher: Bencher) {
        let sos = digital_sos(8);
        warm(bencher, || signal::sosfilt_zi(black_box(&sos)).unwrap());
    }

    #[divan::bench]
    fn sosfiltfilt(bencher: Bencher) {
        let sos = digital_sos(8);
        let x = sine_signal(N);
        warm(bencher, || {
            signal::sosfiltfilt(black_box(&sos), black_box(&x), 27).unwrap()
        });
    }

    #[divan::bench]
    fn medfilt(bencher: Bencher) {
        let x = sine_signal(N);
        bencher.bench(|| signal::medfilt(black_box(&x), 5).unwrap());
    }

    #[divan::bench]
    fn order_filter(bencher: Bencher) {
        let x = sine_signal(N);
        bencher.bench(|| signal::order_filter(black_box(&x), 5, 1).unwrap());
    }

    #[divan::bench]
    fn wiener(bencher: Bencher) {
        let x = sine_signal(N);
        bencher.bench(|| signal::wiener(black_box(&x), 5, None).unwrap());
    }

    #[divan::bench]
    fn savgol_coeffs(bencher: Bencher) {
        warm(bencher, || {
            signal::savgol_coeffs(black_box(11), black_box(3), 0, 1.).unwrap()
        });
    }

    #[divan::bench]
    fn savgol_filter(bencher: Bencher) {
        let x = sine_signal(N);
        warm(bencher, || {
            signal::savgol_filter(black_box(&x), 11, 3, 0, 1.).unwrap()
        });
    }

    #[divan::bench]
    fn detrend(bencher: Bencher) {
        let x = sine_signal(N);
        bencher.bench(|| signal::detrend(black_box(&x), true).unwrap());
    }

    #[divan::bench]
    fn symiirorder1(bencher: Bencher) {
        let x = sine_signal(N);
        bencher.bench(|| signal::symiirorder1(black_box(&x), 1., 0.5, 1e-8).unwrap());
    }

    #[divan::bench]
    fn symiirorder2(bencher: Bencher) {
        let x = sine_signal(N);
        bencher.bench(|| signal::symiirorder2(black_box(&x), 0.5, 0.6, 1e-8).unwrap());
    }
}

mod fir_benches {
    use super::*;

    #[divan::bench]
    fn kaiser_beta(bencher: Bencher) {
        bencher.bench(|| signal::kaiser_beta(black_box(60.)));
    }

    #[divan::bench]
    fn kaiser_atten(bencher: Bencher) {
        bencher.bench(|| signal::kaiser_atten(black_box(31), black_box(0.1)).unwrap());
    }

    #[divan::bench]
    fn kaiserord(bencher: Bencher) {
        bencher.bench(|| signal::kaiserord(black_box(60.), black_box(0.1)).unwrap());
    }

    #[divan::bench]
    fn firwin(bencher: Bencher) {
        let window = Window::Kaiser(8.);
        bencher.bench(|| {
            signal::firwin(
                black_box(63),
                black_box(&[0.2, 0.4]),
                false,
                &window,
                true,
                2.,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn firwin2(bencher: Bencher) {
        let freq = [0., 0.4, 0.6, 1.];
        let gain = [1., 1., 0., 0.];
        bencher.bench(|| {
            signal::firwin2(
                black_box(63),
                black_box(&freq),
                black_box(&gain),
                129,
                &Window::Hamming,
                false,
                2.,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn firwin_2d(bencher: Bencher) {
        bencher.bench(|| {
            signal::firwin_2d(
                black_box((15, 15)),
                black_box(&[0.3]),
                true,
                &Window::Hamming,
                &Window::Hann,
                true,
                2.,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn firls(bencher: Bencher) {
        let bands = [0., 0.3, 0.5, 1.];
        let desired = [1., 1., 0., 0.];
        let weight = [1., 1.];
        warm(bencher, || {
            signal::firls(
                black_box(31),
                black_box(&bands),
                black_box(&desired),
                black_box(&weight),
                2.,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn remez(bencher: Bencher) {
        let bands = [0., 0.3, 0.5, 1.];
        let desired = [1., 0.];
        let weight = [1., 1.];
        warm(bencher, || {
            signal::remez(
                black_box(7),
                black_box(&bands),
                black_box(&desired),
                black_box(&weight),
                2.,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn minimum_phase(bencher: Bencher) {
        let h = signal::firwin(31, &[0.3], true, &Window::Hamming, true, 2.).unwrap();
        bencher.bench(|| signal::minimum_phase(black_box(&h), 256, true).unwrap());
    }
}

mod multidimensional_benches {
    use super::*;

    #[divan::bench]
    fn convolve2d(bencher: Bencher) {
        let x = image(32, 1);
        let h = image(5, 2);
        bencher.bench(|| {
            signal::convolve2d(black_box(&x), black_box(&h), Mode::Same, Boundary::Fill(0.))
                .unwrap()
        });
    }

    #[divan::bench]
    fn correlate2d(bencher: Bencher) {
        let x = image(32, 1);
        let h = image(5, 2);
        bencher.bench(|| {
            signal::correlate2d(
                black_box(&x),
                black_box(&h),
                Mode::Same,
                Boundary::Symmetric,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn sepfir2d(bencher: Bencher) {
        let x = image(32, 1);
        let row = [0.0625, 0.25, 0.375, 0.25, 0.0625];
        let column = [0.25, 0.5, 0.25];
        bencher.bench(|| {
            signal::sepfir2d(black_box(&x), black_box(&row), black_box(&column)).unwrap()
        });
    }

    #[divan::bench]
    fn medfilt2d(bencher: Bencher) {
        let x = image(32, 1);
        bencher.bench(|| signal::medfilt2d(black_box(&x), [3, 3]).unwrap());
    }

    #[divan::bench]
    fn hilbert2(bencher: Bencher) {
        let x = image(32, 1);
        bencher.bench(|| signal::hilbert2(black_box(&x)).unwrap());
    }
}

mod spline_benches {
    use super::*;

    #[divan::bench]
    fn cspline1d(bencher: Bencher) {
        let x = sine_signal(N);
        bencher.bench(|| signal::cspline1d(black_box(&x), 0.).unwrap());
    }

    #[divan::bench]
    fn cspline1d_smoothing(bencher: Bencher) {
        let x = sine_signal(256);
        bencher.bench(|| signal::cspline1d(black_box(&x), 0.5).unwrap());
    }

    #[divan::bench]
    fn qspline1d(bencher: Bencher) {
        let x = sine_signal(N);
        bencher.bench(|| signal::qspline1d(black_box(&x)).unwrap());
    }

    #[divan::bench]
    fn cspline1d_eval(bencher: Bencher) {
        let c = signal::cspline1d(&sine_signal(128), 0.).unwrap();
        let x = linspace(0., 127., N);
        bencher.bench(|| signal::cspline1d_eval(black_box(&c), black_box(&x), 1., 0.).unwrap());
    }

    #[divan::bench]
    fn qspline1d_eval(bencher: Bencher) {
        let c = signal::qspline1d(&sine_signal(128)).unwrap();
        let x = linspace(0., 127., N);
        bencher.bench(|| signal::qspline1d_eval(black_box(&c), black_box(&x), 1., 0.).unwrap());
    }

    #[divan::bench]
    fn cspline2d(bencher: Bencher) {
        let x = image(32, 1);
        bencher.bench(|| signal::cspline2d(black_box(&x), 0., 1e-6).unwrap());
    }

    #[divan::bench]
    fn qspline2d(bencher: Bencher) {
        let x = image(32, 1);
        bencher.bench(|| signal::qspline2d(black_box(&x), 1e-6).unwrap());
    }

    #[divan::bench]
    fn spline_filter(bencher: Bencher) {
        let x = image(32, 1);
        bencher.bench(|| signal::spline_filter(black_box(&x), 0.1).unwrap());
    }

    #[divan::bench]
    fn whittaker_henderson(bencher: Bencher) {
        let y = sine_signal(128);
        let weights = vec![1.; 128];
        warm(bencher, || {
            signal::whittaker_henderson(black_box(&y), 10., 2, black_box(&weights)).unwrap()
        });
    }
}

mod peak_benches {
    use super::*;

    #[divan::bench]
    fn argrelextrema(bencher: Bencher) {
        let x = sine_signal(N);
        bencher.bench(|| {
            signal::argrelextrema(black_box(&x), |a, b| a > b, 3, EdgeMode::Clip).unwrap()
        });
    }

    #[divan::bench]
    fn argrelmax(bencher: Bencher) {
        let x = sine_signal(N);
        bencher.bench(|| signal::argrelmax(black_box(&x), 3, EdgeMode::Clip).unwrap());
    }

    #[divan::bench]
    fn argrelmin(bencher: Bencher) {
        let x = sine_signal(N);
        bencher.bench(|| signal::argrelmin(black_box(&x), 3, EdgeMode::Wrap).unwrap());
    }

    #[divan::bench]
    fn find_peaks(bencher: Bencher) {
        let x = sine_signal(N);
        bencher.bench(|| signal::find_peaks(black_box(&x)).unwrap());
    }

    #[divan::bench]
    fn peak_prominences(bencher: Bencher) {
        let x = sine_signal(N);
        let peaks = signal::find_peaks(&x).unwrap();
        bencher.bench(|| signal::peak_prominences(black_box(&x), black_box(&peaks), None).unwrap());
    }

    #[divan::bench]
    fn peak_widths(bencher: Bencher) {
        let x = sine_signal(N);
        let peaks = signal::find_peaks(&x).unwrap();
        bencher.bench(|| signal::peak_widths(black_box(&x), black_box(&peaks), 0.5, None).unwrap());
    }

    #[divan::bench]
    fn find_peaks_cwt(bencher: Bencher) {
        let x = sine_signal(256);
        let widths = [1., 2., 3., 4., 6., 8.];
        warm(bencher, || {
            signal::find_peaks_cwt(black_box(&x), black_box(&widths)).unwrap()
        });
    }
}

mod spectral_benches {
    use super::*;

    #[divan::bench]
    fn hilbert(bencher: Bencher) {
        let x = sine_signal(N);
        warm(bencher, || signal::hilbert(black_box(&x)).unwrap());
    }

    #[divan::bench]
    fn envelope(bencher: Bencher) {
        let x = sine_signal(N);
        warm(bencher, || {
            signal::envelope(black_box(&x), (0, 200), 512, false, true).unwrap()
        });
    }

    #[divan::bench]
    fn resample(bencher: Bencher) {
        let x = sine_signal(N);
        warm(bencher, || signal::resample(black_box(&x), 768).unwrap());
    }

    #[divan::bench]
    fn czt_points(bencher: Bencher) {
        let a = Complex { re: 1., im: 0. };
        bencher.bench(|| signal::czt_points(black_box(256), None, black_box(a)).unwrap());
    }

    #[divan::bench]
    fn czt(bencher: Bencher) {
        let x = complex_values(128, 1);
        let a = Complex { re: 1., im: 0. };
        warm(bencher, || {
            signal::czt(black_box(&x), 128, None, black_box(a)).unwrap()
        });
    }

    #[divan::bench]
    fn zoom_fft(bencher: Bencher) {
        let x = complex_values(128, 1);
        warm(bencher, || {
            signal::zoom_fft(black_box(&x), 40., 140., 128, FS, false).unwrap()
        });
    }

    #[divan::bench]
    fn periodogram(bencher: Bencher) {
        let x = sine_signal(N);
        let window = windows::hann(N, false).unwrap();
        warm(bencher, || {
            signal::periodogram(
                black_box(&x),
                black_box(&window),
                N,
                FS,
                true,
                Scaling::Density,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn welch(bencher: Bencher) {
        let x = sine_signal(N);
        let window = windows::hann(256, false).unwrap();
        warm(bencher, || {
            signal::welch(
                black_box(&x),
                black_box(&window),
                128,
                256,
                FS,
                true,
                Scaling::Density,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn csd(bencher: Bencher) {
        let x = sine_signal(N);
        let y = values(N, 5);
        let window = windows::hann(256, false).unwrap();
        warm(bencher, || {
            signal::csd(
                black_box(&x),
                black_box(&y),
                black_box(&window),
                128,
                256,
                FS,
                true,
                Scaling::Spectrum,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn coherence(bencher: Bencher) {
        let x = sine_signal(N);
        let y = values(N, 5);
        let window = windows::hann(256, false).unwrap();
        warm(bencher, || {
            signal::coherence(
                black_box(&x),
                black_box(&y),
                black_box(&window),
                128,
                256,
                FS,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn spectrogram(bencher: Bencher) {
        let x = sine_signal(N);
        let window = windows::hann(128, false).unwrap();
        warm(bencher, || {
            signal::spectrogram(
                black_box(&x),
                black_box(&window),
                64,
                128,
                FS,
                true,
                Scaling::Density,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn stft(bencher: Bencher) {
        let x = sine_signal(N);
        let window = windows::hann(128, false).unwrap();
        warm(bencher, || {
            signal::stft(black_box(&x), black_box(&window), 64, 128, FS).unwrap()
        });
    }

    #[divan::bench]
    fn istft(bencher: Bencher) {
        let window = windows::hann(128, false).unwrap();
        let (_, _, spectrum) = signal::stft(&sine_signal(N), &window, 64, 128, FS).unwrap();
        warm(bencher, || {
            signal::istft(black_box(&spectrum), black_box(&window), 64, 128, FS).unwrap()
        });
    }

    #[divan::bench]
    fn check_cola(bencher: Bencher) {
        let window = windows::hann(256, false).unwrap();
        bencher.bench(|| signal::check_cola(black_box(&window), 128, 1e-10).unwrap());
    }

    #[divan::bench]
    fn check_nola(bencher: Bencher) {
        let window = windows::hann(256, false).unwrap();
        bencher.bench(|| signal::check_nola(black_box(&window), 128, 1e-10).unwrap());
    }

    #[divan::bench]
    fn closest_stft_dual_window(bencher: Bencher) {
        let window = windows::hann(256, false).unwrap();
        let desired = windows::hamming(256, false).unwrap();
        bencher.bench(|| {
            signal::closest_stft_dual_window(black_box(&window), 64, black_box(&desired), true)
                .unwrap()
        });
    }

    #[divan::bench]
    fn lombscargle(bencher: Bencher) {
        let times: Vec<f64> = unit_values(128, 3)
            .iter()
            .enumerate()
            .map(|(i, u)| (i as f64 + u) / FS)
            .collect();
        let x: Vec<f64> = times
            .iter()
            .map(|t| (2. * std::f64::consts::PI * 50. * t).sin())
            .collect();
        let freqs = linspace(10., 2000., 32);
        bencher.bench(|| {
            signal::lombscargle(
                black_box(&times),
                black_box(&x),
                black_box(&freqs),
                true,
                false,
            )
            .unwrap()
        });
    }
}

mod system_benches {
    use super::*;

    fn continuous() -> StateSpace {
        let (b, a) = analog_butter(4);
        signal::tf2ss(&b, &a).unwrap()
    }

    fn discrete() -> StateSpace {
        let (b, a) = digital_butter(4);
        signal::tf2ss(&b, &a).unwrap()
    }

    #[divan::bench]
    fn abcd_normalize(bencher: Bencher) {
        let s = continuous();
        bencher.bench(|| {
            signal::abcd_normalize(
                black_box(&s.a),
                black_box(&s.b),
                black_box(&s.c),
                black_box(&s.d),
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn tf2ss(bencher: Bencher) {
        let (b, a) = analog_butter(8);
        bencher.bench(|| signal::tf2ss(black_box(&b), black_box(&a)).unwrap());
    }

    #[divan::bench]
    fn zpk2ss(bencher: Bencher) {
        let zpk = signal::buttap(8).unwrap();
        bencher.bench(|| signal::zpk2ss(black_box(&zpk)).unwrap());
    }

    #[divan::bench]
    fn ss2tf(bencher: Bencher) {
        let s = continuous();
        warm(bencher, || signal::ss2tf(black_box(&s), 0).unwrap());
    }

    #[divan::bench]
    fn ss2zpk(bencher: Bencher) {
        let s = continuous();
        warm(bencher, || signal::ss2zpk(black_box(&s), 0, 0).unwrap());
    }

    #[divan::bench]
    fn cont2discrete(bencher: Bencher) {
        let s = continuous();
        warm(bencher, || {
            signal::cont2discrete(black_box(&s), 0.1, Discretization::ZeroOrderHold).unwrap()
        });
    }

    #[divan::bench]
    fn dlsim(bencher: Bencher) {
        let s = discrete();
        let u: Vec<Vec<f64>> = sine_signal(256).into_iter().map(|v| vec![v]).collect();
        let initial = [0.; 4];
        bencher.bench(|| {
            signal::dlsim(black_box(&s), black_box(&u), 1., black_box(&initial)).unwrap()
        });
    }

    #[divan::bench]
    fn lsim(bencher: Bencher) {
        let s = continuous();
        let t = linspace(0., 25.5, 256);
        let u: Vec<Vec<f64>> = sine_signal(256).into_iter().map(|v| vec![v]).collect();
        let initial = [0.; 4];
        warm(bencher, || {
            signal::lsim(
                black_box(&s),
                black_box(&u),
                black_box(&t),
                black_box(&initial),
                true,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn step(bencher: Bencher) {
        let s = continuous();
        let t = linspace(0., 25.5, 256);
        warm(bencher, || {
            signal::step(black_box(&s), black_box(&t), 0).unwrap()
        });
    }

    #[divan::bench]
    fn impulse(bencher: Bencher) {
        let s = continuous();
        let t = linspace(0., 25.5, 256);
        warm(bencher, || {
            signal::impulse(black_box(&s), black_box(&t), 0).unwrap()
        });
    }

    #[divan::bench]
    fn dstep(bencher: Bencher) {
        let s = discrete();
        bencher.bench(|| signal::dstep(black_box(&s), 1., 256, 0).unwrap());
    }

    #[divan::bench]
    fn dimpulse(bencher: Bencher) {
        let s = discrete();
        bencher.bench(|| signal::dimpulse(black_box(&s), 1., 256, 0).unwrap());
    }

    #[divan::bench]
    fn place_poles(bencher: Bencher) {
        let a = vec![
            vec![0., 1., 0., 0.],
            vec![0., 0., 1., 0.],
            vec![0., 0., 0., 1.],
            vec![-1., -2., -3., -4.],
        ];
        let b = [0., 0., 0., 1.];
        let poles: Vec<Complex> = (1..=4)
            .map(|k| Complex {
                re: -(k as f64),
                im: 0.,
            })
            .collect();
        warm(bencher, || {
            signal::place_poles(black_box(&a), black_box(&b), black_box(&poles)).unwrap()
        });
    }
}

mod waveform_benches {
    use super::*;

    fn times() -> Vec<f64> {
        linspace(0., 1., N)
    }

    #[divan::bench]
    fn sawtooth(bencher: Bencher) {
        let t = times();
        bencher.bench(|| {
            black_box(&t)
                .iter()
                .map(|&t| signal::sawtooth(20. * t, 0.5))
                .collect::<Vec<_>>()
        });
    }

    #[divan::bench]
    fn square(bencher: Bencher) {
        let t = times();
        bencher.bench(|| {
            black_box(&t)
                .iter()
                .map(|&t| signal::square(20. * t, 0.3))
                .collect::<Vec<_>>()
        });
    }

    #[divan::bench]
    fn chirp(bencher: Bencher) {
        let t = times();
        bencher.bench(|| {
            black_box(&t)
                .iter()
                .map(|&t| signal::chirp(t, 5., 1., 50., ChirpMethod::Logarithmic, 0.).unwrap())
                .collect::<Vec<_>>()
        });
    }

    #[divan::bench]
    fn sweep_poly(bencher: Bencher) {
        let t = times();
        let coefficients = [1., -2., 3., 4.];
        bencher.bench(|| {
            black_box(&t)
                .iter()
                .map(|&t| signal::sweep_poly(t, black_box(&coefficients), 0.).unwrap())
                .collect::<Vec<_>>()
        });
    }

    #[divan::bench]
    fn gausspulse(bencher: Bencher) {
        let t = linspace(-1., 1., N);
        bencher.bench(|| {
            black_box(&t)
                .iter()
                .map(|&t| signal::gausspulse(t, 5., 0.5, -6.).unwrap())
                .collect::<Vec<_>>()
        });
    }

    #[divan::bench]
    fn gauss_spline(bencher: Bencher) {
        let x = linspace(-3., 3., N);
        bencher.bench(|| {
            black_box(&x)
                .iter()
                .map(|&x| signal::gauss_spline(x, 3))
                .collect::<Vec<_>>()
        });
    }

    #[divan::bench]
    fn unit_impulse(bencher: Bencher) {
        bencher.bench(|| signal::unit_impulse(black_box(N), black_box(N / 2)).unwrap());
    }

    #[divan::bench]
    fn vectorstrength(bencher: Bencher) {
        let events: Vec<f64> = positive_values(N, 9).iter().map(|v| v / 10.).collect();
        bencher.bench(|| signal::vectorstrength(black_box(&events), 0.1).unwrap());
    }

    #[divan::bench]
    fn max_len_seq(bencher: Bencher) {
        let state = [1u8; 10];
        bencher.bench(|| signal::max_len_seq(10, black_box(&state), 1023, None).unwrap());
    }
}

mod window_benches {
    use super::*;

    const M: usize = 256;

    macro_rules! simple_windows {
        ($($name:ident),+ $(,)?) => {
            $(
                #[divan::bench]
                fn $name(bencher: Bencher) {
                    bencher.bench(|| windows::$name(black_box(M), true).unwrap());
                }
            )+
        };
    }

    simple_windows!(
        boxcar,
        bartlett,
        triang,
        cosine,
        lanczos,
        barthann,
        bohman,
        parzen,
        hann,
        hamming,
        blackman,
        blackmanharris,
        nuttall,
        flattop,
    );

    #[divan::bench]
    fn general_cosine(bencher: Bencher) {
        let coefficients = [0.4, 0.5, 0.1];
        bencher.bench(|| {
            windows::general_cosine(black_box(M), black_box(&coefficients), true).unwrap()
        });
    }

    #[divan::bench]
    fn general_hamming(bencher: Bencher) {
        bencher.bench(|| windows::general_hamming(black_box(M), 0.54, true).unwrap());
    }

    #[divan::bench]
    fn gaussian(bencher: Bencher) {
        bencher.bench(|| windows::gaussian(black_box(M), 30., true).unwrap());
    }

    #[divan::bench]
    fn general_gaussian(bencher: Bencher) {
        bencher.bench(|| windows::general_gaussian(black_box(M), 1.5, 30., true).unwrap());
    }

    #[divan::bench]
    fn exponential(bencher: Bencher) {
        bencher.bench(|| windows::exponential(black_box(M), None, 30., true).unwrap());
    }

    #[divan::bench]
    fn tukey(bencher: Bencher) {
        bencher.bench(|| windows::tukey(black_box(M), 0.5, true).unwrap());
    }

    #[divan::bench]
    fn kaiser(bencher: Bencher) {
        bencher.bench(|| windows::kaiser(black_box(M), 8., true).unwrap());
    }

    #[divan::bench]
    fn kaiser_bessel_derived(bencher: Bencher) {
        bencher.bench(|| windows::kaiser_bessel_derived(black_box(M), 8., true).unwrap());
    }

    #[divan::bench]
    fn chebwin(bencher: Bencher) {
        bencher.bench(|| windows::chebwin(black_box(M), 60., true).unwrap());
    }

    #[divan::bench]
    fn taylor(bencher: Bencher) {
        bencher.bench(|| windows::taylor(black_box(M), 4, 30., true, true).unwrap());
    }

    #[divan::bench]
    fn dpss(bencher: Bencher) {
        warm(bencher, || {
            windows::dpss(black_box(48), 2.5, 4, true).unwrap()
        });
    }

    #[divan::bench]
    fn get_window(bencher: Bencher) {
        let window = Window::Taylor(4, 30., true);
        bencher.bench(|| signal::get_window(black_box(&window), black_box(M), true).unwrap());
    }
}
