mod common;
use common::{
    indices, integer, matrix_rows as matrix, number, number as boolean,
    optional as optional_number, optional as optional_integer, vector,
};

use sci::{
    Error,
    complex::Complex,
    signal::{
        windows::{self, Window},
        *,
    },
};

fn complex(s: &str) -> Complex {
    let v = vector(s);
    Complex { re: v[0], im: v[1] }
}
fn complex_vector(s: &str) -> Vec<Complex> {
    matrix(s)
        .into_iter()
        .map(|v| Complex { re: v[0], im: v[1] })
        .collect()
}
fn complex_row(v: Vec<Complex>) -> Vec<f64> {
    v.into_iter().flat_map(|z| [z.re, z.im]).collect()
}
fn mode(s: &str) -> Mode {
    match s {
        "full" => Mode::Full,
        "same" => Mode::Same,
        "valid" => Mode::Valid,
        _ => panic!(),
    }
}
fn scaling(s: &str) -> Scaling {
    match s {
        "density" => Scaling::Density,
        "spectrum" => Scaling::Spectrum,
        _ => panic!(),
    }
}
fn edge(s: &str) -> EdgeMode {
    match s {
        "clip" => EdgeMode::Clip,
        "wrap" => EdgeMode::Wrap,
        _ => panic!(),
    }
}
fn window(s: &str) -> Window {
    match s {
        "hann" => Window::Hann,
        "hamming" => Window::Hamming,
        "blackman" => Window::Blackman,
        _ => panic!(),
    }
}
fn sos(s: &str) -> Vec<[f64; 6]> {
    matrix(s)
        .into_iter()
        .map(|v| v.try_into().unwrap())
        .collect()
}
fn pair((a, b): (Vec<f64>, Vec<f64>)) -> Vec<Vec<f64>> {
    vec![a, b]
}
fn spectral((a, b, c): Spectrogram) -> Vec<Vec<f64>> {
    let mut rows = vec![a, b];
    rows.extend(c);
    rows
}
fn short_time((a, b, c): ShortTimeSpectrum) -> Vec<Vec<f64>> {
    let mut rows = vec![a, b];
    rows.extend(c.into_iter().map(complex_row));
    rows
}

fn evaluate(name: &str, a: &[&str]) -> Vec<Vec<f64>> {
    let v = |i| vector(a[i]);
    let n = |i| integer(a[i]);
    let f = |i| number(a[i]);
    let b = |i| boolean(a[i]);
    match name {
        "besselap" => return root_rows(besselap(n(0), bessel_norm(a[1])).unwrap()),
        "ellipap" => return root_rows(ellipap(n(0), f(1), f(2)).unwrap()),
        "bessel" => {
            return pair(
                bessel(
                    n(0),
                    &v(1),
                    band(a[2]),
                    optional_number(a[3]),
                    bessel_norm(a[4]),
                )
                .unwrap(),
            );
        }
        "ellip" => {
            return pair(
                ellip(n(0), f(1), f(2), &v(3), band(a[4]), optional_number(a[5])).unwrap(),
            );
        }
        "ellipord" => {
            return {
                let (n, w) = ellipord(f(0), f(1), f(2), f(3), optional_number(a[4])).unwrap();
                vec![vec![n as f64], vec![w]]
            };
        }
        "iirdesign" => {
            return pair(
                iirdesign(f(0), f(1), f(2), f(3), optional_number(a[4]), family(a[5])).unwrap(),
            );
        }
        "band_stop_obj" => {
            return vec![vec![
                band_stop_obj(
                    f(0),
                    n(1),
                    v(2).try_into().unwrap(),
                    v(3).try_into().unwrap(),
                    f(4),
                    f(5),
                    family(a[6]),
                )
                .unwrap(),
            ]];
        }
        "iircomb" => return pair(iircomb(f(0), f(1), b(2), f(3), b(4)).unwrap()),
        "gammatone" => return pair(gammatone(f(0), b(1), n(2), n(3), f(4)).unwrap()),
        "findfreqs" => return vec![findfreqs(&v(0), &v(1), n(2)).unwrap()],
        "abcd_normalize" => {
            return state_rows(
                abcd_normalize(&matrix(a[0]), &matrix(a[1]), &matrix(a[2]), &matrix(a[3])).unwrap(),
            );
        }
        "tf2ss" => return state_rows(tf2ss(&v(0), &v(1)).unwrap()),
        "zpk2ss" => return state_rows(zpk2ss(&zpk(a[0])).unwrap()),
        "ss2tf" => {
            return {
                let (b, a) = ss2tf(&state_args(a), n(4)).unwrap();
                let mut r = b;
                r.push(a);
                r
            };
        }
        "ss2zpk" => return root_rows(ss2zpk(&state_args(a), n(4), n(5)).unwrap()),
        "cont2discrete" => {
            return state_rows(
                cont2discrete(
                    &state_args(a),
                    f(4),
                    match a[5] {
                        "zoh" => Discretization::ZeroOrderHold,
                        "bilinear" => Discretization::Bilinear,
                        "euler" => Discretization::Euler,
                        _ => Discretization::BackwardDifference,
                    },
                )
                .unwrap(),
            );
        }
        "lsim" => {
            return sim_rows(lsim(&state_args(a), &matrix(a[4]), &v(5), &v(6), b(7)).unwrap());
        }
        "dlsim" => return sim_rows(dlsim(&state_args(a), &matrix(a[4]), f(5), &v(6)).unwrap()),
        "step" => return sim_rows(step(&state_args(a), &v(4), n(5)).unwrap()),
        "impulse" => return sim_rows(impulse(&state_args(a), &v(4), n(5)).unwrap()),
        "dstep" => return sim_rows(dstep(&state_args(a), f(4), n(5), n(6)).unwrap()),
        "dimpulse" => return sim_rows(dimpulse(&state_args(a), f(4), n(5), n(6)).unwrap()),
        "place_poles" => {
            return vec![place_poles(&matrix(a[0]), &v(1), &complex_vector(a[2])).unwrap()];
        }
        "unique_roots" => {
            return {
                let (r, m) = unique_roots(
                    &complex_vector(a[0]),
                    f(1),
                    match a[2] {
                        "avg" => RootRepresentative::Mean,
                        "min" => RootRepresentative::Minimum,
                        _ => RootRepresentative::Maximum,
                    },
                )
                .unwrap();
                vec![complex_row(r), m.into_iter().map(|v| v as f64).collect()]
            };
        }
        "residue" => return fraction_rows(residue(&v(0), &v(1), f(2)).unwrap()),
        "residuez" => return fraction_rows(residuez(&v(0), &v(1), f(2)).unwrap()),
        "invres" => {
            return pair(invres(&complex_vector(a[0]), &complex_vector(a[1]), &v(2)).unwrap());
        }
        "invresz" => {
            return pair(invresz(&complex_vector(a[0]), &complex_vector(a[1]), &v(2)).unwrap());
        }
        "tf2sos" => return sos_response(tf2sos(&v(0), &v(1)).unwrap()),
        "zpk2sos" => return sos_response(zpk2sos(&zpk(a[0])).unwrap()),
        "firls" => return vec![firls(n(0), &v(1), &v(2), &v(3), f(4)).unwrap()],
        "firwin2" => {
            return vec![firwin2(n(0), &v(1), &v(2), n(3), &window(a[4]), b(5), f(6)).unwrap()];
        }
        "firwin_2d" => {
            return firwin_2d(
                (indices(a[0])[0], indices(a[0])[1]),
                &v(1),
                b(2),
                &window(a[3]),
                &window(a[4]),
                b(5),
                f(6),
            )
            .unwrap();
        }
        "minimum_phase" => return vec![minimum_phase(&v(0), n(1), b(2)).unwrap()],
        "remez" => return vec![remez(n(0), &v(1), &v(2), &v(3), f(4)).unwrap()],
        "cspline1d" => return vec![cspline1d(&v(0), f(1)).unwrap()],
        "qspline1d" => return vec![qspline1d(&v(0)).unwrap()],
        "cspline1d_eval" => return vec![cspline1d_eval(&v(0), &v(1), f(2), f(3)).unwrap()],
        "qspline1d_eval" => return vec![qspline1d_eval(&v(0), &v(1), f(2), f(3)).unwrap()],
        "cspline2d" => return cspline2d(&matrix(a[0]), f(1), f(2)).unwrap(),
        "qspline2d" => return qspline2d(&matrix(a[0]), f(1)).unwrap(),
        "spline_filter" => return spline_filter(&matrix(a[0]), f(1)).unwrap(),
        "symiirorder1" => return vec![symiirorder1(&v(0), f(1), f(2), f(3)).unwrap()],
        "symiirorder2" => return vec![symiirorder2(&v(0), f(1), f(2), f(3)).unwrap()],
        "whittaker_henderson" => {
            return vec![whittaker_henderson(&v(0), f(1), n(2), &v(3)).unwrap()];
        }
        "choose_conv_method" => {
            return vec![vec![
                if choose_conv_method(&v(0), &v(1), mode(a[2])).unwrap() == ConvolutionMethod::Fft {
                    1.
                } else {
                    0.
                },
            ]];
        }
        "oaconvolve" => return vec![oaconvolve(&v(0), &v(1), mode(a[2])).unwrap()],
        "closest_STFT_dual_window" => {
            return {
                let (w, s) = closest_STFT_dual_window(&v(0), n(1), &v(2), b(3)).unwrap();
                vec![w, vec![s]]
            };
        }
        "envelope" => {
            return {
                let e = envelope(
                    &v(0),
                    (indices(a[1])[0], indices(a[1])[1]),
                    n(2),
                    b(3),
                    b(4),
                )
                .unwrap();
                vec![e.amplitude, e.residual]
            };
        }
        "max_len_seq" => {
            return {
                let state: Vec<_> = indices(a[1]).into_iter().map(|b| b as u8).collect();
                let taps = if a[3] == "none" {
                    None
                } else {
                    Some(indices(a[3]))
                };
                let (x, s) = max_len_seq(n(0), &state, n(2), taps.as_deref()).unwrap();
                vec![
                    x.into_iter().map(f64::from).collect(),
                    s.into_iter().map(f64::from).collect(),
                ]
            };
        }
        "find_peaks_cwt" => {
            return vec![
                find_peaks_cwt(&v(0), &v(1))
                    .unwrap()
                    .into_iter()
                    .map(|i| i as f64)
                    .collect(),
            ];
        }
        _ => {}
    }
    if let Some(name) = name.strip_prefix("windows.") {
        let result = match name {
            "boxcar" => windows::boxcar(n(0), b(1)),
            "bartlett" => windows::bartlett(n(0), b(1)),
            "triang" => windows::triang(n(0), b(1)),
            "cosine" => windows::cosine(n(0), b(1)),
            "lanczos" => windows::lanczos(n(0), b(1)),
            "barthann" => windows::barthann(n(0), b(1)),
            "bohman" => windows::bohman(n(0), b(1)),
            "parzen" => windows::parzen(n(0), b(1)),
            "hann" => windows::hann(n(0), b(1)),
            "hamming" => windows::hamming(n(0), b(1)),
            "blackman" => windows::blackman(n(0), b(1)),
            "blackmanharris" => windows::blackmanharris(n(0), b(1)),
            "nuttall" => windows::nuttall(n(0), b(1)),
            "flattop" => windows::flattop(n(0), b(1)),
            "general_cosine" => windows::general_cosine(n(0), &v(1), b(2)),
            "general_hamming" => windows::general_hamming(n(0), f(1), b(2)),
            "gaussian" => windows::gaussian(n(0), f(1), b(2)),
            "general_gaussian" => windows::general_gaussian(n(0), f(1), f(2), b(3)),
            "exponential" => windows::exponential(n(0), optional_number(a[1]), f(2), b(3)),
            "tukey" => windows::tukey(n(0), f(1), b(2)),
            "kaiser" => windows::kaiser(n(0), f(1), b(2)),
            "chebwin" => windows::chebwin(n(0), f(1), b(2)),
            "taylor" => windows::taylor(n(0), n(1), f(2), b(3), b(4)),
            "kaiser_bessel_derived" => windows::kaiser_bessel_derived(n(0), f(1), b(2)),
            "dpss" => return windows::dpss(n(0), f(1), n(2), b(3)).unwrap(),
            "get_window" => windows::get_window(&window(a[0]), n(1), b(2)),
            _ => panic!("unknown window {name}"),
        };
        return vec![result.unwrap()];
    }
    let result = match name {
        "resample_poly" => {
            let taps = if a[3] == "none" { None } else { Some(v(3)) };
            resample_poly(&v(0), n(1), n(2), taps.as_deref()).unwrap()
        }
        "decimate" => decimate(&v(0), n(1), n(2), b(3)).unwrap(),
        "convolve2d" => {
            return convolve2d(
                &matrix(a[0]),
                &matrix(a[1]),
                mode(a[2]),
                boundary(a[3], f(4)),
            )
            .unwrap();
        }
        "correlate2d" => {
            return correlate2d(
                &matrix(a[0]),
                &matrix(a[1]),
                mode(a[2]),
                boundary(a[3], f(4)),
            )
            .unwrap();
        }
        "sepfir2d" => return sepfir2d(&matrix(a[0]), &v(1), &v(2)).unwrap(),
        "medfilt2d" => return medfilt2d(&matrix(a[0]), indices(a[1]).try_into().unwrap()).unwrap(),
        "hilbert2" => {
            return hilbert2(&matrix(a[0]))
                .unwrap()
                .into_iter()
                .map(complex_row)
                .collect();
        }
        "lp2lp" => return pair(lp2lp(&v(0), &v(1), f(2)).unwrap()),
        "lp2hp" => return pair(lp2hp(&v(0), &v(1), f(2)).unwrap()),
        "lp2bp" => return pair(lp2bp(&v(0), &v(1), f(2), f(3)).unwrap()),
        "lp2bs" => return pair(lp2bs(&v(0), &v(1), f(2), f(3)).unwrap()),
        "bilinear" => return pair(bilinear(&v(0), &v(1), f(2)).unwrap()),
        "buttap" => return root_rows(buttap(n(0)).unwrap()),
        "cheb1ap" => return root_rows(cheb1ap(n(0), f(1)).unwrap()),
        "cheb2ap" => return root_rows(cheb2ap(n(0), f(1)).unwrap()),
        "lp2lp_zpk" => return root_rows(lp2lp_zpk(&zpk(a[0]), f(1)).unwrap()),
        "lp2hp_zpk" => return root_rows(lp2hp_zpk(&zpk(a[0]), f(1)).unwrap()),
        "lp2bp_zpk" => return root_rows(lp2bp_zpk(&zpk(a[0]), f(1), f(2)).unwrap()),
        "lp2bs_zpk" => return root_rows(lp2bs_zpk(&zpk(a[0]), f(1), f(2)).unwrap()),
        "bilinear_zpk" => return root_rows(bilinear_zpk(&zpk(a[0]), f(1)).unwrap()),
        "zpk2tf" => return pair(zpk2tf(&zpk(a[0])).unwrap()),
        "normalize" => return pair(normalize(&v(0), &v(1)).unwrap()),
        "tf2zpk" => return root_rows(tf2zpk(&v(0), &v(1)).unwrap()),
        "sos2zpk" => return root_rows(sos2zpk(&sos(a[0])).unwrap()),
        "freqs_zpk" => complex_row(freqs_zpk(&zpk(a[0]), &v(1)).unwrap()),
        "freqz_zpk" => complex_row(freqz_zpk(&zpk(a[0]), &v(1)).unwrap()),
        "butter" => return pair(butter(n(0), &v(1), band(a[2]), optional_number(a[3])).unwrap()),
        "cheby1" => {
            return pair(cheby1(n(0), f(1), &v(2), band(a[3]), optional_number(a[4])).unwrap());
        }
        "cheby2" => {
            return pair(cheby2(n(0), f(1), &v(2), band(a[3]), optional_number(a[4])).unwrap());
        }
        "iirfilter" => {
            return pair(
                iirfilter(
                    n(0),
                    &v(1),
                    band(a[2]),
                    optional_number(a[3]),
                    match a[4] {
                        "butter" => IirFamily::Butterworth,
                        "cheby1" => IirFamily::ChebyshevI(1.),
                        "cheby2" => IirFamily::ChebyshevII(40.),
                        _ => panic!(),
                    },
                )
                .unwrap(),
            );
        }
        "buttord" | "cheb1ord" | "cheb2ord" => {
            let (order, w) = match name {
                "buttord" => buttord(f(0), f(1), f(2), f(3), optional_number(a[4])),
                "cheb1ord" => cheb1ord(f(0), f(1), f(2), f(3), optional_number(a[4])),
                _ => cheb2ord(f(0), f(1), f(2), f(3), optional_number(a[4])),
            }
            .unwrap();
            return vec![vec![order as f64], vec![w]];
        }
        "bode" => return pair(bode(&v(0), &v(1), &v(2)).unwrap()),
        "dbode" => return pair(dbode(&v(0), &v(1), &v(2)).unwrap()),
        "freqresp" => complex_row(freqresp(&v(0), &v(1), &v(2)).unwrap()),
        "dfreqresp" => complex_row(dfreqresp(&v(0), &v(1), &v(2)).unwrap()),
        "get_window" => get_window(&window(a[0]), n(1), b(2)).unwrap(),
        "convolve" => convolve(&v(0), &v(1), mode(a[2])).unwrap(),
        "correlate" => correlate(&v(0), &v(1), mode(a[2])).unwrap(),
        "fftconvolve" => fftconvolve(&v(0), &v(1), mode(a[2])).unwrap(),
        "correlation_lags" => correlation_lags(n(0), n(1), mode(a[2]))
            .unwrap()
            .into_iter()
            .map(|v| v as f64)
            .collect(),
        "deconvolve" => return pair(deconvolve(&v(0), &v(1)).unwrap()),
        "upfirdn" => upfirdn(&v(0), &v(1), n(2), n(3)).unwrap(),
        "sawtooth" => vec![sawtooth(f(0), f(1))],
        "square" => vec![square(f(0), f(1))],
        "chirp" => vec![
            chirp(
                f(0),
                f(1),
                f(2),
                f(3),
                match a[4] {
                    "linear" => ChirpMethod::Linear,
                    "quadratic" => ChirpMethod::Quadratic,
                    "logarithmic" => ChirpMethod::Logarithmic,
                    "hyperbolic" => ChirpMethod::Hyperbolic,
                    _ => panic!(),
                },
                f(5),
            )
            .unwrap(),
        ],
        "sweep_poly" => vec![sweep_poly(f(0), &v(1), f(2)).unwrap()],
        "gausspulse" => {
            let (i, q, e) = gausspulse(f(0), f(1), f(2), f(3)).unwrap();
            return vec![vec![i], vec![q], vec![e]];
        }
        "gauss_spline" => vec![gauss_spline(f(0), n(1))],
        "unit_impulse" => unit_impulse(n(0), n(1)).unwrap(),
        "vectorstrength" => {
            let (s, p) = vectorstrength(&v(0), f(1)).unwrap();
            return vec![vec![s], vec![p]];
        }
        "detrend" => detrend(&v(0), b(1)).unwrap(),
        "lfilter" => lfilter(&v(0), &v(1), &v(2)).unwrap(),
        "lfilter_with_state" => {
            return pair(lfilter_with_state(&v(0), &v(1), &v(2), &v(3)).unwrap());
        }
        "lfilter_zi" => lfilter_zi(&v(0), &v(1)).unwrap(),
        "lfiltic" => lfiltic(&v(0), &v(1), &v(2), &v(3)).unwrap(),
        "filtfilt" => filtfilt(&v(0), &v(1), &v(2), n(3)).unwrap(),
        "sosfilt" => sosfilt(&sos(a[0]), &v(1)).unwrap(),
        "sosfilt_zi" => {
            return sosfilt_zi(&sos(a[0]))
                .unwrap()
                .into_iter()
                .map(|r| r.to_vec())
                .collect();
        }
        "sos2tf" => return pair(sos2tf(&sos(a[0])).unwrap()),
        "sosfiltfilt" => sosfiltfilt(&sos(a[0]), &v(1), n(2)).unwrap(),
        "freqz" => complex_row(freqz(&v(0), &v(1), &v(2)).unwrap()),
        "freqs" => complex_row(freqs(&v(0), &v(1), &v(2)).unwrap()),
        "freqz_sos" => complex_row(freqz_sos(&sos(a[0]), &v(1)).unwrap()),
        "sosfreqz" => complex_row(sosfreqz(&sos(a[0]), &v(1)).unwrap()),
        "group_delay" => group_delay(&v(0), &v(1), &v(2)).unwrap(),
        "iirnotch" => return pair(iirnotch(f(0), f(1), f(2)).unwrap()),
        "iirpeak" => return pair(iirpeak(f(0), f(1), f(2)).unwrap()),
        "kaiser_beta" => vec![kaiser_beta(f(0))],
        "kaiser_atten" => vec![kaiser_atten(n(0), f(1)).unwrap()],
        "kaiserord" => {
            let (n, b) = kaiserord(f(0), f(1)).unwrap();
            return vec![vec![n as f64], vec![b]];
        }
        "firwin" => firwin(n(0), &v(1), b(2), &window(a[3]), b(4), f(5)).unwrap(),
        "medfilt" => medfilt(&v(0), n(1)).unwrap(),
        "order_filter" => order_filter(&v(0), n(1), n(2)).unwrap(),
        "wiener" => wiener(&v(0), n(1), optional_number(a[2])).unwrap(),
        "savgol_coeffs" => savgol_coeffs(n(0), n(1), n(2), f(3)).unwrap(),
        "savgol_filter" => savgol_filter(&v(0), n(1), n(2), n(3), f(4)).unwrap(),
        "hilbert" => complex_row(hilbert(&v(0)).unwrap()),
        "resample" => resample(&v(0), n(1)).unwrap(),
        "czt_points" => complex_row(
            czt_points(
                n(0),
                if a[1] == "none" {
                    None
                } else {
                    Some(complex(a[1]))
                },
                complex(a[2]),
            )
            .unwrap(),
        ),
        "czt" => complex_row(
            czt(
                &complex_vector(a[0]),
                n(1),
                if a[2] == "none" {
                    None
                } else {
                    Some(complex(a[2]))
                },
                complex(a[3]),
            )
            .unwrap(),
        ),
        "zoom_fft" => {
            complex_row(zoom_fft(&complex_vector(a[0]), f(1), f(2), n(3), f(4), b(5)).unwrap())
        }
        "periodogram" => {
            return pair(periodogram(&v(0), &v(1), n(2), f(3), b(4), scaling(a[5])).unwrap());
        }
        "welch" => {
            return pair(welch(&v(0), &v(1), n(2), n(3), f(4), b(5), scaling(a[6])).unwrap());
        }
        "csd" => {
            let (f, p) = csd(&v(0), &v(1), &v(2), n(3), n(4), f(5), b(6), scaling(a[7])).unwrap();
            return vec![f, complex_row(p)];
        }
        "coherence" => return pair(coherence(&v(0), &v(1), &v(2), n(3), n(4), f(5)).unwrap()),
        "spectrogram" => {
            return spectral(
                spectrogram(&v(0), &v(1), n(2), n(3), f(4), b(5), scaling(a[6])).unwrap(),
            );
        }
        "stft" => return short_time(stft(&v(0), &v(1), n(2), n(3), f(4)).unwrap()),
        "istft" => {
            let z = stft(&v(0), &v(1), n(2), n(3), f(4)).unwrap().2;
            return pair(istft(&z, &v(1), n(2), n(3), f(4)).unwrap());
        }
        "check_cola" => vec![f64::from(check_cola(&v(0), n(1), f(2)).unwrap())],
        "check_COLA" => vec![f64::from(check_COLA(&v(0), n(1), f(2)).unwrap())],
        "check_nola" => vec![f64::from(check_nola(&v(0), n(1), f(2)).unwrap())],
        "check_NOLA" => vec![f64::from(check_NOLA(&v(0), n(1), f(2)).unwrap())],
        "lombscargle" => lombscargle(&v(0), &v(1), &v(2), b(3), b(4)).unwrap(),
        "find_peaks" => find_peaks(&v(0))
            .unwrap()
            .into_iter()
            .map(|v| v as f64)
            .collect(),
        "argrelmax" => argrelmax(&v(0), n(1), edge(a[2]))
            .unwrap()
            .into_iter()
            .map(|v| v as f64)
            .collect(),
        "argrelmin" => argrelmin(&v(0), n(1), edge(a[2]))
            .unwrap()
            .into_iter()
            .map(|v| v as f64)
            .collect(),
        "argrelextrema" => argrelextrema(&v(0), |a, b| a > b, n(1), edge(a[2]))
            .unwrap()
            .into_iter()
            .map(|v| v as f64)
            .collect(),
        "peak_prominences" => {
            let p = peak_prominences(&v(0), &indices(a[1]), optional_integer(a[2])).unwrap();
            return vec![
                p.values,
                p.left_bases.into_iter().map(|v| v as f64).collect(),
                p.right_bases.into_iter().map(|v| v as f64).collect(),
            ];
        }
        "peak_widths" => {
            let p = peak_widths(&v(0), &indices(a[1]), f(2), optional_integer(a[3])).unwrap();
            return vec![p.widths, p.heights, p.left_ips, p.right_ips];
        }
        _ => panic!("unknown function {name}"),
    };
    vec![result]
}

#[track_caller]
fn near(actual: f64, expected: f64, tolerance: f64, context: &str) {
    common::assert_close(actual, expected, tolerance, tolerance, context);
}

#[test]
fn scipy_signal_fixtures() {
    let inputs = common::lines(include_str!("fixtures/signal_inputs.csv"));
    let expected = common::lines(include_str!("fixtures/signal_expected.csv"));
    assert_eq!(inputs.clone().count(), expected.clone().count());
    let mut failures = vec![];
    for (input, reference) in inputs.zip(expected) {
        let a: Vec<_> = common::fields(input);
        let (id, values) = reference.split_once(',').unwrap();
        assert_eq!(a[0], id);
        let context = format!("case {} {}", a[0], a[1]);
        let result = std::panic::catch_unwind(|| {
            let actual = evaluate(a[1], &a[2..]);
            let expected: Vec<_> = values.split('/').map(vector).collect();
            assert_eq!(actual.len(), expected.len(), "{context} row count");
            // Direct f64 formulas/FFT normally agree to <1e-13; the 2e-11
            // combined absolute/relative bound also covers small SVD/eigen fits.
            // Remez proxy/grid approximation has a separate measured bound,
            // documented in function_coverages/SIGNAL_COVERAGE.md.
            for (actual, expected) in actual.iter().zip(expected) {
                assert_eq!(
                    actual.len(),
                    expected.len(),
                    "{context} row shape: {actual:?} versus {expected:?}"
                );
                for (&x, y) in actual.iter().zip(expected) {
                    near(x, y, if a[1] == "remez" { 1e-10 } else { 2e-11 }, &context);
                }
            }
        });
        if result.is_err() {
            failures.push(context);
        }
    }
    assert!(
        failures.is_empty(),
        "{} failing cases: {:?}",
        failures.len(),
        failures
    );
}

fn zpk(s: &str) -> Zpk {
    let v: Vec<_> = s.split('|').collect();
    Zpk {
        zeros: if v[0].is_empty() {
            vec![]
        } else {
            complex_vector(v[0])
        },
        poles: if v[1].is_empty() {
            vec![]
        } else {
            complex_vector(v[1])
        },
        gain: number(v[2]),
    }
}
fn root_rows(z: Zpk) -> Vec<Vec<f64>> {
    vec![complex_row(z.zeros), complex_row(z.poles), vec![z.gain]]
}
fn band(s: &str) -> BandType {
    match s {
        "lowpass" => BandType::Lowpass,
        "highpass" => BandType::Highpass,
        "bandpass" => BandType::Bandpass,
        "bandstop" => BandType::Bandstop,
        _ => panic!(),
    }
}

#[test]
fn streaming_filters_preserve_state_and_sos_interoperates() {
    let (b, a) = butter(3, &[0.3], BandType::Lowpass, Some(2.)).unwrap();
    let x = [1., 2., 0., -1., 3., 2., 4.];
    let zero = vec![0.; a.len() - 1];
    let (mut left, state) = lfilter_with_state(&b, &a, &x[..3], &zero).unwrap();
    let (right, _) = lfilter_with_state(&b, &a, &x[3..], &state).unwrap();
    left.extend(right);
    for (u, v) in left.iter().zip(lfilter(&b, &a, &x).unwrap()) {
        near(*u, v, 1e-14, "streaming");
    }
    let zi = lfilter_zi(&b, &a).unwrap();
    let steady = lfilter_with_state(&b, &a, &[1.; 20], &zi).unwrap().0;
    for y in steady {
        near(y, 1., 1e-13, "steady step");
    }
    let rows = [
        [0.2, 0.4, 0.2, 1., -0.4, 0.1],
        [1., -0.2, 0.1, 1., -0.5, 0.2],
    ];
    let (b, a) = sos2tf(&rows).unwrap();
    for (u, v) in sosfilt(&rows, &x)
        .unwrap()
        .iter()
        .zip(lfilter(&b, &a, &x).unwrap())
    {
        near(*u, v, 1e-13, "SOS/TF");
    }
}

#[test]
fn transforms_spectra_and_smoothing_obey_independent_identities() {
    for n in [3, 4, 7, 8] {
        let x: Vec<_> = (0..n).map(|j| (j as f64).sin()).collect();
        let z = hilbert(&x).unwrap();
        for (&a, b) in x.iter().zip(&z) {
            near(b.re, a, 1e-13, "analytic real part");
        }
        let (f, p) = periodogram(&x, &vec![1.; n], n, 4., false, Scaling::Density).unwrap();
        let energy = p.iter().sum::<f64>() * (f[1] - f[0]);
        near(
            energy,
            x.iter().map(|v| v * v).sum::<f64>() / n as f64,
            1e-13,
            "Parseval",
        );
        let cx: Vec<_> = x.iter().map(|&re| Complex { re, im: 0. }).collect();
        let z = czt(&cx, n, None, Complex { re: 1., im: 0. }).unwrap();
        for (u, v) in z.iter().zip(fft(&cx)) {
            near(u.re, v.re, 1e-13, "CZT/FFT real");
            near(u.im, v.im, 1e-13, "CZT/FFT imaginary");
        }
        let round = resample(&resample(&x, n * 2).unwrap(), n).unwrap();
        for (&u, v) in x.iter().zip(round) {
            near(u, v, 1e-13, "resampling round trip");
        }
    }
    let x: Vec<_> = (0..17)
        .map(|j| 1. + 2. * j as f64 + 3. * (j as f64).powi(2))
        .collect();
    for (&u, v) in x.iter().zip(savgol_filter(&x, 7, 2, 0, 1.).unwrap()) {
        near(u, v, 1e-12, "polynomial preservation");
    }
    let w = windows::hann(8, false).unwrap();
    let z = stft(&x, &w, 4, 8, 1.).unwrap();
    let (time, round) = istft(&z.2, &w, 4, 8, 1.).unwrap();
    assert_eq!(time.len(), round.len());
    for (&u, &v) in x.iter().zip(&round) {
        near(u, v, 1e-12, "STFT round trip");
    }
    let tapers = windows::dpss(32, 2.5, 4, true).unwrap();
    for (i, a) in tapers.iter().enumerate() {
        for (j, b) in tapers.iter().enumerate() {
            near(
                sci::linalg::dot(a, b).unwrap(),
                if i == j { 1. } else { 0. },
                1e-12,
                "DPSS orthonormality",
            );
        }
    }
    let window_variants = [
        Window::Boxcar,
        Window::Hann,
        Window::Hamming,
        Window::Blackman,
        Window::BlackmanHarris,
        Window::Nuttall,
        Window::FlatTop,
        Window::Bartlett,
        Window::Triang,
        Window::Barthann,
        Window::Bohman,
        Window::Parzen,
        Window::Cosine,
        Window::Lanczos,
        Window::Kaiser(4.),
        Window::Gaussian(2.),
        Window::GeneralGaussian(2., 3.),
        Window::Tukey(0.5),
        Window::Exponential(2.),
        Window::Chebwin(60.),
        Window::Taylor(4, 30., true),
        Window::GeneralHamming(0.6),
        Window::GeneralCosine(vec![0.5, 0.5]),
    ];
    for w in window_variants {
        let symmetric = get_window(&w, 8, false).unwrap();
        for (&a, &b) in symmetric.iter().zip(symmetric.iter().rev()) {
            near(a, b, 1e-13, "window symmetry");
        }
        let periodic = get_window(&w, 7, true).unwrap();
        assert_eq!(periodic, symmetric[..7]);
    }
}

#[test]
fn filter_design_round_trips_and_meets_frequency_constraints() {
    let analog = buttap(3).unwrap();
    let w = [0., 0.5, 1., 2.];
    let h = freqs_zpk(&analog, &w).unwrap();
    for (&w, h) in w.iter().zip(h) {
        near(
            h.re * h.re + h.im * h.im,
            1. / (1. + w.powi(6)),
            1e-13,
            "Butterworth magnitude",
        );
    }
    for (n, w) in [
        buttord(0.2, 0.4, 1., 40., Some(2.)).unwrap(),
        cheb1ord(0.2, 0.4, 1., 40., Some(2.)).unwrap(),
        cheb2ord(0.2, 0.4, 1., 40., Some(2.)).unwrap(),
    ]
    .into_iter()
    .enumerate()
    {
        let (order, cutoff) = w;
        let (b, a) = match n {
            0 => butter(order, &[cutoff], BandType::Lowpass, Some(2.)),
            1 => cheby1(order, 1., &[cutoff], BandType::Lowpass, Some(2.)),
            _ => cheby2(order, 40., &[cutoff], BandType::Lowpass, Some(2.)),
        }
        .unwrap();
        let h = freqz(
            &b,
            &a,
            &[std::f64::consts::PI * 0.2, std::f64::consts::PI * 0.4],
        )
        .unwrap();
        let db: Vec<_> = h
            .iter()
            .map(|z| -20. * sci::complex::abs(*z).log10())
            .collect();
        assert!(db[0] <= 1. + 1e-8 && db[1] >= 40. - 1e-8);
    }
    let (b, a) = butter(3, &[0.3], BandType::Lowpass, Some(2.)).unwrap();
    let z = tf2zpk(&b, &a).unwrap();
    let (b2, a2) = zpk2tf(&z).unwrap();
    for (&u, v) in b.iter().zip(b2) {
        near(u, v, 1e-12, "numerator factoring");
    }
    for (&u, v) in a.iter().zip(a2) {
        near(u, v, 1e-12, "denominator factoring");
    }
    let x = lfilter(&b, &a, &[1., 0., 0., 0.]).unwrap();
    assert!(sci::stats::mean(&x).unwrap().is_finite());
}

#[test]
fn invalid_domains_shapes_and_work_limits_are_errors() {
    let invalid = [f64::NAN];
    let x = [1., 2., 3., 4.];
    let w = [1.; 3];
    for f in [convolve, correlate, fftconvolve] {
        assert!(f(&[], &x, Mode::Full).is_err());
        assert!(f(&invalid, &x, Mode::Full).is_err());
    }
    assert!(convolve(&vec![1.; 5000], &vec![1.; 5000], Mode::Full).is_err());
    assert!(correlation_lags(usize::MAX, 1, Mode::Full).is_err());
    assert!(correlation_lags(0, 1, Mode::Full).is_err());
    assert!(deconvolve(&x, &[0.]).is_err());
    assert!(upfirdn(&w, &x, 0, 1).is_err());
    assert!(upfirdn(&w, &x, usize::MAX, 1).is_err());
    assert!(lfilter(&w, &[0.], &x).is_err());
    assert!(lfilter_with_state(&w, &[1.], &x, &[]).is_err());
    assert!(lfilter_zi(&[1.], &[1., -1.]).is_err());
    assert!(lfiltic(&w, &[1.], &invalid, &[]).is_err());
    assert!(filtfilt(&w, &[1.], &x, 4).is_err());
    let bad = [[1., 0., 0., 2., 0., 0.]];
    assert!(sosfilt(&bad, &x).is_err());
    assert!(sosfilt_zi(&bad).is_err());
    assert!(sosfiltfilt(&bad, &x, 0).is_err());
    assert!(sos2tf(&bad).is_err());
    assert!(freqz_sos(&bad, &x).is_err());
    assert!(freqz(&[1.], &[1., -1.], &[0.]).is_err());
    assert!(freqs(&[1.], &[0.], &[1.]).is_err());
    assert!(group_delay(&[1., -1.], &[1.], &[0.]).is_err());
    assert!(iirnotch(1., 1., 2.).is_err());
    assert!(iirpeak(0.1, 0., 2.).is_err());
    assert!(kaiser_atten(0, 0.1).is_err());
    assert!(kaiserord(7., 0.1).is_err());
    assert!(firwin(4, &[0.2], false, &Window::Hann, true, 2.).is_err());
    assert!(medfilt(&x, 2).is_err());
    assert!(order_filter(&x, 3, 3).is_err());
    assert!(wiener(&x, 3, Some(-1.)).is_err());
    assert!(savgol_coeffs(3, 3, 0, 1.).is_err());
    assert!(savgol_filter(&x, 3, 1, 0, 0.).is_err());
    assert!(chirp(0., 1., 0., 2., ChirpMethod::Linear, 0.).is_err());
    assert!(chirp(0., 0., 1., 2., ChirpMethod::Logarithmic, 0.).is_err());
    assert!(sweep_poly(0., &[], 0.).is_err());
    assert!(gausspulse(0., -1., 0.5, -6.).is_err());
    assert!(unit_impulse(3, 3).is_err());
    assert!(vectorstrength(&[], 1.).is_err());
    assert!(detrend(&invalid, true).is_err());
    assert!(hilbert(&[]).is_err());
    assert!(resample(&x, 0).is_err());
    assert!(czt_points(0, None, Complex { re: 1., im: 0. }).is_err());
    assert!(czt_points(3, Some(Complex::default()), Complex { re: 1., im: 0. }).is_err());
    assert!(czt(&[], 3, None, Complex { re: 1., im: 0. }).is_err());
    assert!(zoom_fft(&[Complex::default()], 0., 1., usize::MAX, 2., false).is_err());
    assert!(periodogram(&x, &w, 4, 1., false, Scaling::Density).is_err());
    assert!(welch(&x, &w, 3, 4, 1., false, Scaling::Density).is_err());
    assert!(csd(&x, &w, &w, 1, 4, 1., false, Scaling::Density).is_err());
    assert!(coherence(&x, &w, &w, 1, 4, 1.).is_err());
    assert!(spectrogram(&x, &w, 1, 2, 1., false, Scaling::Density).is_err());
    assert!(stft(&x, &[0.; 3], 1, 4, 1.).is_err());
    assert!(istft(&[vec![Complex::default()]], &w, 1, 4, 1.).is_err());
    assert!(check_cola(&w, 3, 1e-10).is_err());
    assert!(check_nola(&w, 1, 0.).is_err());
    assert!(lombscargle(&x, &x, &[0.], false, false).is_err());
    assert!(argrelmax(&x, 0, EdgeMode::Clip).is_err());
    assert!(argrelmin(&invalid, 1, EdgeMode::Wrap).is_err());
    assert!(argrelextrema(&x, |a, b| a > b, usize::MAX, EdgeMode::Clip).is_err());
    assert!(find_peaks(&invalid).is_err());
    assert!(peak_prominences(&x, &[4], None).is_err());
    assert!(peak_widths(&x, &[1], -0.5, None).is_err());
    assert!(windows::hann(usize::MAX, true).is_err());
    assert!(windows::general_cosine(4, &[], true).is_err());
    assert!(windows::general_hamming(4, f64::NAN, true).is_err());
    assert!(windows::gaussian(4, 0., true).is_err());
    assert!(windows::general_gaussian(4, 0., 1., true).is_err());
    assert!(windows::exponential(4, Some(1.), 1., true).is_err());
    assert!(windows::tukey(4, f64::NAN, true).is_err());
    assert!(windows::kaiser(4, f64::INFINITY, true).is_err());
    assert!(windows::kaiser_bessel_derived(3, 1., true).is_err());
    assert!(windows::chebwin(4, 601., true).is_err());
    assert!(windows::taylor(4, 0, 30., true, true).is_err());
    assert!(windows::dpss(257, 2., 2, true).is_err());
    assert!(get_window(&Window::Hann, usize::MAX, true).is_err());
    assert!(buttap(65).is_err());
    assert!(cheb1ap(3, 0.).is_err());
    assert!(cheb2ap(3, f64::INFINITY).is_err());
    let z = buttap(3).unwrap();
    assert!(lp2lp_zpk(&z, 0.).is_err());
    assert!(lp2hp_zpk(&z, -1.).is_err());
    assert!(lp2bp_zpk(&z, 1., 0.).is_err());
    assert!(lp2bs_zpk(&z, 0., 1.).is_err());
    assert!(bilinear_zpk(&z, 0.).is_err());
    assert!(
        zpk2tf(&Zpk {
            zeros: vec![Complex { re: 0., im: 1. }],
            poles: z.poles.clone(),
            gain: 1.
        })
        .is_err()
    );
    assert!(tf2zpk(&[1., 2., 3.], &[1., 1.]).is_err());
    assert!(normalize(&[1.], &[0.]).is_err());
    assert!(sos2zpk(&bad).is_err());
    assert!(freqs_zpk(&z, &invalid).is_err());
    assert!(freqz_zpk(&z, &invalid).is_err());
    assert!(butter(3, &[1.], BandType::Lowpass, Some(2.)).is_err());
    assert!(cheby1(3, 0., &[0.2], BandType::Lowpass, Some(2.)).is_err());
    assert!(cheby2(3, 40., &[0.2], BandType::Bandpass, Some(2.)).is_err());
    assert!(
        iirfilter(
            3,
            &[0.3, 0.2],
            BandType::Bandstop,
            None,
            IirFamily::Butterworth
        )
        .is_err()
    );
    assert!(bode(&[1.], &[0.], &x).is_err());
    assert!(dbode(&[1.], &[0.], &x).is_err());
    assert!(buttord(0.2, 0.2, 1., 40., None).is_err());
    assert!(cheb1ord(0.2, 0.4, 40., 1., None).is_err());
    assert!(cheb2ord(0.2, 1., 1., 40., Some(2.)).is_err());
    assert_eq!(
        lfilter(&[1.], &[1.], &[f64::MAX, f64::MAX]).unwrap(),
        vec![f64::MAX; 2]
    );
    assert_eq!(
        convolve(&[f64::MAX], &[2.], Mode::Full),
        Err(Error::NonFiniteEvaluation)
    );
}

#[test]
fn documented_special_values_and_adaptations() {
    assert!(square(0., f64::NAN).is_nan());
    assert!(square(f64::INFINITY, 0.5).is_nan());
    assert!(sawtooth(f64::NAN, 0.5).is_nan());
    assert_eq!(kaiser_beta(f64::NAN), 0.);
    assert_eq!(kaiser_beta(f64::INFINITY), f64::INFINITY);
    assert_eq!(wiener(&[1., 2., 3.], 1, None).unwrap(), vec![1., 2., 3.]);
    assert_eq!(wiener(&[0.; 4], 3, Some(0.)).unwrap(), vec![0.; 4]);
    assert_eq!(detrend(&[], true).unwrap(), Vec::<f64>::new());
    assert_eq!(find_peaks(&[]).unwrap(), Vec::<usize>::new());
    assert_eq!(lfilter(&[1.], &[1.], &[]).unwrap(), Vec::<f64>::new());
    let (_, p) = coherence(&[0.; 4], &[0.; 4], &[1.; 2], 1, 2, 1.).unwrap();
    assert!(p.iter().all(|v| v.is_nan()));
    let (db, _) = dbode(&[1., -1.], &[1.], &[0.]).unwrap();
    assert_eq!(db[0], f64::NEG_INFINITY);
    assert_eq!(square(-0., 0.5), square(0., 0.5));
    assert_eq!(sawtooth(-0., 1.), -1.);
    let impulse = unit_impulse(3, 1).unwrap();
    assert!(impulse[0].is_sign_positive());
}

fn boundary(s: &str, fill: f64) -> Boundary {
    match s {
        "fill" => Boundary::Fill(fill),
        "wrap" => Boundary::Wrap,
        "symm" => Boundary::Symmetric,
        _ => panic!(),
    }
}

#[test]
fn multidimensional_and_resampling_validation_and_interoperability() {
    let image = vec![vec![1., 2., 3.], vec![4., 5., 6.]];
    let kernel = vec![
        vec![0.25, 0.5, 0.25],
        vec![0.5, 1., 0.5],
        vec![0.25, 0.5, 0.25],
    ];
    let a = sepfir2d(&image, &[0.25, 0.5, 0.25], &[1., 2., 1.]).unwrap();
    let b = convolve2d(&image, &kernel, Mode::Same, Boundary::Symmetric).unwrap();
    for (a, b) in a.iter().flatten().zip(b.iter().flatten()) {
        near(*a, *b, 1e-14, "separable product");
    }
    assert!(
        sci::linalg::matvec(&a, &[1.; 3])
            .unwrap()
            .iter()
            .all(|v| v.is_finite())
    );
    let ragged = [vec![1.], vec![]];
    for f in [convolve2d, correlate2d] {
        assert_eq!(
            f(
                &[vec![f64::MAX]],
                &[vec![2.]],
                Mode::Full,
                Boundary::Fill(0.)
            ),
            Err(Error::NonFiniteEvaluation)
        );
        assert!(f(&ragged, &image, Mode::Full, Boundary::Fill(0.)).is_err());
        assert!(f(&image, &image, Mode::Same, Boundary::Fill(f64::NAN)).is_err());
        assert!(
            f(
                &[vec![1., 2.]],
                &[vec![1.], vec![2.]],
                Mode::Valid,
                Boundary::Wrap
            )
            .is_err()
        );
    }
    assert!(sepfir2d(&image, &[1., 1.], &[1.]).is_err());
    assert!(medfilt2d(&image, [2, 3]).is_err());
    assert!(hilbert2(&ragged).is_err());
    assert!(resample_poly(&[1.], 0, 1, None).is_err());
    assert!(resample_poly(&[1.], 2, 1, Some(&[])).is_err());
    assert!(resample_poly(&[1.], usize::MAX, 1, None).is_err());
    assert_eq!(resample_poly(&[1., 2.], 4, 4, None).unwrap(), vec![1., 2.]);
    assert!(decimate(&[1.], 1, 8, true).is_err());
    assert!(decimate(&[1.], 2, usize::MAX, false).is_err());
    assert!(lp2lp(&[1.], &[0.], 1.).is_err());
    assert!(lp2hp(&[1.], &[1., 0.], 1.).is_err());
    assert!(lp2bp(&[1.], &[1., 1.], 1., 0.).is_err());
    assert!(lp2bs(&[1.], &[1., 1.], 0., 1.).is_err());
    assert!(bilinear(&[1.], &[1., 1.], 0.).is_err());
}

fn bessel_norm(s: &str) -> BesselNorm {
    match s {
        "phase" => BesselNorm::Phase,
        "delay" => BesselNorm::Delay,
        _ => BesselNorm::Magnitude,
    }
}
fn family(s: &str) -> IirFamily {
    match s {
        "butter" => IirFamily::Butterworth,
        "cheby1" => IirFamily::ChebyshevI(1.),
        "cheby2" => IirFamily::ChebyshevII(40.),
        "ellip" => IirFamily::Elliptic(1., 40.),
        _ => panic!(),
    }
}
fn state_args(a: &[&str]) -> StateSpace {
    abcd_normalize(&matrix(a[0]), &matrix(a[1]), &matrix(a[2]), &matrix(a[3])).unwrap()
}
fn state_rows(s: StateSpace) -> Vec<Vec<f64>> {
    s.a.into_iter().chain(s.b).chain(s.c).chain(s.d).collect()
}
fn sim_rows(s: Simulation) -> Vec<Vec<f64>> {
    std::iter::once(s.time)
        .chain(s.output)
        .chain(s.state)
        .collect()
}

fn fraction_rows(p: PartialFractions) -> Vec<Vec<f64>> {
    vec![complex_row(p.residues), complex_row(p.poles), p.direct]
}
fn sos_response(s: Vec<[f64; 6]>) -> Vec<Vec<f64>> {
    vec![
        vec![s.len() as f64],
        complex_row(freqz_sos(&s, &[0., 0.2, 0.9, 2., 3.]).unwrap()),
    ]
}

#[test]
fn state_space_responses_and_feedback_obey_system_identities() {
    let continuous = tf2ss(&[1.], &[1., 1.]).unwrap();
    let t = [0., 0.1, 0.2, 0.3];
    let response = impulse(&continuous, &t, 0).unwrap();
    let step_response = step(&continuous, &t, 0).unwrap();
    for (i, &time) in t.iter().enumerate() {
        near(
            response.output[i][0],
            (-time).exp(),
            1e-13,
            "analytic impulse",
        );
        near(
            step_response.output[i][0],
            1. - (-time).exp(),
            1e-13,
            "analytic step",
        );
    }
    let discrete = cont2discrete(&continuous, 0.1, Discretization::ZeroOrderHold).unwrap();
    let discrete_response = dstep(&discrete, 0.1, 4, 0).unwrap();
    for (row, expected) in discrete_response.output.iter().zip(step_response.output) {
        near(row[0], expected[0], 1e-13, "ZOH step identity");
    }
    let digital = tf2ss(&[0., 1.], &[1., -0.5]).unwrap();
    let input = [1., 2., -1., 0.];
    let simulation = dlsim(
        &digital,
        &input.iter().map(|&v| vec![v]).collect::<Vec<_>>(),
        1.,
        &[0.],
    )
    .unwrap();
    let filtered = lfilter(&[0., 1.], &[1., -0.5], &input).unwrap();
    for (row, value) in simulation.output.iter().zip(filtered) {
        near(row[0], value, 1e-13, "state/filter interoperability");
    }
    let a = vec![vec![0., 1.], vec![0., 0.]];
    let target = [Complex { re: -1., im: 0. }, Complex { re: -2., im: 0. }];
    let gain = place_poles(&a, &[0., 1.], &target).unwrap();
    let feedback = abcd_normalize(
        &[vec![0., 1.], vec![-gain[0], -gain[1]]],
        &[vec![0.], vec![1.]],
        &[vec![1., 0.]],
        &[vec![0.]],
    )
    .unwrap();
    let (_, den) = ss2tf(&feedback, 0).unwrap();
    for (a, b) in den.into_iter().zip([1., 3., 2.]) {
        near(a, b, 1e-13, "feedback characteristic polynomial");
    }
    let direct = abcd_normalize(&[], &[], &[vec![]], &[vec![2., 3.]]).unwrap();
    let y = dlsim(&direct, &[vec![1., 2.]], 1., &[]).unwrap();
    assert_eq!(y.output, vec![vec![8.]]);
    assert_eq!(ss2tf(&direct, 1).unwrap(), (vec![vec![3.]], vec![1.]));
    assert_eq!(
        cont2discrete(&direct, 1., Discretization::ZeroOrderHold).unwrap(),
        direct
    );
}

#[test]
fn advanced_filter_representations_preserve_transfer_functions() {
    for digital in [false, true] {
        for (b, a) in [
            (vec![1., 2., 3.], vec![1., -1., 0.25]),
            (vec![1., 2., 3., 4.], vec![1., 1., 1.]),
            (vec![1.], vec![1., -1.5, 0.75, -0.125]),
        ] {
            let fractions = if digital {
                residuez(&b, &a, 1e-3)
            } else {
                residue(&b, &a, 1e-3)
            }
            .unwrap();
            let (rb, ra) = if digital {
                invresz(&fractions.residues, &fractions.poles, &fractions.direct)
            } else {
                invres(&fractions.residues, &fractions.poles, &fractions.direct)
            }
            .unwrap();
            let w = [0.2, 0.7, 1.3, 2.5];
            let expected = if digital {
                freqz(&b, &a, &w)
            } else {
                freqs(&b, &a, &w)
            }
            .unwrap();
            let actual = if digital {
                freqz(&rb, &ra, &w)
            } else {
                freqs(&rb, &ra, &w)
            }
            .unwrap();
            for (x, y) in actual.into_iter().zip(expected) {
                near(x.re, y.re, 1e-9, "partial fractions real");
                near(x.im, y.im, 1e-9, "partial fractions imaginary");
            }
        }
    }
    for n in [1, 2, 3, 6] {
        let (b, a) = ellip(n, 1., 40., &[0.3], BandType::Lowpass, Some(2.)).unwrap();
        let sections = tf2sos(&b, &a).unwrap();
        assert_eq!(sections.len(), n.div_ceil(2));
        assert!(sections.iter().all(|s| s[3] == 1.));
        let w = [0., 0.1, 0.4, 1., 2., 3.];
        for (x, y) in freqz_sos(&sections, &w)
            .unwrap()
            .into_iter()
            .zip(freqz(&b, &a, &w).unwrap())
        {
            near(x.re, y.re, 1e-9, "SOS real response");
            near(x.im, y.im, 1e-9, "SOS imaginary response");
        }
    }
    for n in 1..=12 {
        let delay = besselap(n, BesselNorm::Delay).unwrap();
        let w = [0., 1e-5];
        let response = freqs_zpk(&delay, &w).unwrap();
        near(response[0].re, 1., 1e-11, "Bessel DC gain");
        near(-response[1].im / 1e-5, 1., 1e-8, "Bessel unit group delay");
        let magnitude = besselap(n, BesselNorm::Magnitude).unwrap();
        let h = freqs_zpk(&magnitude, &[1.]).unwrap()[0];
        near(
            h.re * h.re + h.im * h.im,
            0.5,
            1e-12,
            "Bessel half-power normalization",
        );
    }
}

#[test]
fn spline_penalties_dual_windows_and_binary_sequences_obey_properties() {
    let line = [1., 3., 5., 7., 9.];
    for weights in [[1.; 5], [1., 0., 1., 0., 1.]] {
        let out = whittaker_henderson(&line, 1000., 2, &weights).unwrap();
        for (x, y) in out.into_iter().zip(line) {
            near(x, y, 1e-10, "unpenalized linear trend");
        }
    }
    let win = [0.5, 1., 0.7, 0.3, 0.8, 1., 0.5, 0.2];
    for hop in [1, 3, 4, 8] {
        for scaled in [false, true] {
            let (dual, _) = closest_STFT_dual_window(&win, hop, &[1.; 8], scaled).unwrap();
            for r in 0..hop {
                near(
                    (r..win.len()).step_by(hop).map(|i| win[i] * dual[i]).sum(),
                    1.,
                    1e-13,
                    "dual constraint",
                );
            }
        }
    }
    for n in [2, 3, 5, 8] {
        let length = (1usize << n) - 1;
        let (seq, state) = max_len_seq(n, &vec![1; n], length, None).unwrap();
        assert_eq!(state, vec![1; n]);
        assert_eq!(seq.iter().filter(|&&b| b == 1).count(), 1usize << (n - 1));
        let (first, state) = max_len_seq(n, &vec![1; n], length / 2, None).unwrap();
        let (second, _) = max_len_seq(n, &state, length - length / 2, None).unwrap();
        assert_eq!(first.into_iter().chain(second).collect::<Vec<_>>(), seq);
    }
    let coefficients = [1., 2., -1., 3.];
    for x in [0.2, 0.7, 1.3, 2.4] {
        let cubic = cspline1d_eval(&coefficients, &[x, -x, 6. - x, x + 6000.], 1., 0.).unwrap();
        let quadratic = qspline1d_eval(&coefficients, &[x, -x, 6. - x, x + 6000.], 1., 0.).unwrap();
        for out in [cubic, quadratic] {
            for value in &out {
                near(*value, out[0], 1e-11, "spline mirror periodicity");
            }
        }
    }
}

#[test]
fn remez_preserves_linear_phase_and_scaling_conventions() {
    // The 255-tap boundary is compared with SciPy in the CSV fixtures.
    for n in [2, 3, 20, 35, 63] {
        let bands = [0., 0.3, 0.32, 1.];
        let h = remez(n, &bands, &[1., 0.], &[1., 10.], 2.)
            .unwrap_or_else(|error| panic!("{n} taps: {error}"));
        assert_eq!(h.len(), n);
        for (&x, &y) in h.iter().zip(h.iter().rev()) {
            near(x, y, 1e-13, "Remez type I/II symmetry");
        }
        let scaled = remez(n, &[0., 300., 320., 1000.], &[1., 0.], &[1., 10.], 2000.).unwrap();
        let gain = remez(n, &bands, &[-3., 0.], &[7., 70.], 2.).unwrap();
        for ((&x, y), z) in h.iter().zip(scaled).zip(gain) {
            near(x, y, 1e-11, "Remez sample-rate units");
            near(-3. * x, z, 1e-11, "Remez gain and common weight scaling");
        }
    }
    for n in [0, 1, 256, usize::MAX] {
        assert!(remez(n, &[0., 0.3, 0.5, 1.], &[1., 0.], &[1., 1.], 2.).is_err());
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(remez(9, &[0., 0.3, 0.5, 1.], &[value, 0.], &[1., 1.], 2.).is_err());
        assert!(remez(9, &[0., 0.3, 0.5, 1.], &[1., 0.], &[1., value], 2.).is_err());
        assert!(remez(9, &[0., 0.3, 0.5, 1.], &[1., 0.], &[1., 1.], value).is_err());
    }
    // Type II symmetry forces zero gain at Nyquist.
    for n in [2, 20] {
        assert!(remez(n, &[0., 0.3, 0.5, 1.], &[0., 1.], &[1., 1.], 2.).is_err());
    }
}

#[test]
fn fir_designs_and_envelopes_obey_frequency_and_energy_properties() {
    let h = remez(35, &[0., 0.3, 0.5, 1.], &[1., 0.], &[1., 1.], 2.).unwrap();
    for (&x, &y) in h.iter().zip(h.iter().rev()) {
        near(x, y, 1e-13, "linear phase symmetry");
    }
    let f: Vec<_> = (0..=10000)
        .map(|i| std::f64::consts::PI * i as f64 / 10000.)
        .collect();
    let response = freqz(&h, &[1.], &f).unwrap();
    let errors: Vec<_> = response
        .iter()
        .enumerate()
        .filter_map(|(i, z)| {
            let freq = i as f64 / 10000.;
            if freq > 0.3 && freq < 0.5 {
                return None;
            }
            let phase = 17. * f[i];
            let amplitude = z.re * phase.cos() - z.im * phase.sin();
            Some((freq, (amplitude - if freq <= 0.3 { 1. } else { 0. }).abs()))
        })
        .collect();
    let ripple = errors.iter().map(|(_, e)| *e).fold(0., f64::max);
    assert!(ripple < 0.001);
    let extrema: Vec<_> = errors
        .windows(3)
        .filter(|v| v[1].1 > v[0].1 && v[1].1 > v[2].1)
        .map(|v| v[1].1)
        .collect();
    assert!(extrema.len() >= 15);
    for e in extrema {
        near(e, ripple, 2e-8, "continuous minimax ripple");
    }
    let original = [0.1, 0.2, 0.4, 0.2, 0.1];
    let minimum = minimum_phase(&original, 16384, false).unwrap();
    let energy = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
    near(
        energy(&minimum),
        energy(&original),
        1e-7,
        "minimum-phase energy",
    );
    assert!(minimum[0].abs() > original[0].abs());
    let n = 32;
    let x: Vec<_> = (0..n)
        .map(|i| 2. * (2. * std::f64::consts::PI * 3. * i as f64 / n as f64).cos())
        .collect();
    let envelope = envelope(&x, (1, 16), n, false, true).unwrap();
    for a in envelope.amplitude {
        near(a, 2., 1e-13, "sinusoid envelope");
    }
    for r in envelope.residual {
        near(r, 0., 1e-13, "sinusoid residual");
    }
    let x: Vec<_> = (0..300).map(|i| (i as f64 * 0.1).sin()).collect();
    let kernel = [0.2, 0.6, 0.2];
    for mode in [Mode::Full, Mode::Same, Mode::Valid] {
        for (a, b) in oaconvolve(&x, &kernel, mode)
            .unwrap()
            .into_iter()
            .zip(convolve(&x, &kernel, mode).unwrap())
        {
            near(a, b, 1e-13, "overlap-add linear convolution");
        }
    }
}

#[test]
fn remaining_functions_reject_invalid_domains_and_shapes() {
    assert!(besselap(13, BesselNorm::Phase).is_err());
    assert!(
        bessel(
            2,
            &[0.3, 0.2],
            BandType::Bandpass,
            Some(2.),
            BesselNorm::Delay
        )
        .is_err()
    );
    assert!(ellipap(65, 1., 40.).is_err());
    assert!(ellipap(3, 40., 1.).is_err());
    assert!(ellip(3, 1., 40., &[1.], BandType::Lowpass, Some(2.)).is_err());
    assert!(ellipord(0.2, 0.2, 1., 40., Some(2.)).is_err());
    assert!(band_stop_obj(1., 2, [1., 4.], [2., 3.], 1., 40., IirFamily::Butterworth).is_err());
    assert!(
        iirdesign(
            0.2,
            0.3,
            1.,
            40.,
            Some(2.),
            IirFamily::Bessel(BesselNorm::Phase)
        )
        .is_err()
    );
    assert!(iircomb(30., 20., false, 200., false).is_err());
    assert!(iircomb(20., 0.1, false, 200., false).is_err());
    assert!(iircomb(20., 0.75, false, 200., false).is_err());
    assert!(gammatone(500., true, 4, 16, 1000.).is_err());
    assert!(gammatone(100., true, 0, 16, 1000.).is_err());
    assert!(gammatone(100., false, 4, 1, 1000.).is_err());
    assert!(findfreqs(&[1.], &[0.], 4).is_err());
    let bad = Complex {
        re: f64::NAN,
        im: 0.,
    };
    let pole = Complex { re: 0.5, im: 0. };
    assert!(unique_roots(&[bad], 1e-3, RootRepresentative::Mean).is_err());
    assert!(unique_roots(&[pole], 0., RootRepresentative::Mean).is_err());
    assert!(residue(&[1.], &[0.], 1e-3).is_err());
    assert!(residuez(&[1.], &[0., 1.], 1e-3).is_err());
    assert!(invres(&[pole], &[], &[]).is_err());
    assert!(invresz(&[bad], &[pole], &[]).is_err());
    assert!(
        zpk2sos(&Zpk {
            zeros: vec![],
            poles: vec![Complex { re: 0.5, im: 0.2 }],
            gain: 1.
        })
        .is_err()
    );
    assert!(tf2sos(&[1.], &[0.]).is_err());
    assert!(abcd_normalize(&[vec![1., 2.]], &[vec![1.]], &[vec![1.]], &[vec![0.]]).is_err());
    assert!(tf2ss(&[1., 1.], &[1.]).is_err());
    assert!(
        zpk2ss(&Zpk {
            zeros: vec![pole],
            poles: vec![],
            gain: 1.
        })
        .is_err()
    );
    let sys = tf2ss(&[1.], &[1., 1.]).unwrap();
    assert!(ss2tf(&sys, 1).is_err());
    assert!(ss2zpk(&sys, 0, 1).is_err());
    assert!(cont2discrete(&sys, 0., Discretization::ZeroOrderHold).is_err());
    assert!(dlsim(&sys, &[vec![1., 2.]], 1., &[0.]).is_err());
    assert!(lsim(&sys, &vec![vec![1.]; 3], &[0., 1., 3.], &[0.], true).is_err());
    assert!(step(&sys, &[-1.], 0).is_err());
    assert!(impulse(&sys, &[0.], 1).is_err());
    assert!(dstep(&sys, 1., 0, 0).is_err());
    assert!(dimpulse(&sys, 1., 2, 1).is_err());
    assert!(place_poles(&[vec![0.]], &[0.], &[pole]).is_err());
    assert!(firls(4, &[0., 0.3], &[1., 1.], &[1.], 2.).is_err());
    assert!(firls(5, &[0., 0.3], &[1.], &[1.], 2.).is_err());
    assert!(
        firwin2(
            5,
            &[0., 0.5, 0.5, 0.5, 1.],
            &[1.; 5],
            9,
            &Window::Hann,
            false,
            2.
        )
        .is_err()
    );
    assert!(firwin_2d((0, 5), &[0.3], true, &Window::Hann, &Window::Hann, true, 2.).is_err());
    assert!(minimum_phase(&[0.; 5], 32, true).is_err());
    assert!(minimum_phase(&[1., 2., 3.], 32, true).is_err());
    assert!(remez(10, &[0., 0.5, 0.4, 1.], &[1., 0.], &[1., 1.], 2.).is_err());
    assert!(cspline1d(&[1., 2., 3.], 0.001).is_err());
    assert!(qspline1d(&[]).is_err());
    assert!(cspline1d_eval(&[], &[1.], 1., 0.).is_err());
    assert!(qspline1d_eval(&[1.], &[1.], 0., 0.).is_err());
    assert!(cspline2d(&[vec![1., 2.], vec![1.]], 0., 1e-6).is_err());
    assert!(qspline2d(&vec![vec![1.; 4]; 4], 0.).is_err());
    assert!(spline_filter(&vec![vec![1.; 4]; 4], -1.).is_err());
    assert!(symiirorder1(&[1.; 8], 1., 1., 1e-6).is_err());
    assert!(symiirorder1(&[1.; 8], 1., 0.99, 1e-6).is_err());
    assert!(symiirorder2(&[1.; 8], 1., 0.2, 1e-6).is_err());
    assert!(whittaker_henderson(&[1., 2., 3.], 1., 2, &[0.; 3]).is_err());
    assert!(choose_conv_method(&[], &[1.], Mode::Full).is_err());
    assert!(oaconvolve(&[f64::INFINITY], &[1.], Mode::Full).is_err());
    assert!(closest_STFT_dual_window(&[0.; 4], 2, &[1.; 4], true).is_err());
    assert!(envelope(&[1.; 4], (0, 3), 4, false, true).is_err());
    assert!(max_len_seq(3, &[0; 3], 10, None).is_err());
    assert!(max_len_seq(3, &[1; 3], 10, Some(&[2, 2])).is_err());
    assert!(find_peaks_cwt(&[1.; 10], &[2., 1.]).is_err());
    assert!(find_peaks_cwt(&[f64::NAN], &[1.]).is_err());
    let many_peaks: Vec<_> = (0..8192)
        .map(|i| if i % 2 == 0 { 1. } else { 0. })
        .collect();
    assert!(matches!(
        find_peaks_cwt(&many_peaks, &[0.3, 0.5]),
        Err(Error::InvalidInput(
            "numerical work exceeds 16777216 operations"
        ))
    ));
}

#[test]
fn cropped_convolutions_match_scipy_for_both_length_orderings() {
    let inputs = common::lines(include_str!("fixtures/cropped_convolution_inputs.csv"));
    let expected: Vec<_> =
        common::lines(include_str!("fixtures/cropped_convolution_expected.csv")).collect();
    assert_eq!(inputs.clone().count(), expected.len());
    for (input, expected) in inputs.zip(expected) {
        let row = common::fields(input);
        let reference = common::fields(expected);
        assert_eq!(row[0], reference[0]);
        let x = common::vector(row[2]);
        let h = common::vector(row[3]);
        let mode = match row[4] {
            "full" => sci::signal::Mode::Full,
            "same" => sci::signal::Mode::Same,
            "valid" => sci::signal::Mode::Valid,
            _ => panic!("unknown mode"),
        };
        let function = if row[1] == "convolve" {
            sci::signal::convolve
        } else {
            sci::signal::correlate
        };
        assert_eq!(
            function(&x, &h, mode).unwrap(),
            common::vector(reference[1]),
            "{input}"
        );
    }
}
