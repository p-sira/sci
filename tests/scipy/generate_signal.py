# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Generate checked-in signal fixtures independently using SciPy.

Run: uv run tests/scipy/generate_signal.py (or the pinned project .venv).
Inputs live in signal_cases.json; expected values are never computed by Rust.
CSV input arrays use semicolons and matrix rows use colons. Output rows use
slashes; complex vectors are interleaved real/imaginary components. Every
output row's length is checked by the Rust fixture test, including empty rows.
"""

import csv
import json
from pathlib import Path
import warnings

import numpy as np
import scipy
from scipy import signal

if (np.__version__, scipy.__version__) != ("2.5.4", "1.18.1"):
    raise RuntimeError("Use NumPy 2.5.4 and SciPy 1.18.1")
ROOT = Path(__file__).resolve().parents[1]


def convert(value):
    if isinstance(value, list):
        return [convert(v) for v in value]
    if value in ("nan", "inf", "-inf"):
        return float(value)
    return value


def encode(value):
    if isinstance(value, list):
        separator = ":" if value and isinstance(value[0], list) else ";"
        return separator.join(encode(v) for v in value)
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def output_rows(value):
    if isinstance(value, tuple):
        return [row for component in value for row in output_rows(component)]
    array = np.asarray(value)
    if array.ndim > 1:
        return [row for component in array for row in output_rows(component)]
    array = array.reshape(-1)
    if np.iscomplexobj(array):
        return [np.column_stack((array.real, array.imag)).reshape(-1).tolist()]
    return [array.astype(float).tolist()]


def reference(name, a):
    if name == "unique_roots":
        roots, m = signal.unique_roots(
            [complex(*z) for z in a[0]], tol=a[1], rtype=a[2]
        )
        return np.asarray(roots, dtype=complex), m
    if name in ("residue", "residuez"):
        r, p, k = getattr(signal, name)(*a[:2], tol=a[2], rtype="avg")
        # Equal-magnitude conjugates sort lexicographically to avoid backend ordering differences.
        # Stable blocks preserve ascending powers for repeated poles.
        groups = []
        i = 0
        while i < len(p):
            j = i + 1
            while j < len(p) and p[j] == p[i]:
                j += 1
            groups.append((p[i], r[i:j]))
            i = j
        groups.sort(key=lambda v: (abs(v[0]), v[0].real, v[0].imag))
        return (
            np.asarray([r for p, rs in groups for r in rs], dtype=complex),
            np.asarray([p for p, rs in groups for r in rs], dtype=complex),
            k,
        )
    if name in ("invres", "invresz"):
        b, den = getattr(signal, name)(
            np.asarray([complex(*z) for z in a[0]]),
            np.asarray([complex(*z) for z in a[1]]),
            a[2],
            tol=1e-12,
            rtype="avg",
        )
        b = np.real_if_close(b)
        den = np.real_if_close(den)
        b = np.trim_zeros(b, "b" if name.endswith("z") else "f")
        return b if len(b) else np.array([0.0]), den
    if name in ("tf2sos", "zpk2sos"):
        if name == "tf2sos":
            sos = signal.tf2sos(*a)
        else:
            sos = signal.zpk2sos(
                [complex(*z) for z in a[0][0]], [complex(*z) for z in a[0][1]], a[0][2]
            )
        return len(sos), signal.freqz_sos(sos, worN=[0.0, 0.2, 0.9, 2.0, 3.0])[1]
    if name == "firls":
        return signal.firls(a[0], a[1], a[2], weight=a[3], fs=a[4])
    if name == "firwin2":
        return signal.firwin2(
            a[0], a[1], a[2], nfreqs=a[3], window=a[4], antisymmetric=a[5], fs=a[6]
        )
    if name == "firwin_2d":
        return np.outer(
            signal.firwin(
                a[0][0], a[1], pass_zero=a[2], window=a[3], scale=a[5], fs=a[6]
            ),
            signal.firwin(
                a[0][1], a[1], pass_zero=a[2], window=a[4], scale=a[5], fs=a[6]
            ),
        )
    if name == "minimum_phase":
        return signal.minimum_phase(a[0], n_fft=a[1], half=a[2], method="homomorphic")
    if name == "remez":
        return signal.remez(
            a[0], a[1], a[2], weight=a[3], fs=a[4], grid_density=32768, maxiter=100
        )
    if name == "qspline1d":
        return signal.qspline1d(np.asarray(a[0]))
    if name in (
        "cspline1d",
        "cspline1d_eval",
        "qspline1d_eval",
        "symiirorder1",
        "symiirorder2",
        "cspline2d",
    ):
        return getattr(signal, name)(np.asarray(a[0], dtype=float), *a[1:])
    if name == "qspline2d":
        return signal.qspline2d(np.asarray(a[0]), precision=a[1])
    if name == "spline_filter":
        return signal.spline_filter(np.asarray(a[0]), lmbda=a[1])
    if name == "whittaker_henderson":
        return signal.whittaker_henderson(a[0], lamb=a[1], order=a[2], weights=a[3]).x
    if name == "choose_conv_method":
        return 1 if signal.choose_conv_method(a[0], a[1], mode=a[2]) == "fft" else 0
    if name == "oaconvolve":
        return signal.oaconvolve(a[0], a[1], mode=a[2])
    if name == "closest_STFT_dual_window":
        return signal.closest_STFT_dual_window(
            np.asarray(a[0]), a[1], np.asarray(a[2]), scaled=a[3]
        )
    if name == "envelope":
        return signal.envelope(
            np.asarray(a[0]),
            bp_in=tuple(a[1]),
            n_out=a[2],
            squared=a[3],
            residual="all" if a[4] else "lowpass",
        )
    if name == "max_len_seq":
        return signal.max_len_seq(a[0], state=a[1], length=a[2], taps=a[3])
    if name == "find_peaks_cwt":
        return signal.find_peaks_cwt(a[0], a[1])
    if name in ("besselap", "ellipap"):
        if name == "besselap":
            z, p, k = signal.besselap(a[0], norm={"magnitude": "mag"}.get(a[1], a[1]))
            p = np.asarray(sorted(p, key=lambda z: (z.real, z.imag)), dtype=complex)
        else:
            z, p, k = signal.ellipap(*a)
        return np.asarray(z, dtype=complex), np.asarray(p, dtype=complex), k
    if name == "bessel":
        return signal.bessel(
            a[0],
            a[1][0] if len(a[1]) == 1 else a[1],
            btype=a[2],
            analog=a[3] is None,
            fs=a[3],
            norm={"magnitude": "mag"}.get(a[4], a[4]),
        )
    if name == "ellip":
        return signal.ellip(
            *a[:3],
            a[3][0] if len(a[3]) == 1 else a[3],
            btype=a[4],
            analog=a[5] is None,
            fs=a[5],
        )
    if name == "ellipord":
        return signal.ellipord(*a[:4], analog=a[4] is None, fs=a[4])
    if name == "band_stop_obj":
        return signal.band_stop_obj(
            a[0],
            a[1],
            np.asarray(a[2]),
            np.asarray(a[3]),
            a[4],
            a[5],
            "cheby" if a[6].startswith("cheby") else a[6],
        )
    if name == "iirdesign":
        return signal.iirdesign(*a[:4], analog=a[4] is None, fs=a[4], ftype=a[5])
    if name == "iircomb":
        return signal.iircomb(
            a[0], a[1], ftype="peak" if a[2] else "notch", fs=a[3], pass_zero=a[4]
        )
    if name == "gammatone":
        return signal.gammatone(
            a[0],
            "fir" if a[1] else "iir",
            order=a[2] if a[1] else None,
            numtaps=a[3] if a[1] else None,
            fs=a[4],
        )
    if name == "abcd_normalize":
        return signal.abcd_normalize(*a)
    if name == "zpk2ss":
        return signal.zpk2ss(
            [complex(*z) for z in a[0][0]], [complex(*z) for z in a[0][1]], a[0][2]
        )
    if name == "ss2tf":
        return signal.ss2tf(*a[:4], input=a[4])
    if name == "ss2zpk":
        b, den = signal.ss2tf(*a[:4], input=a[4])
        z, p, k = signal.tf2zpk(b[a[5]], den)
        sort = lambda v: np.asarray(
            sorted(v, key=lambda z: (z.real, z.imag)), dtype=complex
        )
        return sort(z), sort(p), k
    if name == "cont2discrete":
        return signal.cont2discrete(
            tuple(np.asarray(v) for v in a[:4]), a[4], method=a[5]
        )[:4]
    if name == "place_poles":
        return signal.place_poles(
            np.asarray(a[0]), np.asarray(a[1])[:, None], [complex(*z) for z in a[2]]
        ).gain_matrix[0]
    if name in ("lsim", "dlsim"):
        system = tuple(np.asarray(v) for v in a[:4])
        if name == "lsim":
            t, y, x = signal.lsim(system, np.asarray(a[4]), a[5], X0=a[6], interp=a[7])
        else:
            t, y, x = signal.dlsim((*system, a[5]), np.asarray(a[4]), x0=a[6])
        return (
            t,
            np.asarray(y).reshape(len(t), len(a[3])),
            np.asarray(x).reshape(len(t), len(a[0])),
        )
    if name in ("step", "impulse", "dstep", "dimpulse"):
        system = tuple(np.asarray(v) for v in a[:4])
        input = a[-1]
        if name.startswith("d"):
            dt, length = a[4:6]
            u = np.zeros((length, len(a[3][0])))
            if name == "dstep":
                u[:, input] = 1.0
            else:
                u[0, input] = 1.0
            t, y, x = signal.dlsim((*system, dt), u)
        else:
            t = np.asarray(a[4])
            u = np.zeros((len(t), len(a[3][0])))
            if name == "step":
                u[:, input] = 1.0
                initial = np.zeros(len(a[0]))
            else:
                initial = np.asarray(a[1])[:, input]
            t, y, x = signal.lsim(system, u, t, X0=initial, interp=False)
        return (
            t,
            np.asarray(y).reshape(len(t), len(a[3])),
            np.asarray(x).reshape(len(t), len(a[0])),
        )
    if name in (
        "lp2lp_zpk",
        "lp2hp_zpk",
        "lp2bp_zpk",
        "lp2bs_zpk",
        "bilinear_zpk",
        "zpk2tf",
        "freqs_zpk",
        "freqz_zpk",
    ):

        def roots(v):
            return np.asarray([complex(*z) for z in v])

        zpk = a[0]
        args = [roots(zpk[0]), roots(zpk[1]), zpk[2], *a[1:]]
        result = getattr(signal, name)(*args)
        if name in ("freqs_zpk", "freqz_zpk"):
            return result[1]
        if name == "zpk2tf":
            return result
        sort = lambda v: np.asarray(
            sorted(v, key=lambda z: (complex(z).real, complex(z).imag)), dtype=complex
        )
        return sort(result[0]), sort(result[1]), result[2]
    if name in ("tf2zpk", "sos2zpk"):
        z, p, k = getattr(signal, name)(*a)

        def sorted_roots(v):
            return np.asarray(sorted(v, key=lambda z: (z.real, z.imag)), dtype=complex)

        if name == "sos2zpk":
            # Sci groups roots by section, with sorted section roots.
            sections = [signal.tf2zpk(row[:3], row[3:]) for row in a[0]]
            return (
                np.concatenate([sorted_roots(z) for z, p, k in sections]),
                np.concatenate([sorted_roots(p) for z, p, k in sections]),
                np.prod([k for z, p, k in sections]),
            )
        return sorted_roots(z), sorted_roots(p), k
    if name in ("butter", "cheby1", "cheby2"):
        args = a[:-2].copy()
        if len(args[-1]) == 1:
            args[-1] = args[-1][0]
        return getattr(signal, name)(*args, btype=a[-2], analog=a[-1] is None, fs=a[-1])
    if name == "iirfilter":
        return signal.iirfilter(
            a[0],
            a[1][0] if len(a[1]) == 1 else a[1],
            btype=a[2],
            fs=a[3],
            ftype=a[4],
            rp=1.0,
            rs=40.0,
        )
    if name in ("buttord", "cheb1ord", "cheb2ord"):
        return getattr(signal, name)(*a[:-1], analog=a[-1] is None, fs=a[-1])
    if name in ("bode", "dbode", "freqresp", "dfreqresp"):
        digital = name.startswith("d")
        if name.endswith("freqresp"):
            return (signal.freqz if digital else signal.freqs)(a[0], a[1], a[2])[1]
        if digital:
            # dbode accepts descending z polynomials; pad numerator at the
            # end to reference Sci ascending-delay coefficients.
            b = np.pad(a[0], (0, max(0, len(a[1]) - len(a[0]))))
            return signal.dbode((b, a[1], 1.0), w=a[2])[1:]
        return signal.bode((a[0], a[1]), w=a[2])[1:]
    if name in ("convolve2d", "correlate2d"):
        return getattr(signal, name)(
            np.asarray(a[0]), np.asarray(a[1]), mode=a[2], boundary=a[3], fillvalue=a[4]
        )
    if name == "sepfir2d":
        return signal.sepfir2d(np.asarray(a[0]), np.asarray(a[1]), np.asarray(a[2]))
    if name == "medfilt2d":
        return signal.medfilt2d(np.asarray(a[0]), kernel_size=a[1])
    if name == "hilbert2":
        return signal.hilbert2(np.asarray(a[0]))
    if name == "resample_poly":
        return signal.resample_poly(
            *a[:3], window=("kaiser", 5.0) if a[3] is None else np.asarray(a[3])
        )
    if name == "decimate":
        return signal.decimate(a[0], a[1], n=a[2], ftype="fir", zero_phase=a[3])
    if name.startswith("windows."):
        name = name.removeprefix("windows.")
        if name == "get_window":
            return signal.windows.get_window(a[0], a[1], fftbins=a[2])
        if name == "dpss":
            return signal.windows.dpss(a[0], a[1], Kmax=a[2], sym=a[3], norm=2)
        return getattr(signal.windows, name)(*a[:-1], sym=a[-1])
    if name == "get_window":
        return signal.get_window(a[0], a[1], fftbins=a[2])
    if name == "detrend":
        return signal.detrend(a[0], type="linear" if a[1] else "constant")
    if name == "gausspulse":
        return signal.gausspulse(*a, retquad=True, retenv=True)
    if name == "lfilter_with_state":
        return signal.lfilter(*a[:3], zi=a[3])
    if name in ("filtfilt", "sosfiltfilt"):
        return getattr(signal, name)(*a[:-1], padlen=a[-1])
    if name in ("freqz", "freqs", "freqz_sos", "sosfreqz", "group_delay"):
        if name == "group_delay":
            return signal.group_delay((a[0], a[1]), w=a[2])[1]
        return getattr(signal, name)(*a)[1]
    if name == "firwin":
        return signal.firwin(
            a[0], a[1], pass_zero=a[2], window=a[3], scale=a[4], fs=a[5]
        )
    if name == "order_filter":
        return signal.order_filter(a[0], np.ones(a[1]), a[2])
    if name in ("czt", "czt_points"):
        if name == "czt":
            x = np.asarray(a[0])
            x = x[:, 0] + 1j * x[:, 1]
            return signal.czt(
                x, m=a[1], w=None if a[2] is None else complex(*a[2]), a=complex(*a[3])
            )
        return signal.czt_points(
            a[0], w=None if a[1] is None else complex(*a[1]), a=complex(*a[2])
        )
    if name == "zoom_fft":
        x = np.asarray(a[0])
        return signal.zoom_fft(
            x[:, 0] + 1j * x[:, 1], [a[1], a[2]], m=a[3], fs=a[4], endpoint=a[5]
        )
    if name == "periodogram":
        return signal.periodogram(
            a[0],
            window=a[1],
            nfft=a[2],
            fs=a[3],
            detrend="constant" if a[4] else False,
            scaling=a[5],
        )
    if name in ("welch", "csd", "spectrogram", "coherence", "stft", "istft"):
        offset = int(name in ("csd", "coherence"))
        args = [np.asarray(v) for v in a[: offset + 1]]
        w, overlap, nfft, fs = a[offset + 1 : offset + 5]
        kw = dict(
            window=np.asarray(w), nperseg=len(w), noverlap=overlap, nfft=nfft, fs=fs
        )
        if name == "istft":
            z = signal.stft(*args, **kw)[2]
            return signal.istft(z, **kw)
        if name in ("welch", "csd", "spectrogram"):
            kw.update(detrend="constant" if a[-2] else False, scaling=a[-1])
        return getattr(signal, name)(*args, **kw)
    if name.lower() in ("check_cola", "check_nola"):
        f = signal.check_COLA if name.lower() == "check_cola" else signal.check_NOLA
        return f(a[0], len(a[0]), a[1], tol=a[2])
    if name in ("argrelmax", "argrelmin", "argrelextrema"):
        if name == "argrelextrema":
            return signal.argrelextrema(
                np.asarray(a[0]), np.greater, order=a[1], mode=a[2]
            )[0]
        return getattr(signal, name)(np.asarray(a[0]), order=a[1], mode=a[2])[0]
    if name == "find_peaks":
        return signal.find_peaks(a[0])[0]
    if name == "peak_prominences":
        return signal.peak_prominences(a[0], a[1], wlen=a[2])
    if name == "peak_widths":
        return signal.peak_widths(a[0], a[1], rel_height=a[2], wlen=a[3])
    return getattr(signal, name)(*a)


cases = json.loads((ROOT / "scipy/signal_cases.json").read_text())
(ROOT / "fixtures").mkdir(exist_ok=True)
with (
    (ROOT / "fixtures/signal_inputs.csv").open("w", newline="") as inputs,
    (ROOT / "fixtures/signal_expected.csv").open("w", newline="") as expected,
):
    iw = csv.writer(inputs, lineterminator="\n")
    ew = csv.writer(expected, lineterminator="\n")
    iw.writerow(["case", "function", *[f"arg{i}" for i in range(8)]])
    ew.writerow(["case", "rows"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, case in enumerate(cases):
            args = [convert(a) for a in case["arguments"]]
            result = reference(case["function"], args)
            encoded = [encode(a) for a in args]
            if case["function"] in (
                "lp2lp_zpk",
                "lp2hp_zpk",
                "lp2bp_zpk",
                "lp2bs_zpk",
                "bilinear_zpk",
                "zpk2tf",
                "freqs_zpk",
                "freqz_zpk",
                "zpk2ss",
                "zpk2sos",
            ):
                encoded[0] = "|".join(encode(v) for v in args[0])
            iw.writerow([i, case["function"], *encoded, *[""] * (8 - len(args))])
            ew.writerow([i, "/".join(encode(row) for row in output_rows(result))])
print(f"Generated {len(cases)} signal reference cases")
