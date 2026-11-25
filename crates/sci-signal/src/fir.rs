use super::windows::Window;
use super::{checked, fft, firwin, get_window, ifft, nonempty, positive, samples, size, work};
use crate::{
    Error, Result,
    complex::{Complex, native, owned},
};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64 as C;
use std::f64::consts::PI;

fn taps(n: usize) -> Result<()> {
    if n == 0 || n > 1023 {
        return Err(Error::InvalidInput("FIR length must be 1..1023"));
    }
    Ok(())
}
fn bands_validate(bands: &[f64], desired: &[f64], weight: &[f64], fs: f64) -> Result<()> {
    positive(fs)?;
    nonempty(bands)?;
    samples(desired)?;
    samples(weight)?;
    if !bands.len().is_multiple_of(2)
        || desired.len() != bands.len() / 2
        || weight.len() != desired.len()
        || bands.windows(2).any(|p| p[0] >= p[1])
        || bands[0] < 0.
        || bands[bands.len() - 1] > fs / 2.
        || weight.iter().any(|w| *w <= 0.)
    {
        return Err(Error::InvalidInput(
            "increasing band pairs, one desired gain and positive weight per band required",
        ));
    }
    work(bands.len(), bands.len())?;
    Ok(())
}
fn icos(k: usize, lo: f64, hi: f64) -> f64 {
    if k == 0 {
        hi - lo
    } else {
        ((k as f64 * hi).sin() - (k as f64 * lo).sin()) / k as f64
    }
}
fn iwcos(k: usize, lo: f64, hi: f64) -> f64 {
    if k == 0 {
        (hi * hi - lo * lo) / 2.
    } else {
        let k = k as f64;
        (hi * (k * hi).sin() - lo * (k * lo).sin()) / k
            + ((k * hi).cos() - (k * lo).cos()) / (k * k)
    }
}

fn remez_two_taps(
    bands: &[f64],
    desired: &[f64],
    weight: &[f64],
    fs: f64,
    amplitude: f64,
    weight_scale: f64,
) -> Result<Vec<f64>> {
    // A(f) = 2*a*cos(pi*f/fs). On each band its weighted error reaches
    // its maximum at an endpoint. For endpoint constraints |s_i*a-t_i| <= r,
    // the smallest feasible r is max_ij |s_j*t_i-s_i*t_j|/(s_i+s_j).
    // A nonzero slope always exists for validated, strictly increasing bands.
    // Intersect their feasible intervals to recover a minimizing tap.
    let endpoints: Vec<_> = bands
        .iter()
        .enumerate()
        .map(|(i, &f)| {
            let w = weight[i / 2] / weight_scale;
            let cosine = if f == fs / 2. {
                0.
            } else {
                (PI * (f / fs)).cos()
            };
            (2. * w * cosine, w * (desired[i / 2] / amplitude))
        })
        .collect();
    let mut ripple: f64 = 0.;
    for &(s, t) in &endpoints {
        for &(u, v) in &endpoints {
            if s + u > 0. {
                ripple = ripple.max((u * t - s * v).abs() / (s + u));
            }
        }
    }
    let (mut lower, mut upper) = (f64::NEG_INFINITY, f64::INFINITY);
    for (s, t) in endpoints {
        if s > 0. {
            lower = lower.max((t - ripple) / s);
            upper = upper.min((t + ripple) / s);
        }
    }
    let tap = (lower / 2. + upper / 2.) * amplitude;
    checked(vec![tap; 2])
}

/// Minimizes `sum_b w_b integral_b (D(omega)-sum_j a_j cos(j omega))^2 d omega`.
///
/// ## Parameters
///
/// numtaps is odd in 1..511. bands contains increasing frequency pairs in fs units; desired gives a gain at each band endpoint, interpolated linearly; weight has one positive entry per band. Returns symmetric type-I FIR coefficients, length numtaps.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Integrates the normal equations analytically and delegates the positive-definite solve to private nalgebra Cholesky. Tiny specified bands can make the system singular or ill conditioned.
///
/// # Examples
///
/// ```
/// use sci_signal::fir::*;
/// let _=firls(11,&[0.,0.3,0.5,1.],&[1.,1.,0.,0.],&[1.,1.],2.)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy firls](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firls.html).
pub fn firls(
    numtaps: usize,
    bands: &[f64],
    desired: &[f64],
    weight: &[f64],
    fs: f64,
) -> Result<Vec<f64>> {
    taps(numtaps)?;
    if numtaps.is_multiple_of(2) || numtaps > 511 {
        return Err(Error::InvalidInput("firls requires odd length <=511"));
    }
    if desired.len() != bands.len() {
        return Err(Error::DimensionMismatch);
    }
    bands_validate(bands, &vec![0.; bands.len() / 2], weight, fs)?;
    samples(desired)?;
    let m = numtaps / 2;
    work((m + 1) * (m + 1), bands.len())?;
    let mut q = DMatrix::zeros(m + 1, m + 1);
    let mut rhs = DVector::zeros(m + 1);
    for band in 0..weight.len() {
        let lo = 2. * PI * bands[2 * band] / fs;
        let hi = 2. * PI * bands[2 * band + 1] / fs;
        let slope = (desired[2 * band + 1] - desired[2 * band]) / (hi - lo);
        let intercept = desired[2 * band] - slope * lo;
        for i in 0..=m {
            rhs[i] += weight[band] * (intercept * icos(i, lo, hi) + slope * iwcos(i, lo, hi));
            for j in 0..=m {
                q[(i, j)] +=
                    weight[band] * 0.5 * (icos(i.abs_diff(j), lo, hi) + icos(i + j, lo, hi));
            }
        }
    }
    let a = q
        .cholesky()
        .ok_or(Error::NumericalFailure("singular FIR least squares"))?
        .solve(&rhs);
    let mut h = vec![0.; numtaps];
    h[m] = a[0];
    for j in 1..=m {
        h[m - j] = a[j] / 2.;
        h[m + j] = a[j] / 2.;
    }
    checked(h)
}

/// Designs a linear-phase FIR by interpolating a requested gain and applying an inverse FFT and taper.
///
/// ## Parameters
///
/// numtaps is 1..1023; freq/gain have equal length, freq runs from 0 to fs/2 and is nondecreasing. Interior frequencies may repeat twice for jumps. nfreqs>numtaps sets the uniform mesh, bounded by FFT length. window is a symmetric typed taper; antisymmetric chooses type III/IV and enforces endpoint zeros. Returns numtaps taps.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses private RustFFT. Mesh resolution controls interpolation error; repeated frequencies are separated by machine epsilon times Nyquist as in SciPy.
///
/// # Examples
///
/// ```
/// use sci_signal::fir::*;
/// use sci_signal::windows;
/// let _=firwin2(9,&[0.,0.4,0.6,1.],&[1.,1.,0.,0.],17,&windows::Window::Hamming,false,2.)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy firwin2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin2.html).
pub fn firwin2(
    numtaps: usize,
    freq: &[f64],
    gain: &[f64],
    nfreqs: usize,
    window: &Window,
    antisymmetric: bool,
    fs: f64,
) -> Result<Vec<f64>> {
    taps(numtaps)?;
    positive(fs)?;
    nonempty(freq)?;
    samples(gain)?;
    if freq.len() < 2
        || freq.len() != gain.len()
        || nfreqs <= numtaps
        || freq[0] != 0.
        || freq[freq.len() - 1] != fs / 2.
        || freq.windows(2).any(|v| v[0] > v[1])
        || freq.windows(3).any(|v| v[0] == v[2])
        || freq[1] == 0.
        || freq[freq.len() - 2] == fs / 2.
    {
        return Err(Error::InvalidInput("invalid frequency mesh or endpoints"));
    }
    let fftlen = nfreqs
        .checked_sub(1)
        .and_then(|n| n.checked_mul(2))
        .ok_or(Error::InvalidInput("FFT size overflow"))?;
    size(fftlen)?;
    if (!antisymmetric && numtaps.is_multiple_of(2) && gain[gain.len() - 1] != 0.)
        || (antisymmetric && (gain[0] != 0. || (numtaps % 2 == 1 && gain[gain.len() - 1] != 0.)))
    {
        return Err(Error::InvalidInput(
            "linear-phase type requires zero endpoint gain",
        ));
    }
    let mut freq = freq.to_vec();
    for k in 0..freq.len() - 1 {
        if freq[k] == freq[k + 1] {
            freq[k] -= f64::EPSILON * fs / 2.;
            freq[k + 1] += f64::EPSILON * fs / 2.;
        }
    }
    if freq.windows(2).any(|v| v[0] >= v[1]) {
        return Err(Error::InvalidInput("frequency spacing is too small"));
    }
    let mut spectrum = vec![C::new(0., 0.); fftlen];
    let mut j = 0;
    for (i, z) in spectrum.iter_mut().enumerate().take(nfreqs) {
        let x = (fs / 2.) * i as f64 / (nfreqs - 1) as f64;
        while j + 2 < freq.len() && x > freq[j + 1] {
            j += 1;
        }
        let g = gain[j] + (gain[j + 1] - gain[j]) * (x - freq[j]) / (freq[j + 1] - freq[j]);
        *z = C::from_polar(g, -((numtaps - 1) as f64) / 2. * PI * x / (fs / 2.));
        if antisymmetric {
            *z *= C::new(0., 1.);
        }
    }
    // irfft discards imaginary components at self-conjugate bins.
    spectrum[0].im = 0.;
    spectrum[nfreqs - 1].im = 0.;
    for i in 1..nfreqs - 1 {
        spectrum[fftlen - i] = spectrum[i].conj();
    }
    let y = ifft(&spectrum.into_iter().map(owned).collect::<Vec<_>>());
    let w = get_window(window, numtaps, false)?;
    let mut h: Vec<_> = y
        .iter()
        .zip(w)
        .map(|(z, w)| z.re * w)
        .take(numtaps)
        .collect();
    if antisymmetric && numtaps % 2 == 1 {
        h[numtaps / 2] = 0.;
    }
    checked(h)
}

/// Constructs a separable two-dimensional FIR `h[i,j]=h_row[i] h_col[j]`.
///
/// ## Parameters
///
/// shape=(rows,columns), each in 1..1023; cutoffs/pass_zero/scale/fs follow firwin. row_window and column_window are typed symmetric tapers. Returns row-major shape. SciPy circular radial interpolation is unsupported.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic; cancellation and ill conditioning can reduce accuracy.
///
/// # Examples
///
/// ```
/// use sci_signal::fir::*;
/// use sci_signal::windows;
/// let _=firwin_2d((5,7),&[0.3],true,&windows::Window::Hamming,&windows::Window::Hann,true,2.)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy firwin_2d](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin_2d.html).
pub fn firwin_2d(
    shape: (usize, usize),
    cutoffs: &[f64],
    pass_zero: bool,
    row_window: &Window,
    column_window: &Window,
    scale: bool,
    fs: f64,
) -> Result<Vec<Vec<f64>>> {
    taps(shape.0)?;
    taps(shape.1)?;
    work(shape.0, shape.1)?;
    let row = firwin(shape.0, cutoffs, pass_zero, row_window, scale, fs)?;
    let col = firwin(shape.1, cutoffs, pass_zero, column_window, scale, fs)?;
    row.into_iter()
        .map(|r| checked(col.iter().map(|c| r * c).collect()))
        .collect()
}

/// Constructs a minimum-phase FIR from a real cepstrum: `h_min=IFFT(exp(FFT((2u-delta)*IFFT(log|H|))))`.
///
/// ## Parameters
///
/// h is finite, nonzero, symmetric linear-phase taps, length>=3. nfft>=h.len() is explicit and <=1048576. half=true halves the log magnitude and returns ceil(length/2), otherwise preserves magnitude and length. Supports the homomorphic method only.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses private RustFFT and SciPy magnitude regularization (1e-7 times the least positive spectral magnitude). Finite FFT length and zeros on the unit circle limit accuracy.
///
/// # Examples
///
/// ```
/// use sci_signal::fir::*;
/// let _=minimum_phase(&[0.25,0.5,0.25],128,true)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy minimum_phase](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.minimum_phase.html).
pub fn minimum_phase(h: &[f64], nfft: usize, half: bool) -> Result<Vec<f64>> {
    nonempty(h)?;
    size(nfft)?;
    if h.len() < 3
        || nfft < h.len()
        || h.iter().all(|v| *v == 0.)
        || h.iter()
            .zip(h.iter().rev())
            .any(|(a, b)| (a - b).abs() > 1e-8 + 1e-6 * a.abs())
    {
        return Err(Error::InvalidInput(
            "nonzero symmetric FIR and nfft>=length>=3 required",
        ));
    }
    let mut x = vec![Complex { re: 0., im: 0. }; nfft];
    for (i, &v) in h.iter().enumerate() {
        x[i].re = v;
    }
    let spectrum = fft(&x);
    let mag: Vec<_> = spectrum.into_iter().map(|z| native(z).norm()).collect();
    let eps = 1e-7
        * mag
            .iter()
            .filter(|&&v| v > 0.)
            .copied()
            .fold(f64::INFINITY, f64::min);
    let logmag: Vec<_> = mag
        .into_iter()
        .map(|v| Complex {
            re: (v + eps).ln() * if half { 0.5 } else { 1. },
            im: 0.,
        })
        .collect();
    let cep = ifft(&logmag);
    let mut causal = vec![Complex { re: 0., im: 0. }; nfft];
    causal[0].re = cep[0].re;
    for i in 1..nfft / 2 {
        causal[i].re = 2. * cep[i].re;
    }
    causal[nfft / 2].re = cep[nfft / 2].re * (1 + nfft % 2) as f64;
    let x: Vec<_> = fft(&causal)
        .into_iter()
        .map(|z| owned(native(z).exp()))
        .collect();
    let out = ifft(&x);
    checked(
        out.into_iter()
            .take(if half { h.len().div_ceil(2) } else { h.len() })
            .map(|z| z.re)
            .collect(),
    )
}

/// Finds a symmetric minimax FIR minimizing
/// `max_b sup_{f in band_b} weight_b |A(f) - desired_b|`, where the zero-phase
/// amplitude is `A(f) = sum_k h[k] cos(2*pi*f*(k-(numtaps-1)/2)/fs)`.
///
/// ## Parameters
///
/// numtaps is 2..255; bands are strictly increasing pairs in fs units with one constant desired gain and positive weight per band. Returns symmetric type I/II taps. Bandpass amplitude design is supported; differentiator and Hilbert dispatch are unsupported.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
/// Even-length designs require zero desired gain in a band ending at Nyquist
/// (`fs/2`), because type II symmetry forces the amplitude there to zero.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Two-tap filters solve the scalar weighted minimax problem directly at band
/// endpoints, with quadratic work in the number of endpoints. Larger filters
/// delegate the exchange algorithm to private pm-remez, reusing nalgebra for
/// eigenvalues, with at most 100 exchanges and degree-24 Chebyshev proxies.
/// The backend controls Schur convergence internally and exposes no separate
/// eigenvalue iteration limit. Its continuous extrema differ from SciPy grid
/// extrema; filters agree in minimax behavior, not necessarily every coefficient.
/// Failure to converge is an error.
///
/// # Examples
///
/// ```
/// use sci_signal::fir::*;
/// let _=remez(21,&[0.,0.3,0.5,1.],&[1.,0.],&[1.,1.],2.)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [Parks-McClellan backend](https://pm-remez.readthedocs.io/en/latest/)
/// and [SciPy remez](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.remez.html).
pub fn remez(
    numtaps: usize,
    bands: &[f64],
    desired: &[f64],
    weight: &[f64],
    fs: f64,
) -> Result<Vec<f64>> {
    if !(2..=255).contains(&numtaps) {
        return Err(Error::InvalidInput("Remez length must be 2..255"));
    }
    bands_validate(bands, desired, weight, fs)?;
    work(numtaps * numtaps, bands.len())?;
    let amplitude = desired.iter().map(|v| v.abs()).fold(0., f64::max);
    if amplitude == 0. {
        return Ok(vec![0.; numtaps]);
    }
    let weight_scale = weight.iter().copied().fold(0., f64::max);
    if numtaps == 2 {
        if bands[bands.len() - 1] == fs / 2. && desired[desired.len() - 1] != 0. {
            return Err(Error::InvalidInput("incompatible Remez constraints"));
        }
        return remez_two_taps(bands, desired, weight, fs, amplitude, weight_scale);
    }
    use pm_remez::{BandSetting, ParametersBuilder, constant, pm_parameters, pm_remez};
    let settings: Vec<_> = bands
        .as_chunks::<2>()
        .0
        .iter()
        .enumerate()
        .map(|(i, b)| {
            BandSetting::with_weight(
                b[0] / fs,
                b[1] / fs,
                constant(desired[i] / amplitude),
                constant(weight[i] / weight_scale),
            )
        })
        .collect::<std::result::Result<_, _>>()
        .map_err(|_| Error::InvalidInput("invalid Remez bands"))?;
    let mut params = pm_parameters(numtaps, &settings)
        .map_err(|_| Error::InvalidInput("incompatible Remez constraints"))?;
    params.set_max_iterations(100);
    params.set_flatness_threshold(1e-12);
    params.set_chebyshev_proxy_degree(24);
    let design = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pm_remez(&params)))
        .map_err(|_| Error::NumericalFailure("Parks-McClellan backend failed"))?
        .map_err(|_| Error::NumericalFailure("Parks-McClellan design failed"))?;
    if design.flatness > 1e-10 {
        return Err(Error::NumericalFailure("Parks-McClellan did not converge"));
    }
    checked(
        design
            .impulse_response
            .into_iter()
            .map(|v| v * amplitude)
            .collect(),
    )
}
