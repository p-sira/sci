// SciPy heuristics and ridge detection adaptations: LICENSES/scipy-signal.txt.
use super::{Mode, checked, convolve, fft, ifft, nonempty, resample, samples, size, work};
use crate::{
    Error, Result,
    complex::{Complex, native},
};
use std::f64::consts::PI;

/// Frequency-domain envelope and its real residual.
#[derive(Clone, Debug, PartialEq)]
pub struct Envelope {
    /// Nonnegative amplitude (or squared amplitude) at output sample times.
    pub amplitude: Vec<f64>,
    /// Real residual, containing all rejected positive frequencies or only lower ones.
    pub residual: Vec<f64>,
}

/// Computes zero-extended convolution `y[k]=sum_j x[j] h[k-j]` with FFT overlap-add.
///
/// ## Parameters
///
/// Nonempty finite real slices; mode follows convolve. Returns Full n+m-1, Same x.len(), or Valid abs(n-m)+1 samples. The shorter vector is the block kernel.
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
/// Uses private RustFFT with power-of-two blocks and a cached kernel spectrum. Rounding differs from direct summation; total estimated FFT work is bounded.
///
/// # Examples
///
/// ```
/// use sci_signal::utilities::*;
/// use sci_signal::Mode;
/// let _=oaconvolve(&[1.,2.,3.],&[1.,-1.],Mode::Full)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy oaconvolve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.oaconvolve.html).
pub fn oaconvolve(x: &[f64], h: &[f64], mode: Mode) -> Result<Vec<f64>> {
    nonempty(x)?;
    nonempty(h)?;
    let len = x.len() + h.len() - 1;
    size(len)?;
    let (input, kernel) = if x.len() >= h.len() { (x, h) } else { (h, x) };
    if kernel.len() == 1 {
        return convolve(x, h, mode);
    }
    let nfft = (kernel.len() * 2).next_power_of_two();
    size(nfft)?;
    let block = nfft - kernel.len() + 1;
    work(
        input.len().div_ceil(block),
        nfft * (nfft.ilog2() as usize + 1),
    )?;
    let mut planner = rustfft::FftPlanner::<f64>::new();
    let forward = planner.plan_fft_forward(nfft);
    let inverse = planner.plan_fft_inverse(nfft);
    let mut ks = vec![num_complex::Complex64::new(0., 0.); nfft];
    for (i, &v) in kernel.iter().enumerate() {
        ks[i].re = v;
    }
    forward.process(&mut ks);
    let mut full = vec![0.; len];
    for (start, chunk) in input.chunks(block).enumerate() {
        let start = start * block;
        let mut spectrum = vec![num_complex::Complex64::new(0., 0.); nfft];
        for (i, &v) in chunk.iter().enumerate() {
            spectrum[i].re = v;
        }
        forward.process(&mut spectrum);
        for (i, z) in spectrum.iter_mut().enumerate() {
            *z *= ks[i];
        }
        inverse.process(&mut spectrum);
        for i in 0..chunk.len() + kernel.len() - 1 {
            full[start + i] += spectrum[i].re / nfft as f64;
        }
    }
    let (start, length) = match mode {
        Mode::Full => (0, len),
        Mode::Same => ((h.len() - 1) / 2, x.len()),
        Mode::Valid => (x.len().min(h.len()) - 1, x.len().abs_diff(h.len()) + 1),
    };
    checked(full[start..start + length].to_vec())
}

/// Projects a desired window onto the dual-window constraint `sum_{j mod hop=r} win[j] dual[j]=1`.
///
/// ## Parameters
///
/// win/desired are equally long nonempty real windows; hop is 1..length. scaled=false returns the closest Euclidean dual and alpha=1; scaled=true jointly optimizes a nonzero alpha. Returns (dual,alpha). SciPy complex windows are unsupported.
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
/// use sci_signal::utilities::*;
/// let _=closest_stft_dual_window(&[1.;8],4,&[1.;8],true)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy closest_STFT_dual_window](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.closest_STFT_dual_window.html).
pub fn closest_stft_dual_window(
    win: &[f64],
    hop: usize,
    desired: &[f64],
    scaled: bool,
) -> Result<(Vec<f64>, f64)> {
    nonempty(win)?;
    samples(desired)?;
    if win.len() != desired.len() || hop == 0 || hop > win.len() {
        return Err(Error::DimensionMismatch);
    }
    let mut energy = vec![0.; hop];
    let mut dot = vec![0.; hop];
    for (i, (&w, &d)) in win.iter().zip(desired).enumerate() {
        energy[i % hop] += w * w;
        dot[i % hop] += w * d;
    }
    if energy.iter().any(|e| !e.is_finite() || *e <= f64::EPSILON) {
        return Err(Error::InvalidInput("window fails NOLA"));
    }
    let canonical: Vec<_> = win
        .iter()
        .enumerate()
        .map(|(i, w)| w / energy[i % hop])
        .collect();
    let q: Vec<_> = canonical
        .iter()
        .enumerate()
        .map(|(i, w)| w * dot[i % hop])
        .collect();
    let alpha = if scaled {
        let numerator = q.iter().zip(&canonical).map(|(q, w)| q * w).sum::<f64>();
        let denominator = q.iter().map(|q| q * q).sum::<f64>();
        if numerator.abs() == 0. || denominator <= 1e-15 {
            return Err(Error::NumericalFailure("unstable dual scaling"));
        }
        numerator / denominator
    } else {
        1.
    };
    if !alpha.is_finite() {
        return Err(Error::NonFiniteEvaluation);
    }
    Ok((
        checked(
            canonical
                .iter()
                .zip(desired)
                .zip(q)
                .map(|((w, d), q)| w + alpha * (d - q))
                .collect(),
        )?,
        alpha,
    ))
}

/// Computes the envelope `abs(IFFT(analytic band spectrum))` and the rejected-band residual.
///
/// ## Parameters
///
/// x is nonempty real periodic samples. band=(lo,hi) selects integer positive DFT bins, 0<=lo<hi<=ceil(n/2). output_length>0 permits Fourier resampling; squared selects squared amplitude; residual_all=false retains only frequencies below lo, true all rejected frequencies. Returns Envelope. Complex/negative-band dispatch is unsupported; even Nyquist remains in the all residual.
/// Follows SciPy 1.18.1: a band starting at DC produces a zero residual, even in all mode.
/// Output sample times span the same period; amplitude units equal x units, or their square when squared=true.
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
/// Uses private RustFFT and a periodic signal convention. Finite records and frequency discontinuities can produce ringing; output times are evenly spaced over the original period.
///
/// # Examples
///
/// ```
/// use sci_signal::utilities::*;
/// let _=envelope(&[1.,0.,-1.,0.],(1,2),4,false,true)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy envelope](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.envelope.html).
pub fn envelope(
    x: &[f64],
    band: (usize, usize),
    output_length: usize,
    squared: bool,
    residual_all: bool,
) -> Result<Envelope> {
    nonempty(x)?;
    size(output_length)?;
    let n = x.len();
    if output_length == 0 || band.0 >= band.1 || band.1 > n.div_ceil(2) {
        return Err(Error::InvalidInput(
            "invalid positive-frequency band or output length",
        ));
    }
    let z = fft(&x
        .iter()
        .map(|&v| Complex { re: v, im: 0. })
        .collect::<Vec<_>>());
    let mut bb = vec![Complex { re: 0., im: 0. }; output_length];
    for (k, b) in bb.iter_mut().enumerate().take(band.1 - band.0) {
        let i = band.0 + k;
        let factor = if i == 0 { 1. } else { 2. };
        b.re = z[i].re * factor;
        b.im = z[i].im * factor;
    }
    let amplitude = checked(
        ifft(&bb)
            .into_iter()
            .map(|z| {
                let a = native(z).norm() * output_length as f64 / n as f64;
                if squared { a * a } else { a }
            })
            .collect(),
    )?;
    let mut residual = z;
    for i in 0..=n / 2 {
        let rejected =
            (i >= band.0 && i < band.1) || (!residual_all && i >= band.1 && i < n.div_ceil(2));
        if rejected {
            residual[i] = Complex { re: 0., im: 0. };
            if i > 0 {
                residual[n - i] = Complex { re: 0., im: 0. };
            }
        }
    }
    if band.0 == 0 {
        residual.fill(Complex { re: 0., im: 0. });
    }
    let residual = checked(ifft(&residual).into_iter().map(|z| z.re).collect())?;
    let residual = if output_length == n {
        residual
    } else {
        resample(&residual, output_length)?
    };
    Ok(Envelope {
        amplitude,
        residual,
    })
}

/// Generates a binary linear-feedback shift-register sequence `feedback=state[0] XOR state[tap] ...`.
///
/// ## Parameters
///
/// nbits is 2..32; state has nbits binary entries and must be nonzero; length<=1048576. taps=None selects SciPy primitive taps; explicit taps are unique indices 1..nbits-1. Returns (binary sequence, state for the next call). Only default taps guarantee maximal period 2^nbits-1.
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
/// use sci_signal::utilities::*;
/// let _=max_len_seq(3,&[1,1,1],7,None)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy max_len_seq](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.max_len_seq.html).
pub fn max_len_seq(
    nbits: usize,
    state: &[u8],
    length: usize,
    taps: Option<&[usize]>,
) -> Result<(Vec<u8>, Vec<u8>)> {
    size(length)?;
    if !(2..=32).contains(&nbits)
        || state.len() != nbits
        || state.iter().any(|b| *b > 1)
        || state.iter().all(|b| *b == 0)
    {
        return Err(Error::InvalidInput(
            "invalid bit count or nonzero binary state",
        ));
    }
    let defaults: &[&[usize]] = &[
        &[1],
        &[2],
        &[3],
        &[3],
        &[5],
        &[6],
        &[7, 6, 1],
        &[5],
        &[7],
        &[9],
        &[11, 10, 4],
        &[12, 11, 8],
        &[13, 12, 2],
        &[14],
        &[15, 13, 4],
        &[14],
        &[11],
        &[18, 17, 14],
        &[17],
        &[19],
        &[21],
        &[18],
        &[23, 22, 17],
        &[22],
        &[25, 24, 20],
        &[26, 25, 22],
        &[25],
        &[27],
        &[29, 28, 7],
        &[28],
        &[31, 30, 10],
    ];
    let taps = taps.unwrap_or(defaults[nbits - 2]);
    if taps.is_empty()
        || taps.len() > nbits
        || taps
            .iter()
            .enumerate()
            .any(|(i, t)| *t == 0 || *t >= nbits || taps[..i].contains(t))
    {
        return Err(Error::InvalidInput(
            "taps must be unique indices 1..nbits-1",
        ));
    }
    work(length, taps.len() + 1)?;
    let mut state = state.to_vec();
    let mut seq = Vec::with_capacity(length);
    let mut index = 0;
    for _ in 0..length {
        let mut feedback = state[index];
        seq.push(feedback);
        for tap in taps {
            feedback ^= state[(tap + index) % nbits];
        }
        state[index] = feedback;
        index = (index + 1) % nbits;
    }
    state.rotate_left(index);
    Ok((seq, state))
}

#[derive(Clone)]
struct Ridge {
    points: Vec<(usize, usize)>,
    gap: usize,
}
fn maxima(row: &[f64]) -> Vec<usize> {
    (1..row.len().saturating_sub(1))
        .filter(|&i| row[i] > row[i - 1] && row[i] > row[i + 1])
        .collect()
}

/// Finds peaks by connecting local maxima across Ricker continuous-wavelet scales and filtering ridge length and signal-to-noise ratio.
///
/// ## Parameters
///
/// x is nonempty real samples; widths are positive increasing widths in samples. Uses SciPy default Ricker wavelet, maximum distance width/4, gap ceil(widths\[0\]), minimum length ceil(widths.len()/4), minimum SNR=1, noise percentile=10 in ceil(x.len()/20) windows. Returns sorted peak indices; customization keywords are unsupported.
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
/// Uses bounded direct Ricker convolutions and SciPy ridge criteria (Bioinformatics 2006, DOI 10.1093/bioinformatics/btl355). Zero noise gives infinite SNR; 0/0 follows SciPy acceptance without warnings.
/// Nearly tied wavelet maxima can yield backend-dependent peak indices due to rounding.
///
/// # Examples
///
/// ```
/// use sci_signal::utilities::*;
/// let _=find_peaks_cwt(&[0.,0.,1.,0.,0.,0.,2.,0.,0.],&[1.,2.,3.])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy find_peaks_cwt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks_cwt.html).
pub fn find_peaks_cwt(x: &[f64], widths: &[f64]) -> Result<Vec<usize>> {
    nonempty(x)?;
    nonempty(widths)?;
    if widths.iter().any(|w| *w <= 0.) || widths.windows(2).any(|w| w[0] >= w[1]) {
        return Err(Error::InvalidInput("widths must be positive increasing"));
    }
    work(x.len(), widths.len())?;
    size(x.len() * widths.len())?;
    let mut cwt = vec![];
    let mut cost = 0usize;
    for &width in widths {
        let points = (10. * width).min(x.len() as f64);
        let length = points.ceil() as usize;
        work(length, x.len())?;
        cost = cost
            .checked_add(length * x.len())
            .ok_or(Error::InvalidInput("wavelet work overflow"))?;
        work(cost, 1)?;
        let norm = 2. / ((3. * width).sqrt() * PI.powf(0.25));
        let wavelet: Vec<_> = (0..length)
            .map(|i| {
                let t = (i as f64 - (points - 1.) / 2.) / width;
                norm * (1. - t * t) * (-t * t / 2.).exp()
            })
            .collect();
        cwt.push(convolve(x, &wavelet, Mode::Same)?);
    }
    let maxima: Vec<_> = cwt.iter().map(|r| maxima(r)).collect();
    let Some(start) = maxima.iter().rposition(|r| !r.is_empty()) else {
        return Ok(vec![]);
    };
    let mut ridges: Vec<_> = maxima[start]
        .iter()
        .map(|&c| Ridge {
            points: vec![(start, c)],
            gap: 0,
        })
        .collect();
    let mut finished = vec![];
    for row in (0..start).rev() {
        for line in &mut ridges {
            line.gap += 1;
        }
        let previous: Vec<_> = ridges.iter().map(|r| r.points.last().unwrap().1).collect();
        cost = cost
            .checked_add(
                maxima[row]
                    .len()
                    .checked_mul(previous.len())
                    .ok_or(Error::InvalidInput("ridge work overflow"))?,
            )
            .ok_or(Error::InvalidInput("ridge work overflow"))?;
        work(cost, 1)?;
        for &col in &maxima[row] {
            let closest = previous
                .iter()
                .enumerate()
                .min_by_key(|(_, c)| col.abs_diff(**c))
                .map(|(i, c)| (i, col.abs_diff(*c)));
            if let Some((i, _)) = closest.filter(|(_, d)| *d as f64 <= widths[row] / 4.) {
                ridges[i].points.push((row, col));
                ridges[i].gap = 0;
            } else {
                ridges.push(Ridge {
                    points: vec![(row, col)],
                    gap: 0,
                });
            }
        }
        for i in (0..ridges.len()).rev() {
            if ridges[i].gap > widths[0].ceil() as usize {
                finished.push(ridges.remove(i));
            }
        }
    }
    finished.extend(ridges);
    let window = x.len().div_ceil(20);
    let half = window / 2;
    let mut noise = vec![0.; x.len()];
    work(x.len(), window)?;
    for (i, value) in noise.iter_mut().enumerate() {
        let mut local =
            cwt[0][i.saturating_sub(half)..(i + half + window % 2).min(x.len())].to_vec();
        local.sort_by(f64::total_cmp);
        let position = 0.1 * (local.len() - 1) as f64;
        let lower = position.floor() as usize;
        let upper = position.ceil() as usize;
        *value = local[lower] + (local[upper] - local[lower]) * (position - lower as f64);
    }
    let mut peaks = vec![];
    for mut ridge in finished {
        if ridge.points.len() < widths.len().div_ceil(4) {
            continue;
        }
        ridge.points.sort_by_key(|&(r, _)| r);
        let (row, col) = ridge.points[0];
        let snr = (cwt[row][col] / noise[col]).abs();
        if snr >= 1. || snr.is_nan() {
            peaks.push(col);
        }
    }
    peaks.sort_unstable();
    Ok(peaks)
}
