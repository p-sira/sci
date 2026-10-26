# `signal` coverage and SciPy adaptations

Audited against `SCIPY_FUNCTIONS.md` and `scipy-func-list/scipy.signal.md`:
**168 of 175 public inventory function names remain exported**. Six redundant
spellings are removed: `freqresp` → `freqs`, `dfreqresp` → `freqz`, `sosfreqz` →
`freqz_sos`, `check_COLA` → `check_cola`, `check_NOLA` → `check_nola`, and
`closest_STFT_dual_window` → `closest_stft_dual_window`.
`choose_conv_method` and `ConvolutionMethod` are removed because their cost model
was calibrated to SciPy rather than Sci’s Rust backends.
The underlying numerical operations retain bounded Rust signatures and
independent reference cases; fixtures may keep SciPy's original labels.
Compatibility namespaces and Cython duplicates add no separate Rust APIs.

The `fft`, `ifft`, and `rfft` functions are re-exported from `sci-fft` and
retain their signatures and normalization. `lfilter_with_state`, `check_cola`,
and `check_nola` provide Rust functional spellings/extensions. Backend types
remain private; data uses f64 slices, row-major vectors, and shared Complex.

## Verification

`tests/scipy/signal_cases.json` contains inputs only. The generator
`tests/scipy/generate_signal.py` pins SciPy 1.18.1 and NumPy 2.5.4 and writes
1411 cases to separate input and expected CSV files. Rust tests check every
output row and its length, complex interleaving, indices, state ordering, and
frequency/time axes. The justified combined tolerance is
`2e-11 * (1 + abs(expected))`; most direct formulas agree within 1e-13.
Remez coefficients use `1e-10` in the same combined comparison.
Remez references use SciPy grid_density=32768 and maxiter=100: a density
comparison at 2048/32768 identified up to 3.4e-9 grid errors. Increasing the
backend Chebyshev proxy degree from 8 to 24 reduced even-length design
differences from 2e-9 to about 2e-11. Bessel roots use the ordinary `2e-11`
bound at every supported order (0..12), in all three normalizations. Scaled
Schur factorization supplies initial poles; bounded Newton refinement on
`exp(z) K_(n+1/2)(z)` avoids the companion polynomial's cancellation and
platform-sensitive root errors. Exact conjugate symmetry preserves real
transfer functions. These tolerances are unchanged by the shared `approx`
comparison helper, which retains absolute-plus-relative reference bounds.
NaN and signed infinities are compared explicitly. Independent checks cover
streaming state, steady steps, FIR polynomial preservation, Parseval energy,
CZT/FFT equivalence, STFT and resampling round trips, DPSS orthogonality,
filter frequency constraints, and interoperability with algebra and statistics.
Additional checks cover continuous/discrete step identities, analytic exponential
responses, feedback characteristic polynomials, partial-fraction reconstruction,
SOS response equivalence, minimax ripple, minimum-phase energy, spline reflection,
unpenalized smoothing trends, dual-window constraints, MLS periods and streaming,
and overlap-add equivalence. Each new function has invalid-domain/shape checks.
SOS reference cases compare section count and complex cascade responses because
our documented pairing order differs from SciPy; independent tests verify section
shapes and normalized denominators.
Invalid domains, overflow, shapes, and work limits are checked separately.
The original Fourier fixture and round-trip tests also remain active.

Run `python tests/scipy/audit_signal.py` to detect exports without fixture cases
and compute inventory coverage. Regenerate with
`uv run tests/scipy/generate_signal.py`. Ordinary Rust tests require no Python.

## Signature and behavior adaptations

- Arrays are one-dimensional unless the function explicitly accepts a matrix.
  Axis, dtype, device, broadcasting, plotting and Python object dispatch are
  unsupported. Inputs to fallible functions must be finite.
- Convolution/correlation use explicit Full, Same, or Valid modes. One-dimensional
  Same follows SciPy convolution cropping; correlation_lags follows SciPy's
  separate lag convention. Two-dimensional correlate2d uses SciPy's one-sample
  even-kernel offset. Matrix boundary extensions are explicit.
- Waveforms accept scalar phase/time parameters. Chirp supports real linear,
  quadratic (vertex at zero), logarithmic and hyperbolic sweeps; phase is degrees.
  Gausspulse returns (in-phase, quadrature, envelope), without cutoff-time dispatch.
  Unit impulse is one-dimensional with an explicit index. Vector strength takes
  one period and returns (strength, phase in radians).
- Nonfinite square-wave phase/duty returns NaN. SciPy 1.18.1 returns -1 for NaN
  duty; that corner is intentionally adapted and tested directly.
- Filtering uses ascending-delay coefficients, normalized by a[0]. State is
  transposed direct form II. SOS rows are [b0,b1,b2,1,a1,a2]. SOS streaming states
  are not an argument to sosfilt; use lfilter_with_state section by section.
  Forward/backward filters support odd extension and explicit padlen only.
- Frequency responses take explicit frequency slices and return responses only.
  Digital frequencies are radians/sample; analog frequencies are radians/time.
  Response poles and undefined group delays are errors. Bode functions return
  (dB magnitude, phase in degrees); exact zeros preserve negative infinity dB.
  Digital Bode uses ascending-delay coefficients, whereas SciPy's system tuple
  describes descending z polynomials. The reference generator accounts for this.
- Butterworth, Chebyshev I/II, elliptic, and Bessel design support all four band types and analog
  or digital systems; output is numerator/denominator only. fs=None selects
  analog angular frequencies, Some(fs) selects digital cutoff units with
  prewarping. iirfilter supports all five prototype families. Minimum-order
  selection and iirdesign accept scalar lowpass/highpass edges only. iirdesign
  selects the four magnitude prototype families; Bessel has no order selector.
- ZPK and coefficient transforms support proper real systems. Transformed roots
  are sorted by real and imaginary components; prototype order is documented.
  Root conversion uses bounded Schur factorization, can lose accuracy for
  repeated roots, and follows SciPy's normalization/trimming threshold. SOS root
  conversion factors sections individually. Coefficient transformations reuse
  the shared ZPK transformations rather than independent polynomial algorithms.
- FIR design uses explicit increasing cutoffs, pass-zero selection, a typed
  window, scaling and sample rate; width-based dispatch is unsupported.
  Savitzky–Golay coefficients are centered and in convolution order. Filtering
  supports odd windows and mode=interp only. Derivatives exceeding the degree
  return zeros. Detrending supports one constant or linear segment.
- Median/rank filters use zero-padded full neighborhoods; order_filter adapts
  SciPy masks to a one-dimensional all-ones neighborhood. Wiener filtering
  returns the local mean when variance is zero, including zero noise, avoiding
  SciPy's 0/0 result for this corner. This adaptation has independent tests.
- Fourier resampling accepts real periodic signals with no spectral window or
  timestamp dispatch. Polyphase resampling supports default Kaiser-beta-5 taps
  or explicit taps, constant-zero extension and delay removal. Decimation
  supports FIR/Hamming only, with explicit order and zero-phase selection.
- CZT accepts shared Complex inputs and parameters and evaluates the z-transform
  directly with a work bound. Zoom FFT accepts an explicit interval, length,
  rate and endpoint flag. These direct evaluations can be slower than SciPy's
  Bluestein implementation and ill-conditioned away from the unit circle.
- Power/spectral functions accept explicit windows, overlap, FFT lengths and
  sample rate, real signals, one-sided outputs and arithmetic mean averaging.
  CSD/coherence require equal input lengths. Zero-power coherence is NaN.
  Spectrogram returns [frequency][time] powers. STFT uses zero boundary extension,
  final-frame padding, no detrending, and spectrum scaling. ISTFT uses that
  convention, checks NOLA, and retains trailing padded samples.
  Lomb–Scargle supports classical power or normalized power, optional mean
  centering, no weights, floating mean, or amplitude mode.
- hilbert preserves DC and even-length Nyquist. hilbert2 follows SciPy 1.18's
  single-orthant convention, zeroing even-length Nyquist along both axes; its
  real part need not equal the original input.
- find_peaks returns all local maxima with plateau midpoint rounding; selection
  options are unsupported. Relative extrema use a scalar comparator and explicit
  Clip/Wrap edges. Prominences/widths support optional local windows, return
  structured fields in caller peak order, and yield zero for nonmaximal peaks
  rather than emitting Python warnings. Precomputed prominence dispatch is absent.
- Window symmetry is explicit. get_window uses a typed Window rather than strings,
  with periodic=true matching SciPy fftbins=true. DPSS accepts k tapers, norm=2,
  and no concentration-ratio/singleton approximate-normalization dispatch.
  KBD requires an even symmetric window and its own explicit function.

- Bessel prototypes explicitly select phase, delay, or half-power magnitude
  normalization. Elliptic functions use parameter m=k^2. band_stop_obj accepts
  ordered analog bandstop candidates and returns fractional order; family ripple
  payloads are replaced by explicit gpass/gstop. findfreqs takes coefficients only.
- Comb design has explicit peak/pass-zero flags and fs/w0 integer divisibility;
  Q>1 ensures the supported stable bandwidth. Gammatone supports explicit FIR
  order/taps or its fixed eighth-order IIR, with IIR order=4 and numtaps=0.
- StateSpace owns row-major A/B/C/D; abcd_normalize requires all matrices
  explicitly. tf2ss/zpk2ss return controller canonical SISO realizations;
  ss2tf selects an input and returns all output numerator rows, while ss2zpk
  selects both input/output. cont2discrete accepts StateSpace and explicit dt,
  with ZOH, bilinear, Euler, or backward difference. Other conversion methods
  and dynamic system tuple dispatch are unsupported.
- Simulations return time, [time][output], and [time][state]. dlsim takes samples
  and explicit dt with no input-time interpolation. lsim requires equally spaced
  nonnegative times and supports ZOH or linear interpolation. step/impulse and
  discrete counterparts select one input and return every output/state; continuous
  impulse rejects nonzero feedthrough because a Dirac delta cannot be sampled.
  place_poles supports controllable single-input real systems and conjugate
  complex targets through Ackermann feedback; multi-input YT/KNV placement is
  unsupported. Zero-state feedthrough models work across conversion/simulation.
- SOS design pairs conjugates and real roots deterministically by magnitude,
  places gain in the first section, and pads digital roots at zero. It supports
  real digital systems only; analog/nearest/keep-odd dispatch is unsupported.
  tf2sos pads ascending-delay polynomials before factoring to preserve delays.
  It does not optimize per-section peak gain as SciPy nearest pairing does.
- unique_roots uses greedy distance clusters, with mean or lexicographic min/max
  representatives. Partial fractions support repeated poles with mean clustering
  and ascending powers in consecutive blocks, sorted by magnitude then real/imag.
  residue uses descending s powers; residuez ascending delays. Inverse functions
  accept exactly repeated consecutive poles, with real coefficients required;
  tolerance/rtype keyword dispatch and complex coefficient outputs are unsupported.
- firls supports odd type-I least squares with linearly varying band gains and
  explicit weights. firwin2 supports all four linear-phase types, endpoint-zero
  constraints, explicit mesh, repeated interior frequencies, and typed tapers.
  firwin_2d supports separable row/column designs; circular radial interpolation
  is unsupported. minimum_phase supports homomorphic conversion with explicit
  nfft and half selection; Hilbert-method dispatch is unsupported. remez supports
  symmetric amplitude minimax design with constant band gains/weights; Hilbert
  and differentiator types are unsupported. pm-remez 0.3.5 reuses private
  nalgebra 0.35 eigenvalues, with 100 exchanges and degree-24 Chebyshev proxies;
  backend failures are converted to shared errors. The backend controls Schur
  convergence internally and does not expose an eigenvalue iteration limit.
  Two-tap designs use a direct scalar minimax solution over band endpoints,
  avoiding an upstream constant-proxy failure.
  This removes the older faer stack and its default features. Rust 1.89
  compatibility is checked with the workspace's minimum supported toolchain.
- Cardinal cubic/quadratic spline evaluation uses explicit spacing/origin and
  periodic endpoint reflection with clamped coefficient indices. cspline1d
  supports lambda=0 or lambda>1/144; small positive smoothing is unsupported.
  Quadratic functions support interpolation only. Two-dimensional cubic analysis
  uses interpolation for lambda<=1/144 and smoothing otherwise. Boundary sums
  have explicit precision and must converge within the input length; short
  records or slowly decaying poles can error. symiirorder2 uses radius in [0,1)
  and angle in [0,pi]. spline_filter preserves SciPy's float32 reconstruction
  tap rounding while filtering real f64 matrices. whittaker_henderson supports
  an explicit nonnegative penalty and nonnegative weights; REML selection and
  nonfinite data masked by zero weights are unsupported. It returns samples
  instead of SciPy's result object; lambda=0 returns the original data.
- oaconvolve uses FFT overlap-add with cached kernel spectra and bounded blocks.
  closest_stft_dual_window accepts
  real desired windows, explicit hop, and optional scaling, returning (dual,alpha).
- envelope supports real periodic samples, nonnegative integer frequency-bin
  bands, output-length resampling, squared amplitude, and all/lowpass residual.
  Complex/negative-band dispatch is unsupported. SciPy 1.18.1's DC-inclusive
  band convention returns a zero residual; this behavior is preserved and tested.
- max_len_seq supports 2..32 bits, explicit nonzero binary states and lengths,
  SciPy's default primitive taps or explicit unique tap indices. It returns
  sequence and continuation state. find_peaks_cwt supports the default Ricker
  wavelet/ridge/noise criteria with explicit increasing widths; customization
  options are unsupported. Exact or nearly tied wavelet maxima can change peak
  indices through floating-point rounding. Fixtures use peaks with stable maxima.

## Numerical limits

Fallible sample vectors and matrices are limited to 1,048,576 entries. Direct
numerical work is limited to 16,777,216 operations. FFT padding also obeys the
length limit. New functions report overflow rather than returning nonfinite
results, except the documented IEEE scalar functions, zero-power coherence,
and Bode zeros. Existing Fourier helpers preserve IEEE propagation.
Polynomial/filter order is at most 64 for classical design, 12 for Bessel, and
128 for general conversion. State-space dimensions are at most 32; single-input
pole placement at most 16. Partial fractions support pole order<=32 and numerator
degree<=64. Dense Whittaker smoothing supports at most 256 samples, orders 1..8.
FIR design lengths are at most 1023, least squares requires odd length<=511,
and Remez supports 2..255 taps. Matrix exponential input norms are bounded by
1e6. Convolution FFT work, CWT convolutions, and matrix/fit sizes are checked
before numerical work and allocation.
DPSS length is 2..256 (periodic extension may use 257); its bounded nalgebra
symmetric eigensolver allows 100,000 iterations. Savitzky–Golay length is at
most 4095 and degree at most 32, with an additional fit-work limit and bounded
SVD. Savitzky–Golay uses one centered, scaled Vandermonde pseudoinverse per
call, sharing it between interior weights and both boundary polynomial fits.
Fit construction has O(window*(degree+1)²) work; boundary evaluation reuses
the fitted coefficients instead of repeating an SVD for every edge sample.
Chebyshev-window attenuation and Taylor suppression are at most 600 dB;
Taylor nbar is 1..128. These bounds are part of the supported API.
No uniform relative-error guarantee applies near cancellation, repeated roots,
singularities, poorly conditioned filters, or extreme parameter values.

## Implemented numerical operations (canonical Rust names)

- `abcd_normalize`
- `argrelextrema`
- `argrelmax`
- `argrelmin`
- `band_stop_obj`
- `bessel`
- `besselap`
- `bilinear`
- `bilinear_zpk`
- `bode`
- `buttap`
- `butter`
- `buttord`
- `cheb1ap`
- `cheb1ord`
- `cheb2ap`
- `cheb2ord`
- `cheby1`
- `cheby2`
- `check_cola`
- `check_nola`
- `chirp`
- `closest_stft_dual_window`
- `coherence`
- `cont2discrete`
- `convolve`
- `convolve2d`
- `correlate`
- `correlate2d`
- `correlation_lags`
- `csd`
- `cspline1d`
- `cspline1d_eval`
- `cspline2d`
- `czt`
- `czt_points`
- `dbode`
- `decimate`
- `deconvolve`
- `detrend`
- `dimpulse`
- `dlsim`
- `dstep`
- `ellip`
- `ellipap`
- `ellipord`
- `envelope`
- `fftconvolve`
- `filtfilt`
- `find_peaks`
- `find_peaks_cwt`
- `findfreqs`
- `firls`
- `firwin`
- `firwin2`
- `firwin_2d`
- `freqs`
- `freqs_zpk`
- `freqz`
- `freqz_sos`
- `freqz_zpk`
- `gammatone`
- `gauss_spline`
- `gausspulse`
- `get_window`
- `group_delay`
- `hilbert`
- `hilbert2`
- `iircomb`
- `iirdesign`
- `iirfilter`
- `iirnotch`
- `iirpeak`
- `impulse`
- `invres`
- `invresz`
- `istft`
- `kaiser_atten`
- `kaiser_beta`
- `kaiserord`
- `lfilter`
- `lfilter_zi`
- `lfiltic`
- `lombscargle`
- `lp2bp`
- `lp2bp_zpk`
- `lp2bs`
- `lp2bs_zpk`
- `lp2hp`
- `lp2hp_zpk`
- `lp2lp`
- `lp2lp_zpk`
- `lsim`
- `max_len_seq`
- `medfilt`
- `medfilt2d`
- `minimum_phase`
- `normalize`
- `oaconvolve`
- `order_filter`
- `peak_prominences`
- `peak_widths`
- `periodogram`
- `place_poles`
- `qspline1d`
- `qspline1d_eval`
- `qspline2d`
- `remez`
- `resample`
- `resample_poly`
- `residue`
- `residuez`
- `savgol_coeffs`
- `savgol_filter`
- `sawtooth`
- `sepfir2d`
- `sos2tf`
- `sos2zpk`
- `sosfilt`
- `sosfilt_zi`
- `sosfiltfilt`
- `spectrogram`
- `spline_filter`
- `square`
- `ss2tf`
- `ss2zpk`
- `step`
- `stft`
- `sweep_poly`
- `symiirorder1`
- `symiirorder2`
- `tf2sos`
- `tf2ss`
- `tf2zpk`
- `unique_roots`
- `unit_impulse`
- `upfirdn`
- `vectorstrength`
- `welch`
- `whittaker_henderson`
- `wiener`
- `windows.barthann`
- `windows.bartlett`
- `windows.blackman`
- `windows.blackmanharris`
- `windows.bohman`
- `windows.boxcar`
- `windows.chebwin`
- `windows.cosine`
- `windows.dpss`
- `windows.exponential`
- `windows.flattop`
- `windows.gaussian`
- `windows.general_cosine`
- `windows.general_gaussian`
- `windows.general_hamming`
- `windows.get_window`
- `windows.hamming`
- `windows.hann`
- `windows.kaiser`
- `windows.kaiser_bessel_derived`
- `windows.lanczos`
- `windows.nuttall`
- `windows.parzen`
- `windows.taylor`
- `windows.triang`
- `windows.tukey`
- `zoom_fft`
- `zpk2sos`
- `zpk2ss`
- `zpk2tf`
