# scipy.signal

[Back to SciPy function index](../SCIPY_FUNCTIONS.md)

## Namespaces

- [`scipy.signal`](#scipy-signal)
- [`scipy.signal.bsplines`](#scipy-signal-bsplines) (additional / compatibility)
- [`scipy.signal.filter_design`](#scipy-signal-filter_design) (additional / compatibility)
- [`scipy.signal.fir_filter_design`](#scipy-signal-fir_filter_design) (additional / compatibility)
- [`scipy.signal.lti_conversion`](#scipy-signal-lti_conversion) (additional / compatibility)
- [`scipy.signal.ltisys`](#scipy-signal-ltisys) (additional / compatibility)
- [`scipy.signal.signaltools`](#scipy-signal-signaltools) (additional / compatibility)
- [`scipy.signal.spectral`](#scipy-signal-spectral) (additional / compatibility)
- [`scipy.signal.spline`](#scipy-signal-spline) (additional / compatibility)
- [`scipy.signal.waveforms`](#scipy-signal-waveforms) (additional / compatibility)
- [`scipy.signal.wavelets`](#scipy-signal-wavelets) (additional / compatibility)
- [`scipy.signal.windows`](#scipy-signal-windows)
- [`scipy.signal.windows.windows`](#scipy-signal-windows-windows) (additional / compatibility)

## Public API namespaces

### scipy.signal

[Module reference](https://docs.scipy.org/doc/scipy/reference/signal.html#module-scipy.signal)

Function entries: **149**. Callable objects: **0**. Classes: **9**.

#### Functions

- [`abcd_normalize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.abcd_normalize.html#scipy.signal.abcd_normalize)
- [`argrelextrema`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.argrelextrema.html#scipy.signal.argrelextrema)
- [`argrelmax`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.argrelmax.html#scipy.signal.argrelmax)
- [`argrelmin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.argrelmin.html#scipy.signal.argrelmin)
- [`band_stop_obj`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.band_stop_obj.html#scipy.signal.band_stop_obj)
- [`bessel`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.bessel.html#scipy.signal.bessel)
- [`besselap`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.besselap.html#scipy.signal.besselap)
- [`bilinear`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.bilinear.html#scipy.signal.bilinear)
- [`bilinear_zpk`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.bilinear_zpk.html#scipy.signal.bilinear_zpk)
- [`bode`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.bode.html#scipy.signal.bode)
- [`buttap`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.buttap.html#scipy.signal.buttap)
- [`butter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html#scipy.signal.butter)
- [`buttord`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.buttord.html#scipy.signal.buttord)
- [`cheb1ap`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheb1ap.html#scipy.signal.cheb1ap)
- [`cheb1ord`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheb1ord.html#scipy.signal.cheb1ord)
- [`cheb2ap`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheb2ap.html#scipy.signal.cheb2ap)
- [`cheb2ord`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheb2ord.html#scipy.signal.cheb2ord)
- [`cheby1`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheby1.html#scipy.signal.cheby1)
- [`cheby2`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheby2.html#scipy.signal.cheby2)
- [`check_COLA`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.check_COLA.html#scipy.signal.check_cola)
- [`check_NOLA`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.check_NOLA.html#scipy.signal.check_nola)
- [`chirp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.chirp.html#scipy.signal.chirp)
- [`choose_conv_method`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.choose_conv_method.html#scipy.signal.choose_conv_method)
- [`closest_STFT_dual_window`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.closest_STFT_dual_window.html#scipy.signal.closest_stft_dual_window)
- [`coherence`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html#scipy.signal.coherence)
- [`cont2discrete`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cont2discrete.html#scipy.signal.cont2discrete)
- [`convolve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html#scipy.signal.convolve)
- [`convolve2d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html#scipy.signal.convolve2d)
- [`correlate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html#scipy.signal.correlate)
- [`correlate2d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate2d.html#scipy.signal.correlate2d)
- [`correlation_lags`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlation_lags.html#scipy.signal.correlation_lags)
- [`csd`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.csd.html#scipy.signal.csd)
- [`cspline1d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cspline1d.html#scipy.signal.cspline1d)
- [`cspline1d_eval`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cspline1d_eval.html#scipy.signal.cspline1d_eval)
- [`cspline2d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cspline2d.html#scipy.signal.cspline2d)
- [`czt`](https://docs.scipy.org/doc/scipy/reference/generated/czt-function.html#scipy.signal.czt)
- [`czt_points`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.czt_points.html#scipy.signal.czt_points)
- [`dbode`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.dbode.html#scipy.signal.dbode)
- [`decimate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.decimate.html#scipy.signal.decimate)
- [`deconvolve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.deconvolve.html#scipy.signal.deconvolve)
- [`detrend`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.detrend.html#scipy.signal.detrend)
- [`dfreqresp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.dfreqresp.html#scipy.signal.dfreqresp)
- [`dimpulse`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.dimpulse.html#scipy.signal.dimpulse)
- [`dlsim`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.dlsim.html#scipy.signal.dlsim)
- [`dstep`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.dstep.html#scipy.signal.dstep)
- [`ellip`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ellip.html#scipy.signal.ellip)
- [`ellipap`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ellipap.html#scipy.signal.ellipap)
- [`ellipord`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ellipord.html#scipy.signal.ellipord)
- [`envelope`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.envelope.html#scipy.signal.envelope)
- [`fftconvolve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html#scipy.signal.fftconvolve)
- [`filtfilt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html#scipy.signal.filtfilt)
- [`find_peaks`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html#scipy.signal.find_peaks)
- [`find_peaks_cwt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks_cwt.html#scipy.signal.find_peaks_cwt)
- [`findfreqs`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.findfreqs.html#scipy.signal.findfreqs)
- [`firls`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firls.html#scipy.signal.firls)
- [`firwin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html#scipy.signal.firwin)
- [`firwin2`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin2.html#scipy.signal.firwin2)
- [`firwin_2d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin_2d.html#scipy.signal.firwin_2d)
- [`freqresp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqresp.html#scipy.signal.freqresp)
- [`freqs`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqs.html#scipy.signal.freqs)
- [`freqs_zpk`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqs_zpk.html#scipy.signal.freqs_zpk)
- [`freqz`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqz.html#scipy.signal.freqz)
- [`freqz_sos`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqz_sos.html#scipy.signal.freqz_sos)
- [`freqz_zpk`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqz_zpk.html#scipy.signal.freqz_zpk)
- [`gammatone`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.gammatone.html#scipy.signal.gammatone)
- [`gauss_spline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.gauss_spline.html#scipy.signal.gauss_spline)
- [`gausspulse`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.gausspulse.html#scipy.signal.gausspulse)
- [`get_window`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html#scipy.signal.get_window)
- [`group_delay`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.group_delay.html#scipy.signal.group_delay)
- [`hilbert`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html#scipy.signal.hilbert)
- [`hilbert2`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert2.html#scipy.signal.hilbert2)
- [`iircomb`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iircomb.html#scipy.signal.iircomb)
- [`iirdesign`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirdesign.html#scipy.signal.iirdesign)
- [`iirfilter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirfilter.html#scipy.signal.iirfilter)
- [`iirnotch`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html#scipy.signal.iirnotch)
- [`iirpeak`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirpeak.html#scipy.signal.iirpeak)
- [`impulse`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.impulse.html#scipy.signal.impulse)
- [`invres`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.invres.html#scipy.signal.invres)
- [`invresz`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.invresz.html#scipy.signal.invresz)
- [`istft`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.istft.html#scipy.signal.istft)
- [`kaiser_atten`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.kaiser_atten.html#scipy.signal.kaiser_atten)
- [`kaiser_beta`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.kaiser_beta.html#scipy.signal.kaiser_beta)
- [`kaiserord`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.kaiserord.html#scipy.signal.kaiserord)
- [`lfilter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html#scipy.signal.lfilter)
- [`lfilter_zi`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter_zi.html#scipy.signal.lfilter_zi)
- [`lfiltic`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfiltic.html#scipy.signal.lfiltic)
- [`lombscargle`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lombscargle.html#scipy.signal.lombscargle)
- [`lp2bp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lp2bp.html#scipy.signal.lp2bp)
- [`lp2bp_zpk`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lp2bp_zpk.html#scipy.signal.lp2bp_zpk)
- [`lp2bs`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lp2bs.html#scipy.signal.lp2bs)
- [`lp2bs_zpk`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lp2bs_zpk.html#scipy.signal.lp2bs_zpk)
- [`lp2hp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lp2hp.html#scipy.signal.lp2hp)
- [`lp2hp_zpk`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lp2hp_zpk.html#scipy.signal.lp2hp_zpk)
- [`lp2lp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lp2lp.html#scipy.signal.lp2lp)
- [`lp2lp_zpk`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lp2lp_zpk.html#scipy.signal.lp2lp_zpk)
- [`lsim`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lsim.html#scipy.signal.lsim)
- [`max_len_seq`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.max_len_seq.html#scipy.signal.max_len_seq)
- [`medfilt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt.html#scipy.signal.medfilt)
- [`medfilt2d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt2d.html#scipy.signal.medfilt2d)
- [`minimum_phase`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.minimum_phase.html#scipy.signal.minimum_phase)
- [`normalize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.normalize.html#scipy.signal.normalize)
- [`oaconvolve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.oaconvolve.html#scipy.signal.oaconvolve)
- [`order_filter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.order_filter.html#scipy.signal.order_filter)
- [`peak_prominences`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html#scipy.signal.peak_prominences)
- [`peak_widths`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_widths.html#scipy.signal.peak_widths)
- [`periodogram`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html#scipy.signal.periodogram)
- [`place_poles`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.place_poles.html#scipy.signal.place_poles)
- [`qspline1d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.qspline1d.html#scipy.signal.qspline1d)
- [`qspline1d_eval`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.qspline1d_eval.html#scipy.signal.qspline1d_eval)
- [`qspline2d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.qspline2d.html#scipy.signal.qspline2d)
- [`remez`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.remez.html#scipy.signal.remez)
- [`resample`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html#scipy.signal.resample)
- [`resample_poly`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html#scipy.signal.resample_poly)
- [`residue`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.residue.html#scipy.signal.residue)
- [`residuez`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.residuez.html#scipy.signal.residuez)
- [`savgol_coeffs`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_coeffs.html#scipy.signal.savgol_coeffs)
- [`savgol_filter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html#scipy.signal.savgol_filter)
- [`sawtooth`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sawtooth.html#scipy.signal.sawtooth)
- [`sepfir2d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sepfir2d.html#scipy.signal.sepfir2d)
- [`sos2tf`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sos2tf.html#scipy.signal.sos2tf)
- [`sos2zpk`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sos2zpk.html#scipy.signal.sos2zpk)
- [`sosfilt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html#scipy.signal.sosfilt)
- [`sosfilt_zi`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt_zi.html#scipy.signal.sosfilt_zi)
- [`sosfiltfilt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html#scipy.signal.sosfiltfilt)
- [`sosfreqz`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfreqz.html#scipy.signal.sosfreqz)
- [`spectrogram`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html#scipy.signal.spectrogram)
- [`spline_filter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spline_filter.html#scipy.signal.spline_filter)
- [`square`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.square.html#scipy.signal.square)
- [`ss2tf`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ss2tf.html#scipy.signal.ss2tf)
- [`ss2zpk`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ss2zpk.html#scipy.signal.ss2zpk)
- [`step`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.step.html#scipy.signal.step)
- [`stft`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html#scipy.signal.stft)
- [`sweep_poly`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sweep_poly.html#scipy.signal.sweep_poly)
- [`symiirorder1`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.symiirorder1.html#scipy.signal.symiirorder1)
- [`symiirorder2`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.symiirorder2.html#scipy.signal.symiirorder2)
- [`tf2sos`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.tf2sos.html#scipy.signal.tf2sos)
- [`tf2ss`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.tf2ss.html#scipy.signal.tf2ss)
- [`tf2zpk`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.tf2zpk.html#scipy.signal.tf2zpk)
- [`unique_roots`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.unique_roots.html#scipy.signal.unique_roots)
- [`unit_impulse`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.unit_impulse.html#scipy.signal.unit_impulse)
- [`upfirdn`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.upfirdn.html#scipy.signal.upfirdn)
- [`vectorstrength`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.vectorstrength.html#scipy.signal.vectorstrength)
- [`welch`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html#scipy.signal.welch)
- [`whittaker_henderson`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.whittaker_henderson.html#scipy.signal.whittaker_henderson)
- [`wiener`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.wiener.html#scipy.signal.wiener)
- [`zoom_fft`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.zoom_fft.html#scipy.signal.zoom_fft)
- [`zpk2sos`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.zpk2sos.html#scipy.signal.zpk2sos)
- [`zpk2ss`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.zpk2ss.html#scipy.signal.zpk2ss)
- [`zpk2tf`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.zpk2tf.html#scipy.signal.zpk2tf)

#### Classes and their public methods

- [`BadCoefficients`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.BadCoefficients.html#scipy.signal.badcoefficients)
  - [`BadCoefficients.add_note`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.BadCoefficients.html#scipy.signal.badcoefficients.add_note)
  - [`BadCoefficients.with_traceback`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.BadCoefficients.html#scipy.signal.badcoefficients.with_traceback)
- [`CZT`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.CZT.html#scipy.signal.czt)
  - [`CZT.points`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.CZT.points.html#scipy.signal.czt.points)
- [`ShortTimeFFT`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.html#scipy.signal.shorttimefft)
  - [`ShortTimeFFT.extent`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.extent.html#scipy.signal.shorttimefft.extent)
  - [`ShortTimeFFT.from_dual`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.from_dual.html#scipy.signal.shorttimefft.from_dual)
  - [`ShortTimeFFT.from_win_equals_dual`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.from_win_equals_dual.html#scipy.signal.shorttimefft.from_win_equals_dual)
  - [`ShortTimeFFT.from_window`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.from_window.html#scipy.signal.shorttimefft.from_window)
  - [`ShortTimeFFT.istft`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.istft.html#scipy.signal.shorttimefft.istft)
  - [`ShortTimeFFT.k_max`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.k_max.html#scipy.signal.shorttimefft.k_max)
  - [`ShortTimeFFT.nearest_k_p`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.nearest_k_p.html#scipy.signal.shorttimefft.nearest_k_p)
  - [`ShortTimeFFT.p_max`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.p_max.html#scipy.signal.shorttimefft.p_max)
  - [`ShortTimeFFT.p_num`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.p_num.html#scipy.signal.shorttimefft.p_num)
  - [`ShortTimeFFT.p_range`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.p_range.html#scipy.signal.shorttimefft.p_range)
  - [`ShortTimeFFT.scale_to`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.scale_to.html#scipy.signal.shorttimefft.scale_to)
  - [`ShortTimeFFT.spectrogram`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.spectrogram.html#scipy.signal.shorttimefft.spectrogram)
  - [`ShortTimeFFT.stft`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.stft.html#scipy.signal.shorttimefft.stft)
  - [`ShortTimeFFT.stft_detrend`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.stft_detrend.html#scipy.signal.shorttimefft.stft_detrend)
  - [`ShortTimeFFT.t`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.t.lower.html#scipy.signal.shorttimefft.t)
  - [`ShortTimeFFT.upper_border_begin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.upper_border_begin.html#scipy.signal.shorttimefft.upper_border_begin)
- [`StateSpace`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.StateSpace.html#scipy.signal.statespace)
  - [`StateSpace.to_ss`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.StateSpace.to_ss.html#scipy.signal.statespace.to_ss)
  - [`StateSpace.to_tf`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.StateSpace.to_tf.html#scipy.signal.statespace.to_tf)
  - [`StateSpace.to_zpk`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.StateSpace.to_zpk.html#scipy.signal.statespace.to_zpk)
- [`TransferFunction`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.TransferFunction.html#scipy.signal.transferfunction)
  - [`TransferFunction.to_ss`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.TransferFunction.to_ss.html#scipy.signal.transferfunction.to_ss)
  - [`TransferFunction.to_tf`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.TransferFunction.to_tf.html#scipy.signal.transferfunction.to_tf)
  - [`TransferFunction.to_zpk`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.TransferFunction.to_zpk.html#scipy.signal.transferfunction.to_zpk)
- [`ZerosPolesGain`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ZerosPolesGain.html#scipy.signal.zerospolesgain)
  - [`ZerosPolesGain.to_ss`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ZerosPolesGain.to_ss.html#scipy.signal.zerospolesgain.to_ss)
  - [`ZerosPolesGain.to_tf`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ZerosPolesGain.to_tf.html#scipy.signal.zerospolesgain.to_tf)
  - [`ZerosPolesGain.to_zpk`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ZerosPolesGain.to_zpk.html#scipy.signal.zerospolesgain.to_zpk)
- [`ZoomFFT`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ZoomFFT.html#scipy.signal.zoomfft)
  - [`ZoomFFT.points`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ZoomFFT.points.html#scipy.signal.zoomfft.points)
- [`dlti`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.dlti.html#scipy.signal.dlti)
  - [`dlti.bode`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.dlti.bode.html#scipy.signal.dlti.bode)
  - [`dlti.freqresp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.dlti.freqresp.html#scipy.signal.dlti.freqresp)
  - [`dlti.impulse`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.dlti.impulse.html#scipy.signal.dlti.impulse)
  - [`dlti.output`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.dlti.output.html#scipy.signal.dlti.output)
  - [`dlti.step`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.dlti.step.html#scipy.signal.dlti.step)
- [`lti`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lti.html#scipy.signal.lti)
  - [`lti.bode`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lti.bode.html#scipy.signal.lti.bode)
  - [`lti.freqresp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lti.freqresp.html#scipy.signal.lti.freqresp)
  - [`lti.impulse`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lti.impulse.html#scipy.signal.lti.impulse)
  - [`lti.output`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lti.output.html#scipy.signal.lti.output)
  - [`lti.step`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lti.step.html#scipy.signal.lti.step)
  - [`lti.to_discrete`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lti.to_discrete.html#scipy.signal.lti.to_discrete)

### scipy.signal.windows

[Module reference](https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows)

Function entries: **26**. Callable objects: **0**. Classes: **0**.

#### Functions

- [`barthann`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.barthann.html#scipy.signal.windows.barthann)
- [`bartlett`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.bartlett.html#scipy.signal.windows.bartlett)
- [`blackman`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.blackman.html#scipy.signal.windows.blackman)
- [`blackmanharris`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.blackmanharris.html#scipy.signal.windows.blackmanharris)
- [`bohman`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.bohman.html#scipy.signal.windows.bohman)
- [`boxcar`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.boxcar.html#scipy.signal.windows.boxcar)
- [`chebwin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.chebwin.html#scipy.signal.windows.chebwin)
- [`cosine`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.cosine.html#scipy.signal.windows.cosine)
- [`dpss`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.dpss.html#scipy.signal.windows.dpss)
- [`exponential`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.exponential.html#scipy.signal.windows.exponential)
- [`flattop`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.flattop.html#scipy.signal.windows.flattop)
- [`gaussian`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.gaussian.html#scipy.signal.windows.gaussian)
- [`general_cosine`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.general_cosine.html#scipy.signal.windows.general_cosine)
- [`general_gaussian`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.general_gaussian.html#scipy.signal.windows.general_gaussian)
- [`general_hamming`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.general_hamming.html#scipy.signal.windows.general_hamming)
- [`get_window`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.get_window.html#scipy.signal.windows.get_window)
- [`hamming`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.hamming.html#scipy.signal.windows.hamming)
- [`hann`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.hann.html#scipy.signal.windows.hann)
- [`kaiser`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.kaiser.html#scipy.signal.windows.kaiser)
- [`kaiser_bessel_derived`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.kaiser_bessel_derived.html#scipy.signal.windows.kaiser_bessel_derived)
- [`lanczos`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.lanczos.html#scipy.signal.windows.lanczos)
- [`nuttall`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.nuttall.html#scipy.signal.windows.nuttall)
- [`parzen`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.parzen.html#scipy.signal.windows.parzen)
- [`taylor`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.taylor.html#scipy.signal.windows.taylor)
- [`triang`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.triang.html#scipy.signal.windows.triang)
- [`tukey`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.tukey.html#scipy.signal.windows.tukey)

## Additional and compatibility namespaces

Many of these paths are deprecated compatibility aliases. Class and object method lists are omitted in these sections to avoid repeating the public API inventory.

### scipy.signal.bsplines

Function entries: **8**. Callable objects: **0**. Classes: **0**.

#### Functions

- `cspline1d`
- `cspline1d_eval`
- `cspline2d`
- `gauss_spline`
- `qspline1d`
- `qspline1d_eval`
- `sepfir2d`
- `spline_filter`

### scipy.signal.filter_design

Function entries: **46**. Callable objects: **0**. Classes: **1**.

#### Functions

- `band_stop_obj`
- `bessel`
- `besselap`
- `bilinear`
- `bilinear_zpk`
- `buttap`
- `butter`
- `buttord`
- `cheb1ap`
- `cheb1ord`
- `cheb2ap`
- `cheb2ord`
- `cheby1`
- `cheby2`
- `ellip`
- `ellipap`
- `ellipord`
- `findfreqs`
- `freqs`
- `freqs_zpk`
- `freqz`
- `freqz_sos`
- `freqz_zpk`
- `gammatone`
- `group_delay`
- `iircomb`
- `iirdesign`
- `iirfilter`
- `iirnotch`
- `iirpeak`
- `lp2bp`
- `lp2bp_zpk`
- `lp2bs`
- `lp2bs_zpk`
- `lp2hp`
- `lp2hp_zpk`
- `lp2lp`
- `lp2lp_zpk`
- `normalize`
- `sos2tf`
- `sos2zpk`
- `sosfreqz`
- `tf2sos`
- `tf2zpk`
- `zpk2sos`
- `zpk2tf`

#### Classes

- `BadCoefficients`

### scipy.signal.fir_filter_design

Function entries: **9**. Callable objects: **0**. Classes: **0**.

#### Functions

- `firls`
- `firwin`
- `firwin2`
- `firwin_2d`
- `kaiser_atten`
- `kaiser_beta`
- `kaiserord`
- `minimum_phase`
- `remez`

### scipy.signal.lti_conversion

Function entries: **9**. Callable objects: **0**. Classes: **0**.

#### Functions

- `abcd_normalize`
- `cont2discrete`
- `normalize`
- `ss2tf`
- `ss2zpk`
- `tf2ss`
- `tf2zpk`
- `zpk2ss`
- `zpk2tf`

### scipy.signal.ltisys

Function entries: **24**. Callable objects: **0**. Classes: **5**.

#### Functions

- `abcd_normalize`
- `bode`
- `cont2discrete`
- `dbode`
- `dfreqresp`
- `dimpulse`
- `dlsim`
- `dstep`
- `freqresp`
- `freqs`
- `freqs_zpk`
- `freqz`
- `freqz_zpk`
- `impulse`
- `lsim`
- `normalize`
- `place_poles`
- `ss2tf`
- `ss2zpk`
- `step`
- `tf2ss`
- `tf2zpk`
- `zpk2ss`
- `zpk2tf`

#### Classes

- `StateSpace`
- `TransferFunction`
- `ZerosPolesGain`
- `dlti`
- `lti`

### scipy.signal.signaltools

Function entries: **36**. Callable objects: **0**. Classes: **1**.

#### Functions

- `cheby1`
- `choose_conv_method`
- `convolve`
- `convolve2d`
- `correlate`
- `correlate2d`
- `correlation_lags`
- `decimate`
- `deconvolve`
- `detrend`
- `fftconvolve`
- `filtfilt`
- `firwin`
- `get_window`
- `hilbert`
- `hilbert2`
- `invres`
- `invresz`
- `lfilter`
- `lfilter_zi`
- `lfiltic`
- `medfilt`
- `medfilt2d`
- `oaconvolve`
- `order_filter`
- `resample`
- `resample_poly`
- `residue`
- `residuez`
- `sosfilt`
- `sosfilt_zi`
- `sosfiltfilt`
- `unique_roots`
- `upfirdn`
- `vectorstrength`
- `wiener`

#### Classes

- `dlti`

### scipy.signal.spectral

Function entries: **11**. Callable objects: **0**. Classes: **0**.

#### Functions

- `check_COLA`
- `check_NOLA`
- `coherence`
- `csd`
- `get_window`
- `istft`
- `lombscargle`
- `periodogram`
- `spectrogram`
- `stft`
- `welch`

### scipy.signal.spline

Function entries: **1**. Callable objects: **0**. Classes: **0**.

#### Functions

- `sepfir2d`

### scipy.signal.waveforms

Function entries: **6**. Callable objects: **0**. Classes: **0**.

#### Functions

- `chirp`
- `gausspulse`
- `sawtooth`
- `square`
- `sweep_poly`
- `unit_impulse`

### scipy.signal.wavelets

Function entries: **0**. Callable objects: **0**. Classes: **0**.

This namespace contains submodules or has no exported callable entries in this runtime.

### scipy.signal.windows.windows

Function entries: **24**. Callable objects: **0**. Classes: **0**.

#### Functions

- `barthann`
- `bartlett`
- `blackman`
- `blackmanharris`
- `bohman`
- `boxcar`
- `chebwin`
- `cosine`
- `dpss`
- `exponential`
- `flattop`
- `gaussian`
- `general_cosine`
- `general_gaussian`
- `general_hamming`
- `get_window`
- `hamming`
- `hann`
- `kaiser`
- `nuttall`
- `parzen`
- `taylor`
- `triang`
- `tukey`
