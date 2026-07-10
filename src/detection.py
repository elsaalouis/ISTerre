"""
detection.py
============
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : April 2026

Detection algorithms and associated helpers:
  - classic STA/LTA (scripts 01/02)
  - SNR computation for detected windows (script 04)
  - denoiser fidelity metrics — raw vs. denoised correlation/energy ratio (script 03d)
  - window merging for the spectrogram-based detector (script 04)
  - kurtosis-based onset refiner for rockslides (Fuchs et al. 2018)
"""

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis
from obspy.signal.trigger import classic_sta_lta, trigger_onset


# =============================================================================
# CLASSIC STA/LTA (scripts 01/02)
# =============================================================================

def run_sta_lta(tr, sta_s, lta_s, thres_on, thres_off):
    """
    Compute the STA/LTA characteristic function and detect trigger windows

    Parameters
    ----------
    tr        : obspy.Trace — preprocessed vertical-component trace
    sta_s     : float — Short-Term Average window length in seconds
    lta_s     : float — Long-Term Average window length in seconds
    thres_on  : float — STA/LTA ratio above which a trigger is declared ON
    thres_off : float — STA/LTA ratio below which the trigger turns OFF

    Returns
    -------
    cft    : numpy array — STA/LTA ratio at every sample of the trace
    on_off : list of [on_sample, off_sample] pairs
    """
    fs   = tr.stats.sampling_rate
    nsta = int(sta_s * fs)     # window lengths in samples
    nlta = int(lta_s * fs)

    cft    = classic_sta_lta(tr.data, nsta, nlta)
    on_off = trigger_onset(cft, thres_on, thres_off)
    return cft, on_off


def summarise_detections(tr, on_off, t_start, thres_on):
    """
    Print detection results for one trace and return the trigger times

    Parameters
    ----------
    tr       : obspy.Trace
    on_off   : list of [on_sample, off_sample] pairs (from run_sta_lta)
    t_start  : UTCDateTime — absolute start time of the trace
    thres_on : float — threshold used (printed in the "no trigger" message)

    Returns
    -------
    results : list of (t_on, t_off) UTCDateTime tuples
    """
    results = []
    sta     = f"{tr.stats.network}.{tr.stats.station}"
    fs      = tr.stats.sampling_rate

    if len(on_off) == 0:
        print(f"    {sta:>15s}  — no trigger (ratio never exceeded {thres_on})")
    else:
        for k, (i_on, i_off) in enumerate(on_off):
            t_on  = t_start + i_on  / fs
            t_off = t_start + i_off / fs
            print(f"    {sta:>15s}  trigger {k+1}: "
                  f"ON={t_on.strftime('%H:%M:%S')}  "
                  f"OFF={t_off.strftime('%H:%M:%S')}  "
                  f"duration={t_off - t_on:.1f}s")
            results.append((t_on, t_off))
    return results



# =============================================================================
# SNR COMPUTATION (script 04)
# =============================================================================

# did anything extreme happen in the signal window compared to the ambient noise level?
def signal2noise_median(y_noise, y_signal):
    mad_pre_event = np.median(np.abs(y_noise - np.mean(y_noise))) # MAD: median absolute deviation
    percentile    = np.percentile(np.abs(y_signal - np.mean(y_signal)), 99.5) # amplitude that 99.5% of samples are below
    if mad_pre_event > 0:
        return percentile / mad_pre_event # near-maximum amplitude during the event / typical fluctuation amplitude in the noise
    else:
        return 0


def compute_snr(tr_filt, t_on, t_off):
    """
    Compute five SNR measures for a detected event, following Groult et al.
    All measures use the envelope (absolute value) of the filtered trace.

    Parameters
    ----------
    tr_filt   : obspy.Trace — bandpass-filtered velocity trace (full segment)
    t_on/off  : UTCDateTime — start and end of the detected event

    Returns
    -------
    snr_dict : dict with keys:
        SNR               — peak-centred 5 s signal vs 5 s post-event noise
        SNR_picking_5_5   — 5 s after onset vs 5 s before onset
        SNR_picking_3_3   — 3 s after onset vs 3 s before onset
        SNR_picking_1_3   — 1 s after onset vs 3 s before onset
        SNR_full_mean     — mean of full detection window vs equal-length noise window
        SNR_full_median   — median of full detection window vs equal-length noise window
                            (Groult et al. 2026 use both mean AND median > 3 as quality gate)
        SNR_s2n_median    — tutor's robust metric: 99.5th percentile of |signal| / MAD of noise
                            (same noise/signal windows as SNR_full_mean/median)
    """
    def _mean_env(tr_slice):
        """Mean absolute amplitude of a trace slice; returns 1.0 if empty."""
        if tr_slice is None or tr_slice.stats.npts == 0:
            return 1.0
        return float(np.mean(np.abs(tr_slice.data))) or 1.0

    snr_dict = {}
    duration = t_off - t_on
    fs       = tr_filt.stats.sampling_rate
    t_start  = tr_filt.stats.starttime
    t_end    = tr_filt.stats.endtime

    # -- SNR: peak-centred (5 s window around the amplitude peak) -------------
    try:
        seg   = tr_filt.slice(t_on, t_off)
        env   = np.abs(seg.data)
        i_max = int(env.argmax())
        hw    = int(2.5 * fs)                        # half-width = 2.5 s
        i1    = max(0, i_max - hw)
        i2    = min(len(env), i_max + hw)
        sig   = float(np.mean(env[i1:i2])) or 1.0
        nz    = (tr_filt.slice(t_off, t_off + 5) if t_off + 5 <= t_end
                 else tr_filt.slice(t_on - 5, t_on) if t_on - 5 >= t_start
                 else seg)
        snr_dict['SNR'] = sig / _mean_env(nz)
    except Exception:
        snr_dict['SNR'] = np.nan

    # -- SNR picking: signal window vs noise window just before onset ----------
    for sig_sec, noi_sec, key in [(5, 5, 'SNR_picking_5_5'),
                                  (3, 3, 'SNR_picking_3_3'),
                                  (1, 3, 'SNR_picking_1_3')]:
        try:
            s = tr_filt.slice(t_on, t_on + sig_sec)
            n = (tr_filt.slice(t_on - noi_sec, t_on)
                 if t_on - noi_sec >= t_start else s)
            snr_dict[key] = _mean_env(s) / _mean_env(n)
        except Exception:
            snr_dict[key] = np.nan

    # -- SNR full mean / median: event window vs equal-length noise window ----
    try:
        s = tr_filt.slice(t_on, t_off)
        if t_on - duration >= t_start:
            n = tr_filt.slice(t_on - duration, t_on)
        elif t_off + duration <= t_end:
            n = tr_filt.slice(t_off, t_off + duration)
        else:
            n = s
        env_s    = np.abs(s.data)
        env_n    = np.abs(n.data)
        mean_n   = float(np.mean(env_n))   or 1.0
        median_n = float(np.median(env_n)) or 1.0
        snr_dict['SNR_full_mean']   = float(np.mean(env_s))   / mean_n
        snr_dict['SNR_full_median'] = float(np.median(env_s)) / median_n
        # Tutor's robust metric — uses raw (non-envelope) data, same noise/signal windows
        snr_dict['SNR_s2n_median']  = signal2noise_median(n.data, s.data)
    except Exception:
        snr_dict['SNR_full_mean']   = np.nan
        snr_dict['SNR_full_median'] = np.nan
        snr_dict['SNR_s2n_median']  = np.nan

    return snr_dict


def compute_snr_numpy(full_signal, itp, det_duration_s, sps):
    """
    Numpy-only version of compute_snr — same metrics, no ObsPy required.

    Used by scripts that work with raw numpy arrays (e.g. 03d post-denoiser
    processing) where an ObsPy Trace is not available.

    The window layout mirrors the DeepDenoiser 30 s NPZ convention:
      noise_window  = full_signal[    0 : itp]              (pre-onset)
      signal_window = full_signal[  itp : itp + dur_samp]   (post-onset)
    where dur_samp = min(int(det_duration_s * sps), len(full_signal) - itp).

    Parameters
    ----------
    full_signal    : 1D numpy array — waveform (e.g. 3000 samples at 100 Hz)
    itp            : int   — onset sample index (noise/signal split point)
    det_duration_s : float — event duration in seconds (from catalog)
    sps            : float — sampling rate in Hz

    Returns
    -------
    dict with keys: SNR, SNR_picking_5_5, SNR_picking_3_3, SNR_picking_1_3,
                    SNR_full_mean, SNR_full_median, SNR_s2n_median
    """
    _NAN_DICT = {k: np.nan for k in ('SNR', 'SNR_picking_5_5', 'SNR_picking_3_3',
                                      'SNR_picking_1_3', 'SNR_full_mean',
                                      'SNR_full_median', 'SNR_s2n_median')}
    n = len(full_signal)

    # Noise window: everything before the onset
    noise_win = full_signal[:itp]

    # Signal window: onset to onset + duration (capped at array length)
    dur_samp = max(int(det_duration_s * sps), 200)   # at least 200 samples
    sig_end  = min(n, itp + dur_samp)
    sig_win  = full_signal[itp:sig_end]

    if len(noise_win) == 0 or len(sig_win) == 0:
        return _NAN_DICT

    env_s    = np.abs(sig_win)
    env_n    = np.abs(noise_win)
    mean_n   = float(np.mean(env_n))   or 1.0
    median_n = float(np.median(env_n)) or 1.0

    snr = {}

    # Full-window mean / median  (primary quality-gate metrics)
    snr['SNR_full_mean']   = float(np.mean(env_s))   / mean_n
    snr['SNR_full_median'] = float(np.median(env_s)) / median_n
    snr['SNR_s2n_median']  = signal2noise_median(noise_win, sig_win)

    # Peak-centred SNR (5 s half-window around the amplitude maximum)
    i_max    = int(env_s.argmax())
    hw       = int(2.5 * sps)
    peak_env = env_s[max(0, i_max - hw) : min(len(env_s), i_max + hw)]
    snr['SNR'] = float(np.mean(peak_env)) / mean_n if len(peak_env) else np.nan

    # Picking-window SNRs (signal N s after onset vs noise M s before onset)
    for sig_sec, noi_sec, key in [(5, 5, 'SNR_picking_5_5'),
                                   (3, 3, 'SNR_picking_3_3'),
                                   (1, 3, 'SNR_picking_1_3')]:
        s_w = full_signal[itp : min(n, itp + int(sig_sec * sps))]
        n_w = full_signal[max(0, itp - int(noi_sec * sps)) : itp]
        if len(s_w) == 0 or len(n_w) == 0:
            snr[key] = np.nan
        else:
            snr[key] = float(np.mean(np.abs(s_w))) / (float(np.mean(np.abs(n_w))) or 1.0)

    return snr



# =============================================================================
# DENOISER FIDELITY METRICS (script 03d)
# =============================================================================

def compute_denoise_correlation(raw_signal, denoised_signal, itp, det_duration_s, sps):
    """
    Compare a denoised waveform against its own raw (pre-denoiser) input to check
    whether DeepDenoiser preserved genuine signal structure or is hallucinating it.

    SNR alone can go up even when a denoiser just invents smooth-looking structure
    rather than recovering the real event, so 03d_rescue_feature_extraction.py uses
    this alongside the before/after SNR comparison as a sanity check: it reports how
    well the denoised trace still resembles the raw input, separately in the noise
    window and the signal window.

    Window layout matches compute_snr_numpy:
      noise_window  = full_signal[    0 : itp]              (pre-onset)
      signal_window = full_signal[  itp : itp + dur_samp]   (post-onset)

    Parameters
    ----------
    raw_signal      : 1D numpy array — original waveform before denoising
    denoised_signal : 1D numpy array — DeepDenoiser output, aligned with raw_signal
    itp             : int   — onset sample index (noise/signal split point)
    det_duration_s  : float — event duration in seconds (from catalog)
    sps             : float — sampling rate in Hz

    Returns
    -------
    dict with keys:
      corr_signal          — Pearson r between raw and denoised, signal window.
                              Expect moderate-to-high if real structure was recovered;
                              near-zero despite a large SNR gain is a red flag for
                              hallucination.
      corr_noise            — Pearson r between raw and denoised, noise window.
                              Expect low — a well-behaved denoiser should not reproduce
                              the noise it removed.
      energy_ratio_signal   — std(denoised) / std(raw) in the signal window.
                              Expect close to 1 — signal energy preserved, not erased.
      energy_ratio_noise    — std(denoised) / std(raw) in the noise window.
                              Expect << 1 — noise energy suppressed.
    """
    _NAN_DICT = {k: np.nan for k in ('corr_signal', 'corr_noise',
                                      'energy_ratio_signal', 'energy_ratio_noise')}

    n = min(len(raw_signal), len(denoised_signal))
    raw_signal      = raw_signal[:n]
    denoised_signal = denoised_signal[:n]

    noise_raw = raw_signal[:itp]
    noise_den = denoised_signal[:itp]

    dur_samp = max(int(det_duration_s * sps), 200)
    sig_end  = min(n, itp + dur_samp)
    sig_raw  = raw_signal[itp:sig_end]
    sig_den  = denoised_signal[itp:sig_end]

    if len(noise_raw) < 2 or len(sig_raw) < 2:
        return _NAN_DICT

    def _safe_corr(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    def _safe_ratio(raw_win, den_win):
        std_raw = np.std(raw_win)
        return float(np.std(den_win) / std_raw) if std_raw > 0 else np.nan

    return {
        'corr_signal'        : _safe_corr(sig_raw, sig_den),
        'corr_noise'         : _safe_corr(noise_raw, noise_den),
        'energy_ratio_signal': _safe_ratio(sig_raw, sig_den),
        'energy_ratio_noise' : _safe_ratio(noise_raw, noise_den),
    }



# =============================================================================
# KURTOSIS ONSET REFINER — Fuchs et al. (2018)
# =============================================================================

def refine_onset_kurtosis(tr, t_on, dt_s=5.0, search_before=10.0, search_after=1.0):
    """
    Refine a preliminary STA/LTA onset time using the kurtosis-based picker described in Fuchs (2018) 

    Designed for rockslide signals whose onsets are emergent, so the STA/LTA fires later than the true onset 
     -> the kurtosis rises sharply when impulsive seismic energy enters the sliding window, allowing precise detection of the true signal start

    Algorithm (Fuchs eq. 1–3):
      1. Slide a window of dt_s seconds from (t_on - search_before) to (t_on + search_after) and compute kurtosis β at each step → CF(t)
         β = 3 for Gaussian noise; rises above 3 when a signal arrives
      2. Build cCF(k) = cumulative sum of only the *positive* slopes of CF(t) -> accumulates rises and ignores flat/decreasing parts
      3. Refined onset = time where d(cCF)/dt is maximum (steepest kurtosis rise)
         If the maximum is a plateau, the first occurrence is used (Fuchs).
         The onset corresponds to the START of the kurtosis window at that step.

    Parameters
    ----------
    tr   : obspy.Trace — bandpass-filtered trace, Fuchs uses 1–5 Hz (suppresses microseism and enhances the emergent onset)
    t_on : UTCDateTime — preliminary onset from the spectrogram STA/LTA detector (DetecteurV3)
    dt_s : float       — kurtosis sliding window length in seconds, Fuchs: 5 s
    search_before : float — search start = t_on − search_before seconds, Fuchs: 10 s
    search_after  : float — search end   = t_on + search_after  seconds, Fuchs: 1 s

    Returns
    -------
    t_refined : UTCDateTime — refined onset time, falls back to t_on if refinement fails
    info : dict
        Diagnostic arrays for plotting (all times are seconds from trace start):
          't0'          — UTCDateTime of the trace slice start
          'cf_times_s'  — time axis of CF (seconds from t0), length M
          'cf_values'   — CF(t) values, length M
          'ccf_values'  — cCF values, length M
          'dccf'        — d(cCF)/dt, length M-1 (positive slopes)
          't_on_rel'    — preliminary onset in seconds from t0
          't_refined_rel'  — refined onset in seconds from t0
    """
    fs   = tr.stats.sampling_rate
    nwin = max(2, int(dt_s * fs))   # number of samples in the kurtosis window -> converting the window duration dt_s [s] into samples using the sampling rate fs

    # Slice the trace to cover the full search window plus one window-length of tail so that the last 
    # kurtosis window (starting at t_on + search_after) has enough samples to be computed.
    t_slice_start = max(t_on - search_before,         tr.stats.starttime)
    t_slice_end   = min(t_on + search_after + dt_s,   tr.stats.endtime)

    tr_slice = tr.slice(t_slice_start, t_slice_end)
    if tr_slice.stats.npts < nwin + 2:
        return t_on, {}

    data = tr_slice.data   # raw array of amplitude values
    n    = len(data)       # length in samples
    t0   = tr_slice.stats.starttime   # UTCDateTime

    # ---- Step 1: compute CF(t) — kurtosis of the window starting at sample i ----
    # We timestamp each value at the window START so that the onset (= argmax of d(cCF)/dt) gives the pick time without any dt_s correction
    cf_all   = []   # time series of kurtosis values (one per window position)
    tcf_all  = []   # timestamps of the START of each kurtosis window (seconds from t0)

    for i in range(n - nwin):        # stops at n-nwin so the last window data[n-nwin : n] doesn't go out of bounds
        window = data[i : i + nwin]  # extract the samples inside the current window
        beta   = float(scipy_kurtosis(window, fisher=False))   # standard Pearson kurtosis (β=3 for Gaussian)
        cf_all.append(beta)
        tcf_all.append(i / fs)       # window START time (seconds from t0)

    cf_all  = np.array(cf_all)    # convert lists to numpy arrays
    tcf_all = np.array(tcf_all)   # seconds from t0

    # Restrict to the search window [t_on - search_before, t_on + search_after]
    t_on_rel         = float(t_on - t0)
    search_start_rel = t_on_rel - search_before
    search_end_rel   = t_on_rel + search_after

    # Keep only the CF values whose window START falls inside the search zone
    mask = (tcf_all >= search_start_rel) & (tcf_all <= search_end_rel)
    if mask.sum() < 3:    # if fewer than 3 values survive, fall back to the original onset
        return t_on, {}

    cf   = cf_all[mask]
    tcf  = tcf_all[mask]   # time of window START, seconds from t0

    # ---- Step 2: cCF = cumulative sum of positive slopes only ----
    slopes     = np.diff(cf)    # compute the difference between consecutive CF values: slopes[k] = cf[k+1] - cf[k]
    pos_slopes = np.where(slopes > 0, slopes, 0.0)               # only accumulate rises
    ccf        = np.concatenate([[0.0], np.cumsum(pos_slopes)])  # cumsum computes the running total of pos_slopes

    # ---- Step 3: refined onset = argmax of d(cCF)/dt ----
    dccf = np.diff(ccf)   # = pos_slopes (same values, length M-1)

    # With window-START timestamps and the corrected search zone [t_on-search_before,
    # t_on+search_after], argmax(dccf) no longer locks onto the Groult onset:
    # it finds the window start where the kurtosis rise is largest within the
    # actual signal content of the search zone — the tallest orange bar in the
    # diagnostic plot. This is the standard Fuchs (2018) criterion.
    i_peak = int(np.argmax(dccf))

    # tcf[i_peak] is the START of the kurtosis window at the onset step.
    t_refined_rel = tcf[i_peak]
    t_refined     = t0 + t_refined_rel

    # Clamp to trace boundaries
    t_refined = max(t_refined, tr.stats.starttime)
    t_refined = min(t_refined, tr.stats.endtime)

    info = {
        't0'            : t0,
        'cf_times_s'    : tcf,           # window START times of CF, seconds from t0
        'cf_values'     : cf,
        'ccf_values'    : ccf,
        'dccf'          : dccf,
        'i_peak'        : i_peak,        # index used for the refined onset (in cf/tcf/ccf)
        't_on_rel'      : t_on_rel,
        't_refined_rel' : t_refined_rel,
    }
    return t_refined, info


# =============================================================================
# WINDOW MERGING FOR SLIDING-WINDOW DETECTOR (script 04)
# =============================================================================

def merge_window_events(total_events, total_thresholds, new_events, new_thresholds):
    """
    Merge detections from a new 10-min window into the running total

    If the first event of the new window starts within 60 s of the last event
    of the previous window, the two are merged into a single detection
     -> the new event's start is replaced by the previous one's start

    Parameters
    ----------
    total_events, total_thresholds : dicts of already-accumulated detections
    new_events, new_thresholds     : dicts from the current window

    Returns
    -------
    Updated (total_events, total_thresholds) dicts
    """
    if not new_events:
        return total_events, total_thresholds

    if not total_events:
        return new_events.copy(), new_thresholds.copy()

    last_key  = f"Event_{len(total_events)}"
    first_key = "Event_1"

    gap = new_events[first_key][0] - total_events[last_key][1]
    if gap < 60.0:
        # extend the last event to cover the start of the new one
        new_events[first_key][0]     = total_events[last_key][0]
        new_thresholds[first_key][0] = total_thresholds[last_key][0]
        del total_events[last_key]
        del total_thresholds[last_key]

    offset = len(total_events)
    for k, ev_key in enumerate(new_events, 1):
        total_events[f"Event_{offset + k}"]     = new_events[ev_key]
        total_thresholds[f"Event_{offset + k}"] = new_thresholds[ev_key]

    return total_events, total_thresholds
