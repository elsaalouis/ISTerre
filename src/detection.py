"""
detection.py
============
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : April 2026

Detection algorithms and associated helpers:
  - classic STA/LTA (scripts 01/02)
  - SNR computation for detected windows (script 04)
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



# =============================================================================
# KURTOSIS ONSET REFINER — Fuchs et al. (2018) / Hibert et al. (2014)
# =============================================================================

def refine_onset_kurtosis(tr, t_on, dt_s=5.0, search_before=10.0, search_after=1.0):
    """
    Refine a preliminary STA/LTA onset time using the kurtosis-based picker
    described in Fuchs et al. (2018) and Hibert et al. (2014).

    Designed for rockslide signals whose onsets are emergent (gradual build-up),
    so the STA/LTA fires later than the true onset. The kurtosis rises sharply
    when impulsive seismic energy first enters the sliding window, allowing
    precise detection of the true signal start.

    Algorithm (Fuchs eq. 1–3):
      1. Slide a window of dt_s seconds from (t_on - search_before) to
         (t_on + search_after) and compute kurtosis β at each step → CF(t).
         β = 3 for Gaussian noise; rises above 3 when a signal arrives.
      2. Build cCF(k) = cumulative sum of only the *positive* slopes of CF(t).
         This accumulates rises and ignores flat/decreasing parts.
      3. Refined onset = time where d(cCF)/dt is maximum (steepest kurtosis rise).
         If the maximum is a plateau, the first occurrence is used (Fuchs).
         The onset corresponds to the START of the kurtosis window at that step.

    Parameters
    ----------
    tr            : obspy.Trace
        Bandpass-filtered trace. Fuchs et al. use 1–5 Hz (suppresses
        microseism and enhances the emergent onset). Pass a 1–5 Hz filtered
        copy, not the 1–20 Hz copy used for SNR.
    t_on          : UTCDateTime
        Preliminary onset from the spectrogram STA/LTA detector (DetecteurV3).
    dt_s          : float
        Kurtosis sliding window length in seconds. Fuchs: 5 s.
    search_before : float
        Search start = t_on − search_before seconds. Fuchs: 10 s.
    search_after  : float
        Search end   = t_on + search_after  seconds. Fuchs: 1 s.

    Returns
    -------
    t_refined : UTCDateTime
        Refined onset time. Falls back to t_on if refinement fails
        (trace too short, insufficient samples, flat CF, etc.).
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
    nwin = max(2, int(dt_s * fs))   # kurtosis window in samples

    # Slice the trace: need dt_s of lead-in before the search window
    t_slice_start = max(t_on - search_before - dt_s, tr.stats.starttime)
    t_slice_end   = min(t_on + search_after,          tr.stats.endtime)

    tr_slice = tr.slice(t_slice_start, t_slice_end)
    if tr_slice.stats.npts < nwin + 2:
        return t_on, {}

    data = tr_slice.data
    n    = len(data)
    t0   = tr_slice.stats.starttime   # UTCDateTime

    # ---- Step 1: compute CF(t) — kurtosis of the window ending at sample i+nwin ----
    cf_all   = []
    tcf_all  = []   # time of the END of each kurtosis window (seconds from t0)

    for i in range(n - nwin):
        window = data[i : i + nwin]
        beta   = float(scipy_kurtosis(window, fisher=False))   # standard kurtosis (β=3 for Gaussian)
        cf_all.append(beta)
        tcf_all.append((i + nwin) / fs)

    cf_all  = np.array(cf_all)
    tcf_all = np.array(tcf_all)   # seconds from t0

    # Restrict to the search window [t_on - search_before, t_on + search_after]
    t_on_rel         = float(t_on - t0)
    search_start_rel = t_on_rel - search_before
    search_end_rel   = t_on_rel + search_after

    mask = (tcf_all >= search_start_rel) & (tcf_all <= search_end_rel)
    if mask.sum() < 3:
        return t_on, {}

    cf   = cf_all[mask]
    tcf  = tcf_all[mask]   # time of window END, seconds from t0

    # ---- Step 2: cCF = cumulative sum of positive slopes only ----
    slopes     = np.diff(cf)
    pos_slopes = np.where(slopes > 0, slopes, 0.0)
    ccf        = np.concatenate([[0.0], np.cumsum(pos_slopes)])

    # ---- Step 3: refined onset = time where d(cCF)/dt is maximum ----
    dccf   = np.diff(ccf)           # same as pos_slopes; length M-1
    i_peak = int(np.argmax(dccf))   # index of steepest positive kurtosis rise

    # tcf[i_peak] is the END of the kurtosis window at the peak step.
    # The onset corresponds to the START of that window: subtract dt_s.
    t_refined_rel = tcf[i_peak] - dt_s
    t_refined     = t0 + t_refined_rel

    # Clamp to trace boundaries
    t_refined = max(t_refined, tr.stats.starttime)
    t_refined = min(t_refined, tr.stats.endtime)

    info = {
        't0'            : t0,
        'cf_times_s'    : tcf,           # time axis of CF, seconds from t0
        'cf_values'     : cf,
        'ccf_values'    : ccf,
        'dccf'          : dccf,
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
