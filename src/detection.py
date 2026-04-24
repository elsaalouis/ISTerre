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
"""

import numpy as np
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

    # -- SNR full mean: event window vs equal-length noise window --------------
    try:
        s = tr_filt.slice(t_on, t_off)
        if t_on - duration >= t_start:
            n = tr_filt.slice(t_on - duration, t_on)
        elif t_off + duration <= t_end:
            n = tr_filt.slice(t_off, t_off + duration)
        else:
            n = s
        snr_dict['SNR_full_mean'] = _mean_env(s) / _mean_env(n)
    except Exception:
        snr_dict['SNR_full_mean'] = np.nan

    return snr_dict



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
