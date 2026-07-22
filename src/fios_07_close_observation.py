"""
FIOS LANDSLIDE — CLOSE-UP MICROSEISMICITY OBSERVATION
======================================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : July 2026

Outputs (all saved to OUTPUT_DIR/<date>/)
------------------------------------------
  fig_01_night_10h_<date>.png        — full night: waveform + spectrogram + sliding kurtosis
  fig_interactive_<date>.html        — auto-picked INTERACTIVE_WINDOW_MIN window, Plotly: zoom/pan yourself on time or frequency in a browser
  fig_03_loudest_pick_<date>.png     — few-second raw waveform zoom (PICK band) on the single loudest fine-picker event in that window
  fig_04_interevent_hist_<date>.png  — histogram of inter-event times (fine picker)
  events_<date>.csv                  — one row per picked event (+ 99 features if ENABLE_FEATURE_EXTRACTION=True)
  day_selection_summary.csv           — which dates were selected and why (OUTPUT_DIR root)
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

DATA_ROOT  = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\data\FIOS"
OUTPUT_DIR = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\fios\07_close_observation"

# Existing hourly PSD table from fios_04_spectral_energy_psd.py — reused for the automatic day-ranking
PSD_HOURLY_CSV = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\fios\04_psd\psd_hourly.csv"

NETWORK  = "XT"
STATION  = "FIO1"
LOCATION = "01"
CHANNEL  = "DHZ"

# ---- Day selection ----------------------------------------------------------
KEY_EVENTS = [
    ("2026-04-13", "1st destabilisation"),
    ("2026-05-25", "2nd destabilisation"),
    ("2026-06-23", "3rd destabilisation"),
]
N_AUTO_DAYS   = 0     # how many extra nights to add, ranked by night-only 1-5 Hz energy
MANUAL_DATES  = []    # add dates here manually, e.g. ["2026-06-02"], to force them in

# ---- Night time window (UTC) -------------------------------------------------
WINDOW_START_UTC  = 18
WINDOW_DURATION_H = 10

# ---- Display bandpass: spectrogram band vs waveform band, kept separate ------
FILT_FMIN = 1.0    # spectrogram band (also used for the Level 1 context figure)
FILT_FMAX = 60.0

# Waveform panel only (Level 1 + the interactive figure)
WAVE_FMIN = 1.0
WAVE_FMAX = 20.0

# ---- Spectrogram window length: Level 1 only (context, kept as a plain scipy.signal.spectrogram)
LEVEL1_WINDOW_S = 10.0    # full night -> Δf = 0.10 Hz
SPEC_OVERLAP    = 0.90

SUBWINDOW_BAND = (2.0, 10.0)   # band used to auto-pick the loudest sub-window

# ---- Interactive zoom window (auto-picked, exported as Plotly .html) ---------
INTERACTIVE_WINDOW_MIN = 120   # length of the auto-picked window you'll explore yourself
#   INTERACTIVE_HOP_S     : Δt of the final spectrogram (how far the analysis bin
#                 slides between columns) — "how often do we get a new column";
#                 does not by itself change blur.
#   INTERACTIVE_BIN_S     : length of data feeding ONE column's Welch average —
#                 the real temporal "blur" width. Smaller = sharper in time,
#                 but fewer sub-segments fit inside -> noisier, unless
#                 compensated with a smaller SUBSEG_S and/or higher overlap.
#   INTERACTIVE_SUBSEG_S  : length of each Welch sub-segment inside a bin -> Δf = 1/SUBSEG_S
#   INTERACTIVE_SUBSEG_OVERLAP : overlap between sub-segments inside a bin ->
#                 more overlap = more sub-segments averaged = smoother
INTERACTIVE_BIN_S          = 0.5
INTERACTIVE_SUBSEG_S       = 0.5
INTERACTIVE_SUBSEG_OVERLAP = 0.75
INTERACTIVE_HOP_S          = 0.2

# 'cdn'    -> small file, needs internet to view
# 'inline' -> larger file (+~3-4 MB for plotly.js), fully offline-viewable
PLOTLYJS_MODE = 'cdn'

FREQ_MIN_PLOT = 1.0
FREQ_MAX_PLOT = 60.0

# Level-1 colour scale fixed across nights for cross-night comparability
# (same convention as fios_05's VSCALE_MODE="fixed")
LEVEL1_VMIN_DB = -15
LEVEL1_VMAX_DB =  55

# Raw waveform zoom (seconds each side) around the single loudest pick in Level 3
FINE_TRACE_MARGIN_S = 3.0

# ---- Fine-scale event picker --------------------------------------------------
PICK_FREQMIN   = 10.0
PICK_FREQMAX   = 60.0
PICK_STA_S     = 0.08
PICK_LTA_S     = 4.0
PICK_THR_ON    = 6.0
PICK_THR_OFF   = 2.0
PICK_MIN_DUR_S = 0.1
PICK_MIN_GAP_S = 0.3

FEATURE_PAD_S = 0.2   # padding added on each side of a pick before feature extraction

# The 99-feature Maggi/Hibert/Provost set (seismic_params.py / features.py)
ENABLE_FEATURE_EXTRACTION = False

# Sliding kurtosis (envelope "smoothness" diagnostic, computed on tr_filt)
KURT_WIN_S  = 2.0
KURT_STEP_S = 0.5



# =============================================================================
# SECTION 2 — IMPORTS
# =============================================================================

import os
import sys
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates  as mdates

from obspy                 import UTCDateTime, read, Stream
from scipy.signal          import spectrogram as sp_spectrogram, welch
from scipy.signal.windows  import hann as hann_window
from scipy.stats           import kurtosis as scipy_kurtosis

import plotly.graph_objects as go
from plotly.subplots        import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detection import run_sta_lta
from features  import extract_features, FEATURE_NAMES

os.makedirs(OUTPUT_DIR, exist_ok=True)



# =============================================================================
# SECTION 3 — HELPER FUNCTIONS
# =============================================================================

def select_target_dates(psd_csv_path, key_events, n_auto, manual_dates):
    """
    Build the final list of nights to process, with a human-readable reason
    for each one, so the choice is transparent (saved to day_selection_summary.csv).

    Ranking metric for the automatic picks: median night-only 1-5 Hz band
    energy (dB) per calendar day, taken directly from fios_04's psd_hourly.csv
    (column 'energy_db' + 'is_night'). No waveform is re-read for this step.
    """
    rows = []
    seen = set()

    for d, label in key_events:
        rows.append({'date': d, 'reason': label, 'energy_db_1_5hz_night': np.nan})
        seen.add(d)

    for d in manual_dates:
        if d not in seen:
            rows.append({'date': d, 'reason': 'manual', 'energy_db_1_5hz_night': np.nan})
            seen.add(d)

    energy_by_day = {}
    if os.path.exists(psd_csv_path):
        df = pd.read_csv(psd_csv_path, parse_dates=['datetime'])
        df_night = df[df['is_night'] == True]
        daily_night_energy = (df_night.groupby('day')['energy_db']
                              .median().sort_values(ascending=False))
        energy_by_day = daily_night_energy.to_dict()

        n_added = 0
        for day_str, e_db in daily_night_energy.items():
            if day_str in seen:
                continue
            rows.append({'date': day_str,
                        'reason': f'auto top-{n_auto} energy (1-5 Hz, night)',
                        'energy_db_1_5hz_night': round(float(e_db), 1)})
            seen.add(day_str)
            n_added += 1
            if n_added >= n_auto:
                break
    else:
        print(f"  [WARN] psd_hourly.csv not found at:\n    {psd_csv_path}")
        print(f"         Run fios_04_spectral_energy_psd.py first for auto day-selection.")
        print(f"         Continuing with key/manual dates only.\n")

    # Backfill energy values for key/manual dates too, for the printout
    for r in rows:
        if np.isnan(r['energy_db_1_5hz_night']) and r['date'] in energy_by_day:
            r['energy_db_1_5hz_night'] = round(float(energy_by_day[r['date']]), 1)

    rows.sort(key=lambda r: r['date'])
    return rows


def load_night_window(data_root, date_str, network, station, location, channel,
                      window_start_utc, window_duration_h):
    """
    Load window_duration_h hours starting at window_start_utc (UTC) on date_str,
    merging as many hourly MiniSEED files as needed across calendar-day boundaries.
    Returns a cleaned (demean+detrend), UNFILTERED obspy.Trace, or None if no data.
    """
    t_start = UTCDateTime(date_str) + window_start_utc * 3600
    t_end   = t_start + window_duration_h * 3600

    st  = Stream()
    day = UTCDateTime(t_start.strftime('%Y-%m-%d'))
    while day < t_end:
        month_str = day.strftime('%Y%m')
        date_s    = day.strftime('%Y%m%d')
        pattern   = os.path.join(
            data_root, month_str,
            f"{network}.{station}.{location}.{channel}_{date_s}_*.miniseed"
        )
        for f in sorted(glob.glob(pattern)):
            try:
                st += read(f)
            except Exception as e:
                print(f"    [WARN] Skipping {os.path.basename(f)}: {e}")
        day += 86400

    if len(st) == 0:
        return None
    try:
        st.merge(fill_value=0)
    except Exception:
        st.merge(method=0, fill_value=0)

    st.trim(t_start, t_end)
    if len(st) == 0 or st[0].stats.npts == 0:
        return None

    tr = st[0]
    tr.detrend('demean')
    tr.detrend('linear')
    return tr


def compute_spectrogram(data, fs, window_s, overlap_frac):
    """
    Spectrogram computed FROM SCRATCH with the given window length -> the
    parameter that actually controls time resolution (unlike cropping).
    Returns (t_sec, f_hz, Sxx_db).
    """
    nperseg  = max(8, int(window_s * fs))
    noverlap = min(nperseg - 1, int(nperseg * overlap_frac))
    f_hz, t_sec, Sxx = sp_spectrogram(
        data, fs=fs, window=hann_window(nperseg),
        nperseg=nperseg, noverlap=noverlap,
        scaling='density', mode='psd'
    )
    Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-30))
    return t_sec, f_hz, Sxx_db


def compute_welch_spectrogram(data, fs, bin_s, subseg_s, subseg_overlap, hop_s):
    """
    "Spectrogram" built as a sequence of Welch PSD estimates (scipy.signal.welch),
    one per sliding bin, instead of scipy.signal.spectrogram's one-raw-periodogram-
    per-column approach. A raw periodogram is NOT a low-variance estimator —
    its per-bin variance stays ~constant no matter how short the window is
    (only Δf improves), which is why a plain short-window spectrogram looks
    speckled/noisy up close. Averaging several overlapping sub-segments per
    column (exactly what scipy.signal.welch does, already used for the PSD
    figures in fios_04/06) removes that per-pixel noise.

    Parameters
    ----------
    bin_s            : length of data (s) that feeds ONE output column
    subseg_s         : length of each Welch sub-segment inside a bin (s) -> Δf = 1/subseg_s
    subseg_overlap   : overlap fraction between sub-segments inside a bin
    hop_s            : how far the bin slides between columns (s) -> Δt of the output

    Returns (t_sec, f_hz, Sxx_db) — t_sec = bin CENTER time, seconds from data start.
    """
    nbin = max(8, int(bin_s * fs))
    nseg = max(8, min(int(subseg_s * fs), nbin))
    noverlap_seg = min(nseg - 1, int(nseg * subseg_overlap))
    nhop = max(1, int(hop_s * fs))

    n = len(data)
    starts = np.arange(0, max(1, n - nbin + 1), nhop)
    if len(starts) == 0:
        starts = np.array([0])

    f_ref = None
    cols  = []
    used_starts = []
    for s0 in starts:
        seg = data[s0:s0 + nbin]
        if len(seg) < nseg:
            continue
        freqs, psd = welch(seg, fs=fs, nperseg=nseg, noverlap=noverlap_seg,
                           window='hann', scaling='density')
        if f_ref is None:
            f_ref = freqs
        cols.append(psd)
        used_starts.append(s0)

    if not cols:
        return np.array([]), np.array([]), np.zeros((0, 0))

    Sxx    = np.array(cols).T   # shape (n_freq, n_time)
    Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-30))
    t_sec  = (np.array(used_starts) + nbin / 2.0) / fs
    return t_sec, f_ref, Sxx_db


def find_best_subwindow(t_abs_spec, f_plot, S_plot, band, duration_min):
    """
    Slide a duration_min-long window across an already-computed spectrogram
    grid and return the UTCDateTime start of the window that maximises mean
    energy (dB) in [band[0], band[1]] Hz. Generalises fios_05's zoom-picker
    so it can be reused at every cascade level, on any band/duration.
    """
    band_mask       = (f_plot >= band[0]) & (f_plot <= band[1])
    energy_per_time = np.nanmean(S_plot[band_mask, :], axis=0)

    if len(t_abs_spec) < 3:
        return UTCDateTime(t_abs_spec[0])

    dt_spec    = (t_abs_spec[1] - t_abs_spec[0]).total_seconds()
    n_win_col  = max(1, int(duration_min * 60 / dt_spec))
    if n_win_col >= len(energy_per_time):
        return UTCDateTime(t_abs_spec[0])

    kernel   = np.ones(n_win_col) / n_win_col
    smoothed = np.convolve(energy_per_time, kernel, mode='valid')
    best_col = int(np.nanargmax(smoothed))
    return UTCDateTime(t_abs_spec[best_col])


def sliding_kurtosis(data, fs, win_s, step_s):
    """
    Kurtosis of the raw signal in successive sliding windows -> a rough
    "impulsiveness" curve over time.
      ~3 (Gaussian) and flat  -> leans towards continuous tremor
      spiky / >> 3            -> leans towards impulsive/transient energy
    Returns (t_centers_s, kurt_values) — t_centers_s in seconds from data start.
    """
    nwin  = max(8, int(win_s * fs))
    nstep = max(1, int(step_s * fs))
    n     = len(data)
    starts = np.arange(0, max(1, n - nwin), nstep)
    if len(starts) == 0:
        return np.array([]), np.array([])
    kurt = np.empty(len(starts))
    for i, s0 in enumerate(starts):
        kurt[i] = scipy_kurtosis(data[s0:s0 + nwin], fisher=False)
    t_centers_s = (starts + nwin / 2.0) / fs
    return t_centers_s, kurt


def run_fine_picker(tr_pick, sta_s, lta_s, thr_on, thr_off, min_dur_s, min_gap_s=0.0):
    """
    Short-window STA/LTA tuned for brief, closely-spaced impulses (not the
    STA=1s/LTA=80s config used for the daily counts in fios_01).

    min_gap_s : float
        If the gap between one event's end and the next event's start is
        below this, the two are merged into a single event (extends the
        first event's t_off to the second one's t_off). Without this, a
        single burst with a ringing/oscillating envelope gets chopped into
        many near-duplicate "events" by classic_sta_lta.

    Returns a list of dicts: {t_on, t_off, duration_s, max_cft} (UTCDateTime).
    """
    fs = tr_pick.stats.sampling_rate
    cft, on_off = run_sta_lta(tr_pick, sta_s, lta_s, thr_on, thr_off)
    t0 = tr_pick.stats.starttime

    raw_events = []
    for i_on, i_off in on_off:
        t_on  = t0 + i_on / fs
        t_off = t0 + i_off / fs
        dur   = t_off - t_on
        if dur < min_dur_s:
            continue
        max_cft = float(np.max(cft[i_on:i_off + 1])) if i_off > i_on else float(thr_on)
        raw_events.append({'t_on': t_on, 't_off': t_off, 'max_cft': max_cft})

    if not raw_events:
        return []

    # ---- merge events separated by less than min_gap_s ------------------------
    merged = [raw_events[0]]
    for ev in raw_events[1:]:
        prev = merged[-1]
        if (ev['t_on'] - prev['t_off']) < min_gap_s:
            prev['t_off']   = max(prev['t_off'], ev['t_off'])
            prev['max_cft'] = max(prev['max_cft'], ev['max_cft'])
        else:
            merged.append(ev)

    events = [{'t_on': ev['t_on'], 't_off': ev['t_off'],
              'duration_s': round(ev['t_off'] - ev['t_on'], 3),
              'max_cft': round(ev['max_cft'], 3)} for ev in merged]
    return events



# =============================================================================
# SECTION 4 — DAY SELECTION
# =============================================================================

print(f"\n{'='*70}")
print(f"  FIOS Close-up microseismicity observation")
print(f"{'='*70}\n")

selection = select_target_dates(PSD_HOURLY_CSV, KEY_EVENTS, N_AUTO_DAYS, MANUAL_DATES)
TARGET_DATES = [r['date'] for r in selection]

print(f"  {len(TARGET_DATES)} night(s) selected:")
for r in selection:
    e = f"{r['energy_db_1_5hz_night']:.1f} dB" if not np.isnan(r['energy_db_1_5hz_night']) else "n/a"
    print(f"    {r['date']}  —  {r['reason']:<32s}  night 1-5Hz energy = {e}")

df_sel = pd.DataFrame(selection)
sel_csv = os.path.join(OUTPUT_DIR, "day_selection_summary.csv")
df_sel.to_csv(sel_csv, index=False)
print(f"\n  [SAVED] {sel_csv}\n")



# =============================================================================
# SECTION 5 — MAIN LOOP (one cascade per selected night)
# =============================================================================

_w_end_utc    = (WINDOW_START_UTC + WINDOW_DURATION_H) % 24
_window_label = (f"UTC {WINDOW_START_UTC:02d}:00 -> {_w_end_utc:02d}:00"
                + (" (+1 day)" if WINDOW_DURATION_H + WINDOW_START_UTC >= 24 else ""))

for TARGET_DATE in TARGET_DATES:

    print(f"\n{'='*70}")
    print(f"  Processing : {TARGET_DATE}  |  {_window_label}")
    print(f"{'='*70}")

    date_dir = os.path.join(OUTPUT_DIR, TARGET_DATE)
    os.makedirs(date_dir, exist_ok=True)

    try:
        # ---- Load -----------------------------------------------------------------
        tr_raw = load_night_window(
            DATA_ROOT, TARGET_DATE, NETWORK, STATION, LOCATION, CHANNEL,
            WINDOW_START_UTC, WINDOW_DURATION_H
        )
        if tr_raw is None:
            print(f"  [SKIP] No data found for {TARGET_DATE}.")
            continue

        fs = tr_raw.stats.sampling_rate
        print(f"  Loaded : {tr_raw.stats.starttime} -> {tr_raw.stats.endtime}  "
              f"({tr_raw.stats.npts/fs/3600:.2f} h  @ {fs:.0f} Hz)")

        fmax_disp_safe = min(FILT_FMAX, 0.45 * fs)
        fmax_pick_safe = min(PICK_FREQMAX, 0.45 * fs)
        fmax_wave_safe = min(WAVE_FMAX, 0.45 * fs)

        tr_filt = tr_raw.copy()
        tr_filt.filter('bandpass', freqmin=FILT_FMIN, freqmax=fmax_disp_safe,
                       corners=4, zerophase=True)

        tr_pick = tr_raw.copy()
        tr_pick.filter('bandpass', freqmin=PICK_FREQMIN, freqmax=fmax_pick_safe,
                       corners=4, zerophase=True)

        # Separate, narrower band for waveform panels (Level 1 + interactive
        # figure) so they aren't dominated by broadband/high-freq content
        # you can already see clearly in the spectrogram.
        tr_wave = tr_raw.copy()
        tr_wave.filter('bandpass', freqmin=WAVE_FMIN, freqmax=fmax_wave_safe,
                       corners=4, zerophase=True)

        # ---- Fine-scale picker over the WHOLE night --------------------------------
        print(f"  Running fine picker (STA={PICK_STA_S}s LTA={PICK_LTA_S}s "
              f"band={PICK_FREQMIN}-{fmax_pick_safe:.0f}Hz, min_gap={PICK_MIN_GAP_S}s) ...")
        events = run_fine_picker(tr_pick, PICK_STA_S, PICK_LTA_S,
                                 PICK_THR_ON, PICK_THR_OFF, PICK_MIN_DUR_S,
                                 min_gap_s=PICK_MIN_GAP_S)
        print(f"  Picked {len(events)} candidate event(s) over the night.")
        if len(events) > 500:
            print(f"  [WARN] {len(events)} events in one night is a lot — the picker is "
                  f"probably still over-triggering. Consider raising PICK_THR_ON / "
                  f"PICK_LTA_S before trusting fig_04 or the events CSV.")

        # ---- Feature extraction per picked event (optional, off by default) -------
        event_rows = []
        if ENABLE_FEATURE_EXTRACTION:
            t_start_tr = tr_filt.stats.starttime
            t_end_tr   = tr_filt.stats.endtime
            for ev in events:
                t0 = max(ev['t_on']  - FEATURE_PAD_S, t_start_tr)
                t1 = min(ev['t_off'] + FEATURE_PAD_S, t_end_tr)
                seg = tr_filt.slice(t0, t1)
                if seg.stats.npts < 20:
                    continue
                feats = extract_features(seg.data.astype(float), fs)
                row = {'t_on': str(ev['t_on']), 't_off': str(ev['t_off']),
                      'duration_s': ev['duration_s'], 'max_cft': ev['max_cft']}
                row.update(dict(zip(FEATURE_NAMES, feats)))
                event_rows.append(row)

            if event_rows:
                df_ev = pd.DataFrame(event_rows)
                ev_csv = os.path.join(date_dir, f"events_{TARGET_DATE}.csv")
                df_ev.to_csv(ev_csv, index=False)
                print(f"  [SAVED] {os.path.basename(ev_csv)}  ({len(event_rows)} events x "
                      f"{len(FEATURE_NAMES)} features)")
        else:
            # still save a lightweight events CSV (time/duration/max_cft only,
            # no 99-feature columns) so the picks are inspectable either way
            if events:
                df_ev = pd.DataFrame([
                    {'t_on': str(ev['t_on']), 't_off': str(ev['t_off']),
                    'duration_s': ev['duration_s'], 'max_cft': ev['max_cft']}
                    for ev in events
                ])
                ev_csv = os.path.join(date_dir, f"events_{TARGET_DATE}.csv")
                df_ev.to_csv(ev_csv, index=False)
                print(f"  [SAVED] {os.path.basename(ev_csv)}  ({len(events)} events, "
                      f"no feature columns — ENABLE_FEATURE_EXTRACTION is False)")

        # =========================================================================
        # LEVEL 1 — full night (context) + sliding kurtosis
        # =========================================================================
        print("  Level 1 — full night spectrogram ...")
        t_sec1, f_hz1, Sxx1_db = compute_spectrogram(tr_filt.data, fs, LEVEL1_WINDOW_S, SPEC_OVERLAP)
        freq_mask1 = (f_hz1 >= FREQ_MIN_PLOT) & (f_hz1 <= FREQ_MAX_PLOT)
        f_plot1    = f_hz1[freq_mask1]
        S_plot1    = Sxx1_db[freq_mask1, :]
        t_abs1     = np.array([(tr_filt.stats.starttime + float(t)).datetime for t in t_sec1])

        # Gap masking (zero-variance windows -> NaN, same logic as fios_05)
        nperseg1 = max(8, int(LEVEL1_WINDOW_S * fs))
        i_ctrs1  = np.round(t_sec1 * fs).astype(int)
        gap_cols1 = np.array([
            np.var(tr_raw.data[max(0, i - nperseg1 // 2):min(len(tr_raw.data), i + nperseg1 // 2)]) < 1.0
            for i in i_ctrs1
        ])
        S_plot1[:, gap_cols1] = np.nan

        kurt_t_s, kurt_vals = sliding_kurtosis(tr_filt.data, fs, KURT_WIN_S, KURT_STEP_S)
        kurt_t_abs = np.array([(tr_filt.stats.starttime + float(t)).datetime for t in kurt_t_s])

        # Downsampled waveform for display
        step_ds = max(1, tr_filt.stats.npts // 10000)
        t_wave1 = np.array([(tr_filt.stats.starttime + i / fs).datetime
                            for i in range(0, tr_filt.stats.npts, step_ds)])
        d_wave1 = tr_filt.data[::step_ds]

        fig, (ax_w, ax_s, ax_k) = plt.subplots(
            3, 1, figsize=(18, 11),
            gridspec_kw={'height_ratios': [1, 3, 1]}, sharex=True
        )
        ax_w.plot(t_wave1, d_wave1, color='black', lw=0.4, rasterized=True)
        ax_w.set_ylabel('Amplitude\n(counts)', fontsize=9)
        ax_w.set_title(
            f'FIO1 — {TARGET_DATE}  ({_window_label})  |  Bandpass {FILT_FMIN}-{fmax_disp_safe:.0f} Hz\n'
            f'Level 1 (context): {LEVEL1_WINDOW_S:.0f}-s windows -> '
            f'Δf={1/LEVEL1_WINDOW_S:.2f} Hz  |  {len(events)} events picked by fine STA/LTA '
            f'(STA={PICK_STA_S}s LTA={PICK_LTA_S}s, {PICK_FREQMIN}-{fmax_pick_safe:.0f} Hz)',
            fontsize=10
        )
        ax_w.grid(axis='y', lw=0.3, alpha=0.4)
        # Rug of picked event onsets along the top of the waveform panel
        if events:
            y_rug = ax_w.get_ylim()[1] * 0.9
            ev_times = [ev['t_on'].datetime for ev in events]
            ax_w.plot(ev_times, [y_rug] * len(ev_times), '|', color='crimson',
                      markersize=8, alpha=0.6, label=f'{len(events)} picks (fine STA/LTA)')
            ax_w.legend(fontsize=8, loc='upper right')

        cmap_spec = plt.cm.inferno.copy()
        cmap_spec.set_bad(color='black')
        im1 = ax_s.pcolormesh(t_abs1, f_plot1, S_plot1, cmap=cmap_spec,
                              vmin=LEVEL1_VMIN_DB, vmax=LEVEL1_VMAX_DB,
                              shading='auto', rasterized=True)
        cbar1 = plt.colorbar(im1, ax=ax_s, pad=0.01, fraction=0.015)
        cbar1.set_label('PSD (dB re counts²/Hz)', fontsize=8)
        ax_s.set_ylabel('Frequency (Hz)', fontsize=9)
        ax_s.set_ylim(FREQ_MIN_PLOT, FREQ_MAX_PLOT)

        if len(kurt_t_abs) > 0:
            ax_k.plot(kurt_t_abs, kurt_vals, color='teal', lw=0.8)
            ax_k.axhline(3.0, color='grey', lw=1.0, ls='--', label='Gaussian (β=3)')
            ax_k.set_ylabel(f'Kurtosis\n({KURT_WIN_S:.0f}s win)', fontsize=8)
            ax_k.legend(fontsize=7, loc='upper right')
            ax_k.grid(axis='y', lw=0.3, alpha=0.4)
        ax_k.set_xlabel(f'Time UTC  ({TARGET_DATE}  {_window_label})', fontsize=9)

        date_fmt = mdates.DateFormatter('%H:%M')
        ax_k.xaxis.set_major_formatter(date_fmt)
        ax_k.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.setp(ax_k.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        fig_path = os.path.join(date_dir, f"fig_01_night_10h_{TARGET_DATE}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [SAVED] {os.path.basename(fig_path)}")

        # ---- Inter-event time histogram --------------------------------------------
        if len(events) >= 3:
            t_ons = sorted(ev['t_on'] for ev in events)
            inter_s = [float(t_ons[i+1] - t_ons[i]) for i in range(len(t_ons) - 1)]
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.hist(inter_s, bins=50, color='steelblue', edgecolor='white')
            ax.set_yscale('log')
            ax.set_xlabel('Inter-event time (s)', fontsize=11)
            ax.set_ylabel('Count (log scale)', fontsize=11)
            ax.set_title(
                f'FIO1 — {TARGET_DATE}  |  Inter-event times, fine picker (n={len(events)})\n'
                f'Clustered at short lags -> leans dense microfissure swarm  |  '
                f'~uniform spread -> leans continuous tremor',
                fontsize=10
            )
            ax.grid(axis='y', lw=0.3, alpha=0.4)
            plt.tight_layout()
            fig_path = os.path.join(date_dir, f"fig_04_interevent_hist_{TARGET_DATE}.png")
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  [SAVED] {os.path.basename(fig_path)}")
        else:
            print("  [SKIP] Fewer than 3 picked events — skipping inter-event histogram.")

        # =========================================================================
        # INTERACTIVE ZOOM — auto-picked INTERACTIVE_WINDOW_MIN window, exported
        # as a Plotly .html: zoom/pan yourself on time or frequency in a browser,
        # instead of relying on a pre-picked "Level 2/3" PNG (see module docstring)
        # =========================================================================
        t_zoom = find_best_subwindow(t_abs1, f_plot1, S_plot1, SUBWINDOW_BAND, INTERACTIVE_WINDOW_MIN)
        t_zoom_end = t_zoom + INTERACTIVE_WINDOW_MIN * 60
        print(f"  Interactive zoom — auto {INTERACTIVE_WINDOW_MIN}-min window : {t_zoom}  ->  {t_zoom_end}")

        tr_int      = tr_filt.slice(t_zoom, t_zoom_end)   # spectrogram band
        tr_int_wave = tr_wave.slice(t_zoom, t_zoom_end)   # separate waveform band

        picks_in_window = []

        if tr_int.stats.npts < 20:
            print("  [WARN] Interactive zoom window too short — skipping fig_interactive.")
        else:
            t_sec_i, f_hz_i, Sxx_i_db = compute_welch_spectrogram(
                tr_int.data, fs, INTERACTIVE_BIN_S, INTERACTIVE_SUBSEG_S,
                INTERACTIVE_SUBSEG_OVERLAP, INTERACTIVE_HOP_S
            )
            if Sxx_i_db.size == 0:
                print("  [WARN] Interactive Welch spectrogram produced no columns — skipping fig_interactive.")
            else:
                freq_mask_i = (f_hz_i >= FREQ_MIN_PLOT) & (f_hz_i <= FREQ_MAX_PLOT)
                f_plot_i = f_hz_i[freq_mask_i]
                S_plot_i = Sxx_i_db[freq_mask_i, :]
                t_abs_i  = [(tr_int.stats.starttime + float(t)).datetime for t in t_sec_i]

                n_cells = S_plot_i.shape[0] * S_plot_i.shape[1]
                print(f"  Interactive grid: {S_plot_i.shape[1]} time cols x {S_plot_i.shape[0]} "
                      f"freq rows = {n_cells:,} cells")

                picks_in_window = [ev for ev in events
                                   if tr_int.stats.starttime <= ev['t_on'] <= tr_int.stats.endtime]

                step_ds_i = max(1, tr_int_wave.stats.npts // 20000)
                t_wave_i = [(tr_int_wave.stats.starttime + i / fs).datetime
                           for i in range(0, tr_int_wave.stats.npts, step_ds_i)]
                d_wave_i = tr_int_wave.data[::step_ds_i]

                fig_int = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.25, 0.75], vertical_spacing=0.03,
                    subplot_titles=("Waveform", "Spectrogram")
                )
                fig_int.add_trace(
                    go.Scatter(x=t_wave_i, y=d_wave_i, mode='lines',
                              line=dict(color='black', width=0.6),
                              name=f'Waveform ({WAVE_FMIN:.0f}-{fmax_wave_safe:.0f} Hz)',
                              hovertemplate='%{x}<br>%{y:.0f} counts<extra></extra>'),
                    row=1, col=1
                )
                if picks_in_window:
                    y_rug = float(np.max(np.abs(d_wave_i))) * 1.05 if len(d_wave_i) else 1.0
                    fig_int.add_trace(
                        go.Scatter(
                            x=[ev['t_on'].datetime for ev in picks_in_window],
                            y=[y_rug] * len(picks_in_window),
                            mode='markers', marker=dict(color='crimson', symbol='line-ns', size=10,
                                                        line=dict(width=1.5, color='crimson')),
                            name=f'{len(picks_in_window)} fine-picker picks',
                            hovertemplate='pick: %{x}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                fig_int.add_trace(
                    go.Heatmap(
                        z=S_plot_i, x=t_abs_i, y=f_plot_i,
                        colorscale='Inferno',
                        colorbar=dict(title='PSD (dB re counts²/Hz)', len=0.75, y=0.35),
                        hovertemplate='Time: %{x}<br>Freq: %{y:.1f} Hz<br>PSD: %{z:.1f} dB<extra></extra>',
                    ),
                    row=2, col=1
                )
                fig_int.update_layout(
                    title=(
                        f'FIO1 — {TARGET_DATE}  |  {tr_int.stats.starttime.strftime("%H:%M:%S")} -> '
                        f'{tr_int.stats.endtime.strftime("%H:%M:%S")} UTC (auto-picked on max '
                        f'{SUBWINDOW_BAND[0]:.0f}-{SUBWINDOW_BAND[1]:.0f} Hz energy)  |  '
                        f'Spectrogram band {FILT_FMIN:.0f}-{fmax_disp_safe:.0f} Hz, '
                        f'waveform band {WAVE_FMIN:.0f}-{fmax_wave_safe:.0f} Hz<br>'
                        f'<sub>Welch spectrogram: Δf={1/INTERACTIVE_SUBSEG_S:.1f} Hz, '
                        f'Δt={INTERACTIVE_HOP_S*1000:.0f} ms (fixed grid — zooming magnifies these '
                        f'pixels, it does not compute new ones)  |  '
                        f'{len(picks_in_window)} fine-picker event(s) in this window (rug above waveform)  |  '
                        f'scroll/drag to zoom, double-click to reset</sub>'
                    ),
                    height=750,
                    hovermode='x unified',
                    xaxis2=dict(rangeslider=dict(visible=True), type='date', title='Time UTC'),
                    yaxis=dict(title='Amplitude (counts)'),
                    yaxis2=dict(title='Frequency (Hz)', range=[FREQ_MIN_PLOT, FREQ_MAX_PLOT]),
                )
                fig_path = os.path.join(date_dir, f"fig_interactive_{TARGET_DATE}.html")
                fig_int.write_html(fig_path, include_plotlyjs=PLOTLYJS_MODE)
                print(f"  [SAVED] {os.path.basename(fig_path)}")

        # =========================================================================
        # LOUDEST PICK — few-second raw waveform zoom (PICK band, not WAVE band)
        # on the single loudest fine-picker event in the interactive window, to
        # inspect pulse SHAPE without the general-purpose waveform band flattening it
        # =========================================================================
        if picks_in_window:
            loudest = max(picks_in_window, key=lambda ev: ev['max_cft'])
            t_fine0 = loudest['t_on']  - FINE_TRACE_MARGIN_S
            t_fine1 = loudest['t_off'] + FINE_TRACE_MARGIN_S
            tr_fine = tr_pick.slice(t_fine0, t_fine1)
            t_fine_ax = np.array([(tr_fine.stats.starttime + i / fs).datetime
                                  for i in range(tr_fine.stats.npts)])

            fig, ax_zw = plt.subplots(figsize=(10, 4))
            ax_zw.plot(t_fine_ax, tr_fine.data, color='black', lw=1.0)
            ax_zw.axvspan(loudest['t_on'].datetime, loudest['t_off'].datetime,
                         color='crimson', alpha=0.15)
            ax_zw.set_title(
                f'FIO1 — {TARGET_DATE}  |  Raw waveform (Bandpass {PICK_FREQMIN:.0f}-{fmax_pick_safe:.0f} Hz '
                f'— picker band), loudest pick in the interactive window '
                f'(max_cft={loudest["max_cft"]:.1f}, dur={loudest["duration_s"]:.2f}s)',
                fontsize=10
            )
            ax_zw.set_ylabel('Amplitude (counts)', fontsize=9)
            ax_zw.set_xlabel('Time UTC', fontsize=9)
            ax_zw.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S.%f'))
            plt.setp(ax_zw.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
            ax_zw.grid(axis='y', lw=0.3, alpha=0.4)
            plt.tight_layout()
            fig_path = os.path.join(date_dir, f"fig_03_loudest_pick_{TARGET_DATE}.png")
            plt.savefig(fig_path, dpi=160, bbox_inches='tight')
            plt.close(fig)
            print(f"  [SAVED] {os.path.basename(fig_path)}")
        else:
            print("  [SKIP] No fine-picker event in the interactive window — skipping fig_03_loudest_pick.")


    except Exception as e:
        print(f"  [ERROR] {TARGET_DATE} failed with: {e!r} -- skipping to next night.")
        plt.close('all')
        continue



# =============================================================================
# END
# =============================================================================

print(f"\n[DONE]  All outputs saved to: {OUTPUT_DIR}")
print(f"        Nights processed  : {len(TARGET_DATES)}")
print(f"        See day_selection_summary.csv for why each date was picked.")
