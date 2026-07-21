"""
FIOS LANDSLIDE — CLOSE-UP MICROSEISMICITY OBSERVATION
======================================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : July 2026

What this script does
----------------------
1. DAY SELECTION — decide which nights are worth a close look:
     - the 3 known destabilisations (13 Apr, 25 May, 23 Jun)
     - + an automatic top-N ranking of the most energetic nights, using the night-only 1-5 Hz band energy already computed in psd_hourly.csv (fios_04.py)
   -> written to day_selection_summary.csv so the choice is transparent and reproducible
      add dates to MANUAL_DATES below to force extra nights in

2. MULTI-SCALE ZOOM CASCADE — for each selected night, three spectrograms are computed FROM SCRATCH (not cropped) with shrinking window length:
     Level 1 : full night   (10 h),  10-s windows -> fine Δf,  coarse Δt (context)
     Level 2 : auto 10-min window,    2-s windows -> medium Δt
     Level 3 : auto 1-min window,     1-s windows -> Δt ~0.1s, still a clean, low-noise spectrogram
   ALL THREE waveform+spectrogram panels use the SAME bandpass (FILT_FMIN-FILT_FMAX,
   1-60 Hz by default) — that band is now stated explicitly in every figure title,
   not just fig_01, to avoid ambiguity. Change FILT_FMIN/FILT_FMAX in the config to
   look at a narrower band (e.g. 1-20 Hz) instead; everything downstream follows.
   Level 2/3 windows are auto-picked as the sub-window that maximises mean 2-10 Hz energy within the level above (same logic as fios_05, generalised)
   Level 3 also plots a raw-waveform zoom (a few seconds) around the single most energetic pick, to inspect pulse SHAPE directly, as asked

3. FINE-SCALE EVENT PICKER — a short-window STA/LTA (STA~0.08 s / LTA~4 s, band
   10-60 Hz, picks closer than PICK_MIN_GAP_S merged) run on the whole night,
   to try to resolve INDIVIDUAL events inside what looks like a continuous block.
   This is a first tuning guess, not validated — check the printed event count
   and fig_01's rug plot after each run (see PICK_* comments in the config).
   Picked events are:
     - overlaid on the Level 1 and Level 3 figures
     - saved to events_<date>.csv (time, duration, max_cft)
     - optionally (ENABLE_FEATURE_EXTRACTION=True) also run through
       features.extract_features() — the 99-feature Maggi/Hibert/Provost set
       used elsewhere in this project for event-type classification; OFF by
       default here since it isn't needed to answer the tremor-vs-swarm
       question on its own
     - used to build two diagnostic plots per night:
         fig_04_interevent_hist : histogram of the time gaps between
           consecutive picks (clustered at short lags -> leans swarm;
           spread out -> leans continuous tremor) — only meaningful once the
           picker isn't over-triggering
         (in fig_01) a sliding-window kurtosis-of-envelope curve under the
           spectrogram (flat, ~3 -> leans tremor; spiky -> leans impulsive)

Outputs (all saved to OUTPUT_DIR/<date>/)
------------------------------------------
  fig_01_night_10h_<date>.png        — full night: waveform + spectrogram + sliding kurtosis
  fig_02_zoom_10min_<date>.png       — auto 10-min window, medium time resolution
  fig_03_zoom_1min_<date>.png        — auto 1-min window, fine time resolution + picks + a few-second raw waveform zoom on the top pick
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
N_AUTO_DAYS   = 5     # how many extra nights to add, ranked by night-only 1-5 Hz energy
MANUAL_DATES  = []    # add dates here manually, e.g. ["2026-06-02"], to force them in

# ---- Night time window (UTC) -------------------------------------------------
WINDOW_START_UTC  = 18
WINDOW_DURATION_H = 10

# ---- Display bandpass (waveform + spectrogram) -------------------------------
FILT_FMIN = 1.0
FILT_FMAX = 60.0

# ---- Spectrogram window length per cascade level -----------------------------
# Shorter window -> coarser Δf, but finer Δt.
# Level 3 was tried at 0.25 s first (Δt ~25 ms) but each pixel is then a single
# ~62-sample FFT with no averaging -> very high per-pixel variance -> the
# speckled/"mauvaise qualité" look. 1.0 s keeps Δt ~0.1 s at 90% overlap
# (still ~10x finer than fios_05's 10-s zoom, and shorter than a typical
# picked event duration here, ~0.5-2 s per fios_01's stats) while the spectral
# estimate at each pixel is far more stable.
LEVEL1_WINDOW_S = 10.0    # full night   -> Δf = 0.10 Hz
LEVEL2_WINDOW_S = 2.0     # 10-min zoom  -> Δf = 0.50 Hz
LEVEL3_WINDOW_S = 1.0     # 1-min zoom   -> Δf = 1.0  Hz,  Δt ~ 0.1 s steps
SPEC_OVERLAP    = 0.90

LEVEL2_DURATION_MIN = 10
LEVEL3_DURATION_MIN = 1
SUBWINDOW_BAND       = (2.0, 10.0)   # band used to auto-pick the loudest sub-window

FREQ_MIN_PLOT = 1.0
FREQ_MAX_PLOT = 60.0

# Level-1 colour scale fixed across nights for cross-night comparability
# (same convention as fios_05's VSCALE_MODE="fixed")
LEVEL1_VMIN_DB = -15
LEVEL1_VMAX_DB =  55

# Raw waveform zoom (seconds each side) around the single loudest pick in Level 3
FINE_TRACE_MARGIN_S = 3.0

# ---- Fine-scale event picker --------------------------------------------------
# First attempt (STA=0.05s LTA=2s ON=4.0 band 5-60Hz) fired 7496 times over a
# single 10-h night, i.e. one "event" every ~5 s on average, with the rug plot
# in fig_01 essentially solid red -> the picker was just tracking background
# fluctuations, not distinct events. Retuned below:
#  - PICK_FREQMIN raised to 10 Hz to get away from the persistent, highly
#    variable <10 Hz band (that's what was destabilising the 2-s LTA baseline)
#  - LTA lengthened and thresholds raised for a much more stable baseline
#  - PICK_MIN_GAP_S: picks separated by less than this are merged into one,
#    so a single ringing burst isn't chopped into a dozen "events"
# These are still a first guess, not validated — check events_<date>.csv and
# the fig_01 rug plot after a run and retune if the count still looks off
# (expect tens, not thousands, per night outside a destabilisation burst).
PICK_FREQMIN   = 10.0
PICK_FREQMAX   = 60.0
PICK_STA_S     = 0.08
PICK_LTA_S     = 4.0
PICK_THR_ON    = 6.0
PICK_THR_OFF   = 2.0
PICK_MIN_DUR_S = 0.1
PICK_MIN_GAP_S = 0.3

FEATURE_PAD_S = 0.2   # padding added on each side of a pick before feature extraction

# The 99-feature Maggi/Hibert/Provost set (seismic_params.py / features.py) is
# used elsewhere in this project for automatic event-type classification
# (Provost et al. 2017, Hibert et al. 2017, Pirot et al. 2024). It is NOT
# needed to answer "is this tremor or a microfissure swarm" by itself — kept
# here only as an optional extra in case you want per-event features for a
# classifier later. Off by default: turn on if/when you actually need it.
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
from scipy.signal          import spectrogram as sp_spectrogram
from scipy.signal.windows  import hann as hann_window
from scipy.stats           import kurtosis as scipy_kurtosis

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

        tr_filt = tr_raw.copy()
        tr_filt.filter('bandpass', freqmin=FILT_FMIN, freqmax=fmax_disp_safe,
                       corners=4, zerophase=True)

        tr_pick = tr_raw.copy()
        tr_pick.filter('bandpass', freqmin=PICK_FREQMIN, freqmax=fmax_pick_safe,
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
        # LEVEL 2 — auto 10-min window, medium time resolution (RECOMPUTED, not cropped)
        # =========================================================================
        t_zoom10 = find_best_subwindow(t_abs1, f_plot1, S_plot1, SUBWINDOW_BAND, LEVEL2_DURATION_MIN)
        t_zoom10_end = t_zoom10 + LEVEL2_DURATION_MIN * 60
        print(f"  Level 2 — auto 10-min window : {t_zoom10}  ->  {t_zoom10_end}")

        tr_l2 = tr_filt.slice(t_zoom10, t_zoom10_end)
        if tr_l2.stats.npts < 20:
            print("  [WARN] Level 2 window too short — skipping fig_02.")
            t_abs2, f_plot2, S_plot2 = None, None, None
        else:
            t_sec2, f_hz2, Sxx2_db = compute_spectrogram(tr_l2.data, fs, LEVEL2_WINDOW_S, SPEC_OVERLAP)
            freq_mask2 = (f_hz2 >= FREQ_MIN_PLOT) & (f_hz2 <= FREQ_MAX_PLOT)
            f_plot2 = f_hz2[freq_mask2]
            S_plot2 = Sxx2_db[freq_mask2, :]
            t_abs2  = np.array([(tr_l2.stats.starttime + float(t)).datetime for t in t_sec2])

            vmin2 = float(np.nanpercentile(S_plot2, 5))
            vmax2 = float(np.nanpercentile(S_plot2, 99))

            t_wave2 = np.array([(tr_l2.stats.starttime + i / fs).datetime for i in range(tr_l2.stats.npts)])

            fig, (ax_w, ax_s) = plt.subplots(
                2, 1, figsize=(14, 7), gridspec_kw={'height_ratios': [1, 2.5]}, sharex=True
            )
            ax_w.plot(t_wave2, tr_l2.data, color='black', lw=0.6, rasterized=True)
            ax_w.set_ylabel('Amplitude (counts)', fontsize=9)
            ax_w.set_title(
                f'FIO1 — {LEVEL2_DURATION_MIN}-min zoom  |  '
                f'{t_zoom10.datetime.strftime("%Y-%m-%d %H:%M")} -> '
                f'{t_zoom10_end.datetime.strftime("%H:%M")} UTC  |  '
                f'Bandpass {FILT_FMIN}-{fmax_disp_safe:.0f} Hz\n'
                f'RECOMPUTED with {LEVEL2_WINDOW_S:.0f}-s windows -> Δf={1/LEVEL2_WINDOW_S:.2f} Hz  '
                f'(auto-picked on max {SUBWINDOW_BAND[0]:.0f}-{SUBWINDOW_BAND[1]:.0f} Hz energy)',
                fontsize=10
            )
            ax_w.grid(axis='y', lw=0.3, alpha=0.4)
            im2 = ax_s.pcolormesh(t_abs2, f_plot2, S_plot2, cmap=cmap_spec,
                                  vmin=vmin2, vmax=vmax2, shading='auto', rasterized=True)
            cbar2 = plt.colorbar(im2, ax=ax_s, pad=0.01, fraction=0.015)
            cbar2.set_label('PSD (dB re counts²/Hz)', fontsize=8)
            ax_s.set_ylabel('Frequency (Hz)', fontsize=9)
            ax_s.set_xlabel('Time UTC', fontsize=9)
            ax_s.set_ylim(FREQ_MIN_PLOT, FREQ_MAX_PLOT)
            ax_s.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.setp(ax_s.xaxis.get_majorticklabels(), rotation=45, ha='right')
            plt.tight_layout()
            fig_path = os.path.join(date_dir, f"fig_02_zoom_10min_{TARGET_DATE}.png")
            plt.savefig(fig_path, dpi=160, bbox_inches='tight')
            plt.close(fig)
            print(f"  [SAVED] {os.path.basename(fig_path)}")

        # =========================================================================
        # LEVEL 3 — auto 1-min window, fine time resolution (RECOMPUTED) + picks
        # =========================================================================
        if S_plot2 is not None:
            t_zoom1 = find_best_subwindow(t_abs2, f_plot2, S_plot2, SUBWINDOW_BAND, LEVEL3_DURATION_MIN)
        else:
            t_zoom1 = t_zoom10
        t_zoom1_end = t_zoom1 + LEVEL3_DURATION_MIN * 60
        print(f"  Level 3 — auto 1-min window  : {t_zoom1}  ->  {t_zoom1_end}")

        tr_l3 = tr_filt.slice(t_zoom1, t_zoom1_end)
        if tr_l3.stats.npts < 20:
            print("  [WARN] Level 3 window too short — skipping fig_03.")
            continue

        t_sec3, f_hz3, Sxx3_db = compute_spectrogram(tr_l3.data, fs, LEVEL3_WINDOW_S, SPEC_OVERLAP)
        freq_mask3 = (f_hz3 >= FREQ_MIN_PLOT) & (f_hz3 <= FREQ_MAX_PLOT)
        f_plot3 = f_hz3[freq_mask3]
        S_plot3 = Sxx3_db[freq_mask3, :]
        t_abs3  = np.array([(tr_l3.stats.starttime + float(t)).datetime for t in t_sec3])

        vmin3 = float(np.nanpercentile(S_plot3, 5))
        vmax3 = float(np.nanpercentile(S_plot3, 99))

        t_wave3 = np.array([(tr_l3.stats.starttime + i / fs).datetime for i in range(tr_l3.stats.npts)])

        # Picks falling inside this 1-min window
        picks_in_window = [ev for ev in events if tr_l3.stats.starttime <= ev['t_on'] <= tr_l3.stats.endtime]

        fig = plt.figure(figsize=(14, 10))
        gs  = fig.add_gridspec(3, 1, height_ratios=[1, 2.5, 1])
        ax_w  = fig.add_subplot(gs[0])
        ax_s  = fig.add_subplot(gs[1], sharex=ax_w)
        ax_zw = fig.add_subplot(gs[2])   # NOT sharing x — this is a separate, finer time zoom

        ax_w.plot(t_wave3, tr_l3.data, color='black', lw=0.8, rasterized=True)
        ax_w.set_ylabel('Amplitude (counts)', fontsize=9)
        for ev in picks_in_window:
            ax_w.axvline(ev['t_on'].datetime, color='crimson', lw=1.0, alpha=0.7)
        ax_w.set_title(
            f'FIO1 — {LEVEL3_DURATION_MIN}-min zoom  |  '
            f'{t_zoom1.datetime.strftime("%Y-%m-%d %H:%M:%S")} -> '
            f'{t_zoom1_end.datetime.strftime("%H:%M:%S")} UTC  |  '
            f'Bandpass {FILT_FMIN}-{fmax_disp_safe:.0f} Hz\n'
            f'RECOMPUTED with {LEVEL3_WINDOW_S*1000:.0f}-ms windows -> Δf={1/LEVEL3_WINDOW_S:.1f} Hz  |  '
            f'{len(picks_in_window)} fine-picker event(s) in this window (red lines)',
            fontsize=10
        )
        ax_w.grid(axis='y', lw=0.3, alpha=0.4)

        im3 = ax_s.pcolormesh(t_abs3, f_plot3, S_plot3, cmap=cmap_spec,
                              vmin=vmin3, vmax=vmax3, shading='auto', rasterized=True)
        cbar3 = plt.colorbar(im3, ax=ax_s, pad=0.01, fraction=0.015)
        cbar3.set_label('PSD (dB re counts²/Hz)', fontsize=8)
        for ev in picks_in_window:
            ax_s.axvline(ev['t_on'].datetime, color='white', lw=1.0, ls='--', alpha=0.8)
        for f_g in [2, 4, 6, 8, 10, 12, 14]:
            if FREQ_MIN_PLOT < f_g < FREQ_MAX_PLOT:
                ax_s.axhline(f_g, color='white', lw=0.4, ls=':', alpha=0.35)
        ax_s.set_ylabel('Frequency (Hz)', fontsize=9)
        ax_s.set_xlabel('Time UTC', fontsize=9)
        ax_s.set_ylim(FREQ_MIN_PLOT, FREQ_MAX_PLOT)
        ax_s.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax_s.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # ---- Raw waveform zoom (a few seconds) on the single loudest pick ---------
        if picks_in_window:
            loudest = max(picks_in_window, key=lambda ev: ev['max_cft'])
            t_fine0 = loudest['t_on']  - FINE_TRACE_MARGIN_S
            t_fine1 = loudest['t_off'] + FINE_TRACE_MARGIN_S
            tr_fine = tr_filt.slice(t_fine0, t_fine1)
            t_fine_ax = np.array([(tr_fine.stats.starttime + i / fs).datetime
                                  for i in range(tr_fine.stats.npts)])
            ax_zw.plot(t_fine_ax, tr_fine.data, color='black', lw=1.0)
            ax_zw.axvspan(loudest['t_on'].datetime, loudest['t_off'].datetime,
                         color='crimson', alpha=0.15)
            ax_zw.set_title(
                f'Raw waveform (Bandpass {FILT_FMIN}-{fmax_disp_safe:.0f} Hz), loudest pick in '
                f'this window (max_cft={loudest["max_cft"]:.1f}, dur={loudest["duration_s"]:.2f}s)  '
                f'-> look at the pulse shape here',
                fontsize=9
            )
            ax_zw.set_ylabel('Amplitude\n(counts)', fontsize=8)
            ax_zw.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S.%f'))
            plt.setp(ax_zw.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=7)
            ax_zw.grid(axis='y', lw=0.3, alpha=0.4)
        else:
            ax_zw.text(0.5, 0.5, 'No fine-picker event in this 1-min window',
                      ha='center', va='center', transform=ax_zw.transAxes, fontsize=10, color='grey')
            ax_zw.set_xticks([])
            ax_zw.set_yticks([])

        plt.tight_layout()
        fig_path = os.path.join(date_dir, f"fig_03_zoom_1min_{TARGET_DATE}.png")
        plt.savefig(fig_path, dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f"  [SAVED] {os.path.basename(fig_path)}")


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
