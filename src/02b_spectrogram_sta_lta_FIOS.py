"""
FIOS LANDSLIDE — SPECTROGRAM-BASED STA/LTA MICROSEISMICITY DETECTION
=====================================================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : May 2026

Goal
----
- scan continuous 24-hour waveforms from local MiniSEED files (Val d'Anniviers, Switzerland), day by day
- detect microseismic events using the spectrogram-based bidirectional STA/LTA (DetecteurV3, Groult et al. 2026).
- produces daily event count output 

Data source
-----------
Local MiniSEED files:
 {DATA_ROOT}/{YYYYMM}/{NET}.{STA}.{LOC}.{CHAN}_{YYYYMMDD}_{HHMMSS}.miniseed
 -> one file per hour, stored in monthly subfolders (202603, 202604, 202605)

Output 
------
  detections_YYYYMMDD.csv    — one row per detection: starttime, endtime, duration, CFT, SNR
  detections_all_<stamp>.csv — all detections across the full period
  daily_counts.csv           — one row per day: day, n_detections, data_available
  fig_YYYYMMDD.png           — per-day waveform + energy with detected windows
  fig_daily_counts_<stamp>.png — summary bar chart (April 15 marked)
  run.log                    — full console output
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# All parameters to adapt are grouped here
# =============================================================================

# -- Data paths ---------------------------------------------------------------
DATA_ROOT  = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\data\FIOS"
OUTPUT_DIR = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\02b_fios_spectrogram_sta_lta"

# -- Station ------------------------------------------------------------------
NETWORK  = "XT"
STATION  = "FIO1"
LOCATION = "01"        # 01 Z-component; 02 N-component; 03 E-component
CHANNEL  = "DHZ"       # DHZ; DHN; DHE

# -- Time window to process ---------------------------------------------------
T_START = "2026-04-12"   # first day to process
T_END   = "2026-04-18"   # last  day to process (exclusive)

# -- Preprocessing (no response removal — relative detection) -----------------
FREQ_MIN = 10.0    # bandpass low  cutoff -> above the 5–6 Hz resonance peak
FREQ_MAX = 80.0    # bandpass high cutoff

# -- DetecteurV3 parameters ---------------------------------------------------
NSTA    = 2       # STA window (spectrogram time steps)
NLTA    = 80      # LTA window (spectrogram time steps)  [was 20 -> raised to ~30s]
THR_ON  = 8.0     # STA/LTA trigger ON  threshold        [was 5.0 -> raised]
THR_OFF = 3.0     # STA/LTA trigger OFF threshold        [was 2.0 -> raised]

# -- Spectrogram parameters ---------------------------------------------------
NWIN_SEC  = 0.5    # spectrogram window length (s)
NOVER_PCT = 0.25   # overlap fraction between spectrogram windows

# -- Sliding window segmentation of the 24-hour trace ------------------------
WINDOW_SEC  = 10 * 60   # 10-min processing chunk
OVERLAP_SEC =  1 * 60   # 1-min overlap between consecutive chunks

# -- Minimum trace length to attempt detection --------------------------------
MIN_TRACE_SEC = 60.0

# -- Post-detection quality filter --------------------------------------------
MIN_EVENT_DUR_SEC = 0.5   # discard detections shorter than this  [0.5 s: good balance for local microseismicity]

# -- Reference date -----------------------------------------------------------
DESTAB_DATE = "2026-04-15"   # destabilisation onset

# -- Instrument sensitivity ---------------------------------------------------
SENSITIVITY = None    # e.g. 4e8  [counts / (m/s)]

# -- Plotting -----------------------------------------------------------------
PLOT_DAYS     = True   # save a per-day waveform + energy figure
WAVEFORM_YLIM = None   # fixed y-axis amplitude for all daily waveform panels
                       # set to None on first run (auto-scale), then set to the
                       # max amplitude you observe, e.g. WAVEFORM_YLIM = 5000



# =============================================================================
# SECTION 2 — SETUP
# =============================================================================

import os
import sys
import glob
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates  as mdates
import numpy  as np
import pandas as pd

from obspy import UTCDateTime, Stream, read

# Shared utility modules (same src/ folder)
from detecteurV3_fonctions import DetecteurV3
from detection             import merge_window_events
from run_setup             import create_run_dir, setup_logging, set_matplotlib_defaults


# ---------------------------------------------------------------------------
# Local file loader — same function as fios_02a
# ---------------------------------------------------------------------------

def load_day_local(data_root, date_utc, network, station, location, channel):
    """
    Glob all 1-hour MiniSEED files for the given date and return a merged trace
    """
    date_str  = date_utc.strftime('%Y%m%d')
    month_str = date_utc.strftime('%Y%m')
    pattern   = os.path.join(
        data_root, month_str,
        f"{network}.{station}.{location}.{channel}_{date_str}_*.miniseed"
    )
    files = sorted(glob.glob(pattern))

    if not files:
        return None, 0

    st = Stream()
    for f in files:
        try:
            st += read(f)
        except Exception as e:
            print(f"    [WARN] Skipping {os.path.basename(f)}: {e}")

    if len(st) == 0:
        return None, 0

    try:
        st.merge(fill_value=0)
    except Exception:
        st.merge(method=0, fill_value=0)

    return st[0], len(files)


# ---------------------------------------------------------------------------
# Preprocessing helper (no instrument response — raw counts)
# ---------------------------------------------------------------------------

def preprocess_trace(tr, freq_min, freq_max, sensitivity=None):
    """
    Detrend + bandpass. If sensitivity (counts per m/s) is provided,
    also converts the data to velocity in µm/s.
    Returns a processed copy; does not modify the original trace.
    """
    tr_out = tr.copy()
    tr_out.detrend('demean')
    tr_out.detrend('linear')
    tr_out.taper(max_percentage=0.01, type='cosine')
    fs = tr_out.stats.sampling_rate
    fmax_safe = min(freq_max, 0.45 * fs)
    tr_out.filter('bandpass', freqmin=freq_min, freqmax=fmax_safe,
                  corners=4, zerophase=True)
    if sensitivity is not None:
        tr_out.data = (tr_out.data / sensitivity) * 1e6   # counts → µm/s
    return tr_out


# ---------------------------------------------------------------------------
# Per-day figure (waveform + spectrogram energy)  — same layout as fios_02a
# ---------------------------------------------------------------------------

def plot_day(tr_filt, energy_t, energy_v, detections, day_str, freq_min, freq_max,
            nsta, nlta, thr_on, thr_off, out_dir, sensitivity=None, waveform_ylim=None):
    """
    Two-panel figure:
      Top    — filtered waveform (µm/s if sensitivity was applied, counts otherwise)
               with detected windows shaded in red
      Bottom — spectrogram energy (sum_cft from DetecteurV3) with threshold line
    X-axis: hours of the day (00:00 → 24:00 UTC)
    """
    fs         = tr_filt.stats.sampling_rate
    t_start    = tr_filt.stats.starttime
    n_det      = len(detections)

    # Time axes in hours from midnight
    t_midnight = UTCDateTime(day_str)   # 00:00:00 UTC of that day
    offset_h   = float(t_start - t_midnight) / 3600.0
    t_ax_h     = offset_h + np.arange(len(tr_filt.data)) / fs / 3600.0

    # Convert energy time series (seconds from trace start) to hours from midnight
    energy_t_h = [offset_h + t_s / 3600.0 for t_s in energy_t]

    ylabel_wave = 'Velocity (µm/s)' if sensitivity is not None else 'Amplitude'

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={'height_ratios': [3, 2]}
    )

    # -- Panel 1: waveform ---------------------------------------------------
    ax1.plot(t_ax_h, tr_filt.data, 'k-', lw=0.35, alpha=0.85)
    for det in detections:
        t_on_h  = float(UTCDateTime(det['starttime']) - t_midnight) / 3600.0
        t_off_h = float(UTCDateTime(det['endtime'])   - t_midnight) / 3600.0
        ax1.axvspan(t_on_h, t_off_h, color='red', alpha=0.25, lw=0)
    ax1.set_ylabel(ylabel_wave)
    if waveform_ylim is not None:
        ax1.set_ylim(-waveform_ylim, waveform_ylim)
    ax1.set_title(
        f"FIO1 — {day_str} — {n_det} detection(s)   [Spectrogram STA/LTA]\n"
        f"Band: {freq_min}–{freq_max} Hz  |  "
        f"NSTA={nsta}  NLTA={nlta}  |  "
        f"thr ON={thr_on} / OFF={thr_off}",
        fontsize=10
    )
    ax1.grid(axis='x', lw=0.3, alpha=0.4)

    # -- Panel 2: spectrogram energy -----------------------------------------
    if len(energy_t_h) > 0 and len(energy_v) > 0:
        ax2.plot(energy_t_h, energy_v, color='steelblue', lw=0.6, alpha=0.85)
        ax2.axhline(thr_on,  color='red',    ls='--', lw=1.0, label=f'ON  = {thr_on}')
        ax2.axhline(thr_off, color='orange', ls='--', lw=1.0, label=f'OFF = {thr_off}')
        ax2.set_ylabel('STA/LTA ratio (spectrogram, bidirectional)')
        ax2.set_ylim(bottom=0)
    else:
        # Fallback: stem plot of detection onset times
        onset_h = [float(UTCDateTime(det['starttime']) - t_midnight) / 3600.0
                   for det in detections]
        if onset_h:
            ax2.vlines(onset_h, 0, 1, color='red', lw=1.2, alpha=0.7)
        ax2.set_ylabel('Detections (onset)')
        ax2.set_yticks([])

    for det in detections:
        t_on_h  = float(UTCDateTime(det['starttime']) - t_midnight) / 3600.0
        t_off_h = float(UTCDateTime(det['endtime'])   - t_midnight) / 3600.0
        ax2.axvspan(t_on_h, t_off_h, color='red', alpha=0.15, lw=0)
    ax2.set_xlabel('Time (UTC)')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(axis='x', lw=0.3, alpha=0.4)

    # -- X-axis: hours from midnight, ticks every 2 hours -------------------
    ax1.set_xlim(0, 24)
    ax1.set_xticks(range(0, 25, 2))
    ax1.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 2)])

    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"fig_{day_str.replace('-', '')}.png")
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Run setup
# ---------------------------------------------------------------------------

RUN_DIR, _RUN_STAMP = create_run_dir(OUTPUT_DIR)
_log_file, _log_filename = setup_logging(
    RUN_DIR, "02b_spectrogram_sta_lta_FIOS.py",
    extra_info=(
        f"Station : {NETWORK}.{STATION}.{LOCATION}.{CHANNEL}  |  "
        f"Band : {FREQ_MIN}–{FREQ_MAX} Hz  |  "
        f"NSTA={NSTA}  NLTA={NLTA}  ON={THR_ON}  OFF={THR_OFF}  |  "
        f"NWIN={NWIN_SEC}s"
    )
)
set_matplotlib_defaults()



# =============================================================================
# SECTION 3 — MAIN LOOP (day by day)
# =============================================================================

t0   = UTCDateTime(T_START)
t1   = UTCDateTime(T_END)
days = []
d = t0
while d < t1:
    days.append(d)
    d += 86400

print(f"\n{'='*70}")
print(f"  FIOS Spectrogram STA/LTA — {len(days)} day(s) to process")
print(f"  Station : {NETWORK}.{STATION}.{LOCATION}.{CHANNEL}")
print(f"  Band    : {FREQ_MIN}–{FREQ_MAX} Hz")
print(f"  NSTA={NSTA}  NLTA={NLTA}  ON={THR_ON}  OFF={THR_OFF}  NWIN={NWIN_SEC}s")
print(f"  Min duration : {MIN_EVENT_DUR_SEC}s")
print(f"{'='*70}\n")

all_rows     = []
daily_counts = []

for day_utc in days:
    day_str = day_utc.strftime('%Y-%m-%d')
    print(f"\n{'='*60}")
    print(f"  Day : {day_str}")
    print(f"{'='*60}")

    # ---- Load ----------------------------------------------------------------
    tr_raw, n_files = load_day_local(
        DATA_ROOT, day_utc, NETWORK, STATION, LOCATION, CHANNEL
    )

    if tr_raw is None:
        print(f"  [SKIP] No MiniSEED files found for {day_str}.")
        daily_counts.append({'day': day_str, 'n_detections': 0,
                              'data_available': False})
        continue

    seg_dur = tr_raw.stats.endtime - tr_raw.stats.starttime
    fs      = tr_raw.stats.sampling_rate
    print(f"  {n_files} file(s) loaded  |  {seg_dur/3600:.2f} h  |  {fs:.0f} Hz")

    # ---- Preprocess ---------------------------------------------------------
    tr_vel = preprocess_trace(tr_raw, FREQ_MIN, FREQ_MAX, sensitivity=SENSITIVITY)

    # ---- Spectrogram parameters (depend on sampling rate) -------------------
    nwin  = int(NWIN_SEC * fs)
    nover = int(nwin * NOVER_PCT)
    nfft  = 2 ** int(np.ceil(np.log2(nwin)))
    # Ensure enough frequency resolution for our band
    df = fs / nfft
    if FREQ_MAX / df < 2:
        nfft = 2 ** int(np.ceil(np.log2(FREQ_MAX * 4)))

    # ---- Detection on sliding windows ---------------------------------------
    total_events     = {}
    total_thresholds = {}
    n_windows        = 0

    # Lists to collect energy time series for the day figure
    energy_t_all = []
    energy_v_all = []

    t_start_tr = tr_vel.stats.starttime
    win_start  = t_start_tr

    while win_start < tr_vel.stats.endtime:
        win_end = min(win_start + WINDOW_SEC, tr_vel.stats.endtime)
        tr_win  = tr_vel.slice(win_start, win_end)

        win_dur       = tr_win.stats.endtime - tr_win.stats.starttime
        dt_nrj_approx = NWIN_SEC * (1 - NOVER_PCT)
        if win_dur / dt_nrj_approx <= NLTA:
            break

        try:
            _, t_nrj, _, sum_cft, events_dt, thresholds_dt = DetecteurV3(
                tr_win, FREQ_MIN, FREQ_MAX,
                NSTA, NLTA, THR_ON, THR_OFF,
                nwin, nover, nfft, 'True'
            )
        except Exception as e:
            print(f"    [WARN] DetecteurV3 failed on window starting "
                  f"{win_start.strftime('%H:%M:%S')}: {e}")
            if win_end >= tr_vel.stats.endtime:
                break
            win_start = win_end - OVERLAP_SEC
            continue

        # Collect energy for plotting
        sum_cft_flat = sum_cft.flatten()
        for i_t, t_dt in enumerate(t_nrj):
            if i_t < len(sum_cft_flat):
                t_s = UTCDateTime(str(t_dt)) - t_start_tr
                energy_t_all.append(t_s)
                energy_v_all.append(sum_cft_flat[i_t])

        # Parse detected windows, filter by minimum duration
        win_events     = {}
        win_thresholds = {}
        k = 1
        for orig_key, val in events_dt.items():
            t_on  = UTCDateTime(str(val[0]))
            t_off = UTCDateTime(str(val[1]))
            if (t_off - t_on) >= MIN_EVENT_DUR_SEC:
                win_events[f"Event_{k}"]     = [t_on, t_off]
                win_thresholds[f"Event_{k}"] = thresholds_dt.get(orig_key, [0.0, 0.0])
                k += 1

        total_events, total_thresholds = merge_window_events(
            total_events, total_thresholds, win_events, win_thresholds
        )

        if win_end >= tr_vel.stats.endtime:
            break
        win_start = win_end - OVERLAP_SEC
        n_windows += 1

    print(f"  {n_windows} windows scanned  →  {len(total_events)} raw detection(s)")

    # ---- Build detection rows with SNR --------------------------------------
    detections = []

    for ev_key, (t_on, t_off) in total_events.items():
        dur      = t_off - t_on
        thr_pair = total_thresholds.get(ev_key, [0.0, 0.0])
        detections.append({
            'day'        : day_str,
            'starttime'  : str(t_on),
            'endtime'    : str(t_off),
            'duration_s' : round(dur, 3),
            'max_cft'    : round(float(thr_pair[0]), 3),
        })

    n_det = len(detections)
    print(f"  Final detections : {n_det}")

    all_rows.extend(detections)
    daily_counts.append({'day': day_str, 'n_detections': n_det,
                         'data_available': True})

    # ---- Save daily CSV ------------------------------------------------------
    if detections:
        df_day  = pd.DataFrame(detections)
        day_csv = os.path.join(RUN_DIR, f"detections_{day_str.replace('-','')}.csv")
        df_day.to_csv(day_csv, index=False)
        print(f"  [SAVED] {os.path.basename(day_csv)}")

    # ---- Per-day figure ------------------------------------------------------
    if PLOT_DAYS:
        plot_day(tr_vel, energy_t_all, energy_v_all, detections, day_str,
                 FREQ_MIN, FREQ_MAX, NSTA, NLTA, THR_ON, THR_OFF,
                 RUN_DIR, sensitivity=SENSITIVITY, waveform_ylim=WAVEFORM_YLIM)



# =============================================================================
# SECTION 4 — SUMMARY: DAILY COUNTS CSV + BAR CHART
# =============================================================================

# Daily counts CSV
df_counts  = pd.DataFrame(daily_counts)
counts_csv = os.path.join(RUN_DIR, "daily_counts.csv")
df_counts.to_csv(counts_csv, index=False)
print(f"\n[SAVED] Daily counts → {counts_csv}")

# All-detections CSV
if all_rows:
    df_all  = pd.DataFrame(all_rows)
    all_csv = os.path.join(RUN_DIR, f"detections_all_{_RUN_STAMP}.csv")
    df_all.to_csv(all_csv, index=False)
    print(f"[SAVED] All detections → {all_csv}")
    print(f"        {df_all.shape[0]} detections  |  "
          f"{df_all['day'].nunique()} days with at least 1 event")
    print("\n  Detections per day (top 10):")
    for row in sorted(daily_counts, key=lambda r: r['n_detections'], reverse=True)[:10]:
        if row['data_available']:
            print(f"    {row['day']} : {row['n_detections']:5d}")
else:
    print("[WARN] No detections found across all days — check thresholds and frequency band.")

# Daily count bar chart  (identical layout to fios_02a for comparison)
fig, ax = plt.subplots(figsize=(16, 5))

t_destab_dt = datetime.strptime(DESTAB_DATE, '%Y-%m-%d')
avail       = df_counts[df_counts['data_available']]
days_dt  = [datetime.strptime(d, '%Y-%m-%d') for d in avail['day']]
counts_v = avail['n_detections'].values

ax.bar(days_dt, counts_v, color='#1f77b4', width=0.8, alpha=0.85, zorder=2)
ax.axvline(t_destab_dt, color='black', ls='--', lw=1.5,
           label=f'Destabilisation onset: {DESTAB_DATE}', zorder=3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
plt.xticks(rotation=45, ha='right')
ax.set_ylabel('Number of detections per day')
ax.set_title(
    f"FIO1 — Daily microseismicity count  [Spectrogram STA/LTA]\n"
    f"Band: {FREQ_MIN}–{FREQ_MAX} Hz  |  "
    f"NSTA={NSTA}  NLTA={NLTA}  |  "
    f"thr ON={THR_ON} / OFF={THR_OFF}  |  min dur={MIN_EVENT_DUR_SEC}s\n"
    f"Dashed line = destabilisation onset ({DESTAB_DATE})"
)
ax.legend(fontsize=9)
ax.grid(axis='y', lw=0.4, alpha=0.5, zorder=0)
plt.tight_layout()

fig_counts = os.path.join(RUN_DIR, f"fig_daily_counts_{_RUN_STAMP}.png")
plt.savefig(fig_counts, dpi=150)
plt.close(fig)
print(f"[SAVED] Daily counts figure → {fig_counts}")



# =============================================================================
# END
# =============================================================================

n_days_with_data = sum(1 for r in daily_counts if r['data_available'])
n_days_with_det  = sum(1 for r in daily_counts if r['n_detections'] > 0)

print("\n" + "=" * 70)
print(f"  Run finished         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Days processed       : {n_days_with_data} / {len(days)}")
print(f"  Days with detections : {n_days_with_det}")
print(f"  Total detections     : {len(all_rows)}")
print(f"  All outputs          : {RUN_DIR}")
print(f"  Log file             : {_log_filename}")
print("=" * 70)

_log_file.close()
