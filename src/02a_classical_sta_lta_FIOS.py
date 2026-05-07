"""
FIOS LANDSLIDE — CLASSICAL STA/LTA MICROSEISMICITY DETECTION
=============================================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : May 2026

Goal
----
- scan continuous 24-hour waveforms from local MiniSEED files (Val d'Anniviers, Switzerland), day by day over the full monitoring period
- detect microseismic events using the classical STA/LTA algorithm
- produces daily event counts to quantify the increase in microseismicity

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
  fig_YYYYMMDD.png           — per-day waveform + STA/LTA CFT with detected windows
  fig_daily_counts_<stamp>.png — summary bar chart 
  run.log                    — full console output
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# All parameters to adapt are grouped here
# =============================================================================

# -- Data paths ---------------------------------------------------------------
DATA_ROOT  = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\data\FIOS"
OUTPUT_DIR = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\02a_fios_classical_sta_lta"

# -- Station ------------------------------------------------------------------
NETWORK  = "XT"
STATION  = "FIO1"
LOCATION = "01"        # 01 Z-component; 02 N-component; 03 E-component
CHANNEL  = "DHZ"       # DHZ; DHN; DHE

# -- Time window to process ---------------------------------------------------
T_START = "2026-03-19"   # first day to process
T_END   = "2026-05-06"   # last  day to process (exclusive)

# -- Preprocessing ------------------------------------------------------------
FREQMIN = 10.0    # bandpass low  cutoff -> above the 5–6 Hz resonance peak
FREQMAX = 80.0    # bandpass high cutoff

# -- STA/LTA parameters -------------------------------------------------------
STA_S     = 0.5    # Short-Term Average window (seconds)
LTA_S     = 60.0   # Long-Term Average window  (seconds)  [was 10 → raised]
THRES_ON  = 5.0    # STA/LTA ratio to declare event onset  [was 3.0 → raised]
THRES_OFF = 2.0    # STA/LTA ratio to declare event end    [was 1.5 → raised]

# -- Post-detection quality filter --------------------------------------------
MIN_DURATION_S = 0.5   # discard detections shorter than this  [0.5 s: good balance for local microseismicity]

# -- Reference date -----------------------------------------------------------
DESTAB_DATE = "2026-04-13"   # apparently destabilisation onset

# -- Instrument sensitivity ---------------------------------------------------
SENSITIVITY = None    # e.g. 4e8  [counts / (m/s)]

# -- Night-time window (UTC) for seismicity-only counting ---------------------
# Workers and machinery inactive → detections in this window are mostly seismic
# Switzerland is CEST = UTC+2 in spring  →  22:00–05:00 local = 20:00–03:00 UTC
NIGHT_START_UTC = 18   # 18:00 UTC = 20:00 local
NIGHT_END_UTC   = 4    # 04:00 UTC = 06:00 local

# -- Plotting -----------------------------------------------------------------
PLOT_DAYS     = True   # save a per-day waveform + CFT figure
WAVEFORM_YLIM = 20000  # fixed y-axis amplitude for all daily waveform panels



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
from detection  import run_sta_lta
from run_setup  import create_run_dir, setup_logging, set_matplotlib_defaults


# ---------------------------------------------------------------------------
# Local file loader — replaces the SDS client 
# ---------------------------------------------------------------------------

def load_day_local(data_root, date_utc, network, station, location, channel):
    """
    Glob all 1-hour MiniSEED files for the given date and return a merged Stream

    Files are expected at:
        {data_root}/{YYYYMM}/{network}.{station}.{location}.{channel}_{YYYYMMDD}_*.miniseed

    Returns
    -------
    tr : obspy.Trace or None — single merged trace for the full day; None if no files found
    n_files : int — nmber of hourly files loaded
    """
    date_str  = date_utc.strftime('%Y%m%d')
    month_str = date_utc.strftime('%Y%m')

    pattern = os.path.join(
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
# Preprocessing helper (no instrument response)
# ---------------------------------------------------------------------------

def preprocess_trace(tr, freqmin, freqmax, sensitivity=None):
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
    fmax_safe = min(freqmax, 0.45 * fs)
    tr_out.filter('bandpass', freqmin=freqmin, freqmax=fmax_safe,
                  corners=4, zerophase=True)
    if sensitivity is not None:
        tr_out.data = (tr_out.data / sensitivity) * 1e6   # counts → µm/s
    return tr_out


# ---------------------------------------------------------------------------
# Per-day figure (waveform + CFT)
# ---------------------------------------------------------------------------

def plot_day(tr_filt, cft, detections, day_str, sta_s, lta_s, thres_on, thres_off,
            freqmin, freqmax, out_dir, sensitivity=None, waveform_ylim=None):
    """
    Two-panel figure:
      Top    — filtered waveform (µm/s if sensitivity was applied, counts otherwise)
               with detected windows shaded in red
      Bottom — STA/LTA characteristic function with threshold lines
    X-axis: hours of the day (00:00 → 24:00 UTC)
    """
    fs         = tr_filt.stats.sampling_rate
    t_start    = tr_filt.stats.starttime
    n_det      = len(detections)

    # Time axes in hours from midnight
    t_midnight = UTCDateTime(day_str)   # 00:00:00 UTC of that day
    offset_h   = float(t_start - t_midnight) / 3600.0
    t_ax_h     = offset_h + np.arange(len(tr_filt.data)) / fs / 3600.0
    cft_ax_h   = offset_h + np.arange(len(cft))          / fs / 3600.0

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
        f"FIO1 — {day_str} — {n_det} detection(s)   [Classical STA/LTA]\n"
        f"Band: {freqmin}–{freqmax} Hz  |  "
        f"STA={sta_s}s  LTA={lta_s}s  |  "
        f"thr ON={thres_on} / OFF={thres_off}",
        fontsize=10
    )
    ax1.grid(axis='x', lw=0.3, alpha=0.4)

    # -- Panel 2: STA/LTA CFT ------------------------------------------------
    ax2.plot(cft_ax_h, cft, color='steelblue', lw=0.5)
    ax2.axhline(thres_on,  color='red',    ls='--', lw=1.0, label=f'ON  = {thres_on}')
    ax2.axhline(thres_off, color='orange', ls='--', lw=1.0, label=f'OFF = {thres_off}')
    for det in detections:
        t_on_h  = float(UTCDateTime(det['starttime']) - t_midnight) / 3600.0
        t_off_h = float(UTCDateTime(det['endtime'])   - t_midnight) / 3600.0
        ax2.axvspan(t_on_h, t_off_h, color='red', alpha=0.15, lw=0)
    ax2.set_ylabel('STA/LTA ratio')
    ax2.set_xlabel('Time (UTC)')
    ax2.set_ylim(bottom=0)
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
    RUN_DIR, "02a_classical_sta_lta_FIOS.py",
    extra_info=(
        f"Station : {NETWORK}.{STATION}.{LOCATION}.{CHANNEL}  |  "
        f"Band : {FREQMIN}–{FREQMAX} Hz  |  "
        f"STA={STA_S}s  LTA={LTA_S}s  ON={THRES_ON}  OFF={THRES_OFF}"
    )
)
set_matplotlib_defaults()



# =============================================================================
# SECTION 3 — MAIN LOOP (day by day)
# =============================================================================

# Build list of days
t0   = UTCDateTime(T_START)
t1   = UTCDateTime(T_END)
days = []
d = t0
while d < t1:
    days.append(d)
    d += 86400

print(f"\n{'='*70}")
print(f"  FIOS Classical STA/LTA — {len(days)} day(s) to process")
print(f"  Station : {NETWORK}.{STATION}.{LOCATION}.{CHANNEL}")
print(f"  Band    : {FREQMIN}–{FREQMAX} Hz")
print(f"  STA={STA_S}s  LTA={LTA_S}s  ON={THRES_ON}  OFF={THRES_OFF}")
print(f"  Min duration : {MIN_DURATION_S}s")
print(f"{'='*70}\n")

all_rows     = []   # one dict per detection across all days
daily_counts = []   # one dict per day (even if no data / no detections)

for day_utc in days:
    day_str = day_utc.strftime('%Y-%m-%d')
    print(f"\n{'='*60}")
    print(f"  Day : {day_str}")
    print(f"{'='*60}")

    # ---- Load -----------------------------------------------------------------
    tr_raw, n_files = load_day_local(
        DATA_ROOT, day_utc, NETWORK, STATION, LOCATION, CHANNEL
    )

    if tr_raw is None:
        print(f"  [SKIP] No MiniSEED files found for {day_str}.")
        daily_counts.append({'day': day_str, 'n_detections': 0,
                             'n_night': 0, 'data_available': False})
        continue

    seg_dur = tr_raw.stats.endtime - tr_raw.stats.starttime
    fs      = tr_raw.stats.sampling_rate
    print(f"  {n_files} file(s) loaded  |  {seg_dur/3600:.2f} h  |  {fs:.0f} Hz")

    # ---- Preprocess ----------------------------------------------------------
    tr_filt = preprocess_trace(tr_raw, FREQMIN, FREQMAX, sensitivity=SENSITIVITY)

    # ---- STA/LTA -------------------------------------------------------------
    cft, on_off = run_sta_lta(tr_filt, STA_S, LTA_S, THRES_ON, THRES_OFF)

    # ---- Filter by minimum duration + collect detections --------------------
    t_start_tr = tr_filt.stats.starttime
    detections = []

    for i_on, i_off in on_off:
        t_on  = t_start_tr + i_on  / fs
        t_off = t_start_tr + i_off / fs
        dur   = t_off - t_on

        if dur < MIN_DURATION_S:
            continue

        max_cft = float(np.max(cft[i_on : i_off + 1])) if i_off > i_on else float(THRES_ON)

        detections.append({
            'day'        : day_str,
            'starttime'  : str(t_on),
            'endtime'    : str(t_off),
            'duration_s' : round(dur, 3),
            'max_cft'    : round(max_cft, 3),
        })

    n_det = len(detections)
    raw_n = len(on_off)
    print(f"  Raw triggers : {raw_n}  →  after min_duration filter : {n_det}")

    # Night-only count: detections whose onset falls in the night window
    n_night = sum(
        1 for det in detections
        if (UTCDateTime(det['starttime']).hour >= NIGHT_START_UTC or
            UTCDateTime(det['starttime']).hour <  NIGHT_END_UTC)
    )
    print(f"  Night detections (UTC {NIGHT_START_UTC:02d}–{NIGHT_END_UTC:02d}) : {n_night}")

    all_rows.extend(detections)
    daily_counts.append({'day': day_str, 'n_detections': n_det,
                         'n_night': n_night, 'data_available': True})

    # ---- Save daily CSV ------------------------------------------------------
    if detections:
        df_day   = pd.DataFrame(detections)
        day_csv  = os.path.join(RUN_DIR, f"detections_{day_str.replace('-','')}.csv")
        df_day.to_csv(day_csv, index=False)
        print(f"  [SAVED] {os.path.basename(day_csv)}")

    # ---- Per-day figure ------------------------------------------------------
    if PLOT_DAYS:
        plot_day(tr_filt, cft, detections, day_str,
                 STA_S, LTA_S, THRES_ON, THRES_OFF,
                 FREQMIN, FREQMAX, RUN_DIR,
                 sensitivity=SENSITIVITY, waveform_ylim=WAVEFORM_YLIM)



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
    df_all   = pd.DataFrame(all_rows)
    all_csv  = os.path.join(RUN_DIR, f"detections_all_{_RUN_STAMP}.csv")
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

# Daily count bar chart — total (light blue) + night-only (dark blue) stacked
fig, ax = plt.subplots(figsize=(16, 5))

t_destab_dt  = datetime.strptime(DESTAB_DATE, '%Y-%m-%d')
avail        = df_counts[df_counts['data_available']]
days_dt      = [datetime.strptime(d, '%Y-%m-%d') for d in avail['day']]
counts_all   = avail['n_detections'].values
counts_night = avail['n_night'].values

bar_w = 0.8
ax.bar(days_dt, counts_all,   color='#aec7e8', width=bar_w, alpha=0.95, zorder=2,
       label='All hours')
ax.bar(days_dt, counts_night, color='#1f4e79', width=bar_w, alpha=0.95, zorder=3,
       label=f'Night only  (UTC {NIGHT_START_UTC:02d}:00 – {NIGHT_END_UTC:02d}:00)')
ax.axvline(t_destab_dt, color='black', ls='--', lw=1.5,
           label=f'Destabilisation onset: {DESTAB_DATE}', zorder=4)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
plt.xticks(rotation=45, ha='right')
ax.set_ylabel('Number of detections per day')
ax.set_title(
    f"FIO1 — Daily microseismicity count  [Classical STA/LTA]\n"
    f"Band: {FREQMIN}–{FREQMAX} Hz  |  "
    f"STA={STA_S}s  LTA={LTA_S}s  |  "
    f"thr ON={THRES_ON} / OFF={THRES_OFF}  |  min dur={MIN_DURATION_S}s\n"
    f"Dashed line = destabilisation onset ({DESTAB_DATE})"
)
ax.legend(fontsize=9)
ax.grid(axis='y', lw=0.4, alpha=0.5, zorder=0)
plt.tight_layout()

fig_counts = os.path.join(RUN_DIR, f"fig_daily_counts_{_RUN_STAMP}.png")
plt.savefig(fig_counts, dpi=150)
plt.close(fig)
print(f"[SAVED] Daily counts figure → {fig_counts}")


# Detection rate heatmap — date × hour-of-day
if all_rows:
    # Build (day, hour) pairs from all detections
    det_hours = pd.DataFrame([
        {'day' : det['day'],
         'hour': UTCDateTime(det['starttime']).hour}
        for det in all_rows
    ])
    # Pivot: rows = date (sorted), columns = hours 0–23, values = count per hour
    heat = (det_hours.groupby(['day', 'hour'])
                     .size()
                     .unstack(fill_value=0)
                     .reindex(columns=range(24), fill_value=0))
    heat = heat.sort_index()   # chronological order

    n_days_h = len(heat)
    fig_h, ax_h = plt.subplots(figsize=(16, max(5, n_days_h * 0.38)))

    im = ax_h.pcolormesh(
        range(25), range(n_days_h + 1),
        heat.values,
        cmap='YlOrRd', shading='flat'
    )
    plt.colorbar(im, ax=ax_h, label='Detections per hour', pad=0.01, shrink=0.8)

    # Night-time shading (two blocks: [0, NIGHT_END) and [NIGHT_START, 24))
    for x0, x1 in [(0, NIGHT_END_UTC), (NIGHT_START_UTC, 24)]:
        ax_h.axvspan(x0, x1, color='steelblue', alpha=0.10, zorder=0)

    # Mark destabilisation onset as horizontal dashed lines around that row
    day_list = heat.index.tolist()
    if DESTAB_DATE in day_list:
        i_d = day_list.index(DESTAB_DATE)
        ax_h.axhline(i_d,       color='black', lw=1.5, ls='--')
        ax_h.axhline(i_d + 1.0, color='black', lw=1.5, ls='--')

    # Y-axis: one tick per day
    ax_h.set_yticks(np.arange(n_days_h) + 0.5)
    ax_h.set_yticklabels(day_list, fontsize=7)

    # X-axis: every 2 hours
    ax_h.set_xticks(np.arange(0, 25, 2) + 0.5)
    ax_h.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 2)])

    ax_h.set_xlabel('Hour of day (UTC)')
    ax_h.set_ylabel('Date')
    ax_h.set_title(
        f"FIO1 — Detection rate heatmap  [Classical STA/LTA]\n"
        f"Band: {FREQMIN}–{FREQMAX} Hz  |  "
        f"STA={STA_S}s  LTA={LTA_S}s  |  "
        f"thr ON={THRES_ON} / OFF={THRES_OFF}\n"
        f"Dashed lines = destabilisation onset ({DESTAB_DATE})"
    )
    plt.tight_layout()

    fig_heat = os.path.join(RUN_DIR, f"fig_heatmap_{_RUN_STAMP}.png")
    plt.savefig(fig_heat, dpi=150)
    plt.close(fig_h)
    print(f"[SAVED] Heatmap figure → {fig_heat}")
else:
    print("[INFO] No detections — skipping heatmap.")



# =============================================================================
# END
# =============================================================================

n_days_with_data = sum(1 for r in daily_counts if r['data_available'])
n_days_with_det  = sum(1 for r in daily_counts if r['n_detections'] > 0)

print("\n" + "=" * 70)
print(f"  Run finished       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Days processed     : {n_days_with_data} / {len(days)}")
print(f"  Days with detections : {n_days_with_det}")
print(f"  Total detections   : {len(all_rows)}")
print(f"  All outputs        : {RUN_DIR}")
print(f"  Log file           : {_log_filename}")
print("=" * 70)

_log_file.close()
