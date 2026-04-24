"""
EVENT DETECTION — SPECTROGRAM-BASED STA/LTA
=======================================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : April 2026

Goal
----
- Scan continuous 24-hour waveforms from the SDS archive, station by station and day by day
- Detect seismic events without relying on any pre-existing catalog

Detection algorithm (adapted from Groult et al. 2026)
------------------------------------------------------
For each 24-hour trace:
  1. Remove instrument response -> ground velocity [m/s]
  2. Bandpass filter to FREQ_MIN-FREQ_MAX Hz
  3. Slide a 10-min window (1-min overlap) along the day
  4. For each window: compute the short-time Fourier spectrogram, stack spectral energy across FREQ_MIN-FREQ_MAX Hz
     -> 1-D energy time series
  5. Run a bidirectional STA/LTA on that energy series (forward + backward, then summed)
  6. Merge detections separated by < 60 s; keep only events > 5 s
  7. Merge detections across consecutive 10-min windows
  8. For each detected window: extract 99 seismic features + 5 SNR measures

Key differences from Groult et al.
-----------------------------------
  - Data loaded from ISTerre SDS archive (not pre-downloaded mseed files)
  - Station list derived from FDSN inventory (not a fixed text file)
  - Output: CSV per day, columns compatible with script 03

Data sources
------------
  Waveforms  : ISTerre SDS archive  /data/sig/SDS  (cluster only)
  Inventory  : ISTerre FDSN server  http://ist-sc3-geobs.osug.fr:8080

Output
------
  detections_YYYYMMDD.csv: one row per (detection × station) metadata + 5 SNR cols + 99 feature cols
  run.log: full console output of the run
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# All parameters to adapt are grouped here
# =============================================================================

# -- Paths --------------------------------------------------------------------
SDS_ROOT    = "/data/sig/SDS"
ISTERRE_URL = "http://ist-sc3-geobs.osug.fr:8080"
OUTPUT_DIR  = "/data/failles/louisels/project/results/outputs_04"

# -- Time window to process ---------------------------------------------------
T_START = "2022-06-01"
T_END   = "2022-06-02"   # exclusive

# -- Spatial bounding box -----------------------------------------------------
LAT_MIN, LAT_MAX = 45.5, 46.0       # to filter the FDSN station inventory
LON_MIN, LON_MAX = 6.5,  7.2

# -- Channel selection --------------------------------------------------------
Z_CHANNEL = "??Z"       # vertical component only

# -- Detection parameters (Groult et al. 2026) --------------------------------
FREQ_MIN  = 1.0
FREQ_MAX  = 20.0   # cover ice quakes and quarry blasts that have energy above 10 Hz

NSTA      = 1      # STA window length
NLTA      = 15     # LTA window length
THR_ON    = 8.0    # STA/LTA ratio to trigger event onset
THR_OFF   = 2.0    # STA/LTA ratio to trigger event offset

# -- Spectrogram parameters ---------------------------------------------------
NWIN_SEC  = 5.0    # spectrogram window length [s]
NOVER_PCT = 0.20   # overlap fraction between spectrogram windows

# -- Sliding window segmentation of the 24-hour trace ------------------------
WINDOW_SEC  = 10 * 60   # 10-minute processing window
OVERLAP_SEC = 1  * 60   # 1-minute overlap between consecutive windows

# -- Event filtering ----------------------------------------------------------
MIN_EVENT_DUR_SEC = 5.0   # discard detections shorter than this

# -- Minimum trace length to attempt detection --------------------------------
# A trace shorter than LTA × spectrogram_dt cannot run STA/LTA
# spectrogram_dt = NWIN_SEC × (1 - NOVER_PCT) = 4.5 s
# -> minimum useful trace = NLTA × 4.5 = 67.5 s; we use a safe 120 s
MIN_TRACE_SEC = 120.0

# -- Feature extraction -------------------------------------------------------
FEATURE_FLAG = 0    # 0 = 99 features (vertical component only)
N_FEATURES   = 99

# -- Output -------------------------------------------------------------------
CHECKPOINT_EVERY = 10   # save intermediate CSV every N station-days processed



# =============================================================================
# SECTION 2 — SETUP
# =============================================================================

# ------------- Imports ----------------
import os
import sys
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from obspy import UTCDateTime

from detecteurV3_fonctions import DetecteurV3        # detection function from Groult et al. (2026)

from run_setup import (
    create_run_dir,
    setup_logging,
    connect_sds,
    connect_fdsn,
    fetch_inventory,
)
from preprocessing import preprocess_day
from features import FEATURE_NAMES, extract_features
from detection import compute_snr, merge_window_events
from catalog_helpers import build_station_list_from_inventory


# ----------- Run setup ----------------
RUN_DIR, _RUN_STAMP = create_run_dir(OUTPUT_DIR)
_log_file, _log_filename = setup_logging(
    RUN_DIR, "04_detection_pipeline.py",
    extra_info=(f"Period: {T_START} → {T_END}  |  "
                f"Detect band: {FREQ_MIN}–{FREQ_MAX} Hz  |  "
                f"STA/LTA: nsta={NSTA}  nlta={NLTA}  thr_on={THR_ON}  thr_off={THR_OFF}")
)


# --------- Connections ----------------
client_sds  = connect_sds(SDS_ROOT)
client_fdsn = connect_fdsn(ISTERRE_URL)

if client_sds is None or client_fdsn is None:
    print("[ERROR] Cannot proceed without SDS and FDSN. Exiting.")
    sys.exit(1)

inventory = fetch_inventory(client_fdsn, T_START, T_END,
                            lat_min=LAT_MIN, lat_max=LAT_MAX,
                            lon_min=LON_MIN, lon_max=LON_MAX)
if inventory is None:
    print("[ERROR] Could not fetch inventory. Exiting.")
    sys.exit(1)


# -------- Catalog query --------------
station_list = build_station_list_from_inventory(inventory) # build the list of (network, station, location, channel) to process from the inventory

print(f"\n{len(station_list)} station-channel pairs to process:")
for net, sta, loc, chan in station_list:
    print(f"  {net}.{sta}.{loc}.{chan}")



# =============================================================================
# SECTION 3 — MAIN PROCESSING LOOP
# Outer loop: days  |  Inner loop: stations
# For each (day, station): load 24h -> preprocess -> detect -> extract features
# =============================================================================

# Build list of days to process
t0   = UTCDateTime(T_START)
t1   = UTCDateTime(T_END)
days = []
d = t0
while d < t1:
    days.append(d)
    d += 86400

print(f"\n--- Processing {len(days)} day(s) × {len(station_list)} station(s) "
      f"= {len(days) * len(station_list)} station-days ---\n")

all_rows           = []
n_station_days     = 0
n_detections_total = 0

for day_utc in days:
    day_str   = day_utc.strftime('%Y-%m-%d')
    day_start = day_utc
    day_end   = day_utc + 86400

    print(f"\n{'='*60}")
    print(f"  Day : {day_str}")
    print(f"{'='*60}")

    day_rows = []

    for net, sta, loc, chan in station_list:

        print(f"\n  Station: {net}.{sta}.{loc}.{chan}")

        # ---- Load full-day waveform from SDS --------------------------------
        try:
            st = client_sds.get_waveforms(
                network   = net,
                station   = sta,
                location  = loc,
                channel   = chan,
                starttime = day_start,
                endtime   = day_end
            )
        except Exception as e:
            print(f"    [SKIP] SDS load failed: {e}")
            continue

        if len(st) == 0:
            print(f"    [SKIP] No data in SDS for this station-day.")
            continue

        # ---- Merge gaps -----------------------------------------------------
        try:
            st.merge(fill_value=None)
        except Exception:
            st.merge(fill_value=0)

        print(f"    Loaded {len(st)} continuous segment(s)")

        # ---- Process each continuous segment --------------------------------
        for seg_idx, tr_raw in enumerate(st):

            seg_dur = tr_raw.stats.endtime - tr_raw.stats.starttime
            if seg_dur < MIN_TRACE_SEC:
                print(f"    [SKIP] Segment {seg_idx+1}: too short "
                      f"({seg_dur:.0f} s < {MIN_TRACE_SEC} s)")
                continue

            print(f"    Segment {seg_idx+1}: "
                  f"{tr_raw.stats.starttime.strftime('%H:%M:%S')} → "
                  f"{tr_raw.stats.endtime.strftime('%H:%M:%S')} "
                  f"({seg_dur/3600:.2f} h)")

            fs = tr_raw.stats.sampling_rate

            # Spectrogram window parameters (computed from sampling rate)
            nwin  = int(NWIN_SEC  * fs)
            nover = int(nwin * NOVER_PCT)
            nfft  = 2 ** int(np.ceil(np.log2(nwin)))

            # Verify nfft gives enough frequency resolution for our band
            df = fs / nfft
            if FREQ_MAX / df < 2:
                nfft = 2 ** int(np.ceil(np.log2(FREQ_MAX * 4)))

            # ---- Preprocessing: response removal -> velocity ----------------
            tr_vel = preprocess_day(tr_raw, inventory)
            if tr_vel is None:
                print(f"    [SKIP] Preprocessing failed for segment {seg_idx+1}.")
                continue

            # ---- Filtered trace for SNR computation -------------------------
            tr_filt = tr_vel.copy()
            tr_filt.filter('bandpass',
                           freqmin  = FREQ_MIN,
                           freqmax  = min(FREQ_MAX, 0.9 * fs / 2),
                           corners  = 4,
                           zerophase= True)

            # ---- Detection on sliding 10-min windows ------------------------
            total_events     = {}
            total_thresholds = {}

            win_start = tr_vel.stats.starttime
            n_windows = 0

            while win_start < tr_vel.stats.endtime:
                win_end = min(win_start + WINDOW_SEC, tr_vel.stats.endtime)
                tr_win  = tr_vel.slice(win_start, win_end)

                win_dur       = tr_win.stats.endtime - tr_win.stats.starttime
                dt_nrj_approx = NWIN_SEC * (1 - NOVER_PCT)
                if win_dur / dt_nrj_approx <= NLTA:
                    break

                _, _, _, sum_cft, events_dt, thresholds_dt = DetecteurV3(
                    tr_win, FREQ_MIN, FREQ_MAX,
                    NSTA, NLTA, THR_ON, THR_OFF,
                    nwin, nover, nfft, 'True'
                )

                events     = {}
                thresholds = {}
                k = 1
                for orig_key, val in events_dt.items():
                    t_on  = UTCDateTime(str(val[0]))
                    t_off = UTCDateTime(str(val[1]))
                    if (t_off - t_on) >= MIN_EVENT_DUR_SEC:
                        events[f"Event_{k}"]     = [t_on, t_off]
                        thresholds[f"Event_{k}"] = thresholds_dt.get(orig_key, [0.0, 0.0])
                        k += 1

                total_events, total_thresholds = merge_window_events(
                    total_events, total_thresholds,
                    events,       thresholds
                )

                if win_end >= tr_vel.stats.endtime:
                    break
                win_start = win_end - OVERLAP_SEC
                n_windows += 1

            print(f"    Detection: {n_windows} windows scanned → "
                  f"{len(total_events)} event(s) found")

            # ---- Feature extraction for each detection ----------------------
            n_ok = 0
            for ev_key, (t_on, t_off) in total_events.items():

                try:
                    tr_cut = tr_vel.slice(t_on, t_off)
                except Exception:
                    continue

                if tr_cut.stats.npts < 10:
                    continue

                try:
                    coords = inventory.get_coordinates(
                        f"{net}.{sta}.{loc}.{chan}", t_on
                    )
                    lat = coords['latitude']
                    lon = coords['longitude']
                except Exception:
                    lat, lon = np.nan, np.nan

                feats = extract_features(tr_cut.data, tr_cut.stats.sampling_rate,
                                         n_features=N_FEATURES, feature_flag=FEATURE_FLAG)
                if np.all(np.isnan(feats)):
                    continue

                snr = compute_snr(tr_filt, t_on, t_off)

                row = {
                    'trace_id'        : f"{net}.{sta}.{loc}.{chan}",
                    'network'         : net,
                    'station'         : sta,
                    'location'        : loc,
                    'channel'         : chan,
                    'latitude'        : lat,
                    'longitude'       : lon,
                    'day'             : day_str,
                    'starttime'       : str(t_on),
                    'endtime'         : str(t_off),
                    'duration_s'      : round(t_off - t_on, 2),
                    'trigger_on_cft'  : round(total_thresholds[ev_key][0], 4),
                    'trigger_off_cft' : round(total_thresholds[ev_key][1], 4),
                    **snr,
                }
                for fname, fval in zip(FEATURE_NAMES, feats):
                    row[fname] = fval

                day_rows.append(row)
                all_rows.append(row)
                n_ok += 1

            print(f"    Features extracted: {n_ok}/{len(total_events)} detection(s)")
            n_detections_total += n_ok

        n_station_days += 1

        # ---- Checkpoint: save CSV of all rows so far ------------------------
        if CHECKPOINT_EVERY > 0 and n_station_days % CHECKPOINT_EVERY == 0 and all_rows:
            df_chk = pd.DataFrame(all_rows)
            chk_path = os.path.join(RUN_DIR, f"detections_checkpoint_{n_station_days}.csv")
            df_chk.to_csv(chk_path, index=False)
            print(f"\n  [CHECKPOINT] {len(all_rows)} rows → "
                  f"{os.path.basename(chk_path)}\n")

    # ---- Save daily CSV -----------------------------------------------------
    if day_rows:
        df_day = pd.DataFrame(day_rows)
        day_csv = os.path.join(RUN_DIR, f"detections_{day_str.replace('-', '')}.csv")
        df_day.to_csv(day_csv, index=False)
        print(f"\n  [SAVED] {len(day_rows)} detections → "
              f"{os.path.basename(day_csv)}")
    else:
        print(f"\n  [INFO] No detections on {day_str}.")



# =============================================================================
# SECTION 4 — SAVE FINAL FEATURE TABLE
# =============================================================================

if not all_rows:
    print("\n[WARN] No detections extracted — output CSV will not be written.")
else:
    df = pd.DataFrame(all_rows)

    meta_cols = [
        'trace_id', 'network', 'station', 'location', 'channel',
        'latitude', 'longitude', 'day', 'starttime', 'endtime',
        'duration_s', 'trigger_on_cft', 'trigger_off_cft',
        'SNR', 'SNR_picking_5_5', 'SNR_picking_3_3',
        'SNR_picking_1_3', 'SNR_full_mean',
    ]
    ordered_cols = meta_cols + FEATURE_NAMES
    df = df[[c for c in ordered_cols if c in df.columns]]

    csv_path = os.path.join(RUN_DIR, f"detections_all_{_RUN_STAMP}.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n[SAVED] Final detection table → {csv_path}")
    print(f"        Shape   : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"        Days    : {df['day'].nunique()}")
    print(f"        Stations: {df['station'].nunique()}")
    print(f"\n  Detections per day:")
    for day_name, grp in df.groupby('day'):
        print(f"    {day_name} : {len(grp):4d} detections on "
              f"{grp['station'].nunique()} stations")



# =============================================================================
# END
# =============================================================================

from datetime import datetime
print("\n" + "=" * 70)
print(f"  Run finished      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Total detections  : {n_detections_total}")
print(f"  Station-days      : {n_station_days}")
print(f"  All outputs       : {RUN_DIR}")
print(f"  Log file          : {_log_filename}")
print("=" * 70)

_log_file.close()
