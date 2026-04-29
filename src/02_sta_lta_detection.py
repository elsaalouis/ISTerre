"""
CATALOG EVENT PRECISE WINDOWING — GROULT SPECTROGRAM STA/LTA
=============================================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : April 2026

Goal
----
For each seismic event already known in the ISTerre FDSN catalog:
  1. Load the vertical-component waveform with PRE_EVENT seconds of pre-noise
  2. Remove instrument response -> ground velocity [m/s]
  3. Run Groult's spectrogram-based bidirectional STA/LTA (DetecteurV3) on the full trace to detect the precise start and end of the event
  4. Flag: is the catalog origin time inside or outside the detection window?
     -> if outside: the catalog time is a poor proxy for the signal onset (expected for rockfalls whose P-wave is buried in the onset)
     Flag: is the station's P-wave pick time inside the detection window?
     -> more physically justified ground truth (teacher feedback, April 2026):
        origin time is before wave arrival for earthquakes, and unreliable for rockfalls;
        the P pick at each station is the actual observed arrival and should fall inside a correct detection
  5. Extract 99 seismic features (Maggi/Hibert) from the detected window
  6. Compute Groult's full set of 6 SNR metrics (Groult et al. 2026 use both mean AND median > 3 as quality gate)
  7. Save results to CSV and produce one diagnostic plot per station

Key difference with script 04
------------------------------
Because the trace is only ~240 s long (PRE_EVENT + POST_EVENT), it fits i a single DetecteurV3 call 
 -> no 10-minute sliding window needed

Data sources
------------
  Catalog + picks : ISTerre FDSN server  http://ist-sc3-geobs.osug.fr:8080
  Waveforms       : ISTerre SDS archive  /data/sig/SDS  (cluster only)

Output
------
  catalog_windows_<stamp>.csv
      one row per (event x station x detection):
      event metadata  |  station  |  det_starttime / det_endtime / det_duration_s
      origin_inside_det  |  origin_lag_s  |  pick_inside_det  |  pick_lag_s  |  quality_ok
      6 SNR cols  |  99 feature cols

  window_<etype>_<time>_<net>.<sta>.png
      diagnostic figure per station: waveform + sum_cft with detected windows and catalog origin time marked
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# All parameters to adapt are grouped here
# =============================================================================

# -- Paths --------------------------------------------------------------------
SDS_ROOT    = "/data/sig/SDS"
ISTERRE_URL = "http://ist-sc3-geobs.osug.fr:8080"
OUTPUT_DIR  = "/data/failles/louisels/project/results/outputs_02"

# -- Catalog query window -----------------------------------------------------
T_START = "2022-06-01"
T_END   = "2022-07-01"

LAT_MIN, LAT_MAX = 45.5, 46.0
LON_MIN, LON_MAX = 6.5, 7.2

# Event types to process (tutor: focus on earthquakes and rockfalls)
TARGET_TYPES = ["earthquake", "rockslide", "ice quake"]

# -- Waveform extraction window -----------------------------------------------
PRE_EVENT  = 150   # s before the first pick (long enough for LTA background)
POST_EVENT = 90    # s after origin time

Z_CHANNELS = "??Z"

# -- DetecteurV3 parameters (identical to script 04) --------------------------
FREQ_MIN  = 1.0    # Hz — lower bound for spectral detection and SNR filter
FREQ_MAX  = 20.0   # Hz — upper bound
NSTA      = 1      # STA window length (in spectrogram time steps)
NLTA      = 15     # LTA window length (in spectrogram time steps)
THR_ON    = 8.0    # sum_cft threshold to trigger onset
THR_OFF   = 2.0    # sum_cft threshold to trigger offset
NWIN_SEC  = 5.0    # spectrogram window length [s]
NOVER_PCT = 0.20   # spectral overlap fraction (script 04 convention: 20%)
MIN_DURAT = 2      # minimum number of seconds a STA/LTA exceedance must span in the spectrogram

# -- Feature extraction window padding ----------------------------------------
PAD_SEC = 5        # seconds added before t_on and after t_off when cutting the trace for
                   # feature extraction only — detection boundaries in the CSV are NOT padded.
                   # Padding ensures the 99 features have enough samples even for 1-step detections
                   # (~4 s wide). Set to 0 to disable.

# -- Quality flag thresholds (Groult: both mean AND median > 3) ---------------
SNR_MEAN_MIN   = 3.0
SNR_MEDIAN_MIN = 3.0

# -- Feature extraction -------------------------------------------------------
FEATURE_FLAG = 0   # 0 = 99 features, vertical component only
N_FEATURES   = 99

# -- Events to process --------------------------------------------------------
# Leave empty [] to process ALL catalog events of TARGET_TYPES, or list exact origin times to restrict the events
TARGET_EVENT_TIMES = [        # format: "YYYY-MM-DDTHH:MM:SS"
    "2022-06-26T07:27:02",
    "2022-06-29T20:22:22",
    "2022-06-01T00:06:33",
]



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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from obspy import UTCDateTime

from detecteurV3_fonctions import DetecteurV3

from catalog_helpers import (
    find_event_by_time,
    summarise_catalog,
    query_catalog,
    get_stations_from_picks,
    get_pick_times,
)
from preprocessing import (
    load_waveforms_sds,
    build_station_times_df,
    remove_response_or_fallback,
)
from run_setup import (
    create_run_dir,
    setup_logging,
    connect_sds,
    connect_fdsn,
    fetch_inventory,
    set_matplotlib_defaults,
)
from features import FEATURE_NAMES, extract_features
from detection import compute_snr
from visualization import plot_windowing


# ----------- Run setup ----------------
RUN_DIR, _RUN_STAMP = create_run_dir(OUTPUT_DIR)
_log_file, _log_filename = setup_logging(
    RUN_DIR, "02_sta_lta_detection.py",
    extra_info=(
        f"DetecteurV3  FREQ={FREQ_MIN}–{FREQ_MAX} Hz  "
        f"nsta={NSTA}  nlta={NLTA}  thr_on={THR_ON}  thr_off={THR_OFF}  "
        f"PRE_EVENT={PRE_EVENT}s  POST_EVENT={POST_EVENT}s"
    )
)
set_matplotlib_defaults()


# --------- Connections ----------------
client_sds  = connect_sds(SDS_ROOT)
client_fdsn = connect_fdsn(ISTERRE_URL)
inventory   = fetch_inventory(client_fdsn, T_START, T_END) if client_fdsn else None

if client_sds is None:
    print("\n[ERROR] SDS client unavailable — cannot load waveforms. Exiting.")
    sys.exit(1)

if client_fdsn is None:
    print("[ERROR] Cannot query catalog — FDSN client unavailable. Exiting.")
    sys.exit(1)


# -------- Catalog query --------------
events = query_catalog(client_fdsn, T_START, T_END,
                       LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, TARGET_TYPES)
summarise_catalog(events)

# Restrict to hand-picked events if TARGET_EVENT_TIMES is non-empty
if TARGET_EVENT_TIMES:
    batch = [find_event_by_time(events, t) for t in TARGET_EVENT_TIMES]
    batch = [ev for ev in batch if ev is not None]
    print(f"\nRestricted to {len(batch)} hand-picked events.")
else:
    batch = events
    print(f"\nProcessing all {len(batch)} catalog events of types: {TARGET_TYPES}")



# =============================================================================
# SECTION 3 — MAIN PROCESSING LOOP
# For each catalog event x station: load waveform -> DetecteurV3 -> flag origin -> features + SNR -> plot + row
# =============================================================================

print(f"\n--- Processing {len(batch)} catalog events ---\n")

all_rows    = []  # list of (event × station × detection)
n_ev_ok     = 0   # events found into the cluster
n_ev_skip   = 0   # events with no valid trace
n_sta_total = 0   # stations processed
n_det_total = 0   # toal detections
n_no_det    = 0   # stations where detector found nothing

for i, ev in enumerate(batch):
    origin       = ev.preferred_origin() or ev.origins[0]  # .origins carries location and time
    etype        = str(ev.event_type) if ev.event_type else "unknown"
    t_orig       = origin.time
    stas         = get_stations_from_picks(ev)
    picks_by_sta = get_pick_times(ev)   # {station_code: {'P': UTCDateTime or None, 'S': UTCDateTime or None}}

    print(f"\n{'='*60}")
    print(f"  Event {i+1}/{len(batch)}: {etype}  |  {t_orig}  |  {len(stas)} station(s)")

    # ---- Load waveforms for all pick-stations --------------------------------
    st_raw, t_start, t_end = load_waveforms_sds(
        client_sds, ev, Z_CHANNELS, PRE_EVENT, POST_EVENT
    )
    if len(st_raw) == 0:
        print("    [SKIP] No waveforms found in SDS.")
        n_ev_skip += 1
        continue

    station_times_df = build_station_times_df(st_raw, t_start, t_end)
    st_vel = remove_response_or_fallback(st_raw, inventory, station_times_df)
    if len(st_vel) == 0:
        print("    [SKIP] No valid traces after response removal.")
        n_ev_skip += 1
        continue

    n_ev_ok += 1

    # Bandpass-filtered copy for SNR computation (same band as detection)
    st_filt = st_vel.copy()
    for tr in st_filt:
        nyq = tr.stats.sampling_rate / 2
        tr.filter('bandpass', freqmin  = FREQ_MIN, freqmax  = min(FREQ_MAX, 0.9 * nyq), corners  = 2, zerophase= True)

    # ---- Per-station detection loop -----------------------------------------
    station_data = []   # accumulate per-station results for the multi-station figure

    for tr_vel in st_vel:           # each trace in st_vel is one station's vertical component recording
        net  = tr_vel.stats.network
        sta  = tr_vel.stats.station
        chan = tr_vel.stats.channel
        fs   = tr_vel.stats.sampling_rate
        n_sta_total += 1

        # Matching filtered trace
        filt_sel = st_filt.select(network=net, station=sta)
        if len(filt_sel) == 0:
            print(f"    [{net}.{sta}] SKIP — no filtered trace.")
            continue
        tr_filt = filt_sel[0]

        # Spectrogram parameters (same formula as script 04)
        nwin  = int(NWIN_SEC * fs)          # e.g. 5s × 100 Hz = 500 samples per FFT window
        nover = int(nwin * NOVER_PCT)       # e.g. 500 × 0.20 = 100 samples overlap
        nfft  = 2 ** int(np.ceil(np.log2(nwin))) # next power of 2 above nwin (here = 512)

        # Minimum trace length check
        trace_dur   = tr_vel.stats.endtime - tr_vel.stats.starttime
        dt_nrj      = NWIN_SEC * (1 - NOVER_PCT)   # spectrogram time step [s]  -> e.g. = 5 × 0.80 = 4.0 s per step
        n_nrj_steps = trace_dur / dt_nrj

        if n_nrj_steps <= NLTA:
            print(f"    [{net}.{sta}] SKIP — trace too short "
                  f"({trace_dur:.0f}s -> {n_nrj_steps:.0f} steps <= nlta={NLTA})")
            continue

        print(f"    [{net}.{sta}] Running DetecteurV3 on {trace_dur:.0f}s trace ...")

        # ---- Run DetecteurV3 (single call on the full window) ---------------
        try:
            # t_nrj         : list of datetime.datetime timestamps for each step of the energy time series
            # sum_cft       : the characteristic function (forward STA/LTA + backward STA/LTA summed)
            # events_dt     : dict of {"Event_1": [datetime_on, datetime_off], …} for each detected window
            # thresholds_dt : dict of the actual sum_cft values at trigger on and off for each detection
            _, t_nrj, _, sum_cft, events_dt, thresholds_dt = DetecteurV3(
                tr_vel, FREQ_MIN, FREQ_MAX, NSTA, NLTA, THR_ON, THR_OFF, nwin, nover, nfft, 'True', MIN_DURAT
            )
            sum_cft = np.array(sum_cft).flatten()   # DetecteurV3 returns shape (1, N) instead of (N,)
        except Exception as e:
            print(f"    [{net}.{sta}] DetecteurV3 failed: {e}")
            continue

        # Convert events_dt keys -> consistent naming
        detections = {}
        thresholds = {}
        for k, (raw_key, val) in enumerate(events_dt.items(), start=1):
            t_on  = UTCDateTime(str(val[0]))
            t_off = UTCDateTime(str(val[1]))
            detections[f"Det_{k}"] = [t_on, t_off]
            thresholds[f"Det_{k}"] = thresholds_dt.get(raw_key, [0.0, 0.0])

        n_det = len(detections)
        print(f"    [{net}.{sta}] -> {n_det} detection(s)")

        # Accumulate this station's data for the per-event multi-station figure
        # (always added, even when n_det == 0, so every station appears in the plot)
        station_data.append({
            'tr_vel'    : tr_vel,
            'tr_filt'   : tr_filt,
            'detections': detections,
            'picks'     : picks_by_sta.get(sta, {}),
            't_nrj'     : t_nrj,      # time axis from DetecteurV3 (datetime.datetime list)
            'sum_cft'   : sum_cft,    # bidirectional STA/LTA characteristic function
        })

        if n_det == 0:
            n_no_det += 1
            continue

        n_det_total += n_det

        # ---- Features + SNR + flags for each detection ----------------------
        for det_key, (t_on, t_off) in detections.items():

            # Key diagnostic 1: is the catalog origin inside the detected window?
            origin_inside = bool(t_on <= t_orig <= t_off)
            origin_lag_s  = round(float(t_orig - t_on), 2)   # positive = origin after detection start

            # Key diagnostic 2: is this station's P-wave pick inside the detected window?
            p_pick = picks_by_sta.get(sta, {}).get('P', None)
            if p_pick is not None:
                pick_inside_det = bool(t_on <= p_pick <= t_off)
                pick_lag_s      = round(float(p_pick - t_on), 2)  # positive = pick after detection start
            else:
                pick_inside_det = None   # no P pick available for this station
                pick_lag_s      = None

            try:
                # Padded window for feature extraction: PAD_SEC before t_on and after t_off.
                # Clamped to trace boundaries so slice never fails at the edges of the stream.
                t_cut_on  = max(t_on  - PAD_SEC, tr_vel.stats.starttime)
                t_cut_off = min(t_off + PAD_SEC, tr_vel.stats.endtime)
                tr_cut = tr_vel.slice(t_cut_on, t_cut_off)
            except Exception:
                continue
            if tr_cut.stats.npts < 10:
                continue

            feats = extract_features(      # computes the 99 Maggi/Hibert seismic features on the padded window
                tr_cut.data, fs,
                n_features   = N_FEATURES,
                feature_flag = FEATURE_FLAG,
            )

            snr = compute_snr(tr_filt, t_on, t_off)   # SNR on the unpadded detection window (signal vs pre-noise)

            quality_ok = (      # Groult's quality gate
                snr.get('SNR_full_mean',   0) >= SNR_MEAN_MIN and
                snr.get('SNR_full_median', 0) >= SNR_MEDIAN_MIN
            )

            # Everything assembled into a single flat dictionary and appended to all-rows
            row = {
                # Catalog event metadata
                'event_time'       : str(t_orig),
                'event_type'       : etype,
                'catalog_lat'      : origin.latitude,
                'catalog_lon'      : origin.longitude,
                'catalog_depth_km' : (origin.depth / 1000.0
                                      if origin.depth is not None else np.nan),
                # Station
                'network'          : net,
                'station'          : sta,
                'channel'          : chan,
                # Detection window
                'det_starttime'    : str(t_on),
                'det_endtime'      : str(t_off),
                'det_duration_s'   : round(t_off - t_on, 2),
                'trigger_on_cft'   : round(thresholds[det_key][0], 4),
                'trigger_off_cft'  : round(thresholds[det_key][1], 4),
                # Quality flags
                'origin_inside_det': origin_inside,
                'origin_lag_s'     : origin_lag_s,
                'pick_inside_det'  : pick_inside_det,
                'pick_lag_s'       : pick_lag_s,
                'quality_ok'       : quality_ok,
                # 6 SNR metrics (Groult full set)
                **snr,
            }
            for fname, fval in zip(FEATURE_NAMES, feats):
                row[fname] = fval

            all_rows.append(row)

    # ---- One diagnostic figure per event (all stations stacked) -------------
    if station_data:
        plot_windowing(
            station_data, t_orig,
            thr_on=THR_ON, thr_off=THR_OFF,
            etype=etype, run_dir=RUN_DIR,
            freq_min=FREQ_MIN, freq_max=FREQ_MAX,
            nsta=NSTA, nlta=NLTA,
            pre_event=PRE_EVENT,
        )



# =============================================================================
# SECTION 4 — SAVE CSV + PRINT SUMMARY
# =============================================================================

if not all_rows:
    print("\n[WARN] No detections extracted — CSV will not be written.")
else:
    df = pd.DataFrame(all_rows)

    meta_cols = [
        'event_time', 'event_type', 'catalog_lat', 'catalog_lon',
        'catalog_depth_km', 'network', 'station', 'channel',
        'det_starttime', 'det_endtime', 'det_duration_s',
        'trigger_on_cft', 'trigger_off_cft',
        'origin_inside_det', 'origin_lag_s',
        'pick_inside_det', 'pick_lag_s', 'quality_ok',
        'SNR', 'SNR_picking_5_5', 'SNR_picking_3_3',
        'SNR_picking_1_3', 'SNR_full_mean', 'SNR_full_median', 'SNR_s2n_median',
    ]
    ordered_cols = meta_cols + FEATURE_NAMES
    df = df[[c for c in ordered_cols if c in df.columns]]

    csv_path = os.path.join(RUN_DIR, f"catalog_windows_{_RUN_STAMP}.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n[SAVED] {csv_path}")
    print(f"        {df.shape[0]} rows x {df.shape[1]} columns")

    # --- Summary by event type
    print(f"\n  {'Event type':<22} {'n_rows':>7}  "
          f"{'origin_inside':>14}  {'pick_inside':>12}  {'quality_ok':>10}")
    print("  " + "-" * 72)
    for etype_name, grp in df.groupby('event_type'):
        origin_pct = grp['origin_inside_det'].mean() * 100
        # pick_inside_det may contain None → use dropna for the mean
        pick_col   = grp['pick_inside_det'].dropna()
        pick_pct   = pick_col.mean() * 100 if len(pick_col) > 0 else float('nan')
        qual_pct   = grp['quality_ok'].mean() * 100
        print(f"  {etype_name:<22} {len(grp):>7}  "
              f"{origin_pct:>13.1f}%  {pick_pct:>11.1f}%  {qual_pct:>9.1f}%")

    # --- Lag stats by type — both origin and pick
    print(f"\n  Origin lag from detection start "
          f"(+ means time is AFTER the detected onset):")
    for etype_name, grp in df.groupby('event_type'):
        lag = grp['origin_lag_s']
        print(f"    origin / {etype_name:<22}  "
              f"median = {lag.median():+.1f}s  "
              f"mean = {lag.mean():+.1f}s  "
              f"range = [{lag.min():+.1f}s ... {lag.max():+.1f}s]")
        pick_lag = grp['pick_lag_s'].dropna()
        if len(pick_lag) > 0:
            print(f"    pick   / {etype_name:<22}  "
                  f"median = {pick_lag.median():+.1f}s  "
                  f"mean = {pick_lag.mean():+.1f}s  "
                  f"range = [{pick_lag.min():+.1f}s ... {pick_lag.max():+.1f}s]")



# =============================================================================
# END
# =============================================================================

from datetime import datetime
print("\n" + "=" * 70)
print(f"  Run finished      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Events OK         : {n_ev_ok}  |  skipped: {n_ev_skip}")
print(f"  Stations processed: {n_sta_total}")
print(f"  Total detections  : {n_det_total}")
print(f"  Stations with no detection: {n_no_det}")
print(f"  All outputs       : {RUN_DIR}")
print(f"  Log file          : {_log_filename}")
print("=" * 70)

_log_file.close()
