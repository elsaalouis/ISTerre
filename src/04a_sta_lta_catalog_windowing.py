"""
CATALOG EVENT PRECISE WINDOWING
================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : May 2026

Goal
----
For each seismic event in the ISTerre FDSN catalog:
  1. Load the vertical-component waveform with PRE_EVENT seconds of pre-noise
  2. Remove instrument response -> ground velocity [m/s]
  3. Detect the precise start and end of the event using the chosen method (classical or Groult 2026)
  4. Flag: does the station's P-wave pick fall inside the detected window?
  5. Optionally refine the onset with the kurtosis picker (rockslides, Fuchs 2018)
  6. Extract 99 seismic features (Maggi/Hibert) from the detected window
  7. Compute 7 SNR metrics
  8. Save results to CSV and produce one diagnostic plot per event

Detection method
----------------
  'groult': Groult et al. (2026) spectrogram-based bidirectional STA/LTA
     - runs DetecteurV3 on the full trace (single call, no sliding window)
     - stacks spectral energy across FREQ_MIN–FREQ_MAX Hz, then applies a forward + backward STA/LTA on that energy time series

  'sta_lta': classical STA/LTA applied directly on the bandpass-filtered waveform
     - computes the ratio of short-term to long-term average of |signal|

Data sources
------------
  Catalog + picks : ISTerre FDSN server  http://ist-sc3-geobs.osug.fr:8080
  Waveforms       : ISTerre SDS archive  /data/sig/SDS

Output
------
  catalog_windows_<stamp>.csv: one row per (event × station × detection)
  event metadata  |  station  |  det_starttime / det_endtime / det_duration_s | origin_inside_det  |  origin_lag_s  |  pick_inside_det  |  pick_lag_s  |  quality_ok | 7 SNR cols  |  99 feature cols

  window_<etype>_<time>.png: diagnostic figure per event: waveform + characteristic function with detected windows
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# All parameters to adapt are grouped here
# =============================================================================

# -- Detection method ---------------------------------------------------------
DETECTION_METHOD = 'groult'  # 'groult' or 'sta_lta'

# -- Paths --------------------------------------------------------------------
SDS_ROOT    = "/data/sig/SDS"
ISTERRE_URL = "http://ist-sc3-geobs.osug.fr:8080"
OUTPUT_DIR  = "/data/failles/louisels/project/results/outputs_04a"

# -- Catalog query window -----------------------------------------------------
T_START = "2022-02-01"
T_END   = "2022-08-01"

LAT_MIN, LAT_MAX = 45.5, 46.0
LON_MIN, LON_MAX = 6.5, 7.2

TARGET_TYPES = ["earthquake", "rockslide", "ice quake"]

# -- Waveform extraction window -----------------------------------------------
PRE_EVENT  = 150   # [s] before the first pick, must be > LTA_S (classical) or enough for DetecteurV3 LTA warm-up
POST_EVENT = 90    # [s] after origin time

Z_CHANNELS = "??Z"

# -- Shared frequency band (used by both methods and SNR computation) ---------
FREQ_MIN = 1.0    # Hz
FREQ_MAX = 20.0   # Hz

# -- Groult spectrogram STA/LTA parameters ------------------------------------
#    (used only when DETECTION_METHOD = 'groult')
NSTA      = 1      # STA window length [spectrogram time steps]
NLTA      = 15     # LTA window length [spectrogram time steps]
THR_ON    = 8.0    # sum_cft threshold to trigger onset
THR_OFF   = 2.0    # sum_cft threshold to trigger offset
NWIN_SEC  = 5.0    # spectrogram window length [s]
NOVER_PCT = 0.20   # spectral overlap fraction

# -- Classical STA/LTA parameters ---------------------------------------------
#    (used only when DETECTION_METHOD = 'sta_lta')
STA_S     = 5      # Short-Term Average window [s]
LTA_S     = 100    # Long-Term Average window  [s]  — PRE_EVENT must be > LTA_S
THRES_ON  = 2.0    # STA/LTA ratio to trigger onset
THRES_OFF = 1.3    # STA/LTA ratio to trigger offset

# -- Minimum detection duration [s] (both methods) ----------------------------
MIN_DURAT_S = 2.0

# -- Feature extraction window padding ----------------------------------------
PAD_SEC = 5        # seconds added before t_on and after t_off for feature extraction only

# -- Quality flag thresholds — from ROC analysis (script 05a) -----------------
# SNR_full_mean  : AUC=0.653, ROC-optimal threshold=2.70  (TPR=0.507, FPR=0.271)
# SNR_s2n_median : AUC=0.663, ROC-optimal threshold=20.99 (TPR=0.517, FPR=0.272)
SNR_MEAN_MIN  = 2.70    # SNR_full_mean  >= this
SNR_S2N_MIN   = 20.99   # SNR_s2n_median >= this

# -- Kurtosis onset refiner (Fuchs 2018) — rockslides only --------------------
KURTOSIS_REFINE        = True
KURTOSIS_FREQ_MIN      = 1.0
KURTOSIS_FREQ_MAX      = 5.0
KURTOSIS_DT_S          = 5.0
KURTOSIS_SEARCH_BEFORE = 10.0
KURTOSIS_SEARCH_AFTER  = 1.0
KURTOSIS_ETYPES        = ('rockslide', 'landslide')

# -- Feature extraction -------------------------------------------------------
FEATURE_FLAG = 0   # 0 = 99 features, vertical component only
N_FEATURES   = 99

# -- Events to process --------------------------------------------------------
# Leave empty [] to process ALL catalog events, or list exact origin times
TARGET_EVENT_TIMES = [        # format: "YYYY-MM-DDTHH:MM:SS"

]



# =============================================================================
# SECTION 2 — SETUP
# =============================================================================

# Validate method choice immediately so the error is clear
if DETECTION_METHOD not in ('groult', 'sta_lta'):
    raise ValueError(
        f"DETECTION_METHOD must be 'groult' or 'sta_lta', got '{DETECTION_METHOD}'"
    )

# ------------- Imports ----------------
import os
import sys
import warnings
import datetime as _dt

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')

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
from detection import compute_snr, refine_onset_kurtosis
if DETECTION_METHOD == 'sta_lta':
    from detection import run_sta_lta
from visualization import plot_windowing


# ----------- Run setup ----------------
if DETECTION_METHOD == 'groult':
    _method_info = (
        f"Groult spectrogram STA/LTA  "
        f"FREQ={FREQ_MIN}–{FREQ_MAX} Hz  "
        f"nsta={NSTA}  nlta={NLTA}  thr_on={THR_ON}  thr_off={THR_OFF}"
    )
else:
    _method_info = (
        f"Classical STA/LTA  "
        f"FREQ={FREQ_MIN}–{FREQ_MAX} Hz  "
        f"STA={STA_S}s  LTA={LTA_S}s  ON={THRES_ON}  OFF={THRES_OFF}"
    )

RUN_DIR, _RUN_STAMP = create_run_dir(OUTPUT_DIR)
_log_file, _log_filename = setup_logging(
    RUN_DIR, "04a_catalog_windowing.py",
    extra_info=(
        f"Method: {DETECTION_METHOD}  |  {_method_info}  |  "
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
events = query_catalog(client_fdsn, T_START, T_END, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, TARGET_TYPES)
summarise_catalog(events)

if TARGET_EVENT_TIMES:
    batch = [find_event_by_time(events, t) for t in TARGET_EVENT_TIMES]
    batch = [ev for ev in batch if ev is not None]
    print(f"\nRestricted to {len(batch)} hand-picked events.")
else:
    batch = events
    print(f"\nProcessing all {len(batch)} catalog events of types: {TARGET_TYPES}")



# =============================================================================
# SECTION 3 — DETECTION FUNCTION
# Encapsulates the method-specific logic so the main loop stays clean
# =============================================================================

def detect_event(tr_vel, tr_filt, fs):
    """
    Run the selected detection method on one station trace

    Parameters
    ----------
    tr_vel  : obspy.Trace — response-removed velocity [m/s], broadband (no bandpass)
    tr_filt : obspy.Trace — same trace bandpass-filtered to FREQ_MIN–FREQ_MAX Hz
    fs      : float — sampling rate [Hz]

    Returns
    -------
    detections : dict  {"Det_k": [UTCDateTime t_on, UTCDateTime t_off]}
    thresholds : dict  {"Det_k": [float cft_at_on, float cft_at_off]}
    t_nrj      : list of datetime.datetime — time axis of the characteristic function
    sum_cft    : numpy array (1-D) — characteristic function values
                 (STA/LTA ratio for 'sta_lta', sum_cft for 'groult')
    skip_msg   : str or None — if not None, detection was skipped; message explains why
    """
    trace_dur = tr_vel.stats.endtime - tr_vel.stats.starttime

    # ---- Groult spectrogram-based bidirectional STA/LTA ---------------------
    if DETECTION_METHOD == 'groult':

        # Check trace is long enough for the LTA in spectrogram steps
        dt_nrj      = NWIN_SEC * (1 - NOVER_PCT)   # spectrogram time step [s]
        n_nrj_steps = trace_dur / dt_nrj
        if n_nrj_steps <= NLTA:
            return {}, {}, [], np.array([]), (f"trace too short ({trace_dur:.0f}s → {n_nrj_steps:.0f} steps ≤ nlta={NLTA})")

        nwin = int(NWIN_SEC * fs)
        nover = int(nwin * NOVER_PCT)
        nfft  = 2 ** int(np.ceil(np.log2(nwin)))

        try:
            _, t_nrj, _, sum_cft_raw, events_dt, thresholds_dt = DetecteurV3(
                tr_vel, FREQ_MIN, FREQ_MAX,
                NSTA, NLTA, THR_ON, THR_OFF,
                nwin, nover, nfft, 'True', MIN_DURAT_S
            )
            sum_cft = np.array(sum_cft_raw).flatten()
        except Exception as e:
            return {}, {}, [], np.array([]), f"DetecteurV3 failed: {e}"

        # Convert events_dt keys to consistent naming and UTCDateTimes
        detections = {}
        thresholds = {}
        for k, (raw_key, val) in enumerate(events_dt.items(), start=1):
            t_on  = UTCDateTime(str(val[0]))
            t_off = UTCDateTime(str(val[1]))
            detections[f"Det_{k}"] = [t_on, t_off]
            thresholds[f"Det_{k}"] = thresholds_dt.get(raw_key, [0.0, 0.0])

        return detections, thresholds, t_nrj, sum_cft, None

    # ---- Classical STA/LTA on the bandpass-filtered waveform ----------------
    else:  # DETECTION_METHOD == 'sta_lta'

        # Check trace is long enough for the LTA window
        if trace_dur < LTA_S + STA_S:
            return {}, {}, [], np.array([]), (
                f"trace too short ({trace_dur:.0f}s < LTA + STA = {LTA_S + STA_S}s)"
            )

        try:
            cft, on_off = run_sta_lta(tr_filt, STA_S, LTA_S, THRES_ON, THRES_OFF)
        except Exception as e:
            return {}, {}, [], np.array([]), f"STA/LTA failed: {e}"

        # Convert sample-index pairs to UTCDateTime detections
        t_start = tr_vel.stats.starttime
        detections = {}
        thresholds = {}
        k = 1
        for (i_on, i_off) in on_off:
            t_on  = t_start + i_on  / fs
            t_off = t_start + i_off / fs
            if (t_off - t_on) >= MIN_DURAT_S:
                detections[f"Det_{k}"] = [t_on, t_off]
                thresholds[f"Det_{k}"] = [
                    float(cft[i_on])  if i_on  < len(cft) else THRES_ON,
                    float(cft[i_off]) if i_off < len(cft) else THRES_OFF,
                ]
                k += 1

        # Build a datetime.datetime time axis for the CFT
        # (same format as DetecteurV3's t_nrj, so plot_windowing works for both methods)
        t_nrj = [
            _dt.datetime.utcfromtimestamp((t_start + idx / fs).timestamp)
            for idx in range(len(cft))
        ]

        return detections, thresholds, t_nrj, cft, None



# =============================================================================
# SECTION 4 — MAIN PROCESSING LOOP
# For each catalog event × station: load → detect → flag → features + SNR → row + plot
# =============================================================================

print(f"\n--- Processing {len(batch)} catalog events  [{DETECTION_METHOD}] ---\n")

all_rows    = []
n_ev_ok     = 0
n_ev_skip   = 0
n_sta_total = 0
n_det_total = 0
n_no_det    = 0

for i, ev in enumerate(batch):
    origin       = ev.preferred_origin() or ev.origins[0]
    etype        = str(ev.event_type) if ev.event_type else "unknown"
    t_orig       = origin.time
    stas         = get_stations_from_picks(ev)
    picks_by_sta = get_pick_times(ev)

    print(f"\n{'='*60}")
    print(f"  Event {i+1}/{len(batch)}: {etype}  |  {t_orig}  |  {len(stas)} station(s)")

    # ---- Load waveforms -------------------------------------------------------
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

    # Bandpass-filtered stream — used by STA/LTA detection and SNR computation
    st_filt = st_vel.copy()
    for tr in st_filt:
        nyq = tr.stats.sampling_rate / 2
        tr.filter('bandpass', freqmin=FREQ_MIN, freqmax=min(FREQ_MAX, 0.9 * nyq),
                  corners=4, zerophase=True)

    # Narrow-band stream for kurtosis onset picker (rockslides, Fuchs 2018)
    st_kurtosis = st_vel.copy()
    for tr in st_kurtosis:
        nyq = tr.stats.sampling_rate / 2
        tr.filter('bandpass', freqmin=KURTOSIS_FREQ_MIN,
                  freqmax=min(KURTOSIS_FREQ_MAX, 0.9 * nyq),
                  corners=2, zerophase=True)

    # ---- Per-station loop ----------------------------------------------------
    station_data = []

    for tr_vel in st_vel:
        net  = tr_vel.stats.network
        sta  = tr_vel.stats.station
        chan = tr_vel.stats.channel
        fs   = tr_vel.stats.sampling_rate
        n_sta_total += 1

        filt_sel = st_filt.select(network=net, station=sta)
        if len(filt_sel) == 0:
            print(f"    [{net}.{sta}] SKIP — no filtered trace.")
            continue
        tr_filt = filt_sel[0]

        print(f"    [{net}.{sta}] Running {DETECTION_METHOD} on "
              f"{tr_vel.stats.endtime - tr_vel.stats.starttime:.0f}s trace ...")

        # ---- Detection (method-specific) -------------------------------------
        detections, thresholds, t_nrj, sum_cft, skip_msg = detect_event(
            tr_vel, tr_filt, fs
        )

        if skip_msg is not None:
            print(f"    [{net}.{sta}] SKIP — {skip_msg}")
            continue

        n_det = len(detections)
        print(f"    [{net}.{sta}] -> {n_det} detection(s)")

        # Accumulate for the per-event multi-station figure
        station_data.append({
            'tr_vel'    : tr_vel,
            'tr_filt'   : tr_filt,
            'detections': detections,
            'picks'     : picks_by_sta.get(sta, {}),
            't_nrj'     : t_nrj,
            'sum_cft'   : sum_cft,
        })

        if n_det == 0:
            n_no_det += 1
            continue

        n_det_total += n_det

        # ---- Features + SNR + flags for each detection -----------------------
        for det_key, (t_on, t_off) in detections.items():

            # Flag 1: catalog origin inside the detected window
            origin_inside = bool(t_on <= t_orig <= t_off)
            origin_lag_s  = round(float(t_orig - t_on), 2)

            # Kurtosis onset refiner (Fuchs 2018) — rockslides only
            t_on_raw       = t_on
            onset_refine_s = 0.0

            if KURTOSIS_REFINE and etype.lower() in KURTOSIS_ETYPES:
                kurt_sel = st_kurtosis.select(network=net, station=sta)
                if len(kurt_sel) > 0:
                    t_refined, _ = refine_onset_kurtosis(
                        kurt_sel[0], t_on,
                        dt_s          = KURTOSIS_DT_S,
                        search_before = KURTOSIS_SEARCH_BEFORE,
                        search_after  = KURTOSIS_SEARCH_AFTER,
                    )
                    onset_refine_s = round(float(t_refined - t_on), 2)
                    t_on = t_refined
                    print(f"      [kurtosis] {onset_refine_s:+.2f}s  "
                          f"({str(t_on_raw)[11:19]} → {str(t_on)[11:19]})")

            # Flag 2: P-wave pick inside the detected window
            p_pick = picks_by_sta.get(sta, {}).get('P', None)
            if p_pick is not None:
                pick_inside_det = bool(t_on <= p_pick <= t_off)
                pick_lag_s      = round(float(p_pick - t_on), 2)
            else:
                pick_inside_det = None
                pick_lag_s      = None

            # Padded window for feature extraction
            try:
                t_cut_on  = max(t_on  - PAD_SEC, tr_vel.stats.starttime)
                t_cut_off = min(t_off + PAD_SEC, tr_vel.stats.endtime)
                tr_cut    = tr_vel.slice(t_cut_on, t_cut_off)
            except Exception:
                continue
            if tr_cut.stats.npts < 10:
                continue

            feats = extract_features(
                tr_cut.data, fs,
                n_features   = N_FEATURES,
                feature_flag = FEATURE_FLAG,
            )

            snr = compute_snr(tr_filt, t_on, t_off)

            quality_ok = (
                snr.get('SNR_full_mean',   0) >= SNR_MEAN_MIN and
                snr.get('SNR_s2n_median',  0) >= SNR_S2N_MIN
            )

            row = {
                # Event metadata
                'event_time'       : str(t_orig),
                'event_type'       : etype,
                'catalog_lat'      : origin.latitude,
                'catalog_lon'      : origin.longitude,
                'catalog_depth_km' : (origin.depth / 1000.0 if origin.depth is not None else np.nan),
                # Station
                'network'          : net,
                'station'          : sta,
                'channel'          : chan,
                # Detection window
                'det_starttime'    : str(t_on),       # refined onset (after kurtosis)
                'det_starttime_raw': str(t_on_raw),   # raw detector onset (before kurtosis)
                'onset_refine_s'   : onset_refine_s,
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
                # 7 SNR metrics
                **snr,
            }
            for fname, fval in zip(FEATURE_NAMES, feats):
                row[fname] = fval

            all_rows.append(row)

    # ---- One diagnostic figure per event (all stations stacked) -------------
    if station_data:
        plot_windowing(
            station_data, t_orig,
            thr_on   = THR_ON    if DETECTION_METHOD == 'groult' else THRES_ON,
            thr_off  = THR_OFF   if DETECTION_METHOD == 'groult' else THRES_OFF,
            etype    = etype,
            run_dir  = RUN_DIR,
            freq_min = FREQ_MIN,
            freq_max = FREQ_MAX,
            nsta     = NSTA      if DETECTION_METHOD == 'groult' else STA_S,
            nlta     = NLTA      if DETECTION_METHOD == 'groult' else LTA_S,
            pre_event= PRE_EVENT,
        )



# =============================================================================
# SECTION 5 — SAVE CSV + PRINT SUMMARY
# =============================================================================

if not all_rows:
    print("\n[WARN] No detections extracted — CSV will not be written.")
else:
    df = pd.DataFrame(all_rows)

    meta_cols = [
        'event_time', 'event_type', 'catalog_lat', 'catalog_lon',
        'catalog_depth_km', 'network', 'station', 'channel',
        'det_starttime', 'det_starttime_raw', 'onset_refine_s',
        'det_endtime', 'det_duration_s',
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
    print(f"        {df.shape[0]} rows × {df.shape[1]} columns  [{DETECTION_METHOD}]")

    print(f"\n  {'Event type':<22} {'n_rows':>7}  "
          f"{'origin_inside':>14}  {'pick_inside':>12}  {'quality_ok':>10}")
    print("  " + "-" * 72)
    for etype_name, grp in df.groupby('event_type'):
        origin_pct = grp['origin_inside_det'].mean() * 100
        pick_col   = grp['pick_inside_det'].dropna()
        pick_pct   = pick_col.mean() * 100 if len(pick_col) > 0 else float('nan')
        qual_pct   = grp['quality_ok'].mean() * 100
        print(f"  {etype_name:<22} {len(grp):>7}  "
              f"{origin_pct:>13.1f}%  {pick_pct:>11.1f}%  {qual_pct:>9.1f}%")

    print(f"\n  Pick lag from detection start (+ = pick AFTER onset):")
    for etype_name, grp in df.groupby('event_type'):
        pick_lag = grp['pick_lag_s'].dropna()
        if len(pick_lag) > 0:
            print(f"    {etype_name:<22}  "
                  f"median={pick_lag.median():+.1f}s  "
                  f"mean={pick_lag.mean():+.1f}s  "
                  f"range=[{pick_lag.min():+.1f}s … {pick_lag.max():+.1f}s]")



# =============================================================================
# END
# =============================================================================

from datetime import datetime
print("\n" + "=" * 70)
print(f"  Run finished      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Method            : {DETECTION_METHOD}")
print(f"  Events OK         : {n_ev_ok}  |  skipped: {n_ev_skip}")
print(f"  Stations processed: {n_sta_total}")
print(f"  Total detections  : {n_det_total}")
print(f"  Stations with no detection: {n_no_det}")
print(f"  All outputs       : {RUN_DIR}")
print(f"  Log file          : {_log_filename}")
print("=" * 70)

_log_file.close()
