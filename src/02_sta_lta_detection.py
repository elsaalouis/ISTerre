"""
STA/LTA EVENT DETECTION
=======================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : April 2026

Goal
----
For each target seismic event, apply the STA/LTA (Short-Term Average / Long-Term Average) algorithm 
to each station's vertical-component waveform to automatically detect the start and end of the event
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

TARGET_TYPES = ["earthquake", "quarry blast", "ice quake", "rockslide"]

# -- Waveform extraction ------------------------------------------------------
PRE_EVENT  = 150  # seconds before first pick  (must be > LTA_S)
POST_EVENT = 90   # seconds after origin time

Z_CHANNELS = "??Z"

# -- Preprocessing ------------------------------------------------------------
FREQMIN_DEFAULT = 0.1
FREQMAX_DEFAULT = 100.0

FREQ_RANGES = {
    "earthquake":   (1.0, 40.0),
    "ice quake":    (1.0, 20.0),
    "rockslide":    (0.5, 15.0),
    "quarry blast": (2.0, 50.0),
}

# -- STA/LTA parameters -------------------------------------------------------
STA_S     = 5      # Short-Term Average window  (seconds)
LTA_S     = 100    # Long-Term Average window   (seconds)
THRES_ON  = 2.0    # Trigger ON  threshold (ratio above = event detected)
THRES_OFF = 1.3    # Trigger OFF threshold (ratio below = event ended)

# -- Events to process --------------------------------------------------------
TARGET_EVENT_TIMES = [        # Format: "YYYY-MM-DDTHH:MM:SS"
    "2022-06-26T07:27:02",
    "2022-06-29T20:22:22",
    "2022-06-30T12:45:22",
]



# =============================================================================
# SECTION 2 — SETUP
# =============================================================================

# ------------- Imports ----------------
import sys
import warnings

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

from catalog_helpers import (
    find_event_by_time,
    summarise_catalog,
    query_catalog,
    get_freq_range,
)
from preprocessing import (
    load_waveforms_sds,
    build_station_times_df,
    remove_response_or_fallback,
    apply_bandpass,
)
from run_setup import (
    create_run_dir,
    setup_logging,
    connect_sds,
    connect_fdsn,
    fetch_inventory,
    set_matplotlib_defaults,
)
from detection import run_sta_lta, summarise_detections
from visualization import plot_sta_lta


# ----------- Run setup ----------------
RUN_DIR, _RUN_STAMP = create_run_dir(OUTPUT_DIR)
_log_file, _log_filename = setup_logging(
    RUN_DIR, "02_sta_lta_detection.py",
    extra_info=f"STA={STA_S}s  LTA={LTA_S}s  ON={THRES_ON}  OFF={THRES_OFF}"
)

set_matplotlib_defaults()


# --------- Connections ----------------
client_sds  = connect_sds(SDS_ROOT)
client_fdsn = connect_fdsn(ISTERRE_URL)
inventory   = fetch_inventory(client_fdsn, T_START, T_END) if client_fdsn else None


# -------- Catalog query --------------
if client_fdsn is None:
    print("[ERROR] Cannot query catalog. Exiting.")
    sys.exit(1)

events = query_catalog(client_fdsn, T_START, T_END,
                       LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, TARGET_TYPES)
catalog_summary = summarise_catalog(events)



# =============================================================================
# SECTION 3 — MAIN PROCESSING LOOP
# =============================================================================

if client_sds is None:
    print("\n[ERROR] SDS client unavailable — skipping waveform loop.")
else:
    print(f"\n--- Processing {len(TARGET_EVENT_TIMES)} events ---")
    batch = [find_event_by_time(events, t) for t in TARGET_EVENT_TIMES]

    for i, ev in enumerate(batch):
        origin           = ev.preferred_origin() or ev.origins[0]
        etype            = str(ev.event_type) if ev.event_type else "unknown"
        freqmin, freqmax = get_freq_range(ev, FREQ_RANGES, FREQMIN_DEFAULT, FREQMAX_DEFAULT)

        print(f"\n{'='*60}")
        print(f"  Event {i+1}/{len(batch)}: {etype}  |  {origin.time}")
        print(f"  Frequency filter : {freqmin}–{freqmax} Hz")
        print(f"  Loading {PRE_EVENT}s pre-event data ...")

        st_raw, t_start, t_end = load_waveforms_sds(client_sds, ev, Z_CHANNELS, PRE_EVENT, POST_EVENT)
        if len(st_raw) == 0:
            print("    [SKIP] No waveforms found in SDS.")
            continue

        station_times_df = build_station_times_df(st_raw, t_start, t_end)

        # Step 1: remove instrument response -> ground velocity [m/s]
        st_vel = remove_response_or_fallback(st_raw, inventory, station_times_df)

        if len(st_vel) == 0:
            print("    [SKIP] No valid traces after response removal.")
            continue

        # Step 2: bandpass filter the velocity stream
        st_proc = apply_bandpass(st_vel, freqmin, freqmax)
        print(f"    Bandpass filter applied : {freqmin}–{freqmax} Hz")

        # Step 3: STA/LTA detection — compute CFT and detections per station
        print(f"  STA/LTA detections:")
        detections = {}
        cfts       = {}
        for tr in st_proc:
            cft, on_off = run_sta_lta(tr, STA_S, LTA_S, THRES_ON, THRES_OFF)
            cfts[tr.stats.station]       = cft
            detections[tr.stats.station] = summarise_detections(tr, on_off, t_start, THRES_ON)

        n_triggered = sum(1 for d in detections.values() if d)
        print(f"  → {n_triggered}/{len(st_proc)} stations triggered")

        plot_sta_lta(st_proc, ev, t_start, detections, cfts,
                     thres_on=THRES_ON, thres_off=THRES_OFF,
                     run_dir=RUN_DIR, sta_s=STA_S, lta_s=LTA_S,
                     freqmin=freqmin, freqmax=freqmax)



# =============================================================================
# END
# =============================================================================

from datetime import datetime
print("\n" + "=" * 70)
print(f"  Run finished : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  All outputs  : {RUN_DIR}")
print(f"  Log file     : {_log_filename}")
print("=" * 70)
_log_file.close()
