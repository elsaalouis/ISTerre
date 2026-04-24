"""
DYNAMIC STATION SELECTION FROM CATALOG PICKS
=============================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : April 2026

Goal
----
For each seismic event in the ISTerre catalog:
 - find which stations have associated P or S picks
 - download the vertical-component waveforms for ALL those stations from the SDS archive on the cluster
 - preprocess them
 - produce two kinds of plots: waveform + PSD figure per event (one row per station), station coverage summary figures (histogram and box plot)

Data sources
------------
  Catalog + picks : ISTerre FDSN server  http://ist-sc3-geobs.osug.fr:8080
  Waveforms       : ISTerre SDS archive  /data/sig/SDS  (cluster only)
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# All parameters to adapt are grouped here
# =============================================================================

# -- Paths --------------------------------------------------------------------
SDS_ROOT    = "/data/sig/SDS"                                    # waveform archive on cluster
ISTERRE_URL = "http://ist-sc3-geobs.osug.fr:8080"                # FDSN catalog server
OUTPUT_DIR  = "/data/failles/louisels/project/results/outputs_01"

# -- Catalog query window -----------------------------------------------------
T_START = "2022-06-01"
T_END   = "2022-07-01"

# Spatial bounding box (Mont Blanc massif and surroundings)
LAT_MIN, LAT_MAX = 45.5, 46.0
LON_MIN, LON_MAX = 6.5, 7.2

# Event types to keep (exact strings used in the ISTerre catalog)
TARGET_TYPES = ["earthquake", "quarry blast", "ice quake", "rockslide"]

# -- Waveform -----------------------------------------------------------------
PRE_EVENT  = 10   # seconds to include BEFORE the first pick
POST_EVENT = 90   # seconds to include AFTER the origin time

# Channel wildcard for the SDS client
Z_CHANNELS = "??Z"  # matches any 3-character channel ending in Z

# Normalization : 'common' -> all traces divided by the global max (amplitudes comparable across stations)
#                 'individual' -> each trace divided by its own max (every trace fills the plot)
NORMALIZE = 'individual'

# -- Preprocessing ------------------------------------------------------------
# Default bandpass range (Hz)
FREQMIN_DEFAULT = 0.1
FREQMAX_DEFAULT = 100.0

# Per-type frequency ranges
FREQ_RANGES = {
    "earthquake":   (1.0, 40.0),
    "ice quake":    (1.0, 20.0),
    "rockslide":    (0.5, 15.0),
    "quarry blast": (2.0, 50.0),
}

# -- Events to process --------------------------------------------------------
# Hand-picked event times (origin times from the catalog)
TARGET_EVENT_TIMES = [       # Format: "YYYY-MM-DDTHH:MM:SS"
    "2022-06-26T07:27:02",
    "2022-06-29T20:22:22",
    "2022-06-01T00:06:33",
]



# =============================================================================
# SECTION 2 — SETUP
# =============================================================================

# ------------- Imports ----------------
import sys
import warnings

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')   # non-interactive backend (required for cluster)

from catalog_helpers import (
    get_stations_from_picks,
    find_event_by_time,
    summarise_catalog,
    query_catalog,
    get_freq_range,
    compute_station_coverage,
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
from visualization import plot_event_waveforms, plot_station_coverage


# ----------- Run setup ----------------
RUN_DIR, _RUN_STAMP = create_run_dir(OUTPUT_DIR)
_log_file, _log_filename = setup_logging(RUN_DIR, "01_dynamic_station_selection.py")

set_matplotlib_defaults()


# --------- Connections ----------------
client_sds  = connect_sds(SDS_ROOT)
client_fdsn = connect_fdsn(ISTERRE_URL)
inventory   = fetch_inventory(client_fdsn, T_START, T_END) if client_fdsn else None


# -------- Catalog query --------------
if client_fdsn is None:
    print("[ERROR] Cannot query catalog — FDSN client unavailable. Exiting.")
    sys.exit(1)

events = query_catalog(client_fdsn, T_START, T_END,
                       LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, TARGET_TYPES)
catalog_summary = summarise_catalog(events)



# =============================================================================
# SECTION 3 — STATION STATISTICS
# =============================================================================

# Compute how many events each station recorded, and how many stations recorded each event
station_counts, n_stations_per_event, counts_by_type = compute_station_coverage(events)

print(f"\nTotal unique (network, station) pairs across all events: "
      f"{len(station_counts)}")
print("\nTop 15 most-recorded stations:")
for net_sta, count in sorted(station_counts.items(), key=lambda x: -x[1])[:15]:
    net, sta = net_sta
    print(f"  {net}.{sta:>10s}  ->  {count:3d} events")


# Station coverage figures
print("\n--- Generating station coverage figures ---")

plot_station_coverage(
    station_counts       = station_counts,
    n_stations_per_event = n_stations_per_event,
    counts_by_type       = counts_by_type,
    t_start_str          = T_START,
    t_end_str            = T_END,
    run_dir              = RUN_DIR,
    n_events             = len(events),
)



# =============================================================================
# SECTION 4 — MAIN PROCESSING LOOP
# For each hand-picked event: load waveforms, preprocess, plot
# =============================================================================

if client_sds is None:
    print("\n[ERROR] SDS client unavailable — skipping waveform loop (not on cluster).")
else:
    print(f"\n--- Processing {len(TARGET_EVENT_TIMES)} hand-picked events ---")

    batch = [find_event_by_time(events, t) for t in TARGET_EVENT_TIMES]

    for i, ev in enumerate(batch):
        origin           = ev.preferred_origin() or ev.origins[0]
        etype            = str(ev.event_type) if ev.event_type else "unknown"
        stas             = get_stations_from_picks(ev)
        freqmin, freqmax = get_freq_range(ev, FREQ_RANGES, FREQMIN_DEFAULT, FREQMAX_DEFAULT)

        print(f"\n{'='*60}")
        print(f"  Event {i+1}/{len(batch)}: {etype}  |  {origin.time}  |  {len(stas)} stations")
        print(f"  Frequency filter : {freqmin}–{freqmax} Hz")

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

        # Step 2: bandpass filter the velocity stream -> waveform display
        st_proc = apply_bandpass(st_vel, freqmin, freqmax)
        print(f"    Bandpass filter applied : {freqmin}–{freqmax} Hz")

        # Step 3: PSD uses the velocity stream WITHOUT bandpass (true full spectrum)
        st_psd = st_vel

        plot_event_waveforms(st_proc, ev, t_start, RUN_DIR, normalize=NORMALIZE, freqmin=freqmin, freqmax=freqmax, st_psd=st_psd)



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
