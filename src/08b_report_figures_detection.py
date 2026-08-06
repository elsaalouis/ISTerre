"""
REPORT FIGURE — CLASSICAL vs SPECTROGRAM STA/LTA, SIDE BY SIDE
================================================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : August 2026

Goal
----
Make the difference between the two detection methods used elsewhere in the
pipeline (02a/04a 'sta_lta' vs 02b/04a 'groult') visually obvious for the
report, on a small number of real stations from a single catalog event:

  Classical STA/LTA (02a / 04a 'sta_lta')
    - operates directly on the raw waveform AMPLITUDE (bandpass-filtered)
    - characteristic function = ratio of short-term to long-term average of
      |signal|

  Groult et al. (2026) spectrogram STA/LTA (02b / 04a 'groult')
    - operates on a SPECTRAL ENERGY time series: the spectrogram is computed
      first, then power is stacked over freq_min-freq_max Hz into a 1-D
      energy curve, THEN STA/LTA is run on that curve (forward + backward,
      summed) -- not on the raw amplitude at all

Both methods are run with the SAME shared frequency band (freq_min-freq_max)
and on the SAME stations/event, so the only thing that differs between the
two detected windows is the algorithm itself.

04b (method_comparison.py) already answers "which method performs better,
in aggregate, across the whole catalog?" with statistics. This script answers
a different, complementary question -- "what does the difference actually
LOOK like on one real event?" -- which is why it produces one illustrative
multi-panel figure instead of summary statistics.

Output
------
  method_comparison_<etype>_<time>.png : one row per station, 3 columns
      (waveform with both methods' windows overlaid | classical CFT | Groult CFT)
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# -- Paths ----------------------------------------------------------------
SDS_ROOT    = "/data/sig/SDS"
ISTERRE_URL = "http://ist-sc3-geobs.osug.fr:8080"
OUTPUT_DIR  = "/data/failles/louisels/project/results/outputs_08b"

# -- Catalog query window ---------------------------------------------------
T_START = "2022-06-01"
T_END   = "2022-07-01"

LAT_MIN, LAT_MAX = 45.5, 46.0
LON_MIN, LON_MAX = 6.5, 7.2

TARGET_TYPES = ["earthquake", "ice quake", "rockslide"]

# -- Event to illustrate -----------------------------------------------------
# Same earthquake already used for the 02a reference figure (M0.5,
# 2022-06-30T12:45:22, 6C.CI18/CH.SALAN/FR.OGCY/FR.RSL/GU.LSD/GU.TRAV) so this
# figure reads as a direct "same event, method-by-method" companion to it.
TARGET_EVENT_TIME = "2022-06-30T12:45:22"

# -- Waveform extraction -----------------------------------------------------
PRE_EVENT  = 150   # [s] before origin -- must be > LTA_S (classical)
POST_EVENT = 90    # [s] after origin

Z_CHANNELS = "??Z"

# -- Shared detection frequency band (IDENTICAL for both methods -- this is
#    what makes the comparison fair). Matches the production 04a config. ----
FREQ_MIN = 1.0
FREQ_MAX = 20.0

# -- Classical STA/LTA parameters (matches 02a / 04a 'sta_lta' mode) --------
STA_S     = 5      # Short-Term Average window [s]
LTA_S     = 100    # Long-Term Average window  [s]
THRES_ON  = 2.0
THRES_OFF = 1.3

# -- Groult spectrogram STA/LTA parameters (matches 02b / 04a 'groult' mode) -
NSTA      = 1      # STA window [spectrogram time steps]
NLTA      = 15     # LTA window [spectrogram time steps]
THR_ON    = 8.0
THR_OFF   = 2.0
NWIN_SEC  = 5.0    # spectrogram window length [s]
NOVER_PCT = 0.20   # spectral overlap fraction

# -- Minimum detection duration, both methods (matches 04a) ------------------
MIN_DURAT_S = 2.0

# -- How many stations to show ------------------------------------------------
# Fewer rows than the original reference figure (6 stations) so the figure
# stays readable with 3 columns instead of 2. Stations where BOTH methods
# triggered are shown first (clearest comparison); if fewer than
# N_STATIONS_TO_SHOW stations have a double-trigger, the remainder is filled
# with any other station that triggered at least one method.
N_STATIONS_TO_SHOW = 3



# =============================================================================
# SECTION 2 — SETUP
# =============================================================================

import sys
import warnings

warnings.filterwarnings('ignore')

import numpy as np

import matplotlib
matplotlib.use('Agg')

from obspy import UTCDateTime

from catalog_helpers import find_event_by_time, summarise_catalog, query_catalog, get_pick_times
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
from detection import run_sta_lta
from detecteurV3_fonctions import DetecteurV3
from visualization import plot_method_comparison_windowing


# ----------- Run setup ----------------
RUN_DIR, _RUN_STAMP = create_run_dir(OUTPUT_DIR)
_log_file, _log_filename = setup_logging(
    RUN_DIR, "08b_report_figures_detection.py",
    extra_info=(f"Event: {TARGET_EVENT_TIME}  |  Band: {FREQ_MIN}-{FREQ_MAX} Hz  |  "
                f"Classical: sta={STA_S}s lta={LTA_S}s on={THRES_ON} off={THRES_OFF}  |  "
                f"Groult: nsta={NSTA} nlta={NLTA} on={THR_ON} off={THR_OFF}")
)

set_matplotlib_defaults()


# --------- Connections ----------------
client_sds  = connect_sds(SDS_ROOT)
client_fdsn = connect_fdsn(ISTERRE_URL)
inventory   = fetch_inventory(client_fdsn, T_START, T_END) if client_fdsn else None

if client_sds is None or client_fdsn is None:
    print("[ERROR] Cannot proceed without SDS and FDSN. Exiting.")
    sys.exit(1)


# -------- Catalog query --------------
events = query_catalog(client_fdsn, T_START, T_END,
                       LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, TARGET_TYPES)
summarise_catalog(events)

ev = find_event_by_time(events, TARGET_EVENT_TIME)
if ev is None:
    print(f"[ERROR] Event {TARGET_EVENT_TIME} not found in the queried catalog window.")
    sys.exit(1)

origin       = ev.preferred_origin() or ev.origins[0]
etype        = str(ev.event_type) if ev.event_type else "unknown"
picks_by_sta = get_pick_times(ev)

print(f"\n{'='*65}")
print(f"  Event: {etype}  |  {origin.time}")
print(f"  Detection band (both methods): {FREQ_MIN}-{FREQ_MAX} Hz")
print(f"{'='*65}")



# =============================================================================
# SECTION 3 — LOAD + PREPROCESS
# =============================================================================

print(f"\n  Loading {PRE_EVENT}s pre-event / {POST_EVENT}s post-event data ...")
st_raw, t_start, t_end = load_waveforms_sds(client_sds, ev, Z_CHANNELS, PRE_EVENT, POST_EVENT)
if len(st_raw) == 0:
    print("[ERROR] No waveforms found in SDS for this event. Exiting.")
    sys.exit(1)

station_times_df = build_station_times_df(st_raw, t_start, t_end)

# Response removal -> ground velocity [m/s], broadband (used by Groult directly)
st_vel = remove_response_or_fallback(st_raw, inventory, station_times_df)
if len(st_vel) == 0:
    print("[ERROR] No valid traces after response removal. Exiting.")
    sys.exit(1)

# Bandpass-filtered copy -> used for the waveform panel AND classical STA/LTA
st_filt = apply_bandpass(st_vel, FREQ_MIN, FREQ_MAX)
print(f"  {len(st_vel)} station(s) loaded, response removed, bandpass {FREQ_MIN}-{FREQ_MAX} Hz applied")



# =============================================================================
# SECTION 4 — RUN BOTH DETECTION METHODS PER STATION
# =============================================================================

print(f"\n  Running both detection methods per station:")

station_data_all = []

for tr_vel in st_vel:
    net  = tr_vel.stats.network
    sta  = tr_vel.stats.station
    fs   = tr_vel.stats.sampling_rate

    sel = st_filt.select(network=net, station=sta)
    if len(sel) == 0:
        print(f"    [{net}.{sta}] SKIP -- no filtered trace.")
        continue
    tr_filt = sel[0]

    # ---- Classical STA/LTA on the bandpass-filtered waveform ----------------
    try:
        cft_c, on_off_c = run_sta_lta(tr_filt, STA_S, LTA_S, THRES_ON, THRES_OFF)
        dets_c = {}
        k = 1
        for (i_on, i_off) in on_off_c:
            t_on  = tr_filt.stats.starttime + i_on / fs
            t_off = tr_filt.stats.starttime + i_off / fs
            if (t_off - t_on) >= MIN_DURAT_S:
                dets_c[f"Det_{k}"] = [t_on, t_off]
                k += 1
        t_cft_c = np.arange(len(cft_c)) / fs
    except Exception as e:
        print(f"    [{net}.{sta}] SKIP -- classical STA/LTA failed: {e}")
        continue

    # ---- Groult spectrogram STA/LTA on the broadband velocity trace ---------
    try:
        nwin  = int(NWIN_SEC * fs)
        nover = int(nwin * NOVER_PCT)
        nfft  = 2 ** int(np.ceil(np.log2(nwin)))
        _, t_nrj, _, sum_cft_raw, events_dt, _thr_dt = DetecteurV3(
            tr_vel, FREQ_MIN, FREQ_MAX,
            NSTA, NLTA, THR_ON, THR_OFF,
            nwin, nover, nfft, 'True', MIN_DURAT_S,
        )
        cft_g = np.array(sum_cft_raw).flatten()
        dets_g = {}
        for k, (raw_key, val) in enumerate(events_dt.items(), start=1):
            t_on  = UTCDateTime(str(val[0]))
            t_off = UTCDateTime(str(val[1]))
            dets_g[f"Det_{k}"] = [t_on, t_off]
        t_cft_g = np.array([UTCDateTime(str(t)) - tr_vel.stats.starttime for t in t_nrj])
    except Exception as e:
        print(f"    [{net}.{sta}] Groult STA/LTA failed: {e} (classical result kept)")
        dets_g, t_cft_g, cft_g = {}, np.array([]), np.array([])

    n_c, n_g = len(dets_c), len(dets_g)
    print(f"    [{net}.{sta}]  classical={n_c} det(s)   groult={n_g} det(s)")

    station_data_all.append({
        'tr_filt'   : tr_filt,
        'picks'     : picks_by_sta.get(sta, {}),
        'classical' : {'detections': dets_c, 't_cft': t_cft_c, 'cft': cft_c},
        'groult'    : {'detections': dets_g, 't_cft': t_cft_g, 'cft': cft_g},
        '_both'     : (n_c > 0 and n_g > 0),
    })



# =============================================================================
# SECTION 5 — SELECT STATIONS + PLOT
# =============================================================================

both_triggered = [s for s in station_data_all if s['_both']]
other_stations = [s for s in station_data_all if not s['_both']]
selected        = (both_triggered + other_stations)[:N_STATIONS_TO_SHOW]
for s in selected:
    del s['_both']

selected_ids = [f"{s['tr_filt'].stats.network}.{s['tr_filt'].stats.station}" for s in selected]
print(f"\n  {len(both_triggered)}/{len(station_data_all)} station(s) triggered BOTH methods")
print(f"  Showing {len(selected)} station(s): {selected_ids}")

if not selected:
    print("\n[WARN] No station produced a plottable trace -- nothing to plot.")
else:
    plot_method_comparison_windowing(
        selected, origin.time, etype, RUN_DIR,
        freq_min=FREQ_MIN, freq_max=FREQ_MAX,
        classical_params=(STA_S, LTA_S, THRES_ON, THRES_OFF),
        groult_params=(NSTA, NLTA, THR_ON, THR_OFF),
    )



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
