"""
04d_noise_window_extraction.py
================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : July 2026

Goal
----
Build the 4th classification class: 
LOCAL NOISE — a real detected fluctuation that is confirmed NOT to be a network-wide seismic event

New approach
------------ 
1. run a classical STA/LTA detector (same algorithm and parameters as `02a_classical_sta_lta_detection.py`) over a full random day at a random station 
   -> every trigger it finds is guaranteed to be a real local fluctuation, not silence
2. each detected window is put through the local-noise checks:
   - reject if it overlaps a CATALOGED event, of ANY type
   - reject if any OTHER station within NEIGHBOR_RADIUS_KM also triggers during the same window (a seismic signal shows up on more than one station)

What survives both checks is a real, locally-confirmed, non-catalogued fluctuation: 
exactly the kind of "hard negative" the classifier needs to see to be useful for filtering real continuous-scan false triggers

Pipeline
--------
  1. Fetch the FDSN inventory for the whole bounding box -> resolve every candidate station's exact channel + operational epochs + coordinates
  2. Query the FULL catalog (all event types, no filter) over the same period and bounding box
  3. Repeatedly: pick a random station + a random day within its operational epochs 
     -> load that day's continuous trace, remove instrument response, bandpass filter, run classical STA/LTA 
     -> take up to MAX_CANDIDATES_PER_DAY of that day's triggers
  4. For each candidate trigger: reject if it overlaps a catalog exclusion interval, reject if any neighbor station also triggers during the same window
  5. Extract the 99/103 Maggi/Hibert features from the (t_on-PAD_SEC, t_off+PAD_SEC) window
  6. Save a CSV with the SAME column layout as `catalog_windows_<stamp>.csv` so it can be loaded and concatenated directly by 06a/06b/06c

Data sources
------------
  Catalog   : ISTerre FDSN server  http://ist-sc3-geobs.osug.fr:8080
  Waveforms : ISTerre SDS archive  /data/sig/SDS

Output
------
  noise_windows_<stamp>.csv : one row per accepted noise window, same
  metadata + 7 SNR (NaN) + 99/103 feature columns as 04a's catalog_windows CSV
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# -- Paths ----------------------------------------------------------------
SDS_ROOT    = "/data/sig/SDS"
ISTERRE_URL = "http://ist-sc3-geobs.osug.fr:8080"
OUTPUT_DIR  = "/data/failles/louisels/project/results/outputs_04d"

# -- Time range to sample from (full continuous archive) -------------------
T_START = "2015-01-01"
T_END   = "2026-07-01"

# -- Bounding box: every station in/around the Mont Blanc massif -----------
# Must match the box used everywhere else in the pipeline (01/02a/03a/04a) so
# "the network" means the same thing here as it does downstream.
LAT_MIN, LAT_MAX = 45.5, 46.0
LON_MIN, LON_MAX = 6.5, 7.2

# -- Chunked catalog query (avoids FDSN server timeout on a 10+ year span) ---
CHUNK_DAYS = 90
CATALOG_CACHE_FILE = "/data/failles/louisels/project/results/catalog_cache_all_types.xml"

# -- Exclusion margin kept clear around every CATALOGED event ---------------
EXCLUSION_BUFFER_S = 600.0

# -- Cross-station locality check (catches real but UN-catalogued signals) --
# UNCHANGED from the previous version — same mechanism, now applied to
# STA/LTA-detected windows instead of randomly-drawn ones.
NEIGHBOR_RADIUS_KM      = 15.0
MIN_NEIGHBORS_REQUIRED  = 1
COINCIDENCE_STA_S       = 2.0
COINCIDENCE_LTA_S       = 30.0
COINCIDENCE_THR_ON      = 3.5
COINCIDENCE_THR_OFF     = 1.5
COINCIDENCE_FREQ_MIN    = 1.0
COINCIDENCE_FREQ_MAX    = 20.0

# -- Primary detector: classical STA/LTA, SAME parameters as 02a ------------
# 02a has no single frequency band (it uses per-event-type FREQ_RANGES, since
# it already knows the catalog type) — continuous scanning doesn't have that
# luxury, so PRIMARY_FREQ_MIN/MAX reuse the generic 1-20 Hz band already used
# elsewhere for continuous scanning (02b, and the coincidence check above).
PRIMARY_STA_S     = 5      # 02a: STA_S
PRIMARY_LTA_S     = 100    # 02a: LTA_S
PRIMARY_THR_ON    = 2.0    # 02a: THRES_ON
PRIMARY_THR_OFF   = 1.3    # 02a: THRES_OFF
PRIMARY_FREQ_MIN  = 1.0
PRIMARY_FREQ_MAX  = 20.0
MIN_DET_DUR_S     = 2.0    # discard degenerate near-zero-length triggers

# -- Day-trace handling -------------------------------------------------------
MIN_TRACE_SEC          = max(120.0, PRIMARY_LTA_S + PRIMARY_STA_S)  # a segment shorter than this can't run STA/LTA meaningfully
PAD_SEC                = 5     # feature-extraction padding around each detected window, same convention as 04a
MAX_CANDIDATES_PER_DAY = 5     # cap how many triggers from ONE day/station are kept, so a single noisy day can't dominate the class

# -- How many noise windows to collect ---------------------------------------
# Target = min(number of earthquake rows in EQ_COUNT_SOURCE_CSV, N_NOISE_WINDOWS_CAP)
# Set EQ_COUNT_SOURCE_CSV to None to just use N_NOISE_WINDOWS_CAP directly.
EQ_COUNT_SOURCE_CSV               = None
APPLY_GATE_TO_EQ_COUNT            = True
SNR_MIN_FOR_EQ_COUNT              = 1.70
SNR_FULL_MEDIAN_MIN_FOR_EQ_COUNT  = 1.99
N_NOISE_WINDOWS_CAP               = 10000

# -- Random station selection --------------------------------------------------
STATION_WEIGHT_BY_AVAILABILITY = False

# MAX_TOTAL_ATTEMPTS = target * this. An "attempt" is now a (station, day)
# draw rather than a single-window draw — each successful day can yield up to
# MAX_CANDIDATES_PER_DAY rows, so this can be much lower than before, but each
# attempt is also far heavier (a full day of waveform + response removal +
# STA/LTA per draw, vs. one short window before) — see the runtime note below.
ATTEMPTS_MULTIPLIER = 30

# -- Feature extraction -------------------------------------------------------
LOAD_3C = True

# -- Reproducibility -----------------------------------------------------------
RANDOM_SEED = 42

# -- Checkpoint ------------------------------------------------------------
CHECKPOINT_EVERY = 100



# =============================================================================
# SECTION 2 — SETUP
# =============================================================================

import os
import sys
import bisect
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth

from catalog_helpers import query_catalog_chunked, build_station_list_from_inventory
from preprocessing import preprocess_day
from run_setup import (
    create_run_dir, setup_logging,
    connect_sds, connect_fdsn, fetch_inventory,
)
from features import (
    FEATURE_NAMES, FEATURE_NAMES_3C, N_FEATURES_1C, N_FEATURES_3C,
    extract_features,
)
from detection import run_sta_lta

_FEAT_NAMES = FEATURE_NAMES_3C if LOAD_3C else FEATURE_NAMES

RUN_DIR, _RUN_STAMP = create_run_dir(OUTPUT_DIR)
_log_file, _log_filename = setup_logging(
    RUN_DIR, "04d_noise_window_extraction.py",
    extra_info=(f"Bounding box: lat[{LAT_MIN},{LAT_MAX}] lon[{LON_MIN},{LON_MAX}]  |  "
                f"Period: {T_START} -> {T_END}  |  Primary detector: classical STA/LTA "
                f"(STA={PRIMARY_STA_S}s LTA={PRIMARY_LTA_S}s ON={PRIMARY_THR_ON} OFF={PRIMARY_THR_OFF}, "
                f"same as 02a)  |  NEIGHBOR_RADIUS_KM={NEIGHBOR_RADIUS_KM}")
)

client_sds  = connect_sds(SDS_ROOT)
client_fdsn = connect_fdsn(ISTERRE_URL)
if client_sds is None or client_fdsn is None:
    print("[ERROR] Cannot proceed without SDS and FDSN. Exiting.")
    sys.exit(1)


# ---- Resolve every candidate station in the bounding box -------------------
print(f"\n[SETUP] Fetching inventory for lat[{LAT_MIN},{LAT_MAX}] lon[{LON_MIN},{LON_MAX}] ...")
inventory = fetch_inventory(client_fdsn, T_START, T_END,
                            lat_min=LAT_MIN, lat_max=LAT_MAX,
                            lon_min=LON_MIN, lon_max=LON_MAX)
if inventory is None:
    print("[ERROR] Could not fetch inventory. Exiting.")
    sys.exit(1)

candidates = build_station_list_from_inventory(inventory)   # [(net, sta, loc, chan), ...]
if not candidates:
    print("[ERROR] No vertical-component stations found in this bounding box. Exiting.")
    sys.exit(1)

_t0_cfg, _t1_cfg = UTCDateTime(T_START), UTCDateTime(T_END)


def _operational_periods(net, sta, loc, chan):
    """ List of (UTCDateTime, UTCDateTime) epochs this exact channel was live, clipped to [T_START, T_END]. """
    periods = []
    for network in inventory:
        if network.code != net:
            continue
        for station in network:
            if station.code != sta:
                continue
            for channel in station:
                if channel.code != chan or channel.location_code != loc:
                    continue
                ep_start = max(UTCDateTime(channel.start_date or _t0_cfg), _t0_cfg)
                ep_end   = min(UTCDateTime(channel.end_date   or _t1_cfg), _t1_cfg)
                if ep_end > ep_start:
                    periods.append((ep_start, ep_end))
    return periods


station_channel = {}   # (net, sta) -> (loc, chan)
station_periods = {}   # (net, sta) -> [(t0, t1), ...]
station_coords  = {}   # (net, sta) -> (lat, lon)

for net, sta, loc, chan in candidates:
    periods = _operational_periods(net, sta, loc, chan)
    if not periods:
        continue
    station_channel[(net, sta)] = (loc, chan)
    station_periods[(net, sta)] = periods
    try:
        coords = inventory.get_coordinates(f"{net}.{sta}.{loc}.{chan}", periods[-1][1] - 1)
        station_coords[(net, sta)] = (coords["latitude"], coords["longitude"])
    except Exception:
        station_coords[(net, sta)] = (np.nan, np.nan)

if not station_periods:
    print("[ERROR] No station has a usable operational epoch in the requested period. Exiting.")
    sys.exit(1)

station_keys = list(station_periods.keys())
print(f"[OK] {len(station_keys)} station(s) resolved in the bounding box:")
for net, sta in station_keys:
    loc, chan = station_channel[(net, sta)]
    n_days = sum(p1 - p0 for p0, p1 in station_periods[(net, sta)]) / 86400
    print(f"    {net}.{sta}.{loc}.{chan}  —  {n_days:.0f} days available")

if STATION_WEIGHT_BY_AVAILABILITY:
    _sta_days = np.array([sum(p1 - p0 for p0, p1 in station_periods[k]) for k in station_keys])
    station_weights = _sta_days / _sta_days.sum()
else:
    station_weights = np.full(len(station_keys), 1.0 / len(station_keys))



# =============================================================================
# SECTION 3 — CATALOG EXCLUSION ZONES (ALL event types, not just the 3 classes)
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 1 — Building catalog exclusion zones")
print(f"{'='*65}")

all_events = query_catalog_chunked(
    client_fdsn,
    T_START, T_END,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    target_types = None,     # keep EVERY event type -> noise must avoid all of them
    chunk_days   = CHUNK_DAYS,
    cache_path   = CATALOG_CACHE_FILE,
)
print(f"[OK] {len(all_events)} catalog events of any type found in the bounding box.")

_raw_intervals = []
for ev in all_events:
    origin = ev.preferred_origin() or ev.origins[0]
    t0 = origin.time - EXCLUSION_BUFFER_S
    t1 = origin.time + EXCLUSION_BUFFER_S
    _raw_intervals.append((t0, t1))
_raw_intervals.sort(key=lambda iv: iv[0])

excl_intervals = []
for t0, t1 in _raw_intervals:
    if excl_intervals and t0 <= excl_intervals[-1][1]:
        excl_intervals[-1] = (excl_intervals[-1][0], max(excl_intervals[-1][1], t1))
    else:
        excl_intervals.append((t0, t1))
_excl_starts = [iv[0] for iv in excl_intervals]

print(f"[OK] {len(excl_intervals)} merged exclusion interval(s) "
      f"(±{EXCLUSION_BUFFER_S:.0f}s around every event).")


def overlaps_exclusion(t0, t1):
    """ True if [t0, t1] intersects any merged exclusion interval. """
    idx = bisect.bisect_right(_excl_starts, t1)
    for iv in excl_intervals[max(0, idx - 2):idx + 1]:
        if iv[0] < t1 and t0 < iv[1]:
            return True
    return False



# =============================================================================
# SECTION 4 — CLASS SIZE TARGET
# =============================================================================

rng = np.random.default_rng(RANDOM_SEED)

print(f"\n{'='*65}")
print("  STEP 2 — Noise class size target")
print(f"{'='*65}")

if EQ_COUNT_SOURCE_CSV and os.path.isfile(EQ_COUNT_SOURCE_CSV):
    _df_eq = pd.read_csv(EQ_COUNT_SOURCE_CSV, low_memory=False)
    _df_eq = _df_eq[_df_eq["event_type"] == "earthquake"]
    if APPLY_GATE_TO_EQ_COUNT and {"SNR", "SNR_full_median"}.issubset(_df_eq.columns):
        _df_eq = _df_eq[
            (_df_eq["SNR"]             >= SNR_MIN_FOR_EQ_COUNT) &
            (_df_eq["SNR_full_median"] >= SNR_FULL_MEDIAN_MIN_FOR_EQ_COUNT)
        ]
    eq_count = len(_df_eq)
    N_NOISE_WINDOWS = min(eq_count, N_NOISE_WINDOWS_CAP)
    print(f"[OK] {eq_count:,} earthquake rows found in {os.path.basename(EQ_COUNT_SOURCE_CSV)} "
          f"({'gate applied' if APPLY_GATE_TO_EQ_COUNT else 'no gate'}).")
    if eq_count > N_NOISE_WINDOWS_CAP:
        print(f"[INFO] Capped at N_NOISE_WINDOWS_CAP={N_NOISE_WINDOWS_CAP}.")
else:
    N_NOISE_WINDOWS = N_NOISE_WINDOWS_CAP
    print(f"[INFO] No EQ_COUNT_SOURCE_CSV — using N_NOISE_WINDOWS_CAP={N_NOISE_WINDOWS_CAP} directly.")

MAX_TOTAL_ATTEMPTS = N_NOISE_WINDOWS * ATTEMPTS_MULTIPLIER
print(f"[OK] Target: {N_NOISE_WINDOWS} noise windows  (never more than the earthquake class)")
print(f"[OK] Up to {MAX_TOTAL_ATTEMPTS} (station, day) draws allowed "
      f"(each successful day can yield up to {MAX_CANDIDATES_PER_DAY} rows).")


def draw_random_day(net, sta):
    """ Pick a random UTC day (truncated to 00:00:00) inside this station's operational epochs. """
    periods = station_periods[(net, sta)]
    durs    = np.array([p1 - p0 for p0, p1 in periods])
    p_idx   = rng.choice(len(periods), p=durs / durs.sum())
    p0, p1  = periods[p_idx]
    t       = p0 + rng.uniform(0, p1 - p0)
    day_start = UTCDateTime(t.year, t.month, t.day)
    return day_start



# =============================================================================
# SECTION 5 — 3-COMPONENT FETCH HELPER  (adapted from 04a's _fetch_3c_array)
# =============================================================================

def _fetch_3c_array(client_sds, net, sta, loc, chan_z, t0, t1, z_data, fs):
    """ Fetch N/E channels and stack with the already-loaded Z data -> (3, n). """
    base    = chan_z[:-1]
    n       = len(z_data)
    h_pairs = [("N", "E"), ("1", "2")]
    data_n = data_e = None

    for suf_n, suf_e in h_pairs:
        for suf, which in [(suf_n, "N"), (suf_e, "E")]:
            if which == "N" and data_n is not None:
                continue
            if which == "E" and data_e is not None:
                continue
            try:
                st_h = client_sds.get_waveforms(net, sta, loc, base + suf, t0, t1)
                if not st_h or len(st_h[0].data) == 0:
                    continue
                tr_h = st_h[0].copy()
                tr_h.detrend("demean")
                if abs(tr_h.stats.sampling_rate - fs) > 1:
                    tr_h.resample(fs)
                d = tr_h.data[:n].astype(float)
                if len(d) < n:
                    d = np.pad(d, (0, n - len(d)))
                if which == "N":
                    data_n = d
                else:
                    data_e = d
            except Exception:
                continue
        if data_n is not None and data_e is not None:
            break

    if data_n is None or data_e is None:
        return None
    return np.stack([z_data.astype(float), data_n, data_e])



# =============================================================================
# SECTION 6 — CROSS-STATION LOCALITY CHECK  (unchanged from the previous version)
# =============================================================================

_neighbor_cache = {}


def _neighbors_of(net, sta):
    if (net, sta) in _neighbor_cache:
        return _neighbor_cache[(net, sta)]
    lat0, lon0 = station_coords.get((net, sta), (np.nan, np.nan))
    result = []
    if not (np.isnan(lat0) or np.isnan(lon0)):
        for (n2, s2), (lat, lon) in station_coords.items():
            if (n2, s2) == (net, sta) or np.isnan(lat) or np.isnan(lon):
                continue
            try:
                dist_km = gps2dist_azimuth(lat0, lon0, lat, lon)[0] / 1000.0
            except Exception:
                continue
            if dist_km <= NEIGHBOR_RADIUS_KM:
                result.append((n2, s2))
    _neighbor_cache[(net, sta)] = result
    return result


def is_confirmed_local(net, sta, t_start_w, t_end_w):
    """
    True  -> either no neighbor could be checked (assumed OK, logged elsewhere)
             or every checked neighbor stayed quiet during the window
    False -> at least one neighbor also triggered -> looks like a real,
             un-catalogued, network-wide signal, not station-local noise
    """
    neighbors = _neighbors_of(net, sta)
    if len(neighbors) < MIN_NEIGHBORS_REQUIRED:
        return True, 0

    warmup  = COINCIDENCE_LTA_S + COINCIDENCE_STA_S
    checked = 0
    for n2, s2 in neighbors:
        loc2, chan2 = station_channel[(n2, s2)]
        try:
            st2 = client_sds.get_waveforms(n2, s2, loc2, chan2, t_start_w - warmup, t_end_w)
        except Exception:
            continue
        if len(st2) != 1:
            continue
        tr2 = st2[0]
        if (tr2.stats.endtime - tr2.stats.starttime) < 0.9 * warmup:
            continue
        try:
            tr2.detrend("demean")
            nyq2 = tr2.stats.sampling_rate / 2
            tr2.filter("bandpass", freqmin=COINCIDENCE_FREQ_MIN,
                      freqmax=min(COINCIDENCE_FREQ_MAX, 0.9 * nyq2),
                      corners=4, zerophase=True)
            _, on_off2 = run_sta_lta(tr2, COINCIDENCE_STA_S, COINCIDENCE_LTA_S,
                                     COINCIDENCE_THR_ON, COINCIDENCE_THR_OFF)
        except Exception:
            continue
        checked += 1
        if len(on_off2) > 0:
            return False, checked
    return True, checked



# =============================================================================
# SECTION 7 — MAIN SAMPLING LOOP
# =============================================================================

print(f"\n{'='*65}")
print(f"  STEP 3 — Sampling noise windows  (target: {N_NOISE_WINDOWS})")
print(f"{'='*65}")

all_rows          = []
used_windows      = set()   # (net, sta, str(t_on)) -> guards against a re-drawn day producing duplicate rows
n_attempts        = 0
n_days_no_trigger = 0
n_rej_nodata      = 0
n_rej_short       = 0
n_rej_response    = 0
n_rej_excluded    = 0
n_rej_coincidence = 0
n_rej_feat        = 0
n_uncheckable     = 0
n_rej_merge       = 0
n_rej_unexpected  = 0

while len(all_rows) < N_NOISE_WINDOWS and n_attempts < MAX_TOTAL_ATTEMPTS:
    n_attempts += 1

    s_idx    = rng.choice(len(station_keys), p=station_weights)
    net, sta = station_keys[s_idx]
    loc, chan = station_channel[(net, sta)]

    day_start = draw_random_day(net, sta)
    day_end   = day_start + 86400

    # ---- Load the full day, response-removed, bandpass-filtered -----------
    try:
        st_raw = client_sds.get_waveforms(net, sta, loc, chan, day_start, day_end)
    except Exception:
        n_rej_nodata += 1
        continue
    if len(st_raw) == 0:
        n_rej_nodata += 1
        continue

    # ---- Merge + split, defensively -----------------------------------------
    # Two independent real-world failure modes have shown up running this on
    # the cluster, neither of which should ever be allowed to kill a 48h job:
    #  1. An internal data gap makes merge(fill_value=None) represent the
    #     Trace as a numpy MASKED array -> remove_response() can't run on
    #     masked data ("Trace with masked values found"). Fixed by split()
    #     right after merging (ObsPy's own suggested fix): breaks a masked
    #     trace back into its contiguous unmasked pieces at the gap
    #     boundaries; the per-segment loop below already handles multiple
    #     segments per day.
    #  2. A station/day whose SDS segments were written with different dtypes
    #     (e.g. int32 STEIM-compressed vs float32/64 — happens across an
    #     11-year, multi-instrument archive) makes ObsPy refuse to merge at
    #     all: "Can't merge traces with same ids but differing data types" —
    #     this raised from BOTH the fill_value=None try AND the fill_value=0
    #     except-branch, so it was propagating out uncaught and crashing the
    #     whole run (this is what killed run 3887087). Fixed by normalizing
    #     every trace to float64 before merging, which recovers the day's
    #     data instead of losing it.
    try:
        for _tr in st_raw:
            if _tr.data.dtype != np.float64:
                _tr.data = _tr.data.astype(np.float64)
        try:
            st_raw.merge(fill_value=None)
        except Exception:
            st_raw.merge(fill_value=0)
        st_raw = st_raw.split()
    except Exception as e:
        n_rej_merge += 1
        if n_rej_merge <= 20:
            print(f"      [WARN] Merge failed for {net}.{sta} {day_start.date}: {e}")
        continue

    accepted_today = 0

    # Everything below processes ONE (station, day) draw. Wrapped in a
    # catch-all so any further unforeseen data quirk (there will always be
    # something new across 45 stations x 11 years) skips just this draw
    # instead of taking down an unattended multi-day cluster job.
    try:
        for tr_raw in st_raw:
            if accepted_today >= MAX_CANDIDATES_PER_DAY:
                break

            seg_dur = tr_raw.stats.endtime - tr_raw.stats.starttime
            if seg_dur < MIN_TRACE_SEC:
                n_rej_short += 1
                continue

            tr_vel = preprocess_day(tr_raw, inventory)
            if tr_vel is None:
                n_rej_response += 1
                continue

            fs = tr_vel.stats.sampling_rate
            tr_filt = tr_vel.copy()
            nyq = fs / 2
            tr_filt.filter("bandpass", freqmin=PRIMARY_FREQ_MIN,
                           freqmax=min(PRIMARY_FREQ_MAX, 0.9 * nyq),
                           corners=4, zerophase=True)

            # ---- Primary detector: classical STA/LTA, same as 02a ---------
            try:
                cft, on_off = run_sta_lta(tr_filt, PRIMARY_STA_S, PRIMARY_LTA_S,
                                          PRIMARY_THR_ON, PRIMARY_THR_OFF)
            except Exception:
                continue

            if len(on_off) == 0:
                n_days_no_trigger += 1
                continue

            # Shuffle so capping at MAX_CANDIDATES_PER_DAY doesn't always keep
            # only the earliest-in-day triggers.
            order = rng.permutation(len(on_off))

            for k in order:
                if accepted_today >= MAX_CANDIDATES_PER_DAY or len(all_rows) >= N_NOISE_WINDOWS:
                    break

                i_on, i_off = on_off[k]
                t_on  = tr_filt.stats.starttime + i_on / fs
                t_off = tr_filt.stats.starttime + i_off / fs
                if (t_off - t_on) < MIN_DET_DUR_S:
                    continue

                win_key = (net, sta, str(t_on))
                if win_key in used_windows:
                    continue

                # ---- Catalog exclusion (this permissive detector WILL re-find
                # real regional earthquakes too — filter those back out) ----
                if overlaps_exclusion(t_on, t_off):
                    n_rej_excluded += 1
                    continue

                # ---- Cross-station locality check --------------------------
                local_ok, n_checked = is_confirmed_local(net, sta, t_on, t_off)
                if not local_ok:
                    n_rej_coincidence += 1
                    continue
                if n_checked == 0:
                    n_uncheckable += 1

                # ---- Feature extraction window: PAD_SEC padding, like 04a --
                t_cut_on  = max(t_on  - PAD_SEC, tr_vel.stats.starttime)
                t_cut_off = min(t_off + PAD_SEC, tr_vel.stats.endtime)
                tr_cut    = tr_vel.slice(t_cut_on, t_cut_off)
                if tr_cut.stats.npts < 10:
                    continue

                data_3c = None
                if LOAD_3C:
                    data_3c = _fetch_3c_array(client_sds, net, sta, loc, chan,
                                              t_cut_on, t_cut_off, tr_cut.data, fs)

                feats = extract_features(tr_cut.data, fs, data_3c=data_3c)
                if np.all(np.isnan(feats)):
                    n_rej_feat += 1
                    continue

                lat, lon = station_coords.get((net, sta), (np.nan, np.nan))

                row = {
                    "event_time"        : str(t_on),
                    "event_type"        : "noise",
                    "catalog_lat"       : lat,
                    "catalog_lon"       : lon,
                    "catalog_depth_km"  : np.nan,
                    "network"           : net,
                    "station"           : sta,
                    "channel"           : chan,
                    "det_starttime"     : str(t_on),
                    "det_starttime_raw" : str(t_on),
                    "onset_refine_s"    : 0.0,
                    "det_endtime"       : str(t_off),
                    "det_duration_s"    : round(t_off - t_on, 2),   # REAL STA/LTA-measured duration, not bootstrapped
                    "trigger_on_cft"    : round(float(cft[i_on]),  4) if i_on  < len(cft) else np.nan,
                    "trigger_off_cft"   : round(float(cft[i_off]), 4) if i_off < len(cft) else np.nan,
                    "origin_inside_det" : None,
                    "origin_lag_s"      : np.nan,
                    "pick_inside_det"   : None,
                    "pick_lag_s"        : np.nan,
                    # Same convention as before: no SNR-based quality question
                    # applies to a catalog-clear, locality-confirmed background
                    # window -> always kept.
                    "quality_ok"        : True,
                    "SNR"               : np.nan,
                    "SNR_picking_5_5"   : np.nan,
                    "SNR_picking_3_3"   : np.nan,
                    "SNR_picking_1_3"   : np.nan,
                    "SNR_full_mean"     : np.nan,
                    "SNR_full_median"   : np.nan,
                    "SNR_s2n_median"    : np.nan,
                }
                for fname, fval in zip(_FEAT_NAMES, feats):
                    row[fname] = fval
                all_rows.append(row)
                used_windows.add(win_key)
                accepted_today += 1

                if len(all_rows) % 100 == 0:
                    print(f"  ... {len(all_rows):5d}/{N_NOISE_WINDOWS} accepted "
                          f"({n_attempts} day-draws, "
                          f"{100*len(all_rows)/max(n_attempts,1):.1f} rows/draw so far)")

                if CHECKPOINT_EVERY > 0 and len(all_rows) % CHECKPOINT_EVERY == 0:
                    pd.DataFrame(all_rows).to_csv(
                        os.path.join(RUN_DIR, f"noise_windows_checkpoint_{len(all_rows)}.csv"), index=False)

            if accepted_today >= MAX_CANDIDATES_PER_DAY or len(all_rows) >= N_NOISE_WINDOWS:
                break
    except Exception as e:
        n_rej_unexpected += 1
        if n_rej_unexpected <= 20:
            print(f"      [WARN] Unexpected error processing {net}.{sta} {day_start.date}: {e}")
        continue



# =============================================================================
# SECTION 8 — SAVE CSV + SUMMARY
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 4 — Save + summary")
print(f"{'='*65}")

if not all_rows:
    print("\n[WARN] No noise windows extracted — CSV will not be written.")
else:
    df = pd.DataFrame(all_rows)

    meta_cols = [
        "event_time", "event_type", "catalog_lat", "catalog_lon",
        "catalog_depth_km", "network", "station", "channel",
        "det_starttime", "det_starttime_raw", "onset_refine_s",
        "det_endtime", "det_duration_s",
        "trigger_on_cft", "trigger_off_cft",
        "origin_inside_det", "origin_lag_s",
        "pick_inside_det", "pick_lag_s", "quality_ok",
        "SNR", "SNR_picking_5_5", "SNR_picking_3_3",
        "SNR_picking_1_3", "SNR_full_mean", "SNR_full_median", "SNR_s2n_median",
    ]
    ordered_cols = meta_cols + _FEAT_NAMES
    df = df[[c for c in ordered_cols if c in df.columns]]

    csv_path = os.path.join(RUN_DIR, f"noise_windows_{_RUN_STAMP}.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n[SAVED] {csv_path}")
    print(f"        {df.shape[0]} rows x {df.shape[1]} columns  "
          f"[{len(_FEAT_NAMES)} features, 3C={'on' if LOAD_3C else 'off'}]")
    print(f"        Duration: real STA/LTA-measured (median {df['det_duration_s'].median():.1f}s, "
          f"range [{df['det_duration_s'].min():.1f}, {df['det_duration_s'].max():.1f}]s)")

    print(f"\n  Rows per station:")
    for (net, sta), n in df.groupby(["network", "station"]).size().items():
        print(f"    {net}.{sta:<8} {n:5d}")


print(f"\n  (Station, day) draws     : {n_attempts} / {MAX_TOTAL_ATTEMPTS}")
print(f"  Accepted                 : {len(all_rows)} / {N_NOISE_WINDOWS}")
print(f"  Days/segments with no STA/LTA trigger : {n_days_no_trigger}")
print(f"  Rejected — no data       : {n_rej_nodata}")
print(f"  Rejected — merge failed  : {n_rej_merge}  (dtype mismatch or other unmergeable segments)")
print(f"  Rejected — segment short : {n_rej_short}")
print(f"  Rejected — response      : {n_rej_response}")
print(f"  Rejected — unexpected    : {n_rej_unexpected}  (caught, logged above, run continued)")
print(f"  Rejected — catalog excl. : {n_rej_excluded}  (re-found a real cataloged event)")
print(f"  Rejected — coincidence   : {n_rej_coincidence}  (real signal seen on a neighbor station)")
print(f"  Rejected — feat failed   : {n_rej_feat}")
print(f"  Accepted but unverified  : {n_uncheckable}  (fewer than {MIN_NEIGHBORS_REQUIRED} neighbor(s) within {NEIGHBOR_RADIUS_KM} km)")

if len(all_rows) < N_NOISE_WINDOWS:
    print(f"\n[WARN] Stopped after MAX_TOTAL_ATTEMPTS without reaching the target count.")
    print(f"       Consider raising ATTEMPTS_MULTIPLIER or MAX_CANDIDATES_PER_DAY,")
    print(f"       widening the bounding box / date range, or lowering PRIMARY_THR_ON")
    print(f"       (more sensitive -> more triggers per day).")



# =============================================================================
# END
# =============================================================================

from datetime import datetime
print("\n" + "=" * 70)
print(f"  Run finished : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Stations     : {len(station_keys)}")
print(f"  Accepted     : {len(all_rows)}")
print(f"  All outputs  : {RUN_DIR}")
print(f"  Log file     : {_log_filename}")
print("=" * 70)

_log_file.close()
