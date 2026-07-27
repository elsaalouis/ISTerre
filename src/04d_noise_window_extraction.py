"""
04d_noise_window_extraction.py
================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : July 2026

Goal
----
Build the 4th classification class: pure background NOISE (no cataloged seismic event of any kind, and no un-catalogued regional signal either)
Random time windows are drawn from ALL the stations in and around the Mont Blanc massif, one random station + one random time per draw so the class represents the whole network

"Local" means confirmed local, not single-station-only
--------------------------------------------------------
A window is only accepted if it looks like noise local to the ONE station it was drawn from. Concretely:
  1. Reject any window within EXCLUSION_BUFFER_S of a CATALOGED event (any type)
  2. For the surviving candidates, run a quick classical STA/LTA on every OTHER station within NEIGHBOR_RADIUS_KM of the drawn station, over the same time window 
     -> reject if ANY neighbor also triggers (coincident, uncatalogued signal, not station-local noise)

Why window duration is bootstrapped, not fixed
-----------------------------------------------
`duration` (feature #1 — see seismic_params.py) is the most important RF/HGB feature (see 06a/06c results) 
 -> if every noise window had the same fixed length, the classifier could learn to separate "noise" from real events purely on window length 
 -> to avoid that, each noise window's duration is drawn from the empirical distribution of real feature-extraction windows
(`det_duration_s + 2*PAD_SEC` from a 04a `catalog_windows_<stamp>.csv`), the same window structure event rows use in their `extract_features()` call

Pipeline
--------
  1. Fetch the FDSN inventory for the whole bounding box -> resolve every candidate station's exact channel + operational epochs + coordinates
  2. Query the FULL catalog (all event types, no filter) over the same period and bounding box 
     -> build a merged list of "exclusion intervals" (event origin time +/- EXCLUSION_BUFFER_S)
  3. Repeatedly: pick a random station, draw a random start time (weighted by that station's operational epochs) + a random duration 
     -> reject if it overlaps an exclusion interval, has no data, has a gap, or is too short
  4. Coincidence check: reject if any nearby station triggers during the same window (see above)
  5. Remove instrument response -> velocity [m/s], optionally fetch horizontals for 3C polarization features, extract the 99/103 Maggi/ Hibert features exactly like 04a does for real events
  6. Save a CSV with the SAME column layout as `catalog_windows_<stamp>.csv`
     (event_type = "noise", SNR/quality columns not applicable -> NaN /always-pass) so it can be loaded and concatenated directly by 06a/06b/06c

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
# Must match the box used everywhere else in the pipeline (01/02a/03a/04a) so "the network" means the same thing here as it does downstream
LAT_MIN, LAT_MAX = 45.5, 46.0
LON_MIN, LON_MAX = 6.5, 7.2

# -- Chunked catalog query (avoids FDSN server timeout on a 10+ year span) ---
CHUNK_DAYS = 90
CATALOG_CACHE_FILE = "/data/failles/louisels/project/results/catalog_cache_all_types.xml"

# -- Exclusion margin kept clear around every CATALOGED event ---------------
# 04a uses PRE_EVENT=150s / POST_EVENT=90s to window real events; 600s gives a comfortable margin 
EXCLUSION_BUFFER_S = 600.0

# -- Cross-station locality check (catches real but UN-catalogued signals) --
NEIGHBOR_RADIUS_KM      = 15.0   # stations within this radius are checked for a coincident trigger
MIN_NEIGHBORS_REQUIRED  = 1      # fewer neighbors within radius -> can't verify, skip the check for that station (logged, not rejected)
COINCIDENCE_STA_S       = 2.0
COINCIDENCE_LTA_S       = 30.0
COINCIDENCE_THR_ON      = 3.5    # classical STA/LTA ratio above this on ANY neighbor -> reject as a coincident (real) signal
COINCIDENCE_THR_OFF     = 1.5
COINCIDENCE_FREQ_MIN    = 1.0
COINCIDENCE_FREQ_MAX    = 20.0

# -- Noise window duration ----------------------------------------------------
# Set to a 04a `catalog_windows_<stamp>.csv` to bootstrap durations from real event windows 
DURATION_SOURCE_CSV = "/data/failles/louisels/project/results/outputs_04a/all-99-features-recent+3C/catalog_windows_20260708_174019.csv"
PAD_SEC              = 5      # must match 04a's PAD_SEC if DURATION_SOURCE_CSV is set
DUR_MIN_S, DUR_MAX_S = 10.0, 120.0   # fallback range, only used without a source CSV

# -- How many noise windows to collect ---------------------------------------
# Target = min(number of earthquake rows in EQ_COUNT_SOURCE_CSV, N_NOISE_WINDOWS_CAP)
# Set EQ_COUNT_SOURCE_CSV to None to just use N_NOISE_WINDOWS_CAP directly
EQ_COUNT_SOURCE_CSV               = None   # a 04a catalog_windows_<stamp>.csv (often same file as DURATION_SOURCE_CSV)
APPLY_GATE_TO_EQ_COUNT            = True    # count only EQ rows that would actually survive 06a/06c's quality gate
SNR_MIN_FOR_EQ_COUNT              = 1.70    # 05b Tier 2 — mirrors 06a/06c's gate
SNR_FULL_MEDIAN_MIN_FOR_EQ_COUNT  = 1.99
N_NOISE_WINDOWS_CAP               = 10000    # hard ceiling regardless of EQ count — script much slower per-window than 04a (multi-neighbor coincidence check)

# -- Random station selection --------------------------------------------------
# False -> every station gets an equal share of draws, regardless of how many years of archive it has
# True -> stations with longer operational history get proportionally more draws.
STATION_WEIGHT_BY_AVAILABILITY = False

ATTEMPTS_MULTIPLIER = 80   # MAX_TOTAL_ATTEMPTS = target * this (most draws miss: gaps / exclusion / coincidence / no data)

# -- Feature extraction -------------------------------------------------------
LOAD_3C = True   # fetch N/E channels too -> 103 features (99 Z + 4 polarization)

# -- Reproducibility -----------------------------------------------------------
RANDOM_SEED = 42

# -- Checkpoint ------------------------------------------------------------
CHECKPOINT_EVERY = 100   # save a partial CSV every N accepted windows (0 = disabled)



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
from preprocessing import build_station_times_df, remove_response_or_fallback
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
                f"Period: {T_START} -> {T_END}  |  EXCLUSION_BUFFER_S={EXCLUSION_BUFFER_S}  |  "
                f"NEIGHBOR_RADIUS_KM={NEIGHBOR_RADIUS_KM}")
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

# Merge overlapping/adjacent intervals so the overlap check below stays cheap
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
# SECTION 4 — NOISE WINDOW DURATION DISTRIBUTION + CLASS SIZE TARGET
# =============================================================================

rng = np.random.default_rng(RANDOM_SEED)

duration_pool = None
if DURATION_SOURCE_CSV and os.path.isfile(DURATION_SOURCE_CSV):
    _df_dur = pd.read_csv(DURATION_SOURCE_CSV, usecols=["det_duration_s"])
    duration_pool = (_df_dur["det_duration_s"].dropna().values + 2 * PAD_SEC)
    duration_pool = duration_pool[duration_pool > 1]
    print(f"\n[OK] Duration bootstrap pool: {len(duration_pool)} real event windows "
          f"from {os.path.basename(DURATION_SOURCE_CSV)} "
          f"(median {np.median(duration_pool):.1f}s, range "
          f"[{duration_pool.min():.1f}, {duration_pool.max():.1f}]s).")
else:
    print(f"\n[INFO] No DURATION_SOURCE_CSV — using uniform fallback "
          f"[{DUR_MIN_S}, {DUR_MAX_S}]s. Set DURATION_SOURCE_CSV to a 04a output "
          f"for realistic (non-fixed) durations.")


def draw_duration():
    if duration_pool is not None and len(duration_pool) > 0:
        return float(rng.choice(duration_pool))
    return float(rng.uniform(DUR_MIN_S, DUR_MAX_S))


def draw_random_start(net, sta):
    periods = station_periods[(net, sta)]
    durs    = np.array([p1 - p0 for p0, p1 in periods])
    p_idx   = rng.choice(len(periods), p=durs / durs.sum())
    p0, p1  = periods[p_idx]
    return p0 + rng.uniform(0, p1 - p0)


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
        print(f"[INFO] Capped at N_NOISE_WINDOWS_CAP={N_NOISE_WINDOWS_CAP} "
              f"(raise the cap to approach the full earthquake count — this script "
              f"is much slower per-window than 04a).")
else:
    N_NOISE_WINDOWS = N_NOISE_WINDOWS_CAP
    print(f"[INFO] No EQ_COUNT_SOURCE_CSV — using N_NOISE_WINDOWS_CAP={N_NOISE_WINDOWS_CAP} directly.")

MAX_TOTAL_ATTEMPTS = N_NOISE_WINDOWS * ATTEMPTS_MULTIPLIER
print(f"[OK] Target: {N_NOISE_WINDOWS} noise windows  (never more than the earthquake class)")



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
# SECTION 6 — CROSS-STATION LOCALITY CHECK
# =============================================================================
# A candidate window survives the catalog exclusion but might still be a real,
# un-catalogued regional signal. Confirm it is genuinely LOCAL to the drawn
# station by checking that no other nearby station triggers at the same time.

_neighbor_cache = {}   # (net, sta) -> list of (net, sta) within NEIGHBOR_RADIUS_KM


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
n_attempts        = 0
n_rej_excluded    = 0
n_rej_nodata      = 0
n_rej_gap         = 0
n_rej_response    = 0
n_rej_coincidence = 0
n_rej_feat        = 0
n_uncheckable     = 0   # accepted windows where locality couldn't be verified (too few neighbors)

while len(all_rows) < N_NOISE_WINDOWS and n_attempts < MAX_TOTAL_ATTEMPTS:
    n_attempts += 1

    s_idx    = rng.choice(len(station_keys), p=station_weights)
    net, sta = station_keys[s_idx]
    loc, chan = station_channel[(net, sta)]

    wdur      = draw_duration()
    t_start_w = draw_random_start(net, sta)
    t_end_w   = t_start_w + wdur

    if overlaps_exclusion(t_start_w, t_end_w):
        n_rej_excluded += 1
        continue

    # ---- Load primary waveform --------------------------------------------
    try:
        st_raw = client_sds.get_waveforms(net, sta, loc, chan, t_start_w, t_end_w)
    except Exception:
        n_rej_nodata += 1
        continue
    if len(st_raw) == 0:
        n_rej_nodata += 1
        continue
    if len(st_raw) > 1:
        n_rej_gap += 1
        continue

    tr_raw = st_raw[0]
    actual_dur = tr_raw.stats.endtime - tr_raw.stats.starttime
    if actual_dur < 0.95 * wdur:
        n_rej_gap += 1
        continue

    # ---- Cross-station locality check --------------------------------------
    local_ok, n_checked = is_confirmed_local(net, sta, t_start_w, t_end_w)
    if not local_ok:
        n_rej_coincidence += 1
        continue
    if n_checked == 0:
        n_uncheckable += 1

    # ---- Instrument response removal -------------------------------------
    try:
        station_times_df = build_station_times_df(st_raw, t_start_w, t_end_w)
        st_vel = remove_response_or_fallback(st_raw, inventory, station_times_df)
    except Exception:
        n_rej_response += 1
        continue
    if len(st_vel) == 0:
        n_rej_response += 1
        continue

    tr_vel = st_vel[0]
    fs     = tr_vel.stats.sampling_rate

    # ---- Optional 3C polarization features ---------------------------------
    data_3c = None
    if LOAD_3C:
        data_3c = _fetch_3c_array(client_sds, net, sta, loc, chan,
                                  t_start_w, t_end_w, tr_vel.data, fs)

    feats = extract_features(tr_vel.data, fs, data_3c=data_3c)
    if np.all(np.isnan(feats)):
        n_rej_feat += 1
        continue

    lat, lon = station_coords.get((net, sta), (np.nan, np.nan))

    # ---- Build row (schema-compatible with catalog_windows_<stamp>.csv) ---
    row = {
        "event_time"        : str(t_start_w),     # unique id: window start time
        "event_type"        : "noise",
        "catalog_lat"       : lat,
        "catalog_lon"       : lon,
        "catalog_depth_km"  : np.nan,
        "network"           : net,
        "station"           : sta,
        "channel"           : chan,
        "det_starttime"     : str(t_start_w),
        "det_starttime_raw" : str(t_start_w),
        "onset_refine_s"    : 0.0,
        "det_endtime"       : str(t_end_w),
        "det_duration_s"    : round(wdur - (2 * PAD_SEC if duration_pool is not None else 0), 2),
        "trigger_on_cft"    : np.nan,
        "trigger_off_cft"   : np.nan,
        "origin_inside_det" : None,
        "origin_lag_s"      : np.nan,
        "pick_inside_det"   : None,
        "pick_lag_s"        : np.nan,
        # No SNR-based quality question applies to a catalog-clear, locality-
        # confirmed background window by construction -> always kept ('True'),
        # same dtype as 04a's boolean quality_ok column. SNR columns are
        # genuinely not applicable (no on/off signal window to compare
        # against a "noise" reference).
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

    if len(all_rows) % 100 == 0:
        print(f"  ... {len(all_rows):5d}/{N_NOISE_WINDOWS} accepted "
              f"({n_attempts} attempts, {100*len(all_rows)/n_attempts:.1f}% acceptance so far)")

    if CHECKPOINT_EVERY > 0 and len(all_rows) % CHECKPOINT_EVERY == 0:
        pd.DataFrame(all_rows).to_csv(
            os.path.join(RUN_DIR, f"noise_windows_checkpoint_{len(all_rows)}.csv"), index=False)



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
    print(f"        Duration source: "
          f"{'bootstrapped from ' + os.path.basename(DURATION_SOURCE_CSV) if duration_pool is not None else 'uniform fallback'}")

    print(f"\n  Rows per station:")
    for (net, sta), n in df.groupby(["network", "station"]).size().items():
        print(f"    {net}.{sta:<8} {n:5d}")


print(f"\n  Attempts                : {n_attempts} / {MAX_TOTAL_ATTEMPTS}")
print(f"  Accepted                : {len(all_rows)} / {N_NOISE_WINDOWS}")
if n_attempts:
    print(f"  Acceptance rate         : {100*len(all_rows)/n_attempts:.1f}%")
print(f"  Rejected — exclusion    : {n_rej_excluded}")
print(f"  Rejected — no data      : {n_rej_nodata}")
print(f"  Rejected — gap/short    : {n_rej_gap}")
print(f"  Rejected — coincidence  : {n_rej_coincidence}  (real signal seen on a neighbor station)")
print(f"  Rejected — response     : {n_rej_response}")
print(f"  Rejected — feat failed  : {n_rej_feat}")
print(f"  Accepted but unverified : {n_uncheckable}  (fewer than {MIN_NEIGHBORS_REQUIRED} neighbor(s) within {NEIGHBOR_RADIUS_KM} km)")

if len(all_rows) < N_NOISE_WINDOWS:
    print(f"\n[WARN] Stopped after MAX_TOTAL_ATTEMPTS without reaching the target count.")
    print(f"       Consider raising ATTEMPTS_MULTIPLIER, widening the bounding box / date range,")
    print(f"       or relaxing NEIGHBOR_RADIUS_KM / EXCLUSION_BUFFER_S / the noise target.")



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
