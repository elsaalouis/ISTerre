"""
04c_regional_event_extraction.py
==================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : August 2026

Goal
----
Build the 5th classification class: REGIONAL 
Earthquakes ~150-1000 km from the massif, recorded on the SAME stations as the local classes (earthquake,
rockslide, ice quake, noise), but physically and operationally distinct from all of them

Data sources
------------
  Regional catalog     : EMSC FDSN event webservice (obspy shortcut "EMSC" -> https://www.seismicportal.eu/fdsnws/event/1)
  Station inventory    : ISTerre FDSN server (same bounding box as 04d)
  Waveforms            : ISTerre SDS archive

Pipeline
--------
  1. Resolve every candidate station in the bounding box (like 04d)
  2. Query the regional catalog: 150 km <= distance <= 1000 km from the massif centroid, magnitude >= MIN_MAGNITUDE, over the SDS archive's span
  3. (Optional) randomly subsample to N_EVENTS_CAP events for a bounded run
  4. For each (event, station) pair, if the station was operational then:
       a. Predicted arrival = event origin time + obspy.taup travel time
       b. Fetch [predicted_arrival - PRE_ARRIVAL_S, + POST_ARRIVAL_S] from SDS
       c. Remove instrument response -> ground velocity [m/s] 
       d. Classical STA/LTA (1-20 Hz, same band as the local pipeline) to refine the onset/end around the predicted arrival
       e. Compute 7 SNR metrics (same detection.py function as 04a/04d)
       f. Extract 99/103 Maggi/Hibert(+polarization) features from the padded window 
  5. Save a CSV with the SAME base column layout as 04a's catalog_windows / 04d's noise_windows (safe to concatenate)
     + plus regional-specific columns (distance_km, azimuth_deg, magnitude, phase_used, ...)

Output
------
  regional_windows_<stamp>.csv : one row per (event x station) accepted window 
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# -- Paths ----------------------------------------------------------------
SDS_ROOT    = "/data/sig/SDS"
ISTERRE_URL = "http://ist-sc3-geobs.osug.fr:8080"          # station inventory (local FDSN)
OUTPUT_DIR  = "/data/failles/louisels/project/results/outputs_04c"

# -- Regional event catalog source ------------------------------------------
CATALOG_URL = "EMSC"

# -- Physical local/regional/teleseismic split (see docstring) --------------
DIST_MIN_KM    = 150.0     # below this: local Pg/Sg, already covered by the existing classes
DIST_MAX_KM    = 1000.0    # above this: teleseismic (out of scope for this script)
MIN_MAGNITUDE  = 3       # first-pass guess at regional visibility — tunable, see docstring
MAX_MAGNITUDE  = None       # None = no upper bound

# -- Massif reference point (centroid of the bounding box used everywhere else) --
LAT_MIN, LAT_MAX = 45.5, 46.0
LON_MIN, LON_MAX = 6.5, 7.2
MASSIF_CENTER_LAT = (LAT_MIN + LAT_MAX) / 2
MASSIF_CENTER_LON = (LON_MIN + LON_MAX) / 2

# -- Time range to sample from (must match the SDS archive's continuous span) --
T_START = "2015-01-01"
T_END   = "2026-07-01"

# -- Chunked catalog query (avoids FDSN server timeout on a 10+ year span) ---
CHUNK_DAYS          = 90
CATALOG_CACHE_FILE  = "/data/failles/louisels/project/results/regional_catalog_cache.xml"

# -- Bound the run: cap total events processed (event x ~30-45 stations) ----
N_EVENTS_CAP = 10000
RANDOM_SEED  = 42

# -- TauPy predicted-arrival model ------------------------------------------
TAUP_MODEL       = "iasp91"
TAUP_PHASE_LIST  = ["Pg", "P", "Pn"]   # crustal-direct and Moho-refracted phases, see docstring caveat
DEFAULT_DEPTH_KM = 10.0    # used only if the catalog event has no depth

# -- Waveform extraction window (anchored on the PREDICTED arrival) ---------
PRE_ARRIVAL_S  = 45     # [s] before predicted arrival — must be > LTA_S for STA/LTA warm-up
POST_ARRIVAL_S = 150    # [s] after predicted arrival — regional coda is much shorter than
                        # teleseismic; raise if larger regional events need more room

Z_CHANNELS = "??Z"

# -- Detection band: SAME as the local pipeline (1-20 Hz) — regional Pn/Sn
# content overlaps this band much more than teleseismic ever did -----------
FREQ_MIN = 1.0     # Hz
FREQ_MAX = 20.0    # Hz

# -- Onset-refinement classical STA/LTA (search window, not blind scanning) --
STA_S     = 5.0
LTA_S     = 30.0    # must stay < PRE_ARRIVAL_S so there is a warm-up period
THRES_ON  = 2.0     # same as local defaults (02a/04d) — regional SNR should generally be
THRES_OFF = 1.3     # comparable to or better than the near-threshold teleseismic case
MIN_DURAT_S = 2.0

# Tolerance: how far (seconds) a STA/LTA trigger may sit from the predicted
# arrival and still be considered "the same arrival". WIDER than the
# teleseismic-era value (30s) because regional travel times from a coarse
# global crustal model (iasp91) are LESS reliable than teleseismic ones —
# see docstring caveat.
ARRIVAL_MATCH_TOL_S = 20.0

# Fallback window when no STA/LTA trigger matches the predicted arrival —
# kept for inspection/completeness (flagged `detected=False`), not assumed
# to be good training data; downstream SNR gate (05b thresholds) filters it out.
FALLBACK_PRE_S  = 5.0
FALLBACK_POST_S = 60.0

# -- Feature extraction window padding ----------------------------------------
PAD_SEC = 5

# -- Quality flag thresholds — SAME pipeline-wide SNR gate as 03c/03d/06b/06c
# (see plan_detection_algorithm memory, resolved 2026-07-20; informational only,
# 06b/06c always recompute this explicitly rather than trusting the column) ---
SNR_MIN             = 1.70
SNR_FULL_MEDIAN_MIN = 1.99

# -- Feature extraction -------------------------------------------------------
LOAD_3C = True

# -- Checkpoint ---------------------------------------------------------------
CHECKPOINT_EVERY = 50



# =============================================================================
# SECTION 2 — SETUP
# =============================================================================

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth, locations2degrees
from obspy.taup import TauPyModel

from catalog_helpers import (
    query_catalog_by_distance_chunked,
    build_station_list_from_inventory,
)
from preprocessing import preprocess_day
from run_setup import (
    create_run_dir, setup_logging,
    connect_sds, connect_fdsn, fetch_inventory,
)
from features import (
    FEATURE_NAMES, FEATURE_NAMES_3C, N_FEATURES_1C, N_FEATURES_3C,
    extract_features,
)
from detection import run_sta_lta, compute_snr

_FEAT_NAMES = FEATURE_NAMES_3C if LOAD_3C else FEATURE_NAMES

RUN_DIR, _RUN_STAMP = create_run_dir(OUTPUT_DIR)
_log_file, _log_filename = setup_logging(
    RUN_DIR, "04c_regional_event_extraction.py",
    extra_info=(f"Catalog: {CATALOG_URL}  |  "
                f"dist [{DIST_MIN_KM:.0f}, {DIST_MAX_KM:.0f}] km  |  mag >= {MIN_MAGNITUDE}  |  "
                f"Period: {T_START} -> {T_END}  |  "
                f"TauPy model: {TAUP_MODEL}")
)

client_sds     = connect_sds(SDS_ROOT)
client_fdsn    = connect_fdsn(ISTERRE_URL)     # station inventory
client_regional = connect_fdsn(CATALOG_URL)     # regional event catalog
if client_sds is None or client_fdsn is None or client_regional is None:
    print("[ERROR] Cannot proceed without SDS, ISTerre FDSN, and the regional catalog client. Exiting.")
    sys.exit(1)

taup_model = TauPyModel(model=TAUP_MODEL)
rng = np.random.default_rng(RANDOM_SEED)



# =============================================================================
# SECTION 3 — RESOLVE STATIONS (same approach as 04d)
# =============================================================================

print(f"\n[SETUP] Fetching inventory for lat[{LAT_MIN},{LAT_MAX}] lon[{LON_MIN},{LON_MAX}] ...")
inventory = fetch_inventory(client_fdsn, T_START, T_END,
                            lat_min=LAT_MIN, lat_max=LAT_MAX,
                            lon_min=LON_MIN, lon_max=LON_MAX)
if inventory is None:
    print("[ERROR] Could not fetch inventory. Exiting.")
    sys.exit(1)

candidates = build_station_list_from_inventory(inventory)
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
print(f"[OK] {len(station_keys)} station(s) resolved in the bounding box.")


def _station_live(net, sta, t):
    """ True if this station's channel was operational at time t. """
    for p0, p1 in station_periods.get((net, sta), []):
        if p0 <= t <= p1:
            return True
    return False



# =============================================================================
# SECTION 4 — QUERY REGIONAL CATALOG
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 1 — Querying regional catalog")
print(f"{'='*65}")

regional_events = query_catalog_by_distance_chunked(
    client_regional,
    T_START, T_END,
    MASSIF_CENTER_LAT, MASSIF_CENTER_LON,
    DIST_MIN_KM, DIST_MAX_KM,
    MIN_MAGNITUDE, MAX_MAGNITUDE,
    chunk_days = CHUNK_DAYS,
    cache_path = CATALOG_CACHE_FILE if CATALOG_CACHE_FILE else None,
)

if not regional_events:
    print(f"\n[ERROR] No regional events found. If this is unexpected, check that "
          f"{CATALOG_URL} covers this distance/magnitude range (try CATALOG_URL="
          f"'USGS' instead, or back to 'https://api.franceseisme.fr' — see the "
          f"docstring for why EMSC replaced it). Exiting.")
    sys.exit(1)

print(f"[OK] {len(regional_events)} candidate regional event(s) found "
      f"(dist [{DIST_MIN_KM:.0f}, {DIST_MAX_KM:.0f}] km, mag >= {MIN_MAGNITUDE}).")

if len(regional_events) > N_EVENTS_CAP:
    idx = rng.choice(len(regional_events), size=N_EVENTS_CAP, replace=False)
    regional_events = [regional_events[i] for i in sorted(idx)]
    print(f"[INFO] Capped at N_EVENTS_CAP={N_EVENTS_CAP} (random subsample, seed={RANDOM_SEED}). "
          f"Raise N_EVENTS_CAP for a larger training set.")



# =============================================================================
# SECTION 5 — PREDICTED-ARRIVAL HELPER (obspy.taup)
# =============================================================================

def predicted_arrival(origin_time, event_lat, event_lon, event_depth_km, sta_lat, sta_lon):
    """
    Predicted first P-type arrival at a station, from ray theory.
    See docstring caveat: regional (Pg/Pn) predictions are less reliable
    than teleseismic ones, because iasp91's crustal model is coarse.

    Returns
    -------
    t_pred     : UTCDateTime or None
    phase_used : str or None — name of the phase used (e.g. "Pg", "Pn")
    dist_km    : float — great-circle epicentral distance
    az_deg     : float — azimuth event -> station
    """
    dist_deg = locations2degrees(sta_lat, sta_lon, event_lat, event_lon)
    dist_km, az_deg, _ = gps2dist_azimuth(event_lat, event_lon, sta_lat, sta_lon)
    dist_km /= 1000.0

    depth_km = event_depth_km if (event_depth_km is not None and event_depth_km >= 0) else DEFAULT_DEPTH_KM

    try:
        arrivals = taup_model.get_travel_times(
            source_depth_in_km = depth_km,
            distance_in_degree = dist_deg,
            phase_list         = TAUP_PHASE_LIST,
        )
    except Exception:
        return None, None, dist_km, az_deg

    if not arrivals:
        return None, None, dist_km, az_deg

    first = min(arrivals, key=lambda a: a.time)
    t_pred = origin_time + first.time
    return t_pred, first.name, dist_km, az_deg



# =============================================================================
# SECTION 6 — MAIN PROCESSING LOOP  (event x station)
# =============================================================================

print(f"\n{'='*65}")
print(f"  STEP 2 — Processing {len(regional_events)} event(s) x {len(station_keys)} station(s)")
print(f"{'='*65}\n")

all_rows        = []
n_ev_processed  = 0
n_pairs_total   = 0
n_rej_notlive   = 0
n_rej_noarrival = 0
n_rej_nodata    = 0
n_rej_response  = 0
n_rej_feat      = 0
n_detected      = 0
n_fallback      = 0

for i, ev in enumerate(regional_events):
    origin = ev.preferred_origin() or ev.origins[0]
    t_orig = origin.time
    ev_lat, ev_lon = origin.latitude, origin.longitude
    ev_depth_km = (origin.depth / 1000.0) if origin.depth is not None else None

    mag_obj = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
    magnitude      = mag_obj.mag if mag_obj else np.nan
    magnitude_type = mag_obj.magnitude_type if mag_obj else None

    print(f"{'='*60}")
    print(f"  Event {i+1}/{len(regional_events)}: {t_orig}  |  "
          f"M{magnitude_type or '?'} {magnitude:.1f}  |  "
          f"({ev_lat:.2f}, {ev_lon:.2f})  depth={ev_depth_km if ev_depth_km is not None else '?'} km")

    n_ev_processed += 1
    any_row_this_event = False

    for net, sta in station_keys:
        loc, chan = station_channel[(net, sta)]
        sta_lat, sta_lon = station_coords.get((net, sta), (np.nan, np.nan))
        if np.isnan(sta_lat) or np.isnan(sta_lon):
            continue

        t_pred, phase_used, dist_km, az_deg = predicted_arrival(
            t_orig, ev_lat, ev_lon, ev_depth_km, sta_lat, sta_lon
        )
        if t_pred is None:
            n_rej_noarrival += 1
            continue
        # Safety guard — should not trigger given the catalog's own
        # minradius/maxradius filter, but a single station could in
        # principle sit outside the band even when the massif centroid
        # doesn't (negligible given the massif is a ~50 km box).
        if dist_km < DIST_MIN_KM or (DIST_MAX_KM is not None and dist_km > DIST_MAX_KM):
            continue

        n_pairs_total += 1

        if not _station_live(net, sta, t_pred):
            n_rej_notlive += 1
            continue

        t_win_start = t_pred - PRE_ARRIVAL_S
        t_win_end   = t_pred + POST_ARRIVAL_S

        # ---- Fetch + response removal ----------------------------------------
        try:
            st_raw = client_sds.get_waveforms(net, sta, loc, chan, t_win_start, t_win_end)
        except Exception:
            n_rej_nodata += 1
            continue
        if len(st_raw) == 0:
            n_rej_nodata += 1
            continue
        st_raw.merge(method=1, fill_value="interpolate")

        # preprocess_day() (pre_filt-tapered) kept as the safe default,
        # inherited from the teleseismic-era design — it only tapers content
        # below 0.01 Hz / above 0.95x Nyquist, so it doesn't touch the 1-20 Hz
        # regional detection band either way. See docstring.
        tr_vel = preprocess_day(st_raw[0], inventory)
        if tr_vel is None:
            n_rej_response += 1
            continue
        fs = tr_vel.stats.sampling_rate

        # ---- Bandpass for onset refinement + SNR (1-20 Hz, local pipeline band) --
        tr_filt = tr_vel.copy()
        nyq = fs / 2
        tr_filt.filter("bandpass", freqmin=FREQ_MIN,
                       freqmax=min(FREQ_MAX, 0.9 * nyq),
                       corners=4, zerophase=True)

        # ---- Onset refinement: classical STA/LTA search near predicted arrival --
        t_on = t_off = None
        detected = False
        trig_on_cft = trig_off_cft = np.nan

        trace_dur = tr_filt.stats.endtime - tr_filt.stats.starttime
        if trace_dur >= LTA_S + STA_S:
            try:
                cft, on_off = run_sta_lta(tr_filt, STA_S, LTA_S, THRES_ON, THRES_OFF)
            except Exception:
                cft, on_off = np.array([]), []

            best_pair = None
            best_dt   = None
            for (i_on, i_off) in on_off:
                cand_t_on  = tr_filt.stats.starttime + i_on / fs
                cand_t_off = tr_filt.stats.starttime + i_off / fs
                if (cand_t_off - cand_t_on) < MIN_DURAT_S:
                    continue
                dt = min(abs(cand_t_on - t_pred), abs(cand_t_off - t_pred))
                if cand_t_on <= t_pred <= cand_t_off:
                    dt = 0.0
                if dt <= ARRIVAL_MATCH_TOL_S and (best_dt is None or dt < best_dt):
                    best_dt = dt
                    best_pair = (i_on, i_off)

            if best_pair is not None:
                i_on, i_off = best_pair
                t_on  = tr_filt.stats.starttime + i_on / fs
                t_off = tr_filt.stats.starttime + i_off / fs
                trig_on_cft  = float(cft[i_on])  if i_on  < len(cft) else np.nan
                trig_off_cft = float(cft[i_off]) if i_off < len(cft) else np.nan
                detected = True

        if not detected:
            # Fallback: fixed window anchored on the theoretical prediction —
            # kept for inspection, not assumed to be good training data.
            t_on  = t_pred - FALLBACK_PRE_S
            t_off = t_pred + FALLBACK_POST_S
            t_on  = max(t_on,  tr_vel.stats.starttime)
            t_off = min(t_off, tr_vel.stats.endtime)

        arrival_inside_det = bool(t_on <= t_pred <= t_off)
        arrival_lag_s      = round(float(t_pred - t_on), 2)

        if detected:
            n_detected += 1
        else:
            n_fallback += 1

        # ---- SNR (same 7 metrics as 04a/04d) ----------------------------------
        snr = compute_snr(tr_filt, t_on, t_off)

        # ---- Feature extraction window (padded, broadband — same as 04a) -----
        try:
            t_cut_on  = max(t_on  - PAD_SEC, tr_vel.stats.starttime)
            t_cut_off = min(t_off + PAD_SEC, tr_vel.stats.endtime)
            tr_cut    = tr_vel.slice(t_cut_on, t_cut_off)
        except Exception:
            n_rej_feat += 1
            continue
        if tr_cut.stats.npts < 10:
            n_rej_feat += 1
            continue

        data_3c = None
        if LOAD_3C:
            base = chan[:-1]
            n_pts = tr_cut.stats.npts
            data_n = data_e = None
            for suf_n, suf_e in [("N", "E"), ("1", "2")]:
                try:
                    st_n = client_sds.get_waveforms(net, sta, loc, base + suf_n, t_cut_on, t_cut_off)
                    st_e = client_sds.get_waveforms(net, sta, loc, base + suf_e, t_cut_on, t_cut_off)
                    if len(st_n) == 0 or len(st_e) == 0:
                        continue
                    tr_n, tr_e = st_n[0].copy(), st_e[0].copy()
                    tr_n.detrend("demean"); tr_e.detrend("demean")
                    if abs(tr_n.stats.sampling_rate - fs) > 1:
                        tr_n.resample(fs)
                    if abs(tr_e.stats.sampling_rate - fs) > 1:
                        tr_e.resample(fs)
                    dn = tr_n.data[:n_pts].astype(float)
                    de = tr_e.data[:n_pts].astype(float)
                    if len(dn) < n_pts:
                        dn = np.pad(dn, (0, n_pts - len(dn)))
                    if len(de) < n_pts:
                        de = np.pad(de, (0, n_pts - len(de)))
                    data_n, data_e = dn, de
                    break
                except Exception:
                    continue
            if data_n is not None and data_e is not None:
                data_3c = np.stack([tr_cut.data.astype(float), data_n, data_e])

        feats = extract_features(tr_cut.data, fs, data_3c=data_3c)
        if np.all(np.isnan(feats)):
            n_rej_feat += 1
            continue

        quality_ok = (
            snr.get("SNR", 0)             >= SNR_MIN and
            snr.get("SNR_full_median", 0) >= SNR_FULL_MEDIAN_MIN
        )

        row = {
            # Event metadata — TRUE regional hypocenter, not the massif
            "event_time"        : str(t_orig),
            "event_type"        : "regional",
            "catalog_lat"       : ev_lat,
            "catalog_lon"       : ev_lon,
            "catalog_depth_km"  : ev_depth_km if ev_depth_km is not None else np.nan,
            "magnitude"         : magnitude,
            "magnitude_type"    : magnitude_type,
            # Station
            "network"           : net,
            "station"           : sta,
            "channel"           : chan,
            # Geometry
            "distance_km"       : round(dist_km, 1),
            "azimuth_deg"       : round(az_deg, 1),
            "phase_used"        : phase_used,
            "predicted_arrival" : str(t_pred),
            # Detection window
            "det_starttime"     : str(t_on),
            "det_starttime_raw" : str(t_on),
            "onset_refine_s"    : 0.0,
            "det_endtime"       : str(t_off),
            "det_duration_s"    : round(t_off - t_on, 2),
            "trigger_on_cft"    : round(trig_on_cft, 4)  if not np.isnan(trig_on_cft)  else np.nan,
            "trigger_off_cft"   : round(trig_off_cft, 4) if not np.isnan(trig_off_cft) else np.nan,
            "detected"          : detected,
            # Quality flags — reuse 04a's column names: "origin"=theoretical
            # arrival (no true pick exists for regional phases at these stations)
            "origin_inside_det" : arrival_inside_det,
            "origin_lag_s"      : arrival_lag_s,
            "pick_inside_det"   : None,
            "pick_lag_s"        : np.nan,
            "quality_ok"        : quality_ok,
            **snr,
        }
        for fname, fval in zip(_FEAT_NAMES, feats):
            row[fname] = fval

        all_rows.append(row)
        any_row_this_event = True

        if len(all_rows) % 100 == 0:
            print(f"  ... {len(all_rows)} rows so far "
                  f"({n_detected} detected, {n_fallback} fallback)")

        if CHECKPOINT_EVERY > 0 and len(all_rows) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(all_rows).to_csv(
                os.path.join(RUN_DIR, f"regional_windows_checkpoint_{len(all_rows)}.csv"),
                index=False)

    if any_row_this_event:
        print(f"    -> {sum(1 for r in all_rows if r['event_time'] == str(t_orig))} station row(s) kept")



# =============================================================================
# SECTION 7 — SAVE CSV + SUMMARY
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 3 — Save + summary")
print(f"{'='*65}")

if not all_rows:
    print("\n[WARN] No regional windows extracted — CSV will not be written.")
else:
    df = pd.DataFrame(all_rows)

    meta_cols = [
        "event_time", "event_type", "catalog_lat", "catalog_lon",
        "catalog_depth_km", "magnitude", "magnitude_type",
        "network", "station", "channel",
        "distance_km", "azimuth_deg", "phase_used", "predicted_arrival",
        "det_starttime", "det_starttime_raw", "onset_refine_s",
        "det_endtime", "det_duration_s",
        "trigger_on_cft", "trigger_off_cft", "detected",
        "origin_inside_det", "origin_lag_s",
        "pick_inside_det", "pick_lag_s", "quality_ok",
        "SNR", "SNR_picking_5_5", "SNR_picking_3_3",
        "SNR_picking_1_3", "SNR_full_mean", "SNR_full_median", "SNR_s2n_median",
    ]
    ordered_cols = meta_cols + _FEAT_NAMES
    df = df[[c for c in ordered_cols if c in df.columns]]

    csv_path = os.path.join(RUN_DIR, f"regional_windows_{_RUN_STAMP}.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n[SAVED] {csv_path}")
    print(f"        {df.shape[0]} rows x {df.shape[1]} columns  "
          f"[{len(_FEAT_NAMES)} features, 3C={'on' if LOAD_3C else 'off'}]")
    print(f"        Distance range : [{df['distance_km'].min():.0f}, {df['distance_km'].max():.0f}] km")
    print(f"        Magnitude range: [{df['magnitude'].min():.1f}, {df['magnitude'].max():.1f}]")
    print(f"        Detected (STA/LTA-confirmed) : {(df['detected']).sum()} / {len(df)}")
    print(f"        Quality gate pass (SNR>={SNR_MIN}, SNR_full_median>={SNR_FULL_MEDIAN_MIN}) : "
          f"{df['quality_ok'].sum()} / {len(df)}")

    print(f"\n  Rows per station:")
    for (net, sta), n in df.groupby(["network", "station"]).size().items():
        print(f"    {net}.{sta:<8} {n:5d}")

print(f"\n  Events processed         : {n_ev_processed} / {len(regional_events)}")
print(f"  (event, station) pairs   : {n_pairs_total}")
print(f"  Rejected — no predicted arrival : {n_rej_noarrival}")
print(f"  Rejected — station not live     : {n_rej_notlive}")
print(f"  Rejected — no waveform data     : {n_rej_nodata}")
print(f"  Rejected — response removal     : {n_rej_response}")
print(f"  Rejected — feature extraction   : {n_rej_feat}")
print(f"  STA/LTA-confirmed onset         : {n_detected}")
print(f"  Fallback (theoretical) window   : {n_fallback}")



# =============================================================================
# END
# =============================================================================

from datetime import datetime
print("\n" + "=" * 70)
print(f"  Run finished : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Events       : {n_ev_processed}")
print(f"  Rows         : {len(all_rows)}")
print(f"  All outputs  : {RUN_DIR}")
print(f"  Log file     : {_log_filename}")
print("=" * 70)

_log_file.close()
