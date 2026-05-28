"""
SEISMIC FEATURE EXTRACTION
==========================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : April 2026

Goal
----
For each seismic event in the ISTerre catalog:
- load the vertical-component waveforms from the SDS archive
- preprocess them (instrument response removal)
- compute the 99 seismic attributes defined in seismic_params.py
(Maggi / Hibert feature set: waveform shape, spectral, pseudo-spectrogram, frequency-band energies, kurtosis, autocorrelation, SNR)

The output is a single CSV table ("feature table") -> ready to feed a classifier (e.g. Random Forest, as in Groult et al. 2026)

Data sources
------------
  Catalog + picks : ISTerre FDSN server  http://ist-sc3-geobs.osug.fr:8080
  Waveforms       : ISTerre SDS archive  /data/sig/SDS  (cluster only)

Output
------
  features_<run_stamp>.csv   -> one row per (event × station), 99 feature cols
  run.log                    -> full console output of the run
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# All parameters to adapt are grouped here
# =============================================================================

# -- Paths --------------------------------------------------------------------
SDS_ROOT    = "/data/sig/SDS"                                     # waveform archive on cluster
ISTERRE_URL = "http://ist-sc3-geobs.osug.fr:8080"                 # FDSN catalog server
OUTPUT_DIR  = "/data/failles/louisels/project/results/outputs_03"

# -- Catalog query window -----------------------------------------------------
T_START = "2022-06-01"
T_END   = "2022-06-07"

# Spatial bounding box (Mont Blanc massif and surroundings)
LAT_MIN, LAT_MAX = 45.5, 46.0
LON_MIN, LON_MAX = 6.5, 7.2

# Event types to keep (exact strings used in the ISTerre catalog)
TARGET_TYPES = ["earthquake", "quarry blast", "ice quake", "rockslide"]

# -- Waveform extraction window -----------------------------------------------
PRE_EVENT  = 10   # seconds to include BEFORE the first pick
POST_EVENT = 90   # seconds to include AFTER the origin time

# Channel wildcard for the SDS client
Z_CHANNELS = "??Z"   # matches any 3-character channel ending in Z

# -- Feature extraction -------------------------------------------------------
# flag=0 -> 99 features (vertical component only)
# flag=1 -> 62 features (3-component polarisation mode, not used here)
FEATURE_FLAG = 0
N_FEATURES   = 99

# -- Output -------------------------------------------------------------------
MIN_STATIONS_PER_EVENT = 1    # minimum number of valid stations required to keep an event's rows in the table
CHECKPOINT_EVERY = 50         # save intermediate CSV every N events (0 = only save at the very end)



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

from catalog_helpers import (
    get_stations_from_picks,
    find_event_by_time,
    summarise_catalog,
    query_catalog,
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
)
from features import FEATURE_NAMES, extract_features

import seismic_params; print("Loaded from:", seismic_params.__file__)


# ----------- Run setup ----------------
RUN_DIR, _RUN_STAMP = create_run_dir(OUTPUT_DIR)
_log_file, _log_filename = setup_logging(
    RUN_DIR, "03_feature_extraction.py",
    extra_info=f"Feature flag: {FEATURE_FLAG}  ({N_FEATURES} features per trace)"
)


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
# SECTION 3 — MAIN PROCESSING LOOP
# For each event: load waveforms, preprocess, extract 99 features per station
# =============================================================================

if client_sds is None:
    print("\n[ERROR] SDS client unavailable — cannot load waveforms. Exiting.")
    sys.exit(1)

print(f"\n--- Extracting features for {len(events)} events ---")
print(f"    Columns: event metadata (7) + {N_FEATURES} features = "
      f"{7 + N_FEATURES} total columns per row\n")

# Accumulate rows here, convert to DataFrame at the end
rows = []

# Counters
n_processed     = 0
n_skipped       = 0
n_failed_traces = 0

# Output CSV path (final)
csv_path = os.path.join(RUN_DIR, f"features_{_RUN_STAMP}.csv")

for i, ev in enumerate(events):
    origin   = ev.preferred_origin() or ev.origins[0]
    etype    = str(ev.event_type) if ev.event_type else "unknown"
    mag_obj  = ev.preferred_magnitude()
    mag      = mag_obj.mag if mag_obj else np.nan
    lat      = origin.latitude
    lon      = origin.longitude
    depth_km = origin.depth / 1000.0 if origin.depth is not None else np.nan
    t_origin = origin.time

    # Unique event identifier: ISO origin time, safe for filenames / CSV keys
    event_id = str(t_origin)[:19].replace(":", "").replace("T", "_")

    print(f"{'='*60}")
    print(f"  Event {i+1}/{len(events)}: {etype}  |  {t_origin}  |  M{mag:.1f}" if not np.isnan(mag)
          else f"  Event {i+1}/{len(events)}: {etype}  |  {t_origin}  |  M?")

    # ---- Load waveforms ----
    st_raw, t_start, t_end = load_waveforms_sds(client_sds, ev, Z_CHANNELS, PRE_EVENT, POST_EVENT)
    if len(st_raw) == 0:
        print("    [SKIP] No waveforms found in SDS.")
        n_skipped += 1
        continue

    station_times_df = build_station_times_df(st_raw, t_start, t_end)

    # ---- Preprocess: instrument response removal ----
    st_vel = remove_response_or_fallback(st_raw, inventory, station_times_df)

    if len(st_vel) == 0:
        print("    [SKIP] No valid traces after response removal.")
        n_skipped += 1
        continue

    # ---- Extract features per station trace ----
    n_ok = 0
    for tr in st_vel:
        network = tr.stats.network
        station = tr.stats.station

        feats = extract_features(tr.data, tr.stats.sampling_rate, n_features=N_FEATURES, feature_flag=FEATURE_FLAG)

        if np.all(np.isnan(feats)):
            n_failed_traces += 1
            continue

        # Build one row: metadata + 99 features
        row = {
            'event_id'   : event_id,
            'event_time' : str(t_origin),
            'event_type' : etype,
            'magnitude'  : mag,
            'latitude'   : lat,
            'longitude'  : lon,
            'depth_km'   : depth_km,
            'network'    : network,
            'station'    : station,
        }
        for feat_name, feat_val in zip(FEATURE_NAMES, feats):
            row[feat_name] = feat_val

        rows.append(row)
        n_ok += 1

    print(f"    -> {n_ok}/{len(st_vel)} station(s) with valid features")
    n_processed += 1

    # ---- Checkpoint save ----
    if CHECKPOINT_EVERY > 0 and n_processed % CHECKPOINT_EVERY == 0 and rows:
        df_chk = pd.DataFrame(rows)
        chk_path = os.path.join(RUN_DIR, f"features_checkpoint_{n_processed}.csv")
        df_chk.to_csv(chk_path, index=False)
        print(f"\n  [CHECKPOINT] {len(rows)} rows saved -> {os.path.basename(chk_path)}\n")



# =============================================================================
# SECTION 4 — SAVE FEATURE TABLE
# =============================================================================

if not rows:
    print("\n[WARN] No features extracted — output CSV will not be written.")
else:
    df = pd.DataFrame(rows)

    # Column order: metadata first, then features in order
    meta_cols    = ['event_id', 'event_time', 'event_type', 'magnitude',
                    'latitude', 'longitude', 'depth_km', 'network', 'station']
    ordered_cols = meta_cols + FEATURE_NAMES
    df = df[ordered_cols]

    df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] Feature table -> {csv_path}")
    print(f"        Shape : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"        Events with data : {df['event_id'].nunique()}")
    print(f"        Unique stations  : {df['station'].nunique()}")

    # Per-type summary
    print("\n  Rows per event type:")
    for etype_name, grp in df.groupby('event_type'):
        print(f"    {etype_name:>15s} : {len(grp):5d} rows  "
              f"({grp['event_id'].nunique()} events, "
              f"{grp['station'].nunique()} unique stations)")

    # NaN report (any feature with > 5 % NaN values)
    nan_frac = df[FEATURE_NAMES].isna().mean()
    bad_feats = nan_frac[nan_frac > 0.05]
    if len(bad_feats):
        print(f"\n  [WARN] Features with >5% NaN values ({len(bad_feats)}/{N_FEATURES}):")
        for fname, frac in bad_feats.sort_values(ascending=False).items():
            print(f"    {fname} : {frac*100:.1f}%")
    else:
        print(f"\n  All {N_FEATURES} features have <5% NaN values.")



# =============================================================================
# END
# =============================================================================

from datetime import datetime
print("\n" + "=" * 70)
print(f"  Run finished    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Events processed: {n_processed}/{len(events)}")
print(f"  Events skipped  : {n_skipped}")
print(f"  Traces failed   : {n_failed_traces}")
print(f"  Total rows      : {len(rows)}")
print(f"  All outputs     : {RUN_DIR}")
print(f"  Log file        : {_log_filename}")
print("=" * 70)

_log_file.close()
