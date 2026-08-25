"""
08g_multistation_coincidence_check.py
=======================================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : August 2026

Goal
----
Answers Elsa's question directly first: NEITHER 09a NOR 09b currently check
this. Both scripts run DetecteurV3's STA/LTA scan completely independently,
one station at a time (see their own "for net, sta, loc, chan in
station_list" loops) -- a detected window on station A has zero awareness of
what station B saw at the same time. predictions_<month>.csv pools every
station's rows together in one file, but nothing in 09a/09b ever cross-
references them. So a single noisy transient on one sensor can be classified
earthquake/regional/rockslide/ice quake with no check at all that anything
else on the massif saw it too.

This script adds that check as a POST-HOC pass over an existing
predictions_<month>.csv (no re-detection, no re-classification, no SDS
needed) -- exactly the "keep only events seen by several stations" filter
Elsa asked for, done at the level that's actually principled:

  - Coincidence is checked on RAW DETECTIONS, not on predicted_class. Every
    DetecteurV3 trigger gets a row in predictions_<month>.csv regardless of
    what it was classified as (see 09a/09b's own docstrings -- one row per
    detected window, no subsetting). So "did station B also detect
    something around this time" is checked against ALL of station B's rows,
    whatever class they got assigned -- a real event weakly recorded at a
    second station and (wrongly) called "noise" there still counts as
    physical corroboration. Requiring an IDENTICAL predicted_class across
    stations would be a much stronger, and much less defensible, test --
    the classifiers already don't agree with each other station-to-station
    even on the training catalog, so that's not what "correlated" should
    mean here.

  - IMPORTANT CAVEAT -- do not apply this uniformly to every class. Regional
    and (teleseismic-ish) earthquake signals are far-field: if real, they
    should register across most/all of the massif, so "no other station
    triggered nearby" is a strong argument the label is spurious for those
    two classes. Rockslide and ice quake are near-source, spatially
    localized cryospheric/slope processes -- a real one can legitimately
    register on only the one or two stations closest to where it happened,
    even at full physical validity. Blanket-requiring multi-station
    coincidence for those two classes risks discarding genuine detections,
    not just noise, and would quietly bias the report toward under-counting
    exactly the classes the internship cares most about. CLASSES_REQUIRING_
    COINCIDENCE below defaults to ["earthquake", "regional"] ONLY for that
    reason -- rockslide/ice quake/noise are reported on (this script still
    computes and prints their coincidence rate, which is itself a useful
    number -- if it comes out low that's expected/physical, not a red flag)
    but never filtered out on this basis alone.

  - COINCIDENCE_TOLERANCE_S is a single fixed time window applied network-
    wide, not scaled by inter-station distance / apparent velocity. That's
    a simplification, not a hidden claim of precision -- pick a value
    generous enough to cover the true P-wave (or slower, for
    rockslide-type surface signals) travel-time spread across the massif's
    aperture (bounding box in Section 1 is roughly 50x55 km -- a few km/s
    apparent velocity puts cross-network delay at a handful to ~15s for a
    genuinely regional source) plus some slack for onset-picking jitter
    between stations/classifiers. The default below (20s) is a reasonable
    starting point, not a validated constant -- sanity-check it against
    your own station geometry / STA-LTA pick jitter before trusting the
    filtered output for the report, and feel free to re-run with a couple
    of values to see how sensitive the numbers are.

What it does
------------
  1. Reads one or more predictions_<month>.csv (PIPELINE = "HGB" or "CNN",
     same convention as 08d/08e/08f -- pick one at a time; run twice to
     check both).
  2. Pools every station's rows for that month, sorts by window_start.
  3. For each window, finds every OTHER station with a detection (any
     class) within +/- COINCIDENCE_TOLERANCE_S, and records how many
     distinct other stations that is.
  4. Prints, per predicted class: % of windows with >=1 corroborating
     station, and the same split further by whether that corroborating
     detection got the SAME predicted class or not (an extra, optional,
     stricter signal -- see step 1's reasoning for why it's secondary).
  5. Writes the full annotated table (every row + its coincidence count) and
     a second, FILTERED table that drops rows in CLASSES_REQUIRING_
     COINCIDENCE with zero corroboration, leaving every other class
     untouched.
  6. Bar chart of % multi-station-corroborated by class -- itself a useful
     report figure/sanity check: earthquake/regional should sit much higher
     than rockslide/ice quake if the near-source hypothesis above is right.

Needs nothing but the predictions CSV -- no SDS, no FDSN, no cluster.

Output layout
-------------
  OUTPUT_DIR/run_YYYYMMDD_HHMMSS/
      coincidence_annotated_<pipeline>.csv   <- every row + coincidence columns
      coincidence_filtered_<pipeline>.csv    <- annotated table with the
                                                 CLASSES_REQUIRING_COINCIDENCE
                                                 zero-corroboration rows removed
      fig_coincidence_rate_by_class_<stamp>.png
      run.log
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# "HGB" -- 06c/09b (scalar-feature classifier)  |  "CNN" -- 07b/09a (spectrogram classifier)
# Run this script once per pipeline you want checked -- only the matching
# PREDICTIONS_CSVS list below is used.
PIPELINE = "HGB"

# -- Input: one or more predictions_<month>.csv (pooled, same convention as
# 08d/08e/08f). Needs every station's rows for the month(s) you're checking,
# in the SAME file(s) -- that's what makes the coincidence check possible.
PREDICTIONS_CSVS_HGB = [
    "/data/failles/louisels/project/results/outputs_09b/run_20260821_173942/predictions_2025-01.csv",
]
PREDICTIONS_CSVS_CNN = [
    "/data/failles/louisels/project/results/outputs_09a/predictions_2025-01.csv",
]

CLASS_ORDER  = ["earthquake", "regional", "rockslide", "ice quake", "noise"]
CLASS_COLORS = {"earthquake": "#1f77b4", "rockslide": "#d62728",
                "ice quake": "#2ca02c", "noise": "#7f7f7f", "regional": "#9467bd"}

# -- How close in time do two DIFFERENT stations' windows have to be to count
# as "the same physical event"? See CAVEAT in the module docstring -- a
# single network-wide value, not distance-scaled. Tune and re-run if unsure.
COINCIDENCE_TOLERANCE_S = 20

# -- How many OTHER stations must corroborate a window for it to count as
# "multi-station" (1 = at least 2 stations total, i.e. Elsa's "plusieurs
# stations" ask) ------------------------------------------------------------
MIN_OTHER_STATIONS = 1

# -- Classes the multi-station requirement is actually used to FILTER on --
# see the module docstring's CAVEAT for why rockslide/ice quake are left out
# by default (near-source, can be legitimately single-station). noise is
# never filtered on this basis either way -- it's not what's being checked.
# Coincidence rate is still computed and reported for EVERY class regardless
# of whether it's in this list.
CLASSES_REQUIRING_COINCIDENCE = ["earthquake", "regional"]

OUTPUT_DIR = "/data/failles/louisels/project/results/outputs_08g"



# =============================================================================
# SECTION 2 — SETUP
# =============================================================================

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_setup import create_run_dir, setup_logging, set_matplotlib_defaults

if PIPELINE not in ("HGB", "CNN"):
    print(f"[ERROR] PIPELINE must be 'HGB' or 'CNN', got '{PIPELINE}'.")
    sys.exit(1)

_PREDICTIONS_CSVS = PREDICTIONS_CSVS_HGB if PIPELINE == "HGB" else PREDICTIONS_CSVS_CNN

RUN_DIR, STAMP = create_run_dir(OUTPUT_DIR)
log_file, log_path = setup_logging(
    RUN_DIR, "08g_multistation_coincidence_check.py",
    extra_info=(f"PIPELINE={PIPELINE}  |  COINCIDENCE_TOLERANCE_S={COINCIDENCE_TOLERANCE_S}  |  "
                f"MIN_OTHER_STATIONS={MIN_OTHER_STATIONS}  |  "
                f"{len(_PREDICTIONS_CSVS)} predictions file(s)")
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
set_matplotlib_defaults()



# =============================================================================
# SECTION 3 — LOAD + POOL ALL STATIONS
# =============================================================================

print(f"\n{'='*70}")
print(f"  LOADING PREDICTIONS  (PIPELINE={PIPELINE})")
print(f"{'='*70}")

_frames = []
for fpath in _PREDICTIONS_CSVS:
    if not os.path.isfile(fpath):
        print(f"  [WARN] file not found, skipping: {fpath}")
        continue
    df_f = pd.read_csv(fpath, low_memory=False)
    print(f"  [OK] {os.path.basename(fpath)}  {len(df_f):,} row(s)")
    _frames.append(df_f)

if not _frames:
    print("[ERROR] No predictions file could be read -- check Section 1 paths.")
    log_file.close()
    sys.exit(1)

df = pd.concat(_frames, ignore_index=True)

required_cols = {"network", "station", "window_start", "predicted_class"}
missing = required_cols - set(df.columns)
if missing:
    print(f"[ERROR] predictions CSV is missing required column(s): {sorted(missing)}")
    log_file.close()
    sys.exit(1)

df["_station_id"] = df["network"].astype(str) + "." + df["station"].astype(str)
# format="ISO8601" (not the default format-sniffing path) so a mix of
# fractional-second precisions across rows can't make pandas silently
# mis-parse/NaT rows it guessed the wrong format from.
df["_t"] = pd.to_datetime(df["window_start"], utc=True, errors="coerce", format="ISO8601")
n_before = len(df)
df = df.dropna(subset=["_t"]).copy()
if len(df) < n_before:
    print(f"  [WARN] {n_before - len(df):,} row(s) had an unparseable window_start -- dropped.")

n_stations_total = df["_station_id"].nunique()
print(f"\n  {len(df):,} window(s) total, across {n_stations_total} station(s), "
      f"{df['_t'].dt.date.min()} to {df['_t'].dt.date.max()}")



# =============================================================================
# SECTION 4 — COINCIDENCE COMPUTATION
# =============================================================================

print(f"\n{'='*70}")
print(f"  COMPUTING CROSS-STATION COINCIDENCE  (tolerance = +/-{COINCIDENCE_TOLERANCE_S}s)")
print(f"{'='*70}")

df = df.sort_values("_t").reset_index(drop=True)
# Seconds since epoch, sorted. NOT df["_t"].astype("int64") -- pandas'
# datetime64 storage resolution (ns vs us) is version/parse-path dependent
# (e.g. pandas >=2.x commonly parses to datetime64[us]), so a raw int64
# view silently means different units depending on that resolution and
# throws the tolerance off by orders of magnitude. This subtraction is
# resolution-independent and always comes out in seconds.
t_epoch = ((df["_t"] - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy()
sta_id  = df["_station_id"].to_numpy()
cls_arr = df["predicted_class"].to_numpy()

tol = float(COINCIDENCE_TOLERANCE_S)
left_idx  = np.searchsorted(t_epoch, t_epoch - tol, side="left")
right_idx = np.searchsorted(t_epoch, t_epoch + tol, side="right")

n_other_stations   = np.empty(len(df), dtype=int)
n_other_same_class = np.empty(len(df), dtype=int)
other_stations_list = [None] * len(df)

for i in range(len(df)):
    lo, hi = left_idx[i], right_idx[i]
    window_sta = sta_id[lo:hi]
    window_cls = cls_arr[lo:hi]
    is_other = window_sta != sta_id[i]
    other_sta_here = set(window_sta[is_other])
    n_other_stations[i] = len(other_sta_here)
    other_stations_list[i] = ",".join(sorted(other_sta_here))
    same_cls_other = is_other & (window_cls == cls_arr[i])
    n_other_same_class[i] = len(set(window_sta[same_cls_other]))

df["n_other_stations_within_tol"]    = n_other_stations
df["other_stations_within_tol"]      = other_stations_list
df["n_other_stations_same_class"]    = n_other_same_class
df["is_multistation"]                = df["n_other_stations_within_tol"] >= MIN_OTHER_STATIONS

annotated_path = os.path.join(RUN_DIR, f"coincidence_annotated_{PIPELINE}.csv")
df.drop(columns=["_station_id", "_t"]).to_csv(annotated_path, index=False)
print(f"\n  [SAVED] {os.path.basename(annotated_path)}  ({len(df):,} row(s))")



# =============================================================================
# SECTION 5 — SUMMARY BY CLASS
# =============================================================================

print(f"\n{'='*70}")
print(f"  COINCIDENCE RATE BY PREDICTED CLASS")
print(f"{'='*70}")

summary_rows = []
for c in CLASS_ORDER:
    sub = df[df["predicted_class"] == c]
    if sub.empty:
        continue
    n = len(sub)
    n_multi = int(sub["is_multistation"].sum())
    pct_multi = 100.0 * n_multi / n
    n_multi_same_cls = int((sub["n_other_stations_same_class"] >= MIN_OTHER_STATIONS).sum())
    pct_multi_same_cls = 100.0 * n_multi_same_cls / n
    filtered_flag = "FILTERED if uncorroborated" if c in CLASSES_REQUIRING_COINCIDENCE else "reported only, not filtered"
    print(f"\n  {c:<11s} n={n:6,d}   {filtered_flag}")
    print(f"    >=1 other station (any class) within {COINCIDENCE_TOLERANCE_S}s : "
          f"{n_multi:6,d} / {n:6,d}  ({pct_multi:5.1f}%)")
    print(f"    >=1 other station with the SAME predicted class (stricter)  : "
          f"{n_multi_same_cls:6,d} / {n:6,d}  ({pct_multi_same_cls:5.1f}%)")
    summary_rows.append({
        "predicted_class": c, "n": n,
        "pct_multistation_any_class": pct_multi,
        "pct_multistation_same_class": pct_multi_same_cls,
        "filtered_on_this_basis": c in CLASSES_REQUIRING_COINCIDENCE,
    })

summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(RUN_DIR, f"coincidence_summary_by_class_{PIPELINE}.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\n  [SAVED] {os.path.basename(summary_path)}")



# =============================================================================
# SECTION 6 — FILTERED OUTPUT
# =============================================================================

drop_mask = df["predicted_class"].isin(CLASSES_REQUIRING_COINCIDENCE) & (~df["is_multistation"])
df_filtered = df[~drop_mask].copy()

print(f"\n{'='*70}")
print(f"  FILTERING  (drop uncorroborated rows only for {CLASSES_REQUIRING_COINCIDENCE})")
print(f"{'='*70}")
for c in CLASSES_REQUIRING_COINCIDENCE:
    n_before_c = int((df["predicted_class"] == c).sum())
    n_after_c  = int((df_filtered["predicted_class"] == c).sum())
    print(f"  {c:<11s} {n_before_c:6,d} -> {n_after_c:6,d}  "
          f"({n_before_c - n_after_c:,} dropped, no corroborating station within "
          f"{COINCIDENCE_TOLERANCE_S}s)")

filtered_path = os.path.join(RUN_DIR, f"coincidence_filtered_{PIPELINE}.csv")
df_filtered.drop(columns=["_station_id", "_t"]).to_csv(filtered_path, index=False)
print(f"\n  [SAVED] {os.path.basename(filtered_path)}  ({len(df_filtered):,} / {len(df):,} row(s) kept)")



# =============================================================================
# SECTION 7 — PLOT
# =============================================================================

if not summary_df.empty:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(summary_df))
    colors = [CLASS_COLORS.get(c, "grey") for c in summary_df["predicted_class"]]
    bars = ax.bar(x, summary_df["pct_multistation_any_class"], color=colors, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["predicted_class"], rotation=20, ha="right")
    ax.set_ylabel("% of windows with >=1 corroborating station\n(any class, within "
                  f"{COINCIDENCE_TOLERANCE_S}s)", fontweight="bold")
    ax.set_title(f"Multi-station coincidence rate by predicted class ({PIPELINE})", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, pct, n in zip(bars, summary_df["pct_multistation_any_class"], summary_df["n"]):
        ax.text(bar.get_x() + bar.get_width() / 2, pct + 1.5, f"{pct:.0f}%\n(n={n:,})",
                ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    fig_path = os.path.join(RUN_DIR, f"fig_coincidence_rate_by_class_{STAMP}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [SAVED] {os.path.basename(fig_path)}")



print(f"\n{'='*70}")
print(f"  DONE")
print(f"{'='*70}")
print(f"  Run folder: {RUN_DIR}")

log_file.close()
