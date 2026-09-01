"""
03c_denoiser_event_data.py
===========================
ISTerre internship
Author : Elsa Louis
Date   : June 2026 (generalized July 2026)

Prepare data for fine-tuning DeepDenoiser (Zhu et al., 2019) on a chosen event class's signals
Run DeepDenoiser inference to produce denoised waveforms for the events of that class that currently fail the SNR quality gate

Reference
---------
 Zhu et al. (2019). Seismic signal denoising and decomposition using deep neural networks.

Pipeline
--------
  STEP 1 — Load catalog CSV 
           RESCUE targets = EVENT_TYPE rows failing the rescue gate
           TRAINING pairs = rows (pooled across TRAIN_EVENT_TYPES, optionally mixed to TRAIN_MIX_RATIO) passing the stricter TRAIN_* gate
  STEP 2 — Extract 30 s signal windows from SDS (one .npz per event × station)
  STEP 3 — Extract 30 s pre-event noise windows from the same stations
  STEP 4 — Write signal_list.csv and noise_list.csv (DeepDenoiser index files)
  STEP 5 — Run DeepDenoiser inference via predict.py

Output layout
-------------
  outputs_03c/<event_slug>/run_YYYYMMDD_HHMMSS/
      signal/              ← signal .npz files
      noise/               ← noise  .npz files
      signal_list.csv      ← fname + channels, for DeepDenoiser training/inference
      noise_list.csv       ← fname + channels, for DeepDenoiser training
      pred_list.csv        ← fname only, used by predict.py
      denoised/            ← predict.py outputs (only if MODEL_DIR is set)
      run.log

.npz file (required by DeepDenoiser data_reader.py)
---------------------------------------------------
  data      float32  shape (3000, 3)  — 30 s at 100 Hz, 3 components (E, N, Z). If only Z is available on the station, it is replicated into all three slots so the existing DeepDenoiser code needs no edit.
  itp       int      — sample index of the signal onset within the 30 s window = PRE_PAD_S × TARGET_FS  (e.g. 10 s × 100 = sample 1000)
  channels  str      — station key "NET.STA" used to match signal ↔ noise files from the same station during training.
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# -- Target event class ---------------------------------------------------------
# Any value present in the catalog's `event_type` column: "ice quake", "rockslide", "earthquake", ...
EVENT_TYPE = "ice quake"
EVENT_SLUG = EVENT_TYPE.lower().replace(" ", "_")   # used in output folder names

# -- Inputs -------------------------------------------------------------------
CSV_PATH = (
    r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\04a_spectrogram_sta_lta_catalog"
    r"\all-99-features-recent+3C\catalog_windows_20260708_174019.csv"
)

SDS_ROOT    = "/data/sig/SDS"
OUTPUT_DIR  = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\03c_denoiser_event_data\icequake\stricter_20260722_120109"

# -- Quality gate -------------------------------------------------------------
# Defines RESCUE targets: EVENT_TYPE rows that fail this gate are what we try to denoise
SNR_MIN             = 1.70   # 05b Tier 2 — metric 'SNR' (peak/noise), AUC=0.627
SNR_FULL_MEDIAN_MIN = 1.99   # 05b Tier 2 — metric 'SNR_full_median', AUC=0.642 (best)

# -- Training-target gate (stricter than the rescue gate above) ---------------
# Used ONLY to select which rows are clean enough to serve as "signal" training examples for DeepDenoiser
# -> independent of the rescue gate above
TRAIN_SNR_MIN             = SNR_MIN
TRAIN_SNR_FULL_MEDIAN_MIN = SNR_FULL_MEDIAN_MIN

# -- Training-set composition (mix in other event classes?) -------------------
TRAIN_EVENT_TYPES = [EVENT_TYPE, "ice quake"]

# Target proportion of the final training set, by row count, e.g. {"rockslide": 0.5,"earthquake": 0.5} for a 50/50 mix 
TRAIN_MIX_RATIO = None #{EVENT_TYPE: 0.5, "earthquake": 0.5}
TRAIN_MIX_SEED  = 42   # reproducible subsampling

# -- DeepDenoiser directory (where predict.py lives) --------------------------
DEEPDENOISER_DIR = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\src\deepdenoiser"

# -- Trained model checkpoint for inference -----------------------------------
# Set to the path of a trained checkpoint folder, e.g.: MODEL_DIR = "/data/failles/louisels/project/results/deepdenoiser/log/260601-120000"
#  -> set to None to SKIP inference and prepare data only
MODEL_DIR = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\03c_denoiser_event_data\icequake\model-260722-124145"

# -- Existing run directory (inference-only shortcut) -------------------------
# When BOTH MODEL_DIR and EXISTING_RUN_DIR are set, the script skips ALL extraction steps (Sections 3–6b: SDS, signal, noise, rescue) and runs predict.py directly on the rescue files from that previous run 
#  -> set to None to run the full pipeline (extract everything from SDS)
EXISTING_RUN_DIR = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\03c_denoiser_event_data\icequake\stricter_20260722_120109"

# -- Waveform extraction parameters ------------------------------------------
TARGET_FS  = 100      # [Hz]  target sampling rate (DeepDenoiser default)
WINDOW_S   = 30       # [s]   total window length  (= 3000 samples at 100 Hz)
PRE_PAD_S  = 10       # [s]   seconds of pre-signal padding → itp = PRE_PAD_S × TARGET_FS

# -- Noise window extraction --------------------------------------------------
NOISE_OFFSET_S = 120  # [s] gap between noise window end and detection onset noise window

# -- Channel fallback strategy ------------------------------------------------
# For each Z-channel in the catalog we try to also load the two horizontal components by replacing the last letter (Z → N, Z → E or Z → 2, Z → 1)
HORIZONTAL_SUFFIXES = [("N", "E"), ("2", "1")]



# =============================================================================
# SECTION 2 — SETUP
# =============================================================================

import os
import sys
import warnings

import numpy as np
import pandas as pd
from obspy import UTCDateTime

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_setup import create_run_dir, setup_logging, connect_sds, set_matplotlib_defaults
from preprocessing import load_3component
from denoiser_utils import run_deepdenoiser_predict   


if MODEL_DIR is not None and EXISTING_RUN_DIR is not None:
    rescue_dir      = os.path.join(EXISTING_RUN_DIR, "rescue")
    rescue_csv_path = os.path.join(EXISTING_RUN_DIR, "rescue_list.csv")
    denoised_dir    = os.path.join(EXISTING_RUN_DIR, "denoised")

    print(f"\n{'='*65}")
    print("  INFERENCE-ONLY MODE — skipping SDS extraction")
    print(f"{'='*65}")
    print(f"  Event type   : {EVENT_TYPE}")
    print(f"  Existing run : {EXISTING_RUN_DIR}")
    print(f"  Rescue dir   : {rescue_dir}")
    print(f"  Model        : {MODEL_DIR}")
    print(f"  Output       : {denoised_dir}")

    ok, _ = run_deepdenoiser_predict(
        data_dir=rescue_dir, csv_path=rescue_csv_path, model_dir=MODEL_DIR,
        output_dir=denoised_dir, deepdenoiser_dir=DEEPDENOISER_DIR,
        sampling_rate=TARGET_FS,
    )
    sys.exit(0 if ok else 1)


RUN_DIR, STAMP = create_run_dir(os.path.join(OUTPUT_DIR, EVENT_SLUG))
log_file, log_path = setup_logging(
    RUN_DIR,
    script_name="03c_denoiser_event_data.py",
    extra_info=f"EVENT_TYPE: {EVENT_TYPE}  |  CSV: {CSV_PATH}",
)
set_matplotlib_defaults()

signal_dir  = os.path.join(RUN_DIR, "signal")
noise_dir   = os.path.join(RUN_DIR, "noise")
denoised_dir = os.path.join(RUN_DIR, "denoised")
os.makedirs(signal_dir,  exist_ok=True)
os.makedirs(noise_dir,   exist_ok=True)

# Connect to SDS archive
client_sds = connect_sds(SDS_ROOT)
if client_sds is None:
    print("[ERROR] SDS client unavailable — cannot extract waveforms. Exiting.")
    log_file.close()
    sys.exit(1)

ITP = int(PRE_PAD_S * TARGET_FS)          # sample index of onset in 30-s window
NT  = int(WINDOW_S  * TARGET_FS)          # total number of samples



# =============================================================================
# SECTION 3 — LOAD AND FILTER CATALOG
# =============================================================================

print(f"\n{'='*65}")
print(f"  STEP 1 — Load catalog: '{EVENT_TYPE}' rescue targets + training pairs")
print(f"{'='*65}")
print(f"  Rescue gate   : SNR >= {SNR_MIN}  "
      f"AND  SNR_full_median >= {SNR_FULL_MEDIAN_MIN}   (defines what needs denoising)")
print(f"  Training gate : SNR >= {TRAIN_SNR_MIN}  "
      f"AND  SNR_full_median >= {TRAIN_SNR_FULL_MEDIAN_MIN}   (defines clean training examples)")
if (TRAIN_SNR_MIN < SNR_MIN or
        TRAIN_SNR_FULL_MEDIAN_MIN < SNR_FULL_MEDIAN_MIN):
    print("  [WARN] Training gate is LOOSER than the rescue gate on at least one metric — "
          "check TRAIN_SNR_* if that's not intended.")

df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"Loaded {len(df):,} rows × {len(df.columns)} columns.")

# ── Rescue targets: EVENT_TYPE rows failing the rescue gate ─────────
df_event = df[df["event_type"] == EVENT_TYPE].copy()
print(f"After event_type filter: {len(df_event):,} '{EVENT_TYPE}' rows.")

mask_rescue_gate = (
    (df_event["SNR"]             >= SNR_MIN) &
    (df_event["SNR_full_median"] >= SNR_FULL_MEDIAN_MIN)
)
df_rescue = df_event[~mask_rescue_gate].copy()

print(f"\n  RESCUE (fail rescue gate, targets for denoising): "
      f"{len(df_rescue):,} rows  ({df_rescue['event_time'].nunique():,} events)")

# ── Training pairs: stricter gate, optionally pooled across multiple classes ──
print(f"\n  Training pairs pooled from: {TRAIN_EVENT_TYPES}"
      + (f"  (target mix: {TRAIN_MIX_RATIO})" if TRAIN_MIX_RATIO else "  (unbalanced — all qualifying rows)"))

per_type_avail = {}
for et in TRAIN_EVENT_TYPES:
    df_et  = df[df["event_type"] == et]
    mask_t = (
        (df_et["SNR"]             >= TRAIN_SNR_MIN) &
        (df_et["SNR_full_median"] >= TRAIN_SNR_FULL_MEDIAN_MIN)
    )
    per_type_avail[et] = df_et[mask_t].copy()
    print(f"    {et:<14s}: {len(per_type_avail[et]):,} rows qualify "
          f"(of {len(df_et):,} total '{et}' rows)")

if TRAIN_MIX_RATIO is None:
    df_good = pd.concat(per_type_avail.values())
else:
    # The class that is scarcest RELATIVE TO ITS TARGET SHARE sets the ceiling: we can subsample other classes down to match, never invent extra rows
    _limits = [len(per_type_avail[et]) / TRAIN_MIX_RATIO[et]
               for et in TRAIN_EVENT_TYPES if TRAIN_MIX_RATIO.get(et, 0) > 0]
    limiting_total = min(_limits) if _limits else 0
    rng   = np.random.default_rng(TRAIN_MIX_SEED)
    parts = []
    for et in TRAIN_EVENT_TYPES:
        target_frac = TRAIN_MIX_RATIO.get(et, 0)
        n_target    = int(round(limiting_total * target_frac))
        df_avail    = per_type_avail[et]
        if len(df_avail) > n_target:
            df_avail = df_avail.sample(n=n_target, random_state=TRAIN_MIX_SEED)
        parts.append(df_avail)
        print(f"    {et:<14s}: using {len(df_avail):,} rows "
              f"(target {target_frac:.0%} of {int(limiting_total):,})")
    df_good = pd.concat(parts)

print(f"\n  GOOD (training pairs, all classes combined) : "
      f"{len(df_good):,} rows  ({df_good['event_time'].nunique():,} events)")
print(f"\n  → Training data extracted from GOOD rows (clean signal examples, "
      f"may span multiple classes).")
print(f"  → predict.py will be run on RESCUE rows (low-SNR '{EVENT_TYPE}' → denoise → re-check gate).")



# =============================================================================
# SECTION 4 — EXTRACT SIGNAL WINDOWS  →  signal/*.npz
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 2 — Extract signal windows from SDS")
print(f"{'='*65}")
print(f"  Window: [{-PRE_PAD_S:.0f}s, +{WINDOW_S - PRE_PAD_S:.0f}s] around detection onset")
print(f"  itp   : sample {ITP}  (= {PRE_PAD_S} s × {TARGET_FS} Hz)")
print(f"  Target: {NT} samples at {TARGET_FS} Hz  ({WINDOW_S} s total)")


n_signal_ok    = 0
n_signal_skip  = 0
signal_records = []   # list of {"fname": ..., "channels": ..., "net_sta": ...}

for idx, row in df_good.iterrows():
    net   = row["network"]
    sta   = row["station"]
    chan  = row["channel"]          # e.g. "HHZ"
    loc   = ""                      # location code (empty for most stations)
    t_on  = UTCDateTime(row["det_starttime"])

    # Signal window: PRE_PAD_S before onset → WINDOW_S total
    t_sig_start = t_on - PRE_PAD_S
    t_sig_end   = t_sig_start + WINDOW_S

    fname_sig  = f"signal_{net}_{sta}_{chan}_{STAMP}_{idx}.npz"
    fpath_sig  = os.path.join(signal_dir, fname_sig)
    channel_id = f"{net}.{sta}"    # used to match signal ↔ noise from same station

    try:
        data3 = load_3component(client_sds, net, sta, loc, chan, t_sig_start, t_sig_end,
                                target_fs=TARGET_FS, window_s=WINDOW_S, horizontal_suffixes=HORIZONTAL_SUFFIXES)

        # reject only degenerate waveforms (data gaps, all-zeros)
        z_std = np.std(data3[:, 2])
        if z_std == 0 or not np.isfinite(z_std):
            n_signal_skip += 1
            continue

        np.savez(fpath_sig, data=data3, itp=np.int32(ITP), channels=channel_id)

        signal_records.append({"fname": fname_sig, "channels": channel_id})
        n_signal_ok += 1

        if n_signal_ok % 50 == 0:
            print(f"  [{n_signal_ok:4d} saved | {n_signal_skip:4d} skipped]  "
                  f"last: {net}.{sta}")

    except Exception as e:
        n_signal_skip += 1
        if n_signal_skip <= 10:
            print(f"  [SKIP] {net}.{sta} det={row['det_starttime'][:19]}  — {e}")

print(f"\n  Signal files saved : {n_signal_ok}")
print(f"  Signal files skipped: {n_signal_skip}")



# =============================================================================
# SECTION 5 — EXTRACT NOISE WINDOWS  →  noise/*.npz
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 3 — Extract noise windows from SDS")
print(f"{'='*65}")
print(f"  Noise window: [{-(NOISE_OFFSET_S + WINDOW_S):.0f}s, "
      f"-{NOISE_OFFSET_S:.0f}s] before detection onset")

n_noise_ok    = 0
n_noise_skip  = 0
noise_records = []

for idx, row in df_good.iterrows():
    net   = row["network"]
    sta   = row["station"]
    chan  = row["channel"]
    loc   = ""
    t_on  = UTCDateTime(row["det_starttime"])

    # Noise window ends NOISE_OFFSET_S before onset
    t_noise_end   = t_on  - NOISE_OFFSET_S
    t_noise_start = t_noise_end - WINDOW_S

    fname_noise  = f"noise_{net}_{sta}_{chan}_{STAMP}_{idx}.npz"
    fpath_noise  = os.path.join(noise_dir, fname_noise)
    channel_id   = f"{net}.{sta}"

    try:
        data3 = load_3component(client_sds, net, sta, loc, chan, t_noise_start, t_noise_end,
                                target_fs=TARGET_FS, window_s=WINDOW_S, horizontal_suffixes=HORIZONTAL_SUFFIXES)

        # Sanity check: noise window should have low energy (no hidden event)
        z_std = np.std(data3[:, 2])
        if z_std == 0 or not np.isfinite(z_std):
            n_noise_skip += 1
            continue

        np.savez(fpath_noise, data=data3, channels=channel_id)

        noise_records.append({"fname": fname_noise, "channels": channel_id})
        n_noise_ok += 1

    except Exception as e:
        n_noise_skip += 1
        if n_noise_skip <= 10:
            print(f"  [SKIP noise] {net}.{sta} det={row['det_starttime'][:19]}  — {e}")

print(f"  Noise files saved  : {n_noise_ok}")
print(f"  Noise files skipped: {n_noise_skip}")


# =============================================================================
# SECTION 5b — VALIDATE SIGNAL/NOISE STATION COVERAGE
# =============================================================================
# data_reader.py samples ONE noise file,per station on the fly during training, keyed by the 'channels' (NET.STA) value
#  -> it has no fallback if a station has signal windows but zero matching noise windows, and crashes deep in training

signal_stations = {r["channels"] for r in signal_records}
noise_stations   = {r["channels"] for r in noise_records}
orphan_stations  = signal_stations - noise_stations

if orphan_stations:
    n_before = len(signal_records)
    signal_records = [r for r in signal_records if r["channels"] not in orphan_stations]
    n_dropped = n_before - len(signal_records)
    print(f"\n  [WARN] {len(orphan_stations)} station(s) have signal windows but "
          f"ZERO matching noise windows — the DataReader would crash on these "
          f"mid-training. Dropping {n_dropped} orphaned signal row(s):")
    for st in sorted(orphan_stations):
        print(f"           {st}")
else:
    print("\n  [OK] Every station in signal_list.csv has at least one matching noise window.")


# =============================================================================
# SECTION 6 — WRITE CSV INDEX FILES
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 4 — Write CSV index files")
print(f"{'='*65}")

signal_csv_path = os.path.join(RUN_DIR, "signal_list.csv")
noise_csv_path  = os.path.join(RUN_DIR, "noise_list.csv")
rescue_csv_path = os.path.join(RUN_DIR, "rescue_list.csv")

# signal_list.csv and noise_list.csv: fname + channels (used by DataReader for training)
pd.DataFrame(signal_records).to_csv(signal_csv_path, index=False)
pd.DataFrame(noise_records ).to_csv(noise_csv_path,  index=False)

print(f"  [SAVED] {signal_csv_path}  ({len(signal_records)} entries — training signals)")
print(f"  [SAVED] {noise_csv_path}   ({len(noise_records)} entries — training noise)")

# =============================================================================
# SECTION 6b — EXTRACT RESCUE TARGETS  →  rescue/*.npz
# Rows of EVENT_TYPE that FAIL the quality gate: extract their raw waveforms so DeepDenoiser can denoise them
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 4b — Extract rescue targets from SDS  "
      f"(low-SNR '{EVENT_TYPE}' rows)")
print(f"{'='*65}")
print(f"  Source: {len(df_rescue):,} rows that failed the quality gate")

rescue_dir = os.path.join(RUN_DIR, "rescue")
os.makedirs(rescue_dir, exist_ok=True)

n_rescue_ok    = 0
n_rescue_skip  = 0
rescue_records = []

for idx, row in df_rescue.iterrows():
    net  = row["network"]
    sta  = row["station"]
    chan = row["channel"]
    loc  = ""
    t_on = UTCDateTime(row["det_starttime"])

    t_start = t_on - PRE_PAD_S
    t_end   = t_start + WINDOW_S

    fname_rescue  = f"rescue_{net}_{sta}_{chan}_{STAMP}_{idx}.npz"
    fpath_rescue  = os.path.join(rescue_dir, fname_rescue)
    channel_id    = f"{net}.{sta}"

    try:
        data3 = load_3component(client_sds, net, sta, loc, chan, t_start, t_end,
                                target_fs=TARGET_FS, window_s=WINDOW_S, horizontal_suffixes=HORIZONTAL_SUFFIXES)

        # Keep only if the trace is not flat (degenerate waveform)
        if np.std(data3[:, 2]) == 0 or not np.isfinite(np.std(data3[:, 2])):
            n_rescue_skip += 1
            continue

        np.savez(fpath_rescue, data=data3, itp=np.int32(ITP),
                 channels=channel_id)
        rescue_records.append({"fname": fname_rescue})
        n_rescue_ok += 1

        if n_rescue_ok % 100 == 0:
            print(f"  [{n_rescue_ok:4d} saved | {n_rescue_skip:4d} skipped]  "
                  f"last: {net}.{sta}")

    except Exception as e:
        n_rescue_skip += 1
        if n_rescue_skip <= 10:
            print(f"  [SKIP] {net}.{sta} det={row['det_starttime'][:19]}  — {e}")

print(f"  Rescue files saved  : {n_rescue_ok}")
print(f"  Rescue files skipped: {n_rescue_skip}")

# rescue_list.csv: fname only -> used by DataReader_pred (predict.py) for inference
pd.DataFrame(rescue_records).to_csv(rescue_csv_path, index=False)
print(f"  [SAVED] {rescue_csv_path}  ({len(rescue_records)} entries — rescue targets)")

# How to use these files for DeepDenoiser training:
print(f"""
  ── Training DeepDenoiser on your '{EVENT_TYPE}' data ─────────────────
  Once having a validated dataset, run from {DEEPDENOISER_DIR}:

    python train.py \\
      --train_signal_dir {signal_dir}/ \\
      --train_signal_list {signal_csv_path} \\
      --train_noise_dir  {noise_dir}/ \\
      --train_noise_list {noise_csv_path} \\
      --epochs 50 --batch_size 20 --learning_rate 0.001 \\
      --model_dir <checkpoint_to_finetune_from>   # optional: fine-tune pretrained
  ─────────────────────────────────────────────────────────────────────""")



# =============================================================================
# SECTION 7 — (OPTIONAL) RUN DEEPDENOISER INFERENCE
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 5 — DeepDenoiser inference")
print(f"{'='*65}")

if MODEL_DIR is None:
    print("  MODEL_DIR is None — skipping inference.")
    print("  Set MODEL_DIR to a trained checkpoint folder, then re-run.")
    print("  The rescue targets are ready in rescue/ and rescue_list.csv.")

elif len(rescue_records) == 0:
    print("  No rescue targets were extracted — nothing to denoise.")

else:
    print(f"  Running predict.py on {len(rescue_records)} rescue targets ...")
    print(f"  Input  : low-SNR '{EVENT_TYPE}' waveforms (failed quality gate)")
    print(f"  Model  : {MODEL_DIR}")
    print(f"  Output : {denoised_dir}")
    print(f"  After denoising, recompute SNR and re-apply the quality gate")
    print(f"  to identify which rescued events can join the training set.")

    run_deepdenoiser_predict(
        data_dir=rescue_dir, csv_path=rescue_csv_path, model_dir=MODEL_DIR,
        output_dir=denoised_dir, deepdenoiser_dir=DEEPDENOISER_DIR,
        sampling_rate=TARGET_FS,
    )



# =============================================================================
# SECTION 8 — SUMMARY
# =============================================================================

print(f"\n{'='*65}")
print("  SUMMARY")
print(f"{'='*65}")
print(f"  Rescue target class                : {EVENT_TYPE}")
print(f"  '{EVENT_TYPE}' rows total{'':<{max(1, 18-len(EVENT_TYPE))}}: {len(df_event):>6,}")
print(f"  └─ RESCUE (fail rescue gate, to denoise) : {len(df_rescue):>6,}")
print(f"  Training pairs (GOOD, stricter gate), by class:")
for et in TRAIN_EVENT_TYPES:
    n_et = int((df_good['event_type'] == et).sum()) if 'event_type' in df_good.columns else 0
    print(f"    {et:<14s}: {n_et:>6,}")
print(f"  GOOD total (all classes combined)  : {len(df_good):>6,}")
print(f"  Signal .npz extracted              : {n_signal_ok:>6,}  → {signal_dir}/")
if len(signal_records) != n_signal_ok:
    print(f"  Signal rows in signal_list.csv     : {len(signal_records):>6,}  "
          f"({n_signal_ok - len(signal_records)} dropped — orphan stations, see STEP 4 warning above)")
print(f"  Noise  .npz  (training)           : {n_noise_ok:>6,}  → {noise_dir}/")
print(f"  Rescue .npz  (inference targets)  : {n_rescue_ok:>6,}  → {rescue_dir}/")
print(f"  signal_list.csv                   : {signal_csv_path}")
print(f"  noise_list.csv                    : {noise_csv_path}")
print(f"  rescue_list.csv                   : {rescue_csv_path}")
if MODEL_DIR is not None:
    print(f"  Denoised output                   : {denoised_dir}/")
print(f"\n  Next steps:")
print(f"    1. Visually inspect a few signal/noise pairs to check quality")
print(f"    2. Train DeepDenoiser (see training command printed above)")
print(f"    3. Set MODEL_DIR and re-run this script to denoise all '{EVENT_TYPE}' rescue rows")
print(f"    4. Recompute SNR on denoised waveforms → rescue low-SNR events (03d)")
print(f"    5. Recompute 99 features on rescued events → grow training set for 06c")

print(f"\n{'='*70}")
print(f"  Run finished  : {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Run folder    : {RUN_DIR}")
print(f"  Log           : {log_path}")
print(f"{'='*70}")

log_file.close()
