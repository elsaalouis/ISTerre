"""
03c_denoiser_icequake_data.py
==============================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : June 2026

Prepare data for fine-tuning DeepDenoiser (Zhu et al., 2019) on ice quake signals, and run DeepDenoiser inference to produce denoised waveforms

Why?
----
The ice quake class only has 894 quality-filtered windows (3 % of the training set)
Many more ice quake events exist in the catalog but fail the SNR quality gate
 -> denoising those low-SNR signals may boost their SNR enough to pass the gate, growing the training set for scripts 06a/06b

Reference
---------
 Zhu et al. (2019). Seismic signal denoising and decomposition using deep neural networks.

Pipeline
--------
  STEP 1 — Load catalog CSV, keep quality ice quake rows
  STEP 2 — Extract 30 s signal windows from SDS (one .npz per event × station)
  STEP 3 — Extract 30 s pre-event noise windows from the same stations
  STEP 4 — Write signal_list.csv and noise_list.csv (DeepDenoiser index files)
  STEP 5 — Run DeepDenoiser inference via predict.py

Output layout
-------------
  outputs_03c/run_YYYYMMDD_HHMMSS/
      signal/              ← signal .npz files
      noise/               ← noise  .npz files
      signal_list.csv      ← fname + channels, for DeepDenoiser training/inference
      noise_list.csv       ← fname + channels, for DeepDenoiser training
      pred_list.csv        ← fname only, used by predict.py
      denoised/            ← predict.py outputs (only if MODEL_DIR is set)
      run.log

.npz file (required by DeepDenoiser data_reader.py)
---------------------------------------------------
  data      float32  shape (3000, 3)  — 30 s at 100 Hz, 3 components (E, N, Z).
                     If only Z is available on the station, it is replicated into
                     all three slots so the existing DeepDenoiser code needs no edit.
  itp       int      — sample index of the signal onset within the 30 s window.
                     = PRE_PAD_S × TARGET_FS  (e.g. 10 s × 100 = sample 1000)
  channels  str      — station key "NET.STA" used to match signal ↔ noise files
                     from the same station during training.
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# -- Inputs -------------------------------------------------------------------
CSV_PATH = (
    "/data/failles/louisels/project/results/outputs_04a/groult/"
    "run_20260531_104936/catalog_windows_20260531_104936.csv"
)

SDS_ROOT    = "/data/sig/SDS"
OUTPUT_DIR  = "/data/failles/louisels/project/results/outputs_03c"

# -- Quality gate (same thresholds as 03b / 06b) ------------------------------
SNR_FULL_MEAN_MIN  = 2.70
SNR_S2N_MEDIAN_MIN = 20.99

# -- DeepDenoiser directory (where predict.py lives) --------------------------
DEEPDENOISER_DIR = "/data/failles/louisels/project/src/deepdenoiser"

# -- Trained model checkpoint for inference -----------------------------------
# Set to the path of a trained checkpoint folder, e.g.: MODEL_DIR = "/data/failles/louisels/project/results/deepdenoiser/log/260601-120000"
# Set to None to SKIP inference and prepare data only
MODEL_DIR = None

# -- Waveform extraction parameters ------------------------------------------
TARGET_FS  = 100      # [Hz]  target sampling rate (DeepDenoiser default)
WINDOW_S   = 30       # [s]   total window length  (= 3000 samples at 100 Hz)
PRE_PAD_S  = 10       # [s]   seconds of pre-signal padding → itp = PRE_PAD_S × TARGET_FS

# -- Noise window extraction --------------------------------------------------
# The noise window ends NOISE_OFFSET_S before the detection onset
NOISE_OFFSET_S = 120  # [s] gap between noise window end and detection onset noise window

# -- Channel fallback strategy ------------------------------------------------
# For each Z-channel in the catalog we try to also load the two horizontal components by replacing the last letter (Z → N, Z → E or Z → 2, Z → 1)
HORIZONTAL_SUFFIXES = [("N", "E"), ("2", "1")]



# =============================================================================
# SECTION 2 — SETUP
# =============================================================================

import os
import sys
import subprocess
import traceback
import warnings

import numpy as np
import pandas as pd
from obspy import UTCDateTime

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_setup import create_run_dir, setup_logging, connect_sds, set_matplotlib_defaults
from preprocessing import load_3component


RUN_DIR, STAMP = create_run_dir(OUTPUT_DIR)
log_file, log_path = setup_logging(
    RUN_DIR,
    script_name="03c_denoiser_icequake_data.py",
    extra_info=f"CSV: {CSV_PATH}",
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
print("  STEP 1 — Load catalog and split ice quakes into good / rescue")
print(f"{'='*65}")
print(f"  Quality gate : SNR_full_mean >= {SNR_FULL_MEAN_MIN}  "
      f"AND  SNR_s2n_median >= {SNR_S2N_MEDIAN_MIN}")

df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"Loaded {len(df):,} rows × {len(df.columns)} columns.")

# Keep only ice quake rows
df_iq = df[df["event_type"] == "ice quake"].copy()
print(f"After event_type filter: {len(df_iq):,} ice quake rows.")

# Split into two populations — quality gate is the boundary
mask_q = (
    (df_iq["SNR_full_mean"]  >= SNR_FULL_MEAN_MIN) &
    (df_iq["SNR_s2n_median"] >= SNR_S2N_MEDIAN_MIN)
)

# Good: already pass the gate → used as clean training examples for train.py
df_iq_good = df_iq[mask_q].copy()

# Rescue targets: fail the gate → what we actually want to denoise
df_iq_rescue = df_iq[~mask_q].copy()

print(f"\n  GOOD (pass gate, used for training data) : "
      f"{len(df_iq_good):,} rows  ({df_iq_good['event_time'].nunique():,} events)")
print(f"  RESCUE (fail gate, targets for denoising): "
      f"{len(df_iq_rescue):,} rows  ({df_iq_rescue['event_time'].nunique():,} events)")
print(f"\n  → Training data extracted from GOOD rows (clean signal examples).")
print(f"  → predict.py will be run on RESCUE rows (low-SNR → denoise → re-check gate).")



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

for idx, row in df_iq_good.iterrows():
    net   = row["network"]
    sta   = row["station"]
    chan  = row["channel"]          # e.g. "HHZ"
    loc   = ""                      # location code (empty for most SISMalp stations)
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

        # Flat-trace guard: reject only degenerate waveforms (data gaps, all-zeros)
        z_std = np.std(data3[:, 2])
        if z_std == 0 or not np.isfinite(z_std):
            n_signal_skip += 1
            continue

        np.savez(fpath_sig, data=data3, itp=np.int32(ITP), channels=np.bytes_(channel_id))

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

for idx, row in df_iq_good.iterrows():
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

        np.savez(fpath_noise, data=data3, channels=np.bytes_(channel_id))

        noise_records.append({"fname": fname_noise, "channels": channel_id})
        n_noise_ok += 1

    except Exception as e:
        n_noise_skip += 1
        if n_noise_skip <= 10:
            print(f"  [SKIP noise] {net}.{sta} det={row['det_starttime'][:19]}  — {e}")

print(f"  Noise files saved  : {n_noise_ok}")
print(f"  Noise files skipped: {n_noise_skip}")



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
# IQ that FAIL the quality gate: extract their raw waveforms so DeepDenoiser can denoise them
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 4b — Extract rescue targets from SDS  (low-SNR ice quakes)")
print(f"{'='*65}")
print(f"  Source: {len(df_iq_rescue):,} rows that failed the quality gate")

rescue_dir = os.path.join(RUN_DIR, "rescue")
os.makedirs(rescue_dir, exist_ok=True)

n_rescue_ok    = 0
n_rescue_skip  = 0
rescue_records = []

for idx, row in df_iq_rescue.iterrows():
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
        data3 = load_3component(client_sds, net, sta, loc, chan,
                                t_start, t_end,
                                target_fs=TARGET_FS, window_s=WINDOW_S,
                                horizontal_suffixes=HORIZONTAL_SUFFIXES)

        # Keep only if the trace is not flat (degenerate waveform)
        if np.std(data3[:, 2]) == 0 or not np.isfinite(np.std(data3[:, 2])):
            n_rescue_skip += 1
            continue

        np.savez(fpath_rescue, data=data3, itp=np.int32(ITP),
                 channels=np.bytes_(channel_id))
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

# rescue_list.csv: fname only — used by DataReader_pred (predict.py) for inference
pd.DataFrame(rescue_records).to_csv(rescue_csv_path, index=False)
print(f"  [SAVED] {rescue_csv_path}  ({len(rescue_records)} entries — rescue targets)")

# How to use these files for DeepDenoiser training:
print(f"""
  ── Training DeepDenoiser on your ice quake data ──────────────────────
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
    os.makedirs(denoised_dir, exist_ok=True)
    predict_script = os.path.join(DEEPDENOISER_DIR, "predict.py")

    if not os.path.isfile(predict_script):
        print(f"  [ERROR] predict.py not found at {predict_script}")
        print("  Make sure DEEPDENOISER_DIR points to the deepdenoiser source folder.")
    else:
        print(f"  Running predict.py on {len(rescue_records)} rescue targets ...")
        print(f"  Input  : low-SNR ice quake waveforms (failed quality gate)")
        print(f"  Model  : {MODEL_DIR}")
        print(f"  Output : {denoised_dir}")
        print(f"  After denoising, recompute SNR and re-apply the quality gate")
        print(f"  to identify which rescued events can join the training set.")

        cmd = [
            sys.executable, predict_script,
            "--format",        "numpy",
            "--data_dir",      rescue_dir,
            "--data_list",     rescue_csv_path,
            "--model_dir",     MODEL_DIR,
            "--output_dir",    denoised_dir,
            "--save_signal",        # save denoised .npz → next script checks SNR
            "--sampling_rate", str(TARGET_FS),
        ]

        print(f"  Command  : {' '.join(cmd)}\n")
        try:
            subprocess.run(cmd, check=True)
            print(f"\n  [OK] DeepDenoiser inference completed.")
            print(f"       Denoised files → {denoised_dir}/results/")
        except subprocess.CalledProcessError as e:
            print(f"\n  [ERROR] predict.py exited with code {e.returncode}")
            traceback.print_exc()



# =============================================================================
# SECTION 8 — SUMMARY
# =============================================================================

print(f"\n{'='*65}")
print("  SUMMARY")
print(f"{'='*65}")
print(f"  Ice quake rows total              : {len(df_iq):>6,}")
print(f"  ├─ GOOD  (pass gate, training)    : {len(df_iq_good):>6,}")
print(f"  └─ RESCUE (fail gate, to denoise) : {len(df_iq_rescue):>6,}")
print(f"  Signal .npz  (training, clean)    : {n_signal_ok:>6,}  → {signal_dir}/")
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
print(f"    3. Set MODEL_DIR and re-run this script to denoise all ice quakes")
print(f"    4. Recompute SNR on denoised waveforms → rescue low-SNR events")
print(f"    5. Recompute 99 features on rescued events → grow training set for 06b")

print(f"\n{'='*70}")
print(f"  Run finished  : {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Run folder    : {RUN_DIR}")
print(f"  Log           : {log_path}")
print(f"{'='*70}")

log_file.close()
