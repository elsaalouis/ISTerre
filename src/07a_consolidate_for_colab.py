"""
07a_consolidate_for_colab.py
=============================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : July 2026

Goal
----
Pack the many small per-sample .npz files produced by 07a_spectrogram_dataset_build.py into a handful of large archives (one per
train/val/test split), so the dataset can be uploaded to and read from Google Drive without hitting Drive's FUSE-mount limitations.

Usage
-----
  1. Set RUN_DIR below to the run folder printed at the end of your 07a_spectrogram_dataset_build.py run 
  2. Run this script on the cluster: python3 07a_consolidate_for_colab.py
  3. Upload ONLY the contents of RUN_DIR/colab_package/ to Google Drive

Output layout
-------------
  <RUN_DIR>/colab_package/
      spectrograms_train.npz   <- images (N,H,W,3) float16 + labels + metadata
      spectrograms_val.npz
      spectrograms_test.npz
      freq_axis.npy
      time_axis.npy
      image_list.csv           <- copied for reference/provenance only
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# -- EDIT: paste the run folder path printed at the end of your 07a run -------
RUN_DIR = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\07a_spectrogram_dataset_build\run_20260722_095625"

# -- Storage dtype for the packed image arrays --------------------------------
PACK_DTYPE = "float16"   # halves size vs. float32; safe for log-power dB values

# -- Output subfolder (created inside RUN_DIR) ---------------------------------
PACKAGE_DIRNAME = "colab_package"



# =============================================================================
# SECTION 2 — SETUP
# =============================================================================

import os
import sys
import shutil

import numpy as np
import pandas as pd


if not os.path.isdir(RUN_DIR):
    print(f"[ERROR] RUN_DIR not found: {RUN_DIR}")
    print("        Edit RUN_DIR in Section 1 to point at your 07a run folder.")
    sys.exit(1)

images_dir     = os.path.join(RUN_DIR, "images")
manifest_path  = os.path.join(RUN_DIR, "image_list.csv")
freq_axis_path = os.path.join(RUN_DIR, "freq_axis.npy")
time_axis_path = os.path.join(RUN_DIR, "time_axis.npy")

for label, path in [("images/", images_dir), ("image_list.csv", manifest_path),
                    ("freq_axis.npy", freq_axis_path), ("time_axis.npy", time_axis_path)]:
    if not os.path.exists(path):
        print(f"[ERROR] Expected {label} not found at: {path}")
        print("        Is RUN_DIR pointing at a completed 07a run?")
        sys.exit(1)

package_dir = os.path.join(RUN_DIR, PACKAGE_DIRNAME)
os.makedirs(package_dir, exist_ok=True)

print("=" * 70)
print("  07a_consolidate_for_colab.py")
print(f"  RUN_DIR : {RUN_DIR}")
print(f"  Output  : {package_dir}")
print("=" * 70)



# =============================================================================
# SECTION 3 — LOAD MANIFEST
# =============================================================================

manifest = pd.read_csv(manifest_path)
print(f"\nManifest: {len(manifest):,} rows")
print(manifest.groupby(["split", "event_type"]).size().unstack(fill_value=0).to_string())

freq_axis = np.load(freq_axis_path)
time_axis = np.load(time_axis_path)
n_freq, n_time = len(freq_axis), len(time_axis)
print(f"\nImage shape per sample: ({n_freq}, {n_time}, 3)")



# =============================================================================
# SECTION 4 — PACK EACH SPLIT
# =============================================================================

meta_cols = ["event_time", "network", "station", "channel", "det_starttime"]

for split_name in ["train", "val", "test"]:
    rows = manifest[manifest["split"] == split_name].reset_index(drop=True)
    n = len(rows)
    print(f"\n{'='*65}")
    print(f"  Packing split '{split_name}': {n:,} samples")
    print(f"{'='*65}")

    if n == 0:
        print(f"  [WARN] No rows for split '{split_name}' — skipping.")
        continue

    images = np.empty((n, n_freq, n_time, 3), dtype=PACK_DTYPE)
    labels = np.empty(n, dtype=object)
    meta   = {col: np.empty(n, dtype=object) for col in meta_cols}

    n_missing = 0
    for i, row in rows.iterrows():
        fpath = os.path.join(images_dir, row["fname"])
        try:
            with np.load(fpath) as d:
                images[i] = d["image"].astype(PACK_DTYPE)
        except Exception as e:
            n_missing += 1
            images[i] = 0
            if n_missing <= 10:
                print(f"    [WARN] Could not load {row['fname']}: {e}")

        labels[i] = row["event_type"]
        for col in meta_cols:
            meta[col][i] = row[col]

        if (i + 1) % 5000 == 0:
            print(f"    [{i+1:,}/{n:,}] packed")

    if n_missing:
        print(f"  [WARN] {n_missing:,}/{n:,} files failed to load "
              f"(kept as zero-filled placeholders — check RUN_DIR/images/ integrity).")

    out_path = os.path.join(package_dir, f"spectrograms_{split_name}.npz")
    np.savez(
        out_path,
        images=images,
        labels=labels.astype(str),
        **{col: meta[col].astype(str) for col in meta_cols},
    )

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  [SAVED] {out_path}  ({size_mb:.1f} MB, dtype={PACK_DTYPE})")



# =============================================================================
# SECTION 5 — COPY SMALL REFERENCE FILES
# =============================================================================

shutil.copy2(freq_axis_path, os.path.join(package_dir, "freq_axis.npy"))
shutil.copy2(time_axis_path, os.path.join(package_dir, "time_axis.npy"))
shutil.copy2(manifest_path,  os.path.join(package_dir, "image_list.csv"))

print(f"\n[SAVED] freq_axis.npy, time_axis.npy, image_list.csv -> {package_dir}")



# =============================================================================
# SECTION 6 — SUMMARY
# =============================================================================

total_size_mb = sum(
    os.path.getsize(os.path.join(package_dir, f))
    for f in os.listdir(package_dir)
) / 1e6

print(f"\n{'='*70}")
print("  DONE")
print(f"{'='*70}")
print(f"  Package folder : {package_dir}")
print(f"  Files          : {len(os.listdir(package_dir))}  (was {len(manifest):,} individual .npz)")
print(f"  Total size     : {total_size_mb:.1f} MB")
print(f"""
  Next steps
  ----------
  1. Upload the CONTENTS of {package_dir}/ to Google Drive, e.g.:
       MyDrive/colab_cnn_training_spectrogram/
           spectrograms_train.npz
           spectrograms_val.npz
           spectrograms_test.npz
           freq_axis.npy
           time_axis.npy
           image_list.csv
     (do NOT upload the original images/ folder — that's the 50k-file one
     that breaks Drive's FUSE mount; it's no longer needed once this
     package is uploaded)
  2. Open 07b_train_cnn_classifier_colab.ipynb and run through it as usual —
     it now reads these 3 packed archives instead of individual files.
""")
print(f"{'='*70}")
