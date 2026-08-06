"""
07a_spectrogram_dataset_build.py
=================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : July 2026

Goal
----
Build a fixed-size 3-component spectrogram image dataset for CNN-based event-type
classification (earthquake / rockslide / ice quake / noise / regional)
 -> starting from the quality-filtered detection windows already produced by
    04a_sta_lta_catalog_windowing.py (local classes), 04d_noise_window_extraction.py
    (noise, 4th class) and 04c_regional_event_extraction.py (regional, 5th class).

 Training happens separately on Google Colab (07b_train_cnn_classifier_colab.ipynb)
 This script only extracts waveforms from the cluster, builds spectrogram images, and writes them to disk together with a manifest CSV, ready to be uploaded to Google Drive

Pipeline
--------
  1. Load catalog_windows CSV (04a output) + optional NOISE_CSV (04d) + optional
     REGIONAL_CSV (04c), filter by TARGET_CLASSES, apply the SAME explicit
     SNR-based quality gate as 03c/03d/06b/06c to the local classes + regional
     (noise skips it — see the FILTER_QUALITY comment in Section 1), concatenate
  2. For each row: fetch Z/N/E waveforms from SDS around the (already-refined) detection onset
                   remove instrument response -> velocity [m/s]
                   resample to a common rate
                   trim/pad to a fixed-duration window
  3. Compute a log-power spectrogram per channel (fixed STFT params -> every image has exactly the same shape) and stack [Z, N, E] like an RGB image
  4. Save each image as one .npz file + light metadata
  5. Split events into train/val/test (stratified by class, split BY EVENT so no station from the same event leaks across splits) and write image_list.csv

Output layout
-------------
  outputs_07a/run_YYYYMMDD_HHMMSS/
      images/*.npz         <- one file per (event x station) sample: image (n_freq, n_time, 3) float32, dB-scaled
      freq_axis.npy         <- shared frequency axis [Hz], same for every image
      time_axis.npy         <- shared time axis [s] relative to window start
      image_list.csv         <- fname, event metadata, split (train/val/test)
      run.log

Next step: upload the run folder to Google Drive and point 07b_train_cnn_classifier_colab.ipynb at it (see printed instructions at the end)
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# -- Input CSV (output of 04a_sta_lta_catalog_windowing.py) -------------------
CSV_PATH = "/data/failles/louisels/project/results/outputs_04a/all-99-features-recent+3C/catalog_windows_20260708_174019.csv"

# -- Noise CSV (output of 04d_noise_window_extraction.py, optional 4th class) --
NOISE_CSV = "/data/failles/louisels/project/results/outputs_04d/run_20260803_174514/noise_windows_20260803_174514.csv"

# -- Regional CSV (output of 04c_regional_event_extraction.py, optional 5th class) --
REGIONAL_CSV = "/data/failles/louisels/project/results/outputs_04c/run_20260805_135512/regional_windows_20260805_135512.csv"

# -- Paths ---------------------------------------------------------------------
SDS_ROOT    = "/data/sig/SDS"
ISTERRE_URL = "http://ist-sc3-geobs.osug.fr:8080"
OUTPUT_DIR  = "/data/failles/louisels/project/results/outputs_07a"

# -- Classes to keep -------------------------------------------------------------
TARGET_CLASSES = ["earthquake", "rockslide", "ice quake", "noise", "regional"]

# -- Quality filtering -----------------------------------------------------------
FILTER_QUALITY      = True
SNR_MIN             = 1.70    # 05b Tier 2 — metric 'SNR'
SNR_FULL_MEDIAN_MIN = 1.99    # 05b Tier 2 — metric 'SNR_full_median'

# -- Waveform extraction: fixed window anchored on the (kurtosis-refined) onset --
# det_starttime in the CSV is already the refined onset from 04a. The window is [onset - WINDOW_PRE_S, onset + WINDOW_POST_S]
# ADAPT THESE to the data: check the det_duration_s distribution in the CSV first 
# -- WINDOW_POST_S must comfortably cover the longest events (rockslide codas in particular can run long) or the spectrogram will cut off real signal
TARGET_FS      = 100     # [Hz] common resampling rate for every trace
WINDOW_PRE_S   = 5       # [s] seconds of window BEFORE the onset
WINDOW_POST_S  = 95      # [s] seconds of window AFTER the onset
WINDOW_S       = WINDOW_PRE_S + WINDOW_POST_S
NT             = int(WINDOW_S * TARGET_FS)          # fixed number of samples per trace

# -- Channel fallback strategy (for horizontals N/E) ------------------------------
# For each Z-channel we try to also load the two horizontal components by replacing the last letter: Z -> N/E, or Z -> 2/1 (older naming)
#  -> if neither pair is available, the Z channel is duplicated into N and E slots so every image still has 3 channels
HORIZONTAL_SUFFIXES = [("N", "E"), ("2", "1")]

# -- Spectrogram (STFT) parameters ------------------------------------------------
# Fixed in samples (not seconds) because TARGET_FS is constant -> every image comes out with exactly the same (n_freq, n_time) shape, no resizing needed
SPEC_NPERSEG_S = 2.0     # [s] STFT segment length
SPEC_NOVERLAP_FRAC = 0.75   # fraction of SPEC_NPERSEG_S overlapping between segments
SPEC_NFFT      = 256     # zero-padded FFT length (>= nperseg for frequency resolution)
FREQ_MAX_KEEP  = 45.0    # [Hz] drop bins above this (avoid Nyquist-edge artifacts at fs=100)

SPEC_NPERSEG   = int(SPEC_NPERSEG_S * TARGET_FS)
SPEC_NOVERLAP  = int(SPEC_NPERSEG * SPEC_NOVERLAP_FRAC)

# -- Train / val / test split (by EVENT, stratified by class) --------------------
VAL_SIZE     = 0.20
TEST_SIZE    = 0.15
RANDOM_STATE = 42

# -- Debugging / quick sanity runs -------------------------------------------------
MAX_ROWS = 0   # 0 = process every row; set e.g. 50 for a quick end-to-end smoke test

# -- Output ------------------------------------------------------------------------
CHECKPOINT_EVERY = 100   # save a partial manifest CSV every N successfully processed rows (0 = disabled)



# =============================================================================
# SECTION 2 — SETUP
# =============================================================================

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.signal import spectrogram

from obspy import UTCDateTime

from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_setup import create_run_dir, setup_logging, connect_sds, connect_fdsn
from preprocessing import cosine_taper


RUN_DIR, STAMP = create_run_dir(OUTPUT_DIR)
log_file, log_path = setup_logging(
    RUN_DIR, "07a_spectrogram_dataset_build.py",
    extra_info=(f"CSV: {CSV_PATH}\nNOISE_CSV: {NOISE_CSV}\nREGIONAL_CSV: {REGIONAL_CSV}\n"
                f"Classes: {TARGET_CLASSES}  |  "
                f"Window: -{WINDOW_PRE_S}s..+{WINDOW_POST_S}s @ {TARGET_FS} Hz "
                f"({NT} samples)")
)

images_dir = os.path.join(RUN_DIR, "images")
os.makedirs(images_dir, exist_ok=True)


# --------- Connections ----------------
client_sds  = connect_sds(SDS_ROOT)
client_fdsn = connect_fdsn(ISTERRE_URL)

if client_sds is None:
    print("\n[ERROR] SDS client unavailable — cannot extract waveforms. Exiting.")
    log_file.close()
    sys.exit(1)
if client_fdsn is None:
    print("\n[ERROR] FDSN client unavailable — cannot fetch instrument inventory "
          "for response removal. Exiting.")
    log_file.close()
    sys.exit(1)



# =============================================================================
# SECTION 3 — LOAD AND FILTER CATALOG
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 1 — Loading catalog CSV(s)")
print(f"{'='*65}")

if not os.path.isfile(CSV_PATH):
    print(f"[ERROR] CSV not found: {CSV_PATH}")
    print("        Update CSV_PATH in Section 1 and rerun.")
    log_file.close()
    sys.exit(1)

df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"Loaded {len(df):,} rows x {df.shape[1]} columns from CSV_PATH.")

df = df[df["event_type"].isin(TARGET_CLASSES)].copy()
print(f"After class filter ({TARGET_CLASSES}): {len(df):,} rows.")

# -- Optional 5th class: regional (04c output), concatenated BEFORE the quality
# gate below — unlike noise (added AFTER the gate further down, since noise
# rows have SNR=NaN by construction), regional rows carry REAL computed SNR
# from 04c's own detection pipeline and need to pass the SAME gate as the local
# classes, not skip it. Same convention as 06b/06c.
if REGIONAL_CSV is not None:
    if os.path.isfile(REGIONAL_CSV):
        df_regional = pd.read_csv(REGIONAL_CSV, low_memory=False)
        df_regional = df_regional[df_regional["event_type"].isin(TARGET_CLASSES)].copy()
        print(f"Loaded {len(df_regional):,} regional rows from {os.path.basename(REGIONAL_CSV)}.")
        df = pd.concat([df, df_regional], ignore_index=True)
    else:
        print(f"[WARN] REGIONAL_CSV not found: {REGIONAL_CSV} — continuing without the regional class.")

if FILTER_QUALITY:
    n_before = len(df)
    # Explicit SNR-based gate (03c/03d/06b/06c), NOT the catalog's quality_ok
    # column — see the FILTER_QUALITY comment in Section 1 for why.
    if {"SNR", "SNR_full_median"}.issubset(df.columns):
        mask_quality = (df["SNR"] >= SNR_MIN) & (df["SNR_full_median"] >= SNR_FULL_MEDIAN_MIN)
        df = df[mask_quality].copy()
        print(f"After quality filter (SNR>={SNR_MIN}, SNR_full_median>={SNR_FULL_MEDIAN_MIN}): "
              f"{len(df):,} rows kept ({n_before - len(df):,} dropped).")
    else:
        print("[WARN] 'SNR'/'SNR_full_median' column(s) not found — skipping quality filter.")

required_cols = ["event_time", "event_type", "network", "station", "channel", "det_starttime"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print(f"[ERROR] CSV is missing required column(s): {missing}")
    log_file.close()
    sys.exit(1)

df = df.dropna(subset=required_cols).reset_index(drop=True)
print(f"After dropping rows with missing key columns: {len(df):,} rows.")

# -- Optional 4th class: noise (04d output), added AFTER the quality gate —
# noise rows have SNR=NaN by construction (they'd fail the mask above); 04d
# already guarantees each row is a real, locality-confirmed, catalog-clear
# detection with no SNR question to ask. Same convention as 06b/06c.
if NOISE_CSV is not None:
    if os.path.isfile(NOISE_CSV):
        df_noise = pd.read_csv(NOISE_CSV, low_memory=False)
        df_noise = df_noise[df_noise["event_type"].isin(TARGET_CLASSES)].copy()
        df_noise = df_noise.dropna(subset=required_cols).reset_index(drop=True)
        print(f"Loaded {len(df_noise):,} noise rows from {os.path.basename(NOISE_CSV)}.")
        df = pd.concat([df, df_noise], ignore_index=True)
    else:
        print(f"[WARN] NOISE_CSV not found: {NOISE_CSV} — continuing without the noise class.")

if MAX_ROWS > 0:
    df = df.iloc[:MAX_ROWS].copy()
    print(f"[DEBUG] MAX_ROWS={MAX_ROWS} -> restricted to first {len(df):,} rows.")

print("\nClass distribution:")
for cls, n in df["event_type"].value_counts().items():
    print(f"  {cls:<20s}  {n:6,} rows")

# Fetch instrument inventory — restricted to the networks that actually appear in the filtered catalog (not "*"), and fetched ONE NETWORK AT A TIME
print(f"\n{'='*65}")
print("  STEP 1b — Fetching instrument inventory (per network)")
print(f"{'='*65}")

t_min = pd.to_datetime(df["event_time"]).min()
t_max = pd.to_datetime(df["event_time"]).max()
inv_t_start = UTCDateTime((t_min - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
inv_t_end   = UTCDateTime((t_max + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))

networks = sorted(df["network"].dropna().unique())
print(f"Networks in catalog: {networks}")

inventory = None
failed_networks = []
for net in networks:
    try:
        inv_net = client_fdsn.get_stations(
            network=net, station="*",
            starttime=inv_t_start, endtime=inv_t_end,
            level="response",
        )
        n_sta = sum(len(n.stations) for n in inv_net.networks)
        inventory = inv_net if inventory is None else inventory + inv_net
        print(f"  [OK]   {net:<4s}  {n_sta} station(s)")
    except Exception as e:
        failed_networks.append(net)
        print(f"  [WARN] {net:<4s}  failed: {e}")

if failed_networks:
    print(f"\n[WARN] {len(failed_networks)} network(s) failed to fetch: {failed_networks}")
    print(f"       Rows for these networks will be skipped at the per-trace "
          f"response-removal step, not the whole run.")

if inventory is None:
    print("\n[ERROR] Could not fetch instrument inventory for ANY network — "
          "response removal impossible. Exiting.")
    log_file.close()
    sys.exit(1)

print(f"\nInventory ready: {len(networks) - len(failed_networks)}/{len(networks)} "
      f"network(s) loaded.")



# =============================================================================
# SECTION 4 — TRAIN / VAL / TEST SPLIT (BY EVENT, STRATIFIED)
# =============================================================================
# Computed once, up front, on the full filtered catalog -> baked into the
# manifest CSV so the Colab training notebook never needs to re-derive it
# (no sklearn / no access to the original catalog needed on Colab's side).

print(f"\n{'='*65}")
print("  STEP 2 — Train / val / test split (by event)")
print(f"{'='*65}")

event_info = (
    df.groupby("event_time")["event_type"]
    .agg(lambda x: x.mode().iloc[0])
    .reset_index()
    .rename(columns={"event_type": "event_type_label"})
)

min_class_events = event_info["event_type_label"].value_counts().min()
if min_class_events < 5:
    print(f"[WARN] Smallest class has only {min_class_events} event(s). "
          f"Stratified split may fail -> falling back to unstratified split.")
    stratify_col = None
else:
    stratify_col = event_info["event_type_label"]

train_ev, temp_ev = train_test_split(
    event_info, test_size=(VAL_SIZE + TEST_SIZE),
    stratify=stratify_col, random_state=RANDOM_STATE,
)

if stratify_col is not None and len(temp_ev) >= 5:
    stratify_temp = temp_ev["event_type_label"]
else:
    stratify_temp = None

rel_test_size = TEST_SIZE / (VAL_SIZE + TEST_SIZE)
val_ev, test_ev = train_test_split(
    temp_ev, test_size=rel_test_size,
    stratify=stratify_temp, random_state=RANDOM_STATE,
)

split_by_event = {}
for t in train_ev["event_time"]: split_by_event[t] = "train"
for t in val_ev["event_time"]:   split_by_event[t] = "val"
for t in test_ev["event_time"]:  split_by_event[t] = "test"

print(f"Events: train={len(train_ev):4d}  val={len(val_ev):4d}  test={len(test_ev):4d}")
for split_name, ev_df in [("train", train_ev), ("val", val_ev), ("test", test_ev)]:
    print(f"  {split_name:<6s} class distribution: "
          f"{dict(ev_df['event_type_label'].value_counts())}")



# =============================================================================
# SECTION 5 — SPECTROGRAM SHAPE (PRECOMPUTED — FIXED FOR EVERY SAMPLE)
# =============================================================================
# nperseg/noverlap/nfft/fs are all constants, and every trace is trimmed/padded
# to exactly NT samples -> the spectrogram shape is identical for every sample,
# computed once here rather than re-derived per row.

_dummy = np.zeros(NT, dtype=np.float32)
_f_full, _t_axis, _ = spectrogram(
    _dummy, fs=TARGET_FS, window="hann",
    nperseg=SPEC_NPERSEG, noverlap=SPEC_NOVERLAP, nfft=SPEC_NFFT,
    scaling="density", mode="psd",
)
_freq_keep_mask = _f_full <= FREQ_MAX_KEEP
FREQ_AXIS = _f_full[_freq_keep_mask]
TIME_AXIS = _t_axis
N_FREQ    = len(FREQ_AXIS)
N_TIME    = len(TIME_AXIS)

np.save(os.path.join(RUN_DIR, "freq_axis.npy"), FREQ_AXIS)
np.save(os.path.join(RUN_DIR, "time_axis.npy"), TIME_AXIS)

print(f"\nSpectrogram image shape (fixed for every sample): "
      f"({N_FREQ}, {N_TIME}, 3)  [freq x time x (Z,N,E)]")
print(f"  nperseg={SPEC_NPERSEG} ({SPEC_NPERSEG_S}s)  noverlap={SPEC_NOVERLAP}  "
      f"nfft={SPEC_NFFT}  freq range=[0, {FREQ_MAX_KEEP}] Hz")



# =============================================================================
# SECTION 6 — HELPER FUNCTIONS
# =============================================================================

def _process_trace(tr_raw, inventory, target_fs, t_start, t_end, nt):
    """
    Clean one raw trace -> response-removed velocity [m/s], resampled, trimmed
    to exactly `nt` samples.

    Returns
    -------
    np.ndarray, shape (nt,), float32  — or None if any step fails / the result
    is degenerate (all-zero, NaN, Inf).
    """
    from obspy import Stream

    try:
        tr = tr_raw.copy()
        tr.detrend("demean")
        tr.detrend("linear")
        cosine_taper(Stream(traces=[tr]), max_percentage=0.05)

        tr.remove_response(inventory=inventory, output="VEL", water_level=60, pre_filt=None)

        if abs(tr.stats.sampling_rate - target_fs) > 0.5:
            tr.resample(target_fs)

        tr.trim(starttime=t_start, endtime=t_end, pad=True, fill_value=0)

        d = tr.data.astype(np.float32)
        if len(d) < nt:
            d = np.pad(d, (0, nt - len(d)))
        elif len(d) > nt:
            d = d[:nt]

        if d is None or len(d) == 0 or np.any(np.isnan(d)) or np.any(np.isinf(d)) or np.max(np.abs(d)) == 0:
            return None
        return d

    except Exception:
        return None


def fetch_3c_window(client_sds, net, sta, chan_z, t_start, t_end, inventory, target_fs, nt,
                    horizontal_suffixes=HORIZONTAL_SUFFIXES):
    """
    Fetch and clean Z, N, E components for one station over [t_start, t_end].

    Channel order is always [Z, N, E]. If horizontals are unavailable, Z is
    duplicated into the N and E slots (documented fallback, same convention as
    preprocessing.load_3component / 04a's _fetch_3c_array).

    Returns
    -------
    np.ndarray, shape (nt, 3), float32  [Z, N, E]  — or None if the Z component
    itself fails (mandatory channel).
    """
    try:
        st_z = client_sds.get_waveforms(net, sta, "*", chan_z, t_start, t_end)
        if not st_z:
            return None
        st_z.merge(method=1, fill_value="interpolate")
        z_data = _process_trace(st_z[0], inventory, target_fs, t_start, t_end, nt)
        if z_data is None:
            return None
    except Exception:
        return None

    base = chan_z[:-1]   # e.g. "HH" from "HHZ"
    n_data = None
    e_data = None

    for suf_n, suf_e in horizontal_suffixes:
        if n_data is None:
            try:
                st_n = client_sds.get_waveforms(net, sta, "*", base + suf_n, t_start, t_end)
                if st_n:
                    st_n.merge(method=1, fill_value="interpolate")
                    n_data = _process_trace(st_n[0], inventory, target_fs, t_start, t_end, nt)
            except Exception:
                pass
        if e_data is None:
            try:
                st_e = client_sds.get_waveforms(net, sta, "*", base + suf_e, t_start, t_end)
                if st_e:
                    st_e.merge(method=1, fill_value="interpolate")
                    e_data = _process_trace(st_e[0], inventory, target_fs, t_start, t_end, nt)
            except Exception:
                pass
        if n_data is not None and e_data is not None:
            break

    if n_data is None:
        n_data = z_data.copy()
    if e_data is None:
        e_data = z_data.copy()

    return np.stack([z_data, n_data, e_data], axis=1)   # (nt, 3) [Z, N, E]


def spectrogram_image(data3, fs, nperseg, noverlap, nfft, freq_keep_mask):
    """
    Compute the log-power spectrogram of each of the 3 channels in `data3` and
    stack them like an RGB image.

    Parameters
    ----------
    data3 : np.ndarray, shape (nt, 3)  [Z, N, E]

    Returns
    -------
    image : np.ndarray, shape (n_freq, n_time, 3), float32, dB-scaled
    """
    # Floor epsilon, chosen to sit well BELOW real PSD values for this dataset --
    # confirmed via 07a_debug_amplitude_check.py: background PSD is ~1e-18
    # (m/s)^2/Hz and even the strongest observed event peaks only reach
    # ~1e-13. The previous epsilon (1e-12) was *larger* than nearly every real
    # value, so `Sxx + 1e-12` was dominated by the epsilon itself almost
    # everywhere -- silently flattening nearly the entire dynamic range to a
    # near-constant -120 dB floor regardless of whether a bin held background
    # noise or an actual event (worst for ice quake, the weakest class, whose
    # real PSD never got close to escaping the old floor). 1e-20 is ~2 orders
    # of magnitude below the smallest real values seen, so it only guards
    # against literal log(0) without swallowing real signal.
    PSD_FLOOR_EPS = 1e-20

    channels = []
    for c in range(3):
        _, _, Sxx = spectrogram(
            data3[:, c], fs=fs, window="hann",
            nperseg=nperseg, noverlap=noverlap, nfft=nfft,
            scaling="density", mode="psd",
        )
        Sxx_db = 10 * np.log10(Sxx[freq_keep_mask, :] + PSD_FLOOR_EPS)
        channels.append(Sxx_db.astype(np.float32))
    return np.stack(channels, axis=-1)   # (n_freq, n_time, 3)



# =============================================================================
# SECTION 7 — MAIN PROCESSING LOOP
# =============================================================================

print(f"\n{'='*65}")
print(f"  STEP 3 — Extracting {len(df):,} spectrogram images")
print(f"{'='*65}")

manifest_rows = []
n_ok   = 0
n_skip = 0
n_shape_mismatch = 0

for idx, row in df.iterrows():
    net   = row["network"]
    sta   = row["station"]
    chan  = row["channel"]
    etype = row["event_type"]
    t_on  = UTCDateTime(row["det_starttime"])

    t_start = t_on - WINDOW_PRE_S
    t_end   = t_start + WINDOW_S

    data3 = fetch_3c_window(
        client_sds, net, sta, chan, t_start, t_end,
        inventory, TARGET_FS, NT,
    )
    if data3 is None:
        n_skip += 1
        continue

    image = spectrogram_image(data3, TARGET_FS, SPEC_NPERSEG, SPEC_NOVERLAP, SPEC_NFFT, _freq_keep_mask)

    if image.shape != (N_FREQ, N_TIME, 3):
        n_shape_mismatch += 1
        n_skip += 1
        continue

    etype_slug = etype.lower().replace(" ", "_")
    fname = f"spec_{net}_{sta}_{chan}_{etype_slug}_{STAMP}_{idx}.npz"
    fpath = os.path.join(images_dir, fname)

    np.savez(
        fpath, image=image,
        event_type=etype, event_time=row["event_time"],
        network=net, station=sta, channel=chan,
        det_starttime=row["det_starttime"],
    )

    manifest_rows.append({
        "fname"        : fname,
        "event_time"   : row["event_time"],
        "event_type"   : etype,
        "network"      : net,
        "station"      : sta,
        "channel"      : chan,
        "det_starttime": row["det_starttime"],
        "split"        : split_by_event.get(row["event_time"], "train"),
    })
    n_ok += 1

    if n_ok % 50 == 0:
        print(f"  [{n_ok:5d} saved | {n_skip:5d} skipped]  last: {net}.{sta}.{chan}  ({etype})")

    if CHECKPOINT_EVERY > 0 and n_ok % CHECKPOINT_EVERY == 0:
        pd.DataFrame(manifest_rows).to_csv(
            os.path.join(RUN_DIR, f"image_list_checkpoint_{n_ok}.csv"), index=False
        )

print(f"\nExtraction done: {n_ok:,} images saved, {n_skip:,} rows skipped "
      f"({n_shape_mismatch:,} due to spectrogram shape mismatch — should be rare).")



# =============================================================================
# SECTION 8 — WRITE MANIFEST CSV
# =============================================================================

if not manifest_rows:
    print("\n[WARN] No images were extracted — image_list.csv will not be written.")
else:
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(RUN_DIR, "image_list.csv")
    manifest.to_csv(manifest_path, index=False)

    print(f"\n[SAVED] {manifest_path}")
    print(f"        {len(manifest):,} rows")

    print("\nFinal split x class breakdown:")
    print(manifest.groupby(["split", "event_type"]).size().unstack(fill_value=0).to_string())



# =============================================================================
# SECTION 9 — SUMMARY + NEXT STEPS
# =============================================================================

print(f"\n{'='*70}")
print(f"  Run finished       : {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Images saved       : {n_ok:,}  -> {images_dir}/")
print(f"  Manifest           : {os.path.join(RUN_DIR, 'image_list.csv')}")
print(f"  Image shape        : ({N_FREQ}, {N_TIME}, 3)")
print(f"  Run folder         : {RUN_DIR}")
print(f"  Log file           : {log_path}")
print(f"""
  Next steps
  ----------
  1. Run 07a_consolidate_for_colab.py (set RUN_DIR = "{RUN_DIR}") to pack
     the {n_ok:,} individual .npz files above into a handful of large
     archives — DO NOT upload the images/ folder directly to Google Drive
  2. Upload the contents of {RUN_DIR}/colab_package/ to Google Drive:
       MyDrive/colab_cnn_training_spectrogram/
           spectrograms_train.npz
           spectrograms_val.npz
           spectrograms_test.npz
           image_list.csv
           freq_axis.npy
           time_axis.npy
  3. Open 07b_train_cnn_classifier_colab.ipynb, set the Drive folder path in
     Cell 3, and run through the notebook (Runtime > Change runtime type > T4 GPU).
""")
print(f"{'='*70}")

log_file.close()
