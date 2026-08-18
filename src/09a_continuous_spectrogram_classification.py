"""
09a_continuous_spectrogram_classification.py
==============================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : August 2026

Goal
----
End-to-end validation of the CNN branch (07a/07b) on a RAW CONTINUOUS STREAM,
never chained before on data the pipeline hasn't already seen as pre-curated
catalog windows.

TWO PHASES, RUN SEPARATELY -- why
-----------------------------------
The ISTerre cluster's Python cannot get a clean TensorFlow install: the
`python/python3.11` module's environment already carries a large pre-existing
scientific stack (torch, pymc, beat, numba, ...) that hard-conflicts with
TensorFlow's own numpy/protobuf version pins, even inside a fresh venv. Rather
than keep fighting that, this script splits into two independently-runnable
phases controlled by RUN_EXTRACTION / RUN_CLASSIFICATION (Section 1):

  Phase 1 -- EXTRACTION (run on the cluster, needs SDS access, NO TensorFlow):
    per station-day, run STA/LTA detection on the full day, then for each
    detected event build ONE 07a-style spectrogram window -> pack every
    station-day's images into ONE .npz file (not one file per window --
    keeps the file count sane over month-scale runs).

  Phase 2 -- CLASSIFICATION (run locally, needs TensorFlow, NOT SDS access):
    load the packed .npz files (copied from the cluster to EXTRACTION_DIR
    locally), classify every detected window with the saved 5-class CNN,
    write one predictions CSV per month, optionally save a copy of individual
    spectrogram images for visual review now that the predicted class is
    actually known.

Typical usage
-------------
  1. On the cluster : RUN_EXTRACTION=True,  RUN_CLASSIFICATION=False
                       -> produces EXTRACTION_DIR/packed/*.npz
  2. Copy EXTRACTION_DIR (or at least its packed/ subfolder) to your local
     machine, e.g. into the local results folder.
  3. Locally         : RUN_EXTRACTION=False, RUN_CLASSIFICATION=True
                       EXTRACTION_DIR pointed at the local copy
                       -> produces predictions_<month>.csv + review images

There is no scoring against ground truth here -- the output is meant for
visual/plausibility review: does the class distribution look sane (mostly
noise/rare events), do spectrograms saved for non-noise predictions actually
look like real events, etc.

Output layout
-------------
  EXTRACTION_DIR/
      packed/spec_<net>_<sta>_<YYYYMMDD>.npz   <- one file per station-day
                                                   WITH >=1 detection, written
                                                   by Phase 1, read by Phase 2
  outputs_09a/run_YYYYMMDD_HHMMSS/    (this invocation's own log + phase-2 output)
      predictions_<month_tag>.csv     <- Phase 2 only: one row per classified detection
      review_images/*.npz             <- Phase 2 only: images saved for visual review
      review_gallery/*.png            <- Phase 2 only: rendered spectrograms for review
      run.log
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# -- Run mode -- see the module docstring "TWO PHASES" section for why -----------
RUN_EXTRACTION     = False    # Phase 1 -- cluster: fetch/preprocess/window/spectrogram, no TF import
RUN_CLASSIFICATION = True    # Phase 2 -- local: load packed spectrograms + model, classify

# -- Interchange directory between the two phases ---------------------------------
# Phase 1 writes packed per-station-day spectrogram archives here
EXTRACTION_DIR = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\09a_continuous_data_test"

# -- Paths (Phase 1 only -- cluster) ----------------------------------------------
SDS_ROOT    = "/data/sig/SDS"
ISTERRE_URL = "http://ist-sc3-geobs.osug.fr:8080"

# -- Output for THIS invocation's own log + (Phase 2 only) predictions/review imgs -
OUTPUT_DIR = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\09a_continuous_data_test"

# -- Trained CNN model (Phase 2 only -- local, downloaded from Google Drive) -----
MODEL_PATH      = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\07b_cnn_classifier\5classes_80Hz_20260818_100256\spectrogram_cnn_final.keras"
NORM_STATS_PATH = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\07b_cnn_classifier\5classes_80Hz_20260818_100256\normalization_stats.npz"

# -- Class order ------------------------------------------------------------------
# MUST exactly match CLASS_NAMES in 07b's Cell 3 at the time this model was trained
CLASS_NAMES = ['earthquake', 'regional', 'rockslide', 'ice quake', 'noise']

# -- Spatial bounding box (Mont Blanc massif --------------------------------------
LAT_MIN, LAT_MAX = 45.5, 46.0
LON_MIN, LON_MAX = 6.5,  7.2

# -- Channel selection (Phase 1 only) ---------------------------------------------
Z_CHANNEL = "??Z"
HORIZONTAL_SUFFIXES = [("N", "E"), ("2", "1")]   # same fallback convention as 07a

# -- Months to scan (Phase 1 only) ------------------------------------------------
MONTHS_TO_SCAN = [
    ("2025-01-01", "2025-02-01"),   # January
    ("2025-07-01", "2025-08-01"),   # July
]

# -- High-frequency mode -- MUST match whatever 07a/07b used to build+train the model MODEL_PATH points at below
# extend the spectrograms frequency window from 45 Hz (False -> baseline) to 80 Hz (True -> high freq mode)
HIGH_FREQ_MODE = True

if HIGH_FREQ_MODE:
    TARGET_FS     = 200     # [Hz] reaches FREQ_MAX_KEEP=80 Hz
    SPEC_NFFT     = 512     # nfft >= nperseg (400 samples @ 200Hz); same 0.39 Hz/bin resolution as the baseline below
    FREQ_MAX_KEEP = 80.0    # [Hz]
else:
    TARGET_FS     = 100     # [Hz] validated baseline
    SPEC_NFFT     = 256
    FREQ_MAX_KEEP = 45.0    # [Hz]

# -- Window / spectrogram parameters -- MUST match 07a EXACTLY (same trained model) --
WINDOW_PRE_S   = 5      # kept for documentation only -- there is no onset here,
WINDOW_POST_S  = 95     # window length is just WINDOW_PRE_S + WINDOW_POST_S = 100 s
WINDOW_S       = WINDOW_PRE_S + WINDOW_POST_S
NT             = int(WINDOW_S * TARGET_FS)

SPEC_NPERSEG_S      = 2.0
SPEC_NOVERLAP_FRAC  = 0.75
SPEC_NPERSEG        = int(SPEC_NPERSEG_S * TARGET_FS)
SPEC_NOVERLAP       = int(SPEC_NPERSEG * SPEC_NOVERLAP_FRAC)

# -- Detection: spectrogram-based STA/LTA (Groult et al. 2026), Phase 1 only ------
DET_FREQ_MIN  = 1.0
DET_FREQ_MAX  = 20.0    # covers ice quakes with energy above 10 Hz
DET_NSTA      = 1       # STA window length [s]
DET_NLTA      = 15      # LTA window length [s]
DET_THR_ON    = 8.0     # STA/LTA ratio to trigger event onset
DET_THR_OFF   = 2.0     # STA/LTA ratio to trigger event offset
DET_NWIN_SEC  = 5.0     # spectrogram window length for the detector's own internal STFT [s]
DET_NOVER_PCT = 0.20    # overlap fraction between the detector's internal STFT windows
DET_WINDOW_SEC  = 10 * 60   # 10-minute sliding processing window for the detector itself
DET_OVERLAP_SEC = 1  * 60   # 1-minute overlap between consecutive detector windows
DET_MIN_EVENT_DUR_SEC = 5.0     # discard detections shorter than this
DET_MIN_TRACE_SEC     = 120.0   # minimum day-segment length to attempt detection at all

# -- Packed spectrogram storage dtype (Phase 1 writes / Phase 2 reads) ------------
PACK_DTYPE = "float16"   # halves size vs float32, same convention as 07a_consolidate_for_colab.py's PACK_DTYPE

# -- Consolidation (Phase 1 only) -- combine one month's per-station-day packed
# files into ONE compressed archive, same idea as 07a_consolidate_for_colab.py.
# Per-station-day files are kept too (Phase 1's skip/resume logic depends on
# them existing) -- only the consolidated file needs to be copied off the cluster.
CONSOLIDATE_PER_MONTH = True

# -- Classification batch size (matches BATCH_SIZE used in 07b training) ----------
BATCH_SIZE = 128

# -- Review image saving policy (Phase 2 only) ------------------------------------
SAVE_IMAGES_FOR_NONNOISE = True    # save every window NOT predicted as 'noise'
SAVE_EVERY_NTH_NOISE     = 5      

# -- Review gallery (Phase 2 only) -------------------------------------------------
PLOT_REVIEW_GALLERY = True
N_GALLERY_PER_CLASS = 10    # how many examples per predicted class to plot

# -- Probability summary plot (Phase 2 only) ----------------------------------------
PLOT_PROBABILITY_SUMMARY = True   # confidence boxplot + mean-probability-vector heatmap per month

# -- Checkpointing (Phase 1: station-days, Phase 2: packed files) -----------------
CHECKPOINT_EVERY_STATION_DAYS = 5
CHECKPOINT_EVERY_PACKED_FILES = 20

# -- SMOKE TEST (Phase 1 only) -- strongly recommended before the full run --------
# Restricts every month to MAX_DAYS_SMOKE_TEST day(s) and MAX_STATIONS_SMOKE_TEST station(s)
SMOKE_TEST              = True
MAX_DAYS_SMOKE_TEST     = 1
MAX_STATIONS_SMOKE_TEST = 1



# =============================================================================
# SECTION 2 — SETUP
# =============================================================================

import os
import sys
import glob
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.signal import spectrogram as scipy_spectrogram

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_setup import create_run_dir, setup_logging

RUN_DIR, STAMP = create_run_dir(OUTPUT_DIR)
log_file, log_path = setup_logging(
    RUN_DIR, "09a_continuous_spectrogram_classification.py",
    extra_info=(f"RUN_EXTRACTION={RUN_EXTRACTION}  RUN_CLASSIFICATION={RUN_CLASSIFICATION}  |  "
                f"EXTRACTION_DIR: {EXTRACTION_DIR}  |  DET_THR_ON={DET_THR_ON}  |  SMOKE_TEST={SMOKE_TEST}")
)

packed_dir = os.path.join(EXTRACTION_DIR)
os.makedirs(packed_dir, exist_ok=True)

if not RUN_EXTRACTION and not RUN_CLASSIFICATION:
    print("[ERROR] Both RUN_EXTRACTION and RUN_CLASSIFICATION are False -- nothing to do.")
    log_file.close()
    sys.exit(1)

if SMOKE_TEST and RUN_EXTRACTION:
    print(f"\n[SMOKE TEST MODE] Restricting extraction to {MAX_DAYS_SMOKE_TEST} day(s) x "
          f"{MAX_STATIONS_SMOKE_TEST} station(s) per month. Set SMOKE_TEST=False once "
          f"you've checked the timing + storage estimate below.\n")


# --------- Phase 1 setup: SDS/FDSN connections + station selection ----------------
if RUN_EXTRACTION:
    from obspy import UTCDateTime, Stream
    from run_setup import connect_sds, connect_fdsn, fetch_inventory
    from catalog_helpers import build_station_list_from_inventory
    from preprocessing import cosine_taper, preprocess_day
    from detecteurV3_fonctions import DetecteurV3   # Groult et al. 2026 -- third-party, do not modify
    from detection import compute_snr, merge_window_events

    client_sds  = connect_sds(SDS_ROOT)
    client_fdsn = connect_fdsn(ISTERRE_URL)
    if client_sds is None or client_fdsn is None:
        print("[ERROR] Cannot proceed with extraction without SDS and FDSN. Exiting.")
        log_file.close()
        sys.exit(1)

    _t_min = min(UTCDateTime(t0) for t0, _ in MONTHS_TO_SCAN)
    _t_max = max(UTCDateTime(t1) for _, t1 in MONTHS_TO_SCAN)
    inventory = fetch_inventory(client_fdsn, _t_min.strftime("%Y-%m-%d"), _t_max.strftime("%Y-%m-%d"),
                                lat_min=LAT_MIN, lat_max=LAT_MAX,
                                lon_min=LON_MIN, lon_max=LON_MAX)
    if inventory is None:
        print("[ERROR] Could not fetch inventory. Exiting.")
        log_file.close()
        sys.exit(1)

    station_list = build_station_list_from_inventory(inventory)

    # -- HIGH_FREQ_MODE station filter -- keep only stations natively sampled >=
    # TARGET_FS, same reasoning as 07a's catalog filter (see Section 1 comment).
    if HIGH_FREQ_MODE:
        n_before = len(station_list)
        filtered = []
        for net, sta, loc, chan in station_list:
            try:
                sel = inventory.select(network=net, station=sta, channel=chan)
                chans_found = [c for n_ in sel.networks for s_ in n_.stations for c in s_.channels]
                sr = chans_found[0].sample_rate if chans_found else 0
            except Exception:
                sr = 0
            if sr >= TARGET_FS:
                filtered.append((net, sta, loc, chan))
            else:
                print(f"  [SKIP] {net}.{sta}.{loc}.{chan} — native {sr:.0f} Hz < "
                      f"TARGET_FS={TARGET_FS} Hz (HIGH_FREQ_MODE)")
        station_list = filtered
        print(f"\n[HIGH_FREQ_MODE] Station filter: {len(station_list)}/{n_before} "
              f"station(s) natively >= {TARGET_FS} Hz kept.")

    if SMOKE_TEST:
        station_list = station_list[:MAX_STATIONS_SMOKE_TEST]

    print(f"\n{len(station_list)} station(s) selected in the Mont Blanc massif bounding box:")
    for net, sta, loc, chan in station_list:
        print(f"  {net}.{sta}.{loc}.{chan}")


# --------- Phase 2 setup: load model + normalization stats (TensorFlow only here) -
model = None
CHANNEL_MEAN = CHANNEL_STD = None

if RUN_CLASSIFICATION:
    if not os.path.isfile(MODEL_PATH):
        print(f"[ERROR] MODEL_PATH not found: {MODEL_PATH}")
        print("        Update MODEL_PATH in Section 1 to your local copy.")
        log_file.close()
        sys.exit(1)
    if not os.path.isfile(NORM_STATS_PATH):
        print(f"[ERROR] NORM_STATS_PATH not found: {NORM_STATS_PATH}")
        log_file.close()
        sys.exit(1)

    print(f"\n[LOAD] Model  : {MODEL_PATH}")
    print(f"[LOAD] Norm   : {NORM_STATS_PATH}")

    import tensorflow as tf   # deferred import -- Phase 1 (cluster) never reaches this line
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)   # inference only

    _norm = np.load(NORM_STATS_PATH)
    CHANNEL_MEAN = _norm["mean"].astype(np.float32)   # shape (1,1,1,3)
    CHANNEL_STD  = _norm["std"].astype(np.float32)

    print(f"[OK] Model loaded. Output classes (index order) = {CLASS_NAMES}")
    print(f"[OK] Per-channel mean={CHANNEL_MEAN.ravel()}  std={CHANNEL_STD.ravel()}")

    review_dir = os.path.join(RUN_DIR, "review_images")
    os.makedirs(review_dir, exist_ok=True)

    if PLOT_REVIEW_GALLERY or PLOT_PROBABILITY_SUMMARY:
        from run_setup import set_matplotlib_defaults
        import matplotlib.pyplot as plt
        set_matplotlib_defaults()
        if PLOT_REVIEW_GALLERY:
            from visualization import plot_spectrogram_rgb_example
            gallery_dir = os.path.join(RUN_DIR, "review_gallery")
            os.makedirs(gallery_dir, exist_ok=True)
        if PLOT_PROBABILITY_SUMMARY:
            summary_dir = os.path.join(RUN_DIR, "probability_summary")
            os.makedirs(summary_dir, exist_ok=True)



# =============================================================================
# SECTION 3 — SPECTROGRAM SHAPE (fixed, precomputed once — must match 07a exactly)
# =============================================================================
# Cheap (scipy/numpy only), needed by both phases -- Phase 1 to build images,
# Phase 2 to sanity-check packed images against what the model expects.

_dummy = np.zeros(NT, dtype=np.float32)
_f_full, _t_axis, _ = scipy_spectrogram(
    _dummy, fs=TARGET_FS, window="hann",
    nperseg=SPEC_NPERSEG, noverlap=SPEC_NOVERLAP, nfft=SPEC_NFFT,
    scaling="density", mode="psd",
)
_freq_keep_mask = _f_full <= FREQ_MAX_KEEP
FREQ_AXIS = _f_full[_freq_keep_mask]
TIME_AXIS = _t_axis
N_FREQ    = len(FREQ_AXIS)
N_TIME    = len(TIME_AXIS)

print(f"\nSpectrogram image shape: ({N_FREQ}, {N_TIME}, 3)")

if RUN_CLASSIFICATION and model is not None:
    _expected_shape = tuple(model.input_shape[1:])
    if _expected_shape != (N_FREQ, N_TIME, 3):
        print(f"[ERROR] Model expects input shape {_expected_shape}, but this script "
              f"computes ({N_FREQ}, {N_TIME}, 3). Check SPEC_* / FREQ_MAX_KEEP / TARGET_FS "
              f"match the values used to build the packed spectrograms. Exiting.")
        log_file.close()
        sys.exit(1)



# =============================================================================
# SECTION 4 — HELPER FUNCTIONS
# =============================================================================

def _process_event_trace(tr_raw, inventory, target_fs, t_start, t_end, nt):
    """
    Phase 1 only. Identical to 07a_spectrogram_dataset_build.py's
    _process_trace() -- literal copy, not an import, same reasoning as
    spectrogram_image() below. Clean one SHORT (~WINDOW_S-second) raw trace ->
    response-removed velocity [m/s], resampled, trimmed/padded to exactly `nt`
    samples.

    Deliberately plain remove_response (pre_filt=None, water_level=60), NOT
    the pre_filt-tapered approach used for the full-day detection trace below
    -- this MUST match 07a's per-event convention exactly, since that's what
    the model was trained to normalize against. See the module docstring's
    "WHY THIS SCRIPT NOW USES STA/LTA DETECTION" section for why this matters.

    Returns
    -------
    np.ndarray, shape (nt,), float32 -- or None if any step fails / degenerate.
    """
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


def fetch_3c_event_window(client_sds, net, sta, chan_z, t_start, t_end, inventory, target_fs, nt,
                          horizontal_suffixes=HORIZONTAL_SUFFIXES):
    """
    Phase 1 only. Identical to 07a's fetch_3c_window() -- fetch and clean Z, N,
    E components for one station over [t_start, t_end), a fresh short-window
    SDS fetch (NOT sliced from the day-scale detection trace -- see
    _process_event_trace docstring for why that distinction matters). Channel
    order always [Z, N, E]; horizontals default to a copy of Z if unavailable.

    Returns
    -------
    np.ndarray, shape (nt, 3), float32 [Z, N, E] -- or None if Z fails.
    """
    try:
        st_z = client_sds.get_waveforms(net, sta, "*", chan_z, t_start, t_end)
        if not st_z:
            return None
        st_z.merge(method=1, fill_value="interpolate")
        z_data = _process_event_trace(st_z[0], inventory, target_fs, t_start, t_end, nt)
        if z_data is None:
            return None
    except Exception:
        return None

    base = chan_z[:-1]
    n_data = None
    e_data = None

    for suf_n, suf_e in horizontal_suffixes:
        if n_data is None:
            try:
                st_n = client_sds.get_waveforms(net, sta, "*", base + suf_n, t_start, t_end)
                if st_n:
                    st_n.merge(method=1, fill_value="interpolate")
                    n_data = _process_event_trace(st_n[0], inventory, target_fs, t_start, t_end, nt)
            except Exception:
                pass
        if e_data is None:
            try:
                st_e = client_sds.get_waveforms(net, sta, "*", base + suf_e, t_start, t_end)
                if st_e:
                    st_e.merge(method=1, fill_value="interpolate")
                    e_data = _process_event_trace(st_e[0], inventory, target_fs, t_start, t_end, nt)
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
    """Phase 1 only. Identical to 07a_spectrogram_dataset_build.py's
    spectrogram_image() -- same PSD floor epsilon, same STFT params. Kept as
    a literal copy (not an import) so this script has no hidden runtime
    dependency on 07a's own CONFIGURATION section changing underneath it."""
    PSD_FLOOR_EPS = 1e-20
    channels = []
    for c in range(3):
        _, _, Sxx = scipy_spectrogram(
            data3[:, c], fs=fs, window="hann",
            nperseg=nperseg, noverlap=noverlap, nfft=nfft,
            scaling="density", mode="psd",
        )
        Sxx_db = 10 * np.log10(Sxx[freq_keep_mask, :] + PSD_FLOOR_EPS)
        channels.append(Sxx_db.astype(np.float32))
    return np.stack(channels, axis=-1)


def classify_batch(images_f32):
    """Phase 2 only. images_f32: (B, N_FREQ, N_TIME, 3) float32 -> (labels, probs)"""
    batch = (images_f32 - CHANNEL_MEAN) / CHANNEL_STD
    proba = model.predict(batch, verbose=0)
    labels = np.argmax(proba, axis=1)
    return labels, proba


_noise_review_counter = [0]   # mutable cell -- persists across the whole Phase 2 run

def _emit_row(meta, img_f32, label_idx, proba, rows_list):
    """
    Phase 2 only. Build one output row (metadata + predicted class + all 5
    probabilities), append it to rows_list, and -- per SAVE_IMAGES_FOR_NONNOISE
    / SAVE_EVERY_NTH_NOISE -- optionally save the spectrogram image itself to
    review_dir for visual review.
    """
    cls_name = CLASS_NAMES[label_idx]

    row = dict(meta)
    row["predicted_class"] = cls_name
    for i, cname in enumerate(CLASS_NAMES):
        row[f"proba_{cname.replace(' ', '_')}"] = float(proba[i])

    save_image = False
    if cls_name != "noise" and SAVE_IMAGES_FOR_NONNOISE:
        save_image = True
    elif cls_name == "noise" and SAVE_EVERY_NTH_NOISE > 0:
        _noise_review_counter[0] += 1
        save_image = (_noise_review_counter[0] % SAVE_EVERY_NTH_NOISE == 0)

    if save_image:
        safe_time = meta["window_start"].replace(":", "").replace("-", "").replace(".", "")
        fname = (f"spec_{meta['network']}_{meta['station']}_{meta['channel']}_"
                 f"{safe_time}_{cls_name.replace(' ', '_')}.npz")
        # np.savez can't hold a bare None (needs allow_pickle=True to reload) --
        # sanitize any missing SNR fields (older packed files) to NaN first.
        meta_clean = {k: (np.nan if v is None else v) for k, v in meta.items()}
        np.savez(os.path.join(review_dir, fname),
                 image=img_f32, predicted_class=cls_name, proba=proba, **meta_clean)
        row["review_image_saved"] = fname
    else:
        row["review_image_saved"] = ""

    rows_list.append(row)


def consolidate_month_packed_files(packed_dir, month_tag):
    """
    Phase 1 only, run once per month right after that month's station-day
    loop finishes. Globs every per-station-day spec_<net>_<sta>_<YYYYMMDD>.npz
    in packed_dir whose "day" field falls in this month, and concatenates
    them into ONE compressed archive: packed_dir/consolidated_<month_tag>.npz
    -- same idea as 07a_consolidate_for_colab.py bundling train/test/val into
    single files, so there's one thing to copy off the cluster instead of one
    file per station-day. Per-station-day scalar metadata (network/station/
    location/channel/day) becomes a per-window array here so Phase 2 can
    still tell which detection came from which station-day. The individual
    per-station-day files are left in place -- Phase 1's skip/resume logic
    (the `if os.path.isfile(out_path)` check) depends on them existing.
    """
    candidates = sorted(glob.glob(os.path.join(packed_dir, "spec_*.npz")))
    imgs, ws, we, snr_v, snr_fm, nets, stas, locs, chans, days = [], [], [], [], [], [], [], [], [], []

    for fpath in candidates:
        try:
            with np.load(fpath, allow_pickle=False) as d:
                day = str(d["day"])
                if day[:7] != month_tag:
                    continue
                n = d["images"].shape[0]
                imgs.append(d["images"])
                ws.append(d["window_start"])
                we.append(d["window_end"])
                snr_v.append(d["snr"])
                snr_fm.append(d["snr_full_median"])
                nets.append(np.full(n, str(d["network"])))
                stas.append(np.full(n, str(d["station"])))
                locs.append(np.full(n, str(d["location"])))
                chans.append(np.full(n, str(d["channel"])))
                days.append(np.full(n, day))
        except Exception as e:
            print(f"  [WARN] consolidate: could not read {os.path.basename(fpath)}: {e}")

    if not imgs:
        print(f"  [WARN] consolidate: no packed files found for month {month_tag} -- skipping.")
        return None

    out_path = os.path.join(packed_dir, f"consolidated_{month_tag}.npz")
    np.savez_compressed(
        out_path,
        images=np.concatenate(imgs, axis=0),
        window_start=np.concatenate(ws), window_end=np.concatenate(we),
        snr=np.concatenate(snr_v), snr_full_median=np.concatenate(snr_fm),
        network=np.concatenate(nets), station=np.concatenate(stas),
        location=np.concatenate(locs), channel=np.concatenate(chans),
        day=np.concatenate(days),
    )
    n_total = sum(a.shape[0] for a in imgs)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  [CONSOLIDATED] {month_tag}: {n_total} window(s) from {len(imgs)} station-day file(s) "
          f"-> {os.path.basename(out_path)} ({size_mb:.1f} MB)")
    return out_path


def plot_probability_summary(df_month, class_names, month_tag, out_dir):
    """
    Phase 2 only. Two-panel figure per month, built directly from the
    proba_<class> columns already in the predictions CSV -- makes the
    model's confidence/confusion visible without digging through the raw
    numbers.

    Left panel  : boxplot of the WINNING probability (the value argmax was
                  actually taken on), grouped by predicted class. A class
                  sitting near chance level (1 / n_classes) most of the time
                  is a class this validation run can't be trusted on yet.
    Right panel : heatmap of the MEAN full probability vector for each
                  predicted class (rows = predicted class, cols = probability
                  mass on each class). A confusion-matrix stand-in for
                  continuous data with no ground truth -- real off-diagonal
                  mass in a row means that class is being confused with
                  another one even when it "wins" the argmax.
    """
    proba_cols = [f"proba_{c.replace(' ', '_')}" for c in class_names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # -- Left: winning-probability boxplot by predicted class --------------
    ax = axes[0]
    data, labels_present = [], []
    for c in class_names:
        col = f"proba_{c.replace(' ', '_')}"
        vals = df_month.loc[df_month["predicted_class"] == c, col]
        if len(vals) > 0:
            data.append(vals.values)
            labels_present.append(f"{c}\n(n={len(vals)})")
    if data:
        ax.boxplot(data, labels=labels_present, showmeans=True)
    ax.axhline(1.0 / len(class_names), color="gray", linestyle="--", linewidth=1,
               label=f"chance level (1/{len(class_names)})")
    ax.set_ylabel("Winning probability")
    ax.set_title(f"Confidence by predicted class — {month_tag}")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="lower right")
    ax.tick_params(axis="x", labelsize=8)

    # -- Right: mean full probability vector per predicted class -----------
    ax = axes[1]
    present_classes = [c for c in class_names if (df_month["predicted_class"] == c).any()]
    matrix = np.array([
        df_month.loc[df_month["predicted_class"] == c, proba_cols].mean().values
        for c in present_classes
    ])
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(present_classes)))
    ax.set_yticklabels(present_classes, fontsize=8)
    ax.set_xlabel("Probability mass on class")
    ax.set_ylabel("Predicted class")
    ax.set_title("Mean probability vector by predicted class")
    for i in range(len(present_classes)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                     color="white" if matrix[i, j] < 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Predicted-class probability summary — {month_tag}", fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"probability_summary_{month_tag}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] Probability summary -> {out_path}")



# =============================================================================
# SECTION 5A — PHASE 1: EXTRACTION (cluster, no TensorFlow)
# =============================================================================

if RUN_EXTRACTION:

    print(f"\n{'='*70}")
    print(f"  PHASE 1 — EXTRACTION (STA/LTA detection, Groult et al. 2026)")
    print(f"  {'[SMOKE TEST]' if SMOKE_TEST else '[FULL RUN]'}")
    print(f"{'='*70}")

    for month_idx, (t_start_str, t_end_str) in enumerate(MONTHS_TO_SCAN, 1):
        month_tag = t_start_str[:7]
        t0 = UTCDateTime(t_start_str)
        t1 = UTCDateTime(t_end_str)

        days = []
        d = t0
        while d < t1:
            days.append(d)
            d += 86400
        if SMOKE_TEST:
            days = days[:MAX_DAYS_SMOKE_TEST]

        print(f"\n{'='*70}")
        print(f"  MONTH {month_idx}/{len(MONTHS_TO_SCAN)} : {month_tag}  "
              f"({len(days)} day(s) x {len(station_list)} station(s))")
        print(f"{'='*70}")

        n_station_days   = 0
        n_events_total   = 0

        for day_utc in days:
            day_str   = day_utc.strftime("%Y-%m-%d")
            day_start = day_utc
            day_end   = day_utc + 86400

            for net, sta, loc, chan in station_list:
                t_station_day = time.time()

                out_fname = f"spec_{net}_{sta}_{day_utc.strftime('%Y%m%d')}.npz"
                out_path  = os.path.join(packed_dir, out_fname)
                n_station_days += 1

                if os.path.isfile(out_path):
                    print(f"  [SKIP] {day_str} {net}.{sta} — already packed ({out_fname})")
                    continue

                # ---- Load full-day Z, detect candidate events -----------------------
                try:
                    st = client_sds.get_waveforms(net, sta, "*", chan, day_start, day_end)
                except Exception as e:
                    print(f"  [SKIP] {day_str} {net}.{sta} — SDS load failed: {e}")
                    continue
                if len(st) == 0:
                    print(f"  [SKIP] {day_str} {net}.{sta} — no data")
                    continue
                try:
                    st.merge(fill_value=None)
                except Exception:
                    st.merge(fill_value=0)

                day_events = []   # list of (t_on, t_off, snr_dict)

                for tr_raw in st:
                    seg_dur = tr_raw.stats.endtime - tr_raw.stats.starttime
                    if seg_dur < DET_MIN_TRACE_SEC:
                        continue

                    tr_vel = preprocess_day(tr_raw, inventory)
                    if tr_vel is None:
                        continue

                    fs = tr_vel.stats.sampling_rate
                    nwin  = int(DET_NWIN_SEC * fs)
                    nover = int(nwin * DET_NOVER_PCT)
                    nfft  = 2 ** int(np.ceil(np.log2(nwin)))
                    df = fs / nfft
                    if DET_FREQ_MAX / df < 2:
                        nfft = 2 ** int(np.ceil(np.log2(DET_FREQ_MAX * 4)))

                    tr_filt = tr_vel.copy()
                    tr_filt.filter('bandpass', freqmin=DET_FREQ_MIN,
                                   freqmax=min(DET_FREQ_MAX, 0.9 * fs / 2),
                                   corners=4, zerophase=True)

                    total_events     = {}
                    total_thresholds = {}
                    win_start = tr_vel.stats.starttime

                    while win_start < tr_vel.stats.endtime:
                        win_end = min(win_start + DET_WINDOW_SEC, tr_vel.stats.endtime)
                        tr_win  = tr_vel.slice(win_start, win_end)

                        win_dur       = tr_win.stats.endtime - tr_win.stats.starttime
                        dt_nrj_approx = DET_NWIN_SEC * (1 - DET_NOVER_PCT)
                        if win_dur / dt_nrj_approx <= DET_NLTA:
                            break

                        _, _, _, _, events_dt, thresholds_dt = DetecteurV3(
                            tr_win, DET_FREQ_MIN, DET_FREQ_MAX,
                            DET_NSTA, DET_NLTA, DET_THR_ON, DET_THR_OFF,
                            nwin, nover, nfft, 'True'
                        )

                        events, thresholds, k = {}, {}, 1
                        for orig_key, val in events_dt.items():
                            ev_on  = UTCDateTime(str(val[0]))
                            ev_off = UTCDateTime(str(val[1]))
                            if (ev_off - ev_on) >= DET_MIN_EVENT_DUR_SEC:
                                events[f"Event_{k}"]     = [ev_on, ev_off]
                                thresholds[f"Event_{k}"] = thresholds_dt.get(orig_key, [0.0, 0.0])
                                k += 1

                        total_events, total_thresholds = merge_window_events(
                            total_events, total_thresholds, events, thresholds
                        )

                        if win_end >= tr_vel.stats.endtime:
                            break
                        win_start = win_end - DET_OVERLAP_SEC

                    for ev_key, (ev_on, ev_off) in total_events.items():
                        snr = compute_snr(tr_filt, ev_on, ev_off)
                        day_events.append((ev_on, ev_off, snr))

                # ---- For each detected event: build ONE 07a-style spectrogram window --
                images, starts, ends, snr_vals, snr_full_med = [], [], [], [], []

                for ev_on, ev_off, snr in day_events:
                    w_start = ev_on - WINDOW_PRE_S
                    w_end   = w_start + WINDOW_S

                    data3 = fetch_3c_event_window(client_sds, net, sta, chan, w_start, w_end,
                                                  inventory, TARGET_FS, NT)
                    if data3 is None:
                        continue

                    img = spectrogram_image(data3, TARGET_FS, SPEC_NPERSEG, SPEC_NOVERLAP,
                                            SPEC_NFFT, _freq_keep_mask)
                    if img.shape != (N_FREQ, N_TIME, 3):
                        continue

                    images.append(img.astype(PACK_DTYPE))
                    starts.append(str(w_start))
                    ends.append(str(w_end))
                    snr_vals.append(float(snr.get('SNR', np.nan)))
                    snr_full_med.append(float(snr.get('SNR_full_median', np.nan)))

                dt = time.time() - t_station_day

                if not images:
                    print(f"  {day_str} {net}.{sta:<6s}  0 event(s) detected  [{dt:6.1f}s]")
                    continue

                images_arr = np.stack(images, axis=0)   # (n_events, N_FREQ, N_TIME, 3)
                np.savez(out_path,
                         images=images_arr,
                         window_start=np.array(starts), window_end=np.array(ends),
                         snr=np.array(snr_vals), snr_full_median=np.array(snr_full_med),
                         network=net, station=sta, location=loc, channel=chan, day=day_str)

                size_mb = os.path.getsize(out_path) / 1e6
                n_events_total += len(images)
                print(f"  {day_str} {net}.{sta:<6s}  {len(images):4d} event(s) detected+packed "
                      f"({size_mb:6.1f} MB) in {dt:6.1f}s")

        print(f"\n  [DONE] Month {month_tag}: {n_events_total} event(s) across "
              f"{n_station_days} station-day(s) -> packed files in {packed_dir}/")

        if CONSOLIDATE_PER_MONTH:
            consolidate_month_packed_files(packed_dir, month_tag)

    print(f"\n[PHASE 1 COMPLETE] Packed spectrograms -> {packed_dir}/")
    print(f"  Copy this folder to your local machine, then run Phase 2 with")
    print(f"  RUN_EXTRACTION=False, RUN_CLASSIFICATION=True, and EXTRACTION_DIR")
    print(f"  pointed at the local copy.")



# =============================================================================
# SECTION 5B — PHASE 2: CLASSIFICATION (local, TensorFlow)
# =============================================================================

if RUN_CLASSIFICATION:

    # Prefer consolidated monthly archives (see CONSOLIDATE_PER_MONTH in Section 1)
    # over raw per-station-day files -- never both, that would double-count.
    consolidated_files = sorted(glob.glob(os.path.join(packed_dir, "consolidated_*.npz")))
    if consolidated_files:
        packed_files = consolidated_files
        print(f"\n{'='*70}")
        print(f"  PHASE 2 — CLASSIFICATION")
        print(f"  {len(packed_files)} consolidated monthly archive(s) found in {packed_dir}")
        print(f"{'='*70}")
    else:
        packed_files = sorted(glob.glob(os.path.join(packed_dir, "spec_*.npz")))
        print(f"\n{'='*70}")
        print(f"  PHASE 2 — CLASSIFICATION")
        print(f"  {len(packed_files)} packed station-day file(s) found in {packed_dir} "
              f"(no consolidated_*.npz found -- see CONSOLIDATE_PER_MONTH in Section 1)")
        print(f"{'='*70}")

    if not packed_files:
        print(f"\n[WARN] No packed .npz files found in {packed_dir}. "
              f"Did you copy Phase 1's output here and set EXTRACTION_DIR correctly?")

    rows_by_month = {}   # month_tag -> list of row dicts
    n_files_done  = 0

    for fpath in packed_files:
        t_file = time.time()
        with np.load(fpath, allow_pickle=False) as d:
            images = d["images"]              # (n, N_FREQ, N_TIME, 3), PACK_DTYPE
            starts = d["window_start"]
            ends   = d["window_end"]
            snrs   = d["snr"]              if "snr" in d.files              else None
            snr_fm = d["snr_full_median"]  if "snr_full_median" in d.files  else None

            n = images.shape[0]
            # network/station/location/channel/day are scalars in a per-station-day
            # file, but per-window arrays in a consolidated_*.npz -- normalize both
            # to a length-n array here so the rest of the loop doesn't care which.
            def _per_window(key):
                val = np.atleast_1d(d[key])
                return val if val.shape[0] == n else np.full(n, str(val.reshape(-1)[0]))
            nets  = _per_window("network")
            stas  = _per_window("station")
            locs  = _per_window("location")
            chans = _per_window("channel")
            days  = _per_window("day")

        if images.shape[1:] != (N_FREQ, N_TIME, 3):
            print(f"  [SKIP] {os.path.basename(fpath)} — shape mismatch "
                  f"{images.shape[1:]} != ({N_FREQ}, {N_TIME}, 3)")
            continue

        for b0 in range(0, n, BATCH_SIZE):
            b1 = min(b0 + BATCH_SIZE, n)
            batch = images[b0:b1].astype(np.float32)
            labels, proba = classify_batch(batch)
            for i in range(b1 - b0):
                idx = b0 + i
                month_tag = str(days[idx])[:7]
                rows_by_month.setdefault(month_tag, [])
                meta = {
                    "network": str(nets[idx]), "station": str(stas[idx]),
                    "location": str(locs[idx]), "channel": str(chans[idx]),
                    "day": str(days[idx]), "window_start": str(starts[idx]), "window_end": str(ends[idx]),
                    "snr": float(snrs[idx]) if snrs is not None else None,
                    "snr_full_median": float(snr_fm[idx]) if snr_fm is not None else None,
                }
                _emit_row(meta, batch[i], labels[i], proba[i], rows_by_month[month_tag])

        n_files_done += 1
        dt = time.time() - t_file
        print(f"  [{n_files_done:4d}/{len(packed_files)}] {os.path.basename(fpath)}  "
              f"{n:5d} windows classified in {dt:5.1f}s")

        if CHECKPOINT_EVERY_PACKED_FILES > 0 and n_files_done % CHECKPOINT_EVERY_PACKED_FILES == 0:
            for month_tag, rows in rows_by_month.items():
                pd.DataFrame(rows).to_csv(
                    os.path.join(RUN_DIR, f"predictions_{month_tag}_checkpoint.csv"), index=False
                )

    for month_tag, rows in rows_by_month.items():
        if not rows:
            continue
        df_month = pd.DataFrame(rows)
        out_csv = os.path.join(RUN_DIR, f"predictions_{month_tag}.csv")
        df_month.to_csv(out_csv, index=False)
        print(f"\n  [SAVED] {len(df_month):,} windows -> {out_csv}")
        print(f"  Class distribution ({month_tag}):")
        vc = df_month["predicted_class"].value_counts()
        for c in CLASS_NAMES:
            n_c = int(vc.get(c, 0))
            pct = 100 * n_c / max(len(df_month), 1)
            print(f"    {c:<12s} {n_c:8,d}  ({pct:5.1f}%)")

        if PLOT_PROBABILITY_SUMMARY:
            plot_probability_summary(df_month, CLASS_NAMES, month_tag, summary_dir)

    # ---- Review gallery: render the N most confident review images per class ----
    if PLOT_REVIEW_GALLERY and N_GALLERY_PER_CLASS > 0:
        print(f"\n  Building review gallery ({N_GALLERY_PER_CLASS} example(s) per class)...")
        review_files = glob.glob(os.path.join(review_dir, "*.npz"))

        by_class = {c: [] for c in CLASS_NAMES}
        for rf in review_files:
            with np.load(rf, allow_pickle=False) as d:
                cls = str(d["predicted_class"])
                if cls in by_class:
                    by_class[cls].append((float(np.max(d["proba"])), rf))

        for cls, items in by_class.items():
            if not items:
                continue
            items.sort(key=lambda x: -x[0])   # most confident (highest max-proba) first
            for rank, (p, rf) in enumerate(items[:N_GALLERY_PER_CLASS], 1):
                with np.load(rf, allow_pickle=False) as d:
                    img    = d["image"]
                    net_i  = str(d["network"])
                    sta_i  = str(d["station"])
                    w_start = str(d["window_start"])
                out_png = os.path.join(gallery_dir, f"{cls.replace(' ', '_')}_{rank:02d}.png")
                plot_spectrogram_rgb_example(
                    FREQ_AXIS, TIME_AXIS, img.astype(np.float32),
                    title_lines=(f"{cls}  (p={p:.2f})", f"{net_i}.{sta_i}  {w_start}"),
                    out_path=out_png,
                )
            print(f"    {cls:<12s} {min(len(items), N_GALLERY_PER_CLASS):3d} image(s) plotted "
                  f"(of {len(items)} saved review image(s) for this class)")

        print(f"  [SAVED] Review gallery -> {gallery_dir}/")

    print(f"\n[PHASE 2 COMPLETE] Predictions + review images -> {RUN_DIR}/")



# =============================================================================
# SECTION 6 — END
# =============================================================================

print(f"\n{'='*70}")
print(f"  Run finished : {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  All outputs  : {RUN_DIR}")
print(f"  Log file     : {log_path}")
if RUN_EXTRACTION and SMOKE_TEST:
    print(f"\n  This was a SMOKE TEST run (SMOKE_TEST=True). Check the per-station-day")
    print(f"  timing and detection counts printed above, sanity-check that the number")
    print(f"  of events/day looks physically reasonable for this station, then set")
    print(f"  SMOKE_TEST=False for the full MONTHS_TO_SCAN.")
print(f"{'='*70}")

log_file.close()
