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

WHY THIS SCRIPT NOW USES STA/LTA DETECTION (v2 -- it didn't at first)
------------------------------------------------------------------------
The first version of this script deliberately skipped detection entirely: it
computed a spectrogram on a fixed-hop continuous sliding window over the WHOLE
raw stream and classified every window directly, letting the CNN's 'noise'
class do the separating. In practice this produced predictions dominated by
'rockslide' (83-93%) instead of the expected mostly-noise distribution -- a
systematic bias, not just noisy/uncertain predictions. Two compounding causes:

  1. Onset-alignment mismatch: 07a/07b only ever trained the CNN on windows
     anchored at [-WINDOW_PRE_S, +WINDOW_POST_S] around a real (kurtosis-
     refined) onset -- the true onset always sits ~5s into the window. A
     continuous sliding window has no such anchoring; the model was being fed
     inputs unlike anything it saw in training.
  2. Preprocessing mismatch: v1's full-day response removal used the
     pre_filt-based approach from preprocessing.preprocess_day() (needed to
     avoid amplifying broadband noise across 24h), but 07a's actual per-EVENT
     training windows were built with plain remove_response(pre_filt=None,
     water_level=60) on short ~100s segments -- a materially different
     frequency response than what the model was trained to normalize.

v2 fixes both by going back to a real detector (Groult et al. 2026
spectrogram-based STA/LTA, same method + parameters as
02b_spectrogram_sta_lta_detection.py's catalog-less scanner) to find candidate
onsets, and then building each CNN input window EXACTLY the way 07a does for
training data: a fresh short-window fetch + remove_response(pre_filt=None,
water_level=60), anchored so the onset sits ~WINDOW_PRE_S seconds into the
window. Spectrograms are only computed for detected windows now, not the
entire continuous trace -- far cheaper too.

Note on the 'noise' class in this design: since STA/LTA only fires on
above-threshold energy bursts, most ambient background is filtered out before
the CNN ever sees it. 'noise' predictions here mostly mean "STA/LTA triggered
on something that isn't a real event" (a spurious/cultural trigger), not
"ambient background in general" the way 04d's training 'noise' class was
built. Worth keeping in mind when reading the class distribution.

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
     machine, e.g. into the OneDrive results folder.
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
RUN_EXTRACTION     = True    # Phase 1 -- cluster: fetch/preprocess/window/spectrogram, no TF import
RUN_CLASSIFICATION = True    # Phase 2 -- local: load packed spectrograms + model, classify
# (both True together still works for a small local-only smoke test, if you
# ever have SDS access AND a working TensorFlow in the same place)

# -- Interchange directory between the two phases ---------------------------------
# Phase 1 writes packed per-station-day spectrogram archives here (under
# EXTRACTION_DIR/packed/). Copy that folder from the cluster to your local
# machine, then repoint EXTRACTION_DIR at the local copy before Phase 2.
EXTRACTION_DIR = "/data/failles/louisels/project/results/outputs_09a_packed"

# -- Paths (Phase 1 only -- cluster) ----------------------------------------------
SDS_ROOT    = "/data/sig/SDS"
ISTERRE_URL = "http://ist-sc3-geobs.osug.fr:8080"

# -- Output for THIS invocation's own log + (Phase 2 only) predictions/review imgs -
OUTPUT_DIR = "/data/failles/louisels/project/results/outputs_09a"

# -- Trained CNN model (Phase 2 only -- local, downloaded from Google Drive) -----
# From run_log_20260812_130248 -- a representative 5-class custom-CNN result
# (acc 91%, macro F1 0.82), NOT cherry-picked as the single best of the 3 repeats.
# MUST come from the SAME run as NORM_STATS_PATH below -- mixing model weights
# from one run with normalization stats from another will silently corrupt
# every prediction. Adjust these two paths to wherever you put them locally.
MODEL_PATH      = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\cnn_models\run_20260812_130248\spectrogram_cnn_final.keras"
NORM_STATS_PATH = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\cnn_models\run_20260812_130248\normalization_stats.npz"

# -- Class order -- MUST exactly match CLASS_NAMES in 07b's Cell 3 at the time ---
# this model was trained (baked into the Dense(5) output index order; get this
# wrong and every prediction is silently mislabeled). Confirmed against the
# "Label encoding" line printed at the top of run_log_20260812_130248.txt:
#   {'earthquake': 0, 'regional': 1, 'rockslide': 2, 'ice quake': 3, 'noise': 4}
CLASS_NAMES = ['earthquake', 'regional', 'rockslide', 'ice quake', 'noise']

# -- Spatial bounding box (Mont Blanc massif -- same box used everywhere else
#    in this project: 01/02b/03a/04a/04c/04d/06c/08a/08b) -- Phase 1 only ---------
LAT_MIN, LAT_MAX = 45.5, 46.0
LON_MIN, LON_MAX = 6.5,  7.2

# -- Channel selection (Phase 1 only) ----------------------------------------------
Z_CHANNEL = "??Z"
HORIZONTAL_SUFFIXES = [("N", "E"), ("2", "1")]   # same fallback convention as 07a

# -- Months to scan (Phase 1 only) -- EDIT to your exact dates; each entry is
# [T_START, T_END) (T_END exclusive, ISO date strings). Confirmed: SDS archive
# has continuous data reaching 2026 -- no coverage concern for these. --------------
MONTHS_TO_SCAN = [
    ("2025-01-01", "2025-02-01"),   # January
    ("2025-07-01", "2025-08-01"),   # a summer month
]

# -- Window / spectrogram parameters -- MUST match 07a EXACTLY (same trained model) --
TARGET_FS      = 100
WINDOW_PRE_S   = 5      # kept for documentation only -- there is no onset here,
WINDOW_POST_S  = 95     # window length is just WINDOW_PRE_S + WINDOW_POST_S = 100 s
WINDOW_S       = WINDOW_PRE_S + WINDOW_POST_S
NT             = int(WINDOW_S * TARGET_FS)

SPEC_NPERSEG_S      = 2.0
SPEC_NOVERLAP_FRAC  = 0.75
SPEC_NFFT           = 256
FREQ_MAX_KEEP       = 45.0
SPEC_NPERSEG        = int(SPEC_NPERSEG_S * TARGET_FS)
SPEC_NOVERLAP       = int(SPEC_NPERSEG * SPEC_NOVERLAP_FRAC)

# -- Detection: spectrogram-based STA/LTA (Groult et al. 2026), Phase 1 only ------
# Same method + parameters as 02b_spectrogram_sta_lta_detection.py's catalog-less
# scanner -- proven values already used elsewhere in this project, not re-tuned
# here. Spectrograms are now only built for detected events, not every window,
# so this drives both compute AND storage cost (far less of either than v1).
DET_FREQ_MIN  = 1.0
DET_FREQ_MAX  = 20.0    # covers ice quakes and quarry blasts with energy above 10 Hz
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
PACK_DTYPE = "float16"   # halves size vs float32, same convention as
                         # 07a_consolidate_for_colab.py's PACK_DTYPE

# -- Classification batch size (matches BATCH_SIZE used in 07b training) ----------
BATCH_SIZE = 128

# -- Review image saving policy (Phase 2 only -- now that the predicted class is
#    actually known, unlike Phase 1 which must keep everything) -------------------
SAVE_IMAGES_FOR_NONNOISE = True    # save every window NOT predicted as 'noise'
SAVE_EVERY_NTH_NOISE     = 2000    # + 1 in every N noise-predicted windows, as a
                                    # background sanity sample (0 = save none)

# -- Review gallery (Phase 2 only) -- render a sample of the saved review images
#    as PNGs (RGB composite, R=Z G=N B=E), so you can actually look at what the
#    model is calling each class without opening .npz files by hand ------------
PLOT_REVIEW_GALLERY = True
N_GALLERY_PER_CLASS = 10    # how many examples per predicted class to plot
                            # (0 = skip that class); picks the HIGHEST-CONFIDENCE
                            # examples first (highest max-probability), so you're
                            # looking at the model's clearest calls, not a random draw

# -- Checkpointing (Phase 1: station-days, Phase 2: packed files) -----------------
CHECKPOINT_EVERY_STATION_DAYS = 5
CHECKPOINT_EVERY_PACKED_FILES = 20

# -- SMOKE TEST (Phase 1 only) -- strongly recommended before the full run --------
# Restricts every month to MAX_DAYS_SMOKE_TEST day(s) and MAX_STATIONS_SMOKE_TEST
# station(s), so the printed timing + storage estimate reflects a real, small run
# you can extrapolate from before committing hours of cluster time / hundreds of
# GB of storage to the full MONTHS_TO_SCAN.
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

packed_dir = os.path.join(EXTRACTION_DIR, "packed")
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

    if PLOT_REVIEW_GALLERY:
        from run_setup import set_matplotlib_defaults
        from visualization import plot_spectrogram_rgb_example
        set_matplotlib_defaults()
        gallery_dir = os.path.join(RUN_DIR, "review_gallery")
        os.makedirs(gallery_dir, exist_ok=True)



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

    print(f"\n[PHASE 1 COMPLETE] Packed spectrograms -> {packed_dir}/")
    print(f"  Copy this folder to your local machine, then run Phase 2 with")
    print(f"  RUN_EXTRACTION=False, RUN_CLASSIFICATION=True, and EXTRACTION_DIR")
    print(f"  pointed at the local copy.")



# =============================================================================
# SECTION 5B — PHASE 2: CLASSIFICATION (local, TensorFlow)
# =============================================================================

if RUN_CLASSIFICATION:

    packed_files = sorted(glob.glob(os.path.join(packed_dir, "*.npz")))
    print(f"\n{'='*70}")
    print(f"  PHASE 2 — CLASSIFICATION")
    print(f"  {len(packed_files)} packed station-day file(s) found in {packed_dir}")
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
            net    = str(d["network"])
            sta    = str(d["station"])
            loc    = str(d["location"])
            chan   = str(d["channel"])
            day    = str(d["day"])

        if images.shape[1:] != (N_FREQ, N_TIME, 3):
            print(f"  [SKIP] {os.path.basename(fpath)} — shape mismatch "
                  f"{images.shape[1:]} != ({N_FREQ}, {N_TIME}, 3)")
            continue

        month_tag = day[:7]
        rows_by_month.setdefault(month_tag, [])

        n = images.shape[0]
        for b0 in range(0, n, BATCH_SIZE):
            b1 = min(b0 + BATCH_SIZE, n)
            batch = images[b0:b1].astype(np.float32)
            labels, proba = classify_batch(batch)
            for i in range(b1 - b0):
                meta = {
                    "network": net, "station": sta, "location": loc, "channel": chan,
                    "day": day, "window_start": str(starts[b0 + i]), "window_end": str(ends[b0 + i]),
                    "snr": float(snrs[b0 + i]) if snrs is not None else None,
                    "snr_full_median": float(snr_fm[b0 + i]) if snr_fm is not None else None,
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
