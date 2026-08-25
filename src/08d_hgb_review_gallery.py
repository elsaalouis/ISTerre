"""
08d_hgb_review_gallery.py
==========================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : August 2026

Goal
----
09b's (HGB) and 09a's (CNN) own review galleries only save N_GALLERY_PER_CLASS
=10 images per class, always the HIGHEST-confidence windows for each
predicted class. That's fine as a quick sanity check, but it can't answer
"is this class over-triggering on marginal/borderline detections?" -- the
exact question raised by the earthquake-heavy class distribution seen in the
retrained 09b runs (Jan/Aug 2025: 63-69% of all detections predicted
'earthquake', with a mean-proba confusion matrix showing earthquake as the
single largest runner-up on every OTHER predicted class too).

PIPELINE (Section 1) picks which classifier's results to review -- "HGB" or
"CNN" -- everything else below (N_PER_CLASS, SELECTION_MODE, CLASSES_TO_REVIEW,
filters) works identically for both:
  - reads one or more EXISTING predictions_<month>.csv file(s) that 09b's
    (HGB) or 09a's (CNN) Phase 2 already produced -- no re-extraction, no
    re-classification;
  - lets you pick MANY MORE examples per class (N_PER_CLASS, default 50
    vs 09a/09b's own galleries' 10);
  - lets you choose HOW they're picked (SELECTION_MODE): a random sample
    (unbiased plausibility check), the lowest-confidence calls (the most
    ambiguous/borderline windows -- exactly what you want when checking
    "does this class over-trigger on weak detections?"), the lowest-SNR
    calls, or the same top-confidence behaviour 09a/09b's own galleries use
    (for side-by-side comparison).
  - optionally restricts to one class at a time (CLASSES_TO_REVIEW) so you
    can, e.g., pull 100 more 'earthquake' examples without touching the
    other four classes.

HOW THE TWO PIPELINES DIFFER UNDER THE HOOD
---------------------------------------------
  PIPELINE = "HGB" -- there is no pre-built image to reuse (HGB classifies
    scalar features, not pictures), so this RE-FETCHES the raw waveform from
    SDS for each selected window and builds a fresh 2-panel waveform+
    spectrogram figure, same as 09b's Phase 2d / 08a's report figures. Needs
    SDS (and ideally FDSN for calibrated amplitudes -- see the uncalibrated
    fallback below) -- cluster or VPN.

  PIPELINE = "CNN" -- 09a's Phase 2 already saved the EXACT spectrogram image
    the CNN was fed (review_images/*.npz, one per reviewed window, alongside
    predictions_<month>.csv in the same run folder), so this just loads and
    re-renders that image with the same RGB-composite style as 09a's own
    gallery (visualization.plot_spectrogram_rgb_example). No SDS/FDSN, no
    waveform re-fetch, no cluster access needed at all -- point
    PREDICTIONS_CSVS_CNN at your local copy of a 09a Phase 2 run folder and
    run this locally. Only windows 09a actually chose to save an image for
    (SAVE_IMAGES_FOR_NONNOISE / SAVE_EVERY_NTH_NOISE in 09a Section 1) are
    reviewable here -- rows with no saved image are skipped.

Output layout
-------------
  OUTPUT_DIR/run_YYYYMMDD_HHMMSS/
      review_gallery/<class>/<rank>_p<proba>_snr<snr>_<net>_<sta>_<time>.png
      run.log
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# -- Which pipeline's results to review ----------------------------------------
# "HGB" -- 06c/09b (scalar-feature classifier)  |  "CNN" -- 07b/09a (spectrogram classifier)
PIPELINE = "CNN"

# -- Input: one or more predictions_<month>.csv files written by 09b Phase 2 --
PREDICTIONS_CSVS_HGB = [
    "/data/failles/louisels/project/results/outputs_09b/run_20260821_173942/predictions_2025-01.csv",
    "/data/failles/louisels/project/results/outputs_09b/run_20260821_173942/predictions_2025-08.csv",
]

# -- Input: one or more predictions_<month>.csv files written by 09a Phase 2 --
PREDICTIONS_CSVS_CNN = [
    r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\09a_continuous_data_test\2025-01_45Hz_20260817_150800\predictions_2025-01.csv",
    r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\09a_continuous_data_test\2025-08_45Hz\predictions_2025-08.csv",
]

# -- CNN spectrogram-image axes (only used when PIPELINE == "CNN") ------------
CNN_HIGH_FREQ_MODE = False   # must match that 09a run's HIGH_FREQ_MODE
if CNN_HIGH_FREQ_MODE:
    CNN_TARGET_FS, CNN_SPEC_NFFT, CNN_FREQ_MAX_KEEP = 200, 512, 80.0
else:
    CNN_TARGET_FS, CNN_SPEC_NFFT, CNN_FREQ_MAX_KEEP = 100, 256, 45.0
CNN_WINDOW_PRE_S, CNN_WINDOW_POST_S = 5, 95     # must match 09a's WINDOW_PRE_S/POST_S
CNN_WINDOW_S           = CNN_WINDOW_PRE_S + CNN_WINDOW_POST_S
CNN_SPEC_NPERSEG_S     = 2.0
CNN_SPEC_NOVERLAP_FRAC = 0.75
CNN_SPEC_NPERSEG       = int(CNN_SPEC_NPERSEG_S * CNN_TARGET_FS)
CNN_SPEC_NOVERLAP      = int(CNN_SPEC_NPERSEG * CNN_SPEC_NOVERLAP_FRAC)

# -- Paths (waveform re-fetch, HGB only, same convention as 09b/08a) ----------
SDS_ROOT    = "/data/sig/SDS"
ISTERRE_URL = "http://ist-sc3-geobs.osug.fr:8080"
OUTPUT_DIR  = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\08d_end-to-end_review_gallery"

# -- Spatial bounding box (Mont Blanc massif, same as everywhere else) --------
LAT_MIN, LAT_MAX = 45.5, 46.0
LON_MIN, LON_MAX = 6.5,  7.2

# -- What to review -------------------------------------------------------------
CLASS_ORDER = ["earthquake", "regional", "rockslide", "ice quake", "noise"]
CLASSES_TO_REVIEW = list(CLASS_ORDER)   # trim to e.g. ["earthquake"] to focus on one class
N_PER_CLASS = 50                        # vs 09b's N_GALLERY_PER_CLASS=10

# -- How to pick which N_PER_CLASS windows to plot, per class -----------------
#   "random"          -- uniform random sample (unbiased overall plausibility check)
#   "low_confidence"  -- lowest winning probability first (the most ambiguous/
#                         borderline calls -- best choice for "is this class
#                         over-triggering on weak detections?")
#   "low_snr"         -- lowest SNR first (weakest raw signal, independent of
#                         what the model itself thought)
#   "top_confidence"  -- highest winning probability first (matches 09b's own
#                         gallery selection -- use for side-by-side comparison)
SELECTION_MODE = "low_confidence"
RANDOM_SEED    = 42   # only used when SELECTION_MODE == "random"

# -- Optional pre-filters, applied BEFORE selection (None = disabled) ---------
SNR_MIN   = None   # e.g. 0.0  to drop rows with missing/degenerate SNR
SNR_MAX   = None   # e.g. 5.0  to focus on already-suspicious low-SNR calls
PROBA_MIN = None   # e.g. 0.2  to drop near-chance-level calls entirely
PROBA_MAX = None   # e.g. 0.5  to focus on genuinely uncertain calls

# -- Review figure style (identical to 09b's Phase 2d / 08a's example gallery) -
REVIEW_PRE_S       = 10     # seconds BEFORE the detection onset shown in the figure
REVIEW_WINDOW_S    = 100    # total fixed display window [s]
REVIEW_TARGET_FS   = 200    # [Hz] resample target before the spectrogram
REVIEW_FETCH_PAD_S = 60     # extra context fetched (not shown) so filtering has run-in room
REVIEW_WAVE_FREQMIN, REVIEW_WAVE_FREQMAX = 1.0, 20.0

REVIEW_SPEC_NPERSEG_S     = 2.0     # [s] STFT segment length
REVIEW_SPEC_NOVERLAP_FRAC = 0.75
REVIEW_SPEC_NFFT          = 512
REVIEW_FREQ_MAX_KEEP      = 95.0    # [Hz] 95% of Nyquist at REVIEW_TARGET_FS=200Hz
REVIEW_SPEC_VMIN, REVIEW_SPEC_VMAX = -200, -120   # dB color scale
REVIEW_SPEC_NPERSEG  = int(REVIEW_SPEC_NPERSEG_S * REVIEW_TARGET_FS)
REVIEW_SPEC_NOVERLAP = int(REVIEW_SPEC_NPERSEG * REVIEW_SPEC_NOVERLAP_FRAC)
REVIEW_PSD_FLOOR_EPS = 1e-20



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
from run_setup import create_run_dir, setup_logging, connect_sds, connect_fdsn, fetch_inventory, set_matplotlib_defaults

if PIPELINE not in ("HGB", "CNN"):
    print(f"[ERROR] PIPELINE must be 'HGB' or 'CNN', got '{PIPELINE}'.")
    sys.exit(1)

_PREDICTIONS_CSVS = PREDICTIONS_CSVS_HGB if PIPELINE == "HGB" else PREDICTIONS_CSVS_CNN

RUN_DIR, STAMP = create_run_dir(OUTPUT_DIR)
log_file, log_path = setup_logging(
    RUN_DIR, "08d_hgb_review_gallery.py",
    extra_info=(f"PIPELINE={PIPELINE}  |  SELECTION_MODE={SELECTION_MODE}  |  "
                f"N_PER_CLASS={N_PER_CLASS}  |  CLASSES_TO_REVIEW={CLASSES_TO_REVIEW}  |  "
                f"{len(_PREDICTIONS_CSVS)} predictions file(s)")
)

if SELECTION_MODE not in ("random", "low_confidence", "low_snr", "top_confidence"):
    print(f"[ERROR] SELECTION_MODE must be one of random/low_confidence/low_snr/top_confidence, "
          f"got '{SELECTION_MODE}'.")
    log_file.close()
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import spectrogram as scipy_spectrogram
from obspy import UTCDateTime

set_matplotlib_defaults()
from visualization import plot_waveform_spectrogram_example, plot_spectrogram_rgb_example

review_dir = os.path.join(RUN_DIR, "review_gallery")
os.makedirs(review_dir, exist_ok=True)



# =============================================================================
# SECTION 3 — LOAD PREDICTIONS
# =============================================================================

print(f"\n{'='*70}")
print(f"  LOADING PREDICTIONS  (PIPELINE={PIPELINE})")
print(f"{'='*70}")

_frames = []
for fpath in _PREDICTIONS_CSVS:
    if not os.path.isfile(fpath):
        print(f"  [WARN] Not found, skipping: {fpath}")
        continue
    df = pd.read_csv(fpath, low_memory=False)
    if PIPELINE == "CNN":
        # review_images/*.npz lives as a sibling folder of predictions_<month>.csv
        # in 09a's own output layout -- tag each row so Section 5 knows where
        # to look, even when pooling CSVs from more than one 09a run folder.
        df["_source_review_dir"] = os.path.join(os.path.dirname(fpath), "review_images")
    print(f"  [OK] {os.path.basename(fpath)}  {len(df):,} row(s)")
    _frames.append(df)

if not _frames:
    print(f"[ERROR] No predictions CSV could be loaded. Check PREDICTIONS_CSVS_{PIPELINE}. Exiting.")
    log_file.close()
    sys.exit(1)

df_all = pd.concat(_frames, ignore_index=True)

_required = {"network", "station", "channel", "window_start", "predicted_class"}
_missing  = _required - set(df_all.columns)
if _missing:
    print(f"[ERROR] Predictions CSV missing required column(s): {_missing}. Exiting.")
    log_file.close()
    sys.exit(1)

# 09a's CNN predictions use lowercase "snr" (09b's HGB predictions use "SNR")
# -- normalize to "SNR" so the SNR_MIN/MAX filters and titles below work
# unchanged regardless of PIPELINE.
if "SNR" not in df_all.columns and "snr" in df_all.columns:
    df_all = df_all.rename(columns={"snr": "SNR"})

print(f"\n  Pooled: {len(df_all):,} window(s) across {len(_frames)} file(s)")



# =============================================================================
# SECTION 4 — HELPER FUNCTIONS (literal copies from 09b / 08a -- see their
# own docstrings for why these are copied rather than imported: no hidden
# dependency on another script's CONFIGURATION section changing underneath it)
# =============================================================================

def _fetch_padded_trace(client_sds, inventory, net, sta, chan, det_start,
                        pre_s, window_s, target_fs, fetch_pad_s):
    """Literal copy of 08a/09b's _fetch_padded_trace()."""
    from preprocessing import build_station_times_df, remove_response_or_fallback

    t_on    = det_start - pre_s
    t_off   = t_on + window_s
    t_on_p  = t_on - fetch_pad_s
    t_off_p = t_off + fetch_pad_s
    try:
        st_raw = client_sds.get_waveforms(net, sta, "*", chan, t_on_p, t_off_p)
        if len(st_raw) == 0:
            return None, None, None, "no waveform in SDS"
        st_raw.merge(method=1, fill_value="interpolate")

        sdf    = build_station_times_df(st_raw, t_on_p, t_off_p)
        st_vel = remove_response_or_fallback(st_raw, inventory, sdf)
        if len(st_vel) == 0:
            return None, None, None, "response removal failed"

        tr = st_vel[0].copy()
        if abs(tr.stats.sampling_rate - target_fs) > 0.5:
            tr.resample(target_fs)

        if not np.all(np.isfinite(tr.data)) or np.max(np.abs(tr.data)) == 0:
            return None, None, None, "degenerate trace (NaN/Inf/all-zero)"
        return tr, t_on, t_off, None
    except Exception as e:
        return None, None, None, str(e)


def _trim_to_fixed_length(tr, t_on, t_off, target_fs, window_s):
    """Literal copy of 08a/09b's _trim_to_fixed_length()."""
    tr = tr.copy()
    tr.trim(t_on, t_off, pad=True, fill_value=0)
    nt = int(round(window_s * target_fs))
    if len(tr.data) < nt:
        tr.data = np.pad(tr.data, (0, nt - len(tr.data)))
    elif len(tr.data) > nt:
        tr.data = tr.data[:nt]
    return tr


def plot_review_waveform(client_sds, inventory, row, cls_name, top_proba, out_path, calibrated=True):
    """
    Same 2-panel style (bandpassed waveform + broadband dB spectrogram) as
    09b's Phase 2d / 08a's report figures -- adapted from 09b's function of
    the same name, unchanged in behaviour.

    calibrated : bool -- False when called with inventory=None (FDSN down).
    remove_response_or_fallback() already degrades gracefully to cleaned-but-
    uncalibrated raw counts in that case (see preprocessing.py), so this
    still produces a usable figure -- shape/duration/frequency content are
    still roughly indicative, only the amplitude axis is not true ground
    velocity. This flag just controls the title annotation so that's obvious
    at a glance rather than silently mislabeled.
    """
    det_start = UTCDateTime(row["window_start"])

    tr_padded, t_on, t_off, err = _fetch_padded_trace(
        client_sds, inventory, row["network"], row["station"], row["channel"], det_start,
        REVIEW_PRE_S, REVIEW_WINDOW_S, REVIEW_TARGET_FS, REVIEW_FETCH_PAD_S,
    )
    if tr_padded is None:
        return False

    tr_broadband = _trim_to_fixed_length(tr_padded, t_on, t_off, REVIEW_TARGET_FS, REVIEW_WINDOW_S)
    if not (np.all(np.isfinite(tr_broadband.data)) and np.max(np.abs(tr_broadband.data)) > 0):
        return False

    tr_wave_padded = tr_padded.copy()
    nyq = tr_wave_padded.stats.sampling_rate / 2.0
    tr_wave_padded.filter("bandpass", freqmin=REVIEW_WAVE_FREQMIN,
                          freqmax=min(REVIEW_WAVE_FREQMAX, 0.9 * nyq),
                          corners=4, zerophase=True)
    tr_wave = _trim_to_fixed_length(tr_wave_padded, t_on, t_off, REVIEW_TARGET_FS, REVIEW_WINDOW_S)

    f_full, t_full, Sxx = scipy_spectrogram(
        tr_broadband.data, fs=tr_broadband.stats.sampling_rate, window="hann",
        nperseg=REVIEW_SPEC_NPERSEG, noverlap=REVIEW_SPEC_NOVERLAP, nfft=REVIEW_SPEC_NFFT,
        scaling="density", mode="psd",
    )
    freq_mask = f_full <= REVIEW_FREQ_MAX_KEEP
    freq_axis = f_full[freq_mask]
    Sxx_db    = 10 * np.log10(Sxx[freq_mask, :] + REVIEW_PSD_FLOOR_EPS)

    # REVIEW_SPEC_VMIN/VMAX are tuned for CALIBRATED m/s amplitudes. Raw
    # digitizer counts run ~10^6-10^9x larger (the same scale factor as the
    # 04a units-bug investigation), so their PSD in dB lands far above that
    # fixed range -- every pixel saturates to vmax, i.e. a uniformly red
    # spectrogram, not a meaningful one. When uncalibrated, auto-scale from
    # this window's own dB range instead, so the relative time-frequency
    # STRUCTURE (still physically meaningful for a shape/pattern check,
    # unlike absolute amplitude) is actually visible.
    if calibrated:
        spec_vmin, spec_vmax = REVIEW_SPEC_VMIN, REVIEW_SPEC_VMAX
    else:
        _finite = Sxx_db[np.isfinite(Sxx_db)]
        if _finite.size:
            spec_vmin, spec_vmax = np.percentile(_finite, [5, 99.5])
            if spec_vmax - spec_vmin < 1e-6:
                spec_vmax = spec_vmin + 1.0
        else:
            spec_vmin, spec_vmax = REVIEW_SPEC_VMIN, REVIEW_SPEC_VMAX

    snr_val   = row.get("SNR", np.nan)
    snr_str   = f"SNR={snr_val:.2f}" if pd.notna(snr_val) else "SNR=n/a"
    cal_tag   = "" if calibrated else "  [UNCALIBRATED — raw counts, not m/s]"
    title_l1  = f"{cls_name} (predicted, p={top_proba:.2f}) — {str(row['window_start'])[:19]}{cal_tag}"
    title_l2  = f"{row['network']}.{row['station']} | {snr_str} | duration={row.get('duration_s', float('nan')):.1f}s"

    plot_waveform_spectrogram_example(
        times_wave     = tr_wave.times() - REVIEW_PRE_S,
        wave_data      = tr_wave.data,
        times_spec     = t_full - REVIEW_PRE_S,
        freq_axis      = freq_axis,
        spec_db        = Sxx_db,
        det_duration_s = row.get("duration_s", 0.0),
        title_lines    = (title_l1, title_l2),
        out_path       = out_path,
        spec_vmin      = spec_vmin,
        spec_vmax      = spec_vmax,
    )
    return True


# -- CNN pipeline (PIPELINE == "CNN") only --------------------------------------
# Freq/time axes computed once from a dummy spectrogram, literal copy of 09a
# Section 3's own precompute -- these only LABEL the image, whose pixels are
# read verbatim from the saved .npz, so they must match the CNN_* constants
# in Section 1 (which in turn must match the 09a run being reviewed).
if PIPELINE == "CNN":
    _cnn_nt = int(CNN_WINDOW_S * CNN_TARGET_FS)
    _cnn_dummy = np.zeros(_cnn_nt, dtype=np.float32)
    _cnn_f_full, CNN_TIME_AXIS, _ = scipy_spectrogram(
        _cnn_dummy, fs=CNN_TARGET_FS, window="hann",
        nperseg=CNN_SPEC_NPERSEG, noverlap=CNN_SPEC_NOVERLAP, nfft=CNN_SPEC_NFFT,
        scaling="density", mode="psd",
    )
    _cnn_freq_mask = _cnn_f_full <= CNN_FREQ_MAX_KEEP
    CNN_FREQ_AXIS  = _cnn_f_full[_cnn_freq_mask]
    print(f"  [CNN] Expected image shape: ({len(CNN_FREQ_AXIS)}, {len(CNN_TIME_AXIS)}, 3) "
          f"-- must match review_images/*.npz (check CNN_HIGH_FREQ_MODE if it doesn't)")


def plot_review_spectrogram_cnn(npz_path, cls_name, top_proba, out_path):
    """
    CNN pipeline (PIPELINE == "CNN") counterpart to plot_review_waveform()
    above. Loads a spectrogram image ALREADY saved by 09a Phase 2's
    _emit_row() -- the exact (N_FREQ, N_TIME, 3) [Z, N, E] dB array the CNN
    was fed -- and renders it with the same RGB-composite style as 09a's own
    review gallery (visualization.plot_spectrogram_rgb_example), so this
    looks identical to 09a's gallery, just with 08d's richer N_PER_CLASS /
    SELECTION_MODE / CLASSES_TO_REVIEW controls layered on top. No SDS/FDSN,
    no live re-fetch -- returns False if the .npz is missing/unreadable or
    its image shape doesn't match CNN_FREQ_AXIS/CNN_TIME_AXIS above (most
    likely cause: CNN_HIGH_FREQ_MODE doesn't match the 09a run reviewed).
    """
    if not npz_path or not os.path.isfile(npz_path):
        return False
    try:
        with np.load(npz_path, allow_pickle=False) as d:
            img     = d["image"].astype(np.float32)
            net_i   = str(d["network"])
            sta_i   = str(d["station"])
            w_start = str(d["window_start"])
    except Exception:
        return False

    if img.shape != (len(CNN_FREQ_AXIS), len(CNN_TIME_AXIS), 3):
        return False

    plot_spectrogram_rgb_example(
        CNN_FREQ_AXIS, CNN_TIME_AXIS, img,
        title_lines=(f"{cls_name} (predicted, p={top_proba:.2f})", f"{net_i}.{sta_i}  {w_start}"),
        out_path=out_path,
    )
    return True



# =============================================================================
# SECTION 5 — SELECT WINDOWS PER CLASS + GENERATE FIGURES
# =============================================================================

print(f"\n{'='*70}")
print(f"  SELECTING WINDOWS  (mode={SELECTION_MODE}, {N_PER_CLASS}/class)")
print(f"{'='*70}")

selected_by_class = {}

for cls in CLASSES_TO_REVIEW:
    proba_col = f"proba_{str(cls).replace(' ', '_')}"
    df_cls = df_all[df_all["predicted_class"] == cls].copy()

    if proba_col in df_cls.columns:
        df_cls["_winning_proba"] = df_cls[proba_col]
    else:
        print(f"  [WARN] {cls}: no '{proba_col}' column found -- low_confidence/"
              f"top_confidence selection unavailable, falling back to random.")
        df_cls["_winning_proba"] = np.nan

    n_before = len(df_cls)
    if SNR_MIN is not None and "SNR" in df_cls.columns:
        df_cls = df_cls[df_cls["SNR"] >= SNR_MIN]
    if SNR_MAX is not None and "SNR" in df_cls.columns:
        df_cls = df_cls[df_cls["SNR"] <= SNR_MAX]
    if PROBA_MIN is not None:
        df_cls = df_cls[df_cls["_winning_proba"] >= PROBA_MIN]
    if PROBA_MAX is not None:
        df_cls = df_cls[df_cls["_winning_proba"] <= PROBA_MAX]
    if PIPELINE == "CNN":
        # Only windows 09a actually saved an image for (SAVE_IMAGES_FOR_NONNOISE
        # / SAVE_EVERY_NTH_NOISE) are reviewable -- filter BEFORE selection so
        # N_PER_CLASS reflects what will actually get plotted, not candidates
        # that would silently fail in the generation loop below.
        if "review_image_saved" in df_cls.columns:
            df_cls = df_cls[df_cls["review_image_saved"].fillna("") != ""]
        else:
            print(f"  [WARN] {cls}: no 'review_image_saved' column -- predictions CSV "
                  f"may not be from 09a Phase 2. Nothing reviewable for this class.")
            df_cls = df_cls.iloc[0:0]
    n_after_filter = len(df_cls)

    if df_cls.empty:
        print(f"  {cls:<12s} 0 candidate(s) after filters (had {n_before} before) -- skipping.")
        selected_by_class[cls] = df_cls
        continue

    mode = SELECTION_MODE
    if mode in ("low_confidence", "top_confidence") and df_cls["_winning_proba"].isna().all():
        mode = "random"

    if mode == "random":
        picked = df_cls.sample(n=min(N_PER_CLASS, len(df_cls)), random_state=RANDOM_SEED)
    elif mode == "low_confidence":
        picked = df_cls.sort_values("_winning_proba", ascending=True).head(N_PER_CLASS)
    elif mode == "top_confidence":
        picked = df_cls.sort_values("_winning_proba", ascending=False).head(N_PER_CLASS)
    elif mode == "low_snr":
        if "SNR" not in df_cls.columns:
            print(f"  [WARN] {cls}: no 'SNR' column -- falling back to random.")
            picked = df_cls.sample(n=min(N_PER_CLASS, len(df_cls)), random_state=RANDOM_SEED)
        else:
            picked = df_cls.dropna(subset=["SNR"]).sort_values("SNR", ascending=True).head(N_PER_CLASS)

    selected_by_class[cls] = picked
    print(f"  {cls:<12s} {len(picked):4d} selected  ({n_after_filter} candidate(s) after filters, "
          f"{n_before} total predicted)")

_all_selected = pd.concat([d for d in selected_by_class.values() if not d.empty], ignore_index=True) \
                if any(not d.empty for d in selected_by_class.values()) else pd.DataFrame()
if _all_selected.empty:
    print("[ERROR] Nothing selected across any class (check filters/CLASSES_TO_REVIEW). Exiting.")
    log_file.close()
    sys.exit(1)

if PIPELINE == "HGB":
    print(f"\n{'='*70}")
    print(f"  FETCHING INVENTORY + GENERATING FIGURES  (re-fetching waveforms from SDS)")
    print(f"{'='*70}")

    client_sds  = connect_sds(SDS_ROOT)
    client_fdsn = connect_fdsn(ISTERRE_URL)
    if client_sds is None:
        # Only SDS is a hard requirement -- it's where the actual waveform samples
        # live. FDSN/inventory is needed for CALIBRATED (m/s) amplitudes, but
        # remove_response_or_fallback() already degrades gracefully to cleaned,
        # uncalibrated raw counts when inventory=None (see preprocessing.py) --
        # so an FDSN outage no longer has to block review figures entirely, only
        # their amplitude calibration. See the inventory fetch below.
        print("[ERROR] Cannot proceed without SDS. Exiting.")
        log_file.close()
        sys.exit(1)

    inventory = None
    if client_fdsn is not None:
        _t_min = pd.to_datetime(_all_selected["window_start"]).min()
        _t_max = pd.to_datetime(_all_selected["window_start"]).max()
        inventory = fetch_inventory(
            client_fdsn, str(_t_min.date()), str((_t_max + pd.Timedelta(days=1)).date()),
            lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX,
        )

    if inventory is None:
        print("  [WARN] No instrument inventory (FDSN unreachable or inventory fetch failed) --")
        print("         continuing with UNCALIBRATED raw-counts waveforms. Shape, duration, and")
        print("         frequency content are still roughly indicative for a visual class check;")
        print("         amplitude/true ground velocity are NOT. Figures are tagged 'UNCALIBRATED'")
        print("         in the title and '_UNCAL' in the filename -- rerun once FDSN is back for")
        print("         calibrated figures if precise amplitudes matter.")
    else:
        print(f"  [OK] Instrument inventory fetched -- figures will be calibrated ground velocity.")

    for cls, picked in selected_by_class.items():
        if picked.empty:
            continue
        cls_dir = os.path.join(review_dir, cls.replace(" ", "_"))
        os.makedirs(cls_dir, exist_ok=True)

        n_plotted = 0
        for rank, (_, row) in enumerate(picked.iterrows(), 1):
            top_p    = float(row.get("_winning_proba", np.nan)) if pd.notna(row.get("_winning_proba", np.nan)) else 0.0
            snr_val  = row.get("SNR", np.nan)
            snr_tag  = f"{snr_val:05.1f}" if pd.notna(snr_val) else "na"
            safe_time  = str(row["window_start"]).replace(":", "").replace("-", "").replace(".", "")
            cal_suffix = "" if inventory is not None else "_UNCAL"
            out_png = os.path.join(
                cls_dir,
                f"{rank:03d}_p{top_p:.2f}_snr{snr_tag}_{row['network']}_{row['station']}_{safe_time}{cal_suffix}.png",
            )
            try:
                ok = plot_review_waveform(client_sds, inventory, row, cls, top_p, out_png,
                                          calibrated=(inventory is not None))
                if ok:
                    n_plotted += 1
            except Exception:
                continue

        print(f"  {cls:<12s} {n_plotted:4d}/{len(picked)} waveform(s) plotted -> {cls_dir}/")

else:   # PIPELINE == "CNN"
    print(f"\n{'='*70}")
    print(f"  GENERATING FIGURES  (re-rendering saved 09a spectrogram images -- no SDS/FDSN)")
    print(f"{'='*70}")

    for cls, picked in selected_by_class.items():
        if picked.empty:
            continue
        cls_dir = os.path.join(review_dir, cls.replace(" ", "_"))
        os.makedirs(cls_dir, exist_ok=True)

        n_plotted = 0
        n_no_image = 0
        for rank, (_, row) in enumerate(picked.iterrows(), 1):
            top_p   = float(row.get("_winning_proba", np.nan)) if pd.notna(row.get("_winning_proba", np.nan)) else 0.0
            snr_val = row.get("SNR", np.nan)
            snr_tag = f"{snr_val:05.1f}" if pd.notna(snr_val) else "na"
            safe_time = str(row["window_start"]).replace(":", "").replace("-", "").replace(".", "")
            fname   = row.get("review_image_saved", "")
            if not fname:
                n_no_image += 1
                continue
            npz_path = os.path.join(row.get("_source_review_dir", ""), fname)
            out_png  = os.path.join(
                cls_dir,
                f"{rank:03d}_p{top_p:.2f}_snr{snr_tag}_{row['network']}_{row['station']}_{safe_time}.png",
            )
            try:
                ok = plot_review_spectrogram_cnn(npz_path, cls, top_p, out_png)
                if ok:
                    n_plotted += 1
            except Exception:
                continue

        _skip_note = f"  ({n_no_image} had no saved image)" if n_no_image else ""
        print(f"  {cls:<12s} {n_plotted:4d}/{len(picked)} spectrogram(s) plotted -> {cls_dir}/{_skip_note}")



# =============================================================================
# SECTION 6 — END
# =============================================================================

print(f"\n{'='*70}")
print(f"  Run finished : {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  All outputs  : {RUN_DIR}")
print(f"  Log file     : {log_path}")
print(f"{'='*70}")

log_file.close()
