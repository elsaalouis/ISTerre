"""
08e_hgb_cnn_qualitative_comparison.py
=======================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : August 2026

Goal
----
09a (CNN) and 09b (HGB) both run the SAME detector (DetecteurV3, byte-for-byte
identical detection -- see 09b's own docstring) over the same continuous
data, so for a given month/station they see the SAME candidate windows,
just classified independently and differently by each model. That makes a
clean, honest qualitative comparison possible -- not "model A beats model B"
(no ground truth exists on continuous data to support that), but: "on an
UNBIASED random sample, how often does each model's prediction visually
match the expected profile, and when the two models disagree, do they
disagree in a similar way or a genuinely different way?"

This script:
  1. Pairs up HGB (09b) and CNN (09a) predictions for the SAME underlying
     detected window (matched on network+station+onset time, NOT cherry-
     picked -- see MATCHING below), for one month.
  2. Draws a RANDOM sample (not "the cases that stood out") of N_PER_CLASS
     windows per PREDICTED CLASS, independently for each model -- so you get
     an honest cross-section of what each model is actually calling that
     class, not a hand-picked highlight reel.
  3. For every window that lands in either sample, renders BOTH models'
     review figure side by side in the same folder: HGB's 2-panel waveform+
     spectrogram (re-fetched from SDS, same as 08d/09b) and CNN's RGB
     spectrogram (re-rendered from 09a's saved review image, same as 08d).
  4. Writes ONE manifest CSV with both models' predicted class + confidence
     for every sampled window, plus BLANK columns for you to fill in by eye
     while looking at the figures (hgb_visual_match / cnn_visual_match /
     notes) -- MODE="summarize" then turns your filled-in manifest straight
     into the report sentence: "sur un echantillon aleatoire de N fenetres
     classees X (modele Y), M ne correspondent pas visuellement au profil
     attendu (2.1.3)", plus a disagreement-type breakdown for the cases
     where the two models actually predicted different classes -- that's
     the potentially interesting, defensible observation even on a small
     sample: not just "the two models disagree at rate R", but "when they
     disagree, do they fail in the SAME way (both getting fooled by the
     same ambiguous signal) or in DIFFERENT ways (each with its own bias)?"

MATCHING (how a HGB row and a CNN row get paired as "the same window")
-------------------------------------------------------------------------
HGB's "window_start" (09b) is the raw detection onset time itself.
CNN's "window_start" (09a) is onset - WINDOW_PRE_S (it packs a FIXED
window starting WINDOW_PRE_S seconds before onset -- see 09a Section 1).
So for the SAME underlying detection, cnn_window_start + CNN_WINDOW_PRE_S_
FOR_MATCH should equal hgb_window_start almost exactly. Pairing is done
with pandas' merge_asof (nearest match within MATCH_TOLERANCE_S, grouped by
network+station) on that basis -- NOT a blind nearest-time guess. The
script prints the observed offset distribution for matched pairs as a
self-check: it should cluster tightly around 0s. If it doesn't, HGB's own
window-construction convention has likely changed since this was written --
widen/adjust MATCH_TOLERANCE_S / CNN_WINDOW_PRE_S_FOR_MATCH accordingly
rather than trusting the pairing blindly.

Needs SDS (+ ideally FDSN, same uncalibrated fallback as 08d/09b) for the
HGB figures -- so this script effectively has to run on the cluster/VPN.
MODE="summarize" needs neither, it only reads a CSV.

CNN IMAGES: consolidated_<month>.npz, NOT review_images/*.npz
-------------------------------------------------------------------------
08d's CNN mode reads 09a Phase 2's review_images/*.npz (a subset of images
re-saved locally, alongside predictions_<month>.csv, on whatever machine ran
Phase 2 -- typically NOT the cluster, since Phase 2 needs TensorFlow). That
folder can be large and inconvenient to copy to the cluster, and this script
needs to run WHERE SDS IS REACHABLE for the HGB side -- so relying on
review_images here would force choosing between SDS access and CNN images.

Instead, this script reads CNN images straight from 09a Phase 1's
consolidated_<month>.npz (CNN_CONSOLIDATED_NPZ below). That file is a Phase-1
(cluster-side, no-TensorFlow) output -- it already lives on the cluster and
never needs copying anywhere. It contains the spectrogram image for EVERY
window CNN classified (not just the review_images subset), indexed here by
(network, station, window_start) against predictions_<month>.csv. All you
need to move to the cluster is predictions_<month>.csv itself (a small CSV,
Phase 2's only lightweight output) -- copy that up next to where
consolidated_<month>.npz already sits, or anywhere convenient, and point
CNN_PREDICTIONS_CSV / CNN_CONSOLIDATED_NPZ at them.

Output layout
-------------
  OUTPUT_DIR/run_YYYYMMDD_HHMMSS/           (MODE == "sample")
      figures/<net>_<sta>_<safe_time>/hgb_p<conf>_<class>.png
      figures/<net>_<sta>_<safe_time>/cnn_p<conf>_<class>.png
      paired_sample_manifest_<month>.csv    <- fill in hgb_visual_match /
                                                cnn_visual_match / notes by
                                                eye, then re-run with
                                                MODE="summarize" pointed at it
      run.log
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# -- "sample" builds the paired random sample + figures + blank manifest.
# -- "summarize" reads a manifest YOU annotated and prints the report numbers.
MODE = "sample"

# -- Input: ONE month's predictions from each pipeline (same month, so the
# comparison is apples-to-apples -- see the module docstring). Used only in
# MODE == "sample". Both paths are CLUSTER paths -- this script needs SDS for
# the HGB figures, so it has to run where SDS is reachable, not locally.
HGB_PREDICTIONS_CSV = "/data/failles/louisels/project/results/outputs_09b/run_20260821_173942/predictions_2025-01.csv"
# Small file -- copy it up from wherever 09a Phase 2 wrote it locally.
CNN_PREDICTIONS_CSV = "/data/failles/louisels/project/results/outputs_09a/predictions_2025-01.csv"
# 09a Phase 1's consolidated archive -- already on the cluster, no copy needed
# (see "CNN IMAGES" in the module docstring for why this replaces review_images).
CNN_CONSOLIDATED_NPZ = "/data/failles/louisels/project/results/09a_continuous_data_test/2025-01_45Hz/packed/consolidated_2025-01.npz"

# -- Optional station restriction (None = every station present in both files) -
STATIONS_FILTER = None   # e.g. ["STA1", "STA2"]

# -- What to sample --------------------------------------------------------------
CLASS_ORDER  = ["earthquake", "regional", "rockslide", "ice quake", "noise"]
N_PER_CLASS  = 10          # per predicted class, per model (her "~10 events" ask)
RANDOM_SEED  = 42          # fixed seed -- reproducible, and NOT hand-picked

# -- Optional pre-filters, applied BEFORE sampling (None = disabled) -------------
# Left off by default on purpose -- an unbiased sample means no cherry-picking,
# including no implicit cherry-picking via aggressive filters.
SNR_MIN   = None
SNR_MAX   = None
PROBA_MIN = None
PROBA_MAX = None

# -- Pairing (see MATCHING in the module docstring) -------------------------------
CNN_WINDOW_PRE_S_FOR_MATCH = 5     # must match 09a's WINDOW_PRE_S for the run used
MATCH_TOLERANCE_S          = 2.0   # slack around the expected offset, seconds

# -- MODE == "summarize" only: path to YOUR annotated copy of a manifest CSV
# this script produced (hgb_visual_match / cnn_visual_match filled in with
# "yes"/"no"/"unsure") -----------------------------------------------------------
ANNOTATED_MANIFEST_CSV = None

# -- Paths (HGB figures only -- waveform re-fetch, same convention as 08d/09b) ---
SDS_ROOT    = "/data/sig/SDS"
ISTERRE_URL = "http://ist-sc3-geobs.osug.fr:8080"
OUTPUT_DIR  = "/data/failles/louisels/project/results/outputs_08e"
LAT_MIN, LAT_MAX = 45.5, 46.0
LON_MIN, LON_MAX = 6.5,  7.2

# -- HGB review figure style (identical to 08d/09b/08a) ---------------------------
REVIEW_PRE_S       = 10
REVIEW_WINDOW_S    = 100
REVIEW_TARGET_FS   = 200
REVIEW_FETCH_PAD_S = 60
REVIEW_WAVE_FREQMIN, REVIEW_WAVE_FREQMAX = 1.0, 20.0
REVIEW_SPEC_NPERSEG_S     = 2.0
REVIEW_SPEC_NOVERLAP_FRAC = 0.75
REVIEW_SPEC_NFFT          = 512
REVIEW_FREQ_MAX_KEEP      = 95.0
REVIEW_SPEC_VMIN, REVIEW_SPEC_VMAX = -200, -120
REVIEW_SPEC_NPERSEG  = int(REVIEW_SPEC_NPERSEG_S * REVIEW_TARGET_FS)
REVIEW_SPEC_NOVERLAP = int(REVIEW_SPEC_NPERSEG * REVIEW_SPEC_NOVERLAP_FRAC)
REVIEW_PSD_FLOOR_EPS = 1e-20

# -- CNN spectrogram-image axes (identical to 08d -- must match the 09a run used) -
CNN_HIGH_FREQ_MODE = False
if CNN_HIGH_FREQ_MODE:
    CNN_TARGET_FS, CNN_SPEC_NFFT, CNN_FREQ_MAX_KEEP = 200, 512, 80.0
else:
    CNN_TARGET_FS, CNN_SPEC_NFFT, CNN_FREQ_MAX_KEEP = 100, 256, 45.0
CNN_WINDOW_PRE_S, CNN_WINDOW_POST_S = 5, 95
CNN_WINDOW_S           = CNN_WINDOW_PRE_S + CNN_WINDOW_POST_S
CNN_SPEC_NPERSEG_S     = 2.0
CNN_SPEC_NOVERLAP_FRAC = 0.75
CNN_SPEC_NPERSEG       = int(CNN_SPEC_NPERSEG_S * CNN_TARGET_FS)
CNN_SPEC_NOVERLAP      = int(CNN_SPEC_NPERSEG * CNN_SPEC_NOVERLAP_FRAC)



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

if MODE not in ("sample", "summarize"):
    print(f"[ERROR] MODE must be 'sample' or 'summarize', got '{MODE}'.")
    sys.exit(1)

if MODE == "summarize":
    if not ANNOTATED_MANIFEST_CSV or not os.path.isfile(ANNOTATED_MANIFEST_CSV):
        print(f"[ERROR] MODE='summarize' needs ANNOTATED_MANIFEST_CSV pointed at an "
              f"existing, annotated manifest CSV. Got: {ANNOTATED_MANIFEST_CSV}")
        sys.exit(1)

RUN_DIR, STAMP = create_run_dir(OUTPUT_DIR)
log_file, log_path = setup_logging(
    RUN_DIR, "08e_hgb_cnn_qualitative_comparison.py",
    extra_info=(f"MODE={MODE}  |  N_PER_CLASS={N_PER_CLASS}  |  "
                f"HGB_PREDICTIONS_CSV={HGB_PREDICTIONS_CSV if MODE == 'sample' else 'n/a'}  |  "
                f"CNN_PREDICTIONS_CSV={CNN_PREDICTIONS_CSV if MODE == 'sample' else 'n/a'}  |  "
                f"CNN_CONSOLIDATED_NPZ={CNN_CONSOLIDATED_NPZ if MODE == 'sample' else 'n/a'}  |  "
                f"ANNOTATED_MANIFEST_CSV={ANNOTATED_MANIFEST_CSV if MODE == 'summarize' else 'n/a'}")
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import spectrogram as scipy_spectrogram
from obspy import UTCDateTime

set_matplotlib_defaults()
from visualization import plot_waveform_spectrogram_example, plot_spectrogram_rgb_example

figures_dir = os.path.join(RUN_DIR, "figures")
if MODE == "sample":
    os.makedirs(figures_dir, exist_ok=True)



# =============================================================================
# SECTION 3 — HELPER FUNCTIONS (literal copies from 08d -- see 08d's own
# docstring for why: no hidden dependency on another script's CONFIGURATION
# section changing underneath it)
# =============================================================================

def _fetch_padded_trace(client_sds, inventory, net, sta, chan, det_start,
                        pre_s, window_s, target_fs, fetch_pad_s):
    """Literal copy of 08a/09b/08d's _fetch_padded_trace()."""
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
    """Literal copy of 08a/09b/08d's _trim_to_fixed_length()."""
    tr = tr.copy()
    tr.trim(t_on, t_off, pad=True, fill_value=0)
    nt = int(round(window_s * target_fs))
    if len(tr.data) < nt:
        tr.data = np.pad(tr.data, (0, nt - len(tr.data)))
    elif len(tr.data) > nt:
        tr.data = tr.data[:nt]
    return tr


def plot_review_waveform(client_sds, inventory, net, sta, chan, window_start, snr_val,
                         cls_name, top_proba, out_path, calibrated=True, duration_s=0.0):
    """
    HGB side of the pair. Same 2-panel style as 08d/09b/08a -- adapted to take
    plain values instead of a predictions-CSV row, since here the row is a
    merged pair with suffixed column names.
    """
    det_start = UTCDateTime(window_start)

    tr_padded, t_on, t_off, err = _fetch_padded_trace(
        client_sds, inventory, net, sta, chan, det_start,
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

    snr_str  = f"SNR={snr_val:.2f}" if pd.notna(snr_val) else "SNR=n/a"
    cal_tag  = "" if calibrated else "  [UNCALIBRATED — raw counts, not m/s]"
    title_l1 = f"HGB: {cls_name} (p={top_proba:.2f}) — {str(window_start)[:19]}{cal_tag}"
    title_l2 = f"{net}.{sta} | {snr_str}"

    plot_waveform_spectrogram_example(
        times_wave     = tr_wave.times() - REVIEW_PRE_S,
        wave_data      = tr_wave.data,
        times_spec     = t_full - REVIEW_PRE_S,
        freq_axis      = freq_axis,
        spec_db        = Sxx_db,
        det_duration_s = duration_s,
        title_lines    = (title_l1, title_l2),
        out_path       = out_path,
        spec_vmin      = spec_vmin,
        spec_vmax      = spec_vmax,
    )
    return True


# -- CNN side of the pair: freq/time axes, literal copy of 08d's own precompute --
if MODE == "sample":
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
          f"-- must match consolidated_<month>.npz (check CNN_HIGH_FREQ_MODE if it doesn't)")


def plot_review_spectrogram_cnn(img, net_i, sta_i, w_start, cls_name, top_proba, out_path):
    """
    CNN side of the pair. img is sliced directly out of the consolidated
    archive already loaded into memory in Section 4 -- no per-row file access
    (unlike 08d's version of this function, which opens one review_images/
    *.npz per row -- see the module docstring's "CNN IMAGES" section for why).
    """
    img = np.asarray(img, dtype=np.float32)
    if img.shape != (len(CNN_FREQ_AXIS), len(CNN_TIME_AXIS), 3):
        return False

    plot_spectrogram_rgb_example(
        CNN_FREQ_AXIS, CNN_TIME_AXIS, img,
        title_lines=(f"CNN: {cls_name} (p={top_proba:.2f})", f"{net_i}.{sta_i}  {w_start}"),
        out_path=out_path,
    )
    return True



# =============================================================================
# SECTION 4 — MODE == "sample"
# =============================================================================

if MODE == "sample":

    # ---- Load + normalize both predictions files -------------------------------
    print(f"\n{'='*70}")
    print(f"  LOADING PREDICTIONS")
    print(f"{'='*70}")

    if not os.path.isfile(HGB_PREDICTIONS_CSV):
        print(f"[ERROR] HGB_PREDICTIONS_CSV not found: {HGB_PREDICTIONS_CSV}")
        log_file.close(); sys.exit(1)
    if not os.path.isfile(CNN_PREDICTIONS_CSV):
        print(f"[ERROR] CNN_PREDICTIONS_CSV not found: {CNN_PREDICTIONS_CSV}")
        log_file.close(); sys.exit(1)
    if not os.path.isfile(CNN_CONSOLIDATED_NPZ):
        print(f"[ERROR] CNN_CONSOLIDATED_NPZ not found: {CNN_CONSOLIDATED_NPZ}")
        log_file.close(); sys.exit(1)

    df_hgb = pd.read_csv(HGB_PREDICTIONS_CSV, low_memory=False)
    df_cnn = pd.read_csv(CNN_PREDICTIONS_CSV, low_memory=False)
    print(f"  [OK] HGB: {os.path.basename(HGB_PREDICTIONS_CSV)}  {len(df_hgb):,} row(s)")
    print(f"  [OK] CNN: {os.path.basename(CNN_PREDICTIONS_CSV)}  {len(df_cnn):,} row(s)")

    if "SNR" not in df_hgb.columns and "snr" in df_hgb.columns:
        df_hgb = df_hgb.rename(columns={"snr": "SNR"})
    if "SNR" not in df_cnn.columns and "snr" in df_cnn.columns:
        df_cnn = df_cnn.rename(columns={"snr": "SNR"})

    if STATIONS_FILTER:
        df_hgb = df_hgb[df_hgb["station"].isin(STATIONS_FILTER)].copy()
        df_cnn = df_cnn[df_cnn["station"].isin(STATIONS_FILTER)].copy()
        print(f"  After STATIONS_FILTER={STATIONS_FILTER}: HGB {len(df_hgb):,} row(s), "
              f"CNN {len(df_cnn):,} row(s)")

    # ---- CNN images: index the consolidated archive by (network, station,
    # window_start) -- see "CNN IMAGES" in the module docstring for why this
    # replaces review_images/*.npz. Loaded once, kept in memory for the whole
    # run (Section 4's generation loop slices _cnn_images[idx] directly).
    print(f"  Loading CNN images from {os.path.basename(CNN_CONSOLIDATED_NPZ)} ...")
    with np.load(CNN_CONSOLIDATED_NPZ, allow_pickle=False) as _d:
        _cnn_images = _d["images"]
        _cnn_index = {
            (str(net), str(sta), str(ws)): i
            for i, (net, sta, ws) in enumerate(zip(_d["network"], _d["station"], _d["window_start"]))
        }
    print(f"  [OK] {len(_cnn_index):,} image(s) indexed from the consolidated archive")

    # Only CNN rows with a matching image in the consolidated archive are
    # usable -- filter BEFORE pairing so we don't pair against a window we
    # can't actually show. A predictions CSV row missing here most likely
    # means it came from a different month/run than CNN_CONSOLIDATED_NPZ.
    df_cnn["_cnn_img_idx"] = df_cnn.apply(
        lambda r: _cnn_index.get((str(r["network"]), str(r["station"]), str(r["window_start"]))), axis=1
    )
    n_before = len(df_cnn)
    df_cnn = df_cnn[df_cnn["_cnn_img_idx"].notna()].copy()
    df_cnn["_cnn_img_idx"] = df_cnn["_cnn_img_idx"].astype(int)
    print(f"  CNN rows with a matching image in the consolidated archive: {len(df_cnn):,} / {n_before:,}")
    if n_before and len(df_cnn) == 0:
        print("  [WARN] Zero matches -- check that CNN_PREDICTIONS_CSV and CNN_CONSOLIDATED_NPZ "
              "really are the same month/run.")

    if df_hgb.empty or df_cnn.empty:
        print("[ERROR] Nothing left to pair after loading/filtering. Exiting.")
        log_file.close(); sys.exit(1)

    # ---- Pair HGB and CNN rows for the SAME underlying window (see MATCHING
    # in the module docstring) ----------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  PAIRING  (tolerance={MATCH_TOLERANCE_S}s around a {CNN_WINDOW_PRE_S_FOR_MATCH}s offset)")
    print(f"{'='*70}")

    df_hgb = df_hgb.copy()
    df_cnn = df_cnn.copy()
    df_hgb["_t_key"] = pd.to_datetime(df_hgb["window_start"], utc=True, errors="coerce")
    # CNN's window_start is onset - CNN_WINDOW_PRE_S_FOR_MATCH -- add it back so
    # both sides are keyed on the same thing (the estimated true onset time).
    df_cnn["_t_key"] = (
        pd.to_datetime(df_cnn["window_start"], utc=True, errors="coerce")
        + pd.Timedelta(seconds=CNN_WINDOW_PRE_S_FOR_MATCH)
    )
    df_hgb = df_hgb.dropna(subset=["_t_key"]).sort_values("_t_key")
    df_cnn = df_cnn.dropna(subset=["_t_key"]).sort_values("_t_key")

    df_paired = pd.merge_asof(
        df_hgb, df_cnn,
        on="_t_key", by=["network", "station"],
        direction="nearest",
        tolerance=pd.Timedelta(seconds=MATCH_TOLERANCE_S),
        suffixes=("_hgb", "_cnn"),
    )
    n_paired_before_drop = len(df_paired)
    df_paired = df_paired.dropna(subset=["predicted_class_cnn"]).copy()   # no CNN match within tolerance

    print(f"  HGB rows (post-filter): {len(df_hgb):,}")
    print(f"  CNN rows (post-filter): {len(df_cnn):,}")
    print(f"  Paired (within tolerance): {len(df_paired):,} / {len(df_hgb):,} HGB rows "
          f"({100 * len(df_paired) / max(len(df_hgb), 1):.1f}%)")

    if df_paired.empty:
        print("[ERROR] No pairs found within tolerance. Check CNN_WINDOW_PRE_S_FOR_MATCH, "
              "MATCH_TOLERANCE_S, and that both CSVs really cover the same month/stations. Exiting.")
        log_file.close(); sys.exit(1)

    # Self-check: the observed offset (cnn's estimated onset minus hgb's onset)
    # should cluster tightly around 0s if the pairing assumption is right.
    _hgb_onset = pd.to_datetime(df_paired["window_start_hgb"], utc=True, errors="coerce")
    _cnn_onset_est = df_paired["_t_key"]
    _offset_s = (_cnn_onset_est - _hgb_onset).dt.total_seconds()
    print(f"  Offset check (should be ~0s): median={_offset_s.median():.2f}s, "
          f"std={_offset_s.std():.2f}s, max|offset|={_offset_s.abs().max():.2f}s")
    if abs(_offset_s.median()) > MATCH_TOLERANCE_S / 2:
        print("  [WARN] Median offset is not close to 0s -- the pairing assumption in the "
              "module docstring may not hold for this data. Treat pairs with caution.")

    # ---- Confidence + agreement columns -----------------------------------------
    def _winning_proba(row, model_suffix):
        col = f"proba_{str(row[f'predicted_class_{model_suffix}']).replace(' ', '_')}_{model_suffix}"
        return float(row[col]) if col in row and pd.notna(row[col]) else np.nan

    df_paired["hgb_confidence"] = df_paired.apply(lambda r: _winning_proba(r, "hgb"), axis=1)
    df_paired["cnn_confidence"] = df_paired.apply(lambda r: _winning_proba(r, "cnn"), axis=1)
    df_paired["agree"] = df_paired["predicted_class_hgb"] == df_paired["predicted_class_cnn"]

    # ---- Objective (no manual judgement needed) disagreement cross-tab, over
    # the FULL paired pool, printed as an immediate sanity-check / defensible
    # frequency-based observation on its own -------------------------------------
    _disagree = df_paired[~df_paired["agree"]]
    print(f"\n  Agreement over the full paired pool: "
          f"{df_paired['agree'].sum():,}/{len(df_paired):,} "
          f"({100 * df_paired['agree'].mean():.1f}%) agree")
    if not _disagree.empty:
        print("  Disagreement cross-tab (rows=HGB class, cols=CNN class, full paired pool):")
        _ctab = pd.crosstab(_disagree["predicted_class_hgb"], _disagree["predicted_class_cnn"])
        print(_ctab.to_string().replace("\n", "\n  "))

    # ---- Optional pre-filters (applied identically to both models' columns) ----
    if SNR_MIN is not None and "SNR_hgb" in df_paired.columns:
        df_paired = df_paired[df_paired["SNR_hgb"] >= SNR_MIN]
    if SNR_MAX is not None and "SNR_hgb" in df_paired.columns:
        df_paired = df_paired[df_paired["SNR_hgb"] <= SNR_MAX]
    if PROBA_MIN is not None:
        df_paired = df_paired[(df_paired["hgb_confidence"] >= PROBA_MIN) &
                              (df_paired["cnn_confidence"] >= PROBA_MIN)]
    if PROBA_MAX is not None:
        df_paired = df_paired[(df_paired["hgb_confidence"] <= PROBA_MAX) &
                              (df_paired["cnn_confidence"] <= PROBA_MAX)]

    # ---- Stratified random sampling, independently per model ------------------
    print(f"\n{'='*70}")
    print(f"  SAMPLING  ({N_PER_CLASS}/class, per model, seed={RANDOM_SEED})")
    print(f"{'='*70}")

    df_paired["_pair_key"] = (
        df_paired["network"] + "|" + df_paired["station"] + "|" + df_paired["window_start_hgb"].astype(str)
    )

    def _stratified_sample(df, class_col, seed):
        picked_frames = []
        for cls in CLASS_ORDER:
            df_cls = df[df[class_col] == cls]
            n = min(N_PER_CLASS, len(df_cls))
            print(f"    {class_col:<20s} {cls:<12s} {n:3d}/{N_PER_CLASS} available={len(df_cls)}")
            if n > 0:
                picked_frames.append(df_cls.sample(n=n, random_state=seed))
        return pd.concat(picked_frames, ignore_index=True) if picked_frames else df.iloc[0:0]

    print("  By HGB predicted class:")
    sample_hgb = _stratified_sample(df_paired, "predicted_class_hgb", RANDOM_SEED)
    print("  By CNN predicted class:")
    sample_cnn = _stratified_sample(df_paired, "predicted_class_cnn", RANDOM_SEED + 1)

    hgb_keys = set(sample_hgb["_pair_key"])
    cnn_keys = set(sample_cnn["_pair_key"])
    union_keys = hgb_keys | cnn_keys
    df_union = df_paired[df_paired["_pair_key"].isin(union_keys)].drop_duplicates("_pair_key").copy()
    df_union["selected_for"] = df_union["_pair_key"].apply(
        lambda k: "both" if (k in hgb_keys and k in cnn_keys)
        else ("hgb_stratum" if k in hgb_keys else "cnn_stratum")
    )
    print(f"\n  Union of both samples: {len(df_union):,} unique window(s) "
          f"({len(hgb_keys)} from HGB strata, {len(cnn_keys)} from CNN strata, "
          f"{len(hgb_keys & cnn_keys)} selected by both)")

    # ---- Generate both models' figures for every window in the union ----------
    print(f"\n{'='*70}")
    print(f"  GENERATING FIGURES")
    print(f"{'='*70}")

    client_sds  = connect_sds(SDS_ROOT)
    client_fdsn = connect_fdsn(ISTERRE_URL)
    if client_sds is None:
        print("[ERROR] Cannot proceed without SDS (needed for the HGB figures). Exiting.")
        log_file.close(); sys.exit(1)

    inventory = None
    if client_fdsn is not None:
        _t_min = pd.to_datetime(df_union["window_start_hgb"]).min()
        _t_max = pd.to_datetime(df_union["window_start_hgb"]).max()
        inventory = fetch_inventory(
            client_fdsn, str(_t_min.date()), str((_t_max + pd.Timedelta(days=1)).date()),
            lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX,
        )
    if inventory is None:
        print("  [WARN] No instrument inventory -- HGB figures will be UNCALIBRATED "
              "(shape/frequency content still indicative, see 08d for why). CNN figures "
              "are unaffected (09a always requires FDSN at extraction time).")
    else:
        print("  [OK] Instrument inventory fetched -- HGB figures will be calibrated.")

    manifest_rows = []
    n_hgb_ok = n_cnn_ok = 0

    for _, row in df_union.iterrows():
        safe_time = str(row["window_start_hgb"]).replace(":", "").replace("-", "").replace(".", "")
        pair_dir  = os.path.join(figures_dir, f"{row['network']}_{row['station']}_{safe_time}")
        os.makedirs(pair_dir, exist_ok=True)

        hgb_conf = row["hgb_confidence"] if pd.notna(row["hgb_confidence"]) else 0.0
        cnn_conf = row["cnn_confidence"] if pd.notna(row["cnn_confidence"]) else 0.0
        hgb_cls  = row["predicted_class_hgb"]
        cnn_cls  = row["predicted_class_cnn"]

        hgb_png = os.path.join(pair_dir, f"hgb_p{hgb_conf:.2f}_{str(hgb_cls).replace(' ', '_')}.png")
        cnn_png = os.path.join(pair_dir, f"cnn_p{cnn_conf:.2f}_{str(cnn_cls).replace(' ', '_')}.png")

        hgb_ok = False
        try:
            hgb_ok = plot_review_waveform(
                client_sds, inventory, row["network"], row["station"], row["channel_hgb"],
                row["window_start_hgb"], row.get("SNR_hgb", np.nan),
                hgb_cls, hgb_conf, hgb_png, calibrated=(inventory is not None),
                duration_s=row.get("duration_s", 0.0) if pd.notna(row.get("duration_s", np.nan)) else 0.0,
            )
        except Exception:
            pass
        n_hgb_ok += int(hgb_ok)

        cnn_ok = False
        try:
            img = _cnn_images[int(row["_cnn_img_idx"])]
            cnn_ok = plot_review_spectrogram_cnn(
                img, row["network"], row["station"], row["window_start_cnn"],
                cnn_cls, cnn_conf, cnn_png,
            )
        except Exception:
            pass
        n_cnn_ok += int(cnn_ok)

        manifest_rows.append({
            "network": row["network"], "station": row["station"],
            "window_start_hgb": row["window_start_hgb"], "window_start_cnn": row["window_start_cnn"],
            "hgb_predicted_class": hgb_cls, "hgb_confidence": round(hgb_conf, 4),
            "cnn_predicted_class": cnn_cls, "cnn_confidence": round(cnn_conf, 4),
            "agree": bool(row["agree"]), "selected_for": row["selected_for"],
            "hgb_figure": hgb_png if hgb_ok else "",
            "cnn_figure": cnn_png if cnn_ok else "",
            "hgb_visual_match": "",   # <- fill in by eye: yes / no / unsure
            "cnn_visual_match": "",   # <- fill in by eye: yes / no / unsure
            "notes": "",
        })

    print(f"  HGB figures: {n_hgb_ok}/{len(df_union)} plotted")
    print(f"  CNN figures: {n_cnn_ok}/{len(df_union)} plotted")

    month_tag = str(df_union["window_start_hgb"].iloc[0])[:7] if len(df_union) else "unknown"
    manifest_path = os.path.join(RUN_DIR, f"paired_sample_manifest_{month_tag}.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"\n  [SAVED] Manifest -> {manifest_path}")
    print(f"  Next step: open the figures in {figures_dir}/, fill in hgb_visual_match / "
          f"cnn_visual_match ('yes'/'no'/'unsure') + notes in the manifest CSV, then set "
          f"MODE='summarize' and ANNOTATED_MANIFEST_CSV='{manifest_path}' and re-run.")



# =============================================================================
# SECTION 5 — MODE == "summarize"
# =============================================================================

if MODE == "summarize":

    print(f"\n{'='*70}")
    print(f"  SUMMARIZING  {ANNOTATED_MANIFEST_CSV}")
    print(f"{'='*70}")

    df_m = pd.read_csv(ANNOTATED_MANIFEST_CSV, low_memory=False)
    _required = {"hgb_predicted_class", "cnn_predicted_class", "hgb_visual_match", "cnn_visual_match",
                 "agree", "selected_for"}
    _missing  = _required - set(df_m.columns)
    if _missing:
        print(f"[ERROR] Manifest missing required column(s): {_missing}. Exiting.")
        log_file.close(); sys.exit(1)

    def _norm(v):
        return str(v).strip().lower() if pd.notna(v) else ""

    df_m["_hgb_match_norm"] = df_m["hgb_visual_match"].apply(_norm)
    df_m["_cnn_match_norm"] = df_m["cnn_visual_match"].apply(_norm)

    n_unannotated = int(((df_m["_hgb_match_norm"] == "") | (df_m["_cnn_match_norm"] == "")).sum())
    if n_unannotated:
        print(f"  [WARN] {n_unannotated} row(s) still have a blank hgb_visual_match or "
              f"cnn_visual_match -- those are excluded from the counts below.")

    print(f"\n  {'='*70}")
    print(f"  REPORT-READY SENTENCES")
    print(f"  {'='*70}")

    # IMPORTANT: restrict each model's count to rows actually drawn into THAT
    # model's own random stratum ("hgb_stratum"/"both" for HGB, "cnn_stratum"/
    # "both" for CNN). A window can appear in the manifest only because it was
    # sampled for the OTHER model's stratum (e.g. selected for CNN's "noise"
    # bucket but happens to also be predicted_class_hgb=="earthquake") -- that
    # row is NOT part of an unbiased random draw from HGB's "earthquake"
    # predictions, so counting it toward HGB's N here would quietly bias the
    # reported N/M numbers upward without it being a real random sample.
    for model, cls_col, match_col, stratum_tags in [
        ("HGB", "hgb_predicted_class", "_hgb_match_norm", ("hgb_stratum", "both")),
        ("CNN", "cnn_predicted_class", "_cnn_match_norm", ("cnn_stratum", "both")),
    ]:
        print(f"\n  -- {model} --")
        df_model = df_m[df_m["selected_for"].isin(stratum_tags)]
        for cls in CLASS_ORDER:
            sub = df_model[(df_model[cls_col] == cls) & (df_model[match_col] != "")]
            if sub.empty:
                continue
            n = len(sub)
            m = int((sub[match_col] == "no").sum())
            u = int((sub[match_col] == "unsure").sum())
            note = f"  ({u} unsure)" if u else ""
            print(f"    Sur un echantillon aleatoire de {n} fenetres classees '{cls}' par {model}, "
                  f"{m} ne correspondent pas visuellement au profil attendu (2.1.3).{note}")

    # ---- Disagreement / error-type breakdown ------------------------------------
    print(f"\n  {'='*70}")
    print(f"  DISAGREEMENT / ERROR-TYPE BREAKDOWN  (rows where HGB and CNN predicted "
          f"DIFFERENT classes for the same window)")
    print(f"  {'='*70}")

    _dis = df_m[(df_m["agree"] == False) &
               (df_m["_hgb_match_norm"] != "") & (df_m["_cnn_match_norm"] != "")]
    if _dis.empty:
        print("  No fully-annotated disagreement rows to analyze.")
    else:
        print(f"  {len(_dis)} fully-annotated disagreement window(s).")
        both_wrong  = ((_dis["_hgb_match_norm"] == "no") & (_dis["_cnn_match_norm"] == "no")).sum()
        both_right  = ((_dis["_hgb_match_norm"] == "yes") & (_dis["_cnn_match_norm"] == "yes")).sum()
        only_hgb_ok = ((_dis["_hgb_match_norm"] == "yes") & (_dis["_cnn_match_norm"] == "no")).sum()
        only_cnn_ok = ((_dis["_hgb_match_norm"] == "no") & (_dis["_cnn_match_norm"] == "yes")).sum()
        print(f"    Both visually match their own prediction : {both_right}")
        print(f"    Neither visually matches (same failure)  : {both_wrong}")
        print(f"    Only HGB visually matches (CNN off)      : {only_hgb_ok}")
        print(f"    Only CNN visually matches (HGB off)      : {only_cnn_ok}")
        if only_hgb_ok or only_cnn_ok:
            print(f"\n    -> {only_hgb_ok + only_cnn_ok}/{len(_dis)} disagreement window(s) show a "
                  f"DIFFERENT error type between the two models (one visually plausible, the "
                  f"other not) rather than both models simply failing on the same ambiguous "
                  f"signal -- worth citing even at this sample size.")

        print("\n  Disagreement pairs, restricted to windows where at least one model "
              "visually mismatched (rows=HGB class, cols=CNN class):")
        _mismatch_rows = _dis[(_dis["_hgb_match_norm"] == "no") | (_dis["_cnn_match_norm"] == "no")]
        if not _mismatch_rows.empty:
            _ctab = pd.crosstab(_mismatch_rows["hgb_predicted_class"], _mismatch_rows["cnn_predicted_class"])
            print("  " + _ctab.to_string().replace("\n", "\n  "))

    summary_path = os.path.join(RUN_DIR, "summary.csv")
    df_m.drop(columns=["_hgb_match_norm", "_cnn_match_norm"]).to_csv(summary_path, index=False)
    print(f"\n  [SAVED] {summary_path}")



# =============================================================================
# SECTION 6 — END
# =============================================================================

print(f"\n{'='*70}")
print(f"  Run finished : {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  All outputs  : {RUN_DIR}")
print(f"  Log file     : {log_path}")
print(f"{'='*70}")

log_file.close()
