"""
09b_continuous_tabular_classification.py
=========================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : August 2026

Goal
----
End-to-end validation of the CHAPTER 3 branch (scalar-feature HGB classifier,
06a/06b/06c) on a RAW CONTINUOUS STREAM, never chained before on data the
pipeline hasn't already seen as pre-curated catalog windows.

This is the tabular-feature counterpart of 09a_continuous_spectrogram_
classification.py, which did the same thing for the CNN branch (chapter 4).
Same idea, same MONTHS_TO_SCAN, same underlying Groult et al. (2026)
continuous-scan detector -- so the two runs are directly comparable for the
chapter 5 synthesis ("does AI outperform conventional detection?").

Why ONE script instead of 09a's two phases
-------------------------------------------
09a is split into RUN_EXTRACTION / RUN_CLASSIFICATION because TensorFlow
cannot be cleanly installed in the cluster's Python (see 09a's docstring).
Nothing here needs TensorFlow -- scikit-learn/imbalanced-learn install fine
in the same environment used for SDS/FDSN access, so extraction (needs SDS)
and classification (needs the training CSVs + sklearn) can run in the same
process. The RUN_EXTRACTION / RUN_CLASSIFICATION toggle is kept anyway,
purely for workflow convenience -- extraction is the slow, cluster-bound
step, while classification is cheap to re-run (e.g. testing a different
TOP_N_FEATURES or SNR gate) without re-scanning a month of continuous data.

There is no saved trained model to load (06c never persists one -- it
retrains from scratch every run and only saves figures/metrics). So Section
5B here retrains the exact "final classifier" described in the report
(chapter 3.4): HGB, Top-60 features, trained on the quality-gated catalog
PLUS the DeepDenoiser-rescued ice-quake events (06c's "Run B"). Retraining
is cheap (seconds), and an internal held-out-test evaluation is printed and
plotted first, as a sanity check that this run reproduces the report's
numbers (accuracy 95.5%, macro F1 0.847) before applying the model to new
unlabelled data.

Windowing note
--------------
Unlike the CNN branch's fixed 100 s window, the scalar-feature classifier's
single strongest feature is `duration` (Table 2, rank 1, ΔF1=+0.232) --
using a fixed-length window here would destroy the very signal the model
relies on most. So detections keep their natural variable length: features
are extracted from [t_on - PAD_SEC, t_off + PAD_SEC], exactly as 04a did
when building the training catalog.

TWO PHASES (kept for workflow convenience, see above) -----------------------
  Phase 1 -- EXTRACTION (needs SDS/FDSN):
    per station-day, run the same continuous STA/LTA scan as 09a Phase 1,
    then for each detected event extract the 99 Maggi/Hibert Z-features +
    4 polarization features (103 total, matching the training catalog's 3C
    mode) + 7 SNR measures -> one CSV per station-day, consolidated per
    month.

  Phase 2 -- CLASSIFICATION (needs the training catalog CSVs + sklearn):
    (a) retrain the final HGB classifier (06c Run B recipe) and sanity-check
        it against the report's reference numbers,
    (b) classify every extracted continuous-data window,
    (c) write predictions_<month>.csv + a probability summary figure +
        (optional, needs SDS again) a small waveform review gallery.

There is no scoring against ground truth here -- exactly like 09a, the
output is meant for visual/plausibility review: does the class distribution
look sane (mostly noise/rare events), do the review waveforms for non-noise
predictions actually look like real events, etc.

Output layout
-------------
  EXTRACTION_DIR/
      feats_<net>_<sta>_<YYYYMMDD>.csv   <- one file per station-day WITH
                                             >=1 detection, written by Phase 1,
                                             read by Phase 2
      consolidated_<month>.csv           <- Phase 1, one file per scanned month
  outputs_09b/run_YYYYMMDD_HHMMSS/    (this invocation's own log + phase-2 output)
      fig_reference_confusion_<stamp>.png  <- Phase 2a: sanity check vs report
      model_cache_<stamp>.joblib           <- Phase 2a: trained model+imputer+features
      predictions_<month_tag>.csv          <- Phase 2b: one row per classified detection
      probability_summary/*.png            <- Phase 2c
      review_waveforms/*.png               <- Phase 2d (optional, needs SDS)
      run.log
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# -- Run mode -- see the module docstring for why this split is kept ---------
RUN_EXTRACTION     = False   # Phase 1 -- needs SDS/FDSN, no sklearn dependency
RUN_CLASSIFICATION = True    # Phase 2 -- needs the training CSVs + sklearn/imblearn

# -- Interchange directory between the two phases -----------------------------
# Phase 1 writes per-station-day + consolidated feature CSVs here
EXTRACTION_DIR = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\09b_continuous_data_test\run_20260818_090000"

# -- Paths (Phase 1 -- SDS/FDSN) -----------------------------------------------
SDS_ROOT    = "/data/sig/SDS"
ISTERRE_URL = "http://ist-sc3-geobs.osug.fr:8080"

# -- Output for THIS invocation's own log + Phase 2 outputs -------------------
OUTPUT_DIR = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\09b_continuous_data_test\run_20260818_090000"

# -- Spatial bounding box (Mont Blanc massif, same as everywhere else) --------
LAT_MIN, LAT_MAX = 45.5, 46.0
LON_MIN, LON_MAX = 6.5,  7.2

# -- Channel selection (Phase 1) -----------------------------------------------
Z_CHANNEL = "??Z"
HORIZONTAL_SUFFIXES = [("N", "E"), ("2", "1")]   # same fallback convention as 04a/09a

# -- Months to scan (Phase 1) -- SAME as 09a so the two chapters are ----------
# directly comparable on identical raw data.
MONTHS_TO_SCAN = [
    ("2025-01-01", "2025-02-01"),   # January
    ("2025-07-01", "2025-08-01"),   # July
]

# -- Detection: spectrogram-based STA/LTA (Groult et al. 2026), Phase 1 -------
# Identical to 09a's Phase 1 detector so both chapters see the same events.
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

# -- Feature extraction window (Phase 1) -- variable length, matches 04a ------
PAD_SEC = 5        # seconds added before t_on and after t_off, same as 04a
LOAD_3C = True     # fetch N/E for polarization features -> 103 features (matches training)

# -- Checkpointing / resume (Phase 1: per station-day file existence) ---------
CONSOLIDATE_PER_MONTH = True   # bundle each month's per-station-day CSVs into one file

# -- SMOKE TEST (Phase 1) -- strongly recommended before the full run ---------
SMOKE_TEST              = True
MAX_DAYS_SMOKE_TEST     = 1
MAX_STATIONS_SMOKE_TEST = 1


# ------------------------------------------------------------------------------
# Phase 2a — training data for the FINAL chapter-3 classifier
# Identical paths/hyperparameters to 06c's "Run B" (original + denoised
# rescued ice quakes) -- this IS the model chapter 3.4 reports as final:
# HGB, Top-60 features, accuracy 95.5%, macro F1 0.847.
# ------------------------------------------------------------------------------
ORIGINAL_CSV = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\04a_spectrogram_sta_lta_catalog\all-99-features-recent+3C\catalog_windows_20260708_174019.csv"
RESCUE_CATALOG_CSV = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\03d_rescue_feature_extraction\stricter_IQ100_20260722_145529\rescue_catalog_20260722_145529.csv"
NOISE_CSV = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\04d_noise_window_extraction\run_20260803_174514\noise_windows_20260803_174514.csv"
REGIONAL_CSV = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\04c_regional_EQ_extraction\run_20260805_135512\regional_windows_20260805_135512.csv"

TARGET_CLASSES = ["earthquake", "regional", "rockslide", "ice quake", "noise"]
CLASS_ORDER    = ["earthquake", "regional", "rockslide", "ice quake", "noise"]
CLASS_ABBR     = {"earthquake": "eq", "regional": "re", "rockslide": "rs", "ice quake": "iq", "noise": "no"}
CLASS_COLORS   = {"earthquake": "#1f77b4", "rockslide": "#d62728",
                  "ice quake": "#2ca02c", "noise": "#7f7f7f", "regional": "#9467bd"}

FEATURE_IMPORTANCES_CSV = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\03b_feature_selection\run_20260710_144246\feature_importances_20260710_144246.csv"
TOP_N_FEATURES          = 60   # matches chapter 3.1's "Top-60" result (best macro F1 in Table 3)

SNR_MIN             = 1.70    # 05b Tier 2 gate, same as 06c
SNR_FULL_MEDIAN_MIN = 1.99

TEST_SIZE    = 0.20
RANDOM_STATE = 42
SMOTE_K      = 5

HGB_N_EST     = 200
HGB_MAX_DEPTH = 6
HGB_LR        = 0.1

# -- Model cache (Phase 2a) -- avoids retraining on every re-run of Phase 2b/c/d --
USE_CACHED_MODEL = True
MODEL_CACHE_PATH = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\09b_continuous_data_test\hgb_final_model_cache.joblib"

# -- Phase 2d: review waveform gallery (optional, needs SDS) ------------------
SAVE_REVIEW_WAVEFORMS   = True
SAVE_IMAGES_FOR_NONNOISE = True    # save every window NOT predicted as 'noise'
SAVE_EVERY_NTH_NOISE     = 5
N_GALLERY_PER_CLASS      = 10
REVIEW_PAD_SEC           = 10      # extra context padding around the window, display only
REVIEW_FREQ_MIN, REVIEW_FREQ_MAX = 1.0, 20.0

# -- Probability summary plot (Phase 2c) ---------------------------------------
PLOT_PROBABILITY_SUMMARY = True



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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_setup import create_run_dir, setup_logging

RUN_DIR, STAMP = create_run_dir(OUTPUT_DIR)
log_file, log_path = setup_logging(
    RUN_DIR, "09b_continuous_tabular_classification.py",
    extra_info=(f"RUN_EXTRACTION={RUN_EXTRACTION}  RUN_CLASSIFICATION={RUN_CLASSIFICATION}  |  "
                f"EXTRACTION_DIR: {EXTRACTION_DIR}  |  DET_THR_ON={DET_THR_ON}  |  SMOKE_TEST={SMOKE_TEST}")
)

os.makedirs(EXTRACTION_DIR, exist_ok=True)

if not RUN_EXTRACTION and not RUN_CLASSIFICATION:
    print("[ERROR] Both RUN_EXTRACTION and RUN_CLASSIFICATION are False -- nothing to do.")
    log_file.close()
    sys.exit(1)

if SMOKE_TEST and RUN_EXTRACTION:
    print(f"\n[SMOKE TEST MODE] Restricting extraction to {MAX_DAYS_SMOKE_TEST} day(s) x "
          f"{MAX_STATIONS_SMOKE_TEST} station(s) per month. Set SMOKE_TEST=False once "
          f"you've checked the timing + storage estimate below.\n")


# --------- Phase 1 setup: SDS/FDSN connections + station selection -----------
if RUN_EXTRACTION:
    from obspy import UTCDateTime
    from run_setup import connect_sds, connect_fdsn, fetch_inventory
    from catalog_helpers import build_station_list_from_inventory
    from preprocessing import preprocess_day
    from detecteurV3_fonctions import DetecteurV3   # Groult et al. 2026 -- third-party, do not modify
    from detection import compute_snr, merge_window_events
    from features import FEATURE_NAMES_3C, N_FEATURES_3C, extract_features

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

    _FEAT_NAMES = FEATURE_NAMES_3C if LOAD_3C else FEATURE_NAMES_3C[:99]
    N_FEATURES  = N_FEATURES_3C    if LOAD_3C else 99


# --------- Phase 2 setup: sklearn/imblearn (no TensorFlow anywhere here) -----
if RUN_CLASSIFICATION:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import (
        classification_report, accuracy_score,
        confusion_matrix, ConfusionMatrixDisplay,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.impute import SimpleImputer
    from imblearn.over_sampling import SMOTE
    try:
        import joblib
    except ImportError:
        joblib = None
        print("[WARN] joblib not importable -- model caching disabled for this run.")

    from features import FEATURE_NAMES, FEATURE_NAMES_3C, rename_legacy_columns
    from run_setup import set_matplotlib_defaults
    set_matplotlib_defaults()

    summary_dir = os.path.join(RUN_DIR, "probability_summary")
    os.makedirs(summary_dir, exist_ok=True)
    if SAVE_REVIEW_WAVEFORMS:
        review_dir = os.path.join(RUN_DIR, "review_waveforms")
        os.makedirs(review_dir, exist_ok=True)



# =============================================================================
# SECTION 3 — HELPER FUNCTIONS: PHASE 1 (EXTRACTION)
# =============================================================================

if RUN_EXTRACTION:

    def _fetch_3c_array(client_sds, net, sta, chan_z, t0, t1, z_data, fs):
        """
        Literal copy of 04a_sta_lta_catalog_windowing.py's _fetch_3c_array() --
        kept as a copy (not an import) so this script has no hidden runtime
        dependency on 04a's own CONFIGURATION section changing underneath it,
        same reasoning 09a used for its own copied helpers.

        IMPORTANT convention (inherited from 04a, not changed here): Z is
        already response-removed velocity (z_data), but N/E are only demeaned
        + resampled, NOT response-removed. This mismatched-units convention is
        exactly how the TRAINING catalog's polarization features were built --
        replicating it here (rather than "fixing" it) is what keeps this
        script's features consistent with what the classifier was trained on.

        Returns
        -------
        arr : np.ndarray, shape (3, n_samples), rows [Z, N, E]  or  None
        """
        base    = chan_z[:-1]
        n       = len(z_data)
        h_pairs = [("N", "E"), ("1", "2")]

        data_n = data_e = None
        for suf_n, suf_e in h_pairs:
            for suf, which in [(suf_n, "N"), (suf_e, "E")]:
                if which == "N" and data_n is not None:
                    continue
                if which == "E" and data_e is not None:
                    continue
                try:
                    st_h = client_sds.get_waveforms(net, sta, "*", base + suf, t0, t1)
                    if not st_h or len(st_h[0].data) == 0:
                        continue
                    tr_h = st_h[0].copy()
                    tr_h.detrend("demean")
                    if abs(tr_h.stats.sampling_rate - fs) > 1:
                        tr_h.resample(fs)
                    d = tr_h.data[:n].astype(float)
                    if len(d) < n:
                        d = np.pad(d, (0, n - len(d)))
                    if which == "N":
                        data_n = d
                    else:
                        data_e = d
                except Exception:
                    continue
            if data_n is not None and data_e is not None:
                break

        if data_n is None or data_e is None:
            return None
        return np.stack([z_data.astype(float), data_n, data_e])   # (3, n)


    def extract_window_row(client_sds, inventory, net, sta, loc, chan, day_trace_bounds,
                           t_on, t_off, trigger_cft, day_str):
        """
        Build one output row for a single detection.

        IMPORTANT: unlike the detection step (which runs on preprocess_day's
        pre_filt-tapered, whole-day trace -- necessary for continuous
        scanning), features here are extracted from a FRESH short SDS fetch
        of the padded window, response-removed the SAME way 04a built the
        training catalog (plain remove_response, no pre_filt taper -- see
        preprocessing.remove_response_or_fallback). Reusing the day-scale
        pre_filt trace instead would systematically shift amplitude/energy
        features relative to what the classifier was trained on, especially
        near the pre_filt taper's band edges. 09a made the exact same choice
        for the CNN branch (see its _process_event_trace docstring) for the
        same reason -- matching training-time preprocessing exactly matters
        more than saving one extra SDS round-trip.

        Returns
        -------
        row : dict, or None if the window is unusable
        """
        from preprocessing import build_station_times_df, remove_response_or_fallback

        day_start, day_end = day_trace_bounds
        t_cut_on  = max(t_on  - PAD_SEC, day_start)
        t_cut_off = min(t_off + PAD_SEC, day_end)

        try:
            st_raw = client_sds.get_waveforms(net, sta, loc if loc else "*", chan, t_cut_on, t_cut_off)
        except Exception:
            return None
        if len(st_raw) == 0:
            return None
        st_raw.merge(method=1, fill_value="interpolate")

        sdf    = build_station_times_df(st_raw, t_cut_on, t_cut_off)
        st_vel = remove_response_or_fallback(st_raw, inventory, sdf)
        if len(st_vel) == 0:
            return None
        tr_cut = st_vel[0]
        if tr_cut.stats.npts < 10:
            return None
        fs = tr_cut.stats.sampling_rate

        tr_filt_local = tr_cut.copy()
        nyq = fs / 2
        tr_filt_local.filter('bandpass', freqmin=DET_FREQ_MIN,
                             freqmax=min(DET_FREQ_MAX, 0.9 * nyq), corners=4, zerophase=True)

        data_3c = None
        if LOAD_3C:
            data_3c = _fetch_3c_array(client_sds, net, sta, chan, t_cut_on, t_cut_off,
                                      tr_cut.data, fs)

        feats = extract_features(tr_cut.data, fs, data_3c=data_3c)
        snr   = compute_snr(tr_filt_local, t_on, t_off)

        row = {
            "network": net, "station": sta, "location": loc, "channel": chan,
            "day": day_str,
            "window_start": str(t_on), "window_end": str(t_off),
            "duration_s": round(t_off - t_on, 2),
            "trigger_on_cft": round(trigger_cft, 4),
            **snr,
        }
        for fname, fval in zip(_FEAT_NAMES, feats):
            row[fname] = fval
        return row


    def consolidate_month_csvs(extraction_dir, month_tag):
        """Concatenate every station-day feats_*.csv whose 'day' falls in
        month_tag into one consolidated_<month_tag>.csv, same purpose as
        09a's consolidate_month_packed_files() for the CNN branch."""
        candidates = sorted(glob.glob(os.path.join(extraction_dir, "feats_*.csv")))
        frames = []
        for fpath in candidates:
            try:
                df = pd.read_csv(fpath, low_memory=False)
                if df.empty:
                    continue
                if str(df["day"].iloc[0])[:7] != month_tag:
                    continue
                frames.append(df)
            except Exception as e:
                print(f"  [WARN] consolidate: could not read {os.path.basename(fpath)}: {e}")

        if not frames:
            print(f"  [WARN] consolidate: no feature CSVs found for month {month_tag} -- skipping.")
            return None

        out_df   = pd.concat(frames, ignore_index=True)
        out_path = os.path.join(extraction_dir, f"consolidated_{month_tag}.csv")
        out_df.to_csv(out_path, index=False)
        print(f"  [CONSOLIDATED] {month_tag}: {len(out_df)} window(s) from {len(frames)} "
              f"station-day file(s) -> {os.path.basename(out_path)}")
        return out_path



# =============================================================================
# SECTION 4 — PHASE 1: EXTRACTION (continuous STA/LTA scan, SDS/FDSN needed)
# =============================================================================

if RUN_EXTRACTION:

    print(f"\n{'='*70}")
    print(f"  PHASE 1 — EXTRACTION (STA/LTA detection, Groult et al. 2026)")
    print(f"  {'[SMOKE TEST]' if SMOKE_TEST else '[FULL RUN]'}  |  {N_FEATURES} feature(s)/window")
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

        n_station_days = 0
        n_events_total = 0

        for day_utc in days:
            day_str   = day_utc.strftime("%Y-%m-%d")
            day_start = day_utc
            day_end   = day_utc + 86400

            for net, sta, loc, chan in station_list:
                t_station_day = time.time()

                out_fname = f"feats_{net}_{sta}_{day_utc.strftime('%Y%m%d')}.csv"
                out_path  = os.path.join(EXTRACTION_DIR, out_fname)
                n_station_days += 1

                if os.path.isfile(out_path):
                    print(f"  [SKIP] {day_str} {net}.{sta} — already extracted ({out_fname})")
                    continue

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

                day_rows = []

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
                    df_hz = fs / nfft
                    if DET_FREQ_MAX / df_hz < 2:
                        nfft = 2 ** int(np.ceil(np.log2(DET_FREQ_MAX * 4)))

                    # Note: no day-scale filtered trace is built here (unlike 02b/09a) --
                    # SNR is computed per-event from a freshly-fetched, plain-response-
                    # removed window inside extract_window_row (see its docstring), not
                    # from this day-scale pre_filt-tapered trace. Detection itself still
                    # runs on tr_vel below, which is fine -- only the amplitude-sensitive
                    # feature/SNR extraction needs to match 04a's exact preprocessing.

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

                    # ---- Feature + SNR extraction for each detected event --------
                    # (fresh short SDS fetch per event, see extract_window_row docstring
                    # for why this doesn't just slice tr_vel/tr_filt above)
                    for ev_key, (t_on, t_off) in total_events.items():
                        row = extract_window_row(
                            client_sds, inventory, net, sta, loc, chan,
                            (tr_vel.stats.starttime, tr_vel.stats.endtime),
                            t_on, t_off, total_thresholds[ev_key][0], day_str,
                        )
                        if row is not None:
                            day_rows.append(row)

                dt = time.time() - t_station_day

                if not day_rows:
                    print(f"  {day_str} {net}.{sta:<6s}  0 event(s) detected  [{dt:6.1f}s]")
                    continue

                pd.DataFrame(day_rows).to_csv(out_path, index=False)
                n_events_total += len(day_rows)
                print(f"  {day_str} {net}.{sta:<6s}  {len(day_rows):4d} event(s) detected+extracted "
                      f"in {dt:6.1f}s")

        print(f"\n  [DONE] Month {month_tag}: {n_events_total} event(s) across "
              f"{n_station_days} station-day(s) -> feature CSVs in {EXTRACTION_DIR}/")

        if CONSOLIDATE_PER_MONTH:
            consolidate_month_csvs(EXTRACTION_DIR, month_tag)

    print(f"\n[PHASE 1 COMPLETE] Extracted features -> {EXTRACTION_DIR}/")
    print(f"  Set RUN_EXTRACTION=False, RUN_CLASSIFICATION=True, EXTRACTION_DIR")
    print(f"  pointed at this folder (locally or on the cluster -- both work,")
    print(f"  see the module docstring) to classify.")



# =============================================================================
# SECTION 5 — PHASE 2a: TRAIN THE FINAL CHAPTER-3 CLASSIFIER (06c "Run B")
# =============================================================================

def train_final_classifier():
    """
    Reproduces 06c's Run B exactly: original quality-gated catalog (+ regional,
    + noise) combined with the DeepDenoiser-rescued ice-quake events, Top-60
    features by permutation importance, event-stratified 80/20 split, SMOTE on
    the training fold, HistGradientBoostingClassifier. This IS the model
    chapter 3.4 of the report calls "the final classifier".

    Returns
    -------
    model, imputer, features : fitted sklearn objects + the ordered feature list
    """
    print(f"\n{'='*70}")
    print(f"  PHASE 2a — Training the final chapter-3 classifier (06c Run B recipe)")
    print(f"{'='*70}")

    orig = pd.read_csv(ORIGINAL_CSV, low_memory=False)
    orig = rename_legacy_columns(orig)
    orig = orig[orig["event_type"].isin(TARGET_CLASSES)].copy()

    if REGIONAL_CSV is not None and os.path.exists(str(REGIONAL_CSV)):
        regional = pd.read_csv(REGIONAL_CSV, low_memory=False)
        regional = rename_legacy_columns(regional)
        regional = regional[regional["event_type"].isin(TARGET_CLASSES)].copy()
        orig = pd.concat([orig, regional], ignore_index=True)
        print(f"  Regional catalog added (pre-gate): {len(regional):,} rows")

    mask = (orig["SNR"] >= SNR_MIN) & (orig["SNR_full_median"] >= SNR_FULL_MEDIAN_MIN)
    orig = orig[mask].copy()
    z_feat_cols = [f for f in FEATURE_NAMES if f in orig.columns]
    orig = orig.dropna(subset=z_feat_cols).copy()
    print(f"  Original(+regional) after quality gate: {len(orig):,} rows")

    if NOISE_CSV is not None and os.path.exists(str(NOISE_CSV)):
        noise = pd.read_csv(NOISE_CSV, low_memory=False)
        noise = rename_legacy_columns(noise)
        z_feat_cols_noise = [f for f in FEATURE_NAMES if f in noise.columns]
        noise = noise.dropna(subset=z_feat_cols_noise).copy()
        orig = pd.concat([orig, noise], ignore_index=True)
        print(f"  Noise catalog added: {len(noise):,} rows")

    has_rescue = RESCUE_CATALOG_CSV is not None and os.path.exists(str(RESCUE_CATALOG_CSV))
    if has_rescue:
        rescue = pd.read_csv(RESCUE_CATALOG_CSV, low_memory=False)
        rescue = rename_legacy_columns(rescue)
        rescue = rescue[rescue["event_type"].isin(TARGET_CLASSES)].copy()
        z_feat_cols_r = [f for f in FEATURE_NAMES if f in rescue.columns]
        rescue = rescue.dropna(subset=z_feat_cols_r).copy()
        print(f"  DeepDenoiser-rescued ice quakes added: {len(rescue):,} rows")
    else:
        rescue = pd.DataFrame()
        print(f"  [WARN] RESCUE_CATALOG_CSV not found — training WITHOUT the rescued "
              f"events (this will NOT match the report's 0.847 macro F1 reference).")

    combined = pd.concat([orig, rescue], ignore_index=True) if has_rescue else orig.copy()
    print(f"\n  Combined training dataset: {len(combined):,} rows")
    for cls in CLASS_ORDER:
        n = (combined["event_type"] == cls).sum()
        print(f"    {cls:<12} {n:>6,}  ({100*n/len(combined):.1f} %)")

    if FEATURE_IMPORTANCES_CSV is not None and os.path.exists(FEATURE_IMPORTANCES_CSV):
        imp_df   = pd.read_csv(FEATURE_IMPORTANCES_CSV)
        features = imp_df["feature"].head(TOP_N_FEATURES).tolist()
        print(f"\n  Loaded Top-{TOP_N_FEATURES} features from: {FEATURE_IMPORTANCES_CSV}")
    else:
        raise FileNotFoundError(
            f"FEATURE_IMPORTANCES_CSV not found: {FEATURE_IMPORTANCES_CSV}. "
            f"This is required to reproduce the report's Top-60 feature set."
        )
    missing = [f for f in features if f not in combined.columns]
    if missing:
        raise ValueError(f"Features missing from combined catalog: {missing}")

    events = combined[["event_time", "event_type"]].drop_duplicates("event_time")
    train_ev, test_ev = train_test_split(
        events["event_time"], test_size=TEST_SIZE,
        stratify=events["event_type"], random_state=RANDOM_STATE,
    )
    train_mask = combined["event_time"].isin(train_ev)
    test_mask  = combined["event_time"].isin(test_ev)

    X_tr_raw = combined.loc[train_mask, features].values
    y_tr_raw = combined.loc[train_mask, "event_type"].values
    X_te     = combined.loc[test_mask,  features].values
    y_te     = combined.loc[test_mask,  "event_type"].values
    print(f"\n  Train: {train_mask.sum():,} rows  |  Test: {test_mask.sum():,} rows")

    imputer  = SimpleImputer(strategy="median")
    X_tr_raw = imputer.fit_transform(X_tr_raw)
    X_te     = imputer.transform(X_te)

    sm = SMOTE(k_neighbors=SMOTE_K, random_state=RANDOM_STATE)
    X_tr, y_tr = sm.fit_resample(X_tr_raw, y_tr_raw)
    print(f"  After SMOTE: {len(X_tr):,} rows")

    model = HistGradientBoostingClassifier(
        max_iter=HGB_N_EST, max_depth=HGB_MAX_DEPTH,
        learning_rate=HGB_LR, early_stopping=True,
        n_iter_no_change=15, random_state=RANDOM_STATE,
    )
    t0 = time.time()
    model.fit(X_tr, y_tr)
    elapsed = time.time() - t0

    y_pred   = model.predict(X_te)
    report   = classification_report(y_te, y_pred, labels=CLASS_ORDER,
                                     target_names=CLASS_ORDER, output_dict=True, zero_division=0)
    acc      = accuracy_score(y_te, y_pred)
    macro_f1 = report["macro avg"]["f1-score"]

    print(f"\n  [SANITY CHECK vs report chapter 3.4] Accuracy={acc:.3f}  MacroF1={macro_f1:.3f}  "
          f"Time={elapsed:.1f}s")
    print(f"  Report reference: accuracy=0.955, macro F1=0.847, ice-quake F1=0.490")
    for cls in CLASS_ORDER:
        print(f"    {CLASS_ABBR[cls].upper()}-F1={report[cls]['f1-score']:.3f}  "
              f"(precision={report[cls]['precision']:.3f}  recall={report[cls]['recall']:.3f})")

    cm = confusion_matrix(y_te, y_pred, labels=CLASS_ORDER, normalize="true")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_ORDER)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format=".2f")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_title(f"HGB — Original + rescued (reference)\nMacroF1={macro_f1:.3f}  Acc={acc:.3f}",
                fontsize=10)
    plt.tight_layout()
    ref_path = os.path.join(RUN_DIR, f"fig_reference_confusion_{STAMP}.png")
    plt.savefig(ref_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {ref_path}")

    if joblib is not None and USE_CACHED_MODEL:
        try:
            os.makedirs(os.path.dirname(MODEL_CACHE_PATH), exist_ok=True)
            joblib.dump({"model": model, "imputer": imputer, "features": features,
                        "acc": acc, "macro_f1": macro_f1}, MODEL_CACHE_PATH)
            print(f"  [CACHED] Trained model -> {MODEL_CACHE_PATH}")
        except Exception as e:
            print(f"  [WARN] Could not write model cache: {e}")

    return model, imputer, features


if RUN_CLASSIFICATION:
    _cached = None
    if USE_CACHED_MODEL and joblib is not None and os.path.isfile(MODEL_CACHE_PATH):
        try:
            _cached = joblib.load(MODEL_CACHE_PATH)
            print(f"\n[LOAD] Using cached model from {MODEL_CACHE_PATH} "
                  f"(acc={_cached['acc']:.3f}, macro F1={_cached['macro_f1']:.3f}). "
                  f"Delete this file or set USE_CACHED_MODEL=False to retrain.")
        except Exception as e:
            print(f"[WARN] Could not load model cache ({e}) — retraining.")
            _cached = None

    if _cached is not None:
        final_model, final_imputer, final_features = (
            _cached["model"], _cached["imputer"], _cached["features"]
        )
    else:
        final_model, final_imputer, final_features = train_final_classifier()



# =============================================================================
# SECTION 6 — PHASE 2b/2c/2d: CLASSIFY CONTINUOUS DATA + REVIEW OUTPUTS
# =============================================================================

def plot_probability_summary(df_month, class_names, month_tag, out_dir):
    """Same idea as 09a's function of the same name -- confidence boxplot +
    mean-probability-vector heatmap per predicted class, adapted for the
    5-class tabular output (no image needed, works directly off the CSV)."""
    proba_cols = [f"proba_{c.replace(' ', '_')}" for c in class_names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

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


def plot_review_waveform(client_sds, inventory, row, cls_name, top_proba, out_path):
    """
    Re-fetch and plot the padded waveform for one classified detection --
    the tabular-branch equivalent of 09a's spectrogram review image. Needs
    SDS access; caller should catch exceptions and skip gracefully.
    """
    from obspy import UTCDateTime
    from preprocessing import build_station_times_df, remove_response_or_fallback

    t_on  = UTCDateTime(row["window_start"]) - REVIEW_PAD_SEC
    t_off = UTCDateTime(row["window_end"])   + REVIEW_PAD_SEC

    st_raw = client_sds.get_waveforms(row["network"], row["station"], "*", row["channel"], t_on, t_off)
    if len(st_raw) == 0:
        return False
    st_raw.merge(fill_value=0)

    sdf    = build_station_times_df(st_raw, t_on, t_off)
    st_vel = remove_response_or_fallback(st_raw, inventory, sdf)
    if len(st_vel) == 0:
        return False

    tr = st_vel[0].copy()
    nyq = tr.stats.sampling_rate / 2
    tr.filter("bandpass", freqmin=REVIEW_FREQ_MIN, freqmax=min(REVIEW_FREQ_MAX, 0.9 * nyq),
              corners=4, zerophase=True)

    t_axis = tr.times() - REVIEW_PAD_SEC   # 0 = window_start
    dur    = UTCDateTime(row["window_end"]) - UTCDateTime(row["window_start"])

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(t_axis, tr.data * 1e6, lw=0.6, color=CLASS_COLORS.get(cls_name, "black"))
    ax.axvspan(0, dur, color="grey", alpha=0.15)
    ax.set_ylabel("Velocity (µm/s)")
    ax.set_xlabel("Time (s), 0 = detection onset")
    ax.set_title(f"{row['network']}.{row['station']}  {row['window_start'][:19]}Z\n"
                f"predicted = {cls_name}  (p={top_proba:.2f})  duration={dur:.1f}s",
                fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


if RUN_CLASSIFICATION:

    consolidated_files = sorted(glob.glob(os.path.join(EXTRACTION_DIR, "consolidated_*.csv")))
    if consolidated_files:
        feature_files = consolidated_files
        print(f"\n{'='*70}")
        print(f"  PHASE 2b — CLASSIFICATION")
        print(f"  {len(feature_files)} consolidated monthly file(s) found in {EXTRACTION_DIR}")
        print(f"{'='*70}")
    else:
        feature_files = sorted(glob.glob(os.path.join(EXTRACTION_DIR, "feats_*.csv")))
        print(f"\n{'='*70}")
        print(f"  PHASE 2b — CLASSIFICATION")
        print(f"  {len(feature_files)} station-day file(s) found in {EXTRACTION_DIR} "
              f"(no consolidated_*.csv found)")
        print(f"{'='*70}")

    if not feature_files:
        print(f"\n[WARN] No extracted feature CSVs found in {EXTRACTION_DIR}. "
              f"Did Phase 1 run and point EXTRACTION_DIR here?")

    meta_cols = ["network", "station", "location", "channel", "day",
                "window_start", "window_end", "duration_s", "trigger_on_cft",
                "SNR", "SNR_picking_5_5", "SNR_picking_3_3", "SNR_picking_1_3",
                "SNR_full_mean", "SNR_full_median", "SNR_s2n_median"]

    rows_by_month = {}
    _noise_review_counter = [0]

    for fpath in feature_files:
        df_feat = pd.read_csv(fpath, low_memory=False)
        if df_feat.empty:
            continue

        missing_feats = [f for f in final_features if f not in df_feat.columns]
        if missing_feats:
            print(f"  [SKIP] {os.path.basename(fpath)} — missing feature column(s): "
                  f"{missing_feats[:5]}{'...' if len(missing_feats) > 5 else ''}")
            continue

        X = df_feat[final_features].values
        X = final_imputer.transform(X)
        proba = final_model.predict_proba(X)
        labels_idx = np.argmax(proba, axis=1)
        class_order_model = list(final_model.classes_)

        month_tag = str(df_feat["day"].iloc[0])[:7]
        rows_by_month.setdefault(month_tag, [])

        for i in range(len(df_feat)):
            cls_name = class_order_model[labels_idx[i]]
            row = {c: df_feat.iloc[i][c] for c in meta_cols if c in df_feat.columns}
            row["predicted_class"] = cls_name
            for j, cname in enumerate(class_order_model):
                row[f"proba_{cname.replace(' ', '_')}"] = float(proba[i, j])
            rows_by_month[month_tag].append(row)

        print(f"  [OK] {os.path.basename(fpath)}  {len(df_feat):5d} window(s) classified")

    for month_tag, rows in rows_by_month.items():
        if not rows:
            continue
        df_month = pd.DataFrame(rows)
        out_csv = os.path.join(RUN_DIR, f"predictions_{month_tag}.csv")
        df_month.to_csv(out_csv, index=False)
        print(f"\n  [SAVED] {len(df_month):,} windows -> {out_csv}")
        print(f"  Class distribution ({month_tag}):")
        vc = df_month["predicted_class"].value_counts()
        for c in CLASS_ORDER:
            n_c = int(vc.get(c, 0))
            pct = 100 * n_c / max(len(df_month), 1)
            print(f"    {c:<12s} {n_c:8,d}  ({pct:5.1f}%)")

        if PLOT_PROBABILITY_SUMMARY:
            plot_probability_summary(df_month, CLASS_ORDER, month_tag, summary_dir)

    # ---- Phase 2d: review waveform gallery (optional, needs SDS) ------------
    if SAVE_REVIEW_WAVEFORMS and rows_by_month:
        print(f"\n  Building review waveform gallery...")
        from run_setup import connect_sds, connect_fdsn, fetch_inventory

        _client_sds  = connect_sds(SDS_ROOT)
        _client_fdsn = connect_fdsn(ISTERRE_URL)

        if _client_sds is None or _client_fdsn is None:
            print("  [WARN] SDS/FDSN unavailable — skipping review waveform gallery "
                  "(this step only works on the cluster / with VPN access).")
        else:
            all_rows_df = pd.concat([pd.DataFrame(r) for r in rows_by_month.values()], ignore_index=True)
            _t_min = pd.to_datetime(all_rows_df["window_start"]).min()
            _t_max = pd.to_datetime(all_rows_df["window_end"]).max()
            _inventory = fetch_inventory(
                _client_fdsn, str(_t_min.date()), str((_t_max + pd.Timedelta(days=1)).date()),
                lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX,
            )
            if _inventory is None:
                print("  [WARN] Inventory fetch failed — skipping review waveform gallery.")
            else:
                by_class = {c: [] for c in CLASS_ORDER}
                for _, r in all_rows_df.iterrows():
                    cls = r["predicted_class"]
                    proba_col = f"proba_{str(cls).replace(' ', '_')}"
                    top_p = float(r.get(proba_col, np.nan))
                    save = False
                    if cls != "noise" and SAVE_IMAGES_FOR_NONNOISE:
                        save = True
                    elif cls == "noise" and SAVE_EVERY_NTH_NOISE > 0:
                        _noise_review_counter[0] += 1
                        save = (_noise_review_counter[0] % SAVE_EVERY_NTH_NOISE == 0)
                    if save and cls in by_class:
                        by_class[cls].append((top_p, r))

                for cls, items in by_class.items():
                    if not items:
                        continue
                    items.sort(key=lambda x: -x[0])
                    n_plotted = 0
                    for rank, (p, r) in enumerate(items[:N_GALLERY_PER_CLASS], 1):
                        safe_time = str(r["window_start"]).replace(":", "").replace("-", "").replace(".", "")
                        out_png = os.path.join(
                            review_dir,
                            f"{cls.replace(' ', '_')}_{rank:02d}_{r['network']}_{r['station']}_{safe_time}.png",
                        )
                        try:
                            ok = plot_review_waveform(_client_sds, _inventory, r, cls, p, out_png)
                            if ok:
                                n_plotted += 1
                        except Exception:
                            continue
                    print(f"    {cls:<12s} {n_plotted:3d}/{min(len(items), N_GALLERY_PER_CLASS)} "
                          f"waveform(s) plotted (of {len(items)} candidate(s))")

                print(f"  [SAVED] Review waveform gallery -> {review_dir}/")

    print(f"\n[PHASE 2 COMPLETE] Predictions + review outputs -> {RUN_DIR}/")



# =============================================================================
# SECTION 7 — END
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
