"""
09b_continuous_tabular_classification.py
=========================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : August 2026

Goal
----
End-to-end validation of the scalar-feature HGB classifier, on a RAW CONTINUOUS STREAM, 
never chained before on data the pipeline hasn't already seen as pre-curated catalog windows

TWO PHASES (kept for workflow convenience, see above) -----------------------
  Phase 1 -- EXTRACTION (needs SDS/FDSN + the saved model's feature list):
    per station-day, run the same continuous STA/LTA scan as 09a Phase 1,
    then for each detected event extract ONLY the features MODEL_PATH's
    saved bundle actually uses (Top-60, not the full 99+4) + 7 SNR measures
    -> one CSV per station-day, consolidated per month. The N/E fetch +
    polarization computation is skipped entirely when none of those features
    are in the model's list (checked once at startup, see NEED_3C).

  Phase 2 -- CLASSIFICATION (needs the saved model bundle + sklearn):
    (a) load the fixed model bundle saved by 06c,
    (b) classify every extracted continuous-data window,
    (c) write predictions_<month>.csv + a probability summary figure +
        (optional, needs SDS again) a waveform+spectrogram review gallery,
        same 2-panel style as 08a's report figures.

Output layout
-------------
  EXTRACTION_DIR/
      feats_<net>_<sta>_<YYYYMMDD>.csv   <- one file per station-day WITH
                                             >=1 detection, written by Phase 1,
                                             read by Phase 2
      consolidated_<month>.csv           <- Phase 1, one file per scanned month
  outputs_09b/run_YYYYMMDD_HHMMSS/    (this invocation's own log + phase-2 output)
      predictions_<month_tag>.csv          <- Phase 2b: one row per classified detection
      probability_summary/*.png            <- Phase 2c
      review_waveforms/*.png               <- Phase 2d (optional, needs SDS)
      run.log
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# -- Run mode -- see the module docstring for why this split is kept ---------
RUN_EXTRACTION     = True   # Phase 1 -- needs SDS/FDSN, no sklearn dependency
RUN_CLASSIFICATION = True    # Phase 2 -- needs the training CSVs + sklearn/imblearn

# -- Interchange directory between the two phases -----------------------------
# Phase 1 writes per-station-day + consolidated feature CSVs here
EXTRACTION_DIR = "/data/failles/louisels/project/results/outputs_09b/feature_csv"

# -- Paths (Phase 1 -- SDS/FDSN) -----------------------------------------------
SDS_ROOT    = "/data/sig/SDS"
ISTERRE_URL = "http://ist-sc3-geobs.osug.fr:8080"

# -- Output for THIS invocation's own log + Phase 2 outputs -------------------
OUTPUT_DIR = "/data/failles/louisels/project/results/outputs_09b"

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
LOAD_3C = True     # master switch for polarization features; the ACTUAL decision
                   # (NEED_3C) also checks whether MODEL_PATH's saved feature list
                   # contains a polarization feature at all -- set this False to
                   # force-skip the N/E fetch even if the model happens to use one
                   # (features.py fills NaN, the model's own imputer handles it)

# -- Checkpointing / resume (Phase 1: per station-day file existence) ---------
CONSOLIDATE_PER_MONTH = True   # bundle each month's per-station-day CSVs into one file

# -- SMOKE TEST (Phase 1) -- strongly recommended before the full run ---------
SMOKE_TEST              = True
MAX_DAYS_SMOKE_TEST     = 1
MAX_STATIONS_SMOKE_TEST = 1

MODEL_PATH = "/data/failles/louisels/project/results/outputs_06c/IQ_rescue_raw_ablation_noise_regional_20260818_152559/hgb_final_model_20260818_152559.joblib"

CLASS_ORDER  = ["earthquake", "regional", "rockslide", "ice quake", "noise"]   # display order only
CLASS_COLORS = {"earthquake": "#1f77b4", "rockslide": "#d62728",
                "ice quake": "#2ca02c", "noise": "#7f7f7f", "regional": "#9467bd"}

# -- Phase 2d: review waveform+spectrogram gallery (optional, needs SDS) ------
SAVE_REVIEW_WAVEFORMS    = True
SAVE_IMAGES_FOR_NONNOISE = True    # save every window NOT predicted as 'noise'
SAVE_EVERY_NTH_NOISE     = 5
N_GALLERY_PER_CLASS      = 10

REVIEW_PRE_S       = 10     # seconds BEFORE the detection onset shown in the figure
REVIEW_WINDOW_S    = 100    # total fixed display window [s] -- matches 08a's FIXED_WINDOW_S
REVIEW_TARGET_FS   = 200    # [Hz] resample target before the spectrogram, matches 08a
REVIEW_FETCH_PAD_S = 60     # extra context fetched (not shown) so filtering has run-in room
REVIEW_WAVE_FREQMIN, REVIEW_WAVE_FREQMAX = 1.0, 20.0

REVIEW_SPEC_NPERSEG_S     = 2.0     # [s] STFT segment length
REVIEW_SPEC_NOVERLAP_FRAC = 0.75
REVIEW_SPEC_NFFT          = 512
REVIEW_FREQ_MAX_KEEP      = 95.0    # [Hz] 95% of Nyquist at REVIEW_TARGET_FS=200Hz
REVIEW_SPEC_VMIN, REVIEW_SPEC_VMAX = -200, -120   # dB color scale, matches 08a
REVIEW_SPEC_NPERSEG  = int(REVIEW_SPEC_NPERSEG_S * REVIEW_TARGET_FS)
REVIEW_SPEC_NOVERLAP = int(REVIEW_SPEC_NPERSEG * REVIEW_SPEC_NOVERLAP_FRAC)
REVIEW_PSD_FLOOR_EPS = 1e-20   # same floor as 07a/08a: guards log(0) without swallowing signal

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
    from features import FEATURE_NAMES_3C, extract_features, POLARIZATION_NAMES

    # Only extract the features the saved model actually uses (Top-60, not all 103) -- both cheaper to store
    try:
        import joblib as _joblib_extract
    except ImportError:
        print("[ERROR] joblib is required to read the saved model's feature list. Exiting.")
        log_file.close()
        sys.exit(1)
    if not os.path.isfile(MODEL_PATH):
        print(f"[ERROR] MODEL_PATH not found: {MODEL_PATH}")
        print("        Run 06c_train_HGB_classifier.py once with SAVE_FINAL_MODEL=True,")
        print("        then point MODEL_PATH at the hgb_final_model_<stamp>.joblib it writes.")
        log_file.close()
        sys.exit(1)
    _FEAT_NAMES = list(_joblib_extract.load(MODEL_PATH)["features"])
    N_FEATURES  = len(_FEAT_NAMES)
    NEED_3C     = LOAD_3C and any(f in POLARIZATION_NAMES for f in _FEAT_NAMES)
    print(f"\n[OK] Extracting only the {N_FEATURES} feature(s) the saved model at "
          f"{MODEL_PATH} actually uses.")
    print(f"     Polarization fetch (N/E, 3C): {'enabled' if NEED_3C else 'skipped'} "
          f"({'a polarization feature is' if NEED_3C else 'no polarization feature is'} "
          f"in that list).")

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


# --------- Phase 2 setup: load the fixed model bundle (no TensorFlow, no ------
# training here -- see the module docstring "Model source" section) ----------
if RUN_CLASSIFICATION:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram as scipy_spectrogram
    try:
        import joblib
    except ImportError:
        print("[ERROR] joblib is required to load the saved model bundle. Exiting.")
        log_file.close()
        sys.exit(1)

    from run_setup import set_matplotlib_defaults
    from visualization import plot_waveform_spectrogram_example
    set_matplotlib_defaults()

    if not os.path.isfile(MODEL_PATH):
        print(f"[ERROR] MODEL_PATH not found: {MODEL_PATH}")
        print("        Run 06c_train_HGB_classifier.py once with SAVE_FINAL_MODEL=True,")
        print("        then point MODEL_PATH at the hgb_final_model_<stamp>.joblib it writes.")
        log_file.close()
        sys.exit(1)

    print(f"\n[LOAD] Model bundle : {MODEL_PATH}")
    _bundle = joblib.load(MODEL_PATH)
    final_model    = _bundle["model"]
    final_imputer  = _bundle["imputer"]
    final_features = _bundle["features"]
    _metrics       = _bundle.get("metrics", {})
    print(f"[OK] Loaded '{_bundle.get('run_label', '?')}' classifier "
          f"(trained {_bundle.get('trained_on', '?')} by {_bundle.get('source_script', '?')})")
    print(f"     acc={_metrics.get('acc', float('nan')):.3f}  "
          f"macro F1={_metrics.get('macro_f1', float('nan')):.3f}  "
          f"n_features={len(final_features)}")
    print(f"     Classes (model order): {list(final_model.classes_)}")

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

        # Only fetch N/E (and pay for the polarization computation inside
        # extract_features) when the saved model actually uses a polarization
        # feature -- NEED_3C is decided once, at startup, from MODEL_PATH's
        # own feature list (see Phase 1 setup above).
        data_3c = None
        if NEED_3C:
            data_3c = _fetch_3c_array(client_sds, net, sta, chan, t_cut_on, t_cut_off,
                                      tr_cut.data, fs)

        # extract_features() always computes the full 99 Z-features in one
        # monolithic call (seismic_params.calculate_all_attributes, third-party,
        # can't be asked for a subset) + 4 more if data_3c is given -- so we
        # still compute everything NEED_3C implies, then keep only the columns
        # _FEAT_NAMES (the model's Top-N) actually needs. This is where the
        # real saving happens: skipping the N/E fetch above when NEED_3C is
        # False, not the Z-feature computation itself (which can't be split).
        full_names = FEATURE_NAMES_3C if NEED_3C else FEATURE_NAMES_3C[:99]
        feats      = extract_features(tr_cut.data, fs, data_3c=data_3c)
        feat_dict  = dict(zip(full_names, feats))
        snr        = compute_snr(tr_filt_local, t_on, t_off)

        row = {
            "network": net, "station": sta, "location": loc, "channel": chan,
            "day": day_str,
            "window_start": str(t_on), "window_end": str(t_off),
            "duration_s": round(t_off - t_on, 2),
            "trigger_on_cft": round(trigger_cft, 4),
            **snr,
        }
        for fname in _FEAT_NAMES:
            row[fname] = feat_dict.get(fname, np.nan)
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
# SECTION 5 — CLASSIFY CONTINUOUS DATA + REVIEW OUTPUTS
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


def _fetch_padded_trace(client_sds, inventory, net, sta, chan, det_start,
                        pre_s, window_s, target_fs, fetch_pad_s):
    """
    Literal copy of 08a_report_figures_events.py's _fetch_padded_trace() --
    kept as a copy (not an import) for the same reason as this script's other
    copied helpers: no hidden dependency on 08a's own CONFIGURATION section.
    Fetches a window LARGER than needed (fetch_pad_s of extra context each
    side) so response removal / filtering have run-in room before the region
    actually shown in the figure.

    Returns
    -------
    (trace, t_on, t_off, None) on success, (None, None, None, reason_str) on failure.
    """
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
    """Literal copy of 08a's _trim_to_fixed_length() -- trim a COPY of tr to
    exactly [t_on, t_off] -> exactly window_s*target_fs samples."""
    tr = tr.copy()
    tr.trim(t_on, t_off, pad=True, fill_value=0)
    nt = int(round(window_s * target_fs))
    if len(tr.data) < nt:
        tr.data = np.pad(tr.data, (0, nt - len(tr.data)))
    elif len(tr.data) > nt:
        tr.data = tr.data[:nt]
    return tr


def plot_review_waveform(client_sds, inventory, row, cls_name, top_proba, out_path):
    """
    Same 2-panel style (bandpassed waveform + broadband dB spectrogram) as
    08a's report figure gallery, via visualization.plot_waveform_spectrogram_
    example() -- the tabular-branch review image now looks the same as the
    training-catalog example figures, just with predicted class/probability
    standing in for the (unknown, here) true class and catalog distance.
    Needs SDS access; caller should catch exceptions and skip gracefully.
    """
    from obspy import UTCDateTime

    det_start = UTCDateTime(row["window_start"])

    tr_padded, t_on, t_off, err = _fetch_padded_trace(
        client_sds, inventory, row["network"], row["station"], row["channel"], det_start,
        REVIEW_PRE_S, REVIEW_WINDOW_S, REVIEW_TARGET_FS, REVIEW_FETCH_PAD_S,
    )
    if tr_padded is None:
        return False

    # -- broadband copy for the spectrogram: trim only, no filtering ---------
    tr_broadband = _trim_to_fixed_length(tr_padded, t_on, t_off, REVIEW_TARGET_FS, REVIEW_WINDOW_S)
    if not (np.all(np.isfinite(tr_broadband.data)) and np.max(np.abs(tr_broadband.data)) > 0):
        return False

    # -- waveform panel: bandpass the PADDED trace first, trim after ---------
    tr_wave_padded = tr_padded.copy()
    nyq = tr_wave_padded.stats.sampling_rate / 2.0
    tr_wave_padded.filter("bandpass", freqmin=REVIEW_WAVE_FREQMIN,
                          freqmax=min(REVIEW_WAVE_FREQMAX, 0.9 * nyq),
                          corners=4, zerophase=True)
    tr_wave = _trim_to_fixed_length(tr_wave_padded, t_on, t_off, REVIEW_TARGET_FS, REVIEW_WINDOW_S)

    # -- spectrogram: broadband (unfiltered) ----------------------------------
    f_full, t_full, Sxx = scipy_spectrogram(
        tr_broadband.data, fs=tr_broadband.stats.sampling_rate, window="hann",
        nperseg=REVIEW_SPEC_NPERSEG, noverlap=REVIEW_SPEC_NOVERLAP, nfft=REVIEW_SPEC_NFFT,
        scaling="density", mode="psd",
    )
    freq_mask = f_full <= REVIEW_FREQ_MAX_KEEP
    freq_axis = f_full[freq_mask]
    Sxx_db    = 10 * np.log10(Sxx[freq_mask, :] + REVIEW_PSD_FLOOR_EPS)

    snr_val   = row.get("SNR", np.nan)
    snr_str   = f"SNR={snr_val:.2f}" if pd.notna(snr_val) else "SNR=n/a"
    title_l1  = f"{cls_name} (predicted, p={top_proba:.2f}) — {str(row['window_start'])[:19]}"
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
        spec_vmin      = REVIEW_SPEC_VMIN,
        spec_vmax      = REVIEW_SPEC_VMAX,
    )
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
