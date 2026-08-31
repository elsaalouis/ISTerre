"""
09b_continuous_tabular_classification.py
=========================================
ISTerre internship
Author : Elsa Louis
Date   : August 2026

Goal
----
End-to-end validation of the scalar-feature HGB classifier, on a RAW CONTINUOUS STREAM, never chained before on data the pipeline hasn't already seen 

TWO PHASES:
  Phase 1 -- EXTRACTION (needs SDS/FDSN + the saved model's feature list):
    per station-day, run the same continuous STA/LTA scan as 09a Phase 1, then for each detected event extract the features

  Phase 2 -- CLASSIFICATION (needs the saved model bundle + sklearn):
    (a) load the fixed model bundle saved by 06c,
    (b) classify every extracted continuous-data window,
    (c) write predictions_<month>.csv + a probability summary figure + a waveform+spectrogram review gallery

Output layout
-------------
  EXTRACTION_DIR/
      feats_<net>_<sta>_<YYYYMMDD>.csv   <- one file per station-day WITH >=1 detection, written by Phase 1, read by Phase 2
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
RUN_EXTRACTION     = True   # Phase 1 
RUN_CLASSIFICATION = True    # Phase 2 

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

# -- Months to scan (Phase 1) ---------------------------------
MONTHS_TO_SCAN = [
    ("2025-01-01", "2025-02-01"),   # January
]

# -- Detection: spectrogram-based STA/LTA (Groult et al. 2026), Phase 1 -------
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

# -- Cross-station coincidence (Phase 1) -----------------------
COINCIDENCE_TOLERANCE_S = 20

# -- MULTI-STATION-ONLY CLASSIFICATION (Phase 2) ---------------------
REQUIRE_MULTISTATION_FOR_CLASSIFICATION = True
MIN_OTHER_STATIONS_FOR_CLASSIFICATION   = 1

# -- Feature extraction window (Phase 1) ------------------------
PAD_SEC = 5        # seconds added before t_on and after t_off
LOAD_3C = True     

# -- Checkpointing / resume (Phase 1: per station-day file existence) ---------
CONSOLIDATE_PER_MONTH = True   # bundle each month's per-station-day CSVs into one file

# -- SMOKE TEST (Phase 1) -- recommended before the full run ---------
SMOKE_TEST              = True
MAX_DAYS_SMOKE_TEST     = 1
MAX_STATIONS_SMOKE_TEST = 1

MODEL_PATH = "/data/failles/louisels/project/results/outputs_06c/IQ_rescue_raw_ablation_noise_regional_20260818_152559/hgb_final_model_20260818_152559.joblib"

CLASS_ORDER  = ["earthquake", "regional", "rockslide", "ice quake", "noise"]   # display order only
CLASS_COLORS = {"earthquake": "#1f77b4", "rockslide": "#d62728",
                "ice quake": "#2ca02c", "noise": "#7f7f7f", "regional": "#9467bd"}

# -- Phase 2d: review waveform+spectrogram gallery (optional) ------
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
REVIEW_PSD_FLOOR_EPS = 1e-20   # same floor as 07a: guards log(0) without swallowing signal

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
                f"EXTRACTION_DIR: {EXTRACTION_DIR}  |  DET_THR_ON={DET_THR_ON}  |  "
                f"COINCIDENCE_TOLERANCE_S={COINCIDENCE_TOLERANCE_S}  |  "
                f"REQUIRE_MULTISTATION_FOR_CLASSIFICATION={REQUIRE_MULTISTATION_FOR_CLASSIFICATION}  |  "
                f"SMOKE_TEST={SMOKE_TEST}")
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
    from detection import compute_snr, merge_window_events, compute_cross_station_coincidence
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


# --------- Phase 2 setup: load the fixed model bundle ----------
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
        Literal copy of 04a_sta_lta_catalog_windowing.py's _fetch_3c_array()
        Returns: arr : np.ndarray, shape (3, n_samples), rows [Z, N, E]  or  None
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
        Returns: row : dict, or None if the window is unusable
        """
        from preprocessing import build_station_times_df, remove_response_or_fallback

        day_start, day_end = day_trace_bounds
        t_cut_on  = max(t_on  - PAD_SEC, day_start)
        t_cut_off = min(t_off + PAD_SEC, day_end)

        snr_pad   = min(max(30.0, 1.5 * (t_off - t_on)), 120.0)
        t_snr_on  = max(t_on  - snr_pad, day_start)
        t_snr_off = min(t_off + snr_pad, day_end)

        try:
            st_raw = client_sds.get_waveforms(net, sta, loc if loc else "*", chan, t_snr_on, t_snr_off)
        except Exception:
            return None
        if len(st_raw) == 0:
            return None
        st_raw.merge(method=1, fill_value="interpolate")

        sdf    = build_station_times_df(st_raw, t_snr_on, t_snr_off)
        st_vel = remove_response_or_fallback(st_raw, inventory, sdf)
        if len(st_vel) == 0:
            return None
        tr_wide = st_vel[0]
        if tr_wide.stats.npts < 10:
            return None
        fs = tr_wide.stats.sampling_rate

        # Feature-extraction window: exactly [t_on-PAD_SEC, t_off+PAD_SEC]
        tr_cut = tr_wide.slice(t_cut_on, t_cut_off)
        if tr_cut.stats.npts < 10:
            return None

        tr_filt_local = tr_wide.copy()
        nyq = fs / 2
        tr_filt_local.filter('bandpass', freqmin=DET_FREQ_MIN,
                             freqmax=min(DET_FREQ_MAX, 0.9 * nyq), corners=4, zerophase=True)

        # Only fetch N/E 
        data_3c = None
        if NEED_3C:
            data_3c = _fetch_3c_array(client_sds, net, sta, chan, t_cut_on, t_cut_off,
                                      tr_cut.data, fs)

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
            val = feat_dict.get(fname, np.nan)
            row[fname] = val if np.isfinite(val) else np.nan
        return row


    def consolidate_month_csvs(extraction_dir, month_tag):
        """Concatenate every station-day feats_*.csv whose 'day' falls in month_tag into one consolidated_<month_tag>.csv"""
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

            # ---- PASS 1: STA/LTA-detect EVERY station for this day first --------------
            station_day_events  = {}   # station_key -> [(t_on, t_off, trigger_cft, trace_bounds), ...]
            station_needs_write = set()
            station_timing      = {}

            for net, sta, loc, chan in station_list:
                station_key = f"{net}.{sta}"
                t_station_day = time.time()

                out_fname = f"feats_{net}_{sta}_{day_utc.strftime('%Y%m%d')}.csv"
                out_path  = os.path.join(EXTRACTION_DIR, out_fname)
                n_station_days += 1

                if os.path.isfile(out_path):
                    has_coincidence = False
                    _read_error = None
                    try:
                        _existing_cols = pd.read_csv(out_path, nrows=0).columns
                        has_coincidence = "n_other_stations_within_tol" in _existing_cols
                    except Exception as e:
                        _read_error = e
                    if has_coincidence:
                        onsets = [UTCDateTime(s) for s in
                                  pd.read_csv(out_path, usecols=["window_start"])["window_start"]]
                        station_day_events[station_key] = [(t_on, None, None, None) for t_on in onsets]
                        print(f"  [SKIP-DETECT] {day_str} {net}.{sta} — already extracted "
                              f"({out_fname}), reusing {len(onsets)} onset(s) for coincidence")
                        continue
                    elif _read_error is not None:
                        print(f"  [REDO] {day_str} {net}.{sta} — {out_fname} unreadable "
                              f"({_read_error}), re-detecting")
                    else:
                        print(f"  [REDO] {day_str} {net}.{sta} — {out_fname} predates the "
                              f"cross-station coincidence check, re-detecting")

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

                day_events = []   # list of (t_on, t_off, trigger_cft, trace_bounds)

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

                    trace_bounds = (tr_vel.stats.starttime, tr_vel.stats.endtime)
                    for ev_key, (t_on, t_off) in total_events.items():
                        day_events.append((t_on, t_off, total_thresholds[ev_key][0], trace_bounds))

                station_day_events[station_key] = day_events
                station_needs_write.add(station_key)
                station_timing[station_key] = t_station_day

            # ---- PASS 2: cross-station coincidence for this day's onsets --------------
            coincidence_by_station = compute_cross_station_coincidence(
                station_day_events, COINCIDENCE_TOLERANCE_S
            )

            # ---- PASS 3: feature-extract ---------------------------------------------
            for net, sta, loc, chan in station_list:
                station_key = f"{net}.{sta}"
                if station_key not in station_needs_write:
                    continue

                t_station_day = station_timing[station_key]
                day_events    = station_day_events[station_key]
                coinc         = coincidence_by_station[station_key]

                out_fname = f"feats_{net}_{sta}_{day_utc.strftime('%Y%m%d')}.csv"
                out_path  = os.path.join(EXTRACTION_DIR, out_fname)

                day_rows = []
                for (t_on, t_off, trigger_cft, trace_bounds), (n_other, other_str) in zip(day_events, coinc):
                    row = extract_window_row(
                        client_sds, inventory, net, sta, loc, chan,
                        trace_bounds, t_on, t_off, trigger_cft, day_str,
                    )
                    if row is not None:
                        row["n_other_stations_within_tol"] = int(n_other)
                        row["other_stations_within_tol"]   = other_str
                        day_rows.append(row)

                dt = time.time() - t_station_day

                if not day_rows:
                    print(f"  {day_str} {net}.{sta:<6s}  0 event(s) detected  [{dt:6.1f}s]")
                    continue

                pd.DataFrame(day_rows).to_csv(out_path, index=False)
                n_events_total += len(day_rows)
                n_multi = sum(1 for r in day_rows if r["n_other_stations_within_tol"] >= 1)
                print(f"  {day_str} {net}.{sta:<6s}  {len(day_rows):4d} event(s) detected+extracted "
                      f"({n_multi}/{len(day_rows)} multi-station) in {dt:6.1f}s")

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
    """Same idea as 09a's function of the same name"""
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
    Literal copy of 08_report_figures_events.py's _fetch_padded_trace()
    Returns: (trace, t_on, t_off, None) on success, (None, None, None, reason_str) on failure
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
    """Literal copy of 08's _trim_to_fixed_length()"""
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
    Same 2-panel style (bandpassed waveform + broadband dB spectrogram) as 08's report figure gallery, 
    via visualization.plot_waveform_spectrogram_example() 
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
                "SNR_full_mean", "SNR_full_median", "SNR_s2n_median",
                "n_other_stations_within_tol", "other_stations_within_tol"]

    rows_by_month = {}
    _noise_review_counter = [0]

    for fpath in feature_files:
        df_feat = pd.read_csv(fpath, low_memory=False)
        if df_feat.empty:
            continue

        if REQUIRE_MULTISTATION_FOR_CLASSIFICATION:
            if "n_other_stations_within_tol" not in df_feat.columns:
                print(f"  [WARN] {os.path.basename(fpath)} — no n_other_stations_within_tol "
                      f"column (predates the cross-station coincidence check) — classifying "
                      f"ALL windows in this file, REQUIRE_MULTISTATION_FOR_CLASSIFICATION "
                      f"cannot be applied. Re-run Phase 1 to get coincidence annotations.")
            else:
                n_before_mc = len(df_feat)
                df_feat = df_feat[
                    df_feat["n_other_stations_within_tol"] >= MIN_OTHER_STATIONS_FOR_CLASSIFICATION
                ].copy()
                n_dropped_mc = n_before_mc - len(df_feat)
                if n_dropped_mc:
                    print(f"  [MULTISTATION-ONLY] {os.path.basename(fpath)} — skipping "
                          f"classification on {n_dropped_mc:,} / {n_before_mc:,} "
                          f"single-station-only window(s).")
                if df_feat.empty:
                    continue

        missing_feats = [f for f in final_features if f not in df_feat.columns]
        if missing_feats:
            print(f"  [SKIP] {os.path.basename(fpath)} — missing feature column(s): "
                  f"{missing_feats[:5]}{'...' if len(missing_feats) > 5 else ''}")
            continue

        X = df_feat[final_features].values.astype(float)
        n_inf = int(np.isinf(X).sum())
        if n_inf:
            X[np.isinf(X)] = np.nan
            print(f"         [WARN] {n_inf} infinite feature value(s) in "
                  f"{os.path.basename(fpath)} -> treated as missing (imputed).")
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

        if _client_sds is None:
            print("  [WARN] SDS unavailable — skipping review waveform gallery "
                  "(this step only works on the cluster / with VPN access).")
        else:
            all_rows_df = pd.concat([pd.DataFrame(r) for r in rows_by_month.values()], ignore_index=True)

            _inventory = None
            if _client_fdsn is not None:
                _t_min = pd.to_datetime(all_rows_df["window_start"]).min()
                _t_max = pd.to_datetime(all_rows_df["window_end"]).max()
                _inventory = fetch_inventory(
                    _client_fdsn, str(_t_min.date()), str((_t_max + pd.Timedelta(days=1)).date()),
                    lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX,
                )

            if _inventory is None:
                print("  [WARN] No instrument inventory (FDSN unreachable or inventory fetch")
                print("         failed) -- continuing with UNCALIBRATED raw-counts waveforms.")
                print("         Shape/duration/frequency content are still roughly indicative;")
                print("         amplitude/true ground velocity are NOT. Figures are tagged")
                print("         'UNCALIBRATED' in the title and '_UNCAL' in the filename --")
                print("         rerun once FDSN is back for calibrated figures.")
            else:
                print(f"  [OK] Instrument inventory fetched -- figures will be calibrated ground velocity.")

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
                    safe_time  = str(r["window_start"]).replace(":", "").replace("-", "").replace(".", "")
                    cal_suffix = "" if _inventory is not None else "_UNCAL"
                    out_png = os.path.join(
                        review_dir,
                        f"{cls.replace(' ', '_')}_{rank:02d}_{r['network']}_{r['station']}_{safe_time}{cal_suffix}.png",
                    )
                    try:
                        ok = plot_review_waveform(_client_sds, _inventory, r, cls, p, out_png,
                                                  calibrated=(_inventory is not None))
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
