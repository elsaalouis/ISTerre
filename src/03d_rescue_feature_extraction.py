"""
03d_rescue_feature_extraction.py
=================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : July 2026

Post-denoiser processing: 
- load the DeepDenoiser-denoised waveforms produced by 03c_denoiser_event_data.py
- recompute SNR metrics on the clean signal
- re-apply the quality gate
- extract the Maggi/Hibert features on events that pass
- save a rescue catalog CSV (rescue_catalog_<stamp>.csv) in the exact same column schema as the 04a output

Pipeline position
-----------------
  03c ✓  →  [03d this script]  →  06c

How filenames encode metadata
------------------------------
  rescue_{net}_{sta}_{cha}_{run_date}_{run_time}_{row_idx}.npz     e.g.   rescue_CH_FULLY_HHZ_20260605_102106_4880.npz
                                                  ^^^^
                                                  row index in the 04a catalog CSV gives us all original metadata (event_time, lat/lon, det window…)

SNR computation
---------------
The DeepDenoiser writes 3000-sample windows (30 s at 100 Hz)
itp = 1000 is hardcoded by 03c (10 s of noise before the onset)
  - noise_window  : signal[0   : itp]       (10 s × 100 Hz = 1000 samples)
  - signal_window : signal[itp : 3000]      (20 s × 100 Hz = 2000 samples)
SNR metrics (same definitions as detection.py):
  SNR_full_mean   = mean(|sig|)   / mean(|noise|)
  SNR_full_median = median(|sig|) / median(|noise|)
  SNR_s2n_median  = 99.5th pct(|sig|) / MAD(noise)   

Quality gate (per-event-type ROC-optimal thresholds from 05a — see snr_summary_by_type_*.csv)
  SNR_full_mean  >= SNR_FULL_MEAN_MIN
  SNR_s2n_median >= SNR_S2N_MEDIAN_MIN

Note: SNR_full_median is tracked and plotted for QC purposes, but is NOT part of the
accept/reject gate above.

Outputs
-------
  rescue_catalog_<stamp>.csv         — accepted (gate-passing) rows, schema matches 04a
  denoise_qc_<stamp>.csv             — one row per rescue candidate (before/after SNR + fidelity metrics)
  snr_improvement_summary_<stamp>.csv — gate-INDEPENDENT summary: how much SNR improved overall, across every denoised candidate regardless of pass/fail
  snr_before_after_<stamp>.png       — paired SNR scatter, raw vs denoised, per metric (all candidates)
  snr_delta_distribution_<stamp>.png — histogram of log10(SNR after/before), per metric (all candidates)
  rescue_funnel_<stamp>.png          — candidates -> passed gate, as a funnel bar chart
  denoise_fidelity_<stamp>.png       — waveform correlation (raw vs denoised) vs SNR gain
  waveform_compare_<fname>_<stamp>.png — raw vs denoised waveform, one per selected event (see MAKE_WAVEFORM_PLOTS / WAVEFORM_SELECT_MODE)
"""


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# ── Input: denoised NPZ files from 03c ───────────────────────────────────────
DENOISED_DIR = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\03c_denoiser_event_data\rockslide\run_20260709_180801\denoised\results"

# ── Input: original (pre-denoising) rescue NPZ files — needed for itp ────────
# Same run as DENOISED_DIR above — must always point at the matching run_.../rescue folder.
RESCUE_DIR   = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\03c_denoiser_event_data\rockslide\run_20260709_180801\rescue"

# ── Input: master catalog from 04a — used for metadata lookup by row index ───
CATALOG_CSV  = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\04a_spectrogram_sta_lta_catalog\all-99-features-recent\catalog_windows_20260707_165719.csv"

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR   = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\03d_rescue_feature_extraction"

# ── NPZ filename prefix ───────────────────────────────────────────────────────
FILE_PREFIX  = "rescue"   

# ── Signal parameters ─────────────────────────────────────────────────────────
SPS    = 100.0    # sampling rate (Hz) — fixed by 03c --sampling_rate 100
ITP    = 1000     # P-arrival sample index (10 s × 100 Hz, fixed by 03c windowing)
N_SAMP = 3000     # total samples in each NPZ window

# ── Quality gate (same thresholds as 05a / 06b) ───────────────────────────────
SNR_FULL_MEAN_MIN  = 1.91    # 05a ROC-optimal   
SNR_FULL_MEDIAN_MIN = 1.60

# ── Minimum signal samples for feature extraction ────────────────────────────
MIN_SIGNAL_SAMPLES = 200   # < 2 s at 100 Hz → skip (too short for 99 features)

# ── QC diagnostic plots ───────────────────────────────────────────────────────
# Signal-quality plots only (before/after SNR, fidelity) — covers every rescue candidate, not just the ones that pass the gate
MAKE_QC_PLOTS = True

# ── Single-event waveform comparison plots (raw vs denoised) ─────────────────
# Two selection modes:
#   "top_delta" — auto-pick the N candidates with the biggest SNR change (see WAVEFORM_RANK_METRIC / WAVEFORM_RANK_ABS below)
#   "manual"    — plot exactly the fnames listed in WAVEFORM_MANUAL_FNAMES
MAKE_WAVEFORM_PLOTS   = True
WAVEFORM_SELECT_MODE  = "top_delta"      # "top_delta" or "manual"

# -- "top_delta" mode settings --
WAVEFORM_TOP_N        = 10
WAVEFORM_RANK_METRIC  = "SNR_full_median"  # which QC metric ranks "biggest delta"
WAVEFORM_RANK_ABS     = False             # False = biggest IMPROVEMENT only (delta desc)
                                          # True  = biggest change either direction (|delta| desc)

# -- "manual" mode settings --
# the rescue .npz basename without extension, e.g. "rescue_8D_AMID2_EHZ_20260709_180801_35284" 
#  -> see denoise_qc_<stamp>.csv's 'fname' column for valid values from a given run
WAVEFORM_MANUAL_FNAMES = []

# Display-only bandpass applied to the plotted waveforms 
#  -> set either to None to disable that side (e.g. FREQMIN=None, FREQMAX=20 -> lowpass)
WAVEFORM_FILTER_FREQMIN = 1.0    # Hz
WAVEFORM_FILTER_FREQMAX = 20.0   # Hz


# =============================================================================
# SECTION 2 — SETUP
# =============================================================================

import os
import sys
import glob
import time
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

# Add project src to path so all project modules are importable
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
from features import FEATURE_NAMES, POLARIZATION_NAMES, extract_features
from detection import compute_snr_numpy, compute_denoise_correlation   # numpy-only, no ObsPy needed
from run_setup import create_run_dir, setup_logging, set_matplotlib_defaults
from visualization import (
    plot_snr_before_after,
    plot_delta_snr_distribution,
    plot_rescue_funnel,
    plot_denoise_fidelity,
    plot_waveform_comparison,
)

RUN_DIR, STAMP = create_run_dir(OUTPUT_DIR)
log_file, log_path = setup_logging(
    RUN_DIR,
    script_name="03d_rescue_feature_extraction.py",
    extra_info=f"DENOISED_DIR: {DENOISED_DIR}",
)
set_matplotlib_defaults()


# =============================================================================
# SECTION 3 — LOAD MASTER CATALOG
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 1 — Loading master catalog")
print(f"{'='*65}")

catalog = pd.read_csv(CATALOG_CSV, low_memory=False)
print(f"Loaded {len(catalog):,} rows × {len(catalog.columns)} columns.")
catalog_cols = list(catalog.columns)

# Detect whether the catalog was produced with LOAD_3C=True (103 features) or not (99).
# If polarization column names are present, rescued events will get NaN for those columns
# (DeepDenoiser is Z-only; horizontal data is not available for the denoised windows).
_has_3c = all(p in catalog_cols for p in POLARIZATION_NAMES)
print(f"  Catalog feature mode : {'103 features (3C) — polarization will be NaN for rescued events' if _has_3c else '99 features (Z-only)'}")


# =============================================================================
# SECTION 4 — DISCOVER DENOISED FILES
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 2 — Discovering denoised NPZ files")
print(f"{'='*65}")

denoised_files = sorted(glob.glob(os.path.join(DENOISED_DIR, f"{FILE_PREFIX}_*.npz")))
print(f"Found {len(denoised_files):,} denoised NPZ files.")


# =============================================================================
# SECTION 5 — MAIN PROCESSING LOOP
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 3 — SNR + feature extraction")
print(f"{'='*65}")
print(f"  Quality gate: SNR_full_mean >= {SNR_FULL_MEAN_MIN}  "
      f"AND  SNR_full_median >= {SNR_FULL_MEDIAN_MIN}")

rescue_rows  = []    # list of dicts — one per accepted event
qc_rows      = []    # list of dicts — one per EVERY evaluated candidate (pass or fail),
                      # feeds the QC plots below; independent of the classification step
n_total      = len(denoised_files)
n_snr_fail   = 0
n_feat_fail  = 0
n_missing_orig  = 0
n_raw_load_fail = 0
_raw_load_errs  = []   # first few exceptions, for diagnostics

for fpath in tqdm(denoised_files, desc="Processing"):
    fname = os.path.basename(fpath)[:-4]   # strip .npz

    # ── Parse filename ────────────────────────────────────────────────────────
    # format: rescue_{net}_{sta}_{cha}_{run_date}_{run_time}_{row_idx}
    parts   = fname.split('_')
    row_idx = int(parts[-1])
    # cha is parts[-4], sta is parts[-5], net is parts[-6]
    # (run_date = parts[-3], run_time = parts[-2])
    cha = parts[-4]
    sta = parts[-5]
    net = parts[-6]

    # ── Load denoised waveform  ───────────────────────────────────────────────
    try:
        den = np.load(fpath, allow_pickle=True)
        denoised_signal = den['data'][:, 0, 0].astype(np.float64)  # shape (3000,)
    except Exception as e:
        print(f"  [WARN] Could not load denoised file {fname}: {e}")
        continue

    # ── Load original rescue NPZ (itp + raw waveform, needed for the before/after QC) ──
    orig_path   = os.path.join(RESCUE_DIR, os.path.basename(fpath))
    raw_signal  = None
    if not os.path.exists(orig_path):
        n_missing_orig += 1
        itp = ITP   # fallback to constant (always 1000 from 03c)
    else:
        try:
            orig       = np.load(orig_path, allow_pickle=True)
            itp        = int(orig['itp'])
            # Raw rescue NPZ stores data as (3000, 3) — E, N, Z components (03c line ~495);
            # Z is index 2 (see 03c's own flat-trace check: `data3[:, 2]`). This is a
            # different shape than the denoised NPZ (3000, 1, 1), which uses [:, 0, 0].
            raw_signal = orig['data'][:, 2].astype(np.float64)
        except Exception as e:
            # File exists on disk (checked above) but failed to load — this is the
            # OneDrive Files-On-Demand signature (placeholder present, content not
            # actually downloaded yet), NOT a missing file. Count + log it distinctly
            # from n_missing_orig so it's diagnosable instead of silently zeroing out
            # every before/after SNR comparison.
            n_raw_load_fail += 1
            if len(_raw_load_errs) < 5:
                _raw_load_errs.append((os.path.basename(orig_path), str(e)))
            itp = ITP

    # ── Metadata from master catalog ──────────────────────────────────────────
    if row_idx < 0 or row_idx >= len(catalog):
        print(f"  [WARN] row_idx {row_idx} out of catalog range — skipping {fname}")
        continue
    meta = catalog.iloc[row_idx]

    # ── SNR on denoised signal ─────────────────────────────────────────────────
    det_dur = float(meta.get('det_duration_s', 10.0))
    snr_dict = compute_snr_numpy(denoised_signal, itp, det_dur, SPS)

    # ── QC: SNR on the raw (pre-denoiser) signal + waveform-fidelity check ──────
    # Only possible when the original rescue .npz was found above (raw_signal is not None).
    # This is what lets the plots distinguish "genuinely denoised" from "SNR went up
    # because the model invented structure" — independent of the quality gate below.
    if raw_signal is not None:
        snr_dict_raw = compute_snr_numpy(raw_signal, itp, det_dur, SPS)
        fidelity     = compute_denoise_correlation(raw_signal, denoised_signal, itp, det_dur, SPS)
    else:
        snr_dict_raw = {'SNR_full_mean': np.nan, 'SNR_full_median': np.nan, 'SNR_s2n_median': np.nan}
        fidelity     = {'corr_signal': np.nan, 'corr_noise': np.nan,
                         'energy_ratio_signal': np.nan, 'energy_ratio_noise': np.nan}

    # ── Quality gate ──────────────────────────────────────────────────────────
    snr_mean   = snr_dict.get('SNR_full_mean',  0.0)
    snr_median    = snr_dict.get('SNR_full_median', 0.0)
    passes     = (not np.isnan(snr_mean) and snr_mean  >= SNR_FULL_MEAN_MIN and
                  not np.isnan(snr_median)  and snr_median   >= SNR_FULL_MEDIAN_MIN)

    qc_rows.append({
        'fname'               : fname,
        'row_idx'             : row_idx,
        'event_time'          : meta.get('event_time', None),
        'event_type'          : meta.get('event_type', None),
        'network'             : net, 'station': sta, 'channel': cha,
        'SNR_full_mean_raw'   : snr_dict_raw.get('SNR_full_mean',  np.nan),
        'SNR_full_mean'       : snr_dict.get('SNR_full_mean',      np.nan),
        'SNR_full_median_raw'  : snr_dict_raw.get('SNR_full_median', np.nan),
        'SNR_full_median'      : snr_dict.get('SNR_full_median',     np.nan),
        'SNR_s2n_median_raw'  : snr_dict_raw.get('SNR_s2n_median',  np.nan),
        'SNR_s2n_median'      : snr_dict.get('SNR_s2n_median',      np.nan),
        'corr_signal'         : fidelity.get('corr_signal',         np.nan),
        'corr_noise'          : fidelity.get('corr_noise',          np.nan),
        'energy_ratio_signal' : fidelity.get('energy_ratio_signal', np.nan),
        'energy_ratio_noise'  : fidelity.get('energy_ratio_noise',  np.nan),
        'passed_gate'         : passes,
    })

    if not passes:
        n_snr_fail += 1
        continue

    # ── Feature extraction on signal window ───────────────────────────────────
    sig_end = min(N_SAMP, itp + max(int(det_dur * SPS), MIN_SIGNAL_SAMPLES))
    sig_window = denoised_signal[itp:sig_end]

    if len(sig_window) < MIN_SIGNAL_SAMPLES:
        n_feat_fail += 1
        continue

    feats = extract_features(sig_window, sps=SPS)
    if np.any(np.isnan(feats)):
        n_feat_fail += 1
        continue

    # ── Build output row (same schema as 04a catalog) ─────────────────────────
    row = {col: meta.get(col, np.nan) for col in catalog_cols}

    # Overwrite SNR columns with denoised-signal values
    for k, v in snr_dict.items():
        if k in row:
            row[k] = v
    row['quality_ok'] = True

    # Overwrite the 99 Z-component features
    for feat_name, val in zip(FEATURE_NAMES, feats):
        row[feat_name] = val

    # If the catalog has 3C polarization columns, write NaN for rescued events.
    # Horizontal channels are not available from the Z-only denoised NPZ files.
    if _has_3c:
        for pol_name in POLARIZATION_NAMES:
            row[pol_name] = np.nan

    # Also overwrite snr (lowercase alias used by some scripts)
    if 'snr' in row:
        row['snr'] = snr_dict.get('SNR', np.nan)

    # Add provenance columns (appended at the end, won't break 06c concat)
    row['source']           = 'denoised_rescue'
    row['original_row_idx'] = row_idx

    rescue_rows.append(row)


# =============================================================================
# SECTION 6 — SUMMARY
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 4 — Results")
print(f"{'='*65}")
print(f"  Total denoised files   : {n_total:,}")
print(f"  Failed SNR gate        : {n_snr_fail:,}  "
      f"({100*n_snr_fail/max(n_total,1):.1f} %)")
print(f"  Feature extraction err : {n_feat_fail:,}")
print(f"  Missing original NPZ   : {n_missing_orig:,}")
print(f"  Original NPZ load fail : {n_raw_load_fail:,}  "
      f"({100*n_raw_load_fail/max(n_total,1):.1f} %)  "
      f"[before/after SNR + fidelity QC unavailable for these]")
if n_raw_load_fail > 0:
    for fname_err, err in _raw_load_errs:
        print(f"           {fname_err}: {err}")
    if n_raw_load_fail > len(_raw_load_errs):
        print(f"           ... and {n_raw_load_fail - len(_raw_load_errs)} more")
if n_raw_load_fail / max(n_total, 1) > 0.3:
    _err_text = " ".join(e for _, e in _raw_load_errs).lower()
    if any(s in _err_text for s in ("no data left", "eof", "zip", "0 bytes", "not a zip file")):
        print("  [WARN] Most/all original rescue files failed to load even though they exist on disk —")
        print("         this is the OneDrive Files-On-Demand signature (placeholder present, content not")
        print("         actually downloaded). In File Explorer, right-click the RESCUE_DIR folder ->")
        print("         'Always keep on this device', then re-run once sync finishes.")
    else:
        print("  [WARN] Most/all original rescue files failed to load with a consistent error — this")
        print("         looks like a data-format issue (e.g. array shape mismatch), not a sync problem.")
        print("         Check the error text above against how RESCUE_DIR's NPZ files were written.")
print(f"  ─────────────────────────────────────────")
print(f"  Accepted (rescued)     : {len(rescue_rows):,}  "
      f"({100*len(rescue_rows)/max(n_total,1):.1f} %)")


# =============================================================================
# SECTION 6b — QC DIAGNOSTICS: signal-quality plots (no classification here — see 06c)
# =============================================================================
# Covers every evaluated candidate (qc_rows), not just the ones that pass the gate —
# needed to see the full picture: did the denoiser help, by how much, and is it
# recovering real signal or just inventing smooth structure?

if MAKE_QC_PLOTS and qc_rows:
    print(f"\n{'='*65}")
    print("  STEP 5 — QC diagnostics (signal quality, not classification)")
    print(f"{'='*65}")

    qc_df = pd.DataFrame(qc_rows)
    qc_csv_path = os.path.join(RUN_DIR, f"denoise_qc_{STAMP}.csv")
    qc_df.to_csv(qc_csv_path, index=False)
    print(f"  [SAVED] {qc_csv_path}  ({len(qc_df):,} candidates)")

    _event_type_label = (
        qc_df['event_type'].mode().iat[0]
        if qc_df['event_type'].notna().any() else ""
    )

    _metric_pairs = [
        ('SNR_full_mean_raw',   'SNR_full_mean',   'SNR_full_mean'),
        ('SNR_full_median_raw', 'SNR_full_median', 'SNR_full_median'),
        ('SNR_s2n_median_raw',  'SNR_s2n_median',  'SNR_s2n_median'),
    ]
    # Gate metrics are SNR_full_mean and SNR_s2n_median — SNR_full_median is tracked
    # for QC only and has no threshold line drawn on its panel.
    _thresholds = {
        'SNR_full_mean':  SNR_FULL_MEAN_MIN,
        'SNR_full_median': SNR_FULL_MEDIAN_MIN,
    }

    # ── Gate-INDEPENDENT summary: how much did the denoiser improve the SNR overall? ──
    # This ignores passed_gate entirely (every evaluated candidate counts)
    print("\n  SNR improvement summary — ALL denoised candidates, gate not considered:")
    summary_rows = []
    for before_col, after_col, label in _metric_pairs:
        x = qc_df[before_col].to_numpy(dtype=float)
        y = qc_df[after_col].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        n_valid = int(valid.sum())
        if n_valid == 0:
            print(f"    {label:<16s}  no valid before/after pairs")
            continue
        xv, yv    = x[valid], y[valid]
        log_ratio = np.log10(yv / xv)
        n_improved = int((yv > xv).sum())
        row = {
            'metric'             : label,
            'n_valid'            : n_valid,
            'median_before'      : float(np.median(xv)),
            'median_after'       : float(np.median(yv)),
            'mean_before'        : float(np.mean(xv)),
            'mean_after'         : float(np.mean(yv)),
            'median_log10_ratio' : float(np.median(log_ratio)),
            'mean_log10_ratio'   : float(np.mean(log_ratio)),
            'median_gain_factor' : float(10 ** np.median(log_ratio)),
            'pct_improved'       : 100.0 * n_improved / n_valid,
        }
        summary_rows.append(row)
        print(f"    {label:<16s}  median {row['median_before']:.2f} -> {row['median_after']:.2f}  "
              f"(x{row['median_gain_factor']:.2f} median gain)   "
              f"{row['pct_improved']:.1f}% of candidates improved   (n={n_valid})")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_csv_path = os.path.join(RUN_DIR, f"snr_improvement_summary_{STAMP}.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"  [SAVED] {summary_csv_path}")

    plot_snr_before_after(
        qc_df, _metric_pairs, _thresholds, rescued_col='passed_gate',
        run_dir=RUN_DIR, stamp=STAMP, event_type=_event_type_label,
    )
    plot_delta_snr_distribution(
        qc_df, _metric_pairs,
        run_dir=RUN_DIR, stamp=STAMP, event_type=_event_type_label,
    )
    plot_rescue_funnel(
        {"Denoised candidates": len(qc_df), "Passed quality gate": len(rescue_rows)},
        run_dir=RUN_DIR, stamp=STAMP,
        title=f"Rescue funnel — {_event_type_label}".strip(),
        fname=f"rescue_funnel_{STAMP}.png",
    )
    plot_denoise_fidelity(
        qc_df, corr_col='corr_signal',
        snr_before_col='SNR_s2n_median_raw', snr_after_col='SNR_s2n_median',
        rescued_col='passed_gate', run_dir=RUN_DIR, stamp=STAMP,
        event_type=_event_type_label, noise_corr_col='corr_noise',
    )
elif MAKE_QC_PLOTS:
    print("\n  [WARN] No candidates were evaluated — skipping QC plots.")


# =============================================================================
# SECTION 6c — WAVEFORM COMPARISON PLOTS (raw vs denoised, individual events)
# =============================================================================

if MAKE_WAVEFORM_PLOTS and qc_rows:
    print(f"\n{'='*65}")
    print("  STEP 6 — Waveform comparison plots (raw vs denoised)")
    print(f"{'='*65}")

    raw_col   = f"{WAVEFORM_RANK_METRIC}_raw"
    after_col = WAVEFORM_RANK_METRIC

    if raw_col not in qc_df.columns or after_col not in qc_df.columns:
        print(f"  [WARN] WAVEFORM_RANK_METRIC={WAVEFORM_RANK_METRIC!r} not found in QC columns — skipping.")
        selected = pd.DataFrame()
    elif WAVEFORM_SELECT_MODE == "manual":
        if not WAVEFORM_MANUAL_FNAMES:
            print("  [WARN] WAVEFORM_SELECT_MODE='manual' but WAVEFORM_MANUAL_FNAMES is empty — skipping.")
            selected = pd.DataFrame()
        else:
            selected = qc_df[qc_df['fname'].isin(WAVEFORM_MANUAL_FNAMES)]
            missing  = set(WAVEFORM_MANUAL_FNAMES) - set(selected['fname'])
            if missing:
                print(f"  [WARN] {len(missing)} requested fname(s) not found in this run: "
                      f"{sorted(missing)[:5]}")
    elif WAVEFORM_SELECT_MODE == "top_delta":
        valid = qc_df.dropna(subset=[raw_col, after_col]).copy()
        valid = valid[(valid[raw_col] > 0) & (valid[after_col] > 0)]
        if len(valid) == 0:
            print(f"  [WARN] No valid before/after pairs for {WAVEFORM_RANK_METRIC} — skipping.")
            selected = pd.DataFrame()
        else:
            valid['_delta'] = np.log10(valid[after_col] / valid[raw_col])
            rank_col = valid['_delta'].abs() if WAVEFORM_RANK_ABS else valid['_delta']
            top_idx  = rank_col.sort_values(ascending=False).index[:WAVEFORM_TOP_N]
            selected = valid.loc[top_idx]
            print(f"  Selecting top {len(selected)} candidates by "
                  f"{'|delta|' if WAVEFORM_RANK_ABS else 'delta'} of {WAVEFORM_RANK_METRIC} "
                  f"(log10 after/before)")
    else:
        print(f"  [WARN] Unknown WAVEFORM_SELECT_MODE={WAVEFORM_SELECT_MODE!r} — skipping.")
        selected = pd.DataFrame()

    if len(selected) == 0:
        print("  No events selected for waveform comparison plots.")
    else:
        for _, wrow in selected.iterrows():
            fname     = wrow['fname']
            den_path  = os.path.join(DENOISED_DIR, fname + ".npz")
            orig_path = os.path.join(RESCUE_DIR,   fname + ".npz")
            try:
                den   = np.load(den_path,  allow_pickle=True)
                orig  = np.load(orig_path, allow_pickle=True)
                w_denoised = den['data'][:, 0, 0].astype(np.float64)
                w_raw      = orig['data'][:, 2].astype(np.float64)
                w_itp      = int(orig['itp'])
            except Exception as e:
                print(f"  [WARN] Could not reload waveforms for {fname}: {e}")
                continue

            plot_waveform_comparison(
                w_raw, w_denoised, w_itp, SPS,
                run_dir=RUN_DIR, stamp=STAMP, fname_tag=fname,
                event_type=wrow.get('event_type', '') or '',
                snr_before=wrow.get(raw_col, np.nan),
                snr_after=wrow.get(after_col, np.nan),
                metric_label=WAVEFORM_RANK_METRIC,
                freqmin=WAVEFORM_FILTER_FREQMIN,
                freqmax=WAVEFORM_FILTER_FREQMAX,
            )
elif MAKE_WAVEFORM_PLOTS:
    print("\n  [WARN] No candidates were evaluated — skipping waveform comparison plots.")


# =============================================================================
# SECTION 6d — RESCUE CATALOG CSV (only rows that pass the quality gate)
# =============================================================================

if not rescue_rows:
    print("\n[WARN] No events passed the quality gate — rescue_catalog CSV not written. "
          "See denoise_qc_*.csv / QC plots above to check SNR thresholds or denoiser quality.")
    rescue_df = pd.DataFrame()
else:
    rescue_df = pd.DataFrame(rescue_rows)

    # Show event-level counts (some events have multiple station observations)
    n_events = rescue_df['event_time'].nunique()
    print(f"\n  Unique events rescued  : {n_events:,}")
    print(f"  Station-obs rows       : {len(rescue_df):,}  "
          f"(avg {len(rescue_df)/n_events:.1f} stations/event)")

    # Save
    out_csv = os.path.join(RUN_DIR, f"rescue_catalog_{STAMP}.csv")
    rescue_df.to_csv(out_csv, index=False)
    print(f"\n  [SAVED] {out_csv}")

print(f"\n{'='*65}")
print(f"  Run finished : {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Run folder   : {RUN_DIR}")
print(f"  Log          : {log_path}")
print(f"{'='*65}")

log_file.close()
