"""
05b_snr_quality_threshold.py
=============================
ISTerre internship
Author : Elsa Louis
Date   : July 2026

Scope
-----
Companion to 05a_snr_windowing_validation.py
 -> 05a answers "does SNR predict whether the detector correctly bracketed the true onset"
 -> 05b answers a different question: "what SNR level predicts actual downstream usefulness for classification / DeepDenoiser training"

Both tiers below are restricted to WELL-ALIGNED detections only 
 -> alignment is treated as a hard, separate prerequisite here, not something SNR is asked to predict
 -> this isolates the amplitude/quality question from the timing question

Tier 1 — Unsupervised (no ground truth needed)
------------------------------------------------
Within the aligned population, look for a natural split in the SNR distribution between "real event" and "noise trigger" populations:
  - 2-component Gaussian mixture fit on log10(SNR); candidate threshold = the crossover point between the two fitted components
  - Otsu's method (maximizes between-class variance on the histogram) cheap, no labels, no classifier training required

Tier 2 — Classification-correctness-based (supervised)
----------------------------------------------------------
06c only ever trains/evaluates on rows that already pass the CURRENT SNR gate, so its output can't tell us whether that gate is in the right place 
 -> this tier trains a fresh, INDEPENDENT HGB classifier (same hyperparameters as 06c, for rough comparability) on the aligned population WITHOUT any SNR-based filtering
 -> performance across the FULL amplitude range (down to near-noise-level) is observed

Input
-----
  catalog_windows_<stamp>.csv           (output of script 04a)
  feature_importances_<stamp>.csv       (output of script 03b, optional)
  05a's snr_summary_<stamp>.csv         (optional — reference line only)

Output
------
  quality_threshold_tier1_<stamp>.csv   : per-metric GMM/Otsu/reference thresholds
  quality_threshold_tier2_<stamp>.csv   : per-metric AUC/threshold vs classification correctness
  fig_quality_threshold_<stamp>.png     : Tier 1 histogram + GMM/Otsu/reference overlay
  fig_roc_quality_<stamp>.png           : Tier 2 ROC curves (correctness vs SNR)
"""


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# -- Input CSV (output of script 04a) — same catalog used by 05a, so Tier 1 and Tier 2 both operate on the identical population/snapshot --
INPUT_CSV  = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\04a_spectrogram_sta_lta_catalog\all-99-features-recent\catalog_windows_20260707_165719.csv"

OUTPUT_DIR = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\05b_snr_quality_threshold"

# -- Alignment pre-filter — both tiers only look at well-aligned rows ---------
GROUND_TRUTH = 'pick_inside_det'   # origin_inside_det or pick_inside_det

# -- SNR metrics to evaluate ----------------------------------------------------
SNR_METRICS = [
    'SNR',
    'SNR_picking_5_5',
    'SNR_picking_3_3',
    'SNR_picking_1_3',
    'SNR_full_mean',
    'SNR_full_median',
    'SNR_s2n_median',
]
SNR_SHORT = {
    'SNR'              : 'SNR\n(peak/noise)',
    'SNR_picking_5_5'  : 'pick_5-5\n(±5s)',
    'SNR_picking_3_3'  : 'pick_3-3\n(±3s)',
    'SNR_picking_1_3'  : 'pick_1-3\n(1s/3s)',
    'SNR_full_mean'    : 'full_mean',
    'SNR_full_median'  : 'full_median',
    'SNR_s2n_median'   : 's2n_median\n(MAD noise)',
}
SNR_LONG = {
    'SNR'              : 'SNR  (peak-centred 5s / 5s post-event noise)',
    'SNR_picking_5_5'  : 'SNR_picking_5-5  (5s after onset / 5s before onset)',
    'SNR_picking_3_3'  : 'SNR_picking_3-3  (3s after onset / 3s before onset)',
    'SNR_picking_1_3'  : 'SNR_picking_1-3  (1s after onset / 3s before onset)',
    'SNR_full_mean'    : 'SNR_full_mean  (mean envelope: signal window / noise window)',
    'SNR_full_median'  : 'SNR_full_median  (median envelope: signal window / noise window)',
    'SNR_s2n_median'   : 'SNR_s2n_median  (99.5th percentile signal / MAD noise)',
}

# -- Reference: current pooled windowing-validation threshold (05a output), shown on the Tier 1 plot for comparison ONLY (None to skip it)
REFERENCE_SNR_SUMMARY_CSV = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\05a_snr_windowing_validation\pick_inside_det\run_20260710_141258\snr_summary_20260710_141258.csv"

# -- Tier 1: unsupervised GMM / Otsu ---------------------------------------------
GMM_RANDOM_STATE = 42

# -- Tier 2: classifier (mirrors 06c's hyperparameters for rough comparability) ----------------
RUN_TIER_2               = True
TARGET_CLASSES           = ["earthquake", "rockslide", "ice quake"]
FEATURE_IMPORTANCES_CSV  = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\03b_feature_selection\run_20260709_145058\feature_importances_20260709_145058.csv"
TOP_N_FEATURES           = 60      # None -> use every feature column present
TEST_SIZE                = 0.20
RANDOM_STATE              = 42
SMOTE_K                   = 5
HGB_N_EST                 = 200
HGB_MAX_DEPTH             = 6
HGB_LR                    = 0.1



# =============================================================================
# SECTION 2 — SETUP
# =============================================================================

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')

from sklearn.mixture import GaussianMixture
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_curve, auc as sklearn_auc, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
from features import FEATURE_NAMES, FEATURE_NAMES_3C, rename_legacy_columns
from run_setup import create_run_dir, setup_logging, set_matplotlib_defaults
from visualization import plot_snr_quality_threshold, plot_roc_pooled

RUN_DIR, STAMP = create_run_dir(OUTPUT_DIR)
log_file, log_path = setup_logging(
    RUN_DIR, "05b_snr_quality_threshold.py",
    extra_info=f"Alignment ground truth: {GROUND_TRUTH} | Tier 2: {RUN_TIER_2}",
)
set_matplotlib_defaults()



# =============================================================================
# SECTION 3 — LOAD CATALOG + ALIGNMENT PRE-FILTER
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 1 — Load catalog, keep well-aligned rows only")
print(f"{'='*65}")

df_all = pd.read_csv(INPUT_CSV, low_memory=False)
print(f"Loaded {len(df_all):,} rows x {df_all.shape[1]} columns.")

SNR_METRICS = [m for m in SNR_METRICS if m in df_all.columns]
if not SNR_METRICS:
    print("[ERROR] None of the expected SNR columns found in the CSV. Exiting.")
    sys.exit(1)

if GROUND_TRUTH not in df_all.columns:
    print(f"[ERROR] '{GROUND_TRUTH}' column not found. Exiting.")
    sys.exit(1)

# Alignment is a hard prerequisite here, not an optimization target 
df_aligned = df_all[df_all[GROUND_TRUTH] == True].copy()
print(f"  Well-aligned rows ({GROUND_TRUTH} == True) : {len(df_aligned):,} / {len(df_all):,}"
      f"  ({100*len(df_aligned)/len(df_all):.1f}%)")

for et, cnt in df_aligned['event_type'].value_counts().items():
    print(f"    {et:<22s} : {cnt}")

# Reference threshold (05a pooled, windowing-validation) — comparison line only
reference_thresholds = {}
if REFERENCE_SNR_SUMMARY_CSV and os.path.isfile(REFERENCE_SNR_SUMMARY_CSV):
    _ref_df = pd.read_csv(REFERENCE_SNR_SUMMARY_CSV)
    if {'metric', 'best_threshold'}.issubset(_ref_df.columns):
        reference_thresholds = dict(zip(_ref_df['metric'], _ref_df['best_threshold']))
        print(f"\n  Loaded reference (05a windowing) thresholds from: {REFERENCE_SNR_SUMMARY_CSV}")
    else:
        print(f"\n  [WARN] {REFERENCE_SNR_SUMMARY_CSV} missing expected columns — skipping reference line.")
else:
    print("\n  [INFO] No REFERENCE_SNR_SUMMARY_CSV set/found — Tier 1 plot will skip the comparison line.")



# =============================================================================
# SECTION 4 — HELPER FUNCTIONS (Otsu, GMM crossover)
# =============================================================================

def otsu_threshold(values, n_bins=256):
    """
    Otsu's method: find the threshold that maximizes between-class variance of a histogram 
     -> a standard unsupervised bimodal-split finder, no label or distributional assumption required (unlike the GMM fit above it)

    Parameters
    ----------
    values : 1-D array-like — already in the space you want the threshold in (here, log10(SNR))
    n_bins : int — histogram resolution

    Returns
    -------
    threshold : float or np.nan if the input has no spread
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or np.ptp(values) == 0:
        return np.nan

    hist, bin_edges = np.histogram(values, bins=n_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    hist = hist.astype(float)
    total = hist.sum()
    sum_total = (hist * bin_centers).sum()

    w0 = 0.0
    sum0 = 0.0
    best_var = -1.0
    best_thr = bin_centers[0]
    for i in range(len(hist)):
        w0 += hist[i]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        sum0 += hist[i] * bin_centers[i]
        mean0 = sum0 / w0
        mean1 = (sum_total - sum0) / w1
        between_var = w0 * w1 * (mean0 - mean1) ** 2
        if between_var > best_var:
            best_var = between_var
            best_thr = bin_centers[i]
    return float(best_thr)


def _gauss_pdf(x, mean, std):
    std = max(std, 1e-9)
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


def gmm_crossover(means, stds, weights):
    """
    Crossover point (in the same units as `means`) between two fitted Gaussian components 
     -> the point in the valley between the two peaks where neither component dominates
     -> returns None if the components don't actually cross between their means (e.g. heavily overlapping / one totally dominates)
    """
    order = np.argsort(means)
    m = np.asarray(means)[order]
    s = np.asarray(stds)[order]
    w = np.asarray(weights)[order]
    if m[0] == m[1]:
        return None
    xs = np.linspace(m[0], m[1], 2000)
    diff = w[0] * _gauss_pdf(xs, m[0], s[0]) - w[1] * _gauss_pdf(xs, m[1], s[1])
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_changes) == 0:
        return None
    return float(xs[sign_changes[0]])



# =============================================================================
# SECTION 5 — TIER 1: UNSUPERVISED GMM / OTSU
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 2 — Tier 1: unsupervised quality threshold (GMM + Otsu)")
print(f"{'='*65}")

gmm_params_by_metric  = {}
thresholds_by_metric  = {}
tier1_rows            = []

for metric in SNR_METRICS:
    x = df_aligned[metric].dropna()
    x = x[x > 0]
    if len(x) < 20:
        print(f"  {metric:<18s}  SKIP — too few valid values (n={len(x)})")
        continue

    logx = np.log10(x).values.reshape(-1, 1)

    gmm = GaussianMixture(n_components=2, random_state=GMM_RANDOM_STATE, n_init=3)
    gmm.fit(logx)
    means   = gmm.means_.flatten()
    stds    = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()

    gmm_params_by_metric[metric] = {
        'means': tuple(means), 'stds': tuple(stds), 'weights': tuple(weights),
    }

    crossover_log = gmm_crossover(means, stds, weights)
    gmm_thr = float(10 ** crossover_log) if crossover_log is not None else None

    otsu_log = otsu_threshold(logx.flatten())
    otsu_thr = float(10 ** otsu_log) if np.isfinite(otsu_log) else None

    ref_thr = reference_thresholds.get(metric)

    thresholds_by_metric[metric] = {
        'GMM crossover'          : gmm_thr,
        'Otsu'                   : otsu_thr,
        '05a pooled (windowing)' : ref_thr,
    }

    print(f"  {metric:<18s}  GMM crossover={gmm_thr if gmm_thr is None else f'{gmm_thr:.2f}'}   "
          f"Otsu={otsu_thr if otsu_thr is None else f'{otsu_thr:.2f}'}   "
          f"05a pooled={ref_thr if ref_thr is None else f'{ref_thr:.2f}'}   (n={len(x)})")

    tier1_rows.append({
        'metric': metric, 'n_valid': len(x),
        'gmm_crossover_threshold': gmm_thr,
        'otsu_threshold': otsu_thr,
        'reference_windowing_threshold': ref_thr,
        'gmm_vs_reference_delta': (round(gmm_thr - ref_thr, 3)
                                    if gmm_thr is not None and ref_thr is not None else np.nan),
    })

df_tier1 = pd.DataFrame(tier1_rows)
tier1_csv = os.path.join(RUN_DIR, f"quality_threshold_tier1_{STAMP}.csv")
df_tier1.to_csv(tier1_csv, index=False)
print(f"\n[SAVED] {tier1_csv}")

plot_snr_quality_threshold(
    df_aligned, SNR_METRICS, SNR_LONG, gmm_params_by_metric, thresholds_by_metric,
    RUN_DIR, STAMP, event_type="all types, well-aligned only",
)



# =============================================================================
# SECTION 6 — TIER 2: CLASSIFICATION-CORRECTNESS-BASED THRESHOLD
# =============================================================================

tier2_csv = None   # stays None if Tier 2 is skipped/disabled

if not RUN_TIER_2:
    print("\n[INFO] RUN_TIER_2 = False — skipping Tier 2.")
else:
    print(f"\n{'='*65}")
    print("  STEP 3 — Tier 2: train a wide-SNR-range classifier, no changes to 06c")
    print(f"{'='*65}")

    df_cls = df_aligned[df_aligned["event_type"].isin(TARGET_CLASSES)].copy()
    df_cls = rename_legacy_columns(df_cls)
    print(f"  Aligned + target-class rows (NO SNR gate applied) : {len(df_cls):,}")
    for cls in TARGET_CLASSES:
        n = (df_cls["event_type"] == cls).sum()
        print(f"    {cls:<14s} : {n:>6,}  ({100*n/max(len(df_cls),1):.1f}%)")

    # ── Feature selection — mirrors 06c's logic exactly, independently ──────────
    if TOP_N_FEATURES is None:
        features = [f for f in FEATURE_NAMES_3C if f in df_cls.columns]
        if not features:
            features = [f for f in FEATURE_NAMES if f in df_cls.columns]
        print(f"  TOP_N_FEATURES=None -> using all {len(features)} feature columns found.")
    elif FEATURE_IMPORTANCES_CSV is not None and os.path.exists(FEATURE_IMPORTANCES_CSV):
        _imp_df  = pd.read_csv(FEATURE_IMPORTANCES_CSV)
        features = _imp_df["feature"].head(TOP_N_FEATURES).tolist()
        features = [f for f in features if f in df_cls.columns]
        print(f"  Loaded Top-{TOP_N_FEATURES} features from: {FEATURE_IMPORTANCES_CSV}  ({len(features)} usable)")
    else:
        features = [f for f in FEATURE_NAMES if f in df_cls.columns][:TOP_N_FEATURES]
        print(f"  [WARN] Feature importances CSV unavailable — using first {len(features)} FEATURE_NAMES columns.")

    df_cls = df_cls.dropna(subset=features).copy()
    print(f"  Rows after dropping NaN features : {len(df_cls):,}")

    # ── Event-level train/test split + SMOTE + HGB — mirrors 06c's train_and_eval,
    # ── kept as an independent copy here so 06c itself is never touched ──────────
    events = df_cls[["event_time", "event_type"]].drop_duplicates("event_time")
    class_counts = events["event_type"].value_counts()
    min_class_size = class_counts.min() if len(class_counts) else 0
    effective_k = min(SMOTE_K, min_class_size - 1)

    if effective_k < 1:
        print(f"  [WARN] Smallest class has {min_class_size} events at the event level — "
              f"cannot split/SMOTE. Skipping Tier 2.")
    else:
        train_ev, test_ev = train_test_split(
            events["event_time"], test_size=TEST_SIZE,
            stratify=events["event_type"], random_state=RANDOM_STATE,
        )
        train_mask = df_cls["event_time"].isin(train_ev)
        test_mask  = df_cls["event_time"].isin(test_ev)
        print(f"  Train: {train_mask.sum():,} rows  |  Test: {test_mask.sum():,} rows"
              f"  (split at event level, {len(train_ev):,}/{len(test_ev):,} events)")

        X_tr_raw = df_cls.loc[train_mask, features].values
        y_tr_raw = df_cls.loc[train_mask, "event_type"].values
        X_te     = df_cls.loc[test_mask,  features].values
        y_te     = df_cls.loc[test_mask,  "event_type"].values

        imp = SimpleImputer(strategy="median")
        X_tr_raw = imp.fit_transform(X_tr_raw)
        X_te     = imp.transform(X_te)

        sm = SMOTE(k_neighbors=effective_k, random_state=RANDOM_STATE)
        X_tr, y_tr = sm.fit_resample(X_tr_raw, y_tr_raw)
        print(f"  After SMOTE: {len(X_tr):,} training rows")

        model = HistGradientBoostingClassifier(
            max_iter=HGB_N_EST, max_depth=HGB_MAX_DEPTH, learning_rate=HGB_LR,
            early_stopping=True, n_iter_no_change=15, random_state=RANDOM_STATE,
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        acc      = accuracy_score(y_te, y_pred)
        macro_f1 = f1_score(y_te, y_pred, average='macro')
        print(f"  Wide-range classifier check: accuracy={acc:.3f}  macro F1={macro_f1:.3f}")
        print(f"  (For reference only — this is NOT the production classifier and is trained\n"
              f"   on a much wider, ungated SNR range on purpose. Expect lower scores than 06c.)")

        # ── Build the "correctly classified" ground truth on the TEST rows only ──
        test_idx    = df_cls.index[test_mask]
        correct     = (y_pred == y_te)
        df_quality  = df_cls.loc[test_idx].copy()
        df_quality['label'] = correct

        n_correct = int(correct.sum())
        print(f"\n  Test-set correctness: {n_correct:,}/{len(correct):,} correct "
              f"({100*n_correct/len(correct):.1f}%)")

        # ── ROC / Youden threshold per SNR metric, using correctness as ground truth ──
        print("\n--- Tier 2: ROC AUC per metric, vs classification correctness ---")

        roc_results_quality = {}
        tier2_rows = []
        for metric in SNR_METRICS:
            valid = df_quality[['label', metric]].dropna()
            if valid['label'].nunique() < 2:
                print(f"  {metric:<18s}  SKIP — only one class in this subset")
                continue

            fpr, tpr, thresholds = roc_curve(valid['label'].astype(int), valid[metric])
            auc_val  = sklearn_auc(fpr, tpr)
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)

            roc_results_quality[metric] = {
                'fpr': fpr, 'tpr': tpr, 'auc': float(auc_val),
                'youden_threshold': float(thresholds[best_idx]),
                'youden_tpr': float(tpr[best_idx]),
                'youden_fpr': float(fpr[best_idx]),
            }
            r = roc_results_quality[metric]
            print(f"  {metric:<18s}  AUC={r['auc']:.3f}  best threshold={r['youden_threshold']:.2f}  "
                  f"-> TPR={r['youden_tpr']:.2f}  FPR={r['youden_fpr']:.2f}  (n={len(valid):,})")

            ref_thr = reference_thresholds.get(metric)
            tier2_rows.append({
                'metric': metric, 'n_valid': len(valid),
                'auc': round(r['auc'], 4),
                'best_threshold': round(r['youden_threshold'], 3),
                'tpr_at_best_thr': round(r['youden_tpr'], 3),
                'fpr_at_best_thr': round(r['youden_fpr'], 3),
                'reference_windowing_threshold': ref_thr,
            })

        df_tier2 = pd.DataFrame(tier2_rows).sort_values('auc', ascending=False, na_position='last')
        tier2_csv = os.path.join(RUN_DIR, f"quality_threshold_tier2_{STAMP}.csv")
        df_tier2.to_csv(tier2_csv, index=False)
        print(f"\n[SAVED] {tier2_csv}")

        if roc_results_quality:
            plot_roc_pooled(
                roc_results_quality, SNR_METRICS, SNR_SHORT, RUN_DIR, STAMP,
                title='ROC curves — SNR vs. classification correctness (wide SNR range, no gate)',
                subtitle='Diamond = Youden-optimal threshold  |  higher AUC = SNR more predictive of getting the class right',
                fname=f"fig_roc_quality_{STAMP}.png",
            )
        else:
            print("  [SKIP] No Tier 2 ROC results to plot.")



# =============================================================================
# SECTION 7 — SUMMARY
# =============================================================================

print(f"\n{'='*65}")
print("  SUMMARY")
print(f"{'='*65}")
print("  Compare candidate thresholds below — none is automatically \"correct\"; the")
print("  point is to see whether Tier 1 (unsupervised), Tier 2 (classification-based),")
print("  and the current pooled 05a (windowing) threshold roughly agree or diverge.")
print(f"\n  Tier 1 (GMM / Otsu)         : {tier1_csv}")
if tier2_csv is not None:
    print(f"  Tier 2 (classification ROC) : {tier2_csv}")
print(f"\n  Run folder : {RUN_DIR}")
print(f"  Log        : {log_path}")
print(f"{'='*65}")

log_file.close()
