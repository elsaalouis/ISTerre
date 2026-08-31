"""
06b_compare_classifiers.py
==========================
ISTerre internship
Author : Elsa Louis
Date   : May 2026

Goal
----
Compare 5 classifiers on the same event-stratified train/test split:
  - Random Forest            (baseline, same config as 06a)
  - Hist. Gradient Boost HGB (sklearn HistGradientBoostingClassifier — same algorithm as XGBoost)
  - KNeighbors
  - SVM with RBF kernel      subsampled for cluster speed (see SVM_MAX_ROWS)
  - MLP                      (multi-layer perceptron)

Class balance: SMOTE on training set (same k and seed as 06a)
Evaluation   : per-class F1 / precision / recall, macro F1, accuracy, training time

Outputs
-------
  fig_confusion_<short>_<stamp>.png : normalised confusion matrix per classifier
  fig_comparison_<stamp>.png        : 3-panel unified comparison (per-class F1, macro F1, training time)
  comparison_results_<stamp>.csv    : full metrics table
"""


# =============================================================================
# SECTION 1 — CONFIGURATION
# All user-facing parameters are here
# =============================================================================

# -- Input CSV (output of script 04a) -----------------------------------------
CSV_PATH   = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\04a_spectrogram_sta_lta_catalog\all-99-features-recent+3C\catalog_windows_20260819_171211.csv"

# -- Noise CSV (output of script 04d, optional 4th class) ----------------------
# Set to a 04d `noise_windows_<stamp>.csv` to add the "noise" class
NOISE_CSV  = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\04d_noise_window_extraction\run_20260803_174514\noise_windows_20260803_174514.csv"

# -- Regional CSV (output of script 04c, optional 5th class) -------------------
# Set to a 04c `regional_windows_<stamp>.csv` to add the "regional" class
REGIONAL_CSV = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\04c_regional_EQ_extraction\run_20260805_135512\regional_windows_20260805_135512.csv"

# -- Output directory ----------------------------------------------------------
OUTPUT_DIR = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\06b_compare_classifiers"

# -- Classes -------------------------------------------------------------------
TARGET_CLASSES = ["earthquake", "regional", "rockslide", "ice quake", "noise"]
CLASS_ORDER    = ["earthquake", "regional", "rockslide", "ice quake", "noise"]   # table / figure order
CLASS_ABBR     = {"earthquake": "eq", "regional": "re", "rockslide": "rs", "ice quake": "iq", "noise": "no"}

# -- Feature set ---------------------------------------------------------------
# TOP_N_FEATURES = None  → use ALL feature columns present in the catalog CSV
# TOP_N_FEATURES = int   → use only the top-N features, ranked by:
#                            · FEATURE_IMPORTANCES_CSV if provided
#                            · FALLBACK_TOP20 otherwise (hardcoded list below)
FEATURE_IMPORTANCES_CSV = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\03b_feature_selection\run_20260709_145058\feature_importances_20260709_145058.csv"   # or None
TOP_N_FEATURES          = 60   # None → all features; int → top-N

# Hardcoded Top-20 fallback
FALLBACK_TOP20 = [
    "duration",                "ediff_3_10__10_20",   "eratio_3_10__10_20",
    "spec_kurtosis_median_env","kurtosis_10_20Hz",    "kurtosis_3_10Hz",
    "kurtosis_1_8Hz",          "fft_energy_1_nyq4",   "kurtosis_1_3Hz",
    "energy_1_3Hz",            "fft_freq_at_max",     "fft_spread_peaks",
    "dist_q3_q1",              "eratio_1_3__3_10",    "eratio_0.1_1__1_3",
    "fft_n_peaks",             "ediff_1_3__10_20",    "ascend_descend_ratio",
    "kurtosis_20_nyq",         "ediff_1_3__3_10",
]

# -- Quality gate (05b Tier 2 classification-based) ----------------------
SNR_MIN             = 1.70    # 05b Tier 2 — metric 'SNR', AUC=0.627
SNR_FULL_MEDIAN_MIN = 1.99    # 05b Tier 2 — metric 'SNR_full_median', AUC=0.642 (best)

# -- Train / test split --------------------------------------------------------
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# -- SMOTE ---------------------------------------------------------------------
SMOTE_K = 5

# ── Earthquake over-prediction rebalancing (optional experiment) ──────────────
EARTHQUAKE_REBALANCE_MODE    = "sample_weight"   # None | "undersample" | "sample_weight"
EARTHQUAKE_UNDERSAMPLE_RATIO = 2.0    # used only if mode == "undersample"
EARTHQUAKE_SAMPLE_WEIGHT     = 0.5    # used only if mode == "sample_weight"

# -- Classifier hyperparameters ------------------------------------------------

# Random Forest
RF_N_EST      = 200          # nb of trees, more trees = more stable predictions bc average out more random variance

# Histogram Gradient Boosting  (sklearn built-in, same algorithm as XGBoost hist)
HGB_N_EST     = 200          # max number of boosting iterations (trees)
HGB_MAX_DEPTH = 6            # max depth of each individual tree -> depth-6 tree can make 2⁶=64 distinct predictions
HGB_LR        = 0.1          # learning rate -> lower = more conservative steps = needs more iterations to converge but generalises better

# K-Nearest Neighbours
KNN_K         = 7            # number of neighbours

# SVM (RBF)
SVM_C         = 10.0         # regularisation strength -> maximise the margin between classes (want C small) vs. minimise classification errors on the training set (want C large)
SVM_MAX_ROWS  = 10_000       # max training rows (subsampled if exceeded)
                              # Set to None to use the full SMOTE set (slooooow)

# MLP
MLP_HIDDEN    = (128, 64)    # hidden layer sizes -> first processes the 20 input features into 128 intermediate representations, the second compresses them into 64, and the output layer maps to 3 class probabilities
MLP_MAX_ITER  = 300          # max epochs (early stopping is also enabled)



# =============================================================================
# SECTION 2 — SETUP
# =============================================================================

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# Add project src directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FEATURE_NAMES, FEATURE_NAMES_3C, rename_legacy_columns
from run_setup import create_run_dir, setup_logging

warnings.filterwarnings("ignore")

# create timestamped run folder, then redirect stdout → terminal + run.log
RUN_DIR, STAMP = create_run_dir(OUTPUT_DIR)
log_file, log_path = setup_logging(
    RUN_DIR,
    script_name="06b_compare_classifiers.py",
    extra_info=f"CSV: {CSV_PATH}\nEARTHQUAKE_REBALANCE_MODE: {EARTHQUAKE_REBALANCE_MODE}",
)



# =============================================================================
# SECTION 3 — LOAD AND FILTER DATA  (same logic as 06a)
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 1 — Loading CSV")
print(f"{'='*65}")

df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"Loaded {len(df):,} rows × {len(df.columns)} columns.")

# Rename legacy feat_XX columns to descriptive names if needed
df = rename_legacy_columns(df)

# Class filter
df = df[df["event_type"].isin(TARGET_CLASSES)].copy()
print(f"After class filter {TARGET_CLASSES}: {len(df):,} rows.")

# -- Optional 5th class: regional (output of 04c) -----------------
if REGIONAL_CSV is not None:
    if os.path.isfile(REGIONAL_CSV):
        df_regional = pd.read_csv(REGIONAL_CSV, low_memory=False)
        df_regional = rename_legacy_columns(df_regional)
        df_regional = df_regional[df_regional["event_type"].isin(TARGET_CLASSES)].copy()
        print(f"Loaded {len(df_regional):,} regional rows from {os.path.basename(REGIONAL_CSV)}.")
        df = pd.concat([df, df_regional], ignore_index=True)
    else:
        print(f"[WARN] REGIONAL_CSV not found: {REGIONAL_CSV} — continuing without the regional class.")

# Quality gate: always recompute explicitly from SNR / SNR_full_median
# NOTE: do NOT trust the precomputed 'quality_ok' catalog column, it was baked by 04a using the OLD 05a thresholds (SNR_full_mean/SNR_s2n_median)
mask_quality = (
    (df["SNR"]             >= SNR_MIN) &
    (df["SNR_full_median"] >= SNR_FULL_MEDIAN_MIN)
)
df = df[mask_quality].copy()
print(f"After quality filter: {len(df):,} rows kept.")

# Drop rows where any Z-component feature is NaN
z_feat_cols = [f for f in FEATURE_NAMES if f in df.columns]
df = df.dropna(subset=z_feat_cols).copy()
print(f"After NaN drop (Z features): {len(df):,} rows.")

# -- Optional 4th class: noise (output of 04d), added AFTER the quality gate --
if NOISE_CSV is not None:
    if os.path.isfile(NOISE_CSV):
        df_noise = pd.read_csv(NOISE_CSV, low_memory=False)
        df_noise = rename_legacy_columns(df_noise)
        z_feat_cols_n = [f for f in FEATURE_NAMES if f in df_noise.columns]
        df_noise = df_noise.dropna(subset=z_feat_cols_n).copy()
        print(f"Loaded {len(df_noise):,} noise rows from {os.path.basename(NOISE_CSV)}.")
        df = pd.concat([df, df_noise], ignore_index=True)
    else:
        print(f"[WARN] NOISE_CSV not found: {NOISE_CSV} — continuing without the noise class.")

print("\n  CLASS DISTRIBUTION")
print("  " + "─" * 45)
for cls in CLASS_ORDER:
    n = (df["event_type"] == cls).sum()
    print(f"  {cls:<22} {n:>6,}  ({100 * n / len(df):.1f} %)")



# =============================================================================
# SECTION 4 — SELECT FEATURES
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 2 — Feature selection")
print(f"{'='*65}")

if TOP_N_FEATURES is None:
    features = [f for f in FEATURE_NAMES_3C if f in df.columns] # use every feature column present in the catalog
    if not features:
        # Fallback: catalog may only have the 99 Z features
        features = [f for f in FEATURE_NAMES if f in df.columns]
    print(f"  TOP_N_FEATURES=None → using all {len(features)} feature columns found in catalog.")
elif FEATURE_IMPORTANCES_CSV is not None:
    imp_df   = pd.read_csv(FEATURE_IMPORTANCES_CSV)
    features = imp_df["feature"].head(TOP_N_FEATURES).tolist()
    print(f"  Loaded Top-{TOP_N_FEATURES} features from: {FEATURE_IMPORTANCES_CSV}")
else:
    features = list(FALLBACK_TOP20[:TOP_N_FEATURES])
    print(f"  Using hardcoded Top-{TOP_N_FEATURES} fallback list ({len(features)} features).")

# Sanity check
missing = [f for f in features if f not in df.columns]
if missing:
    raise ValueError(f"Features missing from CSV: {missing}")

print(f"  n_features = {len(features)}")
for i, f in enumerate(features, 1):
    print(f"    {i:>2}. {f}")



# =============================================================================
# SECTION 5 — TRAIN / TEST SPLIT  +  SMOTE
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 3 — Train / test split  (by event)  +  SMOTE")
print(f"{'='*65}")

# Split on unique events (not rows) → prevents the same event appearing in both train and test
events = df[["event_time", "event_type"]].drop_duplicates("event_time")
train_events, test_events = train_test_split(
    events["event_time"],
    test_size=TEST_SIZE,
    stratify=events["event_type"],
    random_state=RANDOM_STATE,
)
train_mask = df["event_time"].isin(train_events)
test_mask  = df["event_time"].isin(test_events)

# -- Optional earthquake row-level undersampling (mirrors 06c) ---------------
if EARTHQUAKE_REBALANCE_MODE == "undersample":
    train_types  = df.loc[train_mask, "event_type"]
    eq_train_idx = train_types.index[train_types == "earthquake"]
    other_row_counts = train_types[
        ~train_types.isin(["earthquake", "noise"])
    ].value_counts()
    if len(eq_train_idx) > 0 and len(other_row_counts) > 0:
        target_n = int(round(EARTHQUAKE_UNDERSAMPLE_RATIO * other_row_counts.max()))
        if len(eq_train_idx) > target_n >= 1:
            _rng     = np.random.RandomState(RANDOM_STATE)
            keep_idx = _rng.choice(eq_train_idx, size=target_n, replace=False)
            drop_idx = eq_train_idx.difference(keep_idx)
            train_mask.loc[drop_idx] = False
            print(f"  [REBALANCE] earthquake training ROWS "
                  f"undersampled {len(eq_train_idx):,} -> {target_n:,} "
                  f"({EARTHQUAKE_UNDERSAMPLE_RATIO}x next-largest real "
                  f"training class by rows = {other_row_counts.max():,} "
                  f"[{other_row_counts.idxmax()}] rows)")

X_train_raw = df.loc[train_mask, features].values
y_train_raw = df.loc[train_mask, "event_type"].values
X_test      = df.loc[test_mask,  features].values
y_test      = df.loc[test_mask,  "event_type"].values

print(f"Train : {train_mask.sum():,} rows  ({len(train_events):,} events)")
print(f"Test  : {test_mask.sum():,} rows  ({len(test_events):,} events)")

# Impute NaN with column median 
nan_cols = [f for i, f in enumerate(features) if np.isnan(X_train_raw[:, i]).any()]
if nan_cols:
    print(f"\n  [NaN] {len(nan_cols)} feature(s) contain NaN → imputing with training-set median:")
    for c in nan_cols:
        print(f"    · {c}")
imputer = SimpleImputer(strategy="median")
X_train_raw = imputer.fit_transform(X_train_raw)
X_test      = imputer.transform(X_test)

print(f"\nApplying SMOTE (k_neighbors={SMOTE_K}) on training set ...")
sm = SMOTE(k_neighbors=SMOTE_K, random_state=RANDOM_STATE)
X_train, y_train = sm.fit_resample(X_train_raw, y_train_raw)
print(f"After SMOTE: {len(X_train):,} rows")
for cls in CLASS_ORDER:
    n = (y_train == cls).sum()
    print(f"  {cls:<22} {n:>6,}")

# -- Optional earthquake sample_weight down-weighting (mirrors 06c) ----------
sample_weight_train = None
if EARTHQUAKE_REBALANCE_MODE == "sample_weight":
    sample_weight_train = np.where(y_train == "earthquake", EARTHQUAKE_SAMPLE_WEIGHT, 1.0)
    print(f"  [REBALANCE] earthquake sample_weight={EARTHQUAKE_SAMPLE_WEIGHT} vs 1.0 for other classes "
          f"({int((y_train == 'earthquake').sum()):,} / {len(y_train):,} training rows affected)")



# =============================================================================
# SECTION 6 — DEFINE CLASSIFIERS
# =============================================================================
#
# Each entry in CLASSIFIER_CONFIGS is a dict with:
#   name          : used in logs and figure titles
#   short         : short code used in file names
#   model         : sklearn-compatible estimator
#   needs_scaling : True  → fit StandardScaler on X_train, transform X_test
#                   False → use raw features (RF and XGBoost are scale-invariant)
#   svm_subsample : True  → subsample training rows to SVM_MAX_ROWS before fit
#   color         : bar colour in the comparison figure
#   skip          : True  → skip this classifier (used when package unavailable)
# =============================================================================

CLASSIFIER_CONFIGS = [

    {   # ── Random Forest ── baseline identical to 06a ──────────────────────
        "name":          "Random Forest",
        "short":         "RF",
        "model":         RandomForestClassifier(
                             n_estimators=RF_N_EST,     # nb of trees, more trees = more stable predictions bc average out more random variance
                             max_features="sqrt",       # at each node split, the tree only considers a random subset of √20 ≈ 4 features per node -> diversity between trees
                             class_weight="balanced",   # automatically weights each class inversely proportional to its frequency
                             random_state=RANDOM_STATE, # makes the bootstrap sampling and feature selection deterministic
                             n_jobs=-1,     # use all available CPU cores in parallel
                         ),
        "needs_scaling": False,
        "svm_subsample": False,
        "color":         "#1f77b4",
        "skip":          False,
        "supports_sample_weight": True,
    },

    {   # ── Histogram Gradient Boosting ─────────────────────────────────────
        "name":          "Hist. GradBoost",
        "short":         "HGB",
        "model":         HistGradientBoostingClassifier(
                             max_iter=HGB_N_EST,      # maximum number of boosting rounds (trees)
                             max_depth=HGB_MAX_DEPTH, # max depth of each individual tree
                             learning_rate=HGB_LR,    # learning rate
                             early_stopping=True,     # stops training when it stops improving, saving time on the cluster
                             n_iter_no_change=15,     # patience before stopping -> 15 rounds conservative enough to not stop prematurely on a temporary plateau
                             random_state=RANDOM_STATE,
                         ),
        "needs_scaling": False,
        "svm_subsample": False,
        "color":         "#ff7f0e",
        "skip":          False,
        "supports_sample_weight": True,
    },

    {   # ── K-Nearest Neighbours ────────────────────────────────────────────
        "name":          "KNeighbors",
        "short":         "KNN",
        "model":         KNeighborsClassifier(
                             n_neighbors=KNN_K,   # number of neighbours
                             weights="distance",  # closer neighbours vote more strongly -> weight of 1/distance (instead of giving each of the neighbours one equal vote)
                             n_jobs=-1,           # use all available CPU cores in parallel
                         ),
        "needs_scaling": True,   # distance meaningful only on scaled features (otherwise a large-scale feature dominates the metric, e.g. duration in hundred of sec)
        "svm_subsample": False,
        "color":         "#2ca02c",
        "skip":          False,
        "supports_sample_weight": False,   # KNeighborsClassifier.fit() has no sample_weight param
    },

    {   # ── SVM with RBF kernel ─────────────────────────────────────────────
        "name":          "SVM (RBF)",
        "short":         "SVM",
        "model":         SVC(
                             kernel="rbf",  # projects data into a space where a linear separator becomes a curved boundary in the original feature space
                             C=SVM_C,       # regularisation strength -> maximise the margin between classes (want C small) vs. minimise classification errors on the training set (want C large)
                             gamma="scale", # controls the reach of the RBF kernel -> uses 1 / (n_features * X.var())
                             class_weight="balanced", # compensates for class imbalance in the (possibly subsampled) training set
                         ),
        "needs_scaling": True,
        "svm_subsample": True,          # training rows are capped at SVM_MAX_ROWS
        "color":         "#d62728",
        "skip":          False,
        "supports_sample_weight": True,
    },

    {   # ── MLP (Multi-Layer Perceptron) ─────────────────────────────────────
        "name":          "MLP",
        "short":         "MLP",
        "model":         MLPClassifier(
                             hidden_layer_sizes=MLP_HIDDEN, # hidden layer sizes ->  first layer learns low-level feature combinations, second learns higher-level patterns
                             activation="relu",             # 2 hidden layers (128 → 64 neurons), ReLU activation -> avoids the vanishing gradient problem
                             max_iter=MLP_MAX_ITER,         # max epochs (early stopping is also enabled)
                             early_stopping=True,           # stops training when it stops improving, saving time on the cluster
                             n_iter_no_change=15,           # patience before stopping -> 15 rounds conservative enough to not stop prematurely on a temporary plateau
                             random_state=RANDOM_STATE,
                         ),
        "needs_scaling": True,
        "svm_subsample": False,
        "color":         "#9467bd",
        "skip":          False,
        "encode_labels": True,   # MLP early_stopping calls np.isnan() on predictions → fails with string labels
        "supports_sample_weight": False,   # MLPClassifier.fit() has no sample_weight param
    },
]



# =============================================================================
# SECTION 7 — TRAINING LOOP
# =============================================================================
#
# For each classifier:
#   1. Optionally subsample training rows (SVM only)
#   2. Optionally fit a StandardScaler
#   3. Train and measure wall-clock time
#   4. Predict on the test set
#   5. Compute metrics and save confusion matrix figure
#   6. Append results to RESULTS list
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 4 — Training and evaluation")
print(f"{'='*65}")

RESULTS = []   # list of dicts, one per successful classifier

for cfg in CLASSIFIER_CONFIGS:

    name  = cfg["name"]
    short = cfg["short"]
    model = cfg["model"]

    if cfg.get("skip", False):
        print(f"\n  [{name}] SKIPPED")
        continue

    print(f"\n  [{name}]")
    print("  " + "─" * 55)

    try:
        # ── Prepare training data ────────────────────────────────────────────
        X_fit = X_train.copy()   # copies to not affect the originals for the next classifier
        y_fit = y_train.copy()
        weight_fit = sample_weight_train.copy() if sample_weight_train is not None else None

        # SVM subsampling — stratified to preserve class balance
        if cfg["svm_subsample"] and SVM_MAX_ROWS is not None and len(X_fit) > SVM_MAX_ROWS:
            rng        = np.random.default_rng(RANDOM_STATE)
            classes    = np.unique(y_fit)
            n_per_cls  = SVM_MAX_ROWS // len(classes)  # integer division: 10,000 // 3 = 3,333 rows per class
            keep_idx   = []
            for cls in classes:
                cls_idx = np.where(y_fit == cls)[0]    # finds all row indices belonging to that class
                keep_idx.extend(rng.choice(cls_idx, n_per_cls, replace=False))  # picks 3,333 of them randomly without replacement
            keep_idx = np.array(keep_idx)
            X_fit    = X_fit[keep_idx]    # slice to keep only those rows
            y_fit    = y_fit[keep_idx]
            if weight_fit is not None:
                weight_fit = weight_fit[keep_idx]   # keep weights aligned with the subsampled rows
            print(f"  SVM subsampled: {len(X_fit):,} rows "
                  f"({n_per_cls:,} per class, from {len(X_train):,} SMOTE rows)")

        # Feature scaling — mandatory for KNN / SVM / MLP
        if cfg["needs_scaling"]:
            scaler    = StandardScaler()
            X_fit_sc  = scaler.fit_transform(X_fit)  # StandardScaler is fit on X_fit only (no leakage from test set)
            X_test_sc = scaler.transform(X_test)
        else:
            X_fit_sc  = X_fit
            X_test_sc = X_test

        # ── Label encoding (MLP + early_stopping requires integer labels) ──────
        if cfg.get("encode_labels", False):
            from sklearn.preprocessing import LabelEncoder
            le        = LabelEncoder().fit(CLASS_ORDER)
            y_fit_enc = le.transform(y_fit)
            y_test_enc = le.transform(y_test)
        else:
            y_fit_enc  = y_fit
            y_test_enc = y_test

        # ── Train ────────────────────────────────────────────────────────────
        t0 = time.time()
        if weight_fit is not None and cfg.get("supports_sample_weight", False):
            model.fit(X_fit_sc, y_fit_enc, sample_weight=weight_fit)
        else:
            if weight_fit is not None:
                print(f"  [REBALANCE] {name} has no sample_weight support — trained without it")
            model.fit(X_fit_sc, y_fit_enc)     # sklearn universal interface: every classifier accepts the same call
        train_time = time.time() - t0  # wall-clock training time
        t_str      = f"{train_time:.1f}s" if train_time < 60 else f"{train_time/60:.1f}m"
        print(f"  Training time : {t_str}  ({len(X_fit):,} training rows)")

        # ── Predict and evaluate ─────────────────────────────────────────────
        y_pred_raw = model.predict(X_test_sc)
        # decode integer predictions back to class name strings if needed
        y_pred = le.inverse_transform(y_pred_raw) if cfg.get("encode_labels", False) else y_pred_raw

        report   = classification_report(   # computes precision, recall, and F1 for each class, plus averages
            y_test, y_pred,
            labels=CLASS_ORDER,       # force class ordering (alphabetical default would swap rockslide/ice quake)
            target_names=CLASS_ORDER,
            output_dict=True,
            zero_division=0,
        )
        acc      = accuracy_score(y_test, y_pred)
        macro_f1 = report["macro avg"]["f1-score"]

        print(f"  Accuracy      : {acc:.3f}   Macro F1 : {macro_f1:.3f}")
        for cls in CLASS_ORDER:
            r = report[cls]
            print(f"  {cls:<22} F1={r['f1-score']:.3f}  "
                  f"P={r['precision']:.3f}  R={r['recall']:.3f}")

        # ── Confusion matrix figure ──────────────────────────────────────────
        # normalize='true' -> rows sum to 1.0 (recall per class)
        #  -> makes matrices comparable regardless of class sizes
        cm   = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER, normalize="true")
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_ORDER)
        disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format=".2f")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_title(
            f"{name}\n"
            f"Macro F1 = {macro_f1:.3f}   Acc = {acc:.3f}",
            fontsize=10,
        )
        plt.tight_layout()
        cm_path = os.path.join(RUN_DIR, f"fig_confusion_{short}_{STAMP}.png")
        plt.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] {cm_path}")

        # ── Store results ────────────────────────────────────────────────────
        RESULTS.append({
            "name":       name,
            "short":      short,
            "color":      cfg["color"],
            "acc":        acc,
            "macro_f1":   macro_f1,
            "train_time": train_time,
            **{
                f"{CLASS_ABBR[cls]}_{metric}": round(report[cls][full], 4)
                for cls in CLASS_ORDER
                for metric, full in [
                    ("f1", "f1-score"),
                    ("p",  "precision"),
                    ("r",  "recall"),
                ]
            },
        })

    except Exception as exc:
        print(f"  [ERROR] {name} failed: {exc}")
        import traceback
        traceback.print_exc()
        continue


# =============================================================================
# SECTION 8 — COMPARISON FIGURES + SUMMARY TABLE
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 5 — Comparison figures and summary")
print(f"{'='*65}")

if not RESULTS:
    print("[ERROR] No results collected — all classifiers failed.")
    sys.exit(1)

# Convenience lists for plotting
names     = [r["name"]     for r in RESULTS]
shorts    = [r["short"]    for r in RESULTS]
colors    = [r["color"]    for r in RESULTS]
macro_f1s = [r["macro_f1"] for r in RESULTS]
accs      = [r["acc"]      for r in RESULTS]
times     = [r["train_time"] for r in RESULTS]
# per-class F1 lists, one array per class in CLASS_ORDER (generalizes to any n classes)
class_f1s = {cls: [r[f"{CLASS_ABBR[cls]}_f1"] for r in RESULTS] for cls in CLASS_ORDER}

n = len(RESULTS)
x = np.arange(n)

# ── Figure: unified 3-panel comparison ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
_feat_label = (
    f"All {len(features)} features"
    if TOP_N_FEATURES is None
    else f"Top-{len(features)} features"
)
_class_counts_str = "  ".join(
    f"{sum(y_test == cls):,} {CLASS_ABBR[cls].upper()}" for cls in CLASS_ORDER
)
fig.suptitle(
    f"Classifier comparison — {_feat_label}  "
    f"({len(X_test):,} test rows, {_class_counts_str})",
    fontsize=11, fontweight="bold",
)

# -- Panel 1: per-class F1 grouped bars --------------------------------------
ax  = axes[0]
n_cls   = len(CLASS_ORDER)
w       = min(0.8 / n_cls, 0.22)
_bar_colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]
bar_kw = dict(alpha=0.85, edgecolor="white", linewidth=0.8)
offsets = [(i - (n_cls - 1) / 2) * w for i in range(n_cls)]
for off, cls, bc in zip(offsets, CLASS_ORDER, _bar_colors):
    vals = class_f1s[cls]
    ax.bar(x + off, vals, w, label=cls, color=bc, **bar_kw)
    for i, val in enumerate(vals):
        ax.text(i + off, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(shorts, fontsize=10)
ax.set_ylabel("F1-score")
ax.set_title("Per-class F1")
ax.set_ylim(0, 1.08)
ax.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.4)
ax.legend(fontsize=8, loc="upper right")

# -- Panel 2: macro F1 -------------------------------------------------------
ax   = axes[1]
bars = ax.bar(x, macro_f1s, color=colors, alpha=0.85, edgecolor="white", linewidth=0.8)
# RF baseline dashed reference line
ax.axhline(macro_f1s[0], color=colors[0], lw=1.5, ls="--", alpha=0.7,
           label=f"RF baseline ({macro_f1s[0]:.3f})")
for bar, val in zip(bars, macro_f1s):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(shorts, fontsize=10)
ax.set_ylabel("Macro F1-score  (equal class weight)")
ax.set_title("Macro F1")
ax.set_ylim(0, 1.0)
ax.legend(fontsize=8)

# -- Panel 3: training time (log scale) --------------------------------------
ax   = axes[2]
bars = ax.bar(x, times, color=colors, alpha=0.85, edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, times):
    lbl = f"{val:.0f}s" if val < 60 else f"{val/60:.1f}m"
    ax.text(bar.get_x() + bar.get_width() / 2, val * 1.15,
            lbl, ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(shorts, fontsize=10)
ax.set_ylabel("Training time (s, log scale)")
ax.set_title("Training time")
ax.set_yscale("log")   # log scale: times can span milliseconds to minutes

plt.tight_layout()
comp_path = os.path.join(RUN_DIR, f"fig_comparison_{STAMP}.png")
plt.savefig(comp_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[SAVED] {comp_path}")

# ── Save CSV ─────────────────────────────────────────────────────────────────
res_df = pd.DataFrame([
    {
        "classifier":   r["name"],
        "n_features":   len(features),
        "earthquake_rebalance_mode": EARTHQUAKE_REBALANCE_MODE,
        "accuracy":     round(r["acc"],        4),
        "macro_f1":     round(r["macro_f1"],   4),
        **{
            f"{CLASS_ABBR[cls]}_{metric}": round(r[f"{CLASS_ABBR[cls]}_{key}"], 4)
            for cls in CLASS_ORDER
            for metric, key in [("f1", "f1"), ("precision", "p"), ("recall", "r")]
        },
        "train_time_s": round(r["train_time"], 2),
    }
    for r in RESULTS
])

csv_path = os.path.join(RUN_DIR, f"comparison_results_{STAMP}.csv")
res_df.to_csv(csv_path, index=False)
print(f"[SAVED] {csv_path}")

# ── Print summary table ───────────────────────────────────────────────────────
COL = 19
print(f"\n  {'='*85}")
_feat_src = (
    "all features from catalog"
    if TOP_N_FEATURES is None
    else f"Top-{TOP_N_FEATURES} from 03b"
)
print(f"  SUMMARY — {len(features)}-feature set  ({_feat_src})")
print(f"  {'='*85}")
_cls_headers = "".join(f"{CLASS_ABBR[cls].upper() + ' F1':>7} " for cls in CLASS_ORDER)
print(f"  {'Classifier':{COL}} {'Acc':>6} {'MacroF1':>8} {_cls_headers}{'Time':>8}")
print(f"  {'-'*85}")
for r in RESULTS:
    t     = r["train_time"]
    t_str = f"{t:.0f}s" if t < 60 else f"{t/60:.1f}m"
    _cls_vals = "".join(f"{r[f'{CLASS_ABBR[cls]}_f1']:>7.3f} " for cls in CLASS_ORDER)
    print(
        f"  {r['name']:{COL}} "
        f"{r['acc']:>6.3f} "
        f"{r['macro_f1']:>8.3f} "
        f"{_cls_vals}"
        f"{t_str:>8}"
    )
print(f"  {'='*85}")

print(f"\n{'='*70}")
print(f"  Run finished  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Run folder    : {RUN_DIR}")
print(f"  Log           : {log_path}")
print(f"{'='*70}")

log_file.close()
