"""
06a_train_RF_classifier.py
=======================
ISTerre internship
Author : Elsa Louis
Date   : May 2026

Goal
----
Train a Random Forest classifier to distinguish between seismic event types (earthquake / rockslide / ice quake) 
 -> using the 99 Maggi/Hibert features extracted by script 04a

Pipeline
--------
  1. Load the catalog_windows CSV produced by 04a
  2. Filter on quality_ok flag (SNR thresholds from 05a)
  3. Print class distribution and check for imbalance
  4. Split by EVENT (not by row) -> avoids data leakage across stations
  5. Apply SMOTE on the training set to balance minority classes
  6. Train a RandomForestClassifier
  7. Evaluate on the held-out test set
       - confusion matrix
       - per-class precision / recall / F1
       - ROC curves (one-vs-rest, one per class)
       - top-20 feature importances
  8. Save the trained model as clf_RF.pkl  (reload with joblib or pickle)
  9. Optionally: score all rows in the original CSV and append predictions

References
----------
  Maggi et al. (2017) — 99-feature set
  Groult et al. (2026) — clf_RF.pkl inference approach
  Chawla et al. (2002) — SMOTE
"""


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# -- Input CSV (output of script 04a) -----------------------------------------
CSV_PATH = "/data/failles/louisels/project/results/outputs_04a/catalog_windows_XXXX_XXXX.csv"

# -- Noise CSV (output of script 04d, optional 4th class) ----------------------
# Set to a 04d `noise_windows_<stamp>.csv` to add the "noise" class
NOISE_CSV = None

# -- Regional CSV (output of script 04c, optional 5th class) -------------------
# Set to a 04c `regional_windows_<stamp>.csv` to add the "regional" class
REGIONAL_CSV = None

# -- Output directory ----------------------------------------------------------
OUTPUT_DIR = "/data/failles/louisels/project/results/outputs_06a"

# -- Classes to keep -----------------------------------------------------------
TARGET_CLASSES = ["earthquake", "rockslide", "ice quake", "noise", "regional"]

# -- Quality filtering ---------------------------------------------------------
FILTER_QUALITY = True   # True  → keep only quality_ok == True rows
                        # False → keep all rows (more data, more noise)

# -- Train / test split --------------------------------------------------------
TEST_SIZE    = 0.20   # fraction of EVENTS (not rows) reserved for testing
RANDOM_STATE = 42

# -- SMOTE (minority class oversampling on the training set) -------------------
USE_SMOTE     = True
SMOTE_STRATEGY = "auto"   # "auto" = resample all classes to match the majority
                            # or e.g. {"ice quake": 500} for fine-grained control

# -- Random Forest hyper-parameters --------------------------------------------
RF_N_ESTIMATORS  = 200      # nb of trees in the forest
RF_MAX_FEATURES  = "sqrt"   # classical choice: sqrt(n_features) ≈ 10 out of 99 (each tree has different 10 features)
RF_MIN_SAMPLES_LEAF = 2     # a knot (leaf) has at least 2 training examples
RF_CLASS_WEIGHT  = "balanced"   # extra safeguard against residual imbalance

# -- Output filenames ----------------------------------------------------------
MODEL_FILENAME       = "clf_RF.pkl"
SCORED_CSV_FILENAME  = "catalog_windows_scored.csv"



# =============================================================================
# SECTION 2 — SETUP
# =============================================================================

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("[WARN] imbalanced-learn not found (pip install imbalanced-learn).")
    print("       SMOTE will be disabled.")
    USE_SMOTE = False

import joblib   # preferred over pickle for sklearn models

sys.path.insert(0, os.path.dirname(__file__))
from features import FEATURE_NAMES, rename_legacy_columns
from run_setup import create_run_dir, setup_logging, set_matplotlib_defaults


# ---- Run directory + logging ------------------------------------------------
RUN_DIR, STAMP = create_run_dir(OUTPUT_DIR)
log_file, _    = setup_logging(RUN_DIR, "06a_train_RF_classifier.py",
                               extra_info=f"CSV: {CSV_PATH}")
set_matplotlib_defaults()


# =============================================================================
# SECTION 3 — LOAD AND FILTER DATA
# =============================================================================

print(f"\n{'='*65}")
print(f"  STEP 1 — Loading CSV")
print(f"{'='*65}")

if not os.path.isfile(CSV_PATH):
    print(f"[ERROR] CSV not found: {CSV_PATH}")
    print("        Update CSV_PATH in Section 1 and rerun.")
    sys.exit(1)

df_raw = pd.read_csv(CSV_PATH)
rename_legacy_columns(df_raw)   # renames feat_01…feat_99 → descriptive names if needed
print(f"Loaded {len(df_raw):,} rows × {df_raw.shape[1]} columns.")

# -- Optional 4th class: noise (output of 04d) --------------------------------
if NOISE_CSV is not None:
    if os.path.isfile(NOISE_CSV):
        df_noise = pd.read_csv(NOISE_CSV)
        rename_legacy_columns(df_noise)
        print(f"Loaded {len(df_noise):,} noise rows × {df_noise.shape[1]} columns "
              f"from {os.path.basename(NOISE_CSV)}.")
        df_raw = pd.concat([df_raw, df_noise], ignore_index=True)
    else:
        print(f"[WARN] NOISE_CSV not found: {NOISE_CSV} — continuing without the noise class.")

# -- Optional 5th class: regional (output of 04c) ------------------------------
if REGIONAL_CSV is not None:
    if os.path.isfile(REGIONAL_CSV):
        df_regional = pd.read_csv(REGIONAL_CSV)
        rename_legacy_columns(df_regional)
        print(f"Loaded {len(df_regional):,} regional rows × {df_regional.shape[1]} columns "
              f"from {os.path.basename(REGIONAL_CSV)}.")
        df_raw = pd.concat([df_raw, df_regional], ignore_index=True)
    else:
        print(f"[WARN] REGIONAL_CSV not found: {REGIONAL_CSV} — continuing without the regional class.")

# -- Keep only target classes -------------------------------------------------
df_raw = df_raw[df_raw["event_type"].isin(TARGET_CLASSES)].copy()
print(f"After class filter ({TARGET_CLASSES}): {len(df_raw):,} rows.")

# -- Quality filter -----------------------------------------------------------
if FILTER_QUALITY and "quality_ok" in df_raw.columns:
    n_before = len(df_raw)
    df_raw = df_raw[df_raw["quality_ok"] == True].copy()
    print(f"After quality filter (quality_ok==True): "
          f"{len(df_raw):,} rows kept  ({n_before - len(df_raw):,} dropped).")
elif FILTER_QUALITY:
    print("[WARN] 'quality_ok' column not found — skipping quality filter.")

# -- Class distribution -------------------------------------------------------
print(f"\n{'─'*55}")
print("  CLASS DISTRIBUTION  (after filtering)")
print(f"{'─'*55}")
class_counts = df_raw["event_type"].value_counts()
for cls, n in class_counts.items():
    pct = 100 * n / len(df_raw)
    print(f"  {cls:<20s}  {n:6,} rows  ({pct:.1f} %)")
print(f"  {'TOTAL':<20s}  {len(df_raw):6,} rows")
print(f"{'─'*55}\n")

n_events = df_raw["event_time"].nunique()
print(f"Unique events : {n_events}")
print(f"Avg rows/event: {len(df_raw)/n_events:.1f}")

# -- Drop rows missing any feature --------------------------------------------
missing_feats = [f for f in FEATURE_NAMES if f not in df_raw.columns]
if missing_feats:
    print(f"[WARN] {len(missing_feats)} feature columns not found in CSV "
          f"(e.g. {missing_feats[:3]}). Check FEATURE_NAMES.")
    sys.exit(1)

df_raw = df_raw.dropna(subset=FEATURE_NAMES).copy()
print(f"After dropping NaN-feature rows: {len(df_raw):,} rows.")


# =============================================================================
# SECTION 4 — TRAIN / TEST SPLIT BY EVENT
# =============================================================================

print(f"\n{'='*65}")
print(f"  STEP 2 — Train / test split  (by event, not by row)")
print(f"{'='*65}")

# One row per event (df_raw has one row per detection)
event_info = (
    df_raw.groupby("event_time")["event_type"]   # group together all rows that share same origin time
    .agg(lambda x: x.mode().iloc[0])   # get the dominant event_type for each event_time, should be unique per event, but .mode()[0] handles edge cases
    .reset_index()
    .rename(columns={"event_type": "event_type_label"})
)

# Check that every class has enough events for a stratified split
min_class_events = event_info["event_type_label"].value_counts().min()
if min_class_events < 5:
    print(f"[WARN] Smallest class has only {min_class_events} event(s). "
          f"Stratified split may fail.")
    stratify_col = None
else:
    stratify_col = event_info["event_type_label"]

train_event_times, test_event_times = train_test_split(
    event_info["event_time"],
    test_size    = TEST_SIZE,
    stratify     = stratify_col,
    random_state = RANDOM_STATE,
)
train_event_times = set(train_event_times)
test_event_times  = set(test_event_times)

df_train = df_raw[df_raw["event_time"].isin(train_event_times)].copy()
df_test  = df_raw[df_raw["event_time"].isin(test_event_times)].copy()

print(f"Train : {len(train_event_times):4d} events  →  {len(df_train):6,} rows")
print(f"Test  : {len(test_event_times):4d} events  →  {len(df_test):6,} rows")

print("\nTrain class distribution:")
for cls, n in df_train["event_type"].value_counts().items():
    print(f"  {cls:<20s}  {n:6,}")

print("\nTest class distribution:")
for cls, n in df_test["event_type"].value_counts().items():
    print(f"  {cls:<20s}  {n:6,}")


# =============================================================================
# SECTION 5 — FEATURE MATRIX + SMOTE
# =============================================================================

print(f"\n{'='*65}")
print(f"  STEP 3 — Feature matrix + SMOTE")
print(f"{'='*65}")

X_train = df_train[FEATURE_NAMES].values.astype(np.float32)
y_train = df_train["event_type"].values

X_test  = df_test[FEATURE_NAMES].values.astype(np.float32)
y_test  = df_test["event_type"].values

print(f"X_train shape : {X_train.shape}")
print(f"X_test  shape : {X_test.shape}")

if USE_SMOTE:
    # SMOTE requires at least k_neighbors+1 samples per class (default k=5 → need ≥6)
    min_class_rows = pd.Series(y_train).value_counts().min()
    k_neighbors = min(5, min_class_rows - 1)
    if k_neighbors < 1:
        print(f"[WARN] Smallest training class has {min_class_rows} sample(s). "
              f"SMOTE disabled (need ≥ 2).")
    else:
        print(f"\nApplying SMOTE  (k_neighbors={k_neighbors}, strategy={SMOTE_STRATEGY}) ...")
        sm = SMOTE(sampling_strategy=SMOTE_STRATEGY,
                   k_neighbors=k_neighbors,
                   random_state=RANDOM_STATE)
        X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
        print(f"After SMOTE: {X_train_sm.shape[0]:,} rows")
        for cls, n in pd.Series(y_train_sm).value_counts().items():
            print(f"  {cls:<20s}  {n:6,}")
        X_train, y_train = X_train_sm, y_train_sm
else:
    print("SMOTE disabled.")


# =============================================================================
# SECTION 6 — TRAIN RANDOM FOREST
# =============================================================================

print(f"\n{'='*65}")
print(f"  STEP 4 — Training Random Forest")
print(f"{'='*65}")
print(f"  n_estimators     = {RF_N_ESTIMATORS}")
print(f"  max_features     = {RF_MAX_FEATURES}")
print(f"  min_samples_leaf = {RF_MIN_SAMPLES_LEAF}")
print(f"  class_weight     = {RF_CLASS_WEIGHT}")

clf = RandomForestClassifier(
    n_estimators     = RF_N_ESTIMATORS,
    max_features     = RF_MAX_FEATURES,
    min_samples_leaf = RF_MIN_SAMPLES_LEAF,
    class_weight     = RF_CLASS_WEIGHT,
    n_jobs           = -1,        # use all available CPU cores
    random_state     = RANDOM_STATE,
)
clf.fit(X_train, y_train)
print("[OK] Training complete.")

# Determine class order used internally by sklearn
classes = list(clf.classes_)
print(f"Classes (RF order): {classes}")


# =============================================================================
# SECTION 7 — EVALUATION
# =============================================================================

print(f"\n{'='*65}")
print(f"  STEP 5 — Evaluation on test set")
print(f"{'='*65}")

y_pred      = clf.predict(X_test)
y_proba     = clf.predict_proba(X_test)   # shape: (n_test, n_classes)

# -- Classification report (precision / recall / F1 per class) ----------------
print("\n" + classification_report(y_test, y_pred, target_names=classes))

# -- Confusion matrix ----------------------------------------------------------
cm = confusion_matrix(y_test, y_pred, labels=classes)
print("Confusion matrix (rows=true, cols=predicted):")
print(pd.DataFrame(cm, index=classes, columns=classes).to_string())

fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
ax_cm.set_title("Confusion matrix — test set")
plt.tight_layout()
cm_path = os.path.join(RUN_DIR, "confusion_matrix.png")
fig_cm.savefig(cm_path, dpi=150)
plt.close(fig_cm)
print(f"\n[SAVED] {cm_path}")

# -- ROC curves (one-vs-rest, one curve per class) ----------------------------
y_test_bin = label_binarize(y_test, classes=classes)  # (n_test, n_classes)
n_classes  = len(classes)

fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
for i, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc      = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, lw=2, label=f"{cls}  (AUC={roc_auc:.3f})")

ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC curves — one-vs-rest (test set)")
ax_roc.legend(loc="lower right")
plt.tight_layout()
roc_path = os.path.join(RUN_DIR, "roc_curves.png")
fig_roc.savefig(roc_path, dpi=150)
plt.close(fig_roc)
print(f"[SAVED] {roc_path}")

# -- Feature importances (top 20) ---------------------------------------------
importances = clf.feature_importances_
feat_imp_df = pd.DataFrame({
    "feature":    FEATURE_NAMES,
    "importance": importances,
}).sort_values("importance", ascending=False).head(20)

print("\nTop-20 feature importances:")
for _, row in feat_imp_df.iterrows():
    print(f"  {row['feature']:<12s}  {row['importance']:.4f}")

fig_fi, ax_fi = plt.subplots(figsize=(8, 6))
ax_fi.barh(feat_imp_df["feature"][::-1], feat_imp_df["importance"][::-1])
ax_fi.set_xlabel("Mean decrease in impurity")
ax_fi.set_title("Top-20 feature importances (Random Forest)")
plt.tight_layout()
fi_path = os.path.join(RUN_DIR, "feature_importances.png")
fig_fi.savefig(fi_path, dpi=150)
plt.close(fig_fi)
print(f"[SAVED] {fi_path}")


# =============================================================================
# SECTION 8 — SAVE MODEL
# =============================================================================

print(f"\n{'='*65}")
print(f"  STEP 6 — Saving model")
print(f"{'='*65}")

model_path = os.path.join(RUN_DIR, MODEL_FILENAME)
joblib.dump(clf, model_path)
print(f"[SAVED] Model → {model_path}")
print(f"        Reload with: clf = joblib.load('{model_path}')")


# =============================================================================
# SECTION 9 — SCORE ALL ROWS IN THE ORIGINAL CSV (optional)
# =============================================================================
# Append predicted class + per-class probabilities back onto the full filtered CSV, so we can inspect each detection's score later

print(f"\n{'='*65}")
print(f"  STEP 7 — Scoring all rows in the dataset")
print(f"{'='*65}")

X_all   = df_raw[FEATURE_NAMES].values.astype(np.float32)
y_all   = clf.predict(X_all)
p_all   = clf.predict_proba(X_all)

df_scored = df_raw.copy()
df_scored["predicted_class"] = y_all
for i, cls in enumerate(classes):
    col_name = f"proba_{cls.replace(' ', '_')}"
    df_scored[col_name] = p_all[:, i]

# Add train/test split flag for traceability
df_scored["split"] = df_scored["event_time"].apply(
    lambda t: "train" if t in train_event_times else "test"
)

scored_path = os.path.join(RUN_DIR, SCORED_CSV_FILENAME)
df_scored.to_csv(scored_path, index=False)
print(f"[SAVED] Scored CSV → {scored_path}")
print(f"        Columns added: predicted_class, proba_<class>, split")

# Quick sanity check: accuracy on the rows flagged as test
df_test_check = df_scored[df_scored["split"] == "test"]
acc = (df_test_check["predicted_class"] == df_test_check["event_type"]).mean()
print(f"\nOverall accuracy on test rows: {acc:.3f}")


# =============================================================================
# DONE
# =============================================================================

print(f"\n{'='*65}")
print(f"  ALL DONE")
print(f"  Run folder : {RUN_DIR}")
print(f"  Model      : {model_path}")
print(f"  Figures    : confusion_matrix.png  roc_curves.png  feature_importances.png")
print(f"  Scored CSV : {scored_path}")
print(f"{'='*65}\n")

log_file.close()
