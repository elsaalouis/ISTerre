"""
06c_train_HGB_classifier.py
========================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : July 2026

Train a HistGradientBoosting classifier on the full dataset (original catalog + rescued ice quakes from 03d denoising pipeline)

Scientific question answered
----------------------------
Did the DeepDenoiser rescue pipeline meaningfully improve classification — and is
that improvement actually coming from the DENOISER, or just from adding more real
low-SNR examples to training regardless of denoising? Answered with three conditions:
  (A)  Original catalog only        (same data as 06b)
  (B)  Original + denoised rescued  (after 03d feature extraction)
  (C)  Original + RAW rescued       (same events as B, undenoised features —
                                      needs 03d's rescue_catalog_raw_<stamp>.csv)
If C performs about as well as B, the gain is from added data volume, not denoising.
If B clearly beats C, that's evidence the denoiser itself is adding value.

Pipeline position
-----------------
  03d ✓  +  06b ✓  →  [06c this script]

Key differences from 06b
------------------------
  - Loads and concatenates RESCUE_CATALOG_CSV (from 03d) with the original
  - Optionally also loads RESCUE_CATALOG_RAW_CSV for the Run C ablation
  - Trains only HGB (+ RF as baseline); removes KNN / SVM / MLP for speed
  - Runs up to three full train-eval cycles: A, B, and C
  - Produces a direct before/after/ablation comparison figure

Outputs
-------
  fig_confusion_A_<stamp>.png  : HGB confusion matrix — original only
  fig_confusion_B_<stamp>.png  : HGB confusion matrix — original + denoised rescued
  fig_confusion_C_<stamp>.png  : HGB confusion matrix — original + raw rescued (ablation)
  fig_comparison_<stamp>.png   : A / B / C panel (IQ or RS F1 / precision / recall, Macro F1)
  results_<stamp>.csv          : full metrics for all runs present
"""


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# ── Original catalog (04a output) ─────────────────────────────────────────────
ORIGINAL_CSV = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\04a_spectrogram_sta_lta_catalog\all-99-features-recent+3C\catalog_windows_20260708_174019.csv"

# ── Rescue catalog (03d output) ───────────────────────────────────────────────
RESCUE_CATALOG_CSV = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\03d_rescue_feature_extraction\stricter_IQ100_20260722_145529\rescue_catalog_20260722_145529.csv"

# ── Raw-ablation rescue catalog (03d output, Run C) ───────────────────────────
# Same accepted events as RESCUE_CATALOG_CSV, but features extracted from the RAW (pre-denoise) signal instead
# Set to None to skip Run C
RESCUE_CATALOG_RAW_CSV = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\03d_rescue_feature_extraction\stricter_IQ100_20260722_145529\rescue_catalog_raw_20260722_145529.csv"

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\06c_HGB_classifier"

# ── Classes ───────────────────────────────────────────────────────────────────
TARGET_CLASSES = ["earthquake", "rockslide", "ice quake"]
CLASS_ORDER    = ["earthquake", "rockslide", "ice quake"]

# ── Feature set ───────────────────────────────────────────────────────────────
FEATURE_IMPORTANCES_CSV = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\03b_feature_selection\run_20260710_144246\feature_importances_20260710_144246.csv"
# TOP_N_FEATURES = None  → use ALL feature columns present in the catalog
# TOP_N_FEATURES = int   → use the top-N features ranked by FEATURE_IMPORTANCES_CSV (or FALLBACK_TOP20 if that file is unavailable)
TOP_N_FEATURES          = 60   

FALLBACK_TOP20 = [
    "duration",                "ediff_3_10__10_20",   "eratio_3_10__10_20",
    "spec_kurtosis_median_env","kurtosis_10_20Hz",    "kurtosis_3_10Hz",
    "kurtosis_1_8Hz",          "fft_energy_1_nyq4",   "kurtosis_1_3Hz",
    "energy_1_3Hz",            "fft_freq_at_max",     "fft_spread_peaks",
    "dist_q3_q1",              "eratio_1_3__3_10",    "eratio_0.1_1__1_3",
    "fft_n_peaks",             "ediff_1_3__10_20",    "ascend_descend_ratio",
    "kurtosis_20_nyq",         "ediff_1_3__3_10",
]

# ── Quality gate (applied to original catalog only; rescue catalog already ────
SNR_MIN             = 1.70    # 05b Tier 2 — metric 'SNR', AUC=0.627
SNR_FULL_MEDIAN_MIN = 1.99    # 05b Tier 2 — metric 'SNR_full_median', AUC=0.642 (best)

# ── Train / test split ────────────────────────────────────────────────────────
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# ── SMOTE ─────────────────────────────────────────────────────────────────────
SMOTE_K = 5

# ── Classifier hyperparameters ────────────────────────────────────────────────
# HGB — primary classifier
HGB_N_EST     = 200
HGB_MAX_DEPTH = 6
HGB_LR        = 0.1

# RF — baseline comparison
RF_N_EST      = 200


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
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
from features import FEATURE_NAMES, FEATURE_NAMES_3C, POLARIZATION_NAMES, rename_legacy_columns
from run_setup import create_run_dir, setup_logging

warnings.filterwarnings("ignore")

RUN_DIR, STAMP = create_run_dir(OUTPUT_DIR)
log_file, log_path = setup_logging(
    RUN_DIR,
    script_name="06c_HGB_with_rescues.py",
    extra_info=(
        f"ORIGINAL_CSV: {ORIGINAL_CSV}\n"
        f"RESCUE_CATALOG_CSV: {RESCUE_CATALOG_CSV}\n"
        f"RESCUE_CATALOG_RAW_CSV: {RESCUE_CATALOG_RAW_CSV}"
    ),
)


# =============================================================================
# SECTION 3 — LOAD AND COMBINE DATA
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 1 — Loading catalogs")
print(f"{'='*65}")

# ── Original catalog ──────────────────────────────────────────────────────────
orig = pd.read_csv(ORIGINAL_CSV, low_memory=False)
orig = rename_legacy_columns(orig)
orig = orig[orig["event_type"].isin(TARGET_CLASSES)].copy()

# Apply quality gate to original (rescue rows already satisfy this same gate,
# by construction, from 03d). Always recompute explicitly from SNR/SNR_full_median
# — do NOT trust the precomputed 'quality_ok' column, which was baked by 04a
# using the OLD 05a thresholds and is stale relative to the 05b Tier 2 values
# used here (only refreshed if 04a itself is rerun, out of scope for now).
mask = (
    (orig["SNR"]             >= SNR_MIN) &
    (orig["SNR_full_median"] >= SNR_FULL_MEDIAN_MIN)
)
orig = orig[mask].copy()
z_feat_cols = [f for f in FEATURE_NAMES if f in orig.columns]
orig = orig.dropna(subset=z_feat_cols).copy()
orig["source"] = "original"

print(f"  Original catalog  : {len(orig):,} rows after quality gate + NaN drop")
for cls in CLASS_ORDER:
    n = (orig["event_type"] == cls).sum()
    print(f"    {cls:<22} {n:>6,}  ({100*n/len(orig):.1f} %)")

# ── Rescue catalog (optional) ─────────────────────────────────────────────────
has_rescue = RESCUE_CATALOG_CSV is not None and os.path.exists(str(RESCUE_CATALOG_CSV))

if has_rescue:
    rescue = pd.read_csv(RESCUE_CATALOG_CSV, low_memory=False)
    rescue = rename_legacy_columns(rescue)
    rescue = rescue[rescue["event_type"].isin(TARGET_CLASSES)].copy()
    z_feat_cols_r = [f for f in FEATURE_NAMES if f in rescue.columns]
    rescue = rescue.dropna(subset=z_feat_cols_r).copy()
    if "source" not in rescue.columns:
        rescue["source"] = "denoised_rescue"
    print(f"\n  Rescue catalog    : {len(rescue):,} rows")
    for cls in CLASS_ORDER:
        n = (rescue["event_type"] == cls).sum()
        print(f"    {cls:<22} {n:>6,}  ({100*n/len(rescue):.1f} %)")
else:
    rescue = pd.DataFrame()
    if RESCUE_CATALOG_CSV is None:
        print("\n  [INFO] RESCUE_CATALOG_CSV is None — running original-only mode.")
        print("         Set RESCUE_CATALOG_CSV to the 03d output to enable the")
        print("         before/after comparison.")
    else:
        print(f"\n  [WARN] RESCUE_CATALOG_CSV not found: {RESCUE_CATALOG_CSV}")
        print("         Running original-only mode.")

# Combined dataset (Run B: original + denoised rescue)
combined = pd.concat([orig, rescue], ignore_index=True) if has_rescue else orig.copy()
print(f"\n  Combined dataset  : {len(combined):,} rows")
for cls in CLASS_ORDER:
    n = (combined["event_type"] == cls).sum()
    print(f"    {cls:<22} {n:>6,}  ({100*n/len(combined):.1f} %)")

# ── Raw-ablation rescue catalog (optional, Run C) ─────────────────────────────
# Same accepted events as `rescue`, but features from the undenoised signal.
has_rescue_raw = (
    RESCUE_CATALOG_RAW_CSV is not None and os.path.exists(str(RESCUE_CATALOG_RAW_CSV))
)

if has_rescue_raw:
    rescue_raw = pd.read_csv(RESCUE_CATALOG_RAW_CSV, low_memory=False)
    rescue_raw = rename_legacy_columns(rescue_raw)
    rescue_raw = rescue_raw[rescue_raw["event_type"].isin(TARGET_CLASSES)].copy()
    z_feat_cols_rr = [f for f in FEATURE_NAMES if f in rescue_raw.columns]
    rescue_raw = rescue_raw.dropna(subset=z_feat_cols_rr).copy()
    rescue_raw["source"] = "raw_undenoised_rescue"
    print(f"\n  Raw-ablation catalog : {len(rescue_raw):,} rows (Run C — same events, no denoising)")
    for cls in CLASS_ORDER:
        n = (rescue_raw["event_type"] == cls).sum()
        print(f"    {cls:<22} {n:>6,}  ({100*n/len(rescue_raw):.1f} %)")
    if "raw_passes_gate_alone" in rescue_raw.columns:
        n_alone = int(rescue_raw["raw_passes_gate_alone"].sum())
        print(f"    Already passed the gate on raw SNR alone: {n_alone:,} / {len(rescue_raw):,} "
              f"({100*n_alone/max(len(rescue_raw),1):.1f} %) — denoising wasn't necessary for these")
else:
    rescue_raw = pd.DataFrame()
    if RESCUE_CATALOG_RAW_CSV is None:
        print("\n  [INFO] RESCUE_CATALOG_RAW_CSV is None — Run C (raw ablation) will be skipped.")
    else:
        print(f"\n  [WARN] RESCUE_CATALOG_RAW_CSV not found: {RESCUE_CATALOG_RAW_CSV}")
        print("         Run C (raw ablation) will be skipped. Re-run 03d with the current")
        print("         code to generate the sibling rescue_catalog_raw_<stamp>.csv.")

# Combined dataset (Run C: original + raw/undenoised rescue)
combined_raw = (
    pd.concat([orig, rescue_raw], ignore_index=True) if has_rescue_raw else pd.DataFrame()
)
if has_rescue_raw:
    print(f"\n  Combined dataset (raw ablation) : {len(combined_raw):,} rows")
    for cls in CLASS_ORDER:
        n = (combined_raw["event_type"] == cls).sum()
        print(f"    {cls:<22} {n:>6,}  ({100*n/len(combined_raw):.1f} %)")


# =============================================================================
# SECTION 4 — FEATURE SELECTION
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 2 — Feature selection")
print(f"{'='*65}")

if TOP_N_FEATURES is None:
    # Use ALL feature columns present in the combined catalog.
    # FEATURE_NAMES_3C is the ordered superset (103: 99 Z + 4 polarization).
    # Intersect with combined.columns to handle 99-feature and 103-feature catalogs.
    features = [f for f in FEATURE_NAMES_3C if f in combined.columns]
    if not features:
        features = [f for f in FEATURE_NAMES if f in combined.columns]
    print(f"  TOP_N_FEATURES=None → using all {len(features)} feature columns found in catalog.")
elif FEATURE_IMPORTANCES_CSV is not None and os.path.exists(FEATURE_IMPORTANCES_CSV):
    imp_df   = pd.read_csv(FEATURE_IMPORTANCES_CSV)
    features = imp_df["feature"].head(TOP_N_FEATURES).tolist()
    print(f"  Loaded Top-{TOP_N_FEATURES} features from: {FEATURE_IMPORTANCES_CSV}")
else:
    features = list(FALLBACK_TOP20[:TOP_N_FEATURES])
    print(f"  Using hardcoded Top-{TOP_N_FEATURES} fallback list.")

missing = [f for f in features if f not in combined.columns]
if missing:
    raise ValueError(f"Features missing from combined CSV: {missing}")
print(f"  n_features = {len(features)}")


# =============================================================================
# SECTION 5 — TRAIN / EVALUATE HELPER
# =============================================================================

def train_and_eval(df, label, features, smote_k, test_size, rs):
    """
    Full pipeline: event-stratified split → SMOTE → HGB + RF → metrics dict.
    Returns (results_dict, cm_hgb, cm_rf, X_test, y_test) for plotting.
    """
    # ── Split by unique events ─────────────────────────────────────────────────
    events = df[["event_time", "event_type"]].drop_duplicates("event_time")
    # Guard: if a class has < 2 events it can't be stratified
    class_counts = events["event_type"].value_counts()
    min_class_size = class_counts.min()
    effective_k = min(smote_k, min_class_size - 1)
    if effective_k < 1:
        print(f"  [WARN] {label}: smallest class has {min_class_size} events — cannot split or SMOTE.")
        return None

    train_ev, test_ev = train_test_split(
        events["event_time"],
        test_size=test_size,
        stratify=events["event_type"],
        random_state=rs,
    )
    train_mask = df["event_time"].isin(train_ev)
    test_mask  = df["event_time"].isin(test_ev)

    X_tr_raw = df.loc[train_mask, features].values
    y_tr_raw = df.loc[train_mask, "event_type"].values
    X_te     = df.loc[test_mask,  features].values
    y_te     = df.loc[test_mask,  "event_type"].values

    print(f"\n  [{label}]  Train: {train_mask.sum():,} rows  |  Test: {test_mask.sum():,} rows")

    # Impute NaN with training-set median (fits on train only — no leakage).
    # Handles polarization features that are NaN when horizontal channels were
    # unavailable; SMOTE and RF cannot accept NaN arrays.
    _imp     = SimpleImputer(strategy="median")
    X_tr_raw = _imp.fit_transform(X_tr_raw)
    X_te     = _imp.transform(X_te)

    sm = SMOTE(k_neighbors=effective_k, random_state=rs)
    X_tr, y_tr = sm.fit_resample(X_tr_raw, y_tr_raw)
    print(f"  After SMOTE: {len(X_tr):,} rows")

    results = {"label": label, "n_train_raw": train_mask.sum(), "n_test": test_mask.sum()}
    cms = {}

    for name, short, model in [
        ("Hist. GradBoost", "HGB",
         HistGradientBoostingClassifier(
             max_iter=HGB_N_EST, max_depth=HGB_MAX_DEPTH,
             learning_rate=HGB_LR, early_stopping=True,
             n_iter_no_change=15, random_state=rs,
         )),
        ("Random Forest", "RF",
         RandomForestClassifier(
             n_estimators=RF_N_EST, max_features="sqrt",
             class_weight="balanced", random_state=rs, n_jobs=-1,
         )),
    ]:
        t0 = time.time()
        model.fit(X_tr, y_tr)
        elapsed = time.time() - t0
        y_pred  = model.predict(X_te)

        report   = classification_report(
            y_te, y_pred,
            labels=CLASS_ORDER, target_names=CLASS_ORDER,
            output_dict=True, zero_division=0,
        )
        acc      = accuracy_score(y_te, y_pred)
        macro_f1 = report["macro avg"]["f1-score"]

        t_str = f"{elapsed:.1f}s" if elapsed < 60 else f"{elapsed/60:.1f}m"
        print(f"  {name}: Accuracy={acc:.3f}  MacroF1={macro_f1:.3f}  "
              f"IQ-F1={report['ice quake']['f1-score']:.3f}  Time={t_str}")

        results[short] = {
            "acc":      acc,
            "macro_f1": macro_f1,
            **{
                f"{abbr}_{m}": round(report[cls][full], 4)
                for cls, abbr in [("earthquake","eq"),("rockslide","rs"),("ice quake","iq")]
                for m, full in [("f1","f1-score"),("p","precision"),("r","recall")]
            },
            "train_time_s": round(elapsed, 2),
        }
        cms[short] = confusion_matrix(y_te, y_pred, labels=CLASS_ORDER, normalize="true")

    return results, cms, X_te, y_te


# =============================================================================
# SECTION 6 — RUN A: ORIGINAL ONLY
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 3 — Run A: original catalog only")
print(f"{'='*65}")

out_A = train_and_eval(orig, "A — Original only", features, SMOTE_K, TEST_SIZE, RANDOM_STATE)
if out_A is None:
    print("[ERROR] Run A failed. Aborting.")
    sys.exit(1)
results_A, cms_A, Xte_A, yte_A = out_A


# =============================================================================
# SECTION 7 — RUN B: ORIGINAL + RESCUED  (only if rescue data is available)
# =============================================================================

results_B = None
cms_B     = None

if has_rescue:
    print(f"\n{'='*65}")
    print("  STEP 4 — Run B: original + rescued ice quakes")
    print(f"{'='*65}")
    out_B = train_and_eval(combined, "B — Original + rescued", features, SMOTE_K, TEST_SIZE, RANDOM_STATE)
    if out_B is not None:
        results_B, cms_B, Xte_B, yte_B = out_B
    else:
        print("  [WARN] Run B failed — skipping comparison figure.")
else:
    print(f"\n  [INFO] No rescue catalog loaded — skipping Run B.")
    print(f"         To enable, set RESCUE_CATALOG_CSV at the top of this script.")


# =============================================================================
# SECTION 7b — RUN C: ORIGINAL + RAW/UNDENOISED RESCUE (ablation, only if available)
# =============================================================================
# Same accepted events as Run B, same everything, EXCEPT the rescue rows' features
# come from the raw (pre-denoise) signal instead of the denoised one. If Run C
# performs about as well as Run B, the Run B gain is coming from "more real
# examples" rather than from the denoiser itself.

results_C = None
cms_C     = None

if has_rescue_raw:
    print(f"\n{'='*65}")
    print("  STEP 4b — Run C: original + RAW rescued (denoiser ablation)")
    print(f"{'='*65}")
    out_C = train_and_eval(combined_raw, "C — Original + raw rescued (ablation)",
                            features, SMOTE_K, TEST_SIZE, RANDOM_STATE)
    if out_C is not None:
        results_C, cms_C, Xte_C, yte_C = out_C
    else:
        print("  [WARN] Run C failed — skipping ablation comparison.")
else:
    print(f"\n  [INFO] No raw-ablation catalog loaded — skipping Run C.")
    print(f"         To enable, set RESCUE_CATALOG_RAW_CSV at the top of this script")
    print(f"         (produced by 03d as rescue_catalog_raw_<stamp>.csv).")


# =============================================================================
# SECTION 8 — FIGURES
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 5 — Figures")
print(f"{'='*65}")


def save_cm_figure(cm, title, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_ORDER)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format=".2f")
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {path}")


# Confusion matrices — Run A
for short in ["HGB", "RF"]:
    cm   = cms_A[short]
    mf1  = results_A[short]["macro_f1"]
    acc  = results_A[short]["acc"]
    save_cm_figure(
        cm,
        f"{short} — Original only\nMacroF1={mf1:.3f}  Acc={acc:.3f}",
        os.path.join(RUN_DIR, f"fig_confusion_A_{short}_{STAMP}.png"),
    )

# Confusion matrices — Run B
if results_B is not None:
    for short in ["HGB", "RF"]:
        cm  = cms_B[short]
        mf1 = results_B[short]["macro_f1"]
        acc = results_B[short]["acc"]
        save_cm_figure(
            cm,
            f"{short} — Original + rescued\nMacroF1={mf1:.3f}  Acc={acc:.3f}",
            os.path.join(RUN_DIR, f"fig_confusion_B_{short}_{STAMP}.png"),
        )

# Confusion matrices — Run C (raw ablation)
if results_C is not None:
    for short in ["HGB", "RF"]:
        cm  = cms_C[short]
        mf1 = results_C[short]["macro_f1"]
        acc = results_C[short]["acc"]
        save_cm_figure(
            cm,
            f"{short} — Original + raw rescued (ablation)\nMacroF1={mf1:.3f}  Acc={acc:.3f}",
            os.path.join(RUN_DIR, f"fig_confusion_C_{short}_{STAMP}.png"),
        )

# Before / after / ablation comparison figure
# Auto-detect which class the rescue catalog actually targets (rockslide as of
# 2026-07, historically ice quake) so the panel tracks the metric that can
# actually move — hardcoding "ice quake" here would make Run C's effect
# invisible whenever the rescue target is a different class.
_class_abbr = {"earthquake": "eq", "rockslide": "rs", "ice quake": "iq"}
if has_rescue and len(rescue) > 0:
    _target_class = rescue["event_type"].mode().iat[0]
elif has_rescue_raw and len(rescue_raw) > 0:
    _target_class = rescue_raw["event_type"].mode().iat[0]
else:
    _target_class = "ice quake"
_abbr = _class_abbr.get(_target_class, "iq")

if results_B is not None or results_C is not None:
    metrics_shown = [
        (f"{_target_class.title()} F1",        f"{_abbr}_f1", "F1-score"),
        (f"{_target_class.title()} Precision", f"{_abbr}_p",  "Precision"),
        (f"{_target_class.title()} Recall",    f"{_abbr}_r",  "Recall"),
        ("Macro F1",     "macro_f1", "Macro F1"),
    ]

    # Only include runs that actually completed
    run_defs = [("A", "Original\nonly", "#1f77b4", results_A)]
    if results_B is not None:
        run_defs.append(("B", "Original\n+ denoised", "#ff7f0e", results_B))
    if results_C is not None:
        run_defs.append(("C", "Original\n+ raw (ablation)", "#2ca02c", results_C))

    n_bars = len(run_defs)
    positions = [0.5 * i for i in range(n_bars)]

    fig, axes = plt.subplots(1, len(metrics_shown), figsize=(4 * len(metrics_shown), 4.5))
    _feat_pfx = "All" if TOP_N_FEATURES is None else f"Top-{len(features)}"
    fig.suptitle(
        f"DeepDenoiser rescue impact on {_target_class} classification\n"
        f"HGB  |  {_feat_pfx} features  |  Test set n≈{Xte_A.shape[0]}  |  "
        f"B vs C isolates the denoiser's own contribution from added data volume",
        fontsize=11, fontweight="bold",
    )
    bar_kw = dict(edgecolor="white", linewidth=0.8, alpha=0.85, width=0.4)

    for ax, (title, key, ylabel) in zip(axes, metrics_shown):
        vals = [res["HGB"][key] for _, _, _, res in run_defs]
        for pos, (_, _, color, _), val in zip(positions, run_defs, vals):
            ax.bar(pos, val, color=color, **bar_kw)
            ax.text(pos, val + 0.01, f"{val:.3f}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")
        # Delta annotations relative to Run A
        val_A = vals[0]
        for pos, (short, _, _, _), val in list(zip(positions, run_defs, vals))[1:]:
            delta = val - val_A
            sign  = "+" if delta >= 0 else ""
            ax.text(pos, val + 0.065, f"Δ{short}={sign}{delta:.3f}",
                    ha="center", va="bottom", fontsize=8,
                    color="darkgreen" if delta >= 0 else "red")
        ax.set_xlim(-0.4, positions[-1] + 0.4)
        ax.set_ylim(0, 1.2)
        ax.set_xticks(positions)
        ax.set_xticklabels([label for _, label, _, _ in run_defs], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.axhline(val_A, color=run_defs[0][2], lw=0.8, ls="--", alpha=0.5)

    plt.tight_layout()
    comp_path = os.path.join(RUN_DIR, f"fig_before_after_{STAMP}.png")
    plt.savefig(comp_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {comp_path}")


# =============================================================================
# SECTION 9 — SUMMARY TABLE + CSV
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 6 — Summary")
print(f"{'='*65}")

runs = [("A — Original only",       results_A)]
if results_B is not None:
    runs.append(("B — Original + denoised rescued", results_B))
if results_C is not None:
    runs.append(("C — Original + raw rescued (ablation)", results_C))

COL = 40
print(f"\n  {'Run':{COL}} {'Clf':>5} {'Acc':>6} {'MacroF1':>8} "
      f"{'EQ F1':>7} {'RS F1':>7} {'IQ F1':>7} {'IQ P':>6} {'IQ R':>6}")
print(f"  {'-'*100}")

csv_rows = []
for run_label, res in runs:
    for short in ["HGB", "RF"]:
        r = res[short]
        print(
            f"  {run_label:{COL}} {short:>5} "
            f"{r['acc']:>6.3f} {r['macro_f1']:>8.3f} "
            f"{r['eq_f1']:>7.3f} {r['rs_f1']:>7.3f} "
            f"{r['iq_f1']:>7.3f} {r['iq_p']:>6.3f} {r['iq_r']:>6.3f}"
        )
        csv_rows.append({
            "run":         run_label,
            "classifier":  short,
            "n_features":  len(features),
            **{k: v for k, v in r.items() if k not in ("label",)},
        })

res_df   = pd.DataFrame(csv_rows)
csv_path = os.path.join(RUN_DIR, f"results_{STAMP}.csv")
res_df.to_csv(csv_path, index=False)
print(f"\n  [SAVED] {csv_path}")

print(f"\n{'='*65}")
print(f"  Run finished : {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Run folder   : {RUN_DIR}")
print(f"  Log          : {log_path}")
print(f"{'='*65}")

log_file.close()
