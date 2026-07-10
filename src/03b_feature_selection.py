"""
FEATURE SELECTION & CORRELATION ANALYSIS
=========================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : May 2026

Goal
----
- Understand the redundancy and discriminative power of the 99 Maggi/Hibert features before feeding them to a classifier
- Find the smallest reliable subset

Pipeline
--------
  1. Load the catalog_windows CSV (output of script 04a)
  2. Correlation analysis: Pearson and Spearman 99×99 heatmaps, cluster extraction (group features with |r| > CORR_THRESHOLD into clusters)
  3. HGB feature importances: train HGB (same config as 06b) on all 99 features, rank them by permutation importance (how much Macro F1 drops when each feature is shuffled on the test set)
  4. Feature subset experiments: for each subset (top-20 / top-40 / top-60 / all-99 / cluster-representatives), train HGB with event-stratified split + SMOTE → compare precision / recall / F1
  5. PCA exploration: cumulative explained variance, loadings of the first 3 principal components

Output
------
  fig_correlation_pearson_<stamp>.png    : clustered Pearson heatmap
  fig_correlation_spearman_<stamp>.png   : clustered Spearman heatmap
  fig_importances_<stamp>.png            : top-N HGB permutation importances, coloured by group
  fig_importances_grouped_<stamp>.png    : box plot of importances per feature group
  fig_subset_comparison_<stamp>.png      : macro F1 / per-class F1 vs subset size
  fig_pca_variance_<stamp>.png           : PCA cumulative explained variance
  fig_pca_loadings_<stamp>.png           : feature loadings for PC1 / PC2 / PC3
  feature_clusters_<stamp>.csv           : cluster assignment for every feature
  feature_importances_<stamp>.csv        : ranked importances + group + cluster
  subset_results_<stamp>.csv             : per-class F1 for every subset tested
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# All parameters to adapt are grouped here
# =============================================================================

# -- Input CSV (output of script 04a) -----------------------------------------
CSV_PATH = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\04a_spectrogram_sta_lta_catalog\all-99-features-recent\catalog_windows_20260707_165719.csv"

# -- Output directory ----------------------------------------------------------
OUTPUT_DIR = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\03b_feature_selection"

# -- Classes to keep -----------------------------------------------------------
TARGET_CLASSES = ["earthquake", "rockslide", "ice quake"]

# -- Quality filtering ---------------------------------------------------------
FILTER_QUALITY = True   # True → keep only quality_ok == True rows

# -- Correlation clustering ----------------------------------------------------
# Features whose pairwise |r| > CORR_THRESHOLD are placed in the same cluster lower value = finer clusters (more features considered distinct)
CORR_THRESHOLD = 0.70   # |Pearson r| above which two features are "redundant"

# -- HGB training (same config as 06b, now the reference classifier) ----------
TEST_SIZE         = 0.20
RANDOM_STATE      = 42
USE_SMOTE         = True
SMOTE_K           = 5     # k_neighbors for SMOTE — must match 06b SMOTE_K
HGB_N_ESTIMATORS  = 200
HGB_MAX_DEPTH     = 6
HGB_LEARNING_RATE = 0.1

# -- Permutation importance (HGB has no built-in MDI like RF) -----------------
# n_repeats: how many times each feature is shuffled — higher = more reliable but slower
#  -> 10 good balance for ~6000 test rows × 99 features on cluster
N_PERMUTATION_REPEATS = 10

# -- Feature subsets to test ---------------------------------------------------
SUBSET_CONFIGS = [        # each entry is (label_for_plot, n_features_or_sentinel)
    ("Top 20",   20),
    ("Top 40",   40),
    ("Top 60",   60),
    ("All",      "all"),  # "all" → auto-resolved to n_features after the CSV is loaded (99 or 103)
    ("Clusters", None),   # None  → cluster representatives (one best feature per correlation cluster)
]

# -- Top-N importances to display in the bar chart ----------------------------
N_PLOT_IMPORTANCES = 10



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
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm
import matplotlib.ticker as mticker

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.decomposition import PCA

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("[WARN] imbalanced-learn not found. SMOTE will be disabled.")
    USE_SMOTE = False

import joblib

sys.path.insert(0, os.path.dirname(__file__))
from features import (
    FEATURE_NAMES, FEATURE_NAMES_3C, FEATURE_GROUPS, FEATURE_GROUPS_3C,
    POLARIZATION_NAMES, feature_group_array, get_feature_group, rename_legacy_columns,
)
from run_setup import create_run_dir, setup_logging, set_matplotlib_defaults


# ---- Run directory + logging ------------------------------------------------
RUN_DIR, STAMP = create_run_dir(OUTPUT_DIR)
log_file, _    = setup_logging(RUN_DIR, "03b_feature_selection.py",
                               extra_info=f"CSV: {CSV_PATH}")
set_matplotlib_defaults()

# Group colour palette — built after the feature set is auto-detected from the CSV
# (GROUP_NAMES, GROUP_COLORS, FEAT_GROUP_ARRAY are finalised in Section 3)
GROUP_PALETTE = plt.cm.tab10.colors          # up to 10 distinct colours



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
rename_legacy_columns(df_raw)
print(f"Loaded {len(df_raw):,} rows × {df_raw.shape[1]} columns.")

# -- Keep only target classes -------------------------------------------------
df_raw = df_raw[df_raw["event_type"].isin(TARGET_CLASSES)].copy()
print(f"After class filter ({TARGET_CLASSES}): {len(df_raw):,} rows.")

# -- Quality filter -----------------------------------------------------------
if FILTER_QUALITY and "quality_ok" in df_raw.columns:
    n_before = len(df_raw)
    df_raw = df_raw[df_raw["quality_ok"] == True].copy()
    print(f"After quality filter: {len(df_raw):,} rows kept  "
          f"({n_before - len(df_raw):,} dropped).")
elif FILTER_QUALITY:
    # Fall back to SNR thresholds if quality_ok column is absent
    snr_cols = {"SNR_full_mean": 1.856, "SNR_s2n_median": 10.503}
    available = {c: t for c, t in snr_cols.items() if c in df_raw.columns}
    if available:
        n_before = len(df_raw)
        mask = np.ones(len(df_raw), dtype=bool)
        for col, thr in available.items():
            mask &= (df_raw[col] >= thr).values
        df_raw = df_raw[mask].copy()
        print(f"After quality filter (SNR thresholds): {len(df_raw):,} rows kept  "
              f"({n_before - len(df_raw):,} dropped).")
    else:
        print("[WARN] No quality column found — skipping quality filter.")

# -- Auto-detect feature set (99 Z-only or 103 Z + polarization) --------------
_has_3c     = all(f in df_raw.columns for f in POLARIZATION_NAMES)
_feat_names = FEATURE_NAMES_3C if _has_3c else FEATURE_NAMES
n_features  = len(_feat_names)
_feat_groups = FEATURE_GROUPS_3C if _has_3c else FEATURE_GROUPS
print(f"\n  Feature set : {n_features} features  "
      f"({'Z + polarization (3C)' if _has_3c else 'Z only (1C)'})")

# -- Require all Z-component features -----------------------------------------
missing = [f for f in FEATURE_NAMES if f not in df_raw.columns]
if missing:
    print(f"[ERROR] {len(missing)} Z-feature columns not found (e.g. {missing[:3]}). Exiting.")
    sys.exit(1)

# Drop rows with NaN in Z-component features only
# Polarization NaN rows (~3-5 % of rows when LOAD_3C=True) are kept; they will be median-imputed in Section 4 before SMOTE and classifier training
z_feat_cols = [f for f in FEATURE_NAMES if f in df_raw.columns]
df_raw = df_raw.dropna(subset=z_feat_cols).copy()
print(f"After dropping NaN-feature rows: {len(df_raw):,} rows.")

# -- Group colour setup (depends on whether 3C features are present) ----------
GROUP_NAMES       = list(_feat_groups.keys())
GROUP_COLORS      = {g: GROUP_PALETTE[i % 10] for i, g in enumerate(GROUP_NAMES)}
FEAT_GROUP_ARRAY  = feature_group_array(use_3c=_has_3c)
FEAT_COLORS_ARRAY = [GROUP_COLORS[g] for g in FEAT_GROUP_ARRAY]

# Print class distribution
print(f"\n{'─'*45}")
print("  CLASS DISTRIBUTION")
print(f"{'─'*45}")
for cls, n in df_raw["event_type"].value_counts().items():
    print(f"  {cls:<20s}  {n:6,}  ({100*n/len(df_raw):.1f} %)")

# -- Feature matrix and labels ------------------------------------------------
y_all = df_raw["event_type"].values



# =============================================================================
# SECTION 4 — TRAIN / TEST SPLIT (shared across all experiments)
# =============================================================================

print(f"\n{'='*65}")
print(f"  STEP 2 — Train / test split  (by event)")
print(f"{'='*65}")

# Use drop_duplicates (same approach as 06b) — one row per event keeps the first
# occurrence of event_type, which is consistent since each event has a single type.
event_info = (
    df_raw[["event_time", "event_type"]]
    .drop_duplicates("event_time")
)

train_evt, test_evt = train_test_split(
    event_info["event_time"],
    test_size    = TEST_SIZE,
    stratify     = event_info["event_type"],
    random_state = RANDOM_STATE,
)
train_evt = set(train_evt);  test_evt = set(test_evt)

df_train = df_raw[df_raw["event_time"].isin(train_evt)].copy()
df_test  = df_raw[df_raw["event_time"].isin(test_evt)].copy()

print(f"Train : {len(train_evt):4d} events  →  {len(df_train):6,} rows")
print(f"Test  : {len(test_evt):4d}  events  →  {len(df_test):6,} rows")

X_train_full = df_train[_feat_names].values.astype(np.float32)
y_train      = df_train["event_type"].values
X_test_full  = df_test[_feat_names].values.astype(np.float32)
y_test       = df_test["event_type"].values

# Impute NaN with training-set median (fits on train only — no leakage)
# Only affects polarization features when _has_3c=True and some rows lacked horizontal channels.  SMOTE and classifiers need NaN-free arrays.
_imputer     = SimpleImputer(strategy="median")
X_train_full = _imputer.fit_transform(X_train_full)
X_test_full  = _imputer.transform(X_test_full)

# SMOTE on the full feature training set — used ONLY for the baseline HGB (permutation importance in Section 6) 
# Subset experiments (Section 7) each apply SMOTE on their own feature columns so the interpolation happens in the same feature space as 06b does
if USE_SMOTE:
    k_nb = min(SMOTE_K, pd.Series(y_train).value_counts().min() - 1)
    if k_nb >= 1:
        print(f"\nApplying SMOTE (k_neighbors={k_nb}) on full training set  "
              f"[used for baseline HGB / permutation importance only] ...")
        sm = SMOTE(sampling_strategy="auto", k_neighbors=k_nb,
                   random_state=RANDOM_STATE)
        X_train_full_sm, y_train_sm = sm.fit_resample(X_train_full, y_train)
        print(f"After SMOTE: {X_train_full_sm.shape[0]:,} rows")
        for cls, n in pd.Series(y_train_sm).value_counts().items():
            print(f"  {cls:<20s}  {n:6,}")
    else:
        print("[WARN] SMOTE disabled — smallest class has < 2 training rows.")
        X_train_full_sm, y_train_sm = X_train_full, y_train
else:
    X_train_full_sm, y_train_sm = X_train_full, y_train



# =============================================================================
# SECTION 5 — CORRELATION ANALYSIS
# =============================================================================

print(f"\n{'='*65}")
print(f"  STEP 3 — Correlation analysis")
print(f"{'='*65}")

df_feats = df_train[_feat_names]

print("  Computing Pearson correlation matrix ...")
corr_pearson  = np.clip(
    np.nan_to_num(df_feats.corr(method='pearson').values,  nan=0.0), -1, 1
)
print("  Computing Spearman correlation matrix ...")
corr_spearman = np.clip(
    np.nan_to_num(df_feats.corr(method='spearman').values, nan=0.0), -1, 1
)
print(f"  Correlation matrices computed ({n_features} × {n_features}).")

# -- Hierarchical clustering on Pearson |r| -----------------------------------
print(f"\n  Clustering features  (threshold |r| > {CORR_THRESHOLD} = same cluster) ...")

dist_matrix  = 1.0 - np.abs(corr_pearson)   # distance = 1 - |r|  so that perfectly correlated features (r=±1) → dist=0
np.fill_diagonal(dist_matrix, 0.0)          # every feature has distance 0 with itself on the diag

dist_condensed = squareform(dist_matrix, checks=False)  # squareform converts symmetric matrix → condensed 1D for linkage
Z              = linkage(dist_condensed, method='ward') # builds a dendrogram (leaves are individual features and branches merge similar features into groups)

# any 2 features that would merge at a distance below 0.30 (i.e. |r| > 0.70) end up in the same cluster
cluster_labels = fcluster(Z, t=1.0 - CORR_THRESHOLD, criterion='distance')  # fcluster cut the dendrogram to extract clusters
n_clusters     = cluster_labels.max()
print(f"  → {n_clusters} clusters found at |r| threshold = {CORR_THRESHOLD}")

# Feature cluster table: one row per feature, three columns: its name, its semantic group (waveform shape, spectral, ...), and its cluster ID
cluster_df = pd.DataFrame({
    "feature"    : _feat_names,
    "group"      : FEAT_GROUP_ARRAY,
    "cluster_id" : cluster_labels,
}).sort_values(["cluster_id", "group"])

# Print summary: large clusters
big_clusters = cluster_df.groupby("cluster_id").filter(lambda x: len(x) >= 3)  # keeps only the rows that belong to a cluster of size 3 or more
for cid, sub in big_clusters.groupby("cluster_id"):             # cid = cluster integer ID
    print(f"\n  Cluster {cid:2d}  ({len(sub)} features):")
    for feat in sub["feature"].tolist():
        print(f"    {feat}")

# -- Save cluster CSV (will be enriched with importances in Section 6) --------
cluster_csv = os.path.join(RUN_DIR, f"feature_clusters_{STAMP}.csv")
cluster_df.to_csv(cluster_csv, index=False)
print(f"\n[SAVED] {cluster_csv}")


# ---- Helper: reorder features by hierarchical clustering ----------------
# Returns the leaf order from the dendrogram (same order clustermap uses)
def _leaf_order(Z, n):
    """ Return feature indices in dendrogram leaf order """
    from scipy.cluster.hierarchy import leaves_list
    return leaves_list(Z)

leaf_idx     = _leaf_order(Z, n_features)
feat_ordered = [_feat_names[i] for i in leaf_idx]


# ---- Plot helper: draw heatmap with group boundary rectangles ---------------
def _plot_corr_heatmap(corr_matrix, title, out_path, vmin=-1, vmax=1):
    """
    Plot a correlation heatmap reordered by hierarchical clustering.
    Works for both 99 (1C) and 103 (3C) feature sets.
    Feature names are shown (small font); coloured rectangles mark FEATURE_GROUP boundaries.
    """
    n = len(feat_ordered)   # dynamic — 99 or 103 depending on catalog
    # Reorder matrix rows + columns by dendrogram leaf order
    C = corr_matrix[np.ix_(leaf_idx, leaf_idx)]   # C[i, j] gives the correlation between the i-th and j-th features in dendrogram order

    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(C, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto", interpolation="none")

    # Axis tick labels — feature names in dendrogram order
    ax.set_xticks(range(n))
    ax.set_xticklabels(feat_ordered, rotation=90, fontsize=5, ha='center')
    ax.set_yticks(range(n))
    ax.set_yticklabels(feat_ordered, fontsize=5)

    # Draw group boundary lines (solid black)
    # Compute positions of group boundaries in the reordered axis
    group_positions = {}
    for i, fname in enumerate(feat_ordered):
        grp = get_feature_group(fname, use_3c=_has_3c)
        group_positions.setdefault(grp, []).append(i)

    prev_end = -0.5
    for grp in GROUP_NAMES:
        positions = group_positions.get(grp, [])
        if not positions:
            continue
        start = min(positions) - 0.5
        end   = max(positions) + 0.5
        # Draw boundary lines
        for boundary in [start, end]:
            ax.axhline(boundary, color='black', lw=0.8, alpha=0.6)
            ax.axvline(boundary, color='black', lw=0.8, alpha=0.6)
        # Group label on x-axis (centred on the group's features)
        mid = (min(positions) + max(positions)) / 2
        ax.text(mid, n + 1.5, grp.replace('\n', ' '),
                ha='center', va='top', fontsize=7,
                color=GROUP_COLORS[grp], fontweight='bold', rotation=0)

    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01,
                 label='Correlation coefficient')
    ax.set_title(title, fontsize=13, pad=20)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVED] {out_path}")


print("\n  Plotting Pearson heatmap ...")
_plot_corr_heatmap(
    corr_pearson,
    title    = (f"Pearson correlation — {n_features} features  "
                f"(reordered by hierarchical clustering)\n"
                f"Red=positive  Blue=negative  |  "
                f"Black lines = feature groups"),
    out_path = os.path.join(RUN_DIR, f"fig_correlation_pearson_{STAMP}.png"),
)

print("  Plotting Spearman heatmap ...")
_plot_corr_heatmap(
    corr_spearman,
    title    = (f"Spearman correlation — {n_features} features  "
                f"(reordered by hierarchical clustering)\n"
                f"Captures non-linear monotonic relationships"),
    out_path = os.path.join(RUN_DIR, f"fig_correlation_spearman_{STAMP}.png"),
)



# =============================================================================
# SECTION 6 — HGB FEATURE IMPORTANCES  (permutation importance)
# =============================================================================
# HGB does not expose MDI (Mean Decrease in Impurity) like Random Forest does
#  -> instead, use permutation importance: each feature is randomly shuffled in the TEST set and we measure the drop in macro F1
#  -> a large drop = the model relied heavily on that feature
#  -> this is evaluated on the test set (not train), so it is free of overfitting bias

print(f"\n{'='*65}")
print(f"  STEP 4 — HGB feature importances  (permutation, {N_PERMUTATION_REPEATS} repeats)")
print(f"{'='*65}")

print("  Training HGB on full feature set ...")
clf_full = HistGradientBoostingClassifier(
    max_iter      = HGB_N_ESTIMATORS,   # number of boosting rounds (= number of trees)
    max_depth     = HGB_MAX_DEPTH,      # maximum depth of each tree
    learning_rate = HGB_LEARNING_RATE,  # shrinks each tree's contribution → regularisation
    early_stopping = True,              # stops if validation loss stops improving
    n_iter_no_change = 15,              # patience: stop after 15 rounds without improvement
    random_state  = RANDOM_STATE,
)
clf_full.fit(X_train_full_sm, y_train_sm)
classes = list(clf_full.classes_)
print(f"  Classes (HGB order): {classes}")

# Evaluate baseline so the user can see the model quality before feature selection
from sklearn.metrics import accuracy_score
y_pred_base = clf_full.predict(X_test_full)
acc_base    = accuracy_score(y_test, y_pred_base)
rep_base    = classification_report(y_test, y_pred_base,
                                    labels=classes, target_names=classes,
                                    output_dict=True, zero_division=0)
print(f"  Baseline (all {n_features} features) — "
      f"Accuracy={acc_base:.3f}  Macro F1={rep_base['macro avg']['f1-score']:.3f}")

# -- Permutation importance on the TEST set -----------------------------------
# Scoring = 'f1_macro' so the drop matches our main evaluation metric.
# n_jobs=-1 parallelises across features.
print(f"\n  Computing permutation importance  "
      f"({N_PERMUTATION_REPEATS} repeats × {X_test_full.shape[1]} features) ...")
print(f"  [This may take 5-15 min on the cluster — parallelised on all cores]")

perm_result = permutation_importance(
    clf_full, X_test_full, y_test,
    n_repeats  = N_PERMUTATION_REPEATS,
    random_state = RANDOM_STATE,
    n_jobs     = -1,
    scoring    = "f1_macro",   # macro F1 = our target metric
)
importances      = perm_result.importances_mean   # mean drop across repeats
importances_std  = perm_result.importances_std    # std across repeats (measure of stability)

# Build importance DataFrame
imp_df = pd.DataFrame({
    "feature"         : _feat_names,
    "importance"      : importances,
    "importance_std"  : importances_std,
    "group"           : FEAT_GROUP_ARRAY,
    "cluster_id"      : cluster_labels,
}).sort_values("importance", ascending=False).reset_index(drop=True)
imp_df["rank"] = imp_df.index + 1

# Print top 20
print(f"\n  Top-20 feature importances  (mean Macro F1 drop when shuffled):")
for _, row in imp_df.head(20).iterrows():
    print(f"    {row['rank']:3d}. {row['feature']:<32s}  "
          f"{row['importance']:+.4f} ± {row['importance_std']:.4f}  "
          f"[{row['group'].replace(chr(10),' ')}]")

# Save importance CSV (merged with cluster info)
imp_csv = os.path.join(RUN_DIR, f"feature_importances_{STAMP}.csv")
imp_df.to_csv(imp_csv, index=False)
print(f"\n[SAVED] {imp_csv}")

# -- Identify cluster representatives: best-importance feature per cluster ---
rep_df = (
    imp_df.sort_values("importance", ascending=False)
    .groupby("cluster_id", sort=False)  # groups all 99 rows by their cluster ID
    .first()                            # takes the first row from each group
    .reset_index()
)

# Sorts the cluster representatives themselves by importance (most important first)
cluster_reps = rep_df.sort_values("importance", ascending=False)["feature"].tolist()
print(f"\n  Cluster representatives ({len(cluster_reps)} features, one per cluster):")
for i, fname in enumerate(cluster_reps[:20]):
    grp = get_feature_group(fname)
    imp = imp_df.loc[imp_df["feature"] == fname, "importance"].values[0]
    print(f"    {i+1:3d}. {fname:<32s}  imp={imp:+.4f}  [{grp.replace(chr(10),' ')}]")
if len(cluster_reps) > 20:
    print(f"    … and {len(cluster_reps)-20} more")


# -- Figure: top-N importances bar chart with error bars, coloured by group ---
top_n   = imp_df.head(N_PLOT_IMPORTANCES)
colours = [GROUP_COLORS[g] for g in top_n["group"]]

fig, ax = plt.subplots(figsize=(10, N_PLOT_IMPORTANCES * 0.32 + 1.5))
y_pos = np.arange(N_PLOT_IMPORTANCES)
ax.barh(y_pos, top_n["importance"][::-1],
        xerr=top_n["importance_std"][::-1],
        color=colours[::-1], edgecolor='white', linewidth=0.4,
        error_kw=dict(ecolor='grey', lw=0.8, capsize=2))
ax.set_yticks(y_pos)
ax.set_yticklabels(top_n["feature"][::-1].tolist(), fontsize=8)
ax.axvline(0, color='black', lw=0.8, ls='--', alpha=0.5)   # zero line: negative = feature hurts
ax.set_xlabel("Permutation importance  (mean Macro F1 drop when feature is shuffled)")
ax.set_title(f"Top-{N_PLOT_IMPORTANCES} HGB feature importances\n"
             f"(permutation on test set, {N_PERMUTATION_REPEATS} repeats  |  "
             f"error bars = ±1 std across repeats)")
# Legend for groups
legend_patches = [
    mpatches.Patch(facecolor=GROUP_COLORS[g], label=g.replace('\n', ' '))
    for g in GROUP_NAMES if g in top_n["group"].values
]
ax.legend(handles=legend_patches, fontsize=8, loc='lower right')
plt.tight_layout()
path = os.path.join(RUN_DIR, f"fig_importances_{STAMP}.png")
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\n[SAVED] {path}")


# -- Figure: importance distribution per group (box plot) ---------------------
groups_with_data = [g for g in GROUP_NAMES if g in imp_df["group"].values]
data_per_group   = [
    imp_df.loc[imp_df["group"] == g, "importance"].values
    for g in groups_with_data
]
short_labels = [g.replace('\n', ' ') for g in groups_with_data]

fig, ax = plt.subplots(figsize=(11, 5))
bp = ax.boxplot(data_per_group, patch_artist=True, vert=True,
                medianprops=dict(color='black', lw=2))
for patch, grp in zip(bp['boxes'], groups_with_data):
    patch.set_facecolor(GROUP_COLORS[grp])
    patch.set_alpha(0.8)
ax.set_xticks(range(1, len(groups_with_data) + 1))
ax.set_xticklabels(short_labels, rotation=20, ha='right', fontsize=9)
ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
ax.set_ylabel("Permutation importance  (mean Macro F1 drop)")
ax.set_title("HGB permutation importance distribution by semantic group\n"
             "Boxes = IQR  |  Line = median  |  Whiskers = 1.5×IQR")
ax.grid(axis='y', lw=0.4, alpha=0.4)
plt.tight_layout()
path = os.path.join(RUN_DIR, f"fig_importances_grouped_{STAMP}.png")
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"[SAVED] {path}")



# =============================================================================
# SECTION 7 — FEATURE SUBSET EXPERIMENTS
# For each subset: select features → train HGB → evaluate on test set
# =============================================================================

print(f"\n{'='*65}")
print(f"  STEP 5 — Feature subset experiments")
print(f"{'='*65}")

# Ordered list of feature names by HGB permutation importance (used for top-N subsets)
features_by_importance = imp_df["feature"].tolist()


def _run_subset(label, feature_subset):
    """
    Train and evaluate HGB on feature_subset.
    SMOTE is applied on the subset columns (same feature space as 06b).
    Returns a dict with macro/per-class F1 and accuracy.
    """
    idx = [_feat_names.index(f) for f in feature_subset]
    Xtr_raw = X_train_full[:, idx]   # pre-SMOTE (but already imputed), subset of columns
    Xte     = X_test_full[:, idx]

    # Apply SMOTE in the subset feature space — identical to 06b's approach
    if USE_SMOTE:
        k_nb = min(SMOTE_K, pd.Series(y_train).value_counts().min() - 1)
        sm   = SMOTE(sampling_strategy="auto", k_neighbors=k_nb,
                     random_state=RANDOM_STATE)
        Xtr, ytr = sm.fit_resample(Xtr_raw, y_train)
    else:
        Xtr, ytr = Xtr_raw, y_train

    clf = HistGradientBoostingClassifier(
        max_iter         = HGB_N_ESTIMATORS,
        max_depth        = HGB_MAX_DEPTH,
        learning_rate    = HGB_LEARNING_RATE,
        early_stopping   = True,
        n_iter_no_change = 15,
        random_state     = RANDOM_STATE,
    )
    clf.fit(Xtr, ytr)
    y_pred = clf.predict(Xte)

    report = classification_report(y_test, y_pred,
                                   labels=classes, target_names=classes,
                                   output_dict=True, zero_division=0)
    macro_f1 = report["macro avg"]["f1-score"]
    accuracy  = report["accuracy"]

    result = {
        "subset"    : label,
        "n_features": len(feature_subset),
        "macro_f1"  : round(macro_f1, 4),
        "accuracy"  : round(accuracy, 4),
    }
    for cls in classes:
        safe = cls.replace(" ", "_")
        result[f"f1_{safe}"]        = round(report[cls]["f1-score"], 4)
        result[f"precision_{safe}"] = round(report[cls]["precision"], 4)
        result[f"recall_{safe}"]    = round(report[cls]["recall"], 4)

    return result


# Build feature lists for each subset config
subset_results = []

for label, n_feat in SUBSET_CONFIGS:
    if n_feat is None:
        # Cluster representatives — one best-importance feature per correlation cluster
        feat_list    = cluster_reps
        actual_label = f"Clusters\n({len(feat_list)} feat.)"
    elif n_feat == "all":
        # All features from the catalog (n_features = 99 or 103 depending on LOAD_3C in 04a)
        feat_list    = features_by_importance   # all ranked
        actual_label = f"All {n_features}\n({n_features} feat.)"
    else:
        feat_list    = features_by_importance[:n_feat]
        actual_label = label

    print(f"\n  [{actual_label.replace(chr(10),' ')}]  "
          f"({len(feat_list)} features) — training ...")
    res = _run_subset(actual_label, feat_list)
    subset_results.append(res)

    print(f"    Accuracy : {res['accuracy']:.3f}   "
          f"Macro F1 : {res['macro_f1']:.3f}")
    for cls in classes:
        safe = cls.replace(" ", "_")
        print(f"    {cls:<20s}  "
              f"F1={res[f'f1_{safe}']:.3f}  "
              f"P={res[f'precision_{safe}']:.3f}  "
              f"R={res[f'recall_{safe}']:.3f}")

df_subsets = pd.DataFrame(subset_results)
subset_csv = os.path.join(RUN_DIR, f"subset_results_{STAMP}.csv")
df_subsets.to_csv(subset_csv, index=False)
print(f"\n[SAVED] {subset_csv}")


# -- Figure: subset comparison ------------------------------------------------
# Two panels: macro F1 (left) and per-class F1 (right)
subset_labels = [r["subset"].replace('\n', ' ') for r in subset_results]
x             = np.arange(len(subset_results))

class_colors = {
    "earthquake": "#2166ac",
    "rockslide" : "#d6604d",
    "ice quake" : "#4dac26",
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Left panel: macro F1 ---
macro_vals = [r["macro_f1"] for r in subset_results]
bars = ax1.bar(x, macro_vals, color="#7570b3", edgecolor='white', width=0.6)
for bar, val in zip(bars, macro_vals):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f"{val:.3f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(subset_labels, fontsize=9)
ax1.set_ylabel("Macro F1-score")
ax1.set_ylim(0, 1.0)
ax1.set_title("Macro F1 by feature subset  (HGB)\n(equal weight to all classes)")
_all_idx = next((i for i, c in enumerate(SUBSET_CONFIGS) if c[1] == "all"), None)
if _all_idx is not None and _all_idx < len(macro_vals):
    ax1.axhline(macro_vals[_all_idx], color='grey', lw=1.2, ls='--',
                label=f'All {n_features} features')
ax1.grid(axis='y', lw=0.4, alpha=0.4)
ax1.legend(fontsize=8)

# --- Right panel: per-class F1 grouped bars ---
n_classes = len(classes)
bar_width  = 0.22
offsets    = np.linspace(-(n_classes - 1) / 2, (n_classes - 1) / 2, n_classes) * bar_width

for j, cls in enumerate(classes):
    safe = cls.replace(" ", "_")
    vals = [r[f"f1_{safe}"] for r in subset_results]
    bars2 = ax2.bar(x + offsets[j], vals, width=bar_width,
                    color=class_colors.get(cls, 'grey'), label=cls,
                    edgecolor='white', alpha=0.85)
    for bar, val in zip(bars2, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.2f}", ha='center', va='bottom', fontsize=7, rotation=60)

ax2.set_xticks(x)
ax2.set_xticklabels(subset_labels, fontsize=9)
ax2.set_ylabel("F1-score per class")
ax2.set_ylim(0, 1.15)
ax2.set_title("Per-class F1 by feature subset  (HGB)")
ax2.legend(fontsize=9)
ax2.grid(axis='y', lw=0.4, alpha=0.4)

fig.suptitle("Feature subset experiments — HGB classifier  "
             "(does reducing the feature set hurt accuracy?)",
             fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
path = os.path.join(RUN_DIR, f"fig_subset_comparison_{STAMP}.png")
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"[SAVED] {path}")



# =============================================================================
# SECTION 9 — SUMMARY + CONCLUSIONS
# =============================================================================

print(f"\n{'='*65}")
print(f"  SUMMARY")
print(f"{'='*65}")

print(f"\n  Correlation clusters (|Pearson r| > {CORR_THRESHOLD}):")
print(f"    {n_clusters} clusters found from {n_features} features")
sizes = cluster_df.groupby("cluster_id").size().value_counts().sort_index()
for size, count in sizes.items():
    print(f"    {count:3d} cluster(s) of size {size}")

print(f"\n  HGB feature importances (permutation, full set):")
print(f"    Top feature: {imp_df.iloc[0]['feature']}  "
      f"(importance={imp_df.iloc[0]['importance']:+.4f})")
print(f"    Group with highest median importance: "
      f"{imp_df.groupby('group')['importance'].median().idxmax().replace(chr(10),' ')}")

print(f"\n  Subset experiment results:")
header_classes = "  ".join(f"{c.replace(' ','_')[:8]:>10}" for c in classes)
print(f"  {'Subset':<20s}  {'n_feat':>6}  {'Macro F1':>8}  {'Accuracy':>9}  {header_classes}")
print("  " + "-" * 70)
for res in subset_results:
    line = (f"  {res['subset'].replace(chr(10),' '):<20s}  "
            f"{res['n_features']:>6}  {res['macro_f1']:>8.3f}  "
            f"{res['accuracy']:>9.3f}")
    for cls in classes:
        safe = cls.replace(" ", "_")
        line += f"  {res[f'f1_{safe}']:>10.3f}"
    print(line)

print(f"\n  Cluster representatives ({len(cluster_reps)} features):")
print(f"    {cluster_reps[:10]}")
if len(cluster_reps) > 10:
    print(f"    … {len(cluster_reps) - 10} more (see {os.path.basename(imp_csv)})")

print(f"\n  All outputs saved to: {RUN_DIR}")


# =============================================================================
# END
# =============================================================================

from datetime import datetime
print("\n" + "=" * 70)
print(f"  Run finished  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Run folder    : {RUN_DIR}")
print(f"  Log file      : {_}")
print("=" * 70)

log_file.close()
