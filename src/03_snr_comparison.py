"""
SNR COMPARISON DIAGNOSTIC
==========================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : April 2026

Goal
----
Compare the 7 SNR metrics computed by script 02 to identify which one best separates
high-quality detections from poor ones.

Ground truth: origin_inside_det
  True  -> catalog origin time falls inside the detected window (good detection)
  False -> origin time is outside the window (missed onset)

Ground truth: pick_inside_det
  True  -> P-wave pick time falls inside the detected window
  False -> P-wave pick time is outside the window 

Analyses
--------
  3.1  Basic distribution statistics per metric (mean, median, std, IQR)
       + mean for good vs bad detections separately
  3.2  Pearson correlation matrix between the 7 metrics
       (are some metrics redundant?)
  3.3  ROC curves + AUC for each metric
       (which metric best discriminates good from bad detections?)
  3.4  Youden J optimal threshold per metric
  3.5  Threshold sensitivity: pass rate, TPR, FPR vs threshold value
  3.6  Per-station and per-event-type summary
  3.7  Save summary CSV

Input
-----
  catalog_windows_<stamp>.csv  (output of script 02)
  ISTerre SDS archive          (for waveform panel, cluster only)

Output
------
  snr_summary_<stamp>.csv               : per-metric table (AUC, best threshold, ...)
  fig_distributions_<stamp>.png         : histograms per metric, blue=good / red=bad
  fig_correlation_<stamp>.png           : Pearson correlation heatmap
  fig_roc_<stamp>.png                   : ROC curves for all 7 metrics
  fig_threshold_sensitivity_<stamp>.png : pass rate / TPR / FPR vs threshold
  fig_boxplots_type_<stamp>.png         : per event-type boxplots
  fig_station_heatmap_<stamp>.png       : per-station mean SNR values
  fig_waveforms_<stamp>.png             : waveform panel for sampled detections
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# All parameters to adapt are grouped here
# =============================================================================

# -- Input CSV (output of script 02) ------------------------------------------
# Leave INPUT_CSV = "" to auto-detect the most recent catalog_windows_*.csv in SEARCH_DIR
INPUT_CSV  = ""
SEARCH_DIR = "/data/failles/louisels/project/results/outputs_02"

# -- Paths --------------------------------------------------------------------
SDS_ROOT    = "/data/sig/SDS"
ISTERRE_URL = "http://ist-sc3-geobs.osug.fr:8080"
OUTPUT_DIR  = "/data/failles/louisels/project/results/outputs_03"

# -- Key diagnostic -----------------------------------------------------------
GROUND_TRUTH = 'pick_inside_det'  # origin_inside_det or pick_inside_det

# -- SNR metrics to compare ---------------------------------------------------
SNR_METRICS = [
    'SNR',
    'SNR_picking_5_5',
    'SNR_picking_3_3',
    'SNR_picking_1_3',
    'SNR_full_mean',
    'SNR_full_median',
    'SNR_s2n_median',
]
# Short labels used in plot axes and legends
SNR_SHORT = {
    'SNR'              : 'SNR\n(peak/noise)',
    'SNR_picking_5_5'  : 'pick_5-5\n(±5s)',
    'SNR_picking_3_3'  : 'pick_3-3\n(±3s)',
    'SNR_picking_1_3'  : 'pick_1-3\n(1s/3s)',
    'SNR_full_mean'    : 'full_mean',
    'SNR_full_median'  : 'full_median',
    'SNR_s2n_median'   : 's2n_median\n(robust)',
}
# Full descriptions for figure titles
SNR_LONG = {
    'SNR'              : 'SNR  (peak-centred 5s / 5s post-event noise)',
    'SNR_picking_5_5'  : 'SNR_picking_5-5  (5s after onset / 5s before onset)',
    'SNR_picking_3_3'  : 'SNR_picking_3-3  (3s after onset / 3s before onset)',
    'SNR_picking_1_3'  : 'SNR_picking_1-3  (1s after onset / 3s before onset)',
    'SNR_full_mean'    : 'SNR_full_mean  (mean envelope: signal window / noise window)',
    'SNR_full_median'  : 'SNR_full_median  (median envelope: signal window / noise window)',
    'SNR_s2n_median'   : 'SNR_s2n_median  (99.5th percentile signal / MAD noise)',
}

# -- Threshold sensitivity sweep (section 3.5) --------------------------------
THRESHOLD_RANGE = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0]

# -- Waveform panel -----------------------------------------------------------
N_WAVEFORM_SAMPLES   = 20             # number of traces shown (spread across the SNR range)
WAVEFORM_RANK_METRIC = 'SNR_full_mean'# metric used to rank detections in the panel
WAVEFORM_PAD_S       = 10.0           # seconds added before/after the detection window
WAVEFORM_FREQMIN     = 1.0            # Hz — bandpass filter for display
WAVEFORM_FREQMAX     = 20.0           # Hz
WAVEFORM_CHANNEL     = "??Z"          # channel wildcard for SDS query



# =============================================================================
# SECTION 2 — SETUP
# =============================================================================

import os
import sys
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from obspy import UTCDateTime
from sklearn.metrics import roc_curve, auc as sklearn_auc
from run_setup import (
    create_run_dir,
    setup_logging,
    connect_sds,
    set_matplotlib_defaults,
)


# ----------- Run setup ---------------
RUN_DIR, _RUN_STAMP = create_run_dir(OUTPUT_DIR)
_log_file, _log_filename = setup_logging(
    RUN_DIR, "03_snr_comparison.py",
    extra_info=f"Ground truth: {GROUND_TRUTH} | Metrics: {SNR_METRICS}"
)
set_matplotlib_defaults()


# ----------- Locate input CSV --------
if INPUT_CSV and os.path.isfile(INPUT_CSV):
    csv_path = INPUT_CSV
else:
    # Auto-detect the most recent catalog_windows_*.csv produced by script 02
    pattern    = os.path.join(SEARCH_DIR, "**", "catalog_windows_*.csv")
    candidates = sorted(glob.glob(pattern, recursive=True))
    if not candidates:
        print(f"[ERROR] No catalog_windows_*.csv found under {SEARCH_DIR}.")
        print(f"        Set INPUT_CSV manually in SECTION 1.")
        sys.exit(1)
    csv_path = candidates[-1]   # most recent file
    print(f"[AUTO] Using most recent CSV:\n       {csv_path}")

print(f"\nLoading CSV ...")
df_all = pd.read_csv(csv_path)
print(f"  {len(df_all)} rows x {df_all.shape[1]} columns")

# Keep only SNR metrics that actually exist in the CSV
SNR_METRICS = [m for m in SNR_METRICS if m in df_all.columns]
if not SNR_METRICS:
    print("[ERROR] None of the expected SNR columns found in the CSV. Exiting.")
    sys.exit(1)
print(f"  SNR metrics found : {SNR_METRICS}")

# Remove rows where every SNR metric is NaN (no usable data)
df = df_all.dropna(subset=SNR_METRICS, how='all').copy()
print(f"  {len(df)} rows kept  ({len(df_all) - len(df)} dropped — all SNR values were NaN)")

# Create the ground truth column: True = good detection, False = missed onset
if 'origin_inside_det' not in df.columns:
    print("[ERROR] 'origin_inside_det' column not found. Exiting.")
    sys.exit(1)
df['label'] = df[GROUND_TRUTH].astype(bool)

n_pos = df['label'].sum()          # number of good detections
n_neg = (~df['label']).sum()       # number of bad detections
print(f"\n  Ground truth ({GROUND_TRUTH}):")
print(f"    True  — inside window  : {n_pos}  ({100*n_pos/len(df):.1f}%)")
print(f"    False — outside window : {n_neg}  ({100*n_neg/len(df):.1f}%)")
print(f"\n  Rows by event type:")
for et, cnt in df['event_type'].value_counts().items():
    print(f"    {et:<22s} : {cnt}")


# ----------- SDS connection (needed only for waveform panel) -----------------
client_sds = connect_sds(SDS_ROOT)


# ----------- Color scheme used in all figures --------------------------------
C_POS  = '#2166ac'       # blue  — origin inside (good detection)
C_NEG  = '#d6604d'       # red   — origin outside (bad detection)
CMAP10 = plt.cm.tab10.colors   # 10 distinct colors, one per metric



# =============================================================================
# SECTION 3 — STATISTICAL ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("  STATISTICAL ANALYSIS")
print("=" * 70)


# --- 3.1  Basic distribution statistics per metric ---------------------------
#
# For each SNR metric we compute:
#   - overall stats: mean, median, std, IQR (Q3-Q1)
#   - mean and median split by ground truth: good (label=True) vs bad (label=False)
#
# Key: difference between mean_inside and mean_outside
# if good detections have clearly higher SNR than bad ones

print("\n--- 3.1  Distribution statistics ---")

dist_rows = []
for metric in SNR_METRICS:
    col     = df[metric].dropna()       # .loc[condition, column] keeps only the rows where the condition is True
    col_pos = df.loc[ df['label'], metric].dropna()   # SNR values for good detections
    col_neg = df.loc[~df['label'], metric].dropna()   # SNR values for bad detections

    row = {
        'metric'         : metric,
        'n_valid'        : len(col),
        'n_nan'          : int(df[metric].isna().sum()),
        'mean'           : round(col.mean(), 2),
        'median'         : round(col.median(), 2),
        'std'            : round(col.std(), 2),            # sample standard deviation: quantifies the variation in a dataset
        'Q1'             : round(col.quantile(0.25), 2),   # 25th percentile (returns the value below which 25% of the data falls)
        'Q3'             : round(col.quantile(0.75), 2),   # 75th percentile
        'IQR'            : round(col.quantile(0.75) - col.quantile(0.25), 2), # range of the middle 50% of values
        'mean_inside'    : round(col_pos.mean(), 2),       # good detections
        'mean_outside'   : round(col_neg.mean(), 2),       # bad detections
        'median_inside'  : round(col_pos.median(), 2),
        'median_outside' : round(col_neg.median(), 2),
    }
    dist_rows.append(row)

    print(f"\n  {metric}")
    print(f"    All   :  mean={row['mean']}  median={row['median']}  "
          f"std={row['std']}  IQR=[{row['Q1']}, {row['Q3']}]")
    print(f"    Inside  (good) :  mean={row['mean_inside']}   median={row['median_inside']}")
    print(f"    Outside (bad)  :  mean={row['mean_outside']}  median={row['median_outside']}")

df_dist = pd.DataFrame(dist_rows)  # converts the list of 7 dictionaries into a table


# --- 3.2  Pearson correlation matrix -----------------------------------------
#
# Pearson correlation r between two metrics ranges from -1 to +1
# r close to 1 means the two metrics move together (linearly) -> carry redundant information

print("\n--- 3.2  Pearson correlation matrix ---")

corr = df[SNR_METRICS].dropna().corr(method='pearson')
print(corr.round(2).to_string())


# --- 3.3  ROC curves and AUC -------------------------------------------------
#
# For each metric we sweep all possible threshold values and compute:
#   TPR (True Positive Rate)  = fraction of good detections that pass the threshold
#   FPR (False Positive Rate) = fraction of bad  detections that also pass
#
# The ROC curve plots TPR vs FPR as the threshold moves from -inf to +inf
# AUC (Area Under Curve) summarises the whole curve in one number:
#   AUC = 1.0 -> perfect discrimination
#   AUC = 0.5 -> no better than random
#   AUC > 0.8 -> good discriminator

print("\n--- 3.3  ROC AUC per metric ---")

roc_results = {}
for metric in SNR_METRICS:
    valid = df[['label', metric]].dropna()
    if valid['label'].nunique() < 2:            # nunique() counts the number of distinct values (if only one class -> skip ROC)
        print(f"  {metric:<22s}  SKIP — only one class in this subset")
        continue

    fpr, tpr, thresholds = roc_curve(valid['label'].astype(int), valid[metric])
    auc_val = sklearn_auc(fpr, tpr)     # area under the ROC curve -> close to 1.0 = close to the elbow

    # TPR - FPR: the threshold that maximises this, is the best tradeoff
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)

    roc_results[metric] = {
        'fpr'             : fpr,
        'tpr'             : tpr,
        'auc'             : float(auc_val),
        'youden_threshold': float(thresholds[best_idx]),
        'youden_tpr'      : float(tpr[best_idx]),
        'youden_fpr'      : float(fpr[best_idx]),
        'youden_j'        : float(j_scores[best_idx]),
    }
    r = roc_results[metric]
    print(f"  {metric:<22s}  AUC={r['auc']:.3f}  "
            f"best threshold={r['youden_threshold']:.2f}  "
            f"-> TPR={r['youden_tpr']:.2f}  FPR={r['youden_fpr']:.2f}")


# --- 3.4  Threshold sensitivity ----------------------------------------------
#
# For each metric and each threshold value in THRESHOLD_RANGE, we compute:
#   pass_rate = fraction of ALL detections that pass (practical data retention)
#   TPR       = fraction of GOOD detections that pass (we want this high)
#   FPR       = fraction of BAD  detections that pass (we want this low)
#
# This is the practical complement to the ROC curve: instead of all possible
# thresholds, we look at specific round numbers (1, 2, 3 ...) that you might
# actually use as a quality gate.

print("\n--- 3.4  Threshold sensitivity ---")

sens_rows = []
for metric in SNR_METRICS:
    valid   = df[['label', metric]].dropna()
    n_pos_v = int(valid['label'].sum())    # how many good detections exist for this metric
    n_neg_v = int((~valid['label']).sum()) # bad detections
    for thr in THRESHOLD_RANGE:
        passing   = valid[metric] >= thr   # true for every detection whose SNR is above the threshold
        n_pass    = int(passing.sum())     # counts the trues
        pass_rate = n_pass / len(valid) if len(valid) > 0 else np.nan  # fraction of all detections that would survive this threshold
        tpr_val   = (passing &  valid['label']).sum() / n_pos_v if n_pos_v > 0 else np.nan  # detections that pass the threshold AND are good
        fpr_val   = (passing & ~valid['label']).sum() / n_neg_v if n_neg_v > 0 else np.nan  # detections that pass the threshold AND are bad
        sens_rows.append({
            'metric': metric, 'threshold': thr,
            'n_pass': n_pass, 'pass_rate': pass_rate,
            'tpr': tpr_val,   'fpr': fpr_val,
        })
df_sens = pd.DataFrame(sens_rows)


# --- 3.5  Per-station and per-event-type summaries ---------------------------
print("\n--- 3.5  Per-station mean SNR ---")
df_sta = df.groupby('station')[SNR_METRICS].mean().round(2)
print(df_sta.to_string())

print("\n--- 3.5  Per-event-type median SNR ---")
df_type = df.groupby('event_type')[SNR_METRICS].median().round(2)
print(df_type.to_string())


# --- 3.6  Save summary CSV ---------------------------------------------------
print("\n--- 3.6  Summary table ---")

summary_rows = []
for metric in SNR_METRICS:
    base = df_dist[df_dist['metric'] == metric].iloc[0]
    roc  = roc_results.get(metric, {})
    summary_rows.append({
        'metric'           : metric,
        'n_valid'          : int(base['n_valid']),
        'mean'             : base['mean'],
        'median'           : base['median'],
        'std'              : base['std'],
        'IQR'              : base['IQR'],
        'mean_inside'      : base['mean_inside'],    # good detections
        'mean_outside'     : base['mean_outside'],   # bad detections
        'auc'              : round(roc.get('auc', np.nan), 4),
        'best_threshold'   : round(roc.get('youden_threshold', np.nan), 3),
        'tpr_at_best_thr'  : round(roc.get('youden_tpr', np.nan), 3),
        'fpr_at_best_thr'  : round(roc.get('youden_fpr', np.nan), 3),
    })

df_summary = pd.DataFrame(summary_rows).sort_values('auc', ascending=False, na_position='last')
print(df_summary.to_string(index=False))

summary_path = os.path.join(RUN_DIR, f"snr_summary_{_RUN_STAMP}.csv")
df_summary.to_csv(summary_path, index=False)
print(f"\n[SAVED] {summary_path}")



# =============================================================================
# SECTION 4 — FIGURES
# =============================================================================

print("\n" + "=" * 70)
print("  GENERATING FIGURES")
print("=" * 70)

N_M = len(SNR_METRICS)


# ---- Figure 1: Distributions (histogram + KDE per metric) ------------------
print("\n  Fig 1: Distributions ...")

from scipy import stats as scipy_stats

ncols = min(4, N_M)
nrows = int(np.ceil(N_M / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = np.array(axes).flatten()

for k, metric in enumerate(SNR_METRICS):
    ax      = axes[k]
    col_pos = df.loc[ df['label'], metric].dropna()
    col_neg = df.loc[~df['label'], metric].dropna()

    # Clip x-axis at 99th percentile to avoid extreme outliers stretching the plot
    xmax = df[metric].quantile(0.99)
    bins = np.linspace(0, xmax, 40)

    ax.hist(col_neg, bins=bins, alpha=0.45, color=C_NEG, density=True,
            label=f'origin outside  (n={len(col_neg)})')
    ax.hist(col_pos, bins=bins, alpha=0.45, color=C_POS, density=True,
            label=f'origin inside   (n={len(col_pos)})')

    # Smooth KDE curve on top of each histogram
    x_kde = np.linspace(0, xmax, 300)
    for col_grp, color in [(col_neg, C_NEG), (col_pos, C_POS)]:
        grp = col_grp[(col_grp > 0) & (col_grp <= xmax)]
        if len(grp) > 5:
            try:
                kde = scipy_stats.gaussian_kde(grp)
                ax.plot(x_kde, kde(x_kde), color=color, lw=1.8)
            except Exception:
                pass

    # Vertical line at the Youden optimal threshold for this metric
    if metric in roc_results:
        thr = roc_results[metric]['youden_threshold']
        ax.axvline(thr, color='black', lw=1.4, ls='--',
                   label=f'best threshold = {thr:.2f}')

    auc_str = f"AUC = {roc_results[metric]['auc']:.3f}" if metric in roc_results else ""
    ax.set_title(f"{SNR_LONG.get(metric, metric)}\n{auc_str}", fontsize=8)
    ax.set_xlabel('SNR value', fontsize=8)
    ax.set_ylabel('Density', fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_xlim(left=0)

for idx in range(N_M, len(axes)):
    axes[idx].set_visible(False)

fig.suptitle(
    'SNR distributions — blue: origin inside window (good)  |  red: origin outside (bad)\n'
    'The more separated the two colors, the more useful the metric',
    fontsize=10
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
path = os.path.join(RUN_DIR, f"fig_distributions_{_RUN_STAMP}.png")
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"    [SAVED] {path}")


# ---- Figure 2: Pearson correlation heatmap ----------------------------------
print("  Fig 2: Correlation heatmap ...")

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
short_tick = [SNR_SHORT.get(m, m) for m in SNR_METRICS]
ax.set_xticks(range(N_M)); ax.set_xticklabels(short_tick, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(N_M)); ax.set_yticklabels(short_tick, fontsize=9)
for i in range(N_M):
    for j in range(N_M):
        val = corr.values[i, j]
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=9, color='white' if abs(val) > 0.65 else 'black')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title('Pearson correlation between SNR metrics\n'
             'Values close to 1 = metrics carry the same information (redundant)', fontsize=10)
plt.tight_layout()
path = os.path.join(RUN_DIR, f"fig_correlation_{_RUN_STAMP}.png")
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"    [SAVED] {path}")


# ---- Figure 3: ROC curves ---------------------------------------------------
print("  Fig 3: ROC curves ...")

if not roc_results:
    print("    [SKIP] No ROC results (sklearn unavailable).")
else:
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Random (AUC = 0.50)')
    for k, metric in enumerate(SNR_METRICS):
        if metric not in roc_results:
            continue
        r   = roc_results[metric]
        lbl = f"{SNR_SHORT.get(metric, metric).replace(chr(10),' ')}  (AUC={r['auc']:.3f})"
        ax.plot(r['fpr'], r['tpr'], lw=2.2, color=CMAP10[k % 10], label=lbl)
        # Diamond marks the Youden optimal threshold for this metric
        ax.scatter(r['youden_fpr'], r['youden_tpr'],
                   color=CMAP10[k % 10], s=70, zorder=5, marker='D',
                   edgecolors='black', linewidths=0.5)

    ax.set_xlabel('False Positive Rate  (fraction of bad detections kept)', fontsize=10)
    ax.set_ylabel('True Positive Rate  (fraction of good detections kept)', fontsize=10)
    ax.set_title('ROC curves — higher AUC = better metric\n'
                 'Diamond = best threshold (Youden J)', fontsize=10)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, lw=0.4, alpha=0.4)
    plt.tight_layout()
    path = os.path.join(RUN_DIR, f"fig_roc_{_RUN_STAMP}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [SAVED] {path}")


# ---- Figure 4: Threshold sensitivity ----------------------------------------
print("  Fig 4: Threshold sensitivity ...")

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
panel_cfg = [
    ('pass_rate', 'Pass rate (fraction of all detections kept)'),
    ('tpr',       'TPR — fraction of good detections kept'),
    ('fpr',       'FPR — fraction of bad detections kept'),
]
for ax, (key, ylabel) in zip(axes, panel_cfg):
    for k, metric in enumerate(SNR_METRICS):
        sub = df_sens[df_sens['metric'] == metric]
        ax.plot(sub['threshold'], sub[key], lw=2,
                color=CMAP10[k % 10],
                label=SNR_SHORT.get(metric, metric).replace('\n', ' '))
        if metric in roc_results:
            thr = roc_results[metric]['youden_threshold']
            # Mark the Youden threshold on the sensitivity curve
            row_near = sub.iloc[(sub['threshold'] - thr).abs().argsort().iloc[0]]
            ax.scatter(row_near['threshold'], row_near[key],
                       color=CMAP10[k % 10], s=50, zorder=5, marker='D')

    ax.set_xlabel('Threshold value', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.4, alpha=0.4)
    ax.set_xlim(THRESHOLD_RANGE[0], THRESHOLD_RANGE[-1])
    ax.set_ylim(0, 1.05)

fig.suptitle(
    'Threshold sensitivity  |  diamond = Youden optimal threshold per metric\n'
    'Good metric: TPR stays high while FPR drops fast as threshold increases',
    fontsize=10
)
plt.tight_layout(rect=[0, 0, 1, 0.93])
path = os.path.join(RUN_DIR, f"fig_threshold_sensitivity_{_RUN_STAMP}.png")
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"    [SAVED] {path}")


# ---- Figure 5: Boxplots by event type ---------------------------------------
print("  Fig 5: Boxplots by event type ...")

event_types = sorted(df['event_type'].dropna().unique())
ncols = min(4, N_M)
nrows = int(np.ceil(N_M / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = np.array(axes).flatten()

for k, metric in enumerate(SNR_METRICS):
    ax = axes[k]
    data_by_type = [
        df.loc[df['event_type'] == et, metric].dropna().values
        for et in event_types
    ]
    bp = ax.boxplot(data_by_type, patch_artist=True,
                    medianprops={'color': 'black', 'lw': 2.0},
                    flierprops={'marker': '.', 'markersize': 3, 'alpha': 0.5})
    for patch, color in zip(bp['boxes'], plt.cm.Set2.colors):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(event_types) + 1))
    ax.set_xticklabels(event_types, rotation=30, ha='right', fontsize=8)
    ax.set_title(SNR_LONG.get(metric, metric), fontsize=8)
    ax.set_ylabel('SNR value', fontsize=8)
    ax.tick_params(labelsize=7)

for idx in range(N_M, len(axes)):
    axes[idx].set_visible(False)

fig.suptitle('SNR metric distributions per event type', fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.97])
path = os.path.join(RUN_DIR, f"fig_boxplots_type_{_RUN_STAMP}.png")
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"    [SAVED] {path}")


# ---- Figure 6: Per-station mean SNR heatmap ---------------------------------
print("  Fig 6: Per-station heatmap ...")

df_sta_plot = df.groupby('station')[SNR_METRICS].mean()
n_sta = len(df_sta_plot)

fig, ax = plt.subplots(figsize=(max(9, N_M * 1.6), max(4, n_sta * 0.55)))
im = ax.imshow(df_sta_plot.values, aspect='auto', cmap='YlOrRd',
               vmin=0, vmax=np.nanpercentile(df_sta_plot.values, 95))
ax.set_xticks(range(N_M))
ax.set_xticklabels([SNR_SHORT.get(m, m).replace('\n', ' ') for m in SNR_METRICS],
                   rotation=40, ha='right', fontsize=9)
ax.set_yticks(range(n_sta))
ax.set_yticklabels(df_sta_plot.index.tolist(), fontsize=9)
for i in range(n_sta):
    for j in range(N_M):
        val = df_sta_plot.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=8)
plt.colorbar(im, ax=ax, label='Mean SNR  (color scale capped at 95th percentile)')
ax.set_title('Per-station mean SNR values\n'
             'Stations with anomalously high values may need to be excluded', fontsize=10)
plt.tight_layout()
path = os.path.join(RUN_DIR, f"fig_station_heatmap_{_RUN_STAMP}.png")
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"    [SAVED] {path}")



# =============================================================================
# SECTION 5 — WAVEFORM PANEL
# =============================================================================

print("\n" + "=" * 70)
print("  WAVEFORM PANEL")
print("=" * 70)

if client_sds is None:
    print("[SKIP] SDS client unavailable — waveform panel requires cluster access.")
else:
    # Sort all detections by WAVEFORM_RANK_METRIC and sample N evenly across the range
    df_ranked = df.dropna(subset=[WAVEFORM_RANK_METRIC]).sort_values(WAVEFORM_RANK_METRIC)
    df_ranked = df_ranked.reset_index(drop=True)

    if len(df_ranked) == 0:
        print("[SKIP] No valid rows for waveform panel.")
    else:
        n_sample  = min(N_WAVEFORM_SAMPLES, len(df_ranked))
        indices   = np.linspace(0, len(df_ranked) - 1, n_sample, dtype=int)
        df_sample = df_ranked.iloc[indices].reset_index(drop=True)

        print(f"\n  Sampling {n_sample} detections across {WAVEFORM_RANK_METRIC} range")
        print(f"  (min={df_ranked[WAVEFORM_RANK_METRIC].iloc[0]:.2f}  "
              f"max={df_ranked[WAVEFORM_RANK_METRIC].iloc[-1]:.2f})")

        fig, axes = plt.subplots(n_sample, 1, figsize=(19, max(6, n_sample * 1.3)),
                                 sharex=False)
        if n_sample == 1:
            axes = [axes]

        n_loaded = 0
        for row_idx, (_, row) in enumerate(df_sample.iterrows()):
            ax  = axes[row_idx]
            net = str(row['network'])
            sta = str(row['station'])
            chan = str(row.get('channel', WAVEFORM_CHANNEL))

            try:
                t_on   = UTCDateTime(row['det_starttime'])
                t_off  = UTCDateTime(row['det_endtime'])
                t_orig = UTCDateTime(row['event_time'])
            except Exception:
                ax.text(0.5, 0.5, 'Bad timestamps in CSV',
                        transform=ax.transAxes, ha='center', va='center', fontsize=8)
                continue

            # Try several location codes (location is not stored in the script 02 CSV)
            tr = None
            for loc in ['', '*', '00', '10', '20']:
                try:
                    st_tmp = client_sds.get_waveforms(
                        network=net, station=sta, location=loc, channel=chan,
                        starttime=t_on - WAVEFORM_PAD_S,
                        endtime=t_off + WAVEFORM_PAD_S,
                    )
                    if len(st_tmp) > 0:
                        tr = st_tmp[0]; break
                except Exception:
                    continue

            if tr is None:
                ax.text(0.5, 0.5, f'{net}.{sta}.{chan} — no SDS data',
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=8, color='gray')
                continue

            # Bandpass filter
            try:
                tr.detrend('demean')
                nyq = tr.stats.sampling_rate / 2.0
                tr.filter('bandpass',
                          freqmin=WAVEFORM_FREQMIN,
                          freqmax=min(WAVEFORM_FREQMAX, 0.9 * nyq),
                          corners=2, zerophase=True)
            except Exception:
                pass

            # Time axis: 0 = detection onset (t_on)
            t_sec = (np.arange(tr.stats.npts) / tr.stats.sampling_rate
                     + float(tr.stats.starttime - t_on))

            ax.plot(t_sec, tr.data * 1e6, color='#333333', lw=0.5)
            ax.axvspan(0, float(t_off - t_on), color='steelblue', alpha=0.12)
            ax.axvline(0, color='steelblue', lw=1.1, ls='--', alpha=0.8)
            ax.axvline(float(t_off - t_on), color='steelblue', lw=1.1, ls='--', alpha=0.8)
            ax.axvline(float(t_orig - t_on), color='red', lw=1.3, ls=':')

            inside_str = 'IN' if row['origin_inside_det'] else 'OUT'
            snr_annot  = '   '.join([
                f"{SNR_SHORT.get(m, m).replace(chr(10),' ')}={row.get(m, float('nan')):.1f}"
                for m in SNR_METRICS if m in row
            ])
            ax.set_title(
                f"Rank {row_idx+1}/{n_sample}  {net}.{sta}  {row.get('event_type','?')}"
                f"  [{WAVEFORM_RANK_METRIC}={row[WAVEFORM_RANK_METRIC]:.2f}  origin={inside_str}]\n"
                f"{snr_annot}",
                fontsize=6.5, loc='left', pad=2
            )
            ax.set_ylabel('µm/s', fontsize=6)
            ax.tick_params(labelsize=6)
            ax.set_xlim(t_sec[0], t_sec[-1])
            n_loaded += 1

        axes[-1].set_xlabel(
            'Time relative to detection onset (s)  |  '
            'Blue shade = detected window  |  Red dotted = catalog origin',
            fontsize=9
        )
        fig.suptitle(
            f'Waveform panel — {n_sample} detections from lowest (rank 1) to highest '
            f'(rank {n_sample}) {WAVEFORM_RANK_METRIC}\n'
            f'Filter: {WAVEFORM_FREQMIN}–{WAVEFORM_FREQMAX} Hz',
            fontsize=10
        )
        plt.tight_layout(rect=[0, 0.01, 1, 0.97])
        path = os.path.join(RUN_DIR, f"fig_waveforms_{_RUN_STAMP}.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  {n_loaded}/{n_sample} waveforms loaded.")
        print(f"  [SAVED] {path}")



# =============================================================================
# END
# =============================================================================

from datetime import datetime
print("\n" + "=" * 70)
print(f"  Run finished  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Input CSV     : {csv_path}")
print(f"  Rows analysed : {len(df)}")
print(f"  All outputs   : {RUN_DIR}")
print(f"  Log file      : {_log_filename}")
print("=" * 70)

_log_file.close()
