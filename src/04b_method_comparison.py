"""
DETECTION METHOD COMPARISON — GROULT SPECTROGRAM vs CLASSICAL STA/LTA
======================================================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : May 2026

Goal
----
Compare two precise-windowing methods applied to the same set of catalog events:
  Method A — Groult spectrogram-based bidirectional STA/LTA  (script 04a output)
  Method B — Classical STA/LTA on bandpass-filtered waveform  (script 04b output)

The primary question: 
which method produces more accurate detection windows, measured by how often the station's P-wave pick falls inside the detected window?

Metrics
-------
  1. pick_inside_det rate  —  fraction of detections where the P pick is inside the window
  2. Detection coverage    —  fraction of catalog events detected by at least one station per method (missed events = false negatives)
  3. Window precision      —  distribution of pick_lag_s (seconds between the detected onset and the P pick): a tighter distribution closer to 0 = more precise onset
  4. Window duration       —  distribution of det_duration_s
  5. Quality gate rate     —  fraction of detections passing SNR_full_mean ≥ 3 AND SNR_full_median ≥ 3 (Groult quality gate)
  6. SNR comparison        —  box plots of SNR_s2n_median for each method

Input
-----
  CSV_04A  — catalog_windows_<stamp>.csv produced by 04a (Groult)
  CSV_04B  — catalog_windows_<stamp>.csv produced by 04a (classical STA/LTA)

Output
------
  comparison_summary_<stamp>.csv    : per-metric, per-method, per-event-type table
  fig_pickrate_<stamp>.png          : pick_inside_det rate per method × event type
  fig_coverage_<stamp>.png          : detection coverage per method × event type
  fig_picklag_<stamp>.png           : pick_lag_s distributions per method × event type
  fig_duration_<stamp>.png          : det_duration_s distributions per method
  fig_quality_<stamp>.png           : quality_ok rate + SNR box plots per method
  fig_overview_<stamp>.png          : all-in-one 2×3 summary panel
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# -- Input CSVs ---------------------------------------------------------------
# Point each path to the catalog_windows_<stamp>.csv file from the relevant run
CSV_04A = "/data/failles/louisels/project/results/outputs_04a/groult/run_XXXXXXXX_XXXXXX/catalog_windows_XXXXXXXX_XXXXXX.csv"
CSV_04B = "/data/failles/louisels/project/results/outputs_04a/sta_lta/run_XXXXXXXX_XXXXXX/catalog_windows_XXXXXXXX_XXXXXX.csv"

# -- Output -------------------------------------------------------------------
OUTPUT_DIR = "/data/failles/louisels/project/results/outputs_04b"

# -- Event types to include in comparison (set [] to keep all) ----------------
TARGET_TYPES = []    # ["earthquake", "rockslide", ...] — empty = all types in the data

# -- Ground truth column ------------------------------------------------------
GROUND_TRUTH = 'pick_inside_det'   # or 'origin_inside_det'

# -- Quality gate thresholds (must match 04a) ---------------------------------
SNR_MEAN_MIN  = 1.856    # SNR_full_mean  >= this  (05a ROC-optimal, AUC=0.700)
SNR_S2N_MIN   = 10.503   # SNR_s2n_median >= this  (05a ROC-optimal, AUC=0.703)

# -- Display labels -----------------------------------------------------------
LABEL_A = "Groult\n(spectrogram STA/LTA)"
LABEL_B = "Classical\nSTA/LTA"
COLOR_A = '#1f77b4'   # blue
COLOR_B = '#ff7f0e'   # orange



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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from run_setup import create_run_dir, setup_logging, set_matplotlib_defaults


# ----------- Run setup ---------------
RUN_DIR, _RUN_STAMP = create_run_dir(OUTPUT_DIR)
_log_file, _log_filename = setup_logging(
    RUN_DIR, "04b_method_comparison.py",
    extra_info=f"04a: {os.path.basename(CSV_04A)}  |  04b: {os.path.basename(CSV_04B)}"
)
set_matplotlib_defaults()


# ----------- Load CSVs ---------------
def _load_csv(path, method_label):
    if not os.path.isfile(path):
        print(f"[ERROR] CSV not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path, low_memory=False)
    df['method'] = method_label
    print(f"  {method_label:<12}: {len(df)} rows  ({df['event_type'].value_counts().to_dict()})")
    return df

print("\nLoading CSVs ...")
df_a = _load_csv(CSV_04A, 'groult')
df_b = _load_csv(CSV_04B, 'sta_lta')

# Optionally filter to specific event types
if TARGET_TYPES:
    df_a = df_a[df_a['event_type'].isin(TARGET_TYPES)]
    df_b = df_b[df_b['event_type'].isin(TARGET_TYPES)]
    print(f"  Filtered to types: {TARGET_TYPES}")

# Combined table for joint analyses
df_all = pd.concat([df_a, df_b], ignore_index=True)

# Ground truth as boolean (NaN rows are kept but excluded from rate calculations)
for df in [df_a, df_b, df_all]:
    df['label'] = df[GROUND_TRUTH].map(lambda x: bool(x) if pd.notna(x) else np.nan)

# Event types present in either CSV
all_etypes = sorted(df_all['event_type'].dropna().unique())
n_types    = len(all_etypes)

print(f"\n  Event types: {all_etypes}")
print(f"  Ground truth column: {GROUND_TRUTH}")



# =============================================================================
# SECTION 3 — METRIC COMPUTATIONS
# =============================================================================

print("\n" + "=" * 70)
print("  METRIC COMPUTATIONS")
print("=" * 70)


def pick_rate(df):
    """
    Fraction of rows where pick_inside_det is True (NaN excluded from denominator)
     -> measures window precision (high pick-rate = windows well-placed)
    """
    col = df['label'].dropna()
    return float(col.mean()) if len(col) > 0 else np.nan


def coverage_rate(df, all_event_times):
    """
    Fraction of catalog events that generated at least one detection on any station (regardless of pick quality)
     -> measures sensitivity (a method can have a high pick-rate but low coverage)
    """
    detected = set(df['event_time'].dropna().unique())
    return len(detected & all_event_times) / len(all_event_times) if all_event_times else np.nan


# All unique event times across both methods (union)
all_event_times = set(df_all['event_time'].dropna().unique())
print(f"\n  Total unique events in either CSV: {len(all_event_times)}")


# --- 3.1  Overall metrics per method -----------------------------------------
print("\n--- 3.1  Overall metrics ---")

summary_rows = []
for method_label, df_m in [('groult', df_a), ('sta_lta', df_b)]:
    for etype in all_etypes:
        sub = df_m[df_m['event_type'] == etype]
        sub_all_events = set(df_all.loc[df_all['event_type'] == etype, 'event_time'].dropna().unique())

        pr     = pick_rate(sub)
        cov    = coverage_rate(sub, sub_all_events)
        qual   = sub['quality_ok'].mean() if 'quality_ok' in sub.columns else np.nan
        n_det  = len(sub)
        n_ev   = sub['event_time'].nunique()

        pick_lag = pd.to_numeric(sub['pick_lag_s'], errors='coerce').dropna()
        dur      = pd.to_numeric(sub['det_duration_s'], errors='coerce').dropna()
        snr_med  = pd.to_numeric(sub.get('SNR_s2n_median', pd.Series(dtype=float)), errors='coerce').dropna()

        summary_rows.append({
            'method'            : method_label,
            'event_type'        : etype,
            'n_detections'      : n_det,
            'n_events_detected' : n_ev,
            'n_events_total'    : len(sub_all_events),
            'pick_inside_rate'  : round(pr,   3) if not np.isnan(pr)   else np.nan,
            'coverage_rate'     : round(cov,  3) if not np.isnan(cov)  else np.nan,
            'quality_ok_rate'   : round(qual, 3) if not np.isnan(qual) else np.nan,
            'pick_lag_median_s' : round(pick_lag.median(), 2) if len(pick_lag) > 0 else np.nan,
            'pick_lag_mean_s'   : round(pick_lag.mean(),   2) if len(pick_lag) > 0 else np.nan,
            'pick_lag_std_s'    : round(pick_lag.std(),    2) if len(pick_lag) > 0 else np.nan,
            'det_dur_median_s'  : round(dur.median(), 1) if len(dur) > 0 else np.nan,
            'snr_s2n_median'    : round(snr_med.median(), 2) if len(snr_med) > 0 else np.nan,
        })

df_summary = pd.DataFrame(summary_rows)

# Print formatted comparison table
print(f"\n  {'Method':<12} {'Type':<22} {'n_det':>7} "
      f"{'pick_rate':>10} {'coverage':>10} {'quality':>8} "
      f"{'lag_med':>9} {'dur_med':>8}")
print("  " + "-" * 92)
for _, row in df_summary.iterrows():
    print(f"  {row['method']:<12} {row['event_type']:<22} {row['n_detections']:>7} "
          f"{row['pick_inside_rate']:>9.1%} {row['coverage_rate']:>9.1%} "
          f"{row['quality_ok_rate']:>7.1%} "
          f"{row['pick_lag_median_s']:>+8.1f}s {row['det_dur_median_s']:>7.0f}s")


# --- 3.2  Event-level comparison: matched pairs ------------------------------
# For each (event_time, station) pair, record detection status in each method
print("\n--- 3.2  Matched event×station analysis ---")

# Build a pivot: index=(event_time, station, event_type), columns=method
def _summarise_pair(df_m):
    """One row per (event_time, station): was it detected? pick inside?"""
    if len(df_m) == 0:
        return pd.Series({'detected': False, 'pick_inside': np.nan})
    # If multiple detections at same station, a pick_inside=True anywhere counts as True
    pick_vals = df_m['label'].dropna()
    return pd.Series({
        'detected'   : True,
        'pick_inside': bool(pick_vals.any()) if len(pick_vals) > 0 else np.nan,
    })

key_cols = ['event_time', 'event_type', 'station']

grp_a = df_a.groupby(key_cols).apply(_summarise_pair).add_prefix('A_').reset_index()
grp_b = df_b.groupby(key_cols).apply(_summarise_pair).add_prefix('B_').reset_index()

df_matched = pd.merge(grp_a, grp_b, on=key_cols, how='outer')         # 'outer' keeps a station that is in one csv (if one method detected it and the other didn't)
df_matched['A_detected']    = df_matched['A_detected'].fillna(False)  # fills missing detections as False for the method that didn't detect
df_matched['B_detected']    = df_matched['B_detected'].fillna(False)

# Detection agreement categories
def _agreement(row):
    if row['A_detected'] and row['B_detected']:
        return 'both'
    elif row['A_detected']:
        return 'A_only'
    elif row['B_detected']:
        return 'B_only'
    return 'neither'

df_matched['agreement'] = df_matched.apply(_agreement, axis=1)

print(f"\n  Agreement across {len(df_matched)} (event × station) pairs:")
for cat, cnt in df_matched['agreement'].value_counts().items():
    pct = cnt / len(df_matched) * 100
    print(f"    {cat:<12}: {cnt:4d}  ({pct:.1f}%)")

print(f"\n  Per event type:")
for etype, grp in df_matched.groupby('event_type'):
    agree_pct = (grp['agreement'] == 'both').mean() * 100
    a_only    = (grp['agreement'] == 'A_only').mean() * 100
    b_only    = (grp['agreement'] == 'B_only').mean() * 100
    print(f"    {etype:<22}  both={agree_pct:.1f}%  A_only={a_only:.1f}%  B_only={b_only:.1f}%")


# --- 3.3  Save summary CSV ---------------------------------------------------
summary_path = os.path.join(RUN_DIR, f"comparison_summary_{_RUN_STAMP}.csv")
df_summary.to_csv(summary_path, index=False)
print(f"\n[SAVED] {summary_path}")



# =============================================================================
# SECTION 4 — FIGURES
# =============================================================================

print("\n" + "=" * 70)
print("  GENERATING FIGURES")
print("=" * 70)

# Helper: bar group positions for N event types, 2 methods
def _bar_positions(n_types, width=0.35):
    x = np.arange(n_types)
    return x - width / 2, x + width / 2, x, width


# ---- Figure 1: pick_inside_det rate per method × event type ----------------
print("\n  Fig 1: pick_inside_det rate ...")

fig, ax = plt.subplots(figsize=(max(7, n_types * 2.5), 5))

xA, xB, x, w = _bar_positions(n_types)
for xi, etype in zip(xA, all_etypes):
    row = df_summary[(df_summary['method'] == 'groult') & (df_summary['event_type'] == etype)]
    ax.bar(xi, row['pick_inside_rate'].iloc[0] if len(row) else 0,
           width=w, color=COLOR_A, alpha=0.85, edgecolor='white')

for xi, etype in zip(xB, all_etypes):
    row = df_summary[(df_summary['method'] == 'sta_lta') & (df_summary['event_type'] == etype)]
    ax.bar(xi, row['pick_inside_rate'].iloc[0] if len(row) else 0,
           width=w, color=COLOR_B, alpha=0.85, edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(all_etypes, fontsize=11)
ax.set_ylabel('P-pick inside detected window (%)', fontsize=12)
ax.set_ylim(0, 1.1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.set_title(f'Pick accuracy — fraction of detections where the P pick falls inside the window\n'
             f'(ground truth: {GROUND_TRUTH})', fontsize=11)
ax.legend(handles=[
    mpatches.Patch(color=COLOR_A, alpha=0.85, label=LABEL_A.replace('\n', ' ')),
    mpatches.Patch(color=COLOR_B, alpha=0.85, label=LABEL_B.replace('\n', ' ')),
], fontsize=10)
ax.grid(axis='y', alpha=0.3, lw=0.5)
plt.tight_layout()
path = os.path.join(RUN_DIR, f"fig_pickrate_{_RUN_STAMP}.png")
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"    [SAVED] {path}")


# ---- Figure 2: detection coverage per method × event type ------------------
print("  Fig 2: Detection coverage ...")

fig, ax = plt.subplots(figsize=(max(7, n_types * 2.5), 5))

xA, xB, x, w = _bar_positions(n_types)
for xi, etype in zip(xA, all_etypes):
    row = df_summary[(df_summary['method'] == 'groult') & (df_summary['event_type'] == etype)]
    ax.bar(xi, row['coverage_rate'].iloc[0] if len(row) else 0,
           width=w, color=COLOR_A, alpha=0.85, edgecolor='white')

for xi, etype in zip(xB, all_etypes):
    row = df_summary[(df_summary['method'] == 'sta_lta') & (df_summary['event_type'] == etype)]
    ax.bar(xi, row['coverage_rate'].iloc[0] if len(row) else 0,
           width=w, color=COLOR_B, alpha=0.85, edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(all_etypes, fontsize=11)
ax.set_ylabel('Events detected / total events (%)', fontsize=12)
ax.set_ylim(0, 1.1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.set_title('Detection coverage — fraction of catalog events detected by ≥ 1 station',
             fontsize=11)
ax.legend(handles=[
    mpatches.Patch(color=COLOR_A, alpha=0.85, label=LABEL_A.replace('\n', ' ')),
    mpatches.Patch(color=COLOR_B, alpha=0.85, label=LABEL_B.replace('\n', ' ')),
], fontsize=10)
ax.grid(axis='y', alpha=0.3, lw=0.5)
plt.tight_layout()
path = os.path.join(RUN_DIR, f"fig_coverage_{_RUN_STAMP}.png")
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"    [SAVED] {path}")


# ---- Figure 3: pick_lag_s distribution per method × event type -------------
print("  Fig 3: Pick lag distribution ...")

# Keep ALL rows that have a P pick (pick_lag_s not NaN), regardless of pick_inside_det.
# This exposes negative lags (detector fired AFTER the P pick → onset missed) as well
# as large positive lags (detector fired very early, pick far inside a wide window).
df_lag_a = df_a[df_a['pick_lag_s'].notna()].copy()
df_lag_b = df_b[df_b['pick_lag_s'].notna()].copy()
df_lag_a['pick_lag_s'] = pd.to_numeric(df_lag_a['pick_lag_s'], errors='coerce')
df_lag_b['pick_lag_s'] = pd.to_numeric(df_lag_b['pick_lag_s'], errors='coerce')

# Scatter colour: green = pick inside window (good), red = pick outside (missed/late)
COLOR_INSIDE  = '#2e7d32'   # dark green
COLOR_OUTSIDE = '#c62828'   # dark red

n_cols = max(1, n_types)
fig, axes = plt.subplots(1, n_cols, figsize=(max(6, n_cols * 4), 6), sharey=False)
if n_cols == 1:
    axes = [axes]

for ax, etype in zip(axes, all_etypes):
    sub_a = df_lag_a[df_lag_a['event_type'] == etype].dropna(subset=['pick_lag_s'])
    sub_b = df_lag_b[df_lag_b['event_type'] == etype].dropna(subset=['pick_lag_s'])

    lag_a = sub_a['pick_lag_s'].values
    lag_b = sub_b['pick_lag_s'].values

    # Clip display range to 2nd–98th percentile across both methods so extreme
    # outliers don't flatten the violin (raw data are still used for the violin shape)
    all_lags = np.concatenate([lag_a, lag_b])
    y_lo = np.percentile(all_lags, 2)
    y_hi = np.percentile(all_lags, 98)
    # Always show at least a ±5 s window around zero
    y_lo = min(y_lo, -5)
    y_hi = max(y_hi,  5)

    # Violin on the full (unclipped) distribution
    parts = ax.violinplot([lag_a, lag_b], positions=[1, 2],
                          showmedians=True, showextrema=False)
    for pc, col in zip(parts['bodies'], [COLOR_A, COLOR_B]):
        pc.set_facecolor(col)
        pc.set_alpha(0.5)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)

    # Scatter: colour by pick_inside_det
    for xi, sub, col_method in [(1, sub_a, COLOR_A), (2, sub_b, COLOR_B)]:
        inside  = sub[sub['label'] == True]['pick_lag_s']
        outside = sub[sub['label'] != True]['pick_lag_s']
        jitter_in  = np.random.normal(0, 0.05, size=len(inside))
        jitter_out = np.random.normal(0, 0.05, size=len(outside))
        ax.scatter(xi + jitter_in,  inside,  s=10, color=COLOR_INSIDE,
                   alpha=0.5, zorder=3, label='pick inside'  if xi == 1 else '')
        ax.scatter(xi + jitter_out, outside, s=10, color=COLOR_OUTSIDE,
                   alpha=0.3, zorder=3, label='pick outside' if xi == 1 else '')

    ax.axhline(0, color='black', lw=1.2, ls='--', alpha=0.6, label='lag = 0')
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([LABEL_A, LABEL_B], fontsize=9)
    ax.set_title(etype, fontsize=10, fontweight='bold')
    ax.set_ylabel('pick_lag_s  [s]', fontsize=9)
    ax.grid(axis='y', alpha=0.3, lw=0.5)

    # Annotate median (all picks) and fraction with negative lag
    for xi, lag in [(1, lag_a), (2, lag_b)]:
        if len(lag) == 0:
            continue
        med = np.median(lag)
        pct_neg = (lag < 0).mean() * 100
        ax.text(xi, y_hi * 0.97, f'med={med:+.1f}s\n{pct_neg:.0f}% neg',
                ha='center', va='top', fontsize=7.5, fontweight='bold')

# One shared legend on the first axis
axes[0].legend(
    handles=[
        plt.scatter([], [], s=15, color=COLOR_INSIDE,  label='pick inside window'),
        plt.scatter([], [], s=15, color=COLOR_OUTSIDE, label='pick outside window'),
        Line2D([0], [0], color='black', ls='--', lw=1.2, label='lag = 0'),
    ],
    fontsize=8, loc='lower right'
)

fig.suptitle(
    'Pick lag (P-pick time − detected onset) — all detections with a known P pick\n'
    'Green = pick inside window  |  Red = pick outside  |  '
    'Negative lag = detector fired AFTER the P pick (onset missed)',
    fontsize=10
)
plt.tight_layout()
path = os.path.join(RUN_DIR, f"fig_picklag_{_RUN_STAMP}.png")
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"    [SAVED] {path}")


# ---- Figure 4: detection duration distribution per method ------------------
print("  Fig 4: Detection duration ...")

fig, ax = plt.subplots(figsize=(8, 5))

dur_a = pd.to_numeric(df_a['det_duration_s'], errors='coerce').dropna()
dur_b = pd.to_numeric(df_b['det_duration_s'], errors='coerce').dropna()

xmax = max(np.percentile(dur_a, 95) if len(dur_a) else 1,
           np.percentile(dur_b, 95) if len(dur_b) else 1)
bins = np.linspace(0, xmax, 50)

ax.hist(dur_a, bins=bins, density=True, alpha=0.5, color=COLOR_A,
        label=f'{LABEL_A.replace(chr(10)," ")}  (n={len(dur_a)}, med={dur_a.median():.0f}s)')
ax.hist(dur_b, bins=bins, density=True, alpha=0.5, color=COLOR_B,
        label=f'{LABEL_B.replace(chr(10)," ")}  (n={len(dur_b)}, med={dur_b.median():.0f}s)')
ax.axvline(dur_a.median(), color=COLOR_A, lw=2, ls='--')
ax.axvline(dur_b.median(), color=COLOR_B, lw=2, ls='--')
ax.set_xlabel('Detection duration [s]', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Distribution of detected window durations\n'
             '(shorter and tighter = more precise windowing)', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, lw=0.5)
plt.tight_layout()
path = os.path.join(RUN_DIR, f"fig_duration_{_RUN_STAMP}.png")
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"    [SAVED] {path}")


# ---- Figure 5: quality_ok rate + SNR_s2n_median box plots ------------------
print("  Fig 5: Quality rate + SNR ...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: quality_ok rate per method × event type
ax = axes[0]
xA, xB, x, w = _bar_positions(n_types)
for xi, etype in zip(xA, all_etypes):
    row = df_summary[(df_summary['method'] == 'groult') & (df_summary['event_type'] == etype)]
    ax.bar(xi, row['quality_ok_rate'].iloc[0] if len(row) else 0,
           width=w, color=COLOR_A, alpha=0.85, edgecolor='white')
for xi, etype in zip(xB, all_etypes):
    row = df_summary[(df_summary['method'] == 'sta_lta') & (df_summary['event_type'] == etype)]
    ax.bar(xi, row['quality_ok_rate'].iloc[0] if len(row) else 0,
           width=w, color=COLOR_B, alpha=0.85, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(all_etypes, fontsize=10)
ax.set_ylabel('quality_ok rate (%)', fontsize=11)
ax.set_ylim(0, 1.1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.set_title(f'Quality gate rate\n(SNR_full_mean ≥ {SNR_MEAN_MIN} AND SNR_s2n_median ≥ {SNR_S2N_MIN})', fontsize=10)
ax.legend(handles=[
    mpatches.Patch(color=COLOR_A, alpha=0.85, label=LABEL_A.replace('\n', ' ')),
    mpatches.Patch(color=COLOR_B, alpha=0.85, label=LABEL_B.replace('\n', ' ')),
], fontsize=9)
ax.grid(axis='y', alpha=0.3, lw=0.5)

# Panel B: SNR_s2n_median box plot per method
ax = axes[1]
if 'SNR_s2n_median' in df_a.columns and 'SNR_s2n_median' in df_b.columns:
    snr_a = pd.to_numeric(df_a['SNR_s2n_median'], errors='coerce').dropna()
    snr_b = pd.to_numeric(df_b['SNR_s2n_median'], errors='coerce').dropna()
    bp = ax.boxplot([snr_a, snr_b], labels=[LABEL_A, LABEL_B],
                    patch_artist=True, notch=False,
                    medianprops=dict(color='white', lw=2.5))
    bp['boxes'][0].set_facecolor(COLOR_A); bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(COLOR_B); bp['boxes'][1].set_alpha(0.7)
    # Scatter individual points with jitter (capped at 300 per method for speed)
    for xi, (snr, col) in enumerate([(snr_a, COLOR_A), (snr_b, COLOR_B)], 1):
        samp = snr.sample(min(len(snr), 300), random_state=0)
        ax.scatter(xi + np.random.normal(0, 0.07, len(samp)), samp,
                   s=8, color=col, alpha=0.35, zorder=3)
    ax.set_ylabel('SNR_s2n_median', fontsize=11)
    ax.set_title('SNR comparison (robust metric)\n'
                 'Higher = cleaner signal in the detected window', fontsize=10)
    ax.grid(axis='y', alpha=0.3, lw=0.5)
    ax.set_ylim(bottom=0)
else:
    ax.text(0.5, 0.5, 'SNR_s2n_median not available\nin one or both CSVs',
            ha='center', va='center', transform=ax.transAxes, fontsize=10, color='grey')

plt.tight_layout()
path = os.path.join(RUN_DIR, f"fig_quality_{_RUN_STAMP}.png")
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"    [SAVED] {path}")


# ---- Figure 6: all-in-one overview panel (2 × 3) ---------------------------
print("  Fig 6: Overview panel ...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    f'Method comparison — Groult spectrogram STA/LTA (04a)  vs  Classical STA/LTA (04b)\n'
    f'Ground truth: {GROUND_TRUTH}',
    fontsize=13, fontweight='bold', y=1.01
)

# Top-left: pick_inside_det rate
ax = axes[0, 0]
xA, xB, x, w = _bar_positions(n_types)
for xi, etype in zip(xA, all_etypes):
    row = df_summary[(df_summary['method'] == 'groult') & (df_summary['event_type'] == etype)]
    ax.bar(xi, row['pick_inside_rate'].iloc[0] if len(row) else 0,
           width=w, color=COLOR_A, alpha=0.85, edgecolor='white')
for xi, etype in zip(xB, all_etypes):
    row = df_summary[(df_summary['method'] == 'sta_lta') & (df_summary['event_type'] == etype)]
    ax.bar(xi, row['pick_inside_rate'].iloc[0] if len(row) else 0,
           width=w, color=COLOR_B, alpha=0.85, edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(all_etypes, fontsize=9)
ax.set_ylim(0, 1.1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.set_title('Pick accuracy (primary metric)', fontsize=10, fontweight='bold')
ax.set_ylabel('pick_inside_det rate', fontsize=9)
ax.grid(axis='y', alpha=0.3, lw=0.5)

# Top-center: detection coverage
ax = axes[0, 1]
for xi, etype in zip(xA, all_etypes):
    row = df_summary[(df_summary['method'] == 'groult') & (df_summary['event_type'] == etype)]
    ax.bar(xi, row['coverage_rate'].iloc[0] if len(row) else 0,
           width=w, color=COLOR_A, alpha=0.85, edgecolor='white')
for xi, etype in zip(xB, all_etypes):
    row = df_summary[(df_summary['method'] == 'sta_lta') & (df_summary['event_type'] == etype)]
    ax.bar(xi, row['coverage_rate'].iloc[0] if len(row) else 0,
           width=w, color=COLOR_B, alpha=0.85, edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(all_etypes, fontsize=9)
ax.set_ylim(0, 1.1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.set_title('Detection coverage', fontsize=10, fontweight='bold')
ax.set_ylabel('Events detected / total', fontsize=9)
ax.grid(axis='y', alpha=0.3, lw=0.5)

# Top-right: quality_ok rate
ax = axes[0, 2]
for xi, etype in zip(xA, all_etypes):
    row = df_summary[(df_summary['method'] == 'groult') & (df_summary['event_type'] == etype)]
    ax.bar(xi, row['quality_ok_rate'].iloc[0] if len(row) else 0,
           width=w, color=COLOR_A, alpha=0.85, edgecolor='white')
for xi, etype in zip(xB, all_etypes):
    row = df_summary[(df_summary['method'] == 'sta_lta') & (df_summary['event_type'] == etype)]
    ax.bar(xi, row['quality_ok_rate'].iloc[0] if len(row) else 0,
           width=w, color=COLOR_B, alpha=0.85, edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(all_etypes, fontsize=9)
ax.set_ylim(0, 1.1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.set_title('Quality gate rate (SNR filter)', fontsize=10, fontweight='bold')
ax.set_ylabel('quality_ok rate', fontsize=9)
ax.grid(axis='y', alpha=0.3, lw=0.5)

# Legend for the three bar charts (add to top-right)
axes[0, 0].legend(handles=[
    mpatches.Patch(color=COLOR_A, alpha=0.85, label=LABEL_A.replace('\n', ' ')),
    mpatches.Patch(color=COLOR_B, alpha=0.85, label=LABEL_B.replace('\n', ' ')),
], fontsize=8, loc='lower right')

# Bottom-left: pick_lag_s violin (all event types merged)
ax = axes[1, 0]
lag_a_all = pd.to_numeric(df_lag_a['pick_lag_s'], errors='coerce').dropna().values
lag_b_all = pd.to_numeric(df_lag_b['pick_lag_s'], errors='coerce').dropna().values
if len(lag_a_all) > 0 and len(lag_b_all) > 0:
    parts = ax.violinplot([lag_a_all, lag_b_all], positions=[1, 2],
                          showmedians=True, showextrema=False)
    parts['bodies'][0].set_facecolor(COLOR_A); parts['bodies'][0].set_alpha(0.6)
    parts['bodies'][1].set_facecolor(COLOR_B); parts['bodies'][1].set_alpha(0.6)
    parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
    ax.axhline(0, color='black', lw=1.0, ls='--', alpha=0.5)
    ax.set_xticks([1, 2]); ax.set_xticklabels([LABEL_A, LABEL_B], fontsize=9)
    ax.set_ylabel('pick_lag_s  [s]', fontsize=9)
    # Clip y-axis to 2nd–98th percentile so the violin is readable
    all_ov = np.concatenate([lag_a_all, lag_b_all])
    ax.set_ylim(min(np.percentile(all_ov, 2), -5),
                max(np.percentile(all_ov, 98), 5))
    for xi, lag in [(1, lag_a_all), (2, lag_b_all)]:
        ax.text(xi, ax.get_ylim()[1] * 0.97,
                f'med={np.median(lag):+.1f}s\n{(lag < 0).mean()*100:.0f}% neg',
                ha='center', va='top', fontsize=7, fontweight='bold')
ax.set_title('Pick lag (all picks)\nGreen=inside  Red=outside  neg=missed onset',
             fontsize=9, fontweight='bold')
ax.grid(axis='y', alpha=0.3, lw=0.5)

# Bottom-center: detection duration
ax = axes[1, 1]
if len(dur_a) > 0 and len(dur_b) > 0:
    xmax_dur = max(np.percentile(dur_a, 95), np.percentile(dur_b, 95))
    bins_dur  = np.linspace(0, xmax_dur, 40)
    ax.hist(dur_a, bins=bins_dur, density=True, alpha=0.5, color=COLOR_A,
            label=f'Groult  (med={dur_a.median():.0f}s)')
    ax.hist(dur_b, bins=bins_dur, density=True, alpha=0.5, color=COLOR_B,
            label=f'STA/LTA  (med={dur_b.median():.0f}s)')
    ax.axvline(dur_a.median(), color=COLOR_A, lw=2, ls='--')
    ax.axvline(dur_b.median(), color=COLOR_B, lw=2, ls='--')
    ax.legend(fontsize=8)
ax.set_xlabel('Duration [s]', fontsize=9)
ax.set_ylabel('Density', fontsize=9)
ax.set_title('Detection window duration', fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3, lw=0.5)

# Bottom-right: agreement pie chart (both / A_only / B_only / neither)
ax = axes[1, 2]
agree_counts = df_matched['agreement'].value_counts()
labels_pie = []
sizes_pie  = []
colors_pie = []
for cat, col in [('both', '#66BB6A'), ('A_only', COLOR_A),
                 ('B_only', COLOR_B), ('neither', '#BDBDBD')]:
    cnt = agree_counts.get(cat, 0)
    labels_pie.append(f'{cat}\n({cnt})')
    sizes_pie.append(cnt)
    colors_pie.append(col)
ax.pie(sizes_pie, labels=labels_pie, colors=colors_pie,
       autopct='%1.0f%%', startangle=90,
       textprops={'fontsize': 9},
       wedgeprops={'edgecolor': 'white', 'linewidth': 1.2})
ax.set_title('Event×station detection agreement', fontsize=10, fontweight='bold')

plt.tight_layout()
path = os.path.join(RUN_DIR, f"fig_overview_{_RUN_STAMP}.png")
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"    [SAVED] {path}")



# =============================================================================
# END
# =============================================================================

from datetime import datetime
print("\n" + "=" * 70)
print(f"  Run finished   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Method A (04a) : {len(df_a)} rows — {CSV_04A}")
print(f"  Method B (04b) : {len(df_b)} rows — {CSV_04B}")
print(f"  All outputs    : {RUN_DIR}")
print(f"  Log file       : {_log_filename}")
print("=" * 70)

_log_file.close()
