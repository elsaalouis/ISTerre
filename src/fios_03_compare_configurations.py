"""
FIOS LANDSLIDE — MULTI-CONFIGURATION COMPARISON
=================================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : May 2026

Goal
----
Overlay the night-only seismicity trend from several 02a runs
(different frequency bands / STA-LTA parameters) on a single figure,
to check whether the post-destabilisation decrease is robust across
frequency bands or is band-specific.

Usage
-----
1. Run 02a_classical_sta_lta_FIOS.py once for each configuration, each time with a different OUTPUT_DIR so the results are kept separate.
2. Fill in CONFIGURATIONS below (label + path to the OUTPUT_DIR used).
3. Run this script: python 02a_compare_configurations.py
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# Each entry: ("Label for legend", r"path\to\OUTPUT_DIR")
# The script automatically picks the most recent run_* subfolder inside each.
CONFIGURATIONS = [
    ("10–80 Hz  (STA=0.5s  LTA=60s)",
     r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\02a_fios_classical_sta_lta\10-80Hz"),

    ("5–20 Hz   (STA=1s    LTA=80s)",
     r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\02a_fios_classical_sta_lta\5-20Hz"),

    ("2–10 Hz   (STA=2s    LTA=120s)",
     r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\02a_fios_classical_sta_lta\2-10Hz"),
]

# Destabilisation onset
DESTAB_DATE = "2026-04-14"

# Rolling-mean window (days)
ROLLING_DAYS = 7

# Night window label (for axis title)
NIGHT_START_UTC = 18
NIGHT_END_UTC   = 4

# Figures saved here
OUTPUT_DIR = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\02a_fios_classical_sta_lta"



# =============================================================================
# SECTION 2 — SETUP & LOAD
# =============================================================================

import os
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Colours for up to 5 configurations
PALETTE = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e']

datasets = []   # list of (label, df_counts) loaded successfully

for label, out_dir in CONFIGURATIONS:
    # look for daily_counts.csv directly in the given folder
    csv_direct = os.path.join(out_dir, "daily_counts.csv")
    run_dirs   = sorted(glob.glob(os.path.join(out_dir, "run_*")))

    if os.path.exists(csv_direct):
        csv_path = csv_direct
    elif run_dirs:
        csv_path = os.path.join(run_dirs[-1], "daily_counts.csv")
    else:
        print(f"[WARN] No daily_counts.csv found in: {out_dir}  →  skipping '{label}'")
        continue

    if not os.path.exists(csv_path):
        print(f"[WARN] daily_counts.csv not found at: {csv_path}  →  skipping '{label}'")
        continue

    df = pd.read_csv(csv_path)
    df = df[df['data_available']].copy()
    df['date'] = pd.to_datetime(df['day'])
    df = df.set_index('date').sort_index()

    if 'n_night' not in df.columns:
        print(f"[WARN] 'n_night' column missing in {csv_path}  →  skipping '{label}'")
        print("       Re-run 02a with NIGHT_START_UTC / NIGHT_END_UTC defined.")
        continue

    datasets.append((label, df))
    print(f"[OK] Loaded '{label}'  ({len(df)} days  —  {csv_path})")

if not datasets:
    raise RuntimeError("No valid datasets loaded. Check your CONFIGURATIONS paths.")



# =============================================================================
# SECTION 3 — FIGURES
# =============================================================================

t_destab = pd.Timestamp(DESTAB_DATE)

def linear_trend(dates, values):
    """Return fitted y values for a least-squares line (needs ≥ 2 points)."""
    x = np.array([(d - dates[0]).days for d in dates], dtype=float)
    if len(x) < 2 or np.all(np.isnan(values)):
        return values * np.nan
    mask = ~np.isnan(values)
    coeffs = np.polyfit(x[mask], values[mask], 1)
    return np.polyval(coeffs, x)


# --------------------------------------------------------------------------
# Figure 1: Rolling mean overlay (main comparison figure)
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 6))

for i, (label, df) in enumerate(datasets):
    color = PALETTE[i % len(PALETTE)]

    # Daily bars (faint, same colour)
    ax.bar(df.index.to_pydatetime(), df['n_night'].values,
           color=color, alpha=0.18, width=0.85)

    # Rolling mean
    rolling = df['n_night'].rolling(ROLLING_DAYS, center=True, min_periods=3).mean()
    ax.plot(rolling.index.to_pydatetime(), rolling.values,
            color=color, lw=2.2, label=label)

    # Linear trend — pre period
    pre  = df[df.index <  t_destab]
    post = df[df.index >= t_destab]
    if len(pre) >= 3:
        ax.plot(pre.index.to_pydatetime(),
                linear_trend(pre.index.to_pydatetime(), pre['n_night'].values),
                color=color, lw=1.2, ls='--', alpha=0.7)
    if len(post) >= 3:
        ax.plot(post.index.to_pydatetime(),
                linear_trend(post.index.to_pydatetime(), post['n_night'].values),
                color=color, lw=1.2, ls=':', alpha=0.7)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
plt.xticks(rotation=45, ha='right')

ax.set_ylabel('Night detections per day')
ax.set_ylim(bottom=0)
ax.set_title(
    f'FIO1 — Night-only seismicity: multi-configuration comparison\n'
    f'Night window: UTC {NIGHT_START_UTC:02d}:00–{NIGHT_END_UTC:02d}:00  |  '
    f'{ROLLING_DAYS}-day rolling mean  |  '
    f'Dashed = pre trend,  Dotted = post trend'
)
ax.legend(fontsize=9, loc='upper right')
ax.grid(axis='y', lw=0.4, alpha=0.4)
plt.tight_layout()

fig_path = os.path.join(OUTPUT_DIR, "fig_compare_night_trend.png")
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"\n[SAVED] {fig_path}")


# --------------------------------------------------------------------------
# Figure 2: Pre / post mean comparison — one panel per configuration
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 5),
                         sharey=False)
if len(datasets) == 1:
    axes = [axes]

for i, (label, df) in enumerate(datasets):
    color = PALETTE[i % len(PALETTE)]
    ax    = axes[i]

    pre_mean  = df.loc[df.index <  t_destab, 'n_night'].mean()
    post_mean = df.loc[df.index >= t_destab, 'n_night'].mean()
    pre_std   = df.loc[df.index <  t_destab, 'n_night'].std()
    post_std  = df.loc[df.index >= t_destab, 'n_night'].std()

    ax.bar(['Pre'],  [pre_mean],  color=color, alpha=0.85,
           yerr=[pre_std],  capsize=6, error_kw={'lw': 1.5})
    ax.bar(['Post'], [post_mean], color=color, alpha=0.45,
           yerr=[post_std], capsize=6, error_kw={'lw': 1.5})

    # Annotate ratio
    ratio = pre_mean / max(post_mean, 0.1)
    direction = "↓" if ratio > 1 else "↑"
    ax.text(0.5, 0.94,
            f'Pre/post = {ratio:.2f}× {direction}',
            ha='center', va='top', transform=ax.transAxes,
            fontsize=9,
            color='#c0392b' if ratio > 1 else '#27ae60')

    ax.set_title(label, fontsize=9)
    ax.set_ylabel('Mean night detections / day')
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', lw=0.4, alpha=0.5)

plt.suptitle(
    f'FIO1 — Mean night detections: pre vs post {DESTAB_DATE}\n'
    f'(bars = mean ± std,  night window UTC {NIGHT_START_UTC:02d}:00–{NIGHT_END_UTC:02d}:00)',
    fontsize=11
)
plt.tight_layout()

fig_path = os.path.join(OUTPUT_DIR, "fig_compare_pre_post.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"[SAVED] {fig_path}")


# --------------------------------------------------------------------------
# Print summary table
# --------------------------------------------------------------------------
print("\n" + "=" * 65)
print(f"  Night-only summary  (onset: {DESTAB_DATE})")
print(f"  {'Configuration':<30s}  {'Pre mean':>9s}  {'Post mean':>9s}  {'Ratio':>7s}")
print("=" * 65)
for label, df in datasets:
    pre_m  = df.loc[df.index <  t_destab, 'n_night'].mean()
    post_m = df.loc[df.index >= t_destab, 'n_night'].mean()
    ratio  = pre_m / max(post_m, 0.1)
    arrow  = "↓ decrease" if ratio > 1.1 else ("↑ increase" if ratio < 0.9 else "≈ stable")
    print(f"  {label:<30s}  {pre_m:>9.0f}  {post_m:>9.0f}  {ratio:>5.2f}×  {arrow}")
print("=" * 65)



# =============================================================================
# END
# =============================================================================

print("\n[DONE] Comparison figures saved.")
print(f"       Output folder: {OUTPUT_DIR}")
