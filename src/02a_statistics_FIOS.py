"""
FIOS LANDSLIDE — EVENT STATISTICS
==================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : May 2026

Goal
----
Read the CSV outputs from 02a_classical_sta_lta_FIOS.py and compute summary statistics to characterise the microseismicity pattern:
  - Weekday vs weekend detection counts
  - Daytime vs night-time split
  - Pre vs post destabilisation comparison (all-hours and night-only)
  - Event duration distribution
  - Diurnal and weekly patterns

Input
-----
  OUTPUT_DIR/run_*/daily_counts.csv        — one row per day
  OUTPUT_DIR/run_*/detections_all_*.csv    — one row per event

Output
------
  statistics_summary.txt  — printed statistics saved as text
  fig_stats_weekday.png   — mean detections by day of week
  fig_stats_hourly.png    — total detections by hour of day (pre vs post)
  fig_stats_duration.png  — event duration histogram (night, pre vs post)
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# Directory that contains the run_* subfolders from 02a
OUTPUT_DIR = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\02a_fios_classical_sta_lta"

# Destabilisation onset
DESTAB_DATE = "2026-04-14"

# Night window (UTC) — must match what was used in 02a
NIGHT_START_UTC = 18   # 18:00 UTC = 20:00 local (CEST)
NIGHT_END_UTC   = 4    # 04:00 UTC = 06:00 local



# =============================================================================
# SECTION 2 — SETUP
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

# ---- Find the most recent run directory -------------------------------------
run_dirs = sorted(glob.glob(os.path.join(OUTPUT_DIR, "run_*")))
if not run_dirs:
    raise FileNotFoundError(
        f"No run_* directories found in {OUTPUT_DIR}.\n"
        "Run 02a_classical_sta_lta_FIOS.py first."
    )
RUN_DIR = run_dirs[-1]
print(f"Using run directory: {RUN_DIR}\n")

# ---- Load daily counts -------------------------------------------------------
counts_csv = os.path.join(RUN_DIR, "daily_counts.csv")
if not os.path.exists(counts_csv):
    raise FileNotFoundError(f"daily_counts.csv not found in {RUN_DIR}")

df_counts = pd.read_csv(counts_csv)
df_counts = df_counts[df_counts['data_available']].copy()
df_counts['date']       = pd.to_datetime(df_counts['day'])
df_counts['dow_num']    = df_counts['date'].dt.dayofweek    # 0=Mon … 6=Sun
df_counts['day_name']   = df_counts['date'].dt.day_name()
df_counts['is_weekend'] = df_counts['dow_num'] >= 5         # Sat & Sun
df_counts['is_pre']     = df_counts['day'] < DESTAB_DATE

has_night = 'n_night' in df_counts.columns

# ---- Load all-detections CSV -------------------------------------------------
det_files = sorted(glob.glob(os.path.join(RUN_DIR, "detections_all_*.csv")))
if det_files:
    df_det = pd.read_csv(det_files[-1])
    df_det['starttime'] = pd.to_datetime(df_det['starttime'], utc=True,
                                          errors='coerce')
    df_det['hour']     = df_det['starttime'].dt.hour
    df_det['dow_num']  = df_det['starttime'].dt.dayofweek
    df_det['is_night'] = ((df_det['hour'] >= NIGHT_START_UTC) |
                          (df_det['hour'] <  NIGHT_END_UTC))
    df_det['is_pre']   = df_det['day'] < DESTAB_DATE
    has_det = True
    print(f"Loaded {len(df_det)} individual detections from {os.path.basename(det_files[-1])}")
else:
    has_det = False
    print("[WARN] No detections_all_*.csv found — per-event statistics skipped.")



# =============================================================================
# SECTION 3 — COMPUTE STATISTICS
# =============================================================================

lines = []

def section(title):
    lines.append("")
    lines.append("=" * 62)
    lines.append(f"  {title}")
    lines.append("=" * 62)

def row(label, value):
    lines.append(f"  {label:<48s} {value}")

# --------------------------------------------------------------------------
# 3.1  OVERVIEW
# --------------------------------------------------------------------------
section("OVERVIEW")
row("Run directory",              os.path.basename(RUN_DIR))
row("Days with data",             str(len(df_counts)))
row("Date range",
    f"{df_counts['day'].min()}  →  {df_counts['day'].max()}")
row("Total detections (all days)",
    str(int(df_counts['n_detections'].sum())))
if has_night:
    row("Total night detections",
        str(int(df_counts['n_night'].sum())))
if has_det:
    row("Individual events in CSV",   str(len(df_det)))

# --------------------------------------------------------------------------
# 3.2  WEEKDAY vs WEEKEND
# --------------------------------------------------------------------------
weekday = df_counts[~df_counts['is_weekend']]
weekend = df_counts[ df_counts['is_weekend']]

section("WEEKDAY vs WEEKEND  —  ALL-HOURS COUNT")
row("Mean detections / weekday",         f"{weekday['n_detections'].mean():.0f}")
row("Mean detections / weekend day",     f"{weekend['n_detections'].mean():.0f}")
ratio = weekday['n_detections'].mean() / max(weekend['n_detections'].mean(), 0.1)
row("Weekday / weekend ratio",           f"{ratio:.1f}×  (weekdays have {ratio:.1f}× more events)")

if has_night:
    section("WEEKDAY vs WEEKEND  —  NIGHT-ONLY COUNT")
    row("Mean night dets / weekday",
        f"{weekday['n_night'].mean():.0f}")
    row("Mean night dets / weekend day",
        f"{weekend['n_night'].mean():.0f}")
    ratio_n = weekday['n_night'].mean() / max(weekend['n_night'].mean(), 0.1)
    row("Weekday / weekend ratio (night)",
        f"{ratio_n:.1f}×  ({'much less' if ratio_n < 1.5 else 'some' if ratio_n < 2.5 else 'large'} weekday bias)")

# --------------------------------------------------------------------------
# 3.3  MEAN DETECTIONS BY DAY OF WEEK
# --------------------------------------------------------------------------
section("MEAN DETECTIONS BY DAY OF WEEK")
dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
for day_name in dow_order:
    mask = df_counts['day_name'] == day_name
    n    = df_counts.loc[mask, 'n_detections'].mean()
    flag = "  ← weekend" if day_name in ('Saturday', 'Sunday') else ""
    row(day_name, f"{n:.0f}{flag}")

# --------------------------------------------------------------------------
# 3.4  PRE vs POST DESTABILISATION
# --------------------------------------------------------------------------
pre  = df_counts[ df_counts['is_pre']]
post = df_counts[~df_counts['is_pre']]

section(f"PRE vs POST DESTABILISATION  (onset: {DESTAB_DATE})")
row("Days before onset (with data)",  str(len(pre)))
row("Days after  onset (with data)",  str(len(post)))

if len(pre) > 0 and len(post) > 0:
    row("Mean dets/day  [pre,  all hours]",  f"{pre['n_detections'].mean():.0f}")
    row("Mean dets/day  [post, all hours]",  f"{post['n_detections'].mean():.0f}")
    ratio_pp = pre['n_detections'].mean() / max(post['n_detections'].mean(), 0.1)
    row("Pre / post ratio  (all hours)",     f"{ratio_pp:.2f}×")

    if has_night:
        row("Mean dets/day  [pre,  night only]",
            f"{pre['n_night'].mean():.0f}")
        row("Mean dets/day  [post, night only]",
            f"{post['n_night'].mean():.0f}")
        ratio_pp_n = pre['n_night'].mean() / max(post['n_night'].mean(), 0.1)
        row("Pre / post ratio  (night only)",
            f"{ratio_pp_n:.2f}×")

# --------------------------------------------------------------------------
# 3.5  PER-EVENT DURATION STATISTICS
# --------------------------------------------------------------------------
if has_det:
    section("EVENT DURATION  —  ALL DETECTIONS")
    row("Count",                  str(len(df_det)))
    row("Mean   (s)",             f"{df_det['duration_s'].mean():.2f}")
    row("Median (s)",             f"{df_det['duration_s'].median():.2f}")
    row("10th percentile (s)",    f"{df_det['duration_s'].quantile(0.10):.2f}")
    row("90th percentile (s)",    f"{df_det['duration_s'].quantile(0.90):.2f}")
    row("Max    (s)",             f"{df_det['duration_s'].max():.1f}")

    night_det = df_det[df_det['is_night']]
    section("EVENT DURATION  —  NIGHT ONLY")
    row("Night events / total",
        f"{len(night_det)} / {len(df_det)}  "
        f"({100*len(night_det)/len(df_det):.1f}%)")
    row("Mean   (s)",             f"{night_det['duration_s'].mean():.2f}")
    row("Median (s)",             f"{night_det['duration_s'].median():.2f}")

    if df_det['is_pre'].any() and (~df_det['is_pre']).any():
        pre_night  = df_det[ df_det['is_pre']  & df_det['is_night']]
        post_night = df_det[~df_det['is_pre']  & df_det['is_night']]
        section("EVENT DURATION  —  NIGHT  PRE vs POST")
        row("Mean duration [pre,  night]",
            f"{pre_night['duration_s'].mean():.2f} s  (n={len(pre_night)})")
        row("Mean duration [post, night]",
            f"{post_night['duration_s'].mean():.2f} s  (n={len(post_night)})")

lines.append("")

# ---- Print & save summary ---------------------------------------------------
summary_text = "\n".join(lines)
print(summary_text)

summary_path = os.path.join(RUN_DIR, "statistics_summary.txt")
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(summary_text)
print(f"\n[SAVED] Summary text → {summary_path}")



# =============================================================================
# SECTION 4 — FIGURES
# =============================================================================

DOW_LABELS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
BAR_COLORS = ['#1f4e79'] * 5 + ['#c0392b', '#c0392b']   # blue weekdays, red weekends

# --------------------------------------------------------------------------
# Figure 1: Mean detections by day of week  (all-hours + night side by side)
# --------------------------------------------------------------------------
n_panels = 2 if has_night else 1
fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
if n_panels == 1:
    axes = [axes]

dow_all = (df_counts.groupby('dow_num')['n_detections']
           .mean().reindex(range(7), fill_value=0))
axes[0].bar(range(7), dow_all.values, color=BAR_COLORS, alpha=0.85)
axes[0].set_xticks(range(7))
axes[0].set_xticklabels(DOW_LABELS)
axes[0].set_ylabel('Mean detections per day')
axes[0].set_title('All-hours count by day of week')
axes[0].grid(axis='y', lw=0.4, alpha=0.5)

if has_night:
    dow_night = (df_counts.groupby('dow_num')['n_night']
                 .mean().reindex(range(7), fill_value=0))
    axes[1].bar(range(7), dow_night.values, color=BAR_COLORS, alpha=0.85)
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(DOW_LABELS)
    axes[1].set_ylabel('Mean night detections per day')
    axes[1].set_title(
        f'Night-only count by day of week\n'
        f'(UTC {NIGHT_START_UTC:02d}:00 – {NIGHT_END_UTC:02d}:00)')
    axes[1].grid(axis='y', lw=0.4, alpha=0.5)

plt.suptitle('FIO1 — Weekly detection pattern  [Classical STA/LTA]',
             fontsize=12, y=1.01)
plt.tight_layout()
fig_path = os.path.join(RUN_DIR, "fig_stats_weekday.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"[SAVED] {os.path.basename(fig_path)}")


# --------------------------------------------------------------------------
# Figure 2: Detections by hour of day — total and pre vs post
# --------------------------------------------------------------------------
if has_det:
    # Night-hour colour mask
    def hour_colors(hours):
        return ['#2c7bb6' if (h >= NIGHT_START_UTC or h < NIGHT_END_UTC)
                else '#d7191c' for h in hours]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Left: total count by hour
    hour_tot = (df_det.groupby('hour').size()
                .reindex(range(24), fill_value=0))
    axes[0].bar(range(24), hour_tot.values,
                color=hour_colors(range(24)), alpha=0.85)
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)],
                             rotation=45, ha='right')
    axes[0].set_ylabel('Total detections')
    axes[0].set_title('Total detections by hour of day\n'
                      '(blue = night window, red = daytime)')
    axes[0].grid(axis='y', lw=0.4, alpha=0.5)

    # Right: pre vs post by hour
    pre_h  = (df_det[ df_det['is_pre']].groupby('hour').size()
              .reindex(range(24), fill_value=0))
    post_h = (df_det[~df_det['is_pre']].groupby('hour').size()
              .reindex(range(24), fill_value=0))
    x = np.arange(24)
    w = 0.4
    axes[1].bar(x - w/2, pre_h.values,  width=w, color='#2166ac',
                alpha=0.85, label=f'Pre  {DESTAB_DATE}')
    axes[1].bar(x + w/2, post_h.values, width=w, color='#d6604d',
                alpha=0.85, label=f'Post {DESTAB_DATE}')
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)],
                             rotation=45, ha='right')
    axes[1].set_ylabel('Total detections')
    axes[1].set_title(f'Pre vs post destabilisation\nby hour of day')
    axes[1].legend(fontsize=8)
    axes[1].grid(axis='y', lw=0.4, alpha=0.5)

    plt.suptitle('FIO1 — Diurnal detection pattern  [Classical STA/LTA]',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fig_path = os.path.join(RUN_DIR, "fig_stats_hourly.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {os.path.basename(fig_path)}")


# --------------------------------------------------------------------------
# Figure 3: Duration histogram — night events, pre vs post
# --------------------------------------------------------------------------
if has_det and df_det['is_pre'].any() and (~df_det['is_pre']).any():
    pre_night  = df_det[ df_det['is_pre']  & df_det['is_night']]['duration_s']
    post_night = df_det[~df_det['is_pre']  & df_det['is_night']]['duration_s']

    cap = min(df_det['duration_s'].quantile(0.98), 30)   # cap at 98th pct
    bins = np.linspace(0, cap, 40)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pre_night,  bins=bins, alpha=0.6, color='#2166ac',
            label=f'Pre  {DESTAB_DATE}  (n={len(pre_night)})')
    ax.hist(post_night, bins=bins, alpha=0.6, color='#d6604d',
            label=f'Post {DESTAB_DATE}  (n={len(post_night)})')
    ax.axvline(pre_night.median(),  color='#2166ac', ls='--', lw=1.5,
               label=f'Median pre  = {pre_night.median():.2f} s')
    ax.axvline(post_night.median(), color='#d6604d', ls='--', lw=1.5,
               label=f'Median post = {post_night.median():.2f} s')
    ax.set_xlabel('Event duration (s)')
    ax.set_ylabel('Count')
    ax.set_title(
        f'FIO1 — Night event duration: pre vs post destabilisation  '
        f'[Classical STA/LTA]\n'
        f'Night window: UTC {NIGHT_START_UTC:02d}:00 – {NIGHT_END_UTC:02d}:00'
    )
    ax.legend(fontsize=9)
    ax.grid(axis='y', lw=0.4, alpha=0.5)
    plt.tight_layout()
    fig_path = os.path.join(RUN_DIR, "fig_stats_duration.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[SAVED] {os.path.basename(fig_path)}")



# --------------------------------------------------------------------------
# Figure 4: Night detection count over time — daily bars + rolling mean
#           → shows whether seismicity drifts upward after destabilisation
# --------------------------------------------------------------------------
if has_night:
    ROLLING_DAYS = 7   # rolling-mean window (days) — adjust if needed

    ts = df_counts.set_index('date').sort_index()
    dates       = ts.index.to_pydatetime()
    night_vals  = ts['n_night'].values.astype(float)

    # Rolling mean (centred, min 3 observations so edges are still drawn)
    rolling = (ts['n_night']
               .rolling(ROLLING_DAYS, center=True, min_periods=3)
               .mean())

    # Linear trend lines for pre and post periods separately
    def _trend(x_dates, y_vals):
        """Return (x_num, y_fitted) for a least-squares line through the data."""
        x_num = np.array([(d - x_dates[0]).days for d in x_dates], dtype=float)
        if len(x_num) < 2:
            return x_num, y_vals * 0
        coeffs = np.polyfit(x_num, y_vals, 1)
        return x_num, np.polyval(coeffs, x_num)

    t_destab = pd.Timestamp(DESTAB_DATE)
    pre_mask  = ts.index <  t_destab
    post_mask = ts.index >= t_destab

    fig, ax = plt.subplots(figsize=(16, 5))

    # Daily bars
    ax.bar(dates, night_vals, color='#1f4e79', alpha=0.55, width=0.85,
           label='Night detections (daily)')

    # Rolling mean
    ax.plot(rolling.index.to_pydatetime(), rolling.values,
            color='#e67e22', lw=2.0, label=f'{ROLLING_DAYS}-day rolling mean')

    # Trend lines
    if pre_mask.sum() >= 3:
        pre_dates = ts.index[pre_mask].to_pydatetime()
        _, pre_fit = _trend(pre_dates, night_vals[pre_mask])
        ax.plot(pre_dates, pre_fit, color='#2980b9', lw=1.5, ls='--',
                label='Linear trend (pre)')

    if post_mask.sum() >= 3:
        post_dates = ts.index[post_mask].to_pydatetime()
        _, post_fit = _trend(post_dates, night_vals[post_mask])
        ax.plot(post_dates, post_fit, color='#c0392b', lw=1.5, ls='--',
                label='Linear trend (post)')

    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    plt.xticks(rotation=45, ha='right')
    ax.set_ylabel('Night detections per day')
    ax.set_ylim(bottom=0)
    ax.set_title(
        f'FIO1 — Night-only microseismicity trend  [Classical STA/LTA]\n'
        f'Band: 10–80 Hz  |  Night window: UTC {NIGHT_START_UTC:02d}:00–{NIGHT_END_UTC:02d}:00  |  '
        f'{ROLLING_DAYS}-day rolling mean  |  Dashed = linear trends'
    )
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', lw=0.4, alpha=0.4)
    plt.tight_layout()

    fig_path = os.path.join(RUN_DIR, "fig_stats_night_trend.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[SAVED] {os.path.basename(fig_path)}")



# =============================================================================
# END
# =============================================================================

print("\n[DONE] All statistics computed and figures saved.")
print(f"       Outputs in: {RUN_DIR}")
