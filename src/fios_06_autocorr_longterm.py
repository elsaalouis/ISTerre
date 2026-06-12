"""
FIOS LANDSLIDE — LONG-TERM AUTOCORRELATION EVOLUTION
=====================================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : June 2026

Goal
----
Track how the autocorrelation coefficient ρ evolves over a multi-day or
multi-week period, to detect transitions between two seismic regimes:

  ρ > 10 %  →  periodic / repetitive signal  (tremor, continuous slip)
  ρ < 5 %   →  random / impulsive signal     (independent microcracks)

The expected pattern for the FIOS landslide:
  - Before 2026-04-13  : low ρ (baseline microcracking, random)
  - Around 2026-05-25  : step increase → tremor phase during second episode
  - After  2026-06-03  : return to lower ρ (system re-stabilises)

Method
------
For each hourly MiniSEED file in [T_START, T_END]:
  1. Skip daytime files if NIGHT_ONLY = True (avoids anthropogenic noise)
  2. Bandpass filter (FILT_FMIN–FILT_FMAX Hz)
  3. Compute ρ on sliding AC_WINDOW_S-second windows (step = AC_STEP_S s)
       ρ = max |r(τ)| / r(0)  for τ ∈ [AC_LAG_MIN, AC_LAG_MAX] s
  4. Skip zero-variance windows (recording gaps filled with zeros)
  5. Aggregate (datetime, ρ) pairs into a multi-day time series

Outputs (all saved to OUTPUT_DIR)
-------
  fig_autocorr_longterm.png  — ρ(t) scatter + rolling median + thresholds
  fig_autocorr_daily.png     — daily median ρ bar chart + rolling mean
  autocorr_longterm.csv      — raw ρ values (one row per 60-s window)
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# All parameters to modify are grouped here.
# =============================================================================

DATA_ROOT  = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\data\FIOS"
OUTPUT_DIR = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\fios\06_autocorr_longterm"

NETWORK  = "XT"
STATION  = "FIO1"
LOCATION = "01"
CHANNEL  = "DHZ"

# ---- Date range -------------------------------------------------------------
T_START = "2026-03-19"   # inclusive
T_END   = "2026-06-10"   # exclusive

# ---- Night filter -----------------------------------------------------------
# If True, only process files whose UTC hour falls in the night window.
# Recommended: avoids anthropogenic daytime signals.
NIGHT_ONLY      = True
NIGHT_START_UTC = 18   # first night hour  (e.g. 18 → UTC 18:00)
NIGHT_END_UTC   = 4    # last  night hour  (e.g.  4 → UTC 04:59)

# ---- Bandpass filter --------------------------------------------------------
FILT_FMIN = 1.0    # Hz
FILT_FMAX = 20.0   # Hz

# ---- Autocorrelation --------------------------------------------------------
AC_WINDOW_S = 60.0   # sliding window length (seconds)
AC_STEP_S   = 30.0   # step between consecutive windows (seconds)
AC_LAG_MIN  = 0.5    # minimum lag searched (s) — avoids trivial lag-0 peak
AC_LAG_MAX  = 10.0   # maximum lag searched (s)

# Gap detection: windows with raw signal variance below this threshold
# are zero-filled recording gaps and are skipped
GAP_VAR_THRESHOLD = 1.0   # counts²

# ---- Smoothing and display --------------------------------------------------
ROLLING_HOURS = 6    # width of the rolling-median window for the long-term plot (hours)
ROLLING_DAYS  = 3    # rolling mean width for the daily bar chart (days)

# ---- Destabilisation markers ------------------------------------------------
DESTAB_DATES = {
    "2026-04-13": "1st destab.",
    "2026-05-25": "2nd destab.",
}



# =============================================================================
# SECTION 2 — SETUP & IMPORTS
# =============================================================================

import os
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot  as plt
import matplotlib.dates   as mdates

from obspy           import UTCDateTime, read
from scipy.signal    import correlate

os.makedirs(OUTPUT_DIR, exist_ok=True)



# =============================================================================
# SECTION 3 — HELPER: AUTOCORRELATION ON ONE TRACE SEGMENT
# =============================================================================

def compute_rho_windows(data, fs, t_start_utc,
                         window_s, step_s, lag_min_s, lag_max_s,
                         gap_var_threshold=1.0):
    """
    Compute the autocorrelation coefficient ρ(t) on sliding windows.

    For each window of window_s seconds:
      1. Skip if raw variance < gap_var_threshold  (zero-filled gap)
      2. Demean the window
      3. Compute the normalised autocorrelation: r(τ) = AC(τ) / AC(0)
      4. ρ = max |r(τ)|  for τ ∈ [lag_min_s, lag_max_s]

    Parameters
    ----------
    data              : 1-D numpy array — bandpassed seismic signal
    fs                : float — sampling rate (Hz)
    t_start_utc       : UTCDateTime — start time of the trace
    window_s          : float — window length (s)
    step_s            : float — step between windows (s)
    lag_min_s         : float — minimum lag (s)
    lag_max_s         : float — maximum lag (s)
    gap_var_threshold : float — variance threshold below which window is skipped

    Returns
    -------
    t_centres : list of datetime — centre time of each valid window
    rho       : list of float   — ρ in [0, 1] for each window
    """
    n_win        = int(window_s * fs)
    n_step       = int(step_s   * fs)
    lag_min_samp = int(lag_min_s * fs)
    lag_max_samp = int(lag_max_s * fs)
    n_data       = len(data)

    t_centres = []
    rho       = []

    i = 0
    while i + n_win <= n_data:
        win = data[i : i + n_win].copy()

        # Skip zero-filled recording gaps
        if np.var(win) < gap_var_threshold:
            i += n_step
            continue

        win -= np.mean(win)

        r_full = correlate(win, win, mode='full')   # 2N-1 values
        r0     = r_full[n_win - 1]                  # lag-0 = signal energy
        if r0 == 0:
            i += n_step
            continue

        r_norm = r_full / r0   # r(0) = 1 by definition

        # Search positive lags only
        r_lags = r_norm[n_win - 1 + lag_min_samp :
                        n_win - 1 + lag_max_samp + 1]
        if len(r_lags) == 0:
            i += n_step
            continue

        rho_val  = float(np.max(np.abs(r_lags)))
        t_centre = (t_start_utc + (i + n_win // 2) / fs).datetime

        t_centres.append(t_centre)
        rho.append(rho_val)

        i += n_step

    return t_centres, rho



# =============================================================================
# SECTION 4 — MAIN LOOP: day by day, file by file
# =============================================================================

def is_night_hour(hour_utc, night_start, night_end):
    """Return True if hour_utc falls in the night window [night_start, night_end)."""
    if night_start > night_end:   # window wraps around midnight
        return (hour_utc >= night_start) or (hour_utc < night_end)
    else:
        return night_start <= hour_utc < night_end

t0   = UTCDateTime(T_START)
t1   = UTCDateTime(T_END)
days = []
d = t0
while d < t1:
    days.append(d)
    d += 86400

print("=" * 65)
print(f"  FIOS Long-term Autocorrelation  —  {T_START} → {T_END}")
print("=" * 65)
print(f"  Station     : {NETWORK}.{STATION}.{LOCATION}.{CHANNEL}")
print(f"  Night only  : {NIGHT_ONLY}  "
      f"(UTC {NIGHT_START_UTC:02d}:00–{NIGHT_END_UTC:02d}:59)")
print(f"  Bandpass    : {FILT_FMIN}–{FILT_FMAX} Hz")
print(f"  AC window   : {AC_WINDOW_S:.0f} s  |  step = {AC_STEP_S:.0f} s  |  "
      f"lags = {AC_LAG_MIN}–{AC_LAG_MAX} s\n")

all_rows      = []   # one dict per 60-s window
n_files_ok    = 0
n_files_skip  = 0

for day_utc in days:
    day_str   = day_utc.strftime('%Y-%m-%d')
    date_str  = day_utc.strftime('%Y%m%d')
    month_str = day_utc.strftime('%Y%m')

    pattern = os.path.join(
        DATA_ROOT, month_str,
        f"{NETWORK}.{STATION}.{LOCATION}.{CHANNEL}_{date_str}_*.miniseed"
    )
    files = sorted(glob.glob(pattern))

    if not files:
        continue

    day_n_windows = 0

    for fpath in files:
        try:
            st = read(fpath)
            st.merge(fill_value=0)
            tr = st[0]
        except Exception as e:
            print(f"    [WARN] {os.path.basename(fpath)}: {e}")
            n_files_skip += 1
            continue

        if tr.stats.npts < 100:
            n_files_skip += 1
            continue

        hour_utc = tr.stats.starttime.hour

        # Night filter
        if NIGHT_ONLY and not is_night_hour(hour_utc, NIGHT_START_UTC, NIGHT_END_UTC):
            continue

        fs = tr.stats.sampling_rate

        # Bandpass filter
        tr_filt = tr.copy()
        tr_filt.filter('bandpass', freqmin=FILT_FMIN, freqmax=FILT_FMAX,
                       corners=4, zerophase=True)

        # Compute ρ windows
        t_centres, rho = compute_rho_windows(
            tr_filt.data, fs, tr_filt.stats.starttime,
            AC_WINDOW_S, AC_STEP_S, AC_LAG_MIN, AC_LAG_MAX,
            gap_var_threshold=GAP_VAR_THRESHOLD
        )

        for t_c, r in zip(t_centres, rho):
            all_rows.append({
                'datetime' : t_c,
                'day'      : day_str,
                'hour_utc' : hour_utc,
                'rho'      : r,
            })

        day_n_windows += len(t_centres)
        n_files_ok    += 1

    if day_n_windows > 0:
        print(f"  {day_str} — {day_n_windows} windows")

if not all_rows:
    raise RuntimeError(
        "No data processed — check DATA_ROOT, T_START/T_END, "
        "and NIGHT_ONLY settings."
    )

print(f"\n  Total : {len(all_rows)} autocorrelation windows  "
      f"({n_files_ok} files processed, {n_files_skip} skipped)")



# =============================================================================
# SECTION 5 — BUILD DATAFRAME AND ROLLING STATISTICS
# =============================================================================

df = pd.DataFrame(all_rows)
df['datetime'] = pd.to_datetime(df['datetime'])
df.sort_values('datetime', inplace=True)
df.set_index('datetime', inplace=True)

# Rolling median (time-based window) — requires DatetimeIndex
roll_win = f"{ROLLING_HOURS}h"
df['rho_rolling'] = (
    df['rho']
    .rolling(roll_win, center=True, min_periods=5)
    .median()
)

# Daily median
daily_rho = df.groupby('day')['rho'].median()
daily_rho.index = pd.to_datetime(daily_rho.index)
daily_rolling = daily_rho.rolling(ROLLING_DAYS, center=True, min_periods=2).mean()

# Save CSV
df_save = df.reset_index()[['datetime', 'day', 'hour_utc', 'rho']]
csv_path = os.path.join(OUTPUT_DIR, "autocorr_longterm.csv")
df_save.to_csv(csv_path, index=False)
print(f"\n[SAVED] {os.path.basename(csv_path)}")

# Summary statistics
print(f"\n  ρ across full period :  "
      f"median = {df['rho'].median():.3f}  |  "
      f"mean = {df['rho'].mean():.3f}  |  "
      f"max = {df['rho'].max():.3f}")



# =============================================================================
# SECTION 6 — FIGURES
# =============================================================================

date_fmt_full  = mdates.DateFormatter('%b %d')
date_fmt_daily = mdates.DateFormatter('%b %d')

destab_colors = ['#333333', '#c0392b']   # dark grey for 1st, red for 2nd

# --------------------------------------------------------------------------
# Figure 1 — ρ(t) scatter + rolling median + thresholds + destab markers
# --------------------------------------------------------------------------
print("\nGenerating Figure 1 — long-term ρ(t) time series ...")

fig, ax = plt.subplots(figsize=(18, 6))

# Individual windows (light scatter, small points)
ax.scatter(
    df.index, df['rho'],
    s=3, color='#5b9bd5', alpha=0.25, rasterized=True,
    label=f'ρ per {AC_WINDOW_S:.0f}-s window'
)

# Rolling median (bold line)
ax.plot(
    df.index, df['rho_rolling'],
    color='#1a3f6f', lw=2.2,
    label=f'{ROLLING_HOURS}-h rolling median'
)

# Interpretation thresholds
ax.axhline(0.10, color='#d62728', lw=1.5, ls='--',
           label='ρ = 10 %  (tremor threshold, Provost et al. 2017)')
ax.axhline(0.05, color='#ff7f0e', lw=1.0, ls=':',
           label='ρ = 5 %  (lower bound)')

# Destabilisation markers
for (date_str, label), col in zip(DESTAB_DATES.items(), destab_colors):
    t_d = pd.Timestamp(date_str)
    if pd.Timestamp(T_START) <= t_d <= pd.Timestamp(T_END):
        ax.axvline(t_d, color=col, lw=1.8, ls='-', alpha=0.8,
                   label=f'Destab. onset ({date_str}) — {label}')

ax.set_ylabel('ρ  (normalised autocorrelation coefficient)', fontsize=10)
ax.set_xlabel('Date (UTC)', fontsize=10)

night_label = f'Night only  (UTC {NIGHT_START_UTC:02d}:00–{NIGHT_END_UTC:02d}:00)' \
              if NIGHT_ONLY else 'All hours'
ax.set_title(
    f'FIO1  —  Long-term autocorrelation  [{T_START} → {T_END}]\n'
    f'Bandpass {FILT_FMIN}–{FILT_FMAX} Hz  |  window = {AC_WINDOW_S:.0f} s  |  '
    f'lags = {AC_LAG_MIN}–{AC_LAG_MAX} s  |  {night_label}',
    fontsize=10
)

ax.set_ylim(0, min(1.0, float(df['rho'].quantile(0.995)) * 1.15))
ax.xaxis.set_major_formatter(date_fmt_full)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax.legend(fontsize=8, loc='upper left', ncol=2)
ax.grid(axis='y', lw=0.3, alpha=0.4)

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "fig_autocorr_longterm.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  [SAVED] {os.path.basename(fig_path)}")


# --------------------------------------------------------------------------
# Figure 2 — Daily median ρ bar chart + rolling mean
# --------------------------------------------------------------------------
print("Generating Figure 2 — daily median ρ ...")

fig, ax = plt.subplots(figsize=(16, 5))

# Colour bars by regime: above 10 % → red, 5–10 % → orange, below 5 % → blue
bar_colors = []
for v in daily_rho.values:
    if np.isnan(v):
        bar_colors.append('#cccccc')
    elif v >= 0.10:
        bar_colors.append('#d62728')
    elif v >= 0.05:
        bar_colors.append('#ff7f0e')
    else:
        bar_colors.append('#5b9bd5')

ax.bar(
    daily_rho.index.to_pydatetime(), daily_rho.values,
    color=bar_colors, alpha=0.8, width=0.8,
    label='Daily median ρ'
)
ax.plot(
    daily_rolling.index.to_pydatetime(), daily_rolling.values,
    color='#1a3f6f', lw=2.2,
    label=f'{ROLLING_DAYS}-day rolling mean'
)

ax.axhline(0.10, color='#d62728', lw=1.5, ls='--',
           label='ρ = 10 %  (tremor)')
ax.axhline(0.05, color='#ff7f0e', lw=1.0, ls=':',
           label='ρ = 5 %  (lower bound)')

for (date_str, label), col in zip(DESTAB_DATES.items(), destab_colors):
    t_d = pd.Timestamp(date_str)
    if pd.Timestamp(T_START) <= t_d <= pd.Timestamp(T_END):
        ax.axvline(t_d, color=col, lw=1.8, ls='-', alpha=0.8,
                   label=f'{date_str} — {label}')

ax.set_ylabel('Daily median ρ', fontsize=10)
ax.set_xlabel('Date (UTC)', fontsize=10)
ax.set_title(
    f'FIO1  —  Daily median autocorrelation coefficient  [{T_START} → {T_END}]\n'
    f'Blue < 5 % (microcracks)   |   Orange 5–10 % (mixed)   |   Red > 10 % (tremor)',
    fontsize=10
)

ax.set_ylim(0, min(1.0, float(daily_rho.max()) * 1.2 + 0.02))
ax.xaxis.set_major_formatter(date_fmt_daily)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax.legend(fontsize=8, loc='upper left', ncol=2)
ax.grid(axis='y', lw=0.3, alpha=0.4)

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "fig_autocorr_daily.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  [SAVED] {os.path.basename(fig_path)}")



# =============================================================================
# SECTION 7 — PRINT SUMMARY
# =============================================================================

print("\n" + "=" * 65)
print(f"  LONG-TERM AUTOCORRELATION SUMMARY  —  {T_START} → {T_END}")
print("=" * 65)

# Per-period stats around each destabilisation date
for date_str, label in DESTAB_DATES.items():
    t_d = pd.Timestamp(date_str)
    if t_d < pd.Timestamp(T_START) or t_d > pd.Timestamp(T_END):
        continue
    pre  = df.loc[df.index < t_d, 'rho']
    post = df.loc[df.index >= t_d, 'rho']
    if len(pre) == 0 or len(post) == 0:
        continue
    print(f"\n  Relative to {date_str} ({label}):")
    print(f"    Before  :  median ρ = {pre.median():.3f}  "
          f"(n = {len(pre)} windows)")
    print(f"    After   :  median ρ = {post.median():.3f}  "
          f"(n = {len(post)} windows)")
    delta = post.median() - pre.median()
    print(f"    Change  :  {delta:+.3f}  "
          f"({'INCREASE → tremor-like' if delta > 0 else 'DECREASE → less periodic'})")

pct_above_10 = float(np.mean(df['rho'] > 0.10)) * 100
pct_above_05 = float(np.mean(df['rho'] > 0.05)) * 100
print(f"\n  Over the full period:")
print(f"    % windows with ρ > 10 % : {pct_above_10:.1f} %  (tremor-like)")
print(f"    % windows with ρ > 5 %  : {pct_above_05:.1f} %  (above lower bound)")
print("=" * 65)

print(f"\n[DONE]  All outputs saved to:")
print(f"        {OUTPUT_DIR}")



# =============================================================================
# END
# =============================================================================
