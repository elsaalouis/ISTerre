"""
FIOS LANDSLIDE — POWER SPECTRAL DENSITY OVER TIME
===================================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : May 2026

Goal
----
Compute the background seismic energy in the 2–10 Hz band for each hour of
the monitoring period, independently of any event detector.

This answers the question: did the background seismic energy (not just
detected events) increase or decrease after April 13?

  - If energy DECREASED  → real seismicity drop  (brittle cracking stopped)
  - If energy INCREASED  → detector was blind     (continuous tremor/noise
    raised the LTA, suppressing event triggers)

Method
------
For each 1-hour MiniSEED file:
  1. Detrend + taper (no bandpass — PSD captures the full spectrum)
  2. Compute PSD with Welch's method (4-s windows, 50 % overlap, Hann)
  3. Integrate PSD over 2–10 Hz  →  hourly band energy [counts² · Hz]

Outputs
-------
  fig_psd_hourly_energy.png   — hourly band energy time series (night highlighted)
  fig_psd_daily_energy.png    — daily median energy + 7-day rolling mean
  fig_psd_spectrogram.png     — time–frequency heatmap (daily median PSD in dB)
  psd_hourly.csv              — raw hourly values
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

DATA_ROOT  = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\data\FIOS"
OUTPUT_DIR = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\fios\04_psd"

NETWORK  = "XT"
STATION  = "FIO1"
LOCATION = "01"
CHANNEL  = "DHZ"

T_START = "2026-03-19"
T_END   = "2026-07-01"   # exclusive

# ---- Multi-band energy analysis -------------------------------------------
# Each tuple (fmin, fmax) produces a separate energy time series + figure
# The first band is also used as the "reference" band in the summary printout
FREQ_BANDS = [
    (1.0,  5.0),   # low-frequency tremor / stick-slip fundamentals
    (5.0,  20.0),  # classical microseismic / impulsive events band
    (20.0, 60.0),  # high-frequency band — turbine harmonics, impact events
    (1.0,  20.0),
]

# Upper frequency limit for the spectrogram heatmap (fig_psd_spectrogram)
FREQ_PLOT_MAX = 60.0

# Night window (UTC)
NIGHT_START_UTC = 18
NIGHT_END_UTC   = 4

# Destabilisation date
DESTAB_DATE = "2026-04-13"

# ---- Key events to annotate on the PSD spectrogram heatmap ----------------
# Each entry: (date_str, label, color)
# These appear as horizontal lines on the time–frequency heatmap.
KEY_EVENTS = [
    ("2026-04-13", "1st destabilisation", "red"),
    ("2026-05-25", "2nd destabilisation", "orange"),
    ("2026-06-23", "3nd destabilisation",  "gold"),
]

# Rolling-mean window (days)
ROLLING_DAYS = 7

# Gap handling
# Recording gaps are filled with zeros by ObsPy (fill_value=0).
# These zero-valued samples dilute the Welch PSD average by a factor of (1 − gap_fraction).
# Correction strategy:
#   gap_fraction < 5 %          → no correction needed
#   5 % ≤ gap_fraction < 95 %  → divide energy by (1 − gap_fraction) to recover true level
#   gap_fraction ≥ 95 %        → NaN  (not enough real signal for a reliable estimate)
GAP_SKIP_THRESHOLD = 0.95   # fraction of zero samples above which the file is discarded



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
import matplotlib.dates  as mdates

from obspy          import UTCDateTime, read, Stream
from scipy.signal   import welch
from scipy.signal.windows import tukey

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: compute Welch PSD of a trace (no bandpass — full spectrum)
# ---------------------------------------------------------------------------
def compute_welch_psd(tr, nperseg):
    """
    Detrend, taper, then compute Welch PSD.
    Returns (freqs [Hz], psd [counts²/Hz], gap_fraction).
    nperseg must be consistent across calls (fixes the frequency axis).

    gap_fraction : float in [0, 1]
        Fraction of raw samples that are zero-filled recording gaps
        (|amplitude| < 0.5 counts). Used by the caller to correct for
        power dilution or to discard unreliable files.
    """
    data = tr.data.astype(float)

    # Detect gap fraction BEFORE any processing (raw counts, gaps = exact zeros)
    gap_fraction = float(np.mean(np.abs(data) < 0.5))

    data -= np.mean(data)
    data -= np.polyval(np.polyfit(np.arange(len(data)), data, 1),
                       np.arange(len(data)))          # linear detrend
    data *= tukey(len(data), alpha=0.05)              # 5 % cosine taper
    freqs, psd = welch(data, fs=tr.stats.sampling_rate,
                       nperseg=nperseg, noverlap=nperseg // 2,  # nperseg number of points per segment, frequency resolution of Δf = 1/4 s = 0.25 Hz < 2-10 Hz
                       window='hann', scaling='density')
    return freqs, psd, gap_fraction


def band_energy(freqs, psd, fmin, fmax):
    """
    Integrate PSD over [fmin, fmax] Hz (trapezoidal). Returns NaN if too few points.
    
    We want a single value representing the energy in the 2–10 Hz band -> this is the integral ∫₂¹⁰ PSD(f) df. 
    The trapezoidal rule (np.trapezoid) calculates this integral numerically by approximating the curve with a 
    series of trapezoids between each consecutive frequency point.
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    if mask.sum() < 2:
        return np.nan
    return float(np.trapezoid(psd[mask], freqs[mask]))



# =============================================================================
# SECTION 3 — MAIN LOOP: day by day, file by file
# =============================================================================

t0   = UTCDateTime(T_START)
t1   = UTCDateTime(T_END)
days = []
d = t0
while d < t1:
    days.append(d)
    d += 86400

band_labels = [f"{fmin:.0f}–{fmax:.0f} Hz" for fmin, fmax in FREQ_BANDS]
print(f"Processing {len(days)} day(s) — station {NETWORK}.{STATION}.{LOCATION}.{CHANNEL}")
print(f"Bands: {', '.join(band_labels)}\n")

nperseg       = None          # initialised from the first file
freqs_ref     = None          # reference frequency axis (fixed for all files)
hourly_rows   = []            # one dict per hourly file
daily_psd_med = {}            # day_str → median PSD array over all hours

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
        print(f"  {day_str} — no files")
        continue

    day_psds = []

    for fpath in files:
        try:
            st = read(fpath)
            st.merge(fill_value=0)
            tr = st[0]
        except Exception as e:
            print(f"    [WARN] {os.path.basename(fpath)}: {e}")
            continue

        if tr.stats.npts < 100:
            continue

        fs = tr.stats.sampling_rate

        # Initialise nperseg once from the first valid file (4-second windows)
        if nperseg is None:  
            nperseg   = int(4 * fs)
            # Compute a dummy PSD to get the reference frequency axis
            freqs_ref, _ = welch(np.zeros(nperseg * 4), fs=fs,
                                  nperseg=nperseg, noverlap=nperseg // 2,
                                  window='hann', scaling='density')
            print(f"  Sampling rate: {fs:.0f} Hz  |  "
                  f"Welch nperseg={nperseg} ({4:.0f} s)  |  "
                  f"df={fs/nperseg:.3f} Hz")

        # Skip files shorter than 2 Welch windows
        if tr.stats.npts < 2 * nperseg:
            continue

        freqs, psd, gap_frac = compute_welch_psd(tr, nperseg)

        t_file   = tr.stats.starttime
        hour_utc = t_file.hour
        is_night = (hour_utc >= NIGHT_START_UTC) or (hour_utc < NIGHT_END_UTC)

        row = {
            'datetime'     : t_file.datetime,
            'day'          : day_str,
            'hour_utc'     : hour_utc,
            'gap_fraction' : gap_frac,
            'is_night'     : is_night,
        }

        if gap_frac >= GAP_SKIP_THRESHOLD:
            print(f"    [GAP]  {os.path.basename(fpath)}: "
                  f"{gap_frac:.0%} zeros → NaN (skipped)")
            for fmin, fmax in FREQ_BANDS:
                key = f"energy_db_{fmin:.0f}_{fmax:.0f}"
                row[key] = np.nan
        else:
            for fmin, fmax in FREQ_BANDS:
                e_raw = band_energy(freqs, psd, fmin, fmax)
                if gap_frac > 0.05:
                    e_raw = e_raw / (1.0 - gap_frac)
                key = f"energy_db_{fmin:.0f}_{fmax:.0f}"
                row[key] = 10 * np.log10(max(e_raw, 1e-30))

        # Keep first band as legacy 'energy_db' for backward-compatible CSV
        row['energy_db'] = row[f"energy_db_{FREQ_BANDS[0][0]:.0f}_{FREQ_BANDS[0][1]:.0f}"]

        hourly_rows.append(row)
        # Only accumulate non-gap files for the daily median PSD heatmap
        if gap_frac < GAP_SKIP_THRESHOLD:
            day_psds.append(psd)

    if day_psds:
        daily_psd_med[day_str] = np.median(np.array(day_psds), axis=0)
        print(f"  {day_str} — {len(day_psds)} file(s) processed")

if not hourly_rows:
    raise RuntimeError("No data processed — check DATA_ROOT and T_START/T_END.")

# Build DataFrames
df_h = pd.DataFrame(hourly_rows)
df_h['datetime'] = pd.to_datetime(df_h['datetime'])
df_h.sort_values('datetime', inplace=True)

# Save hourly CSV
csv_path = os.path.join(OUTPUT_DIR, "psd_hourly.csv")
df_h[['datetime', 'day', 'hour_utc', 'energy_db', 'gap_fraction', 'is_night']].to_csv(
    csv_path, index=False)
print(f"\n[SAVED] {csv_path}")



# =============================================================================
# SECTION 4 — FIGURES
# =============================================================================

t_destab = pd.Timestamp(DESTAB_DATE)

day_mask       = ~df_h['is_night']
night_mask     =  df_h['is_night']
corrected_mask =  df_h['gap_fraction'] > 0.05


# --------------------------------------------------------------------------
# Figures 1 & 2 — one panel per frequency band
# --------------------------------------------------------------------------
n_bands  = len(FREQ_BANDS)
COLORS_B = ['#2c7bb6', '#1a9641', '#d7191c']  # one colour per band

for bi, (fmin, fmax) in enumerate(FREQ_BANDS):
    col      = COLORS_B[bi % len(COLORS_B)]
    edb_col  = f"energy_db_{fmin:.0f}_{fmax:.0f}"
    blabel   = f"{fmin:.0f}–{fmax:.0f} Hz"

    daily_all   = df_h.groupby('day')[edb_col].median()
    daily_night = df_h[night_mask].groupby('day')[edb_col].median()
    daily_all.index   = pd.to_datetime(daily_all.index)
    daily_night.index = pd.to_datetime(daily_night.index)
    rolling_all   = daily_all.rolling(  ROLLING_DAYS, center=True, min_periods=3).mean()
    rolling_night = daily_night.rolling(ROLLING_DAYS, center=True, min_periods=3).mean()

    # --- Figure 1a: hourly scatter -----------------------------------------
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.scatter(df_h.loc[day_mask   & ~corrected_mask, 'datetime'],
               df_h.loc[day_mask   & ~corrected_mask, edb_col],
               s=4, color='#d7191c', alpha=0.4, label='Daytime')
    ax.scatter(df_h.loc[night_mask & ~corrected_mask, 'datetime'],
               df_h.loc[night_mask & ~corrected_mask, edb_col],
               s=6, color=col, alpha=0.7,
               label=f'Night (UTC {NIGHT_START_UTC:02d}–{NIGHT_END_UTC:02d})')
    ax.scatter(df_h.loc[night_mask & corrected_mask, 'datetime'],
               df_h.loc[night_mask & corrected_mask, edb_col],
               s=18, color=col, alpha=0.85, marker='D', linewidths=0.6,
               facecolors='none', edgecolors=col, label='Night (gap-corrected)')
    ax.plot(rolling_night.index.to_pydatetime(), rolling_night.values,
            color=col, lw=2.0, label=f'{ROLLING_DAYS}-day rolling median (night)')
    for ev_date, ev_label, ev_color in KEY_EVENTS:
        ax.axvline(pd.Timestamp(ev_date).to_pydatetime(),
                   color=ev_color, lw=1.4, ls='--', alpha=0.8, label=ev_label)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    plt.xticks(rotation=45, ha='right')
    ax.set_ylabel(f'Band energy [{blabel}]  (dB re counts²·Hz)')
    ax.set_title(
        f'FIO1 — Hourly seismic energy  [{blabel}]  (Welch PSD, no bandpass)\n'
        f'Blue/green/red = night  |  Line = {ROLLING_DAYS}-day rolling median (night)'
    )
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(axis='y', lw=0.3, alpha=0.4)
    plt.tight_layout()
    tag = f"{fmin:.0f}_{fmax:.0f}"
    fig_path = os.path.join(OUTPUT_DIR, f"fig_psd_hourly_energy_{tag}Hz.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[SAVED] {os.path.basename(fig_path)}")

    # --- Figure 1b: daily median bars --------------------------------------
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(daily_all.index.to_pydatetime(),   daily_all.values,
           color='#aec7e8', alpha=0.7, width=0.8, label='All hours (daily median)')
    ax.bar(daily_night.index.to_pydatetime(), daily_night.values,
           color=col, alpha=0.85, width=0.8,
           label=f'Night only (daily median, UTC {NIGHT_START_UTC:02d}–{NIGHT_END_UTC:02d})')
    ax.plot(rolling_all.index.to_pydatetime(),   rolling_all.values,
            color='steelblue', lw=1.5, ls='--', label=f'{ROLLING_DAYS}-d rolling (all)')
    ax.plot(rolling_night.index.to_pydatetime(), rolling_night.values,
            color=col, lw=2.2, label=f'{ROLLING_DAYS}-d rolling (night)')
    for ev_date, ev_label, ev_color in KEY_EVENTS:
        ax.axvline(pd.Timestamp(ev_date).to_pydatetime(),
                   color=ev_color, lw=1.4, ls='--', alpha=0.8, label=ev_label)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    plt.xticks(rotation=45, ha='right')
    ax.set_ylabel('Median band energy  (dB re counts²·Hz)')
    ax.set_title(
        f'FIO1 — Daily median seismic energy  [{blabel}]\n'
        f'Decrease = real seismicity drop  |  Increase = tremor raised LTA → detector blind'
    )
    ax.legend(fontsize=8)
    ax.grid(axis='y', lw=0.3, alpha=0.4)
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f"fig_psd_daily_energy_{tag}Hz.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[SAVED] {os.path.basename(fig_path)}")


# --------------------------------------------------------------------------
# Figure 3 — Time–frequency heatmap (daily median PSD)
# --------------------------------------------------------------------------
if daily_psd_med and freqs_ref is not None:
    sorted_days = sorted(daily_psd_med.keys())
    freq_mask   = freqs_ref <= FREQ_PLOT_MAX
    freqs_plot  = freqs_ref[freq_mask]

    # Build matrix: rows = days, cols = frequencies
    psd_matrix = np.array([
        daily_psd_med[d][freq_mask] for d in sorted_days
    ])
    # Convert to dB
    psd_db = 10 * np.log10(np.maximum(psd_matrix, 1e-30))

    n_days = len(sorted_days)
    fig_h, ax_h = plt.subplots(figsize=(16, max(5, n_days * 0.22)))

    im = ax_h.pcolormesh(
        freqs_plot, range(n_days),
        psd_db,
        cmap='viridis', shading='auto'
    )
    plt.colorbar(im, ax=ax_h, label='PSD  (dB re counts²/Hz)', pad=0.01)

    # Annotate all key events as horizontal lines
    for ev_date, ev_label, ev_color in KEY_EVENTS:
        if ev_date in sorted_days:
            i_ev = sorted_days.index(ev_date)
            ax_h.axhline(i_ev + 0.5, color=ev_color, lw=1.5, ls='--',
                         label=ev_label)
    # Draw vertical guide lines at the band boundaries
    for fmin, fmax in FREQ_BANDS:
        ax_h.axvline(fmin, color='white', lw=0.8, ls=':', alpha=0.6)
        ax_h.axvline(fmax, color='white', lw=0.8, ls=':', alpha=0.6)
    ax_h.legend(fontsize=7, loc='upper right')

    # Y-axis: date labels (every 3 days to avoid crowding)
    tick_idx    = list(range(0, n_days, 3))
    tick_labels = [sorted_days[i] for i in tick_idx]
    ax_h.set_yticks([i + 0.5 for i in tick_idx])
    ax_h.set_yticklabels(tick_labels, fontsize=7)

    ax_h.set_xlabel('Frequency (Hz)')
    ax_h.set_ylabel('Date')
    ax_h.set_xlim(0, FREQ_PLOT_MAX)
    band_desc = "  |  ".join([f"{fmin:.0f}–{fmax:.0f} Hz" for fmin, fmax in FREQ_BANDS])
    ax_h.set_title(
        f'FIO1 — Daily median PSD (time–frequency heatmap, 0–{FREQ_PLOT_MAX:.0f} Hz)\n'
        f'Analysis bands: {band_desc}  |  dashed lines = key events'
    )
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "fig_psd_spectrogram.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig_h)
    print(f"[SAVED] {os.path.basename(fig_path)}")



# =============================================================================
# SECTION 5 — PRINT SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print(f"  PSD SUMMARY  (night only)  —  split at {DESTAB_DATE}")
print("=" * 60)
for fmin, fmax in FREQ_BANDS:
    edb_col    = f"energy_db_{fmin:.0f}_{fmax:.0f}"
    pre_night  = df_h[df_h['is_night'] & (df_h['day'] <  DESTAB_DATE)][edb_col].dropna()
    post_night = df_h[df_h['is_night'] & (df_h['day'] >= DESTAB_DATE)][edb_col].dropna()
    delta = post_night.median() - pre_night.median()
    sign  = "▲ INCREASE" if delta > 0 else "▼ DECREASE"
    print(f"  {fmin:.0f}–{fmax:.0f} Hz :  "
          f"pre={pre_night.median():.1f} dB  post={post_night.median():.1f} dB  "
          f"Δ={delta:+.1f} dB  →  {sign}")
print("=" * 60)

print(f"\n[DONE]  All outputs in: {OUTPUT_DIR}")
