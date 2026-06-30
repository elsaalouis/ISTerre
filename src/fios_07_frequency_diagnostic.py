"""
FIOS LANDSLIDE — FREQUENCY RANGE DIAGNOSTIC
============================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis

Goal
----
Determine the usable frequency band of station FIO1 by computing raw (unfiltered) PSD on quiet periods

Two complementary outputs:
  1. fig_psd_noise_floor.png  — median PSD for each selected period, overlaid on the same log-log axes
     Shows: Nyquist limit, current 1-20 Hz band, sensor roll-off at low frequencies,electronic noise floor at high frequencies
  2. fig_psd_day_vs_night.png — daytime vs night-time median PSD for the same quiet dates → reveals anthropogenic peaks (turbines, machinery) that appear only by day

How to read the plots
---------------------
  - Flat plateau in the middle  → instrument response is flat, this is the useful band
  - Roll-off on the left        → sensor natural frequency (lower limit)
  - Roll-off or floor on the right → electronic noise floor (upper limit ≈ Nyquist)
  - Sharp vertical peaks         → tonal noise (turbine harmonics, machinery)
  - Peaks stronger by day        → anthropogenic source
"""


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

DATA_ROOT  = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\data\FIOS"
OUTPUT_DIR = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\fios\07_frequency_diagnostic"

NETWORK  = "XT"
STATION  = "FIO1"
LOCATION = "01"
CHANNEL  = "DHZ"

# Quiet dates: choose nights clearly before any detected activity (early March)
# The script will load NIGHT_HOURS hours starting at NIGHT_START_UTC for each date, AND a DAY window (DAY_START_UTC → DAY_START_UTC + DAY_HOURS) for day/night comparison.
QUIET_DATES = [
    "2026-03-20",
    "2026-03-23",
    "2026-03-27",
    "2026-03-31",
]

NIGHT_START_UTC = 18   # 18:00 UTC = 20:00 local (CEST)
NIGHT_HOURS     = 8    # hours of night data to use per date

DAY_START_UTC   = 7    # 07:00 UTC = 09:00 local (working hours)
DAY_HOURS       = 8    # hours of daytime data to use per date

# Length of each Welch segment in seconds
# Longer → better frequency resolution (Δf = 1/nperseg_s), heavier compute
WELCH_NPERSEG_S = 120    # 120 s -> Δf = 0.008 Hz, good for resolving sharp tonal peaks

# Current analysis band (plotted as a reference shaded region)
CURRENT_FMIN = 1.0
CURRENT_FMAX = 60.0


# =============================================================================
# SECTION 2 — IMPORTS & HELPERS
# =============================================================================

import os
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy  as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from obspy        import UTCDateTime, Stream, read
from scipy.signal import welch
from scipy.signal.windows import tukey

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Load a time window spanning multiple hourly MiniSEED files
# ---------------------------------------------------------------------------

def load_window(data_root, t_start, duration_h, network, station, location, channel):
    """
    Load `duration_h` hours starting at t_start (UTCDateTime).
    Globs hourly MiniSEED files across as many calendar days as needed,
    merges them, and trims to the exact requested window.

    Returns (tr, n_files) or (None, 0) if no data.
    """
    st = Stream()
    t_end = t_start + duration_h * 3600

    # Collect all calendar days spanned by the window
    day = UTCDateTime(t_start.strftime('%Y-%m-%d'))
    while day < t_end:
        month_str = day.strftime('%Y%m')
        date_str  = day.strftime('%Y%m%d')
        pattern   = os.path.join(
            data_root, month_str,
            f"{network}.{station}.{location}.{channel}_{date_str}_*.miniseed"
        )
        for f in sorted(glob.glob(pattern)):
            try:
                st += read(f)
            except Exception as e:
                print(f"  [WARN] Skipping {os.path.basename(f)}: {e}")
        day += 86400

    if len(st) == 0:
        return None, 0

    try:
        st.merge(fill_value=0)
    except Exception:
        st.merge(method=0, fill_value=0)

    st.trim(t_start, t_end)

    if len(st) == 0 or st[0].stats.npts == 0:
        return None, 0

    return st[0], len(st)


# ---------------------------------------------------------------------------
# Compute median PSD over a trace using Welch's method (no filter)
# ---------------------------------------------------------------------------

def compute_psd(tr, nperseg_s):
    """
    Detrend + taper, then Welch PSD on the raw (unfiltered) trace.
    Returns (freqs [Hz], psd [counts²/Hz]).
    nperseg_s : segment length in seconds → Δf = 1/nperseg_s Hz.
    """
    fs      = tr.stats.sampling_rate
    nperseg = int(nperseg_s * fs)
    noverlap = nperseg // 2

    data = tr.data.astype(float)
    data -= np.mean(data)
    # Linear detrend
    data -= np.polyval(np.polyfit(np.arange(len(data)), data, 1),
                       np.arange(len(data)))
    # 5% cosine taper to reduce edge effects
    data *= tukey(len(data), alpha=0.05)

    freqs, psd = welch(data, fs=fs,
                       nperseg=nperseg, noverlap=noverlap,
                       window='hann', scaling='density')
    return freqs, psd


# =============================================================================
# SECTION 3 — LOAD DATA & COMPUTE PSDs
# =============================================================================

print(f"\n{'='*60}")
print(f"  FIO1 — Frequency range diagnostic")
print(f"  Welch segment length : {WELCH_NPERSEG_S} s  →  Δf = {1/WELCH_NPERSEG_S:.4f} Hz")
print(f"{'='*60}\n")

night_psds = []   # list of (date_str, freqs, psd)
day_psds   = []

nyquist = None

for date_str in QUIET_DATES:

    # --- Night window ----------------------------------------------------------
    t_night = UTCDateTime(date_str) + NIGHT_START_UTC * 3600
    tr_n, n_n = load_window(DATA_ROOT, t_night, NIGHT_HOURS,
                             NETWORK, STATION, LOCATION, CHANNEL)
    if tr_n is not None:
        nyquist = tr_n.stats.sampling_rate / 2.0
        freqs_n, psd_n = compute_psd(tr_n, WELCH_NPERSEG_S)
        night_psds.append((date_str, freqs_n, psd_n))
        print(f"  {date_str} NIGHT  — fs={tr_n.stats.sampling_rate:.0f} Hz  "
              f"Nyquist={nyquist:.0f} Hz  npts={tr_n.stats.npts}")
    else:
        print(f"  {date_str} NIGHT  — [NO DATA]")

    # --- Day window ------------------------------------------------------------
    t_day = UTCDateTime(date_str) + DAY_START_UTC * 3600
    tr_d, n_d = load_window(DATA_ROOT, t_day, DAY_HOURS,
                             NETWORK, STATION, LOCATION, CHANNEL)
    if tr_d is not None:
        freqs_d, psd_d = compute_psd(tr_d, WELCH_NPERSEG_S)
        day_psds.append((date_str, freqs_d, psd_d))
        print(f"  {date_str} DAY    — fs={tr_d.stats.sampling_rate:.0f} Hz")
    else:
        print(f"  {date_str} DAY    — [NO DATA]")

if not night_psds:
    print("\n[ERROR] No data loaded — check DATA_ROOT and QUIET_DATES.")
    raise SystemExit(1)


# =============================================================================
# SECTION 4 — PLOT 1 : NOISE FLOOR (individual nights + median)
# =============================================================================

fig, ax = plt.subplots(figsize=(14, 6))

COLORS = ['#4878CF', '#6ACC65', '#D65F5F', '#B47CC7']

all_psd_arrays = []
freqs_ref = night_psds[0][1]   # common frequency axis

for i, (date_str, freqs, psd) in enumerate(night_psds):
    psd_db = 10 * np.log10(np.maximum(psd, 1e-30))
    ax.plot(freqs[1:], psd_db[1:],   # skip DC (f=0)
            color=COLORS[i % len(COLORS)], lw=0.8, alpha=0.6,
            label=f'{date_str} (night {NIGHT_START_UTC:02d}h–'
                  f'{(NIGHT_START_UTC + NIGHT_HOURS) % 24:02d}h UTC)')
    all_psd_arrays.append(np.interp(freqs_ref, freqs, psd))

# Median across all nights
median_psd    = np.median(all_psd_arrays, axis=0)
median_psd_db = 10 * np.log10(np.maximum(median_psd, 1e-30))
ax.plot(freqs_ref[1:], median_psd_db[1:],
        color='black', lw=2.0, label='Median (all nights)', zorder=5)

# Current analysis band
ax.axvspan(CURRENT_FMIN, CURRENT_FMAX, color='gold', alpha=0.15,
           label=f'Current band ({CURRENT_FMIN}–{CURRENT_FMAX} Hz)', zorder=0)
ax.axvline(CURRENT_FMIN, color='goldenrod', lw=1.2, ls='--')
ax.axvline(CURRENT_FMAX, color='goldenrod', lw=1.2, ls='--')

# Nyquist
if nyquist:
    ax.axvline(nyquist, color='red', lw=1.5, ls=':',
               label=f'Nyquist = {nyquist:.0f} Hz  (fs = {nyquist*2:.0f} Hz)')

ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)', fontsize=12)
ax.set_ylabel('PSD (dB re counts²/Hz)', fontsize=12)
ax.set_title(
    f"FIO1 — Raw (unfiltered) PSD on quiet nights — {', '.join(QUIET_DATES)}\n"
    f"Welch: {WELCH_NPERSEG_S}s windows, 50% overlap, Hann  →  Δf = {1/WELCH_NPERSEG_S:.3f} Hz\n"
    f"Use this plot to identify: sensor roll-off (left), noise floor (right), tonal peaks",
    fontsize=10
)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, which='both', lw=0.3, alpha=0.5)
ax.set_xlim(0.05, nyquist if nyquist else 100)

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "fig_psd_noise_floor.png")
plt.savefig(fig_path, dpi=150)
plt.close(fig)
print(f"\n[SAVED] {fig_path}")


# =============================================================================
# SECTION 5 — PLOT 2 : DAY vs NIGHT (anthropogenic noise check)
# =============================================================================

if day_psds and night_psds:

    fig, ax = plt.subplots(figsize=(14, 6))

    # All individual nights (thin, blue)
    for date_str, freqs, psd in night_psds:
        psd_db = 10 * np.log10(np.maximum(psd, 1e-30))
        ax.plot(freqs[1:], psd_db[1:],
                color='steelblue', lw=0.6, alpha=0.35)

    # All individual days (thin, orange)
    for date_str, freqs, psd in day_psds:
        psd_db = 10 * np.log10(np.maximum(psd, 1e-30))
        ax.plot(freqs[1:], psd_db[1:],
                color='darkorange', lw=0.6, alpha=0.35)

    # Night median
    night_arrays = [np.interp(freqs_ref, f, p) for _, f, p in night_psds]
    night_med_db = 10 * np.log10(np.maximum(np.median(night_arrays, axis=0), 1e-30))
    ax.plot(freqs_ref[1:], night_med_db[1:],
            color='steelblue', lw=2.0,
            label=f'Night median  (UTC {NIGHT_START_UTC:02d}h–'
                  f'{(NIGHT_START_UTC + NIGHT_HOURS) % 24:02d}h, '
                  f'≈ {NIGHT_START_UTC+2:02d}h–{(NIGHT_START_UTC+NIGHT_HOURS+2)%24:02d}h local)')

    # Day median
    day_arrays = [np.interp(freqs_ref, f, p) for _, f, p in day_psds]
    day_med_db = 10 * np.log10(np.maximum(np.median(day_arrays, axis=0), 1e-30))
    ax.plot(freqs_ref[1:], day_med_db[1:],
            color='darkorange', lw=2.0,
            label=f'Day median  (UTC {DAY_START_UTC:02d}h–'
                  f'{DAY_START_UTC + DAY_HOURS:02d}h, '
                  f'≈ {DAY_START_UTC+2:02d}h–{DAY_START_UTC+DAY_HOURS+2:02d}h local)')

    # Difference (day - night) on secondary axis
    ax2 = ax.twinx()
    diff_db = day_med_db - night_med_db
    ax2.fill_between(freqs_ref[1:], 0, diff_db[1:],
                     where=diff_db[1:] > 0,
                     color='red', alpha=0.15, label='Day louder than night')
    ax2.fill_between(freqs_ref[1:], 0, diff_db[1:],
                     where=diff_db[1:] < 0,
                     color='blue', alpha=0.10, label='Night louder than day')
    ax2.axhline(0, color='gray', lw=0.8, ls='--')
    ax2.set_ylabel('Day − Night (dB)', fontsize=10, color='dimgray')
    ax2.tick_params(axis='y', labelcolor='dimgray')
    ax2.set_ylim(-15, 25)

    # Reference lines
    ax.axvspan(CURRENT_FMIN, CURRENT_FMAX, color='gold', alpha=0.12,
               label=f'Current band ({CURRENT_FMIN}–{CURRENT_FMAX} Hz)', zorder=0)
    if nyquist:
        ax.axvline(nyquist, color='red', lw=1.2, ls=':',
                   label=f'Nyquist = {nyquist:.0f} Hz')

    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('PSD (dB re counts²/Hz)', fontsize=12)
    ax.set_title(
        f"FIO1 — Day vs Night PSD on quiet dates ({', '.join(QUIET_DATES)})\n"
        f"Peaks louder by day → anthropogenic source (machinery, turbine, traffic)\n"
        f"Peaks present night & day → geological or permanent noise",
        fontsize=10
    )

    # Merge legends from both axes
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, fontsize=8, loc='upper right')

    ax.grid(True, which='both', lw=0.3, alpha=0.5)
    ax.set_xlim(0.05, nyquist if nyquist else 100)

    plt.tight_layout()
    fig_path2 = os.path.join(OUTPUT_DIR, "fig_psd_day_vs_night.png")
    plt.savefig(fig_path2, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {fig_path2}")
else:
    print("[INFO] Not enough day/night data for comparison plot — skipping.")


# =============================================================================
# END
# =============================================================================

print(f"\n{'='*60}")
print(f"  Outputs in : {OUTPUT_DIR}")
print(f"  Dates used : {', '.join(QUIET_DATES)}")
if nyquist:
    print(f"  Sampling rate : {nyquist*2:.0f} Hz  →  Nyquist = {nyquist:.0f} Hz")
print(f"  Welch Δf : {1/WELCH_NPERSEG_S:.4f} Hz  (segments of {WELCH_NPERSEG_S}s)")
print(f"{'='*60}\n")
