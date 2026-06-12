"""
FIOS LANDSLIDE — HIGH-RESOLUTION NIGHT SPECTROGRAM
====================================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : June 2026

Goal
----
Investigate the nature of the seismic signal during a high-activity night to distinguish between two possible sources:

  Tremor / stick-slip   → harmonic spectral lines (f0, 2f0, 3f0 ...) possibly with "gliding" (slow frequency drift over time)
                          continuous and emergent waveform envelope high autocorrelation coefficient (ρ > 10 %)

  Microcrack impulses   → broadband, short-duration bursts (< 2 s) triangular spectrogram shape (high-f first, then decay)
                          impulsive envelope with clear P-wave onsets low autocorrelation coefficient (ρ < 5 %)

Method
------
For the chosen night (UTC 18:00 → UTC 04:00 next day, i.e. 10 hours):
  1. Load and concatenate all 1-hour MiniSEED files into one continuous trace
  2. Detrend, taper, bandpass filter (FILT_FMIN–FILT_FMAX Hz)
  3. Compute a high-resolution spectrogram using scipy.signal.spectrogram
       - SPECGRAM_WINDOW_S-second Hann windows
       - 90 % overlap  →  time step ≈ 1 s,  Δf = 1/SPECGRAM_WINDOW_S Hz
       - Sufficient to resolve harmonic lines separated by ~1 Hz
  4. Compute the autocorrelation coefficient ρ on sliding 60-s windows
       ρ(t) = max |r(τ)|  for τ ∈ [AC_LAG_MIN, AC_LAG_MAX]
       ρ > 10 %  →  periodic/repetitive signal  (tremor)
       ρ < 5 %   →  random/impulsive signal      (microcracks)

Outputs (all saved to OUTPUT_DIR)
-------
  fig_spectrogram_night.png   — waveform + full-night high-resolution spectrogram
  fig_autocorrelation.png     — waveform + ρ(t) time series with interpretation thresholds
  fig_spectrogram_zoom.png    — 30-min zoom on the most active window
                                (guides at 2, 4, 6, 8, 10 Hz for harmonic reading)
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# All parameters to modify are grouped here.
# =============================================================================

DATA_ROOT  = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\data\FIOS"
OUTPUT_DIR = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\fios\05_night_spectrogram"

NETWORK  = "XT"
STATION  = "FIO1"
LOCATION = "01"
CHANNEL  = "DHZ"

# ---- Target night ----------------------------------------------------------
# Good candidates for now:
#   "2026-05-26"  →  second destabilisation episode
#   "2026-04-14"  →  first destabilisation onset
TARGET_DATE = "2026-05-21"

# ---- Bandpass filter -------------------------------------------------------
FILT_FMIN = 1.0    # Hz — lower corner
FILT_FMAX = 20.0   # Hz — upper corner

# ---- Spectrogram -----------------------------------------------------------
# Window length controls the trade-off between time and frequency resolution:
#   longer window  →  finer Δf  (better harmonic separation),  coarser Δt
#   shorter window →  finer Δt  (better impulse localisation), coarser Δf
SPECGRAM_WINDOW_S = 10.0    # Hann window length in seconds
SPECGRAM_OVERLAP  = 0.90    # fraction: 90 % → time step ≈ 1 s at 250 Hz
FREQ_MIN_PLOT     = 0.5     # Hz — lower bound of colour axis
FREQ_MAX_PLOT     = 20.0    # Hz — upper bound of colour axis

# Colour scale (dB) (set to None to use automatic percentile clipping)
VMIN_DB = None
VMAX_DB = None

# ---- Autocorrelation -------------------------------------------------------
# ρ quantifies how periodic the signal is within each sliding window
# For each window of AC_WINDOW_S seconds, ρ = max|r(τ)| for τ in the lag range
# r(τ) is the normalized autocorrelation of the bandpassed signal
AC_WINDOW_S = 60.0   # sliding window length (seconds)
AC_STEP_S   = 30.0   # step between consecutive windows (seconds)
AC_LAG_MIN  = 0.5    # minimum lag to search (s) — avoids trivial lag-0 = 1
AC_LAG_MAX  = 10.0   # maximum lag to search (s) — looks for periods up to 10 s

# ---- Zoom window -----------------------------------------------------------
# The 30-min zoom spectrogram is centred on the loudest window (auto-detected).
# Override with an explicit UTC string to fix it manually, e.g.:
#   ZOOM_START_UTC = "2026-05-27T00:30:00"
ZOOM_START_UTC    = None   # None → auto-detect loudest 30-min block
ZOOM_DURATION_MIN = 30



# =============================================================================
# SECTION 2 — SETUP & IMPORTS
# =============================================================================

import os
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy  as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot  as plt
import matplotlib.dates   as mdates

from obspy              import UTCDateTime, read, Stream
from scipy.signal       import correlate, spectrogram as sp_spectrogram
from scipy.signal.windows import hann as hann_window

os.makedirs(OUTPUT_DIR, exist_ok=True)



# =============================================================================
# SECTION 3 — LOAD & ASSEMBLE NIGHT TRACE
# =============================================================================
# The night window spans two calendar days (18:00 → 04:00 next day)
# We scan both days' MiniSEED files, keep only files within the UTC window, and merge them into a single continuous ObsPy trace
# =============================================================================

t_night_start = UTCDateTime(TARGET_DATE + "T18:00:00")
t_night_end   = t_night_start + 10 * 3600    # 10 hours later

print("=" * 65)
print(f"  FIOS Night Spectrogram  —  {TARGET_DATE}")
print("=" * 65)
print(f"  Night window : {t_night_start}  →  {t_night_end}")
print(f"  Station      : {NETWORK}.{STATION}.{LOCATION}.{CHANNEL}")
print(f"  Bandpass     : {FILT_FMIN}–{FILT_FMAX} Hz\n")

all_files = []
t_current = t_night_start

while t_current < t_night_end:
    date_str  = t_current.strftime('%Y%m%d')
    month_str = t_current.strftime('%Y%m')
    pattern   = os.path.join(
        DATA_ROOT, month_str,
        f"{NETWORK}.{STATION}.{LOCATION}.{CHANNEL}_{date_str}_*.miniseed"
    )
    for fpath in sorted(glob.glob(pattern)):
        try:
            # Quick header read to check start time
            st_hdr = read(fpath, headonly=True)
            if st_hdr and (t_night_start <= st_hdr[0].stats.starttime < t_night_end):
                all_files.append(fpath)
        except Exception:
            pass
    t_current += 3600    # advance by one hour

print(f"  Found {len(all_files)} hourly file(s) for the night window.")
if not all_files:
    raise RuntimeError(
        "No MiniSEED files found — check DATA_ROOT and TARGET_DATE."
    )

# Load, merge, and filter the full night trace
st = Stream()
for fpath in all_files:
    try:
        st += read(fpath)
    except Exception as e:
        print(f"  [WARN] Cannot read {os.path.basename(fpath)}: {e}")

st.merge(method=1, fill_value=0)
st.detrend('demean')
st.detrend('linear')

tr = st.select(channel=CHANNEL)[0]
fs = tr.stats.sampling_rate

print(f"\n  Assembled trace : {tr.stats.starttime}  →  {tr.stats.endtime}")
print(f"  Sampling rate   : {fs:.0f} Hz")
print(f"  Duration        : {tr.stats.npts / fs / 3600:.2f} h  ({tr.stats.npts} samples)")

# Bandpass-filtered copy used for all figures and computations
tr_filt = tr.copy()
tr_filt.filter('bandpass', freqmin=FILT_FMIN, freqmax=FILT_FMAX,
               corners=4, zerophase=True)



# =============================================================================
# SECTION 4 — HELPER FUNCTIONS
# =============================================================================

def compute_spectrogram(data, fs, window_s, overlap_frac):
    """
    High-resolution spectrogram using scipy.signal.spectrogram.

    Parameters
    ----------
    data        : 1-D numpy array — bandpassed seismic signal
    fs          : float — sampling rate in Hz
    window_s    : float — Hann window length in seconds
                  Controls frequency resolution: Δf = 1 / window_s
    overlap_frac: float in [0, 1) — fraction of overlap between windows
                  Controls time resolution: Δt ≈ window_s × (1 − overlap_frac)

    Returns
    -------
    t_sec  : 1-D array — time in seconds from start of data
    f_hz   : 1-D array — frequency in Hz
    Sxx_db : 2-D array (n_freq × n_time) — PSD in dB re counts²/Hz
    """
    nperseg  = int(window_s * fs)    # if = 2500 points -> covers 10 s of signal
    noverlap = int(nperseg * overlap_frac)

    f_hz, t_sec, Sxx = sp_spectrogram(
        data,
        fs       = fs,
        window   = hann_window(nperseg),  # 10 s window
        nperseg  = nperseg,
        noverlap = noverlap,   # 90 % overlapping -> time step = 10x(1-0.9) = 1 s -> frequency resolution = 1/10 = 0.1 Hz
        scaling  = 'density',
        mode     = 'psd'
    )
    Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-30))
    return t_sec, f_hz, Sxx_db


def compute_autocorr_series(data, fs, t_start_utc,
                            window_s, step_s, lag_min_s, lag_max_s):
    """
    Sliding autocorrelation coefficient ρ(t)

    For each window of length window_s seconds:
      1. Demean the window
      2. Compute the full normalized autocorrelation r(τ) = AC(τ) / AC(0)
      3. ρ = max |r(τ)|  for τ ∈ [lag_min_s, lag_max_s]

    Physical interpretation
    -----------------------
    A purely sinusoidal signal has r(τ) = cos(2π f τ)  → ρ = 1.0
    White noise has r(τ) ≈ 0 for all τ ≠ 0          → ρ ≈ 0
    Tremor (quasi-periodic)                           → ρ > 10 %
    Independent microcrack impulses (random)          → ρ < 5 %

    Parameters
    ----------
    data         : 1-D numpy array — bandpassed seismic signal
    fs           : float — sampling rate
    t_start_utc  : UTCDateTime — start time of the trace (for time axis)
    window_s     : float — window length in seconds
    step_s       : float — step between consecutive windows in seconds
    lag_min_s    : float — minimum lag to search (s), > 0 to avoid lag-0
    lag_max_s    : float — maximum lag to search (s)

    Returns
    -------
    t_centres : list of datetime — centre time of each window
    rho       : list of float — ρ value in [0, 1] for each window
    """
    n_win        = int(window_s * fs)
    n_step       = int(step_s * fs)
    lag_min_samp = int(lag_min_s * fs)
    lag_max_samp = int(lag_max_s * fs)
    n_data       = len(data)

    t_centres = []
    rho       = []

    i = 0
    while i + n_win <= n_data:
        win = data[i : i + n_win].copy()

        # Skip gap-filled (zero-variance) sections
        if np.var(win) < 1e-20:
            i += n_step
            continue

        win -= np.mean(win)

        r_full = correlate(win, win, mode='full') # full autocorrelation via scipy (returns 2N-1 values)
        r0     = r_full[n_win - 1]      # index (N-1) corresponds to lag = 0 (= signal energy)
        if r0 == 0:
            i += n_step
            continue

        r_norm = r_full / r0    # normalize so r(0) = 1

        # Positive lags in [lag_min_samp, lag_max_samp]
        r_lags = r_norm[n_win - 1 + lag_min_samp :
                        n_win - 1 + lag_max_samp + 1]
        if len(r_lags) == 0:
            i += n_step
            continue

        rho_val   = float(np.max(np.abs(r_lags)))
        t_centre  = (t_start_utc + (i + n_win // 2) / fs).datetime

        t_centres.append(t_centre)
        rho.append(rho_val)

        i += n_step

    return t_centres, rho



# =============================================================================
# SECTION 5 — COMPUTE SPECTROGRAM AND AUTOCORRELATION
# =============================================================================

print("\nComputing high-resolution spectrogram ...")
t_sec, f_hz, Sxx_db = compute_spectrogram(
    tr_filt.data, fs, SPECGRAM_WINDOW_S, SPECGRAM_OVERLAP
)

# --- Mask spectrogram columns that correspond to recording gaps -----------
# Gap regions are filled with 0 (fill_value=0 in st.merge)
#  -> those windows produce Sxx ≈ 1e-30 → Sxx_db ≈ -300 dB, which collapses the colorscale and hides all contrast in the real signal

nperseg_spec = int(SPECGRAM_WINDOW_S * fs)
i_ctrs = np.round(t_sec * fs).astype(int)
gap_cols = np.array([                       # detect zero-variance windows from the RAW (unfiltered) trace
    np.var(tr.data[max(0, i - nperseg_spec // 2) :
                   min(len(tr.data), i + nperseg_spec // 2)]) < 1.0
    for i in i_ctrs
])
n_gap_cols = int(gap_cols.sum())
Sxx_db[:, gap_cols] = np.nan            # replace those spectrogram columns with NaN
print(f"  Gap masking: {n_gap_cols} spectrogram columns set to NaN "
      f"({n_gap_cols / max(1, len(t_sec)) * 100:.1f} % of total)")

# Convert t_sec (seconds from trace start) to absolute datetimes
t_abs_spec = np.array([
    (tr_filt.stats.starttime + float(t)).datetime for t in t_sec
])

# Restrict frequency axis for plotting
freq_mask = (f_hz >= FREQ_MIN_PLOT) & (f_hz <= FREQ_MAX_PLOT)
f_plot    = f_hz[freq_mask]
S_plot    = Sxx_db[freq_mask, :]     # shape: (n_freq, n_time)

# Colour scale — use nanpercentile so NaN gap columns don't affect the range
vmin = VMIN_DB if VMIN_DB is not None else float(np.nanpercentile(S_plot, 5))
vmax = VMAX_DB if VMAX_DB is not None else float(np.nanpercentile(S_plot, 99))

# Colormap: inferno with NaN rendered as black (same visual language as gap = no data)
cmap_spec = plt.cm.inferno.copy()
cmap_spec.set_bad(color='black')
print(f"  Spectrogram shape : {S_plot.shape}  (freq × time)")
print(f"  Δf = {1/SPECGRAM_WINDOW_S:.2f} Hz  |  "
      f"Δt ≈ {SPECGRAM_WINDOW_S * (1 - SPECGRAM_OVERLAP):.1f} s")
print(f"  Colour range      : [{vmin:.1f}, {vmax:.1f}] dB")

print("\nComputing autocorrelation coefficient ρ ...")
t_ac, rho = compute_autocorr_series(
    tr_filt.data, fs, tr_filt.stats.starttime,
    AC_WINDOW_S, AC_STEP_S, AC_LAG_MIN, AC_LAG_MAX
)
rho_arr = np.array(rho)
print(f"  {len(rho)} windows  |  "
      f"median ρ = {np.median(rho_arr):.3f}  |  "
      f"max ρ = {np.max(rho_arr):.3f}")

# Downsampled waveform for plots (keep max 10 000 points for speed)
step_ds  = max(1, tr_filt.stats.npts // 10000)
t_wave   = np.array([
    (tr_filt.stats.starttime + i / fs).datetime
    for i in range(0, tr_filt.stats.npts, step_ds)
])
d_wave   = tr_filt.data[::step_ds]
date_fmt = mdates.DateFormatter('%H:%M')



# =============================================================================
# SECTION 6 — FIGURES
# =============================================================================

# --------------------------------------------------------------------------
# Figure 1 — Full night: waveform + high-resolution spectrogram
# --------------------------------------------------------------------------
print("\nGenerating Figure 1 — full-night spectrogram ...")

fig, (ax_w, ax_s) = plt.subplots(
    2, 1, figsize=(18, 9),
    gridspec_kw={'height_ratios': [1, 3]},
    sharex=True
)

ax_w.plot(t_wave, d_wave, color='black', lw=0.4, rasterized=True)
ax_w.set_ylabel('Amplitude (counts)', fontsize=9)
ax_w.set_title(
    f'FIO1  —  Night {TARGET_DATE}  (UTC 18:00 → +10 h)  '
    f'|  Bandpass {FILT_FMIN}–{FILT_FMAX} Hz\n'
    f'Spectrogram: {SPECGRAM_WINDOW_S:.0f}-s Hann windows, '
    f'{int(SPECGRAM_OVERLAP * 100)} % overlap  →  '
    f'Δf = {1/SPECGRAM_WINDOW_S:.2f} Hz,  '
    f'Δt ≈ {SPECGRAM_WINDOW_S * (1 - SPECGRAM_OVERLAP):.0f} s',
    fontsize=10
)
ax_w.grid(axis='y', lw=0.3, alpha=0.4)

im = ax_s.pcolormesh(
    t_abs_spec, f_plot, S_plot,
    cmap=cmap_spec, vmin=vmin, vmax=vmax,
    shading='auto', rasterized=True
)
cbar = plt.colorbar(im, ax=ax_s, pad=0.01, fraction=0.015)
cbar.set_label('PSD (dB re counts²/Hz)', fontsize=8)

ax_s.set_ylabel('Frequency (Hz)', fontsize=9)
ax_s.set_xlabel(f'Time UTC  ({TARGET_DATE}  18:00 → next day 04:00)', fontsize=9)
ax_s.set_ylim(FREQ_MIN_PLOT, FREQ_MAX_PLOT)
ax_s.xaxis.set_major_formatter(date_fmt)
ax_s.xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.setp(ax_s.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, f"fig_spectrogram_night_{TARGET_DATE}.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  [SAVED] {os.path.basename(fig_path)}")


# --------------------------------------------------------------------------
# Figure 2 — Waveform + autocorrelation coefficient ρ(t)
# --------------------------------------------------------------------------
print("Generating Figure 2 — autocorrelation ρ(t) ...")

fig, (ax_w2, ax_r) = plt.subplots(
    2, 1, figsize=(18, 6),
    gridspec_kw={'height_ratios': [1, 1.5]},
    sharex=True
)

ax_w2.plot(t_wave, d_wave, color='black', lw=0.4, rasterized=True)
ax_w2.set_ylabel('Amplitude (counts)', fontsize=9)
ax_w2.set_title(
    f'FIO1  —  Night {TARGET_DATE}  |  '
    f'Autocorrelation coefficient ρ\n'
    f'Window = {AC_WINDOW_S:.0f} s,  step = {AC_STEP_S:.0f} s,  '
    f'lags searched: {AC_LAG_MIN}–{AC_LAG_MAX} s  '
    f'(ρ > 10 %  →  tremor-like  |  ρ < 5 %  →  microcrack-like)',
    fontsize=10
)
ax_w2.grid(axis='y', lw=0.3, alpha=0.4)

ax_r.fill_between(t_ac, rho, 0,
                  color='#2c7bb6', alpha=0.3)
ax_r.plot(t_ac, rho,
          color='#2c7bb6', lw=1.2, label='ρ(t)')
ax_r.axhline(0.10, color='#d62728', lw=1.4, ls='--',
             label='ρ = 10 %  (tremor threshold, Provost et al. 2017)')
ax_r.axhline(0.05, color='#ff7f0e', lw=1.0, ls=':',
             label='ρ = 5 %  (lower bound)')
ax_r.set_ylabel('ρ  (norm. autocorr.)', fontsize=9)
ax_r.set_xlabel(f'Time UTC  ({TARGET_DATE}  18:00 → next day 04:00)', fontsize=9)
ax_r.set_ylim(0, min(1.0, max(0.30, float(np.max(rho_arr)) * 1.2)))
ax_r.legend(fontsize=8, loc='upper right')
ax_r.grid(axis='y', lw=0.3, alpha=0.4)
ax_r.xaxis.set_major_formatter(date_fmt)
ax_r.xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.setp(ax_r.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, f"fig_autocorrelation_{TARGET_DATE}.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  [SAVED] {os.path.basename(fig_path)}")


# --------------------------------------------------------------------------
# Figure 3 — 30-minute zoom spectrogram on the most active window
# --------------------------------------------------------------------------
print("Generating Figure 3 — 30-min zoom spectrogram ...")

# Auto-detect the loudest 30-min block in 2–10 Hz
if ZOOM_START_UTC is None:
    band_mask_zoom  = (f_plot >= 2.0) & (f_plot <= 10.0)
    energy_per_time = S_plot[band_mask_zoom, :].mean(axis=0)    # mean energy between 2-10Hz for each spectrogram's column (each sec)

    dt_spec    = float(t_sec[1] - t_sec[0]) if len(t_sec) > 1 else 1.0
    n_zoom_col = max(1, int(ZOOM_DURATION_MIN * 60 / dt_spec))
    kernel     = np.ones(n_zoom_col) / n_zoom_col
    smoothed   = np.convolve(energy_per_time, kernel, mode='valid')  # smooth the result over a 30-minute window
    best_col   = int(np.argmax(smoothed))   # start index of loudest window (max energy)

    t_zoom_s   = tr_filt.stats.starttime + float(t_sec[best_col])
else:
    t_zoom_s   = UTCDateTime(ZOOM_START_UTC)

t_zoom_e = t_zoom_s + ZOOM_DURATION_MIN * 60
print(f"  Zoom window : {t_zoom_s}  →  {t_zoom_e}")

# Extract zoom slice
zoom_spec_mask = (t_abs_spec >= t_zoom_s.datetime) & \
                 (t_abs_spec <= t_zoom_e.datetime)
zoom_wave_mask = (t_wave >= t_zoom_s.datetime) & \
                 (t_wave <= t_zoom_e.datetime)

if zoom_spec_mask.sum() < 10:
    print("  [WARN] Zoom window has too few columns — skipping Figure 3.")
else:
    S_zoom  = S_plot[:, zoom_spec_mask]
    t_zoom  = t_abs_spec[zoom_spec_mask]
    d_zoom  = d_wave[zoom_wave_mask]
    tw_zoom = t_wave[zoom_wave_mask]

    vmin_z = float(np.nanpercentile(S_zoom, 5))
    vmax_z = float(np.nanpercentile(S_zoom, 99))

    fig, (az_w, az_s) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={'height_ratios': [1, 2.5]},
        sharex=True
    )

    az_w.plot(tw_zoom, d_zoom, color='black', lw=0.7, rasterized=True)
    az_w.set_ylabel('Amplitude (counts)', fontsize=9)
    az_w.set_title(
        f'FIO1  —  {ZOOM_DURATION_MIN}-min zoom  |  '
        f'{t_zoom_s.datetime.strftime("%Y-%m-%d %H:%M")} UTC  →  '
        f'{t_zoom_e.datetime.strftime("%H:%M")} UTC\n'
        f'Δf = {1/SPECGRAM_WINDOW_S:.2f} Hz  '
        f'— white guides at 2, 4, 6, 8, 10, 12 Hz  '
        f'— look for harmonic lines (tremor) vs vertical streaks (microcracks)',
        fontsize=10
    )
    az_w.grid(axis='y', lw=0.3, alpha=0.4)

    im_z = az_s.pcolormesh(
        t_zoom, f_plot, S_zoom,
        cmap=cmap_spec, vmin=vmin_z, vmax=vmax_z,
        shading='auto', rasterized=True
    )
    cbar_z = plt.colorbar(im_z, ax=az_s, pad=0.01, fraction=0.015)
    cbar_z.set_label('PSD (dB re counts²/Hz)', fontsize=8)

    # Horizontal guide lines every 2 Hz to help read harmonic structure
    for f_g in [2, 4, 6, 8, 10, 12, 14]:
        if FREQ_MIN_PLOT < f_g < FREQ_MAX_PLOT:
            az_s.axhline(f_g, color='white', lw=0.5, ls='--', alpha=0.5)

    az_s.set_ylabel('Frequency (Hz)', fontsize=9)
    az_s.set_xlabel('Time UTC', fontsize=9)
    az_s.set_ylim(FREQ_MIN_PLOT, FREQ_MAX_PLOT)
    az_s.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    az_s.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    plt.setp(az_s.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f"fig_spectrogram_zoom_{TARGET_DATE}.png")
    plt.savefig(fig_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {os.path.basename(fig_path)}")



# =============================================================================
# SECTION 7 — PRINT SUMMARY
# =============================================================================

rho_median  = float(np.median(rho_arr))
rho_max_val = float(np.max(rho_arr))
pct_above   = float(np.mean(rho_arr > 0.10)) * 100

print("\n" + "=" * 65)
print(f"  AUTOCORRELATION SUMMARY  —  night {TARGET_DATE}")
print(f"  Lags searched  : {AC_LAG_MIN}–{AC_LAG_MAX} s  "
      f"|  window = {AC_WINDOW_S:.0f} s")
print("=" * 65)
print(f"  Median ρ           : {rho_median:.3f}")
print(f"  Maximum ρ          : {rho_max_val:.3f}")
print(f"  % windows > 10 %   : {pct_above:.1f} %")
print()
if rho_median > 0.10:
    print("  → HIGH ρ  :  signal is predominantly PERIODIC / REPETITIVE")
    print("    Consistent with TREMOR or rapid stick-slip at a stable frequency.")
    print("    Check fig_spectrogram_zoom.png for harmonic lines at f0, 2f0, 3f0.")
elif rho_median > 0.05:
    print("  → MODERATE ρ  :  mixed signal.")
    print("    Possible interpretation: rapid microcrack bursts with partial")
    print("    overlap, OR tremor episodes alternating with quieter windows.")
    print("    Compare spectrogram zoom (vertical vs horizontal features).")
else:
    print("  → LOW ρ  :  signal is predominantly RANDOM / IMPULSIVE.")
    print("    Consistent with a sequence of INDEPENDENT MICROCRACK events.")
    print("    Check fig_spectrogram_zoom.png for broadband vertical streaks.")
print("=" * 65)

print(f"\n[DONE]  All outputs saved to:")
print(f"        {OUTPUT_DIR}")



# =============================================================================
# END
# =============================================================================
