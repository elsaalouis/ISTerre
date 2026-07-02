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
                          continuous and emergent waveform envelope 

  Microcrack impulses   → broadband, short-duration bursts (< 2 s) triangular spectrogram shape (high-f first, then decay)
                          impulsive envelope with clear P-wave onsets 

Method
------
For each chosen night (UTC 18:00 → UTC 04:00 next day, i.e. 10 hours):
  1. Load and concatenate all 1-hour MiniSEED files into one continuous trace
  2. Detrend, taper, bandpass filter (FILT_FMIN–FILT_FMAX Hz)
  3. Compute a high-resolution spectrogram using scipy.signal.spectrogram
       - SPECGRAM_WINDOW_S-second Hann windows
       - 90 % overlap  →  time step ≈ 1 s,  Δf = 1/SPECGRAM_WINDOW_S Hz
       - Sufficient to resolve harmonic lines separated by ~1 Hz

Outputs (all saved to OUTPUT_DIR / one subfolder per date)
-------
  fig_spectrogram_night_YYYYMMDD.png  — waveform + full-night high-resolution spectrogram
  fig_spectrogram_zoom_YYYYMMDD.png   — 30-min zoom on the most active window
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

# ---- Target nights ---------------------------------------------------------
TARGET_DATES = [
    "2026-03-24","2026-03-28","2026-04-01","2026-04-04","2026-04-08",
    "2026-04-11","2026-04-12","2026-04-13","2026-04-14","2026-04-15",
    "2026-04-16","2026-04-17","2026-04-18","2026-04-19","2026-04-20",
    "2026-04-21","2026-04-22","2026-04-23","2026-04-24","2026-04-25",
    "2026-04-29","2026-05-03","2026-05-08","2026-05-10","2026-05-14",
    "2026-05-17","2026-05-21","2026-05-24","2026-05-25","2026-05-26",
    "2026-05-27","2026-05-28","2026-05-29","2026-05-30","2026-05-31",
    "2026-06-01","2026-06-02","2026-06-03","2026-06-04","2026-06-05",
    "2026-06-08","2026-06-12","2026-06-16","2026-06-20","2026-06-22",
    "2026-06-23","2026-06-24","2026-06-25","2026-06-26","2026-06-27",
    "2026-06-28","2026-06-29","2026-06-30",
]

"""
    "2026-03-24","2026-03-28","2026-04-01","2026-04-04","2026-04-08",
    "2026-04-11","2026-04-12","2026-04-13","2026-04-14","2026-04-15",
    "2026-04-16","2026-04-17","2026-04-18","2026-04-19","2026-04-20",
    "2026-04-21","2026-04-22","2026-04-23","2026-04-24","2026-04-25",
    "2026-04-29","2026-05-03","2026-05-08","2026-05-10","2026-05-14",
    "2026-05-17","2026-05-21","2026-05-24","2026-05-25","2026-05-26",
    "2026-05-27","2026-05-28","2026-05-29","2026-05-30","2026-05-31",
    "2026-06-01","2026-06-02","2026-06-03","2026-06-04","2026-06-05",
    "2026-06-08","2026-06-12","2026-06-16","2026-06-20","2026-06-22",
    "2026-06-23","2026-06-24","2026-06-25","2026-06-26","2026-06-27",
    "2026-06-28","2026-06-29","2026-06-30",
"""

# ---- Time window -----------------------------------------------------------
WINDOW_START_UTC  = 18   # UTC hour at which the window begins (0–23)
WINDOW_DURATION_H = 10   # duration in hours

# ---- Per-date zoom override -------------------------------------------------
# By default the zoom window is auto-detected (loudest 30-min block in 2-10 Hz)
# To fix a specific zoom start for a given date:
#   ZOOM_OVERRIDES = {
#       "2026-04-14": "2026-04-13T18:00:00",
#       "2026-05-25": "2026-05-26T00:15:00",
#   }
ZOOM_OVERRIDES = {}

# ---- Bandpass filter -------------------------------------------------------
FILT_FMIN = 1.0    # Hz — lower corner
FILT_FMAX = 60.0   # Hz — upper corner

# ---- Spectrogram -----------------------------------------------------------
# Window length controls the trade-off between time and frequency resolution:
#   longer window  →  finer Δf  (better harmonic separation),  coarser Δt
#   shorter window →  finer Δt  (better impulse localisation), coarser Δf
SPECGRAM_WINDOW_S = 10.0    # Hann window length in seconds
SPECGRAM_OVERLAP  = 0.90    # fraction: 90 % → time step ≈ 1 s at 250 Hz
FREQ_MIN_PLOT     = 1.0     # Hz — lower bound of colour axis
FREQ_MAX_PLOT     = 60.0    # Hz — upper bound of colour axis

# ---- Colour scale (dB) -------------------------------------------------------
# VSCALE_MODE controls how the spectrogram colour axis is set:
#   "per_night"  — each night auto-scales independently (p5–p99 of that night)#
#   "global"     — a first pass scans all nights to find the global p5 and p99, then uses these for ALL plots
#   "fixed"      — uses the exact values in VMIN_DB / VMAX_DB below
VSCALE_MODE = "fixed"  

# Used only when VSCALE_MODE = "fixed" (ignored otherwise):
VMIN_DB = -15
VMAX_DB = 55

# Percentiles used to clip the colour scale (both "per_night" and "global")
VSCALE_PLOW  =  2   # lower percentile  (e.g. 2 → clips the darkest 2 %)
VSCALE_PHIGH = 99   # upper percentile  (e.g. 99 → clips the brightest 1 %)

# ---- Zoom window -----------------------------------------------------------
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
# SECTION 3 — HELPER FUNCTIONS
# =============================================================================

def load_night(data_root, target_date, network, station, location, channel,
               filt_fmin, filt_fmax,
               window_start_utc=18, window_duration_h=10):
    """
    Load a configurable time window starting at window_start_utc (UTC hour)
    on target_date and lasting window_duration_h hours.
    Returns (tr_raw, tr_filt, fs) or raises RuntimeError if no data.

    Default: night window (UTC 18:00 → 04:00 next day, 10 h).
    For a daytime window pass e.g. window_start_utc=6, window_duration_h=12.
    """
    t_night_start = (UTCDateTime(target_date + "T00:00:00")
                     + window_start_utc * 3600)
    t_night_end   = t_night_start + window_duration_h * 3600

    all_files = []
    t_current = t_night_start
    while t_current < t_night_end:
        date_str  = t_current.strftime('%Y%m%d')
        month_str = t_current.strftime('%Y%m')
        pattern   = os.path.join(
            data_root, month_str,
            f"{network}.{station}.{location}.{channel}_{date_str}_*.miniseed"
        )
        for fpath in sorted(glob.glob(pattern)):
            try:
                st_hdr = read(fpath, headonly=True)
                if st_hdr and (t_night_start <= st_hdr[0].stats.starttime < t_night_end):
                    all_files.append(fpath)
            except Exception:
                pass
        t_current += 3600

    print(f"  Found {len(all_files)} hourly file(s) for the night window.")
    if not all_files:
        raise RuntimeError(
            f"No MiniSEED files found for {target_date} — check DATA_ROOT."
        )

    st = Stream()
    for fpath in all_files:
        try:
            st += read(fpath)
        except Exception as e:
            print(f"  [WARN] Cannot read {os.path.basename(fpath)}: {e}")

    st.merge(method=1, fill_value=0)
    st.detrend('demean')
    st.detrend('linear')

    tr = st.select(channel=channel)[0]
    fs = tr.stats.sampling_rate

    tr_filt = tr.copy()
    tr_filt.filter('bandpass', freqmin=filt_fmin, freqmax=filt_fmax,
                   corners=4, zerophase=True)

    return tr, tr_filt, fs


def compute_spectrogram(data, fs, window_s, overlap_frac):
    """
    High-resolution spectrogram using scipy.signal.spectrogram.
    Returns (t_sec, f_hz, Sxx_db).
    """
    nperseg  = int(window_s * fs)
    noverlap = int(nperseg * overlap_frac)
    f_hz, t_sec, Sxx = sp_spectrogram(
        data, fs=fs,
        window=hann_window(nperseg),
        nperseg=nperseg, noverlap=noverlap,
        scaling='density', mode='psd'
    )
    Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-30))
    return t_sec, f_hz, Sxx_db



# =============================================================================
# SECTION 4 — MAIN LOOP (one iteration per date)
# =============================================================================

_w_end_utc = (WINDOW_START_UTC + WINDOW_DURATION_H) % 24
_window_label = (f"UTC {WINDOW_START_UTC:02d}:00 → "
                 f"{_w_end_utc:02d}:00"
                 + (" (+1 day)" if WINDOW_DURATION_H + WINDOW_START_UTC >= 24 else ""))

print(f"\n{'='*65}")
print(f"  FIOS Spectrogram  —  {len(TARGET_DATES)} date(s) to process")
print(f"  Station : {NETWORK}.{STATION}.{LOCATION}.{CHANNEL}")
print(f"  Window  : {_window_label}  ({WINDOW_DURATION_H} h)")
print(f"  Band    : {FILT_FMIN}–{FILT_FMAX} Hz")
print(f"  Scale   : {VSCALE_MODE}")
print(f"{'='*65}\n")

# ---- Pre-scan : compute global colour scale if requested --------------------
_global_vmin = None
_global_vmax = None

if VSCALE_MODE == "global":
    print("  Pre-scan: computing global colour scale across all dates ...")
    _p_lows, _p_highs = [], []
    for _date in TARGET_DATES:
        try:
            _, _tr_f, _fs = load_night(
                DATA_ROOT, _date,
                NETWORK, STATION, LOCATION, CHANNEL,
                FILT_FMIN, FILT_FMAX,
                window_start_utc=WINDOW_START_UTC,
                window_duration_h=WINDOW_DURATION_H,
            )
            _, _f_hz, _Sxx = compute_spectrogram(
                _tr_f.data, _fs, SPECGRAM_WINDOW_S, SPECGRAM_OVERLAP
            )
            _fmask = (_f_hz >= FREQ_MIN_PLOT) & (_f_hz <= FREQ_MAX_PLOT)
            _S = _Sxx[_fmask, :]
            _p_lows.append(float(np.nanpercentile(_S, VSCALE_PLOW)))
            _p_highs.append(float(np.nanpercentile(_S, VSCALE_PHIGH)))
            del _tr_f, _Sxx, _S
            print(f"    {_date}  →  p{VSCALE_PLOW}={_p_lows[-1]:.1f} dB  "
                  f"p{VSCALE_PHIGH}={_p_highs[-1]:.1f} dB")
        except RuntimeError:
            print(f"    {_date}  →  [NO DATA — skipped]")

    if _p_lows:
        _global_vmin = float(np.min(_p_lows))
        _global_vmax = float(np.max(_p_highs))
        print(f"\n  Global scale : [{_global_vmin:.1f}, {_global_vmax:.1f}] dB  "
              f"(p{VSCALE_PLOW} min → p{VSCALE_PHIGH} max across all dates)\n")
    else:
        print("  [WARN] No data found during pre-scan — falling back to per_night.")
        VSCALE_MODE = "per_night"

for TARGET_DATE in TARGET_DATES:

    print(f"\n{'='*65}")
    print(f"  Processing : {TARGET_DATE}  |  {_window_label}")
    print(f"{'='*65}")

    # ---- Load ----------------------------------------------------------------
    try:
        tr, tr_filt, fs = load_night(
            DATA_ROOT, TARGET_DATE,
            NETWORK, STATION, LOCATION, CHANNEL,
            FILT_FMIN, FILT_FMAX,
            window_start_utc=WINDOW_START_UTC,
            window_duration_h=WINDOW_DURATION_H,
        )
    except RuntimeError as e:
        print(f"  [SKIP] {e}")
        continue

    print(f"  Assembled trace : {tr.stats.starttime}  →  {tr.stats.endtime}")
    print(f"  Sampling rate   : {fs:.0f} Hz  |  "
          f"Duration : {tr.stats.npts / fs / 3600:.2f} h")

    # ---- Compute spectrogram -------------------------------------------------
    print("  Computing spectrogram ...")
    t_sec, f_hz, Sxx_db = compute_spectrogram(
        tr_filt.data, fs, SPECGRAM_WINDOW_S, SPECGRAM_OVERLAP
    )

    # Mask gap columns (zero-variance windows)
    nperseg_spec = int(SPECGRAM_WINDOW_S * fs)
    i_ctrs = np.round(t_sec * fs).astype(int)
    gap_cols = np.array([
        np.var(tr.data[max(0, i - nperseg_spec // 2) :
                       min(len(tr.data), i + nperseg_spec // 2)]) < 1.0
        for i in i_ctrs
    ])
    Sxx_db[:, gap_cols] = np.nan
    print(f"  Gap masking: {gap_cols.sum()} columns set to NaN "
          f"({gap_cols.sum() / max(1, len(t_sec)) * 100:.1f} %)")

    t_abs_spec = np.array([
        (tr_filt.stats.starttime + float(t)).datetime for t in t_sec
    ])

    freq_mask = (f_hz >= FREQ_MIN_PLOT) & (f_hz <= FREQ_MAX_PLOT)
    f_plot    = f_hz[freq_mask]
    S_plot    = Sxx_db[freq_mask, :]

    if VSCALE_MODE == "fixed" and VMIN_DB is not None and VMAX_DB is not None:
        vmin, vmax = VMIN_DB, VMAX_DB
    elif VSCALE_MODE == "global" and _global_vmin is not None:
        vmin, vmax = _global_vmin, _global_vmax
    else:  # "per_night" or fallback
        vmin = float(np.nanpercentile(S_plot, VSCALE_PLOW))
        vmax = float(np.nanpercentile(S_plot, VSCALE_PHIGH))

    cmap_spec = plt.cm.inferno.copy()
    cmap_spec.set_bad(color='black')

    print(f"  Spectrogram : {S_plot.shape}  |  "
          f"Δf={1/SPECGRAM_WINDOW_S:.2f} Hz  "
          f"Δt≈{SPECGRAM_WINDOW_S*(1-SPECGRAM_OVERLAP):.1f}s  "
          f"[{vmin:.1f}, {vmax:.1f}] dB")

    # Downsampled waveform for plots
    step_ds = max(1, tr_filt.stats.npts // 10000)
    t_wave  = np.array([
        (tr_filt.stats.starttime + i / fs).datetime
        for i in range(0, tr_filt.stats.npts, step_ds)
    ])
    d_wave   = tr_filt.data[::step_ds]
    date_fmt = mdates.DateFormatter('%H:%M')

    # ---- Figure 1 — Full-night spectrogram -----------------------------------
    print("  Saving Figure 1 — full-night spectrogram ...")
    fig, (ax_w, ax_s) = plt.subplots(
        2, 1, figsize=(18, 9),
        gridspec_kw={'height_ratios': [1, 3]},
        sharex=True
    )
    ax_w.plot(t_wave, d_wave, color='black', lw=0.4, rasterized=True)
    ax_w.set_ylabel('Amplitude (counts)', fontsize=9)
    ax_w.set_title(
        f'FIO1  —  {TARGET_DATE}  ({_window_label})  '
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
    ax_s.set_xlabel(
        f'Time UTC  ({TARGET_DATE}  {_window_label})', fontsize=9
    )
    ax_s.set_ylim(FREQ_MIN_PLOT, FREQ_MAX_PLOT)
    ax_s.xaxis.set_major_formatter(date_fmt)
    ax_s.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.setp(ax_s.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f"fig_spectrogram_night_{TARGET_DATE}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {os.path.basename(fig_path)}")

    # ---- Figure 3 — 30-min zoom spectrogram ----------------------------------
    print("  Saving Figure 3 — 30-min zoom ...")
    ZOOM_START_UTC = ZOOM_OVERRIDES.get(TARGET_DATE, None)

    if ZOOM_START_UTC is None:
        band_mask_zoom  = (f_plot >= 2.0) & (f_plot <= 10.0)
        energy_per_time = S_plot[band_mask_zoom, :].mean(axis=0)
        dt_spec    = float(t_sec[1] - t_sec[0]) if len(t_sec) > 1 else 1.0
        n_zoom_col = max(1, int(ZOOM_DURATION_MIN * 60 / dt_spec))
        kernel     = np.ones(n_zoom_col) / n_zoom_col
        smoothed   = np.convolve(energy_per_time, kernel, mode='valid')
        best_col   = int(np.argmax(smoothed))
        t_zoom_s   = tr_filt.stats.starttime + float(t_sec[best_col])
    else:
        t_zoom_s = UTCDateTime(ZOOM_START_UTC)

    t_zoom_e = t_zoom_s + ZOOM_DURATION_MIN * 60
    print(f"  Zoom window : {t_zoom_s}  →  {t_zoom_e}")

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
# END
# =============================================================================

print(f"\n[DONE]  All outputs saved to: {OUTPUT_DIR}")
