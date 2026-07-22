"""
FIOS LANDSLIDE — INTERACTIVE SPECTROGRAM PROTOTYPE
====================================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : July 2026

What this is
-------------
Export a SINGLE interactive spectrogram as a .html file, scroll/drag to zoom on the time axis (and frequency
axis), double-click to reset, hover to read exact time/freq/dB values

Output
------
  OUTPUT_DIR/fig_interactive_<date>_<HHMM>.html — open directly in a browser
  (needs an internet connection once, to load plotly.js from a CDN — see
  PLOTLYJS_MODE below if you need a fully offline file instead)
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

DATA_ROOT  = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\data\FIOS"
OUTPUT_DIR = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\fios\08_interactive"

NETWORK  = "XT"
STATION  = "FIO1"
LOCATION = "01"
CHANNEL  = "DHZ"

# ---- What to load -------------------------------------------------------------
TARGET_DATE        = "2026-04-13"   # pick any night you've already looked at in fios_07
WINDOW_START_UTC    = 18             # UTC hour to start from
WINDOW_DURATION_MIN = 60             # kept short on purpose for a first test — see docstring

# ---- Display bandpass -----------------------------------------------------
# Spectrogram band (unchanged) vs waveform band (separate on purpose — see
# WAVE_FMIN/WAVE_FMAX below): the two panels don't have to share one filter.
FILT_FMIN = 1.0
FILT_FMAX = 60.0

# Waveform panel only: narrower than the spectrogram band so the trace isn't
# dominated by broadband/high-freq scatter you can already see in the
# spectrogram. 1-20 Hz keeps most of the 1-10 Hz energy you found earlier
# while cutting the noisiest high end. CAUTION: if you're specifically
# checking the SHAPE of a high-frequency impulsive pick (like the fig_03
# bottom panel in fios_07, filtered on the picker's 10-60 Hz band instead),
# 1-20 Hz will flatten exactly that content — match this band to what you're
# trying to see, not the other way around.
WAVE_FMIN = 1.0
WAVE_FMAX = 20.0

# ---- Welch spectrogram (same method as fios_07 Level 2, pushed a bit finer
# now that the interaction model itself is validated) -----------------------
BIN_S           = 2.0    # length of data feeding ONE output column
SUBSEG_S        = 0.5    # length of each Welch sub-segment inside a bin -> Δf = 1/SUBSEG_S
SUBSEG_OVERLAP  = 0.75   # overlap between sub-segments inside a bin
HOP_S           = 0.2    # how far the bin slides between columns -> Δt of the output
                         # (0.2s over a 60-min window -> ~540k cells, still smooth for
                         #  most machines; if the browser starts lagging when you pan,
                         #  raise this back toward 0.5, or shorten WINDOW_DURATION_MIN)

FREQ_MIN_PLOT = 1.0
FREQ_MAX_PLOT = 60.0

# 'cdn'    -> small file (~few hundred KB - MBs of data only), needs internet to view
# 'inline' -> larger file (+~3-4 MB for plotly.js itself), fully offline-viewable
PLOTLYJS_MODE = 'cdn'



# =============================================================================
# SECTION 2 — IMPORTS
# =============================================================================

import os
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np

from obspy        import UTCDateTime, read, Stream
from scipy.signal  import welch

import plotly.graph_objects as go
from plotly.subplots import make_subplots

os.makedirs(OUTPUT_DIR, exist_ok=True)



# =============================================================================
# SECTION 3 — HELPER FUNCTIONS (duplicated from fios_07, kept standalone here)
# =============================================================================

def load_window(data_root, date_str, network, station, location, channel,
                window_start_utc, window_duration_min):
    """
    Load window_duration_min minutes starting at window_start_utc (UTC) on
    date_str, merging as many hourly MiniSEED files as needed across
    calendar-day boundaries. Returns a cleaned (demean+detrend), UNFILTERED
    obspy.Trace, or None if no data.
    """
    t_start = UTCDateTime(date_str) + window_start_utc * 3600
    t_end   = t_start + window_duration_min * 60

    st  = Stream()
    day = UTCDateTime(t_start.strftime('%Y-%m-%d'))
    while day < t_end:
        month_str = day.strftime('%Y%m')
        date_s    = day.strftime('%Y%m%d')
        pattern   = os.path.join(
            data_root, month_str,
            f"{network}.{station}.{location}.{channel}_{date_s}_*.miniseed"
        )
        for f in sorted(glob.glob(pattern)):
            try:
                st += read(f)
            except Exception as e:
                print(f"    [WARN] Skipping {os.path.basename(f)}: {e}")
        day += 86400

    if len(st) == 0:
        return None
    try:
        st.merge(fill_value=0)
    except Exception:
        st.merge(method=0, fill_value=0)

    st.trim(t_start, t_end)
    if len(st) == 0 or st[0].stats.npts == 0:
        return None

    tr = st[0]
    tr.detrend('demean')
    tr.detrend('linear')
    return tr


def compute_welch_spectrogram(data, fs, bin_s, subseg_s, subseg_overlap, hop_s):
    """
    Same method as fios_07's compute_welch_spectrogram(): a sequence of
    Welch PSD estimates (averaged sub-segments), not a single noisy
    periodogram per column. Returns (t_sec, f_hz, Sxx_db).
    """
    nbin = max(8, int(bin_s * fs))
    nseg = max(8, min(int(subseg_s * fs), nbin))
    noverlap_seg = min(nseg - 1, int(nseg * subseg_overlap))
    nhop = max(1, int(hop_s * fs))

    n = len(data)
    starts = np.arange(0, max(1, n - nbin + 1), nhop)
    if len(starts) == 0:
        starts = np.array([0])

    f_ref = None
    cols  = []
    used_starts = []
    for s0 in starts:
        seg = data[s0:s0 + nbin]
        if len(seg) < nseg:
            continue
        freqs, psd = welch(seg, fs=fs, nperseg=nseg, noverlap=noverlap_seg,
                           window='hann', scaling='density')
        if f_ref is None:
            f_ref = freqs
        cols.append(psd)
        used_starts.append(s0)

    if not cols:
        return np.array([]), np.array([]), np.zeros((0, 0))

    Sxx    = np.array(cols).T
    Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-30))
    t_sec  = (np.array(used_starts) + nbin / 2.0) / fs
    return t_sec, f_ref, Sxx_db



# =============================================================================
# SECTION 4 — LOAD + COMPUTE
# =============================================================================

print(f"\n{'='*70}")
print(f"  FIOS Interactive spectrogram prototype")
print(f"{'='*70}\n")

tr_raw = load_window(DATA_ROOT, TARGET_DATE, NETWORK, STATION, LOCATION, CHANNEL,
                     WINDOW_START_UTC, WINDOW_DURATION_MIN)
if tr_raw is None:
    raise RuntimeError(f"No data found for {TARGET_DATE} starting at UTC {WINDOW_START_UTC}:00")

fs = tr_raw.stats.sampling_rate
print(f"  Loaded : {tr_raw.stats.starttime} -> {tr_raw.stats.endtime}  "
      f"({tr_raw.stats.npts/fs/60:.1f} min  @ {fs:.0f} Hz)")

fmax_safe = min(FILT_FMAX, 0.45 * fs)
tr_filt = tr_raw.copy()
tr_filt.filter('bandpass', freqmin=FILT_FMIN, freqmax=fmax_safe, corners=4, zerophase=True)

wave_fmax_safe = min(WAVE_FMAX, 0.45 * fs)
tr_wave = tr_raw.copy()
tr_wave.filter('bandpass', freqmin=WAVE_FMIN, freqmax=wave_fmax_safe, corners=4, zerophase=True)

print(f"  Computing Welch spectrogram (bin={BIN_S}s, subseg={SUBSEG_S}s, "
      f"overlap={int(SUBSEG_OVERLAP*100)}%, hop={HOP_S}s) ...")
t_sec, f_hz, Sxx_db = compute_welch_spectrogram(
    tr_filt.data, fs, BIN_S, SUBSEG_S, SUBSEG_OVERLAP, HOP_S
)
if Sxx_db.size == 0:
    raise RuntimeError("Spectrogram computation produced no columns — check the window/params.")

freq_mask = (f_hz >= FREQ_MIN_PLOT) & (f_hz <= FREQ_MAX_PLOT)
f_plot = f_hz[freq_mask]
S_plot = Sxx_db[freq_mask, :]
t_abs  = [(tr_filt.stats.starttime + float(t)).datetime for t in t_sec]

n_cells = S_plot.shape[0] * S_plot.shape[1]
print(f"  Grid: {S_plot.shape[1]} time columns x {S_plot.shape[0]} freq rows "
      f"= {n_cells:,} cells  (keep under ~1M for a smooth browser experience)")

# Downsampled waveform for the top panel (same convention as fios_07's fig_01)
# — uses tr_wave (WAVE_FMIN-WAVE_FMAX), NOT tr_filt (the spectrogram's band)
step_ds = max(1, tr_wave.stats.npts // 20000)
t_wave  = [(tr_wave.stats.starttime + i / fs).datetime
          for i in range(0, tr_wave.stats.npts, step_ds)]
d_wave  = tr_wave.data[::step_ds]



# =============================================================================
# SECTION 5 — INTERACTIVE FIGURE
# =============================================================================

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.25, 0.75], vertical_spacing=0.03,
    subplot_titles=("Waveform", "Spectrogram")
)

fig.add_trace(
    go.Scatter(x=t_wave, y=d_wave, mode='lines',
              line=dict(color='black', width=0.6),
              name=f'Waveform ({WAVE_FMIN:.0f}-{wave_fmax_safe:.0f} Hz)',
              hovertemplate='%{x}<br>%{y:.0f} counts<extra></extra>'),
    row=1, col=1
)

fig.add_trace(
    go.Heatmap(
        z=S_plot, x=t_abs, y=f_plot,
        colorscale='Inferno',
        colorbar=dict(title='PSD (dB re counts²/Hz)', len=0.75, y=0.35),
        hovertemplate='Time: %{x}<br>Freq: %{y:.1f} Hz<br>PSD: %{z:.1f} dB<extra></extra>',
    ),
    row=2, col=1
)

fig.update_layout(
    title=(
        f'FIO1 — {TARGET_DATE}  |  {tr_filt.stats.starttime.strftime("%H:%M")} -> '
        f'{tr_filt.stats.endtime.strftime("%H:%M")} UTC  |  '
        f'Spectrogram band {FILT_FMIN}-{fmax_safe:.0f} Hz, waveform band {WAVE_FMIN}-{wave_fmax_safe:.0f} Hz<br>'
        f'<sub>Welch spectrogram: Δf={1/SUBSEG_S:.1f} Hz, Δt={HOP_S*1000:.0f} ms (fixed grid — '
        f'zooming magnifies these pixels, it does not compute new ones)  |  '
        f'scroll/drag to zoom on time or frequency, double-click to reset</sub>'
    ),
    height=750,
    hovermode='x unified',
    xaxis2=dict(rangeslider=dict(visible=True), type='date', title='Time UTC'),
    yaxis=dict(title='Amplitude (counts)'),
    yaxis2=dict(title='Frequency (Hz)', range=[FREQ_MIN_PLOT, FREQ_MAX_PLOT]),
)

fname = f"fig_interactive_{TARGET_DATE}_{WINDOW_START_UTC:02d}00.html"
fig_path = os.path.join(OUTPUT_DIR, fname)
fig.write_html(fig_path, include_plotlyjs=PLOTLYJS_MODE)
print(f"\n  [SAVED] {fig_path}")



# =============================================================================
# END
# =============================================================================

print(f"\n[DONE]  Open the .html file above in a browser and try zooming on the")
print(f"        time axis (drag a box on the spectrogram, or use the range")
print(f"        slider at the bottom). If this feels more useful than the")
print(f"        pre-picked cascade in fios_07, tell me and I'll fold it in.")
