"""
08_report_figures.py
=====================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : August 2026

Goal
----
Produce the explanatory figures needed for the internship report: 
 - what data the training sets are built from
 - where it comes from
 - what the raw signals actually look like
 - what quantitatively separates the classes

Figures produced
-----------------
  fig_event_map_<stamp>.png              : map of catalog events used in the training set (EQ/IQ/RS), after the SNR quality gate
  fig_station_map_<stamp>.png            : map of stations recording those events + the noise windows, colored by how many training rows come from each station
  examples/<abbr>/fig_example_*.png      : ONE waveform+spectrogram figure per selected example (N_EXAMPLES_PER_CLASS per class) — cluster-only (needs SDS/FDSN)
  fig_average_spectrogram_<stamp>.png    : "typical fingerprint" per class, built by averaging the example spectrograms above in linear power before the dB conversion
  fig_feature_distributions_<stamp>.png  : violin plots of 4 physically-interpretable features, one panel each, by class
  fig_snr_quality_by_class_<stamp>.png   : SNR / SNR_full_median distributions by class (ungated), with the 05b/06c quality gate threshold overlaid

Data sources
------------
  04a catalog_windows_<stamp>.csv     : earthquake / rockslide / ice quake detections + features
  04d noise_windows_<stamp>.csv       : noise-class detections + features (same schema)
  04c regional_windows_<stamp>.csv    : optional 5th class (regional earthquakes, 150-1000km,
                                        same schema) — REGIONAL_CSV=None to skip. Included in
                                        every figure EXCEPT Map 1 (fig_event_map), which is
                                        deliberately tight to the massif bounding box and would
                                        never show a regional hypocenter anyway — see Section 3.
  ISTerre SDS archive + FDSN inventory : ONLY for the example gallery + average spectrogram
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# -- Input CSVs (04a / 04d outputs) -------------------------------------------
ORIGINAL_CSV = "/data/failles/louisels/project/results/outputs_04a/all-99-features-recent+3C/catalog_windows_20260708_174019.csv"
NOISE_CSV    = "/data/failles/louisels/project/results/outputs_04d/run_20260803_174514/noise_windows_20260803_174514.csv"

# -- Regional catalog (output of script 04c, optional 5th class) --------------
# Set to a 04c `regional_windows_<stamp>.csv` to add the "regional" class.
# None = skip it entirely (unlike NOISE_CSV above, which is currently required).
REGIONAL_CSV = "/data/failles/louisels/project/results/outputs_04c/run_20260805_135512/regional_windows_20260805_135512.csv"

# -- Output ---------------------------------------------------------------------
OUTPUT_DIR = "/data/failles/louisels/project/results/outputs_08"

# -- Cluster access (example gallery + average spectrogram + station map only) --
SDS_ROOT    = "/data/sig/SDS"
ISTERRE_URL = "http://ist-sc3-geobs.osug.fr:8080"

# -- Study area (Mont Blanc massif) — same bounding box used everywhere else in
#    the pipeline (01/02a/02b/03a/04a/04d/06c) -----------------------------------
LAT_MIN, LAT_MAX = 45.5, 46.0
LON_MIN, LON_MAX = 6.5, 7.2
MAP_EXTENT_PAD   = 0.15   # degrees padding around the bbox for the two maps
MONT_BLANC_LON   = 6.865
MONT_BLANC_LAT   = 45.832
CITIES = [   # (name, lon, lat) — same list used in notebooks/01_catalog_exploration.ipynb
    ("Chamonix",         6.870, 45.924),
    ("Annecy",           6.129, 45.899),
    ("Geneva",           6.143, 46.204),
    ("Martigny",         7.074, 46.102),
    ("Aosta",            7.315, 45.737),
    ("Courmayeur",       6.969, 45.794),
    ("Sallanches",       6.633, 45.933),
    ("Cluses",           6.583, 46.067),
    ("Bonneville",       6.408, 46.081),
    ("Thonon-les-Bains", 6.479, 46.371),
    ("Brig",             7.988, 46.317),
]

# -- Classes ----------------------------------------------------------------------
# TARGET_CLASSES : the 3 LOCAL classes from the catalog (04a) only — this list
#   also drives Map 1 (fig_event_map), which is deliberately tight to the
#   massif bounding box. Regional events' true hypocenters are 150-1000km
#   away, so "regional" is NOT added here — it would be invisible (or force
#   the map to zoom out and lose the point) — see Section 3/4 notes below.
TARGET_CLASSES = ["earthquake", "rockslide", "ice quake"]        # from the catalog (04a)
# CLASS_ORDER : everything else (station map, example gallery, average
#   spectrogram, feature distributions, SNR figure) — these are all about what
#   was recorded on the massif STATIONS, which regional legitimately belongs to.
CLASS_ORDER    = ["earthquake", "rockslide", "ice quake", "noise", "regional"]  # incl. 4th/5th class (04d/04c)
CLASS_ABBR     = {"earthquake": "eq", "rockslide": "rs", "ice quake": "iq", "noise": "no", "regional": "re"}
CLASS_COLORS   = {"earthquake": "#1f77b4", "rockslide": "#d62728",
                  "ice quake": "#2ca02c", "noise": "#7f7f7f",
                  "regional": "#9467bd"}

# -- Quality gate (same values used in 05b Tier 2 / 06c) — applied to the
#    earthquake/rockslide/ice quake catalog only; noise is intentionally NOT
#    gated (see plot_snr_quality_by_class docstring) -----------------------------
SNR_MIN             = 1.70    # metric 'SNR'
SNR_FULL_MEDIAN_MIN = 1.99    # metric 'SNR_full_median'

# -- Example gallery (one waveform+spectrogram figure PER example) --------------
N_EXAMPLES_PER_CLASS = 10
# Ranking: earthquake/rockslide/ice quake -> highest SNR_full_median first,
#          AFTER excluding outliers (see SNR_OUTLIER_MULT below)
#          noise                          -> RANDOM sample (fixed seed), NOT
#          sorted by highest STA/LTA trigger ratio. 06c's QC gallery sorts noise
#          by highest trigger_on_cft on purpose (a diagnostic: "show me the
#          noise windows that look most event-like"), but that means the top
#          few are, by construction, the most anomalous outliers in the whole
#          catalog -- exactly the opposite of what a report figure titled
#          "what does noise look like" should show. A random sample is
#          representative; sorting by CFT is not.
NOISE_SAMPLE_SEED = 42

# A handful of stations produce implausibly large SNR_full_median values --
# e.g. XX.B03 reaches >1900 (even >4000 on individual rows) while its OWN
# median is ~3.7, completely ordinary. These are not "exceptionally clean
# earthquakes", they're a recurring data-quality artifact (same station,
# same spike shape/size/position, regardless of which real event) that a
# naive "top-N by SNR" selection cherry-picks every single time. Reject any
# candidate whose SNR_full_median exceeds SNR_OUTLIER_MULT times its class's
# SNR_OUTLIER_PCTL percentile before ranking -- excludes ~0.1% of rows,
# removes every one of the B03-type artifacts (verified against the actual
# 04a CSV: cap=220 for earthquake keeps 42,553/42,615 rows and the resulting
# top-10 becomes a healthy mix of real stations, no repeats).
SNR_OUTLIER_PCTL = 0.99
SNR_OUTLIER_MULT = 5.0

# The class-wide percentile cap above can still miss a B03-type artifact when
# the class has few detections (e.g. rockslide, n=7,656 vs earthquake's
# 42,615): the artifact rows are then a big enough share of the top tail that
# they drag the 99th-percentile value itself upward, which raises the cap
# (5x that already-inflated percentile) enough for some of them to survive.
# XX.B03 and XX.B01 have now BOTH shown the exact same narrow double-spike
# signature -- same shape, same implausible SNR relative to a normal
# station's median -- across THREE different classes (rockslide, regional,
# noise). Two different stations in the same network sharing an identical,
# distinctive artifact is not a coincidence of two unrelated broken sensors;
# it's consistent with a shared hardware/telemetry issue across the whole XX
# deployment (XX-style codes are typically a temporary/nodal experiment
# network, which commonly show periodic GPS-clock-resync spikes). Rather than
# keep whitelisting individual XX station codes one at a time as new ones
# turn up, exclude the entire network -- use ("<network>", "*") to blacklist
# every station in a network.
# GU.BLANC (regional class) shows a related but distinct signature: 2-3
# near-identical spikes recurring at a fixed ~43s interval within the SAME
# 90s window -- a real earthquake doesn't repeat itself perfectly on a timer,
# this is consistent with a periodic instrumental artifact (calibration
# pulse / timing correction / digitizer glitch). Note GU.REMY does NOT show
# this pattern (its examples look like genuine noisy regional arrivals), so
# only GU.BLANC specifically is excluded, not the whole GU network.
# Confirmed the artifact is NOT a filtering issue: it survived both the
# 2-10Hz and a wider 1-20Hz bandpass, meaning it's broadband (impulsive/
# spike-like) energy present in the RAW trace itself, not something our
# processing introduces or a filter can remove.
# Add other (network, station) pairs -- or ("network", "*") -- here if the
# same pattern shows up again.
EXCLUDED_STATIONS = {("XX", "*"), ("GU", "BLANC")}


def _is_station_excluded(net, sta):
    """True if (net, sta) or the whole network ("net", "*") is blacklisted."""
    return (net, sta) in EXCLUDED_STATIONS or (net, "*") in EXCLUDED_STATIONS

# Fixed-length window fetched for every example (also what gets averaged into the
# per-class "typical fingerprint" spectrogram, so the shape MUST be identical for
# every example — hence fixed window length + fixed target sampling rate, no
# per-event trimming to det_duration_s)
PRE_S          = 10     # seconds BEFORE det_starttime
FIXED_WINDOW_S = 100    # total window length [s] (so 90s of post-onset context)
TARGET_FS      = 200    # [Hz] every trace is resampled to this before spectrogram
                        # (raised from 100 -> many stations here are DHZ/HGZ/HHZ
                        # channels sampled well above 100Hz; forcing everything
                        # down to 100Hz Nyquist=50Hz was silently discarding real
                        # high-frequency content for those stations -- see FREQ_MAX_KEEP)

# Extra context fetched (and processed) on EACH side of the window above, then
# discarded before plotting. NOT optional: removing the instrument response
# (water-level deconvolution) or bandpass-filtering right at the edge of a
# tightly-trimmed window creates a large low-frequency transient that looks
# exactly like a real spike/step in the trace and shows up as a solid
# broadband streak in the spectrogram -- it is a processing artifact, not
# signal. Fetching this padding and trimming it off AFTER filtering gives the
# transient room to decay before it reaches the region actually plotted.
FETCH_PAD_S = 60

# Waveform display filter (matches the reference figure the report is modeled on)
WAVE_FREQMIN = 1.0
WAVE_FREQMAX = 20.0

# Spectrogram parameters (same convention as 07a_spectrogram_dataset_build.py,
# but with a higher frequency ceiling -- see TARGET_FS note above)
SPEC_NPERSEG_S     = 2.0     # [s] STFT segment length
SPEC_NOVERLAP_FRAC = 0.75
SPEC_NFFT          = 512
FREQ_MAX_KEEP      = 95.0    # [Hz] 95% of Nyquist at TARGET_FS=200Hz -- stations
                             # whose native rate is lower than this will simply
                             # show (correctly) quiet/blue above their own Nyquist,
                             # not an artifact, just an honest reflection of that
                             # instrument's real bandwidth
SPEC_NPERSEG       = int(SPEC_NPERSEG_S * TARGET_FS)
SPEC_NOVERLAP      = int(SPEC_NPERSEG * SPEC_NOVERLAP_FRAC)
# PSD floor epsilon — see 07a's amplitude-check note: background PSD is ~1e-18
# (m/s)^2/Hz, strongest events ~1e-13; 1e-20 guards log(0) without swallowing signal
PSD_FLOOR_EPS = 1e-20
SPEC_VMIN, SPEC_VMAX = -200, -120   # dB color scale, matches the reference figure

# -- Feature distributions (physically-interpretable subset) --------------------
FEATURES_TO_PLOT = ["duration", "fft_freq_at_max", "kurtosis_signal", "eratio_1_3__3_10"]
FEATURE_LABELS = {
    "duration":          "Duration (s)",
    "fft_freq_at_max":   "Dominant frequency (Hz)",
    "kurtosis_signal":   "Kurtosis (waveform impulsiveness)",
    "eratio_1_3__3_10":  "Energy ratio 1\u20133 Hz / 3\u201310 Hz",
}
LOG_FEATURES = {"eratio_1_3__3_10", "kurtosis_signal"}   # both are heavy-tailed;
# a linear axis lets a handful of extreme outliers squash the violin body flat

# -- SNR / quality distribution panel --------------------------------------------
SNR_METRICS = ("SNR", "SNR_full_median")



# =============================================================================
# SECTION 2 — SETUP
# =============================================================================

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# SSL workaround (same fix applied to notebooks/01_catalog_exploration.ipynb):
# recent OpenSSL builds raise ssl.SSLError: [ASN1: NOT_ENOUGH_DATA] when Python
# falls back to loading the OS certificate store. Forcing certifi's CA bundle
# sidesteps it. Must run before the first HTTPS request (FDSN inventory fetch,
# and the contextily basemap tile downloads used by the two maps below).
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import spectrogram
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from features import FEATURE_NAMES, rename_legacy_columns
from run_setup import (
    create_run_dir, setup_logging, connect_sds, connect_fdsn,
    fetch_inventory, set_matplotlib_defaults,
)
from preprocessing import build_station_times_df, remove_response_or_fallback
from visualization import (
    plot_event_map, plot_station_map, plot_waveform_spectrogram_example,
    plot_average_spectrograms, plot_feature_distributions, plot_snr_quality_by_class,
)

RUN_DIR, STAMP = create_run_dir(OUTPUT_DIR)
log_file, log_path = setup_logging(
    RUN_DIR, "08_report_figures.py",
    extra_info=f"ORIGINAL_CSV: {ORIGINAL_CSV}\nNOISE_CSV: {NOISE_CSV}\nREGIONAL_CSV: {REGIONAL_CSV}",
)
set_matplotlib_defaults()

# Environment fingerprint -- if this ever runs under the wrong interpreter
# (e.g. a cluster's shared /soft/python module instead of the glacier-seismo
# conda env), package ABI mismatches (numpy/rasterio in particular, via
# contextily) fail in ways that are hard to diagnose from the traceback alone.
# This makes a mismatch immediately visible in the run log.
print(f"  Python executable : {sys.executable}")
print(f"  NumPy version     : {np.__version__}")



# =============================================================================
# SECTION 3 — LOAD DATA
# =============================================================================

print(f"\n{'='*65}\n  STEP 1 — Loading catalogs\n{'='*65}")

orig_all = pd.read_csv(ORIGINAL_CSV, low_memory=False)
orig_all = rename_legacy_columns(orig_all)
orig_all = orig_all[orig_all["event_type"].isin(TARGET_CLASSES)].copy()
print(f"  Original catalog (all quality) : {len(orig_all):,} rows")

# Quality gate — applied ONLY to earthquake/rockslide/ice quake, recomputed
# explicitly from SNR/SNR_full_median (do not trust a stale 'quality_ok' column)
quality_mask = (
    (orig_all["SNR"]             >= SNR_MIN) &
    (orig_all["SNR_full_median"] >= SNR_FULL_MEDIAN_MIN)
)
z_feat_cols = [f for f in FEATURE_NAMES if f in orig_all.columns]
orig_gated  = orig_all[quality_mask].dropna(subset=z_feat_cols).copy()
print(f"  After quality gate + NaN drop  : {len(orig_gated):,} rows")
for cls in TARGET_CLASSES:
    n = (orig_gated["event_type"] == cls).sum()
    print(f"    {cls:<22} {n:>6,}")

noise = pd.read_csv(NOISE_CSV, low_memory=False)
noise = rename_legacy_columns(noise)
noise = noise[noise["event_type"] == "noise"].copy()
z_feat_cols_noise = [f for f in FEATURE_NAMES if f in noise.columns]
noise = noise.dropna(subset=z_feat_cols_noise).copy()
print(f"  Noise catalog                  : {len(noise):,} rows")

# -- Optional 5th class: regional (04c output) ---------------------------------
# Kept as SEPARATE variables from orig_all/orig_gated (which stay local-only
# and continue to feed Map 1's tight massif-only geography, via TARGET_CLASSES
# — see Section 1 note). Regional's true hypocenters are 150-1000km away, well
# outside that map's extent, so mixing it into orig_all would either make it
# invisible or force the local map to zoom out and lose its whole point.
# It DOES belong in combined_gated/combined_all below though — its WAVEFORMS
# were recorded on the same massif stations as everything else, which is what
# the station map / example gallery / feature and SNR distributions are about.
if REGIONAL_CSV is not None and os.path.exists(str(REGIONAL_CSV)):
    regional_all = pd.read_csv(REGIONAL_CSV, low_memory=False)
    regional_all = rename_legacy_columns(regional_all)
    regional_all = regional_all[regional_all["event_type"] == "regional"].copy()
    print(f"  Regional catalog (all quality) : {len(regional_all):,} rows")

    # Same gate as the local classes — regional rows carry REAL computed SNR
    # from 04c's own detection pipeline, unlike noise (SNR=NaN by construction).
    regional_mask = (
        (regional_all["SNR"]             >= SNR_MIN) &
        (regional_all["SNR_full_median"] >= SNR_FULL_MEDIAN_MIN)
    )
    z_feat_cols_r  = [f for f in FEATURE_NAMES if f in regional_all.columns]
    regional_gated = regional_all[regional_mask].dropna(subset=z_feat_cols_r).copy()
    print(f"  After quality gate + NaN drop  : {len(regional_gated):,} rows")
else:
    if REGIONAL_CSV is not None:
        print(f"  [WARN] REGIONAL_CSV not found: {REGIONAL_CSV} — continuing without the regional class.")
    regional_all, regional_gated = pd.DataFrame(), pd.DataFrame()

# `combined_gated`  -> what the classifier actually trains on (used for the
#                      station map and the example gallery)
# `combined_all`    -> ungated union, used ONLY for the SNR/quality figure so the
#                      gate's effect is actually visible (see docstring)
combined_gated = pd.concat([orig_gated, noise, regional_gated], ignore_index=True)
combined_all   = pd.concat([orig_all,   noise, regional_all],   ignore_index=True)



# =============================================================================
# SECTION 4 — MAP 1: labelled catalog events used in the training set
# =============================================================================

print(f"\n{'='*65}\n  STEP 2 — Event map (earthquake / rockslide / ice quake)\n{'='*65}")

events_unique = orig_gated.drop_duplicates(subset=["event_time"]).copy()
print(f"  {len(events_unique):,} unique events "
      f"(from {len(orig_gated):,} station-detections, after quality gate)")
for cls in TARGET_CLASSES:
    n = (events_unique["event_type"] == cls).sum()
    print(f"    {cls:<22} {n:>6,}")

map_extent = (LON_MIN - MAP_EXTENT_PAD, LON_MAX + MAP_EXTENT_PAD,
             LAT_MIN - MAP_EXTENT_PAD, LAT_MAX + MAP_EXTENT_PAD)
event_colors = {c: CLASS_COLORS[c] for c in TARGET_CLASSES}

fig1, ax1 = plt.subplots(figsize=(8, 7))
n_plotted = plot_event_map(
    ax1,
    lats            = events_unique["catalog_lat"],
    lons            = events_unique["catalog_lon"],
    event_types     = events_unique["event_type"],
    class_colors    = event_colors,
    map_extent      = map_extent,
    mont_blanc_lon  = MONT_BLANC_LON,
    mont_blanc_lat  = MONT_BLANC_LAT,
    cities          = CITIES,
    title           = f"Catalog events used in the training set  (n={len(events_unique)}, "
                      f"quality-gated: SNR\u2265{SNR_MIN}, SNR_full_median\u2265{SNR_FULL_MEDIAN_MIN})",
)
fig1.tight_layout()
path1 = os.path.join(RUN_DIR, f"fig_event_map_{STAMP}.png")
fig1.savefig(path1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"  {n_plotted} events plotted")
print(f"  [SAVED] {path1}")



# =============================================================================
# SECTION 5 — MAP 2: stations recording the training-set signals
# =============================================================================

print(f"\n{'='*65}\n  STEP 3 — Station map (all 4 classes)\n{'='*65}")

client_sds  = connect_sds(SDS_ROOT)
client_fdsn = connect_fdsn(ISTERRE_URL)

inventory  = None
sta_coords = {}   # {(network, station): (lat, lon)}
if client_fdsn is not None:
    # IMPORTANT: always pass the bounding box here (not a bare network="*"/
    # station="*" query) -- see 06c's note: an unscoped query on ISTerre's FDSN
    # server hits a WADL validation error unrelated to our request. Scoping to
    # the massif bbox is also just correct (the only region we ever want) and
    # faster besides.
    _t_min = pd.to_datetime(combined_gated["det_starttime"]).min()
    _t_max = pd.to_datetime(combined_gated["det_starttime"]).max()
    inventory = fetch_inventory(
        client_fdsn, str(_t_min.date()), str((_t_max + pd.Timedelta(days=1)).date()),
        lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX,
    )
    if inventory is not None:
        for net in inventory:
            for sta in net:
                sta_coords[(net.code, sta.code)] = (sta.latitude, sta.longitude)
        print(f"  Station coordinates fetched: {len(sta_coords)} stations")
    else:
        print("  [WARN] Inventory fetch failed -- station map will be skipped.")
else:
    print("  [WARN] FDSN unavailable -- station map will be skipped "
          "(this section only works on the cluster / with VPN access).")

if sta_coords:
    counts_by_code = {}
    for (net, sta), grp in combined_gated.groupby(["network", "station"]):
        counts_by_code[sta] = counts_by_code.get(sta, 0) + len(grp)
    # plot_station_map keys on station code only (see script 01/05a) — collapse
    # sta_coords the same way if two networks happen to share a station code
    sta_coords_by_code = {}
    for (net, sta), coord in sta_coords.items():
        sta_coords_by_code.setdefault(sta, coord)
    counts_series = pd.Series(counts_by_code)

    fig2, ax2 = plt.subplots(figsize=(8, 7))
    n_ok = plot_station_map(
        ax2, counts_series, sta_coords_by_code,
        title      = f"Stations recording the training set  ({len(counts_series)} stations, "
                     f"{len(combined_gated):,} detections across {len(CLASS_ORDER)} classes)",
        vmin       = 0,
        vmax       = float(counts_series.max()),
        map_extent = map_extent,
        mont_blanc_lon = MONT_BLANC_LON,
        mont_blanc_lat = MONT_BLANC_LAT,
        cmap       = "YlOrRd",
    )
    sm = plt.cm.ScalarMappable(cmap="YlOrRd",
                               norm=plt.Normalize(vmin=0, vmax=counts_series.max()))
    sm.set_array([])
    fig2.colorbar(sm, ax=ax2, label="Number of training detections recorded", shrink=0.8)
    fig2.tight_layout()
    path2 = os.path.join(RUN_DIR, f"fig_station_map_{STAMP}.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  {n_ok} stations plotted")
    print(f"  [SAVED] {path2}")



# =============================================================================
# SECTION 6 — EXAMPLE GALLERY: one waveform+spectrogram figure per example
# =============================================================================

print(f"\n{'='*65}\n  STEP 4 — Example waveform+spectrogram gallery\n{'='*65}")


def _fetch_padded_trace(client_sds, net, sta, chan, det_start, inventory,
                        pre_s, window_s, target_fs, fetch_pad_s):
    """
    Fetch + response-remove + resample a window LARGER than needed (with
    fetch_pad_s of extra context on each side of [det_start-pre_s, ...+window_s]).

    Returns the UN-TRIMMED trace, still including the padding. The caller is
    responsible for any further filtering and for trimming down to the final
    [t_on, t_off] window (via _trim_to_fixed_length) AFTER that filtering --
    both response removal and any bandpass need this padding as "run-in" room,
    or their edge transients (which look exactly like a real spike/step) end
    up sitting right inside the plotted window.

    Returns
    -------
    (trace, t_on, t_off, None) on success, (None, None, None, reason_str) on failure.
    """
    t_on   = det_start - pre_s
    t_off  = t_on + window_s
    t_on_p = t_on - fetch_pad_s
    t_off_p = t_off + fetch_pad_s
    try:
        st_raw = client_sds.get_waveforms(net, sta, "*", chan, t_on_p, t_off_p)
        if len(st_raw) == 0:
            return None, None, None, "no waveform in SDS"
        st_raw.merge(method=1, fill_value="interpolate")

        sdf    = build_station_times_df(st_raw, t_on_p, t_off_p)
        st_vel = remove_response_or_fallback(st_raw, inventory, sdf)
        if len(st_vel) == 0:
            return None, None, None, "response removal failed"

        tr = st_vel[0].copy()
        if abs(tr.stats.sampling_rate - target_fs) > 0.5:
            tr.resample(target_fs)

        if not np.all(np.isfinite(tr.data)) or np.max(np.abs(tr.data)) == 0:
            return None, None, None, "degenerate trace (NaN/Inf/all-zero)"
        return tr, t_on, t_off, None
    except Exception as e:
        return None, None, None, str(e)


def _trim_to_fixed_length(tr, t_on, t_off, target_fs, window_s):
    """ Trim a COPY of tr to exactly [t_on, t_off] -> exactly window_s*target_fs samples """
    tr = tr.copy()
    tr.trim(t_on, t_off, pad=True, fill_value=0)
    nt = int(round(window_s * target_fs))
    if len(tr.data) < nt:
        tr.data = np.pad(tr.data, (0, nt - len(tr.data)))
    elif len(tr.data) > nt:
        tr.data = tr.data[:nt]
    return tr


# Running sums for the average-spectrogram figure (section 7) — accumulated in
# LINEAR power while we're already fetching each example, so no second fetch pass
_avg_sum_linear = {cls: None for cls in CLASS_ORDER}
_avg_count       = {cls: 0    for cls in CLASS_ORDER}
_freq_axis_shared = None
_time_axis_shared = None

if client_sds is None or client_fdsn is None or not sta_coords:
    print("  [WARN] SDS/FDSN unavailable -- skipping example gallery entirely "
          "(cluster-only section).")
else:
    for cls in CLASS_ORDER:
        sub = combined_gated[combined_gated["event_type"] == cls].copy()

        # Hard-exclude known-broken stations/networks first (see
        # EXCLUDED_STATIONS note in section 1) -- applies to EVERY class,
        # including noise: the noise gallery used to skip this check entirely
        # and picked up the exact same XX artifact as a "typical noise"
        # example.
        n_before_excl = len(sub)
        keep_mask = [not _is_station_excluded(net, sta)
                     for net, sta in zip(sub["network"], sub["station"])]
        sub = sub[keep_mask]
        n_excl_station = n_before_excl - len(sub)
        if n_excl_station:
            print(f"  [{cls}] excluded {n_excl_station} detection(s) from "
                  f"hard-excluded station(s)/network(s) {sorted(EXCLUDED_STATIONS)} "
                  f"before ranking")

        if cls == "noise":
            # Random, NOT sorted by trigger_on_cft -- see NOISE_SAMPLE_SEED note
            # in section 1: sorting by highest CFT surfaces the most anomalous
            # outlier windows in the whole catalog, not typical noise.
            sub = sub.sample(frac=1.0, random_state=NOISE_SAMPLE_SEED)
        else:
            # Exclude implausible-outlier detections (see SNR_OUTLIER_MULT note
            # in section 1) BEFORE ranking, so a single glitchy station can't
            # monopolize the gallery.
            p99 = sub["SNR_full_median"].quantile(SNR_OUTLIER_PCTL)
            cap = p99 * SNR_OUTLIER_MULT
            n_before = len(sub)
            sub = sub[sub["SNR_full_median"] <= cap]
            n_excluded = n_before - len(sub)
            if n_excluded:
                print(f"  [{cls}] excluded {n_excluded} outlier detection(s) with "
                      f"SNR_full_median > {cap:.1f} ({SNR_OUTLIER_MULT:g}x the "
                      f"{SNR_OUTLIER_PCTL*100:g}th pct) before ranking -- almost "
                      f"certainly instrumental artifacts, not genuinely clean events")
            sub = sub.sort_values("SNR_full_median", ascending=False)

        out_dir_cls = os.path.join(RUN_DIR, "examples", CLASS_ABBR[cls])
        os.makedirs(out_dir_cls, exist_ok=True)

        n_done = 0
        for _, row in sub.iterrows():
            if n_done >= N_EXAMPLES_PER_CLASS:
                break
            net, sta, chan = row["network"], row["station"], row["channel"]

            tr_padded, t_on, t_off, err = _fetch_padded_trace(
                client_sds, net, sta, chan, UTCDateTime(row["det_starttime"]),
                inventory, PRE_S, FIXED_WINDOW_S, TARGET_FS, FETCH_PAD_S,
            )
            if tr_padded is None:
                continue

            # -- broadband copy for the spectrogram: trim only, no filtering --
            tr_broadband = _trim_to_fixed_length(tr_padded, t_on, t_off, TARGET_FS, FIXED_WINDOW_S)
            if not (np.all(np.isfinite(tr_broadband.data)) and np.max(np.abs(tr_broadband.data)) > 0):
                continue

            # -- waveform panel: bandpass the PADDED trace first, trim after --
            # (filtering before trimming keeps the bandpass's own edge transient
            # out of the padding region too, not just deconvolution's)
            tr_wave_padded = tr_padded.copy()
            nyq = tr_wave_padded.stats.sampling_rate / 2.0
            tr_wave_padded.filter("bandpass", freqmin=WAVE_FREQMIN,
                                 freqmax=min(WAVE_FREQMAX, 0.9 * nyq),
                                 corners=4, zerophase=True)
            tr_wave = _trim_to_fixed_length(tr_wave_padded, t_on, t_off, TARGET_FS, FIXED_WINDOW_S)

            # -- spectrogram: broadband (unfiltered) -------------------------
            f_full, t_full, Sxx = spectrogram(
                tr_broadband.data, fs=tr_broadband.stats.sampling_rate, window="hann",
                nperseg=SPEC_NPERSEG, noverlap=SPEC_NOVERLAP, nfft=SPEC_NFFT,
                scaling="density", mode="psd",
            )
            freq_mask = f_full <= FREQ_MAX_KEEP
            freq_axis = f_full[freq_mask]
            Sxx_lin   = Sxx[freq_mask, :]                       # kept linear, for averaging
            Sxx_db    = 10 * np.log10(Sxx_lin + PSD_FLOOR_EPS)

            if _freq_axis_shared is None:
                _freq_axis_shared = freq_axis
                _time_axis_shared = t_full - PRE_S

            # -- accumulate for the average-spectrogram figure ----------------
            if _avg_sum_linear[cls] is None:
                _avg_sum_linear[cls] = Sxx_lin.copy()
            else:
                _avg_sum_linear[cls] += Sxx_lin
            _avg_count[cls] += 1

            # -- distance from source ------------------------------------------
            sta_key = (net, sta)
            if sta_key in sta_coords:
                lat_s, lon_s = sta_coords[sta_key]
                dist_km = gps2dist_azimuth(row["catalog_lat"], row["catalog_lon"],
                                           lat_s, lon_s)[0] / 1000.0
                dist_label = f"{dist_km:.0f} km from source"
            else:
                dist_label = "distance unknown"

            if cls == "noise" and "trigger_on_cft" in row and pd.notna(row["trigger_on_cft"]):
                quality_str = f"CFT={row['trigger_on_cft']:.2f}"
            else:
                snr_val = row.get("SNR", np.nan)
                quality_str = f"SNR={snr_val:.2f}" if pd.notna(snr_val) else "SNR=n/a"

            title_l1 = f"{cls} \u2014 {str(row['event_time'])[:19]}"
            title_l2 = f"{net}.{sta} | {dist_label} | {quality_str}"

            out_path = os.path.join(
                out_dir_cls,
                f"fig_example_{CLASS_ABBR[cls]}_{n_done+1:02d}_{net}_{sta}_{STAMP}.png",
            )
            plot_waveform_spectrogram_example(
                times_wave     = tr_wave.times() - PRE_S,
                wave_data      = tr_wave.data,
                times_spec     = t_full - PRE_S,
                freq_axis      = freq_axis,
                spec_db        = Sxx_db,
                det_duration_s = row.get("det_duration_s", 0.0),
                title_lines    = (title_l1, title_l2),
                out_path       = out_path,
                spec_vmin      = SPEC_VMIN,
                spec_vmax      = SPEC_VMAX,
            )
            n_done += 1

        if n_done < N_EXAMPLES_PER_CLASS:
            print(f"  [WARN] '{cls}': only {n_done}/{N_EXAMPLES_PER_CLASS} examples "
                  f"plottable (SDS fetch/response-removal failures skipped)")
        else:
            print(f"  [OK] '{cls}': {n_done}/{N_EXAMPLES_PER_CLASS} examples saved to "
                  f"{out_dir_cls}")



# =============================================================================
# SECTION 7 — AVERAGE ("TYPICAL") SPECTROGRAM PER CLASS
# =============================================================================

print(f"\n{'='*65}\n  STEP 5 — Average spectrogram per class\n{'='*65}")

if _freq_axis_shared is None:
    print("  [SKIP] No examples were fetched in section 6 -- nothing to average.")
else:
    class_avg_db = {}
    for cls in CLASS_ORDER:
        if _avg_count[cls] == 0:
            continue
        mean_linear = _avg_sum_linear[cls] / _avg_count[cls]
        class_avg_db[cls] = 10 * np.log10(mean_linear + PSD_FLOOR_EPS)
        print(f"  {cls:<12} averaged over {_avg_count[cls]} example(s)")

    if class_avg_db:
        path_avg = plot_average_spectrograms(
            class_avg_db, _freq_axis_shared, _time_axis_shared, CLASS_ORDER,
            RUN_DIR, STAMP, vmin=SPEC_VMIN, vmax=SPEC_VMAX,
        )



# =============================================================================
# SECTION 8 — FEATURE DISTRIBUTIONS BY CLASS
# =============================================================================

print(f"\n{'='*65}\n  STEP 6 — Feature distributions by class\n{'='*65}")

path_feat = plot_feature_distributions(
    combined_gated, FEATURES_TO_PLOT, FEATURE_LABELS,
    class_col="event_type", class_order=CLASS_ORDER, class_colors=CLASS_COLORS,
    run_dir=RUN_DIR, stamp=STAMP, log_features=LOG_FEATURES,
)



# =============================================================================
# SECTION 9 — SNR / DATA-QUALITY DISTRIBUTION BY CLASS
# =============================================================================

print(f"\n{'='*65}\n  STEP 7 — SNR / quality distribution by class (ungated)\n{'='*65}")

path_snr = plot_snr_quality_by_class(
    combined_all, class_col="event_type", class_order=CLASS_ORDER, class_colors=CLASS_COLORS,
    run_dir=RUN_DIR, stamp=STAMP,
    snr_min=SNR_MIN, snr_full_median_min=SNR_FULL_MEDIAN_MIN, metrics=SNR_METRICS,
)



# =============================================================================
# END
# =============================================================================

from datetime import datetime
print("\n" + "=" * 70)
print(f"  Run finished : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  All outputs  : {RUN_DIR}")
print(f"  Log file     : {log_path}")
print("=" * 70)

log_file.close()
