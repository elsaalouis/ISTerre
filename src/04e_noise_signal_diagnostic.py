"""
04e_noise_signal_diagnostic.py
================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : August 2026

Goal
----
QC / statistics tool for the "noise" class extracted by 04d. Answers 2 questions:
  1. What are we actually looking at? 
     — how many rows per station, what duration distribution (compared against real events), how strongly each candidate triggered (trigger_on_cft), 
     how the population is spread across the 2015-2026 archive
  2. How did the detector actually fire? 
     — for a sample of individual noise windows, re-fetch the surrounding waveform from SDS, recompute the SAME classical STA/LTA (to 04d) over a wider
     context window, and plot the waveform + characteristic function together

Pipeline
--------
  1. Load noise_windows_<stamp>.csv, print descriptive stats 
  2. Optionally compare the noise duration distribution against a real catalog_windows_<stamp>.csv (EVENT_CATALOG_CSV) by class
  3. Select N_EXAMPLES rows (random by default) and for each:
     - re-fetch [det_starttime - CONTEXT_SEC, det_endtime + CONTEXT_SEC]
     - remove instrument response (broadband), compute a spectrogram from it
     - bandpass filter a copy, rerun classical STA/LTA with 04d's exact PRIMARY_* parameters
     - plot waveform + broadband spectrogram + STA/LTA CFT together

Data sources
------------
  Waveforms : ISTerre SDS archive  /data/sig/SDS
  Inventory : ISTerre FDSN server  http://ist-sc3-geobs.osug.fr:8080

Output
------
  noise_summary_<stamp>.csv     : per-station row counts + duration/CFT stats
  fig_duration_hist_<stamp>.png : noise duration distribution (+ event classes if given)
  fig_cft_hist_<stamp>.png      : trigger_on_cft distribution
  fig_station_counts_<stamp>.png: rows per station
  fig_temporal_coverage_<stamp>.png : rows per year
  noise_diagnostic_<tag>_<stamp>.png : one per example (waveform + STA/LTA CFT)
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# -- Input: 04d output ----------------------------------------------------
NOISE_CSV = "/data/failles/louisels/project/results/outputs_04d/run_20260731_141524/noise_windows_20260731_141524.csv"

# -- Optional: 04a output, for a duration comparison against real events ---
EVENT_CATALOG_CSV = None

# -- Paths ------------------------------------------------------------------
SDS_ROOT    = "/data/sig/SDS"
ISTERRE_URL = "http://ist-sc3-geobs.osug.fr:8080"
OUTPUT_DIR  = "/data/failles/louisels/project/results/outputs_04e"

# -- Bounding box, same as everywhere else in the pipeline ------------------
LAT_MIN, LAT_MAX = 45.5, 46.0
LON_MIN, LON_MAX = 6.5, 7.2

# -- Primary detector -------------------------------------------------------
# MUST match 04d's settings, since we're recomputing the same classical STA/LTA purely for visualization
PRIMARY_STA_S    = 5
PRIMARY_LTA_S    = 100
PRIMARY_THR_ON   = 2.0
PRIMARY_THR_OFF  = 1.3
PRIMARY_FREQ_MIN = 1.0
PRIMARY_FREQ_MAX = 20.0

# -- Example-plot spectrogram (broadband, unfiltered) ------------------------
# Same convention as 08_report_figures.py's example gallery, so noise-class
# examples look directly comparable to the EQ/RS/IQ report figures.
SPEC_NPERSEG_S     = 2.0     # [s] STFT segment length
SPEC_NOVERLAP_FRAC = 0.75
SPEC_NFFT          = 512
PSD_FLOOR_EPS      = 1e-20   # guards log(0) without swallowing real signal
SPEC_VMIN, SPEC_VMAX = -200, -120   # dB color scale

# -- Example diagnostic plots -------------------------------------------------
N_EXAMPLES        = 12
EXAMPLE_SELECTION = "random"   # "random" | "stratified" (spreads picks across stations) | "top_cft" (strongest triggers)
CONTEXT_SEC        = 120       # seconds of context before/after the detected window (>= LTA+STA so a real warm-up is visible)
RANDOM_SEED         = 42



# =============================================================================
# SECTION 2 — SETUP
# =============================================================================

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from scipy.signal import spectrogram

from preprocessing import preprocess_day
from run_setup import (
    create_run_dir, setup_logging, set_matplotlib_defaults,
    connect_sds, connect_fdsn, fetch_inventory,
)
from detection import run_sta_lta
from visualization import plot_noise_diagnostic

RUN_DIR, STAMP = create_run_dir(OUTPUT_DIR)
log_file, log_path = setup_logging(
    RUN_DIR, "04e_noise_signal_diagnostic.py",
    extra_info=f"NOISE_CSV: {NOISE_CSV}"
)
set_matplotlib_defaults()

if not os.path.isfile(NOISE_CSV):
    print(f"[ERROR] NOISE_CSV not found: {NOISE_CSV}")
    print("        Set NOISE_CSV in Section 1 to a 04d noise_windows_<stamp>.csv and rerun.")
    sys.exit(1)



# =============================================================================
# SECTION 3 — LOAD + DESCRIPTIVE STATISTICS
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 1 — Loading noise catalog + descriptive statistics")
print(f"{'='*65}")

df = pd.read_csv(NOISE_CSV, low_memory=False)
print(f"Loaded {len(df):,} noise rows x {df.shape[1]} columns.")

df["_t_on"] = pd.to_datetime(df["det_starttime"])

print(f"\n  Time span covered : {df['_t_on'].min()}  ->  {df['_t_on'].max()}")
print(f"  Unique stations   : {df['station'].nunique()}")

print(f"\n  {'Station':<12} {'rows':>8}  {'median dur (s)':>15}  {'median CFT':>11}")
print(f"  {'-'*52}")
station_stats = (
    df.groupby(["network", "station"])
    .agg(rows=("event_time", "count"),
         med_dur=("det_duration_s", "median"),
         med_cft=("trigger_on_cft", "median"))
    .sort_values("rows", ascending=False)
)
for (net, sta), row in station_stats.iterrows():
    print(f"  {net}.{sta:<9} {int(row['rows']):>8}  {row['med_dur']:>15.1f}  {row['med_cft']:>11.2f}")

print(f"\n  Duration (s)   : min={df['det_duration_s'].min():.1f}  "
      f"median={df['det_duration_s'].median():.1f}  "
      f"mean={df['det_duration_s'].mean():.1f}  "
      f"max={df['det_duration_s'].max():.1f}")
print(f"  trigger_on_cft : min={df['trigger_on_cft'].min():.2f}  "
      f"median={df['trigger_on_cft'].median():.2f}  "
      f"mean={df['trigger_on_cft'].mean():.2f}  "
      f"max={df['trigger_on_cft'].max():.2f}  "
      f"(PRIMARY_THR_ON={PRIMARY_THR_ON})")

summary_path = os.path.join(RUN_DIR, f"noise_summary_{STAMP}.csv")
station_stats.to_csv(summary_path)
print(f"\n[SAVED] {summary_path}")



# =============================================================================
# SECTION 4 — SUMMARY FIGURES
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 2 — Summary figures")
print(f"{'='*65}")

# ── Fig A: duration distribution (noise, + real event classes if given) ─────
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["det_duration_s"], bins=50, color="#7f7f7f", alpha=0.7,
        edgecolor="white", label=f"noise (n={len(df):,})", density=True)

if EVENT_CATALOG_CSV and os.path.isfile(EVENT_CATALOG_CSV):
    df_ev = pd.read_csv(EVENT_CATALOG_CSV, low_memory=False)
    colors = {"earthquake": "#1f77b4", "rockslide": "#d62728", "ice quake": "#2ca02c"}
    for etype, color in colors.items():
        sub = df_ev.loc[df_ev["event_type"] == etype, "det_duration_s"].dropna()
        if len(sub) == 0:
            continue
        ax.hist(sub, bins=50, histtype="step", linewidth=1.6, color=color,
                label=f"{etype} (n={len(sub):,})", density=True)
    print(f"  Overlaid event-class durations from {os.path.basename(EVENT_CATALOG_CSV)}")
else:
    print("  [INFO] No EVENT_CATALOG_CSV — noise-only duration histogram.")

ax.set_xlabel("Detection duration (s)", fontsize=12, fontweight="bold")
ax.set_ylabel("Density", fontsize=12, fontweight="bold")
ax.set_title("Duration distribution — noise vs. real event classes", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
fig_dur_path = os.path.join(RUN_DIR, f"fig_duration_hist_{STAMP}.png")
plt.savefig(fig_dur_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  [SAVED] {fig_dur_path}")

# ── Fig B: trigger_on_cft distribution ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["trigger_on_cft"].dropna(), bins=50, color="steelblue", alpha=0.8, edgecolor="white")
ax.axvline(PRIMARY_THR_ON, color="red", linestyle="--", linewidth=1.5,
           label=f"THR_ON = {PRIMARY_THR_ON}")
ax.set_xlabel("STA/LTA ratio at trigger onset (trigger_on_cft)", fontsize=12, fontweight="bold")
ax.set_ylabel("Number of noise windows", fontsize=12, fontweight="bold")
ax.set_title("How strongly did each noise candidate trigger?", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
fig_cft_path = os.path.join(RUN_DIR, f"fig_cft_hist_{STAMP}.png")
plt.savefig(fig_cft_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  [SAVED] {fig_cft_path}")

# ── Fig C: rows per station ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, max(4, 0.28 * len(station_stats))))
labels = [f"{net}.{sta}" for net, sta in station_stats.index]
ax.barh(labels, station_stats["rows"], color="#7f7f7f", edgecolor="white")
ax.invert_yaxis()
ax.set_xlabel("Number of noise rows", fontsize=12, fontweight="bold")
ax.set_title("Noise rows per station", fontsize=13, fontweight="bold")
ax.tick_params(labelsize=8)
plt.tight_layout()
fig_sta_path = os.path.join(RUN_DIR, f"fig_station_counts_{STAMP}.png")
plt.savefig(fig_sta_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  [SAVED] {fig_sta_path}")

# ── Fig D: temporal coverage (rows per year) ─────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
years = df["_t_on"].dt.year
year_counts = years.value_counts().sort_index()
ax.bar(year_counts.index.astype(str), year_counts.values, color="#7f7f7f", edgecolor="white")
ax.set_xlabel("Year", fontsize=12, fontweight="bold")
ax.set_ylabel("Number of noise rows", fontsize=12, fontweight="bold")
ax.set_title("Temporal coverage of the extracted noise class", fontsize=13, fontweight="bold")
ax.tick_params(axis="x", rotation=45, labelsize=9)
plt.tight_layout()
fig_time_path = os.path.join(RUN_DIR, f"fig_temporal_coverage_{STAMP}.png")
plt.savefig(fig_time_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  [SAVED] {fig_time_path}")



# =============================================================================
# SECTION 5 — EXAMPLE DIAGNOSTIC PLOTS: HOW DID STA/LTA ACTUALLY TRIGGER?
# =============================================================================

print(f"\n{'='*65}")
print(f"  STEP 3 — Example diagnostics  (selection: {EXAMPLE_SELECTION}, n={N_EXAMPLES})")
print(f"{'='*65}")

rng = np.random.default_rng(RANDOM_SEED)

if EXAMPLE_SELECTION == "top_cft":
    examples = df.sort_values("trigger_on_cft", ascending=False).head(N_EXAMPLES)
elif EXAMPLE_SELECTION == "stratified":
    # up to one example per station, cycling through stations until N_EXAMPLES is reached
    groups = [g.sample(frac=1, random_state=RANDOM_SEED) for _, g in df.groupby("station")]
    rng.shuffle(groups)
    picked, i = [], 0
    while len(picked) < N_EXAMPLES and any(i < len(g) for g in groups):
        for g in groups:
            if i < len(g):
                picked.append(g.iloc[i])
            if len(picked) >= N_EXAMPLES:
                break
        i += 1
    examples = pd.DataFrame(picked)
else:  # "random"
    examples = df.sample(n=min(N_EXAMPLES, len(df)), random_state=RANDOM_SEED)

print(f"  {len(examples)} example(s) selected.")

client_sds  = connect_sds(SDS_ROOT)
client_fdsn = connect_fdsn(ISTERRE_URL)

if client_sds is None or client_fdsn is None:
    print("\n  [WARN] SDS/FDSN unavailable — skipping example diagnostic plots "
          "(this section only works on the cluster / with VPN access).")
else:
    _t_min = examples["_t_on"].min()
    _t_max = examples["_t_on"].max()
    inventory = fetch_inventory(
        client_fdsn, str(_t_min.date()), str((_t_max + pd.Timedelta(days=1)).date()),
        lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX,
    )
    if inventory is None:
        print("  [WARN] Inventory fetch failed — waveforms will be uncalibrated raw counts.")

    n_ok, n_fail = 0, 0
    for _, row in examples.iterrows():
        try:
            net, sta, chan = row["network"], row["station"], row["channel"]
            t_on  = UTCDateTime(row["det_starttime"])
            t_off = UTCDateTime(row["det_endtime"])
            t_ctx_start = t_on  - CONTEXT_SEC
            t_ctx_end   = t_off + CONTEXT_SEC

            st_raw = client_sds.get_waveforms(net, sta, "*", chan, t_ctx_start, t_ctx_end)
            if len(st_raw) == 0:
                n_fail += 1
                continue

            # Same defensive merge as 04d: normalize dtypes before merging (a
            # station/day whose segments were written with different dtypes
            # makes ObsPy refuse to merge at all — see 04d's fix), then
            # split() so any remaining internal gap doesn't leave a masked
            # array that remove_response() can't handle.
            for _tr in st_raw:
                if _tr.data.dtype != np.float64:
                    _tr.data = _tr.data.astype(np.float64)
            try:
                st_raw.merge(fill_value=None)
            except Exception:
                st_raw.merge(fill_value=0)
            st_raw = st_raw.split()

            if len(st_raw) == 0:
                n_fail += 1
                continue
            tr_raw = max(st_raw, key=lambda t: t.stats.npts)   # longest contiguous segment
            if tr_raw.stats.npts < 10:
                n_fail += 1
                continue

            # Use preprocess_day() here, NOT remove_response_or_fallback()/
            # preprocess_signal_sp() (04a's catalog-event path). That path
            # calls remove_response(pre_filt=None) and relies on "bandpass
            # applied externally" to clean up afterward — fine for 04a's
            # ~240s catalog windows where the real transient is large, but
            # without pre_filt the deconvolution blows up the low-frequency
            # content (dividing by a near-zero instrument response below its
            # corner frequency), and for these low-amplitude noise windows
            # that residual can still dominate the 1-20 Hz band after
            # filtering, making the real signal invisible on a linear
            # auto-scaled axis — this was the actual "we don't see anything"
            # bug. preprocess_day() is the function 04d itself used to
            # accept these rows in the first place: it passes an explicit
            # pre_filt that tapers the response below 0.01 Hz / above 0.95x
            # Nyquist DURING the deconvolution, which avoids the blow-up
            # instead of trying to filter it out afterward. Using it here
            # also means the waveform/CFT shown now matches what 04d
            # actually saw, not a differently-preprocessed reconstruction.
            tr_vel = preprocess_day(tr_raw, inventory)   # broadband, unfiltered
            if tr_vel is None:
                n_fail += 1
                continue

            fs  = tr_vel.stats.sampling_rate
            nyq = fs / 2

            # -- Spectrogram from the BROADBAND trace, before any bandpass --
            # (shows the full frequency content, same convention as script 08's
            # example gallery, so noise examples are visually comparable to
            # the EQ/RS/IQ report figures)
            nperseg  = int(SPEC_NPERSEG_S * fs)
            noverlap = int(nperseg * SPEC_NOVERLAP_FRAC)
            nfft     = max(SPEC_NFFT, nperseg)   # nfft must be >= nperseg
            f_full, t_full, Sxx = spectrogram(
                tr_vel.data, fs=fs, window="hann",
                nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                scaling="density", mode="psd",
            )
            spec_db = 10 * np.log10(Sxx + PSD_FLOOR_EPS)

            # -- Bandpass copy for the waveform panel + STA/LTA (same band --
            # 04d used to detect this window in the first place)
            tr_wave = tr_vel.copy()
            tr_wave.filter("bandpass", freqmin=PRIMARY_FREQ_MIN,
                           freqmax=min(PRIMARY_FREQ_MAX, 0.9 * nyq),
                           corners=4, zerophase=True)

            cft, _ = run_sta_lta(tr_wave, PRIMARY_STA_S, PRIMARY_LTA_S,
                                 PRIMARY_THR_ON, PRIMARY_THR_OFF)

            tag = f"{net}.{sta}_{str(t_on)[:19].replace(':', '-')}"
            title_extra = (f"   |   trigger_on_cft={row['trigger_on_cft']:.2f}"
                           if pd.notna(row.get("trigger_on_cft")) else "")
            plot_noise_diagnostic(
                tr_wave, cft, f_full, t_full, spec_db,
                t_on, t_off, PRIMARY_THR_ON, PRIMARY_THR_OFF,
                RUN_DIR, STAMP, tag, title_extra=title_extra,
                spec_vmin=SPEC_VMIN, spec_vmax=SPEC_VMAX,
            )
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"    [WARN] Failed to plot {row.get('network')}.{row.get('station')} "
                  f"{row.get('det_starttime')}: {e}")
            continue

    print(f"\n  Diagnostics plotted : {n_ok}/{len(examples)}  (failed: {n_fail})")



# =============================================================================
# END
# =============================================================================

from datetime import datetime
print("\n" + "=" * 70)
print(f"  Run finished : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Noise rows   : {len(df):,}")
print(f"  All outputs  : {RUN_DIR}")
print(f"  Log file     : {log_path}")
print("=" * 70)

log_file.close()
