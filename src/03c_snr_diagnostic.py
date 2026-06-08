"""
03c_snr_diagnostic.py
=====================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : June 2026

Quick diagnostic: compute the raw-waveform SNR (Zhu's metric) for all
good-quality ice quake rows, save the distribution, and plot a histogram.

Purpose
-------
Before running 03c with a MIN_SIGNAL_SNR threshold, you need to know what
the actual SNR values look like on your ice quakes. This script extracts a
random sample of waveforms from SDS, computes SNR, and plots the histogram
so you can choose a threshold that keeps real signals and only rejects flat
or degenerate traces.

Interpretation guide
--------------------
  SNR ~ 1    : signal std ≈ noise std — barely above noise, but can be real
  SNR 1–3    : low but detectable — typical for ice quakes at far stations
  SNR 3–10   : moderate — good quality for training
  SNR > 10   : high — Zhu's original threshold (too strict for cryoseismicity)
  SNR → 0    : degenerate waveform (flat trace, data gap) — should be excluded

Expected result: you should see a roughly log-normal distribution starting
around 1–3 and tapering off at high values. The MIN_SIGNAL_SNR cutoff should
be placed just below the left edge of the main population (typically ~1.2–2.0),
NOT at 10.
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

CSV_PATH = (
    "/data/failles/louisels/project/results/outputs_04a/groult/"
    "run_20260531_104936/catalog_windows_20260531_104936.csv"
)
SDS_ROOT   = "/data/sig/SDS"
OUTPUT_DIR = "/data/failles/louisels/project/results/outputs_03c"

# Quality gate (same as 03c)
SNR_FULL_MEAN_MIN  = 2.70
SNR_S2N_MEDIAN_MIN = 20.99

# Waveform parameters (same as 03c)
TARGET_FS  = 100
WINDOW_S   = 30
PRE_PAD_S  = 10

# How many rows to sample for the diagnostic (reduce if SDS is slow)
N_SAMPLE = 300   # set to None to use all rows

# =============================================================================
# SETUP
# =============================================================================

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from obspy import UTCDateTime

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_setup import connect_sds, set_matplotlib_defaults
from preprocessing import load_3component, compute_snr_window

set_matplotlib_defaults()
os.makedirs(OUTPUT_DIR, exist_ok=True)

client_sds = connect_sds(SDS_ROOT)
if client_sds is None:
    print("[ERROR] SDS unavailable. Run on the cluster.")
    sys.exit(1)

ITP = int(PRE_PAD_S * TARGET_FS)
NT  = int(WINDOW_S  * TARGET_FS)

# =============================================================================
# LOAD AND FILTER
# =============================================================================

df = pd.read_csv(CSV_PATH, low_memory=False)
df_iq = df[df["event_type"] == "ice quake"].copy()

mask_q = (
    (df_iq["SNR_full_mean"]  >= SNR_FULL_MEAN_MIN) &
    (df_iq["SNR_s2n_median"] >= SNR_S2N_MEDIAN_MIN)
)
df_good = df_iq[mask_q].copy()
print(f"Good-quality ice quake rows: {len(df_good):,}")

if N_SAMPLE is not None and N_SAMPLE < len(df_good):
    df_sample = df_good.sample(n=N_SAMPLE, random_state=42)
    print(f"Sampling {N_SAMPLE} rows for diagnostic.")
else:
    df_sample = df_good
    print("Using all rows for diagnostic.")

# =============================================================================
# COMPUTE SNR DISTRIBUTION
# =============================================================================

snr_values   = []
catalog_snrs = []   # SNR_full_mean from catalog, for comparison
n_ok = 0
n_skip = 0

for idx, row in df_sample.iterrows():
    net  = row["network"]
    sta  = row["station"]
    chan = row["channel"]
    t_on = UTCDateTime(row["det_starttime"])

    try:
        data3 = load_3component(client_sds, net, sta, "", chan,
                                t_on - PRE_PAD_S,
                                t_on - PRE_PAD_S + WINDOW_S,
                                target_fs=TARGET_FS, window_s=WINDOW_S)
        snr = compute_snr_window(data3[:, 2], ITP)
        snr_values.append(snr)
        catalog_snrs.append(row["SNR_full_mean"])
        n_ok += 1
    except Exception:
        n_skip += 1

print(f"\nExtracted: {n_ok} waveforms  ({n_skip} skipped — SDS gaps)")

if len(snr_values) == 0:
    print("[ERROR] No SNR values computed. Check SDS connection.")
    sys.exit(1)

snr_arr = np.array(snr_values)
print(f"\n  SNR distribution (Zhu's raw-waveform metric):")
for pct in [5, 10, 25, 50, 75, 90, 95]:
    print(f"    p{pct:2d}: {np.percentile(snr_arr, pct):.2f}")
print(f"    mean: {snr_arr.mean():.2f}  ±  {snr_arr.std():.2f}")
print(f"    fraction > 1.5 : {(snr_arr > 1.5).mean()*100:.1f} %")
print(f"    fraction > 2.0 : {(snr_arr > 2.0).mean()*100:.1f} %")
print(f"    fraction > 3.0 : {(snr_arr > 3.0).mean()*100:.1f} %")
print(f"    fraction > 5.0 : {(snr_arr > 5.0).mean()*100:.1f} %")
print(f"    fraction > 10.0: {(snr_arr > 10.0).mean()*100:.1f} %")

# Save to CSV for reference
out_csv = os.path.join(OUTPUT_DIR, "snr_diagnostic.csv")
pd.DataFrame({"snr_zhu": snr_values, "snr_catalog": catalog_snrs}).to_csv(out_csv, index=False)
print(f"\n[SAVED] {out_csv}")

# =============================================================================
# FIGURES
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Left: histogram of Zhu SNR with candidate thresholds ---
ax = axes[0]
ax.hist(snr_arr, bins=60, color="#2171b5", edgecolor="white", lw=0.4, alpha=0.85)
for thr, color, label in [
    (1.5,  "#e6550d", "1.5 (current 03c)"),
    (2.0,  "#31a354", "2.0"),
    (3.0,  "#756bb1", "3.0"),
    (10.0, "#de2d26", "10.0 (Zhu default)"),
]:
    ax.axvline(thr, color=color, lw=1.5, ls="--", label=f"SNR = {label}")
ax.set_xlabel("Raw-waveform SNR  (Zhu's metric)")
ax.set_ylabel("Count")
ax.set_title("SNR distribution — good-quality ice quakes\n"
             "(already pass catalog quality gate)")
ax.legend(fontsize=9)
ax.set_xlim(0, min(snr_arr.max() * 1.05, 30))
ax.grid(axis="y", lw=0.4, alpha=0.4)

# --- Right: scatter — Zhu SNR vs catalog SNR_full_mean ---
ax = axes[1]
ax.scatter(catalog_snrs, snr_values, alpha=0.4, s=15, c="#2171b5")
ax.axhline(1.5,  color="#e6550d", lw=1.2, ls="--", label="SNR_zhu = 1.5")
ax.axhline(10.0, color="#de2d26", lw=1.2, ls="--", label="SNR_zhu = 10")
ax.set_xlabel("Catalog SNR_full_mean (04a quality gate)")
ax.set_ylabel("Raw-waveform SNR (Zhu's metric, 03c)")
ax.set_title("Two SNR metrics compared\n"
             "— they measure different things")
ax.legend(fontsize=9)
ax.grid(lw=0.4, alpha=0.4)

plt.suptitle(f"03c SNR diagnostic — {n_ok} good-quality ice quake waveforms",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "fig_snr_diagnostic.png")
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[SAVED] {fig_path}")

print(f"""
Recommended threshold selection
--------------------------------
Look at the histogram (fig_snr_diagnostic.png):
  - The main population of real ice quakes should form a clear peak.
  - Flat/degenerate traces will cluster near SNR = 0.
  - Set MIN_SIGNAL_SNR just to the right of the near-zero cluster,
    i.e. just below the left edge of the main population.
  - A value of 1.2–2.0 is typical for cryoseismic data.
  - Do NOT use 10 — that would discard {(snr_arr > 10.0).mean()*100:.0f} % of your confirmed events.

Once decided, update MIN_SIGNAL_SNR in 03c and re-run.
""")
