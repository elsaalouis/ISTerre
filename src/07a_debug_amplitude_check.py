"""
07a_debug_amplitude_check.py
=============================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : July 2026

One-off diagnostic script (does NOT run the full 07a pipeline). Pulls ONE raw
waveform per class straight from SDS and runs it through the exact same
cleaning steps as 07a_spectrogram_dataset_build.py (response removal -> VEL
[m/s] -> resample -> trim), then prints real amplitude and PSD numbers.

Why this check
---------------
Cell 6b in 07b_train_cnn_classifier_colab.ipynb showed every class's packed
spectrograms sitting almost entirely at the -120 dB floor. That floor value
is exact: 10*log10(1e-12) = -120.0, where 1e-12 is the epsilon added inside
spectrogram_image() in 07a (`Sxx + 1e-12`) purely to avoid log(0). Ice quake
train images showed essentially ZERO variance across all 1269 samples --
every pixel at the floor. That means the real power spectral density is
landing below 1e-12 (m/s)^2/Hz almost everywhere in almost every image,
which either means (a) these are genuinely extremely weak signals relative
to that floor, or (b) something upstream (response removal, units) is
producing abnormally tiny velocity amplitudes. This script checks which, by
printing actual numbers at each processing stage for one example event per
class.

Run directly on the ISTerre cluster (needs SDS + FDSN access):
    python3 07a_debug_amplitude_check.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.signal import spectrogram

from obspy import UTCDateTime, Stream

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_setup import connect_sds, connect_fdsn
from preprocessing import cosine_taper


# =============================================================================
# CONFIGURATION -- must match 07a_spectrogram_dataset_build.py exactly
# =============================================================================

CSV_PATH    = "/data/failles/louisels/project/results/outputs_04a/all-99-features-recent+3C/catalog_windows_20260708_174019.csv"
SDS_ROOT    = "/data/sig/SDS"
ISTERRE_URL = "http://ist-sc3-geobs.osug.fr:8080"

TARGET_CLASSES = ["earthquake", "rockslide", "ice quake"]
FILTER_QUALITY = True

TARGET_FS     = 100
WINDOW_PRE_S  = 5
WINDOW_POST_S = 55
WINDOW_S      = WINDOW_PRE_S + WINDOW_POST_S
NT            = int(WINDOW_S * TARGET_FS)
HORIZONTAL_SUFFIXES = [("N", "E"), ("2", "1")]

SPEC_NPERSEG_S     = 2.0
SPEC_NOVERLAP_FRAC = 0.75
SPEC_NFFT          = 256
FREQ_MAX_KEEP      = 45.0
SPEC_NPERSEG  = int(SPEC_NPERSEG_S * TARGET_FS)
SPEC_NOVERLAP = int(SPEC_NPERSEG * SPEC_NOVERLAP_FRAC)

PSD_FLOOR_EPS = 1e-12   # same epsilon as spectrogram_image() in 07a



# =============================================================================
# Same cleaning function as 07a (copied verbatim, not imported, so this
# script never triggers 07a's own top-level pipeline)
# =============================================================================

def _process_trace(tr_raw, inventory, target_fs, t_start, t_end, nt):
    try:
        tr = tr_raw.copy()
        tr.detrend("demean")
        tr.detrend("linear")
        cosine_taper(Stream(traces=[tr]), max_percentage=0.05)

        tr.remove_response(inventory=inventory, output="VEL", water_level=60, pre_filt=None)

        if abs(tr.stats.sampling_rate - target_fs) > 0.5:
            tr.resample(target_fs)

        tr.trim(starttime=t_start, endtime=t_end, pad=True, fill_value=0)

        d = tr.data.astype(np.float32)
        if len(d) < nt:
            d = np.pad(d, (0, nt - len(d)))
        elif len(d) > nt:
            d = d[:nt]

        if d is None or len(d) == 0 or np.any(np.isnan(d)) or np.any(np.isinf(d)) or np.max(np.abs(d)) == 0:
            return None
        return d

    except Exception as e:
        print(f"    [FAIL _process_trace] {e}")
        return None



# =============================================================================
# Per-event inspection
# =============================================================================

def inspect_event(client_sds, inventory, net, sta, chan, etype, det_starttime):
    t_on = UTCDateTime(det_starttime)
    t_start = t_on - WINDOW_PRE_S
    t_end = t_start + WINDOW_S

    print(f"\n{'='*70}")
    print(f"  {etype}  |  {net}.{sta}.{chan}  |  onset={t_on}")
    print(f"{'='*70}")

    # -- Raw, pre-response-removal (counts) -------------------------------------
    try:
        st_raw = client_sds.get_waveforms(net, sta, "*", chan, t_start, t_end)
    except Exception as e:
        print(f"  [FAIL] get_waveforms error: {e}")
        return
    if not st_raw:
        print("  [FAIL] no raw waveform returned from SDS for this window.")
        return
    st_raw.merge(method=1, fill_value="interpolate")
    tr_raw = st_raw[0]
    raw = tr_raw.data.astype(np.float64)
    print(f"  RAW (counts)       : n={len(raw)}  peak={np.max(np.abs(raw)):.6g}  "
          f"rms={np.sqrt(np.mean(raw**2)):.6g}")

    # -- Cleaned, post-response-removal (VEL, m/s) -------------------------------
    d = _process_trace(tr_raw, inventory, TARGET_FS, t_start, t_end, NT)
    if d is None:
        print("  [FAIL] _process_trace returned None (response removal or resample failed).")
        return

    n_pre = WINDOW_PRE_S * TARGET_FS
    pre_onset  = d[:n_pre]     # before the detected onset
    post_onset = d[n_pre:]     # after the detected onset

    print(f"  CLEANED (m/s)      : n={len(d)}  peak={np.max(np.abs(d)):.6g}  "
          f"rms={np.sqrt(np.mean(d**2)):.6g}")
    print(f"    pre-onset  ({WINDOW_PRE_S:>2d}s): peak={np.max(np.abs(pre_onset)):.6g}  "
          f"rms={np.sqrt(np.mean(pre_onset**2)):.6g}")
    print(f"    post-onset ({WINDOW_POST_S:>2d}s): peak={np.max(np.abs(post_onset)):.6g}  "
          f"rms={np.sqrt(np.mean(post_onset**2)):.6g}")
    snr = np.max(np.abs(post_onset)) / (np.max(np.abs(pre_onset)) + 1e-30)
    print(f"    post/pre peak ratio (rough SNR): {snr:.2f}x")

    # -- Raw (non-log) PSD -- the exact quantity the +1e-12 floor is compared to --
    _, _, Sxx = spectrogram(
        d, fs=TARGET_FS, window="hann",
        nperseg=SPEC_NPERSEG, noverlap=SPEC_NOVERLAP, nfft=SPEC_NFFT,
        scaling="density", mode="psd",
    )
    print(f"  PSD (m/s)^2/Hz     : max={Sxx.max():.6g}  median={np.median(Sxx):.6g}  "
          f"(floor epsilon = {PSD_FLOOR_EPS:.0e})")
    frac_above_floor = np.mean(Sxx > PSD_FLOOR_EPS)
    print(f"  Fraction of PSD bins above the floor: {frac_above_floor*100:.2f}%")



# =============================================================================
# Main
# =============================================================================

def main():
    print("Connecting to SDS + FDSN ...")
    client_sds  = connect_sds(SDS_ROOT)
    client_fdsn = connect_fdsn(ISTERRE_URL)
    if client_sds is None or client_fdsn is None:
        print("[ERROR] Could not connect to SDS/FDSN.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = df[df["event_type"].isin(TARGET_CLASSES)].copy()
    if FILTER_QUALITY and "quality_ok" in df.columns:
        df = df[df["quality_ok"] == True].copy()
    df = df.dropna(
        subset=["event_time", "event_type", "network", "station", "channel", "det_starttime"]
    ).reset_index(drop=True)
    print(f"Catalog rows after filtering: {len(df):,}")

    print("Fetching instrument inventory (per network) ...")
    networks = sorted(df["network"].dropna().unique())
    t = pd.to_datetime(df["event_time"])
    inv_start = UTCDateTime((t.min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    inv_end   = UTCDateTime((t.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))

    inventory = None
    for net in networks:
        try:
            inv_net = client_fdsn.get_stations(
                network=net, station="*", starttime=inv_start, endtime=inv_end, level="response",
            )
            inventory = inv_net if inventory is None else inventory + inv_net
        except Exception as e:
            print(f"  [WARN] {net} inventory failed: {e}")
    if inventory is None:
        print("[ERROR] No inventory available for any network.")
        sys.exit(1)
    print("Inventory ready.\n")

    # One example row per class. Change .iloc[0] to inspect a different event.
    for etype in TARGET_CLASSES:
        sub = df[df["event_type"] == etype]
        if len(sub) == 0:
            print(f"\n[SKIP] no rows for {etype}")
            continue
        row = sub.iloc[0]
        inspect_event(
            client_sds, inventory,
            row["network"], row["station"], row["channel"], etype, row["det_starttime"],
        )

    print(f"\n{'='*70}")
    print("  Done.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
