"""
05c_kurtosis_onset_comparison.py
=================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : May 2026

Goal
----
Compare the preliminary onset time from the STA/LTA method with the kurtosis-refined onset (Fuchs et al. 2018) for rockslide / landslide events

Two types of output:
  Fig 1 — Statistical overview
      Histogram of onset corrections (onset_refine_s = t_kurtosis - t_detected)
      Negative values mean the kurtosis picker found the onset earlier than in the detection method

  Fig 2 — Per-event kurtosis diagnostic (waveform + CF + cCF)
      For the N_EVENTS_DIAG best-SNR events:
        - waveform (1–20 Hz) with both onset markers
        - CF(t): kurtosis characteristic function with β = 3 reference
        - cCF(t) and d(cCF)/dt with the steepest-rise peak marked

Input
-----
  catalog_windows_<stamp>.csv  produced by script 04a_spectrogram_sta_lta_catalog.py
  Waveforms from the SDS archive (reloaded for the diagnostic plots)
"""


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

SDS_ROOT    = "/data/sig/SDS"
ISTERRE_URL = "http://ist-sc3-geobs.osug.fr:8080"
OUTPUT_DIR  = "/data/failles/louisels/project/results/outputs_05c"

# Path to the script 04 CSV (picks the most recent run automatically)
CSV_GLOB = "/data/failles/louisels/project/results/outputs_04a/run_*/catalog_windows_*.csv"

# Event types that were refined by kurtosis (must match KURTOSIS_ETYPES in script 04)
KURTOSIS_ETYPES = ('rockslide', 'landslide')

# Kurtosis parameters (must match script 04)
KURTOSIS_FREQ_MIN      = 1.0    # Hz
KURTOSIS_FREQ_MAX      = 5.0    # Hz
KURTOSIS_DT_S          = 5.0    # s
KURTOSIS_SEARCH_BEFORE = 10.0   # s
KURTOSIS_SEARCH_AFTER  = 1.0    # s

# Waveform display parameters
WAVEFORM_FREQ_MIN  = 1.0    # Hz — broadband bandpass for the waveform panel
WAVEFORM_FREQ_MAX  = 20.0   # Hz
WAVEFORM_BEFORE    = 15.0   # s before detected onset to display
WAVEFORM_AFTER     = 30.0   # s after detected onset to display

# Number of events to show in the per-event diagnostic figure (Fig 2)
# Events are ranked by SNR_s2n_median (highest first)
N_EVENTS_DIAG = 10

Z_CHANNELS = "??Z"


# =============================================================================
# SECTION 2 — SETUP
# =============================================================================

import os
import sys
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe

from obspy import UTCDateTime

sys.path.insert(0, os.path.dirname(__file__))
from run_setup import (
    create_run_dir, setup_logging, connect_sds, connect_fdsn,
    fetch_inventory, set_matplotlib_defaults,
)
from preprocessing import remove_response_or_fallback, build_station_times_df
from detection import refine_onset_kurtosis

RUN_DIR, _RUN_STAMP = create_run_dir(OUTPUT_DIR)
_log_file, _log_filename = setup_logging(
    RUN_DIR, "05c_kurtosis_onset_comparison.py",
    extra_info=(f"KURTOSIS {KURTOSIS_FREQ_MIN}–{KURTOSIS_FREQ_MAX} Hz  "
                f"dt={KURTOSIS_DT_S}s  search={KURTOSIS_SEARCH_BEFORE}/{KURTOSIS_SEARCH_AFTER}s  "
                f"N_DIAG={N_EVENTS_DIAG}")
)
set_matplotlib_defaults()

client_sds  = connect_sds(SDS_ROOT)
client_fdsn = connect_fdsn(ISTERRE_URL)

if client_sds is None:
    print("[ERROR] SDS client unavailable — waveform diagnostic (Fig 2) will be skipped.")
if client_fdsn is None:
    print("[WARN] FDSN client unavailable — response removal will be skipped.")


# =============================================================================
# SECTION 3 — LOAD AND VALIDATE CSV
# =============================================================================

csv_files = sorted(glob.glob(CSV_GLOB))
if not csv_files:
    print(f"[ERROR] No CSV found matching: {CSV_GLOB}")
    sys.exit(1)

csv_path = csv_files[-1]   # most recent run
print(f"\nLoading CSV: {csv_path}")
df_all = pd.read_csv(csv_path, low_memory=False)
print(f"  Total rows: {len(df_all)}")

# Check required columns
required = ['event_type', 'station', 'network', 'det_starttime',
            'det_starttime_raw', 'onset_refine_s']
missing = [c for c in required if c not in df_all.columns]
if missing:
    print(f"[ERROR] Missing columns in CSV: {missing}")
    print("  Make sure KURTOSIS_REFINE=True was set in script 04 before re-running.")
    sys.exit(1)

# Filter: keep only event types where kurtosis was applied
df_rock = df_all[df_all['event_type'].str.lower().isin(KURTOSIS_ETYPES)].copy()
print(f"  Rockslide / landslide rows: {len(df_rock)}")

# Drop rows where onset_refine_s is NaN (kurtosis failed / trace too short)
df_rock = df_rock.dropna(subset=['onset_refine_s'])
n_refined = (df_rock['onset_refine_s'] != 0.0).sum()
print(f"  Rows with non-zero refinement: {n_refined} / {len(df_rock)}")

if len(df_rock) == 0:
    print("[ERROR] No valid kurtosis refinements found in the CSV. "
          "Check that script 04 was run with KURTOSIS_REFINE=True.")
    sys.exit(1)


# =============================================================================
# SECTION 4 — FIG 1: STATISTICAL OVERVIEW
# =============================================================================

print("\nFig 1: Statistical overview ...")

corrections = df_rock['onset_refine_s'].values          # seconds (negative = earlier)
n_total     = len(corrections)
n_neg       = (corrections < 0).sum()
n_zero      = (corrections == 0).sum()
n_pos       = (corrections > 0).sum()

print(f"  n = {n_total}  |  earlier: {n_neg}  |  unchanged: {n_zero}  |  later: {n_pos}")
print(f"  mean={corrections.mean():.2f}s  median={np.median(corrections):.2f}s  "
      f"std={corrections.std():.2f}s  "
      f"min={corrections.min():.2f}s  max={corrections.max():.2f}s")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"Kurtosis onset correction for {', '.join(KURTOSIS_ETYPES)} events  "
    f"(n = {n_total})\n"
    f"onset_refine_s = t_kurtosis − t_detected  "
    f"(negative = kurtosis onset is earlier)",
    fontsize=11, fontweight='bold'
)

# ── Panel A: histogram ─────────────────────────────────────────────────────
ax = axes[0]
bins = np.arange(
    np.floor(corrections.min()) - 0.5,
    np.ceil(corrections.max())  + 1.0,
    0.5
)
ax.hist(corrections, bins=bins, color='steelblue', edgecolor='white', linewidth=0.5)
ax.axvline(0,                       color='black', lw=1.5, ls='--', label='No correction')
ax.axvline(corrections.mean(),      color='red',   lw=1.5, ls='-',  label=f'Mean = {corrections.mean():.2f} s')
ax.axvline(np.median(corrections),  color='orange',lw=1.5, ls='-',  label=f'Median = {np.median(corrections):.2f} s')
ax.set_xlabel('Onset correction [s]', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.set_title('Distribution of kurtosis corrections', fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, lw=0.5)

# ── Panel B: per-station boxplot ───────────────────────────────────────────
ax = axes[1]
stas_present = df_rock['station'].unique()
data_by_sta  = [df_rock.loc[df_rock['station'] == s, 'onset_refine_s'].values
                for s in stas_present]
# Keep only stations with at least 3 detections
mask_enough  = [len(d) >= 3 for d in data_by_sta]
stas_plot    = [s for s, ok in zip(stas_present, mask_enough) if ok]
data_plot    = [d for d, ok in zip(data_by_sta,  mask_enough) if ok]

if data_plot:
    bp = ax.boxplot(data_plot, labels=stas_plot, vert=True,
                    patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.5),
                    medianprops=dict(color='red', lw=1.5),
                    flierprops=dict(marker='.', ms=4, color='grey'))
    ax.axhline(0, color='black', lw=1.0, ls='--', alpha=0.6)
    ax.set_xlabel('Station', fontsize=10)
    ax.set_ylabel('Onset correction [s]', fontsize=10)
    ax.set_title('Per-station distribution', fontsize=10)
    ax.tick_params(axis='x', labelrotation=45, labelsize=7)
    ax.grid(True, axis='y', alpha=0.3, lw=0.5)
else:
    ax.text(0.5, 0.5, 'Not enough data per station\n(< 3 detections)',
            ha='center', va='center', transform=ax.transAxes, fontsize=10)

plt.tight_layout()
out1 = os.path.join(RUN_DIR, f"fig_kurtosis_stats_{_RUN_STAMP}.png")
fig.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  [SAVED] {out1}")


# =============================================================================
# SECTION 5 — FIG 2: PER-EVENT KURTOSIS DIAGNOSTIC
# =============================================================================

if client_sds is None:
    print("\nFig 2: Skipped (no SDS client).")
else:
    print(f"\nFig 2: Per-event diagnostic ({N_EVENTS_DIAG} events) ...")

    # Select N best-SNR events (non-zero correction preferred)
    df_sel = df_rock.copy()
    if 'SNR_s2n_median' in df_sel.columns:
        df_sel = df_sel.sort_values('SNR_s2n_median', ascending=False)
    df_sel = df_sel.head(N_EVENTS_DIAG).reset_index(drop=True)

    # Fetch inventory once for response removal
    inventory = None
    if client_fdsn:
        try:
            t_min = df_rock['det_starttime_raw'].dropna().min()
            t_max = df_rock['det_starttime_raw'].dropna().max()
            inventory = fetch_inventory(client_fdsn, str(t_min)[:10], str(t_max)[:10])
        except Exception as e:
            print(f"  [WARN] Could not fetch inventory: {e}")

    # ── Figure layout: N_rows × 3 columns ─────────────────────────────────
    n_rows = len(df_sel)
    fig = plt.figure(figsize=(18, max(4, n_rows * 3.2)))
    gs  = gridspec.GridSpec(n_rows, 3, figure=fig,
                            hspace=0.55, wspace=0.35,
                            left=0.07, right=0.97, top=0.94, bottom=0.05)

    col_titles = [
        f'Waveform ({WAVEFORM_FREQ_MIN}–{WAVEFORM_FREQ_MAX} Hz)',
        'CF(t) — kurtosis characteristic function',
        'cCF(t) and d(cCF)/dt',
    ]
    for col, title in enumerate(col_titles):
        fig.text(
            [0.18, 0.50, 0.83][col], 0.97,
            title, ha='center', va='top',
            fontsize=10, fontweight='bold'
        )

    PREL_COLOR     = '#E53935'   # red — preliminary onset
    KURTOSIS_COLOR = '#1565C0'   # blue — kurtosis-refined onset
    BETA3_COLOR    = '#999999'   # grey — β = 3 reference line

    for row_idx, (_, row) in enumerate(df_sel.iterrows()):
        net    = row['network']
        sta    = row['station']
        etype  = row['event_type']
        refine = row['onset_refine_s']
        snr    = row.get('SNR_s2n_median', np.nan)

        t_detected = UTCDateTime(row['det_starttime_raw'])
        t_kurtosis = UTCDateTime(row['det_starttime'])

        # Y-label shared across columns
        label_txt = (f"{sta}  {str(t_detected)[:19]}\n"
                     f"{etype}  Δ={refine:+.2f}s  SNR={snr:.1f}" if not np.isnan(snr)
                     else f"{sta}  {str(t_detected)[:19]}\n{etype}  Δ={refine:+.2f}s")

        # ── Load waveform ──────────────────────────────────────────────
        t_load_start = t_detected - WAVEFORM_BEFORE - KURTOSIS_SEARCH_BEFORE - KURTOSIS_DT_S
        t_load_end   = t_detected + WAVEFORM_AFTER

        try:
            st_raw = client_sds.get_waveforms(
                network=net, station=sta, location='*', channel=Z_CHANNELS,
                starttime=t_load_start, endtime=t_load_end
            )
        except Exception as e:
            print(f"  [{sta} {t_detected}] waveform load failed: {e}")
            for col in range(3):
                ax = fig.add_subplot(gs[row_idx, col])
                ax.text(0.5, 0.5, 'No waveform', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='grey')
            continue

        if len(st_raw) == 0:
            print(f"  [{sta} {t_detected}] empty stream.")
            for col in range(3):
                ax = fig.add_subplot(gs[row_idx, col])
                ax.text(0.5, 0.5, 'No waveform', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='grey')
            continue

        st_raw.merge(fill_value='interpolate')
        tr_raw = st_raw[0]

        # Response removal (or pass-through on failure)
        if inventory:
            try:
                station_times_df = build_station_times_df(st_raw, t_load_start, t_load_end)
                st_vel = remove_response_or_fallback(st_raw, inventory, station_times_df)
                tr_vel = st_vel[0] if len(st_vel) > 0 else tr_raw
            except Exception:
                tr_vel = tr_raw
        else:
            tr_vel = tr_raw

        # ── Broadband trace for display (1–20 Hz) ─────────────────────
        tr_display = tr_vel.copy()
        nyq = tr_display.stats.sampling_rate / 2.0
        tr_display.filter('bandpass',
                          freqmin=WAVEFORM_FREQ_MIN,
                          freqmax=min(WAVEFORM_FREQ_MAX, 0.9 * nyq),
                          corners=2, zerophase=True)

        # ── Narrow band for kurtosis (1–5 Hz) ─────────────────────────
        tr_kurt = tr_vel.copy()
        tr_kurt.filter('bandpass',
                       freqmin=KURTOSIS_FREQ_MIN,
                       freqmax=min(KURTOSIS_FREQ_MAX, 0.9 * nyq),
                       corners=2, zerophase=True)

        # Re-run kurtosis to retrieve the diagnostic info dict
        t_refined_check, kurt_info = refine_onset_kurtosis(
            tr_kurt, t_detected,
            dt_s          = KURTOSIS_DT_S,
            search_before = KURTOSIS_SEARCH_BEFORE,
            search_after  = KURTOSIS_SEARCH_AFTER,
        )
        has_kurt = bool(kurt_info)

        # ── COLUMN 0: waveform ─────────────────────────────────────────
        ax0 = fig.add_subplot(gs[row_idx, 0])
        t0_disp  = t_detected - WAVEFORM_BEFORE
        t1_disp  = t_detected + WAVEFORM_AFTER
        tr_clip  = tr_display.slice(t0_disp, t1_disp)
        if tr_clip.stats.npts > 0:
            t_ax = np.arange(tr_clip.stats.npts) / tr_clip.stats.sampling_rate - WAVEFORM_BEFORE
            ax0.plot(t_ax, tr_clip.data, lw=0.6, color='#333333', alpha=0.9)
        ax0.axvline(0,       color=PREL_COLOR,   lw=1.5, ls='--', label='Detected onset')
        ax0.axvline(refine,  color=KURTOSIS_COLOR, lw=1.5, ls='-',  label='Kurtosis onset')
        ax0.set_xlim(-WAVEFORM_BEFORE, WAVEFORM_AFTER)
        ax0.set_ylabel(label_txt, fontsize=7)
        ax0.tick_params(labelsize=7)
        ax0.grid(True, alpha=0.2, lw=0.4)
        if row_idx == n_rows - 1:
            ax0.set_xlabel('Time from detected onset [s]', fontsize=8)
        if row_idx == 0:
            ax0.legend(fontsize=7, loc='upper right')

        # ── COLUMN 1: CF(t) ────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[row_idx, 1])
        if has_kurt:
            t0_kurt    = kurt_info['t0']
            cf_times   = kurt_info['cf_times_s']                    # seconds from t0_kurt
            cf_rel     = np.array(cf_times) - float(t_detected - t0_kurt)  # seconds from t_detected
            ax1.plot(cf_rel, kurt_info['cf_values'], lw=0.9, color='#555555')
            ax1.axhline(3.0, color=BETA3_COLOR, lw=1.0, ls=':', label='β = 3 (Gaussian)')
            ax1.axvline(0,      color=PREL_COLOR,   lw=1.5, ls='--')
            ax1.axvline(refine, color=KURTOSIS_COLOR, lw=1.5, ls='-')
            ax1.set_xlim(-KURTOSIS_SEARCH_BEFORE - 1, KURTOSIS_SEARCH_AFTER + 1)
            ax1.legend(fontsize=7, loc='upper left')
        else:
            ax1.text(0.5, 0.5, 'CF unavailable', ha='center', va='center',
                     transform=ax1.transAxes, fontsize=9, color='grey')
        ax1.set_ylabel('β (kurtosis)', fontsize=8)
        ax1.tick_params(labelsize=7)
        ax1.grid(True, alpha=0.2, lw=0.4)
        if row_idx == n_rows - 1:
            ax1.set_xlabel('Time from detected onset [s]', fontsize=8)

        # ── COLUMN 2: cCF + d(cCF)/dt ─────────────────────────────────
        ax2  = fig.add_subplot(gs[row_idx, 2])
        ax2b = ax2.twinx()
        if has_kurt:
            t0_kurt  = kurt_info['t0']
            cf_times = kurt_info['cf_times_s']
            cf_rel   = np.array(cf_times) - float(t_detected - t0_kurt)
            ccf      = kurt_info['ccf_values']
            dccf     = kurt_info['dccf']

            ax2.plot(cf_rel, ccf, lw=1.0, color='#1A237E', label='cCF')
            ax2.set_ylabel('cCF', fontsize=8, color='#1A237E')
            ax2.tick_params(axis='y', labelcolor='#1A237E', labelsize=7)

            # d(cCF)/dt on twin axis — time axis is one step shorter
            cf_rel_d = cf_rel[1:]    # diff reduces length by 1
            ax2b.plot(cf_rel_d, dccf, lw=0.8, color='#FF6F00', alpha=0.8, label='d(cCF)/dt')
            ax2b.set_ylabel('d(cCF)/dt', fontsize=8, color='#FF6F00')
            ax2b.tick_params(axis='y', labelcolor='#FF6F00', labelsize=7)

            # Mark the onset step — use the same i_peak returned by the detection
            # function (cCF threshold-based), not a fresh argmax(dccf).
            i_peak = int(kurt_info.get('i_peak', int(np.argmax(dccf))))
            i_peak = min(i_peak, len(cf_rel_d) - 1)   # guard against edge case
            ax2b.axvline(cf_rel_d[i_peak], color='#FF6F00', lw=1.2, ls=':',
                         label='argmax d(cCF)/dt')

            ax2.axvline(0,      color=PREL_COLOR,   lw=1.5, ls='--')
            ax2.axvline(refine, color=KURTOSIS_COLOR, lw=1.5, ls='-')
            ax2.set_xlim(-KURTOSIS_SEARCH_BEFORE - 1, KURTOSIS_SEARCH_AFTER + 1)

            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2b.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')
        else:
            ax2.text(0.5, 0.5, 'cCF unavailable', ha='center', va='center',
                     transform=ax2.transAxes, fontsize=9, color='grey')
        ax2.tick_params(axis='x', labelsize=7)
        ax2.grid(True, alpha=0.2, lw=0.4)
        if row_idx == n_rows - 1:
            ax2.set_xlabel('Time from detected onset [s]', fontsize=8)

        print(f"  [{row_idx+1}/{n_rows}] {sta}  {str(t_detected)[:19]}  "
              f"Δ={refine:+.2f}s  SNR={snr:.1f}" if not np.isnan(snr)
              else f"  [{row_idx+1}/{n_rows}] {sta}  {str(t_detected)[:19]}  Δ={refine:+.2f}s")

    fig.suptitle(
        f"Kurtosis onset diagnostic — {', '.join(KURTOSIS_ETYPES)}  "
        f"(top {n_rows} by SNR_s2n_median)\n"
        f"Red dashed = detected onset  |  Blue solid = Kurtosis-refined (Fuchs 2018)  |  "
        f"Kurtosis band: {KURTOSIS_FREQ_MIN}–{KURTOSIS_FREQ_MAX} Hz  dt = {KURTOSIS_DT_S} s",
        fontsize=10, fontweight='bold', y=0.995
    )

    out2 = os.path.join(RUN_DIR, f"fig_kurtosis_diagnostic_{_RUN_STAMP}.png")
    fig.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  [SAVED] {out2}")


# =============================================================================
# END
# =============================================================================

from datetime import datetime
print("\n" + "=" * 60)
print(f"  Run finished : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Output dir   : {RUN_DIR}")
print(f"  Log file     : {_log_filename}")
print("=" * 60)

_log_file.close()
