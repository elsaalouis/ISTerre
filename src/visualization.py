"""
visualization.py
================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : April 2026

All figure-generating functions used across the pipeline scripts:
  - plot_event_waveforms()    : waveform + PSD panels per station  (script 01)
  - plot_station_coverage()   : histogram + bar chart + box plot    (script 01)
  - plot_sta_lta()            : waveform + STA/LTA ratio panels     (script 02)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.signal import welch


# =============================================================================
# WAVEFORM + PSD (script 01)
# =============================================================================

def plot_event_waveforms(st_proc, event, t_start, run_dir,
                         normalize='individual',
                         freqmin=None, freqmax=None, st_psd=None):
    """
    Produce one figure per event: waveform (left) + PSD (right) per station

    Layout
    ------
    One row per station; left panel = bandpass-filtered seismogram
                         right panel = Power Spectral Density (Welch, log-frequency scale)

    Parameters
    ----------
    st_proc   : ObsPy Stream — bandpass-filtered, used for the waveform panel
    event     : ObsPy Event object
    t_start   : UTCDateTime — start of the time window (time axis reference)
    run_dir   : str — output directory where the figure is saved
    normalize : 'individual' (each trace scaled to its own max) or
                'common' (all traces divided by the global max — amplitudes comparable)
    freqmin, freqmax : float or None — bandpass limits; drawn as reference lines on the PSD
    st_psd    : ObsPy Stream or None — unfiltered stream (demean+detrend+taper only) used for the PSD to show the true full spectrum
    """
    from catalog_helpers import get_pick_times

    origin   = event.preferred_origin() or event.origins[0]
    t_origin = origin.time
    picks    = get_pick_times(event)
    etype    = str(event.event_type) if event.event_type else "unknown"
    mag      = event.preferred_magnitude()
    mag_str  = f"M{mag.mag:.1f}" if mag else "M?"
    n        = len(st_proc)

    if n == 0:
        print("    [SKIP] No traces to plot.")
        return

    global_max = max(np.max(np.abs(tr.data)) for tr in st_proc) or 1.0

    fig, axes = plt.subplots(
        n, 2,
        figsize=(18, max(4, n * 2.2)),
        gridspec_kw={'width_ratios': [3, 1]},
        sharey=False
    )
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f"{etype.upper()}   {mag_str}   |   {t_origin}\n"
        f"lat={origin.latitude:.3f}°   lon={origin.longitude:.3f}°   "
        f"depth={origin.depth/1000:.1f} km   |   {n} stations   |   "
        f"normalization: {normalize}",
        fontsize=15, fontweight='bold', y=1.02
    )

    for row, (ax_row, tr) in enumerate(zip(axes, st_proc)):
        ax_wave = ax_row[0]
        ax_psd  = ax_row[1]

        data  = tr.data.astype(float)
        fs    = tr.stats.sampling_rate
        times = tr.times(reftime=t_start)

        # -- Waveform panel ---------------------------------------------------
        amp = global_max if normalize == 'common' else (np.max(np.abs(data)) or 1.0)

        ax_wave.plot(times, data / amp, 'k-', linewidth=0.7)
        ax_wave.set_ylim(-1.5, 1.5)
        ax_wave.set_yticks([-1, 0, 1])
        ax_wave.set_yticklabels(['-1', '0', '1'], fontsize=9, color='grey')
        ax_wave.axhline(0, color='lightgrey', linewidth=0.5)
        ax_wave.axvline(t_origin - t_start, color='grey', linestyle='--', linewidth=1.2)

        if normalize == 'common':
            local_peak = np.max(np.abs(data)) / global_max
            ax_wave.set_ylabel(
                f"{tr.stats.network}.{tr.stats.station}\n({local_peak:.2f}×max)",
                fontsize=11, fontweight='bold', rotation=0, labelpad=80, va='center'
            )
        else:
            ax_wave.set_ylabel(
                f"{tr.stats.network}.{tr.stats.station}",
                fontsize=12, fontweight='bold', rotation=0, labelpad=60, va='center'
            )

        # P and S pick markers
        sta = tr.stats.station
        if sta in picks:
            if picks[sta]['P']:
                t_P = picks[sta]['P'] - t_start
                ax_wave.axvline(t_P, color='red', linewidth=1.5)
                ax_wave.text(t_P + 0.3, 1.2, 'P', color='red',
                             fontsize=12, fontweight='bold', va='top')
            if picks[sta]['S']:
                t_S = picks[sta]['S'] - t_start
                ax_wave.axvline(t_S, color='blue', linewidth=1.5)
                ax_wave.text(t_S + 0.3, 1.2, 'S', color='blue',
                             fontsize=12, fontweight='bold', va='top')

        # -- PSD panel --------------------------------------------------------
        if st_psd is not None:
            psd_tr = next((t for t in st_psd if t.stats.station == tr.stats.station), tr)
        else:
            psd_tr = tr
        psd_data   = psd_tr.data.astype(float)
        nperseg    = min(int(10 * fs), len(psd_data) // 4)
        freqs, psd = welch(psd_data, fs=fs, nperseg=nperseg)
        psd_db     = 10 * np.log10(psd + 1e-30)   # +1e-30 avoids log(0)

        ax_psd.plot(freqs, psd_db, color='steelblue', linewidth=1.0)
        ax_psd.set_xlim(0.5, fs / 2)
        ax_psd.set_xscale('log')
        ax_psd.set_facecolor('#f5f8fc')
        ax_psd.tick_params(axis='x', labelsize=10)
        ax_psd.tick_params(axis='y', labelsize=9)
        ax_psd.set_ylabel("Power (dB)", fontsize=10)

        if freqmin:
            ax_psd.axvline(freqmin, color='orange', linestyle='--',
                           linewidth=1.2, alpha=0.9)
        if freqmax:
            ax_psd.axvline(freqmax, color='orange', linestyle='--',
                           linewidth=1.2, alpha=0.9)

        if row == 0:
            ax_psd.set_title("Power Spectral\nDensity", fontsize=12, fontweight='bold')

    axes[-1][0].set_xlabel("Time (s) relative to event start", fontsize=14, fontweight='bold')
    axes[-1][1].set_xlabel("Frequency (Hz)", fontsize=12, fontweight='bold')

    legend_elements = [
        Line2D([0], [0], color='grey',   linestyle='--', linewidth=1.5, label='Origin time'),
        Line2D([0], [0], color='red',    linewidth=1.5,  label='P pick (compressional wave)'),
        Line2D([0], [0], color='blue',   linewidth=1.5,  label='S pick (shear wave)'),
        Line2D([0], [0], color='orange', linestyle='--', linewidth=1.5, label='Filter band limits'),
    ]
    axes[0][0].legend(handles=legend_elements, loc='upper right',
                      fontsize=11, framealpha=0.85, edgecolor='grey')

    plt.tight_layout()

    safe_time = str(t_origin)[:19].replace(":", "-").replace("T", "_")
    safe_type = etype.replace(" ", "_")
    out_path  = os.path.join(run_dir, f"waveform_{safe_type}_{safe_time}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [SAVED] {os.path.basename(out_path)}")



# =============================================================================
# STATION COVERAGE SUMMARY (script 01)
# =============================================================================

def plot_station_coverage(station_counts, n_stations_per_event, counts_by_type,
                          t_start_str, t_end_str, run_dir, n_events):
    """
    Save two station coverage figures to run_dir

    Figure 1 — Coverage summary (two panels):
      - Left : histogram of number of stations per event, with median line
      - Right : horizontal bar chart of the top 20 most active stations

    Figure 2 — Box plot of station count grouped by event type

    Parameters
    ----------
    station_counts       : dict (net, sta) -> int (number of events recorded) from catalog_helpers.compute_station_coverage()
    n_stations_per_event : list of int (one entry per event)
    counts_by_type       : dict event_type -> list of int (station counts per event)
    t_start_str, t_end_str : str — date strings for figure titles
    run_dir              : str — output directory
    n_events             : int — total number of events (for title)
    """
    # ---- Figure 1: histogram + top-20 bar chart --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.hist(n_stations_per_event,
            bins=range(0, max(n_stations_per_event) + 2),
            edgecolor='white', color='steelblue', align='left')
    ax.axvline(np.median(n_stations_per_event), color='red', linestyle='--',
               linewidth=2,
               label=f'Median = {np.median(n_stations_per_event):.0f} stations')
    ax.set_xlabel("Number of stations with picks", fontsize=14, fontweight='bold')
    ax.set_ylabel("Number of events", fontsize=14, fontweight='bold')
    ax.set_title("Station coverage per event", fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', labelsize=12)

    ax = axes[1]
    top20  = sorted(station_counts.items(), key=lambda x: -x[1])[:20]
    labels = [f"{net}.{sta}" for (net, sta), _ in top20]
    counts = [c for _, c in top20]
    bars   = ax.barh(range(len(labels)), counts, color='steelblue', edgecolor='white')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlabel("Number of events with picks", fontsize=14, fontweight='bold')
    ax.set_title("Top 20 most-active stations", fontsize=15, fontweight='bold')
    ax.tick_params(axis='x', labelsize=12)
    for bar, count in zip(bars, counts):
        ax.text(count + 0.3, bar.get_y() + bar.get_height() / 2,
                str(count), va='center', fontsize=10, color='navy')

    plt.suptitle(
        f"Station coverage — {n_events} events   ({t_start_str} to {t_end_str})",
        fontsize=16, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    out1 = os.path.join(run_dir, "station_coverage_summary.png")
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] station_coverage_summary.png")

    # ---- Figure 2: box plot by event type -------------------------------------
    types_sorted = sorted(counts_by_type.keys(), key=lambda t: -len(counts_by_type[t]))
    data_sorted  = [counts_by_type[t] for t in types_sorted]

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(data_sorted, labels=types_sorted, patch_artist=True,
                    medianprops=dict(color='white', linewidth=2.5))
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for i, (t, d) in enumerate(zip(types_sorted, data_sorted)):
        jitter = np.random.normal(0, 0.07, size=len(d))
        ax.scatter(np.ones(len(d)) * (i + 1) + jitter, d,
                   alpha=0.5, s=25, color='k', zorder=3)
    ax.set_xlabel("Event type", fontsize=14, fontweight='bold')
    ax.set_ylabel("Number of stations with picks", fontsize=14, fontweight='bold')
    ax.set_title(
        f"Station coverage by event type — {n_events} events   "
        f"({t_start_str} to {t_end_str})",
        fontsize=15, fontweight='bold'
    )
    ax.set_xticklabels(
        [f"{t}\n(n = {len(counts_by_type[t])} events)" for t in types_sorted],
        fontsize=13
    )
    ax.tick_params(axis='y', labelsize=12)
    plt.tight_layout()
    out2 = os.path.join(run_dir, "station_count_by_type.png")
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] station_count_by_type.png")



# =============================================================================
# WAVEFORM + STA/LTA RATIO (script 02)
# =============================================================================

def plot_sta_lta(st_proc, event, t_start, detections, cfts,
                 thres_on, thres_off, run_dir,
                 sta_s=None, lta_s=None, freqmin=None, freqmax=None):
    """
    Produce one figure per event: waveform with detection shading (left) + STA/LTA characteristic function with threshold lines (right)

    Parameters
    ----------
    st_proc    : ObsPy Stream — bandpass-filtered trace for each station
    event      : ObsPy Event object
    t_start    : UTCDateTime — start of the time window (time axis reference)
    detections : dict station_code -> list of (t_on, t_off) UTCDateTime tuples
    cfts       : dict station_code -> numpy array (STA/LTA ratio at each sample)
    thres_on   : float — STA/LTA ratio used to declare trigger ON
    thres_off  : float — STA/LTA ratio used to declare trigger OFF
    run_dir    : str — output directory
    sta_s, lta_s : float or None — STA and LTA window lengths in seconds, shown in the figure title when provided
    freqmin, freqmax : float or None — bandpass frequencies, used only for the output filename
    """
    from catalog_helpers import get_pick_times

    origin   = event.preferred_origin() or event.origins[0]
    t_origin = origin.time
    picks    = get_pick_times(event)
    etype    = str(event.event_type) if event.event_type else "unknown"
    mag      = event.preferred_magnitude()
    mag_str  = f"M{mag.mag:.1f}" if mag else "M?"
    n        = len(st_proc)

    if n == 0:
        print("    [SKIP] No traces to plot.")
        return

    fig, axes = plt.subplots(
        n, 2, figsize=(18, max(4, n * 2.5)),
        gridspec_kw={'width_ratios': [3, 2]}, sharey=False
    )
    if n == 1:
        axes = [axes]

    # Build optional STA/LTA label for the title
    sta_lta_str = ""
    if sta_s is not None and lta_s is not None:
        sta_lta_str = f"\nSTA={sta_s}s  LTA={lta_s}s  ON={thres_on}  OFF={thres_off}"

    fig.suptitle(
        f"{etype.upper()}   {mag_str}   |   {t_origin}\n"
        f"lat={origin.latitude:.3f}°  lon={origin.longitude:.3f}°  "
        f"depth={origin.depth/1000:.1f} km  |  {n} stations"
        + sta_lta_str,
        fontsize=14, fontweight='bold', y=1.02
    )

    for row, (ax_row, tr) in enumerate(zip(axes, st_proc)):
        ax_wave, ax_cft = ax_row[0], ax_row[1]
        data  = tr.data.astype(float)
        fs    = tr.stats.sampling_rate
        times = tr.times(reftime=t_start)
        amp   = np.max(np.abs(data)) or 1.0
        sta   = tr.stats.station

        # -- Waveform ---------------------------------------------------------
        ax_wave.plot(times, data / amp, 'k-', linewidth=0.7)
        ax_wave.set_ylim(-1.5, 1.5)
        ax_wave.set_yticks([-1, 0, 1])
        ax_wave.set_yticklabels(['-1', '0', '1'], fontsize=9, color='grey')
        ax_wave.axhline(0, color='lightgrey', linewidth=0.5)
        ax_wave.set_ylabel(f"{tr.stats.network}.{sta}",
                           fontsize=12, fontweight='bold',
                           rotation=0, labelpad=60, va='center')
        ax_wave.axvline(t_origin - t_start, color='grey', linestyle='--', linewidth=1.2)

        if sta in picks:
            if picks[sta]['P']:
                t_P = picks[sta]['P'] - t_start
                ax_wave.axvline(t_P, color='red', linewidth=1.5)
                ax_wave.text(t_P + 0.5, 1.2, 'P', color='red',
                             fontsize=12, fontweight='bold')
            if picks[sta]['S']:
                t_S = picks[sta]['S'] - t_start
                ax_wave.axvline(t_S, color='blue', linewidth=1.5)
                ax_wave.text(t_S + 0.5, 1.2, 'S', color='blue',
                             fontsize=12, fontweight='bold')

        for t_on, t_off in detections.get(sta, []):
            ax_wave.axvspan(t_on - t_start, t_off - t_start,
                            alpha=0.15, color='green', zorder=0)
            ax_wave.axvline(t_on  - t_start, color='green',     linewidth=1.5, alpha=0.8)
            ax_wave.axvline(t_off - t_start, color='darkgreen', linewidth=1.5, alpha=0.8)

        # -- STA/LTA ratio ----------------------------------------------------
        cft       = cfts.get(sta, np.array([]))
        cft_times = np.arange(len(cft)) / fs

        if len(cft) > 0:
            ax_cft.plot(cft_times, cft, color='steelblue', linewidth=0.8)
        ax_cft.axhline(thres_on,  color='red',    linestyle='--',
                       linewidth=1.5, label=f'ON  = {thres_on}')
        ax_cft.axhline(thres_off, color='orange', linestyle='--',
                       linewidth=1.5, label=f'OFF = {thres_off}')

        for t_on, t_off in detections.get(sta, []):
            ax_cft.axvspan(t_on - t_start, t_off - t_start,
                           alpha=0.15, color='green', zorder=0)

        ax_cft.set_xlim(times[0], times[-1])
        ax_cft.set_ylim(bottom=0)
        ax_cft.set_facecolor('#f5f8fc')
        ax_cft.tick_params(axis='both', labelsize=10)
        ax_cft.set_ylabel("STA/LTA ratio", fontsize=10)

        if row == 0:
            ax_cft.set_title("STA/LTA\nCharacteristic Function",
                              fontsize=12, fontweight='bold')
            ax_cft.legend(loc='upper right', fontsize=10, framealpha=0.85)

    axes[-1][0].set_xlabel("Time (s) relative to window start", fontsize=14, fontweight='bold')
    axes[-1][1].set_xlabel("Time (s) relative to window start", fontsize=12, fontweight='bold')

    legend_elements = [
        Line2D([0], [0], color='grey',     linestyle='--', linewidth=1.5, label='Origin time'),
        Line2D([0], [0], color='red',       linewidth=1.5, label='P pick'),
        Line2D([0], [0], color='blue',      linewidth=1.5, label='S pick'),
        Line2D([0], [0], color='green',     linewidth=1.5, label='Trigger ON'),
        Line2D([0], [0], color='darkgreen', linewidth=1.5, label='Trigger OFF'),
        Patch(facecolor='green', alpha=0.15, label='Detected event window'),
    ]
    axes[0][0].legend(handles=legend_elements, loc='upper right',
                      fontsize=10, framealpha=0.85, edgecolor='grey')

    plt.tight_layout()
    safe_time = str(t_origin)[:19].replace(":", "-").replace("T", "_")
    out_path  = os.path.join(run_dir, f"stalta_{etype.replace(' ','_')}_{safe_time}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [SAVED] {os.path.basename(out_path)}")
