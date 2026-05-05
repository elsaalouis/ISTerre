"""
visualization.py
================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : April 2026

All figure-generating functions used across the pipeline scripts:
  - plot_event_waveforms()    : waveform + PSD panels per station  (script 01)
  - plot_station_coverage()   : histogram + bar chart + box plot    (script 01)
  - plot_windowing()          : waveform + STA/LTA ratio panels     (script 02)
  - plot_station_map()        : geographic station map, colored by SNR (script 05a)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from obspy import UTCDateTime
from scipy.signal import welch


# =============================================================================
# WAVEFORM + PSD (script 01)
# =============================================================================

def plot_event_waveforms(st_proc, event, t_start, run_dir, normalize='individual', freqmin=None, freqmax=None, st_psd=None):
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
    normalize : 'individual' (each trace scaled to its own max) 
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

def plot_station_coverage(station_counts, n_stations_per_event, counts_by_type, t_start_str, t_end_str, run_dir, n_events):
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
# WAVEFORM + STA/LTA CHARACTERISTIC FUNCTION — ALL STATIONS (script 02)
# =============================================================================

def plot_windowing(station_data, t_orig, thr_on, thr_off, etype, run_dir, freq_min=1.0, freq_max=20.0, nsta=1, nlta=15):
    """
    One figure per catalog event —> all stations stacked in rows

    Layout
    ------
    One row per station:
      Left  : velocity waveform (response-removed, unfiltered):
                          - gray dashed vertical line  : catalog origin time
                          - red   vertical line + 'P'  : catalog P pick
                          - blue  vertical line + 'S'  : catalog S pick
                          - green solid/dashed border  : detected window onset / offset
                          - green (or orange) shading  : detected event window
      Right : DetecteurV3 sum_cft (bidirectional STA/LTA characteristic function) with the same detected window shading and threshold lines

    Detection window colour code
    ----------------------------
      Green      : the catalog origin time falls INSIDE the detected window -> the detector agrees with the catalog
      Orange     : the catalog origin time falls OUTSIDE the detected window
      No shading : sum_cft never reached THR_ON on this station, the event was too weak or the noise level too high

    Parameters
    ----------
    station_data : list of dicts, one per station:
        {
          'tr_vel'     : obspy.Trace  — response-removed velocity [m/s], unfiltered
          'tr_filt'    : obspyTrace — bandpass-filtered trace (1–20 Hz)
          'detections' : dict {"Det_k": [UTCDateTime t_on, UTCDateTime t_off]}
          'picks'      : dict {'P': UTCDateTime or None, 'S': UTCDateTime or None}
          't_nrj'      : list of datetime.datetime  — time axis from DetecteurV3
          'sum_cft'    : 1-D numpy array — bidirectional STA/LTA ratio
        }
    t_orig       : UTCDateTime — catalog origin time
    thr_on/off   : float — DetecteurV3 thresholds (drawn as horizontal lines)
    etype        : str   — event type label (title + output filename)
    run_dir      : str   — output directory
    freq_min/max : float — detection frequency band (shown in figure title)
    nsta / nlta  : int   — DetecteurV3 STA/LTA window sizes (shown in title)
    pre_event    : float — seconds of pre-noise loaded before origin (grey line)
    """
    n = len(station_data)
    if n == 0:
        return

    fig, axes = plt.subplots(
        n, 2,
        figsize=(18, max(4, n * 2.5)),
        gridspec_kw={'width_ratios': [3, 1]},
        sharey=False,
    )
    if n == 1:
        axes = [axes]   # ensure list-of-rows even for a single station

    fig.suptitle(
        f"{etype.upper()}   |   {str(t_orig)[:19]}\n"
        f"DetecteurV3  {freq_min}–{freq_max} Hz   "
        f"nsta={nsta}  nlta={nlta}   "
        f"thr_on={thr_on}  thr_off={thr_off}",
        fontsize=14, fontweight='bold', y=1.01,
    )

    for row_idx, (ax_row, sd) in enumerate(zip(axes, station_data)):
        ax_wave = ax_row[0]
        ax_cft  = ax_row[1]

        tr_vel     = sd['tr_vel']
        tr_filt    = sd.get('tr_filt', tr_vel)   # bandpass-filtered trace (1–20 Hz); fall back to tr_vel if absent
        detections = sd['detections']
        picks      = sd.get('picks', {})
        t_nrj      = sd.get('t_nrj', [])
        sum_cft    = sd.get('sum_cft', np.array([]))

        t_start = tr_vel.stats.starttime   # timing reference from tr_vel (identical for tr_filt)
        net     = tr_vel.stats.network
        sta     = tr_vel.stats.station

        # Both panels share the same x-axis: seconds from trace start
        t_wav   = tr_filt.times()                            # waveform samples (filtered trace)
        t_cft   = np.array([UTCDateTime(str(t)) - t_start   # sum_cft steps
                             for t in t_nrj])
        t_orig_s = t_orig - t_start                          # origin position on x-axis

        data    = tr_filt.data.astype(float)   # plot the filtered waveform (1–20 Hz), not the broadband
        data_um = data * 1e6                   # convert m/s → µm/s for a readable y-axis

        # ── Waveform panel ───────────────────────────────────────────────────
        ax_wave.plot(t_wav, data_um, 'k-', linewidth=0.5)
        ax_wave.axhline(0, color='lightgrey', linewidth=0.3, zorder=0)
        # auto y-limits with 10% headroom so picks/labels don't clip
        peak_um = np.max(np.abs(data_um)) or 1.0
        ax_wave.set_ylim(-peak_um * 1.15, peak_um * 1.15)
        ax_wave.tick_params(axis='y', labelsize=7)

        # Catalog origin time
        ax_wave.axvline(t_orig_s, color='dimgrey', linewidth=1.5,
                        linestyle='--', zorder=3)

        # Detection windows: green border (onset solid, offset dashed) + shading
        for det_key, (t_on, t_off) in detections.items():
            t_on_s  = t_on  - t_start
            t_off_s = t_off - t_start
            inside  = t_on <= t_orig <= t_off
            col     = '#2ca02c' if inside else '#ff7f0e'   # green / orange
            ax_wave.axvspan(t_on_s, t_off_s, alpha=0.20, color=col, zorder=1)
            ax_wave.axvline(t_on_s,  color=col, linewidth=1.6, alpha=0.9, zorder=3)
            ax_wave.axvline(t_off_s, color=col, linewidth=1.2, alpha=0.7,
                            linestyle='--', zorder=3)

        # P and S catalog picks
        t_p = picks.get('P')
        t_s = picks.get('S')
        label_y = peak_um * 0.92    # place letter labels near the top of the axis
        if t_p is not None:
            t_p_s = t_p - t_start
            ax_wave.axvline(t_p_s, color='red', linewidth=1.5, zorder=4)
            ax_wave.text(t_p_s + 0.5, label_y, 'P',
                         color='red', fontsize=8, fontweight='bold', va='top')
        if t_s is not None:
            t_s_s = t_s - t_start
            ax_wave.axvline(t_s_s, color='blue', linewidth=1.5, zorder=4)
            ax_wave.text(t_s_s + 0.5, label_y, 'S',
                         color='blue', fontsize=8, fontweight='bold', va='top')

        # No-detection label
        if not detections:
            ax_wave.text(
                0.99, 0.96, "NO DETECTION",
                transform=ax_wave.transAxes,
                ha='right', va='top', fontsize=7.5, color='grey',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          alpha=0.85, edgecolor='lightgrey'),
            )

        ax_wave.set_xlim(t_wav[0], t_wav[-1])
        ax_wave.set_ylabel(
            f"{net}.{sta}\nVelocity (µm/s)\n1–20 Hz",
            fontsize=10, fontweight='bold',
            rotation=0, labelpad=65, va='center',
        )

        # ── STA/LTA characteristic function panel ────────────────────────────
        if len(t_cft) > 0 and len(sum_cft) > 0:
            n_pts = min(len(t_cft), len(sum_cft))
            ax_cft.plot(t_cft[:n_pts], sum_cft[:n_pts],
                        color='steelblue', linewidth=0.8)

        # Threshold lines
        ax_cft.axhline(thr_on,  color='red',       linewidth=1.3,
                       linestyle='--', zorder=3)
        ax_cft.axhline(thr_off, color='darkorange', linewidth=1.1,
                       linestyle=':', zorder=3)

        # Same detection shading as waveform panel
        for det_key, (t_on, t_off) in detections.items():
            t_on_s  = t_on  - t_start
            t_off_s = t_off - t_start
            inside  = t_on <= t_orig <= t_off
            col     = '#2ca02c' if inside else '#ff7f0e'
            ax_cft.axvspan(t_on_s, t_off_s, alpha=0.20, color=col, zorder=1)
            ax_cft.axvline(t_on_s,  color=col, linewidth=1.4, alpha=0.8, zorder=3)
            ax_cft.axvline(t_off_s, color=col, linewidth=1.0, alpha=0.6,
                           linestyle='--', zorder=3)

        ax_cft.axvline(t_orig_s, color='dimgrey', linewidth=1.2,
                       linestyle='--', zorder=3)
        ax_cft.set_xlim(t_wav[0], t_wav[-1])
        ax_cft.set_ylim(bottom=0)
        ax_cft.set_ylabel("sum_cft", fontsize=8)
        ax_cft.tick_params(axis='both', labelsize=8)

        # Threshold legend only on the first row right panel
        if row_idx == 0:
            ax_cft.set_title("STA/LTA\nCharacteristic Function",
                              fontsize=10, fontweight='bold')
            ax_cft.legend(
                handles=[
                    Line2D([0], [0], color='red',       linestyle='--',
                           linewidth=1.3, label=f'THR_ON = {thr_on}'),
                    Line2D([0], [0], color='darkorange', linestyle=':',
                           linewidth=1.1, label=f'THR_OFF = {thr_off}'),
                ],
                loc='upper right', fontsize=8, framealpha=0.85,
            )

    axes[-1][0].set_xlabel("Time (s) relative to window start",
                            fontsize=12, fontweight='bold')
    axes[-1][1].set_xlabel("Time (s) relative to window start",
                            fontsize=10, fontweight='bold')

    # ── Main legend — waveform panel of first row ────────────────────────────
    legend_elements = [
        Line2D([0], [0], color='dimgrey', linestyle='--', linewidth=1.5,
               label='Origin time'),
        Line2D([0], [0], color='red',  linewidth=1.5, label='P pick'),
        Line2D([0], [0], color='blue', linewidth=1.5, label='S pick'),
        Line2D([0], [0], color='#2ca02c', linewidth=1.6,
               label='Trigger ON  (det. onset)'),
        Line2D([0], [0], color='#2ca02c', linewidth=1.2, linestyle='--',
               label='Trigger OFF  (det. offset)'),
        Patch(facecolor='#2ca02c', alpha=0.25,
              label='Detected window — origin INSIDE\n'
                    '(detector agrees with catalog pick)'),
        Patch(facecolor='#ff7f0e', alpha=0.25,
              label='Detected window — origin OUTSIDE\n'
                    '(emergent onset)'),
    ]
    axes[0][0].legend(
        handles=legend_elements,
        loc='upper left', fontsize=8,
        framealpha=0.92, edgecolor='grey',
        ncol=2,
    )

    plt.tight_layout()

    safe_time = str(t_orig)[:19].replace(":", "-").replace("T", "_")
    safe_type = etype.replace(" ", "_")
    fname     = f"window_{safe_type}_{safe_time}.png"
    out_path  = os.path.join(run_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [SAVED] {fname}")



# =============================================================================
# GEOGRAPHIC STATION MAP (script 05a)
# =============================================================================

def plot_station_map(ax, snr_series, sta_coords, title, vmin, vmax, map_extent, mont_blanc_lon, mont_blanc_lat,
                     cmap='YlOrRd', basemap_zoom=9):
    """
    Plot one geographic station map panel on an existing Axes
     -> one dot per station, colored by its mean SNR value

    A satellite basemap (Esri WorldImagery) with a city-label overlay (CartoDB VoyagerOnlyLabels) is added automatically via contextily 
     -> when library available and network reachable (if not: white grid background)

    Parameters
    ----------
    ax              : matplotlib.axes.Axes
    snr_series      : pd.Series — index = station code, values = mean SNR for the metric and subset
    sta_coords      : dict  {station_code: (latitude, longitude)}
    title           : str   — subplot title
    vmin, vmax      : float — shared color scale bounds
    map_extent      : tuple (lon_min, lon_max, lat_min, lat_max)
    mont_blanc_lon  : float — longitude of the Mont Blanc summit reference point
    mont_blanc_lat  : float — latitude  of the Mont Blanc summit reference point
    cmap            : str   — matplotlib colormap name (default 'YlOrRd', same as the SNR heatmap figures)
    basemap_zoom    : int   — tile zoom level for contextily: 8 = fast, 9 = city names visible, 10 = detailed

    Returns
    -------
    n_plotted : int — number of stations successfully drawn on the map
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle
    import matplotlib.patheffects as pe

    lon_min, lon_max, lat_min, lat_max = map_extent
    norm     = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = cm.get_cmap(cmap)

    # ---- Map frame ----------------------------------------------------------
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel('Longitude (°E)', fontsize=8)
    ax.set_ylabel('Latitude (°N)',  fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9, fontweight='bold')

    # ---- Satellite basemap (contextily) -------------------------------------
    _has_basemap = False
    try:
        import contextily as ctx
        # Layer 1: Esri WorldImagery, satellite photograph of the Alps
        ctx.add_basemap(ax, crs='EPSG:4326',                    # EPSG:4326 (lat/lon) so no coordinate reprojection needed
                        source=ctx.providers.Esri.WorldImagery, 
                        zoom=basemap_zoom, attribution_size=5)
        # Layer 2: CartoDB VoyagerOnlyLabels, city names / roads on top
        ctx.add_basemap(ax, crs='EPSG:4326', 
                        source=ctx.providers.CartoDB.VoyagerOnlyLabels, 
                        zoom=basemap_zoom, attribution_size=5, alpha=0.85)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        _has_basemap = True
    except ImportError:         # falls back to a white grid if contextily not installed or network unavailable
        ax.grid(True, lw=0.3, alpha=0.5, ls='--')
        ax.set_facecolor('#f0f0f0')
    except Exception as exc:
        ax.grid(True, lw=0.3, alpha=0.5, ls='--')
        ax.text(0.01, 0.01, f'Basemap unavailable ({exc})',
                transform=ax.transAxes, fontsize=5, color='grey', va='bottom')

    # Annotation colours that stay readable on both satellite and white backgrounds
    _stroke      = [pe.withStroke(linewidth=2.5, foreground='black')]
    label_color  = 'white' if _has_basemap else '#222222'
    marker_color = 'white' if _has_basemap else 'black'
    rect_color   = 'white' if _has_basemap else 'black'

    # ---- Mont Blanc summit — fixed geographic reference ---------------------
    ax.plot(mont_blanc_lon, mont_blanc_lat,
            marker='*', color=marker_color, markersize=13,
            markeredgecolor='black', markeredgewidth=0.5,
            zorder=10)
    ax.annotate('Mont Blanc',
                (mont_blanc_lon, mont_blanc_lat),
                textcoords='offset points', xytext=(5, 5),
                fontsize=7, color=marker_color, fontweight='bold',
                path_effects=_stroke if _has_basemap else [])

    # ---- Mont Blanc massif bounding box (dashed rectangle) ------------------
    massif_rect = Rectangle(
        (6.6, 45.7), width=0.7, height=0.3,
        linewidth=1.5, edgecolor=rect_color, facecolor='none',
        linestyle='--', zorder=5,
    )
    ax.add_patch(massif_rect)

    # ---- Station dots -------------------------------------------------------
    n_plotted = 0
    for sta_code, snr_val in snr_series.items():
        if sta_code not in sta_coords:
            continue
        lat, lon = sta_coords[sta_code]
        color = cmap_obj(norm(snr_val)) if not np.isnan(snr_val) else 'lightgrey'
        ax.scatter(lon, lat, s=130, color=color,
                   edgecolors='black', linewidths=0.8, zorder=6)
        ax.annotate(sta_code, (lon, lat),
                    textcoords='offset points', xytext=(4, 4),
                    fontsize=6.5, color=label_color, fontweight='bold',
                    path_effects=_stroke if _has_basemap else [])
        n_plotted += 1

    return n_plotted