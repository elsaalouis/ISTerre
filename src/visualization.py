"""
visualization.py
================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : April 2026

All figure-generating functions used across the pipeline scripts:
  - plot_event_waveforms()       : waveform + PSD panels per station  (script 01)
  - plot_station_coverage()      : histogram + bar chart + box plot    (script 01)
  - plot_windowing()             : waveform + STA/LTA ratio panels     (script 02)
  - plot_station_map()           : geographic station map, colored by SNR (script 05a)
  - plot_snr_before_after()      : paired raw-vs-denoised SNR scatter  (script 03d)
  - plot_delta_snr_distribution(): log-ratio SNR change histogram      (script 03d)
  - plot_rescue_funnel()         : sequential-stage funnel bar chart   (script 03d)
  - plot_denoise_fidelity()      : waveform correlation vs SNR gain    (script 03d)
  - plot_threshold_by_type()     : Youden threshold per event type     (script 05a)
  - plot_roc_by_type()           : ROC curves faceted by event type    (script 05a)
  - plot_waveform_comparison()   : raw vs denoised waveform, one event (script 03d)
  - plot_snr_quality_threshold() : GMM/Otsu quality-threshold overlay  (script 05b)
  - plot_roc_pooled()            : generic pooled ROC curve plot       (script 05b)
  - plot_noise_diagnostic()      : waveform + broadband spectrogram + classical
                                    STA/LTA CFT, one noise-class example (script 04e)
  - plot_event_map()             : geographic catalog-event map, colored
                                    by event type                        (script 08)
  - plot_waveform_spectrogram_example() : single-event waveform + spectrogram panel (script 08)
  - plot_average_spectrograms()  : "typical fingerprint" spectrogram per class (script 08)
  - plot_feature_distributions() : violin plots of a feature, one panel per class (script 08)
  - plot_snr_quality_by_class()  : SNR/quality violin plots with gate threshold  (script 08)
  - plot_method_comparison_windowing() : classical vs spectrogram STA/LTA, side
                                    by side on the same stations/event (script 08b)
  - plot_spectrogram_rgb_example() : one CNN training input rendered as a single
                                    R=Z/G=N/B=E composite image             (script 08c)
  - plot_gradcam_example()       : raw Z-channel spectrogram + Grad-CAM overlay,
                                    one CNN prediction                      (script 08c)

NOTE: this module has NO TensorFlow dependency, on purpose -- scripts 01-06/08a/08b
never need it installed just to import visualization.py. 08c computes Grad-CAM
itself (TF-specific: gradient tape, conv-layer activations) and only ever hands
plot_gradcam_example a plain numpy array (the already-computed CAM), never a
TF tensor or a Keras model.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from obspy import UTCDateTime
from scipy.signal import welch, butter, filtfilt, spectrogram
from scipy.interpolate import CubicSpline


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

def plot_windowing(station_data, t_orig, thr_on, thr_off, etype, run_dir, freq_min=1.0, freq_max=20.0, nsta=1, nlta=15, pre_event=150):
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
      Green      : P-pick (or S-pick if no P available) falls INSIDE the detected window -> the detector captured the wave arrival
      Orange     : the reference pick falls OUTSIDE the detected window (or no pick available at all)
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
        # Colour: green if P-pick inside the window; if no P-pick, fall back to S-pick
        p_pick   = picks.get('P')
        s_pick   = picks.get('S')
        ref_pick = p_pick if p_pick is not None else s_pick   # P preferred, S as fallback
        for det_key, (t_on, t_off) in detections.items():
            t_on_s  = t_on  - t_start
            t_off_s = t_off - t_start
            inside  = (ref_pick is not None) and (t_on <= ref_pick <= t_off)
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

        # Same detection shading as waveform panel (same ref_pick logic)
        for det_key, (t_on, t_off) in detections.items():
            t_on_s  = t_on  - t_start
            t_off_s = t_off - t_start
            inside  = (ref_pick is not None) and (t_on <= ref_pick <= t_off)
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
              label='Detected window — P pick inside\n'
                    '(S pick used as fallback if no P)'),
        Patch(facecolor='#ff7f0e', alpha=0.25,
              label='Detected window — pick outside\n'
                    '(or no pick available)'),
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
# CLASSICAL vs SPECTROGRAM STA/LTA — SIDE-BY-SIDE COMPARISON (script 08b)
# =============================================================================

def plot_method_comparison_windowing(
    station_data, t_orig, etype, run_dir,
    freq_min=1.0, freq_max=20.0,
    classical_params=(5, 100, 2.0, 1.3),   # (sta_s, lta_s, thres_on, thres_off)
    groult_params=(1, 15, 8.0, 2.0),       # (nsta, nlta, thr_on, thr_off)
):
    """
    One figure for one catalog event -> one row per station, 3 columns:
      Col 1 : velocity waveform (bandpass-filtered, freq_min-freq_max Hz),
              full trace length. Shows BOTH methods' detected windows
              overlaid directly on the same trace (classical = orange,
              spectrogram/Groult = purple), plus the catalog P/S picks, so
              the two windows can be compared at a glance against the true
              wave arrival.
      Col 2 : classical STA/LTA characteristic function (ratio of short-term
              to long-term average of the raw waveform amplitude). Kept at
              full trace length (unlike col 1) so the LTA warm-up/background
              level stays visible for context.
      Col 3 : Groult et al. (2026) spectrogram-based bidirectional STA/LTA
              characteristic function (ratio computed on a spectral-energy
              time series stacked over freq_min-freq_max Hz, not on the raw
              amplitude) -- this is the key conceptual difference between the
              two methods, made visible by putting the two CFTs side by side
              on the same time axis. Also kept at full trace length.

    Parameters
    ----------
    station_data : list of dicts, one per station:
        {
          'tr_filt'   : obspy.Trace — bandpass-filtered velocity [m/s]
                        (freq_min-freq_max Hz), used for the waveform panel
          'picks'     : dict {'P': UTCDateTime or None, 'S': UTCDateTime or None}
                        (optional; omit or leave empty if no pick available)
          'classical' : {'detections': {"Det_k": [UTCDateTime t_on, t_off]},
                         't_cft': 1-D array [s, relative to trace start],
                         'cft':   1-D array — STA/LTA ratio}
          'groult'    : same structure as 'classical', but 'cft' is the
                        DetecteurV3 sum_cft and 't_cft' its (coarser)
                        spectrogram-step time axis
        }
    t_orig            : UTCDateTime — catalog origin time
    etype             : str — event type label (title + output filename)
    run_dir           : str — output directory
    freq_min/freq_max : float — shared detection frequency band (title)
    classical_params  : (sta_s, lta_s, thres_on, thres_off)
    groult_params     : (nsta, nlta, thr_on, thr_off)
    """
    n = len(station_data)
    if n == 0:
        return

    sta_s, lta_s, thres_on, thres_off = classical_params
    nsta, nlta, thr_on, thr_off       = groult_params

    COL_CLASSICAL = '#ff7f0e'   # orange
    COL_GROULT    = '#9467bd'   # purple

    fig, axes = plt.subplots(
        n, 3,
        figsize=(20, max(4, n * 2.6)),
        gridspec_kw={'width_ratios': [3, 1.2, 1.2]},
        sharey=False,
    )
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f"{etype.upper()}   |   {str(t_orig)[:19]}\n"
        f"Classical STA/LTA (sta={sta_s}s lta={lta_s}s on={thres_on} off={thres_off})   "
        f"vs   Groult spectrogram STA/LTA ({freq_min}-{freq_max} Hz, "
        f"nsta={nsta} nlta={nlta} on={thr_on} off={thr_off})",
        fontsize=13, fontweight='bold', y=1.02,
    )

    for row_idx, (ax_row, sd) in enumerate(zip(axes, station_data)):
        ax_wave, ax_classical, ax_groult = ax_row

        tr_filt = sd['tr_filt']
        clas    = sd['classical']
        grou    = sd['groult']
        picks   = sd.get('picks', {}) or {}
        p_pick  = picks.get('P')
        s_pick  = picks.get('S')

        t_start = tr_filt.stats.starttime
        net     = tr_filt.stats.network
        sta     = tr_filt.stats.station

        t_wav    = tr_filt.times()
        t_orig_s = t_orig - t_start
        data_um  = tr_filt.data.astype(float) * 1e6   # m/s -> µm/s

        # ── Waveform panel: both methods' windows on the same trace ──────────
        ax_wave.plot(t_wav, data_um, 'k-', linewidth=0.5)
        ax_wave.axhline(0, color='lightgrey', linewidth=0.3, zorder=0)
        peak_um = np.max(np.abs(data_um)) or 1.0
        ax_wave.set_ylim(-peak_um * 1.15, peak_um * 1.15)
        ax_wave.tick_params(axis='y', labelsize=7)
        ax_wave.axvline(t_orig_s, color='dimgrey', linewidth=1.5,
                        linestyle='--', zorder=3)

        for method_key, col, y_frac in [('classical', COL_CLASSICAL, 0.92),
                                         ('groult',    COL_GROULT,    0.80)]:
            dets = sd[method_key]['detections']
            if not dets:
                ax_wave.text(0.99, y_frac, f"{method_key}: NO DETECTION",
                             transform=ax_wave.transAxes, ha='right', va='top',
                             fontsize=7, color=col, fontweight='bold')
                continue
            for det_key, (t_on, t_off) in dets.items():
                t_on_s, t_off_s = t_on - t_start, t_off - t_start
                ax_wave.axvspan(t_on_s, t_off_s, alpha=0.18, color=col, zorder=1)
                ax_wave.axvline(t_on_s,  color=col, linewidth=1.6, alpha=0.9, zorder=3)
                ax_wave.axvline(t_off_s, color=col, linewidth=1.1, alpha=0.7,
                                linestyle='--', zorder=3)

        # Catalog P/S picks (same style as plot_windowing)
        label_y = peak_um * 0.92
        if p_pick is not None:
            t_p_s = p_pick - t_start
            ax_wave.axvline(t_p_s, color='red', linewidth=1.5, zorder=4)
            ax_wave.text(t_p_s + 0.5, label_y, 'P', color='red',
                         fontsize=9, fontweight='bold', va='top')
        if s_pick is not None:
            t_s_s = s_pick - t_start
            ax_wave.axvline(t_s_s, color='blue', linewidth=1.5, zorder=4)
            ax_wave.text(t_s_s + 0.5, label_y, 'S', color='blue',
                         fontsize=9, fontweight='bold', va='top')

        ax_wave.set_xlim(t_wav[0], t_wav[-1])
        ax_wave.set_ylabel(
            f"{net}.{sta}\nVelocity (µm/s)\n{freq_min}-{freq_max} Hz",
            fontsize=10, fontweight='bold',
            rotation=0, labelpad=65, va='center',
        )

        # ── CFT panels (one per method) ───────────────────────────────────────
        for ax_cft, method_key, col, cft_thr_on, cft_thr_off, label in [
            (ax_classical, 'classical', COL_CLASSICAL, thres_on, thres_off, 'STA/LTA ratio'),
            (ax_groult,    'groult',    COL_GROULT,    thr_on,   thr_off,   'sum_cft'),
        ]:
            t_cft = sd[method_key]['t_cft']
            cft   = sd[method_key]['cft']
            if len(t_cft) > 0 and len(cft) > 0:
                n_pts = min(len(t_cft), len(cft))
                ax_cft.plot(t_cft[:n_pts], cft[:n_pts], color='steelblue', linewidth=0.8)

            ax_cft.axhline(cft_thr_on,  color='red',        linewidth=1.2, linestyle='--', zorder=3)
            ax_cft.axhline(cft_thr_off, color='darkorange', linewidth=1.0, linestyle=':',  zorder=3)

            for det_key, (t_on, t_off) in sd[method_key]['detections'].items():
                t_on_s, t_off_s = t_on - t_start, t_off - t_start
                ax_cft.axvspan(t_on_s, t_off_s, alpha=0.18, color=col, zorder=1)
                ax_cft.axvline(t_on_s,  color=col, linewidth=1.3, alpha=0.8, zorder=3)
                ax_cft.axvline(t_off_s, color=col, linewidth=0.9, alpha=0.6,
                               linestyle='--', zorder=3)

            ax_cft.axvline(t_orig_s, color='dimgrey', linewidth=1.1, linestyle='--', zorder=3)
            ax_cft.set_xlim(t_wav[0], t_wav[-1])
            ax_cft.set_ylim(bottom=0)
            ax_cft.set_ylabel(label, fontsize=8)
            ax_cft.tick_params(axis='both', labelsize=8)

            if row_idx == 0:
                title = "Classical STA/LTA\n(raw amplitude)" if method_key == 'classical' \
                        else "Groult spectrogram STA/LTA\n(spectral energy)"
                ax_cft.set_title(title, fontsize=10, fontweight='bold', color=col)
                ax_cft.legend(
                    handles=[
                        Line2D([0], [0], color='red', linestyle='--', linewidth=1.2,
                               label=f'ON = {cft_thr_on}'),
                        Line2D([0], [0], color='darkorange', linestyle=':', linewidth=1.0,
                               label=f'OFF = {cft_thr_off}'),
                    ],
                    loc='upper right', fontsize=7.5, framealpha=0.85,
                )

    for ax in axes[-1]:
        ax.set_xlabel("Time (s) relative to window start", fontsize=10, fontweight='bold')

    legend_elements = [
        Line2D([0], [0], color='dimgrey', linestyle='--', linewidth=1.5, label='Origin time'),
        Line2D([0], [0], color='red',  linewidth=1.5, label='P pick'),
        Line2D([0], [0], color='blue', linewidth=1.5, label='S pick'),
        Patch(facecolor=COL_CLASSICAL, alpha=0.30, label='Classical STA/LTA window'),
        Patch(facecolor=COL_GROULT,    alpha=0.30, label='Groult spectrogram STA/LTA window'),
    ]
    axes[0][0].legend(handles=legend_elements, loc='upper left', fontsize=8,
                      framealpha=0.92, edgecolor='grey', ncol=2)

    plt.tight_layout()

    safe_time = str(t_orig)[:19].replace(":", "-").replace("T", "_")
    safe_type = etype.replace(" ", "_")
    fname     = f"method_comparison_{safe_type}_{safe_time}.png"
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
    import matplotlib as mpl
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle
    import matplotlib.patheffects as pe

    lon_min, lon_max, lat_min, lat_max = map_extent
    norm     = mcolors.Normalize(vmin=vmin, vmax=vmax)
    # matplotlib.cm.get_cmap() was removed in newer matplotlib (deprecated since
    # 3.7, gone in 3.9+) -- matplotlib.colormaps[name] is the current API.
    cmap_obj = mpl.colormaps[cmap]

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
    except ImportError as exc:   # contextily not installed in this environment
        print(f"    [WARN] Basemap skipped -- contextily not installed ({exc}). "
              f"Falling back to a white grid.")
        ax.grid(True, lw=0.3, alpha=0.5, ls='--')
        ax.set_facecolor('#f0f0f0')
        ax.text(0.01, 0.01, 'Basemap unavailable (contextily not installed)',
                transform=ax.transAxes, fontsize=5, color='grey', va='bottom')
    except Exception as exc:     # network/proxy/SSL error reaching the tile servers
        print(f"    [WARN] Basemap skipped -- contextily raised: {exc}. "
              f"Falling back to a white grid.")
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



# =============================================================================
# DENOISER RESCUE QUALITY DIAGNOSTICS (script 03d)
# =============================================================================
# Signal-quality plots only — did DeepDenoiser genuinely help, and is it recovering
# real signal rather than inventing structure? Classification impact is script 06c.

def plot_snr_before_after(df, metric_pairs, thresholds, rescued_col, run_dir, stamp, event_type=""):
    """
    Paired before/after SNR scatter — one panel per metric

    For each (before_col, after_col) pair: one point per rescue candidate,
    x = SNR computed on the raw (pre-denoiser) waveform, y = SNR computed on the
    DeepDenoiser output. The y=x diagonal marks "no change"; points above it improved.
    Threshold lines mark the quality gate on both axes, splitting each panel into four
    zones (still fails / newly passes / already passed / regressed). Points are colored
    by whether the *overall* quality gate (all metrics together) passes after denoising.

    Parameters
    ----------
    df           : pd.DataFrame — one row per rescue candidate
    metric_pairs : list of (before_col, after_col, label) — label is used for the
                   panel title and to look up the threshold in `thresholds`
    thresholds   : dict {label: float} — quality-gate threshold for that metric
    rescued_col  : str — boolean column in df, True if the candidate passes the full
                   gate after denoising
    run_dir      : str — output directory
    stamp        : str — run timestamp, used in the output filename
    event_type   : str — event class label, shown in the figure title

    Returns
    -------
    out_path : str — path to the saved figure
    """
    n_panels = len(metric_pairs)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    rescued_all = df[rescued_col].astype(bool).to_numpy()

    for ax, (before_col, after_col, label) in zip(axes, metric_pairs):
        x = df[before_col].to_numpy(dtype=float)
        y = df[after_col].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        x, y, resc = x[valid], y[valid], rescued_all[valid]

        if len(x) == 0:
            ax.text(0.5, 0.5, "No valid data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label, fontsize=13, fontweight='bold')
            continue

        ax.scatter(x[~resc], y[~resc], s=22, color='lightgrey', edgecolors='grey',
                   linewidths=0.4, alpha=0.7, label='Still fails gate', zorder=2)
        ax.scatter(x[resc], y[resc], s=28, color='#2ca02c', edgecolors='black',
                   linewidths=0.4, alpha=0.85, label='Passes gate (rescued)', zorder=3)

        lims = [min(x.min(), y.min()) * 0.7, max(x.max(), y.max()) * 1.3]
        ax.plot(lims, lims, color='k', linestyle='--', linewidth=1.0, alpha=0.6, zorder=1,
                label='No change (y = x)')

        thr = thresholds.get(label)
        if thr is not None:
            ax.axhline(thr, color='red', linestyle=':', linewidth=1.3, zorder=1)
            ax.axvline(thr, color='red', linestyle=':', linewidth=1.3, zorder=1,
                       label=f'Quality gate ({thr:g})')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"{label} — raw (pre-denoiser)", fontsize=12, fontweight='bold')
        ax.set_ylabel(f"{label} — denoised", fontsize=12, fontweight='bold')
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.grid(True, which='both', alpha=0.25)

        n_above = int((y > x).sum())
        ax.text(0.02, 0.98, f"{n_above}/{len(x)} improved ({100*n_above/len(x):.0f}%)",
                transform=ax.transAxes, fontsize=10, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='lightgrey'))

    # Guard: if every panel hit the "no valid data" branch, axes[0] has no labeled
    # artists and .legend() would raise a harmless-but-noisy UserWarning.
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(loc='lower right', fontsize=9, framealpha=0.9)
    fig.suptitle(f"SNR before vs. after DeepDenoiser — {event_type}".strip(" —"),
                 fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()

    fname = f"snr_before_after_{stamp}.png"
    out_path = os.path.join(run_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {fname}")
    return out_path


def plot_delta_snr_distribution(df, metric_pairs, run_dir, stamp, event_type=""):
    """
    Histogram of the log10 SNR ratio (after / before) for each metric

    Summarizes the population-level effect of denoising in one number per event:
    0 = no change, positive = improved, negative = degraded. Log scale because SNR
    is a ratio quantity spanning orders of magnitude — an arithmetic difference would
    be dominated by whichever events happen to have the largest raw SNR.

    Parameters
    ----------
    df           : pd.DataFrame — one row per rescue candidate
    metric_pairs : list of (before_col, after_col, label)
    run_dir, stamp, event_type : see plot_snr_before_after

    Returns
    -------
    out_path : str
    """
    n_panels = len(metric_pairs)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, (before_col, after_col, label) in zip(axes, metric_pairs):
        x = df[before_col].to_numpy(dtype=float)
        y = df[after_col].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        ratio = np.log10(y[valid] / x[valid])

        if len(ratio) == 0:
            ax.text(0.5, 0.5, "No valid data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label, fontsize=13, fontweight='bold')
            continue

        ax.hist(ratio, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
        ax.axvline(0, color='k', linestyle='--', linewidth=1.2, label='No change')
        med = float(np.median(ratio))
        ax.axvline(med, color='#d62728', linestyle='-', linewidth=1.5,
                   label=f'Median = {med:+.2f}')
        ax.set_xlabel(f"log$_{{10}}$({label} after / before)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Number of events", fontsize=12, fontweight='bold')
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.tick_params(labelsize=10)

    fig.suptitle(f"Distribution of SNR change after DeepDenoiser — {event_type}".strip(" —"),
                 fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()

    fname = f"snr_delta_distribution_{stamp}.png"
    out_path = os.path.join(run_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {fname}")
    return out_path


def plot_rescue_funnel(counts, run_dir, stamp, title="Rescue funnel", fname=None):
    """
    Horizontal bar chart showing how a population narrows through sequential stages
     -> e.g. denoised candidates -> passed quality gate

    Generic on purpose (any ordered dict of stage -> count), so it can be reused for
    other sequential-narrowing summaries, not just the 03d rescue funnel.

    Parameters
    ----------
    counts  : dict — ordered {stage_label: count}, first entry should be the largest
                     (starting) population
    run_dir : str  — output directory
    stamp   : str  — run timestamp, used in the default output filename
    title   : str  — figure title
    fname   : str or None — output filename; defaults to f"rescue_funnel_{stamp}.png"

    Returns
    -------
    out_path : str
    """
    labels = list(counts.keys())
    values = list(counts.values())
    total  = values[0] if values else 0
    colors = plt.cm.Blues(np.linspace(0.85, 0.4, max(len(values), 1)))

    fig, ax = plt.subplots(figsize=(9, 1.2 + 1.1 * max(len(values), 1)))
    bars = ax.barh(range(len(values)), values, color=colors, edgecolor='white', height=0.6)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels, fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlabel("Count", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', labelsize=11)

    for bar, val in zip(bars, values):
        pct = 100 * val / total if total > 0 else 0
        ax.text(val + max(values, default=0) * 0.015, bar.get_y() + bar.get_height() / 2,
                f"{val:,}  ({pct:.1f}%)", va='center', fontsize=11, color='navy')

    ax.set_xlim(0, max(values, default=1) * 1.22)
    plt.tight_layout()

    fname = fname or f"rescue_funnel_{stamp}.png"
    out_path = os.path.join(run_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {os.path.basename(out_path)}")
    return out_path


def plot_denoise_fidelity(df, corr_col, snr_before_col, snr_after_col, rescued_col,
                          run_dir, stamp, event_type="", noise_corr_col=None):
    """
    Two-panel sanity check on whether DeepDenoiser recovered genuine signal structure
    or just invented smooth-looking content while boosting SNR

    Left panel  : scatter of waveform correlation (raw vs. denoised, signal window)
                  against log10 SNR gain — a real improvement should sit in the
                  upper-right (high correlation AND higher SNR); high SNR gain with
                  near-zero correlation is the signature of hallucinated structure
                  rather than recovered signal.
    Right panel : histograms of the signal-window correlation vs. the noise-window
                  correlation across all candidates — a well-behaved denoiser keeps
                  the signal-window correlation noticeably higher than the noise-window
                  one (noise successfully suppressed, not reproduced).

    Parameters
    ----------
    df              : pd.DataFrame — one row per rescue candidate
    corr_col        : str — column with the signal-window correlation
                      (see detection.compute_denoise_correlation)
    snr_before_col, snr_after_col : str — SNR columns used for the x-axis SNR gain
    rescued_col     : str — boolean column, True if the candidate passes the gate
                      after denoising
    run_dir, stamp, event_type : see plot_snr_before_after
    noise_corr_col  : str or None — column with the noise-window correlation; if
                      given, overlaid on the right panel

    Returns
    -------
    out_path : str
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    x_before = df[snr_before_col].to_numpy(dtype=float)
    x_after  = df[snr_after_col].to_numpy(dtype=float)
    corr     = df[corr_col].to_numpy(dtype=float)
    rescued  = df[rescued_col].astype(bool).to_numpy()

    valid  = np.isfinite(x_before) & np.isfinite(x_after) & np.isfinite(corr) & (x_before > 0) & (x_after > 0)
    gain   = np.log10(x_after[valid] / x_before[valid])
    corr_v = corr[valid]
    resc_v = rescued[valid]

    if len(gain) == 0:
        ax1.text(0.5, 0.5, "No valid data", ha='center', va='center', transform=ax1.transAxes)
    else:
        ax1.scatter(corr_v[~resc_v], gain[~resc_v], s=22, color='lightgrey', edgecolors='grey',
                   linewidths=0.4, alpha=0.7, label='Still fails gate')
        ax1.scatter(corr_v[resc_v], gain[resc_v], s=28, color='#2ca02c', edgecolors='black',
                   linewidths=0.4, alpha=0.85, label='Passes gate (rescued)')
        ax1.axhline(0, color='k', linestyle='--', linewidth=1.0, alpha=0.6)
        ax1.axvspan(-1.05, 0.15, color='red', alpha=0.06, zorder=0)
        ax1.text(0.02, 0.02, "low-correlation zone\n(possible hallucination)",
                 transform=ax1.transAxes, fontsize=8.5, color='#a33', va='bottom', ha='left')
        ax1.legend(fontsize=9, loc='upper left')
    ax1.set_xlim(-1.05, 1.05)
    ax1.set_xlabel("Correlation (raw vs. denoised, signal window)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("log$_{10}$(SNR after / before)", fontsize=11, fontweight='bold')
    ax1.set_title("Waveform fidelity vs. SNR gain", fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.25)

    corr_sig_all = corr[np.isfinite(corr)]
    if len(corr_sig_all) > 0:
        ax2.hist(corr_sig_all, bins=30, range=(-1, 1), color='#2ca02c', alpha=0.6,
                edgecolor='white', label='Signal window')
    if noise_corr_col is not None and noise_corr_col in df.columns:
        noise_corr = df[noise_corr_col].to_numpy(dtype=float)
        noise_corr = noise_corr[np.isfinite(noise_corr)]
        if len(noise_corr) > 0:
            ax2.hist(noise_corr, bins=30, range=(-1, 1), color='grey', alpha=0.5,
                    edgecolor='white', label='Noise window')
    ax2.set_xlabel("Correlation (raw vs. denoised)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Number of events", fontsize=12, fontweight='bold')
    ax2.set_title("Signal vs. noise window correlation", fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    fig.suptitle(f"DeepDenoiser fidelity check — {event_type}".strip(" —"),
                fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()

    fname = f"denoise_fidelity_{stamp}.png"
    out_path = os.path.join(run_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {fname}")
    return out_path



# =============================================================================
# PER-EVENT-TYPE SNR THRESHOLDS (script 05a)
# =============================================================================
# Is one pooled/global SNR threshold a reasonable compromise across classes, or
# does each event type really need its own?

def plot_threshold_by_type(df_roc_by_type, metrics, metric_labels, run_dir, stamp):
    """
    Small-multiples bar chart — one panel per SNR metric, one bar per event type

    Bar height = Youden-optimal threshold for that (metric, event_type) pair, from a
    ROC analysis run separately on each event type's subset. A black dashed line marks
    the pooled/global threshold (computed across all event types together — the value
    the pipeline's quality gate currently uses everywhere), so it's immediately visible
    whether one global threshold is a reasonable compromise or some classes are being
    over/under-filtered by it. Each bar is annotated with its own AUC — a per-type
    threshold computed on a poorly-discriminating subset (low AUC) shouldn't be trusted
    the same as one from a well-separated class; see plot_roc_by_type() for that check.

    Parameters
    ----------
    df_roc_by_type : pd.DataFrame — tidy table, one row per (event_type, metric), with
                     columns: event_type, metric, best_threshold, auc, pooled_threshold
                     (see 05a Section 3.3b)
    metrics        : list of str — metric names, in display order
    metric_labels  : dict {metric: label} — used for panel titles
    run_dir        : str — output directory
    stamp          : str — run timestamp, used in the output filename

    Returns
    -------
    out_path : str
    """
    event_types = sorted(df_roc_by_type['event_type'].dropna().unique())
    n_types     = len(event_types)
    n_m         = len(metrics)
    colors      = plt.cm.tab10.colors

    ncols = min(4, n_m)
    nrows = int(np.ceil(n_m / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 4.3 * nrows))
    axes = np.array(axes).reshape(-1)

    for k, metric in enumerate(metrics):
        ax  = axes[k]
        sub = df_roc_by_type[df_roc_by_type['metric'] == metric].set_index('event_type')

        heights = [sub['best_threshold'].get(et, np.nan) for et in event_types]
        aucs    = [sub['auc'].get(et, np.nan) for et in event_types]
        ax.bar(range(n_types), heights,
               color=[colors[i % 10] for i in range(n_types)], edgecolor='white')

        for i, (h, a) in enumerate(zip(heights, aucs)):
            if np.isnan(h):
                continue
            label = f"{h:.2f}" + (f"\nAUC={a:.2f}" if not np.isnan(a) else "")
            ax.text(i, h, label, ha='center', va='bottom', fontsize=7.5)

        pooled_vals = sub['pooled_threshold'].dropna()
        if len(pooled_vals) > 0:
            pooled_thr = float(pooled_vals.iloc[0])
            ax.axhline(pooled_thr, color='black', linestyle='--', linewidth=1.3,
                       label=f'pooled threshold = {pooled_thr:.2f}')

        ax.set_xticks(range(n_types))
        ax.set_xticklabels(event_types, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Youden-optimal threshold', fontsize=8)
        ax.set_title(metric_labels.get(metric, metric).replace('\n', ' '), fontsize=9, fontweight='bold')
        ax.legend(fontsize=7)
        ax.tick_params(axis='y', labelsize=8)
        # headroom so bar-top annotations don't clip
        finite_h = [h for h in heights if not np.isnan(h)]
        if finite_h:
            ax.set_ylim(0, max(finite_h + [pooled_thr if len(pooled_vals) > 0 else 0]) * 1.35)

    for idx in range(n_m, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        'Best SNR threshold per event type (Youden J on a per-type ROC)\n'
        'Dashed line = current pooled/global threshold   |   bar label = threshold (AUC below it)',
        fontsize=11, fontweight='bold', y=1.02,
    )
    plt.tight_layout()

    fname = f"fig_threshold_by_type_{stamp}.png"
    out_path = os.path.join(run_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [SAVED] {fname}")
    return out_path


def plot_roc_by_type(roc_results_by_type, metrics, metric_labels, run_dir, stamp):
    """
    ROC curves faceted by event type — one panel per event type, one curve per metric

    Complements plot_threshold_by_type(): a per-type threshold is only meaningful if
    that type's ROC curve actually rises well above the diagonal (AUC well above 0.5).
    Lets you see at a glance whether a class has enough separable signal for its own
    threshold to be trustworthy, rather than just reading off a number.

    Parameters
    ----------
    roc_results_by_type : dict {event_type: {metric: {fpr, tpr, auc, youden_fpr,
                           youden_tpr, ...}}}  (see 05a Section 3.3b)
    metrics       : list of str — metric names, in display order
    metric_labels : dict {metric: label} — used in the legend
    run_dir       : str — output directory
    stamp         : str — run timestamp, used in the output filename

    Returns
    -------
    out_path : str
    """
    event_types = sorted(roc_results_by_type.keys())
    n_types     = len(event_types)
    colors      = plt.cm.tab10.colors

    ncols = min(3, max(n_types, 1))
    nrows = int(np.ceil(n_types / ncols)) if n_types else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, et in enumerate(event_types):
        ax = axes[i]
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Random (AUC=0.50)')
        results = roc_results_by_type[et]
        for k, metric in enumerate(metrics):
            r = results.get(metric)
            if r is None:
                continue
            lbl = f"{metric_labels.get(metric, metric).replace(chr(10), ' ')}  (AUC={r['auc']:.3f})"
            ax.plot(r['fpr'], r['tpr'], lw=1.8, color=colors[k % 10], label=lbl)
            ax.scatter(r['youden_fpr'], r['youden_tpr'], color=colors[k % 10],
                       s=55, zorder=5, marker='D', edgecolors='black', linewidths=0.4)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel('False Positive Rate', fontsize=9)
        ax.set_ylabel('True Positive Rate', fontsize=9)
        ax.set_title(et, fontsize=11, fontweight='bold')
        ax.legend(fontsize=6.5, loc='lower right')
        ax.grid(True, lw=0.4, alpha=0.4)

    for idx in range(n_types, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('ROC curves per event type — diamond = Youden-optimal threshold',
                fontsize=12, fontweight='bold')
    plt.tight_layout()

    fname = f"fig_roc_by_type_{stamp}.png"
    out_path = os.path.join(run_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [SAVED] {fname}")
    return out_path



# =============================================================================
# SINGLE-EVENT WAVEFORM COMPARISON (script 03d)
# =============================================================================
# Spot-check individual rescue candidates — complements the aggregate QC plots
# (plot_snr_before_after, plot_delta_snr_distribution, plot_denoise_fidelity),
# which show population-level trends but never show what a single waveform
# actually looks like before/after denoising.

def _bandpass_for_display(sig, sps, freqmin=None, freqmax=None, order=4):
    """
    Zero-phase Butterworth filter — for VISUALIZATION only. Never used upstream
    of SNR/feature computation, which must stay on the unfiltered signal.

    freqmin/freqmax follow ObsPy's convention: both given -> bandpass, only one
    given -> high-/low-pass, neither -> signal returned unchanged.
    """
    if not freqmin and not freqmax:
        return sig
    nyq = 0.5 * sps
    if freqmin and freqmax:
        b, a = butter(order, [freqmin / nyq, min(freqmax / nyq, 0.999)], btype='band')
    elif freqmax:
        b, a = butter(order, min(freqmax / nyq, 0.999), btype='low')
    else:
        b, a = butter(order, freqmin / nyq, btype='high')
    return filtfilt(b, a, sig)


def plot_waveform_comparison(raw_signal, denoised_signal, itp, sps, run_dir, stamp,
                              fname_tag, event_type="", snr_before=None, snr_after=None,
                              metric_label="", freqmin=None, freqmax=None):
    """
    Two-panel time-domain plot for one rescue candidate: raw (pre-denoiser) on
    top, DeepDenoiser output below, sharing a time axis with the P-onset marked

    Panels use independent y-scales (denoised amplitude is often much smaller
    than raw once noise is suppressed) — read amplitude within each panel, not
    across panels.

    Parameters
    ----------
    raw_signal, denoised_signal : 1-D np.ndarray, same length, same sampling rate
    itp            : int   — sample index of the P-wave onset within the window
    sps            : float — sampling rate (Hz)
    run_dir, stamp : str — output directory / run timestamp, used in the filename
    fname_tag      : str — candidate identifier (e.g. the rescue .npz basename),
                     used in the title and to make the output filename unique
    event_type     : str — shown in the title
    snr_before, snr_after : float or None — annotated in the title if both given
    metric_label   : str — name of the SNR metric used for snr_before/after
    freqmin, freqmax : float or None — optional display-only bandpass (Hz),
                     applied to BOTH panels identically before plotting; SNR
                     values passed in via snr_before/snr_after are untouched
                     (they were computed upstream on the unfiltered signal)

    Returns
    -------
    out_path : str
    """
    raw_plot      = _bandpass_for_display(raw_signal,      sps, freqmin, freqmax)
    denoised_plot = _bandpass_for_display(denoised_signal, sps, freqmin, freqmax)

    t       = np.arange(len(raw_signal)) / sps
    t_onset = itp / sps

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)

    panels = [
        (ax1, raw_plot,      "Raw (pre-denoiser)",       '#d62728'),
        (ax2, denoised_plot, "Denoised (DeepDenoiser)",  '#2ca02c'),
    ]
    for ax, sig, label, color in panels:
        ax.plot(t, sig, lw=0.7, color=color)
        ax.axvline(t_onset, color='k', linestyle='--', linewidth=1.2, alpha=0.8)
        ax.set_ylabel("Amplitude", fontsize=10, fontweight='bold')
        ax.set_title(label, fontsize=11, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.25)
        ax.margins(x=0)

    ax1.text(t_onset, ax1.get_ylim()[1] * 0.9, ' onset', fontsize=8, va='top')
    ax2.set_xlabel("Time (s)", fontsize=10, fontweight='bold')

    snr_str = ""
    if (snr_before is not None and snr_after is not None
            and np.isfinite(snr_before) and np.isfinite(snr_after)):
        snr_str = f"   |   {metric_label}: {snr_before:.2f} → {snr_after:.2f}"

    filt_str = ""
    if freqmin or freqmax:
        lo  = f"{freqmin:g}" if freqmin else "0"
        hi  = f"{freqmax:g}" if freqmax else "Nyquist"
        filt_str = f"   |   bandpass {lo}-{hi} Hz (display only)"

    fig.suptitle(f"Waveform comparison — {event_type}   ({fname_tag}){snr_str}{filt_str}".strip(),
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    safe_tag  = fname_tag.replace('/', '_').replace('\\', '_')
    fname_out = f"waveform_compare_{safe_tag}_{stamp}.png"
    out_path  = os.path.join(run_dir, fname_out)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [SAVED] {fname_out}")
    return out_path



# =============================================================================
# SNR QUALITY-THRESHOLD DIAGNOSTICS (script 05b)
# =============================================================================
# Distinct from plot_threshold_by_type/plot_roc_by_type above (script 05a), which
# show the threshold that best predicts DETECTOR/WINDOW ALIGNMENT. These two
# functions show candidate thresholds for actual SIGNAL QUALITY / downstream
# usefulness — see 05b_snr_quality_threshold.py for how they're computed.

def _gaussian_pdf(x, mean, std):
    """Standard normal pdf, evaluated manually (no scipy.stats dependency here)."""
    std = max(std, 1e-9)
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


def plot_snr_quality_threshold(df, metrics, metric_labels, gmm_params, thresholds,
                                run_dir, stamp, event_type=""):
    """
    Histogram of log10(SNR) per metric — restricted to well-aligned detections —
    with candidate "signal quality" thresholds overlaid

    Two independent, label-free methods are shown together so you can see whether
    they agree: a 2-component Gaussian mixture fit (the two components are drawn
    individually plus their sum) and an Otsu threshold (maximizes between-class
    variance on the histogram). The current pooled 05a windowing-validation
    threshold is overlaid too, for reference only — it answers a different
    question (see 05a's docstring) and is not expected to land in the same place.

    Parameters
    ----------
    df             : pd.DataFrame — one row per detection, already restricted to
                     well-aligned rows (e.g. pick_inside_det == True)
    metrics        : list of str — SNR columns to plot, one panel each
    metric_labels  : dict {metric: label} — panel titles
    gmm_params     : dict {metric: {'means': (m0, m1), 'stds': (s0, s1),
                     'weights': (w0, w1)}} in log10(SNR) space, or {} / missing
                     key if the fit failed for that metric (panel just skips the
                     component curves)
    thresholds     : dict {metric: {'GMM crossover': float or None,
                     'Otsu': float or None, '05a pooled (windowing)': float or None}}
                     — linear SNR units; any None entry is not drawn
    run_dir, stamp, event_type : see other plot_* functions in this module

    Returns
    -------
    out_path : str
    """
    n_panels = len(metrics)
    ncols = min(4, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.3 * nrows))
    axes = np.array(axes).reshape(-1)

    thr_colors = {
        'GMM crossover'          : '#2ca02c',
        'Otsu'                   : '#9467bd',
        '05a pooled (windowing)' : '#d62728',
    }

    for k, metric in enumerate(metrics):
        ax = axes[k]
        x  = df[metric].dropna()
        x  = x[x > 0]
        if len(x) < 5:
            ax.text(0.5, 0.5, "No valid data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(metric_labels.get(metric, metric), fontsize=10, fontweight='bold')
            continue

        logx = np.log10(x)
        ax.hist(logx, bins=50, density=True, color='steelblue', alpha=0.55, edgecolor='white')

        gp = gmm_params.get(metric)
        if gp:
            xs = np.linspace(logx.min(), logx.max(), 300)
            total = np.zeros_like(xs)
            for m, s, w in zip(gp['means'], gp['stds'], gp['weights']):
                comp = w * _gaussian_pdf(xs, m, s)
                total += comp
                ax.plot(xs, comp, lw=1.3, ls=':', color='grey')
            ax.plot(xs, total, lw=1.8, color='black', label='GMM fit (2 components)')

        for method, thr_val in thresholds.get(metric, {}).items():
            if thr_val is None or thr_val <= 0:
                continue
            ax.axvline(np.log10(thr_val), color=thr_colors.get(method, 'grey'),
                       linestyle='--', linewidth=1.6,
                       label=f'{method} = {thr_val:.2f}')

        ax.set_title(metric_labels.get(metric, metric), fontsize=10, fontweight='bold')
        ax.set_xlabel('log$_{10}$(SNR)', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=6.5)
        ax.tick_params(labelsize=8)

    for idx in range(n_panels, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f'SNR quality-threshold candidates (well-aligned detections only) — {event_type}'.strip(' —')
        + '\nGreen dotted = GMM components  |  compare candidate thresholds, no single one is "correct" by construction',
        fontsize=11, fontweight='bold', y=1.02,
    )
    plt.tight_layout()

    fname = f"fig_quality_threshold_{stamp}.png"
    out_path = os.path.join(run_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [SAVED] {fname}")
    return out_path


def plot_roc_pooled(roc_results, metrics, metric_labels, run_dir, stamp,
                     title, subtitle="", fname=None):
    """
    Generic pooled ROC curve plot — one curve per metric, all on shared axes

    Reusable version of 05a's inline "Figure 3" — kept generic (caller supplies
    the title/subtitle) so it works for any binary ground truth, not just
    windowing alignment. Used by 05b for the classification-correctness ROC.

    Parameters
    ----------
    roc_results   : dict {metric: {fpr, tpr, auc, youden_fpr, youden_tpr, ...}}
                    same structure as 05a's roc_results
    metrics       : list of str — metric names, in display order
    metric_labels : dict {metric: label} — used in the legend
    run_dir       : str — output directory
    stamp         : str — run timestamp, used in the default output filename
    title         : str — main figure title
    subtitle      : str — optional second title line
    fname         : str or None — output filename; defaults to f"fig_roc_{stamp}.png"

    Returns
    -------
    out_path : str
    """
    CMAP10 = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Random (AUC = 0.50)')

    for k, metric in enumerate(metrics):
        r = roc_results.get(metric)
        if r is None:
            continue
        lbl = f"{metric_labels.get(metric, metric).replace(chr(10), ' ')}  (AUC={r['auc']:.3f})"
        ax.plot(r['fpr'], r['tpr'], lw=2.2, color=CMAP10[k % 10], label=lbl)
        ax.scatter(r['youden_fpr'], r['youden_tpr'],
                   color=CMAP10[k % 10], s=70, zorder=5, marker='D',
                   edgecolors='black', linewidths=0.5)

    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    full_title = f"{title}\n{subtitle}" if subtitle else title
    ax.set_title(full_title, fontsize=10, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, lw=0.4, alpha=0.4)
    plt.tight_layout()

    fname = fname or f"fig_roc_{stamp}.png"
    out_path = os.path.join(run_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [SAVED] {fname}")
    return out_path



# =============================================================================
# CATALOG EVENT MAP (script 08)
# =============================================================================

def plot_event_map(ax, lats, lons, event_types, class_colors, map_extent,
                   mont_blanc_lon, mont_blanc_lat, cities=None, title="", sizes=None):
    """
    Plot catalog event locations on a satellite-basemap panel, colored by event_type

    Same visual style/fallback behaviour as plot_station_map() (contextily satellite
    basemap, graceful white-grid fallback if unavailable, Mont Blanc summit marker)
    but for individual event locations rather than station coordinates.

    Parameters
    ----------
    ax              : matplotlib.axes.Axes
    lats, lons      : array-like — event latitude / longitude, one entry per event
    event_types     : array-like — event_type string, one entry per event (same length)
    class_colors    : dict {event_type: matplotlib color} — also defines legend order
    map_extent      : tuple (lon_min, lon_max, lat_min, lat_max)
    mont_blanc_lon  : float — longitude of the Mont Blanc summit
    mont_blanc_lat  : float — latitude  of the Mont Blanc summit
    cities          : list of (name, lon, lat) or None — optional city labels
    title           : str — subplot title
    sizes           : array-like or None — marker size per event; default constant 18

    Returns
    -------
    n_plotted : int — number of events actually drawn (finite lat/lon only)
    """
    import matplotlib.patheffects as pe

    lon_min, lon_max, lat_min, lat_max = map_extent
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel('Longitude (°E)', fontsize=8)
    ax.set_ylabel('Latitude (°N)',  fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9, fontweight='bold')

    # ---- Satellite basemap (contextily) --------------------------------------
    _has_basemap = False
    try:
        import contextily as ctx
        ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.Esri.WorldImagery,
                        zoom=9, attribution_size=5)
        ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.CartoDB.VoyagerOnlyLabels,
                        zoom=9, attribution_size=5, alpha=0.85)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        _has_basemap = True
    except ImportError as exc:   # contextily not installed in this environment
        print(f"    [WARN] Basemap skipped -- contextily not installed ({exc}). "
              f"Falling back to a white grid.")
        ax.grid(True, lw=0.3, alpha=0.5, ls='--')
        ax.set_facecolor('#f0f0f0')
        ax.text(0.01, 0.01, 'Basemap unavailable (contextily not installed)',
                transform=ax.transAxes, fontsize=5, color='grey', va='bottom')
    except Exception as exc:     # network/proxy/SSL error reaching the tile servers
        print(f"    [WARN] Basemap skipped -- contextily raised: {exc}. "
              f"Falling back to a white grid.")
        ax.grid(True, lw=0.3, alpha=0.5, ls='--')
        ax.text(0.01, 0.01, f'Basemap unavailable ({exc})',
                transform=ax.transAxes, fontsize=5, color='grey', va='bottom')

    _stroke    = [pe.withStroke(linewidth=2, foreground='black')] if _has_basemap else []
    _txt_color = 'white' if _has_basemap else '#222222'

    # ---- City labels ----------------------------------------------------------
    if cities:
        for city, clon, clat in cities:
            if lon_min <= clon <= lon_max and lat_min <= clat <= lat_max:
                ax.plot(clon, clat, marker='o', ms=3, color='white',
                        markeredgecolor='black', markeredgewidth=0.5, zorder=6)
                ax.text(clon + 0.02, clat + 0.02, city, fontsize=6.5, color=_txt_color,
                        zorder=7, path_effects=_stroke)

    # ---- Mont Blanc summit ------------------------------------------------------
    ax.plot(mont_blanc_lon, mont_blanc_lat, marker='*',
            color='white' if _has_basemap else 'black', markersize=13,
            markeredgecolor='black', markeredgewidth=0.5, zorder=10)
    ax.annotate('Mont Blanc', (mont_blanc_lon, mont_blanc_lat), textcoords='offset points',
                xytext=(5, 5), fontsize=7, color=_txt_color, fontweight='bold',
                path_effects=_stroke, zorder=10)

    # ---- Event scatter, colored by class ----------------------------------------
    lats        = np.asarray(lats, dtype=float)
    lons        = np.asarray(lons, dtype=float)
    event_types = np.asarray(event_types)
    valid       = np.isfinite(lats) & np.isfinite(lons)
    sizes_arr   = np.full(len(lats), 18.0) if sizes is None else np.asarray(sizes, dtype=float)

    _ec = 'white' if _has_basemap else 'none'
    _lw = 0.4     if _has_basemap else 0
    n_plotted = 0
    for etype, color in class_colors.items():
        idx = valid & (event_types == etype)
        if not np.any(idx):
            continue
        ax.scatter(lons[idx], lats[idx], s=sizes_arr[idx], color=color, alpha=0.75,
                   label=f"{etype} (n={int(idx.sum())})", edgecolors=_ec, linewidths=_lw, zorder=4)
        n_plotted += int(idx.sum())

    ax.legend(fontsize=6.5, loc='lower left', facecolor='white', framealpha=0.85)
    return n_plotted



# =============================================================================
# WAVEFORM + SPECTROGRAM EXAMPLE PANEL (script 08)
# =============================================================================

def plot_waveform_spectrogram_example(times_wave, wave_data, times_spec, freq_axis, spec_db,
                                      det_duration_s, title_lines, out_path,
                                      wave_color='black', spec_vmin=-200, spec_vmax=-120,
                                      wave_units='m/s'):
    """
    One figure: bandpassed waveform (top) + broadband dB spectrogram (bottom), for
    a single example event — mirrors the layout of a typical published seismic
    event-catalog figure (amplitude trace above, spectrogram below, shaded
    detected window on both panels)

    Parameters
    ----------
    times_wave      : 1D array — seconds relative to the detection window start (0 = det_starttime)
    wave_data       : 1D array — displayed (bandpassed) ground velocity, same length as times_wave
    times_spec      : 1D array — spectrogram time bin centers, same relative-second convention
    freq_axis       : 1D array — spectrogram frequency bins [Hz]
    spec_db         : 2D array, shape (len(freq_axis), len(times_spec)) — dB-scaled power spectrogram
    det_duration_s  : float — duration of the detected window [s], shaded on both panels
    title_lines     : tuple of str (line1, line2) — e.g. (event type + date, "NET.STA | XX km from source")
    out_path        : str — full path to save the PNG
    wave_color      : str — trace color
    spec_vmin/vmax  : float — dB color scale bounds
    wave_units      : str — y-axis label units for the waveform panel

    Returns
    -------
    out_path : str
    """
    fig, (ax_wave, ax_spec) = plt.subplots(
        2, 1, figsize=(6.5, 5.5), sharex=True,
        gridspec_kw={'height_ratios': [1, 1.3]},
    )

    ax_wave.plot(times_wave, wave_data, lw=0.5, color=wave_color)
    ax_wave.axvspan(0, det_duration_s, color='grey', alpha=0.15, zorder=0)
    ax_wave.set_ylabel(f"Ground velocity ({wave_units})", fontsize=9)
    ax_wave.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax_wave.tick_params(labelsize=8)
    ax_wave.grid(True, lw=0.3, alpha=0.3)

    im = ax_spec.pcolormesh(times_spec, freq_axis, spec_db, cmap='jet',
                            vmin=spec_vmin, vmax=spec_vmax, shading='auto')
    ax_spec.axvspan(0, det_duration_s, color='white', alpha=0.12, zorder=2)
    ax_spec.set_ylabel('Frequency (Hz)', fontsize=9)
    ax_spec.set_xlabel('Time (s, 0 = detection onset)', fontsize=9)
    ax_spec.tick_params(labelsize=8)

    cbar = fig.colorbar(im, ax=[ax_wave, ax_spec], shrink=0.85, pad=0.02)
    cbar.set_label('dB', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    line1, line2 = title_lines
    fig.suptitle(f"{line1}\n{line2}", fontsize=10, fontweight='bold', y=0.98)

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path



# =============================================================================
# AVERAGE ("TYPICAL") SPECTROGRAM PER CLASS (script 08)
# =============================================================================

def plot_average_spectrograms(class_avg_db, freq_axis, times_spec, class_order,
                              run_dir, stamp, vmin=-200, vmax=-120, fig_height=4,
                              time_label='Time (s, 0 = detection onset)'):
    """
    Grid of averaged spectrograms, one panel per class — a "typical fingerprint"
    built upstream by averaging the LINEAR power spectrogram across several
    example events of that class, THEN converting the average to dB (averaging
    directly in dB would be biased low by Jensen's inequality)

    Parameters
    ----------
    class_avg_db   : dict {class_name: 2D array (n_freq, n_time), dB-scaled} — one
                     already-averaged spectrogram per class (a class may be
                     missing if no waveform could be fetched — shown as "No data")
    freq_axis      : 1D array — spectrogram frequency bins [Hz], shared across classes
    times_spec     : 1D array — spectrogram time bin centers [s], meaning set by
                     time_label below (0 = detection onset by default)
    class_order    : list of str — panel order (and which classes to attempt to show)
    run_dir, stamp : str — output location / filename stamp
    vmin, vmax     : float — shared dB color scale bounds
    fig_height     : float — figure height in inches (panel width stays 4.2 per
                     class either way) — lower it to make each panel less tall/
                     stretched-looking; default 4 matches this function's
                     original, unchanged look
    time_label     : str — x-axis label, since times_spec's zero point/convention
                     is set by the CALLER (not this function) -- default matches
                     this function's original, onset-centered behaviour; pass a
                     different label if times_spec uses another convention (e.g.
                     "0 = window start") so the axis isn't mislabeled

    Returns
    -------
    out_path : str
    """
    n = len(class_order)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, fig_height), sharey=True)
    if n == 1:
        axes = [axes]

    im = None
    for ax, cls in zip(axes, class_order):
        if cls not in class_avg_db:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(cls, fontsize=10, fontweight='bold')
            continue
        im = ax.pcolormesh(times_spec, freq_axis, class_avg_db[cls], cmap='jet',
                           vmin=vmin, vmax=vmax, shading='auto')
        ax.set_title(cls, fontsize=10, fontweight='bold')
        ax.set_xlabel(time_label, fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel('Frequency (Hz)', fontsize=9)

    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01, label='dB')

    fig.suptitle('Average spectrogram by class ("typical fingerprint")',
                 fontsize=12, fontweight='bold')

    fname = f"fig_average_spectrogram_{stamp}.png"
    out_path = os.path.join(run_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    [SAVED] {fname}")
    return out_path



# =============================================================================
# FEATURE DISTRIBUTIONS BY CLASS (script 08)
# =============================================================================

def plot_feature_distributions(df, features, feature_labels, class_col, class_order,
                               class_colors, run_dir, stamp, log_features=None):
    """
    Grid of violin plots, one panel per feature, showing how its distribution
    differs across event classes — the quantitative counterpart to the example
    waveform gallery: what actually separates the classes numerically

    Parameters
    ----------
    df             : pd.DataFrame — one row per detection, must contain `class_col`
                     and every entry of `features`
    features       : list of str — column names to plot (one panel each)
    feature_labels : dict {feature: axis label str} — human-readable label + units
    class_col      : str — column holding the class label (e.g. 'event_type')
    class_order    : list of str — x-axis category order
    class_colors   : dict {class: color}
    run_dir, stamp : str — output location / filename stamp
    log_features   : set of str or None — features to show on a log y-axis (useful
                     for heavy-tailed quantities like energy ratios)

    Returns
    -------
    out_path : str
    """
    log_features = log_features or set()
    n = len(features)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, feat in zip(axes, features):
        data_by_class = [
            df.loc[df[class_col] == cls, feat].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            for cls in class_order
        ]
        positions = np.arange(1, len(class_order) + 1)

        # violinplot's internal KDE needs >= 2 points per dataset -- a class with
        # 0 or 1 valid values (e.g. a feature that's undefined/NaN for that class
        # by construction) would crash it. Plot violins only for classes with
        # enough data, and label the rest "no data" instead.
        has_data  = np.array([len(d) >= 2 for d in data_by_class])
        ok_pos    = positions[has_data]
        ok_data   = [d for d, ok in zip(data_by_class, has_data) if ok]

        if ok_data:
            parts = ax.violinplot(ok_data, positions=ok_pos, showmedians=True, widths=0.8)
            ok_classes = [c for c, ok in zip(class_order, has_data) if ok]
            for pc, cls in zip(parts['bodies'], ok_classes):
                pc.set_facecolor(class_colors.get(cls, 'grey'))
                pc.set_alpha(0.65)
                pc.set_edgecolor('black')
                pc.set_linewidth(0.6)
            for key in ('cmedians', 'cbars', 'cmins', 'cmaxes'):
                if key in parts:
                    parts[key].set_color('black')
                    parts[key].set_linewidth(0.8)

        blended = ax.get_xaxis_transform()   # x in data coords, y in axes fraction
        for pos, ok in zip(positions, has_data):
            if not ok:
                ax.text(pos, 0.5, 'no data', ha='center', va='center', fontsize=7,
                        color='grey', transform=blended)

        ax.set_xticks(positions)
        ax.set_xticklabels(class_order, rotation=15, ha='right', fontsize=8)
        ax.set_ylabel(feature_labels.get(feat, feat), fontsize=9)
        ax.set_title(feat, fontsize=9, fontweight='bold')
        if feat in log_features:
            ax.set_yscale('log')
        ax.grid(True, axis='y', lw=0.3, alpha=0.3)

    for ax in axes[n:]:
        ax.axis('off')

    fig.suptitle('Feature distributions by class', fontsize=13, fontweight='bold')
    plt.tight_layout()

    fname = f"fig_feature_distributions_{stamp}.png"
    out_path = os.path.join(run_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    [SAVED] {fname}")
    return out_path



# =============================================================================
# SNR / DATA-QUALITY DISTRIBUTION BY CLASS (script 08)
# =============================================================================

def plot_snr_quality_by_class(df, class_col, class_order, class_colors, run_dir, stamp,
                              snr_min=None, snr_full_median_min=None,
                              metrics=('SNR', 'SNR_full_median')):
    """
    Violin plots of SNR-type metrics by class, log-scaled, with the pipeline's
    quality-gate thresholds overlaid — shows directly why some classes (typically
    ice quakes: small, high-frequency, easily attenuated) are harder to acquire
    cleanly than others

    Parameters
    ----------
    df                   : pd.DataFrame — must contain `class_col` and every entry
                           of `metrics`. Pass the UNGATED data here — gating
                           everything to the same threshold first would make the
                           whole point (why the gate matters, and unevenly so
                           across classes) invisible
    class_col            : str
    class_order          : list of str
    class_colors         : dict {class: color}
    run_dir, stamp       : str
    snr_min              : float or None — gate threshold for metrics[0], drawn as a horizontal line
    snr_full_median_min  : float or None — gate threshold for metrics[1]
    metrics              : tuple of column names to show (one panel each)

    Returns
    -------
    out_path : str
    """
    thresholds = {}
    if len(metrics) >= 1:
        thresholds[metrics[0]] = snr_min
    if len(metrics) >= 2:
        thresholds[metrics[1]] = snr_full_median_min

    fig, axes = plt.subplots(1, len(metrics), figsize=(6.5 * len(metrics), 4.5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        data_by_class = [
            df.loc[df[class_col] == cls, metric].replace([np.inf, -np.inf], np.nan).dropna()
            for cls in class_order
        ]
        data_by_class = [d[d > 0].to_numpy() for d in data_by_class]   # log axis needs strictly positive values
        positions = np.arange(1, len(class_order) + 1)

        # Some classes legitimately have no value for a metric (e.g. 'noise' has
        # no defined SNR/SNR_full_median -- there's no detection onset to split
        # signal from noise the same way). violinplot's KDE needs >= 2 points,
        # so plot violins only where there's enough data and label the rest.
        has_data = np.array([len(d) >= 2 for d in data_by_class])
        ok_pos   = positions[has_data]
        ok_data  = [d for d, ok in zip(data_by_class, has_data) if ok]

        if ok_data:
            parts = ax.violinplot(ok_data, positions=ok_pos, showmedians=True, widths=0.8)
            ok_classes = [c for c, ok in zip(class_order, has_data) if ok]
            for pc, cls in zip(parts['bodies'], ok_classes):
                pc.set_facecolor(class_colors.get(cls, 'grey'))
                pc.set_alpha(0.65)
                pc.set_edgecolor('black')

        blended = ax.get_xaxis_transform()
        for pos, ok in zip(positions, has_data):
            if not ok:
                ax.text(pos, 0.5, 'no data', ha='center', va='center', fontsize=7,
                        color='grey', transform=blended)

        thr = thresholds.get(metric)
        if thr is not None:
            ax.axhline(thr, color='red', linestyle='--', linewidth=1.3,
                       label=f'Quality gate ({thr:g})', zorder=5)
            ax.legend(fontsize=8, loc='upper right')

        ax.set_yscale('log')
        ax.set_xticks(positions)
        ax.set_xticklabels(class_order, rotation=15, ha='right', fontsize=8)
        ax.set_ylabel(metric, fontsize=9)
        ax.set_title(metric, fontsize=10, fontweight='bold')
        ax.grid(True, axis='y', which='both', lw=0.3, alpha=0.3)

    fig.suptitle('SNR / data-quality distribution by class (log scale)\n'
                'noise is intentionally NOT quality-gated — shown here for contrast',
                fontsize=11, fontweight='bold')
    plt.tight_layout()

    fname = f"fig_snr_quality_by_class_{stamp}.png"
    out_path = os.path.join(run_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    [SAVED] {fname}")
    return out_path



# =============================================================================
# NOISE-CLASS DETECTION DIAGNOSTIC (script 04e)
# =============================================================================
# Shows, for ONE noise-class example extracted by 04d, exactly what classical
# STA/LTA saw: the waveform on top, its characteristic function below, with
# the threshold lines and the accepted trigger window both marked — the same
# kind of "how did the detector actually fire" view 02c gives for real
# detections, applied here to sanity-check the noise class.

def plot_noise_diagnostic(tr_wave, cft, freq_axis, times_spec, spec_db,
                          t_on, t_off, thr_on, thr_off,
                          run_dir, stamp, fname_tag, title_extra="",
                          spec_vmin=-200, spec_vmax=-120):
    """
    Three-panel figure for one noise-candidate window: bandpassed waveform
    (top) + broadband dB spectrogram (middle) + classical STA/LTA
    characteristic function (bottom), sharing a time axis.

    Mirrors the waveform+spectrogram layout of plot_waveform_spectrogram_example()
    (used for the EQ/RS/IQ report gallery, script 08) so noise examples are
    visually comparable to real events, with the STA/LTA panel kept underneath
    since that ratio is the actual mechanism 04d used to accept this window.

    Parameters
    ----------
    tr_wave     : obspy.Trace — velocity trace (response-removed if an
                  inventory was available, raw counts otherwise), BANDPASS
                  FILTERED to 04d's detection band (PRIMARY_FREQ_MIN/MAX),
                  spanning the detected window plus context padding
    cft         : 1-D numpy array — classical STA/LTA ratio, same sampling
                  rate as tr_wave (from detection.run_sta_lta), same or
                  greater length
    freq_axis   : 1-D array — spectrogram frequency bins [Hz]
    times_spec  : 1-D array — spectrogram time bin centers, seconds relative
                  to tr_wave.stats.starttime (same convention as tr_wave.times())
    spec_db     : 2-D array, shape (len(freq_axis), len(times_spec)) — dB
                  power spectrogram, computed from the BROADBAND (unfiltered)
                  trace so the full frequency content is visible, not just
                  the narrow 1–20 Hz detection band
    t_on, t_off : UTCDateTime — the accepted trigger window (04d's
                  det_starttime/det_endtime for this row)
    thr_on, thr_off : float — STA/LTA thresholds used (drawn as reference lines)
    run_dir, stamp  : str — output directory / run timestamp
    fname_tag   : str — unique identifier for this example (e.g. "NET.STA_time"),
                  used in the output filename
    title_extra : str — appended to the figure title (e.g. a CFT value)
    spec_vmin/vmax : float — dB color scale bounds (same defaults as script 08)

    Returns
    -------
    out_path : str
    """
    t0      = tr_wave.stats.starttime
    t_wav   = tr_wave.times()
    data_um = tr_wave.data.astype(float) * 1e6   # m/s -> µm/s (no-op if response wasn't removed, still fine for shape)
    t_on_s  = t_on  - t0
    t_off_s = t_off - t0

    fig, (ax_wave, ax_spec, ax_cft) = plt.subplots(
        3, 1, figsize=(10, 8.5), sharex=True,
        gridspec_kw={'height_ratios': [1.3, 1.6, 1]},
    )

    # ── Waveform panel ────────────────────────────────────────────────────────
    ax_wave.plot(t_wav, data_um, 'k-', linewidth=0.6)
    ax_wave.axhline(0, color='lightgrey', linewidth=0.4, zorder=0)
    ax_wave.axvspan(t_on_s, t_off_s, color='#7f7f7f', alpha=0.22, zorder=1)
    ax_wave.axvline(t_on_s,  color='#555555', linewidth=1.5, zorder=3)
    ax_wave.axvline(t_off_s, color='#555555', linewidth=1.1, linestyle='--', zorder=3)
    ax_wave.set_ylabel("Velocity (µm/s)\n(1–20 Hz)", fontsize=10, fontweight='bold')
    ax_wave.set_title(
        f"{tr_wave.stats.network}.{tr_wave.stats.station}   {str(t_on)[:19]}Z"
        f"   |   duration={t_off - t_on:.1f}s{title_extra}",
        fontsize=11, fontweight='bold',
    )
    ax_wave.tick_params(labelsize=8)
    ax_wave.margins(x=0)

    # ── Spectrogram panel (broadband, unfiltered) ────────────────────────────
    im = ax_spec.pcolormesh(times_spec, freq_axis, spec_db, cmap='jet',
                            vmin=spec_vmin, vmax=spec_vmax, shading='auto')
    ax_spec.axvspan(t_on_s, t_off_s, color='white', alpha=0.12, zorder=2)
    ax_spec.axvline(t_on_s,  color='white', linewidth=1.2, zorder=3)
    ax_spec.axvline(t_off_s, color='white', linewidth=0.9, linestyle='--', zorder=3)
    ax_spec.set_ylabel("Frequency (Hz)\n(broadband)", fontsize=10, fontweight='bold')
    ax_spec.tick_params(labelsize=8)
    cbar = fig.colorbar(im, ax=ax_spec, pad=0.02, fraction=0.05)
    cbar.set_label('dB', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # ── STA/LTA characteristic function panel ────────────────────────────────
    n = min(len(t_wav), len(cft))
    ax_cft.plot(t_wav[:n], cft[:n], color='steelblue', linewidth=0.8)
    ax_cft.axhline(thr_on,  color='red', linestyle='--', linewidth=1.3,
                   label=f'THR_ON = {thr_on:g}')
    ax_cft.axhline(thr_off, color='darkorange', linestyle=':', linewidth=1.1,
                   label=f'THR_OFF = {thr_off:g}')
    ax_cft.axvspan(t_on_s, t_off_s, color='#7f7f7f', alpha=0.22, zorder=1)
    ax_cft.axvline(t_on_s,  color='#555555', linewidth=1.5, zorder=3)
    ax_cft.axvline(t_off_s, color='#555555', linewidth=1.1, linestyle='--', zorder=3)
    ax_cft.set_ylim(bottom=0)
    ax_cft.set_ylabel("STA/LTA\nratio", fontsize=10, fontweight='bold')
    ax_cft.set_xlabel("Time (s) relative to trace start", fontsize=10, fontweight='bold')
    ax_cft.legend(fontsize=8, loc='upper right')
    ax_cft.tick_params(labelsize=8)
    ax_cft.margins(x=0)

    plt.tight_layout()

    safe_tag = fname_tag.replace('/', '_').replace(':', '-').replace(' ', '_')
    fname    = f"noise_diagnostic_{safe_tag}_{stamp}.png"
    out_path = os.path.join(run_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [SAVED] {fname}")



# =============================================================================
# CNN CLASSIFIER REPORT FIGURES (script 08c)
# =============================================================================

def _percentile_clip_to_unit(channel_db, pctl_lo, pctl_hi):
    """Percentile-clip + min-max scale one 2-D dB array to [0, 1]."""
    lo, hi = np.percentile(channel_db, [pctl_lo, pctl_hi])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((channel_db - lo) / (hi - lo), 0.0, 1.0)


def plot_spectrogram_rgb_example(freq_axis, time_axis, image_db, title_lines, out_path,
                                  pctl_lo=1.0, pctl_hi=99.0):
    """
    Render one CNN training sample -- a (n_freq, n_time, 3) [Z, N, E] dB
    spectrogram image, exactly as 07a_spectrogram_dataset_build.py builds it
    and feeds it to the CNN -- as a single RGB-composite figure (R=Z, G=N,
    B=E) instead of three separate grayscale panels, so it reads as "the
    object the CNN actually sees" rather than three disconnected plots.

    Each channel is percentile-clipped and min-max scaled to [0, 1]
    INDEPENDENTLY before stacking into RGB -- Z/N/E can sit at very different
    absolute dB levels (station orientation, ground coupling), so a single
    shared scale across all three would wash out whichever channel happens to
    be weaker. This is a display-only transform; it has no effect on what the
    CNN was actually trained on (that used the (mean, std) z-score in
    normalization_stats.npz, not this per-image percentile clip).

    Parameters
    ----------
    freq_axis   : 1D array [Hz] — same for every sample (07a's freq_axis.npy)
    time_axis   : 1D array [s]  — same for every sample (07a's time_axis.npy)
    image_db    : 3D array (n_freq, n_time, 3), dB-scaled, channel order [Z, N, E]
    title_lines : tuple of str (line1, line2)
    out_path    : str — full path to save the PNG
    pctl_lo/hi  : float — percentile clip bounds per channel (default 1-99)

    Returns
    -------
    out_path : str
    """
    rgb = np.stack(
        [_percentile_clip_to_unit(image_db[:, :, c], pctl_lo, pctl_hi) for c in range(3)],
        axis=-1,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(
        rgb, aspect='auto', origin='lower',
        extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
    )
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Frequency (Hz)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.text(
        0.01, 0.99, 'R = Z   G = N   B = E', transform=ax.transAxes,
        ha='left', va='top', fontsize=7.5, color='white', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.45, edgecolor='none'),
    )

    line1, line2 = title_lines
    fig.suptitle(f"{line1}\n{line2}", fontsize=10, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_gradcam_example(freq_axis, time_axis, z_channel_db, cam, title_lines, out_path,
                         cam_cmap='jet', cam_alpha=0.45):
    """
    Two-panel figure for one CNN prediction: raw Z-channel spectrogram (left)
    and the same panel with a Grad-CAM overlay (right) -- shows which
    time-frequency region of the input drove that prediction. Same visual
    layout as 07b_train_cnn_classifier_colab.ipynb's Cell 14, generalized to
    an arbitrary number of examples per class instead of exactly one.

    This function is pure matplotlib/numpy -- the CAM array must already be
    computed (0-1 normalized, shape (n_freq, n_time)) by the caller, which is
    where the TensorFlow-specific work (gradient tape over the last Conv2D
    layer's activations) lives. Keeps this module importable without
    TensorFlow installed.

    Parameters
    ----------
    freq_axis    : 1D array [Hz]
    time_axis    : 1D array [s]
    z_channel_db : 2D array (n_freq, n_time) — raw (unnormalized) Z-channel
                   dB spectrogram, for display only
    cam          : 2D array (n_freq, n_time), values in [0, 1] — the
                   already-computed, already-normalized Grad-CAM heatmap
    title_lines  : tuple of str (line1, line2)
    out_path     : str — full path to save the PNG
    cam_cmap     : str — colormap for the CAM overlay
    cam_alpha    : float — overlay transparency

    Returns
    -------
    out_path : str
    """
    extent = [time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]]

    fig, (ax_raw, ax_cam) = plt.subplots(1, 2, figsize=(10, 4.5))

    ax_raw.imshow(z_channel_db, aspect='auto', origin='lower', cmap='viridis', extent=extent)
    ax_raw.set_title('Z channel (raw dB)', fontsize=10)
    ax_raw.set_xlabel('Time (s)', fontsize=9)
    ax_raw.set_ylabel('Frequency (Hz)', fontsize=9)
    ax_raw.tick_params(labelsize=8)

    ax_cam.imshow(z_channel_db, aspect='auto', origin='lower', cmap='gray', extent=extent)
    im = ax_cam.imshow(cam, aspect='auto', origin='lower', cmap=cam_cmap,
                       alpha=cam_alpha, extent=extent)
    ax_cam.set_title('Grad-CAM overlay', fontsize=10)
    ax_cam.set_xlabel('Time (s)', fontsize=9)
    ax_cam.tick_params(labelsize=8)
    cbar = fig.colorbar(im, ax=ax_cam, shrink=0.85, pad=0.02)
    cbar.set_label('Grad-CAM activation', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    line1, line2 = title_lines
    fig.suptitle(f"{line1}\n{line2}", fontsize=10, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path
    return out_path