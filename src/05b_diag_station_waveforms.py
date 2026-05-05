"""
STATION WAVEFORM DIAGNOSTIC
============================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : April 2026

Goal
----
Quick visual inspection of one or several specific stations across catalog events

For each event where the target station has a pick, this script:
  1. Loads the waveform (PRE_EVENT s before origin, POST_EVENT s after)
  2. Applies the same 1–20 Hz bandpass as script 02
  3. Plots raw + filtered waveforms, with:
       - pre-event noise window highlighted (grey)
       - origin time marked (red dashed line)
       - P-pick time marked (green dashed line) if available
       - computed SNR value annotated on the plot
"""


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

SDS_ROOT    = "/data/sig/SDS"
ISTERRE_URL = "http://ist-sc3-geobs.osug.fr:8080"
OUTPUT_DIR  = "/data/failles/louisels/project/results/outputs_05b"

# Catalog query
T_START = "2022-02-01"
T_END   = "2022-08-01"
LAT_MIN, LAT_MAX = 45.5, 46.0
LON_MIN, LON_MAX = 6.5, 7.2
TARGET_TYPES = ["earthquake", "rockslide", "ice quake"]

# Stations to inspect (one figure per station)
TARGET_STATIONS = ["BLANC", "MFERR"]

Z_CHANNELS = "??Z"

# Waveform window
PRE_EVENT  = 150   # s before origin time
POST_EVENT = 90    # s after origin time

# Bandpass
FREQ_MIN = 1.0    # Hz
FREQ_MAX = 20.0   # Hz

# How many events to show per station (first N events with a pick at that station)
N_EVENTS_MAX = 10


# =============================================================================
# SECTION 2 — SETUP
# =============================================================================

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from obspy import UTCDateTime

from catalog_helpers import query_catalog, get_stations_from_picks, get_pick_times
from preprocessing import load_waveforms_sds, build_station_times_df, remove_response_or_fallback
from run_setup import (
    create_run_dir, setup_logging, connect_sds, connect_fdsn,
    fetch_inventory, set_matplotlib_defaults,
)
from detection import compute_snr

RUN_DIR, _RUN_STAMP = create_run_dir(OUTPUT_DIR)
_log_file, _log_filename = setup_logging(
    RUN_DIR, "05b_diag_station_waveforms.py",
    extra_info=f"Stations: {TARGET_STATIONS}  FREQ={FREQ_MIN}–{FREQ_MAX} Hz  "
               f"PRE={PRE_EVENT}s  POST={POST_EVENT}s"
)
set_matplotlib_defaults()

client_sds  = connect_sds(SDS_ROOT)
client_fdsn = connect_fdsn(ISTERRE_URL)
inventory   = fetch_inventory(client_fdsn, T_START, T_END) if client_fdsn else None

if client_sds is None:
    print("[ERROR] SDS client unavailable. Exiting.")
    sys.exit(1)
if client_fdsn is None:
    print("[ERROR] FDSN client unavailable. Exiting.")
    sys.exit(1)

events = query_catalog(client_fdsn, T_START, T_END,
                       LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, TARGET_TYPES)
print(f"\nCatalog: {len(events)} events loaded.\n")


# =============================================================================
# SECTION 3 — COLLECT TRACES PER TARGET STATION
# =============================================================================

# For each target station, collect up to N_EVENTS_MAX events that have a pick there
# Structure: station_events[sta] = list of (ev, tr_raw, tr_filt, t_orig, p_pick)

station_events = {sta: [] for sta in TARGET_STATIONS}

for ev in events:
    origin       = ev.preferred_origin() or ev.origins[0]
    t_orig       = origin.time
    etype        = str(ev.event_type) if ev.event_type else "unknown"
    stas         = get_stations_from_picks(ev)
    picks_by_sta = get_pick_times(ev)

    # Check if any target station has a pick in this event
    relevant_stas = [s for (_, s) in stas if s in TARGET_STATIONS]
    if not relevant_stas:
        continue

    # Check we still need more events for at least one target station
    needed = [s for s in relevant_stas if len(station_events[s]) < N_EVENTS_MAX]
    if not needed:
        continue

    # Load waveforms for this event
    st_raw, t_start, t_end = load_waveforms_sds(client_sds, ev, Z_CHANNELS, PRE_EVENT, POST_EVENT)
    if len(st_raw) == 0:
        continue

    station_times_df = build_station_times_df(st_raw, t_start, t_end)
    st_vel = remove_response_or_fallback(st_raw, inventory, station_times_df)
    if len(st_vel) == 0:
        continue

    for sta in needed:
        if len(station_events[sta]) >= N_EVENTS_MAX:
            continue

        sel = st_vel.select(station=sta)
        if len(sel) == 0:
            continue
        tr_vel = sel[0]

        # Apply bandpass (same as script 02)
        tr_filt = tr_vel.copy()
        nyq = tr_filt.stats.sampling_rate / 2
        tr_filt.filter('bandpass',
                       freqmin  = FREQ_MIN,
                       freqmax  = min(FREQ_MAX, 0.9 * nyq),
                       corners  = 2,
                       zerophase= True)

        p_pick = picks_by_sta.get(sta, {}).get('P', None)

        station_events[sta].append({
            'ev'     : ev,
            'etype'  : etype,
            't_orig' : t_orig,
            'p_pick' : p_pick,
            'tr_vel' : tr_vel,
            'tr_filt': tr_filt,
        })
        print(f"  [{sta}] collected event {t_orig}  ({etype})  —  "
              f"P pick: {p_pick if p_pick else 'none'}")


# =============================================================================
# SECTION 4 — PLOT ONE MULTI-PANEL FIGURE PER STATION
# =============================================================================

COLORS = {
    'earthquake': '#2196F3',
    'rockslide' : '#FF5722',
    'ice quake' : '#4CAF50',
    'unknown'   : '#9E9E9E',
}

for sta, ev_list in station_events.items():
    if not ev_list:
        print(f"\n[{sta}] No events found — skipping.")
        continue

    n = len(ev_list)
    fig, axes = plt.subplots(n, 2, figsize=(16, max(3, n * 2.2)),
                             sharex=False, sharey=False)
    if n == 1:
        axes = [axes]   # make iterable

    fig.suptitle(
        f"Station {sta} — waveform diagnostic\n"
        f"Bandpass {FREQ_MIN}–{FREQ_MAX} Hz  |  {n} events  |  {T_START} → {T_END}",
        fontsize=13, fontweight='bold'
    )

    for row, item in enumerate(ev_list):
        tr_vel  = item['tr_vel']
        tr_filt = item['tr_filt']
        t_orig  = item['t_orig']
        p_pick  = item['p_pick']
        etype   = item['etype']
        color   = COLORS.get(etype, COLORS['unknown'])

        t0   = tr_vel.stats.starttime          # absolute start of trace
        t_ax = np.arange(tr_vel.stats.npts) / tr_vel.stats.sampling_rate  # time axis in seconds from t0

        t_orig_rel = float(t_orig - t0)
        p_pick_rel = float(p_pick - t0) if p_pick is not None else None

        # SNR on the filtered trace (noise = before origin, signal = after origin to end)
        try:
            snr_dict = compute_snr(tr_filt, t_orig, tr_filt.stats.endtime)
            snr_val  = snr_dict.get('SNR_s2n_median', np.nan)
            snr_mean = snr_dict.get('SNR_full_mean', np.nan)
        except Exception:
            snr_val  = np.nan
            snr_mean = np.nan

        for col, (tr, label) in enumerate([(tr_vel, 'Raw [m/s]'),
                                            (tr_filt, f'Filtered {FREQ_MIN}–{FREQ_MAX} Hz')]):
            ax = axes[row][col]
            data = tr.data

            ax.plot(t_ax, data, lw=0.6, color=color, alpha=0.85)

            # Origin time
            ax.axvline(t_orig_rel, color='red', lw=1.2, ls='--',
                       label='Origin time' if row == 0 and col == 0 else '')

            # P-pick
            if p_pick_rel is not None:
                ax.axvline(p_pick_rel, color='green', lw=1.2, ls='-.',
                           label='P pick' if row == 0 and col == 0 else '')

            # Annotate SNR on filtered panel only
            if col == 1:
                ax.text(0.99, 0.97,
                        f"SNR_s2n={snr_val:.1f}  SNR_mean={snr_mean:.1f}",
                        transform=ax.transAxes, ha='right', va='top',
                        fontsize=7.5, color='black',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

            # Y-label on left panel only
            if col == 0:
                short_type = etype[:4].upper()
                ax.set_ylabel(f"{short_type}\n{str(t_orig)[:19]}", fontsize=7.5)

            # X-label on bottom row only
            if row == n - 1:
                ax.set_xlabel("Time from trace start [s]", fontsize=8)

            # Title on top row only
            if row == 0:
                ax.set_title(label, fontsize=9, fontweight='bold')

            ax.tick_params(labelsize=7)
            ax.set_xlim(t_ax[0], t_ax[-1])

    # Legend on the first axes
    handles = [
        plt.Line2D([0], [0], color='red',   lw=1.2, ls='--', label='Origin time'),
        plt.Line2D([0], [0], color='green', lw=1.2, ls='-.', label='P pick'),
    ]
    fig.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(RUN_DIR, f"diag_{sta}_{_RUN_STAMP}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[SAVED] {out_path}  ({n} events)")


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
