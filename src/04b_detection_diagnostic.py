"""
DETECTION DIAGNOSTIC
=================================================================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : April 2026

Goal
----
Quick QC tool to assess whether the detection thresholds in script 04 are well-calibrated
For a chosen station and day, this script:
  1. Loads and preprocesses the full-day trace (same as script 04)
  2. Runs DetecteurV3 on each 10-min window and collects sum_cft
  3. Produces two figures:
       Fig 1 — Full-day waveform with detected event windows overlaid
       Fig 2 — Full-day sum_cft (bidirectional STA/LTA characteristic function) with THR_ON and THR_OFF lines, and event markers

Use to decide if THR_ON / THR_OFF need to be raised (too many detections) or lowered (missing real events)
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# All parameters to adapt are grouped here
# =============================================================================

SDS_ROOT    = "/data/sig/SDS"
ISTERRE_URL = "http://ist-sc3-geobs.osug.fr:8080"
OUTPUT_DIR  = "/data/failles/louisels/project/results/outputs_04"

NETWORK  = "8C"
STATION  = "CI18"
LOCATION = "00"
CHANNEL  = "HHZ"
DAY      = "2022-06-01"

# Detection parameters — keep identical to script 04
FREQ_MIN  = 1.0
FREQ_MAX  = 20.0
NSTA      = 1
NLTA      = 15
THR_ON    = 6.0
THR_OFF   = 2.0
NWIN_SEC  = 5.0
NOVER_PCT = 0.20
WINDOW_SEC        = 10 * 60
OVERLAP_SEC       = 1  * 60
MIN_EVENT_DUR_SEC = 5.0



# =============================================================================
# SECTION 2 — SETUP
# =============================================================================

# ------------- Imports ----------------
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for cluster use
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from obspy import UTCDateTime

from detecteurV3_fonctions import DetecteurV3

from run_setup import connect_sds, connect_fdsn
from preprocessing import preprocess_day


# --------- Connections ----------------
print("Connecting to SDS and FDSN...")
client_sds  = connect_sds(SDS_ROOT)
client_fdsn = connect_fdsn(ISTERRE_URL)
inventory   = client_fdsn.get_stations(
    network=NETWORK, station=STATION,
    starttime=UTCDateTime(DAY), endtime=UTCDateTime(DAY) + 86400,
    level='response'
)
print("[OK] Connected")



# =============================================================================
# SECTION 3 : LOAD AND PREPROCESS
# =============================================================================

day_start = UTCDateTime(DAY)
day_end   = day_start + 86400

print(f"\nLoading {NETWORK}.{STATION}.{LOCATION}.{CHANNEL}  {DAY} ...")
st = client_sds.get_waveforms(
    network=NETWORK, station=STATION, location=LOCATION, channel=CHANNEL,
    starttime=day_start, endtime=day_end
)
st.merge(fill_value=None)
tr_raw = st[0]
print(f"  {tr_raw.stats.npts} samples at {tr_raw.stats.sampling_rate} Hz")

print("  Preprocessing (response removal)...")
tr_vel = preprocess_day(tr_raw, inventory)
if tr_vel is None:
    raise RuntimeError("Preprocessing failed — cannot continue.")
print("  Done.")

fs    = tr_vel.stats.sampling_rate
nwin  = int(NWIN_SEC * fs)
nover = int(nwin * NOVER_PCT)
nfft  = 2 ** int(np.ceil(np.log2(nwin)))



# =============================================================================
# SECTION 4: RUN DETECTOR 
# Collect sum_cft and events across all windows
# =============================================================================

print("\nRunning DetecteurV3 on 10-min sliding windows...")

all_t_cft   = []   # time axis of sum_cft (seconds from midnight)
all_sum_cft = []   # sum_cft values
all_events  = []   # list of (t_on_sec, t_off_sec) tuples

t_midnight = tr_vel.stats.starttime

win_start = tr_vel.stats.starttime
n_windows = 0

while win_start < tr_vel.stats.endtime:
    win_end = min(win_start + WINDOW_SEC, tr_vel.stats.endtime)
    tr_win  = tr_vel.slice(win_start, win_end)

    win_dur       = tr_win.stats.endtime - tr_win.stats.starttime
    dt_nrj_approx = NWIN_SEC * (1 - NOVER_PCT)
    if win_dur / dt_nrj_approx <= NLTA:
        break

    _, t_nrj, _, sum_cft, events_dt, _ = DetecteurV3(
        tr_win, FREQ_MIN, FREQ_MAX,
        NSTA, NLTA, THR_ON, THR_OFF,
        nwin, nover, nfft, 'True'
    )
    sum_cft = np.array(sum_cft).flatten()  # DetecteurV3 returns shape (1, N) — make it 1D

    # t_nrj is a list of datetime.datetime — convert to UTCDateTime then subtract midnight
    t_sec = np.array([UTCDateTime(str(t)) - t_midnight for t in t_nrj])

    # Keep only the non-overlapping portion to avoid double-plotting
    if win_end < tr_vel.stats.endtime:
        cutoff = win_dur - OVERLAP_SEC
        mask   = t_sec - (win_start - t_midnight) <= cutoff
    else:
        mask = np.ones(len(t_sec), dtype=bool)

    all_t_cft.append(t_sec[mask])
    all_sum_cft.append(sum_cft[mask] if len(sum_cft) == len(t_sec) else sum_cft[:len(t_sec)][mask])

    # Store detected events
    for val in events_dt.values():
        t_on  = UTCDateTime(str(val[0]))
        t_off = UTCDateTime(str(val[1]))
        dur   = t_off - t_on
        if dur >= MIN_EVENT_DUR_SEC:
            all_events.append((t_on - t_midnight, t_off - t_midnight))

    if win_end >= tr_vel.stats.endtime:
        break
    win_start = win_end - OVERLAP_SEC
    n_windows += 1

print(f"  {n_windows} windows processed — {len(all_events)} event(s) detected")

# Concatenate and convert to hours
t_cft_full = np.concatenate(all_t_cft)   / 3600
cft_full   = np.concatenate(all_sum_cft)
t_wav      = np.arange(tr_vel.stats.npts) / fs / 3600



# ---------- FIGURE 1: Waveform + detected events ------------------

fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
fig.suptitle(
    f"{NETWORK}.{STATION}.{LOCATION}.{CHANNEL}  —  {DAY}\n"
    f"THR_ON={THR_ON}  THR_OFF={THR_OFF}  FREQ={FREQ_MIN}–{FREQ_MAX} Hz  "
    f"→  {len(all_events)} detections",
    fontsize=12
)

ax = axes[0]
ax.plot(t_wav, tr_vel.data * 1e6, color='black', lw=0.4, label='velocity (µm/s)')
for (t_on_s, t_off_s) in all_events:
    ax.axvspan(t_on_s / 3600, t_off_s / 3600, color='red', alpha=0.25)
ax.set_ylabel('Velocity (µm/s)', fontsize=10)
ax.set_title('Waveform  (red = detected windows)', fontsize=10)
ax.legend(fontsize=8, loc='upper right')

ax = axes[1]
ax.plot(t_cft_full, cft_full, color='steelblue', lw=0.6, label='sum_cft (fwd + bwd STA/LTA)')
ax.axhline(THR_ON,  color='red',    lw=1.2, ls='--', label=f'THR_ON = {THR_ON}')
ax.axhline(THR_OFF, color='orange', lw=1.0, ls=':',  label=f'THR_OFF = {THR_OFF}')
for (t_on_s, t_off_s) in all_events:
    ax.axvspan(t_on_s / 3600, t_off_s / 3600, color='red', alpha=0.15)
ax.set_ylabel('sum_cft', fontsize=10)
ax.set_xlabel('Time (hours from midnight UTC)', fontsize=10)
ax.set_title('Bidirectional STA/LTA characteristic function', fontsize=10)
ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
os.makedirs(OUTPUT_DIR, exist_ok=True)
fig1_path = os.path.join(OUTPUT_DIR,
    f"diag_{NETWORK}_{STATION}_{DAY.replace('-','')}_waveform_cft.png")
fig.savefig(fig1_path, dpi=150, bbox_inches='tight')
print(f"\n[SAVED] Figure 1 → {fig1_path}")
plt.close()



# ---------- FIGURE 2: Distribution of sum_cft peaks -----------------

peaks, props = find_peaks(cft_full, height=THR_ON, distance=5)
peak_vals    = cft_full[peaks]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"{NETWORK}.{STATION}  {DAY}  —  Distribution of sum_cft peaks above THR_ON",
    fontsize=12
)

ax = axes[0]
ax.plot(t_cft_full, cft_full, color='steelblue', lw=0.5, alpha=0.7)
ax.scatter(t_cft_full[peaks], peak_vals, color='red', s=10, zorder=3,
           label=f'{len(peaks)} peaks above {THR_ON}')
ax.axhline(THR_ON,  color='red',    lw=1.2, ls='--', label=f'THR_ON = {THR_ON}')
ax.axhline(THR_OFF, color='orange', lw=1.0, ls=':',  label=f'THR_OFF = {THR_OFF}')
ax.set_xlabel('Time (hours)', fontsize=10)
ax.set_ylabel('sum_cft', fontsize=10)
ax.legend(fontsize=8)
ax.set_title('sum_cft with detected peaks', fontsize=10)

ax = axes[1]
ax.hist(peak_vals, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(THR_ON,  color='red',    lw=1.5, ls='--', label=f'THR_ON = {THR_ON}')
ax.axvline(THR_OFF, color='orange', lw=1.2, ls=':',  label=f'THR_OFF = {THR_OFF}')
ax.set_xlabel('Peak sum_cft value', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.set_title('Histogram of peak heights\n(most peaks near THR_ON → threshold too low?)', fontsize=10)
ax.legend(fontsize=8)

plt.tight_layout()
fig2_path = os.path.join(OUTPUT_DIR,
    f"diag_{NETWORK}_{STATION}_{DAY.replace('-','')}_cft_histogram.png")
fig.savefig(fig2_path, dpi=150, bbox_inches='tight')
print(f"[SAVED] Figure 2 → {fig2_path}")
plt.close()

print(f"\nDone. Summary:")
print(f"  Windows processed : {n_windows}")
print(f"  Events detected   : {len(all_events)}")
print(f"  CFT peaks > THR_ON: {len(peaks)}")
if len(peak_vals) > 0:
    print(f"  Peak cft range    : {peak_vals.min():.1f} – {peak_vals.max():.1f}")
