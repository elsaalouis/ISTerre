"""
04c_check_3c_availability.py
============================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : July 2026

Goal
----
Check whether horizontal (N and E) components actually exist in the SDS archive for ALL event types (EQ, RS, IQ)

Strategy
--------
  - IQ  (1 079 quality-passing rows) : probe ALL rows  — IQ is the bottleneck class and every sample matters
  - EQ  (~41 000 rows)               : random sample of MAX_SAMPLE_EQ rows to keep runtime manageable
  - RS  (~6 500 rows)                : random sample of MAX_SAMPLE_RS rows
  Sampling is stratified by (network, station, channel) so every station appears at least once

Pipeline position
-----------------
  04a ✓  →  [04c this script]  →  (decision: implement 3C or skip)

Output
------
  Prints a per-type and per-station summary table to stdout
  Saves  3c_availability_<stamp>.csv  with one row per probed catalog row
"""


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# ── Input: catalog produced by 04a ───────────────────────────────────────────
CATALOG_CSV = "/data/failles/louisels/project/results/outputs_04a/run_20260531_104936/catalog_windows_20260531_104936.csv"

# ── SDS archive root ──────────────────────────────────────────────────────────
SDS_ROOT = "/data/sig/SDS"

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = "/data/failles/louisels/project/results/04c_3c_availability"

# ── Probe window length (seconds) ────────────────────────────────────────────
# Short window is enough — we only need to know if data exists.
PROBE_DURATION_S = 10.0

# ── Horizontal suffix pairs to try (in order) ────────────────────────────────
# Standard SEED: N/E for broadband, 1/2 for some legacy deployments
HORIZONTAL_PAIRS = [("N", "E"), ("1", "2")]

# ── Sampling limits for large classes ────────────────────────────────────────
MAX_SAMPLE_EQ = 300   # random rows from EQ (stratified by station)
MAX_SAMPLE_RS = 200   # random rows from RS (stratified by station)
RANDOM_SEED   = 42

# ── Quality filter ────────────────────────────────────────────────────────────
QUALITY_OK_ONLY = True   # True = only probe rows where quality_ok==True


# =============================================================================
# SECTION 2 — SETUP
# =============================================================================

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
from obspy import UTCDateTime
from obspy.clients.filesystem.sds import Client as SDS_Client

os.makedirs(OUTPUT_DIR, exist_ok=True)
stamp = time.strftime("%Y%m%d_%H%M%S")

print("=" * 65)
print("  04c — 3-Component Availability Check  (all event types)")
print(f"  Started : {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 65)


# =============================================================================
# SECTION 3 — CONNECT TO SDS
# =============================================================================

if not os.path.isdir(SDS_ROOT):
    print(f"\n[ERROR] SDS_ROOT not found: {SDS_ROOT}")
    print("        Run this script on the cluster.")
    sys.exit(1)

client_sds = SDS_Client(SDS_ROOT)
print(f"\n[OK] SDS client connected: {SDS_ROOT}")


# =============================================================================
# SECTION 4 — LOAD CATALOG AND BUILD PROBE SET
# =============================================================================

print(f"\nLoading catalog: {CATALOG_CSV}")
cat = pd.read_csv(CATALOG_CSV, low_memory=False)
print(f"  Total rows: {len(cat):,}")

if QUALITY_OK_ONLY:
    cat = cat[cat["quality_ok"] == True].copy()
    print(f"  After quality_ok filter: {len(cat):,}")

print(f"\n  Class counts:")
for etype, cnt in cat["event_type"].value_counts().items():
    print(f"    {etype:<12}: {cnt:,}")


def stratified_sample(df, n, seed=42):
    """
    Random sample of n rows, stratified by (network, station, channel) so that every station appears at least once in the sample
    Falls back to plain random if n >= len(df)
    """
    if n <= 0 or n >= len(df):
        return df.copy()

    rng    = np.random.RandomState(seed)
    groups = df.groupby(["network", "station", "channel"])

    # Give each station at least 1 slot, distribute remainder proportionally
    n_groups  = len(groups)
    per_group = max(1, n // n_groups)
    sampled   = []

    for _, grp in groups:
        take = min(per_group, len(grp))
        sampled.append(grp.sample(n=take, random_state=rng))

    result = pd.concat(sampled)

    # If we are still under n, top up with random rows not yet included
    remaining = n - len(result)
    if remaining > 0:
        already = result.index
        pool    = df.drop(index=already, errors="ignore")
        extra   = min(remaining, len(pool))
        if extra > 0:
            result = pd.concat([result, pool.sample(n=extra, random_state=rng)])

    return result.head(n)   # cap at n


# Build probe set per type
type_config = {
    "ice quake":  {"max": 0},            # 0 = all rows
    "earthquake": {"max": MAX_SAMPLE_EQ},
    "rockslide":  {"max": MAX_SAMPLE_RS},
}

probe_frames = []
for etype, cfg in type_config.items():
    sub = cat[cat["event_type"] == etype]
    if len(sub) == 0:
        continue
    sampled = stratified_sample(sub, cfg["max"], seed=RANDOM_SEED)
    print(f"\n  {etype}: {len(sub):,} rows → probing {len(sampled):,}"
          f"  ({len(sub.groupby(['network','station','channel']))} unique stations)")
    probe_frames.append(sampled)

probe_df = pd.concat(probe_frames).reset_index(drop=True)
print(f"\n  Total rows to probe: {len(probe_df):,}")


# =============================================================================
# SECTION 5 — PROBE SDS FOR HORIZONTAL CHANNELS
# =============================================================================

print(f"\n{'='*65}")
print("  Probing SDS for N and E channels ...")
print(f"{'='*65}\n")


def probe_horizontals(client, net, sta, chan_z, t0, duration_s, h_pairs):
    """
    Try to fetch horizontal components from SDS
     -> returns (has_N, has_E, chan_N_found, chan_E_found)
    """
    base  = chan_z[:-1]   # e.g. "HH" from "HHZ"
    t1    = t0 + duration_s
    has_N, has_E = False, False
    chan_N_found = chan_E_found = ""

    for suf_n, suf_e in h_pairs:
        if not has_N:
            try:
                st = client.get_waveforms(net, sta, "*", base + suf_n, t0, t1)
                if st and len(st) > 0 and len(st[0].data) > 0:
                    has_N = True
                    chan_N_found = base + suf_n
            except Exception:
                pass

        if not has_E:
            try:
                st = client.get_waveforms(net, sta, "*", base + suf_e, t0, t1)
                if st and len(st) > 0 and len(st[0].data) > 0:
                    has_E = True
                    chan_E_found = base + suf_e
            except Exception:
                pass

        if has_N and has_E:
            break

    return has_N, has_E, chan_N_found, chan_E_found


results = []

for _, row in tqdm(probe_df.iterrows(), total=len(probe_df), desc="Probing"):
    net  = str(row["network"])
    sta  = str(row["station"])
    chan = str(row["channel"])
    t0   = UTCDateTime(row["det_starttime"])

    has_N, has_E, cn, ce = probe_horizontals(
        client_sds, net, sta, chan, t0, PROBE_DURATION_S, HORIZONTAL_PAIRS
    )

    results.append({
        "event_type":    str(row.get("event_type", "")),
        "network":       net,
        "station":       sta,
        "channel_z":     chan,
        "event_time":    str(row.get("event_time", "")),
        "det_starttime": str(row.get("det_starttime", "")),
        "has_N":         has_N,
        "has_E":         has_E,
        "has_3C":        has_N and has_E,
        "chan_N_found":  cn,
        "chan_E_found":  ce,
    })


# =============================================================================
# SECTION 6 — SUMMARY
# =============================================================================

res_df = pd.DataFrame(results)

print(f"\n{'='*65}")
print("  RESULTS — 3C availability")
print(f"{'='*65}")

# ── Per event-type summary ────────────────────────────────────────────────────
print(f"\n  {'Type':<12} {'Probed':>7} {'3C':>6} {'3C%':>6} {'N-only':>7} {'E-only':>7} {'Z-only':>7}")
print(f"  {'-'*12} {'-'*7} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*7}")
for etype in ["earthquake", "rockslide", "ice quake"]:
    sub = res_df[res_df["event_type"] == etype]
    if len(sub) == 0:
        continue
    n   = len(sub)
    n3  = sub["has_3C"].sum()
    nN  = (sub["has_N"] & ~sub["has_E"]).sum()
    nE  = (~sub["has_N"] & sub["has_E"]).sum()
    nZ  = (~sub["has_N"] & ~sub["has_E"]).sum()
    print(f"  {etype:<12} {n:>7,} {n3:>6,} {100*n3/n:>5.1f}% {nN:>7,} {nE:>7,} {nZ:>7,}")

# ── Per-station breakdown ─────────────────────────────────────────────────────
print(f"\n  Per-station breakdown (all types combined):")
print(f"  {'Net':<6} {'Sta':<8} {'ChanZ':<7}", end="")
for etype in ["EQ", "RS", "IQ"]:
    print(f"  {etype+' 3C%':>8}", end="")
print()
print(f"  {'-'*6} {'-'*8} {'-'*7}" + "  " + "  ".join(["-"*8]*3))

label_map = {"earthquake": "EQ", "rockslide": "RS", "ice quake": "IQ"}
for (net, sta, cz), grp in res_df.groupby(["network", "station", "channel_z"]):
    print(f"  {net:<6} {sta:<8} {cz:<7}", end="")
    for etype in ["earthquake", "rockslide", "ice quake"]:
        sub = grp[grp["event_type"] == etype]
        if len(sub) == 0:
            print(f"  {'—':>8}", end="")
        else:
            pct = 100 * sub["has_3C"].sum() / len(sub)
            print(f"  {pct:>7.0f}%", end="")
    print()

# ── Horizontal channel codes found ───────────────────────────────────────────
n_3c_total = res_df["has_3C"].sum()
if n_3c_total > 0:
    print(f"\n  Horizontal channel codes found:")
    for code in sorted(res_df[res_df["chan_N_found"] != ""]["chan_N_found"].unique()):
        cnt = (res_df["chan_N_found"] == code).sum()
        print(f"    N/1-axis: {code}  ({cnt} rows)")
    for code in sorted(res_df[res_df["chan_E_found"] != ""]["chan_E_found"].unique()):
        cnt = (res_df["chan_E_found"] == code).sum()
        print(f"    E/2-axis: {code}  ({cnt} rows)")
else:
    print("\n  [!] No horizontal channels found at all — 3C not viable.")

# ── Interpretation hint ───────────────────────────────────────────────────────
iq_sub  = res_df[res_df["event_type"] == "ice quake"]
iq_pct  = 100 * iq_sub["has_3C"].sum() / max(len(iq_sub), 1)
eq_sub  = res_df[res_df["event_type"] == "earthquake"]
eq_pct  = 100 * eq_sub["has_3C"].sum() / max(len(eq_sub), 1)

print(f"\n  Interpretation:")
if iq_pct >= 80 and eq_pct >= 80:
    print(f"  ✓ Broad 3C coverage (IQ={iq_pct:.0f}%, EQ={eq_pct:.0f}%) → 2e is viable.")
    print(f"    Rebuild catalog with flag=1 for all events.")
elif iq_pct >= 60:
    print(f"  ~ Partial 3C coverage (IQ={iq_pct:.0f}%, EQ={eq_pct:.0f}%).")
    print(f"    Consider restricting 3C catalog to stations that have horizontals,")
    print(f"    or use NaN-safe imputation for missing polarization features.")
else:
    print(f"  ✗ Low 3C coverage (IQ={iq_pct:.0f}%, EQ={eq_pct:.0f}%) → 2e not viable.")
    print(f"    Horizontal data is too sparse to build a reliable 3C catalog.")


# =============================================================================
# SECTION 7 — SAVE
# =============================================================================

out_csv = os.path.join(OUTPUT_DIR, f"3c_availability_{stamp}.csv")
res_df.to_csv(out_csv, index=False)
print(f"\n[SAVED] {out_csv}")

print(f"\n{'='*65}")
print(f"  Run finished : {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*65}")
