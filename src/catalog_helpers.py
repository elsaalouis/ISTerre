"""
catalog_helpers.py
==================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : April 2026

Everything related to the FDSN event catalog and station/pick metadata:
  - query and filter events from the FDSN server
  - extract station lists and P/S pick times from event objects
  - compute station coverage statistics
  - build a station list from an inventory (for catalog-less scanning)
"""

from collections import defaultdict
import numpy as np
from obspy import UTCDateTime


# =============================================================================
# CATALOG QUERY
# =============================================================================

def query_catalog(client_fdsn, t_start, t_end, lat_min, lat_max, lon_min, lon_max, target_types):
    """
    Query the FDSN catalog and return only the events matching target_types

    Parameters
    ----------
    client_fdsn  : ObsPy FDSN_Client
    t_start, t_end : str, ISO date strings -> "2022-06-01"
    lat_min/max, lon_min/max : float, bounding box
    target_types : list of str -> ["earthquake", "ice quake"]

    Returns
    -------
    events : list of ObsPy Event objects (filtered by type)
    """
    print(f"\nQuerying catalog from {t_start} to {t_end} ...")
    print(f"  Bounding box : lat [{lat_min}, {lat_max}]  lon [{lon_min}, {lon_max}]")

    cat = client_fdsn.get_events(
        starttime       = UTCDateTime(t_start),
        endtime         = UTCDateTime(t_end),
        minlatitude     = lat_min,
        maxlatitude     = lat_max,
        minlongitude    = lon_min,
        maxlongitude    = lon_max,
        includearrivals = True   # attaches pick objects to each event
    )
    print(f"Found {len(cat)} events in total.")

    events = [ev for ev in cat if str(ev.event_type) in target_types]
    print(f"After type filter : {len(events)} events kept — types: {target_types}")

    return events


def summarise_catalog(events):
    """
    Print and return a summary of the catalog grouped by event type
     -> for each type, shows the total count and the list of origin times

    Parameters
    ----------
    events : list of ObsPy Event objects (already filtered by type)

    Returns
    -------
    summary : dict
        keys   = event type string ("earthquake", "ice quake", ...)
        values = dict with 'count' (int) and 'times' (list of str)
    """
    summary = defaultdict(lambda: {'count': 0, 'times': []})

    for ev in events:
        etype  = str(ev.event_type) if ev.event_type else "unknown"
        origin = ev.preferred_origin() or ev.origins[0]
        t_str  = str(origin.time)[:19]
        summary[etype]['count'] += 1
        summary[etype]['times'].append(t_str)

    for etype in summary:
        summary[etype]['times'].sort()

    print(f"\n{'─'*55}")
    print(f"  CATALOG SUMMARY  —  {sum(v['count'] for v in summary.values())} events total")
    print(f"{'─'*55}")
    for etype, data in sorted(summary.items(), key=lambda x: -x[1]['count']):
        print(f"\n  {etype.upper()}  ({data['count']} events)")
        for t in data['times']:
            print(f"      {t}")
    print(f"{'─'*55}\n")

    return dict(summary)



# =============================================================================
# PICKS AND STATION LISTS
# =============================================================================

def get_stations_from_picks(event):
    """
    Return a sorted list of unique (network, station) tuples for all picks associated with this event
    """
    stations = set()
    for pick in event.picks:
        wid = pick.waveform_id
        if wid.network_code and wid.station_code:
            stations.add((wid.network_code, wid.station_code))
    return sorted(stations)


def get_pick_times(event):
    """
    Return a dict mapping station_code -> {'P': UTCDateTime or None, 'S': UTCDateTime or None}
    Accepts any phase whose hint starts with 'P' or 'S' (covers Pg, Sg, Pn, Sn, …)
    """
    pick_dict = defaultdict(lambda: {'P': None, 'S': None})
    for pick in event.picks:
        sta   = pick.waveform_id.station_code
        phase = (pick.phase_hint or '').strip().upper()
        if not sta or not phase:
            continue
        generic = phase[0]   # 'P' or 'S' (ignores Pg → P, Sg → S, Pn → P, etc.)
        if generic in ('P', 'S'):
            # keep the earliest pick if there are duplicates
            if pick_dict[sta][generic] is None:
                pick_dict[sta][generic] = pick.time
    return dict(pick_dict)


def find_event_by_time(events, target_time_str, tolerance_s=200):
    """
    Find the catalog event whose origin time is closest to target_time_str

    Parameters
    ----------
    events          : list of ObsPy Event objects
    target_time_str : ISO string -> "2022-06-26T07:27:02"
    tolerance_s     : warn if the closest event is farther than this (seconds)

    Returns
    -------
    The closest Event object
    """
    target  = UTCDateTime(target_time_str)
    closest = min(
        events,
        key=lambda e: abs((e.preferred_origin() or e.origins[0]).time - target)
    )
    diff   = abs((closest.preferred_origin() or closest.origins[0]).time - target)
    origin = closest.preferred_origin() or closest.origins[0]
    if diff > tolerance_s:
        print(f"[WARN] Closest event is {diff:.1f}s away from {target_time_str}")
    else:
        print(f"[OK]  Found : {closest.event_type}  |  {origin.time}  |  diff={diff:.2f}s")
    return closest


def get_freq_range(event, freq_ranges, freqmin_default, freqmax_default):
    """
    Return the (freqmin, freqmax) bandpass range adapted to the event type
     -> falls back to (freqmin_default, freqmax_default) for unknown types
    """
    etype = str(event.event_type) if event.event_type else "unknown"
    return freq_ranges.get(etype, (freqmin_default, freqmax_default))



# =============================================================================
# STATION COVERAGE STATISTICS
# =============================================================================

def compute_station_coverage(events):
    """
    Compute how many events each station recorded, and per-type station counts

    Parameters
    ----------
    events : list of ObsPy Event objects

    Returns
    -------
    station_counts        : dict (net, sta) → int (number of events recorded)
    n_stations_per_event  : list of int (one entry per event)
    counts_by_type        : dict event_type → list of int (station counts per event)
    """
    station_counts       = defaultdict(int)
    n_stations_per_event = []
    counts_by_type       = defaultdict(list)

    for ev in events:
        stas  = get_stations_from_picks(ev)
        etype = str(ev.event_type) if ev.event_type else "unknown"
        n_stations_per_event.append(len(stas))
        counts_by_type[etype].append(len(stas))
        for net_sta in stas:
            station_counts[net_sta] += 1

    return dict(station_counts), n_stations_per_event, dict(counts_by_type)


# =============================================================================
# STATION LIST FROM INVENTORY (used by script 04)
# =============================================================================

# Channel priority for deduplication: keep the highest-sampling-rate channel available per station 
# (HHZ > BHZ > EHZ > HNZ > SHZ > anything else)
CHANNEL_PRIORITY = {'HHZ': 0, 'BHZ': 1, 'EHZ': 2, 'HNZ': 3, 'SHZ': 4}

def build_station_list_from_inventory(inventory, z_suffix='Z'):
    """
    Build a deduplicated list of (network, station, location, channel) tuples from an ObsPy Inventory
     -> keeping only vertical-component channels
     -> selecting the highest-priority channel per station when several are available

    Parameters
    ----------
    inventory : ObsPy Inventory
    z_suffix  : str — last character of vertical-component channel codes (default 'Z')

    Returns
    -------
    station_list : sorted list of (net, sta, loc, chan) tuples
    """
    candidates = []
    for network in inventory:
        for station in network:
            for channel in station:
                if not channel.code.endswith(z_suffix):
                    continue
                loc   = channel.location_code
                entry = (network.code, station.code, loc, channel.code)
                if entry not in candidates:
                    candidates.append(entry)

    # Keep only the highest-priority channel per (net, sta, loc) group
    best = {}   # (net, sta, loc) -> (priority, chan)
    for net, sta, loc, chan in candidates:
        key = (net, sta, loc)
        pri = CHANNEL_PRIORITY.get(chan, 99)
        if key not in best or pri < best[key][0]:
            best[key] = (pri, chan)

    station_list = [(net, sta, loc, chan)
                    for (net, sta, loc), (_, chan) in best.items()]
    station_list.sort()
    return station_list
