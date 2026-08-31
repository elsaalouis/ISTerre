"""
catalog_helpers.py
==================
ISTerre internship 
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
    target_types : list of str -> ["earthquake", "ice quake"]  or  None -> keep every event type

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

    events = [ev for ev in cat if target_types is None or str(ev.event_type) in target_types]
    label = "ALL types (no filter)" if target_types is None else target_types
    print(f"After type filter : {len(events)} events kept — types: {label}")

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
# CHUNKED CATALOG QUERY WITH DISK CACHE
# =============================================================================

def query_catalog_chunked(client_fdsn, t_start, t_end,
                          lat_min, lat_max, lon_min, lon_max,
                          target_types, chunk_days=90, cache_path=None,
                          max_retries=3, retry_sleep_s=60):
    """
    Query the FDSN catalog in small time chunks to avoid server timeouts
     -> each failed chunk is retried up to max_retries times before being skipped
     -> the cache is only saved if ALL chunks succeeded — a partial cache is never written, so a re-run always fetches the missing windows

    Parameters
    ----------
    client_fdsn      : ObsPy FDSN_Client
    t_start, t_end   : str — ISO date strings e.g. "2022-01-01"
    lat_min/max, lon_min/max : float — bounding box
    target_types     : list of str — event types to keep 
    chunk_days       : int — size of each query window in days (default 90 ≈ 3 months)
    cache_path       : str or None — path to a .xml QuakeML cache file
                       • If the file already exists → load from it, no FDSN query
                       • If it does not exist       → query in chunks, save if complete
                       • None                       → query in chunks, no caching
    max_retries      : int — number of retry attempts per chunk on timeout (default 3)
    retry_sleep_s    : int — seconds to wait between retries (default 60)

    Returns
    -------
    events : list of ObsPy Event objects (filtered by target_types)
    """
    import os
    import time
    from obspy import UTCDateTime, Catalog, read_events

    # ---- Load from cache if available ----------------------------------------
    if cache_path and os.path.isfile(cache_path):
        print(f"\n[CACHE] Loading catalog from {cache_path} ...")
        cat    = read_events(cache_path)
        events = [ev for ev in cat if target_types is None or str(ev.event_type) in target_types]
        label  = "ALL types (no filter)" if target_types is None else target_types
        print(f"[CACHE] {len(cat)} events in file, "
              f"{len(events)} kept after type filter: {label}")
        return events

    # ---- Build chunk list ----------------------------------------------------
    t0        = UTCDateTime(t_start)
    t1        = UTCDateTime(t_end)
    chunk_sec = chunk_days * 86400

    chunks  = []
    current = t0
    while current < t1:
        next_t = min(current + chunk_sec, t1)
        chunks.append((current, next_t))
        current = next_t

    print(f"\nQuerying catalog in {len(chunks)} chunks of ~{chunk_days} days ...")
    print(f"  Full window  : {t_start} → {t_end}")
    print(f"  Bounding box : lat [{lat_min}, {lat_max}]  lon [{lon_min}, {lon_max}]")
    print(f"  Types kept   : {'ALL types (no filter)' if target_types is None else target_types}")
    print(f"  Retry policy : up to {max_retries} attempts, {retry_sleep_s}s sleep between retries")

    # ---- Query chunk by chunk (with retries) ---------------------------------
    all_events  = []
    seen_ids    = set()    # deduplicate events that straddle chunk boundaries
    failed_chunks = []     # (c_start, c_end) pairs that failed all retries

    for k, (c_start, c_end) in enumerate(chunks, 1):
        label = f"{str(c_start)[:10]} → {str(c_end)[:10]}"
        success = False

        for attempt in range(1, max_retries + 1):
            attempt_label = f"attempt {attempt}/{max_retries}" if attempt > 1 else ""
            print(f"  Chunk {k:2d}/{len(chunks)} : {label} {attempt_label}...",
                  end=" ", flush=True)
            try:
                cat_chunk = client_fdsn.get_events(
                    starttime       = c_start,
                    endtime         = c_end,
                    minlatitude     = lat_min,
                    maxlatitude     = lat_max,
                    minlongitude    = lon_min,
                    maxlongitude    = lon_max,
                    includearrivals = True,
                )
                n_new = 0
                for ev in cat_chunk:
                    ev_id = str((ev.preferred_origin() or ev.origins[0]).time)
                    if ev_id not in seen_ids:
                        seen_ids.add(ev_id)
                        all_events.append(ev)
                        n_new += 1
                print(f"{len(cat_chunk)} returned, {n_new} new")
                success = True
                break   # chunk done — move to next

            except Exception as e:
                err_msg = str(e) if str(e) else type(e).__name__
                if attempt < max_retries:
                    print(f"FAILED ({err_msg}) — retrying in {retry_sleep_s}s ...")
                    time.sleep(retry_sleep_s)
                else:
                    print(f"FAILED ({err_msg}) — giving up after {max_retries} attempts.")

        if not success:
            failed_chunks.append((c_start, c_end))

    # ---- Filter by type ------------------------------------------------------
    events = [ev for ev in all_events
              if target_types is None or str(ev.event_type) in target_types]
    print(f"\nTotal : {len(all_events)} events fetched across all chunks, "
          f"{len(events)} kept after type filter.")

    # ---- Report any failures -------------------------------------------------
    if failed_chunks:
        print(f"\n[WARN] {len(failed_chunks)} chunk(s) failed after all retries "
              f"— the following date ranges are MISSING from the catalog:")
        for cs, ce in failed_chunks:
            print(f"         {str(cs)[:10]} → {str(ce)[:10]}")
        print(f"[WARN] Cache will NOT be saved because the catalog is incomplete.")
        print(f"       Re-run the script to retry the missing chunks.")
        print(f"       If the same chunks keep failing, try reducing CHUNK_DAYS (e.g. 45).")
        return events

    # ---- Save cache (only if all chunks succeeded) ---------------------------
    if cache_path and events:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        Catalog(events=events).write(cache_path, format="QUAKEML")
        print(f"[CACHE] All chunks succeeded. Catalog saved → {cache_path}")
        print(f"        Next run will skip the FDSN query entirely.")

    return events



# =============================================================================
# DISTANCE-BASED CATALOG QUERY WITH DISK CACHE — used by script 04c
# =============================================================================

def query_catalog_by_distance_chunked(client_fdsn, t_start, t_end,
                                      center_lat, center_lon,
                                      min_radius_km, max_radius_km,
                                      min_magnitude, max_magnitude=None,
                                      chunk_days=90, cache_path=None,
                                      max_retries=3, retry_sleep_s=60):
    """
    Query an FDSN event catalog for events at a given EPICENTRAL-DISTANCE range from a reference point, above a minimum magnitude

    Parameters
    ----------
    client_fdsn      : ObsPy FDSN_Client (e.g. connect_fdsn("https://api.franceseisme.fr"))
    t_start, t_end   : str — ISO date strings -> "2015-01-01"
    center_lat/lon   : float — reference point (e.g. Mont Blanc massif centroid)
    min_radius_km    : float — minimum epicentral distance in KM
    max_radius_km    : float — maximum epicentral distance in KM 
    min_magnitude    : float — minimum magnitude to keep
    max_magnitude    : float or None — optional upper magnitude bound
    chunk_days       : int — size of each query window in days
    cache_path       : str or None — path to a .xml QuakeML cache file
    max_retries      : int — retry attempts per chunk on timeout 
    retry_sleep_s    : int — seconds to wait between retries 

    Returns
    -------
    events : list of ObsPy Event objects (deduplicated by origin time, already restricted to the [min_radius_km, max_radius_km] annulus)
    """
    import os
    import time
    import math
    from obspy import UTCDateTime, Catalog, read_events
    from obspy.geodetics import gps2dist_azimuth

    # ---- Load from cache if available ----------------------------------------
    if cache_path and os.path.isfile(cache_path):
        print(f"\n[CACHE] Loading distance-filtered catalog from {cache_path} ...")
        cat = read_events(cache_path)
        print(f"[CACHE] {len(cat)} events loaded.")
        return list(cat)

    max_radius_km = max_radius_km if max_radius_km is not None else 20000.0

    # ---- Bounding box that comfortably contains the full max_radius_km circle --
    km_per_deg_lat = 111.32
    dlat = (max_radius_km / km_per_deg_lat) * 1.10
    dlon = (max_radius_km / (km_per_deg_lat * max(math.cos(math.radians(center_lat)), 0.05))) * 1.10
    box_lat_min = max(center_lat - dlat, -90.0)
    box_lat_max = min(center_lat + dlat, 90.0)
    box_lon_min = center_lon - dlon
    box_lon_max = center_lon + dlon

    # ---- Build chunk list ----------------------------------------------------
    t0        = UTCDateTime(t_start)
    t1        = UTCDateTime(t_end)
    chunk_sec = chunk_days * 86400

    chunks  = []
    current = t0
    while current < t1:
        next_t = min(current + chunk_sec, t1)
        chunks.append((current, next_t))
        current = next_t

    print(f"\nQuerying distance-filtered catalog in {len(chunks)} chunks of ~{chunk_days} days ...")
    print(f"  Full window     : {t_start} → {t_end}")
    print(f"  Center point    : ({center_lat}, {center_lon})")
    print(f"  Target distance : [{min_radius_km:.0f}, {max_radius_km:.0f}] km "
          f"(applied client-side, after a bounding-box server query)")
    print(f"  Server bbox     : lat [{box_lat_min:.2f}, {box_lat_max:.2f}]  "
          f"lon [{box_lon_min:.2f}, {box_lon_max:.2f}]")
    print(f"  Magnitude       : >= {min_magnitude}" +
          (f", <= {max_magnitude}" if max_magnitude is not None else ""))

    # ---- Query chunk by chunk (with retries) ---------------------------------
    all_events    = []
    seen_ids      = set()   # deduplicate events that straddle chunk boundaries
    failed_chunks = []      # (c_start, c_end) pairs that failed all retries
    n_seen_in_box = 0
    n_kept_dist   = 0

    for k, (c_start, c_end) in enumerate(chunks, 1):
        label   = f"{str(c_start)[:10]} → {str(c_end)[:10]}"
        success = False

        for attempt in range(1, max_retries + 1):
            attempt_label = f"attempt {attempt}/{max_retries}" if attempt > 1 else ""
            print(f"  Chunk {k:2d}/{len(chunks)} : {label} {attempt_label}...",
                  end=" ", flush=True)
            try:
                kwargs = dict(
                    starttime    = c_start,
                    endtime      = c_end,
                    minlatitude  = box_lat_min,
                    maxlatitude  = box_lat_max,
                    minlongitude = box_lon_min,
                    maxlongitude = box_lon_max,
                    minmagnitude = min_magnitude,
                )
                if max_magnitude is not None:
                    kwargs["maxmagnitude"] = max_magnitude
                cat_chunk = client_fdsn.get_events(**kwargs)
                n_seen_in_box += len(cat_chunk)

                n_new = 0
                for ev in cat_chunk:
                    origin = ev.preferred_origin() or (ev.origins[0] if ev.origins else None)
                    if origin is None or origin.latitude is None or origin.longitude is None:
                        continue
                    dist_km = gps2dist_azimuth(center_lat, center_lon,
                                               origin.latitude, origin.longitude)[0] / 1000.0
                    if not (min_radius_km <= dist_km <= max_radius_km):
                        continue
                    n_kept_dist += 1
                    ev_id = str(origin.time)
                    if ev_id not in seen_ids:
                        seen_ids.add(ev_id)
                        all_events.append(ev)
                        n_new += 1
                print(f"{len(cat_chunk)} in box, {n_new} new in distance range")
                success = True
                break   # chunk done — move to next

            except Exception as e:
                err_msg = str(e) if str(e) else type(e).__name__
                if attempt < max_retries:
                    print(f"FAILED ({err_msg}) — retrying in {retry_sleep_s}s ...")
                    time.sleep(retry_sleep_s)
                else:
                    print(f"FAILED ({err_msg}) — giving up after {max_retries} attempts.")

        if not success:
            failed_chunks.append((c_start, c_end))

    print(f"\nTotal : {n_seen_in_box} event(s) returned inside the bounding box, "
          f"{len(all_events)} kept after distance filter "
          f"([{min_radius_km:.0f}, {max_radius_km:.0f}] km, mag >= {min_magnitude}).")

    # ---- Report any failures -------------------------------------------------
    if failed_chunks:
        print(f"\n[WARN] {len(failed_chunks)} chunk(s) failed after all retries "
              f"— the following date ranges are MISSING from the catalog:")
        for cs, ce in failed_chunks:
            print(f"         {str(cs)[:10]} → {str(ce)[:10]}")
        print(f"[WARN] Cache will NOT be saved because the catalog is incomplete.")
        print(f"       Re-run the script to retry the missing chunks.")
        return all_events

    # ---- Save cache (only if all chunks succeeded) ---------------------------
    if cache_path and all_events:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        Catalog(events=all_events).write(cache_path, format="QUAKEML")
        print(f"[CACHE] All chunks succeeded. Catalog saved → {cache_path}")
        print(f"        Next run will skip the FDSN query entirely.")

    return all_events



# =============================================================================
# STATION LIST FROM INVENTORY (used by script 04)
# =============================================================================

# Channel priority for deduplication: keep the highest-sampling-rate channel available per station (HHZ > BHZ > EHZ > HNZ > SHZ > anything else)
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
