"""
preprocessing.py
================
ISTerre internship
Author : Elsa Louis
Date   : April 2026

Signal preparation pipeline:
  - load waveforms from the SDS archive
  - clean (demean, detrend, taper)
  - remove instrument response -> ground velocity [m/s]
  - apply bandpass filter
"""

import numpy as np
import pandas as pd
from obspy import UTCDateTime, Stream


# =============================================================================
# WAVEFORM LOADING
# =============================================================================

def load_3component(client_sds, net, sta, loc, chan_z, t_start, t_end,
                    target_fs=100, window_s=30, horizontal_suffixes=None):
    """
    Load up to 3 components for a given station and time window from SDS
     -> used by 03c (DeepDenoiser data preparation) to build (3000, 3) .npz files in the format required by Zhu et al. (2019) DeepDenoiser

    Parameters
    ----------
    client_sds          : ObsPy SDS_Client
    net, sta, loc       : str — network, station, location codes
    chan_z              : str — vertical channel code, e.g. "HHZ"
    t_start, t_end      : UTCDateTime — time window to extract
    target_fs           : float — target sampling rate in Hz (default 100)
    window_s            : float — window duration in seconds (default 30)
    horizontal_suffixes : list of (str, str) or None
                          Pairs of (N-suffix, E-suffix) to try when loading horizontal components, e.g. [("N","E")]

    Returns
    -------
    data : np.ndarray, shape (nt, 3), dtype float32
           columns = [E, N, Z], if only Z is available, all 3 columns filled with Z
    """
    if horizontal_suffixes is None:
        horizontal_suffixes = [("N", "E"), ("2", "1")]

    nt = int(window_s * target_fs)

    # ---- Z component (mandatory) --------------------------------------------
    st_z = client_sds.get_waveforms(net, sta, loc, chan_z, t_start, t_end)
    if not st_z:
        raise ValueError(f"No Z trace for {net}.{sta}.{loc}.{chan_z}")
    st_z.merge(fill_value=0)
    tr_z = st_z[0].copy()

    if abs(tr_z.stats.sampling_rate - target_fs) > 1:
        tr_z.resample(target_fs)
    tr_z.detrend("demean")
    tr_z.trim(t_start, t_start + window_s, pad=True, fill_value=0)

    data_z = tr_z.data[:nt].astype("float32")
    if len(data_z) < nt:
        data_z = np.pad(data_z, (0, nt - len(data_z)))

    # ---- Horizontal components (optional) -----------------------------------
    base   = chan_z[:-1]   # e.g. "HH" from "HHZ"
    data_e = None
    data_n = None

    for suf_n, suf_e in horizontal_suffixes:
        try:
            st_h = client_sds.get_waveforms(
                net, sta, loc,
                base + suf_n + "," + base + suf_e,
                t_start, t_end,
            )
            if len(st_h) < 2:
                continue
            st_h.merge(fill_value=0)
            for tr in st_h:
                if abs(tr.stats.sampling_rate - target_fs) > 1:
                    tr.resample(target_fs)
                tr.detrend("demean")
                tr.trim(t_start, t_start + window_s, pad=True, fill_value=0)
                d = tr.data[:nt].astype("float32")
                if len(d) < nt:
                    d = np.pad(d, (0, nt - len(d)))
                if tr.stats.channel.endswith(suf_e):
                    data_e = d
                else:
                    data_n = d
            if data_e is not None and data_n is not None:
                break
        except Exception:
            continue

    # Fall back to Z if horizontals are unavailable
    if data_e is None:
        data_e = data_z.copy()
    if data_n is None:
        data_n = data_z.copy()

    return np.stack([data_e, data_n, data_z], axis=1)   # shape (nt, 3)


def load_waveforms_sds(client_sds, event, z_channels, pre, post):
    """
    Load vertical-component waveforms for all pick-stations of an event
     -> time window anchored to the first pick minus 'pre' seconds, so the earliest arrival is captured with enough pre-event margin

    Parameters
    ----------
    client_sds : ObsPy SDS_Client
    event      : ObsPy Event object
    z_channels : channel wildcard string -> "??Z"
    pre        : float — seconds before the first pick
    post       : float — seconds after origin time

    Returns
    -------
    st_all  : Stream — one trace per station (vertical component)
    t_start : UTCDateTime — start of the time window
    t_end   : UTCDateTime — end of the time window
    """
    from catalog_helpers import get_stations_from_picks

    origin       = event.preferred_origin() or event.origins[0]
    pick_times   = [p.time for p in event.picks if p.time]
    t_first_pick = min(pick_times) if pick_times else origin.time
    t_start      = t_first_pick - pre
    t_end        = origin.time + post

    stations = get_stations_from_picks(event)
    st_all   = Stream()
    seen     = set()

    for net, sta in stations:
        try:
            st = client_sds.get_waveforms(
                network   = net,
                station   = sta,
                location  = "*",
                channel   = z_channels,
                starttime = t_start,
                endtime   = t_end
            )
            if len(st) > 0:
                st.merge(method=1, fill_value="interpolate")
                for tr in st:
                    if tr.stats.station not in seen:
                        st_all += tr
                        seen.add(tr.stats.station)
        except Exception:
            pass

    return st_all, t_start, t_end



# =============================================================================
# SIGNAL CLEANING
# =============================================================================

def cosine_taper(st, max_percentage=0.05):
    """
    Apply a cosine taper in-place to each trace in the stream

    Why a manual implementation?
     ObsPy's built-in .taper() calls scipy.signal.hann internally, which was removed in recent scipy versions

    What does tapering do?
     Smoothly fades the signal to zero at both ends of the window (over the first and last 'max_percentage' of samples)
     Without it, the sharp edge at the window boundary introduces artificial high-frequency noise in the frequency domain (Gibbs phenomenon)
    """
    for tr in st:
        npts = len(tr.data)
        if npts == 0:
            continue
        n_taper = max(2, int(npts * max_percentage))
        taper   = np.ones(npts)
        taper[:n_taper]  = 0.5 * (1 - np.cos(np.pi * np.arange(n_taper) / n_taper))
        taper[-n_taper:] = 0.5 * (1 - np.cos(np.pi * np.arange(n_taper, 0, -1) / n_taper))
        tr.data = tr.data.astype(float) * taper


def preprocess(st, freqmin=None, freqmax=None, corners=4):
    """
    Clean a stream and optionally apply a bandpass filter
     -> steps always applied: merge → demean → detrend → cosine taper

    Parameters
    ----------
    st             : ObsPy Stream (not modified in place)
    freqmin/max    : float or None — filter corner frequencies in Hz
    corners        : int — number of filter poles (default 4)

    Returns
    -------
    st2 : new cleaned Stream
    """
    st2 = st.copy()
    st2.merge(method=1, fill_value="interpolate")
    st2.detrend('demean')
    st2.detrend('linear')
    cosine_taper(st2, max_percentage=0.05)
    if freqmin is not None and freqmax is not None:
        st2.filter('bandpass', freqmin=freqmin, freqmax=freqmax, corners=corners, zerophase=True)
    return st2


def apply_bandpass(st, freqmin, freqmax, corners=4):
    """
    Apply a zero-phase bandpass filter to a copy of the stream

    Use this on a stream that is already clean (demean/detrend/taper done), after instrument response removal

    Parameters
    ----------
    st           : ObsPy Stream (not modified in place)
    freqmin/max  : float — filter corner frequencies in Hz
    corners      : int — number of filter poles (default 4)

    Returns
    -------
    st_filt : new filtered Stream
    """
    st_filt = st.copy()
    st_filt.filter('bandpass',
                   freqmin   = freqmin,
                   freqmax   = freqmax,
                   corners   = corners,
                   zerophase = True)   # forward + backward pass -> zero phase distortion
    return st_filt



# =============================================================================
# INSTRUMENT RESPONSE REMOVAL
# =============================================================================

def build_station_times_df(st, t_start, t_end):
    """
    Build the per-station time window DataFrame required by preprocess_signal_sp()
     -> preprocess_signal_sp() needs to know the exact time window for each station to look up its instrument response in the inventory

    Parameters
    ----------
    st      : ObsPy Stream
    t_start : UTCDateTime — start of the waveform window
    t_end   : UTCDateTime — end of the waveform window

    Returns
    -------
    df : pandas DataFrame with columns ['station', 'on_time', 'off_time']
    """
    return pd.DataFrame([
        {'station': tr.stats.station,
         'on_time':  str(t_start),
         'off_time': str(t_end)}
        for tr in st
    ])


def remove_response_or_fallback(st_raw, inventory, station_times_df):
    """
    Remove instrument response if an inventory is available
     -> falls back to basic cleaning (demean + detrend + taper, no filter) if not

    Parameters
    ----------
    st_raw           : ObsPy Stream — raw waveforms in counts
    inventory        : ObsPy Inventory or None
    station_times_df : DataFrame from build_station_times_df()

    Returns
    -------
    st_vel : ObsPy Stream — ground velocity in m/s (or cleaned raw counts)
    """
    if inventory is not None:
        from seismic_params import preprocess_signal_sp  # does instrument response removal
        st_vel, _ = preprocess_signal_sp(st_raw, inventory, station_times_df)
        print(f"    {len(st_vel)} traces preprocessed (instrument response removed → m/s)")
    else:
        st_vel = preprocess(st_raw)   # no filter: preserve full spectrum
        print(f"    {len(st_vel)} traces preprocessed "
              f"(no inventory — raw counts, response NOT removed)")
    return st_vel


def preprocess_day(tr, inventory):
    """
    Preprocess a full-day continuous trace for detection

    Uses pre_filt (Groult approach) for response removal on long continuous traces:
     tapers the instrument response at both frequency ends to avoid amplifying the broadband noise that accumulates over a full 24-hour segment

    Parameters
    ----------
    tr        : obspy.Trace
    inventory : obspy.Inventory

    Returns
    -------
    tr_vel : obspy.Trace, or None if preprocessing fails
    """
    try:
        tr_vel = tr.copy()
        tr_vel.detrend('demean')
        tr_vel.detrend('linear')

        ny = tr_vel.stats.sampling_rate / 2.0
        # pre_filt: [f_hp_start, f_hp_end, f_lp_start, f_lp_end]
        # tapers response below 0.01 Hz and above 95% of Nyquist
        pre_filt = [0.005, 0.01, 0.95 * ny, 0.99 * ny]

        tr_vel.remove_response(
            inventory   = inventory,
            pre_filt    = pre_filt,
            output      = 'VEL',
            water_level = 60          # regularization safety net
        )
        return tr_vel

    except Exception as e:
        print(f"      [WARN] Response removal failed: {e}")
        return None
