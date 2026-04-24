"""
run_setup.py
============
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : April 2026

Infrastructure functions called once at the start of each script:
  - create run directory + logging
  - connect to SDS and FDSN
  - fetch instrument inventory
  - set matplotlib defaults
"""

import os
import sys
from datetime import datetime


# =============================================================================
# RUN DIRECTORY AND LOGGING
# =============================================================================

def create_run_dir(output_dir):
    """
    Create a timestamped subfolder inside output_dir for this run
     -> every run gets its own folder so outputs and logs from different runs never overwrite each other

    Parameters
    ----------
    output_dir : str — base output directory (created if it does not exist)

    Returns
    -------
    run_dir : str — full path to the newly created run folder
    stamp   : str — timestamp string (YYYYMMDD_HHMMSS) used in the folder name
    """
    stamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(output_dir, f"run_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, stamp


class _Tee:
    """
    Duplicates stdout to a log file so nothing is lost between runs
     -> replacing sys.stdout with a _Tee instance sends every print() call to both the original terminal and to the open log file
    """
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()   # write immediately, no buffering

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return self.terminal.isatty()


def setup_logging(run_dir, script_name, extra_info=""):
    """
    Redirect stdout to both the terminal and a run.log file inside run_dir
     -> call once near the top of each script, after create_run_dir()
     -> at the end of the script, close the returned file handle: log_file.close()

    Parameters
    ----------
    run_dir     : str — path to the run folder
    script_name : str — name of the calling script, shown in the header
    extra_info  : str — optional extra line printed in the header (e.g. key parameters)

    Returns
    -------
    log_file : file handle — must be closed at the end of the script
    log_path : str — full path to the log file
    """
    log_path = os.path.join(run_dir, "run.log")
    log_file = open(log_path, 'w', encoding='utf-8')
    sys.stdout = _Tee(sys.stdout, log_file)

    print("=" * 70)
    print(f"  {script_name}")
    print(f"  Run started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Run folder  : {run_dir}")
    if extra_info:
        print(f"  {extra_info}")
    print("=" * 70)

    return log_file, log_path



# =============================================================================
# CLIENT CONNECTIONS
# =============================================================================

def connect_sds(sds_root):
    """
    Connect to the SDS waveform archive

    Parameters
    ----------
    sds_root : str — path to the SDS archive root directory

    Returns
    -------
    client_sds : SDS_Client or None
    """
    if os.path.isdir(sds_root):
        from obspy.clients.filesystem.sds import Client as SDS_Client
        client = SDS_Client(sds_root)
        print(f"[OK] SDS client connected : {sds_root}")
        return client
    else:
        print(f"[WARN] SDS path not found : {sds_root}")
        print("       Waveform loading will be skipped (not running on cluster).")
        return None


def connect_fdsn(isterre_url):
    """
    Connect to the ISTerre FDSN server

    Parameters
    ----------
    isterre_url : str — base URL of the FDSN server

    Returns
    -------
    client_fdsn : FDSN_Client or None
    """
    try:
        from obspy.clients.fdsn import Client as FDSN_Client
        client = FDSN_Client(isterre_url)
        print(f"[OK] ISTerre FDSN connected : {isterre_url}")
        return client
    except Exception as e:
        print(f"[ERROR] ISTerre FDSN not reachable : {e}")
        print("        Connect to the OSUG VPN or run on the cluster.")
        return None


def fetch_inventory(client_fdsn, t_start, t_end, lat_min=None, lat_max=None, lon_min=None, lon_max=None):
    """
    Fetch the instrument inventory (poles, zeros, sensitivity) from FDSN
     -> the inventory is needed by remove_response_or_fallback() to convert raw counts into ground velocity [m/s]

    Parameters
    ----------
    client_fdsn              : ObsPy FDSN_Client
    t_start, t_end           : str, ISO date strings e.g. "2022-06-01"
    lat_min/max, lon_min/max : float or None — optional spatial filter

    Returns
    -------
    inventory : ObsPy Inventory, or None if the request fails
    """
    from obspy import UTCDateTime
    kwargs = dict(
        network   = "*",
        station   = "*",
        starttime = UTCDateTime(t_start),
        endtime   = UTCDateTime(t_end),
        level     = "response"   # deepest level: includes poles/zeros/gain
    )
    if lat_min is not None:
        kwargs.update(dict(
            minlatitude  = lat_min,
            maxlatitude  = lat_max,
            minlongitude = lon_min,
            maxlongitude = lon_max,
        ))
    try:
        inventory = client_fdsn.get_stations(**kwargs)
        print(f"[OK] Instrument inventory fetched ({len(inventory)} network(s))")
        return inventory
    except Exception as e:
        print(f"[WARN] Could not fetch inventory : {e}")
        print("       Response removal will be skipped.")
        return None



# =============================================================================
# PLOT SETTINGS
# =============================================================================

def set_matplotlib_defaults():
    """
    Apply the standard matplotlib font/size settings used across all scripts.
    Call once at the top of any script that produces figures.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size':        13,
        'axes.titlesize':   15,
        'axes.labelsize':   14,
        'xtick.labelsize':  12,
        'ytick.labelsize':  12,
        'legend.fontsize':  12,
    })
