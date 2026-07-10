"""
features.py
===========
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : April 2026 (3C extension: July 2026)

Two feature sets are defined here:

  1C mode (default, backward compatible)
  ----------------------------------------
  FEATURE_NAMES / N_FEATURES_1C = 99
  Z-component only: waveform shape, spectral, pseudo-spectrogram,
  extended frequency bands, energy differences/ratios, misc.

  3C mode (new, option 2e)
  ----------------------------------------
  FEATURE_NAMES_3C / N_FEATURES_3C = 103
  All 99 Z features PLUS 4 polarization parameters computed from
  the [Z, N, E] particle-motion trajectory at the P-wave onset:
    rectilinP  — rectilinearity (0=spherical, 1=linear)
    azimuthP   — azimuth of principal motion axis [degrees]
    dipP       — dip (vertical incidence angle) of principal axis [degrees]
    Plani      — planarity (0=linear, 1=planar)

  Caller API:
    feats = extract_features(signal_z, sps)              # → (99,)  1C
    feats = extract_features(signal_z, sps, data_3c=X)  # → (103,) 3C
    where X is a (3, n_samples) array with rows [Z, N, E].
    Polarization features are set to NaN when X is None or computation fails;
    HGB handles NaN natively so no imputation is needed.
"""

import numpy as np


# =============================================================================
# FEATURE NAMES AND DESCRIPTIONS
# =============================================================================

# 99 Z-only feature names — CSV column headers, in order matching seismic_params.py
FEATURE_NAMES = [
    # Waveform shape (1–24)
    "duration",                      # 1
    "rapp_max_mean",                 # 2
    "rapp_max_median",               # 3
    "ascend_descend_ratio",          # 4
    "kurtosis_signal",               # 5
    "kurtosis_envelope",             # 6
    "skewness_signal",               # 7
    "skewness_envelope",             # 8
    "autocorr_peak_number",          # 9
    "autocorr_energy_1st3rd",        # 10
    "autocorr_energy_last2_3",       # 11
    "autocorr_energy_ratio",         # 12
    "energy_0.1_1Hz",                # 13
    "energy_1_3Hz",                  # 14
    "energy_3_10Hz",                 # 15
    "energy_10_20Hz",                # 16
    "energy_20_nyq",                 # 17
    "kurtosis_0.1_1Hz",              # 18
    "kurtosis_1_3Hz",                # 19
    "kurtosis_3_10Hz",               # 20
    "kurtosis_10_20Hz",              # 21
    "kurtosis_20_nyq",               # 22
    "dist_dec_amp_env",              # 23
    "env_max_over_duration",         # 24
    # Spectral (25–41)
    "fft_mean",                      # 25
    "fft_max",                       # 26
    "fft_freq_at_max",               # 27
    "fft_centroid",                  # 28
    "fft_quartile1",                 # 29
    "fft_quartile3",                 # 30
    "fft_median",                    # 31
    "fft_variance",                  # 32
    "fft_n_peaks",                   # 33
    "fft_spread_peaks",              # 34
    "fft_energy_1_nyq4",             # 35
    "fft_energy_nyq4_nyq2",          # 36
    "fft_energy_nyq2_3nyq4",         # 37
    "fft_energy_3nyq4_nyq",          # 38
    "spectral_centroid_gamma1",      # 39
    "spectral_gyration_gamma2",      # 40
    "spectral_centroid_width",       # 41
    # Pseudo-spectrogram (42–58)
    "spec_kurtosis_max_env",         # 42
    "spec_kurtosis_median_env",      # 43
    "ratio_env_max_mean",            # 44
    "ratio_env_max_median",          # 45
    "dist_max_mean",                 # 46
    "dist_max_median",               # 47
    "n_peaks_max",                   # 48
    "n_peaks_mean",                  # 49
    "n_peaks_median",                # 50
    "ratio_npeaks_max_mean",         # 51
    "ratio_npeaks_max_median",       # 52
    "n_peaks_freq_center",           # 53
    "n_peaks_freq_max",              # 54
    "ratio_n_freq_peaks",            # 55
    "dist_q2_q1",                    # 56
    "dist_q3_q2",                    # 57
    "dist_q3_q1",                    # 58
    # Extended frequency bands: Emilie additions (59–66)
    "energy_0.01_0.05Hz",            # 59
    "energy_0.05_0.1Hz",             # 60
    "energy_0.01_0.1Hz",             # 61
    "energy_0.1_0.5Hz",              # 62
    "kurtosis_0.01_0.05Hz",          # 63
    "kurtosis_0.05_0.1Hz",           # 64
    "kurtosis_0.01_0.1Hz",           # 65
    "kurtosis_0.1_0.5Hz",            # 66
    # Energy differences (67–81)
    "ediff_0.1_1__1_3",              # 67
    "ediff_0.1_1__3_10",             # 68
    "ediff_0.1_1__10_20",            # 69
    "ediff_0.1_1__0.01_0.05",        # 70
    "ediff_0.1_1__0.05_0.1",         # 71
    "ediff_1_3__3_10",               # 72
    "ediff_1_3__10_20",              # 73
    "ediff_1_3__0.01_0.05",          # 74
    "ediff_1_3__0.05_0.1",           # 75
    "ediff_3_10__10_20",             # 76
    "ediff_3_10__0.01_0.05",         # 77
    "ediff_3_10__0.05_0.1",          # 78
    "ediff_10_20__0.01_0.05",        # 79
    "ediff_10_20__0.05_0.1",         # 80
    "ediff_0.01_0.05__0.05_0.1",     # 81
    # Energy ratios (82–96)
    "eratio_0.1_1__1_3",             # 82
    "eratio_0.1_1__3_10",            # 83
    "eratio_0.1_1__10_20",           # 84
    "eratio_0.1_1__0.01_0.05",       # 85
    "eratio_0.1_1__0.05_0.1",        # 86
    "eratio_1_3__3_10",              # 87
    "eratio_1_3__10_20",             # 88
    "eratio_1_3__0.01_0.05",         # 89
    "eratio_1_3__0.05_0.1",          # 90
    "eratio_3_10__10_20",            # 91
    "eratio_3_10__0.01_0.05",        # 92
    "eratio_3_10__0.05_0.1",         # 93
    "eratio_10_20__0.01_0.05",       # 94
    "eratio_10_20__0.05_0.1",        # 95
    "eratio_0.01_0.05__0.05_0.1",    # 96
    # Misc (97–99)
    "snr",                           # 97
    "energy_1_8Hz",                  # 98
    "kurtosis_1_8Hz",                # 99
]

# Sanity check — catches accidental edits that break the 99-feature contract
assert len(FEATURE_NAMES) == 99, f"FEATURE_NAMES must have 99 entries, got {len(FEATURE_NAMES)}"

# ── 3C extension: 4 polarization features appended after the 99 Z features ───
POLARIZATION_NAMES = [
    "rectilinP",   # 100 — rectilinearity at P-onset (0=spherical, 1=linear)
    "azimuthP",    # 101 — azimuth of principal motion eigenvector [degrees]
    "dipP",        # 102 — dip (vertical incidence angle) [degrees]
    "Plani",       # 103 — planarity (0=linear, 1=planar)
]

FEATURE_NAMES_3C = FEATURE_NAMES + POLARIZATION_NAMES   # length 103

# Convenience constants
N_FEATURES_1C = 99    # 1C mode: Z only
N_FEATURES_3C = 103   # 3C mode: Z + polarization


# =============================================================================
# FEATURE GROUPS — semantic grouping
# =============================================================================

FEATURE_GROUPS = {
    "Waveform shape":          FEATURE_NAMES[0:24],
    "Spectral":               FEATURE_NAMES[24:41],
    "Pseudo-spectrogram":     FEATURE_NAMES[41:58],
    "Ext. freq. bands":       FEATURE_NAMES[58:66],
    "Energy differences":     FEATURE_NAMES[66:81],
    "Energy ratios":          FEATURE_NAMES[81:96],
    "Misc":                   FEATURE_NAMES[96:99],
}

FEATURE_GROUPS_3C = {
    **FEATURE_GROUPS,
    "Polarization":           POLARIZATION_NAMES,
}

# Flat feature → group mappings
_FEAT_TO_GROUP    = {f: g for g, fs in FEATURE_GROUPS.items()    for f in fs}
_FEAT_TO_GROUP_3C = {f: g for g, fs in FEATURE_GROUPS_3C.items() for f in fs}


def get_feature_group(feature_name, use_3c=False):
    """ Return the group label for a given feature name """
    mapping = _FEAT_TO_GROUP_3C if use_3c else _FEAT_TO_GROUP
    return mapping.get(feature_name, "Unknown")


def feature_group_array(use_3c=False):
    """ Return a list of group labels, one per feature, in FEATURE_NAMES order """
    names = FEATURE_NAMES_3C if use_3c else FEATURE_NAMES
    mapping = _FEAT_TO_GROUP_3C if use_3c else _FEAT_TO_GROUP
    return [mapping[f] for f in names]


# =============================================================================
# BACKWARD COMPATIBILITY — legacy feat_01 … feat_99 column names
# =============================================================================

LEGACY_NAMES = [f"feat_{i:02d}" for i in range(1, 100)]   # feat_01 … feat_99

# dict: {"feat_01": "duration", "feat_02": "rapp_max_mean", …}
LEGACY_TO_NAMED = dict(zip(LEGACY_NAMES, FEATURE_NAMES))


def rename_legacy_columns(df):
    """ If a DataFrame still uses old feat_XX column names, rename them in-place to the descriptive names defined in FEATURE_NAMES """
    if LEGACY_NAMES[0] in df.columns:
        df.rename(columns=LEGACY_TO_NAMED, inplace=True)
        print("[INFO] Legacy feature columns (feat_01…feat_99) renamed to descriptive names.")
    return df


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_features(data, sps, data_3c=None):
    """
    Extract seismic features from a Z-component signal.

    Always computes the 99 Z-only features (Maggi/Hibert) from seismic_params.py.
    When data_3c is provided, appends 4 polarization parameters computed from
    the [Z, N, E] particle-motion trajectory → 103 features total.

    Parameters
    ----------
    data    : 1D numpy array — Z-component signal (detection window, onset to end)
              This is the same signal passed to calculate_all_attributes().
    sps     : float — sampling rate in Hz
    data_3c : (3, n_samples) numpy array  or  None
              3-component aligned window with rows [Z, N, E].
              When None → 99 features (backward compatible, same as before).
              When provided → 103 features; polarization set to NaN if computation
              fails (e.g. too few samples, degenerate covariance matrix).
              HGB handles NaN natively — no imputation needed.

    Returns
    -------
    feats : 1D numpy array, shape (99,) or (103,)
            NaN values indicate failed computation for that feature.
    """
    n_out = N_FEATURES_3C if data_3c is not None else N_FEATURES_1C

    try:
        from seismic_params import calculate_all_attributes, get_polarization_stuff

        # ── Step 1: always compute the 99 Z-only features ────────────────────
        feats_99 = calculate_all_attributes(data, sps, flag=0).flatten()

        if data_3c is None:
            return feats_99   # backward-compatible 99-feature path

        # ── Step 2: append 4 polarization features ────────────────────────────
        try:
            rectilinP, azimuthP, dipP, Plani = get_polarization_stuff(data_3c, sps)
            pol = np.array([rectilinP, azimuthP, dipP, Plani], dtype=float)
            if not np.all(np.isfinite(pol)):
                pol = np.full(4, np.nan)
        except Exception as e:
            print(f"        [WARN] Polarization failed: {e}")
            pol = np.full(4, np.nan)

        return np.concatenate([feats_99, pol])   # (103,)

    except Exception as e:
        print(f"        [WARN] Feature extraction failed: {e}")
        return np.full(n_out, np.nan)
