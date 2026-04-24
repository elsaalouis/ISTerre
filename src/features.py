"""
features.py
===========
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : April 2026

The 99-feature set defined in seismic_params.py (Maggi / Hibert):
  - FEATURE_NAMES: column names for the output CSV
  - FEATURE_DESCRIPTIONS: human-readable description of each feature
  - extract_features(): safe wrapper around calculate_all_attributes()
"""

import numpy as np


# =============================================================================
# FEATURE NAMES AND DESCRIPTIONS
# =============================================================================

# Column names feat_01 … feat_99 —> indices match the numbered comments in seismic_params.py
FEATURE_NAMES = [f"feat_{i:02d}" for i in range(1, 100)]

# Human-readable descriptions (same order as FEATURE_NAMES)
FEATURE_DESCRIPTIONS = [
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


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_features(data, sps, n_features=99, feature_flag=0):
    """
    Safe wrapper around calculate_all_attributes() from seismic_params.py
    
    Calculate_all_attributes() returns shape (1, N), a 2D array
     -> .flatten() is needed every time to get the expected 1D (N,) array
    
    try/except: if one trace is corrupted, the whole pipeline (24h of data) must not crash 
     -> the trace gets a row of NaNs that can be filtered out later

    Parameters
    ----------
    data         : numpy array — signal samples (from a preprocessed Trace)
    sps          : float — sampling rate in Hz
    n_features   : int — expected number of features (default 99)
    feature_flag : int — 0 = 99 features (vertical only), 1 = 62 features (3-component)

    Returns
    -------
    feats : numpy array, shape (n_features,)
    """
    try:
        from seismic_params import calculate_all_attributes
        feats = calculate_all_attributes(data, sps, feature_flag)
        return feats.flatten()   # (1, N) → (N,)
    except Exception as e:
        print(f"        [WARN] Feature extraction failed: {e}")
        return np.full(n_features, np.nan)
