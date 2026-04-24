                                                                                                                        # -*- coding: utf-8 -*-
"""
@authors: A.Maggi 2016 > porting from Matlab
          C. Hibert after 22/05/2017 > Original code from Matlab and addition of spectrogram attributes and other stuffs + comments 
          2021+ E. Pirot, C. Groult, C. Hibert
          
This function computes the attributes of a seismic signal later used to perform identification through machine
learning algorithms.

- Example: from ComputeAttributes_CH_V1 import calculate_all_attributes 
        
           all_attributes = calculate_all_attributes(Data,sps,flag)


- Inputs: "Data" is the raw seismic signal of the event (cutted at the onset and at the end of the signal)
          "sps" is the sampling rate of the seismic signal (in samples per second)
          "flag" is used to indicate if the input signal is 3C (flag==1) or 1C (flag==0).
          /!\ 3C PROCESSING NOT FULLY IMPLEMENTED YET /!\ 
          
- Output: "all_attributes" is an array of the attribute values for the input signal, ordered as detailed on lines 69-137

- Tweaks: Possibility to adapt the frequncy bands used to compute the energies and the Kurtosis envelopes 
          (attributes #13-#22) on lines 293-294


- References: 
    
        
        Provost, F., Hibert, C., & Malet, J. P. (2017). Automatic classification of endogenous landslide seismicity 
        using the Random Forest supervised classifier. Geophysical Research Letters, 44(1), 113-120.
        
        Hibert, C., Provost, F., Malet, J. P., Maggi, A., Stumpf, A., & Ferrazzini, V. (2017). Automatic identification 
        of rockfalls and volcano-tectonic earthquakes at the Piton de la Fournaise volcano using a Random Forest 
        algorithm. Journal of Volcanology and Geothermal Research, 340, 130-142.
        
        Pirot, E., Hibert, C., & Mangeney, A. (2024). Enhanced glacial earthquake catalogues443
        with supervised machine learning for more comprehensive analysis. Geophysical Journal444
        International , 236 (2), 849–871
 
"""

import numpy as np
from scipy.signal import (hilbert, lfilter, butter, spectrogram,
                          fftconvolve, sosfilt, sosfiltfilt, find_peaks)
from scipy.stats import kurtosis, skew

import scipy.signal
import scipy.signal.windows
if not hasattr(scipy.signal, 'hann'):  # scipy >= 1.8 removed scipy.signal.hann
    scipy.signal.hann = scipy.signal.windows.hann

from obspy import UTCDateTime
    

# -----------------------------------#
#            Main Function           #
# -----------------------------------#
 

def calculate_all_attributes(Data,sps,flag):
    
    sps=int(sps)

    # for 3C make sure is in right order (Z then horizontals)
    
    if flag==1:
        NATT = 62
        
    if flag==0:
        NATT = 99


        
    all_attributes = np.empty((1, NATT), dtype=float)

    env = envelope(Data,sps)
    
    TesMEAN, TesMEDIAN, TesSTD, env_max = get_TesStuff(env)
    
    RappMaxMean, RappMaxMedian = get_RappMaxStuff(TesMEAN, TesMEDIAN)
    # print('Get_MaxRappStuff done')

    AsDec, DistDecAmpEnv = get_AsDec(Data, env, sps)
    # print('Get_AsDec done')

    KurtoEnv, KurtoSig, SkewnessEnv, SkewnessSig =\
        get_KurtoSkewStuff(Data, env)
    # print('Get_kurtoStuff done')

    CorPeakNumber, INT1, INT2, INT_RATIO = get_CorrStuff(Data, sps)
    # print('Get_CorrStuff done')

    ES, KurtoF = get_freq_band_stuff(Data, sps)
    # print('Get_FreqBandstuff done')

    MeanFFT, MaxFFT, FmaxFFT, MedianFFT, VarFFT, FCentroid, Fquart1, Fquart3,\
        NpeakFFT, MeanPeaksFFT, E1FFT, E2FFT, E3FFT, E4FFT, gamma1, gamma2,\
        gammas = get_full_spectrum_stuff(Data, sps)
    # print('Get_SpectrumStuff done')

    if flag==1: #If signal is 3C then compute polarisation parameter
        rectilinP, azimuthP, dipP, Plani =\
            get_polarization_stuff(Data, env)
    
    SpecKurtoMaxEnv, SpecKurtoMedianEnv, RATIOENVSPECMAXMEAN, RATIOENVSPECMAXMEDIAN, \
    DISTMAXMEAN , DISTMAXMEDIAN, NBRPEAKMAX, NBRPEAKMEAN, NBRPEAKMEDIAN, RATIONBRPEAKMAXMEAN, \
    RATIONBRPEAKMAXMED, NBRPEAKFREQCENTER, NBRPEAKFREQMAX, RATIONBRFREQPEAKS, DISTQ2Q1, DISTQ3Q2, DISTQ3Q1 \
    = get_pseudo_spectral_stuff(Data, sps)
    # print('Get_pseudoSpectro done')

    # waveform
    all_attributes[0, 0] = np.mean(duration(Data,sps))  # 1  Duration of the signal
    all_attributes[0, 1] = np.mean(RappMaxMean)         # 2  Ratio of the Max and the Mean of the normalized envelope
    all_attributes[0, 2] = np.mean(RappMaxMedian)       # 3  Ratio of the Max and the Median of the normalized envelope
    all_attributes[0, 3] = np.mean(AsDec)               # 4  Ascending time/Decreasing time of the envelope
    all_attributes[0, 4] = np.mean(KurtoSig)            # 5  Kurtosis Signal
    all_attributes[0, 5] = np.mean(KurtoEnv)            # 6  Kurtosis Envelope
    all_attributes[0, 6] = np.mean(np.abs(SkewnessSig)) # 7  Skewness Signal
    all_attributes[0, 7] = np.mean(np.abs(SkewnessEnv)) # 8  Skewness envelope
    all_attributes[0, 8] = np.mean(CorPeakNumber)       # 9  Number of peaks in the autocorrelation function
    all_attributes[0, 9] = np.mean(INT1)                #10  Energy in the 1/3 around the origin of the autocorr function
    all_attributes[0, 10] = np.mean(INT2)               #11  Energy in the last 2/3 of the autocorr function
    all_attributes[0, 11] = np.mean(INT_RATIO)          #12  Ratio of the energies above
    all_attributes[0, 12] = np.mean(ES[0])              #13  Energy of the seismic signal in the 0.1-1Hz FBand
    all_attributes[0, 13] = np.mean(ES[1])              #14  Energy of the seismic signal in the 1-3Hz FBand
    all_attributes[0, 14] = np.mean(ES[2])              #15  Energy of the seismic signal in the 3-10Hz FBand
    all_attributes[0, 15] = np.mean(ES[3])              #16  Energy of the seismic signal in the 10-20Hz FBand
    all_attributes[0, 16] = np.mean(ES[4])              #17  Energy of the seismic signal in the 20-Nyquist F FBand
    all_attributes[0, 17] = np.mean(KurtoF[0])          #18  Kurtosis of the signal in the 0.1-1Hz FBand
    all_attributes[0, 18] = np.mean(KurtoF[1])          #19  Kurtosis of the signal in the 1-3Hz FBand
    all_attributes[0, 19] = np.mean(KurtoF[2])          #20  Kurtosis of the signal in the 3-10Hz FBand
    all_attributes[0, 20] = np.mean(KurtoF[3])          #21  Kurtosis of the signal in the 10-20Hz FBand
    all_attributes[0, 21] = np.mean(KurtoF[4])          #22  Kurtosis of the signal in the 20-Nyf Hz FBand
    all_attributes[0, 22] = np.mean(DistDecAmpEnv)      #23  Difference bewteen decreasing coda amplitude and straight line
    all_attributes[0, 23] = np.mean(env_max/duration(Data,sps)) # 24  Ratio between max envlelope and duration

    # spectral
    all_attributes[0, 24] = np.mean(MeanFFT)            #25  Mean FFT
    all_attributes[0, 25] = np.mean(MaxFFT)             #26  Max FFT
    all_attributes[0, 26] = np.mean(FmaxFFT)            #27  Frequence at Max(FFT)
    all_attributes[0, 27] = np.mean(FCentroid)          #28  Fq of spectrum centroid
    all_attributes[0, 28] = np.mean(Fquart1)            #29  Fq of 1st quartile
    all_attributes[0, 29] = np.mean(Fquart3)            #30  Fq of 3rd quartile
    all_attributes[0, 30] = np.mean(MedianFFT)          #31  Median Normalized FFT spectrum
    all_attributes[0, 31] = np.mean(VarFFT)             #32  Var Normalized FFT spectrum
    all_attributes[0, 32] = np.mean(NpeakFFT)           #33  Number of peaks in normalized FFT spectrum
    all_attributes[0, 33] = np.mean(MeanPeaksFFT)       #34  Mean peaks value for peaks>0.7 ## CHANGE IN NEW VERSION > SPREAD OF PEAKS
    all_attributes[0, 34] = np.mean(E1FFT)              #35  Energy in the 1 -- NyF/4 Hz (NyF=Nyqusit Freq.) band
    all_attributes[0, 35] = np.mean(E2FFT)              #36  Energy in the NyF/4 -- NyF/2 Hz band
    all_attributes[0, 36] = np.mean(E3FFT)              #37  Energy in the NyF/2 -- 3*NyF/4 Hz band
    all_attributes[0, 37] = np.mean(E4FFT)              #38  Energy in the 3*NyF/4 -- NyF/2 Hz band
    all_attributes[0, 38] = np.mean(gamma1)             #39  Spectrim centroid
    all_attributes[0, 39] = np.mean(gamma2)             #40  Spectrim gyration radio
    all_attributes[0, 40] = np.mean(gammas)             #41  Spectrim centroid width
    
    # Pseudo-Spectro.
    all_attributes[0, 41] = np.mean(SpecKurtoMaxEnv)    #42  Kurto of the envelope of the maximum energy on spectros
    all_attributes[0, 42] = np.mean(SpecKurtoMedianEnv) #43  Kurto of the envelope of the median energy on spectros
    all_attributes[0, 43] = np.mean(RATIOENVSPECMAXMEAN)#44  Ratio Max DFT(t)/ Mean DFT(t)
    all_attributes[0, 44] = np.mean(RATIOENVSPECMAXMEDIAN)#45  Ratio Max DFT(t)/ Median DFT(t)
    all_attributes[0, 45] = np.mean(DISTMAXMEAN)        #46  Nbr peaks Max DFTs(t)
    all_attributes[0, 46] = np.mean(DISTMAXMEDIAN)      #47  Nbr peaks Mean DFTs(t)
    all_attributes[0, 47] = np.mean(NBRPEAKMAX)         #48  Nbr peaks Median DFTs(t)
    all_attributes[0, 48] = np.mean(NBRPEAKMEAN)        #49  Ratio Max/Mean DFTs(t)
    all_attributes[0, 49] = np.mean(NBRPEAKMEDIAN)      #50  Ratio Max/Median DFTs(t)
    all_attributes[0, 50] = np.mean(RATIONBRPEAKMAXMEAN)#51  Nbr peaks X centroid Freq DFTs(t)
    all_attributes[0, 51] = np.mean(RATIONBRPEAKMAXMED) #52  Nbr peaks X Max Freq DFTs(t)
    all_attributes[0, 52] = np.mean(NBRPEAKFREQCENTER)  #53  Ratio Freq Max/X Centroid DFTs(t)
    all_attributes[0, 53] = np.mean(NBRPEAKFREQMAX)     #54  Mean distance bewteen Max DFT(t) Mean DFT(t)
    all_attributes[0, 54] = np.mean(RATIONBRFREQPEAKS)  #55  Mean distance bewteen Max DFT Median DFT
    all_attributes[0, 55] = np.mean(DISTQ2Q1)           #56  Distance Q2 curve to Q1 curve (QX curve = envelope of X quartile of DTFs)
    all_attributes[0, 56] = np.mean(DISTQ3Q2)           #57  Distance Q3 curve to Q2 curve
    all_attributes[0, 57] = np.mean(DISTQ3Q1)           #58  Distance Q3 curve to Q1 curve

# Add by Emilie (17/09/21)

    all_attributes[0, 58] = np.mean(ES[5])              #59  Energy of the seismic signal in the 0.01-0.05 Hz FBand
    all_attributes[0, 59] = np.mean(ES[6])              #60  Energy of the seismic signal in the 0.05-0.1 Hz FBand
    all_attributes[0, 60] = np.mean(ES[7])              #61  Energy of the seismic signal in the 0.01-0.1 Hz FBand  #Bande bruitée
    all_attributes[0, 61] = np.mean(ES[8])              #62  Energy of the seismic signal in the 0.1-0.5 Hz FBand 

    all_attributes[0, 62] = np.mean(KurtoF[5])          #63  Kurtosis of the signal in the 0.01-0.05Hz FBand
    all_attributes[0, 63] = np.mean(KurtoF[6])          #64  Kurtosis of the signal in the 0.05-0.1Hz FBand
    all_attributes[0, 64] = np.mean(KurtoF[7])          #65  Kurtosis of the signal in the 0.01-0.1Hz FBand
    all_attributes[0, 65] = np.mean(KurtoF[8])          #66  Kurtosis of the signal in the 0.1-0.5Hz FBand

    all_attributes[0, 66] = np.mean(ES[0]) - np.mean(ES[1])         #67  Difference of energy 0.1-1Hz/1-3Hz
    all_attributes[0, 67] = np.mean(ES[0]) - np.mean(ES[2])         #68  Difference of energy 0.1-1Hz/3-10Hz
    all_attributes[0, 68] = np.mean(ES[0]) - np.mean(ES[3])         #69  Difference of energy 0.1-1Hz/10-20Hz
    all_attributes[0, 69] = np.mean(ES[0]) - np.mean(ES[5])         #70  Difference of energy 0.1-1Hz/0.01-0.05Hz
    all_attributes[0, 70] = np.mean(ES[0]) - np.mean(ES[6])         #71  Difference of energy 0.1-1Hz/0.05-0.1Hz
    all_attributes[0, 71] = np.mean(ES[1]) - np.mean(ES[2])         #72  Difference of energy 1-3Hz/3-10Hz
    all_attributes[0, 72] = np.mean(ES[1]) - np.mean(ES[3])         #73  Difference of energy 1-3Hz/10-20Hz
    all_attributes[0, 73] = np.mean(ES[1]) - np.mean(ES[5])         #74  Difference of energy 1-3Hz/0.01-0.05Hz'
    all_attributes[0, 74] = np.mean(ES[1]) - np.mean(ES[6])         #75  Difference of energy 1-3Hz/0.05-0.1Hz
    all_attributes[0, 75] = np.mean(ES[2]) - np.mean(ES[3])         #76  Difference of energy 3-10Hz/10-20Hz
    all_attributes[0, 76] = np.mean(ES[2]) - np.mean(ES[5])         #77  Difference of energy 3-10Hz/0.01-0.05Hz
    all_attributes[0, 77] = np.mean(ES[2]) - np.mean(ES[6])         #78  Difference of energy 3-10Hz/0.05-0.1Hz
    all_attributes[0, 78] = np.mean(ES[3]) - np.mean(ES[5])         #79  Difference of energy 10-20Hz/0.01-0.05Hz
    all_attributes[0, 79] = np.mean(ES[3]) - np.mean(ES[6])         #80  Difference of energy 10-20Hz/0.05-0.1Hz
    all_attributes[0, 80] = np.mean(ES[5]) - np.mean(ES[6])         #81  Difference of energy 0.01-0.05Hz/0.05-0.1Hz
    
    all_attributes[0, 81] = np.mean(ES[0]) / np.mean(ES[1])         #82  Ratio of energy 0.1-1Hz/1-3Hz
    all_attributes[0, 82] = np.mean(ES[0]) / np.mean(ES[2])         #83  Ratio of energy 0.1-1Hz/3-10Hz
    all_attributes[0, 83] = np.mean(ES[0]) / np.mean(ES[3])         #84  Ratio of energy 0.1-1Hz/10-20Hz
    all_attributes[0, 84] = np.mean(ES[0]) / np.mean(ES[5])         #85  Ratio of energy 0.1-1Hz/0.01-0.05Hz
    all_attributes[0, 85] = np.mean(ES[0]) / np.mean(ES[6])         #86  Ratio of energy 0.1-1Hz/0.05-0.1Hz
    all_attributes[0, 86] = np.mean(ES[1]) / np.mean(ES[2])         #87  Ratio of energy 1-3Hz/3-10Hz
    all_attributes[0, 87] = np.mean(ES[1]) / np.mean(ES[3])         #88  Ratio of energy 1-3Hz/10-20Hz
    all_attributes[0, 88] = np.mean(ES[1]) / np.mean(ES[5])         #89  Ratio of energy 1-3Hz/0.01-0.05Hz'
    all_attributes[0, 89] = np.mean(ES[1]) / np.mean(ES[6])         #90  Ratio of energy 1-3Hz/0.05-0.1Hz
    all_attributes[0, 90] = np.mean(ES[2]) / np.mean(ES[3])         #91  Ratio of energy 3-10Hz/10-20Hz
    all_attributes[0, 91] = np.mean(ES[2]) / np.mean(ES[5])         #92  Ratio of energy 3-10Hz/0.01-0.05Hz
    all_attributes[0, 92] = np.mean(ES[2]) / np.mean(ES[6])         #93  Ratio of energy 3-10Hz/0.05-0.1Hz
    all_attributes[0, 93] = np.mean(ES[3]) / np.mean(ES[5])         #94  Ratio of energy 10-20Hz/0.01-0.05Hz
    all_attributes[0, 94] = np.mean(ES[3]) / np.mean(ES[6])         #95  Ratio of energy 10-20Hz/0.05-0.1Hz
    all_attributes[0, 95] = np.mean(ES[5]) / np.mean(ES[6])         #96  Ratio of energy 0.01-0.05Hz/0.05-0.1Hz
    
    all_attributes[0, 96] = signaltonoise(Data)         #97  SNR
    
    # NEW
    all_attributes[0, 97] = np.mean(ES[9])              #98  Energy of the seismic signal in the 1-8 Hz FBand
    all_attributes[0, 98] = np.mean(KurtoF[9])          #99  Kutosis of the seismic signal in the 1-8 Hz FBand


   # all_attributes[0, 97] =        #98  Difference HF/BF normalised to energy globale
   # all_attributes[0, 98] =        #99  Difference HF/BF normalised to 1-20s



    # polarisation
    if flag==1:
        all_attributes[0, 58] = rectilinP
        all_attributes[0, 59] = azimuthP
        all_attributes[0, 60] = dipP
        all_attributes[0, 61] = Plani

    return all_attributes
    

# -----------------------------------#
#        Secondary Functions         #
# -----------------------------------#

import numpy as np
import scipy.io

from obspy import Stream



def preprocess_signal_sp(stream,                # obspy.Stream —> the raw waveforms
                         inventory,             # obspy.Inventory —> obtained via client_fdsn.get_stations(...)
                         station_times_df,      # pd.DataFrame with columns: station / on_time / off_time
                         channel_end="Z",       # keep only channels ending in "Z" (vertical component)
                         broadbandonly=False,   # if True, only keeps HH* and BH* channels
                         water_level=60):       # regularization for response deconvolution (dB below peak)
    """
    Preprocess seismic data: remove instrument response, keep only vertical channels (Z),
    trim traces to station times.

    Parameters:
    - stream (obspy.Stream): Input seismic stream.
    - inventory (obspy.Inventory): Station inventory for response removal.
    - station_times_df (pd.DataFrame): DataFrame with columns ['station', 'on_time', 'off_time']

    Returns:
    - processed_stream (obspy.Stream): Stream with processed traces.
    - sps_list (list of floats): Sampling rate for each trace in the stream.
    """
    processed_stream = Stream()
    sps_list = []

    # Keep only vertical channels
    vertical_channels = [tr for tr in stream if tr.stats.channel.endswith(channel_end)]
    if broadbandonly:
        vertical_channels = [tr for tr in vertical_channels if tr.stats.channel.startswith("HH") or tr.stats.channel.startswith("BH")]

    for tr in vertical_channels:
        sta      = tr.stats.station
        sta_times = station_times_df[station_times_df['station'] == sta] # check if station has times in the dataframe
        if sta_times.empty:
            print(f"[SKIP] No time info for station {sta}.")
            continue

        s_time = UTCDateTime(sta_times['on_time'].values[0])
        e_time = UTCDateTime(sta_times['off_time'].values[0])

        tr_copy = tr.copy()

        # Ensure clean baseline before taper and deconvolution
        tr_copy.detrend('demean')
        tr_copy.detrend('linear')

        # Single taper before deconvolution to reduce spectral leakage at the edges
        # -> max_percentage=0.05 means 5% of the trace length tapered on each side
        tr_copy.taper(max_percentage=0.05)

        # Remove instrument response -> ground velocity [m/s]
        # water_level=60 : denominator floor at 60 dB below peak response
        #   -> prevents division by near-zero
        #   -> values in the stopband of short-period sensors (corner freq artifact)
        try:
            tr_copy.remove_response(
                inventory   = inventory,
                output      = "VEL",        # ground velocity in m/s
                water_level = water_level,  # regularization (60 dB = standard)
                pre_filt    = None          # bandpass applied externally
            )
        except Exception as e:
            print(f"[SKIP] Response removal failed for {sta}-{tr.stats.channel}: {e}")
            continue

        tr_copy.trim(starttime=s_time, # trim to the exact requested window
                     endtime=e_time, 
                     pad=True,         # handles minor timing mismatches (e.g. rounding by the SDS client)
                     fill_value=0)     # used only if the trace is genuinely shorter than the window

        # Validate: reject empty, NaN, Inf, or zero-amplitude traces
        d = tr_copy.data
        if (d is None
                or len(d) == 0
                or np.any(np.isnan(d))
                or np.any(np.isinf(d))
                or np.max(np.abs(d)) == 0):
            print(f"[SKIP] Invalid trace after response removal: {sta}-{tr.stats.channel}")
            continue

        processed_stream.append(tr_copy)
        sps_list.append(tr_copy.stats.sampling_rate)

    return processed_stream, sps_list




def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


    
def duration(Data,sps):

    dur = len(Data) / sps

    return dur


def envelope(Data,sps):
    
    env = np.abs(Data)

    return env


def get_TesStuff(env):

    CoefSmooth=3
    light_filter = np.ones(CoefSmooth) / float(CoefSmooth)

    env_max = np.max(env)
    tmp = lfilter(light_filter, 1, env/env_max)
    TesMEAN = np.mean(tmp)
    TesMEDIAN = np.median(tmp)
    TesSTD = np.std(tmp)

    return TesMEAN, TesMEDIAN, TesSTD, env_max


def get_RappMaxStuff(TesMEAN, TesMEDIAN):


    npts = 1
    RappMaxMean = np.empty(npts, dtype=float)
    RappMaxMedian = np.empty(npts, dtype=float)

    RappMaxMean = 1./TesMEAN
    RappMaxMedian = 1./TesMEDIAN

    return RappMaxMean, RappMaxMedian


def get_AsDec(Data, env, sps):

    strong_filter = np.ones(int(sps)) / float(sps)

    smooth_env = lfilter(strong_filter, 1, env)
    imax = np.argmax(smooth_env)
    
    if float(len(Data) - (imax+1))>0:
        AsDec = (imax+1) / float(len(Data) - (imax+1))
    else:
        AsDec = 0 
    
    dec = Data[imax:]
    lendec = len(dec)
     
    DistDecAmpEnv = np.abs(np.mean(np.abs(dec / np.max(Data)) -
            (1 - ((1 / float(lendec)) * (np.arange(lendec)+1)))))

    return AsDec, DistDecAmpEnv


def get_KurtoSkewStuff(Data, env):

    ntr = 1

    KurtoEnv = np.empty(ntr, dtype=float)
    KurtoSig = np.empty(ntr, dtype=float)
    SkewnessEnv = np.empty(ntr, dtype=float)
    SkewnessSig = np.empty(ntr, dtype=float)
    CoefSmooth = 3
    
    light_filter = np.ones(CoefSmooth) / float(CoefSmooth)

    env_max = np.max(env)
    data_max = np.max(Data)
    tmp = lfilter(light_filter, 1, env/env_max)
    KurtoEnv = kurtosis(tmp, fisher=False)
    SkewnessEnv = skew(tmp)
    KurtoSig = kurtosis(Data / data_max, fisher=False)
    SkewnessSig = skew(Data / data_max)

    return KurtoEnv, KurtoSig, SkewnessEnv, SkewnessSig


def get_CorrStuff(Data,sps):

    strong_filter = np.ones(int(sps)) / float(sps)
    min_peak_height = 0.33

    ntr=1
    CorPeakNumber = np.empty(ntr, dtype=int)
    INT1 = np.empty(ntr, dtype=float)
    INT2 = np.empty(ntr, dtype=float)
    INT_RATIO = np.empty(ntr, dtype=float)

    #cor = np.correlate(Data, Data, mode='full')
    cor = fftconvolve(Data,Data[::-1],mode='full')
    cor = cor / np.max(cor)

    # find number of peaks
    cor_env = np.abs(hilbert(cor))
    cor_smooth = lfilter(strong_filter, 1, cor_env)
    cor_smooth2 = lfilter(strong_filter, 1, cor_smooth/np.max(cor_smooth))
    
    ipeaks, _ = find_peaks(cor_smooth2, height=min_peak_height)
    CorPeakNumber = len(ipeaks)

    # integrate over bands
    npts = len(cor_smooth)
    ilag_0 = np.argmax(cor_smooth)+1
    ilag_third = ilag_0 + npts/6
    
    max_cor = np.max(cor_smooth)
    int1 = np.trapezoid(cor_smooth[int(ilag_0):int(ilag_third)+1]/max_cor, dx=1/sps)
    int2 = np.trapezoid(cor_smooth[int(ilag_third):]/max_cor, dx=1/sps)
    int_ratio = int1 / int2

    INT1 = int1
    INT2 = int2
    INT_RATIO = int_ratio

    return CorPeakNumber, INT1, INT2, INT_RATIO


def get_freq_band_stuff(Data,sps):

    NyF = sps / 2
    Ny_max_network = 10
#    ntr=1

    '''V5'''
    # lower bounds of the different tested freq. bands
    FFI = np.array([ 0.025, 1, 3, 6, 3, 1/80 , 1/100, 1/100, 1/40, 1]) 
    
    # higher bounds of the different tested freq. bands
    FFE = np.array([ 1, 3, 9.99, 9.99, 6, 1/20 , 1/60, 1/20, 1/10, 8]) 
    

#    ntr = len(st)
    nf = len(FFI)

    ES = np.empty(nf, dtype=float)
    KurtoF = np.empty(nf, dtype=float)

 
#    for i in range(ntr):
    for j in range(nf):
#        tr = Data
  
        sos = butter(4, [FFI[j]/NyF, FFE[j]/NyF], 'bandpass', output='sos')
        
        data_filt = sosfilt(sos, Data)
           
        ES[j] = np.log10(np.trapezoid(np.abs(data_filt), dx=1/sps))
        KurtoF[j] = kurtosis(data_filt, fisher=False)
        
    
    return ES, KurtoF


def get_full_spectrum_stuff(Data,sps):

    NyF = sps / 2.0

    Ny_max_network = 10

    ntr = 1
    MeanFFT = np.empty(ntr, dtype=float)
    MaxFFT = np.empty(ntr, dtype=float)
    FmaxFFT = np.empty(ntr, dtype=float)
    MedianFFT = np.empty(ntr, dtype=float)
    VarFFT = np.empty(ntr, dtype=float)
    FCentroid = np.empty(ntr, dtype=float)
    Fquart1 = np.empty(ntr, dtype=float)
    Fquart3 = np.empty(ntr, dtype=float)
    NpeakFFT = np.empty(ntr, dtype=float)
    MeanPeaksFFT = np.empty(ntr, dtype=float)
    E1FFT = np.empty(ntr, dtype=float)
    E2FFT = np.empty(ntr, dtype=float)
    E3FFT = np.empty(ntr, dtype=float)
    E4FFT = np.empty(ntr, dtype=float)
    gamma1 = np.empty(ntr, dtype=float)
    gamma2 = np.empty(ntr, dtype=float)
    gammas = np.empty(ntr, dtype=float)


    bb=50
    b = np.ones(bb) / bb

    data = Data
        
    dt= 1/sps
    fourier =  np.fft.fft(data)
    N=len(data)
    normalize = N/2
    fourier_norm = np.abs(fourier)/normalize
    FFTdata = fourier_norm *2
    Freq1 = np.fft.fftfreq(N, dt)

    FFTsmooth = lfilter(b, 1, FFTdata[0:int(len(FFTdata)/2)])
    FREQsmooth = Freq1[0:int(len(FFTdata)/2)]
    FFTsmooth_norm = FFTsmooth / max(FFTsmooth)

    FFTsmooth     = FFTsmooth[np.where(FREQsmooth < Ny_max_network)]
    FFTsmooth_norm = FFTsmooth_norm[np.where(FREQsmooth < Ny_max_network)]
    FREQsmooth    = FREQsmooth[np.where(FREQsmooth < Ny_max_network)]

    MeanFFT = np.mean(FFTsmooth_norm)
    MedianFFT = np.median(FFTsmooth_norm)
    VarFFT = np.var(FFTsmooth_norm, ddof=1)
    MaxFFT = np.max(FFTsmooth)
    iMaxFFT = np.argmax(FFTsmooth)
    
    FmaxFFT = Freq1[iMaxFFT]

    
    xCenterFFT = np.sum((np.arange(len(FFTsmooth_norm))) *
                                FFTsmooth_norm) / np.sum(FFTsmooth_norm)
    i_xCenterFFT = int(np.round(xCenterFFT))

    xCenterFFT_1quart = np.sum((np.arange(i_xCenterFFT+1)) *
                                  FFTsmooth_norm[0:i_xCenterFFT+1]) /\
            np.sum(FFTsmooth_norm[0:i_xCenterFFT+1])
    
    i_xCenterFFT_1quart = int(np.round(xCenterFFT_1quart))

    xCenterFFT_3quart = np.sum((np.arange(len(FFTsmooth_norm) -
                                              i_xCenterFFT)) *
                                   FFTsmooth_norm[i_xCenterFFT:]) /\
            np.sum(FFTsmooth_norm[i_xCenterFFT:]) + i_xCenterFFT+1
       
    i_xCenterFFT_3quart = int(np.round(xCenterFFT_3quart))

    FCentroid = Freq1[i_xCenterFFT]
    Fquart1 = Freq1[i_xCenterFFT_1quart]
    Fquart3 = Freq1[i_xCenterFFT_3quart]

    DiffSmooth = lfilter(b, 1, np.diff(FFTsmooth_norm)) / max(lfilter(b, 1, np.diff(FFTsmooth_norm)))
    min_peak_height = 0.1
    ipeaks, _ = find_peaks(DiffSmooth, height=min_peak_height)
    NpeakFFT    = len(ipeaks)
    MeanPeaksFFT = float(np.std(ipeaks)) if NpeakFFT > 0 else 0.0

    npts = len(FFTsmooth_norm)
    
    E1FFT = np.trapezoid(FFTsmooth_norm[0:int(npts/4)])#, dx=1/sps)
    E2FFT = np.trapezoid(FFTsmooth_norm[int(npts/4-1):int(2*npts/4)])#, dx=1/sps)
    E3FFT = np.trapezoid(FFTsmooth_norm[int(2*npts/4-1):int(3*npts/4)])#, dx=1/sps)
    E4FFT = np.trapezoid(FFTsmooth_norm[int(3*npts/4-1):int(npts)])#, dx=1/sps)

    moment = np.empty(3, dtype=float)

    for j in range(3):
        moment[j] = np.sum(Freq1[0:int(npts)]**j * FFTsmooth_norm[0:int(npts)]**2)
        # moment[j] = np.sum(Freq1**j * FFTsmooth_norm[0:int(n/2)]**2)
        
    gamma1 = moment[1]/moment[0]
    gamma2 = np.sqrt(moment[2]/moment[0])
    
    gammas = np.sqrt(np.abs(gamma1**2 - gamma2**2))

    return MeanFFT, MaxFFT, FmaxFFT, MedianFFT, VarFFT, FCentroid, Fquart1,\
        Fquart3, NpeakFFT, MeanPeaksFFT, E1FFT, E2FFT, E3FFT, E4FFT, gamma1,\
        gamma2, gammas


def get_polarization_stuff(st, env):

    sps = st[0].stats.sampling_rate
    strong_filter = np.ones(int(sps)) / float(sps)
    smooth_env = lfilter(strong_filter, 1, env[0])
    imax = np.argmax(smooth_env)
    end_window = int(np.round(imax/3.))

    xP = st[2].data[0:end_window]
    yP = st[1].data[0:end_window]
    zP = st[0].data[0:end_window]

    MP = np.cov(np.array([xP, yP, zP]))
    w, v = np.linalg.eig(MP)

    indexes = np.argsort(w)
    DP = w[indexes]
    pP = v[:, indexes]

    rectilinP = 1 - ((DP[0] + DP[1]) / (2*DP[2]))
    azimuthP = np.arctan(pP[1, 2] / pP[0, 2]) * 180./np.pi
    dipP = np.arctan(pP[2, 2] / np.sqrt(pP[1, 2]**2 + pP[0, 2]**2)) * 180/np.pi
    Plani = 1 - (2 * DP[0]) / (DP[1] + DP[2])

    return rectilinP, azimuthP, dipP, Plani


def get_pseudo_spectral_stuff(Data, sps):
    
    ntr=1
    Ny_max_network = 10
    SpecKurtoMaxEnv = np.empty(ntr, dtype=float)
    SpecKurtoMedianEnv = np.empty(ntr, dtype=float)
    RATIOENVSPECMAXMEAN = np.empty(ntr, dtype=float)
    RATIOENVSPECMAXMEDIAN = np.empty(ntr, dtype=float)
    DISTMAXMEAN = np.empty(ntr, dtype=float)
    DISTMAXMEDIAN = np.empty(ntr, dtype=float)
    NBRPEAKMAX = np.empty(ntr, dtype=float)
    NBRPEAKMEAN  = np.empty(ntr, dtype=float)
    NBRPEAKMEDIAN = np.empty(ntr, dtype=float)
    RATIONBRPEAKMAXMEAN = np.empty(ntr, dtype=float)
    RATIONBRPEAKMAXMED = np.empty(ntr, dtype=float)
    NBRPEAKFREQCENTER = np.empty(ntr, dtype=float)
    NBRPEAKFREQMAX = np.empty(ntr, dtype=float)
    RATIONBRFREQPEAKS = np.empty(ntr, dtype=float)
    DISTQ2Q1 = np.empty(ntr, dtype=float)
    DISTQ3Q2 = np.empty(ntr, dtype=float)
    DISTQ3Q1 = np.empty(ntr, dtype=float)
    
    # Spectrogram parametrisation
    # SpecWindow = 50 # Window legnth
    SpecWindow = int(2*sps) # Window legnth
    noverlap = int(0.90 * SpecWindow) # Overlap
    
    # on cherche la puissance de 2 de manière à avoir len(fft)>769 pour les freq>15  (sps 40 Hz comme référence)
    # pour mettre ensuite smooth_spec à la même taille quelque soit la freq d'echantillonnage
     
    n=1048*sps/10
    Freq = np.arange(0,int(sps/2), 0.025)

    # n = 2048 
    # Freq=np.linspace(0,int(sps),int(n/2)) # Sampling of frequency array
    
    b_filt = np.ones(50) / 50.0 # Smoothing param
    #print('Starting Spectro Computation')
    # Spectrogram computation from DFT (Discrete Fourier Transform on a moving window)
    f, t, spec = spectrogram(Data, fs=sps, window='hann',
                                     nperseg=SpecWindow, nfft=n, noverlap=noverlap,
                                     scaling='spectrum')
    #print('Finished Spectro Computation')
    # smooth_spec = lfilter(b_filt, 1, np.abs(spec), axis=1) #smoothing
    smooth_spec = lfilter(b_filt, 1, np.abs(spec[np.where(f<Ny_max_network),:][0]), axis=1)
    smooth_freq=f[np.where(f<Ny_max_network)]
    
    # Envelope of the maximum of each DFT constituting the spectrogram
    SpecMaxEnv, SpecMaxFreq = smooth_spec.max(0), smooth_spec.argmax(0)
    # Envelope of the mean of each DFT constituting the spectrogram
    SpecMeanEnv = smooth_spec.mean(0)
    # Envelope of the median of each DFT constituting the spectrogram
    SpecMedianEnv = np.median(smooth_spec, 0)
    
    # Envelope of different quartiles of each DFT
    CentoiX=np.empty(np.size(smooth_spec,1), dtype=float)
    CentoiX1=np.empty(np.size(smooth_spec,1), dtype=float)
    CentoiX3=np.empty(np.size(smooth_spec,1), dtype=float)
    
    #Freq=Freq[np.where(f<Ny_max_network)]
    # Envelope of the frequencies corresponding to different quartiles of the DFT
    for v in range(0, np.size(smooth_spec, 1)):
        CentoiX[v], CentoiX1[v], CentoiX3[v] = calculate_centroid_moments(smooth_freq, smooth_spec[:, v])
        
        
        
    # Tranform into single values
    SpecKurtoMaxEnv=kurtosis(SpecMaxEnv / SpecMaxEnv.max(axis=0))
    SpecKurtoMedianEnv=kurtosis(SpecMedianEnv / SpecMedianEnv.max(axis=0))
    RATIOENVSPECMAXMEAN = np.mean(SpecMaxEnv / SpecMeanEnv)
    RATIOENVSPECMAXMEDIAN = np.mean(SpecMaxEnv / SpecMedianEnv)
    DISTMAXMEAN = np.mean(np.abs(SpecMaxEnv - SpecMeanEnv))
    DISTMAXMEDIAN = np.mean(np.abs(SpecMaxEnv - SpecMedianEnv))
    NBRPEAKMAX    = len(find_peaks(SpecMaxEnv    / SpecMaxEnv.max(),    height=0.75)[0])
    NBRPEAKMEAN   = len(find_peaks(SpecMeanEnv   / SpecMeanEnv.max(),   height=0.75)[0])
    NBRPEAKMEDIAN = len(find_peaks(SpecMedianEnv / SpecMedianEnv.max(), height=0.75)[0])

    RATIONBRPEAKMAXMEAN = np.divide(NBRPEAKMAX, NBRPEAKMEAN)  if NBRPEAKMEAN   > 0 else 0
    RATIONBRPEAKMAXMED  = np.divide(NBRPEAKMAX, NBRPEAKMEDIAN) if NBRPEAKMEDIAN > 0 else 0

    NBRPEAKFREQCENTER = len(find_peaks(CentoiX    / CentoiX.max(),    height=0.75)[0])
    NBRPEAKFREQMAX    = len(find_peaks(SpecMaxFreq / (SpecMaxFreq.max() or 1), height=0.75)[0])

    RATIONBRFREQPEAKS = NBRPEAKFREQMAX / NBRPEAKFREQCENTER if NBRPEAKFREQCENTER > 0 else 0
        
    DISTQ2Q1 = np.mean(abs(CentoiX-CentoiX1))
    DISTQ3Q2 = np.mean(abs(CentoiX3-CentoiX))
    DISTQ3Q1 = np.mean(abs(CentoiX3-CentoiX1))
                        
    return SpecKurtoMaxEnv, SpecKurtoMedianEnv, RATIOENVSPECMAXMEAN, RATIOENVSPECMAXMEDIAN, \
    DISTMAXMEAN , DISTMAXMEDIAN, NBRPEAKMAX, NBRPEAKMEAN, NBRPEAKMEDIAN, RATIONBRPEAKMAXMEAN, \
    RATIONBRPEAKMAXMED, NBRPEAKFREQCENTER, NBRPEAKFREQMAX, RATIONBRFREQPEAKS, DISTQ2Q1, DISTQ3Q2, DISTQ3Q1


def nextpow2(i):
    n = 1
    while n < i:
        n *= 2
    return n

def calculate_centroid_moments(frequencies, amplitudes):
    centroid = np.sum(frequencies * amplitudes) / np.sum(amplitudes)
    bandwidth = np.sqrt(np.sum((frequencies - centroid) ** 2 * amplitudes) / np.sum(amplitudes))
    slope = np.sum((frequencies - centroid) ** 3 * amplitudes) / (np.sum(amplitudes) * bandwidth)
    return centroid, bandwidth, slope


def l2filter(b, a, x):

    # explicit two-pass filtering with no bells or whistles

    x_01 = lfilter(b, a, x)
    x_02 = lfilter(b, a, x_01[::-1])
    x_02 = x_02[::-1]

def centeroidnpX(arr):
    length = np.arange(1, len(arr)+1)
    CentrX = np.sum(length * arr) / np.sum(arr)
    return CentrX
