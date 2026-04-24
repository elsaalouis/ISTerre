#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 19:11:33 2022

@author: Charlotte Groult
"""



import numpy as np
from obspy.realtime.signal import kurtosis
from obspy.signal.trigger import classic_sta_lta
from obspy.signal.trigger import trigger_onset 
import scipy.signal as ss
from obspy import read, read_inventory
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import datetime
import sqlite3
from obspy import UTCDateTime
import pandas as pd
import os
import time





''' fonctions ''' 



def nearest_ind(items, pivot):
    time_diff = np.abs([date - pivot for date in items])
    return time_diff.argmin(0)





def sliding_median(Trace, win_slide):
    ''' Sliding median for obspy.Trace or np.array'''

    if win_slide%2 == 0:
        win_slide += 1

    try:
        data = Trace.data.copy()
    except:
        data = Trace.copy()

    half_wind = int(win_slide/2)

    start = half_wind
    end = len(data) - half_wind

    count = start

    while count != end+1:
        data[count] = np.median(data[count-half_wind : count+half_wind])
        count += 1
    return data





def gather_event(on_off, time_window_NRJ_in_sec):
    
    Event_sorted = {}
    fin = 0
    iii = 0
    debut = iii
    k = 1

    while fin != np.shape(on_off)[0]:
        fusion = True
    
        while fusion == True and iii < np.shape(on_off)[0]-1:
       
            if (on_off[iii+1,0] - on_off[iii,1])*time_window_NRJ_in_sec <= 60:  #on regroupe si < 60 sec
                fusion = True 
                iii = iii + 1
                fin = iii
        
                
            else :
                fusion = False 
                fin = iii
                iii = iii + 1
        
        Event_sorted['Event_' + str(k)] = np.array([on_off[debut,0], on_off[fin,1]])
        fin = fin + 1 
        debut = fin
        k = k + 1
    return(Event_sorted)






def env(trace): 
    env = np.abs(trace.data)
    return env








def DetecteurV3(trace, freq_min, freq_max, nsta, nlta, thr_on, thr_off, nwin, nover, nfft, gather): 
 
    
    '''Compute spectrogram'''
    fs = trace.stats.sampling_rate
    f, t, Sxx = spectrogram(np.array(list(reversed(trace.data))), fs, nfft=nfft, nperseg=nwin, noverlap=nover)
    
    
    
    '''keep spectral content between 2 and 10 Hz'''
    ind = np.where((f>freq_min) & (f<freq_max))
    M = Sxx[ind[0],:]
    stack = np.sum(M, axis=0)
    NRJ = []
    NRJ = [np.array(list(reversed((stack))))]
    
   
    
    '''sta/lta performed on the stack of the cft forward and the cft backward'''
    
    cft = classic_sta_lta(stack, nsta, nlta)
    cft = np.array(list(reversed((cft))))
    cft_endroit = classic_sta_lta(NRJ[0], nsta, nlta)
    array_cft = np.array([[cft], [cft_endroit]])
    sum_cft = np.sum(array_cft, axis=0)
    
    on_off = np.array(trigger_onset(sum_cft[0], thr_on, thr_off))
    
    
    Event_detected = {}
    Event_detected_sorted = {}
    Event_in_time = {}
    Event_thresholds_sta_lta = {}


    time_NRJ = [trace.stats.starttime + t[i] for i in range(len(t))]
    time_NRJ = [datetime.datetime.utcfromtimestamp(elt.timestamp) for elt in time_NRJ]
    
    debut = trace.stats.starttime
    fin = trace.stats.endtime 
    debut = datetime.datetime.utcfromtimestamp(debut.timestamp)
    fin = datetime.datetime.utcfromtimestamp(fin.timestamp) + datetime.timedelta(seconds=trace.stats.delta)
    time_trace = np.arange(debut, fin, datetime.timedelta(seconds=trace.stats.delta))
    time_trace = time_trace[:len(trace)]
    
 
    
    ''' Aggregating close detections '''
    if gather == 'True':
        time_window_NRJ_in_sec = nwin/fs*(1-nwin/(nover*100))
        Event_detected = gather_event(on_off, time_window_NRJ_in_sec)  
        
    
    
    ''' keep only events > 5sec '''
    iii = 1
    for key, value in Event_detected.items():

        if (value[1]-value[0])*time_window_NRJ_in_sec > 5:
            Event_detected_sorted['Event_'+str(iii)] = value
            iii = iii + 1
            
            
            
    
    ''' retrieve starttime and endtime of the detection '''
    i=1
    start_in_sample = 0 
    end_in_sample = 0
    
    
    for key, value in Event_detected_sorted.items():

        try:
            start_in_sample = int(float(UTCDateTime(time_NRJ[value[0]])-UTCDateTime(str(time_trace[0]))) * fs) # faster method
        except IndexError:
            #print('nearest ind')
            start_in_sample = nearest_ind(time_trace[start_in_sample:].tolist(), time_NRJ[value[0]])  
            
         
        try:
            end_in_sample = int(float(UTCDateTime(time_NRJ[value[1]])-UTCDateTime(str(time_trace[0]))) * fs)   # faster method
        except IndexError:
            #print('nearest ind')
            end_in_sample = nearest_ind(time_trace[end_in_sample:].tolist(), time_NRJ[value[1]])  
            
             
        
        
             
    
        Event_in_time['Event_'+str(i)] = [time_trace[start_in_sample], time_trace[end_in_sample]]
        Event_thresholds_sta_lta['Event_'+str(i)] = [sum_cft[0][value[0]],sum_cft[0][value[1]]]
        i = i + 1
    
      


    return(time_trace, time_NRJ, NRJ, sum_cft, Event_in_time, Event_thresholds_sta_lta)
    
