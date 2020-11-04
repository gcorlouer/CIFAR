#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 11:01:40 2020

@author: guime
"""


import HFB_process
import cifar_load_subject
import scipy as sp
import re 
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd

from mne.viz import plot_filter, plot_ideal_filter
from scipy import signal, fftpack
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection, multipletests


pd.options.display.max_rows = 999

# TODO 
# Replace cohen by tstat (effect size)
#%% 

# Subject parameters
proc = 'preproc' # Line noise removed
ext2save = '.mat'
sub = 'DiAs' 
task = 'stimuli' # stimuli or rest_baseline_1
cat = 'Place' # Face or Place if task=stimuli otherwise cat=Rest
run = '1'
sfreq = 500; 

# Filter parameters 

l_freq = 60
h_freq = 80
band_size = 20
l_trans_bandwidth= 10
h_trans_bandwidth = 10
filter_length = 'auto'
fir_window = 'blackman'
phase = 'minimum'

# croping parameters

tmin = 6.032 # Place presentation
tmax = 10.6 # Face presentation
#%% 

subject = cifar_load_subject.Subject(name=sub, task= task, run = run)
fpath = subject.dataset_path(proc = proc, suffix='lnrmv')
raw = subject.read_eeglab_dataset(fpath)
X = raw.get_data()
#%% Filter properties

h = mne.filter.create_filter(X, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq,
                             filter_length='auto', l_trans_bandwidth=l_trans_bandwidth, 
                             h_trans_bandwidth=h_trans_bandwidth,  fir_window= fir_window,
                             phase=phase)

plot_filter(h, sfreq=500)


#%% Extract events time stamp of 2 sitmulus presentation (Place and Face)

events = mne.events_from_annotations(raw)
events_sample_stamp = events[0][:,0]
events_time_stamp = events_sample_stamp/sfreq

#%% Envelope of a representative visual channel

raw_band = raw.copy().pick_channels(['LTo6']).filter(l_freq=l_freq, h_freq=l_freq+band_size, 
                                 phase=phase, filter_length='auto',
                                 l_trans_bandwidth= l_trans_bandwidth, 
                                 h_trans_bandwidth= h_trans_bandwidth, 
                                     fir_window=fir_window)

X = raw_band.copy().crop(tmin=tmin, tmax = tmax).get_data()
envelope = raw_band.copy().crop(tmin=tmin, tmax = tmax).apply_hilbert(envelope=True).get_data()

time = raw.times
time = time[0:len(X[0])]

plt.plot(time,X[0])
plt.plot(time,envelope[0])

#%% HFB properties

bands = HFB_process.freq_bands() # Select Bands of interests 
HFB = HFB_process.extract_HFB(raw, bands) # Extract HFB

X = HFB.copy().pick_channels(['LGRD61']).crop(tmin=tmin, tmax = tmax).get_data()

#%% Detect visual channels 

HFB_db = HFB_process.raw_to_HFB_db(raw, bands)

events, event_id = mne.events_from_annotations(raw)
face_id = HFB_process.extract_stim_id(event_id)
place_id = HFB_process.extract_stim_id(event_id, cat='Place')
image_id = face_id+place_id

visual_channels, cohen = HFB_process.detect_visual_chan(HFB_db, tmin_pr=-0.4, 
                                                           tmax_pr=-0.1, tmin_po=0.1, tmax_po=0.5)

#%% Plot HFB of one representative visual channel

channel = 'LTo6'
visual_HFB = HFB_db.copy().pick_channels(visual_channels)

HFB_process.plot_HFB_response(HFB_db, image_id, picks=channel)
# %% Compute latency response

A_po = HFB_process.crop_stim_HFB(visual_HFB, image_id, tmin=0, tmax=1.5)
A_pr = HFB_process.crop_stim_HFB(visual_HFB, image_id, tmin=-0.4, tmax=-0.1)
A_baseline = HFB_process.cf_mean(A_pr)

latency_response = [0]*len(visual_channels)

for i in range(0, len(visual_channels)):
    for t in range(0,np.size(A_po,2)):
        tstat = stats.ttest_rel(A_po[:,i,t], A_baseline[:,i])
        pval = tstat[1]
        if pval <= 0.05:
            latency_response[i]=t/500*1e3 # return latency in ms
            break 
        else:
            continue

dfelec = subject.df_electrodes_info()

#%% classify V1 and V2 channels

latency_threshold = 180
group = ['other']*len(visual_channels)

for idx, channel in enumerate(visual_channels):
    if latency_response[idx] <= latency_threshold:
        brodman = dfelec['Brodman'].loc[dfelec['electrode_name']==channel]
        brodman = brodman.to_string(index=False)
        if brodman ==' V1':
            group[idx]='V1'
        elif brodman==' V2':
            group[idx]='V2'
        else:
            continue 
    else:
           continue  
# %% classify Face and Place selective electrodes

alpha = 0.05
A_face = HFB_process.crop_stim_HFB(visual_HFB, face_id, tmin=0.1, tmax=0.5)
A_place = HFB_process.crop_stim_HFB(visual_HFB, place_id, tmin=0.1, tmax=0.5)

A_face = np.mean(A_face, 2)
A_place = np.mean(A_place,2)

tstat = [0]*len(visual_channels)
pval = [0]*len(visual_channels)

for i in range(np.size(A_face,1)):
    tstat[i], pval[i] = stats.ttest_rel(A_face[:,i], A_place[:,i])
reject, pval_correct = fdrcorrection(pval, alpha=alpha)

# Significant electrodes located outside of V1 and V2 that are Face and Place responsive
for idx, channel in enumerate(visual_channels):
    if reject[idx]==False:
        continue
    else:
        if group[idx]=='V1':
            continue
        elif group[idx]=='V2':
            continue
        else:
            if tstat[idx]>0:
               group[idx] = 'Face'
            else:
               group[idx] = 'Place' 
