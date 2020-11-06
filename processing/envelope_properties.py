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
suffix = 'preprocessed_raw'
ext = '.fif'
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
fpath = subject.dataset_path(proc = proc, suffix=suffix, ext = ext)
raw = mne.io.read_raw_fif(fpath, preload=True)
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


#%% Detect visual channels 
bands = HFB_process.freq_bands() # Select Bands of interests 
HFB_db = HFB_process.raw_to_HFB_db(raw, bands)

events, event_id = mne.events_from_annotations(raw)
face_id = HFB_process.extract_stim_id(event_id)
place_id = HFB_process.extract_stim_id(event_id, cat='Place')
image_id = face_id+place_id

visual_channels, tstat = HFB_process.detect_visual_chan(HFB_db, tmin_pr=-0.4, 
                                                           tmax_pr=-0.1, tmin_po=0.1, tmax_po=0.5)

#%% Plot HFB of one representative visual channel

channel = 'LTo6'
visual_HFB = HFB_db.copy().pick_channels(visual_channels)

HFB_process.plot_HFB_response(HFB_db, image_id, picks=channel)
# %% Compute latency response

latency_response = HFB_process.compute_latency(visual_HFB, image_id, visual_channels)

#%% classify V1 and V2 channels

dfelec = subject.df_electrodes_info()
latency_threshold = 180

group = HFB_process.classify_retinotopic(latency_response, visual_channels, 
                                         dfelec, latency_threshold=latency_threshold)
# %% classify Face and Place selective electrodes

group = HFB_process.classify_Face_Place(visual_HFB, face_id, place_id, 
                            visual_channels, group, alpha=0.05);

