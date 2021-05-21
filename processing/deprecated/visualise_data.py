#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 11:50:54 2020

@author: guime
"""


import HFB_process as hf
import cifar_load_subject as cf
import scipy as sp
import re 
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path, PurePath
from mne.viz import plot_filter, plot_ideal_filter
from scipy import signal, fftpack
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection, multipletests

from scipy.io import loadmat,savemat

# %matplotlib
#%% TODO

# -Check that there are output visual_data X is correct with HFB_visual (i.e. check that 
# permutation works)
# - Create a module for category specific electrodes
# - Rearrange HFB module consequently

#%% 
pd.options.display.max_rows = 999

sub_id = 'DiAs'
visual_chan_table = 'visual_channels_BP_montage.csv'
proc = 'preproc' 
sfreq = 500;
picks = ['LGRD60-LGRD61', 'LTo1-LTo2']
tmin_crop = -0.5
tmax_crop = 1.75
suffix = 'preprocessed_raw'
ext = '.fif'

#%% Prepare dictionary for GC analysis

subject = cf.Subject(name=sub_id)
datadir = subject.processing_stage_path(proc=proc)
visual_chan = subject.low_high_chan()
# visual_chan = hf.pick_visual_chan(picks, visual_chan)
HFB = hf.low_high_HFB(visual_chan)

categories = ['Rest', 'Face', 'Place']
columns = ['Rest', 'Face', 'Place','populations']
visual_time_series = {'Rest': [], 'Face': [], 'Place': [], 'time': [], 
                      'populations_to_channel': [], 'channel_to_population': []}

for cat in categories:
    visual_data = hf.HFB_to_visual_data(HFB, visual_chan, sfreq=sfreq, cat=cat, 
                                    tmin_crop = tmin_crop, tmax_crop=tmax_crop)
    X = visual_data['data']
    visual_time_series[cat] = X
    
visual_time_series['populations_to_channel'] = visual_data['populations']
visual_time_series['channel_to_population'] = visual_data['channel_group']
visual_time_series['time'] = visual_data['time']
# %% Compute evoked response 

rest = visual_time_series['Rest']
face = visual_time_series['Face']
place = visual_time_series['Place']

rest_evok = np.average(rest, axis=(0,2))
face_evok = np.average(face, axis=(0,2))
place_evok = np.average(place, axis=(0,2))

time = visual_time_series['time']
#%% Take face and retinotopic channel

retinotopic_channels = visual_time_series['populations_to_channel']['V1'] + visual_time_series['populations_to_channel']['V2']
face_channels = visual_time_series['populations_to_channel']['Face']

for i in range(len(retinotopic_channels)):
    retinotopic_channels[i] = retinotopic_channels[i] -1
    
for i in range(len(face_channels)):
    face_channels[i] = face_channels[i] -1
    
#%% Prefered vs non prefered category

prefered = face[face_channels, :]
non_prefered = place[face_channels, :]

prefered_evok = np.average(prefered, axis=(0,2))
non_prefered_evok = np.average(non_prefered, axis=(0,2))

retinotopic = face[retinotopic_channels,:]
retinotopic_evok =  np.average(retinotopic, axis=(0,2))

plt.plot(time, prefered_evok)
plt.plot(time, non_prefered_evok)
plt.plot(time, retinotopic_evok)

#%% Plot retinotopic

retinotopic = place[retinotopic_channels,:]

for i in range(len(retinotopic_channels)):
    retinotopic_evok =  np.average(retinotopic[i, :], axis=1)
    plt.plot(time, retinotopic_evok)

#%% Plot face selective

