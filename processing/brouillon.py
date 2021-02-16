#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 12:03:50 2021

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

# sub_id = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']

#%% 
pd.options.display.max_rows = 999

sub_id = 'DiAs'
visual_chan_table = 'visual_channels_BP_montage.csv'
proc = 'preproc' 
sfreq = 250;
# picks = ['LGRD58-LGRD59', 'LGRD60-LGRD61', 'LTo1-LTo2', 'LTo3-LTo4']
tmin_crop = -0.5
tmax_crop = 1.75
suffix = 'preprocessed_raw'
ext = '.fif'

#%%

subject = cf.Subject(name=sub_id)
datadir = subject.processing_stage_path(proc=proc)
visual_chan = subject.pick_visual_chan()
# visual_chan = hf.pick_visual_chan(picks, visual_chan)
HFB = hf.visually_responsive_HFB(sub_id = sub_id)

categories = ['Rest', 'Face', 'Place']

_, visual_data = hf.HFB_to_visual_data(HFB, visual_chan, sfreq=sfreq, cat='Face', 
                                    tmin_crop = tmin_crop, tmax_crop=tmax_crop)
visual_time_series = visual_data

for cat in categories:
    X, visual_data = hf.HFB_to_visual_data(HFB, visual_chan, sfreq=sfreq, cat=cat, 
                                    tmin_crop = tmin_crop, tmax_crop=tmax_crop)
    visual_time_series[cat] = X


#%% Compute peak onset

cat = 'Face'
X = visual_time_series[cat]
nchan = X.shape[0]
evok = np.average(X, axis=2)
peak = np.amax(evok, axis=1)
peak_time = [0]*nchan
for i in range(nchan):
    peak_time[i] = np.where(evok[i,:] == peak[i])[0][0]
peak_max = max(peak_time)