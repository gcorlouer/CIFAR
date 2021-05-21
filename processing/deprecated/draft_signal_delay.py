#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 17:32:56 2020

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
cat = 'Face'
proc = 'preproc' 
sfreq = 500;
picks = ['LGRD60-LGRD61', 'LTo1-LTo2']
tmin_crop = 0
tmax_crop = 1.5
suffix = 'preprocessed_raw'
ext = '.fif'

#%%

subject = cf.Subject(name=sub_id)
datadir = subject.processing_stage_path(proc=proc)
visual_chan = subject.low_high_chan()
#visual_chan = hf.pick_visual_chan(picks, visual_chan)
HFB = hf.low_high_HFB(visual_chan)

#%% Plot HFB

HFB_cat = hf.category_specific_HFB(HFB, cat='Face', tmin_crop = tmin_crop, tmax_crop=tmax_crop)
time = HFB_cat.times
evok = HFB_cat.copy().average()
ERP = evok.data

plt.plot(time, ERP[0,:])
plt.plot(time, ERP[1,:])

#%% Find signal delay 

X = ERP
nchan, nobs = X.shape
peak = np.amax(X, axis=1);
peak_time = np.zeros((nchan, 1))
                     
for i in range(0,nchan):
    peak_time[i] = np.where(X[i,:]==peak[i])
    peak_time[i] = peak_time[i]/sfreq
    
