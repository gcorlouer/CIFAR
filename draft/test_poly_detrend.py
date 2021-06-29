#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 15:48:38 2021
This script test polynomial detrending function
@author: guime
"""



import HFB_process as hf
import cifar_load_subject as cf
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import helper_functions as fun

from pathlib import Path, PurePath
from scipy import stats
from scipy.io import savemat
from statsmodels.tsa.tsatools import detrend

#%%

sub_id= 'DiAs'
proc= 'preproc' 
stage= '_BP_montage_HFB_raw.fif'
picks = ['LTo1-LTo2', 'LTo5-LTo6']
sfreq = 100
tmin = 0
tmax = 1.75
win_size = 0.100
step = 0.050
detrend = True
norder = 3
#%%
subject = cf.Subject(sub_id)
datadir = subject.processing_stage_path(proc=proc)
visual_populations = subject.pick_visual_chan()
ts , time = hf.chan_specific_category_ts(picks, proc='preproc', stage='_BP_montage_HFB_raw.fif', 
                     sub_id='DiAs', sfreq=sfreq, tmin_crop=0, tmax_crop=1.75)
(n, m, N, c) = ts.shape
#%%

def detrend_ts(ts, norder = 3, axis=0):
    """
    Detrend time series with statsmodel detrend function 
    """
    (n, m, N, c) = ts.shape
    
    for ichan in range(n):
        for icat in range(c):
            X = ts[ichan,:,:,icat]
            for order in range(norder):
                X = detrend(X, order=order, axis=axis)
            ts[ichan,:,:,icat] = X
    return ts

#%%

ts_detrend = detrend_ts(ts, norder=norder, axis=0)

#%%

hf.plot_trials(ts, time)
hf.plot_trials(ts_detrend, time, label='detrend')