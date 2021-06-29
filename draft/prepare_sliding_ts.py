#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 10:25:04 2021
This script prepare prepare the sliding window time series of specific 
channels.
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
tmax = 1.5
win_size = 0.100
step = 0.020
detrend = False
#%%
subject = cf.Subject(sub_id)
datadir = subject.processing_stage_path(proc=proc)
visual_populations = subject.pick_visual_chan()

#%%

ts, time = hf.sliding_ts(picks, proc=proc, stage=stage, sub_id=sub_id,
               tmin=tmin, tmax=tmax, win_size=win_size, step = step, detrend=detrend, sfreq=sfreq)

#%%

ts_dict = {'data': ts, 'sfreq': sfreq, 'time': time, 'sub_id': sub_id}
fname = sub_id + '_hfb_sliding.mat'
fpath = datadir.joinpath(fname)

savemat(fpath, ts_dict)