#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 12:01:57 2021

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
sfreq = 200
win_size = 20
step = 2
detrend = False
tmin =0
tmax = 265
#%%

subject = cf.Subject(sub_id)
datadir = subject.processing_stage_path(proc=proc)
visual_populations = subject.pick_visual_chan()
hfb, visual_chan = hf.load_visual_hfb(sub_id= sub_id, proc= proc, 
                            stage= stage)
hfb = hfb.pick_channels(picks)

#%% Continuous sliding window

ts, time = hf.category_continous_sliding_ts(hfb, tmin=tmin, tmax=tmax,
                                            step=step, win_size=win_size)

#%% Save ts

ts_dict = {'data': ts, 'sfreq': sfreq, 'time': time, 'sub_id': sub_id}
fname = sub_id + '_continuous_sliding_ts.mat'
fpath = datadir.joinpath(fname)

savemat(fpath, ts_dict)