#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 17:20:48 2021

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

#%%

sub_id= 'DiAs'
proc= 'preproc' 
stage= '_BP_montage_preprocessed_raw.fif'
picks = ['LTo1-LTo2', 'LTo5-LTo6']

sfreq = 250
tmin = 0
tmax = 1.75
win_size = 0.200
step = 0.020
detrend = False

#%% Load data
subject = cf.Subject(sub_id)
datadir = subject.processing_stage_path(proc=proc)
visual_populations = subject.pick_visual_chan()

#%%

ts, time = hf.sliding_lfp(picks, proc=proc, stage=stage, sub_id=sub_id,
               tmin=tmin, tmax=tmax, win_size=win_size, step = step, detrend=detrend, sfreq=sfreq)

#%%

ts_dict = {'data': ts, 'sfreq': sfreq, 'time': time, 'sub_id': sub_id}
fname = sub_id + '_ts_sliding.mat'
fpath = datadir.joinpath(fname)

savemat(fpath, ts_dict)
