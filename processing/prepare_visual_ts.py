#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:54:36 2021

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
sfreq = 250
tmin_crop = 0.010
tmax_crop = 1.75

#%%
subject = cf.Subject(sub_id)
datadir = subject.processing_stage_path(proc=proc)
visual_populations = subject.pick_visual_chan()
hfb, visual_chan =hf.load_visual_hfb(sub_id= sub_id, proc= proc, 
                            stage= stage)
ts, time = hf.category_ts(hfb, visual_chan, tmin_crop=tmin_crop, tmax_crop=tmax_crop)


#%% Detrend ts
ts = hf.substract_AERA(ts, axis=2)
#%% Save time series for GC analysis

ts_dict = {'data': ts, 'sfreq': sfreq, 'time': time, 'sub_id': sub_id}
fname = sub_id + '_hfb_visual.mat'
fpath = datadir.joinpath(fname)

savemat(fpath, ts_dict)

