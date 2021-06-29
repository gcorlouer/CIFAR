#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:19:14 2021

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
picks = ['LGRD58-LGRD59','LTo1-LTo2', 'LTo5-LTo6']
sfreq = 250
tmin_crop = 0
tmax_crop = 1

#%%
subject = cf.Subject(sub_id)
datadir = subject.processing_stage_path(proc=proc)
visual_populations = subject.pick_visual_chan()
ts, time = hf.chan_specific_category_ts(picks, sub_id= sub_id, proc= proc, 
                            stage= stage, tmin_crop=tmin_crop, tmax_crop=tmax_crop)
trend = np.average(ts, axis=2)
trend = trend[:,:,1]

trend_dict = {'data': trend, 'sfreq': sfreq, 'time': time, 'sub_id': sub_id}
fname = sub_id + '_trend.mat'
fpath = datadir.joinpath(fname)

savemat(fpath, trend_dict)

#%%

plt.plot(time, trend[2,:])