#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:41:28 2021

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
import statsmodels.tsa.vector_ar.var_model as tsa

from pathlib import Path, PurePath
from scipy import stats
from scipy.io import savemat
from statsmodels.tsa.tsatools import detrend
#%%

sub_id= 'DiAs'
proc= 'preproc' 
stage= '_BP_montage_HFB_raw.fif'
picks = ['LTo1-LTo2', 'LTo5-LTo6']
sfreq = 250
tmin_crop = 0
tmax_crop = 1.75

#%%
subject = cf.Subject(sub_id)
datadir = subject.processing_stage_path(proc=proc)
visual_populations = subject.pick_visual_chan()
ts, time = hf.chan_specific_category_ts(picks, sub_id= sub_id, proc= proc, sfreq=sfreq,
                            stage= stage, tmin_crop=tmin_crop, tmax_crop=tmax_crop)
(nchan, nobs, ntrials, ncat) = ts.shape
#%%

X = ts[:,:,2,1]
VAR = tsa.VAR(X)

#%%

lag_results = VAR.select_order(trend='n')