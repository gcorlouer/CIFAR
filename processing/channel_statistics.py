#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:12:36 2021

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

#%%

subjects = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']
proc= 'preproc' 
stage= '_BP_montage_HFB_raw.fif'
# Pick face selective channel
sfreq = 250
tmin_crop = -0.5
tmax_crop = 1.75

#%% Cross subject time series

ts, time = hf.cross_subject_ts(subjects, proc=proc, stage=stage, sfreq=sfreq,
                                     tmin_crop=tmin_crop, tmax_crop= tmax_crop)
cross_ts = np.concatenate(ts, axis=0)
(n, m, N, p) = cross_ts.shape

#%% Compute skewness and kurtosis

new_shape = (n, m*N, p)
X = np.reshape(cross_ts, new_shape)
skewness = np.zeros((n,p))
kurtosis = np.zeros((n,p))
for i in range(n):
    for j in range(p):
        a = X[i,:,j]
        skewness[i,j] = stats.skew(a)
        kurtosis[i,j] = stats.kurtosis(a)

