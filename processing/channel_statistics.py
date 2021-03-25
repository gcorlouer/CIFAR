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
(n, m, N, c) = cross_ts.shape

#%% Compute skewness and kurtosis

new_shape = (n, m*N, c)
X = np.reshape(cross_ts, new_shape)
skewness = np.zeros((n,c))
kurtosis = np.zeros((n,c))
for i in range(n):
    for j in range(c):
        a = X[i,:,j]
        skewness[i,j] = stats.skew(a)
        kurtosis[i,j] = stats.kurtosis(a)

#%% Plot skewness, kurtosis
nbin =30
categories = ['Rest', 'Face', 'Place']
skew_xticks = np.arange(-1,1,0.5)
kurto_xticks = np.arange(0,5,1)
sns.set(font_scale=1.6)
f, ax = plt.subplots(2,3)
for i in range(c):
    ax[0,i].hist(skewness[:,i], bins=nbin, density=False)
    ax[0,i].set_xlim(left=-1, right=1)
    ax[0,i].set_ylim(bottom=0, top=35)
    ax[0,i].xaxis.set_ticks(skew_xticks)
    ax[0,i].axvline(x=-0.5, color='k')
    ax[0,i].axvline(x=0.5, color='k')
    ax[0,i].set_xlabel(f'Skewness ({categories[i]})')
    ax[0,i].set_ylabel('Number of channels')
    ax[1,i].hist(kurtosis[:,i], bins=nbin, density=False)
    ax[1,i].set_xlim(left=0, right=5)
    ax[1,i].set_ylim(bottom=0, top=60)
    ax[1,i].axvline(x=1, color='k')
    ax[1,i].xaxis.set_ticks(kurto_xticks)
    ax[1,i].set_xlabel(f'Excess kurtosis ({categories[i]})')
    ax[1,i].set_ylabel('Number of channels')

#%%

skewness, kurtosis = hf.chanel_statistics(cross_ts, nbin=30, fontscale=1.6)