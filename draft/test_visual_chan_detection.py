#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 17:13:40 2021

@author: guime
"""


import HFB_process as hf
import cifar_load_subject as cf
import mne
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import helper_functions as fun
from scipy import stats

from pathlib import Path, PurePath

#%% Parameters
proc = 'preproc'
sub_id = 'DiAs'
stage = '_hfb_db_epo.fif'
epo = True
tmin = -0.5
tmax = 1.75
tmin_prestim=-0.4
tmax_prestim=-0.1
tmin_postim=0.2
tmax_postim=0.5
alpha=0.05
zero_method='pratt'
alternative='two-sided'
matplotlib.rcParams.update({'font.size': 18})

#%%

subject = cf.Subject(name=sub_id)
hfb_db = subject.load_data(proc=proc, stage=stage, epo=epo)
visual_channels, effect_size = hf.detect_visual_chan(hfb_db, tmin_prestim=tmin_prestim, 
                                              tmax_prestim=-tmax_prestim
                                              ,tmin_postim=tmin_postim,
                       tmax_postim=tmax_postim, alpha=alpha, zero_method=zero_method, 
                       alternative=alternative)
hfb_visual = hfb_db.copy().pick_channels(visual_channels).crop()

#%% Estimate non gaussianity


hfb = hfb_visual.pick_channels(['LTo1-LTo2'])
prestim_amplitude = fun.skew_kurtosis(hfb, tmin = tmin_prestim, tmax = tmax_prestim)
postim_amplitude = fun.skew_kurtosis(hfb, tmin = tmin_postim, tmax = tmax_postim)
#%% Compute evok response

evok_stat = fun.epochs_to_evok_stat(hfb)
times = hfb.times
#%% Plot Histogram and envelope
matplotlib.rcParams.update({'font.size': 20})

sns.set()
nbins = 150
step = 0.5
fontsize=20
figsize=(10,10)
fig, ax = plt.subplots(2,2, figsize=figsize)
fun.plot_evok(evok_stat, times, ax[0,0], tmin=tmin, tmax=tmax, step=step)
ax[0,0].axvline(x=tmin_prestim, color='violet')
ax[0,0].axvline(x=tmax_prestim, color='violet')
ax[0,0].axvline(x=tmin_postim, color='brown')
ax[0,0].axvline(x=tmax_postim, color='brown')
ax[0,0].set_xlabel('Time (s)', fontsize=fontsize)
ax[0,0].set_ylabel('Amplitude (dB)', fontsize=fontsize)
ax[0,0].tick_params(axis='both', labelsize=18)

ax[0,1].hist(postim_amplitude, bins=nbins, density=True, color='brown')
ax[0,1].set_xlabel('Postimulus amplitude (dB)', fontsize=fontsize)
ax[0,1].set_ylabel('Density', fontsize=fontsize)
ax[0,1].tick_params(axis='both', labelsize=18)

ax[1,0].hist(prestim_amplitude, bins=nbins, density=True, color='violet')
ax[1,0].set_xlabel('Prestimulus amplitude (dB)', fontsize=fontsize)
ax[1,0].set_ylabel('Density', fontsize=fontsize)
ax[1,0].tick_params(axis='both', labelsize=18)

ax[1,1].axis('off')
#%%
