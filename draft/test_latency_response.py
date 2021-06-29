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
from statsmodels.stats.multitest import fdrcorrection, multipletests

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

#%% Detect visual channels

subject = cf.Subject(name=sub_id)
hfb_db = subject.load_data(proc=proc, stage=stage, epo=epo)
visual_channels, effect_size = hf.detect_visual_chan(hfb_db, tmin_prestim=tmin_prestim, 
                                              tmax_prestim=-tmax_prestim
                                              ,tmin_postim=tmin_postim,
                       tmax_postim=tmax_postim, alpha=alpha, zero_method=zero_method, 
                       alternative=alternative)
hfb_visual = hfb_db.copy().pick_channels(visual_channels).crop()
hfb = hfb_visual.copy().pick_channels(['LTo1-LTo2'])
event_id = hfb.event_id
face_id = hf.extract_stim_id(event_id, cat = 'Face')
place_id = hf.extract_stim_id(event_id, cat='Place')
image_id = face_id+place_id
times = hfb_visual.times

#%%

hfb_postim = hfb_visual.copy().crop(tmin=0, tmax=1.5)
time_postim = hfb_postim.times

pval_serie = hf.pval_series(hfb_visual, image_id, visual_channels, alpha = alpha)
latency = hf.compute_latency(hfb_visual, image_id, visual_channels, alpha = alpha)

evok_stat = fun.epochs_to_evok(hfb)


#%%
ichan = 3
fontsize = 20
latency_chan = 1e-3*latency[ichan]
sns.set()
f, ax = plt.subplots(2,1)
fun.plot_evok(evok_stat, times, ax[0], tmin=-0.5, tmax=1.75, step=0.2)
ax[0].axvline(x=latency_chan, color='brown')
ax[0].set_xlabel('Time (s)', fontsize=fontsize)
ax[0].set_ylabel('Amplitude (dB)', fontsize=fontsize)
ax[0].tick_params(axis='both', labelsize=18)

ax[1].plot(time_postim, pval_serie[1][ichan,:])
ax[1].axhline(y=alpha, label='alpha', color='brown')
ax[1].set_xlabel('Time (s)', fontsize=fontsize)
ax[1].set_ylabel('p-value', fontsize=fontsize)
ax[1].tick_params(axis='both', labelsize=18)
