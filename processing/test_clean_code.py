#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 09:23:48 2021

@author: guime
"""
# TODO Make numbers of trials in rest and stimuli the same
import HFB_process as hf
import cifar_load_subject as cf
import numpy as np
import helper_functions as fun
import matplotlib.pyplot as plt
import mne

from scipy.io import savemat
from statsmodels.tsa.tsatools import detrend

# Todo: use time to sample for parametrisation (easier to interpret in time)
#%%

sub_id = 'DiAs'
visual_chan_table = 'visual_channels_BP_montage.csv'
proc = 'preproc' 
sfreq = 250
categories = ['Rest', 'Face', 'Place']
tmin_crop = -0.5
tmax_crop = 1.75
suffix = 'preprocessed_raw'
ext = '.fif'
sample_start = 125
sample_stop = 375
step = 10 
window_size = 40

#%% Load data

subject = cf.Subject(name=sub_id)
datadir = subject.processing_stage_path(proc=proc)
visual_chan = subject.pick_visual_chan()
HFB_visual = hf.visually_responsive_HFB(sub_id = sub_id)

#%% Sliding window time series per category

ts, ts_time = fun.ts_win_cat(HFB_visual, visual_chan, categories,
                             tmin_crop, tmax_crop, sfreq, sample_start,
                             sample_stop, step, window_size)

# %% Detrend ts

# (nchan, nobs, ntrial, nseg, ncat) = ts.shape
# for w in range(nseg):
#     for c in range(ncat):
#         for itrial in range(ntrial):
#             ts[:, :, itrial, w, c] = detrend(ts[:, :, itrial, w, c], order=2, axis=1)

# %% Save time series for mvgc analysis

ts_dict = {'data': ts, 'sfreq': sfreq}
fname = sub_id + '_slided_category_visual_HFB.mat'
fpath = datadir.joinpath(fname)

savemat(fpath, ts_dict)

# %% Plot evoked response of sliding window

iseg = 0
ichan = 0
icat = 2
evoked = np.average(ts, axis = 2)
evoked_seg = evoked[ichan, :, iseg, icat]
time_seg = ts_time[:, iseg]
plt.plot(time_seg, evoked_seg)

#%%
cat = 'Face'

HFB = hf.category_specific_HFB(HFB_visual, cat=cat, tmin_crop = tmin_crop,
                                       tmax_crop=tmax_crop)


HFB = HFB.resample(sfreq=sfreq)

X = HFB.copy().get_data()
time = HFB.times

X_win, time_win = fun.slide_window(X, time, sample_start, sample_stop, step, window_size)


# %%

iwin = 0
ichan = 3

evok = np.average(X[:,ichan, 125:165], axis=0)
evok_win = np.average(X_win[iwin,:,ichan,:], axis=0)

t = time_win[0,:]
#time[125:165]


plt.plot(t, evok)
plt.plot(t, evok_win)








