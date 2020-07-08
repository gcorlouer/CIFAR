#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:22:22 2020

@author: guime
"""


import HFB_test
import cf_load
import scipy as sp
import re 
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd


preproc = 'preproc'
ext2save = '.mat'
sub = 'DiAs'
task = 'stimuli'
run = '1'
t_pr = -0.75
t_po = 1.5
cat = 'Place'
suffix2save = 'HFB_visual_epoch_' + cat

# cat_id = extract_stim_id(event_id, cat = cat)

path_visual = cf_load.visual_path()
df_visual = pd.read_csv(path_visual)

subject = cf_load.Subject(name=sub, task= task, run = run)
fpath = subject.fpath(preproc = preproc, suffix='lnrmv')
raw = subject.import_data(fpath)

bands = HFB_test.freq_bands() # Select Bands of interests 
HFB = HFB_test.extract_HFB(raw, bands)

events, event_id = mne.events_from_annotations(raw) 
cat_id = HFB_test.extract_stim_id(event_id, cat = cat)
epochs = HFB_test.epoch_HFB(HFB, raw, t_pr = t_pr, t_po = t_po)
epochs = epochs[cat_id].copy()

visual_chan = list(df_visual['chan_name'].loc[df_visual['subject_id']== sub])
category = list(df_visual['category'].loc[df_visual['subject_id']== sub])
brodman = list(df_visual['brodman'].loc[df_visual['subject_id']== sub])
data = epochs.get_data(picks=visual_chan)


ch_idx = mne.pick_channels(raw.info['ch_names'], include=visual_chan)
visual_dict = dict(data=data, chan=visual_chan, 
                   category=category, brodman=brodman, ch_idx=ch_idx)
fpath2save = subject.fpath(preproc = preproc, 
                            suffix = suffix2save, ext=ext2save)
sp.io.savemat(fpath2save, visual_dict)

# %% For continuous time series:

# nepochs = np.size(data, 0)
# nobs = np.size(data,2)
# nchan = np.size(data,1)
# newshape = (nchan, nepochs*nobs)
# continuous_data = np.reshape(data, newshape)