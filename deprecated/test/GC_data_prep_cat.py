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

# %% Parameters
proc = 'preproc'
ext2save = '.mat'
sub = 'DiAs'
task = 'stimuli'
run = '1'
t_pr = -0.1
t_po = 1.75
cat = 'Place'
suffix2save = 'HFB_visual_epoch_' + cat

# cat_id = extract_stim_id(event_id, cat = cat)
# %% Import data
# Load visual channels
path_visual = cf_load.visual_path()
df_visual = pd.read_csv(path_visual)

# Load data
subject = cf_load.Subject(name=sub, task= task, run = run)
fpath = subject.fpath(proc = proc, suffix='lnrmv')
raw = subject.import_data(fpath)

# %% Preprocess data

# Extract HFB envelope
bands = HFB_test.freq_bands() # Select Bands of interests 
HFB = HFB_test.extract_HFB(raw, bands) # Extract HFB

# Epoch category specific envelope

def category_specifc_epochs(HFB, category):
    # TODO : define function
    events, event_id = mne.events_from_annotations(raw) 
    cat_id = HFB_test.extract_stim_id(event_id, cat = cat)
    epochs = HFB_test.epoch_HFB(HFB, raw, t_pr = t_pr, t_po = t_po)
    epochs = epochs[cat_id].copy()
    return epochs   
 
events, event_id = mne.events_from_annotations(raw) 
cat_id = HFB_test.extract_stim_id(event_id, cat = cat)
epochs = HFB_test.epoch_HFB(HFB, raw, t_pr = t_pr, t_po = t_po)
epochs = epochs[cat_id].copy()

# %% Downsample to 250 Hz

epochs = epochs.resample(sfreq=250)

# %% Set up mat file for GC analysis in matlab 

# Prepare dictionary for GC analysis

def make_visual_chan_dictionary(df_visual, sub='DiAs'): 
    
    visual_chan = list(df_visual['chan_name'].loc[df_visual['subject_id']== sub])
    category = list(df_visual['category'].loc[df_visual['subject_id']== sub])
    brodman = list(df_visual['brodman'].loc[df_visual['subject_id']== sub])
    DK = list(df_visual['DK'].loc[df_visual['subject_id']== sub] )
    data = epochs.get_data(picks=visual_chan)
    # ch_idx = mne.pick_channels(epochs.info['ch_names'], include=visual_chan)
    visual_dict = dict(data=data, chan=visual_chan, 
                   category=category, brodman=brodman, DK = DK)
    return visual_dict 

# Save data for GC analysis

visual_dict = make_visual_chan_dictionary(df_visual, sub=sub)

fpath2save = subject.fpath(proc = proc, 
                            suffix = suffix2save, ext=ext2save)
sp.io.savemat(fpath2save, visual_dict)

# %% For continuous time series:

# nepochs = np.size(data, 0)
# nobs = np.size(data,2)
# nchan = np.size(data,1)
# newshape = (nchan, nepochs*nobs)
# continuous_data = np.reshape(data, newshape)