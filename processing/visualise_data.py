#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 11:50:54 2020

@author: guime
"""


import HFB_process
import cf_load
import scipy as sp
import re 
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd

# %matplotlib

# %% Parameters
proc = 'preproc' # Line noise removed
ext2save = '.mat'
sub = 'DiAs' 
task = 'stimuli' # stimuli or rest_baseline_1
cat = 'Place' # Face or Place if task=stimuli otherwise cat=Rest
run = '1'
fs = 250; # resample
duration = 10 # Event duration for resting state
t_pr = -0.01
t_po = 1.5
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
bands = HFB_process.freq_bands() # Select Bands of interests 
HFB = HFB_process.extract_HFB(raw, bands) # Extract HFB

# Epoch  envelope, keeping specific category only

epochs = HFB_process.epoch(HFB, raw, task=task,
                            cat=cat, duration=duration, t_pr = t_pr, t_po = t_po)


#%% 
# Resample
 
epochs = epochs.resample(sfreq=fs)

raw = raw.resample(sfreq=fs)

HFB = HFB.resample(sfreq=fs)

events, event_id = mne.events_from_annotations(raw) # adapt events to sampling rate

#%% Prepare dictionary for GC analysis

visual_dict = HFB_process.make_visual_chan_dictionary(df_visual, raw, HFB, epochs, sub=sub)

# %% Visualise data


