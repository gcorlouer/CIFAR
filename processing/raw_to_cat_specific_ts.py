#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:22:22 2020

@author: guime
"""


import HFB_process
import cifar_load_subject
import scipy as sp
import re 
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path, PurePath
from scipy.io import savemat

# %matplotlib


# %% Parameters
proc = 'preproc' # Line noise removed
sub = 'DiAs' 
task = 'rest_baseline' # stimuli or rest_baseline
cat = 'Rest' # Face or Place if task=stimuli otherwise cat=Rest
run = '1'
suffix = 'preprocessed_raw'
ext = '.fif'
fs = 250; # resample
nband=6

duration = 10 # Event duration for resting state epoching
t_pr = -0.01
t_po = 1.5
suffix2save = 'HFB_visual_epoch_' + cat
ext2save = '.mat'
# cat_id = extract_stim_id(event_id, cat = cat)
# %% Import data
# Load visual channels

# Load data
subject = cifar_load_subject.Subject(name=sub, task= task, run = run)
fpath = subject.dataset_path(proc = proc, suffix=suffix, ext=ext)
raw = mne.io.read_raw_fif(fpath, preload=True)
brain_path = subject.brain_path()
visual_chan_path = brain_path.joinpath('visual_channels.csv')
df_visual = pd.read_csv(visual_chan_path)
# %% Preprocess data

# Extract HFB envelope
bands = HFB_process.freq_bands(nband=nband) # Select Bands of interests 
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

# %% Set up mat file for GC analysis in matlab 

# Prepare dictionary for GC analysis


channels_group = df_visual.to_dict(orient='list')
visual_chans = channels_group['chan_name']
chans_group = channels_group['group']
groups = list(set(chans_group))
group_to_chan = []
group_to_chan_dict = dict()
for group in groups:
    group_to_chan = [i for i, x in enumerate(chans_group) if x == group]
    group_to_chan_dict[group] = group_to_chan
multitrial_ts = HFB_process.log_transform(epochs, picks=visual_chans)
# Save data for GC analysis

visual_populations = dict(channels_group, groups=groups, multitrial_ts=multitrial_ts)
visual_populations.update(group_to_chan_dict)

fpath2save = subject.dataset_path(proc = proc, 
                            suffix = suffix2save, ext=ext2save)
sp.io.savemat(fpath2save, visual_populations)

