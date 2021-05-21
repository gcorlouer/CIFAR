#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 14:20:26 2021

@author: guime
"""

import cifar_load_subject as cf
import HFB_process as hf
import numpy as np

from scipy.io import loadmat,savemat

# sub_id = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']
# NOTE that visually responsive populations have change for other subjects
#%%

sub_id = 'DiAs'
sfreq = 250
proc = 'preproc'
stage = '_BP_montage_preprocessed_raw.fif'
categories = ['Rest', 'Face', 'Place']
tmin = -0.5
tmax = 1.75

#%% Load data

subject = cf.Subject(name=sub_id)
datadir = subject.processing_stage_path(proc=proc)
visual_chan = subject.pick_visual_chan()

LFP = subject.load_raw_data(proc=proc, stage=stage)
visual_LFP = LFP.copy().pick(visual_chan['chan_name'].tolist())

#%%

def LFP_to_dict(LFP, visual_chan, tmin=-0.5, tmax=1.75, sfreq=250):
    """Return dictionary with all category specific LFP and visual channels
    information"""
    visual_LFP = LFP.copy().pick(visual_chan['chan_name'].tolist())
    LFP_dict = visual_chan.to_dict(orient='list')
    categories = ['Rest', 'Face', 'Place']
    for cat in categories:
        epochs, events = hf.epoch_category(visual_LFP, cat=cat, tmin=-0.5, tmax=1.75)
        epochs = epochs.resample(sfreq)
        X = epochs.copy().get_data()
        X = np.transpose(X, axes=(1, 2, 0))
        LFP_dict[cat] = X
    LFP_dict['time'] = epochs.times
    population_to_channel = hf.parcellation_to_indices(visual_chan, parcellation='group')
    DK_to_channel = hf.parcellation_to_indices(visual_chan, parcellation='DK')
    LFP_dict['population_to_channel'] = population_to_channel
    LFP_dict['DK_to_channel'] = DK_to_channel
    return LFP_dict

# %%

LFP_dict = LFP_to_dict(LFP, visual_chan, tmin=-0.5, tmax=1.75, sfreq=250)

#%% 


LFP_dict = visual_chan.to_dict(orient='list')
for cat in categories:
    epochs, events = hf.epoch_category(visual_LFP, cat=cat, tmin=-0.5, tmax=1.75)
    epochs = epochs.resample(sfreq)
    X = epochs.copy().get_data()
    X = np.transpose(X, axes=(1, 2, 0))
    LFP_dict[cat] = X

LFP_dict['time'] = epochs.times
population_to_channel = hf.parcellation_to_indices(visual_chan, parcellation='group')
DK_to_channel = hf.parcellation_to_indices(visual_chan, parcellation='DK')
LFP_dict['population_to_channel'] = population_to_channel
LFP_dict['DK_to_channel'] = DK_to_channel
#%% Save dictionary

fname = sub_id + '_visual_LFP_all_categories.mat'
fpath = datadir.joinpath(fname)

# Save data in Rest x Face x Place array of time series

savemat(fpath, LFP_dict)
