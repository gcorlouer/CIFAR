#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 19:35:35 2020

@author: guime
"""

import HFB_process as hf
import cifar_load_subject as  cf
import scipy as sp
import re 
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path, PurePath
from mne.viz import plot_filter, plot_ideal_filter
from scipy import signal, fftpack
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection, multipletests

from scipy.io import loadmat,savemat

# %matplotlib

pd.options.display.max_rows = 999

sub = 'DiAs'

proc = 'preproc' 
sfreq = 500; 
suffix = 'preprocessed_raw'
ext = '.fif'

latency_threshold = 160

#%% Extract and save HFB visual

subject = cf.Subject(name=sub)
datadir = subject.processing_stage_path(proc=proc)
fname = sub + '_BP_montage_HFB_raw.fif'
fpath = datadir.joinpath(fname)
raw = mne.io.read_raw_fif(fpath, preload=True)

# Pick visual HFB

brain_path = subject.brain_path()
fname = 'visual_channels_BP_montage.csv'
fpath = brain_path.joinpath(fname)
visual_chan = pd.read_csv(fpath)

# Drop unwanted channels

visual_chan = visual_chan[visual_chan.group != 'other']
visual_chan = visual_chan.reset_index(drop=True)
visual_chan_name = visual_chan['chan_name'].values.tolist()
group = visual_chan['group'].unique().tolist()

HFB_visual = raw.copy().pick_channels(visual_chan_name)

# Epoch resting state

events_1 = mne.make_fixed_length_events(HFB_visual, id=32, start=70, stop=200, duration=2, first_samp=False, overlap=0.0)
events_2 = mne.make_fixed_length_events(HFB_visual, id=32, start=280, stop=400, duration=2, first_samp=False, overlap=0.0)

rest_events = np.concatenate((events_1,events_2))
rest_id = {'Rest': 32}

# Return Face and Place id
stim_events, stim_events_id = mne.events_from_annotations(HFB_visual)
face_id = hf.extract_stim_id(stim_events_id, cat = 'Face')
place_id = hf.extract_stim_id(stim_events_id, cat = 'Place')


# Epoch HFB relative to rest, face and place events.

epochs_rest = mne.Epochs(HFB_visual, rest_events, event_id= rest_id, 
                        tmin=-0.5, tmax=1.75, baseline= None, preload=True)

epochs_stim = mne.Epochs(HFB_visual, stim_events, event_id= stim_events_id, 
                        tmin=-0.5, tmax=1.75, baseline= None, preload=True)

epochs_face = epochs_stim[face_id]
events_face = epochs_face.events
epochs_place = epochs_stim[place_id]
events_place = epochs_place.events

# db transform
HFB_db_rest = hf.db_transform_cat(epochs_rest, rest_events, tmin=-0.4, tmax=-0.1, t_pr=-0.5)
HFB_db_face = hf.db_transform_cat(epochs_face, events_face, tmin=-0.4, tmax=-0.1, t_pr=-0.5)
HFB_db_place = hf.db_transform_cat(epochs_place, events_place, tmin=-0.4, tmax=-0.1, t_pr=-0.5)


# Extract postimulus amplitude

HFB_db_rest = HFB_db_rest.crop(tmin=0.5, tmax=1.75)
HFB_db_face = HFB_db_face.crop(tmin=0.5, tmax=1.75)
HFB_db_place = HFB_db_place.crop(tmin=0.5, tmax=1.75)

# Get data and permute into hierarchical order

population_indices = dict.fromkeys(group)
for key in group:
    population_indices[key] = visual_chan[visual_chan['group'] == key].index.to_list()

visual_hierarchy = ['V1', 'V2', 'Place', 'Face']
permuted_population_indices = dict.fromkeys(visual_hierarchy)
permuted_indices = []

# Find index permutation
for key in permuted_population_indices:
    if key in group:
       permuted_population_indices[key] = population_indices[key]
       permuted_indices.extend(population_indices[key])
    else:
        permuted_population_indices[key] = []
    
for key in permuted_population_indices:
    if key in group:
        for idx, i in enumerate(permuted_population_indices[key]):
            permuted_population_indices[key][idx] = permuted_indices.index(i)
    else: 
        continue 

X_rest = HFB_db_rest.get_data()
X_rest_permuted = np.zeros_like(X_rest)

for idx, i in enumerate(permuted_indices):
    X_rest_permuted[:,idx,:] = X_rest[:,i,:]
    
# Permute indices:

# Save data in Rest x Face x Place array of time series

populations = permuted_population_indices
cat = 'Rest'
visual_data = dict(data= X_rest, populations=populations)
fname = sub + '_' + cat + '_' + 'visual_HFB.mat'
fpath = datadir.joinpath(fname)

savemat(fpath, visual_data)

#%% 
def epoch_cat_HFB(HFB_visual, cat='Rest', tmin=-0.5, tmax=1.75):
    if cat == 'Rest':
        events_1 = mne.make_fixed_length_events(HFB_visual, id=32, start=70, 
                                                stop=200, duration=2, first_samp=False, overlap=0.0)
        events_2 = mne.make_fixed_length_events(HFB_visual, id=32, 
                                                start=280, stop=400, duration=2, first_samp=False, overlap=0.0)
        
        events = np.concatenate((events_1,events_2))
        rest_id = {'Rest': 32}
        # epoch
        epochs= mne.Epochs(HFB_visual, events, event_id= rest_id, 
                            tmin=tmin, tmax=tmax, baseline= None, preload=True)
    else:
        stim_events, stim_events_id = mne.events_from_annotations(HFB_visual)
        cat_id = hf.extract_stim_id(stim_events_id, cat = cat)
        epochs = epochs_stim[cat_id]
        events = epochs_face.events
    return epochs, events

def HFB_to_visual(HFB_visual, group, visual_chan, cat='Rest', tmin_crop = 0.5, tmax_crop=1.75) :
        epochs, events = epoch_cat_HFB(HFB_visual, cat=cat, tmin=-0.5, tmax=1.75)
        HFB = hf.db_transform_cat(epochs, events, tmin=-0.4, tmax=-0.1, t_pr=-0.5)
        HFB = HFB.crop(tmin=tmin_crop, tmax=tmax_crop)
        # Get data and permute into hierarchical order

        population_indices = dict.fromkeys(group)
        for key in group:
            population_indices[key] = visual_chan[visual_chan['group'] == key].index.to_list()
        
        visual_hierarchy = ['V1', 'V2', 'Place', 'Face']
        permuted_population_indices = dict.fromkeys(visual_hierarchy)
        permuted_indices = []
        
        # Find index permutation
        for key in permuted_population_indices:
            if key in group:
               permuted_population_indices[key] = population_indices[key]
               permuted_indices.extend(population_indices[key])
            else:
                permuted_population_indices[key] = []
            
        for key in permuted_population_indices:
            if key in group:
                for idx, i in enumerate(permuted_population_indices[key]):
                    permuted_population_indices[key][idx] = permuted_indices.index(i)
            else: 
                continue 
        
        X = HFB_db_rest.get_data()
        X_permuted = np.zeros_like(X)
        
        for idx, i in enumerate(permuted_indices):
            X_permuted[:,idx,:] = X[:,i,:]
            
        visual_data = dict(data= X_permuted, populations=permuted_population_indices)
        
        return visual_data

cats = ['Rest', 'Place', 'Face']

visual_data = HFB_to_visual(HFB_visual, group, visual_chan, cat='Face', tmin_crop = 0.5, tmax_crop=1.75)

#%%
fname = sub + '_visual_HFB.mat'
fpath = datadir.joinpath(fname)

# Save data in Rest x Face x Place array of time series

savemat(fpath, visual_data)