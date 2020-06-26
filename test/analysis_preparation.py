#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:21:32 2020

@author: guime
"""
#%% 
import helper
import mne
import scipy.io
import argparse 

from pathlib import Path, PurePath

#%% 

# Electrodes to pick (ROIs might be better)

picks = ['LGRD60-LGRD61', 'LGRD58-LGRD59','LGRD52-LGRD53', 'LTo7-LTo8', 'LTp3-LTp4', 'LTo5-LTo6', 
         'LTo1-LTo2', 'LTo3-LTo4', 'LGRD50-LGRD51', 'LGRD49-LGRD57']
#picks = ['LTo3-LTo4','LGRD60-LGRD61'] # fusiform

# Parameters 

# Subject and task
subject = 'DiAs'
subject_id = '04'
proc = 'BP'
task = 'stimuli'
run = '1'

# High frequency bands 
l_freq = 60
nband = 6 
band_size = 20 

# electrodes to pick 

picks = ['LTo3-LTo4','LGRD60-LGRD61'] # category selective, face selective

# Epoch parameter

tmin = -0.5 # Prestimulus
tmax = 1.5 # Poststimulus

# Saving paramerters 

save2 = Path('~','projects','CIFAR','data_fun').expanduser()
suffix_save = '_epoch.mat'
suffix_face = '_epoch_face.mat'
suffix_place = '_epoch_place.mat'
#%% 

raw, dfelec = helper.import_data(task=task)
HFB, raw_HFB = helper.HFB(raw, l_freq=l_freq, nband=nband, band_size=band_size);

# Create annotations from initial raw
raw_HFB.set_annotations(raw.annotations)
events, events_id = mne.events_from_annotations(raw_HFB)
# retrieve face and place id
place_id, face_id = helper.stim_id(events_id)
# Epoch data
epochs = mne.Epochs(raw_HFB, events, tmin=tmin, tmax=tmax, picks=picks,  
                    event_repeated='drop', preload=True)

#HFB_db = helper.HFB_norm(epochs, picks, events, tmin)

# Select place and face selective epochs

epoch_face = epochs[face_id]
epoch_place = epochs[place_id]

# Transform in array
epochs_face = epoch_face.get_data()
epochs_place = epoch_place.get_data()
epochs = epochs.get_data()

# Retrieve ROI and index
ROI, ch_index, ch_names = helper.ch_info(picks=picks, dfelec=dfelec, epochs=epoch_face)
# Save data 
fname_place = helper.CIFAR_filename(subid=subject,task=task,proc=proc, suffix=suffix_place)
fname_face = helper.CIFAR_filename(subid=subject,task=task,proc=proc, suffix=suffix_face)
fname_save = helper.CIFAR_filename(subid=subject,task=task,proc=proc, suffix=suffix_save)

fpath_place = save2.joinpath(fname_place)
fpath_face = save2.joinpath(fname_face)
fpath_save = save2.joinpath(fname_save)

scipy.io.savemat(fpath_place, dict(epochs=epochs_place, ch_index=ch_index, ch_names=ch_names, ROI=ROI))
scipy.io.savemat(fpath_face, dict(epochs=epochs_face, ch_index=ch_index, ch_names=ch_names, ROI=ROI))
scipy.io.savemat(fpath_save, dict(epochs=epochs, ch_index=ch_index, ch_names=ch_names, ROI=ROI))
