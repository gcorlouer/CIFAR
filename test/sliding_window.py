#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:25:43 2020

@author: guime
"""
#%% 
import helper
import mne
import scipy.io
import argparse 

from pathlib import Path, PurePath

#%% 


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

picks = ['LGRD60-LGRD61', 'LGRD58-LGRD59','LGRD52-LGRD53', 'LTo7-LTo8', 'LTp3-LTp4', 'LTo5-LTo6', 
         'LTo1-LTo2', 'LTo3-LTo4', 'LGRD50-LGRD51', 'LGRD49-LGRD57']
#picks = ['LTo3-LTo4','LGRD60-LGRD61'] # category selective, face selective

# Epoch parameter

tmin = -0.1 # Prestimulus
tmax = 2 # Poststimulus
start = 0
stop = 200 
duration = 2.
overlap = 0.5
# Saving paramerters 

save2 = Path('~','projects','CIFAR','data_fun').expanduser()
suffix = '_sliding.mat'

#%%

raw, dfelec = helper.import_data(task=task)
HFB, raw_HFB = helper.HFB(raw, l_freq=l_freq, nband=nband, band_size=band_size);

events = mne.make_fixed_length_events(raw_HFB, start=start, stop=stop, duration=duration,
                                      overlap=overlap)

epochs = mne.Epochs(raw_HFB, events, tmin=tmin, tmax=tmax, picks=picks, preload=True)

# Retrieve ROI and index
ROI, ch_index, ch_names = helper.ch_info(picks=picks, dfelec=dfelec, epochs=epochs)

epochs = epochs.get_data()

# Save data
fname = helper.CIFAR_filename(subid=subject,task=task,proc=proc, suffix=suffix)
fpath_save = save2.joinpath(fname)

scipy.io.savemat(fpath_save, dict(epochs=epochs, ch_index=ch_index, ch_names=ch_names, ROI=ROI))

