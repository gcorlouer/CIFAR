#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 12:06:12 2020

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

from scipy.io import loadmat,savemat


# %matplotlib
pd.options.display.max_rows = 999
pd.options.display.max_columns = 5

#%%

subjects = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']

sub = 'SoGi'

proc = 'preproc' # Line noise removed
fs = 500; # resample

suffix_save = '_bipolar_concatenated_bad_chans_marked' 
ext_save = '_raw.fif'

def concatenate_run_dataset(sub_id = sub, proc='bipolar_montage', task = 'rest_baseline', preload = True):
    
    subject = cifar_load_subject.Subject(name=sub_id, task = task, run='1')
    fpath = subject.dataset_path(proc=proc, suffix='BP_montage', ext='.set')
    raw = mne.io.read_raw_eeglab(fpath, preload=preload)
    raw_1 = raw.copy()
    
         
    subject = cifar_load_subject.Subject(name=sub_id, task = task, run='2')
    fpath = subject.dataset_path(proc=proc, suffix='BP_montage', ext='.set')
    raw = mne.io.read_raw_eeglab(fpath, preload=preload)
    raw_2 = raw.copy()
    
    raw_1.append([raw_2])
    return raw_1
    
    
def concatenate_task_dataset(sub_id = sub):
    
    raw_rest =  concatenate_run_dataset(task='rest_baseline')
    raw_stimuli = concatenate_run_dataset(task='stimuli')
    raw_rest.append([raw_stimuli])
    return raw_rest

#%% Concatenate

raw = concatenate_task_dataset(sub_id = sub)

#%% Mark bad channels

raw.plot(scalings=1e-4, duration =200)
#%% Check psd 

raw.plot_psd(xscale = 'log')

# %% Save dataset in fif and mat

subject = cifar_load_subject.Subject(name=sub)
fpath_save = subject.processing_stage_path(proc='preproc')
dataset_save = sub + '_BP_montage_concatenated_bads_marked_raw.fif'
fname_save = fpath_save.joinpath(dataset_save)
raw.save(fname_save, overwrite=True)

# Drop bad chans and save in mat format

bads = raw.info['bads']
raw_drop_bads = raw.copy().drop_channels(bads)

dataset_save = sub+ '_BP_montage_preprocessed_raw.fif'
fname_save = fpath_save.joinpath(dataset_save)

raw_drop_bads.save(fname_save, overwrite=True)
