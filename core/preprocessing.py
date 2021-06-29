#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 12:23:38 2020

@author: guime
"""

# This script drop the bad channels from BP montage and save the resulting 
# data in preprecocessed_raw file which is LFP of bipolar montage. The raw extension
# is there for MNE compatibility.

import HFB_process
import cifar_load_subject
import scipy as sp
import re 
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd

from scipy.io import loadmat

# %matplotlib
pd.options.display.max_rows = 999
pd.options.display.max_columns = 5

#%%

subjects = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']
tasks  = ['rest_baseline','stimuli']
runs = ['1','2']
suffix = 'preprocessed' 
ext = '.mat'

suffix_bads = 'bad_chans_removed_raw' 
ext_bads = '.fif'

suffix2save = 'preprocessed_raw'
ext2save = '.fif'

proc = 'preproc' # Line noise removed
for sub in subjects:
    for task in tasks:
        for run in runs:
            sub = sub 
            task = task # stimuli or rest_baseline
            run = run
            
            #%% 
            
            # Load preprocessed data
            subject = cifar_load_subject.Subject(name=sub, task= task, run = run)
            fpath = subject.dataset_path(proc = proc, suffix=suffix, ext=ext)
            dataset = loadmat(fpath)
            preprocessed_signal = dataset['preprocessed_signal']
            X = preprocessed_signal[0][0][0]
            
            #Load raw with marked bad channels
            
            fpath_bads = subject.dataset_path(proc = proc, suffix=suffix_bads, ext=ext_bads)
            raw = mne.io.read_raw_fif(fpath_bads, preload=True)
            bads = raw.info['bads']
            raw_drop_bads = raw.copy().drop_channels(bads)
            
            raw_preproc = mne.io.RawArray(X, raw_drop_bads.info)
            raw_preproc.set_annotations(raw_drop_bads.annotations) # Annotate new raw structure
            #raw_preproc = raw_drop_bads # uncomment if only want to remove bads channels
            #%%
            
            #raw_drop_bads.plot(duration=130, n_channels= 30, scalings = 1e-4) 
            
            #%% Save resulting preprocessed dataset
            
            fpath2save = subject.dataset_path(proc = proc, 
                                        suffix = suffix2save, ext=ext2save)
            
            raw_preproc.save(fpath2save, overwrite=True)
