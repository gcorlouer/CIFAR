#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 18:16:49 2020

@author: guime
"""

#%% Remove bad chans and save time series as mat file for preprocessing in matlab

import HFB_process
import cifar_load_subject
import scipy as sp
import re 
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd

from scipy.io import savemat
# %matplotlib
pd.options.display.max_rows = 999
pd.options.display.max_columns = 5

#%%

subjects = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']
tasks  = ['rest_baseline','stimuli']
runs = ['1','2']
proc = 'preproc' 
for sub in subjects:
    for task in tasks:
        for run in runs:
            sub = sub
            task = task
            run = run
            
            suffix = 'bad_chans_removed_raw' 
            ext = '.fif'
            
            suffix2save = 'bad_chans_removed' 
            ext2save = '.mat'
            # %% Import data
            
            subject = cifar_load_subject.Subject(name=sub, task= task, run = run)
            fpath = subject.dataset_path(proc = proc, suffix=suffix, ext=ext)
            raw = mne.io.read_raw_fif(fpath, preload=True)
            
            
            dfelec = subject.df_electrodes_info()
            
            # drop bad channels
            
            bads = raw.info['bads']
            raw_drop_bads = raw.copy().drop_channels(bads)
            
            # Extract time series and save to matlab
            
            time_series = raw_drop_bads.get_data()
            
            fpath2save = subject.dataset_path(proc = proc, 
                                        suffix = suffix2save, ext=ext2save)
            time_series = dict(time_series=time_series)
            savemat(fpath2save, time_series)