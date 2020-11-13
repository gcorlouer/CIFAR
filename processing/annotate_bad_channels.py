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

# %matplotlib
pd.options.display.max_rows = 999
pd.options.display.max_columns = 5

#%%

subjects = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']

proc = 'preproc' # Line noise removed
sub = 'AnRa' 
task = 'rest_baseline' # stimuli or rest_baseline
run = '2'
fs = 500; # resample

suffix2save = 'bad_chans_marked' 
ext2save = '_raw.fif'

# cat_id = extract_stim_id(event_id, cat = cat)
# %% Import data

subject = cifar_load_subject.Subject(name=sub, task= task, run = run)
fpath = subject.dataset_path(proc = proc, suffix='lnrmv')
raw = subject.read_eeglab_dataset(fpath)

dfelec = subject.df_electrodes_info()

# dfelec['electrode_name', 'ROI_DK', 'Brodman'] uncomment to check anatomical location

raw.plot(duration=130, n_channels= 30, scalings = 1e-4) 


#%% Check with spectral analysis of channels

raw.plot_psd(xscale = 'log')

#%% Save resulting dataset

fpath2save = subject.dataset_path(proc = proc, 
                            suffix = suffix2save, ext=ext2save)

raw.save(fpath2save, overwrite=True)


#%% Distribution of average values in 99 percentiles 

# X = raw.get_data()

# high_values = np.percentile(X, 99, axis=1)
# bad = np.zeros_like(high_values)

# for i in range(np.size(high_values)):
#     if high_values[i]>= 700e-6:
#         bad[i]=True
#     else:
#         bad[i]=False    

# # Check which channels are bads according to this criteria

# bad_index = np.where(bad==True)[0]
# bad_index = bad_index.tolist()
# print(raw.info['ch_names'][index] for index in bad_index)
# for i in bad_index:
#     print(raw.info['ch_names'][i])
    


