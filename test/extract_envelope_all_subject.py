#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 14:18:38 2020

@author: guime
"""


import HFB_test
import cf_load
import scipy as sp

import mne
import matplotlib.pyplot as plt
import pandas as pd

%matplotlib

plt.rcParams.update({'font.size': 17})


preproc = 'preproc'
suffix2save = 'HFB_visual'
ext2save = '.mat'

cf_subjects = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']
tasks = ['stimuli', 'rest_baseline']
runs = ['1','2']

path_visual = cf_load.visual_path() # pick visual channels for all subjects
df_visual = pd.read_csv(path_visual)

for sub in cf_subjects:
    for task in tasks:
        for run in runs:
            #%% Import data
            subject = cf_load.Subject(name=sub, task= task, run = run)
            fpath = subject.fpath(preproc = preproc, suffix='lnrmv')
            raw = subject.import_data(fpath)
            
            # %% Extract HFB and save
            
            bands = HFB_test.freq_bands() # Select Bands of interests 
            HFB = HFB_test.extract_HFB(raw, bands)
            
            visual_chan = list(df_visual['chan_name'].loc[df_visual['subject_id']== sub])
            category = list(df_visual['category'].loc[df_visual['subject_id']== sub])
            brodman = list(df_visual['brodman'].loc[df_visual['subject_id']== sub])
            data = HFB.get_data(picks=visual_chan)
            ch_idx = mne.pick_channels(raw.info['ch_names'], include=visual_chan)
            visual_dict = dict(data=data, chan=visual_chan,
                               category=category, brodman=brodman, ch_idx=ch_idx)
            fpath2save = subject.fpath(preproc = preproc, 
                                       suffix = suffix2save, ext=ext2save)
            sp.io.savemat(fpath2save, visual_dict)
        