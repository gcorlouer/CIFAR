#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 17:44:39 2020

@author: guime
"""



import HFB_test
import cf_load
import scipy as sp

import mne
import matplotlib.pyplot as plt
import pandas as pd

# Parameters



preproc = 'preproc'
suffix2save = 'HFB_visual_epoch'
ext2save = '.mat'
sub = 'DiAs'
task = 'stimuli'
run = '1'
t_pr = -0.5
t_po = 2
duration = 2; # events duration

path_visual = cf_load.visual_path()
df_visual = pd.read_csv(path_visual)

cf_subjects = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']
tasks = ['stimuli', 'rest_baseline']
runs = ['1','2']

for sub in cf_subjects:
    for task in tasks:
        for run in runs:
            #%% 
            subject = cf_load.Subject(name=sub, task= task, run = run)
            fpath = subject.fpath(preproc = preproc, suffix='lnrmv')
            raw = subject.import_data(fpath)
            bands = HFB_test.freq_bands() # Select Bands of interests 
            HFB = HFB_test.extract_HFB(raw, bands)
            epochs = HFB_test.epoch_HFB(HFB, raw, t_pr = t_pr, t_po = t_po)
            visual_chan = list(df_visual['chan_name'].loc[df_visual['subject_id']== sub])
            category = list(df_visual['category'].loc[df_visual['subject_id']== sub])
            brodman = list(df_visual['brodman'].loc[df_visual['subject_id']== sub])
            data = epochs.get_data(picks=visual_chan)
            ch_idx = mne.pick_channels(raw.info['ch_names'], include=visual_chan)
            visual_dict = dict(data=data, chan=visual_chan, 
                               category=category, brodman=brodman, ch_idx=ch_idx)
            fpath2save = subject.fpath(preproc = preproc, 
                                        suffix = suffix2save, ext=ext2save)
            sp.io.savemat(fpath2save, visual_dict)
            