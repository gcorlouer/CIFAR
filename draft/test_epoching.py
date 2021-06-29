#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:21:10 2021

@author: guime
"""


import HFB_process as hf
import cifar_load_subject as cf
import mne
import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path, PurePath

#%% Parameters
sub_id = 'DiAs'
proc = 'preproc'
stage = '_BP_montage_HFB_raw.fif'
epo = True
t_prestim = -0.5
#%% Load data

subject = cf.Subject(name=sub_id)
hfb= subject.load_data(proc=proc, stage=stage, epo=False)

#%% Test epoching

epochs = hf.epoch_hfb(hfb, t_prestim = -0.5, t_postim = 1.75, baseline=None, preload=True)

epochs.plot_image()

#%%

X= epochs.copy().get_data()
events = epochs.events
event_id = epochs.event_id
del event_id['boundary'] # Drop boundary event
hfb = mne.EpochsArray(X, epochs.info, events=events, 
                             event_id=event_id, tmin=t_prestim)
hfb.plot_image()