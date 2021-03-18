#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:04:56 2021

@author: guime
"""

import HFB_process as hf
import cifar_load_subject as cf
import mne
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path, PurePath

#%% Parameters
sub_id = 'DiAs'
proc = 'preproc'
subject = cf.Subject(name=sub_id)
datadir = subject.processing_stage_path(proc=proc)
fname = sub_id + '_hfb_db_epo.fif'
tmin_prestim=-0.4
tmax_prestim=-0.1
tmin_postim=0.1
tmax_postim=0.5
alpha=0.05
zero_method='zsplit'
alternative='greater'

#%% Load data
fpath = datadir.joinpath(fname)
hfb_db = mne.read_epochs(fpath, preload=True)

#%% Test detect_visual_chans on one subject

visual_channels, effect_size = hf.detect_visual_chan(hfb_db, tmin_prestim=tmin_prestim, 
                                                  tmax_prestim=-tmax_prestim
                                                  ,tmin_postim=tmin_postim,
                           tmax_postim=tmax_postim, alpha=alpha, zero_method=zero_method, 
                           alternative=alternative)

hfb_visual = hfb_db.copy().pick_channels(visual_channels).crop()
A_prestim = hfb_visual.copy().crop(tmin=tmin_prestim, tmax=tmax_prestim).get_data()
A_prestim = np.ndarray.flatten(A_prestim)
A_postim = hfb_visual.copy().crop(tmin=tmin_postim, tmax=tmax_postim).get_data()
A_postim = np.ndarray.flatten(A_postim)


#%% Test visual channel detection for all subjects

subjects = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']
postim_amplitude_list = [0]*len(subjects)
prestim_amplitude_list = [0]*len(subjects)
for i, sub_id in enumerate(subjects):
    visual_channels, effect_size = hf.detect_visual_chan(hfb_db, tmin_prestim=tmin_prestim, 
                                                  tmax_prestim=-tmax_prestim
                                                  ,tmin_postim=tmin_postim,
                           tmax_postim=tmax_postim, alpha=alpha, zero_method=zero_method, 
                           alternative=alternative)
    hfb_visual = hfb_db.copy().pick_channels(visual_channels).crop()
    A_prestim = hfb_visual.copy().crop(tmin=tmin_prestim, tmax=tmax_prestim).get_data()
    A_prestim = np.ndarray.flatten(A_prestim)
    A_postim = hfb_visual.copy().crop(tmin=tmin_postim, tmax=tmax_postim).get_data()
    A_postim = np.ndarray.flatten(A_postim)
    prestim_amplitude_list[i] = A_prestim
    postim_amplitude_list[i] = A_postim

prestim_amplitude = np.stack(prestim_amplitude_list)
prestim_amplitude = np.ndarray.flatten(prestim_amplitude)
postim_amplitude = np.stack(postim_amplitude_list)
postim_amplitude = np.ndarray.flatten(A_postim)

data = {'prestimulus': prestim_amplitude, 'postimulus': postim_amplitude}
#%% Plot pre stim and postim amplitude distributions

x=[prestim_amplitude, postim_amplitude]
y=['presitmulus', 'postimulus']
