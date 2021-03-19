#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 17:44:49 2021

@author: guime
"""


import HFB_process as hf
import cifar_load_subject as cf
import mne
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path, PurePath

#%% Parameters
sub_id = 'DiAs'
proc = 'preproc'
stage = '_BP_montage_HFB_raw.fif'
epo = True
tmin = 100
tmax = 102
l_freq = 70
band_size=20.0
l_trans_bandwidth= 10.0
h_trans_bandwidth= 10.0
filter_length='auto'
phase='minimum'
ichan = 6
figname = 'hfb_normalisation.jpg'
figpath = Path.home().joinpath('projects','CIFAR','figures', figname)

#%% Load data

subject = cf.Subject(name=sub_id)
hfb= subject.load_data(proc=proc, stage=stage, epo=False)
hfb = hfb.pick_channels(['LTo1-LTo2'])
#%% Test epoching

epochs = hf.epoch_hfb(hfb, t_prestim = -0.5, t_postim = 1.75, baseline=None, preload=True)

epochs.plot_image()
A = epochs.copy().get_data()

#%% Test baseline extraction

baseline = hf.extract_baseline(epochs, tmin=-0.400, tmax=-0.100)

#%% Test db transformwith MNE
units = dict(eeg='dB')
hfb_db = hf.db_transform(epochs, tmin=-0.4, tmax=-0.1, t_prestim=-0.5)
hfb_db.plot_image(units=units, scalings =1, combine='median')

#%% Test with extract baseline

# baseline = hf.extract_baseline(epochs)
# events = epochs.events
# event_id = epochs.event_id
# del event_id['boundary'] # Drop boundary event
# A = 10*np.log(np.divide(A, baseline[np.newaxis,:,np.newaxis]))
# hfb_db = mne.EpochsArray(A, epochs.info, events=events, 
#                              event_id=event_id, tmin=-0.5)
# hfb_db.plot_image(scalings=1)
