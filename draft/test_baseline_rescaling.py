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
import seaborn as sns
import scipy.stats as stats
import helper_functions as fun


from pathlib import Path, PurePath

#%% Parameters
sub_id = 'DiAs'
proc = 'preproc'
stage = '_BP_montage_HFB_raw.fif'
epo = True
l_freq = 70
band_size=20.0
l_trans_bandwidth= 10.0
h_trans_bandwidth= 10.0
filter_length='auto'
phase='minimum'
ichan = 6
figname = 'hfb_normalisation.jpg'
figpath = Path.home().joinpath('projects','CIFAR','figures', figname)
matplotlib.rcParams.update({'font.size': 20})

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

#%% Plot evoked response
sns.set()
times = epochs.times
evok_stat = fun.epochs_to_evok(epochs)
evok_db_stat = fun.epochs_to_evok(hfb_db)
f, ax = plt.subplots(2,1, sharex=False, figsize=(11,8), tight_layout=True)
plot_evok(evok_stat, ax[0])
plot_evok(evok_db_stat, ax[1])
ax[0].set_ylabel('Amplitude (V)')
ax[1].set_ylabel('Amplitude (dB)')
plt.xlabel('Time(s)')

#%% Save figure
figname = 'baseline_rescaling.jpg'
figpath = Path.home().joinpath('projects','CIFAR','cifar_notes', figname)
plt.savefig(figpath, format='jpeg')