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

from pathlib import Path, PurePath

#%% Parameters
sub_id = 'DiAs'
proc = 'preproc'
fname = sub_id + '_BP_montage_preprocessed_raw.fif'
tmin = 100
tmax = 102
l_freq = 70
band_size=20.0
l_trans_bandwidth= 10.0
h_trans_bandwidth= 10.0
filter_length='auto'
phase='minimum'
ichan = 6
figname = 'HFB_normalisation.jpg'
figpath = Path.home().joinpath('projects','CIFAR','figures', figname)

#%% Load data

subject = cf.Subject()
fpath = subject.processing_stage_path(proc = proc)
fpath = fpath.joinpath(fname)
raw = mne.io.read_raw_fif(fpath, preload=True)

#%%

HFB = hf.extract_HFB(raw, l_freq=60.0, nband=6, band_size=20.0,
                l_trans_bandwidth= 10.0,h_trans_bandwidth= 10.0,
                filter_length='auto', phase='minimum')

#%% Test epoching

epochs = hf.epoch_HFB(HFB, t_pr = -0.5, t_po = 1.75, baseline=None, preload=True)

epochs.plot_image()

#%% Test baseline extraction

baseline = hf.extract_baseline(epochs, tmin=-0.400, tmax=-0.100)

#%% Test db transform

HFB_db = hf.db_transform(epochs, tmin=-0.2, tmax=-0.050, t_pr=-0.5)
HFB_db.plot_image(scalings=1)