#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:24:48 2021

This script take raw iEEG, extract, epochs, normalise hfb of stimuli iEEG in decibel
and save it in a epo.fif file.

@author: guime
"""


import HFB_process as hf
import cifar_load_subject as cf
import mne

#%% Parameters
subjects = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']
proc = 'preproc' 
l_freq = 60.0
nband = 6
band_size = 20.0
t_prestim = -0.5
t_postim = 1.75
l_trans_bandwidth= 10.0
h_trans_bandwidth= 10.0, 
filter_length='auto'
phase='minimum'
baseline=None
preload=True
tmin=-0.4
tmax=-0.1
mode='logratio'
#%%

for sub_id in subjects:
    subject = cf.Subject(name=sub_id)
    datadir = subject.processing_stage_path(proc=proc)
    fname = sub_id + '_BP_montage_preprocessed_raw.fif'
    fpath = datadir.joinpath(fname)
    raw = mne.io.read_raw_fif(fpath, preload=True)
    hfb_db = hf.raw_to_hfb_db(raw, l_freq=l_freq, nband=nband, band_size=band_size,
                              t_prestim = t_prestim, 
                              l_trans_bandwidth=l_trans_bandwidth, 
                              h_trans_bandwidth= h_trans_bandwidth, 
                              filter_length=filter_length, phase=phase, 
                              t_postim = t_postim, mode=mode,
                              baseline=baseline, preload=preload, tmin=tmin, tmax=tmax)
    fname = sub_id + '_hfb_db_epo.fif'
    fpath = datadir.joinpath(fname)
    hfb_db.save(fpath, overwrite = True)