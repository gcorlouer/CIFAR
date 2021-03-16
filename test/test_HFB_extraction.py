#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:56:13 2021

@author: guime
"""

import HFB_process as hf
import cifar_load_subject as cf
import mne
import matplotlib as plt

from pathlib import Path, PurePath

#%% Parameters
sub_id = 'DiAs'
proc = 'preproc'
fname = sub_id + '_BP_montage_preprocessed_raw.fif'
tmin = 100
tmax = 110
l_freq = 70
band_size=20.0
l_trans_bandwidth= 10.0
h_trans_bandwidth= 10.0
filter_length='auto'
phase='minimum'
ichan = 5
#%% Load data

subject = cf.Subject()
fpath = subject.processing_stage_path(proc = proc)
fpath = fpath.joinpath(fname)
raw = mne.io.read_raw_fif(fpath, preload=True)
raw = raw.crop(tmin=tmin, tmax=tmax)
times = raw.times
#%% Test bands

bands = hf.freq_bands()

#%% Test envelope extraction
envelope = hf.extract_envelope(raw, l_freq = 70, band_size=20.0, l_trans_bandwidth= 10.0, 
                     h_trans_bandwidth= 10.0, filter_length='auto', phase='minimum')
# Filter raw data
raw_filt = raw.copy().filter(l_freq=l_freq, h_freq=l_freq+band_size,
                                 phase=phase, filter_length=filter_length,
                                 l_trans_bandwidth= l_trans_bandwidth, 
                                 h_trans_bandwidth= h_trans_bandwidth,
                                     fir_window='blackman')
LFP = raw_filt.copy().get_data()
# Check one channel envelope extraction over a time segment
plt.plot(times, LFP[ichan, :])
plt.plot(times, envelope[ichan, :])