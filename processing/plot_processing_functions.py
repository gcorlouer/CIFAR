#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:04:25 2021

@author: guime
"""

import HFB_process as hf
import cifar_load_subject as cf
import mne
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path, PurePath


sub_id = 'DiAs'
proc = 'preproc'
fname = sub_id + '_BP_montage_preprocessed_raw.fif'
tmin = 100
tmax = 102
l_freq = 60
band_size=20.0
l_trans_bandwidth= 10.0
h_trans_bandwidth= 10.0
filter_length='auto'
phase='minimum'
ichan = 6
figname = 'HFB_envelope_extraction.jpg'
figpath = Path.home().joinpath('projects','CIFAR','figures', figname)

#%% Load data

subject = cf.Subject()
fpath = subject.processing_stage_path(proc = proc)
fpath = fpath.joinpath(fname)
raw = mne.io.read_raw_fif(fpath, preload=True)
raw = raw.crop(tmin=tmin, tmax=tmax)
times = raw.times

HFB = hf.extract_hfb(raw, l_freq=l_freq, nband=nband, band_size=band_size,
                l_trans_bandwidth= l_trans_bandwidth,h_trans_bandwidth= h_trans_bandwidth,
                filter_length=filter_length, phase=phase)