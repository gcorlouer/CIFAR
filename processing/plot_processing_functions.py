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

#%% Plot HFB extraction

subject = hf.Subject()
fpath = subject.processing_stage_path(proc = proc)
fpath = fpath.joinpath(fname)
raw = mne.io.read_raw_fif(fpath, preload=True)
raw = raw.crop(tmin=tmin, tmax=tmax)
times = raw.times
raw_filt = raw.copy().filter(l_freq=60, h_freq=80,
                                 phase=phase, filter_length=filter_length,
                                 l_trans_bandwidth= l_trans_bandwidth, 
                                 h_trans_bandwidth= h_trans_bandwidth,
                                     fir_window='blackman')
lfp_filt = raw_filt.copy().get_data()*1e6
hfb = hf.Hfb()
hfb = hfb.extract_envelope(raw)
hfb = hfb*1e6


matplotlib.rcParams.update({'font.size': 18})
sns.set()

plt.plot(times, hfb[ichan, :], label='HFB amplitude')
plt.plot(times, lfp_filt[ichan, :], label='LFP')

#%%

