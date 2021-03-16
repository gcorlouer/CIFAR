#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:56:13 2021

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
figname = 'HFB_envelope_extraction.jpg'
figpath = Path.home().joinpath('projects','CIFAR','figures', figname)
#%% Load data

subject = cf.Subject()
fpath = subject.processing_stage_path(proc = proc)
fpath = fpath.joinpath(fname)
raw = mne.io.read_raw_fif(fpath, preload=True)
raw = raw.crop(tmin=tmin, tmax=tmax)
times = raw.times
#%% Test bands

bands = hf.freq_bands()

#%% Test HFB extraction

HFA =  hf.extract_envelope(raw, l_freq = 60, band_size=100.0, l_trans_bandwidth= 10.0, 
                     h_trans_bandwidth= 10.0, filter_length='auto', phase='minimum')

HFB = hf.extract_HFB(raw, l_freq=60.0, nband=6, band_size=20.0,
                l_trans_bandwidth= 10.0,h_trans_bandwidth= 10.0,
                filter_length='auto', phase='minimum')

raw_filt = raw.copy().filter(l_freq=60, h_freq=160,
                                 phase=phase, filter_length=filter_length,
                                 l_trans_bandwidth= l_trans_bandwidth, 
                                 h_trans_bandwidth= h_trans_bandwidth,
                                     fir_window='blackman')

LFP = raw.copy().get_data()*1e6
LFP_filt = raw_filt.copy().get_data()*1e6
HFB = HFB.copy().get_data()*1e6
HFA = HFA * 1e6

#%% Plot result
# Check one channel envelope extraction over 2s time segment
matplotlib.rcParams.update({'font.size': 18})

fig, ax = plt.subplots(3,1)
ax[0].plot(times, LFP_filt[ichan, :], label='Bandpass LFP')
ax[0].plot(times, HFA[ichan, :], label = 'HFB')
ax[0].legend()
ax[0].set_ylabel('Amplitude (muV)')
ax[1].plot(times, HFB[ichan, :], label='1/f corrected HFB')
ax[1].legend()
ax[1].set_ylabel('Amplitude (muV)')
ax[2].plot(times, LFP[ichan, :], label='LFP')
ax[2].legend()
ax[2].set_ylabel('Potential (muV)')
plt.xlabel('time (s)')
plt.savefig(figpath)
