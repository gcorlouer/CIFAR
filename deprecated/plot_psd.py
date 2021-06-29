#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 13:24:28 2021
This script plot power spectral densisty function
@author: guime
"""


import HFB_process as hf
import cifar_load_subject as cf
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import helper_functions as fun

from pathlib import Path, PurePath
from scipy import stats
from scipy.io import savemat
from statsmodels.tsa.tsatools import detrend
#%%

sub_id= 'DiAs'
proc= 'preproc' 
stage_hfb = '_BP_montage_HFB_raw.fif'
stage_lfp = '_BP_montage_preprocessed_raw.fif'
picks = ['LTo1-LTo2','LTo5-LTo6']
sfreq = 250
fmin =  0.1
fmax = 200

#%%

subject = cf.Subject(sub_id)
datadir = subject.processing_stage_path(proc=proc)
visual_populations = subject.pick_visual_chan()
hfb, visual_chan = hf.load_visual_hfb(sub_id= sub_id, proc= proc, 
                            stage= stage_hfb)
hfb = hfb.pick_channels(picks)

lfp = subject.load_data(proc=proc, stage=stage_lfp)
lfp = lfp.pick_channels(picks)

#%%

start = 60
stop = 325
psd_rest_hfb, freqs = fun.hfb_to_psd(hfb, start=start, stop=stop, 
                        duration=20, tmin=-0.1, tmax=20, preload=True, 
                        baseline=None, fmin=fmin, fmax=fmax, adaptive=True,
                        bandwidth=0.5, sfreq=sfreq)

psd_rest_lfp, freqs = fun.hfb_to_psd(lfp, start=start, stop=stop, 
                        duration=20, tmin=-0.1, tmax=20, preload=True, 
                        baseline=None, fmin=fmin, fmax=fmax, adaptive=True,
                        bandwidth=0.5, sfreq=sfreq)

#%%

start = 425
stop = 690
psd_stim_hfb, freqs = fun.hfb_to_psd(hfb, start=start, stop=stop, 
                        duration=20, tmin=-0.1, tmax=20, preload=True, 
                        baseline=None, fmin=fmin, fmax=fmax, adaptive=True,
                        bandwidth=0.5, sfreq=sfreq)

psd_stim_lfp, freqs = fun.hfb_to_psd(lfp, start=start, stop=stop, 
                        duration=20, tmin=-0.1, tmax=20, preload=True, 
                        baseline=None, fmin=fmin, fmax=fmax, adaptive=True,
                        bandwidth=0.5, sfreq=sfreq)

#%% Plot

sns.set()
plt.subplot(2,1,1)
fun.plot_psd(psd_rest_hfb, freqs, average=True, label='PSD HFB', font = {'size':20})
fun.plot_psd(psd_rest_lfp, freqs, average=True, label='PSD LFP', font = {'size':20})
plt.ylabel('Power Rest (dB)', fontsize=20)

plt.subplot(2,1,2)
fun.plot_psd(psd_stim_hfb, freqs, average=True, label='PSD HFB', font = {'size':20})
fun.plot_psd(psd_stim_lfp, freqs, average=True, label='PSD LFP', font = {'size':20})
plt.xlabel('Frequency (Hz)', fontsize=20)
plt.ylabel('Power Stimulus (dB)', fontsize=20)

#%%

