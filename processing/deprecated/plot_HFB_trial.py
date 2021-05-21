#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 10:40:58 2021

@author: guime
"""



import HFB_process as hf
import cifar_load_subject as cf
import scipy as sp
import re 
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path, PurePath
from mne.viz import plot_filter, plot_ideal_filter
from scipy import signal, fftpack
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection, multipletests

from scipy.io import loadmat,savemat

#%% Parameters

pd.options.display.max_rows = 999

sub_id = 'DiAs'
visual_chan_table = 'visual_channels_BP_montage.csv'
proc = 'preproc'
ichan = 2
cat = 'Place'
sfreq = 250;
tmin_crop = -0.5
tmax_crop = 1.75
suffix = 'preprocessed_raw'
ext = '.fif'

#%% Load data

subject = cf.Subject(name=sub_id)
datadir = subject.processing_stage_path(proc=proc)
visual_chan = subject.pick_visual_chan()
HFB = hf.visually_responsive_HFB(sub_id = sub_id)
X, visual_data = hf.HFB_to_visual_data(HFB, visual_chan, sfreq=sfreq, cat=cat, 
                                    tmin_crop = tmin_crop, tmax_crop=tmax_crop)

#%% Exctract observables

latency_response = visual_data['latency'][ichan]/500
channel = X[ichan,:,:]
evok = np.average(channel, axis=1)
time = visual_data['time']

#%%  Plot evoked response
import matplotlib
matplotlib.rcParams.update({'font.size': 22})

plt.plot(time, evok)
plt.axvline(x=0, color = 'k', label= 'stimulus onset')
plt.axvline(x=latency_response, color = 'r', ls='--', label = 'latency response')
plt.axhline(y=0, color='k')
plt.xlabel('Time from stimulus onset (s)')
plt.ylabel('Amplitude (dB)')
plt.legend()