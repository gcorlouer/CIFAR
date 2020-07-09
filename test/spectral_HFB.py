#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:42:34 2020

@author: guime
"""


import cf_load
import HFB_test
import mne
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp 

from pathlib import Path
from mne.time_frequency import (tfr_multitaper, tfr_stockwell, tfr_morlet,
                                tfr_array_morlet)
from mne.time_frequency import csd_fourier, csd_multitaper, csd_morlet
from mne.time_frequency import psd_multitaper, psd_welch
%matplotlib

plt.rcParams.update({'font.size': 34})

# Some notes
# HFB spectra on epochs or continuous data yield quite different results
# Interesting to see variation in specta for prefered and non prefered 
# categories


# %%
t_pr= -0.5
t_po=1.75
baseline = None
preload = True 
preproc = 'preproc'
suffix = 'lnrmv'
task = 'stimuli'
run = '1'
sub = 'DiAs'

cat = 'Face'

# %% 
path_visual = cf_load.visual_path()
df_visual = pd.read_csv(path_visual)

face_chan = list(df_visual['chan_name'].loc[df_visual['subject_id']==sub].loc[df_visual['category']=='Face'])
place_chan = list(df_visual['chan_name'].loc[df_visual['subject_id']==sub].loc[df_visual['category']=='Place'])
    
subject = cf_load.Subject(name=sub, task= task, run = run)
fpath = subject.fpath(proc = preproc, suffix=suffix)
raw = subject.import_data(fpath)

# %% 
bands = HFB_test.freq_bands()
HFB =  HFB_test.extract_HFB (raw, bands)

#%% 
HFB_db = HFB_test.extract_HFB_db(raw, bands, t_pr=t_pr, t_po=t_po)

events, event_id = mne.events_from_annotations(raw) 

cat_id = HFB_test.extract_stim_id(event_id, cat = cat)
epochs = HFB_test.epoch_HFB(HFB, raw, t_pr = t_pr, t_po = t_po)
epochs = epochs[cat_id].copy()

#%% Spectral properties
pick = 'LTo7'
# epochs = HFB_db
psd, freqs = psd_multitaper(HFB.copy().pick(pick))
psd = np.ndarray.flatten(psd)
psd = 10*np.log10(psd)
psd_mean = psd.mean(0)
psd_mean = psd_mean[0]
psd_SE = sp.stats.sem(psd, 0)
psd_SE = psd_SE[0]

plt.plot(freqs, psd_mean)
plt.fill_between(freqs, psd_mean-1.96*psd_SE, psd_mean+1.96*psd_SE, alpha=0.3)
plt.xscale('log')
plt.xlim((0.1, 10))
plt.ylim((0,30))

#%% Other test
pick = 'LGRD60'
# epochs = HFB_db
psd, freqs = psd_multitaper(HFB.copy().pick(pick))
psd = np.ndarray.flatten(psd)
psd = 10*np.log10(psd)

plt.plot(freqs, psd)
plt.xscale('log')
plt.xlim((0.1, 10))