#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 13:44:21 2021

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
from netneurotools import stats as nnstats
from scipy.stats.stats import _chk2_asarray
from sklearn.utils.validation import check_random_state

pd.options.display.max_rows = 999
#%%

sub = 'DiAs'
proc = 'preproc' 
sfreq = 500; 
suffix = '_BP_montage_preprocessed_raw'
ext = '.fif'

#%% Read preprocessed data

subject = cf.Subject(name=sub)
datadir = subject.processing_stage_path(proc=proc)
fname = sub + '_BP_montage_HFB_raw.fif'
fpath = datadir.joinpath(fname)
HFB = mne.io.read_raw_fif(fpath, preload=True)        
dfelec = subject.df_electrodes_info()
HFB_db = hf.HFB_to_db(HFB)
events, event_id = mne.events_from_annotations(HFB)
face_id = hf.extract_stim_id(event_id)
place_id = hf.extract_stim_id(event_id, cat='Place')
image_id = face_id+place_id


#%% 

visual_populations = hf.HFB_to_visual_populations(HFB, dfelec, t_pr = -0.5, t_po = 1.75, baseline=None,
                       preload=True, tmin_pr=-0.2, tmax_pr=0, tmin_po=0.2,
                       tmax_po=1, alpha= 0.01)

#%% Plot pval series

ichan = 7
plt.plot(pval[ichan,:])
plt.axhline(y=0.01, color = 'r')

#%% Plot HFB evoked


ichan = 3
ROL = latency_response[ichan]*1e-3
X = visual_HFB.copy().get_data()
time = visual_HFB.times
channel = X[:,ichan,:]
evok = np.average(channel, axis=0)

plt.plot(time, evok)
plt.axvline(x=0, color = 'k')
plt.axvline(x=ROL, color = 'r', ls='--')
plt.axhline(y=0, color='k')