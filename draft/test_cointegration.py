#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 09:42:35 2021

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
from statsmodels.tsa.stattools import coint
from statsmodels.stats.multitest import fdrcorrection, multipletests

#%%

sub_id= 'DiAs'
proc= 'preproc' 
stage= '_BP_montage_HFB_raw.fif'
picks = ['LTo1-LTo2', 'LTo5-LTo6']
sfreq = 250
tmin_crop = 0
tmax_crop = 1.5
alpha = 0.05
#%%
subject = cf.Subject(sub_id)
datadir = subject.processing_stage_path(proc=proc)
visual_populations = subject.pick_visual_chan()
hfb, visual_chan =hf.load_visual_hfb(sub_id= sub_id, proc= proc, 
                            stage= stage)
ts, time = hf.chan_specific_category_ts(picks, sub_id= sub_id, proc= proc, sfreq=sfreq, 
                            stage= stage, tmin_crop=tmin_crop, tmax_crop=tmax_crop)
(nchan, nobs, ntrials, ncat) = ts.shape
#%%

t_stat = np.zeros((ntrials, ncat))
pval = np.zeros((ntrials, ncat))
reject = np.zeros((ntrials, ncat))
pval_correct = np.zeros((ntrials, ncat))
fraction_rejected = np.zeros((ncat))

for i in range(ntrials):
    for j in range(ncat):
        y0 = ts[0,:,i,j]
        y1 = ts[1,:,i,j]
        t_stat[i,j], pval[i,j], _ = coint(y0, y1)

for i in range(ncat):
    reject[:,i], pval_correct[:,i] = fdrcorrection(pval[:,i], alpha)

for i in range(ncat):
    fraction_rejected[i] = len(np.where(reject[:,i]==1)[0])/ntrials