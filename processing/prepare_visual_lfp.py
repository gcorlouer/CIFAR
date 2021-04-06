#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 10:02:15 2021
This script prepare category specific LFP time series with all
visual channels for mvgc analysis
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
stage= '_BP_montage_preprocessed_raw.fif'
picks = ['LTo1-LTo2', 'LTo5-LTo6']
sfreq = 250
tmin = 0
tmax = 1.75
detrend = False
#%%
subject = cf.Subject(sub_id)
datadir = subject.processing_stage_path(proc=proc)
visual_populations = subject.pick_visual_chan()
lfp, visual_chan =hf.load_visual_hfb(sub_id= sub_id, proc= proc, 
                            stage= stage)
#%%

ts, time = hf.chan_specific_category_lfp(picks, tmin_crop=tmin, sub_id=sub_id,
                                         tmax_crop =tmax, sfreq=sfreq, proc=proc,
                                         stage=stage)


#%%
ts_dict = {'data': ts, 'sfreq': sfreq, 'time': time, 'sub_id': sub_id}
fname = sub_id + '_ts_visual.mat'
fpath = datadir.joinpath(fname)

savemat(fpath, ts_dict)