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
sfreq = 250
tmin = 0.200
tmax = 1.50
detrend = False
#%%
subject = cf.Subject(sub_id)
datadir = subject.processing_stage_path(proc=proc)
visual_populations = subject.pick_visual_chan()
lfp, visual_chan =hf.load_visual_hfb(sub_id= sub_id, proc= proc, 
                            stage= stage)
# High pass filter
lfp = lfp.copy().filter(l_freq=1, h_freq=None)
#%%

ts, time = hf.category_lfp(lfp, visual_chan, tmin_crop=tmin, tmax_crop =tmax, 
                           sfreq=sfreq)


#%%
ts_dict = {'data': ts, 'sfreq': sfreq, 'time': time, 'sub_id': sub_id}
fname = sub_id + '_ts_visual.mat'
fpath = datadir.joinpath(fname)

savemat(fpath, ts_dict)