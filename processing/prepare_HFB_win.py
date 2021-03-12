#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 12:02:32 2021

@author: guime
"""

# Time segments of interest: [60 200 270 410 426 558 564 690]
import cifar_load_subject as cf
import HFB_process as hf
import pandas as pd
import numpy as np
import helper_functions as fun

from scipy.io import savemat


#%%
sub_id = 'DiAs'
proc = 'preproc'
sfreq = 200
subject = cf.Subject(name=sub_id)
HFB = hf.visually_responsive_HFB(sub_id = sub_id)
visual_chan = subject.pick_visual_chan()
#HFB = subject.load_raw_data()
brainpath  = subject.brain_path()
fname = 'BP_channels.csv'
fpath = brainpath.joinpath(fname)
df_BP = pd.read_csv(fpath)
datadir = subject.processing_stage_path(proc=proc)

#%%

HFB_win = HFB.copy().crop(tmin=70, tmax=80, include_tmax=True)
HFB_win = HFB_win.resample(sfreq=sfreq)
ts = HFB_win.copy().get_data()
time = HFB_win.times

ts_dict = {'data': ts, 'sfreq': sfreq, 'time': time, 'sub_id': sub_id}
fname = sub_id + '_HFB_win_all_chan.mat'
fpath = datadir.joinpath(fname)

savemat(fpath, ts_dict)

#%% Plot HFB win:

HFB.plot(duration=10, n_channels=8, scalings=1e-5)
