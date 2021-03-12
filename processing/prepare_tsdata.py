#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 10:10:35 2021

@author: guime
"""


import cifar_load_subject as cf
import HFB_process as hf
import pandas as pd
import numpy as np
import helper_functions as fun

from scipy.io import savemat


#%%
tmin_crop = 0.050
tmax_crop = 0.400
sfreq = 200
sub_id = 'DiAs'
proc = 'preproc'
subject = cf.Subject(name=sub_id)
HFB = subject.load_raw_data()
brainpath  = subject.brain_path()
fname = 'BP_channels.csv'
fpath = brainpath.joinpath(fname)
df_BP = pd.read_csv(fpath)
datadir = subject.processing_stage_path(proc=proc)

#%%

ts, time = fun.ts_all_categories(HFB, sfreq=sfreq, tmin_crop=tmin_crop, tmax_crop=tmax_crop)


ts_dict = {'data': ts, 'sfreq': sfreq, 'time': time, 'sub_id': sub_id}
fname = sub_id + '_HFB_all_chan.mat'
fpath = datadir.joinpath(fname)

savemat(fpath, ts_dict)
