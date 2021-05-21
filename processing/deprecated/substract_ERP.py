#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Feb 22 12:18:43 2021

@author: guime
"""


import HFB_process as hf
import cifar_load_subject as cf
import pandas as pd
import numpy as np 

from scipy.io import savemat

# %matplotlib
#%% TODO

# -Check that there are output visual_data X is correct with HFB_visual (i.e. check that 
# permutation works)
# - Create a module for category specific electrodes
# - Rearrange HFB module consequently
# sub_id = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']

#%% 
pd.options.display.max_rows = 999

sub_id = 'DiAs'
visual_chan_table = 'visual_channels_BP_montage.csv'
proc = 'preproc' 
sfreq = 250;
# picks = ['LGRD58-LGRD59', 'LGRD60-LGRD61', 'LTo1-LTo2', 'LTo3-LTo4']
tmin_crop = 0.050
tmax_crop = 0.3
suffix = 'preprocessed_raw'
ext = '.fif'
categories = ['Rest', 'Face', 'Place']

#%%

subject = cf.Subject(name=sub_id)
datadir = subject.processing_stage_path(proc=proc)
visual_chan = subject.pick_visual_chan()
# visual_chan = hf.pick_visual_chan(picks, visual_chan)
HFB = hf.visually_responsive_HFB(sub_id = sub_id)

def ts_demean_all_categories(HFB, tmin_crop=0.050, tmax_crop=0.250):

    categories = ['Rest', 'Face', 'Place']
    ncat = len(categories)
    ts = [0]*ncat
    for idx, cat in enumerate(categories):
        epochs = hf.category_specific_HFB(HFB, cat=cat, tmin_crop=tmin_crop,
                                       tmax_crop=tmax_crop)
        epochs = epochs.resample(sfreq=sfreq)
        X = epochs.get_data().copy()
        ERP = np.mean(X,axis=0)
        ntrial = X.shape[0]
        for i in range(ntrial):
            X[i,:,:] = X[i,:,:] - ERP
        time = epochs.times
        ts[idx] = X

    ts = np.stack(ts)
    (ncat, ntrial, nchan, nobs) = ts.shape
    ts = np.transpose(ts, (2, 3, 1, 0))
    return ts, time


ts, time = ts_demean_all_categories(HFB, tmin_crop=tmin_crop, tmax_crop=tmax_crop)

ts_dict = {'data': ts, 'sfreq': sfreq, 'time': time, 'sub_id': sub_id}
fname = sub_id + '_visual_HFB_all_categories.mat'
fpath = datadir.joinpath(fname)

savemat(fpath, ts_dict)