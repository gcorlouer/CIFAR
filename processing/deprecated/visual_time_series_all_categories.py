#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 13:25:46 2020

@author: guime
"""


import HFB_process as hf
import cifar_load_subject as cf
import pandas as pd

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
tmin_crop = 0.1
tmax_crop = 0.3
suffix = 'preprocessed_raw'
ext = '.fif'

#%%

subject = cf.Subject(name=sub_id)
datadir = subject.processing_stage_path(proc=proc)
visual_chan = subject.pick_visual_chan()
# visual_chan = hf.pick_visual_chan(picks, visual_chan)
HFB = hf.visually_responsive_HFB(sub_id = sub_id)

categories = ['Rest', 'Face', 'Place']

_, visual_data = hf.HFB_to_visual_data(HFB, visual_chan, sfreq=sfreq, cat='Face', 
                                    tmin_crop = tmin_crop, tmax_crop=tmax_crop)
visual_time_series = visual_data

for cat in categories:
    X, visual_data = hf.HFB_to_visual_data(HFB, visual_chan, sfreq=sfreq, cat=cat, 
                                    tmin_crop = tmin_crop, tmax_crop=tmax_crop)
    visual_time_series[cat] = X

#%% Save dictionary

fname = sub_id + '_visual_HFB_category_specific.mat'
fpath = datadir.joinpath(fname)

# Save data in Rest x Face x Place array of time series

savemat(fpath, visual_time_series)

