#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 09:22:25 2021

@author: guime
"""


import HFB_process as hf
import cifar_load_subject as cf
import mne
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import helper_functions as fun
from scipy import stats

from pathlib import Path, PurePath

#%% Parameters
proc = 'preproc'
sub_id = 'DiAs'
stage = '_hfb_db_epo.fif'
epo = True
tmin = -0.5
tmax = 1.75
tmin_prestim=-0.4
tmax_prestim=-0.1
tmin_postim=0.1
tmax_postim=0.5
alpha=0.05
zero_method='pratt'
alternative='two-sided'
matplotlib.rcParams.update({'font.size': 18})

#%% Detect visual channels

subject = cf.Subject(name=sub_id)
dfelec = subject.df_electrodes_info()
hfb_db = subject.load_data(proc=proc, stage=stage, epo=epo)
visual_channels, effect_size = hf.detect_visual_chan(hfb_db, tmin_prestim=tmin_prestim, 
                                              tmax_prestim=-tmax_prestim
                                              ,tmin_postim=tmin_postim,
                       tmax_postim=tmax_postim, alpha=alpha, zero_method=zero_method, 
                       alternative=alternative)
hfb_visual = hfb_db.copy().pick_channels(visual_channels).crop()
event_id = hfb_visual.event_id
face_id = hf.extract_stim_id(event_id, cat = 'Face')
place_id = hf.extract_stim_id(event_id, cat='Place')
image_id = face_id+place_id
times = hfb_visual.times

#%% Test face and place classification

group, category_selectivity = hf.classify_Face_Place(hfb_visual, face_id, place_id, visual_channels, 
                        tmin_postim=tmin_postim, tmax_postim=tmax_postim, alpha=alpha,
                        zero_method=zero_method)

group = hf.classify_retinotopic(visual_channels, group, dfelec)

print('Face: ' + str(group.count('F')))
print('Place: ' + str(group.count('P')))
print('Retinotopic ' + str(group.count('R')))

#%% Test peak time

peak_time = hf.compute_peak_time(hfb_db, visual_channels)
#%%

visual_populations = hf.hfb_to_visual_populations(hfb_db, dfelec,
                       tmin_prestim=-0.4, tmax_prestim=-0.1, tmin_postim=0.1,
                       tmax_postim=0.5, alpha= 0.05, zero_method='pratt')

