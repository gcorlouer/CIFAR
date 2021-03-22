#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 17:13:40 2021

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
tmin_postim=0.2
tmax_postim=0.5
alpha=0.05
zero_method='pratt'
alternative='two-sided'
matplotlib.rcParams.update({'font.size': 18})

#%% Detect visual channels

subject = cf.Subject(name=sub_id)
hfb_db = subject.load_data(proc=proc, stage=stage, epo=epo)
visual_channels, effect_size = hf.detect_visual_chan(hfb_db, tmin_prestim=tmin_prestim, 
                                              tmax_prestim=-tmax_prestim
                                              ,tmin_postim=tmin_postim,
                       tmax_postim=tmax_postim, alpha=alpha, zero_method=zero_method, 
                       alternative=alternative)
hfb_visual = hfb_db.copy().pick_channels(visual_channels).crop()
hfb = hfb_visual.pick_channels(['LTo1-LTo2'])
events, event_id = mne.events_from_annotations(hfb)
face_id = extract_stim_id(event_id, cat = 'Face')
place_id = extract_stim_id(event_id, cat='Place')
image_id = face_id+place_id
#%%
