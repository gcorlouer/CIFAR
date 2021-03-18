#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:04:56 2021

@author: guime
"""

import HFB_process as hf
import cifar_load_subject as cf
import mne
import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path, PurePath

#%% Parameters
sub_id = 'DiAs'
proc = 'preproc'
subject = cf.Subject(name=sub_id)
datadir = subject.processing_stage_path(proc=proc)
fname = sub_id + '_hfb_db_epo.fif'
tmin_prestim=-0.4
tmax_prestim=-0.1
tmin_postim=0.1
tmax_postim=0.5
alpha=0.05
zero_method='zsplit'
alternative='greater'

#%% Load data
fpath = datadir.joinpath(fname)
hfb_db = mne.read_epochs(fpath, preload=True)

#%% Test detect_visual_chans

visual_channels, effect_size = hf.detect_visual_chan(hfb_db, tmin_prestim=tmin_prestim, 
                                                  tmax_prestim=-tmax_prestim
                                                  ,tmin_postim=tmin_postim,
                           tmax_postim=tmax_postim, alpha=alpha, zero_method=zero_method, 
                           alternative=alternative)


