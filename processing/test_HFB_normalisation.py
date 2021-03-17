#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 17:44:49 2021

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
fname = sub_id + '_BP_montage_preprocessed_raw.fif'
tmin = 100
tmax = 102
l_freq = 70
band_size=20.0
l_trans_bandwidth= 10.0
h_trans_bandwidth= 10.0
filter_length='auto'
phase='minimum'
ichan = 6
figname = 'HFB_normalisation.jpg'
figpath = Path.home().joinpath('projects','CIFAR','figures', figname)