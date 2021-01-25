#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 17:06:32 2021

@author: guime
"""



import HFB_process as hf
import cifar_load_subject as cf
import scipy as sp
import re 
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path, PurePath
from mne.viz import plot_filter, plot_ideal_filter
from scipy import signal, fftpack
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection, multipletests

from scipy.io import loadmat,savemat
# %%
pd.options.display.max_rows = 999

sub_id = 'DiAs'
visual_chan_table = 'visual_channels_BP_montage.csv'
proc = 'preproc' 
sfreq = 100;
picks = ['LGRD60-LGRD61', 'LTo1-LTo2']
tmin_crop = 0.5
tmax_crop = 1.5
suffix = 'preprocessed_raw'
ext = '.fif'

#%%

subject = cf.Subject(name=sub_id)
datadir = subject.processing_stage_path(proc=proc)
visual_chan = subject.low_high_chan()
# visual_chan = hf.pick_visual_chan(picks, visual_chan)
HFB = hf.low_high_HFB(visual_chan)