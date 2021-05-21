#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 13:50:56 2020

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

# %matplotlib
#%% TODO

# -Check that there are output visual_data X is correct with HFB_visual (i.e. check that 
# permutation works)
# - Create a module for category specific electrodes
# - Rearrange HFB module consequently

#%% 
pd.options.display.max_rows = 999

sub_id = 'DiAs'
visual_chan_table = 'visual_channels_BP_montage.csv'
proc = 'preproc'
cat = 'Face' 
sfreq = 100;
picks = ['LGRD60-LGRD61', 'LTo1-LTo2']
tmin_crop = 0.5
tmax_crop = 1.5
suffix = 'preprocessed_raw'
ext = '.fif'


#%%v

subject = cf.Subject(name=sub_id)
datadir = subject.processing_stage_path(proc=proc)
visual_chan = subject.pick_visual_chan()
sorted_visual_chan = visual_chan.sort_values(by='latency')


# %%
group = visual_chan['group'].unique().tolist()



anatomical_indices = hf.parcellation_to_indices(visual_chan, parcellation='DK')