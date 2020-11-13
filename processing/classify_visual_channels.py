#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 18:38:01 2020

@author: guime
"""


import HFB_process
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


pd.options.display.max_rows = 999

#%% Parameters 
sub_id = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']
proc = 'preproc' # Line noise removed
for sub in sub_id:
    # Subject parameters
    sub = sub
    task = 'stimuli' # stimuli or rest_baseline_1
    run = '1'
    sfreq = 500; 
    suffix = 'preprocessed_raw'
    ext = '.fif'
    
    #%% Read preprocessed data
    
    subject = cifar_load_subject.Subject(name=sub, task= task, run = run)
    fpath = subject.dataset_path(proc = proc, suffix=suffix, ext = ext)
    raw = mne.io.read_raw_fif(fpath, preload=True)    
    dfelec = subject.df_electrodes_info()
    
    bands = HFB_process.freq_bands() # Select Bands of interests

    visual_populations = HFB_process.raw_to_visual_populations(raw, bands, dfelec,latency_threshold=170)
    
    df_visual = pd.DataFrame(visual_populations)
    
    brain_path = subject.brain_path()
    fname = 'visual_channels.csv'
    fpath = brain_path.joinpath(fname)
    df_visual.to_csv(fpath, index=False)
    
    df_visual = pd.read_csv(fpath)
#%% Make a table of all visual channels for all subjects

columns = ['chan_name', 'group', 'latency', 'effect_size', 'brodman', 'DK', 'subject_id']
df_visual_table = pd.DataFrame(columns=columns)

for sub in sub_id:
    subject = cifar_load_subject.Subject(name=sub, task= task, run = run)
    brain_path = subject.brain_path()
    fpath = brain_path.joinpath(fname)
    fname = 'visual_channels.csv'
    df_visual = pd.read_csv(fpath)
    subject_id = [sub]*len(df_visual)
    df_visual['subject_id'] = subject_id
    df_visual_table = df_visual_table.append(df_visual)
    
path = cf.cifar_ieeg_path()
fname = 'visual_channels_table'
fpath = path.joinpath(fname)
df_visual_table.to_csv(fpath, index=False)

#%% 

