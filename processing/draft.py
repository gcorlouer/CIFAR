#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 19:35:35 2020

@author: guime
"""

import HFB_process
import cifar_load_subject
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

sub = 'DiAs'
task = 'stimuli' # stimuli or rest_baseline_1
run = '1'
proc = 'preproc' 
sfreq = 500; 
suffix = 'preprocessed_raw'
ext = '.fif'

latency_threshold = 160

#%% Read preprocessed data

subject = cifar_load_subject.Subject(name=sub, task= task, run = run)
fpath = subject.dataset_path(proc = proc, suffix=suffix, ext = ext)
dfelec = subject.df_electrodes_info()   
raw = mne.io.read_raw_fif(fpath, preload=True)
 
bands = HFB_process.freq_bands() # Select Bands of interests

visual_populations = HFB_process.raw_to_visual_populations(raw, bands, dfelec,
                                                           latency_threshold=latency_threshold)