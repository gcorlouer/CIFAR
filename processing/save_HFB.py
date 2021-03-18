#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:39:32 2020

@author: guime
"""

# This script extract HFB from bipolar LFP and save file as .fif extension.

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


pd.options.display.max_rows = 999

#%% Parameters 
sub_id = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']
proc = 'preproc' # Line noise removed
for sub in sub_id:
    subject = cf.Subject(name=sub)
    datadir = subject.processing_stage_path(proc=proc)
    fname = sub + '_BP_montage_preprocessed_raw.fif'
    fpath = datadir.joinpath(fname)
    raw = mne.io.read_raw_fif(fpath, preload=True)
    bands = hf.freq_bands() # Select Bands of interests
    HFB = hf.extract_HFB(raw, bands)
    
    fname = sub + '_BP_montage_HFB_raw.fif'
    fpath = datadir.joinpath(fname)
    HFB.save(fpath, overwrite = True)