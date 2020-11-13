#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 19:35:35 2020

@author: guime
"""

import HFB_process as hf
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
proc = 'preproc' 
sfreq = 500; 
suffix = 'preprocessed_raw'
ext = '.fif'

latency_threshold = 160

#%% Read preprocessed data
sub_id = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']
proc = 'preproc' # Line noise removed




def read_raw(sub_id = 'DiAs', task = 'rest_baseline', run='1', 
                            proc= 'raw_signal', suffix = ''):
    
    subject = cifar_load_subject.Subject(name=sub_id, task= task, run = run)
    fpath = subject.dataset_path(proc = proc, suffix=suffix)
    raw = subject.read_eeglab_dataset(fpath, preload=True)
    
    return raw

def concatenate_run_dataset(sub_id = 'DiAs', task = 'rest_baseline'):
    
    raw = read_raw(sub_id = sub_id, task = task, run='1')
    raw_1 = raw.copy()
    
         
    raw = read_raw(sub_id = sub_id, task = task, run='2')
    raw_2 = raw.copy()
    
    raw_1.append([raw_2])
    return raw_1

def annotate_rest(sub_id = 'DiAs'):
    
    raw_rest =  concatenate_run_dataset(sub_id = sub_id, task = 'rest_baseline')
    
    
def concatenate_task_dataset(sub_id = 'DiAs', proc= 'preproc', suffix = 'lnrmv'):
    
    raw_rest =  concatenate_run_dataset(task='rest_baseline')
    raw_stimuli = concatenate_run_dataset(task='stimuli')
    raw_rest.append([raw_stimuli])
    return raw_rest

#%% 

raw = concatenate_task_dataset()

#%% 

raw.plot(scalings=1e-4, duration =200)

raw.plot_psd(xscale = 'log')