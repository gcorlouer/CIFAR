#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 18:38:01 2020

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


pd.options.display.max_rows = 999

#%% Parameters 

sub_id = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']
proc = 'preproc' # Line noise removed
sfreq = 500; 
suffix = '_BP_montage_preprocessed_raw'
ext = '.fif'
t_pr = -0.5
t_po = 1.75
tmin_pr=-0.2 
tmax_pr=0
tmin_po=0.2
tmax_po=0.5
alpha= 0.05

for sub in sub_id:
    # Subject parameters
    sub = sub
    
    #%% Read preprocessed data
    
    subject = cf.Subject(name=sub)
    datadir = subject.processing_stage_path(proc=proc)
    fname = sub + '_BP_montage_HFB_raw.fif'
    fpath = datadir.joinpath(fname)
    HFB = mne.io.read_raw_fif(fpath, preload=True)        
    dfelec = subject.df_electrodes_info()
        
    visual_populations = hf.HFB_to_visual_populations(HFB, dfelec, t_pr = t_pr, t_po = t_po, baseline=None,
                       preload=True, tmin_pr=tmin_pr, tmax_pr=tmax_pr, tmin_po=tmin_po,
                       tmax_po=tmax_po, alpha= alpha)
    # Create single subject dataframe
    df_visual = pd.DataFrame.from_dict(visual_populations, orient='index')
    df_visual = df_visual.transpose()
    index_names = df_visual[df_visual['latency']==0].index
    df_visual = df_visual.drop(index_names)
    
    brain_path = subject.brain_path()
    fname = 'visual_channels_BP_montage.csv'
    fpath = brain_path.joinpath(fname)
    
    df_visual.to_csv(fpath, index=False)
    
#%% Make a table of all visual channels for all subjects

columns = list(visual_populations.keys())
columns.append('subject_id')
df_visual_table = pd.DataFrame(columns=columns)

for sub in sub_id:
    subject = cf.Subject(name=sub)
    brain_path = subject.brain_path()
    fpath = brain_path.joinpath(fname)
    fname = 'visual_channels_BP_montage.csv'
    df_visual = pd.read_csv(fpath)
    subject_id = [sub]*len(df_visual)
    df_visual['subject_id'] = subject_id
    df_visual_table = df_visual_table.append(df_visual)
    
path = cf.cifar_ieeg_path()
fname = 'visual_channels_BP_montage.csv'
fpath = path.joinpath(fname)
df_visual_table.to_csv(fpath, index=False)

#%% Reject zero latency visual channels

index_names = df_visual_table[df_visual_table['latency']==0].index
# df_visual_table = df_visual_table.drop(index_names)