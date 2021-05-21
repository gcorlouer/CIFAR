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
stage = '_hfb_db_epo.fif'
epo = True
tmin = -0.5
tmax = 1.75
t_prestim = -0.5
t_postim = 1.75
tmin_prestim=-0.4
tmax_prestim=-0.1
tmin_postim=0.1
tmax_postim=0.5
alpha=0.05
zero_method='pratt'
alternative='two-sided'

for sub in sub_id:
    # Subject parameters
    sub = sub
    
    #%% Read preprocessed data
    
    subject = cf.Subject(name=sub)
    datadir = subject.processing_stage_path(proc=proc)
    hfb_db = subject.load_data(proc=proc, stage=stage, epo=epo)
    dfelec = subject.df_electrodes_info()
    
    visual_populations = hf.hfb_to_visual_populations(hfb_db, dfelec,
                       tmin_prestim=tmin_prestim, tmax_prestim=tmax_prestim, tmin_postim=tmin_postim,
                       tmax_postim=tmax_postim, alpha= alpha, zero_method=zero_method)
    # Create single subject dataframe
    df_visual = pd.DataFrame.from_dict(visual_populations)
    df_visual = df_visual.sort_values(by='Y', ignore_index=True)
    # Uncomment if want to remove null latency
    # index_names = df_visual[df_visual['latency']==0].index
    #df_visual = df_visual.drop(index_names)
    
    brain_path = subject.brain_path()
    fname = 'visual_BP_channels.csv'
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
    fname = 'visual_BP_channels.csv'
    df_visual = pd.read_csv(fpath)
    subject_id = [sub]*len(df_visual)
    df_visual['subject_id'] = subject_id
    df_visual_table = df_visual_table.append(df_visual)
    
path = cf.cifar_ieeg_path()
fname = 'cross_subjects_visual_BP_channels.csv'
fpath = path.joinpath(fname)
df_visual_table.to_csv(fpath, index=False)

#%% Reject zero latency visual channels

#index_names = df_visual_table[df_visual_table['latency']==0].index
# df_visual_table = df_visual_table.drop(index_names)