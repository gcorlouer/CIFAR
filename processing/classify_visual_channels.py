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
    proc = 'preproc' 
    sfreq = 500; 
    suffix = '_BP_montage_preprocessed_raw'
    ext = '.fif'
    
    #%% Read preprocessed data
    
    subject = cf.Subject(name=sub)
    datadir = subject.processing_stage_path(proc=proc)
    fname = sub + '_BP_montage_preprocessed_raw.fif'
    fpath = datadir.joinpath(fname)
    raw = mne.io.read_raw_fif(fpath, preload=True)        
    dfelec = subject.df_electrodes_info()
    
    bands = HFB_process.freq_bands() # Select Bands of interests
    
    visual_populations = HFB_process.raw_to_visual_populations(raw, bands, dfelec)
    
    df_visual = pd.DataFrame.from_dict(visual_populations, orient='index')
    df_visual = df_visual.transpose()
    
    brain_path = subject.brain_path()
    fname = 'visual_channels_BP_montage.csv'
    fpath = brain_path.joinpath(fname)
    df_visual.to_csv(fpath, index=False)
    
    df_visual = pd.read_csv(fpath)
#%% Make a table of all visual channels for all subjects

columns = ['chan_name', 'group', 'latency', 'effect_size', 'brodman', 'DK', 'subject_id']
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

#%% Create tables

# populations = ['V1', 'V2', 'other', 'Place', 'Face']
# keys = ['visual'] + populations
# keys = ['subject_id'] + keys

# df_demographic = pd.DataFrame(columns=keys)

# visual = [0]*len(sub_id)
# V1 = [0]*len(sub_id)
# V2 = [0]*len(sub_id)
# other = [0]*len(sub_id)
# place = [0]*len(sub_id)
# face = [0]*len(sub_id)

# def count_values(df_visual_table, sub='DiAS', population='V1'):
#     if population in df_visual_table['group'].loc[df_visual_table['subject_id']==sub].values:
#         nchans = df_visual_table['group'].loc[df_visual_table['subject_id']==sub].value_counts()[population]
#     else:
#         nchans = 0
        
#     return nchans

# for idx, sub in enumerate(sub_id):
    
#     visual[idx] = df_visual_table['subject_id'].value_counts()[sub]
#     V1[idx] =  count_values(df_visual_table, sub=sub, population='V1')
#     V2[idx] = count_values(df_visual_table, sub=sub, population='V2')
#     other[idx] = count_values(df_visual_table, sub=sub, population='other')
#     place[idx] = count_values(df_visual_table, sub=sub, population='Place')
#     face[idx] = count_values(df_visual_table, sub=sub, population='Face')
    
    
# dict_demographic = dict.fromkeys(keys, [])


# df_demographic['subject_id'] = sub_id
# df_demographic['V1'] = V1
# df_demographic['V2'] = V2
# df_demographic['other'] = other
# df_demographic['Place'] = place
# df_demographic['Face'] = face
# df_demographic['visual'] = visual

# path = cf.cifar_ieeg_path()
# fname = 'visual_demographics_BP_montage.csv'
# fpath = path.joinpath(fname)
# df_demographic.to_csv(fpath, index=False)

